"""Разбор страницы по частям: макет (CPU) -> блоки -> пакетное распознавание GLM-OCR или PaddleOCR-VL (GPU) -> Markdown.

parse_pages(images, engine="glm_ocr") -> [{"markdown", "blocks", "sec_layout", "sec_ocr"}]"""
import difflib
import html as htmllib
import re
import time
from html.parser import HTMLParser

from . import layout, textfix
from .structure import layout_unreliable, split_bands, structure_page_text
from . import vlm
from .vlm import generate_batch as vlm_generate_batch
from .weights import WEIGHTS

ENGINES = {
    "glm_ocr": dict(path=str(WEIGHTS / "glm_ocr"), trust=True,
                    prompts={"text": "Text Recognition:", "table": "Table Recognition:", "formula": "Formula Recognition:"}),
    "paddleocr_vl": dict(path=str(WEIGHTS / "paddleocr_vl"), trust=False,
                         prompts={"text": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:"}),
}
ENGINE_BY_KIND = {"page": "glm_ocr", "text": "glm_ocr", "table": "paddleocr_vl", "formula": "paddleocr_vl"}  # по специализации
KIND = {"title": "text", "text": "text", "figure_caption": "text", "table_caption": "text", "table_footnote": "text",
        "formula_caption": "text", "table": "table", "formula": "formula"}
MAX_TOKENS = {"page": 450, "text": 400, "table": 500, "formula": 350}
PAD = 6


# ---------- постобработка ----------
class _Tbl(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.cell, self.span = [], None, None, False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            self.cell = []
            a = dict(attrs)
            if int(a.get("colspan", 1) or 1) > 1 or int(a.get("rowspan", 1) or 1) > 1:
                self.span = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.cur is not None:
            self.cur.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def html_table_to_md(src):
    """HTML-таблица -> Markdown (у pipe-таблиц рамки рисует любой просмотрщик). Со сложными ячейками: HTML с border."""
    p = _Tbl()
    p.feed(src)
    if p.span or not p.rows:
        return re.sub(r"<table[^>]*>", '<table border="1" cellpadding="4">', src.strip(), count=1)
    n = max(len(r) for r in p.rows)
    rows = [r + [""] * (n - len(r)) for r in p.rows]
    esc = lambda c: c.replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def otsl_to_md(src):
    """Формат таблиц PaddleOCR-VL (<fcel>текст<nl>...) -> Markdown-таблица."""
    rows = []
    for line in src.strip().split("<nl>"):
        if not line.strip():
            continue
        cells = []
        for tag, txt in re.findall(r"<(fcel|ecel|lcel|ucel|xcel)>([^<]*)", line):
            cells.append(txt.strip() if tag == "fcel" else "")
        rows.append(cells)
    if not rows:
        return src
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    esc = lambda c: c.replace("|", chr(92) + "|")
    out = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return chr(10).join(out)


def table_to_md(t):
    if "<fcel>" in t or "<ecel>" in t:
        return otsl_to_md(t)
    return html_table_to_md(t) if "<table" in t else t


def fix_text(s):
    """Пробелы вокруг встроенных формул и общая чистка."""
    parts = re.split(r"(\$[^$\n]+\$)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # формула
            if out and out[-1] and out[-1][-1].isalnum():
                out[-1] += " "
            out.append(p)
        else:
            if i > 0 and p and p[0].isalnum():
                p = " " + p
            out.append(p)
    return re.sub(r"[ \t]+\n", "\n", "".join(out)).strip()


def fix_formula(s):
    s = s.strip()
    s = re.sub(r"^\$\$|\$\$$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    s = re.sub(r"\\mathrm\s*\{([^{}]*[А-Яа-яЁё][^{}]*)\}", r"\\text{\1}", s)  # кириллица в \mathrm теряет пробелы в KaTeX
    s = re.sub(r"\s*([_^{}])\s*", r"\1", s)  # GLM пишет LaTeX пословно через пробелы
    return s


def assemble(blocks):
    out = []
    page = [b for b in blocks if b["kind"] == "page"]
    if page:  # режим страницы: полосы склеиваются с учётом промежутков, затем строится структура
        text = ""
        for b in page:
            cur = b.get("text", "").strip()
            prev_lines = text.rstrip().split(chr(10))
            cur_lines = cur.split(chr(10))
            # строка на границе полос попала в обе: если они похожи, оставляем первую
            if text and prev_lines and cur_lines and difflib.SequenceMatcher(
                    None, prev_lines[-1].lower(), cur_lines[0].lower()).ratio() > 0.55:
                cur = chr(10).join(cur_lines[1:])
            text += b.get("sep", "") + cur + chr(10)
        return structure_page_text(text)
    for b in blocks:
        t = b.get("text", "")
        if b["cls"] == "abandon" or not t:
            continue
        if b["kind"] == "page":
            out.append(structure_page_text(t))
        elif b["kind"] == "table":
            out.append(table_to_md(t))
        elif b["kind"] == "formula":
            out.append("$$\n" + fix_formula(t) + "\n$$")
        elif b["cls"] == "title":
            out.append("## " + fix_text(t))
        elif b["cls"] in ("figure_caption", "table_caption", "formula_caption"):
            out.append("*" + fix_text(t) + "*")
        else:
            out.append(fix_text(t))
    return "\n\n".join(out)


# ---------- основной разбор ----------
def parse_pages(images, engine=None, device="cuda", batch=8, hints=None, fix=True):
    """engine=None: модель выбирается по типу блока (ENGINE_BY_KIND), иначе одна модель на всё."""
    pages = []
    t0 = time.time()
    for k, im in enumerate(images):
        hint = hints[k] if hints else None  # тип уже известен (например, классификатор определил «формула» или «таблица»)
        if hint in ("formula", "table"):
            pages.append({"image": im, "blocks": [{"cls": hint, "kind": hint, "box": (0, 0, im.width, im.height), "score": 1.0}]})
            continue
        bl = [dict(b, kind=KIND[b["cls"]]) for b in layout.detect(im) if b["cls"] in KIND]
        bl = [b for b in bl if b["cls"] != "abandon"]
        if layout_unreliable(bl, im.size):  # чеки, сканы: страница целиком, структуру строит structure.py
            bl = [{"cls": "text", "kind": "page", "box": (0, y1, im.width, y2), "score": 1.0, "sep": sep}
                  for y1, y2, sep in split_bands(im)]  # длинную страницу режем на полосы и читаем пакетом
        pages.append({"image": im, "blocks": bl})
    t_layout = time.time() - t0

    # очередь блоков со всех страниц: группируем по типу, внутри по площади (меньше padding)
    queue = [(pi, bi, b) for pi, p in enumerate(pages) for bi, b in enumerate(p["blocks"])]
    t1 = time.time()
    for kind in ("page", "text", "table", "formula"):
        cfg = ENGINES[engine or ENGINE_BY_KIND[kind]]
        items = sorted((q for q in queue if q[2]["kind"] == kind),
                       key=lambda q: (q[2]["box"][2] - q[2]["box"][0]) * (q[2]["box"][3] - q[2]["box"][1]))
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            crops = []
            for pi, bi, b in chunk:
                im = pages[pi]["image"]
                x1, y1, x2, y2 = b["box"]
                crops.append(im.crop((max(0, int(x1 - PAD)), max(0, int(y1 - PAD)), min(im.width, int(x2 + PAD)), min(im.height, int(y2 + PAD)))))
            res = vlm_generate_batch(cfg["path"], crops, [cfg["prompts"]["text" if kind == "page" else kind]] * len(crops), device,
                                     max_new_tokens=MAX_TOKENS[kind], trust=cfg["trust"])
            for (pi, bi, b), text in zip(chunk, res):
                b["text"] = text
    t_ocr = time.time() - t1
    t_fix, fix_stats = 0.0, {}
    if fix:  # правка опечаток: сначала освобождаем видеопамять от OCR-моделей, затем грузим Qwen3-1.7B
        t2 = time.time()
        vlm.free()
        targets = [b for p in pages for b in p["blocks"] if b["kind"] == "text" and len(b.get("text", "")) >= 15]
        if targets:
            fixed, fix_stats = textfix.correct_texts([b["text"] for b in targets], device=device)
            for b, f in zip(targets, fixed):
                if f != b["text"]:
                    b["text_raw"], b["text"] = b["text"], f
        textfix.free()
        t_fix = time.time() - t2
    out = []
    for p in pages:
        out.append({"markdown": assemble(p["blocks"]),
                    "blocks": [{k: v for k, v in b.items() if k != "box"} | {"box": [round(x) for x in b["box"]]} for b in p["blocks"]],
                    "sec_layout": round(t_layout / len(pages), 2), "sec_ocr": round(t_ocr / len(pages), 2),
                    "sec_fix": round(t_fix / len(pages), 2), "fix": fix_stats})
    return out
