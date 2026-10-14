"""python run_all.py [--device cuda] [--only id,id]
Прогоняет все модели по подходящим категориям (каждая модель грузится один раз) и собирает report/report.md."""
import argparse
import difflib
import gc
import html
import importlib
import json
import re
import shutil
import time
import traceback
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from PIL import Image

import common
from common import ROOT, load_image

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--only", default="")
ap.add_argument("--out", default="report")
ap.add_argument("--report-only", action="store_true", help="только пересобрать отчёт из raw_results.json")
ap.add_argument("--merge", action="store_true", help="подмешать к сохранённым результатам report/raw_results.json")
a = ap.parse_args()

MD = MarkdownIt("commonmark").enable("table")
SAMPLES, REPORT = ROOT / "samples", ROOT / a.out
IMG_DIR, BOX_DIR = REPORT / "img", REPORT / "boxes"
for d in (IMG_DIR, BOX_DIR):
    d.mkdir(parents=True, exist_ok=True)

CATS = {  # категория -> (заголовок, пояснение)
    "ru_text": ("Русский текст (синтетика)", "чистый текст, эталон рядом"),
    "mixed": ("Смесь: текст, формулы в строках, таблицы (синтетика)", "эталон в Markdown рядом"),
    "formulas_hq": ("Формулы крупно (синтетика)", "эталон LaTeX"),
    "formulas_complex": ("Сложные формулы (пример пользователя)", "эталон LaTeX"),
    "formulas": ("Формулы, мелкие обрезки (LaTeX_OCR)", "малое разрешение, проверка устойчивости"),
    "tables": ("Таблицы, мелкие обрезки (PubTabNet)", "малое разрешение"),
    "documents": ("Документы (DocVQA)", "реальные сканы, английский"),
    "receipts": ("Чеки (SROIE)", "реальные чеки"),
    "scenes_hq": ("Сцены, высокое разрешение (Unsplash)", "в reference.json эталонное описание"),
    "scenes": ("Сцены, COCO 640 px", "малое разрешение"),
}
TEXT_CATS = ["ru_text", "mixed", "formulas_hq", "formulas_complex", "formulas", "tables", "documents", "receipts"]
SCENE_CATS = ["scenes_hq", "scenes"]
# модель -> (категории, доп. контекст)
OCR_KEYS = {"ru_text": ["text"], "documents": ["text"], "receipts": ["text"], "mixed": ["text"],
            "formulas_hq": ["formula"], "formulas_complex": ["formula"], "formulas": ["formula"], "tables": ["table"]}
PLAN = {
    "text_det": CATS.keys(),
    "siglip2": CATS.keys(),
    "glm_ocr": TEXT_CATS,
    "glm_ocr_fixed": TEXT_CATS,
    "paddleocr_vl": TEXT_CATS,
    "qwen3_fix": ["ru_text", "documents", "receipts"],
    "docparse": TEXT_CATS,
    "florence2": SCENE_CATS,
    "scene_event": SCENE_CATS,
    "qwen3vl": SCENE_CATS,
    "faces_insight": SCENE_CATS,
}
# старые модели и Qwen3-VL запускаются только если названы в --only (в конвейере индексации они заменены)
SKIP_DEFAULT = {"glm_ocr", "glm_ocr_fixed", "paddleocr_vl", "qwen3_fix", "qwen3vl"}

models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
only = set(filter(None, a.only.split(",")))

# индекс изображений
items = []  # {cat, name, path}
for cat in CATS:
    for p in sorted((SAMPLES / cat).glob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            items.append({"cat": cat, "name": p.name, "path": p})
for i, it in enumerate(items):
    it["idx"] = i

results = {}  # (model, item idx) -> dict
timing = {}
if a.merge or a.report_only:
    prev_path = REPORT / "raw_results.json"
    by_name = {f"{it['cat']}/{it['name']}": it["idx"] for it in items}
    for key, val in json.loads(prev_path.read_text(encoding="utf-8")).items():
        m, _, nm = key.partition("|")
        if nm in by_name and not (only and m in only and not a.report_only):
            results[(m, by_name[nm])] = val
    tp = REPORT / "timing.json"
    timing = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}


def clear_vlm_cache():
    common.vlm_generate.__defaults__[-1].clear()


import torch
if a.device == "cuda" and not torch.cuda.is_available():
    a.device = "cpu"

def save_raw():
    (REPORT / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORT / "raw_results.json").write_text(
        json.dumps({f"{k[0]}|{items[k[1]]['cat']}/{items[k[1]]['name']}": v for k, v in results.items() if isinstance(k, tuple)},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")


for mid, cats in PLAN.items():
    if a.report_only or (only and mid not in only) or (not only and mid in SKIP_DEFAULT):
        continue
    sel = [it for it in items if it["cat"] in cats]
    imgs = [load_image(it["path"]) for it in sel]
    ctx = {"out": BOX_DIR, "previous": {}, "paths": [str(it["path"]) for it in sel]}
    ctx["progress"] = lambda i, item, mid=mid, sel=sel: print(f"   {mid} {i + 1}/{len(sel)} {sel[i]['cat']}/{sel[i]['name']} {item.get('_sec', '')}", flush=True)
    if mid in ("glm_ocr", "glm_ocr_fixed", "paddleocr_vl"):
        ctx["ocr_keys"] = [OCR_KEYS[it["cat"]] for it in sel]
    if mid == "docparse":  # тип известен классификатором: одиночные формулы и таблицы идут без макета
        ctx["hints"] = ["formula" if it["cat"].startswith("formulas") else "table" if it["cat"] == "tables" else None for it in sel]
    if mid == "scene_event":
        ctx["previous"]["florence2"] = {"images": [results.get(("florence2", it["idx"]), {}) for it in sel]}
    if mid == "qwen3vl":
        ctx["qwen_keys"] = ["describe", "objects_json", "event"]
    if mid == "qwen3_fix":
        # нужен результат glm_ocr по тем же индексам: пересобираем в ctx
        prev = results.get("_glm_raw")
        if not prev:
            print("пропуск qwen3_fix: нет результата glm_ocr")
            continue
        ctx["previous"]["glm_ocr"] = {"images": [prev.get(it["idx"], {}) for it in sel]}
        ctx["fix_idx"] = set(range(len(sel)))
        sel_all = sel
    print(f"== {mid}: {len(sel)} изображений", flush=True)
    t = time.time()
    # боксы нумеруются по порядку внутри вызова, поэтому запоминаем соответствие
    box_names = {}
    try:
        mod = importlib.import_module(f"adapters.{models[mid]['adapter']}")
        res = mod.run(mid, models[mid], imgs, a.device, ctx)
        per = res["images"]
    except Exception:
        (REPORT / f"{mid}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("FAIL", mid, flush=True)
        per = [{"error": "модель не отработала, см. " + f"{mid}.error.txt"}] * len(sel)
    for n, (it, r) in enumerate(zip(sel, per)):
        results[(mid, it["idx"])] = r
        bp = BOX_DIR / f"boxes_{mid}_{n + 1}.png"
        if bp.exists():
            new = BOX_DIR / f"{mid}__{it['cat']}__{Path(it['name']).stem}.png"
            bp.replace(new)
            r["_box"] = new.name
    if mid == "glm_ocr":
        results["_glm_raw"] = {it["idx"]: results[(mid, it["idx"])] for it in sel}
    timing[mid] = {"sec": round(time.time() - t, 1), "n": len(sel), "device": a.device}
    save_raw()  # результаты не теряются, если следующая модель зависнет
    clear_vlm_cache()
    gc.collect()
    if a.device == "cuda":
        torch.cuda.empty_cache()

(REPORT / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
(REPORT / "raw_results.json").write_text(
    json.dumps({f"{k[0]}|{items[k[1]]['cat']}/{items[k[1]]['name']}": v for k, v in results.items() if isinstance(k, tuple)},
               ensure_ascii=False, indent=1, default=str), encoding="utf-8")


# ---------- отчёт ----------
def letters(s):
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", s).lower()


def sim(a_, b_):
    a_, b_ = letters(a_), letters(b_)
    return difflib.SequenceMatcher(None, a_, b_, autojunk=False).ratio() if a_ and b_ else 0.0


def pre(text):
    body = html.escape(str(text).strip()).replace("$", "&#36;")
    body = body.replace(chr(13), "").replace(chr(10), "&#10;")  # пустая строка обрывает HTML-блок в Markdown
    return "<pre>" + body + "</pre>"


def cell(title, body):
    return f"<td valign=\"top\"><b>{html.escape(title)}</b>{body}</td>"


def thumb(it):
    out = IMG_DIR / f"{it['cat']}__{Path(it['name']).stem}.jpg"
    if not out.exists():
        im = Image.open(it["path"]).convert("RGB")
        im.thumbnail((900, 900))
        im.save(out, quality=88)
    return f"img/{out.name}"


def model_cells(it):
    cells, i, cat = [], it["idx"], it["cat"]
    gt_path = it["path"].with_suffix(".md")
    gt = gt_path.read_text(encoding="utf-8") if gt_path.exists() else None
    r = results.get(("text_det", i))
    if r and "error" not in r:
        scripts = {k: v for k, v in r["scripts"].items() if v}
        body = (f"<p>связный текст: <b>{'да' if r['has_text_block'] else 'нет'}</b>, строк {r['lines']}, "
                f"язык: <b>{html.escape(r.get('language', '—'))}</b>, поворот {r['rotation']}°, "
                f"символы по письменностям {scripts}, время {r.get('_sec', '—')} с</p>")
        if r.get("sample"):
            body += "<p><i>пример распознанных строк</i></p>" + pre("\n".join(r["sample"]))
        cells.append(cell("Детектор текста (PP-OCRv5, CPU)", body))
    r = results.get(("siglip2", i))
    if r and "error" not in r:
        tot = sum(p for _, p in r["zero_shot"]) or 1.0
        top = "<br>".join(f"{l}: {p / tot:.0%}" for l, p in r["zero_shot"][:3])
        cells.append(cell("SigLIP2 (тип изображения)", f"<p>{html.escape(top)}</p>"))
    for key, title in (("glm_ocr", "GLM-OCR (первый прогон, 1024 токена, без даунскейла)"),
                       ("glm_ocr_fixed", "GLM-OCR (с правками)"), ("paddleocr_vl", "PaddleOCR-VL (с правками)")):
        r = results.get((key, i))
        if r and "error" not in r:
            body = "".join(f"<p><i>{k}</i></p>{pre(v)}" for k, v in r.items() if not k.startswith("_"))
            if "_sec" in r:
                body += f"<p>время: <b>{r['_sec']} с</b></p>"
            if gt is not None and "text" in r:
                body += f"<p>сходство букв с эталоном: <b>{sim(r['text'], gt):.2f}</b></p>"
            cells.append(cell(title, body))
    r = results.get(("docparse", i))
    if r and r.get("markdown") is not None:
        body = f"<p>время (в пакете): <b>{r.get('_sec', '—')} с</b>, блоков: {len(r.get('blocks', []))}</p>" + MD.render(r["markdown"])
        if gt is not None:
            body += f"<p>сходство букв с эталоном: <b>{sim(r['markdown'], gt):.2f}</b></p>"
        body += "<details><summary>исходный Markdown</summary>" + pre(r["markdown"]) + "</details>"
        cells.append(cell("Разбор страницы (макет, GLM-OCR, PaddleOCR-VL)", body))
    r = results.get(("qwen3_fix", i))
    if r and r.get("after"):
        body = pre(r["after"])
        if gt is not None:
            body += f"<p>сходство букв с эталоном: до {sim(r['before'], gt):.2f}, после <b>{sim(r['after'], gt):.2f}</b></p>"
        cells.append(cell("Qwen3-1.7B (правка)", body))
    r = results.get(("florence2", i))
    if r and "error" not in r:
        cap = r["<MORE_DETAILED_CAPTION>"]["<MORE_DETAILED_CAPTION>"]
        od = r["<OD>"]["<OD>"]["labels"]
        dense = r["<DENSE_REGION_CAPTION>"]["<DENSE_REGION_CAPTION>"]["labels"]
        body = f"<p>{html.escape(cap)}</p><p><i>объекты:</i> {html.escape(', '.join(od))}</p>" \
               f"<p><i>регионы:</i> {html.escape('; '.join(dense[:8]))}</p>"
        if "_box" in r:
            body += f"<img src=\"boxes/{r['_box']}\" width=\"260\">"
        if "_sec" in r:
            body += f"<p>время (в пакете): <b>{r['_sec']} с</b> на изображение</p>"
        cells.append(cell("Florence-2-large", body))
    r = results.get(("scene_event", i))
    if r and "error" not in r and ("summary" in r or "raw" in r):
        body = f"<p><b>{html.escape(str(r.get('event', '')))}</b>: {html.escape(str(r.get('summary', r.get('raw', ''))))}</p>"
        if r.get("tags"):
            body += "<p><i>теги:</i> " + html.escape(", ".join(map(str, r["tags"]))) + "</p>"
        body += f"<p>время (в пакете): <b>{r.get('_sec', '—')} с</b></p>"
        cells.append(cell("Событие (Qwen3-1.7B по описанию Florence-2)", body))
    r = results.get(("qwen3vl", i))
    if r and "error" not in r:
        body = "".join(f"<p><i>{k}</i></p>{pre(v)}" for k, v in r.items())
        cells.append(cell("Qwen3-VL-2B (первый прогон, больше не используется для описаний)", body))
    r = results.get(("faces_insight", i))
    if r and "error" not in r:
        body = f"<p>лиц: <b>{len(r['faces'])}</b></p>"
        if "_box" in r:
            body += f"<img src=\"boxes/{r['_box']}\" width=\"260\">"
        cells.append(cell("Лица (InsightFace)", body))
    return cells, gt


ref_scenes = {}
rj = SAMPLES / "scenes_hq" / "reference.json"
if rj.exists():
    ref_scenes = json.loads(rj.read_text(encoding="utf-8"))

def weights_info(weights_dir):
    """Параметры (по заголовкам safetensors) и размер файлов весов на диске."""
    from safetensors import safe_open
    files = sorted(Path(weights_dir).glob("*.safetensors"))
    params = 0
    for f in files:
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                n = 1
                for d in sf.get_slice(k).get_shape():
                    n *= d
                params += n
    return params, sum(f.stat().st_size for f in files)


def onnx_info(folder):
    files = list(Path(folder).glob("*.onnx"))
    return None, sum(f.stat().st_size for f in files)


# модель -> (название, источник, пик VRAM из замеров или None)
MODEL_TABLE = [
    ("siglip2", "SigLIP2 base (тип изображения, эмбеддинг)", "weights", 0.79),
    ("text_det", "PP-OCRv5: детектор строк + распознаватель (рус/укр/бел/лат) + классификатор угла (текст и язык)", "files", None),
    ("doclayout", "DocLayout-YOLO DocStructBench (макет страницы, CPU)", "files", None),
    ("glm_ocr", "GLM-OCR (текст и формулы внутри строк)", "weights", 2.38),
    ("paddleocr_vl", "PaddleOCR-VL (таблицы, формулы, мультиязычный OCR)", "weights", 2.01),
    ("qwen3_fix", "Qwen3-1.7B (событие сцены по описанию, правка опечаток)", "weights", None),
    ("florence2", "Florence-2-large (описание, объекты, рамки)", "weights", None),
    ("qwen3vl", "Qwen3-VL-2B-Instruct (только вопросы к изображению, в индексации не используется)", "weights", None),
    ("faces_insight", "InsightFace buffalo_l (лица)", "insight", None),
    ("text_embed", "multilingual-e5-small (эмбеддинг описаний)", "weights", None),
]


def model_size_rows():
    rows = ["| модель | параметры | размер весов на диске | пик VRAM, ГБ (замер) |", "|---|---|---|---|"]
    tot = 0
    for mid, title, src, vram in MODEL_TABLE:
        try:
            if src == "weights":
                params, size = weights_info(common.WEIGHTS / models[mid].get("weights", mid))
            elif src == "files":
                import rapidocr_onnxruntime
                cls = Path(rapidocr_onnxruntime.__file__).parent / "models" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
                lists = {"text_det": [common.WEIGHTS / "ppocr_v5_det" / "inference.onnx",
                                      common.WEIGHTS / "ppocr_eslav_rec" / "rec_with_dict.onnx", cls],
                         "doclayout": [common.WEIGHTS / "doclayout_yolo" / "onnx" / "model.onnx"]}
                params, size = None, sum(f.stat().st_size for f in lists[mid] if f.exists())
            else:
                params, size = onnx_info(Path.home() / ".insightface" / "models" / "buffalo_l")
        except Exception:
            params, size = None, 0
        tot += size
        p = f"{params / 1e9:.2f}B" if params and params >= 1e9 else (f"{params / 1e6:.0f}M" if params else "—")
        rows.append(f"| {title} | {p} | {size / 1e9:.2f} ГБ | {vram if vram else '—'} |")
    rows.append(f"| **Итого** | | **{tot / 1e9:.2f} ГБ** | |")
    return rows


lines = ["# Отчёт: сравнение моделей на тестовых изображениях", "",
         "Все модели не больше 3B параметров. Размер весов посчитан по реальным файлам в `model_lab/weights` "
         "(для RapidOCR и InsightFace по ONNX-файлам). Параметры взяты из заголовков safetensors, для ONNX не определялись.", "",
         *model_size_rows(), "",
         "Пик VRAM указан только там, где он был замерян. Для остальных моделей замер даст `bench.py`.", "",
         "**Время работы**", "",
         "| модель | устройство | изображений | всего, с | среднее на изображение, с |", "|---|---|---|---|---|"]
lines += [f"| {k} | {v['device']} | {v['n']} | {v['sec']} | {v['sec'] / v['n']:.2f} |" for k, v in timing.items()]
if "glm_ocr_fixed" in timing or "paddleocr_vl" in timing:
    lines += ["", "**Дополнительные сведения.** Строки `glm_ocr_fixed` и `paddleocr_vl` получены вторым прогоном с правками: лимит ответа по типу "
              "(text 800, formula 256, table 640 токенов), штраф за повторы 1.05, уменьшение страниц до 1280 px, для `mixed` один проход "
              "(промпт text). PaddleOCR-VL загружается встроенным классом transformers без `trust_remote_code`. "
              "Первый прогон `glm_ocr` шёл с лимитом 1024 токена, без даунскейла и тремя проходами для `mixed`."]
lines += ["", "Среднее включает загрузку модели, поэтому для короткого списка оно завышено. Точные замеры CPU и CUDA: раздел «Бенчмарк» ниже."]
lines += ["", "Сходство букв это доля совпадения буквенно-цифровой последовательности с эталоном (грубая метрика, не CER).", ""]
toc = ["## Оглавление", ""] + [f"- [{t}](#{c})" for c, (t, _) in CATS.items()] + [""]
lines += toc
for cat, (title, note) in CATS.items():
    group = [it for it in items if it["cat"] == cat]
    if not group:
        continue
    lines += [f"<a id=\"{cat}\"></a>", "", f"## {title}", "", f"_{note}, изображений: {len(group)}_", ""]
    for it in group:
        cells, gt = model_cells(it)
        lines += [f"### {cat} / {it['name']}", ""]
        if it["name"] in ref_scenes:
            lines += [f"Эталонное описание Unsplash: *{ref_scenes[it['name']]['ai_description']}*", ""]
        row = f"<td valign=\"top\" width=\"280\"><img src=\"{thumb(it)}\" width=\"270\">" + \
              (f"<p><b>Эталон</b></p>{pre(gt)}" if gt else "") + "</td>"
        one_line = ("<table><tr>" + row + "".join(cells) + "</tr></table>").replace(chr(13), " ").replace(chr(10), " ")
        lines += [one_line, ""]
(REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("Отчёт:", REPORT / "report.md")
