"""python bench_opt.py  Замеры оптимизированного пайплайна и короткий отчёт report_opt.md (время, устройство, до и после)."""
import difflib
import glob
import json
import re
import statistics as st
import time
from pathlib import Path

import torch

import docparse
import layout
import textfix
from common import ROOT, load_image
from structure import layout_unreliable

SETS = [("ru_text", None), ("mixed", None), ("receipts", None), ("formulas_complex", "formula")]
OUT = ROOT / "report_opt.md"


def letters(s):
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", s).lower()


def sim(a, b):
    return difflib.SequenceMatcher(None, letters(a), letters(b), autojunk=False).ratio()


def sync():
    torch.cuda.synchronize()


rows, all_blocks = [], []
raw = json.loads((ROOT / "report" / "raw_results.json").read_text(encoding="utf-8"))
gpu = torch.cuda.get_device_name(0)

for cat, hint in SETS:
    paths = sorted(p for p in glob.glob(str(ROOT / "samples" / cat / "*")) if p.lower().endswith((".png", ".jpg")))
    imgs = [load_image(p) for p in paths]
    hints = [hint] * len(imgs) if hint else None
    docparse.parse_pages(imgs[:1], hints=hints[:1] if hints else None, fix=False)  # прогрев OCR-моделей
    sync(); t = time.time()
    plain = docparse.parse_pages(imgs, hints=hints, fix=False)
    sync(); t_plain = time.time() - t
    sync(); t = time.time()
    fixed = docparse.parse_pages(imgs, hints=hints, fix=True)
    sync(); t_fixed = time.time() - t
    gts = [Path(p).with_suffix(".md") for p in paths]
    sp = [sim(r["markdown"], g.read_text(encoding="utf-8")) for r, g in zip(plain, gts) if g.exists()]
    sf = [sim(r["markdown"], g.read_text(encoding="utf-8")) for r, g in zip(fixed, gts) if g.exists()]
    before = [v["_sec"] for k, v in raw.items() if k.startswith(f"glm_ocr_fixed|{cat}/") and "_sec" in v]
    all_blocks += [b["text"] for r in plain for b in r["blocks"] if b.get("kind") == "text" and len(b.get("text", "")) >= 15]
    rows.append({
        "cat": cat, "n": len(imgs), "before": st.mean(before) if before else None,
        "layout": plain[0]["sec_layout"], "ocr": plain[0]["sec_ocr"], "plain_total": t_plain / len(imgs),
        "fix_total": t_fixed / len(imgs), "fix_sec": fixed[0]["sec_fix"], "fix_stats": fixed[0]["fix"],
        "sim_plain": st.mean(sp) if sp else None, "sim_fixed": st.mean(sf) if sf else None})
    print("RESULT", cat, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

# правка опечаток: по одному блоку и пакетами, на одних и тех же блоках
tf = {}
if all_blocks:
    textfix.correct_texts(all_blocks[:2], batch=2)  # загрузка и прогрев
    for b in (1, 8):
        sync(); t = time.time()
        res, stats = textfix.correct_texts(all_blocks, batch=b)
        sync(); tf[b] = (time.time() - t, stats)
    print("RESULT textfix", {b: (round(v[0], 1), v[1]) for b, v in tf.items()}, flush=True)
    textfix.free()

# компоненты вне разбора страниц
timing = json.loads((ROOT / "report" / "timing.json").read_text(encoding="utf-8"))
det = [v["_sec"] for k, v in raw.items() if k.startswith("text_det|") and "_sec" in v]
flo = [v["_sec"] for k, v in raw.items() if k.startswith("florence2|") and "_sec" in v]
evt = [v["_sec"] for k, v in raw.items() if k.startswith("scene_event|") and "_sec" in v]

L = ["# Оптимизированный пайплайн: время работы", "",
     f"GPU: {gpu}. Время на изображение или страницу в секундах, замер после прогрева, модели уже загружены. "
     "Разбор страниц идёт пакетами по 8 блоков, макет и детектор текста на CPU.", "",
     "## Разбор страниц (макет, GLM-OCR, PaddleOCR-VL)", "",
     "| набор | страниц | было (GLM-OCR, 1 проход) | макет, CPU | OCR, GPU | итого без правки | итого с правкой |",
     "|---|---|---|---|---|---|---|"]
for r in rows:
    b = f"{r['before']:.1f}" if r["before"] else "—"
    L.append(f"| {r['cat']} | {r['n']} | {b} | {r['layout']:.2f} | {r['ocr']:.2f} | {r['plain_total']:.1f} | {r['fix_total']:.1f} |")
L += ["", "«Было» это среднее по первому прогону GLM-OCR с правками (`glm_ocr_fixed`): страница целиком, по одной, без пакетов.", "",
      "## Качество и правка опечаток (Qwen3-1.7B)", "",
      "| набор | сходство букв с эталоном, без правки | с правкой | блоков | изменено | отклонено защитой |", "|---|---|---|---|---|---|"]
for r in rows:
    fs = r["fix_stats"] or {}
    sp = f"{r['sim_plain']:.3f}" if r["sim_plain"] else "—"
    sf = f"{r['sim_fixed']:.3f}" if r["sim_fixed"] else "—"
    L.append(f"| {r['cat']} | {sp} | {sf} | {fs.get('blocks', 0)} | {fs.get('changed', 0)} | {fs.get('rejected', 0)} |")
if tf:
    n = len(all_blocks)
    L += ["", f"Скорость правки на {n} текстовых блоках: по одному {tf[1][0]:.1f} с ({tf[1][0] / n:.2f} с на блок), "
          f"пакетами по 8: **{tf[8][0]:.1f} с ({tf[8][0] / n:.2f} с на блок)**, ускорение в {tf[1][0] / tf[8][0]:.1f} раза."]
L += ["", "## Остальные модели", "",
      "| шаг | устройство | было, с | стало, с | что изменено |", "|---|---|---|---|---|",
      f"| детектор текста и язык (PP-OCRv5) | CPU | 3–4 | {st.mean(det):.2f} | многопоточные сессии ONNX, повороты только при необходимости, 4 строки для языка |",
      f"| Florence-2-large (описание, объекты, регионы) | GPU | 3.09 | {st.mean(flo):.2f} | пакеты по 8, sdpa |",
      f"| событие сцены | GPU | 3.47 (Qwen3-1.7B) | {st.mean(evt):.3f} (SigLIP2) | zero-shot по эмбеддингу вместо LLM |",
      "| макет страницы (DocLayout-YOLO) | CPU | — | 0.8 | новый шаг, режет страницу на блоки |", "",
      "## Что дало ускорение разбора страниц", "",
      "- Включён кэш генерации (у PaddleOCR-VL он был выключен: в 7 раз медленнее).",
      "- Блоки страницы читаются пакетами, а не по одному; длинные страницы и чеки режутся на полосы.",
      "- Таблицы и формулы вне строк идут в PaddleOCR-VL (быстрее и точнее на русском), текст с формулами внутри строк в GLM-OCR.",
      "- Длина ответа ограничена по типу блока, изображения уменьшаются до 1280 px."]
OUT.write_text("\n".join(L), encoding="utf-8")
print("Отчёт:", OUT)
