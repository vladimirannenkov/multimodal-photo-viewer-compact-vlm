"""python bench.py [--n 3] [--cpu text_det,siglip2,faces_insight,florence2]
Точные замеры: среднее время на изображение (после прогрева, без загрузки модели) на CUDA и на CPU.
Пишет report/bench.md и вставляет таблицу в report/report.md."""
import argparse
import gc
import importlib
import time
import warnings

import torch
import yaml

import common
from common import ROOT, load_image

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--cpu", default="text_det,siglip2,faces_insight,florence2")
a = ap.parse_args()
cpu_models = set(a.cpu.split(","))

models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
S = ROOT / "samples"
# модель -> (категория для замера, контекст)
SETUP = {
    "text_det": ("mixed", {}),
    "siglip2": ("scenes_hq", {}),
    "paddleocr_vl": ("ru_text", {"ocr_keys": [["text"]] * 64}),
    "glm_ocr": ("ru_text", {"ocr_keys": [["text"]] * 64}),
    "florence2": ("scenes_hq", {}),
    "qwen3vl": ("scenes_hq", {"qwen_keys": ["describe"]}),
    "faces_insight": ("scenes_hq", {}),
}


def paths(cat):
    return sorted(p for p in (S / cat).glob("*") if p.suffix.lower() in (".png", ".jpg"))[: a.n + 1]


rows = []
for mid, (cat, extra) in SETUP.items():
    imgs = [load_image(p) for p in paths(cat)]
    for device in ("cuda", "cpu"):
        if device == "cpu" and mid not in cpu_models:
            rows.append((mid, device, None, "не измерялось (слишком долго на CPU)"))
            continue
        ctx = {"out": ROOT / "report" / "boxes", "previous": {}, **extra}
        mod = importlib.import_module(f"adapters.{models[mid]['adapter']}")
        try:
            mod.run(mid, models[mid], imgs[:1], device, ctx)  # прогрев
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            mod.run(mid, models[mid], imgs[1:], device, ctx)
            if device == "cuda":
                torch.cuda.synchronize()
            rows.append((mid, device, (time.time() - t) / len(imgs[1:]), ""))
        except Exception as e:
            rows.append((mid, device, None, f"{type(e).__name__}: {str(e)[:80]}"))
        common.vlm_generate.__defaults__[-1].clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(rows[-1], flush=True)

lines = ["## Бенчмарк: среднее время на изображение", "",
         f"Замер после прогрева, без загрузки модели, {a.n} изображения. GPU: {torch.cuda.get_device_name(0)}.", "",
         "| модель | устройство | среднее на изображение, с | примечание |", "|---|---|---|---|"]
lines += [f"| {m} | {d} | {'' if t is None else f'{t:.2f}'} | {note} |" for m, d, t, note in rows]
block = "\n".join(lines) + "\n"
(ROOT / "report" / "bench.md").write_text(block, encoding="utf-8")
rp = ROOT / "report" / "report.md"
if rp.exists():
    txt = rp.read_text(encoding="utf-8")
    head, _, rest = txt.partition("\n")
    rp.write_text(head + "\n\n" + block + "\n" + rest, encoding="utf-8")
print("OK")
