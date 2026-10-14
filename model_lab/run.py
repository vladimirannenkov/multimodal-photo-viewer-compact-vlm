"""python run.py <img> [<img2>] [--only id|group,...] [--device cuda|cpu]"""
import argparse
import gc
import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from common import ROOT, load_image

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--only", default="")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
only = set(filter(None, a.only.split(",")))
sel = {k: v for k, v in models.items() if not only or k in only or v["group"] in only}

import torch
if a.device == "cuda" and not torch.cuda.is_available():
    print("CUDA недоступна, переход на cpu")
    a.device = "cpu"

name = Path(a.images[0]).stem
out = ROOT / "results" / f"{name}__{datetime.now():%Y%m%d_%H%M%S}"
out.mkdir(parents=True)
imgs = [load_image(p) for p in a.images]
ctx = {"paths": a.images, "out": out, "previous": {}}
rows = []

for mid, cfg in sel.items():
    print(f"== {mid}", flush=True)
    row = {"id": mid, "status": "ok", "sec": 0.0, "vram_gb": 0.0, "error": ""}
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        mod = importlib.import_module(f"adapters.{cfg['adapter']}")
        res = mod.run(mid, cfg, imgs, a.device, ctx)
        ctx["previous"][mid] = res
        (out / f"{mid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / f"{mid}.md").write_text(res.get("markdown") or json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        row["status"] = "FAIL"
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        (out / f"{mid}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    row["sec"] = round(time.time() - t, 1)
    if a.device == "cuda":
        row["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rows.append(row)
    gc.collect()
    if a.device == "cuda":
        torch.cuda.empty_cache()

lines = [f"# {name}", "", "| модель | статус | сек (с загрузкой) | VRAM ГБ | ошибка |", "|---|---|---|---|---|"]
lines += [f"| {r['id']} | {r['status']} | {r['sec']} | {r['vram_gb']} | {r['error']} |" for r in rows]
(out / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("Результаты:", out)
