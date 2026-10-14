"""python download.py [--only id,id] [--yes]  Скачивает веса в weights/<id>. Без --yes только показывает размеры."""
import argparse
import shutil
import yaml
from huggingface_hub import HfApi, snapshot_download
from common import ROOT, WEIGHTS

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
only = set(filter(None, a.only.split(",")))
todo = {k: v for k, v in models.items() if v.get("repo") and (not only or k in only)}

api, total = HfApi(), 0
for k, v in todo.items():
    info = api.model_info(v["repo"], files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin", ".json", ".txt", ".model", ".py")))
    total += size
    print(f"{k:16} {v['repo']:45} {size / 1e9:6.2f} GB")
free = shutil.disk_usage(ROOT).free / 1e9
print(f"ИТОГО ~{total / 1e9:.1f} GB, свободно {free:.0f} GB")

if a.yes:
    for k, v in todo.items():
        snapshot_download(v["repo"], local_dir=WEIGHTS / k)
        print("OK", k)
else:
    print("Добавьте --yes для скачивания")
