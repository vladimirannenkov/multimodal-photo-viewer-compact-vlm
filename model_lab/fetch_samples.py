"""python fetch_samples.py  Скачивает небольшие тестовые наборы в samples/<категория>/ (стриминг, первые N)."""
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).parent / "samples"

# категория, датасет HF, split, поле-изображение, сколько, доп. аргументы
SETS = [
    ("formulas", "unsloth/LaTeX_OCR", "test", "image", 8, {}),
    ("tables", "docling-project/PubTabNet_OTSL", "val", "image", 8, {}),
    ("documents", "nielsr/docvqa_1200_examples", "train", "image", 6, {}),
    ("receipts", "jsdnrs/ICDAR2019-SROIE", "test", "image", 5, {}),
    ("scenes", "detection-datasets/coco", "val", "image", 15, {}),
]

import sys
only=set(sys.argv[1:])
for cat, repo, split, field, n, kw in SETS:
    if only and cat not in only:
        continue
    out = ROOT / cat
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(repo, split=split, streaming=True, **kw)
        for i, row in enumerate(ds):
            if i >= n:
                break
            img = row[field].convert("RGB")
            img.thumbnail((1600, 1600))
            img.save(out / f"{cat}_{i + 1:02d}.jpg", quality=90)
        print(f"OK   {cat:10} {repo}")
    except Exception as e:
        print(f"FAIL {cat:10} {repo}: {type(e).__name__}: {str(e)[:150]}")
