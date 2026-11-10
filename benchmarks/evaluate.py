"""Small reproducible dataset evaluation; see benchmarks/README.md."""

import argparse
import ast
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

DATASETS = {
    "caption": ("nlphuji/flickr30k", "TEST", "test"),
    "ocr": ("mychen76/invoices-and-receipts_ocr_v1", "default", "test"),
}


def get(url, **kwargs):
    for attempt in range(5):
        try:
            response = httpx.get(url, timeout=90, follow_redirects=True, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def rows(kind, offset, count):
    dataset, config, split = DATASETS[kind]
    response = get(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": count,
        },
    )
    return response.json()["rows"]


def image(row):
    from imagehub.profiling import measure

    source = row["image"]
    data = source["bytes"] if source.get("bytes") is not None else get(source["src"]).content
    with measure("image.decode"):
        im = Image.open(io.BytesIO(data))
        im.load()
    with measure("image.preprocess"):
        return ImageOps.exif_transpose(im).convert("RGB")


def gold(row):
    parsed = json.loads(row["parsed_data"])["json"]
    data = ast.literal_eval(parsed) if isinstance(parsed, str) else parsed
    return data.get("header", {}).get("invoice_date"), data.get("summary", {}).get("total_gross_worth")


def date_values(text):
    result = set()
    for token in re.findall(r"(?<!\d)\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)", str(text)):
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%y", "%d/%m/%y"):
            try:
                result.add(datetime.strptime(token.replace("-", "/").replace(".", "/"), fmt).date().isoformat())
            except ValueError:
                pass
    return result


def amounts(text):
    found = set()
    for token in re.findall(r"(?<![\d/])\d[\d,. ]*[,.]\d{2}(?![\d/])", str(text)):
        token = token.strip().replace(" ", "")
        value = re.sub(r"[^\d]", "", token)
        if value:
            found.add(int(value))
    return found


def extract_fields(text):
    """Extract the issue date and the final amount from the labelled total row."""
    label = re.search(r"\b(?:date of issue|invoice date|date)\b", text, re.I)
    candidates = (
        re.findall(r"(?<!\d)\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)", text[label.end() : label.end() + 1000]) if label else []
    )
    date = next((candidate for candidate in candidates if date_values(candidate)), None)
    total = None
    for line in text.splitlines():
        if re.search(r"\btotal\b", line, re.I):
            numbers = re.findall(r"(?<![\d/])\d[\d,.]*[,.]\d{2}(?![\d/])", line)
            if numbers:
                total = numbers[-1]
    return date, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=DATASETS)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", help="Local dataset parquet file")
    ap.add_argument("--dataset-id", help="Dataset identifier and revision for local Parquet")
    args = ap.parse_args()
    if not 1 <= args.count <= 100 or args.offset < 0:
        ap.error("count must be 1..100 and offset nonnegative")
    os.environ.setdefault("IMAGEHUB_OCR_MODEL", "")
    from imagehub import profiling
    from imagehub.ml import docparse, florence, siglip

    started = time.perf_counter()
    records = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.parquet:
        import pyarrow.parquet as pq

        table = pq.read_table(args.parquet).slice(args.offset, args.count)
        samples = [{"row_idx": args.offset + i, "row": row} for i, row in enumerate(table.to_pylist())]
    else:
        samples = rows(args.kind, args.offset, args.count)
    for sample in samples:
        r = sample["row"]
        entry = {"row_idx": sample["row_idx"]}
        try:
            im = image(r)
            start = time.perf_counter()
            if args.kind == "caption":
                caption = florence.describe([im], args.device, tasks=["<MORE_DETAILED_CAPTION>"])[0]["caption"]
                encoder = siglip.get(args.device)
                vec = encoder.text([caption] + r["caption"])
                similarities = vec[1:] @ vec[0]
                entry.update(
                    filename=r["filename"],
                    original_split=r.get("split", "unknown"),
                    caption=caption,
                    references=r["caption"],
                    cosine_mean=float(np.mean(similarities)),
                    cosine_max=float(np.max(similarities)),
                )
            else:
                result = docparse.parse_pages([im], device=args.device, fix=False)[0]
                output = result["markdown"]
                date, total = gold(r)
                dv, av = date_values(date), amounts(total)
                pred_date, pred_total = extract_fields(output)
                entry.update(
                    id=r["id"],
                    ocr=output,
                    expected_date=date,
                    expected_total=total,
                    date_found=bool(dv & date_values(output)) if dv else None,
                    total_found=bool(av & amounts(output)) if av else None,
                    predicted_date=pred_date,
                    predicted_total=pred_total,
                    date_correct=bool(dv & date_values(pred_date)) if dv else None,
                    total_correct=bool(av & amounts(pred_total)) if av else None,
                )
            entry["seconds"] = round(time.perf_counter() - start, 2)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        records.append(entry)
        partial = {
            "summary": {"status": "running", "requested": args.count, "processed": len(records)},
            "records": records,
            "profile": profiling.snapshot(),
            "memory": profiling.memory(),
        }
        out.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"row_idx": entry["row_idx"], "error": entry.get("error")}, ensure_ascii=True), flush=True)  # noqa: T201
    ok = [r for r in records if "error" not in r]
    summary = {
        "dataset": DATASETS[args.kind],
        "offset": args.offset,
        "requested": args.count,
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "missing": args.count - len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    if args.parquet:
        summary["dataset"] = args.dataset_id or "local Parquet; dataset identity not specified"
        summary["source_parquet"] = str(Path(args.parquet).resolve())
    if args.kind == "caption" and ok:
        summary["mean_cosine"] = float(np.mean([r["cosine_mean"] for r in ok]))
        summary["mean_max_cosine"] = float(np.mean([r["cosine_max"] for r in ok]))
    if args.kind == "ocr":
        for key in ("date_found", "total_found", "date_correct", "total_correct"):
            vals = [r[key] for r in ok if r[key] is not None]
            metric = "accuracy" if key.endswith("_correct") else "recall"
            summary[key] = {"matched": sum(vals), "eligible": len(vals), metric: sum(vals) / len(vals) if vals else None}
    out.write_text(
        json.dumps(
            {"summary": summary, "records": records, "profile": profiling.snapshot(), "memory": profiling.memory()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)  # noqa: T201
    if len(ok) != args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
