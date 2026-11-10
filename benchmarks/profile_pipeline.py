"""Profile the actual ImageHub pipeline on a folder in an isolated database."""

import argparse
import json
import os
import platform
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["IMAGEHUB_DATA_DIR"] = str(Path(args.data_dir).resolve())
    os.environ["IMAGEHUB_AUTOSTART_PIPELINE"] = "false"
    from imagehub import db, profiling, scanner
    from imagehub.pipeline import WAVE, Pipeline

    start = time.perf_counter()
    folder_id = scanner.add_folder(str(Path(args.folder).resolve()), watched=False)
    scan_start = time.perf_counter()
    scanner.scan_folder(folder_id, deep=True)
    scan_seconds = time.perf_counter() - scan_start
    ids = [r["id"] for r in db.q("SELECT id FROM images WHERE folder_id=? ORDER BY id", (folder_id,))]
    pipe = Pipeline()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for start_idx in range(0, len(ids), WAVE):
        pipe.run_wave(ids[start_idx : start_idx + WAVE])
        try:
            import httpx

            url = pipe.s.llm_base_url.rstrip("/").removesuffix("/v1") + "/api/ps"
            response = httpx.get(url, timeout=2)
            response.raise_for_status()
            external_models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError):
            external_models = None
        counts = dict(
            db.q1(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors FROM images WHERE status IN ('done','error')"
            )
        )
        obj = {
            "summary": {
                "kind": "pipeline-profile",
                "status": "completed" if start_idx + WAVE >= len(ids) else "running",
                "folder": str(Path(args.folder).resolve()),
                "device": pipe.device,
                "platform": platform.platform(),
                "llm_model": pipe.s.llm_model,
                "ocr_model": pipe.s.ocr_model or "local GLM-OCR",
                "images": len(ids),
                "processed": counts["total"],
                "images_with_errors": counts["errors"],
                "scan_seconds": scan_seconds,
                "elapsed_seconds": time.perf_counter() - start,
            },
            "profile": profiling.snapshot(),
            "memory": profiling.memory(),
            "ollama_loaded_models": external_models,
            "errors": [dict(r) for r in db.q("SELECT id, name, error FROM images WHERE error IS NOT NULL")],
        }
        output.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(obj["summary"], ensure_ascii=False), flush=True)  # noqa: T201
    scanner.shutdown_pool()
    if not ids or counts["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
