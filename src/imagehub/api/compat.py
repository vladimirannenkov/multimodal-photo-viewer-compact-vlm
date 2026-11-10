"""Compatibility API: upload, ask, asynchronous folder batches and text search."""

import io
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import db, profiling, scanner, search
from ..config import get_settings
from ..pipeline import get_pipeline
from ..vision import VisionModel, get_vision_model

router = APIRouter(tags=["vqa"])
_jobs = {}
_jobs_lock = threading.Lock()


class BatchRequest(BaseModel):
    path: str


def read_image(data):
    try:
        with Image.open(io.BytesIO(data)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(400, "Некорректное изображение") from exc


def describe_image(im):
    from ..ml import florence, manager

    with manager.gpu(user=True):
        manager.ensure("scene")
        return florence.describe([im], get_settings().resolve_device(), tasks=["<MORE_DETAILED_CAPTION>"])[0]["caption"]


def ask_image(im, question, model):
    from ..ml import manager

    with manager.gpu(user=True):
        manager.ensure("vqa")
        return model.ask(im, question)


@router.post("/describe")
async def describe(file: UploadFile = File(...)):
    baseline = profiling.snapshot()
    im = read_image(await file.read())
    result = await run_in_threadpool(describe_image, im)
    profiling.save_snapshot("describe", {"images": 1, "description": result}, per_event=True, baseline=baseline)
    return {"description": result}


@router.post("/ask")
async def ask(file: UploadFile = File(...), prompt: str = Form(...), model: VisionModel = Depends(get_vision_model)):
    if not prompt.strip():
        raise HTTPException(400, "Пустой вопрос")
    baseline = profiling.snapshot()
    im = read_image(await file.read())
    result = await run_in_threadpool(ask_image, im, prompt, model)
    profiling.save_snapshot("ask", {"images": 1, "prompt": prompt, "answer": result}, per_event=True, baseline=baseline)
    return {"answer": result}


def update_job(job_id, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def export_batch(job_id, ids):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = get_settings().data_dir / "exports" / job_id
    dest.mkdir(parents=True, exist_ok=True)
    rows = [dict(db.q1("SELECT * FROM images WHERE id=?", (i,))) for i in ids]
    pq.write_table(pa.Table.from_pylist(rows), dest / "results.parquet")
    vectors = [(i, db.q1("SELECT data FROM vectors WHERE image_id=? AND kind='img'", (i,))) for i in ids]
    vectors = [(i, db.blob_to_vec(r["data"]) if r else None) for i, r in vectors]
    vectors = [(i, v) for i, v in vectors if v is not None]
    np.save(dest / "image_ids.npy", np.array([i for i, _ in vectors], dtype=np.int64), allow_pickle=False)
    np.save(
        dest / "embeddings.npy",
        np.stack([v for _, v in vectors]) if vectors else np.empty((0, 0), np.float32),
        allow_pickle=False,
    )
    return str(dest)


def run_batch(job_id, path):
    try:
        baseline = profiling.snapshot()
        update_job(job_id, state="scanning")
        folder = scanner.add_folder(path, watched=False)
        result = scanner.scan_folder(folder, deep=True)
        ids = [r["id"] for r in db.q("SELECT id FROM images WHERE folder_id=? AND deleted=0", (folder,))]
        update_job(job_id, state="processing", total=len(ids), scanned=result["seen"])
        pipe = get_pipeline()
        pipe.start()
        pipe.kick()
        errors = 0
        while ids:
            marks = ",".join("?" for _ in ids)
            rows = db.q(f"SELECT status, error FROM images WHERE id IN ({marks})", ids)
            done = sum(r["status"] in ("done", "error") for r in rows)
            errors = sum(bool(r["error"]) or r["status"] == "error" for r in rows)
            update_job(job_id, done=done, errors=errors, pipeline=pipe.status())
            if done == len(ids):
                break
            time.sleep(1)
        output = export_batch(job_id, ids)
        profiling.save_snapshot(
            "batch", {"id": job_id, "images": len(ids), "errors": errors, "export_dir": output}, per_event=True, baseline=baseline
        )
        update_job(
            job_id, state="completed_with_errors" if errors else "completed", export_dir=output, profile=profiling.snapshot()
        )
    except Exception as exc:
        update_job(job_id, state="error", error=f"{type(exc).__name__}: {exc}")


@router.post("/batch", status_code=202)
def batch(body: BatchRequest):
    path = Path(body.path).resolve()
    if not path.is_dir():
        raise HTTPException(400, "Папка не найдена")
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "state": "queued", "done": 0, "total": 0}
    threading.Thread(target=run_batch, args=(job_id, str(path)), daemon=True).start()
    return {"job_id": job_id, "status_url": f"/batch/{job_id}"}


@router.get("/batch/{job_id}")
def batch_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Задача не найдена")
        return dict(job)


@router.get("/search")
def text_search(q: str, limit: int = 20):
    if not q.strip() or not 1 <= limit <= 200:
        raise HTTPException(400, "Укажите запрос и limit от 1 до 200")
    hits = search.image_text_search(q, limit=limit)
    items = []
    for image_id, score in hits:
        row = db.q1("SELECT id, path, name, caption FROM images WHERE id=? AND deleted=0", (image_id,))
        if row:
            items.append(dict(row, score=score))
    return {"total": len(items), "items": items}


@router.get("/api/profile")
def profile():
    import httpx

    from .. import llm

    try:
        url = llm.effective_base_url().rstrip("/").removesuffix("/v1") + "/api/ps"
        response = httpx.get(url, timeout=2)
        response.raise_for_status()
        external = response.json().get("models", [])
    except Exception:  # The configured API need not implement Ollama's status endpoint.
        external = None
    return {"memory": profiling.memory(), "calls": profiling.snapshot(), "ollama_loaded_models": external}


@router.get("/api/evaluations")
def evaluations():
    import json

    folder = get_settings().data_dir / "evaluations"
    records = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            records.append({"name": path.name, "modified": path.stat().st_mtime, **obj})
        except (OSError, ValueError):
            continue
    return {"evaluations": records}


class ComputeRequest(BaseModel):
    cpu_only: bool


@router.get("/api/settings")
def get_compute_settings():
    settings = get_settings()
    return {
        "cpu_only": settings.cpu_only(),
        "device": get_pipeline().device,
        "cpu_server": "http://127.0.0.1:11435/v1" if settings.cpu_only() else None,
    }


@router.put("/api/settings")
def set_compute_settings(body: ComputeRequest):
    import sys

    from .. import cpu_server
    from ..ml import manager

    pipe = get_pipeline()
    with manager.gpu(user=True):
        if pipe.state.get("stage") not in ("idle", "paused"):
            raise HTTPException(409, "Дождитесь завершения текущей волны обработки и повторите")
        if body.cpu_only:
            try:
                cpu_server.ensure()
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
        manager.free_heavy()
        siglip = sys.modules.get("imagehub.ml.siglip")
        if siglip is not None:
            siglip.free()
        flag = get_settings().data_dir / "cpu-only.flag"
        if body.cpu_only:
            flag.write_text("cpu-only\n", encoding="utf-8")
        else:
            flag.unlink(missing_ok=True)
            cpu_server.stop()
        pipe.device = get_settings().resolve_device()
    return get_compute_settings()
