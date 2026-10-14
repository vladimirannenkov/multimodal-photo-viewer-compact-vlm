"""Чат с VLM по изображению (Qwen3-VL-2B)."""
import time

import queue
import threading

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse

from .. import db
from ..config import get_settings
from ..ml import manager, vqa
from .deps import image_row, pil_open

router = APIRouter(prefix="/api")


@router.get("/images/{image_id}/chat")
def history(image_id: int):
    return [dict(r) for r in db.q("SELECT id, role, content, ts FROM chat WHERE image_id=? ORDER BY id", (image_id,))]


@router.delete("/images/{image_id}/chat")
def clear(image_id: int):
    db.ex("DELETE FROM chat WHERE image_id=?", (image_id,))
    return {"ok": True}


@router.post("/images/{image_id}/chat")
def ask(image_id: int, body: dict = Body(...)):
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "пустой вопрос")
    r = image_row(image_id)
    hist = [dict(x) for x in db.q("SELECT role, content FROM chat WHERE image_id=? ORDER BY id", (image_id,))]
    hint = ""
    if r["ocr_md"] and len(r["ocr_md"]) > 20:
        hint = "\n\nРаспознанный текст на изображении (может содержать ошибки):\n" + r["ocr_md"][:1500]
    question = msg + hint if not hist else msg
    img = pil_open(r)
    t0 = time.time()
    with manager.gpu(user=True):
        manager.ensure("vqa")
        answer = vqa.ask(img, question, get_settings().resolve_device(), history=hist)
    db.ex("INSERT INTO chat(image_id, role, content, ts) VALUES(?,?,?,?)", (image_id, "user", msg, time.time()))
    db.ex("INSERT INTO chat(image_id, role, content, ts) VALUES(?,?,?,?)", (image_id, "assistant", answer, time.time()))
    return {"answer": answer, "sec": round(time.time() - t0, 1)}


@router.post("/chat/warmup")
def warmup():
    """Загружает Qwen3-VL заранее, пока пользователь набирает вопрос."""
    with manager.gpu(user=True):
        manager.ensure("vqa")
        vqa.warmup(get_settings().resolve_device())
    return {"ok": True}


@router.post("/images/{image_id}/chat/stream")
def ask_stream(image_id: int, body: dict = Body(...)):
    """Ответ потоком: текст появляется по мере генерации, пользователь не ждёт конца."""
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "пустой вопрос")
    r = image_row(image_id)
    hist = [dict(x) for x in db.q("SELECT role, content FROM chat WHERE image_id=? ORDER BY id", (image_id,))]
    hint = ""
    if r["ocr_md"] and len(r["ocr_md"]) > 20 and not hist:
        hint = chr(10) * 2 + "Распознанный текст на изображении (может содержать ошибки):" + chr(10) + r["ocr_md"][:1500]
    img = pil_open(r)
    q: queue.Queue = queue.Queue()

    def worker():
        parts = []
        try:
            with manager.gpu(user=True):
                manager.ensure("vqa")
                for chunk in vqa.ask_stream(img, msg + hint, get_settings().resolve_device(), history=hist):
                    parts.append(chunk)
                    q.put(chunk)
        except Exception as e:
            q.put(f"[ошибка: {e}]")
        finally:
            ans = "".join(parts).strip()
            if ans:
                db.ex("INSERT INTO chat(image_id, role, content, ts) VALUES(?,?,?,?)", (image_id, "user", msg, time.time()))
                db.ex("INSERT INTO chat(image_id, role, content, ts) VALUES(?,?,?,?)", (image_id, "assistant", ans, time.time()))
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while (c := q.get()) is not None:
            yield c
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
