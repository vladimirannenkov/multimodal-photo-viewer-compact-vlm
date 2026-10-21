"""Каталог: список и карточка изображения, файлы и миниатюры, папки, импорт, очередь обработки."""

import json
import os
import string
import time
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .. import db, queries, scanner, search
from ..config import get_settings
from ..pipeline import get_pipeline
from .deps import image_row, viewable_path

router = APIRouter(prefix="/api")


def _filters(**kw):
    return {k: v for k, v in kw.items() if v not in (None, "", False)}


@router.get("/sidebar")
def sidebar():
    return queries.sidebar()


@router.get("/images")
def images(
    folder: int | None = None,
    collection: int | None = None,
    category: int | None = None,
    experiment: int | None = None,
    person: int | None = None,
    kind: str | None = None,
    favorite: bool = False,
    clipboard: bool = False,
    has_text: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
    sort: str = "date",
    desc: bool = True,
    offset: int = 0,
    limit: int = Query(300, le=2000),
):
    f = _filters(
        folder=folder,
        collection=collection,
        category=category,
        experiment=experiment,
        person=person,
        kind=kind,
        favorite=favorite,
        clipboard=clipboard,
        has_text=has_text,
        date_from=date_from,
        date_to=date_to,
    )
    total, items = queries.list_images(f, sort, desc, offset, limit)
    return {"total": total, "items": items}


@router.get("/images/{image_id}")
def detail(image_id: int):
    d = queries.image_detail(image_id)
    if d is None:
        raise HTTPException(404)
    return d


@router.get("/images/{image_id}/file")
def file(image_id: int, download: bool = False):
    r = image_row(image_id)
    p, mt = viewable_path(r)
    return FileResponse(p, media_type=mt, filename=r["name"] if download else None)


@router.get("/images/{image_id}/original")
def original(image_id: int):
    r = image_row(image_id)
    return FileResponse(r["path"], filename=r["name"])


@router.get("/images/{image_id}/thumb")
def thumb(image_id: int):
    p = scanner.thumb_path(image_id)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


@router.post("/images/{image_id}/prioritize")
def prioritize(image_id: int):
    db.ex("UPDATE images SET priority=10 WHERE id=? AND status IN ('new','queued')", (image_id,))
    get_pipeline().kick()
    return {"ok": True}


@router.patch("/images/{image_id}")
def patch(image_id: int, body: dict = Body(...)):
    r = image_row(image_id)
    sets, args = [], []
    for k in ("favorite", "rating", "experiment_id"):
        if k in body:
            sets.append(f"{k}=?")
            args.append(body[k])
    if "name" in body and body["name"] and body["name"] != r["name"]:
        new = Path(r["path"]).with_name(body["name"])
        if new.exists():
            raise HTTPException(409, "файл с таким именем уже есть")
        if any(c in body["name"] for c in '\\/:*?"<>|'):
            raise HTTPException(400, "недопустимые символы в имени")
        os.rename(r["path"], new)
        sets += ["name=?", "path=?"]
        args += [body["name"], str(new)]
        db.ex(
            "INSERT INTO undo_log(ts, action, payload) VALUES(?,?,?)",
            (time.time(), "rename", json.dumps({"id": image_id, "old": r["path"], "new": str(new)})),
        )
    if sets:
        db.ex(f"UPDATE images SET {', '.join(sets)} WHERE id=?", args + [image_id])
        db.fts_update(image_id)
    return {"ok": True}


@router.post("/images/delete")
def delete(body: dict = Body(...)):
    ids = body.get("ids", [])
    remove_file = bool(body.get("remove_file"))
    for i in ids:
        r = db.q1("SELECT path FROM images WHERE id=?", (i,))
        if r and remove_file and os.path.exists(r["path"]):
            try:
                os.remove(r["path"])
            except OSError:
                pass
        db.ex("UPDATE images SET deleted=1 WHERE id=?", (i,))
    db.ex(
        "INSERT INTO undo_log(ts, action, payload) VALUES(?,?,?)",
        (time.time(), "delete", json.dumps({"ids": ids, "file": remove_file})),
    )
    return {"deleted": len(ids)}


@router.post("/images/restore")
def restore(body: dict = Body(...)):
    for i in body.get("ids", []):
        db.ex("UPDATE images SET deleted=0 WHERE id=?", (i,))
    return {"ok": True}


# ---------- папки и импорт ----------
@router.post("/import/folder")
def import_folder(body: dict = Body(...)):
    path = body.get("path", "")
    if not os.path.isdir(path):
        raise HTTPException(400, "папка не найдена")
    fid = scanner.add_folder(path, watched=body.get("watched", True))
    res = scanner.scan_folder(fid, deep=True)
    from .. import watcher

    watcher.start()
    get_pipeline().kick()
    res.pop("ids", None)
    return {**res, "folder_id": fid}


@router.post("/import/paths")
def import_paths(body: dict = Body(...)):
    ids = scanner.import_files([p for p in body.get("paths", []) if os.path.isfile(p)])
    get_pipeline().kick()
    return {"ids": ids}


@router.post("/import/upload")
async def upload(files: list[UploadFile] = File(...), clipboard: bool = Form(False)):
    """Перетаскивание файлов и вставка из буфера: файл сохраняется в папку «Изображения\\ImageHub»."""
    s = get_settings()
    dest = s.pictures_dir / ("clipboard" if clipboard else "uploads")
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower() or ".png"
        name = f"clip_{time.strftime('%Y%m%d_%H%M%S')}_{len(saved)}{ext}" if clipboard or not f.filename else f.filename
        p = dest / name
        n = 1
        while p.exists():
            p = dest / f"{Path(name).stem}_{n}{ext}"
            n += 1
        p.write_bytes(await f.read())
        saved.append(str(p))
    folder_id = scanner.add_folder(str(dest), watched=False)
    ids = scanner.import_files(saved, folder_id=folder_id, clipboard=clipboard)
    for i in ids:
        db.ex("UPDATE images SET priority=10 WHERE id=?", (i,))
    get_pipeline().kick()
    return {"ids": ids}


@router.post("/scan")
def rescan(body: dict = Body(default={})):
    res = scanner.scan_all(deep=bool(body.get("deep")))
    for r in res:
        r.pop("ids", None)
    get_pipeline().kick()
    return res


@router.delete("/folders/{folder_id}")
def remove_folder(folder_id: int):
    db.ex("UPDATE images SET deleted=1 WHERE folder_id=?", (folder_id,))
    db.ex("DELETE FROM folders WHERE id=?", (folder_id,))
    from .. import watcher

    watcher.start()
    return {"ok": True}


@router.get("/fs/browse")
def browse(path: str = ""):
    """Обзор каталогов сервера (локальный сервис): выбор папки для импорта."""
    if not path:
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")] if os.name == "nt" else ["/"]
        return {"path": "", "parent": None, "dirs": drives, "images": 0}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(400, "не папка")
    dirs, n = [], 0
    try:
        for e in os.scandir(p):
            if e.is_dir() and not e.name.startswith((".", "$")):
                dirs.append(e.name)
            elif e.is_file():
                n += e.name.lower().endswith(tuple(__import__("imagehub.imaging", fromlist=["EXTS"]).EXTS))
    except OSError:
        pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else "", "dirs": sorted(dirs, key=str.lower), "images": n}


# ---------- очередь ----------
@router.get("/queue")
def queue():
    return get_pipeline().status()


@router.post("/queue/{action}")
def queue_action(action: str):
    p = get_pipeline()
    if action == "pause":
        p.paused = True
    elif action == "resume":
        p.paused = False
        p.kick()
    else:
        raise HTTPException(400)
    return p.status()


@router.post("/reprocess")
def reprocess(body: dict = Body(...)):
    ids = body.get("ids") or [r["id"] for r in db.q("SELECT id FROM images WHERE deleted=0")]
    db.exmany("UPDATE images SET status='new', stages='{}' WHERE id=?", [(i,) for i in ids])
    get_pipeline().kick()
    return {"queued": len(ids)}


# ---------- поиск ----------
@router.post("/search")
async def do_search(
    q: str = Form(""),
    weight: float = Form(0.5),
    filters: str = Form("{}"),
    limit: int = Form(200),
    image: UploadFile | None = File(None),
    similar_to: int | None = Form(None),
):
    f = json.loads(filters or "{}")
    vec = None
    if image is not None:
        import io

        from PIL import Image

        from ..ml import siglip

        im = Image.open(io.BytesIO(await image.read())).convert("RGB")
        vec = siglip.get("cpu").images([im])[0]
    if similar_to:
        hits = search.similar(similar_to, limit, f or None)
    else:
        hits = search.search(q, vec, weight, f or None, limit)
    if not hits:
        return {"total": 0, "items": []}
    order = {i: (n, s) for n, (i, s) in enumerate(hits)}
    ids = list(order)
    rows = db.q(f"SELECT {queries.LIST_COLS} FROM images i WHERE i.id IN ({','.join('?' * len(ids))}) AND i.deleted=0", ids)
    items = sorted((dict(r, score=order[r["id"]][1]) for r in rows), key=lambda r: order[r["id"]][0])
    return {"total": len(items), "items": items}


@router.get("/duplicates")
def duplicates(threshold: float = 0.97):
    groups = search.duplicates(threshold)
    for g in groups:
        g["items"] = [
            dict(r)
            for r in db.q(
                f"SELECT id, name, width, height, size, path FROM images WHERE id IN ({','.join('?' * len(g['ids']))})", g["ids"]
            )
        ]
    return groups
