"""Категории (теги, у изображения их несколько), коллекции, эксперименты."""
import json
import time

from fastapi import APIRouter, Body, HTTPException

from .. import db

router = APIRouter(prefix="/api")


def _cat(name: str, auto=0) -> int:
    name = name.strip()
    r = db.q1("SELECT id FROM categories WHERE name=?", (name,))
    return r["id"] if r else db.ex("INSERT INTO categories(name, auto) VALUES(?,?)", (name, auto)).lastrowid


# ---------- категории ----------
@router.get("/categories")
def categories():
    return [dict(r) for r in db.q("""SELECT c.id, c.name, c.auto, COUNT(ic.image_id) n FROM categories c
                                     LEFT JOIN image_categories ic ON ic.category_id=c.id GROUP BY c.id ORDER BY n DESC, c.name""")]


@router.post("/categories/assign")
def assign(body: dict = Body(...)):
    ids, add, remove = body.get("ids", []), body.get("add", []), body.get("remove", [])
    for n in add:
        cid = _cat(n)
        db.exmany("INSERT OR REPLACE INTO image_categories(image_id, category_id, score, source) VALUES(?,?,1,'user')", [(i, cid) for i in ids])
    for n in remove:
        r = db.q1("SELECT id FROM categories WHERE name=?", (n,))
        if r:
            db.exmany("DELETE FROM image_categories WHERE image_id=? AND category_id=?", [(i, r["id"]) for i in ids])
    return {"ok": True}


@router.patch("/categories/{cid}")
def rename_category(cid: int, body: dict = Body(...)):
    other = db.q1("SELECT id FROM categories WHERE name=? AND id<>?", (body["name"], cid))
    if other:  # слияние категорий
        db.ex("UPDATE OR IGNORE image_categories SET category_id=? WHERE category_id=?", (other["id"], cid))
        db.ex("DELETE FROM image_categories WHERE category_id=?", (cid,))
        db.ex("DELETE FROM categories WHERE id=?", (cid,))
    else:
        db.ex("UPDATE categories SET name=?, auto=0 WHERE id=?", (body["name"], cid))
    return {"ok": True}


@router.delete("/categories/{cid}")
def delete_category(cid: int):
    db.ex("DELETE FROM image_categories WHERE category_id=?", (cid,))
    db.ex("DELETE FROM categories WHERE id=?", (cid,))
    return {"ok": True}


# ---------- коллекции ----------
@router.get("/collections")
def collections():
    return [dict(r) for r in db.q("SELECT id, name FROM collections ORDER BY name")]


@router.post("/collections")
def create_collection(body: dict = Body(...)):
    name = body["name"].strip()
    if not name:
        raise HTTPException(400, "пустое имя")
    r = db.q1("SELECT id FROM collections WHERE name=?", (name,))
    cid = r["id"] if r else db.ex("INSERT INTO collections(name, created_at) VALUES(?,?)", (name, time.time())).lastrowid
    if body.get("ids"):
        db.exmany("INSERT OR IGNORE INTO collection_images VALUES(?,?)", [(cid, i) for i in body["ids"]])
    return {"id": cid}


@router.post("/collections/{cid}/images")
def collection_add(cid: int, body: dict = Body(...)):
    db.exmany("INSERT OR IGNORE INTO collection_images VALUES(?,?)", [(cid, i) for i in body.get("add", [])])
    db.exmany("DELETE FROM collection_images WHERE collection_id=? AND image_id=?", [(cid, i) for i in body.get("remove", [])])
    return {"ok": True}


@router.patch("/collections/{cid}")
def rename_collection(cid: int, body: dict = Body(...)):
    db.ex("UPDATE collections SET name=? WHERE id=?", (body["name"], cid))
    return {"ok": True}


@router.delete("/collections/{cid}")
def delete_collection(cid: int):
    db.ex("DELETE FROM collection_images WHERE collection_id=?", (cid,))
    db.ex("DELETE FROM collections WHERE id=?", (cid,))
    return {"ok": True}


# ---------- эксперименты ----------
@router.get("/experiments")
def experiments():
    return [dict(r) for r in db.q("""SELECT e.*, (SELECT COUNT(*) FROM images i WHERE i.experiment_id=e.id AND i.deleted=0) n
                                     FROM experiments e ORDER BY created_at DESC""")]


@router.post("/experiments")
def create_experiment(body: dict = Body(...)):
    """Новые эксперименты создаёт только пользователь. Новые графики после этого попадают в самый свежий эксперимент."""
    name = (body.get("name") or "").strip() or "Эксперимент " + time.strftime("%Y-%m-%d %H:%M")
    eid = db.ex("INSERT INTO experiments(name, created_at, auto, note) VALUES(?,?,0,?)", (name, time.time(), body.get("note", ""))).lastrowid
    return {"id": eid}


@router.patch("/experiments/{eid}")
def patch_experiment(eid: int, body: dict = Body(...)):
    """Переименование не меняет created_at."""
    for k in ("name", "note"):
        if k in body:
            db.ex(f"UPDATE experiments SET {k}=? WHERE id=?", (body[k], eid))
    return {"ok": True}


@router.delete("/experiments/{eid}")
def delete_experiment(eid: int):
    db.ex("UPDATE images SET experiment_id=NULL WHERE experiment_id=?", (eid,))
    db.ex("DELETE FROM experiments WHERE id=?", (eid,))
    return {"ok": True}


@router.post("/experiments/{eid}/images")
def experiment_assign(eid: int, body: dict = Body(...)):
    db.exmany("UPDATE images SET experiment_id=? WHERE id=?", [(eid, i) for i in body.get("ids", [])])
    return {"ok": True}


@router.get("/experiments/{eid}/summary")
def experiment_summary(eid: int):
    """Сводка по графикам эксперимента: серии, максимумы и плато (вычислены кодом по извлечённым точкам)."""
    rows = db.q("SELECT id, name, chart, summary FROM images WHERE experiment_id=? AND deleted=0 ORDER BY COALESCE(taken_at, mtime)", (eid,))
    charts, best = [], None
    for r in rows:
        c = json.loads(r["chart"]) if r["chart"] else None
        if not c:
            charts.append({"id": r["id"], "name": r["name"], "title": None, "series": []})
            continue
        series = c.get("stats", {}).get("series", [])
        charts.append({"id": r["id"], "name": r["name"], "title": c.get("title"), "x": c.get("x_axis"), "y": c.get("y_axis"), "series": series})
        for s in series:
            if best is None or s["max"] > best["max"]:
                best = {**s, "image_id": r["id"], "title": c.get("title")}
    return {"experiment": dict(db.q1("SELECT * FROM experiments WHERE id=?", (eid,)) or {}), "charts": charts, "best": best}
