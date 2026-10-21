"""Люди и лица: глобальная база, имена из чата, объединение, кадры лиц."""

import io
import json

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from .. import db, people
from .deps import image_row, pil_open

router = APIRouter(prefix="/api")

GENERIC = ["прохожий", "студент", "коллега", "родственник", "друг", "незнакомец"]
BY_EVENT = {
    "работа": ["коллега", "сотрудник"],
    "праздник": ["гость", "друг", "родственник"],
    "спорт": ["спортсмен", "тренер"],
    "прогулка": ["прохожий", "друг"],
    "путешествие": ["турист", "прохожий"],
    "город": ["прохожий"],
}


@router.get("/people")
def list_people():
    return [
        dict(r)
        for r in db.q("""SELECT p.id, COALESCE(p.name, p.auto_name) AS name, p.labeled, p.n,
                                      (SELECT f.id FROM faces f WHERE f.person_id=p.id ORDER BY f.score DESC LIMIT 1) AS face_id
                                      FROM people p WHERE p.n>0 ORDER BY p.labeled DESC, p.n DESC""")
    ]


@router.patch("/people/{pid}")
def rename_person(pid: int, body: dict = Body(...)):
    name = body["name"].strip()
    if not name:
        raise HTTPException(400, "пустое имя")
    if body.get("role"):  # общий класс («прохожий») не объединяет разных людей
        db.ex("UPDATE people SET name=?, labeled=1 WHERE id=?", (name, pid))
        return {"id": pid}
    return {"id": people.rename(pid, name)}


@router.post("/people/{pid}/merge")
def merge_person(pid: int, body: dict = Body(...)):
    people.merge(pid, body["into"])
    return {"ok": True}


@router.get("/faces/{face_id}/crop")
def face_crop(face_id: int, pad: float = 0.35, size: int = 160):
    f = db.q1("SELECT image_id, box FROM faces WHERE id=?", (face_id,))
    if f is None:
        raise HTTPException(404)
    im = pil_open(image_row(f["image_id"]))
    x1, y1, x2, y2 = json.loads(f["box"])
    w, h = x2 - x1, y2 - y1
    c = im.crop((max(0, x1 - pad * w), max(0, y1 - pad * h), min(im.width, x2 + pad * w), min(im.height, y2 + pad * h)))
    c.thumbnail((size * 2, size * 2))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


def _suggestions(image_id):
    r = db.q1("SELECT event, kind FROM images WHERE id=?", (image_id,))
    known = [x["name"] for x in db.q("SELECT name FROM people WHERE labeled=1 AND name IS NOT NULL ORDER BY n DESC LIMIT 8")]
    ctx = BY_EVENT.get((r["event"] if r else None) or "", [])
    seen, out = set(), []
    for n in ctx + GENERIC:
        if n not in seen and n not in known:
            seen.add(n)
            out.append(n)
    return {"known": known, "roles": out[:5]}


@router.get("/faces/next")
def next_face():
    """Следующее лицо без имени для диалога в чате."""
    nf = people.next_unlabeled()
    if not nf:
        return {"face": None}
    nf["box"] = json.loads(nf["box"])
    im = db.q1("SELECT id, name, width, height FROM images WHERE id=?", (nf["image_id"],))
    return {
        "face": nf,
        "image": dict(im),
        "suggestions": _suggestions(nf["image_id"]),
        "candidate": people.candidate_for(nf["person_id"]),
        "other_images": db.q1("SELECT COUNT(DISTINCT image_id) c FROM faces WHERE person_id=?", (nf["person_id"],))["c"],
    }


@router.post("/faces/{face_id}/label")
def label_face(face_id: int, body: dict = Body(...)):
    f = db.q1("SELECT person_id FROM faces WHERE id=?", (face_id,))
    if f is None:
        raise HTTPException(404)
    name = body.get("name", "").strip()
    if body.get("skip"):  # «не знаю»: человек помечается обработанным без имени, чат переходит к следующему
        db.ex("UPDATE people SET labeled=1 WHERE id=?", (f["person_id"],))
        return {"ok": True}
    if not name:
        raise HTTPException(400, "пустое имя")
    if body.get("role"):
        db.ex("UPDATE people SET name=?, labeled=1 WHERE id=?", (name, f["person_id"]))
        return {"person_id": f["person_id"]}
    return {"person_id": people.rename(f["person_id"], name)}


@router.post("/faces/{face_id}/reassign")
def reassign(face_id: int, body: dict = Body(...)):
    db.ex("UPDATE faces SET person_id=?, confirmed=1 WHERE id=?", (body["person_id"], face_id))
    for pid in db.q("SELECT id FROM people"):
        n = db.q1("SELECT COUNT(*) c FROM faces WHERE person_id=?", (pid["id"],))["c"]
        db.ex("UPDATE people SET n=? WHERE id=?", (n, pid["id"]))
    return {"ok": True}


@router.delete("/faces/{face_id}")
def delete_face(face_id: int):
    db.ex("DELETE FROM faces WHERE id=?", (face_id,))
    return {"ok": True}
