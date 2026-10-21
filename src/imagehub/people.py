"""Глобальная база лиц: сопоставление эмбеддинга с известными людьми, новые люди получают имя person_N."""

import json

import numpy as np

from . import db

THRESHOLD = 0.45  # косинус ArcFace (buffalo_l): один человек обычно выше 0.45


def _centroids():
    rows = db.q("SELECT id, centroid, n FROM people WHERE centroid IS NOT NULL")
    if not rows:
        return [], np.zeros((0, 512), np.float32), []
    return [r["id"] for r in rows], np.stack([np.frombuffer(r["centroid"], np.float32) for r in rows]), [r["n"] for r in rows]


def add_faces(image_id, faces):
    """Сохраняет лица изображения и привязывает к людям. -> список строк лиц."""
    db.ex("DELETE FROM faces WHERE image_id=? AND confirmed=0", (image_id,))
    out = []
    for f in faces:
        emb = np.asarray(f["emb"], np.float32)
        ids, mat, ns = _centroids()
        pid, sim = None, 0.0
        if len(ids):
            sims = mat @ emb
            j = int(sims.argmax())
            if sims[j] >= THRESHOLD:
                pid, sim = ids[j], float(sims[j])
        if pid is None:
            pid = db.ex("INSERT INTO people(name, centroid, n) VALUES(NULL, ?, 0)", (emb.tobytes(),)).lastrowid
            db.ex("UPDATE people SET auto_name=? WHERE id=?", (f"person_{pid}", pid))
            ids, mat, ns = [pid], emb[None], [0]
        j = ids.index(pid)
        n = ns[j]
        c = mat[j] * n + emb
        c = c / (np.linalg.norm(c) + 1e-9)
        db.ex("UPDATE people SET centroid=?, n=? WHERE id=?", (c.astype(np.float32).tobytes(), n + 1, pid))
        fid = db.ex(
            "INSERT INTO faces(image_id, box, score, emb, person_id) VALUES(?,?,?,?,?)",
            (image_id, json.dumps([round(v, 1) for v in f["box"]]), f["score"], emb.tobytes(), pid),
        ).lastrowid
        out.append({"id": fid, "person_id": pid, "sim": sim})
    return out


def person_label(r):
    return r["name"] or r["auto_name"] or f"person_{r['id']}"


def rename(person_id, name):
    """Присваивает имя; если такое имя уже есть у другого человека, люди объединяются."""
    other = db.q1("SELECT id FROM people WHERE name=? AND id<>?", (name, person_id))
    if other:
        merge(person_id, other["id"])
        return other["id"]
    db.ex("UPDATE people SET name=?, labeled=1 WHERE id=?", (name, person_id))
    return person_id


def merge(src, dst):
    a = db.q1("SELECT centroid, n FROM people WHERE id=?", (src,))
    b = db.q1("SELECT centroid, n FROM people WHERE id=?", (dst,))
    if a and b and a["centroid"] and b["centroid"]:
        c = np.frombuffer(a["centroid"], np.float32) * a["n"] + np.frombuffer(b["centroid"], np.float32) * b["n"]
        c = c / (np.linalg.norm(c) + 1e-9)
        db.ex("UPDATE people SET centroid=?, n=? WHERE id=?", (c.astype(np.float32).tobytes(), a["n"] + b["n"], dst))
    db.ex("UPDATE faces SET person_id=? WHERE person_id=?", (dst, src))
    db.ex("DELETE FROM people WHERE id=?", (src,))


def next_unlabeled():
    """Следующее лицо без имени: самое крупное по числу фотографий человека."""
    r = db.q1("""SELECT f.id AS face_id, f.image_id, f.box, f.person_id, p.n FROM faces f JOIN people p ON p.id=f.person_id
                 JOIN images i ON i.id=f.image_id
                 WHERE p.labeled=0 AND f.confirmed=0 AND i.deleted=0 ORDER BY p.n DESC, f.score DESC LIMIT 1""")
    return dict(r) if r else None


def candidate_for(person_id, min_sim=0.28):
    """Ближайший по эмбеддингу человек с заданным именем: предложение «это X?» для диалога разметки."""
    me = db.q1("SELECT centroid FROM people WHERE id=?", (person_id,))
    if not me or not me["centroid"]:
        return None
    v = np.frombuffer(me["centroid"], np.float32)
    best, name = 0.0, None
    for r in db.q(
        "SELECT name, centroid FROM people WHERE labeled=1 AND name IS NOT NULL AND centroid IS NOT NULL AND id<>?", (person_id,)
    ):
        s = float(np.frombuffer(r["centroid"], np.float32) @ v)
        if s > best:
            best, name = s, r["name"]
    return {"name": name, "sim": best} if name and best >= min_sim else None
