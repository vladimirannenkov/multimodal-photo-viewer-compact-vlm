"""Гибридный поиск (всё на CPU): SigLIP2 текст↔изображение, e5 по описаниям и OCR, BM25 (FTS5), объединение через RRF.

Поиск по изображению: косинус к эмбеддингам. Изображение + текст: взвешенная сумма векторов в пространстве SigLIP2,
чем больше вес текста, тем сильнее он влияет на результат."""
import re
import threading

import numpy as np

from . import db, queries

_lock = threading.Lock()
_idx = {"img": None, "txt": None, "ver": None}
RRF_K = 60


def invalidate():
    with _lock:
        _idx["ver"] = None


def _load():
    with _lock:
        ver = db.q1("SELECT COUNT(*) c, COALESCE(MAX(rowid),0) m FROM vectors")
        ver = (ver["c"], ver["m"])
        if _idx["ver"] == ver:
            return
        for kind in ("img", "txt"):
            rows = db.q("SELECT image_id, data FROM vectors WHERE kind=?", (kind,))
            ids = np.array([r["image_id"] for r in rows], dtype=np.int64)
            mat = np.stack([db.blob_to_vec(r["data"]) for r in rows]) if rows else np.zeros((0, 1), np.float32)
            _idx[kind] = (ids, mat)
        _idx["ver"] = ver


def _score(kind, vec, allowed):
    _load()
    ids, mat = _idx[kind]
    if len(ids) == 0 or mat.shape[1] != len(vec):
        return {}
    s = mat @ vec
    out = {}
    for i, v in zip(ids.tolist(), s.tolist()):
        if allowed is None or i in allowed:
            out[i] = v
    return out


def _rank(scores):
    return {i: r for r, (i, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1]))}


def _fts_query(text):
    toks = re.findall(r"\w+", text, re.U)
    return " OR ".join(f'"{t}"*' for t in toks[:12]) if toks else ""


def bm25(text, allowed, limit=300):
    q = _fts_query(text)
    if not q:
        return {}
    rows = db.q("SELECT rowid, bm25(fts, 2.0, 1.0, 1.0, 1.5) AS s FROM fts WHERE fts MATCH ? ORDER BY s LIMIT ?", (q, limit))
    return {r["rowid"]: -r["s"] for r in rows if allowed is None or r["rowid"] in allowed}


def _query_vec(sg, text):
    """Короткие запросы («bed») SigLIP понимает лучше в виде фразы: усредняем несколько шаблонов."""
    prompts = [text]
    if len(text.split()) <= 4:
        prompts += [f"a photo of {text}", f"a photo of a {text}", f"This is a photo of {text}."] if text.isascii() else [f"фото: {text}"]
    v = sg.text(prompts).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def search(text: str = "", image_vec=None, text_weight: float = 0.5, filters: dict | None = None, limit: int = 200, device="cpu"):
    """-> [(image_id, score)] по убыванию. text и/или image_vec (эмбеддинг SigLIP2 картинки-запроса)."""
    from .ml import siglip, textembed
    allowed = queries.candidate_ids(filters or {}) if filters else None
    text = (text or "").strip()
    sg = siglip.get("cpu") if (text or image_vec is not None) else None
    if image_vec is not None and not text:
        sc = _score("img", np.asarray(image_vec, np.float32), allowed)
        return sorted(sc.items(), key=lambda x: -x[1])[:limit]
    if image_vec is not None and text:
        tv = _query_vec(sg, text)
        v = text_weight * tv + (1 - text_weight) * np.asarray(image_vec, np.float32)
        v /= np.linalg.norm(v) + 1e-9
        sc = _score("img", v, allowed)
        return sorted(sc.items(), key=lambda x: -x[1])[:limit]
    if not text:
        return []
    ranks = [_rank(_score("img", _query_vec(sg, text), allowed))] * 2  # картинка весит вдвое: главный сигнал для объектов
    try:
        ranks.append(_rank(_score("txt", textembed.query(text), allowed)))
    except Exception:
        pass
    b = bm25(text, allowed)
    if b:
        ranks.append(_rank(b))
        ranks.append(_rank(b))  # точные слова (названия, числа) важны: BM25 весит вдвое
    fused = {}
    for r in ranks:
        for i, rk in r.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rk)
    return sorted(fused.items(), key=lambda x: -x[1])[:limit]


def similar(image_id: int, limit=60, filters=None):
    r = db.q1("SELECT data FROM vectors WHERE image_id=? AND kind='img'", (image_id,))
    if not r:
        return []
    allowed = queries.candidate_ids(filters or {}) if filters else None
    sc = _score("img", db.blob_to_vec(r["data"]), allowed)
    sc.pop(image_id, None)
    return sorted(sc.items(), key=lambda x: -x[1])[:limit]


def duplicates(threshold=0.97):
    """Точные дубликаты по хешу и почти-дубликаты по эмбеддингу."""
    exact = db.q("""SELECT hash, GROUP_CONCAT(id) ids FROM images WHERE deleted=0 AND hash IS NOT NULL
                    GROUP BY hash HAVING COUNT(*)>1""")
    groups = [{"type": "exact", "ids": [int(x) for x in r["ids"].split(",")]} for r in exact]
    _load()
    ids, mat = _idx["img"]
    seen = {i for g in groups for i in g["ids"]}
    if len(ids) > 1:
        alive = {r["id"] for r in db.q("SELECT id FROM images WHERE deleted=0")}
        for a in range(0, len(ids), 512):
            sims = mat[a:a + 512] @ mat.T
            for ra, row in enumerate(sims):
                ia = int(ids[a + ra])
                if ia in seen or ia not in alive:
                    continue
                js = [j for j in np.where(row >= threshold)[0].tolist() if int(ids[j]) != ia and int(ids[j]) in alive and int(ids[j]) not in seen]
                if js:
                    grp = [ia] + [int(ids[j]) for j in js]
                    groups.append({"type": "similar", "ids": grp})
                    seen.update(grp)
    return groups
