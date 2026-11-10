"""Import completed results from explicitly registered local evaluation databases."""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from . import db, search
from .config import get_settings

log = logging.getLogger("imagehub.external_results")
_lock = threading.Lock()
_last_check = 0.0
FIELDS = (
    "kind",
    "kind_scores",
    "has_text",
    "text_lang",
    "text_rotation",
    "text_sample",
    "ocr_raw",
    "ocr_md",
    "ocr_blocks",
    "caption",
    "summary",
    "summary_ru",
    "event",
    "objects",
    "regions",
    "tags",
    "tags_ru",
    "chart",
    "stages",
    "error",
    "pipeline_ver",
    "timings",
)


def unchanged(path, size, mtime):
    try:
        st = os.stat(path)
        return st.st_size == size and abs(st.st_mtime - mtime) < 0.001
    except (OSError, TypeError, ValueError):
        return False


def _person(target, source_key, external_id, embedding):
    if external_id is None or not embedding:
        return None
    row = target.execute(
        "SELECT person_id FROM external_people WHERE source=? AND external_id=?", (source_key, external_id)
    ).fetchone()
    person_id = row["person_id"] if row else None
    person = target.execute("SELECT centroid, n FROM people WHERE id=?", (person_id,)).fetchone() if person_id else None
    vector = np.frombuffer(embedding, dtype=np.float32)
    if not len(vector):
        return None
    if person is None:
        person_id = target.execute("INSERT INTO people(name, centroid, n) VALUES(NULL,?,0)", (vector.tobytes(),)).lastrowid
        target.execute("UPDATE people SET auto_name=? WHERE id=?", (f"person_{person_id}", person_id))
        target.execute(
            "INSERT OR REPLACE INTO external_people(source,external_id,person_id) VALUES(?,?,?)",
            (source_key, external_id, person_id),
        )
        centroid, count = vector, 0
    else:
        centroid = np.frombuffer(person["centroid"], dtype=np.float32) if person["centroid"] else vector
        count = person["n"] or 0
    if centroid.shape == vector.shape:
        combined = centroid * count + vector
        norm = np.linalg.norm(combined)
        if norm:
            combined = combined / norm
        target.execute(
            "UPDATE people SET centroid=?,n=? WHERE id=?", (combined.astype(np.float32).tobytes(), count + 1, person_id)
        )
    return person_id


def _experiment(target, source, source_key, external_id):
    if external_id is None:
        return None
    row = target.execute(
        "SELECT experiment_id FROM external_experiments WHERE source=? AND external_id=?", (source_key, external_id)
    ).fetchone()
    if row and target.execute("SELECT 1 FROM experiments WHERE id=?", (row["experiment_id"],)).fetchone():
        return row["experiment_id"]
    old = source.execute("SELECT name,created_at,auto,note FROM experiments WHERE id=?", (external_id,)).fetchone()
    if old is None:
        return None
    new_id = target.execute("INSERT INTO experiments(name,created_at,auto,note) VALUES(?,?,?,?)", tuple(old)).lastrowid
    target.execute(
        "INSERT OR REPLACE INTO external_experiments(source,external_id,experiment_id) VALUES(?,?,?)",
        (source_key, external_id, new_id),
    )
    return new_id


def _copy_one(source, source_key, src, dest_id):
    target = db.conn()
    target.execute("BEGIN IMMEDIATE")
    try:
        dest = target.execute("SELECT * FROM images WHERE id=?", (dest_id,)).fetchone()
        if not dest or dest["status"] not in ("new", "queued") or dest["deleted"] or dest["hash"] != src["hash"]:
            target.rollback()
            return False
        if not unchanged(dest["path"], dest["size"], dest["mtime"]):
            target.rollback()
            return False
        experiment_id = dest["experiment_id"] or _experiment(target, source, source_key, src["experiment_id"])
        values = [src[field] for field in FIELDS]
        assignments = ",".join(f"{field}=?" for field in FIELDS)
        target.execute(
            f"UPDATE images SET {assignments},experiment_id=?,status='done',priority=0 WHERE id=?",
            (*values, experiment_id, dest_id),
        )
        for v in source.execute("SELECT kind,dim,data FROM vectors WHERE image_id=?", (src["id"],)):
            target.execute(
                "INSERT OR REPLACE INTO vectors(image_id,kind,dim,data) VALUES(?,?,?,?)",
                (dest_id, v["kind"], v["dim"], v["data"]),
            )
        for c in source.execute(
            "SELECT c.name,ic.score FROM image_categories ic JOIN categories c ON c.id=ic.category_id WHERE ic.image_id=? AND ic.source='auto'",
            (src["id"],),
        ):
            target.execute("INSERT OR IGNORE INTO categories(name,auto) VALUES(?,1)", (c["name"],))
            cid = target.execute("SELECT id FROM categories WHERE name=?", (c["name"],)).fetchone()["id"]
            target.execute(
                "INSERT OR IGNORE INTO image_categories(image_id,category_id,score,source) VALUES(?,?,?,'auto')",
                (dest_id, cid, c["score"]),
            )
        for face in source.execute("SELECT box,score,emb,person_id FROM faces WHERE image_id=?", (src["id"],)):
            pid = _person(target, source_key, face["person_id"], face["emb"])
            target.execute(
                "INSERT INTO faces(image_id,box,score,emb,person_id,confirmed) VALUES(?,?,?,?,?,0)",
                (dest_id, face["box"], face["score"], face["emb"], pid),
            )
        current = target.execute(
            "SELECT name,summary,summary_ru,tags_ru,ocr_raw,ocr_md,tags,caption FROM images WHERE id=?", (dest_id,)
        ).fetchone()
        text = " ".join(filter(None, [current["summary"], current["caption"], current["summary_ru"]]))
        tags = " ".join(json.loads(current["tags"])) if current["tags"] else ""
        if current["tags_ru"]:
            tags += " " + " ".join(json.loads(current["tags_ru"]))
        target.execute("DELETE FROM fts WHERE rowid=?", (dest_id,))
        target.execute(
            "INSERT INTO fts(rowid,name,summary,ocr,tags) VALUES(?,?,?,?,?)",
            (dest_id, current["name"] or "", text, current["ocr_md"] or current["ocr_raw"] or "", tags),
        )
        target.execute(
            "INSERT OR IGNORE INTO external_imports(source,external_id,image_id) VALUES(?,?,?)", (source_key, src["id"], dest_id)
        )
        target.commit()
        return True
    except Exception:
        target.rollback()
        raise


def sync_sources(force=False):
    """Copy only complete, error-free records with matching hash, size and unchanged files."""
    global _last_check
    if not force and time.monotonic() - _last_check < 8:
        return 0
    if not _lock.acquire(blocking=False):
        return 0
    copied = 0
    try:
        _last_check = time.monotonic()
        root = get_settings().data_dir.resolve()
        registry = root / "evaluation-sources.json"
        if not registry.is_file():
            return 0
        paths = json.loads(registry.read_text(encoding="utf-8")).get("sources", [])
        target = db.conn()
        target.execute(
            "CREATE TABLE IF NOT EXISTS external_imports(source TEXT,external_id INTEGER,image_id INTEGER,PRIMARY KEY(source,external_id,image_id))"
        )
        target.execute(
            "CREATE TABLE IF NOT EXISTS external_people(source TEXT,external_id INTEGER,person_id INTEGER,PRIMARY KEY(source,external_id))"
        )
        target.execute(
            "CREATE TABLE IF NOT EXISTS external_experiments(source TEXT,external_id INTEGER,experiment_id INTEGER,PRIMARY KEY(source,external_id))"
        )
        target.commit()
        for value in paths:
            path = Path(value).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path == get_settings().db_path.resolve():
                continue
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)) as source:
                source.row_factory = sqlite3.Row
                for src in source.execute(
                    "SELECT * FROM images WHERE status='done' AND (error IS NULL OR error='') AND hash IS NOT NULL"
                ):
                    if not unchanged(src["path"], src["size"], src["mtime"]):
                        continue
                    destinations = target.execute(
                        "SELECT i.id FROM images i WHERE i.hash=? AND i.size=? AND i.deleted=0 AND i.status IN ('new','queued') "
                        "AND NOT EXISTS(SELECT 1 FROM external_imports e WHERE e.source=? AND e.external_id=? AND e.image_id=i.id)",
                        (src["hash"], src["size"], str(path), src["id"]),
                    ).fetchall()
                    for dest in destinations:
                        if _copy_one(source, str(path), src, dest["id"]):
                            copied += 1
        if copied:
            search.invalidate()
        return copied
    except (OSError, ValueError, sqlite3.Error, KeyError, IndexError, TypeError):
        log.exception("could not import evaluation results")
        return copied
    finally:
        _lock.release()
