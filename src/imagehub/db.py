"""SQLite (WAL) + FTS5. Соединение на поток, схема создаётся при первом открытии."""

import json
import sqlite3
import threading
import time

import numpy as np

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, watched INTEGER DEFAULT 1,
    dir_mtime REAL DEFAULT 0, last_scan REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS dirs(path TEXT PRIMARY KEY, mtime REAL);
CREATE TABLE IF NOT EXISTS experiments(
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL, auto INTEGER DEFAULT 0, note TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS images(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, folder_id INTEGER REFERENCES folders(id),
    name TEXT, size INTEGER, mtime REAL, hash TEXT, width INTEGER, height INTEGER, format TEXT, frames INTEGER DEFAULT 1,
    taken_at REAL, added_at REAL, from_clipboard INTEGER DEFAULT 0, deleted INTEGER DEFAULT 0,
    favorite INTEGER DEFAULT 0, rating INTEGER DEFAULT 0,
    kind TEXT, kind_scores TEXT,
    has_text INTEGER, text_lang TEXT, text_rotation INTEGER DEFAULT 0, text_sample TEXT,
    ocr_raw TEXT, ocr_md TEXT, ocr_blocks TEXT,
    caption TEXT, summary TEXT, event TEXT, objects TEXT, regions TEXT, tags TEXT,
    chart TEXT, experiment_id INTEGER REFERENCES experiments(id),
    status TEXT DEFAULT 'new', stages TEXT DEFAULT '{}', error TEXT, pipeline_ver INTEGER DEFAULT 0, timings TEXT, priority INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_images_folder ON images(folder_id);
CREATE INDEX IF NOT EXISTS ix_images_hash ON images(hash);
CREATE INDEX IF NOT EXISTS ix_images_status ON images(status);
CREATE INDEX IF NOT EXISTS ix_images_kind ON images(kind);
CREATE INDEX IF NOT EXISTS ix_images_exp ON images(experiment_id);
CREATE TABLE IF NOT EXISTS vectors(
    image_id INTEGER NOT NULL, kind TEXT NOT NULL, dim INTEGER, data BLOB, PRIMARY KEY(image_id, kind));
CREATE TABLE IF NOT EXISTS categories(
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, auto INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS image_categories(
    image_id INTEGER NOT NULL, category_id INTEGER NOT NULL, score REAL DEFAULT 1, source TEXT DEFAULT 'user',
    PRIMARY KEY(image_id, category_id));
CREATE INDEX IF NOT EXISTS ix_ic_cat ON image_categories(category_id);
CREATE TABLE IF NOT EXISTS collections(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at REAL);
CREATE TABLE IF NOT EXISTS collection_images(
    collection_id INTEGER NOT NULL, image_id INTEGER NOT NULL, PRIMARY KEY(collection_id, image_id));
CREATE TABLE IF NOT EXISTS people(
    id INTEGER PRIMARY KEY, name TEXT, auto_name TEXT, centroid BLOB, n INTEGER DEFAULT 0, labeled INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, box TEXT, score REAL, emb BLOB,
    person_id INTEGER REFERENCES people(id), confirmed INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_faces_img ON faces(image_id);
CREATE INDEX IF NOT EXISTS ix_faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS chat(
    id INTEGER PRIMARY KEY, image_id INTEGER, role TEXT, content TEXT, ts REAL, meta TEXT);
CREATE INDEX IF NOT EXISTS ix_chat_img ON chat(image_id);
CREATE TABLE IF NOT EXISTS annotations(
    id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, data TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS undo_log(id INTEGER PRIMARY KEY, ts REAL, action TEXT, payload TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(name, summary, ocr, tags, tokenize='unicode61 remove_diacritics 2');
"""

_local = threading.local()
_init_lock = threading.Lock()
_inited = False


def conn() -> sqlite3.Connection:
    global _inited
    c = getattr(_local, "c", None)
    if c is None:
        s = get_settings()
        c = sqlite3.connect(s.db_path, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.c = c
        with _init_lock:
            if not _inited:
                c.executescript(SCHEMA)
                cols = {r["name"] for r in c.execute("PRAGMA table_info(images)")}
                for col in ("summary_ru", "tags_ru"):  # миграция для баз, созданных до появления перевода
                    if col not in cols:
                        c.execute(f"ALTER TABLE images ADD COLUMN {col} TEXT")
                c.commit()
                _inited = True
    return c


def reset_for_tests():
    global _inited
    _inited = False
    _local.c = None


def q(sql, args=()):
    return conn().execute(sql, args).fetchall()


def q1(sql, args=()):
    return conn().execute(sql, args).fetchone()


def ex(sql, args=()):
    c = conn()
    cur = c.execute(sql, args)
    c.commit()
    return cur


def exmany(sql, rows):
    c = conn()
    c.executemany(sql, rows)
    c.commit()


def row_dict(r, json_fields=()):
    if r is None:
        return None
    d = dict(r)
    for f in json_fields:
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except Exception:
                pass
    return d


def vec_to_blob(v) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()


def blob_to_vec(b) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def set_vector(image_id, kind, v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    ex("INSERT OR REPLACE INTO vectors(image_id, kind, dim, data) VALUES(?,?,?,?)", (image_id, kind, len(v), vec_to_blob(v)))


def fts_update(image_id):
    """Пересобирает запись полнотекстового индекса изображения ."""
    c = conn()
    r = c.execute(
        "SELECT name, summary, summary_ru, tags_ru, ocr_raw, ocr_md, tags, caption FROM images WHERE id=?", (image_id,)
    ).fetchone()
    if r is None:
        return
    c.execute("DELETE FROM fts WHERE rowid=?", (image_id,))
    text = " ".join(filter(None, [r["summary"], r["caption"], r["summary_ru"]]))
    ocr = r["ocr_md"] or r["ocr_raw"] or ""
    tags = " ".join(json.loads(r["tags"])) if r["tags"] else ""
    if r["tags_ru"]:
        tags += " " + " ".join(json.loads(r["tags_ru"]))
    c.execute("INSERT INTO fts(rowid, name, summary, ocr, tags) VALUES(?,?,?,?,?)", (image_id, r["name"] or "", text, ocr, tags))
    c.commit()


def now() -> float:
    return time.time()
