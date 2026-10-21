"""Импорт и быстрое обнаружение новых файлов.

Режимы: быстрый (папка с прежним mtime пропускается целиком, файлы не проверяются) и глубокий (stat каждого файла).
Хеш и миниатюры считаются только для новых и изменённых файлов, пулом процессов."""

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import db
from .config import get_settings
from .imaging import is_image, prepare

_pool = None
_pool_lock = threading.Lock()


def pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=get_settings().workers)
        return _pool


def shutdown_pool():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def thumb_path(image_id: int) -> Path:
    return get_settings().thumbs_dir / f"{image_id}.webp"


def add_folder(path: str, watched: bool = True) -> int:
    path = str(Path(path).resolve())
    r = db.q1("SELECT id FROM folders WHERE path=?", (path,))
    if r:
        return r["id"]
    return db.ex("INSERT INTO folders(path, watched) VALUES(?,?)", (path, int(watched))).lastrowid


def _walk(root: str, deep: bool, seen_files: list, changed_dirs: list):
    """Обход дерева. В быстром режиме файлы каталога читаются только если mtime каталога изменился."""
    stack = [root]
    known = {r["path"]: r["mtime"] for r in db.q("SELECT path, mtime FROM dirs WHERE substr(path, 1, ?) = ?", (len(root), root))}
    while stack:
        d = stack.pop()
        try:
            dm = os.stat(d).st_mtime
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                stack.append(e.path)
        if not deep and known.get(d) == dm:
            continue  # каталог не менялся: новых и удалённых файлов в нём нет
        changed_dirs.append((d, dm))
        for e in entries:
            if e.is_file() and is_image(e.name):
                st = e.stat()
                seen_files.append((e.path, st.st_size, st.st_mtime))


def scan_folder(folder_id: int, deep: bool = False, progress=None) -> dict:
    f = db.q1("SELECT * FROM folders WHERE id=?", (folder_id,))
    if f is None:
        raise KeyError(folder_id)
    root = f["path"]
    t0 = time.time()
    seen, changed_dirs = [], []
    _walk(root, deep, seen, changed_dirs)
    existing = {
        r["path"]: (r["size"], r["mtime"], r["id"], r["deleted"])
        for r in db.q("SELECT id, path, size, mtime, deleted FROM images WHERE folder_id=?", (folder_id,))
    }
    todo, restored = [], 0
    for p, size, mt in seen:
        old = existing.get(p)
        if old is None:
            todo.append((p, None))
        elif old[0] != size or abs(old[1] - mt) > 1e-3:
            todo.append((p, old[2]))
        elif old[3]:
            db.ex("UPDATE images SET deleted=0 WHERE id=?", (old[2],))
            restored += 1
    # удалённые файлы: проверяем в тех каталогах, которые просмотрели
    scanned_dirs = {d for d, _ in changed_dirs}
    removed = 0
    for p, (_, _, iid, deleted) in existing.items():
        if not deleted and os.path.dirname(p) in scanned_dirs and not os.path.exists(p):
            db.ex("UPDATE images SET deleted=1 WHERE id=?", (iid,))
            removed += 1
    new_ids = _ingest(todo, folder_id, progress)
    for d, dm in changed_dirs:
        db.ex("INSERT OR REPLACE INTO dirs(path, mtime) VALUES(?,?)", (d, dm))
    db.ex("UPDATE folders SET last_scan=? WHERE id=?", (time.time(), folder_id))
    return {
        "folder": root,
        "seen": len(seen),
        "new": len([1 for _, o in todo if o is None]),
        "updated": len([1 for _, o in todo if o]),
        "removed": removed,
        "restored": restored,
        "ids": new_ids,
        "sec": round(time.time() - t0, 2),
    }


def _ingest(todo, folder_id, progress=None, clipboard=False):
    """todo: [(path, existing_id | None)]. Хеш и миниатюра в пуле процессов, затем запись в БД."""
    if not todo:
        return []
    s = get_settings()
    ids = {}
    for p, iid in todo:  # сначала строки, чтобы id были известны (миниатюра называется по id)
        st = os.stat(p)
        if iid is None:
            iid = db.ex(
                "INSERT OR IGNORE INTO images(path, folder_id, name, size, mtime, added_at, from_clipboard, status) VALUES(?,?,?,?,?,?,?, 'new')",
                (p, folder_id, os.path.basename(p), st.st_size, st.st_mtime, time.time(), int(clipboard)),
            ).lastrowid
            if not iid:
                iid = db.q1("SELECT id FROM images WHERE path=?", (p,))["id"]
        else:
            db.ex(
                "UPDATE images SET size=?, mtime=?, status='new', stages='{}', deleted=0 WHERE id=?",
                (st.st_size, st.st_mtime, iid),
            )
        ids[p] = iid
    futs = {p: pool().submit(prepare, p, str(thumb_path(i)), s.thumb_side) for p, i in ids.items()}
    done = 0
    for p, fu in futs.items():
        r = fu.result()
        iid = ids[p]
        if "error" in r:
            db.ex("UPDATE images SET status='error', error=? WHERE id=?", (r["error"], iid))
        else:
            db.ex(
                "UPDATE images SET hash=?, width=?, height=?, format=?, frames=?, taken_at=? WHERE id=?",
                (r["hash"], r["width"], r["height"], r["format"], r["frames"], r["taken_at"], iid),
            )
        done += 1
        if progress:
            progress(done, len(futs))
    return list(ids.values())


def import_files(paths, folder_id=None, clipboard=False):
    """Отдельные файлы (перетаскивание, выбор в диалоге, буфер обмена)."""
    todo = []
    for p in paths:
        p = str(Path(p).resolve())
        if not is_image(p):
            continue
        r = db.q1("SELECT id FROM images WHERE path=?", (p,))
        todo.append((p, r["id"] if r else None))
    if folder_id is None and todo:
        folder_id = add_folder(str(Path(todo[0][0]).parent), watched=False)
    return _ingest(todo, folder_id, clipboard=clipboard)


def scan_all(deep=False):
    return [scan_folder(f["id"], deep=deep) for f in db.q("SELECT id FROM folders")]
