"""Общие помощники API."""

import mimetypes
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

from .. import db
from ..config import get_settings
from ..imaging import open_rgb

BROWSER_FORMATS = {"jpeg", "png", "webp", "gif", "bmp", "avif"}


def image_row(image_id: int):
    r = db.q1("SELECT * FROM images WHERE id=?", (image_id,))
    if r is None:
        raise HTTPException(404, "изображение не найдено")
    return r


def viewable_path(r) -> tuple[str, str]:
    """Путь к файлу, который понимает браузер (tif, heic конвертируются в кэшированный png)."""
    p = Path(r["path"])
    if not p.exists():
        raise HTTPException(404, "файл не найден на диске")
    if (r["format"] or "") in BROWSER_FORMATS:
        return str(p), mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    out = get_settings().derived_dir / f"view_{r['id']}_{int(r['mtime'] or 0)}.png"
    if not out.exists():
        open_rgb(p).save(out, "PNG")
    return str(out), "image/png"


def new_derived_path(stem: str, ext: str) -> Path:
    import time

    d = get_settings().derived_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{int(time.time() * 1000)}.{ext}"


def register_derived(path: Path) -> int:
    from .. import scanner

    f = scanner.add_folder(str(path.parent), watched=False)
    ids = scanner.import_files([str(path)], folder_id=f)
    from ..pipeline import get_pipeline

    get_pipeline().kick()
    return ids[0]


def pil_open(r, frame=0) -> Image.Image:
    return open_rgb(r["path"], frame=frame)
