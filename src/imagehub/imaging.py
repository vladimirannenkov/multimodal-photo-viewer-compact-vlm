"""Работа с файлами изображений: открытие (в т.ч. webp/gif/tif/heic, многокадровые), миниатюры, хеш, EXIF."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import xxhash
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # heic необязателен
    pass

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp", ".heic", ".heif", ".apng", ".avif"}


def is_image(path) -> bool:
    return Path(path).suffix.lower() in EXTS


def file_hash(path, chunk=1 << 20) -> str:
    h = xxhash.xxh3_64()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def open_rgb(path, frame=0, max_side=None) -> Image.Image:
    """Открывает кадр изображения как RGB с учётом EXIF-ориентации. Прозрачность кладётся на белый фон."""
    src = Image.open(path)
    try:
        src.seek(frame)
    except EOFError:
        src.seek(0)
    im: Image.Image = ImageOps.exif_transpose(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def exif_date(im: Image.Image):
    try:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        for v in (sub.get(36867), ex.get(306)):
            if v:
                return datetime.strptime(str(v)[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return None


def prepare(path: str, thumb_path: str, side: int) -> dict:
    """Выполняется в пуле процессов: хеш, размеры, формат, кадры, дата съёмки, миниатюра."""
    out: dict[str, Any] = {"path": path}
    try:
        out["hash"] = file_hash(path)
        with Image.open(path) as im:
            out["format"] = (im.format or "").lower()
            out["frames"] = getattr(im, "n_frames", 1)
            out["taken_at"] = exif_date(im)
            im2 = ImageOps.exif_transpose(im)
            out["width"], out["height"] = im2.size
        t = open_rgb(path, max_side=side)
        Path(thumb_path).parent.mkdir(parents=True, exist_ok=True)
        t.save(thumb_path, "WEBP", quality=82)
    except Exception as e:  # битый файл не ломает сканирование
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def stat_tuple(p):
    s = os.stat(p)
    return s.st_size, s.st_mtime
