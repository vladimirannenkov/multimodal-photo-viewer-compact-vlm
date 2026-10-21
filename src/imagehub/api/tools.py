"""Инструменты: правка (копией), экспорт и конвертация, склейка N×M, аннотации-примитивы, сводки."""

import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageOps

from .. import db
from .deps import image_row, new_derived_path, pil_open, register_derived

router = APIRouter(prefix="/api")

FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF", "tif": "TIFF", "tiff": "TIFF", "bmp": "BMP"}


def _save_bytes(im: Image.Image, fmt: str, quality=90) -> bytes:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise HTTPException(400, f"формат {fmt} не поддерживается")
    buf = io.BytesIO()
    if FORMATS[fmt] == "JPEG" and im.mode != "RGB":
        im = im.convert("RGB")
    im.save(buf, FORMATS[fmt], quality=quality) if FORMATS[fmt] in ("JPEG", "WEBP") else im.save(buf, FORMATS[fmt])
    return buf.getvalue()


# ---------- правка: результат всегда новая копия, оригинал не меняется ----------
@router.post("/images/{image_id}/edit")
def edit(image_id: int, body: dict = Body(...)):
    r = image_row(image_id)
    im = pil_open(r)
    for op in body.get("ops", []):
        t = op["type"]
        if t == "rotate":
            im = im.rotate(-float(op["deg"]), expand=True, fillcolor=(255, 255, 255))
        elif t == "flip":
            im = ImageOps.mirror(im) if op.get("axis", "h") == "h" else ImageOps.flip(im)
        elif t == "crop":
            x1, y1, x2, y2 = [int(v) for v in op["box"]]
            im = im.crop((max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)))
        elif t == "resize":
            im = im.resize((int(op["width"]), int(op["height"])), Image.Resampling.LANCZOS)
        elif t == "draw":
            _draw(im, op.get("shapes", []))
    fmt = body.get("format", "png")
    p = new_derived_path(Path(r["name"]).stem, fmt)
    p.write_bytes(_save_bytes(im, fmt, body.get("quality", 92)))
    return {"id": register_derived(p)}


def _hex(c, a=255):
    c = c.lstrip("#")
    if len(c) == 3:  # краткая запись #f00
        c = "".join(ch * 2 for ch in c)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _draw(im: Image.Image, shapes):
    """Примитивы: rect, ellipse, line, arrow, text, freehand. Координаты в пикселях изображения."""
    d = ImageDraw.Draw(im, "RGBA")
    for s in shapes:
        col = _hex(s.get("color", "#e5484d"))
        w = int(s.get("width", 4))
        t = s["type"]
        if t == "rect":
            x1, y1, x2, y2 = s["box"]
            d.rectangle((x1, y1, x2, y2), outline=col, width=w, fill=_hex(s["fill"], 60) if s.get("fill") else None)
        elif t == "ellipse":
            d.ellipse(tuple(s["box"]), outline=col, width=w)
        elif t == "line":
            d.line(tuple(s["points"]), fill=col, width=w)
        elif t == "freehand":
            pts = [tuple(p) for p in s["points"]]
            if len(pts) > 1:
                d.line(pts, fill=col, width=w, joint="curve")
        elif t == "arrow":
            (x1, y1), (x2, y2) = s["points"][0], s["points"][-1]
            d.line((x1, y1, x2, y2), fill=col, width=w)
            import math

            ang = math.atan2(y2 - y1, x2 - x1)
            L = max(14, w * 4)
            for da in (math.pi - 0.45, math.pi + 0.45):
                d.line((x2, y2, x2 + L * math.cos(ang + da), y2 + L * math.sin(ang + da)), fill=col, width=w)
        elif t == "text":
            d.text(tuple(s["at"]), s["text"], fill=col, font_size=int(s.get("size", 28)))


@router.put("/images/{image_id}/annotations")
def save_annotations(image_id: int, body: dict = Body(...)):
    """Слой аннотаций поверх изображения хранится отдельно; оригинал не затрагивается."""
    image_row(image_id)
    db.ex("DELETE FROM annotations WHERE image_id=?", (image_id,))
    db.ex(
        "INSERT INTO annotations(image_id, data, updated_at) VALUES(?,?,?)",
        (image_id, json.dumps(body.get("shapes", [])), time.time()),
    )
    return {"ok": True}


@router.post("/images/{image_id}/annotations/export")
def export_annotations(image_id: int):
    r = image_row(image_id)
    a = db.q1("SELECT data FROM annotations WHERE image_id=?", (image_id,))
    im = pil_open(r)
    _draw(im, json.loads(a["data"]) if a else [])
    p = new_derived_path(Path(r["name"]).stem + "_marked", "png")
    im.save(p)
    return {"id": register_derived(p)}


# ---------- экспорт и конвертация ----------
@router.post("/export")
def export(body: dict = Body(...)):
    ids = body.get("ids", [])
    fmt = body.get("format", "png").lower()
    max_side = body.get("max_side")
    if not ids:
        raise HTTPException(400, "нечего экспортировать")
    outs = []
    for i in ids:
        r = image_row(i)
        im = pil_open(r)
        if max_side:
            im.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
        outs.append((Path(r["name"]).stem + "." + fmt, _save_bytes(im, fmt, body.get("quality", 90))))
    if len(outs) == 1:
        name, data = outs[0]
        return Response(
            data,
            media_type=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    buf = io.BytesIO()
    used = Counter()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in outs:
            used[name] += 1
            z.writestr(name if used[name] == 1 else f"{Path(name).stem}_{used[name]}{Path(name).suffix}", data)
    return Response(
        buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="export.zip"'}
    )


# ---------- склейка N×M ----------
@router.post("/merge")
def merge(body: dict = Body(...)):
    """Сетка rows×cols из выбранных изображений (ячейка cell px, отступ gap). save=true: результат попадает в каталог как новая копия."""
    ids = body.get("ids", [])
    rows, cols = int(body.get("rows", 0)), int(body.get("cols", 0))
    n = len(ids)
    if not n:
        raise HTTPException(400, "нужны изображения")
    if not rows and not cols:
        cols = int(n**0.5 + 0.999)
    cols = cols or -(-n // rows)
    rows = rows or -(-n // cols)
    cell, gap = int(body.get("cell", 512)), int(body.get("gap", 8))
    bg = _hex(body.get("bg", "#ffffff"))[:3]
    fit = body.get("fit", "contain")
    sheet = Image.new("RGB", (cols * cell + (cols + 1) * gap, rows * cell + (rows + 1) * gap), bg)
    for k, i in enumerate(ids[: rows * cols]):
        im = pil_open(image_row(i))
        im = (
            ImageOps.fit(im, (cell, cell), Image.Resampling.LANCZOS)
            if fit == "cover"
            else ImageOps.contain(im, (cell, cell), Image.Resampling.LANCZOS)
        )
        r_, c_ = divmod(k, cols)
        sheet.paste(im, (gap + c_ * (cell + gap) + (cell - im.width) // 2, gap + r_ * (cell + gap) + (cell - im.height) // 2))
    if body.get("save"):
        p = new_derived_path("merge", "png")
        sheet.save(p)
        return {"id": register_derived(p), "rows": rows, "cols": cols}
    return Response(_save_bytes(sheet, body.get("format", "png")), media_type="image/png")


# ---------- сводки ----------
@router.post("/summary")
def summary(body: dict = Body(...)):
    """Сводка по выбранным изображениям или всей библиотеке: типы, события, теги, языки текста, период, объём."""
    ids = body.get("ids") or []
    where, args = ("i.id IN (%s)" % ",".join("?" * len(ids)), ids) if ids else ("1=1", [])
    rows = db.q(f"SELECT * FROM images i WHERE i.deleted=0 AND {where}", args)
    if not rows:
        return {"count": 0}
    kinds, events, langs, tags = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        kinds[r["kind"] or "?"] += 1
        if r["event"]:
            events[r["event"]] += 1
        if r["has_text"]:
            langs[r["text_lang"] or "?"] += 1
        if r["tags"]:
            tags.update(json.loads(r["tags"]))
    dates = [r["taken_at"] or r["mtime"] for r in rows if r["taken_at"] or r["mtime"]]
    people = db.q(
        f"""SELECT COALESCE(p.name, p.auto_name) n, COUNT(DISTINCT f.image_id) c FROM faces f JOIN people p ON p.id=f.person_id
                      WHERE f.image_id IN (SELECT i.id FROM images i WHERE i.deleted=0 AND {where}) GROUP BY p.id ORDER BY c DESC LIMIT 10""",
        args,
    )
    return {
        "count": len(rows),
        "bytes": sum(r["size"] or 0 for r in rows),
        "from": min(dates) if dates else None,
        "to": max(dates) if dates else None,
        "kinds": kinds.most_common(),
        "events": events.most_common(6),
        "languages": langs.most_common(),
        "tags": tags.most_common(15),
        "with_text": sum(1 for r in rows if r["has_text"]),
        "people": [(p["n"], p["c"]) for p in people],
    }
