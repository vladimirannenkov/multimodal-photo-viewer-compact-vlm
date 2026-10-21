"""Фоновая обработка изображений волнами: каждая модель загружается один раз на пачку изображений.

Стадии: embed (SigLIP2: вектор, тип, событие) -> textdet (CPU) -> ocr (макет + GLM-OCR/PaddleOCR-VL)
-> scene (Florence-2) -> faces (InsightFace) -> fix, translate, chart (одна Qwen3-VL-2B) -> textemb (e5) + FTS."""

import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import db, people
from .config import get_settings
from .imaging import open_rgb

log = logging.getLogger("imagehub.pipeline")
VERSION = 2
WAVE = 16
DOC_KINDS = {"document", "receipt", "table", "formula"}
OCR_KINDS = DOC_KINDS | {"screenshot", "chart", "diagram", "meme"}
MAX_SIDE = 2048
KIND_RU = {
    "photo": "фото",
    "screenshot": "скриншот",
    "document": "документ",
    "receipt": "чек",
    "chart": "график",
    "heatmap": "карта",
    "diagram": "схема",
    "formula": "формула",
    "table": "таблица",
    "meme": "мем или рисунок",
}
TEXTY = ("document", "screenshot", "receipt", "table")


def refine_kind(kind, scores, td):
    """Zero-shot по картинке путает страницы текста с формулами и графиками. Число найденных строк текста уточняет тип."""
    lines = td.get("lines", 0)
    if kind in ("document", "receipt", "table", "screenshot") and lines < 3 and scores.get("photo", 0) >= 0.1:
        return "photo"  # страница или чек без текста — это фото
    if kind in ("heatmap", "diagram", "meme") and scores.get(kind, 1) < 0.85 and scores.get("photo", 0) >= 0.15:
        kind = "photo"  # редкие типы принимаем только при высокой уверенности, иначе это обычное фото
    if lines >= 7 and kind in ("formula", "photo", "meme", "diagram", "heatmap"):
        return max(TEXTY, key=lambda k: scores.get(k, 0))
    if (
        lines >= 30 and kind == "chart" and scores.get("chart", 1) < 0.9
    ):  # у графиков много подписей: переопределяем только при слабой уверенности
        return max(TEXTY, key=lambda k: scores.get(k, 0))
    return kind


def scale_boxes(boxes, k):
    return [dict(b, box=[round(v * k, 1) for v in b["box"]]) if "box" in b else b for b in boxes] if k != 1 else boxes


class Pipeline:
    def __init__(self):
        self.s = get_settings()
        self.device = self.s.resolve_device()
        self.thread = None
        self.stop_flag = threading.Event()
        self.wake = threading.Event()
        self.paused = False
        self.state: dict[str, Any] = {"stage": "idle", "done": 0, "total": 0, "current": None, "last_error": None, "sec": {}}
        self.listeners = []

    # ---------- управление ----------
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_flag.clear()
        db.ex("UPDATE images SET status='new' WHERE status='processing'")
        # до версии 2 рамки сохранялись в координатах уменьшенной копии: у больших изображений пересчитываем
        db.ex(
            "UPDATE images SET status='new' WHERE deleted=0 AND pipeline_ver<2 AND MAX(COALESCE(width,0), COALESCE(height,0))>?",
            (MAX_SIDE,),
        )  # прерванная волна обрабатывается заново
        self.thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.wake.set()

    def kick(self):
        self.wake.set()

    def queue_counts(self):
        rows = db.q("SELECT status, COUNT(*) c FROM images WHERE deleted=0 GROUP BY status")
        return {r["status"]: r["c"] for r in rows}

    def status(self):
        c = self.queue_counts()
        return {
            **self.state,
            "paused": self.paused,
            "device": self.device,
            "counts": c,
            "pending": c.get("new", 0) + c.get("queued", 0),
            "alive": bool(self.thread and self.thread.is_alive()),
        }

    # ---------- цикл ----------
    def _loop(self):
        while not self.stop_flag.is_set():
            if self.paused:
                self.state["stage"] = "paused"
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            rows = db.q(
                """SELECT id FROM images WHERE deleted=0 AND status IN ('new','queued')
                           ORDER BY priority DESC, id LIMIT ?""",
                (WAVE,),
            )
            if not rows and self.backfill_translation():
                continue
            if not rows:
                self.state.update(stage="idle", current=None, done=0, total=0)
                self.wake.wait(2.0)
                self.wake.clear()
                continue
            ids = [r["id"] for r in rows]
            try:
                self.run_wave(ids)
            except Exception:
                log.error("wave failed\n%s", traceback.format_exc())
                self.state["last_error"] = traceback.format_exc().splitlines()[-1]
                db.exmany(
                    "UPDATE images SET status='error', error=? WHERE id=? AND status<>'done'",
                    [(self.state["last_error"], i) for i in ids],
                )

    def backfill_translation(self) -> bool:
        """Переводит описания у изображений, обработанных до появления перевода. True, если была работа."""
        rows = db.q(
            "SELECT id, caption, tags FROM images WHERE deleted=0 AND status='done' AND caption IS NOT NULL AND caption<>'' AND summary_ru IS NULL AND (error IS NULL OR error NOT LIKE '%translate%') LIMIT 16"
        )
        if not rows:
            return False
        from .ml import manager, textfix

        self.state.update(stage="translate", total=len(rows), done=0)
        texts, slots = [], []
        for r in rows:
            texts.append(r["caption"])
            slots.append((r["id"], "s"))
            tg = json.loads(r["tags"]) if r["tags"] else []
            if tg:
                texts.append(", ".join(tg))
                slots.append((r["id"], "t"))
        with manager.gpu():
            manager.ensure("vqa")
            tr = textfix.translate_ru(texts, device=self.device)
        res = {}
        for (i, k), t in zip(slots, tr):
            res.setdefault(i, {})[k] = t
        for r in rows:
            x = res.get(r["id"], {})
            tags = [t.strip() for t in x.get("t", "").split(",") if t.strip()]
            db.ex(
                "UPDATE images SET summary_ru=?, tags_ru=? WHERE id=?",
                (x.get("s") or "", json.dumps(tags, ensure_ascii=False) if tags else None, r["id"]),
            )
            db.fts_update(r["id"])
        manager.trim_memory()
        for fn in self.listeners:
            fn([r["id"] for r in rows])
        return True

    # ---------- волна ----------
    def run_wave(self, ids):
        from .ml import manager

        self.state.update(total=len(ids), done=0)
        db.exmany("UPDATE images SET status='processing' WHERE id=?", [(i,) for i in ids])
        items = []
        for i in ids:
            r = db.q1("SELECT * FROM images WHERE id=?", (i,))
            try:
                im = open_rgb(r["path"], max_side=MAX_SIDE)
                k = (r["width"] / im.width) if r["width"] and im.width else 1.0  # масштаб рамок к размеру оригинала
                items.append({"id": i, "row": r, "img": im, "res": {}, "t": {}, "k": k})
            except Exception as e:
                db.ex("UPDATE images SET status='error', error=? WHERE id=?", (f"open: {e}", i))
        if not items:
            return
        for name, fn in (
            ("embed", self.st_embed),
            ("textdet", self.st_textdet),
            ("scene", self.st_scene),
            ("ocr", self.st_ocr),
            ("faces", self.st_faces),
            ("fix", self.st_fix),
            ("translate", self.st_translate),
            ("chart", self.st_chart),
            ("textemb", self.st_textemb),
        ):
            self.state["stage"] = name
            t0 = time.time()
            try:
                fn(items)  # замок GPU берётся внутри стадий на каждую пачку, чтобы чат пользователя мог вклиниться
            except Exception:
                err = traceback.format_exc()
                log.error("stage %s failed\n%s", name, err)
                self.state["last_error"] = f"{name}: {err.splitlines()[-1]}"
                for it in items:
                    it["res"].setdefault("errors", []).append(f"{name}: {err.splitlines()[-1]}")
            self.state["sec"][name] = round((time.time() - t0) / max(1, len(items)), 3)
            self.state["done"] = len(items)
        for it in items:
            try:
                self._auto_categories(it)
            except Exception:
                pass
            self._finish(it)
        self.state["current"] = None
        for it in items:
            it["img"] = None
        items.clear()
        manager.trim_memory()
        for fn in self.listeners:
            try:
                fn(ids)
            except Exception:
                pass

    def _finish(self, it):
        i, res = it["id"], it["res"]
        errs = res.get("errors")
        db.ex(
            "UPDATE images SET status=?, error=?, pipeline_ver=?, timings=?, priority=0 WHERE id=?",
            ("done", "; ".join(errs) if errs else None, VERSION, json.dumps(it["t"]), i),
        )
        db.fts_update(i)

    # ---------- стадии ----------
    def st_embed(self, items):
        from .ml import manager, siglip

        with manager.gpu():
            m = siglip.get(self.device)
            t0 = time.time()
            feats = m.images([it["img"] for it in items])
            cls = m.classify(feats)
        for it, f, c in zip(items, feats, cls):
            db.set_vector(it["id"], "img", f)
            it["res"]["kind"], it["res"]["event"], it["res"]["kind_scores"] = c["kind"], c["event"], c["kind_scores"]
            db.ex(
                "UPDATE images SET kind=?, kind_scores=?, event=? WHERE id=?",
                (c["kind"], json.dumps(c["kind_scores"]), c["event"] if c["kind"] in ("photo", "meme") else None, it["id"]),
            )
            it["t"]["embed"] = round((time.time() - t0) / len(items), 3)

    def st_textdet(self, items):
        from .ml import textdet

        def one(it):
            t = time.time()
            r = textdet.detect(it["img"])
            it["res"]["textdet"] = r
            it["t"]["textdet"] = round(time.time() - t, 3)
            k = refine_kind(it["res"]["kind"], it["res"].get("kind_scores", {}), r)
            if k != it["res"]["kind"]:
                it["res"]["kind"] = k
                db.ex("UPDATE images SET kind=?, event=NULL WHERE id=?", (k, it["id"]))
            db.ex(
                "UPDATE images SET has_text=?, text_lang=?, text_rotation=?, text_sample=? WHERE id=?",
                (int(r["has_text"]), r["language"], r["rotation"], r["sample"], it["id"]),
            )

        with ThreadPoolExecutor(2) as ex:
            list(ex.map(one, items))

    def _needs_ocr(self, it):
        k = it["res"].get("kind")
        td = it["res"].get("textdet", {})
        return td.get("has_text") or k in DOC_KINDS

    def st_ocr(self, items):
        from .ml import docparse, manager

        todo = [it for it in items if self._needs_ocr(it)]
        if not todo:
            return
        for s in range(0, len(todo), 8):
            chunk = todo[s : s + 8]
            imgs = []
            for it in chunk:
                im = it["img"]
                if it["res"].get("textdet", {}).get("rotation") == 90:
                    im = im.rotate(90, expand=True)
                imgs.append(im)
            hints = [it["res"]["kind"] if it["res"]["kind"] in ("formula", "table") else None for it in chunk]
            t0 = time.time()
            with manager.gpu():
                manager.ensure("ocr")
                out = docparse.parse_pages(imgs, hints=hints if any(hints) else None, fix=False)
            for it, o in zip(chunk, out):
                o["blocks"] = [
                    dict(b, box=[round(v * it["k"]) for v in b["box"]])
                    if b.get("box") and it["res"].get("textdet", {}).get("rotation", 0) == 0
                    else b
                    for b in o["blocks"]
                ]
                it["res"]["ocr"] = o
                it["t"]["ocr"] = round((time.time() - t0) / len(chunk), 2)
                db.ex(
                    "UPDATE images SET ocr_raw=?, ocr_md=?, ocr_blocks=? WHERE id=?",
                    (o["markdown"], o["markdown"], json.dumps(o["blocks"], ensure_ascii=False), it["id"]),
                )

    def st_fix(self, items):
        if not self.s.ocr_fix:
            return
        from .ml import docparse, manager, textfix

        targets = []
        for it in items:
            o = it["res"].get("ocr")
            if o:
                targets += [(it, b) for b in o["blocks"] if b.get("kind") == "text" and len(b.get("text", "")) >= 15]
        if not targets:
            return
        t0 = time.time()
        fixed = []
        for k in range(0, len(targets), 16):
            with manager.gpu():
                manager.ensure("vqa")
                fixed += textfix.correct_texts([b["text"] for _, b in targets[k : k + 16]], device=self.device)[0]
        touched = set()
        for (it, b), f in zip(targets, fixed):
            if f != b["text"]:
                b["text_raw"], b["text"] = b["text"], f
                touched.add(it["id"])
        for it in items:
            o = it["res"].get("ocr")
            if o and it["id"] in touched:
                md = docparse.assemble(o["blocks"])
                o["markdown"] = md
                db.ex(
                    "UPDATE images SET ocr_md=?, ocr_blocks=? WHERE id=?",
                    (md, json.dumps(o["blocks"], ensure_ascii=False), it["id"]),
                )
        for it in items:
            it["t"]["fix"] = round((time.time() - t0) / len(items), 3)

    def st_translate(self, items):
        """Русский перевод описания и тегов: поиск работает на русском и английском."""
        from .ml import manager, textfix

        todo = [it for it in items if it["res"].get("scene")]
        if (
            not todo or len(items) <= 3
        ):  # вставка из буфера и единичные файлы готовы без ожидания, перевод доделает backfill_translation
            return
        texts, slots = [], []
        for it in todo:
            sc = it["res"]["scene"]
            texts.append(sc["caption"])
            slots.append((it, "s"))
            if it["res"].get("tags"):
                texts.append(", ".join(it["res"]["tags"]))
                slots.append((it, "t"))
        res = {}
        for k in range(0, len(texts), 16):
            with manager.gpu():
                manager.ensure("vqa")
                tr = textfix.translate_ru(texts[k : k + 16], device=self.device)
            for (it, kind), r in zip(slots[k : k + 16], tr):
                res.setdefault(it["id"], {})[kind] = r
        for it in todo:
            r = res.get(it["id"], {})
            tags = [x.strip() for x in r.get("t", "").split(",") if x.strip()]
            db.ex(
                "UPDATE images SET summary_ru=?, tags_ru=? WHERE id=?",
                (r.get("s") or None, json.dumps(tags, ensure_ascii=False) if tags else None, it["id"]),
            )

    def st_scene(self, items):
        from .ml import florence, manager

        todo = [it for it in items if it["res"].get("kind") not in DOC_KINDS]
        if not todo:
            return
        t0 = time.time()
        out = []
        for k in range(0, len(todo), 8):
            with manager.gpu():
                manager.ensure("scene")
                out += florence.describe([it["img"] for it in todo[k : k + 8]], self.device)
        for it, o in zip(todo, out):
            o["objects"] = scale_boxes(o["objects"], it["k"])
            o["regions"] = scale_boxes(o["regions"], it["k"])
            it["res"]["scene"] = o
            tags = sorted({x["label"] for x in o["objects"]})[:12]
            it["res"]["tags"] = tags
            db.ex(
                "UPDATE images SET caption=?, summary=?, objects=?, regions=?, tags=? WHERE id=?",
                (
                    o["caption"],
                    o["caption"],
                    json.dumps(o["objects"], ensure_ascii=False),
                    json.dumps(o["regions"], ensure_ascii=False),
                    json.dumps(tags, ensure_ascii=False),
                    it["id"],
                ),
            )
            it["t"]["scene"] = round((time.time() - t0) / len(todo), 3)

    def _auto_categories(self, it):
        names = [KIND_RU.get(it["res"].get("kind"), it["res"].get("kind"))]
        if it["res"].get("kind") == "photo" and it["res"].get("event"):
            names.append(it["res"]["event"])
        names += it["res"].get("tags", [])[:4]
        db.ex("DELETE FROM image_categories WHERE image_id=? AND source='auto'", (it["id"],))
        for n in filter(None, names):
            c = db.q1("SELECT id FROM categories WHERE name=?", (n,))
            cid = c["id"] if c else db.ex("INSERT INTO categories(name, auto) VALUES(?,1)", (n,)).lastrowid
            db.ex(
                "INSERT OR IGNORE INTO image_categories(image_id, category_id, score, source) VALUES(?,?,1,'auto')",
                (it["id"], cid),
            )

    def st_faces(self, items):
        from .ml import faces

        todo = [it for it in items if it["res"].get("kind") in ("photo", "screenshot", "meme")]
        for it in todo:
            t0 = time.time()
            found = faces.detect(it["img"], device="cpu")
            for f in found:
                f["box"] = [v * it["k"] for v in f["box"]]
            people.add_faces(it["id"], found)
            it["t"]["faces"] = round(time.time() - t0, 3)

    def st_chart(self, items):
        from .ml import manager, vqa

        todo = [it for it in items if it["res"].get("kind") == "chart"]
        if not todo:
            return
        for it in todo:
            t0 = time.time()
            exp = self.ensure_experiment(
                it["row"]
            )  # первый эксперимент создаётся по первому графику, даже если извлечение не удалось
            db.ex("UPDATE images SET experiment_id=COALESCE(experiment_id, ?) WHERE id=?", (exp, it["id"]))
            with manager.gpu():
                manager.ensure("vqa")
                txt = vqa.ask(it["img"], vqa.CHART_PROMPT, self.device, max_new_tokens=900)
            data = vqa.extract_json(txt) or {}
            if data.get("chart") is False:
                continue
            data["stats"] = vqa.chart_stats(data)
            data["raw"] = txt if not data.get("series") else None
            summary = self._chart_summary(data)
            exp = self.ensure_experiment(it["row"])
            db.ex(
                "UPDATE images SET chart=?, experiment_id=?, summary=COALESCE(summary,'') || ? WHERE id=?",
                (json.dumps(data, ensure_ascii=False), exp, ("\n" + summary) if summary else "", it["id"]),
            )
            it["t"]["chart"] = round(time.time() - t0, 2)

    @staticmethod
    def _chart_summary(d):
        parts = [d.get("title") or "график"]
        if d.get("x_axis") or d.get("y_axis"):
            parts.append(f"оси: {d.get('x_axis', '')} / {d.get('y_axis', '')}")
        st = d.get("stats", {})
        for s in st.get("series", [])[:4]:
            p = f"серия {s['name']}: максимум {s['max']:.4g} при x={s['max_at']:.4g}"
            if s["plateau"] is not None:
                p += f", плато около {s['plateau']:.4g}"
            parts.append(p)
        if st.get("best_series"):
            parts.append(f"лучшая серия: {st['best_series']}")
        return ". ".join(parts)

    def ensure_experiment(self, row):
        """Первый эксперимент создаётся автоматически по дате первого графика. Дальше новые создаёт только пользователь."""
        e = db.q1("SELECT id FROM experiments ORDER BY created_at DESC, id DESC LIMIT 1")
        if e:
            return e["id"]
        ts = row["taken_at"] or row["mtime"] or time.time()
        name = "Эксперимент " + time.strftime("%Y-%m-%d", time.localtime(ts))
        return db.ex("INSERT INTO experiments(name, created_at, auto) VALUES(?,?,1)", (name, ts)).lastrowid

    def st_textemb(self, items):
        from .ml import textembed

        texts, ids = [], []
        for it in items:
            r = db.q1("SELECT summary, ocr_md, tags, event FROM images WHERE id=?", (it["id"],))
            t = " ".join(filter(None, [r["summary"], r["event"], r["tags"], (r["ocr_md"] or "")[:1500]])).strip()
            if t:
                texts.append(t)
                ids.append(it["id"])
        if texts:
            for i, v in zip(ids, textembed.passages(texts)):
                db.set_vector(i, "txt", v)


_pipe = None


def get_pipeline() -> Pipeline:
    global _pipe
    if _pipe is None:
        _pipe = Pipeline()
    return _pipe
