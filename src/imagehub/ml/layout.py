"""Детектор макета страницы (DocLayout-YOLO DocStructBench, ONNX, CPU): блоки текста, таблиц, формул."""
import numpy as np
import onnxruntime as ort
from PIL import Image

from .weights import WEIGHTS

MODEL = str(WEIGHTS / "doclayout_yolo" / "onnx" / "model.onnx")
NAMES = {0: "title", 1: "text", 2: "abandon", 3: "figure", 4: "figure_caption", 5: "table",
         6: "table_caption", 7: "table_footnote", 8: "formula", 9: "formula_caption"}
SIDE = 1024
_sess = None


def _session(threads=6):
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False  # динамические размеры входа раздувают арену
        so.enable_mem_pattern = False
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _sess


def detect(img: Image.Image, conf=0.25):
    """-> список {'cls', 'box': (x1,y1,x2,y2) в пикселях исходного изображения, 'score'}, отсортирован по порядку чтения."""
    w, h = img.size
    r = min(SIDE / w, SIDE / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (SIDE, SIDE), (114, 114, 114))
    px, py = (SIDE - nw) // 2, (SIDE - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    s = _session()
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0]  # (300, 6): x1 y1 x2 y2 score cls
    res = []
    for x1, y1, x2, y2, sc, c in out:
        if sc < conf:
            continue
        box = tuple(float(v) for v in ((x1 - px) / r, (y1 - py) / r, (x2 - px) / r, (y2 - py) / r))
        box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        res.append({"cls": NAMES.get(int(c), str(int(c))), "box": box, "score": float(sc)})
    return reading_order(res, w)


def reading_order(blocks, width):
    """Одна колонка: сверху вниз. Две колонки: слева направо, внутри колонки сверху вниз."""
    if not blocks:
        return blocks
    mid = width / 2
    wide = [b for b in blocks if (b["box"][2] - b["box"][0]) > 0.6 * width]
    narrow = [b for b in blocks if b not in wide]
    left = [b for b in narrow if (b["box"][0] + b["box"][2]) / 2 < mid]
    right = [b for b in narrow if b not in left]
    two_cols = len(left) >= 2 and len(right) >= 2
    if not two_cols:
        return sorted(blocks, key=lambda b: (b["box"][1], b["box"][0]))
    key = lambda b: b["box"][1]
    out, rest = [], sorted(wide, key=key)
    # широкие блоки делят страницу на секции, внутри секции колонки идут по очереди
    bounds = [0] + [b["box"][1] for b in rest] + [1e9]
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        sec = lambda cols: sorted([b for b in cols if lo <= b["box"][1] < hi], key=key)
        out += sec(left) + sec(right)
        if i < len(rest):
            out.append(rest[i])
    return out
