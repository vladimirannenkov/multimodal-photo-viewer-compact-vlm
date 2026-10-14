"""Qwen3-VL-2B: вопросы по изображению (чат), JSON графика. Единственная модель для диалога."""
import json
import re

from . import vlm
from .weights import WEIGHTS

CHART_PROMPT = ('Если это график, верни только JSON: {"title": str, "x_axis": str, "y_axis": str, '
                '"series": [{"name": str, "points": [[x, y], ...]}]}. Не менее 8 точек на серию, значения читай по осям. '
                'Если это не график, верни {"chart": false}.')
FACE_PROMPT = "Кто может быть этот человек по контексту сцены? Предложи 3 коротких варианта роли (например: студент, прохожий, коллега) через запятую."


def path():
    return str(WEIGHTS / "qwen3vl")


STYLE = "Отвечай кратко и по существу, на языке вопроса. "


def warmup(device):
    from . import vlm
    vlm.load(path(), device, True)


def ask(img, question, device, history=None, max_new_tokens=320):
    hist = [{"role": h["role"], "content": [{"type": "text", "text": h["content"]}]} for h in (history or [])][-8:]
    return vlm.generate(path(), [img], [(STYLE if not hist else "") + question], device, max_new_tokens=max_new_tokens, history=hist, max_side=1024)[0]


def ask_stream(img, question, device, history=None, max_new_tokens=320):
    hist = [{"role": h["role"], "content": [{"type": "text", "text": h["content"]}]} for h in (history or [])][-8:]
    return vlm.stream(path(), img, (STYLE if not hist else "") + question, device, max_new_tokens=max_new_tokens, history=hist)


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def chart_stats(chart):
    """Максимум, плато и лучшая серия считаются кодом по извлечённым точкам, а не выдумываются моделью."""
    out = []
    for s in (chart or {}).get("series", []) or []:
        pts = [(float(x), float(y)) for x, y in (s.get("points") or []) if _num(x) and _num(y)]
        if len(pts) < 3:
            continue
        ys = [y for _, y in pts]
        tail = ys[-max(3, len(ys) // 5):]
        mean_tail = sum(tail) / len(tail)
        spread = (max(tail) - min(tail)) / (abs(mean_tail) + 1e-9)
        out.append({"name": s.get("name", ""), "max": max(ys), "max_at": pts[ys.index(max(ys))][0], "min": min(ys),
                    "final": ys[-1], "plateau": round(mean_tail, 4) if spread < 0.05 else None})
    best = max(out, key=lambda s: s["max"]) if out else None
    return {"series": out, "best_series": best["name"] if best else None}


def _num(v):
    try:
        float(v)
        return True
    except Exception:
        return False
