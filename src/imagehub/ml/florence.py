"""Florence-2-large: подробное описание, объекты с рамками, регионы. Пакеты по 8 изображений, sdpa, fp16."""
import gc

import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from .weights import WEIGHTS

TASKS = ["<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"]
BATCH = 8
_M = {}


def _load(device):
    if "m" not in _M:
        p = str(WEIGHTS / "florence2")
        dt = torch.float16 if device == "cuda" else torch.float32
        _M["m"] = Florence2ForConditionalGeneration.from_pretrained(p, dtype=dt, attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True).eval()
        _M["p"] = AutoProcessor.from_pretrained(p)
        _M["dt"] = dt
    return _M["m"], _M["p"], _M["dt"]


def free():
    _M.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe(imgs, device, tasks=TASKS):
    """-> по изображению: caption, objects [{label, box}], regions [{label, box}] (рамки в пикселях исходного изображения)."""
    model, proc, dt = _load(device)
    items = [{} for _ in imgs]
    for task in tasks:
        for i in range(0, len(imgs), BATCH):
            chunk = imgs[i:i + BATCH]
            inp = proc(text=[task] * len(chunk), images=chunk, return_tensors="pt").to(device, dt)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=512, num_beams=3, do_sample=False, use_cache=True)
            for j, (text, im) in enumerate(zip(proc.batch_decode(ids, skip_special_tokens=False), chunk)):
                text = text.replace("<pad>", "")  # padding пакета не должен попадать в текст описания
                items[i + j][task] = proc.post_process_generation(text, task=task, image_size=im.size)[task]
    out = []
    for it in items:
        od, dr = it.get("<OD>", {}), it.get("<DENSE_REGION_CAPTION>", {})
        out.append({
            "caption": it.get("<MORE_DETAILED_CAPTION>", "").strip(),
            "objects": [{"label": l, "box": [round(v, 1) for v in b]} for l, b in zip(od.get("labels", []), od.get("bboxes", []))],
            "regions": [{"label": l, "box": [round(v, 1) for v in b]} for l, b in zip(dr.get("labels", []), dr.get("bboxes", []))],
        })
    return out
