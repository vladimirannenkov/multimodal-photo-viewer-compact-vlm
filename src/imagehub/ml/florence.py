"""Florence-2-large: подробное описание, объекты с рамками, регионы. Пакеты по 8 изображений, sdpa, fp16."""

import gc

import torch
import transformers
from packaging.version import Version
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from ..profiling import tracked
from .weights import WEIGHTS

TASKS = ["<DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"]
BATCH = 8
_M = {}


def _load(device):
    if "m" not in _M:
        p = str(WEIGHTS / "florence2")
        dt = torch.float16 if device == "cuda" else torch.float32
        _M["m"] = Florence2ForConditionalGeneration.from_pretrained(
            p, dtype=dt, attn_implementation="sdpa", device_map=device, low_cpu_mem_usage=True
        ).eval()
        _M["p"] = AutoProcessor.from_pretrained(p)
        _M["dt"] = dt
    return _M["m"], _M["p"], _M["dt"]


def free():
    _M.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@tracked("Florence-2.describe")
def describe(imgs, device, tasks=TASKS):
    """-> по изображению: caption, objects [{label, box}], regions [{label, box}] (рамки в пикселях исходного изображения)."""
    model, proc, dt = _load(device)
    items = [{} for _ in imgs]
    # The image encoder is independent of the task prompt. Reuse its output for
    # the three tasks when this Transformers version supports multimodal caches.
    supports_cache = len(tasks) > 1 and Version(transformers.__version__) >= Version("5.19")
    for i in range(0, len(imgs), BATCH):
        chunk = imgs[i : i + BATCH]
        features = None
        for task in tasks:
            inp = proc(text=[task] * len(chunk), images=chunk, return_tensors="pt").to(device, dt)
            with torch.inference_mode():
                if supports_cache:
                    pixels = inp.pop("pixel_values")
                    if features is None:
                        features = model.get_image_features(pixels)
                    inp["mm_encoder_outputs"] = {"image": features}
                ids = model.generate(**inp, max_new_tokens=512, num_beams=3, do_sample=False, use_cache=True)
            for j, (text, im) in enumerate(zip(proc.batch_decode(ids, skip_special_tokens=False), chunk)):
                text = text.replace("<pad>", "")  # padding пакета не должен попадать в текст описания
                items[i + j][task] = proc.post_process_generation(text, task=task, image_size=im.size)[task]
    out = []
    for it in items:
        od, dr = it.get("<OD>", {}), it.get("<DENSE_REGION_CAPTION>", {})
        out.append(
            {
                "caption": (
                    it.get("<DETAILED_CAPTION>") or it.get("<MORE_DETAILED_CAPTION>") or it.get("<CAPTION>") or ""
                ).strip(),
                "objects": [
                    {"label": l, "box": [round(v, 1) for v in b]} for l, b in zip(od.get("labels", []), od.get("bboxes", []))
                ],
                "regions": [
                    {"label": l, "box": [round(v, 1) for v in b]} for l, b in zip(dr.get("labels", []), dr.get("bboxes", []))
                ],
            }
        )
    return out
