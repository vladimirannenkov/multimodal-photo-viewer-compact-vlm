"""Florence-2-large: описание, объекты (рамки), регионы. Пакетная обработка по 8 изображений (в 2 раза быстрее, чем по одному)."""
import time

import torch
from PIL import ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from common import model_path

TASKS = ["<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"]
BATCH = 8


def run(mid, cfg, imgs, device, ctx):
    p = model_path(mid, cfg)
    model = Florence2ForConditionalGeneration.from_pretrained(
        p, dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    proc = AutoProcessor.from_pretrained(p)
    items = [{} for _ in imgs]
    t0 = time.time()
    for task in TASKS:
        for i in range(0, len(imgs), BATCH):
            chunk = imgs[i:i + BATCH]
            inp = proc(text=[task] * len(chunk), images=chunk, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=512, num_beams=3, do_sample=False, use_cache=True)
            for j, (text, im) in enumerate(zip(proc.batch_decode(ids, skip_special_tokens=False), chunk)):
                items[i + j][task] = proc.post_process_generation(text, task=task, image_size=im.size)
    sec = round((time.time() - t0) / max(1, len(imgs)), 2)
    res = {"images": items}
    for n, (img, item) in enumerate(zip(imgs, items)):
        item["_sec"] = sec
        draw_img = img.copy()
        d = ImageDraw.Draw(draw_img)
        od = item["<OD>"]["<OD>"]
        for box, label in zip(od["bboxes"], od["labels"]):
            d.rectangle(box, outline="red", width=3)
            d.text((box[0] + 3, box[1] + 3), label, fill="red")
        draw_img.save(ctx["out"] / f"boxes_{mid}_{n + 1}.png")
    res["markdown"] = "\n\n".join(
        f"### Изображение {i + 1}\n**Caption:** {r['<MORE_DETAILED_CAPTION>']['<MORE_DETAILED_CAPTION>']}\n\n"
        f"**Объекты:** {r['<OD>']['<OD>']['labels']}\n\n**Регионы:** {r['<DENSE_REGION_CAPTION>']['<DENSE_REGION_CAPTION>']['labels']}"
        for i, r in enumerate(res["images"]))
    return res
