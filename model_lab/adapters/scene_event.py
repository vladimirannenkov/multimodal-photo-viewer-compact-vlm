"""Событие сцены: zero-shot SigLIP2 по списку ситуаций. Описание берётся из Florence-2, теги из найденных объектов.
Текстовая LLM для этого не нужна (Qwen3-1.7B определял событие хуже: 13 из 30 снимков «праздник»)."""
import time

import torch
from transformers import AutoModel, AutoProcessor

from common import model_path

EVENTS = {
    "праздник": "a photo of a party or celebration with people, decorations, cake or gifts",
    "прогулка": "a photo of people walking or relaxing outdoors",
    "работа": "a photo of people working at an office, workshop or construction site",
    "спорт": "a photo of people doing sports or physical activity",
    "путешествие": "a travel photo of a famous landmark or tourist place",
    "бытовая сцена": "a photo of an everyday indoor scene at home",
    "природа": "a photo of nature, landscape, mountains, sea, forest or sky",
    "животные": "a photo of animals",
    "город": "a photo of a city street, buildings or architecture",
    "еда": "a photo of food or drinks on a table",
    "транспорт": "a photo of cars, trains, planes or other transport",
}


def run(mid, cfg, imgs, device, ctx):
    src = ctx["previous"].get("florence2")
    if not src:
        return {"markdown": "Нужен результат florence2 в том же прогоне"}
    p = model_path("siglip2", {"repo": "google/siglip2-base-patch16-256"})
    model = AutoModel.from_pretrained(p, dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(p)
    labels = list(EVENTS)
    t0 = time.time()
    with torch.no_grad():
        tin = proc(text=list(EVENTS.values()), padding="max_length", max_length=64, return_tensors="pt").to(device)
        tf = model.get_text_features(**tin)
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        items = []
        for i in range(0, len(imgs), 16):
            pin = proc(images=imgs[i:i + 16], return_tensors="pt").to(device, torch.float16)
            vf = model.get_image_features(**pin)
            vf = getattr(vf, "pooler_output", vf)
            vf = vf / vf.norm(dim=-1, keepdim=True)
            probs = (100 * vf @ tf.T.to(vf.dtype)).softmax(dim=-1).float().cpu()
            for row in probs:
                top = row.argsort(descending=True)[:3]
                items.append({"event": labels[int(top[0])], "event_top3": [(labels[int(j)], round(float(row[j]), 2)) for j in top]})
    sec = round((time.time() - t0) / max(1, len(imgs)), 3)
    for it, f in zip(items, src["images"]):
        it["summary"] = f["<MORE_DETAILED_CAPTION>"]["<MORE_DETAILED_CAPTION>"]
        it["tags"] = sorted(set(f["<OD>"]["<OD>"]["labels"]))[:10]
        it["_sec"] = sec
    return {"images": items, "markdown": "\n".join(f"{i['event']}: {i['summary']}" for i in items)}
