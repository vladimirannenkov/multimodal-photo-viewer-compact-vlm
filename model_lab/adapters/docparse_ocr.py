"""Разбор страницы: макет (CPU) + GLM-OCR для текста + PaddleOCR-VL для таблиц и формул, на выходе Markdown."""
import time

import torch

import docparse


def run(mid, cfg, imgs, device, ctx):
    hints = ctx.get("hints")
    docparse.parse_pages(imgs[:1], hints=hints[:1] if hints else None)  # прогрев моделей
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    pages = docparse.parse_pages(imgs, device=device, batch=8, hints=hints)
    if device == "cuda":
        torch.cuda.synchronize()
    sec = round((time.time() - t) / max(1, len(imgs)), 2)
    items = [{"markdown": p["markdown"], "blocks": p["blocks"], "_sec": sec} for p in pages]
    return {"images": items, "markdown": "\n\n---\n\n".join(p["markdown"] for p in pages)}
