"""SigLIP2 base: эмбеддинги изображений для поиска, тип изображения и событие (zero-shot), текстовый энкодер для запросов."""

import gc
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from .weights import WEIGHTS

KINDS = {  # тип -> промпты (берётся максимум по промптам)
    "photo": [
        "a photo of food on a plate",
        "a photo of people",
        "a photo of a landscape",
        "a photo of an object",
        "a photo of an animal",
        "a photo of a city street",
    ],
    "screenshot": ["a screenshot of an app", "a screenshot of a website", "a screenshot of a chat or messenger"],
    "document": ["a scanned document with text", "a page of a book or article", "a photo of a printed page"],
    "receipt": ["a photo of a shop receipt", "a restaurant bill"],
    "chart": ["a chart or plot of metrics", "a line plot of training loss", "a bar chart", "a graph with axes and a legend"],
    "heatmap": ["a heatmap or confusion matrix", "an attention map"],
    "diagram": ["a diagram or flowchart", "a schematic with boxes and arrows"],
    "formula": ["an image of a mathematical formula", "handwritten or printed equation"],
    "table": ["an image of a table with rows and columns", "a spreadsheet"],
    "meme": ["a meme with a caption", "a cartoon or illustration"],
}
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
_inst = {}
_lock = threading.Lock()


def _feat(x):
    return getattr(x, "pooler_output", x)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Siglip:
    def __init__(self, device):
        self.device = device
        p = str(WEIGHTS / "siglip2")
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(p, dtype=self.dtype, device_map=device, low_cpu_mem_usage=True).eval()
        self.proc = AutoProcessor.from_pretrained(p)
        self._labels = None

    @torch.no_grad()
    def text(self, texts):
        tin = self.proc(text=list(texts), padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        f = _feat(self.model.get_text_features(**tin)).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def images(self, imgs, batch=16):
        out = []
        for i in range(0, len(imgs), batch):
            pin = self.proc(images=imgs[i : i + batch], return_tensors="pt").to(self.device, self.dtype)
            f = _feat(self.model.get_image_features(**pin)).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def _label_embs(self):
        if self._labels is None:
            flat = [p for ps in KINDS.values() for p in ps]
            idx, k = {}, 0
            for name, ps in KINDS.items():
                idx[name] = (k, k + len(ps))
                k += len(ps)
            self._labels = (self.text(flat), idx, self.text(list(EVENTS.values())))
        return self._labels

    def classify(self, feats):
        """-> по каждому изображению: тип, оценки типов, событие, топ-3 событий."""
        kt, idx, et = self._label_embs()
        names, ev = list(KINDS), list(EVENTS)
        res = []
        for f in feats:
            sims = kt @ f
            pk = _softmax(100 * np.array([sims[a:b].max() for a, b in (idx[n] for n in names)]))
            pe = _softmax(100 * (et @ f))
            top = np.argsort(-pe)[:3]
            res.append(
                {
                    "kind": names[int(pk.argmax())],
                    "kind_scores": {n: round(float(p), 3) for n, p in zip(names, pk)},
                    "event": ev[int(top[0])],
                    "event_top3": [(ev[int(j)], round(float(pe[j]), 3)) for j in top],
                }
            )
        return res


def get(device) -> Siglip:
    with _lock:
        if device not in _inst:
            _inst[device] = Siglip(device)
        return _inst[device]


def free(device=None):
    with _lock:
        for d in [device] if device else list(_inst):
            _inst.pop(d, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
