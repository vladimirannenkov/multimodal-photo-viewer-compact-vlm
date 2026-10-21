"""Лица: InsightFace buffalo_l (SCRFD + ArcFace). Эмбеддинги нормированы, сходство = скалярное произведение."""

import numpy as np
from PIL import Image

from ..config import get_settings

_APP = {}


def _app(device):
    if device not in _APP:
        from insightface.app import FaceAnalysis

        root = get_settings().weights_dir / "insightface"
        kw = {"root": str(root)} if root.exists() else {}
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", providers=prov, **kw)
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        _APP[device] = app
    return _APP[device]


def detect(img: Image.Image, device="cpu", min_score=0.5):
    """-> [{box: [x1,y1,x2,y2], score, emb}] в пикселях изображения."""
    faces = _app(device).get(np.array(img)[:, :, ::-1])
    return [
        {"box": [float(v) for v in f.bbox], "score": float(f.det_score), "emb": f.normed_embedding.astype(np.float32)}
        for f in faces
        if f.det_score >= min_score
    ]


def free():
    _APP.clear()
