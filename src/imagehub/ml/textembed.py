"""multilingual-e5-small: эмбеддинги описаний и OCR-текста и запросов (CPU, 118M)."""
import threading

import numpy as np

from .weights import WEIGHTS

_M = None
_lock = threading.Lock()


def _model():
    global _M
    with _lock:
        if _M is None:
            from sentence_transformers import SentenceTransformer
            _M = SentenceTransformer(str(WEIGHTS / "text_embed"), device="cpu")
        return _M


def passages(texts):
    return _model().encode(["passage: " + t[:2000] for t in texts], normalize_embeddings=True, batch_size=16).astype(np.float32)


def query(text):
    return _model().encode(["query: " + text], normalize_embeddings=True)[0].astype(np.float32)
