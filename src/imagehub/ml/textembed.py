"""Эмбеддинги описаний, OCR-текста и запросов.

Если задана settings.embed_model, векторы берутся с сервера по /v1/embeddings; иначе локальная multilingual-e5-small (CPU, 118M).
При смене модели размерность векторов меняется: сохранённые векторы kind='txt' нужно пересчитать (DELETE FROM vectors WHERE kind='txt')."""

import threading

import numpy as np

from .. import llm
from ..config import get_settings
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
    if get_settings().embed_model:
        return llm.embed(get_settings().embed_model, [t[:2000] for t in texts])
    return _model().encode(["passage: " + t[:2000] for t in texts], normalize_embeddings=True, batch_size=16).astype(np.float32)


def query(text):
    if get_settings().embed_model:
        return llm.embed(get_settings().embed_model, [text])[0]
    return _model().encode(["query: " + text], normalize_embeddings=True)[0].astype(np.float32)
