"""Клиент OpenAI-совместимого API (/v1/chat/completions, /v1/embeddings): Ollama, llama-server, vLLM, облачные провайдеры.

Адрес сервера, ключ и идентификаторы моделей берутся из настроек (переменные окружения IMAGEHUB_LLM_*),
в коде они не зашиты. Изображения передаются как data URL в формате content-частей OpenAI."""

import base64
import io
import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np
from PIL import Image

from .config import get_settings

log = logging.getLogger("imagehub.llm")
API_PREFIX = "api:"  # путь модели вида «api:<id модели на сервере>» отличает серверную модель от локальных весов


class LLMError(RuntimeError):
    """Сервер моделей недоступен или вернул ошибку."""


def is_api(path: str) -> bool:
    return str(path).startswith(API_PREFIX)


def model_id(path: str) -> str:
    return str(path)[len(API_PREFIX) :]


def _http() -> httpx.Client:
    s = get_settings()
    return httpx.Client(
        base_url=s.llm_base_url.rstrip("/"), timeout=s.llm_timeout, headers={"Authorization": f"Bearer {s.llm_api_key}"}
    )


def _explain(e: Exception) -> LLMError:
    s = get_settings()
    if isinstance(e, httpx.HTTPStatusError):
        body = e.response.text[:300]
        return LLMError(f"сервер моделей {s.llm_base_url} ответил {e.response.status_code}: {body}")
    return LLMError(
        f"сервер моделей {s.llm_base_url} недоступен ({type(e).__name__}). Запустите его (ollama serve) "
        f"и загрузите модель (ollama pull {s.llm_model})"
    )


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST с повторами: временные сбои соединения и ответы 5xx. Ошибки клиента (4xx) повторять бессмысленно."""
    tries = max(1, get_settings().llm_retries)
    for k in range(tries):
        try:
            with _http() as c:
                r = c.post(path, json=payload)
                r.raise_for_status()
                data: dict[str, Any] = r.json()
                return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or k == tries - 1:
                raise _explain(e) from e
        except httpx.TransportError as e:
            if k == tries - 1:
                raise _explain(e) from e
        log.warning("llm retry", extra={"path": path, "attempt": k + 1})
        time.sleep(min(2.0**k, 8.0))
    raise LLMError("недостижимо")


def image_url(im: Image.Image, max_side: int = 1280) -> str:
    if max_side and max(im.size) > max_side:  # меньше визуальных токенов, быстрее ответ
        k = max_side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _text_of(content: Any) -> str:
    """Реплика истории бывает строкой или списком частей [{type: text, text: ...}]."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def build_messages(
    image: Image.Image | None, prompt: str, history: list[dict[str, Any]] | None = None, max_side: int = 1280
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": h["role"], "content": _text_of(h["content"])} for h in (history or [])]
    if image is None:
        msgs.append({"role": "user", "content": prompt})
    else:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url(image, max_side)}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
    return msgs


def chat(model: str, messages: list[dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0) -> str:
    t0 = time.time()
    data = _post(
        "/chat/completions",
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False},
    )
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"неожиданный ответ сервера: {str(data)[:200]}") from e
    log.info("llm chat", extra={"model": model, "sec": round(time.time() - t0, 2), "chars": len(text)})
    return text.strip()


def generate(
    model: str,
    images: list[Image.Image | None],
    prompts: list[str],
    max_new_tokens: int = 512,
    history: list[dict[str, Any]] | None = None,
    max_side: int = 1280,
) -> list[str]:
    """Пакет запросов (изображение или None, промпт) параллельно; порядок ответов совпадает с порядком запросов."""

    def one(args: tuple[Image.Image | None, str]) -> str:
        return chat(model, build_messages(args[0], args[1], history, max_side), max_new_tokens)

    jobs = list(zip(images, prompts, strict=True))
    workers = max(1, min(get_settings().llm_concurrency, len(jobs)))
    if workers == 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def stream(
    model: str,
    image: Image.Image | None,
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 320,
    max_side: int = 1024,
) -> Iterator[str]:
    """Потоковый ответ (SSE): куски текста по мере генерации."""
    payload = {
        "model": model,
        "messages": build_messages(image, prompt, history, max_side),
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        with _http() as c, c.stream("POST", "/chat/completions", json=payload) as r:
            if r.is_error:
                r.read()  # тело ошибки нужно для сообщения; в потоковом режиме оно не загружено
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
    except (httpx.HTTPStatusError, httpx.TransportError) as e:
        raise _explain(e) from e


def warmup(model: str) -> None:
    """Короткий запрос, чтобы сервер загрузил модель в память заранее."""
    chat(model, [{"role": "user", "content": "ok"}], max_tokens=1)


def embed(model: str, texts: list[str], batch: int = 32) -> np.ndarray:
    """/v1/embeddings -> нормированные векторы float32 (n, dim)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        data = _post("/embeddings", {"model": model, "input": texts[i : i + batch]})
        out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    mat = np.asarray(out, dtype=np.float32)
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def available_models() -> list[str]:
    """Идентификаторы моделей, которые сервер отдаёт по GET /v1/models."""
    try:
        with _http() as c:
            r = c.get("/models")
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
    except (httpx.HTTPError, ValueError, KeyError) as e:
        raise _explain(e) from e
