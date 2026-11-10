"""Optional translation service for English descriptions; no LLM fallback."""

import json
import logging

import httpx

from . import profiling
from .config import get_settings

log = logging.getLogger(__name__)


def settings():
    path = get_settings().data_dir / "translation-settings.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("provider") in ("qwen", "yandex"):
            return {"provider": value["provider"]}
    except (OSError, ValueError, AttributeError):
        pass
    return {"provider": "qwen"}


def save_settings(provider):
    path = get_settings().data_dir / "translation-settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"provider": provider}, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _parts(text, limit=9000):
    """Split on whitespace below the unofficial client's per-request limit."""
    result, part = [], ""
    for word in text.split():
        if len(word.encode("utf-8")) > limit:
            raise ValueError("a word exceeds the translation service limit")
        candidate = f"{part} {word}" if part else word
        if len(candidate.encode("utf-8")) > limit:
            result.append(part)
            part = word
        else:
            part = candidate
    if part:
        result.append(part)
    return result


@profiling.tracked("translator.Yandex")
def yandex(texts):
    translated = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 YaBrowser/24.1.5.825 Yowser/2.5 Safari/537.36",
        "Referer": "https://www.youtube.com/",
    }
    with httpx.Client(timeout=15, headers=headers) as client:
        for text in texts:
            try:
                parts = []
                for part in _parts(text):
                    response = client.post(
                        "https://browser.translate.yandex.net/api/v1/tr.json/translate",
                        params={"srv": "browser_video_translation", "text": part, "lang": "en-ru"},
                        data={"maxRetryCount": 2, "fetchAbortTimeout": 500},
                    )
                    response.raise_for_status()
                    data = response.json()
                    values = data.get("text")
                    if data.get("code") != 200 or not isinstance(values, list) or not values or not values[0]:
                        raise ValueError("empty translation")
                    parts.append(values[0])
                translated.append(" ".join(parts))
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                log.warning("Yandex translation unavailable: %s", exc)
                translated.append("")
    return translated
