"""Интеграционные тесты с настоящей компактной моделью (Qwen3-VL-2B) на сервере Ollama.

Запуск:  ollama pull qwen3-vl:2b-instruct && pytest -m slow --no-cov
Адрес и модель берутся из IMAGEHUB_LLM_BASE_URL и IMAGEHUB_LLM_MODEL. Если сервер недоступен, тесты пропускаются."""

import pytest
from PIL import Image

from imagehub import llm
from imagehub.config import get_settings

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_server():
    get_settings.cache_clear()
    s = get_settings()
    try:
        have = llm.available_models()
    except llm.LLMError as e:
        pytest.skip(f"сервер моделей недоступен: {e}")
    if s.llm_model not in have and f"{s.llm_model}:latest" not in have:
        pytest.skip(f"на сервере нет модели {s.llm_model}: ollama pull {s.llm_model}")
    return s


def test_text_answer(real_server):
    out = llm.generate(real_server.llm_model, [None], ["Ответь одним словом: какая столица Франции?"], max_new_tokens=16)[0]
    assert "париж" in out.lower() or "paris" in out.lower()


def test_describes_color_of_image(real_server):
    red = Image.new("RGB", (256, 256), (220, 20, 20))
    out = llm.generate(real_server.llm_model, [red], ["Какого цвета это изображение? Ответь одним словом."], max_new_tokens=16)[0]
    assert "красн" in out.lower() or "red" in out.lower()


def test_translation_to_russian(real_server):
    from imagehub.ml import textfix

    out = textfix.translate_ru(["A dog is running on the grass."], device="cpu")[0]
    assert out and any("а" <= ch <= "я" for ch in out.lower())
