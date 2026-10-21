"""Общие фикстуры. Модели в тестах заменены: сервер моделей мокается через respx, нейросети-адаптеры подменяются заглушками."""

import json
import sys
import types
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LLM_URL = "http://llm.test/v1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Изолированные каталоги данных, свежая БД и адрес мнимого сервера моделей для каждого теста."""
    monkeypatch.setenv("IMAGEHUB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEHUB_PICTURES_DIR", str(tmp_path / "pics"))
    monkeypatch.setenv("IMAGEHUB_AUTOSTART_PIPELINE", "false")
    monkeypatch.setenv("IMAGEHUB_WORKERS", "1")
    monkeypatch.setenv("IMAGEHUB_LLM_BASE_URL", LLM_URL)
    monkeypatch.setenv("IMAGEHUB_LLM_MODEL", "test-vl")
    monkeypatch.setenv("IMAGEHUB_OCR_MODEL", "test-ocr")
    monkeypatch.setenv("IMAGEHUB_EMBED_MODEL", "test-emb")
    monkeypatch.setenv("IMAGEHUB_LLM_RETRIES", "2")
    monkeypatch.setenv("IMAGEHUB_LLM_CONCURRENCY", "1")
    from imagehub import db, scanner
    from imagehub.config import get_settings

    get_settings.cache_clear()
    db.reset_for_tests()
    yield tmp_path
    scanner.shutdown_pool()
    get_settings.cache_clear()
    db.reset_for_tests()


@pytest.fixture
def photos(tmp_path):
    d = tmp_path / "photos"
    (d / "sub").mkdir(parents=True)
    for i, name in enumerate(["a.png", "b.jpg", "sub/c.webp", "sub/d.gif"]):
        im = Image.new("RGB", (120 + i * 10, 80), (30 * i, 100, 200 - 30 * i))
        ImageDraw.Draw(im).text((10, 10), f"img {i}", fill="white")
        im.save(d / name)
    (d / "notes.txt").write_text("не картинка")
    return d


def completion(text: str) -> dict:
    """Ответ /v1/chat/completions в формате OpenAI."""
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


@pytest.fixture
def llm_server(env):
    """Мок сервера моделей. mock.chat — маршрут чата, mock.requests() — тела отправленных запросов."""
    with respx.mock(base_url=LLM_URL, assert_all_called=False) as mock:
        mock.chat = mock.post("/chat/completions").respond(json=completion("ответ модели"))
        mock.embeddings = mock.post("/embeddings").mock(
            side_effect=lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [1.0, 0.5, 0.0]} for i, _ in enumerate(json.loads(request.content)["input"])
                    ]
                },
            )
        )
        mock.requests = lambda: [json.loads(c.request.content) for c in mock.calls if c.request.content]
        yield mock


# ---------- заглушки нейросетей (адаптеры требуют весов и тяжёлых библиотек) ----------
class FakeSiglip:
    kind = "photo"  # тип, который «определит» классификатор; тесты меняют FakeSiglip.kind

    def images(self, imgs):
        return [np.full(8, 1.0 / np.sqrt(8), dtype=np.float32) + 0.01 * k for k, _ in enumerate(imgs)]

    def classify(self, feats):
        return [{"kind": self.kind, "event": "прогулка", "kind_scores": {self.kind: 0.9, "photo": 0.05}} for _ in feats]

    def text(self, prompts):
        return np.tile(np.full(8, 1.0 / np.sqrt(8), dtype=np.float32), (len(prompts), 1))


@pytest.fixture
def fake_models(monkeypatch):
    """Подменяет siglip, textdet, florence, faces, layout на заглушки: пайплайн проходит все стадии без весов."""
    FakeSiglip.kind = "photo"
    state = types.SimpleNamespace(
        textdet={"has_text": False, "lines": 0, "language": None, "rotation": 0, "sample": None}, faces=[]
    )
    mods = {
        "siglip": types.SimpleNamespace(get=lambda device="cpu": FakeSiglip()),
        "textdet": types.SimpleNamespace(detect=lambda img: dict(state.textdet)),
        "florence": types.SimpleNamespace(
            describe=lambda imgs, device: [
                {
                    "caption": "A cat is sleeping on a sofa.",
                    "objects": [{"label": "cat", "box": [1.0, 2.0, 30.0, 40.0]}],
                    "regions": [],
                }
                for _ in imgs
            ],
            free=lambda: None,
        ),
        "faces": types.SimpleNamespace(detect=lambda img, device="cpu": [dict(f) for f in state.faces]),
        "layout": types.SimpleNamespace(detect=lambda img: []),
    }
    import imagehub.ml as ml

    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, f"imagehub.ml.{name}", mod)
        monkeypatch.setattr(ml, name, mod, raising=False)
    state.siglip = FakeSiglip
    return state
