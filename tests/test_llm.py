"""Клиент OpenAI-совместимого API: запросы, повторы, потоковый ответ, эмбеддинги. Сервер замокан через respx."""

import httpx
import numpy as np
import pytest
from PIL import Image

from conftest import completion
from imagehub import llm


def test_chat_sends_openai_payload(llm_server):
    out = llm.chat("m1", [{"role": "user", "content": "привет"}], max_tokens=7)
    assert out == "ответ модели"
    body = llm_server.requests()[0]
    assert body["model"] == "m1" and body["max_tokens"] == 7 and body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "привет"}]
    assert llm_server.calls[0].request.headers["authorization"] == "Bearer ollama"


def test_base_url_and_model_come_from_environment(llm_server):
    """Адрес сервера и модель не зашиты в коде: меняются переменными окружения."""
    from imagehub.ml import vqa

    assert vqa.path() == "api:test-vl"
    assert str(llm_server.calls.call_count) == "0"
    llm.chat("x", [{"role": "user", "content": "q"}])
    assert llm_server.calls[0].request.url.host == "llm.test"


def test_image_is_sent_as_data_url(llm_server):
    im = Image.new("RGB", (40, 30), "red")
    llm.generate("m", [im], ["что на фото?"])
    content = llm_server.requests()[0]["messages"][-1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "что на фото?"}


def test_large_image_is_shrunk():
    big = Image.new("RGB", (4000, 2000), "blue")
    import base64
    import io

    url = llm.image_url(big, max_side=500)
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(decoded.size) == 500


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (None, []),
        ([{"role": "user", "content": "строка"}], [{"role": "user", "content": "строка"}]),
        (
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "а"}, {"type": "image", "image": "x"}, {"type": "text", "text": "б"}],
                }
            ],
            [{"role": "assistant", "content": "аб"}],
        ),
    ],
)
def test_history_is_flattened(history, expected):
    msgs = llm.build_messages(None, "вопрос", history)
    assert msgs[:-1] == expected
    assert msgs[-1] == {"role": "user", "content": "вопрос"}


def test_generate_keeps_order_with_parallel_workers(llm_server, monkeypatch):
    monkeypatch.setenv("IMAGEHUB_LLM_CONCURRENCY", "3")
    from imagehub.config import get_settings

    get_settings.cache_clear()
    llm_server.chat.side_effect = lambda request: httpx.Response(
        200, json=completion("эхо: " + __import__("json").loads(request.content)["messages"][-1]["content"])
    )
    out = llm.generate("m", [None] * 5, [f"q{i}" for i in range(5)])
    assert out == [f"эхо: q{i}" for i in range(5)]


def test_retries_on_server_error_then_succeeds(llm_server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm_server.chat.side_effect = [httpx.Response(503, text="busy"), httpx.Response(200, json=completion("ok"))]
    assert llm.chat("m", [{"role": "user", "content": "q"}]) == "ok"
    assert llm_server.calls.call_count == 2


def test_client_error_is_not_retried(llm_server):
    llm_server.chat.side_effect = None
    llm_server.chat.respond(404, text='{"error":"model not found"}')
    with pytest.raises(llm.LLMError, match="404"):
        llm.chat("нет-такой", [{"role": "user", "content": "q"}])
    assert llm_server.calls.call_count == 1


def test_connection_error_gives_actionable_message(llm_server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm_server.chat.side_effect = httpx.ConnectError("refused")
    with pytest.raises(llm.LLMError, match="ollama pull test-vl"):
        llm.chat("m", [{"role": "user", "content": "q"}])
    assert llm_server.calls.call_count == 2  # IMAGEHUB_LLM_RETRIES=2


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": None}]}])
def test_malformed_response(llm_server, payload):
    llm_server.chat.side_effect = None
    llm_server.chat.respond(json=payload)
    with pytest.raises(llm.LLMError, match="неожиданный"):
        llm.chat("m", [{"role": "user", "content": "q"}])


def test_empty_content_becomes_empty_string(llm_server):
    llm_server.chat.respond(json={"choices": [{"message": {"content": None}}]})
    assert llm.chat("m", [{"role": "user", "content": "q"}]) == ""


def sse(*chunks: str) -> str:
    lines = ['data: {"choices":[{"delta":{"content":"%s"}}]}' % c for c in chunks]
    return "\n\n".join([*lines, "data: [DONE]", ""])


def test_stream_yields_deltas(llm_server):
    llm_server.chat.respond(text=sse("При", "вет", "!"), headers={"content-type": "text/event-stream"})
    assert list(llm.stream("m", None, "q")) == ["При", "вет", "!"]
    assert llm_server.requests()[0]["stream"] is True


def test_stream_skips_garbage_lines(llm_server):
    body = 'event: ping\n\ndata: not-json\n\ndata: {"choices":[{"delta":{}}]}\n\ndata: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n'
    llm_server.chat.respond(text=body)
    assert list(llm.stream("m", None, "q")) == ["ok"]


def test_stream_http_error(llm_server):
    llm_server.chat.respond(500, text="boom")
    with pytest.raises(llm.LLMError, match="500"):
        list(llm.stream("m", None, "q"))


def test_embed_normalizes_and_orders(llm_server):
    route = llm_server.post("/embeddings").respond(
        json={"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]}
    )
    mat = llm.embed("emb", ["a", "b"])
    assert route.called
    np.testing.assert_allclose(mat, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6)
    assert mat.dtype == np.float32


def test_embed_batches(llm_server):
    llm_server.post("/embeddings").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]}
                    for i, _ in enumerate(__import__("json").loads(request.content)["input"])
                ]
            },
        )
    )
    assert llm.embed("emb", [str(i) for i in range(70)], batch=32).shape == (70, 2)


def test_available_models_and_warmup(llm_server):
    llm_server.get("/models").respond(json={"data": [{"id": "b"}, {"id": "a"}]})
    assert llm.available_models() == ["a", "b"]
    llm.warmup("a")
    assert llm_server.requests()[-1]["max_tokens"] == 1


def test_available_models_unreachable(llm_server):
    llm_server.get("/models").mock(side_effect=httpx.ConnectError("x"))
    with pytest.raises(llm.LLMError):
        llm.available_models()


def test_api_prefix_helpers():
    assert llm.is_api("api:qwen3-vl:2b-instruct") and not llm.is_api("C:/weights/x")
    assert llm.model_id("api:qwen3-vl:2b-instruct") == "qwen3-vl:2b-instruct"
