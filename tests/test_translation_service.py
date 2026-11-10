import httpx
from fastapi.testclient import TestClient


def test_yandex_translation_and_errors(env, respx_mock):
    from imagehub import translation

    url = "https://browser.translate.yandex.net/api/v1/tr.json/translate"
    route = respx_mock.post(url).mock(
        side_effect=[
            httpx.Response(200, json={"code": 200, "text": ["Собака в парке."]}),
            httpx.Response(503),
        ]
    )
    assert translation.yandex(["A dog in a park.", "A second sentence."]) == ["Собака в парке.", ""]
    assert route.call_count == 2
    assert all(len(part.encode("utf-8")) <= 9000 for part in translation._parts("word " * 3000))


def test_translation_settings_persist_and_validate(env):
    from imagehub.api.app import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/translation-settings").json()["provider"] == "qwen"
        assert client.put("/api/translation-settings", json={"provider": "yandex"}).json()["provider"] == "yandex"
        assert client.get("/api/translation-settings").json()["provider"] == "yandex"
        assert client.put("/api/translation-settings", json={"provider": "invalid"}).status_code == 422


def test_yandex_setting_routes_translation_without_llm(env, respx_mock, monkeypatch):
    from imagehub import translation
    from imagehub.ml import textfix

    translation.save_settings("yandex")
    respx_mock.post("https://browser.translate.yandex.net/api/v1/tr.json/translate").respond(
        json={"code": 200, "text": ["Собака в парке."]}
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("LLM should not be called for Yandex translation")

    monkeypatch.setattr(textfix, "_chat", fail_if_called)
    assert textfix.translate_ru(["A dog in a park."], device="cpu") == ["Собака в парке."]
