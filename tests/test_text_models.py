"""Языковая часть: правка опечаток, перевод, вопросы к изображению, разбор OCR. Сервер моделей замокан (respx)."""

import json

import httpx
import numpy as np
import pytest
from PIL import Image

from conftest import completion
from imagehub import llm
from imagehub.ml import docparse, structure, textembed, textfix, vlm, vqa


def reply_with(llm_server, mapping_or_fn):
    """Ответ сервера зависит от последнего сообщения пользователя."""

    def side_effect(request):
        msgs = json.loads(request.content)["messages"]
        last = msgs[-1]["content"]
        last = last if isinstance(last, str) else last[-1]["text"]
        text = mapping_or_fn(last) if callable(mapping_or_fn) else mapping_or_fn[last]
        return httpx.Response(200, json=completion(text))

    llm_server.chat.side_effect = side_effect


# ---------- маршрутизация vlm: сервер или локальные веса ----------
def test_vlm_routes_api_paths_to_server(llm_server):
    out = vlm.generate("api:test-vl", [None, None], ["a", "b"], "cpu", max_new_tokens=5)
    assert out == ["ответ модели", "ответ модели"]
    assert {r["model"] for r in llm_server.requests()} == {"test-vl"}


def test_vlm_stream_routes_to_server(llm_server):
    llm_server.chat.respond(text='data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n')
    assert list(vlm.stream("api:test-vl", None, "q", "cpu")) == ["x"]


def test_vlm_load_warms_up_server_model(llm_server):
    assert vlm.load("api:test-vl", "cpu") is None
    assert llm_server.requests()[0]["max_tokens"] == 1


def test_vlm_free_without_torch_is_noop():
    vlm.free()


# ---------- исправление опечаток ----------
def test_correct_texts_applies_good_fix_and_rejects_bad_ones(llm_server):
    src = {
        "Срок действия договора составляет дванадцать месяцев с даты подписания.": "Срок действия договора составляет двенадцать месяцев с даты подписания.",  # хорошая правка
        "Короткий текст про кошку на диване дома": "Совсем другой длинный ответ, который выдумала модель про собак и лес и море",  # выдумка
        "Привет мир как дела у вас сегодня тут": "Hello world how are you doing today here",  # смена алфавита
    }
    reply_with(llm_server, src)
    out, stats = textfix.correct_texts(list(src), device="cpu")
    assert out[0] == "Срок действия договора составляет двенадцать месяцев с даты подписания."
    assert out[1:] == list(src)[1:]
    assert stats["changed"] == 1 and stats["rejected"] == 2


def test_correct_texts_protects_formulas(llm_server):
    text = "Потери $L = 1$ вычисляются на валидационой выборке каждую эпоху обучения"
    sent = []

    def echo(t):
        sent.append(t)
        return t.replace("валидационой", "валидационной")

    reply_with(llm_server, echo)
    out, stats = textfix.correct_texts([text], device="cpu")
    assert "[M1]" in sent[0] and "$" not in sent[0]  # формула ушла на сервер меткой
    assert out[0] == "Потери $L = 1$ вычисляются на валидационной выборке каждую эпоху обучения" and stats["changed"] == 1


def test_correct_texts_rejects_when_marker_lost(llm_server):
    reply_with(llm_server, lambda t: "Потери без формулы на выборке")
    out, stats = textfix.correct_texts(["Потери $L$ на выборке"], device="cpu")
    assert out == ["Потери $L$ на выборке"] and stats["rejected"] == 1


def test_correct_texts_strips_code_fences_and_ignores_whitespace_changes(llm_server):
    reply_with(llm_server, lambda t: "```\n" + t + "\n```")
    out, stats = textfix.correct_texts(["Один и тот же текст для проверки."], device="cpu")
    assert out == ["Один и тот же текст для проверки."] and stats["changed"] == 0


def test_correct_texts_sends_few_shots_in_history(llm_server):
    textfix.correct_texts(["Некий текст для проверки правки."], device="cpu")
    msgs = llm_server.requests()[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[2]["role"] == "assistant"
    assert msgs[-1]["content"] == "Некий текст для проверки правки."


# ---------- перевод ----------
def test_translate_ru_keeps_translation_and_drops_untranslated(llm_server):
    reply_with(llm_server, {"A dog runs.": "Собака бежит.", "a cat": "a cat"})
    assert textfix.translate_ru(["A dog runs.", "a cat"], device="cpu") == ["Собака бежит.", ""]


@pytest.mark.parametrize(
    ("a", "b", "expected"), [("привет", "привет hello", True), ("привет", "привет", False), ("hello", "hello мир", False)]
)
def test_script_switch(a, b, expected):
    assert textfix._script_switch(a, b) is expected


def test_textfix_free_is_noop():
    assert textfix.free() is None


# ---------- вопросы к изображению ----------
def test_vqa_ask_uses_style_only_for_first_question(llm_server):
    im = Image.new("RGB", (32, 32), "white")
    assert vqa.ask(im, "Что это?", "cpu") == "ответ модели"
    first = llm_server.requests()[0]["messages"][-1]["content"][-1]["text"]
    assert first.startswith(vqa.STYLE) and first.endswith("Что это?")
    vqa.ask(im, "А ещё?", "cpu", history=[{"role": "user", "content": "Что это?"}, {"role": "assistant", "content": "Кошка"}])
    second = llm_server.requests()[1]["messages"]
    assert second[0] == {"role": "user", "content": "Что это?"} and second[-1]["content"][-1]["text"] == "А ещё?"


def test_vqa_history_is_limited_to_last_eight(llm_server):
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(20)]
    vqa.ask(None, "q", "cpu", history=hist)
    assert len(llm_server.requests()[0]["messages"]) == 9


def test_vqa_ask_stream(llm_server):
    llm_server.chat.respond(
        text='data: {"choices":[{"delta":{"content":"Кош"}}]}\n\ndata: {"choices":[{"delta":{"content":"ка"}}]}\n\ndata: [DONE]\n'
    )
    assert "".join(vqa.ask_stream(Image.new("RGB", (8, 8)), "q", "cpu")) == "Кошка"


def test_vqa_warmup(llm_server):
    vqa.warmup("cpu")
    assert llm_server.calls.call_count == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Вот: {"title": "Loss", "n": 1} конец', {"title": "Loss", "n": 1}),
        ("без json", None),
        ("{сломанный: json}", None),
    ],
)
def test_extract_json(text, expected):
    assert vqa.extract_json(text) == expected


def test_chart_stats_plateau_max_and_best_series():
    rising = {"name": "A", "points": [[i, 1 - 1 / (i + 1)] for i in range(20)]}
    noisy = {"name": "B", "points": [[0, 0.1], [1, 0.9], [2, 0.2], [3, 0.8], [4, 0.1]]}
    short = {"name": "C", "points": [[0, 1], [1, 2]]}
    st = vqa.chart_stats({"series": [rising, noisy, short, {"name": "D", "points": [["x", "y"]] * 5}]})
    assert [s["name"] for s in st["series"]] == ["A", "B"]
    a, b = st["series"]
    assert a["plateau"] is not None and a["max_at"] == 19 and a["final"] == pytest.approx(0.95)
    assert b["plateau"] is None and b["max"] == 0.9
    assert st["best_series"] == "A"


def test_chart_stats_empty():
    assert vqa.chart_stats(None) == {"series": [], "best_series": None}
    assert vqa.chart_stats({"series": None})["best_series"] is None


# ---------- эмбеддинги описаний ----------
def test_textembed_uses_server_when_model_configured(llm_server, monkeypatch):
    monkeypatch.setenv("IMAGEHUB_EMBED_MODEL", "bge-m3")
    from imagehub.config import get_settings

    get_settings.cache_clear()
    route = llm_server.post("/embeddings").mock(
        side_effect=lambda req: httpx.Response(
            200, json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(json.loads(req.content)["input"]))]}
        )
    )
    assert textembed.passages(["a", "b"]).shape == (2, 2)
    np.testing.assert_allclose(textembed.query("q"), [1.0, 0.0])
    assert route.call_count == 2


# ---------- настройки движков OCR ----------
@pytest.mark.parametrize(
    ("ocr_model", "paddle", "engine", "path_prefix", "trust"),
    [
        ("glm-ocr", False, "glm_ocr", "api:glm-ocr", False),
        ("glm-ocr", False, "paddleocr_vl", "api:glm-ocr", False),  # без локальной Paddle таблицы читает GLM-OCR
        ("glm-ocr", True, "paddleocr_vl", "paddleocr_vl", False),
        ("", False, "glm_ocr", "glm_ocr", True),
    ],
)
def test_engine_cfg(env, monkeypatch, ocr_model, paddle, engine, path_prefix, trust):
    monkeypatch.setenv("IMAGEHUB_OCR_MODEL", ocr_model)
    monkeypatch.setenv("IMAGEHUB_OCR_LOCAL_PADDLE", str(paddle))
    from imagehub.config import get_settings

    get_settings.cache_clear()
    cfg = docparse.engine_cfg(engine)
    assert (cfg["path"].startswith(path_prefix) or cfg["path"].endswith(path_prefix)) and cfg["trust"] is trust


def test_parse_pages_sends_crops_to_ocr_model(llm_server, monkeypatch):
    """Подсказка «формула» пропускает разбор макета: блок целиком уходит в OCR-модель на сервере."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "imagehub.ml.layout", types.SimpleNamespace(detect=lambda im: []))
    import imagehub.ml as ml

    monkeypatch.setattr(ml, "layout", sys.modules["imagehub.ml.layout"], raising=False)
    reply_with(llm_server, lambda t: "x ^ { 2 }")
    out = docparse.parse_pages([Image.new("RGB", (80, 40), "white")], hints=["formula"], device="cpu", fix=False)
    assert out[0]["markdown"] == "$$\nx^{2}\n$$"
    body = llm_server.requests()[0]
    assert body["model"] == "test-ocr" and body["messages"][-1]["content"][-1]["text"] == "Formula Recognition:"


# ---------- постобработка OCR ----------
@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>x|y</td></tr></table>", "| A | B |\n|---|---|\n| 1 | x\\|y |"),
        (
            "<table><tr><td colspan=2>шапка</td></tr></table>",
            '<table border="1" cellpadding="4"><tr><td colspan=2>шапка</td></tr></table>',
        ),
        ("<table></table>", '<table border="1" cellpadding="4"></table>'),
    ],
)
def test_html_table_to_md(html, expected):
    assert docparse.html_table_to_md(html) == expected


def test_otsl_and_table_to_md_dispatch():
    otsl = "<fcel>A<fcel>B<nl><fcel>1<ecel><nl>"
    assert docparse.table_to_md(otsl) == "| A | B |\n|---|---|\n| 1 |  |"
    assert docparse.table_to_md("<table><tr><td>z</td></tr></table>").startswith("| z |")
    assert docparse.table_to_md("просто текст") == "просто текст"
    assert docparse.otsl_to_md("<nl>") == "<nl>"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$$ x ^ { 2 } $$", "x^{2}"),
        (r"\[ a_{ i } \]", "a_{i}"),
        (r"\mathrm{скорость обучения}", r"\text{скорость обучения}"),
    ],
)
def test_fix_formula(raw, expected):
    assert docparse.fix_formula(raw) == expected


def test_fix_text_spaces_around_inline_math():
    assert docparse.fix_text("значение$x$равно") == "значение $x$ равно"


def test_assemble_blocks():
    blocks = [
        {"cls": "title", "kind": "text", "text": "Заголовок"},
        {"cls": "text", "kind": "text", "text": "Абзац"},
        {"cls": "table", "kind": "table", "text": "<fcel>A<nl>"},
        {"cls": "figure_caption", "kind": "text", "text": "Рис. 1"},
        {"cls": "formula", "kind": "formula", "text": "$$a$$"},
        {"cls": "abandon", "kind": "text", "text": "колонтитул"},
        {"cls": "text", "kind": "text", "text": ""},
    ]
    md = docparse.assemble(blocks)
    assert md.startswith("## Заголовок") and "*Рис. 1*" in md and "$$\na\n$$" in md and "колонтитул" not in md


def test_assemble_page_mode_merges_overlapping_band_lines():
    blocks = [
        {"cls": "text", "kind": "page", "text": "Итого к оплате 100\nСпасибо", "sep": ""},
        {"cls": "text", "kind": "page", "text": "Спасибо\nДо свидания", "sep": "\n"},
    ]
    md = docparse.assemble(blocks)
    assert md.count("Спасибо") == 1 and "До свидания" in md


# ---------- структура чеков ----------
@pytest.mark.parametrize(
    ("line", "numeric"),
    [("1 2 3", True), ("Молоко 2 45,00 90,00", True), ("Итого", False), ("12.5%", False)],
)
def test_is_numeric_row(line, numeric):
    assert structure.is_numeric_row(line) is numeric


def test_header_row_and_page_text_structure():
    assert structure.is_header_row("Товар Кол Цена Сумма") and not structure.is_header_row("1 2 3")
    text = "Магазин Рога\nИтого: 100\n\nТовар Кол Цена\nМолоко 2 45,00\nХлеб 1 30,00\nСыр 1 99,00"
    md = structure.structure_page_text(text)
    assert "Магазин Рога" in md and "**Итого:** 100" in md


def test_layout_unreliable_and_bands():
    assert structure.layout_unreliable([], (100, 100)) in (True, False)
    page = Image.new("L", (200, 400), 255)
    assert structure.split_bands(page) == [(0, 400, "")]


def test_llm_error_surface(llm_server):
    llm_server.chat.side_effect = httpx.ConnectError("нет связи")
    with pytest.raises(llm.LLMError):
        vqa.ask(None, "q", "cpu")
