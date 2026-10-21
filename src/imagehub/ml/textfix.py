"""Правка опечаток OCR и перевод описаний на русский: текстовые запросы к той же Qwen3-VL-2B, что отвечает в чате (отдельной языковой модели нет).
Пакетная генерация, формулы защищены метками, ответ ограничен по длине, выдумки отбрасываются."""

import difflib
import re
import time
from typing import Any

from . import vlm, vqa

MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)
NL = chr(10)
SYS = (
    "Исправь опечатки, ошибки распознавания текста, слитные и разорванные слова. Текст на русском или английском. "
    "Не добавляй и не удаляй содержание, не перефразируй, не меняй числа и коды. "
    "Метки вида [M1] оставь на месте без изменений. Верни только исправленный текст, сохрани переносы строк."
)
SHOTS = [
    (
        "Договор заключен между сторонами в городе Москве. Срок действия соглашения составляет дванадцать месяцев с даты подписания.",
        "Договор заключен между сторонами в городе Москве. Срок действия соглашения составляет двенадцать месяцев с даты подписания.",
    ),
    (
        "Качество обучения оценивается функцией потерь [M1] навалидационной выборке.",
        "Качество обучения оценивается функцией потерь [M1] на валидационной выборке.",
    ),
]


def _chat(texts, system, shots, device, limit):
    """Текстовые запросы пакетом к Qwen3-VL (без изображений). Возвращает ответы в том же порядке."""
    hist = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for u, a in shots:
        hist += [
            {"role": "user", "content": [{"type": "text", "text": u}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]},
        ]
    return vlm.generate(
        vqa.path(), [None] * len(texts), texts, device, max_new_tokens=limit, repetition_penalty=1.0, history=hist
    )


def _script_switch(orig, fixed):
    cyr = lambda t: sum("Ѐ" <= c <= "ӿ" for c in t)
    lat = lambda t: sum(c.isascii() and c.isalpha() for c in t)
    return cyr(orig) > lat(orig) and lat(fixed) > lat(orig)


def free():
    """Своих весов нет: модель общая с чатом (vlm.free)."""


def _protect(text):
    maths = []

    def sub(m):
        maths.append(m.group(0))
        return f"[M{len(maths)}]"

    return MATH.sub(sub, text), maths


def _restore(text, maths):
    for i, m in enumerate(maths, 1):
        text = text.replace(f"[M{i}]", m)
    return text


def correct_texts(texts, device="cuda", batch=16, min_sim=0.85):
    """-> (исправленные тексты в том же порядке, статистика). Блок остаётся прежним, если правка подозрительна."""
    prot = [_protect(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(prot[i][0]))  # близкие по длине в одном пакете
    out = list(texts)
    stats: dict[str, Any] = {"blocks": len(texts), "changed": 0, "rejected": 0}
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        limit = min(
            700, int(max(len(prot[i][0]) for i in idx) / 2) + 24
        )  # ответ не длиннее исходного текста (около 2,5 символа на токен)
        gen = _chat([prot[i][0] for i in idx], SYS, SHOTS, device, limit)
        for i, fixed in zip(idx, gen):
            fixed = re.sub(r"^```\w*|```$", "", fixed).strip()
            orig, maths = prot[i]
            ok = (
                fixed
                and all(f"[M{k}]" in fixed for k in range(1, len(maths) + 1))
                and 0.8 <= len(fixed) / max(1, len(orig)) <= 1.25
                and difflib.SequenceMatcher(None, orig.split(), fixed.split()).ratio() >= min_sim
            )
            if not ok:
                stats["rejected"] += 1
                continue
            if _script_switch(orig, fixed):  # кириллический текст не должен получить новые латинские буквы («плато» -> «plateau»)
                stats["rejected"] += 1
                continue
            res = _restore(fixed, maths)
            if " ".join(res.split()) != " ".join(texts[i].split()):  # изменения только в пробелах и переносах не считаем
                stats["changed"] += 1
                out[i] = res
    stats["sec"] = round(time.time() - t0, 2)
    return out, stats


TR_SYS = (
    "Переведи текст с английского на русский. Верни только перевод без пояснений. Сохрани порядок и количество строк. "
    "Если на вход подан список слов через запятую, верни список слов через запятую. Переводи только данный текст, ничего не добавляй от себя."
)
TR_SHOTS = [
    ("a dog, a person, a bicycle", "собака, человек, велосипед"),
    ("A cat is sleeping on a sofa next to a window.", "Кошка спит на диване рядом с окном."),
]


def translate_ru(texts, device="cuda", batch=16, max_new=260):
    """Перевод описаний и тегов на русский (Qwen3-VL-2B). Для поиска на любом из двух языков."""
    out = [""] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        limit = min(max_new, int(max(len(texts[i]) for i in idx) / 1.6) + 24)
        gen = _chat([texts[i] for i in idx], TR_SYS, TR_SHOTS, device, limit)
        for i, r in zip(idx, gen):
            cyr = sum("Ѐ" <= c <= "ӿ" for c in r)
            out[i] = r if cyr >= 0.3 * max(1, sum(c.isalpha() for c in r)) else ""  # не перевёл: не сохраняем
    return out
