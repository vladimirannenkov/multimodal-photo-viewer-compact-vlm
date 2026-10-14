"""Правка опечаток OCR (Qwen3-1.7B): пакетная, формулы защищены метками, ответ ограничен по длине, выдумки отбрасываются."""
import difflib
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import WEIGHTS

PATH = str(WEIGHTS / "qwen3_fix")
MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)
NL = chr(10)
SYS = ("Исправь опечатки, ошибки распознавания текста, слитные и разорванные слова. Текст на русском или английском. "
       "Не добавляй и не удаляй содержание, не перефразируй, не меняй числа и коды. "
       "Метки вида [M1] оставь на месте без изменений. Верни только исправленный текст, сохрани переносы строк.")
SHOTS = [
    ("Договор заключен между сторонами в городе Москве. Срок действия соглашения составляет дванадцать месяцев с даты подписания.",
     "Договор заключен между сторонами в городе Москве. Срок действия соглашения составляет двенадцать месяцев с даты подписания."),
    ("Качество обучения оценивается функцией потерь [M1] навалидационной выборке.",
     "Качество обучения оценивается функцией потерь [M1] на валидационной выборке."),
]
_cache = {}


def _load(device):
    if "m" not in _cache:
        tok = AutoTokenizer.from_pretrained(PATH)
        tok.padding_side = "left"
        _cache["tok"] = tok
        _cache["m"] = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).to(device).eval()
    return _cache["tok"], _cache["m"]


def _script_switch(orig, fixed):
    cyr = lambda t: sum("Ѐ" <= c <= "ӿ" for c in t)
    lat = lambda t: sum(c.isascii() and c.isalpha() for c in t)
    return cyr(orig) > lat(orig) and lat(fixed) > lat(orig)


def free():
    _cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def correct_texts(texts, device="cuda", batch=8, min_sim=0.85):
    """-> (исправленные тексты в том же порядке, статистика). Блок остаётся прежним, если правка подозрительна."""
    tok, model = _load(device)
    prot = [_protect(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(prot[i][0]))  # близкие по длине в одном пакете
    out = list(texts)
    stats = {"blocks": len(texts), "changed": 0, "rejected": 0}
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        chats = []
        for i in idx:
            msgs = [{"role": "system", "content": SYS}]
            for u, a in SHOTS:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": prot[i][0]})
            chats.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        limit = min(600, int(1.3 * max(len(tok(prot[i][0]).input_ids) for i in idx)) + 16)  # ответ не длиннее исходного текста
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=limit, do_sample=False, use_cache=True)
        for i, g in zip(idx, gen):
            fixed = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            fixed = re.sub(r"^```\w*|```$", "", fixed).strip()
            orig, maths = prot[i]
            ok = (fixed and all(f"[M{k}]" in fixed for k in range(1, len(maths) + 1))
                  and 0.8 <= len(fixed) / max(1, len(orig)) <= 1.25
                  and difflib.SequenceMatcher(None, orig.split(), fixed.split()).ratio() >= min_sim)
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
