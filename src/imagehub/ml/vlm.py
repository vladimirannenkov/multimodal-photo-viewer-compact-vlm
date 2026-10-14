"""Общий вызов моделей AutoModelForImageTextToText (GLM-OCR, PaddleOCR-VL, Qwen3-VL) с кэшем и пакетной генерацией."""
import gc

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_CACHE = {}  # путь -> (модель, процессор); держим до двух моделей (GLM-OCR и PaddleOCR-VL помещаются в 8 ГБ)


def free():
    _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(path, device, trust=True):
    if path not in _CACHE:
        while len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
            gc.collect()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _CACHE[path] = (AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, trust_remote_code=trust, device_map=device, low_cpu_mem_usage=True).eval(),
                        AutoProcessor.from_pretrained(path, trust_remote_code=trust))
    return _CACHE[path]


def _shrink(im, max_side):
    if max_side and max(im.size) > max_side:  # меньше визуальных токенов, быстрее prefill
        k = max_side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)))
    return im


def generate(path, images, prompts, device, max_new_tokens=512, repetition_penalty=1.05, max_side=1280, trust=True,
             max_time=300, history=None):
    """Пакет (изображение, промпт) за один вызов generate, padding слева. history: предыдущие реплики для диалога."""
    model, proc = load(path, device, trust)
    msgs = []
    for im, p in zip(images, prompts):
        turns = list(history or [])
        turns.append({"role": "user", "content": ([{"type": "image", "image": _shrink(im, max_side)}] if im is not None else [])
                      + [{"type": "text", "text": p}]})
        msgs.append(turns)
    tok = getattr(proc, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", padding=True, padding_side="left").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=repetition_penalty,
                             max_time=max_time, use_cache=True)  # у PaddleOCR-VL кэш по умолчанию выключен: в 7 раз медленнее
    return [t.strip() for t in proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]


def generate_batch(path, images, prompts, device, **kw):
    return generate(path, images, prompts, device, **kw)


def stream(path, image, prompt, device, max_new_tokens=320, repetition_penalty=1.05, max_side=1024, trust=True, history=None):
    """Потоковая генерация: возвращает итератор кусков текста. Вызывающий держит замок GPU на время итерации."""
    from threading import Thread

    from transformers import TextIteratorStreamer
    model, proc = load(path, device, trust)
    turns = list(history or [])
    turns.append({"role": "user", "content": [{"type": "image", "image": _shrink(image, max_side)}, {"type": "text", "text": prompt}]})
    inputs = proc.apply_chat_template([turns], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def work():
        with torch.no_grad():
            model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False,
                           repetition_penalty=repetition_penalty, use_cache=True)
    Thread(target=work, daemon=True).start()
    yield from streamer
