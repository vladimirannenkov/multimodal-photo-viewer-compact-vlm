from pathlib import Path

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"


def model_path(model_id, cfg):
    local = WEIGHTS / cfg.get("weights", model_id)
    return str(local) if local.exists() else cfg["repo"]


def load_image(path):
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def vlm_generate(cfg_path, image, prompt, device, max_new_tokens=512, repetition_penalty=1.05, max_side=1280, trust=True, max_time=120, model_cache={}):
    """Общий вызов AutoModelForImageTextToText (Qwen3-VL, GLM-OCR, PaddleOCR-VL)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if cfg_path not in model_cache:
        while len(model_cache) >= 2:  # держим две модели (GLM-OCR и PaddleOCR-VL помещаются в 8 ГБ)
            model_cache.pop(next(iter(model_cache)))
        model_cache[cfg_path] = (
            AutoModelForImageTextToText.from_pretrained(
                cfg_path, torch_dtype=torch.bfloat16, trust_remote_code=trust
            ).to(device).eval(),
            AutoProcessor.from_pretrained(cfg_path, trust_remote_code=trust),
        )
    model, proc = model_cache[cfg_path]
    if max_side and max(image.size) > max_side:  # меньше визуальных токенов, быстрее prefill
        k = max_side / max(image.size)
        image = image.resize((round(image.width * k), round(image.height * k)))
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=repetition_penalty, max_time=max_time,
                             use_cache=True)  # у PaddleOCR-VL в generation_config кэш выключен, генерация в 7 раз медленнее
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def vlm_generate_batch(cfg_path, images, prompts, device, max_new_tokens=512, repetition_penalty=1.05,
                       max_side=1280, trust=True, max_time=300):
    """Пакетная генерация: несколько изображений (с разными промптами) за один вызов generate, padding слева."""
    import torch
    vlm_generate(cfg_path, images[0], prompts[0], device, max_new_tokens=1, trust=trust)  # гарантирует загрузку модели
    model, proc = vlm_generate.__defaults__[-1][cfg_path]
    imgs = []
    for im in images:
        if max_side and max(im.size) > max_side:
            k = max_side / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)))
        imgs.append(im)
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p}]}]
            for im, p in zip(imgs, prompts)]
    tok = getattr(proc, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", padding=True, padding_side="left").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=repetition_penalty, max_time=max_time, use_cache=True)
    return [t.strip() for t in proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
