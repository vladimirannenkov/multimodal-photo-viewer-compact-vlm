import time

from common import model_path, vlm_generate

MAX_TOKENS = {"text": 800, "formula": 256, "table": 640}


def run(mid, cfg, imgs, device, ctx):
    p = model_path(mid, cfg)
    keys_per_img = ctx.get("ocr_keys") or [None] * len(imgs)
    res = {"images": []}
    for i, (img, keys) in enumerate(zip(imgs, keys_per_img)):
        t = time.time()
        item = {}
        for k in (keys or cfg["prompts"]):
            item[k] = vlm_generate(p, img, cfg["prompts"][k], device, max_new_tokens=MAX_TOKENS[k],
                                   trust=cfg.get("trust_remote_code", True))
        item["_sec"] = round(time.time() - t, 1)
        res["images"].append(item)
        if ctx.get("progress"):
            ctx["progress"](i, item)
    res["markdown"] = "\n\n".join(
        f"### Изображение {i + 1}\n" + "\n\n".join(f"**{k}**\n\n{v}" for k, v in r.items() if not k.startswith("_"))
        for i, r in enumerate(res["images"]))
    return res
