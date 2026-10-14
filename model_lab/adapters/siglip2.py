import torch
from transformers import AutoModel, AutoProcessor
from common import model_path

LABELS = ["a photo of people", "a photo of a landscape", "a screenshot of an app", "a scanned document with text",
          "a receipt", "a chart or plot of metrics", "a heatmap", "a diagram", "a photo of an object", "a meme"]


def run(mid, cfg, imgs, device, ctx):
    p = model_path(mid, cfg)
    model = AutoModel.from_pretrained(p, torch_dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(p)
    res = {"images": []}
    with torch.no_grad():
        for img in imgs:
            inp = proc(text=LABELS, images=img, padding="max_length", max_length=64, return_tensors="pt").to(device)
            o = model(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()})
            probs = torch.sigmoid(o.logits_per_image)[0].float().cpu().tolist()
            res["images"].append({"zero_shot": sorted(zip(LABELS, probs), key=lambda x: -x[1])})
    if len(imgs) == 2:
        with torch.no_grad():
            f = [model.get_image_features(**proc(images=i, return_tensors="pt").to(device, torch.float16)) for i in imgs]
        f = [x / x.norm(dim=-1, keepdim=True) for x in f]
        res["image_cosine"] = float((f[0] * f[1]).sum())
    res["markdown"] = "\n\n".join(
        f"### Изображение {i + 1}\n" + "\n".join(f"- {l}: {p:.3f}" for l, p in r["zero_shot"])
        for i, r in enumerate(res["images"]))
    if "image_cosine" in res:
        res["markdown"] += f"\n\nКосинус между изображениями: {res['image_cosine']:.3f}"
    return res
