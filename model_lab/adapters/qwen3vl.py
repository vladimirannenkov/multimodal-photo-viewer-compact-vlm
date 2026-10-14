from common import model_path, vlm_generate

PROMPTS = {
    "describe": "Опиши изображение подробно: объекты, их свойства, расположение.",
    "objects_json": 'Верни JSON-список объектов: [{"label": str, "bbox_2d": [x1,y1,x2,y2], "attrs": str}]. Координаты в диапазоне 0-1000.',
    "event": "Что за ситуация или событие происходит на изображении (праздник, работа, прогулка и т. д.)? Ответь кратко с обоснованием.",
    "chart_json": ('Если это график, верни JSON: {"title","x_axis","y_axis","series":[{"name","points":[[x,y],...]}],'
                   '"best_series","plateau_value"}. Если это не график, верни {"chart": false}.'),
}


def run(mid, cfg, imgs, device, ctx):
    p = model_path(mid, cfg)
    res = {"images": []}
    for img in imgs:
        keys = ctx.get("qwen_keys") or list(PROMPTS)
        res["images"].append({k: vlm_generate(p, img, PROMPTS[k], device, max_new_tokens=700) for k in keys})
    res["markdown"] = "\n\n".join(
        f"### Изображение {i + 1}\n" + "\n\n".join(f"**{k}**\n\n{v}" for k, v in r.items())
        for i, r in enumerate(res["images"]))
    return res
