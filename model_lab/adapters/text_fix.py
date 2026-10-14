import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import model_path

SYS = ("Исправь опечатки и ошибки распознавания в тексте. Не добавляй и не удаляй информацию, "
       "не меняй формулы и таблицы. Верни только исправленный текст.")


def run(mid, cfg, imgs, device, ctx):
    """Берёт OCR-текст из paddleocr_vl (или glm_ocr), если они запускались в этом же прогоне."""
    src = next((ctx["previous"][k] for k in ("paddleocr_vl", "glm_ocr") if k in ctx["previous"]), None)
    if not src:
        return {"markdown": "Нет OCR-результата: запустите вместе с paddleocr_vl (--only paddleocr_vl,qwen3_fix)"}
    p = model_path(mid, cfg)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16).to(device).eval()
    res = {"images": []}
    only = ctx.get("fix_idx")
    for i, r in enumerate(src["images"]):
        if "text" not in r or (only is not None and i not in only):
            res["images"].append({"before": None, "after": None})
            continue
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["text"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=1024, do_sample=False, use_cache=True)
        res["images"].append({"before": r["text"], "after": tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()})
    res["markdown"] = "\n\n".join(f"### Изображение {i + 1}\n**До**\n\n{r['before']}\n\n**После**\n\n{r['after']}"
                                  for i, r in enumerate(res["images"]) if r["before"] is not None)
    return res
