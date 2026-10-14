from sentence_transformers import SentenceTransformer
from common import model_path


def run(mid, cfg, imgs, device, ctx):
    """Косинус между описаниями двух картинок (из florence2/qwen3vl, если запускались в этом прогоне)."""
    src = ctx["previous"].get("qwen3vl")
    if not src or len(src["images"]) < 2:
        return {"markdown": "Нужны 2 изображения и qwen3vl в том же прогоне (--only qwen3vl,text_embed)"}
    m = SentenceTransformer(model_path(mid, cfg), device=device)
    texts = ["passage: " + r["describe"] for r in src["images"][:2]]
    e = m.encode(texts, normalize_embeddings=True)
    cos = float(e[0] @ e[1])
    return {"description_cosine": cos, "markdown": f"Косинус описаний: {cos:.3f}"}
