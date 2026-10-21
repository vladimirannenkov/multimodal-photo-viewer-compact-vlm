"""Реестр моделей и скачивание весов: imagehub models [--download] [--only id,id].

Языковые модели (Qwen3-VL-2B, GLM-OCR) обслуживает сервер с OpenAI-совместимым API (Ollama), их веса сюда не скачиваются:
`ollama pull <модель>`, проверка командой `imagehub check`. Здесь остаются модели, которых нет в Ollama."""

import logging
from pathlib import Path

from .config import get_settings

log = logging.getLogger("imagehub.models")

REGISTRY = {
    "siglip2": ("google/siglip2-base-patch16-256", "эмбеддинги, тип и событие, поиск", 0.4),
    "glm_ocr": ("zai-org/GLM-OCR", "OCR локально (если IMAGEHUB_OCR_MODEL пуст)", 2.7),
    "paddleocr_vl": ("PaddlePaddle/PaddleOCR-VL", "OCR таблиц и формул локально (IMAGEHUB_OCR_LOCAL_PADDLE=true)", 1.9),
    "florence2": ("florence-community/Florence-2-large", "описание, объекты, регионы", 1.5),
    "text_embed": ("intfloat/multilingual-e5-small", "эмбеддинги описаний и запросов (если IMAGEHUB_EMBED_MODEL пуст)", 0.5),
    "ppocr_v5_det": ("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "детектор текста (ONNX, CPU)", 0.005),
    "ppocr_eslav_rec": ("PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx", "распознавание ru/uk/be/en (ONNX, CPU)", 0.008),
    "doclayout_yolo": ("darknoah99/DocLayout-YOLO-DocStructBench-onnx", "макет страницы (ONNX, CPU)", 0.075),
}
# InsightFace buffalo_l скачивается самой библиотекой в <weights>/insightface/models


def status():
    w = get_settings().weights_dir
    return {k: (w / k).exists() and any((w / k).iterdir()) for k in REGISTRY}


def prepare_rec_dict(weights: Path):
    """RapidOCR читает словарь символов из метаданных ONNX: встраиваем character_dict из inference.yml."""
    import onnx
    import yaml

    d = weights / "ppocr_eslav_rec"
    out = d / "rec_with_dict.onnx"
    if out.exists() or not (d / "inference.onnx").exists():
        return
    chars = yaml.safe_load((d / "inference.yml").read_text(encoding="utf-8"))["PostProcess"]["character_dict"]
    (d / "dict.txt").write_text("\n".join(chars), encoding="utf-8")
    m = onnx.load(str(d / "inference.onnx"))
    p = m.metadata_props.add()
    p.key, p.value = "character", "\n".join(chars)
    onnx.save(m, str(out))


def download(only=None):
    from huggingface_hub import snapshot_download

    w = get_settings().weights_dir
    w.mkdir(parents=True, exist_ok=True)
    for k, (repo, _, _) in REGISTRY.items():
        if only and k not in only:
            continue
        log.info("download", extra={"model": k, "repo": repo})
        snapshot_download(repo, local_dir=w / k)
    prepare_rec_dict(w)
    try:  # лица: buffalo_l
        from insightface.app import FaceAnalysis

        FaceAnalysis(name="buffalo_l", root=str(w / "insightface"), providers=["CPUExecutionProvider"])
    except Exception as e:
        log.warning("insightface download failed", extra={"error": str(e)})
