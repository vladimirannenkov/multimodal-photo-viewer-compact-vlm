"""Быстрый детектор «есть связный текст» + язык (письменность). PP-OCRv5, ONNX, CPU, инвариантен к повороту на 90°."""

import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .weights import WEIGHTS

DET_SIDE, SAMPLE_LINES = 960, 4
_OCR = None


def script_counts(text):
    c = {"cyrillic": 0, "latin": 0, "cjk": 0, "other": 0}
    for ch in text:
        o = ord(ch)
        if 0x400 <= o <= 0x4FF:
            c["cyrillic"] += 1
        elif ch.isascii() and ch.isalpha():
            c["latin"] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
            c["cjk"] += 1
        elif ch.isalpha():
            c["other"] += 1
    return c


def language_label(c):
    total = sum(c.values())
    if total < 8:
        return "не определён"
    top = max(c, key=c.get)
    share = c[top] / total
    names = {"cyrillic": "кириллица", "latin": "латиница", "cjk": "CJK", "other": "другая письменность"}
    return names[top] + ("" if share >= 0.8 else f", смешанный ({share:.0%})")


def _ocr(threads=6):
    """RapidOCR создаёт почти однопоточные сессии (детектор 2.3 с вместо 0.4 с), подменяем на многопоточные."""
    global _OCR
    if _OCR is None:
        det = str(WEIGHTS / "ppocr_v5_det" / "inference.onnx")
        rec = str(WEIGHTS / "ppocr_eslav_rec" / "rec_with_dict.onnx")  # словарь встроен в метаданные ONNX
        ocr = RapidOCR(det_model_path=det, rec_model_path=rec)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False  # динамические размеры входа раздувают арену
        so.enable_mem_pattern = False
        ocr.text_detector.infer.session = ort.InferenceSession(det, so, providers=["CPUExecutionProvider"])
        ocr.text_recognizer.session.session = ort.InferenceSession(rec, so, providers=["CPUExecutionProvider"])
        ocr.text_recognizer([np.zeros((48, 320, 3), np.uint8)])
        ocr.text_detector(np.zeros((64, 64, 3), np.uint8))
        _OCR = ocr
    return _OCR


def detect(img: Image.Image) -> dict:
    ocr = _ocr()
    small = img.copy()
    small.thumbnail((DET_SIDE, DET_SIDE), reducing_gap=2.0)
    arr0 = np.ascontiguousarray(np.array(small))
    boxes, _ = ocr.text_detector(arr0)
    best = {"rotation": 0, "lines": 0 if boxes is None else len(boxes), "arr": arr0, "boxes": boxes}
    if best["lines"] < 3:  # вертикальные строки: проверяем 90° на уменьшенной копии
        tiny = small.copy()
        tiny.thumbnail((512, 512))
        bt, _ = ocr.text_detector(np.ascontiguousarray(np.array(tiny.rotate(90, expand=True))))
        n90 = 0 if bt is None else len(bt)
        if n90 >= 3 and n90 > best["lines"] * 1.3:
            arr = np.ascontiguousarray(np.array(small.rotate(90, expand=True)))
            b, _ = ocr.text_detector(arr)
            best = {"rotation": 90, "lines": 0 if b is None else len(b), "arr": arr, "boxes": b}
    sample, conf = [], []
    if best["lines"]:
        boxes = ocr.sorted_boxes(best["boxes"])
        widths = [np.linalg.norm(b[1] - b[0]) for b in boxes]
        top = [boxes[i] for i in sorted(np.argsort(widths)[::-1][:SAMPLE_LINES])]
        crops = ocr.get_crop_img_list(best["arr"], np.array(top))
        crops, _, _ = ocr.text_cls(crops)
        rec, _ = ocr.text_recognizer(crops)
        sample, conf = [r[0] for r in rec], [float(r[1]) for r in rec]
    text = " ".join(sample)
    sc = script_counts(text)
    mc = sum(conf) / len(conf) if conf else 0.0
    return {
        "has_text": best["lines"] >= 3 and len(text) >= 15 and mc >= 0.5,
        "lines": best["lines"],
        "rotation": best["rotation"],
        "language": language_label(sc),
        "scripts": sc,
        "conf": round(mc, 2),
        "sample": text[:300],
    }
