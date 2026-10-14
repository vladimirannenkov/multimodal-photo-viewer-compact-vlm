"""python eval_formula.py <image> [gt.md]  Прогон формульного промпта GLM-OCR и PaddleOCR-VL: время и сходство с эталоном."""
import difflib
import re
import sys
import time

import torch

from common import load_image, vlm_generate

STRIP = r"\\(?:left|right|bigg|Bigg|big|Big|displaystyle|mathrm|operatorname|quad|qquad)(?![a-zA-Z])|\\[,;! ]"


def norm(s):
    s = re.sub(STRIP, "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", s)


img = load_image(sys.argv[1])
gt = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else None
for mid, repo, trust, prompt in (("glm_ocr", "weights/glm_ocr", True, "Formula Recognition:"),
                                 ("paddleocr_vl", "weights/paddleocr_vl", False, "Formula Recognition:")):
    vlm_generate(repo, load_image("samples/test_text.png"), prompt, "cuda", max_new_tokens=4, trust=trust)  # прогрев
    torch.cuda.synchronize()
    t = time.time()
    out = vlm_generate(repo, img, prompt, "cuda", max_new_tokens=700, trust=trust, max_side=1280)
    torch.cuda.synchronize()
    dt = time.time() - t
    sim = difflib.SequenceMatcher(None, norm(out), norm(gt), autojunk=False).ratio() if gt else None
    print(f"\n===== {mid}: {dt:.1f} с, сходство с эталоном (нормализовано): {sim and round(sim, 3)}")
    print(out)
