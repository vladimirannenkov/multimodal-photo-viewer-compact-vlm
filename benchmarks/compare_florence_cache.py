"""Compare original Florence generation with reuse of image encoder features."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

TASKS = ("<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--weights", default="model_lab/weights/florence2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = Florence2ForConditionalGeneration.from_pretrained(
        args.weights, dtype=dtype, attn_implementation="sdpa", device_map=args.device, low_cpu_mem_usage=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.weights)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    inputs = [processor(text=[task] * len(images), images=images, return_tensors="pt").to(args.device, dtype) for task in TASKS]

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def generate(batch):
        return model.generate(**batch, max_new_tokens=args.max_new_tokens, num_beams=3, do_sample=False, use_cache=True)

    with torch.inference_mode():
        sync()
        started = time.perf_counter()
        original = [generate(batch) for batch in inputs]
        sync()
        original_seconds = time.perf_counter() - started
        sync()
        started = time.perf_counter()
        features = model.get_image_features(inputs[0]["pixel_values"])
        cached = []
        for batch in inputs:
            values = {key: value for key, value in batch.items() if key != "pixel_values"}
            values["mm_encoder_outputs"] = {"image": features}
            cached.append(generate(values))
        sync()
        cached_seconds = time.perf_counter() - started
    equal = {task: torch.equal(a, b) for task, a, b in zip(TASKS, original, cached)}
    result = {
        "device": args.device,
        "images": len(images),
        "max_new_tokens": args.max_new_tokens,
        "threads": args.threads,
        "original_seconds": original_seconds,
        "cached_seconds": cached_seconds,
        "tokens_equal": equal,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))  # noqa: T201
    if not all(equal.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
