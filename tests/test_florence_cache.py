"""Check cache reuse and compatibility without downloading model weights."""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.parametrize("supports_cache", [True, False])
def test_image_features_are_reused_per_batch_when_supported(monkeypatch, supports_cache):
    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "5.19.0" if supports_cache else "5.18.0"
    transformers.AutoProcessor = object
    transformers.Florence2ForConditionalGeneration = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    # Locate the module independently so tests never overwrite the real cached import.
    spec = importlib.util.spec_from_file_location(
        "imagehub.ml._florence_cache_test", Path(__file__).resolve().parents[1] / "src/imagehub/ml/florence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Inputs(dict):
        def to(self, *_args):
            return self

    class Processor:
        def __call__(self, text, images, return_tensors):
            return Inputs(input_ids=text, pixel_values=images)

        def batch_decode(self, ids, **_kwargs):
            return ids

        def post_process_generation(self, text, task, image_size):
            return {task: text if task == "<DETAILED_CAPTION>" else {"labels": [], "bboxes": []}}

    class Model:
        def __init__(self):
            self.encoded = 0
            self.cache_ids = []

        def get_image_features(self, pixels):
            self.encoded += 1
            return object()

        def generate(self, input_ids, **kwargs):
            if supports_cache:
                assert "pixel_values" not in kwargs
                self.cache_ids.append(id(kwargs["mm_encoder_outputs"]["image"]))
            else:
                assert "pixel_values" in kwargs
            return input_ids

    model = Model()
    monkeypatch.setattr(module, "_load", lambda _device: (model, Processor(), None))
    monkeypatch.setattr(module, "BATCH", 2)
    images = [Image.new("RGB", (2, 2)) for _ in range(3)]
    result = module.describe(images, "cpu")
    assert len(result) == 3
    assert all(item["caption"] == "<DETAILED_CAPTION>" for item in result)
    assert model.encoded == (2 if supports_cache else 0)
    if supports_cache:
        assert len(set(model.cache_ids[:3])) == len(set(model.cache_ids[3:])) == 1
