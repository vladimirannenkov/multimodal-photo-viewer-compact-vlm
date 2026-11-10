"""Unified vision interface with lazy API and local Transformers adapters."""

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from PIL import Image


class VisionModel(ABC):
    @abstractmethod
    def ask(self, image: Image.Image, prompt: str) -> str:
        """Answer a question about an image."""


class APIVisionModel(VisionModel):
    def ask(self, image: Image.Image, prompt: str) -> str:
        from .config import get_settings
        from .ml import vqa

        return vqa.ask(image, prompt, get_settings().resolve_device())


class TransformersVisionModel(VisionModel):
    def __init__(self, model_path: str | Path, device: str):
        self.model_path = str(model_path)
        self.device = device

    def ask(self, image: Image.Image, prompt: str) -> str:
        from .config import get_settings
        from .ml import vlm

        device = "cpu" if get_settings().cpu_only() else self.device
        return vlm.generate(self.model_path, [image], [prompt], device)[0]


@lru_cache(maxsize=1)
def get_vision_model() -> VisionModel:
    return APIVisionModel()
