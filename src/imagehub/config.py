"""Настройки сервиса. Переопределяются переменными окружения IMAGEHUB_*."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO = Path(__file__).resolve().parents[2]


def _default_weights() -> Path:
    for p in (REPO / "weights", REPO / "model_lab" / "weights"):
        if p.exists():
            return p
    return REPO / "weights"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMAGEHUB_", env_file=".env", extra="ignore")

    data_dir: Path = REPO / "data"
    weights_dir: Path = _default_weights()
    pictures_dir: Path = Path.home() / "Pictures" / "ImageHub"  # сюда сохраняются вставки из буфера
    device: str = "auto"  # auto | cuda | cpu
    host: str = "127.0.0.1"
    port: int = 8765
    workers: int = 2  # процессы подготовки (декодирование, миниатюры, хеш)
    autostart_pipeline: bool = True
    ocr_fix: bool = True  # правка опечаток OCR (Qwen3-VL-2B)
    thumb_side: int = 320

    @property
    def db_path(self) -> Path:
        return self.data_dir / "imagehub.sqlite"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"  # копии после обрезки, склейки, аннотаций

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def ensure_dirs(self):
        for d in (self.data_dir, self.thumbs_dir, self.derived_dir, self.pictures_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
