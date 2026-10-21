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

    # Языковые модели вызываются только по HTTP через OpenAI-совместимый API (Ollama, llama-server, vLLM, облачные провайдеры).
    # Для переключения сервера достаточно сменить переменные окружения, код не меняется.
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama; llama-server: http://localhost:8080/v1
    llm_api_key: str = "ollama"  # локальным серверам ключ не нужен, облачным нужен настоящий
    llm_model: str = "qwen3-vl:2b-instruct"  # чат, графики, правка опечаток, перевод описаний
    ocr_model: str = "glm-ocr"  # OCR через тот же сервер; пустая строка включает локальные веса GLM-OCR из каталога весов
    ocr_local_paddle: bool = False  # таблицы и формулы читает локальная PaddleOCR-VL (в Ollama её нет); иначе их читает ocr_model
    embed_model: str = ""  # модель для /v1/embeddings (например, bge-m3); пустая строка: локальная multilingual-e5-small
    llm_timeout: float = 300.0  # секунд на один запрос
    llm_concurrency: int = 2  # параллельных запросов к серверу (OLLAMA_NUM_PARALLEL должен быть не меньше)
    llm_retries: int = 3

    log_level: str = "INFO"
    log_json: bool = True  # структурированные логи, одна JSON-строка на запись
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

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.thumbs_dir, self.derived_dir, self.pictures_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
