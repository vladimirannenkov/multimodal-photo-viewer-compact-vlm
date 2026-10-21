"""Структурированные логи: одна JSON-строка на запись (время, уровень, логгер, сообщение, дополнительные поля, трассировка)."""

import json
import logging
import sys
import time
from typing import Any

_STD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update({k: v for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_")})  # поля из extra={...}
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure(level: str = "INFO", as_json: bool = True) -> None:
    """Настраивает корневой логгер; повторный вызов заменяет обработчик. Логи uvicorn идут тем же путём."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "PIL", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
