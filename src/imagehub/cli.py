"""Командная строка: imagehub serve | scan PATH | models [--download] | check.

Результаты команд выводятся в stdout (JSON или таблица), диагностика идёт структурированными логами в stderr."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence

log = logging.getLogger("imagehub.cli")


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="imagehub", description="ImageHub: локальный сервис для работы с изображениями")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="запустить сервер и веб-интерфейс")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    sc = sub.add_parser("scan", help="добавить папку и проиндексировать файлы")
    sc.add_argument("path")
    m = sub.add_parser("models", help="состояние локальных весов; --download скачивает их")
    m.add_argument("--download", action="store_true", help="скачать веса в каталог весов")
    m.add_argument("--only", default="", help="список id через запятую")
    sub.add_parser("check", help="проверить сервер моделей (Ollama и др.) и наличие нужных моделей")
    return ap


def cmd_check() -> int:
    """0: сервер отвечает и все нужные модели на нём есть; 1: нет."""
    from . import llm
    from .config import get_settings

    cfg = get_settings()
    wanted = [m for m in (cfg.llm_model, cfg.ocr_model, cfg.embed_model) if m]
    try:
        have = llm.available_models()
    except llm.LLMError as e:
        _out(json.dumps({"ok": False, "base_url": cfg.llm_base_url, "error": str(e)}, ensure_ascii=False))
        return 1
    # Ollama отдаёт имена с тегом (glm-ocr:latest), в настройках тег можно не указывать
    missing = [m for m in wanted if m not in have and f"{m}:latest" not in have]
    _out(
        json.dumps(
            {
                "ok": not missing,
                "base_url": cfg.llm_base_url,
                "available": have,
                "missing": missing,
                "hint": [f"ollama pull {m}" for m in missing],
            },
            ensure_ascii=False,
        )
    )
    return 0 if not missing else 1


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from .config import get_settings
    from .logs import configure

    cfg = get_settings()
    configure(cfg.log_level, cfg.log_json)
    if a.cmd in (None, "serve"):
        import uvicorn

        uvicorn.run(
            "imagehub.api.app:create_app",
            factory=True,
            host=getattr(a, "host", None) or cfg.host,
            port=getattr(a, "port", None) or cfg.port,
            log_level=cfg.log_level.lower(),
            log_config=None,
        )
    elif a.cmd == "scan":
        from . import scanner

        fid = scanner.add_folder(a.path)
        r = scanner.scan_folder(fid, deep=True)
        r.pop("ids", None)
        _out(json.dumps(r, ensure_ascii=False, default=str))
    elif a.cmd == "models":
        from . import models

        if a.download:
            models.download(set(filter(None, a.only.split(","))) or None)
        _out(f"Каталог весов: {cfg.weights_dir}")
        for k, ok in models.status().items():
            _, what, gb = models.REGISTRY[k]
            _out(f"{'OK ' if ok else '-- '} {k:16} ~{gb:>5} ГБ  {what}")
    elif a.cmd == "check":
        return cmd_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
