"""Пути к весам моделей (каталог из настроек; веса монтируются томом, в образ не входят)."""

from pathlib import Path

from ..config import get_settings


class _W:
    def __truediv__(self, name) -> Path:
        return get_settings().weights_dir / name

    def __str__(self):
        return str(get_settings().weights_dir)


WEIGHTS = _W()
