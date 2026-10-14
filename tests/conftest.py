import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Изолированные каталоги данных и свежая БД для каждого теста."""
    monkeypatch.setenv("IMAGEHUB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEHUB_PICTURES_DIR", str(tmp_path / "pics"))
    monkeypatch.setenv("IMAGEHUB_AUTOSTART_PIPELINE", "false")
    monkeypatch.setenv("IMAGEHUB_WORKERS", "1")
    from imagehub import db, scanner
    from imagehub.config import get_settings
    get_settings.cache_clear()
    db.reset_for_tests()
    yield tmp_path
    scanner.shutdown_pool()
    get_settings.cache_clear()
    db.reset_for_tests()


@pytest.fixture()
def photos(tmp_path):
    d = tmp_path / "photos"
    (d / "sub").mkdir(parents=True)
    for i, name in enumerate(["a.png", "b.jpg", "sub/c.webp", "sub/d.gif"]):
        im = Image.new("RGB", (120 + i * 10, 80), (30 * i, 100, 200 - 30 * i))
        ImageDraw.Draw(im).text((10, 10), f"img {i}", fill="white")
        im.save(d / name)
    (d / "notes.txt").write_text("не картинка")
    return d
