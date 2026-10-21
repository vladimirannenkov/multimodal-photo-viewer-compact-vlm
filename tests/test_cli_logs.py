"""CLI, структурированные логи, настройки и реестр моделей."""

import json
import logging
import sys

import httpx
import pytest
from PIL import Image

from imagehub import cli, imaging, logs, models
from imagehub.config import Settings, get_settings


# ---------- логи ----------
def test_json_formatter_includes_extra_and_exception():
    rec = logging.LogRecord("imagehub.test", logging.ERROR, __file__, 1, "сбой %s", ("x",), None)
    rec.model = "m1"
    try:
        raise ValueError("boom")
    except ValueError:
        rec.exc_info = sys.exc_info()
    data = json.loads(logs.JsonFormatter().format(rec))
    assert data["level"] == "ERROR" and data["msg"] == "сбой x" and data["model"] == "m1" and "ValueError" in data["exc"]
    assert data["ts"].endswith("Z") and data["logger"] == "imagehub.test"


@pytest.mark.parametrize("as_json", [True, False])
def test_configure_replaces_handler(as_json, capsys):
    logs.configure("DEBUG", as_json)
    logs.configure("INFO", as_json)
    root = logging.getLogger()
    assert len(root.handlers) == 1 and root.level == logging.INFO
    logging.getLogger("imagehub.x").info("привет", extra={"k": 1})
    err = capsys.readouterr().err.strip()
    assert (json.loads(err)["k"] == 1) if as_json else "привет" in err


def test_logs_are_valid_json_lines(capsys):
    logs.configure("INFO", True)
    log = logging.getLogger("imagehub.y")
    log.info("one")
    log.warning("two", extra={"n": 2})
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.strip()]
    assert [x["msg"] for x in lines] == ["one", "two"]


# ---------- настройки ----------
def test_settings_read_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGEHUB_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("IMAGEHUB_LLM_MODEL", "some/model")
    monkeypatch.setenv("IMAGEHUB_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("IMAGEHUB_DATA_DIR", str(tmp_path))
    s = Settings()
    assert (s.llm_base_url, s.llm_model, s.llm_api_key) == ("https://api.example.com/v1", "some/model", "sk-test")
    assert s.db_path == tmp_path / "imagehub.sqlite" and s.thumbs_dir.parent == tmp_path


def test_defaults_target_local_ollama():
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://localhost:11434/v1" and s.llm_model.startswith("qwen3-vl")


def test_resolve_device(monkeypatch):
    assert Settings(device="cpu").resolve_device() == "cpu"
    assert Settings(device="auto").resolve_device() in ("cpu", "cuda")


def test_get_settings_creates_dirs(env):
    s = get_settings()
    assert s.thumbs_dir.is_dir() and s.derived_dir.is_dir() and s.pictures_dir.is_dir()


# ---------- CLI ----------
def run_cli(capsys, *args):
    code = cli.main(list(args))
    return code, capsys.readouterr().out


def test_parser_commands():
    p = cli.build_parser()
    assert p.parse_args(["serve", "--port", "9000"]).port == 9000
    assert p.parse_args(["scan", "/x"]).path == "/x"
    assert p.parse_args(["models", "--download", "--only", "a,b"]).only == "a,b"
    with pytest.raises(SystemExit):
        p.parse_args(["unknown"])


def test_cli_scan_prints_json(env, photos, capsys):
    code, out = run_cli(capsys, "scan", str(photos))
    assert code == 0 and json.loads(out)["new"] == 4


def test_cli_models_lists_registry(env, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGEHUB_WEIGHTS_DIR", str(tmp_path / "w"))
    get_settings.cache_clear()
    code, out = run_cli(capsys, "models")
    assert code == 0 and "siglip2" in out and "qwen3vl" not in out  # языковая модель не скачивается, её отдаёт Ollama


def test_cli_models_download_dispatches(env, capsys, monkeypatch):
    called = {}
    monkeypatch.setattr(models, "download", lambda only: called.setdefault("only", only))
    run_cli(capsys, "models", "--download", "--only", "siglip2")
    assert called["only"] == {"siglip2"}


def test_cli_serve_starts_uvicorn(env, monkeypatch):
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    assert cli.main(["serve", "--port", "1234", "--host", "0.0.0.0"]) == 0
    assert seen["port"] == 1234 and seen["host"] == "0.0.0.0" and seen["factory"] and seen["log_config"] is None


def test_cli_check_ok_and_missing(llm_server, capsys):
    llm_server.get("/models").respond(json={"data": [{"id": "test-vl:latest"}, {"id": "test-ocr"}, {"id": "test-emb"}]})
    code, out = run_cli(capsys, "check")
    assert code == 0 and json.loads(out)["ok"] is True
    llm_server.get("/models").respond(json={"data": [{"id": "test-vl"}]})
    code, out = run_cli(capsys, "check")
    res = json.loads(out)
    assert code == 1 and res["missing"] == ["test-ocr", "test-emb"] and res["hint"][0] == "ollama pull test-ocr"


def test_cli_check_server_down(llm_server, capsys, monkeypatch):
    from imagehub import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm_server.get("/models").mock(side_effect=httpx.ConnectError("x"))
    code, out = run_cli(capsys, "check")
    assert code == 1 and json.loads(out)["ok"] is False


# ---------- реестр моделей ----------
def test_registry_has_no_server_models():
    assert "qwen3vl" not in models.REGISTRY and "siglip2" in models.REGISTRY


def test_status_reflects_weights_dir(env, monkeypatch, tmp_path):
    w = tmp_path / "weights"
    (w / "siglip2").mkdir(parents=True)
    (w / "siglip2" / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("IMAGEHUB_WEIGHTS_DIR", str(w))
    get_settings.cache_clear()
    st = models.status()
    assert st["siglip2"] is True and st["florence2"] is False


# ---------- работа с изображениями ----------
def test_open_rgb_variants(tmp_path):
    rgba = tmp_path / "t.png"
    Image.new("RGBA", (40, 20), (255, 0, 0, 0)).save(rgba)
    im = imaging.open_rgb(rgba)
    assert im.mode == "RGB" and im.getpixel((1, 1)) == (255, 255, 255)  # прозрачность на белом фоне
    big = tmp_path / "b.jpg"
    Image.new("RGB", (400, 100)).save(big)
    assert max(imaging.open_rgb(big, max_side=100).size) == 100


def test_prepare_makes_thumbnail_and_hash(tmp_path):
    src = tmp_path / "x.png"
    Image.new("RGB", (300, 200), "red").save(src)
    out = imaging.prepare(str(src), str(tmp_path / "t" / "x.webp"), 64)
    assert out["width"] == 300 and out["height"] == 200 and out["format"] == "png" and out["hash"]
    assert (tmp_path / "t" / "x.webp").exists()


def test_prepare_reports_error_for_broken_file(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert "error" in imaging.prepare(str(bad), str(tmp_path / "t.webp"), 64)


def test_is_image_and_stat(tmp_path):
    f = tmp_path / "a.PNG"
    f.write_bytes(b"x")
    assert imaging.is_image("a.PNG") and not imaging.is_image("a.txt")
    assert imaging.stat_tuple(str(f))[0] == 1
