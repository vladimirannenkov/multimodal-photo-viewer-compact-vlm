import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def test_upload_routes_and_search(env, monkeypatch):
    from imagehub.api import compat
    from imagehub.api.app import create_app

    monkeypatch.setattr(compat, "describe_image", lambda image: f"{image.width}x{image.height}")
    monkeypatch.setattr(compat, "ask_image", lambda image, prompt, model: prompt + str(image.width))
    monkeypatch.setattr(compat.search, "image_text_search", lambda q, limit: [])
    stream = io.BytesIO()
    Image.new("RGB", (12, 8)).save(stream, "PNG")
    content = stream.getvalue()
    with TestClient(create_app()) as client:
        assert client.post("/describe", files={"file": ("x.png", content, "image/png")}).json() == {"description": "12x8"}
        assert client.post("/ask", files={"file": ("x.png", content, "image/png")}, data={"prompt": "size?"}).json() == {
            "answer": "size?12"
        }
        assert client.post("/ask", files={"file": ("x.png", content)}, data={"prompt": " "}).status_code == 400
        assert client.post("/describe", files={"file": ("x.png", b"bad")}).status_code == 400
        assert client.get("/search", params={"q": "cat"}).json() == {"total": 0, "items": []}
        assert client.get("/search", params={"q": ""}).status_code == 400
        assert client.post("/batch", json={"path": str(env / "missing")}).status_code == 400
        assert client.get("/batch/missing").status_code == 404
        assert client.get("/api/profile").status_code == 200


def test_export_and_persisted_evaluation(env):
    from imagehub import db
    from imagehub.api.compat import evaluations, export_batch
    from imagehub.config import get_settings

    image_id = db.ex("INSERT INTO images(path,name) VALUES(?,?)", ("a.png", "a.png")).lastrowid
    db.set_vector(image_id, "img", [3.0, 4.0])
    out = export_batch("test", [image_id])
    from pathlib import Path

    import pyarrow.parquet as pq

    folder = Path(out)
    assert pq.read_table(folder / "results.parquet").to_pylist()[0]["name"] == "a.png"
    np.testing.assert_allclose(np.load(folder / "embeddings.npy")[0], [0.6, 0.8], atol=0.001)
    assert np.load(folder / "image_ids.npy").tolist() == [image_id]
    target = get_settings().data_dir / "evaluations"
    target.mkdir()
    (target / "run.json").write_text(json.dumps({"summary": {"completed": 2}}), encoding="utf-8")
    assert evaluations()["evaluations"][0]["summary"]["completed"] == 2


def test_vqa_cli_uploads_image(tmp_path, respx_mock, capsys):
    import httpx

    from imagehub.vqa_cli import main

    image = tmp_path / "image.png"
    Image.new("RGB", (3, 4)).save(image)
    endpoint = respx_mock.post("http://127.0.0.1:8765/ask").respond(json={"answer": "ok"})
    assert main(["ask", str(image), "What?"]) == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "ok"}
    assert b"What?" in endpoint.calls[0].request.content
    respx_mock.post("http://127.0.0.1:8765/describe").mock(return_value=httpx.Response(503))
    assert main(["describe", str(image)]) == 1


def test_batch_job_exports_results_and_reports_errors(env, monkeypatch):
    from types import SimpleNamespace

    from imagehub import db
    from imagehub.api import compat

    image_id = db.ex(
        "INSERT INTO images(path,name,status,error) VALUES(?,?,?,?)", ("b.png", "b.png", "done", "scene failed")
    ).lastrowid
    monkeypatch.setattr(compat.scanner, "add_folder", lambda *a, **k: 1)
    monkeypatch.setattr(compat.scanner, "scan_folder", lambda *a, **k: {"seen": 1})
    db.ex("UPDATE images SET folder_id=? WHERE id=?", (None, image_id))
    # Use an existing folder to satisfy SQLite foreign keys.
    folder = db.ex("INSERT INTO folders(path) VALUES(?)", ("folder",)).lastrowid
    db.ex("UPDATE images SET folder_id=? WHERE id=?", (folder, image_id))
    monkeypatch.setattr(compat.scanner, "add_folder", lambda *a, **k: folder)
    monkeypatch.setattr(compat, "get_pipeline", lambda: SimpleNamespace(start=lambda: None, kick=lambda: None, status=lambda: {}))
    compat._jobs["test-job"] = {}
    compat.run_batch("test-job", str(env))
    assert compat._jobs["test-job"]["state"] == "completed_with_errors"
    assert compat._jobs["test-job"]["errors"] == 1


def test_vision_model_adapters_use_existing_backends(monkeypatch):
    from types import SimpleNamespace

    from imagehub import vision
    from imagehub.ml import vlm, vqa

    image = Image.new("RGB", (2, 2))
    monkeypatch.setattr(vqa, "ask", lambda im, prompt, device: f"{prompt}:{device}")
    monkeypatch.setattr(vision, "get_vision_model", lambda: vision.APIVisionModel())
    monkeypatch.setattr(
        "imagehub.config.get_settings", lambda: SimpleNamespace(resolve_device=lambda: "cpu", cpu_only=lambda: False)
    )
    assert vision.get_vision_model().ask(image, "test") == "test:cpu"
    monkeypatch.setattr(vlm, "generate", lambda path, images, prompts, device: [f"{path}:{device}:{prompts[0]}"])
    assert vision.TransformersVisionModel("weights", "cpu").ask(image, "test") == "weights:cpu:test"


def test_compat_search_ranks_image_vectors(monkeypatch, fake_models):
    from imagehub import search

    monkeypatch.setattr(search, "_query_vec", lambda model, text: np.array([1.0]))
    monkeypatch.setattr(search, "_score", lambda kind, vector, allowed: {4: 0.1, 3: 0.9, 7: 0.4})
    assert search.image_text_search("cat", limit=2) == [(3, 0.9), (7, 0.4)]


def test_profile_tracks_errors_and_external_model_memory(env, respx_mock):
    import pytest

    from imagehub import profiling

    respx_mock.get("http://llm.test/api/ps").respond(
        json={"models": [{"name": "glm-ocr:latest", "size": 1000, "size_vram": 700}]}
    )
    with profiling.measure("VLM.generate:api:glm-ocr"):
        pass
    profiling._observe_ollama("VLM.generate:api:glm-ocr", "glm-ocr")
    assert profiling.snapshot()["VLM.generate:api:glm-ocr"]["ollama_max_observed_size_vram_bytes"] == 700
    with pytest.raises(RuntimeError), profiling.measure("failure-test"):
        raise RuntimeError("x")
    assert profiling.snapshot()["failure-test"]["errors"] == 1


def test_saved_profile_contains_call_delta(env):
    from imagehub import profiling

    with profiling.measure("delta-test"):
        pass
    baseline = profiling.snapshot()
    with profiling.measure("delta-test"):
        pass
    path = profiling.save_snapshot("delta", {"images": 1}, per_event=True, baseline=baseline)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["profile"]["delta-test"]["calls"] == 1
    assert data["profile"]["delta-test"]["cpu_seconds"] >= 0
    assert not data["summary"]["cumulative_profile"]


def test_cpu_setting_forces_local_device_and_cpu_server(env, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from imagehub import cpu_server, llm
    from imagehub.api import compat
    from imagehub.config import get_settings
    from imagehub.ml import manager

    pipe = SimpleNamespace(state={"stage": "paused"}, device="cuda")
    monkeypatch.setattr(compat, "get_pipeline", lambda: pipe)
    monkeypatch.setattr(manager, "gpu", lambda **kwargs: nullcontext())
    monkeypatch.setattr(manager, "free_heavy", lambda: None)
    monkeypatch.setattr(cpu_server, "ensure", lambda: "http://127.0.0.1:11435/v1")
    monkeypatch.setattr(cpu_server, "stop", lambda: None)
    assert compat.set_compute_settings(compat.ComputeRequest(cpu_only=True))["device"] == "cpu"
    assert get_settings().cpu_only()
    assert llm.effective_base_url() == "http://127.0.0.1:11435/v1"
    assert compat.set_compute_settings(compat.ComputeRequest(cpu_only=False))["cpu_only"] is False
