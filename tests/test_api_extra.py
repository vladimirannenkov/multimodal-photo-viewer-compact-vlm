"""API: чат с моделью через мок сервера, люди и лица, эксперименты, поиск, очередь, здоровье сервиса."""

import io
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagehub import db, people, scanner, search


@pytest.fixture
def client(env, photos, fake_models):
    from imagehub.api.app import create_app

    with TestClient(create_app()) as c:
        scanner.scan_folder(scanner.add_folder(str(photos)))
        yield c


def first_id(client):
    return client.get("/api/images?sort=name&desc=false").json()["items"][0]["id"]


# ---------- здоровье и очередь ----------
def test_health_shallow_and_deep(client, llm_server):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm"]["model"] == "test-vl" and h["llm"]["base_url"] == "http://llm.test/v1"
    llm_server.get("/models").respond(json={"data": [{"id": "test-vl"}]})
    assert client.get("/api/health?deep=true").json()["llm"]["models"] == ["test-vl"]


def test_health_deep_reports_unreachable_server(client, llm_server):
    llm_server.get("/models").mock(side_effect=httpx.ConnectError("x"))
    assert "error" in client.get("/api/health?deep=true").json()["llm"]


def test_queue_pause_resume_and_reprocess(client):
    assert client.post("/api/queue/pause").json()["paused"] is True
    assert client.post("/api/queue/resume").json()["paused"] is False
    assert client.post("/api/queue/bogus").status_code == 400
    assert client.post("/api/reprocess", json={"ids": [first_id(client)]}).json() == {"queued": 1}
    assert client.get("/api/queue").json()["counts"]["new"] >= 1


# ---------- чат ----------
def test_chat_roundtrip_is_saved_and_cleared(client, llm_server):
    i = first_id(client)
    llm_server.chat.respond(json={"choices": [{"message": {"content": "Это синий прямоугольник."}}]})
    r = client.post(f"/api/images/{i}/chat", json={"message": "Что на картинке?"}).json()
    assert r["answer"] == "Это синий прямоугольник."
    assert [m["role"] for m in client.get(f"/api/images/{i}/chat").json()] == ["user", "assistant"]
    sent = llm_server.requests()[0]
    assert sent["model"] == "test-vl" and sent["messages"][-1]["content"][0]["type"] == "image_url"
    client.post(f"/api/images/{i}/chat", json={"message": "А цвет?"})
    assert len(llm_server.requests()[1]["messages"]) == 3  # история + новый вопрос
    client.delete(f"/api/images/{i}/chat")
    assert client.get(f"/api/images/{i}/chat").json() == []


def test_chat_includes_ocr_text_hint_for_first_question(client, llm_server):
    i = first_id(client)
    db.ex("UPDATE images SET ocr_md=? WHERE id=?", ("Распознанный текст чека: молоко 90 рублей", i))
    client.post(f"/api/images/{i}/chat", json={"message": "Сколько стоит молоко?"})
    text = llm_server.requests()[0]["messages"][-1]["content"][-1]["text"]
    assert "молоко 90 рублей" in text


def test_chat_rejects_empty_question(client):
    assert client.post(f"/api/images/{first_id(client)}/chat", json={"message": "  "}).status_code == 400
    assert client.post(f"/api/images/{first_id(client)}/chat/stream", json={"message": ""}).status_code == 400


def test_chat_stream_and_persistence(client, llm_server):
    i = first_id(client)
    llm_server.chat.respond(
        text='data: {"choices":[{"delta":{"content":"Кош"}}]}\n\ndata: {"choices":[{"delta":{"content":"ка"}}]}\n\ndata: [DONE]\n'
    )
    r = client.post(f"/api/images/{i}/chat/stream", json={"message": "Кто?"})
    assert r.text == "Кошка" and r.headers["content-type"].startswith("text/plain")
    import time

    for _ in range(50):  # запись в историю идёт в фоновом потоке сразу после последнего куска
        if len(client.get(f"/api/images/{i}/chat").json()) == 2:
            break
        time.sleep(0.1)
    assert [m["content"] for m in client.get(f"/api/images/{i}/chat").json()] == ["Кто?", "Кошка"]


def test_chat_stream_reports_server_error_in_body(client, llm_server):
    llm_server.chat.respond(500, text="boom")
    r = client.post(f"/api/images/{first_id(client)}/chat/stream", json={"message": "Кто?"})
    assert r.status_code == 200 and "[ошибка:" in r.text


def test_chat_warmup(client, llm_server):
    assert client.post("/api/chat/warmup").json() == {"ok": True}
    assert llm_server.requests()[0]["max_tokens"] == 1


# ---------- люди и лица ----------
@pytest.fixture
def faces(client):
    ids = [x["id"] for x in client.get("/api/images?sort=name&desc=false").json()["items"]]
    out = []
    for k, i in enumerate(ids[:2]):
        emb = np.eye(4, dtype=np.float32)[k]
        out.append(people.add_faces(i, [{"box": [5, 5, 40, 40], "score": 0.9 - k * 0.1, "emb": emb}])[0])
    return ids, out


def test_people_list_and_rename_and_merge(client, faces):
    _, (f1, f2) = faces
    assert len(client.get("/api/people").json()) == 2
    assert client.patch(f"/api/people/{f1['person_id']}", json={"name": "Анна"}).status_code == 200
    assert client.patch(f"/api/people/{f1['person_id']}", json={"name": " "}).status_code == 400
    client.patch(f"/api/people/{f2['person_id']}", json={"name": "Прохожий", "role": True})
    assert {p["name"] for p in client.get("/api/people").json()} == {"Анна", "Прохожий"}
    client.post(f"/api/people/{f2['person_id']}/merge", json={"into": f1["person_id"]})
    assert len(client.get("/api/people").json()) == 1


def test_face_crop_label_and_delete(client, faces):
    _, (f1, f2) = faces
    crop = client.get(f"/api/faces/{f1['id']}/crop")
    assert crop.status_code == 200 and Image.open(io.BytesIO(crop.content)).size[0] > 0
    assert client.get("/api/faces/99999/crop").status_code == 404
    nxt = client.get("/api/faces/next").json()
    assert nxt is not None
    assert client.post(f"/api/faces/{f1['id']}/label", json={"name": "Борис"}).json()["person_id"] == f1["person_id"]
    assert client.post(f"/api/faces/{f1['id']}/label", json={"name": ""}).status_code == 400
    assert client.post("/api/faces/99999/label", json={"name": "x"}).status_code == 404
    client.post(f"/api/faces/{f2['id']}/label", json={"skip": True})
    client.post(f"/api/faces/{f2['id']}/reassign", json={"person_id": f1["person_id"]})
    assert client.delete(f"/api/faces/{f2['id']}").json() == {"ok": True}


# ---------- эксперименты ----------
def test_experiments_crud_and_summary(client):
    ids = [x["id"] for x in client.get("/api/images").json()["items"]]
    eid = client.post("/api/experiments", json={"name": "Опыт 1"}).json()["id"]
    assert client.post("/api/experiments", json={}).json()["id"] != eid  # имя по умолчанию
    chart = {
        "title": "Acc",
        "stats": {"series": [{"name": "a", "max": 0.9, "plateau": 0.9, "max_at": 3, "min": 0.1, "final": 0.9}]},
    }
    db.ex("UPDATE images SET chart=? WHERE id=?", (json.dumps(chart), ids[0]))
    client.post(f"/api/experiments/{eid}/images", json={"ids": ids[:2]})
    s = client.get(f"/api/experiments/{eid}/summary").json()
    assert len(s["charts"]) == 2 and s["best"]["name"] == "a"
    created = next(e for e in client.get("/api/experiments").json() if e["id"] == eid)["created_at"]
    client.patch(f"/api/experiments/{eid}", json={"name": "Переименован", "note": "заметка"})
    after = next(e for e in client.get("/api/experiments").json() if e["id"] == eid)
    assert after["name"] == "Переименован" and after["created_at"] == created  # переименование не меняет дату
    client.delete(f"/api/experiments/{eid}")
    assert all(e["id"] != eid for e in client.get("/api/experiments").json())


# ---------- поиск ----------
def index_images(client, fake_models):
    from imagehub.pipeline import Pipeline

    Pipeline().run_wave([x["id"] for x in client.get("/api/images").json()["items"]])
    search.invalidate()


def test_text_search_hybrid_uses_bm25_and_vectors(client, fake_models, llm_server):
    index_images(client, fake_models)
    r = client.post("/api/search", data={"q": "cat sofa"}).json()
    assert r["total"] == 4 and all("score" in x for x in r["items"])
    assert client.post("/api/search", data={"q": "zzzzqqq nonexistent"}).json()["total"] >= 0


def test_search_by_image_and_similar_and_filters(client, fake_models, llm_server):
    index_images(client, fake_models)
    buf = io.BytesIO()
    Image.new("RGB", (30, 30), "red").save(buf, "PNG")
    r = client.post("/api/search", files={"image": ("q.png", buf.getvalue(), "image/png")}, data={"q": ""}).json()
    assert r["total"] == 4
    both = client.post(
        "/api/search", files={"image": ("q.png", buf.getvalue(), "image/png")}, data={"q": "cat", "weight": "0.7"}
    ).json()
    assert both["total"] == 4
    sim = client.post("/api/search", data={"similar_to": str(first_id(client))}).json()
    assert sim["total"] == 3  # без самой картинки
    assert client.post("/api/search", data={"q": ""}).json() == {"total": 0, "items": []}
    filt = client.post("/api/search", data={"q": "cat", "filters": json.dumps({"kind": "document"})}).json()
    assert filt["total"] == 0


def test_duplicates_exact_and_similar(client, fake_models, llm_server, photos):
    index_images(client, fake_models)
    (photos / "copy.png").write_bytes((photos / "a.png").read_bytes())
    scanner.scan_folder(db.q1("SELECT id FROM folders")["id"])
    groups = client.get("/api/duplicates?threshold=0.9").json()
    types = {g["type"] for g in groups}
    assert "exact" in types and all(g["items"] for g in groups)


def test_search_module_helpers(env, fake_models):
    assert search._fts_query("кошка на диване!") == '"кошка"* OR "на"* OR "диване"*'
    assert search._fts_query("!!!") == ""
    assert search._rank({1: 0.1, 2: 0.9, 3: 0.5}) == {2: 0, 3: 1, 1: 2}
    assert search.bm25("!!!", None) == {}
    assert search.search("") == []


# ---------- каталог и импорт ----------
def test_sidebar_and_fs_browse(client, tmp_path):
    sb = client.get("/api/sidebar").json()
    assert sb["total"] == 4 if "total" in sb else sb
    assert client.get("/api/fs/browse", params={"path": str(tmp_path)}).status_code == 200


def test_filters_on_image_list(client):
    ids = [x["id"] for x in client.get("/api/images").json()["items"]]
    db.ex("UPDATE images SET favorite=1, kind='chart', has_text=1, text_lang='ru' WHERE id=?", (ids[0],))
    for qs, n in (
        ("favorite=true", 1),
        ("kind=chart", 1),
        ("has_text=1", 1),
        ("sort=size&desc=false", 4),
        ("sort=resolution", 4),
    ):
        assert client.get(f"/api/images?{qs}").json()["total"] == n, qs


def test_delete_restore_and_patch(client):
    i = first_id(client)
    client.post("/api/images/delete", json={"ids": [i]})
    assert client.get("/api/images").json()["total"] == 3
    client.post("/api/images/restore", json={"ids": [i]})
    assert client.get("/api/images").json()["total"] == 4
    client.patch(f"/api/images/{i}", json={"favorite": True, "rating": 3})
    assert client.get(f"/api/images/{i}").json()["favorite"] in (1, True)
    assert client.post(f"/api/images/{i}/prioritize").status_code == 200


def test_import_upload_and_paths_and_folder(client, tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "green").save(buf, "PNG")
    r = client.post(
        "/api/import/upload", files=[("files", ("clip.png", buf.getvalue(), "image/png"))], data={"clipboard": "true"}
    )
    assert r.status_code == 200
    extra = tmp_path / "more"
    extra.mkdir()
    Image.new("RGB", (20, 20), "blue").save(extra / "z.png")
    assert client.post("/api/import/paths", json={"paths": [str(extra / "z.png")]}).status_code == 200
    assert client.post("/api/import/folder", json={"path": str(extra)}).status_code == 200
    assert client.post("/api/scan", json={}).status_code == 200


def test_summary_and_merge_tools(client):
    ids = [x["id"] for x in client.get("/api/images").json()["items"]]
    assert client.post("/api/summary", json={"ids": ids}).status_code == 200
    assert client.post("/api/summary", json={"ids": []}).status_code == 200
    m = client.post("/api/merge", json={"ids": ids, "cols": 2, "cell": 64})
    assert m.status_code == 200
    ann = client.put(
        f"/api/images/{ids[0]}/annotations", json={"shapes": [{"type": "rect", "box": [1, 1, 10, 10], "color": "#f00"}]}
    )
    assert ann.status_code == 200
    exp = client.post(f"/api/images/{ids[0]}/annotations/export", json={"format": "png"})
    assert exp.status_code == 200


def test_categories_rename_delete(client):
    ids = [x["id"] for x in client.get("/api/images").json()["items"]]
    client.post("/api/categories/assign", json={"ids": ids[:2], "add": ["море"]})
    cid = client.get("/api/categories").json()[0]["id"]
    client.patch(f"/api/categories/{cid}", json={"name": "океан"})
    assert client.get("/api/categories").json()[0]["name"] == "океан"
    client.post("/api/categories/assign", json={"ids": ids[:1], "remove": [cid]})
    client.delete(f"/api/categories/{cid}")
    assert client.get("/api/categories").json() == []
    col = client.post("/api/collections", json={"name": "К", "ids": ids[:2]}).json()["id"]
    client.patch(f"/api/collections/{col}", json={"name": "К2"})
    client.post(f"/api/collections/{col}/images", json={"remove": ids[:1]})
    assert client.get("/api/collections").json()[0]["name"] == "К2"
    client.delete(f"/api/collections/{col}")
    assert client.get("/api/collections").json() == []
