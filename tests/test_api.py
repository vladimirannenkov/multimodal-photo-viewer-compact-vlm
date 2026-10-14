import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagehub import db, scanner


@pytest.fixture()
def client(env, photos):
    from imagehub.api.app import create_app
    with TestClient(create_app()) as c:
        fid = scanner.add_folder(str(photos))
        scanner.scan_folder(fid)
        yield c


def ids_of(client):
    return [x["id"] for x in client.get("/api/images").json()["items"]]


def test_list_and_detail(client):
    r = client.get("/api/images").json()
    assert r["total"] == 4
    d = client.get(f"/api/images/{r['items'][0]['id']}").json()
    assert d["width"] > 0 and d["categories"] == []
    assert client.get("/api/images/9999").status_code == 404


def test_thumb_keeps_aspect_and_file(client):
    i = ids_of(client)[0]
    assert client.get(f"/api/images/{i}/thumb").headers["content-type"] == "image/webp"
    assert client.get(f"/api/images/{i}/file").status_code == 200


def test_categories_are_multi_tag(client):
    ids = ids_of(client)
    client.post("/api/categories/assign", json={"ids": ids[:2], "add": ["путешествия", "море"]})
    d = client.get(f"/api/images/{ids[0]}").json()
    assert {c["name"] for c in d["categories"]} == {"путешествия", "море"}
    cid = next(c["id"] for c in client.get("/api/categories").json() if c["name"] == "море")
    assert client.get(f"/api/images?category={cid}").json()["total"] == 2


def test_collections_and_favorites(client):
    ids = ids_of(client)
    cid = client.post("/api/collections", json={"name": "Лучшее", "ids": []}).json()["id"]
    client.post(f"/api/collections/{cid}/images", json={"add": ids[:3]})
    assert client.get(f"/api/images?collection={cid}").json()["total"] == 3
    client.patch(f"/api/images/{ids[0]}", json={"favorite": 1})
    assert client.get("/api/images?favorite=true").json()["total"] == 1


def test_experiment_rename_keeps_created_at(client):
    eid = client.post("/api/experiments", json={"name": "Опыт 1"}).json()["id"]
    before = [e for e in client.get("/api/experiments").json() if e["id"] == eid][0]["created_at"]
    client.patch(f"/api/experiments/{eid}", json={"name": "Опыт 1 (новое имя)"})
    after = [e for e in client.get("/api/experiments").json() if e["id"] == eid][0]
    assert after["created_at"] == before and after["name"].endswith("(новое имя)")


def test_merge_grid_and_export(client):
    ids = ids_of(client)
    r = client.post("/api/merge", json={"ids": ids, "rows": 2, "cols": 2, "cell": 64, "gap": 4})
    assert Image.open(io.BytesIO(r.content)).size == (2 * 64 + 3 * 4, 2 * 64 + 3 * 4)
    z = client.post("/api/export", json={"ids": ids, "format": "webp"})
    assert z.headers["content-type"] == "application/zip"
    one = client.post("/api/export", json={"ids": ids[:1], "format": "jpg", "max_side": 50})
    assert max(Image.open(io.BytesIO(one.content)).size) <= 50


def test_edit_creates_copy_without_touching_original(client):
    i = client.get("/api/images?sort=name&desc=false").json()["items"][0]
    orig = open(i["path"], "rb").read()
    r = client.post(f"/api/images/{i['id']}/edit", json={"ops": [{"type": "crop", "box": [0, 0, 40, 30]}], "format": "png"}).json()
    new = client.get(f"/api/images/{r['id']}").json()
    assert (new["width"], new["height"]) == (40, 30) and new["path"] != i["path"]
    assert open(i["path"], "rb").read() == orig


def test_upload_clipboard_flag(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "green").save(buf, "PNG")
    r = client.post("/api/import/upload", data={"clipboard": "true"}, files={"files": ("x.png", buf.getvalue(), "image/png")}).json()
    assert len(r["ids"]) == 1
    assert client.get("/api/images?clipboard=true").json()["total"] == 1
    assert client.post("/api/clipboard/clear").status_code in (404, 405)  # массового удаления вставок нет


def test_fts_search_both_languages_and_prefix(client):
    ids = ids_of(client)
    db.ex("UPDATE images SET summary='A red car on a wet street at night', summary_ru='Красная машина на мокрой улице ночью' WHERE id=?", (ids[0],))
    db.ex("UPDATE images SET ocr_md='Счёт за электричество, март' WHERE id=?", (ids[1],))
    for i in ids[:2]:
        db.fts_update(i)
    from imagehub import search
    assert list(search.bm25("car street", None)) == [ids[0]]
    assert list(search.bm25("машина улице", None)) == [ids[0]]  # поиск по русскому переводу описания
    assert list(search.bm25("электричество", None)) == [ids[1]]
    assert list(search.bm25("элек", None)) == [ids[1]]


def test_summary_and_duplicates(client):
    assert client.post("/api/summary", json={}).json()["count"] == 4
    assert client.get("/api/duplicates").status_code == 200


def test_first_experiment_is_created_automatically_once(env):
    from imagehub.pipeline import Pipeline
    p = Pipeline()
    row = {"taken_at": None, "mtime": 1_700_000_000.0}
    a = p.ensure_experiment(row)
    b = p.ensure_experiment(row)
    assert a == b and db.q1("SELECT COUNT(*) c FROM experiments")["c"] == 1
    assert db.q1("SELECT auto FROM experiments WHERE id=?", (a,))["auto"] == 1
