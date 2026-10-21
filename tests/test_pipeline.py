"""Пайплайн обработки целиком: нейросети-адаптеры заменены заглушками, сервер моделей замокан."""

import json

import httpx
import numpy as np
import pytest
from PIL import Image

from conftest import completion
from imagehub import db, people, scanner
from imagehub.pipeline import MAX_SIDE, Pipeline, refine_kind, scale_boxes


@pytest.fixture
def library(env, photos):
    ids = scanner.scan_folder(scanner.add_folder(str(photos)))["ids"]
    return sorted(ids)


def row(i):
    return db.q1("SELECT * FROM images WHERE id=?", (i,))


# ---------- чистая логика ----------
@pytest.mark.parametrize(
    ("kind", "scores", "lines", "expected"),
    [
        ("document", {"photo": 0.3, "document": 0.6}, 1, "photo"),  # страница без текста — фото
        ("receipt", {"photo": 0.05}, 0, "receipt"),  # фото не правдоподобно: оставляем
        ("meme", {"meme": 0.5, "photo": 0.3}, 5, "photo"),  # редкий тип с низкой уверенностью
        ("diagram", {"diagram": 0.95, "photo": 0.2}, 4, "diagram"),
        ("photo", {"document": 0.7, "screenshot": 0.2}, 12, "document"),  # много строк текста
        ("chart", {"chart": 0.5, "document": 0.4}, 40, "document"),
        ("chart", {"chart": 0.95, "document": 0.4}, 40, "chart"),
        ("photo", {}, 0, "photo"),
    ],
)
def test_refine_kind(kind, scores, lines, expected):
    assert refine_kind(kind, scores, {"lines": lines}) == expected


def test_scale_boxes():
    boxes = [{"label": "a", "box": [10, 20, 30, 40]}, {"label": "no box"}]
    assert scale_boxes(boxes, 1) is boxes
    assert scale_boxes(boxes, 2) == [{"label": "a", "box": [20, 40, 60, 80]}, {"label": "no box"}]


def test_chart_summary_mentions_axes_and_best_series():
    data = {
        "title": "Loss",
        "x_axis": "epoch",
        "y_axis": "loss",
        "stats": {
            "best_series": "A",
            "series": [{"name": "A", "max": 0.9, "plateau": 0.88, "max_at": 10, "min": 0.1, "final": 0.88}],
        },
    }
    text = Pipeline._chart_summary(data)
    assert "Loss" in text and "epoch" in text and "A" in text


# ---------- обработка волны ----------
def test_wave_for_photos_fills_metadata(library, fake_models, llm_server):
    llm_server.chat.respond(json=completion("Кошка спит на диване."))
    pipe = Pipeline()
    pipe.run_wave(library * 1)  # 4 изображения: больше трёх, значит перевод идёт сразу
    for i in library:
        r = row(i)
        assert r["status"] == "done" and r["error"] is None and r["kind"] == "photo"
        assert r["caption"] == "A cat is sleeping on a sofa." and r["summary_ru"] == "Кошка спит на диване."
        assert json.loads(r["tags"]) == ["cat"] and json.loads(r["objects"])[0]["label"] == "cat"
        assert json.loads(r["timings"])["embed"] >= 0
    assert db.q1("SELECT COUNT(*) c FROM vectors WHERE kind='img'")["c"] == 4
    cats = {c["name"] for c in db.q("SELECT name FROM categories")}
    assert {"фото", "cat"} <= cats and "прогулка" in cats
    assert db.q1("SELECT COUNT(*) c FROM fts WHERE fts MATCH 'cat'")["c"] == 4


def test_small_wave_defers_translation_and_backfill_fills_it(library, fake_models, llm_server):
    llm_server.chat.respond(json=completion("Кошка"))
    pipe = Pipeline()
    pipe.run_wave(library[:2])
    assert row(library[0])["summary_ru"] is None  # единичные файлы готовы без ожидания перевода
    assert llm_server.chat.call_count == 0
    seen = []
    pipe.listeners.append(seen.append)
    assert pipe.backfill_translation() is True
    assert row(library[0])["summary_ru"] == "Кошка" and seen
    assert pipe.backfill_translation() is False


def test_wave_ocr_fix_and_chart(library, fake_models, llm_server):
    fake_models.textdet.update(has_text=True, lines=9, language="ru", sample="Привет")
    fake_models.siglip.kind = "document"
    answers = iter(["Текст страницы после распознавания", "Текст страницы после распознавания"])
    llm_server.chat.side_effect = lambda request: httpx.Response(
        200, json=completion(next(answers, "Текст страницы после распознавания"))
    )
    pipe = Pipeline()
    pipe.run_wave(library[:1])
    r = row(library[0])
    assert r["status"] == "done" and r["has_text"] == 1 and r["text_lang"] == "ru"
    assert "Текст страницы" in r["ocr_md"] and json.loads(r["ocr_blocks"])
    assert r["caption"] is None  # у документов описание сцены не строится


def test_wave_chart_creates_experiment(library, fake_models, llm_server):
    fake_models.siglip.kind = "chart"
    chart = {
        "title": "Accuracy",
        "x_axis": "epoch",
        "y_axis": "acc",
        "series": [{"name": "run", "points": [[i, 1 - 1 / (i + 1)] for i in range(12)]}],
    }
    llm_server.chat.respond(json=completion(json.dumps(chart)))
    Pipeline().run_wave(library[:2])
    r = row(library[0])
    data = json.loads(r["chart"])
    assert data["title"] == "Accuracy" and data["stats"]["best_series"] == "run"
    assert r["experiment_id"] is not None and row(library[1])["experiment_id"] == r["experiment_id"]
    exp = db.q("SELECT * FROM experiments")
    assert len(exp) == 1 and exp[0]["auto"] == 1


def test_chart_false_answer_is_skipped(library, fake_models, llm_server):
    fake_models.siglip.kind = "chart"
    llm_server.chat.respond(json=completion('{"chart": false}'))
    Pipeline().run_wave(library[:1])
    assert row(library[0])["chart"] is None


def test_stage_failure_is_recorded_not_fatal(library, fake_models, llm_server):
    llm_server.chat.respond(500, text="down")
    llm_server.chat.side_effect = None
    fake_models.siglip.kind = "chart"
    pipe = Pipeline()
    pipe.run_wave(library[:1])
    r = row(library[0])
    assert r["status"] == "done" and "chart" in r["error"]
    assert "chart" in pipe.state["last_error"]


def test_unreadable_file_marks_error(env, tmp_path, fake_models):
    bad = tmp_path / "broken.png"
    Image.new("RGB", (10, 10)).save(bad)
    ids = scanner.scan_folder(scanner.add_folder(str(tmp_path)))["ids"]
    bad.write_bytes(b"not an image")
    Pipeline().run_wave(ids)
    assert row(ids[0])["status"] == "error" and row(ids[0])["error"].startswith("open:")


def test_faces_stage_groups_people(library, fake_models, llm_server):
    emb = np.eye(4, dtype=np.float32)[0]
    fake_models.faces = [{"box": [5, 5, 20, 20], "score": 0.99, "emb": emb}]
    # fake_models.faces — это список, который читает заглушка faces.detect
    Pipeline().run_wave(library[:2])
    assert db.q1("SELECT COUNT(*) c FROM faces")["c"] == 2
    assert db.q1("SELECT COUNT(*) c FROM people")["c"] == 1  # одинаковое лицо — один человек


def test_people_matching_rename_merge(env):
    a, b = np.eye(4, dtype=np.float32)[0], np.eye(4, dtype=np.float32)[1]
    db.ex("INSERT INTO folders(path) VALUES('x')")
    fid = db.q1("SELECT id FROM folders")["id"]
    for n in ("p1", "p2"):
        db.ex("INSERT INTO images(path,name,folder_id,added_at) VALUES(?,?,?,0)", (n, n, fid))
    i1, i2 = [r["id"] for r in db.q("SELECT id FROM images ORDER BY id")]
    f1 = people.add_faces(i1, [{"box": [0, 0, 1, 1], "score": 1.0, "emb": a}])[0]
    f2 = people.add_faces(i2, [{"box": [0, 0, 1, 1], "score": 1.0, "emb": b}])[0]
    assert f1["person_id"] != f2["person_id"]
    again = people.add_faces(i1, [{"box": [0, 0, 1, 1], "score": 1.0, "emb": a}])[0]
    assert again["person_id"] == f1["person_id"] and again["sim"] > 0.9
    people.rename(f1["person_id"], "Анна")
    assert people.person_label(db.q1("SELECT * FROM people WHERE id=?", (f1["person_id"],))) == "Анна"
    assert people.next_unlabeled()["person_id"] == f2["person_id"]
    assert people.candidate_for(f2["person_id"], min_sim=0.9) is None  # векторы ортогональны
    assert people.rename(f2["person_id"], "Анна") == f1["person_id"]  # то же имя: люди объединяются
    assert db.q1("SELECT COUNT(*) c FROM people")["c"] == 1
    assert people.next_unlabeled() is None


# ---------- цикл и управление ----------
def test_start_requeues_interrupted_and_loop_processes(library, fake_models, llm_server):
    import time

    db.ex("UPDATE images SET status='processing'")
    pipe = Pipeline()
    pipe.start()
    pipe.start()  # повторный запуск безопасен
    try:
        deadline = time.time() + 30
        while time.time() < deadline and pipe.queue_counts().get("done", 0) < len(library):
            pipe.kick()
            time.sleep(0.2)
        st = pipe.status()
    finally:
        pipe.stop()
    assert st["counts"]["done"] == len(library) and st["alive"] and st["pending"] == 0


def test_paused_pipeline_does_not_process(library, fake_models):
    import time

    pipe = Pipeline()
    pipe.paused = True
    pipe.start()
    time.sleep(0.5)
    pipe.stop()
    assert pipe.queue_counts().get("done", 0) == 0 and pipe.state["stage"] == "paused"


def test_large_image_scale_factor(env, tmp_path, fake_models, llm_server):
    big = tmp_path / "big.png"
    Image.new("RGB", (MAX_SIDE * 2, 400), "white").save(big)
    ids = scanner.scan_folder(scanner.add_folder(str(tmp_path)))["ids"]
    Pipeline().run_wave(ids)
    box = json.loads(row(ids[0])["objects"])[0]["box"]
    assert box == [2.0, 4.0, 60.0, 80.0]  # рамки пересчитаны к размеру оригинала (коэффициент 2)


def test_text_embedding_stage_stores_vectors(library, fake_models, llm_server):
    Pipeline().run_wave(library)
    assert db.q1("SELECT COUNT(*) c FROM vectors WHERE kind='txt'")["c"] == len(library)
    assert llm_server.embeddings.called
