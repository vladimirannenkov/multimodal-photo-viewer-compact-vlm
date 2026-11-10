import json
import sqlite3

import numpy as np
from PIL import Image


def test_sync_completed_results_preserves_user_data_and_is_idempotent(env):
    from imagehub import db, external_results
    from imagehub.config import get_settings

    root = get_settings().data_dir
    image = env / "image.png"
    Image.new("RGB", (24, 16)).save(image)
    stat = image.stat()
    target_id = db.ex(
        "INSERT INTO images(path,name,size,mtime,hash,status,favorite) VALUES(?,?,?,?,?,'new',1)",
        (str(image), image.name, stat.st_size, stat.st_mtime, "testhash"),
    ).lastrowid
    source_dir = root / "experiment"
    source_dir.mkdir()
    source_path = source_dir / "source.sqlite"
    with sqlite3.connect(source_path) as source:
        source.executescript(db.SCHEMA)
        source.execute("ALTER TABLE images ADD COLUMN summary_ru TEXT")
        source.execute("ALTER TABLE images ADD COLUMN tags_ru TEXT")
        source.execute(
            "INSERT INTO images(id,path,name,size,mtime,hash,status,caption,summary,tags,kind,pipeline_ver) VALUES(1,?,?,?,?,?,'done','a red cat','a red cat','[\"cat\"]','photo',2)",
            (str(image), image.name, stat.st_size, stat.st_mtime, "testhash"),
        )
        source.execute(
            "INSERT INTO vectors(image_id,kind,dim,data) VALUES(1,'img',2,?)", (np.array([0.6, 0.8], dtype=np.float16).tobytes(),)
        )
        source.execute("INSERT INTO categories(id,name,auto) VALUES(1,'cat',1)")
        source.execute("INSERT INTO image_categories(image_id,category_id,score,source) VALUES(1,1,1,'auto')")
        source.execute("INSERT INTO people(id,auto_name,n) VALUES(1,'person_1',1)")
        source.execute(
            "INSERT INTO faces(image_id,box,score,emb,person_id) VALUES(1,'[1,2,3,4]',0.9,?,1)",
            (np.array([1.0, 0.0], dtype=np.float32).tobytes(),),
        )
    (root / "evaluation-sources.json").write_text(json.dumps({"sources": [str(source_path)]}), encoding="utf-8")
    assert external_results.sync_sources(force=True) == 1
    row = db.q1("SELECT * FROM images WHERE id=?", (target_id,))
    assert row["status"] == "done" and row["caption"] == "a red cat" and row["favorite"] == 1
    assert db.q1("SELECT count(*) AS n FROM faces WHERE image_id=?", (target_id,))["n"] == 1
    assert db.q1("SELECT count(*) AS n FROM image_categories WHERE image_id=?", (target_id,))["n"] == 1
    assert db.q1("SELECT count(*) AS n FROM fts WHERE fts MATCH 'cat'")["n"] == 1
    assert db.q1("SELECT count(*) AS n FROM vectors WHERE image_id=? AND kind='img'", (target_id,))["n"] == 1
    assert external_results.sync_sources(force=True) == 0
    db.ex("UPDATE images SET status='new' WHERE id=?", (target_id,))
    assert external_results.sync_sources(force=True) == 0
    assert db.q1("SELECT status FROM images WHERE id=?", (target_id,))["status"] == "new"


def test_sync_skips_changed_file(env):
    from imagehub import db, external_results
    from imagehub.config import get_settings

    root = get_settings().data_dir
    image = env / "changed.png"
    Image.new("RGB", (24, 16)).save(image)
    stat = image.stat()
    target_id = db.ex(
        "INSERT INTO images(path,name,size,mtime,hash,status) VALUES(?,?,?,?,?,'new')",
        (str(image), image.name, stat.st_size, stat.st_mtime, "hash"),
    ).lastrowid
    source_path = root / "source.sqlite"
    with sqlite3.connect(source_path) as source:
        source.executescript(db.SCHEMA)
        source.execute("ALTER TABLE images ADD COLUMN summary_ru TEXT")
        source.execute("ALTER TABLE images ADD COLUMN tags_ru TEXT")
        source.execute(
            "INSERT INTO images(path,name,size,mtime,hash,status) VALUES(?,?,?,?,?,'done')",
            (str(image), image.name, stat.st_size, stat.st_mtime, "hash"),
        )
    (root / "evaluation-sources.json").write_text(json.dumps({"sources": [str(source_path)]}), encoding="utf-8")
    image.write_bytes(b"changed")
    assert external_results.sync_sources(force=True) == 0
    assert db.q1("SELECT status FROM images WHERE id=?", (target_id,))["status"] == "new"
