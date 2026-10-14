import os
import time

from PIL import Image

from imagehub import db, scanner


def test_scan_finds_new_files_and_thumbs(env, photos):
    fid = scanner.add_folder(str(photos))
    r = scanner.scan_folder(fid)
    assert r["new"] == 4 and r["seen"] == 4
    rows = db.q("SELECT * FROM images")
    assert all(x["hash"] and x["width"] for x in rows)
    assert all(scanner.thumb_path(x["id"]).exists() for x in rows)


def test_fast_rescan_skips_unchanged_dirs(env, photos):
    fid = scanner.add_folder(str(photos))
    scanner.scan_folder(fid)
    r = scanner.scan_folder(fid)
    assert r["seen"] == 0 and r["new"] == 0  # каталоги не менялись: файлы не проверялись
    deep = scanner.scan_folder(fid, deep=True)
    assert deep["seen"] == 4 and deep["new"] == 0


def test_detects_added_and_removed(env, photos):
    fid = scanner.add_folder(str(photos))
    scanner.scan_folder(fid)
    Image.new("RGB", (50, 50), "red").save(photos / "sub" / "new.png")
    os.remove(photos / "a.png")
    time.sleep(0.05)
    r = scanner.scan_folder(fid)
    assert r["new"] == 1 and r["removed"] == 1
    assert db.q1("SELECT COUNT(*) c FROM images WHERE deleted=0")["c"] == 4


def test_modified_file_is_requeued(env, photos):
    fid = scanner.add_folder(str(photos))
    scanner.scan_folder(fid)
    db.ex("UPDATE images SET status='done'")
    time.sleep(0.05)
    Image.new("RGB", (300, 200), "blue").save(photos / "b.jpg")
    r = scanner.scan_folder(fid, deep=True)
    assert r["updated"] == 1
    row = db.q1("SELECT status, width FROM images WHERE name='b.jpg'")
    assert row["status"] == "new" and row["width"] == 300
