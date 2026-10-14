"""Фильтры и списки изображений для сетки и левой панели."""
import json

from . import db

SORTS = {
    "date": "COALESCE(i.taken_at, i.mtime)",
    "added": "i.added_at",
    "size": "i.size",
    "resolution": "(i.width * i.height)",
    "name": "i.name COLLATE NOCASE",
    "kind": "i.kind",
    "rating": "i.rating",
}
LIST_COLS = ("i.id, i.name, i.path, i.width, i.height, i.size, i.kind, i.event, i.status, i.favorite, i.rating, i.has_text, "
             "i.text_lang, i.from_clipboard, i.experiment_id, COALESCE(i.taken_at, i.mtime) AS date, i.folder_id, i.frames, i.format")


def build_where(f: dict):
    """f: folder, collection, category, experiment, person, kind, favorite, has_text, text_lang, date_from, date_to, clipboard, ids."""
    w, a = ["i.deleted=0"], []
    if f.get("folder"):
        r = db.q1("SELECT path FROM folders WHERE id=?", (f["folder"],))
        if r:
            w.append("i.folder_id=?")
            a.append(f["folder"])
    if f.get("path_prefix"):
        w.append("substr(i.path,1,?)=?")
        a += [len(f["path_prefix"]), f["path_prefix"]]
    if f.get("collection"):
        w.append("i.id IN (SELECT image_id FROM collection_images WHERE collection_id=?)")
        a.append(f["collection"])
    if f.get("category"):
        w.append("i.id IN (SELECT image_id FROM image_categories WHERE category_id=?)")
        a.append(f["category"])
    if f.get("experiment"):
        w.append("i.experiment_id=?")
        a.append(f["experiment"])
    if f.get("person"):
        w.append("i.id IN (SELECT image_id FROM faces WHERE person_id=?)")
        a.append(f["person"])
    if f.get("kind"):
        w.append("i.kind=?")
        a.append(f["kind"])
    if f.get("favorite"):
        w.append("i.favorite=1")
    if f.get("clipboard"):
        w.append("i.from_clipboard=1")
    if f.get("has_text") is not None and f.get("has_text") != "":
        w.append("i.has_text=?")
        a.append(int(f["has_text"]))
    if f.get("date_from"):
        w.append("COALESCE(i.taken_at, i.mtime)>=?")
        a.append(f["date_from"])
    if f.get("date_to"):
        w.append("COALESCE(i.taken_at, i.mtime)<=?")
        a.append(f["date_to"])
    if f.get("ids"):
        w.append("i.id IN (%s)" % ",".join("?" * len(f["ids"])))
        a += list(f["ids"])
    return " AND ".join(w), a


def list_images(f: dict, sort="date", desc=True, offset=0, limit=200):
    where, args = build_where(f)
    order = SORTS.get(sort, SORTS["date"]) + (" DESC" if desc else " ASC")
    total = db.q1(f"SELECT COUNT(*) c FROM images i WHERE {where}", args)["c"]
    rows = db.q(f"SELECT {LIST_COLS} FROM images i WHERE {where} ORDER BY {order}, i.id LIMIT ? OFFSET ?", args + [limit, offset])
    return total, [dict(r) for r in rows]


def candidate_ids(f: dict):
    where, args = build_where(f)
    return {r["id"] for r in db.q(f"SELECT i.id FROM images i WHERE {where}", args)}


def sidebar():
    folders = [dict(r) for r in db.q("""SELECT f.id, f.path, f.watched, (SELECT COUNT(*) FROM images i WHERE i.folder_id=f.id AND i.deleted=0) AS n
                                         FROM folders f ORDER BY f.path""")]
    cats = [dict(r) for r in db.q("""SELECT c.id, c.name, c.auto, COUNT(ic.image_id) AS n FROM categories c
                                     JOIN image_categories ic ON ic.category_id=c.id JOIN images i ON i.id=ic.image_id AND i.deleted=0
                                     GROUP BY c.id ORDER BY n DESC""")]
    colls = [dict(r) for r in db.q("""SELECT c.id, c.name, (SELECT COUNT(*) FROM collection_images ci JOIN images i ON i.id=ci.image_id AND i.deleted=0
                                       WHERE ci.collection_id=c.id) AS n FROM collections c ORDER BY c.name""")]
    exps = [dict(r) for r in db.q("""SELECT e.id, e.name, e.created_at, e.auto, (SELECT COUNT(*) FROM images i WHERE i.experiment_id=e.id AND i.deleted=0) AS n
                                      FROM experiments e ORDER BY e.created_at DESC""")]
    ppl = [dict(r) for r in db.q("""SELECT p.id, COALESCE(p.name, p.auto_name, 'person_' || p.id) AS name, p.labeled, p.n,
                                     (SELECT f.id FROM faces f WHERE f.person_id=p.id ORDER BY f.score DESC LIMIT 1) AS face_id
                                     FROM people p WHERE p.n>0 ORDER BY p.labeled DESC, p.n DESC LIMIT 200""")]
    kinds = [dict(r) for r in db.q("SELECT kind, COUNT(*) n FROM images WHERE deleted=0 AND kind IS NOT NULL GROUP BY kind ORDER BY n DESC")]
    total = db.q1("SELECT COUNT(*) c FROM images WHERE deleted=0")["c"]
    return {"total": total, "folders": folders, "top_categories": cats[:5], "categories": cats, "collections": colls,
            "experiments": exps, "people": ppl, "kinds": kinds,
            "clipboard": db.q1("SELECT COUNT(*) c FROM images WHERE from_clipboard=1 AND deleted=0")["c"]}


def image_detail(image_id):
    r = db.q1("SELECT * FROM images WHERE id=?", (image_id,))
    if r is None:
        return None
    d = db.row_dict(r, ("kind_scores", "ocr_blocks", "objects", "regions", "tags", "chart", "stages", "timings"))
    d["categories"] = [dict(x) for x in db.q("""SELECT c.id, c.name, ic.source FROM image_categories ic JOIN categories c ON c.id=ic.category_id
                                                WHERE ic.image_id=? ORDER BY ic.source DESC, c.name""", (image_id,))]
    d["collections"] = [dict(x) for x in db.q("""SELECT c.id, c.name FROM collection_images ci JOIN collections c ON c.id=ci.collection_id
                                                  WHERE ci.image_id=?""", (image_id,))]
    d["faces"] = [{**dict(x), "box": json.loads(x["box"])} for x in db.q(
        """SELECT f.id, f.box, f.score, f.person_id, f.confirmed, COALESCE(p.name, p.auto_name) AS person, p.labeled
           FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE f.image_id=?""", (image_id,))]
    d["experiment"] = dict(e) if (e := db.q1("SELECT id, name FROM experiments WHERE id=?", (d["experiment_id"],))) else None
    d["annotations"] = json.loads(a["data"]) if (a := db.q1("SELECT data FROM annotations WHERE image_id=?", (image_id,))) else []
    return d
