"""Слежение за папками в реальном времени (watchdog) с дребезгом: пачка событий даёт один быстрый скан."""
import threading

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from . import db, scanner

_obs = None
_timers = {}
_lock = threading.Lock()


class _H(FileSystemEventHandler):
    def __init__(self, folder_id, on_new):
        self.folder_id, self.on_new = folder_id, on_new

    def on_any_event(self, event):
        if event.event_type in ("opened", "closed_no_write"):
            return
        with _lock:
            t = _timers.pop(self.folder_id, None)
            if t:
                t.cancel()
            t = threading.Timer(1.5, self._scan)
            _timers[self.folder_id] = t
            t.start()

    def _scan(self):
        try:
            r = scanner.scan_folder(self.folder_id)
            if r["new"] or r["updated"] or r["removed"]:
                self.on_new(r)
        except Exception:
            pass


def start(on_new=lambda r: None):
    global _obs
    stop()
    _obs = Observer()
    for f in db.q("SELECT id, path FROM folders WHERE watched=1"):
        try:
            _obs.schedule(_H(f["id"], on_new), f["path"], recursive=True)
        except Exception:
            pass
    _obs.start()


def stop():
    global _obs
    if _obs:
        _obs.stop()
        _obs = None
