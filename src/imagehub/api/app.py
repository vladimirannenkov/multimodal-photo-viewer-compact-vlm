"""Приложение FastAPI: API под /api, веб-интерфейс из web/static."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import db, scanner, search, watcher
from ..config import get_settings
from ..pipeline import get_pipeline
from . import chat, faces, library, meta, tools

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    db.conn()
    pipe = get_pipeline()
    pipe.listeners.append(lambda ids: search.invalidate())
    if s.autostart_pipeline:
        pipe.start()
    # быстрая проверка новых файлов во всех наблюдаемых папках, затем слежение в реальном времени
    import threading

    def boot():
        try:
            scanner.scan_all(deep=False)
            watcher.start(lambda r: pipe.kick())
            pipe.kick()
        except Exception:
            logging.getLogger("imagehub").exception("startup scan failed")
    threading.Thread(target=boot, daemon=True).start()
    yield
    pipe.stop()
    watcher.stop()
    scanner.shutdown_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="ImageHub", version="0.1.0", lifespan=lifespan)
    for m in (library, meta, faces, chat, tools):
        app.include_router(m.router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/themes", include_in_schema=False)
    def themes():
        return FileResponse(STATIC / "themes.html", headers={"Cache-Control": "no-cache"})

    return app
