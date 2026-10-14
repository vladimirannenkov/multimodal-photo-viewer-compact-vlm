"""Запуск: imagehub serve [--host --port] | imagehub scan PATH | imagehub models."""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="imagehub")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    sc = sub.add_parser("scan")
    sc.add_argument("path")
    m = sub.add_parser("models")
    m.add_argument("--download", action="store_true", help="скачать веса в каталог весов")
    m.add_argument("--only", default="")
    a = ap.parse_args()
    from .config import get_settings
    cfg = get_settings()
    if a.cmd in (None, "serve"):
        import uvicorn
        uvicorn.run("imagehub.api.app:create_app", factory=True, host=getattr(a, "host", None) or cfg.host,
                    port=getattr(a, "port", None) or cfg.port, log_level="info")
    elif a.cmd == "scan":
        from . import scanner
        fid = scanner.add_folder(a.path)
        r = scanner.scan_folder(fid, deep=True)
        r.pop("ids", None)
        print(r)
    elif a.cmd == "models":
        from . import models
        if a.download:
            models.download(set(filter(None, a.only.split(","))) or None)
        print("Каталог весов:", cfg.weights_dir)
        for k, ok in models.status().items():
            repo, what, gb = models.REGISTRY[k]
            print(f"{'OK ' if ok else '-- '} {k:16} ~{gb:>5} ГБ  {what}")


if __name__ == "__main__":
    main()
