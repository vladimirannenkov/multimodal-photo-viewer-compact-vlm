"""Small CLI client for the documented upload API."""

import argparse
import json
import sys

import httpx


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqa")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    sub = ap.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("image")
    ask = sub.add_parser("ask")
    ask.add_argument("image")
    ask.add_argument("prompt")
    args = ap.parse_args(argv)
    try:
        with open(args.image, "rb") as image:
            response = httpx.post(
                f"{args.url.rstrip('/')}/{args.command}",
                files={"file": image},
                data={"prompt": args.prompt} if args.command == "ask" else {},
                timeout=300,
            )
        response.raise_for_status()
    except (OSError, httpx.HTTPError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    sys.stdout.write(json.dumps(response.json(), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
