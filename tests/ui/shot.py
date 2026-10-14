"""Снимки интерфейса через Playwright (системный Chrome): python tests/ui/shot.py [url] [out_prefix]"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/ML/Project-Image/tests/ui/out"

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1500, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(URL)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=OUT + "_main.png")
    cells = pg.locator(".cell")
    print("cells:", cells.count())
    if cells.count():
        cells.nth(0).click()
        pg.wait_for_timeout(1200)
        pg.screenshot(path=OUT + "_sel.png")
        cells.nth(0).dblclick()
        pg.wait_for_timeout(1500)
        pg.screenshot(path=OUT + "_viewer.png")
    print("\n".join(logs) or "no console errors")
    b.close()
