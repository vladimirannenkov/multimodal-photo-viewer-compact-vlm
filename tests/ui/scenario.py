"""Сценарий проверки интерфейса: python tests/ui/scenario.py  (сервер на :8765)"""
import json
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8765"
OUT = "C:/ML/Project-Image/tests/ui/sc_"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
get = lambda p: json.loads(opener.open(BASE + p).read())

docs = get("/api/images?kind=document&limit=50")["items"]
mixed = next(i for i in docs if "mixed" in i["path"])
photo = next(i for i in get("/api/images?kind=photo&limit=50")["items"] if "scene_hq_02" in i["path"] or True)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1500, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(BASE)
    pg.wait_for_timeout(1500)
    # карточка документа: вкладка «Текст»
    pg.evaluate(f"document.querySelector('.cell img[alt=\"{mixed['name']}\"]')?.click()")
    pg.wait_for_timeout(300)
    pg.locator(".cell").first.click()
    pg.click(".tab[data-k=text]")
    pg.wait_for_timeout(1000)
    pg.screenshot(path=OUT + "text.png")
    # фото: описание
    pg.click("text=Фото >> nth=0") if False else None
    pg.goto(BASE)
    pg.wait_for_timeout(800)
    pg.fill("#q", "sunset over the ocean")
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#banner:not([hidden])", timeout=60000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=OUT + "search.png")
    pg.locator(".cell").first.click()
    pg.wait_for_timeout(800)
    pg.screenshot(path=OUT + "photo.png")
    pg.locator(".cell").first.dblclick()
    pg.wait_for_timeout(800)
    pg.click("#bObjects")
    pg.wait_for_timeout(400)
    pg.screenshot(path=OUT + "viewer_objects.png")
    print("\n".join(logs) or "no console errors")
    b.close()
