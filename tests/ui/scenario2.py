"""Проверка навигации стрелками, меню, адаптивности: python tests/ui/scenario2.py"""
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8765"
OUT = "C:/ML/Project-Image/tests/ui/s2_"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    ctx = b.new_context(viewport={"width": 1100, "height": 760})
    ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=BASE)
    pg = ctx.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(BASE)
    pg.wait_for_selector(".cell")
    # стрелки в сетке
    pg.locator(".cell").nth(0).click()
    pg.keyboard.press("ArrowRight"); pg.keyboard.press("ArrowRight")
    print("active after arrows:", pg.evaluate("document.querySelector('.cell.act')?.title"))
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#viewer:not([hidden])")
    pg.wait_for_timeout(800)
    name = lambda: pg.evaluate("document.querySelector('#vinfo').textContent.split('  ·  ')[0]")
    n0 = name()
    pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(500)
    n1 = name()
    # выбрать инструмент и двигать ползунок, затем стрелка должна всё равно листать
    pg.click("[data-tool=rect]")
    pg.locator("#vbar input[type=range]").first.focus()
    pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(500)
    n2 = name()
    print("viewer nav:", n0, "->", n1, "->", n2)
    # рисование
    box = pg.locator("#vstage").bounding_box()
    pg.mouse.move(box["x"] + 300, box["y"] + 300); pg.mouse.down(); pg.mouse.move(box["x"] + 500, box["y"] + 420); pg.mouse.up()
    pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(500)
    print("after draw + left:", name())
    pg.screenshot(path=OUT + "viewer_narrow.png")
    pg.keyboard.press("Escape")
    # контекстное меню
    pg.locator(".cell").nth(1).click(button="right")
    pg.wait_for_timeout(300)
    print("menu:", pg.evaluate("[...document.querySelectorAll('.menu .mi')].map(x=>x.textContent)"))
    pg.screenshot(path=OUT + "menu.png")
    pg.click("text=Копировать изображение")
    pg.wait_for_timeout(800)
    print("clipboard has image:", pg.evaluate("navigator.clipboard.read().then(i=>i.map(x=>x.types.join()))"))
    print("\n".join(logs) or "no console errors")
    b.close()
