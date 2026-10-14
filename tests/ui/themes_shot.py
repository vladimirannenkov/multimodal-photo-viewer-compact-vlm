from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1300, "height": 800})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8765/themes"); pg.wait_for_timeout(800)
    pg.screenshot(path="C:/ML/Project-Image/tests/ui/themes_gallery.png")
    for t in ("ocean", "midnight"):
        pg.goto(f"http://localhost:8765/?theme={t}"); pg.wait_for_selector(".cell"); pg.wait_for_timeout(600)
        pg.click(".cell >> nth=3"); pg.wait_for_timeout(500)
        pg.screenshot(path=f"C:/ML/Project-Image/tests/ui/theme_{t}.png")
    pg.click("#btnTools"); pg.wait_for_timeout(400)
    print("dialog text tail:", pg.evaluate("document.querySelector('dialog .dlg-b').parentElement.textContent.slice(-40)"))
    print(errs or "no errors")
    b.close()
