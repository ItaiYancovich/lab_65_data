#!/usr/bin/env python3
import time
from playwright.sync_api import sync_playwright

PATH = "file:///home/user/lab_65_data/web/hex_lab.html"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME)

    for scheme in ("dark", "light"):
        page = browser.new_page(viewport={"width": 1360, "height": 980}, color_scheme=scheme)
        page.goto(PATH)
        page.wait_for_selector("#board-svg polygon")
        time.sleep(0.3)
        # place a couple of stones so the board isn't empty in the screenshot
        page.eval_on_selector('.hexcell[data-cell="60"]', "el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        page.wait_for_function(
            "document.querySelectorAll('#board-svg .hex-fill.black, #board-svg .hex-fill.white').length >= 2",
            timeout=20000,
        )
        time.sleep(0.3)
        page.screenshot(path=f"/home/user/lab_65_data/web/shot_{scheme}.png", full_page=True)
        print(f"saved shot_{scheme}.png")
        page.close()

    browser.close()
