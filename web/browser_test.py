#!/usr/bin/env python3
"""Drive hex_lab.html in real Chromium: console errors, click-to-play, agent-vs-agent."""
import sys
import time
from playwright.sync_api import sync_playwright

PATH = "file:///home/user/lab_65_data/web/hex_lab.html"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

errors = []
logs = []

def on_console(msg):
    logs.append(f"[{msg.type}] {msg.text}")
    if msg.type == "error":
        errors.append(msg.text)

def on_pageerror(exc):
    errors.append(f"PAGE ERROR: {exc}")

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1300, "height": 900})
    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.goto(PATH)
    time.sleep(1)
    print("--- console so far ---")
    for l in logs:
        print(l)
    if errors:
        print("ERRORS:", errors)
        browser.close()
        sys.exit(1)
    page.wait_for_selector("#board-svg polygon", timeout=15000)
    print("page loaded, board rendered")

    n_cells = page.eval_on_selector_all("#board-svg .hexcell", "els => els.length")
    print(f"cells rendered: {n_cells}")
    assert n_cells == 121, f"expected 121 cells, got {n_cells}"

    # Default: Black=Human, White=AlphaZero. Play a human move and let AZ reply.
    status0 = page.inner_text("#status-text")
    print("initial status:", status0)

    # Click center cell (f6 = row5,col5 -> index 5*11+5=60)
    page.eval_on_selector('.hexcell[data-cell="60"]', "el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    time.sleep(0.3)
    board_after_click = page.eval_on_selector_all("#board-svg .hex-fill.black, #board-svg .hex-fill.white", "els => els.length")
    print("stones after human click:", board_after_click)
    assert board_after_click >= 1, "human move did not register"

    print("waiting for AlphaZero reply (may take several seconds)...")
    page.wait_for_function(
        "document.querySelectorAll('#board-svg .hex-fill.black, #board-svg .hex-fill.white').length >= 2",
        timeout=30000,
    )
    stones = page.eval_on_selector_all("#board-svg .hex-fill.black, #board-svg .hex-fill.white", "els => els.length")
    print(f"stones after AI reply: {stones}")
    status1 = page.inner_text("#status-text")
    print("status after AI reply:", status1)
    movelog_text = page.inner_text("#movelog")
    print("move log:", movelog_text.replace("\n", " "))

    eval_visible = page.eval_on_selector("#eval-row", "el => el.style.display")
    print("eval bar display:", eval_visible)

    # New game, then set up Random vs Random and let it play through quickly.
    page.click("#new-game")
    time.sleep(0.2)
    page.select_option("#seat-black-kind", "random")
    page.select_option("#seat-white-kind", "random")
    page.click("#new-game")
    print("watching Random vs Random to completion...")
    page.wait_for_function("document.getElementById('banner').classList.contains('show')", timeout=15000)
    banner_text = page.inner_text("#banner-text")
    print("banner:", banner_text)
    final_status = page.inner_text("#status-text")
    print("final status:", final_status)

    # Undo semantics check: Human vs Rule-based, human moves, rule-based replies,
    # Undo should return to a fresh human turn with 0 moves recorded.
    page.click("#new-game")
    time.sleep(0.2)
    page.select_option("#seat-black-kind", "human")
    page.select_option("#seat-white-kind", "rule")
    page.click("#new-game")
    time.sleep(0.2)
    page.eval_on_selector('.hexcell[data-cell="60"]', "el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
    page.wait_for_function(
        "document.querySelectorAll('#board-svg .hex-fill.black, #board-svg .hex-fill.white').length >= 2",
        timeout=10000,
    )
    hist_before = page.inner_text("#movelog")
    page.click("#undo")
    time.sleep(0.2)
    hist_after = page.inner_text("#movelog")
    print("movelog before undo:", hist_before.replace("\n", " "))
    print("movelog after undo:", hist_after.replace("\n", " "))
    assert "No moves yet" in hist_after, f"undo did not return to empty history: {hist_after!r}"

    browser.close()

print(f"\n--- console messages: {len(logs)} total, {len(errors)} errors ---")
for e in errors:
    print("ERROR:", e)

if errors:
    sys.exit(1)
print("\nALL BROWSER CHECKS PASSED, NO CONSOLE ERRORS")
