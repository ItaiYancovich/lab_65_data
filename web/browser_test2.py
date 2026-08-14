#!/usr/bin/env python3
"""Keyboard activation + minimax/rollout agents, in real Chromium."""
import sys
import time
from playwright.sync_api import sync_playwright

PATH = "file:///home/user/lab_65_data/web/hex_lab.html"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

errors = []

def on_pageerror(exc):
    errors.append(f"PAGE ERROR: {exc}")

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1300, "height": 900})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", on_pageerror)
    page.goto(PATH)
    page.wait_for_selector("#board-svg polygon", timeout=15000)

    # Keyboard activation: focus a cell directly and press Enter.
    page.select_option("#seat-black-kind", "human")
    page.select_option("#seat-white-kind", "human")
    page.click("#new-game")
    time.sleep(0.2)
    page.eval_on_selector('.hexcell[data-cell="60"]', "el => el.focus()")
    page.keyboard.press("Enter")
    time.sleep(0.2)
    stones = page.eval_on_selector_all("#board-svg .hex-fill.black, #board-svg .hex-fill.white", "els => els.length")
    print("stones after keyboard Enter on cell 60:", stones)
    assert stones == 1, f"keyboard activation failed, stones={stones}"
    focused_label = page.eval_on_selector('.hexcell[data-cell="60"]', "el => el.getAttribute('aria-label')")
    print("aria-label after placing stone:", focused_label)
    assert "black" in focused_label, f"aria-label not updated: {focused_label}"

    # Space key too.
    page.eval_on_selector('.hexcell[data-cell="72"]', "el => el.focus()")
    page.keyboard.press(" ")
    time.sleep(0.2)
    stones2 = page.eval_on_selector_all("#board-svg .hex-fill.black, #board-svg .hex-fill.white", "els => els.length")
    print("stones after keyboard Space on cell 72:", stones2)
    assert stones2 == 2

    # Minimax vs Rollout on a fresh game, watch a handful of plies complete.
    page.select_option("#seat-black-kind", "minimax")
    page.select_option("#seat-white-kind", "rollout")
    page.click("#new-game")
    print("watching Brute-force alpha-beta vs Classic MCTS rollouts...")
    page.wait_for_function(
        "document.querySelectorAll('#board-svg .hex-fill.black, #board-svg .hex-fill.white').length >= 6",
        timeout=60000,
    )
    n_stones = page.eval_on_selector_all("#board-svg .hex-fill.black, #board-svg .hex-fill.white", "els => els.length")
    print(f"minimax-vs-rollout reached {n_stones} stones without error")
    status = page.inner_text("#status-text")
    print("status:", status)

    browser.close()

if errors:
    print("ERRORS:", errors)
    sys.exit(1)
print("\nALL KEYBOARD + MINIMAX/ROLLOUT CHECKS PASSED")
