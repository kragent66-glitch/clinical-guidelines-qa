#!/usr/bin/env python3
"""Capture desktop (laptop 1440x900 @2x) screenshots of the Clinical Guidelines Q&A app."""
import os, time
from playwright.sync_api import sync_playwright

OUT = "/home/ubuntu/screenshots/clinical-qa"
os.makedirs(OUT, exist_ok=True)
URL = "http://127.0.0.1:8501"
W, H = 1440, 900  # laptop 16:10


def shot(page, name, note=""):
    path = os.path.join(OUT, name)
    page.screenshot(path=path, full_page=False)
    size = page.evaluate("() => [window.innerWidth, window.innerHeight]")
    print(f"  saved {name}  viewport={size}  {note}", flush=True)
    return path


def wait_idle(page, seconds=2.0):
    """Wait for Streamlit's run indicator to clear."""
    time.sleep(seconds)
    for _ in range(40):
        busy = page.evaluate(
            "() => !!document.querySelector('[data-testid=\"stStatusWidget\"]')"
        )
        if not busy:
            return
        time.sleep(0.5)


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(
        viewport={"width": W, "height": H},
        device_scale_factor=2,
        locale="en-US",
        user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    )
    page = ctx.new_page()

    print("1) loading app ...", flush=True)
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    # Streamlit hydrates via websocket; wait for the title + input
    page.wait_for_selector("text=Clinical Guidelines Q&A Assistant", timeout=60000)
    page.wait_for_selector("input[type='text']", timeout=60000)
    wait_idle(page, 3)

    print("2) JOURNEY A: landing / empty state", flush=True)
    shot(page, "01_landing.png", "empty state + sidebar")

    q1 = "Is metformin safe in patients with low eGFR?"
    print(f"3) JOURNEY B: typing query -> {q1!r}", flush=True)
    box = page.locator("input[type='text']").first
    box.click()
    box.fill(q1)
    time.sleep(1.2)
    shot(page, "02_query_typed.png", "query typed, pre-submit")

    print("4) JOURNEY B: submitting ...", flush=True)
    page.locator("button:has-text('Get Answer')").first.click()
    # wait for the answer block to render
    page.wait_for_selector("text=Clinical Guideline Answer", timeout=60000)
    seen = False
    for _ in range(120):
        body = page.evaluate("() => document.body.innerText")
        if "Citations" in body and "metformin" in body.lower():
            seen = True
            break
        time.sleep(0.5)
    time.sleep(2.5)
    wait_idle(page, 1)
    shot(page, "03_answer_metformin.png", f"answer rendered (citations={seen})")

    print("5) JOURNEY C: second query (multi-turn)", flush=True)
    q2 = "What is the first-line therapy for stage 2 hypertension?"
    box = page.locator("input[type='text']").first
    box.click()
    box.fill(q2)
    page.locator("button:has-text('Get Answer')").first.click()
    for _ in range(120):
        body = page.evaluate("() => document.body.innerText")
        if "hypertension" in body.lower() and "130/80" in body:
            break
        time.sleep(0.5)
    time.sleep(2.5)
    wait_idle(page, 1)
    shot(page, "04_answer_hypertension.png", "second query answered")

    print("6) JOURNEY D: sidebar settings / web-search toggle", flush=True)
    try:
        # open the settings sidebar if collapsed
        page.locator("[data-testid='stSidebarCollapsedControl']").first.click(timeout=3000)
        time.sleep(1)
    except Exception:
        pass
    time.sleep(1)
    shot(page, "05_sidebar_settings.png", "sidebar settings + web search toggle")

    print("\nFINAL URL:", page.url, flush=True)
    browser.close()

print("DONE ->", OUT)
