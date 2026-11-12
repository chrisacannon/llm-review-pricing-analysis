"""Visit the public Streamlit app and wake it if it has gone to sleep.

Run on a schedule by .github/workflows/keep-awake.yml so visitors
don't land on the "app has gone to sleep" screen.
"""
import re
from playwright.sync_api import sync_playwright

URL = "https://llm-review-pricing-analysis.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(10_000)  # let the sleep screen or the app render

    wake = page.get_by_role("button", name=re.compile(r"back up|wake", re.I))
    if wake.count():
        wake.first.click()
        print("App was asleep: clicked the wake button.")
        page.wait_for_timeout(90_000)  # give it time to boot
    else:
        print("App is awake: no wake button found.")
    browser.close()
