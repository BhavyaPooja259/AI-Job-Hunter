"""
Browser Service — manual smoke test

Verifies that the browser can start, open a URL, take a screenshot, and shut
down cleanly. This is not a unit test — run it manually to confirm Playwright
is installed and working correctly.

Usage (from the project root):
    python -m tests.test_browser
"""

import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser import BrowserService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

TARGET_URL = "https://google.com"
SCREENSHOT_PATH = Path("storage/screenshots/google_smoke_test.png")
WAIT_SECONDS = 3


def main() -> None:
    print("\n--- Browser Service Smoke Test ---\n")

    service = BrowserService(headless=False)

    try:
        # Step 1: Start
        print("[1/4] Starting browser...")
        service.start()

        # Step 2: Open URL
        print(f"[2/4] Opening {TARGET_URL}...")
        service.open(TARGET_URL)

        # Step 3: Wait
        print(f"[3/4] Waiting {WAIT_SECONDS} seconds...")
        time.sleep(WAIT_SECONDS)

        # Step 4: Screenshot
        print(f"[4/4] Taking screenshot → {SCREENSHOT_PATH}")
        saved = service.screenshot(SCREENSHOT_PATH)
        print(f"\n✓ Screenshot saved: {saved}")

    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        sys.exit(1)

    finally:
        service.close()

    print("\n--- Smoke test passed ---\n")


if __name__ == "__main__":
    main()
