"""
Greenhouse scraper — all jobs extraction demo.

Opens the official Greenhouse demo board, extracts every available job,
and prints each one followed by a total count.

Run from the project root:
    python -m tests.test_greenhouse_all_jobs
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser import BrowserService
from scrapers.greenhouse_scraper import GreenhouseScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

DEMO_URL = "https://boards.greenhouse.io/embed/job_board?for=greenhouse"


def main() -> None:
    print("\n--- Greenhouse Scraper: All Jobs ---\n")

    browser = BrowserService(headless=True)
    scraper = GreenhouseScraper(browser)

    try:
        browser.start()
        jobs = scraper.scrape(DEMO_URL)

        if not jobs:
            print("No jobs found.")
            sys.exit(1)

        for i, job in enumerate(jobs, start=1):
            print(f"[{i:02d}] Company:  {job.company}")
            print(f"      Title:    {job.title}")
            print(f"      Location: {job.location or 'Not specified'}")
            print()

        print(f"Total Jobs Extracted: {len(jobs)}")

    finally:
        browser.close()

    print("\n--- Done ---\n")


if __name__ == "__main__":
    main()
