"""
Greenhouse scraper — first real job extraction demo.

Opens the official Greenhouse demo board, extracts the first job posting,
and prints its details.

Run from the project root:
    python -m tests.test_greenhouse_first_job
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
    print("\n--- Greenhouse Scraper: First Job Extraction ---\n")

    browser = BrowserService(headless=True)
    scraper = GreenhouseScraper(browser)

    try:
        browser.start()
        jobs = scraper.scrape(DEMO_URL)

        if not jobs:
            print("No jobs found.")
            sys.exit(1)

        job = jobs[0]
        print(f"Company:   {job.company}")
        print(f"Title:     {job.title}")
        print(f"Location:  {job.location or 'Not specified'}")
        print(f"Job URL:   {job.job_url}")
        print(f"Platform:  {job.source_platform.value}")
        print(f"Fingerprint: {job.fingerprint}")

    finally:
        browser.close()

    print("\n--- Done ---\n")


if __name__ == "__main__":
    main()
