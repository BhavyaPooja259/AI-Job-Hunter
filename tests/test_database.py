"""
Database layer — demo and duplicate-detection test.

Run from the project root:
    python -m tests.test_database

Run twice to confirm duplicates are ignored on the second pass.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser import BrowserService
from database import JobRepository
from scrapers.greenhouse_scraper import GreenhouseScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

DEMO_URL = "https://boards.greenhouse.io/embed/job_board?for=greenhouse"


def main() -> None:
    print("\n--- Database Demo: Save Greenhouse Jobs ---\n")

    # Step 1: Scrape jobs
    print("[1/3] Scraping jobs from Greenhouse demo board...")
    browser = BrowserService(headless=True)
    scraper = GreenhouseScraper(browser)
    try:
        browser.start()
        jobs = scraper.scrape(DEMO_URL)
    finally:
        browser.close()

    if not jobs:
        print("No jobs scraped. Exiting.")
        sys.exit(1)

    print(f"      Jobs scraped: {len(jobs)}\n")

    # Step 2: Save to database
    print("[2/3] Saving to database...")
    with JobRepository() as repo:
        saved, skipped = repo.save_many(jobs)
        total_in_db = repo.count()

        print(f"\n      Jobs Saved:      {saved}")
        print(f"      Duplicates:      {skipped}")
        print(f"      Database Count:  {total_in_db}\n")

    # Step 3: Run again to confirm duplicates are ignored
    print("[3/3] Running save again to verify duplicate prevention...")
    with JobRepository() as repo:
        saved2, skipped2 = repo.save_many(jobs)
        total_in_db2 = repo.count()

        print(f"\n      Jobs Saved:      {saved2}  ← should be 0")
        print(f"      Duplicates:      {skipped2}  ← should equal first-run count")
        print(f"      Database Count:  {total_in_db2}  ← unchanged\n")

    if saved2 == 0 and skipped2 == len(jobs) and total_in_db2 == total_in_db:
        print("Duplicate prevention: PASS")
    else:
        print("Duplicate prevention: FAIL")
        sys.exit(1)

    print("\n--- Done ---\n")


if __name__ == "__main__":
    main()
