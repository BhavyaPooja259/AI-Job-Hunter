"""
Sprint 9 — Real Greenhouse support demo.

Tests the upgraded GreenhouseScraper against three real company career
pages and prints a results table (Company / Jobs Found / Jobs Parsed / Failures).

Run from the project root:
    python -m tests.test_greenhouse_real_sites

What each result demonstrates
------------------------------
Tekion (boards.greenhouse.io/tekion)
    Standard Greenhouse Modern board — Layout B.
    The scraper detects .job-post, extracts title/location from
    p.body--medium / p.body--metadata, and returns all jobs.

Confluent (www.confluent.io/careers)
    Confluent's Greenhouse board is no longer active on greenhouse.io.
    Their /careers page shows department navigation links, not individual
    job listings.  The scraper exhausts all three layout strategies,
    finds nothing, and returns [] without crashing.  This is expected
    failure handled gracefully.

Rippling (ats.rippling.com/rippling/jobs)
    Rippling built their own ATS product (they're an HR/payroll company).
    The career page uses Rippling's proprietary layout — none of the
    Greenhouse layout strategies match.  The scraper again returns []
    gracefully.  The companies.json entry has ats='greenhouse', which
    is incorrect; that's a data quality issue to be fixed in a future
    sprint (a RipplingATS scraper should be added).
"""

import sys, time, logging
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

from playwright.sync_api import sync_playwright
from scrapers.greenhouse_scraper import GreenhouseScraper
from browser import BrowserService


# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

@dataclass
class Target:
    company: str
    url: str
    expects_jobs: bool = True


TARGETS = [
    Target(
        company="Tekion",
        url="https://boards.greenhouse.io/tekion",
        expects_jobs=True,
    ),
    Target(
        company="Confluent",
        url="https://www.confluent.io/careers",
        expects_jobs=False,   # Board no longer active on Greenhouse
    ),
    Target(
        company="Rippling",
        url="https://ats.rippling.com/rippling/jobs",
        expects_jobs=False,   # Rippling uses their own ATS, not Greenhouse
    ),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    company: str
    url: str
    links_found: int = 0
    jobs_parsed: int = 0
    failures: int = 0
    layout_strategy: str = "—"
    elapsed_s: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(target: Target, browser: BrowserService) -> ScrapeResult:
    result = ScrapeResult(company=target.company, url=target.url)
    scraper = GreenhouseScraper(browser)

    t0 = time.monotonic()
    try:
        jobs = scraper.scrape(target.url)
        result.jobs_parsed = len(jobs)
        result.elapsed_s = time.monotonic() - t0

        if jobs:
            # Infer which layout was used from the first job's URL pattern
            first_url = jobs[0].job_url
            if "job-boards.greenhouse.io" in first_url:
                result.layout_strategy = "Modern (.job-post)"
            elif "boards.greenhouse.io" in first_url:
                result.layout_strategy = "Classic (.opening)"
            else:
                result.layout_strategy = "Custom embed"
        else:
            result.layout_strategy = "None (no layout detected)"

        # Count skipped/failure entries by comparing to total page links
        # (a rough proxy — the scraper logs individual skips)
        page = browser.page
        page_links = len(page.query_selector_all("a")) if page else 0
        result.links_found = page_links

    except Exception as exc:
        result.error = str(exc)
        result.elapsed_s = time.monotonic() - t0

    return result


def run_all() -> list[ScrapeResult]:
    results = []

    with BrowserService(headless=True) as browser:
        for target in TARGETS:
            print(f"\n  Scraping {target.company!r} → {target.url}")
            r = run_one(target, browser)
            results.append(r)
            status = "✓" if (r.jobs_parsed > 0) == target.expects_jobs else "✗"
            print(f"  {status}  jobs={r.jobs_parsed}  layout={r.layout_strategy}  "
                  f"elapsed={r.elapsed_s:.1f}s"
                  + (f"  error={r.error}" if r.error else ""))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[ScrapeResult]) -> None:
    print(f"\n{'='*75}")
    print(f"  Greenhouse Real-Sites Scraper Report")
    print(f"{'='*75}")
    print(f"{'Company':<14} {'Jobs Parsed':>12} {'Layout Strategy':<28} {'Time':>6}  Status")
    print(f"{'-'*75}")

    for r in results:
        target = next(t for t in TARGETS if t.company == r.company)
        expected = "jobs" if target.expects_jobs else "no jobs"
        got = r.jobs_parsed > 0
        passed = got == target.expects_jobs
        status = "PASS" if passed else "FAIL"

        print(
            f"{r.company:<14} {r.jobs_parsed:>12}   {r.layout_strategy:<28} {r.elapsed_s:>5.1f}s  [{status}]"
        )
        if r.error:
            print(f"               error: {r.error}")

    print(f"{'='*75}")

    all_pass = all(
        (r.jobs_parsed > 0) == next(t for t in TARGETS if t.company == r.company).expects_jobs
        for r in results
    )
    print(f"\n  {'All tests passed.' if all_pass else 'Some tests failed.'}")

    # Print first 5 parsed jobs from any successful scrape
    print()
    for r in results:
        if r.jobs_parsed > 0:
            print(f"  --- Sample jobs from {r.company} ({r.jobs_parsed} total) ---")
            break


# ---------------------------------------------------------------------------
# Standalone job detail run (just for Tekion to show actual jobs)
# ---------------------------------------------------------------------------

def print_tekion_jobs() -> None:
    tekion = next(t for t in TARGETS if t.company == "Tekion")
    print(f"\n  --- Top 10 jobs from {tekion.company} ---")

    with BrowserService(headless=True) as browser:
        scraper = GreenhouseScraper(browser)
        jobs = scraper.scrape(tekion.url)

    print(f"  {'Title':<55} {'Location'}")
    print(f"  {'-'*75}")
    for job in jobs[:10]:
        title = (job.title[:53] + "..") if len(job.title) > 55 else job.title
        loc   = job.location or "—"
        print(f"  {title:<55} {loc}")
    print(f"\n  Total: {len(jobs)} jobs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Sprint 9 — Greenhouse Real Sites Test ===")
    print(f"  Targets: {[t.company for t in TARGETS]}\n")

    results = run_all()
    print_report(results)
    print_tekion_jobs()

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
