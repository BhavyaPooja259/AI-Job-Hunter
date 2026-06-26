"""
Sprint 10 — Lever scraper demo and unit tests.

Tests the LeverScraper against two confirmed live Lever boards using
both the JSON API path and the HTML fallback path.

Run from the project root:
    python -m tests.test_lever_scraper

Test targets
------------
Employ Inc.  (jobs.lever.co/employ)
    Lever's parent company (they acquired Lever in 2022).  Their board is
    the most reliably active Lever board we can test against.  Confirmed
    5 jobs via both the JSON API and the HTML board as of June 2026.
    Tests the happy path — API available, jobs parsed cleanly.

Lever Inc.   (jobs.lever.co/lever)
    Lever's own board.  Zero open positions as of the test date.
    Tests the empty-board case — API reachable, 0 jobs, no crash.

About Razorpay and Netflix
--------------------------
Sprint 10 originally listed these as example Lever companies, but inspection
(June 2026) revealed they have both migrated away from Lever:

  Razorpay → Greenhouse: https://job-boards.greenhouse.io/razorpaysoftwareprivatelimited
  Netflix  → Custom ATS: https://jobs.netflix.com/jobs  (proprietary, not Lever/Greenhouse)

companies.json has been updated to reflect the correct ATS for each.

Why this matters for scale
--------------------------
ATS migrations happen silently — a company changes their hiring system
without announcing it.  Our scraper discovers the mismatch when it returns
0 jobs or gets a 404.  The correct response is to update companies.json and
add the right scraper, not to assume the scraper is broken.  This is one
reason each scraper's 0-result case should be visible in the nightly report
rather than silently ignored.
"""

import sys, time, logging
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

from scrapers.lever_scraper import LeverScraper, _extract_lever_slug, _company_name_from_url
from browser import BrowserService


# ---------------------------------------------------------------------------
# Unit tests (no browser, no network)
# ---------------------------------------------------------------------------

def test_slug_extraction():
    cases = [
        ("https://jobs.lever.co/employ",                    "employ"),
        ("https://jobs.lever.co/employ/089bf688-e005-4469", "employ"),
        ("https://api.lever.co/v0/postings/employ",         "employ"),
        ("https://razorpay.com/jobs",                       None),
        ("https://jobs.netflix.com/jobs",                   None),
        ("https://jobs.lever.co/lever",                     "lever"),
    ]
    print("--- Unit: _extract_lever_slug ---\n")
    all_pass = True
    for url, expected in cases:
        got = _extract_lever_slug(url)
        ok = got == expected
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}]  {url!r}")
        print(f"         expected={expected!r}  got={got!r}")
    print()
    return all_pass


def test_company_name():
    cases = [
        ("https://jobs.lever.co/employ",    "Employ"),
        ("https://jobs.lever.co/lever",     "Lever"),
        ("https://razorpay.com/jobs",       "Razorpay"),
        ("https://jobs.netflix.com/jobs",   "Netflix"),
    ]
    print("--- Unit: _company_name_from_url ---\n")
    all_pass = True
    for url, expected in cases:
        got = _company_name_from_url(url)
        ok = got == expected
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}]  {url!r} → {got!r}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Live integration targets
# ---------------------------------------------------------------------------

@dataclass
class Target:
    company: str
    url: str
    expect_min_jobs: int = 0
    note: str = ""


LIVE_TARGETS = [
    Target(
        company="Employ Inc.",
        url="https://jobs.lever.co/employ",
        expect_min_jobs=1,
        note="Lever's parent company — should have active postings",
    ),
    Target(
        company="Lever Inc.",
        url="https://jobs.lever.co/lever",
        expect_min_jobs=0,
        note="Lever's own board — historically 0 openings",
    ),
]


# ---------------------------------------------------------------------------
# Live scrape runner
# ---------------------------------------------------------------------------

def run_live_targets() -> list[tuple[Target, list, float, str]]:
    """
    Run the scraper against each target.

    Returns list of (target, jobs, elapsed_s, path_used) tuples.
    `path_used` is 'api' or 'html' based on log analysis.
    """
    results = []

    # The JSON API path doesn't need a browser, but the scraper interface
    # always receives one.  We start a single headless browser and reuse it
    # across targets so that HTML fallback is available if needed.
    with BrowserService(headless=True) as browser:
        scraper = LeverScraper(browser)

        for target in LIVE_TARGETS:
            print(f"\n  Scraping {target.company!r} → {target.url}")
            t0 = time.monotonic()
            jobs = scraper.scrape(target.url)
            elapsed = time.monotonic() - t0

            path = scraper.last_path   # "api" or "html" — set by the scraper
            results.append((target, jobs, elapsed, path))

            ok = len(jobs) >= target.expect_min_jobs
            print(
                f"  {'✓' if ok else '✗'}  jobs={len(jobs):3d}  "
                f"path={path}  elapsed={elapsed:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[tuple]) -> None:
    print(f"\n{'='*72}")
    print(f"  Lever Scraper — Live Results")
    print(f"{'='*72}")
    print(f"{'Company':<18} {'Jobs Found':>10} {'Path':>6} {'Time':>6}  {'Status'}")
    print(f"{'-'*72}")

    all_pass = True
    for target, jobs, elapsed, path in results:
        ok = len(jobs) >= target.expect_min_jobs
        if not ok:
            all_pass = False
        print(
            f"{target.company:<18} {len(jobs):>10}   {path:>6} {elapsed:>5.1f}s  "
            f"[{'PASS' if ok else 'FAIL'}]  {target.note}"
        )

    print(f"{'='*72}")
    print(f"\n  {'All live tests passed.' if all_pass else 'Some live tests FAILED.'}")
    return all_pass


def print_job_detail(target: Target, jobs: list) -> None:
    if not jobs:
        print(f"\n  (No jobs to display for {target.company})")
        return

    print(f"\n  --- All jobs from {target.company} ({len(jobs)} total) ---")
    print(f"  {'Title':<55} {'Location'}")
    print(f"  {'-'*80}")
    for job in jobs:
        title = (job.title[:53] + "..") if len(job.title) > 55 else job.title
        loc = job.location or "—"
        print(f"  {title:<55} {loc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Sprint 10 — Lever Scraper Demo ===\n")

    # Unit tests
    slug_ok = test_slug_extraction()
    name_ok = test_company_name()
    unit_pass = slug_ok and name_ok
    print(f"Unit tests: {'all passed' if unit_pass else 'FAILURES detected'}\n")

    # Live scrape
    print("--- Live Integration ---")
    results = run_live_targets()
    live_pass = print_report(results)

    # Print job details for each target
    for target, jobs, _, _ in results:
        print_job_detail(target, jobs)

    # Summary
    print(f"\n=== Done — {'All tests passed.' if unit_pass and live_pass else 'Some tests FAILED.'} ===\n")


if __name__ == "__main__":
    main()
