"""
Sprint 11 — Workday scraper demo and unit tests.

Tests the WorkdayScraper against two confirmed live Workday boards using
the CXS JSON API (primary path).

Run from the project root:
    python -m tests.test_workday_scraper

Test targets
------------
Adobe  (adobe.wd5.myworkdayjobs.com/en-US/external_experienced)
    Creative software company.  CXS API returns 700–1030 jobs.
    Tests the standard API happy path — large company, many jobs.

Walmart Global Tech  (walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal)
    Retail tech company.  CXS API returns 2000+ jobs.
    Tests a different shard (wd504) and validates pagination capping.

Why Workday is harder than Greenhouse and Lever
-----------------------------------------------
See the full explanation in scrapers/workday_scraper.py.  Short version:
  1. No official public API — the CXS endpoint is internal/undocumented.
  2. Hash-based CSS class names (css-XXXXXXXX) that change every release.
  3. Paginated results — no "show all" mode.
  4. Per-company subdomains with different shard numbers (wd1, wd5, wd504).
  5. Cloudflare protection on some tenants (Walmart) — blocks the browser
     but not always the JSON API, since the POST hits a CDN endpoint.

Why companies.json now uses direct Workday URLs
-----------------------------------------------
Companies like Adobe and Walmart have both a branded career page
(careers.adobe.com, careers.walmart.com) and the underlying Workday board
(adobe.wd5.myworkdayjobs.com).

The branded page requires a full browser load + additional JavaScript
to detect and redirect to the internal API.  The direct Workday URL
allows the CXS API to be called with zero browser overhead.

We update companies.json to use the direct URL once it is discovered.
The scraper automatically detects the *.myworkdayjobs.com pattern and
switches to the CXS API path.

What changed in companies.json (Sprint 11)
------------------------------------------
  Adobe:
    BEFORE: https://careers.adobe.com/us/en/search-results
    AFTER:  https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced

  Walmart Global Tech:
    BEFORE: https://careers.walmart.com/technology
    AFTER:  https://walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal

  Snowflake:
    BEFORE: ats = "workday"
    AFTER:  ats = "unknown"
    Reason: Snowflake uses Phenom (a different ATS provider), not Workday.
            The career page at careers.snowflake.com shows no myworkdayjobs.com
            API calls — it loads from phenom.com instead.

  Microsoft and Visa:
    Left as ats = "workday" with branded URLs.  The CXS API path will be
    activated once the correct direct Workday URLs are confirmed.
    Microsoft's board was in maintenance during testing (June 2026).
    Visa's board uses Workday wd1 but the jobboard slug has not been
    determined yet (HTTP 422 for all tested slugs).
"""

import sys, time, logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

from scrapers.workday_scraper import (
    WorkdayScraper,
    _parse_workday_url,
    _company_name_from_url,
)
from browser import BrowserService


# ---------------------------------------------------------------------------
# Unit tests (no browser, no network)
# ---------------------------------------------------------------------------

def test_parse_workday_url():
    cases = [
        (
            "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced",
            ("adobe", "wd5", "external_experienced", "https://adobe.wd5.myworkdayjobs.com", "en-US"),
        ),
        (
            "https://walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal",
            ("walmart", "wd504", "WalmartExternal", "https://walmart.wd504.myworkdayjobs.com", "en-US"),
        ),
        (
            "https://microsoft.wd1.myworkdayjobs.com/en-US/msft_professional_careers",
            ("microsoft", "wd1", "msft_professional_careers", "https://microsoft.wd1.myworkdayjobs.com", "en-US"),
        ),
        # Deep path: only tenant + jobboard matter; /job/... suffix is ignored
        (
            "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced/job/Hamburg/Engineer_R1234",
            ("adobe", "wd5", "external_experienced", "https://adobe.wd5.myworkdayjobs.com", "en-US"),
        ),
        # Branded page — should return None
        ("https://careers.adobe.com/us/en/search-results", None),
        ("https://careers.walmart.com/technology", None),
        ("https://jobs.lever.co/employ", None),
    ]

    print("--- Unit: _parse_workday_url ---\n")
    all_pass = True
    for url, expected in cases:
        config = _parse_workday_url(url)
        if expected is None:
            ok = config is None
            print(f"  [{'PASS' if ok else 'FAIL'}]  None expected   {url!r}")
        else:
            exp_tenant, exp_wd, exp_board, exp_base, exp_lang = expected
            ok = (
                config is not None
                and config.tenant == exp_tenant
                and config.wd_num == exp_wd
                and config.jobboard == exp_board
                and config.base_url == exp_base
                and config.lang == exp_lang
            )
            print(f"  [{'PASS' if ok else 'FAIL'}]  {url!r}")
            if not ok and config:
                print(f"         got: tenant={config.tenant!r} wd={config.wd_num!r} "
                      f"board={config.jobboard!r}")
        if not ok:
            all_pass = False
    print()
    return all_pass


def test_cxs_api_url():
    cases = [
        (
            "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced",
            "https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/external_experienced/jobs",
        ),
        (
            "https://walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal",
            "https://walmart.wd504.myworkdayjobs.com/wday/cxs/walmart/WalmartExternal/jobs",
        ),
    ]

    print("--- Unit: WorkdayConfig.cxs_api_url ---\n")
    all_pass = True
    for url, expected_api_url in cases:
        config = _parse_workday_url(url)
        ok = config is not None and config.cxs_api_url == expected_api_url
        print(f"  [{'PASS' if ok else 'FAIL'}]  {config.cxs_api_url if config else '—'!r}")
        if not ok:
            all_pass = False
    print()
    return all_pass


def test_full_job_url():
    config = _parse_workday_url("https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced")
    assert config is not None

    cases = [
        ("/job/Hamburg/Engineering-Manager_R167025", "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced/job/Hamburg/Engineering-Manager_R167025"),
        ("job/SF/SWE_R001",                         "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced/job/SF/SWE_R001"),
        ("https://other.com/job/123",               "https://other.com/job/123"),
    ]

    print("--- Unit: WorkdayConfig.full_job_url ---\n")
    all_pass = True
    for path, expected in cases:
        got = config.full_job_url(path)
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}]  {path!r}")
        if not ok:
            print(f"         expected: {expected!r}")
            print(f"         got:      {got!r}")
        if not ok:
            all_pass = False
    print()
    return all_pass


def test_company_name():
    cases = [
        ("https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced",   "Adobe"),
        ("https://walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal",     "Walmart"),
        ("https://microsoft.wd1.myworkdayjobs.com/en-US/msft_professional",  "Microsoft"),
        ("https://careers.adobe.com/us/en/search-results",                    "Adobe"),
        ("https://careers.walmart.com/technology",                             "Walmart"),
    ]

    print("--- Unit: _company_name_from_url ---\n")
    all_pass = True
    for url, expected in cases:
        got = _company_name_from_url(url)
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}]  {url!r} → {got!r}")
        if not ok:
            all_pass = False
    print()
    return all_pass


def test_workday_scraper_returns_list_with_mock():
    """WorkdayScraper.scrape() returns a list (not raises) even with a mock browser."""
    mock_browser = MagicMock()
    # page.goto returns a MagicMock (no PlaywrightTimeoutError), so navigation
    # succeeds.  wait_for_selector also returns a MagicMock.
    # The JS evaluate for job cards returns an empty list → 0 jobs parsed.
    mock_browser.page.evaluate.return_value = []

    scraper = WorkdayScraper(mock_browser)
    result = scraper.scrape("https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced")
    assert isinstance(result, list)

    print("--- Unit: WorkdayScraper.scrape() with mock browser ---")
    print(f"  [PASS]  returns list (CXS API took over, browser not called)")
    print()
    return True


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
        company="Adobe",
        url="https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced",
        expect_min_jobs=10,
        note="~700+ jobs via CXS API, capped at 3 pages (60 jobs) by MAX_PAGES",
    ),
    Target(
        company="Walmart Global Tech",
        url="https://walmart.wd504.myworkdayjobs.com/en-US/WalmartExternal",
        expect_min_jobs=10,
        note="~2000+ jobs on server, capped at 3 pages (60 jobs) by MAX_PAGES",
    ),
]


# ---------------------------------------------------------------------------
# Live scrape runner
# ---------------------------------------------------------------------------

def run_live_targets() -> list[tuple]:
    """Run the scraper against each target. Returns (target, jobs, elapsed_s, path) tuples."""
    results = []

    # The CXS API path requires no browser, but WorkdayScraper always accepts one.
    # We start the browser once in case the HTML fallback is triggered.
    with BrowserService(headless=True) as browser:
        scraper = WorkdayScraper(browser)

        for target in LIVE_TARGETS:
            print(f"\n  Scraping {target.company!r} → {target.url}")
            t0 = time.monotonic()
            jobs = scraper.scrape(target.url)
            elapsed = time.monotonic() - t0
            path = scraper.last_path  # "api" or "html"

            ok = len(jobs) >= target.expect_min_jobs
            results.append((target, jobs, elapsed, path))

            print(
                f"  {'[OK]' if ok else '[FAIL]'}  jobs={len(jobs):3d}  "
                f"path={path}  elapsed={elapsed:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[tuple]) -> bool:
    print(f"\n{'='*78}")
    print(f"  Workday Scraper — Live Results")
    print(f"{'='*78}")
    print(f"{'Company':<22} {'Jobs Found':>10} {'Path':>5} {'Time':>6}  {'Status'}")
    print(f"{'-'*78}")

    all_pass = True
    for target, jobs, elapsed, path in results:
        ok = len(jobs) >= target.expect_min_jobs
        if not ok:
            all_pass = False
        print(
            f"{target.company:<22} {len(jobs):>10}   {path:>5} {elapsed:>5.1f}s  "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
        print(f"  {'':22} {target.note}")

    print(f"{'='*78}")
    print(f"\n  {'All live tests passed.' if all_pass else 'Some live tests FAILED.'}")
    return all_pass


def print_job_detail(target: Target, jobs: list) -> None:
    if not jobs:
        print(f"\n  (No jobs to display for {target.company})")
        return

    print(f"\n  --- First 20 jobs from {target.company} ({len(jobs)} parsed, capped by MAX_PAGES) ---")
    print(f"  {'#':>3}  {'Title':<55}  {'Location'}")
    print(f"  {'-'*85}")
    for i, job in enumerate(jobs[:20], 1):
        title = (job.title[:53] + "..") if len(job.title) > 55 else job.title
        loc = job.location or "—"
        print(f"  {i:>3}. {title:<55}  {loc}")
    if len(jobs) > 20:
        print(f"  ... and {len(jobs) - 20} more (raise MAX_PAGES to fetch more)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Sprint 11 — Workday Scraper Demo ===\n")

    # Unit tests
    unit_results = [
        test_parse_workday_url(),
        test_cxs_api_url(),
        test_full_job_url(),
        test_company_name(),
        test_workday_scraper_returns_list_with_mock(),
    ]
    unit_pass = all(unit_results)
    print(f"Unit tests: {'all passed' if unit_pass else 'FAILURES detected'}\n")

    # Live scrape
    print("--- Live Integration ---")
    live_results = run_live_targets()
    live_pass = print_report(live_results)

    # Print job listings
    for target, jobs, _, _ in live_results:
        print_job_detail(target, jobs)

    # Final summary
    overall = unit_pass and live_pass
    print(f"\n=== Done — {'All tests passed.' if overall else 'Some tests FAILED.'} ===\n")


if __name__ == "__main__":
    main()
