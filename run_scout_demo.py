"""
Demo: Scout / Scraper pipeline.

Shows the full job-discovery pipeline — from reading companies.json to
persisting jobs in jobs.db — with no ranking, tailoring, cover letters,
referrals, or scheduler logic involved.

Pipeline
--------
  1. Load companies.json  — print company count and ATS breakdown.
  2. Select targets       — show which companies will be scraped and which
                            are skipped (no scraper registered for their ATS).
  3. Run ScoutAgent       — opens a headless browser per company, scrapes
                            live job listings, and saves them to the database.
                            Per-company progress appears as INFO log lines.
  4. Verify database      — compare job counts before and after the run.
  5. Sample output        — print the five most recently discovered jobs.

Requirements
------------
  Playwright must be installed (runs a real headless browser):
      playwright install chromium

  This demo makes live network requests.  Each company typically takes
  30–120 seconds.  All 19 companies in companies.json → roughly 10–20 min
  total.  Reduce the active list in data/companies.json if you need a
  shorter run.

Run
---
  python run_scout_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
for _ns in ("agents", "scrapers", "browser", "database", "services"):
    logging.getLogger(_ns).setLevel(logging.INFO)

from config.constants import ATSType
from database.job_repository import JobRepository
from agents.scout_agent import ScoutAgent
from scrapers.scraper_factory import ScraperFactory
from services.company_registry import Company, CompanyRegistry

BANNER = "=" * 60
_SUPPORTED = set(ScraperFactory.supported_platforms())


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _load_registry() -> CompanyRegistry:
    """Load companies.json; exit with a helpful message if the file is missing."""
    try:
        return CompanyRegistry()
    except FileNotFoundError as exc:
        print()
        print("ERROR: companies.json not found.")
        print(str(exc))
        print()
        print("Create data/companies.json with at least one company entry.")
        sys.exit(1)
    except Exception as exc:
        print()
        print(f"ERROR: Could not load companies.json — {exc}")
        sys.exit(1)


def _ats_breakdown(companies: list[Company]) -> dict[str, int]:
    """Count companies per ATS type."""
    counts: dict[str, int] = {}
    for c in companies:
        counts[c.ats.value] = counts.get(c.ats.value, 0) + 1
    return dict(sorted(counts.items()))


def _get_job_count() -> int:
    """Return the current total job count without leaving the DB open."""
    with JobRepository() as repo:
        return repo.count()


def _get_recent_jobs(n: int = 5):
    """Return the N most recently discovered jobs."""
    with JobRepository() as repo:
        return repo.get_all()[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(BANNER)
    print("  Scout / Scraper Demo")
    print(BANNER)

    # ── 1/5 — companies.json ──────────────────────────────────────────────────
    print()
    print("[ 1/5 ]  Loading companies.json")
    print("-" * 40)

    registry = _load_registry()

    all_companies = registry.all()
    active = registry.active()
    breakdown = _ats_breakdown(active)

    print(f"  Total companies  : {len(all_companies)}")
    print(f"  Active           : {len(active)}")
    print(f"  Inactive         : {len(all_companies) - len(active)}")
    print()
    for ats_name, count in breakdown.items():
        marker = "✓" if ats_name in {a.value for a in _SUPPORTED} else "–"
        print(f"  {marker} {ats_name.capitalize():18s}: {count}")

    # ── 2/5 — select targets ──────────────────────────────────────────────────
    print()
    print("[ 2/5 ]  Selecting companies to scrape")
    print("-" * 40)

    scrapeable = sorted(
        [c for c in active if c.ats in _SUPPORTED],
        key=lambda c: (c.priority, c.name),
    )
    no_scraper = sorted(
        [c for c in active if c.ats not in _SUPPORTED],
        key=lambda c: c.name,
    )

    print(f"  Will scrape      : {len(scrapeable)} companies")
    for c in scrapeable:
        login_note = "  [login required — may fail]" if c.login_required else ""
        print(f"    • {c.name:25s}  [{c.ats.value}]  priority={c.priority}{login_note}")

    if no_scraper:
        print()
        print(f"  Skipped (no scraper for ATS) : {len(no_scraper)}")
        for c in no_scraper:
            print(f"    – {c.name:25s}  [{c.ats.value}]")

    if not scrapeable:
        print()
        print("  No companies have a supported scraper registered.")
        print("  Implement scrapers for the ATS types above to continue.")
        sys.exit(0)

    print()
    print("  NOTE: This demo makes live network requests and opens a headless")
    print("  browser per company. Each company takes 30–120 seconds.")
    estimated_min = max(1, len(scrapeable) * 30 // 60)
    estimated_max = len(scrapeable) * 2
    print(f"  Estimated time: {estimated_min}–{estimated_max} minutes "
          f"for {len(scrapeable)} companies.")

    # ── 3/5 — run scraper ─────────────────────────────────────────────────────
    print()
    print("[ 3/5 ]  Running ScoutAgent")
    print("-" * 40)
    print("  Per-company progress appears below as log lines.")
    print("  Companies with unknown ATS are skipped automatically.")
    print()

    before_count = _get_job_count()

    try:
        agent = ScoutAgent(registry=registry)
        result = agent.run()
    except Exception as exc:
        print()
        print(f"ERROR: ScoutAgent raised an unexpected exception: {exc}")
        print("Check that Playwright is installed: playwright install chromium")
        sys.exit(1)

    # ── 4/5 — database verification ───────────────────────────────────────────
    print()
    print("[ 4/5 ]  Database verification")
    print("-" * 40)

    after_count = _get_job_count()

    print(f"  Jobs before this run   : {before_count}")
    print(f"  Jobs discovered        : {result.jobs_scraped}")
    print(f"  New jobs inserted      : {result.jobs_saved}")
    print(f"  Duplicates skipped     : {result.duplicates}")
    print(f"  Total jobs in jobs.db  : {after_count}")

    if result.failures:
        print()
        print(f"  Failed companies ({result.failure_count}):")
        for company_name, reason in result.failures:
            short_reason = reason.split("\n")[0][:100]
            print(f"    x {company_name}: {short_reason}")

    # ── 5/5 — sample output ───────────────────────────────────────────────────
    print()
    print("[ 5/5 ]  Sample output  (5 most recent jobs)")
    print("-" * 40)

    recent = _get_recent_jobs(5)

    if not recent:
        print("  No jobs in the database yet.")
        print("  If all companies failed, check network connectivity and")
        print("  ensure playwright is installed: playwright install chromium")
    else:
        for i, job in enumerate(recent, 1):
            print(f"  [{i}] {job.company} — {job.title}")
            print(f"       Location : {job.location or 'Not specified'}")
            print(f"       URL      : {job.job_url}")
            print()

    # ── final summary ──────────────────────────────────────────────────────────
    print()
    print(BANNER)
    print("  RESULT")
    print(BANNER)
    print(f"  Companies processed  : {result.companies_checked}")
    print(f"  Companies succeeded  : {result.success_count}")
    print(f"  Companies failed     : {result.failure_count}")
    print(f"  Jobs discovered      : {result.jobs_scraped}")
    print(f"  New jobs inserted    : {result.jobs_saved}")
    print(f"  Duplicates skipped   : {result.duplicates}")
    print()
    print(f"  Database total jobs  : {after_count}")

    if result.failures:
        print()
        print("  Failed companies:")
        for company_name, reason in result.failures:
            short_reason = reason.split("\n")[0][:80]
            print(f"    x {company_name}: {short_reason}")

    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
