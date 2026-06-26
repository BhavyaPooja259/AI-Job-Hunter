"""
Sprint 11.5 — Job description extraction demo and tests.

Goal: Demonstrate that all three ATS scrapers can populate the `description`,
`requirements`, and `department` fields on Job objects, and that the matcher
benefits from searching the description text.

Run from the project root:
    python -m tests.test_description_extraction

Why descriptions improve matching quality
-----------------------------------------
The job title alone rarely encodes technology requirements.  "Software Engineer
II" looks identical whether the company uses Java + Spring Boot or Go + gRPC.
Descriptions contain explicit language: "5+ years of Java experience",
"familiarity with Kafka is a plus", "strong SQL skills required".

With title-only matching:
  - Score is driven almost entirely by title keywords and seniority level
  - Two SDE2 jobs at different companies get the same base score regardless of
    tech stack alignment
  - Skills dimension defaults to 0 for most postings (title says nothing about SQL)

With description matching:
  - Skill scorer finds "java", "spring boot", "kafka", "sql" etc. directly in text
  - A Java-focused job correctly scores +25 on skills
  - A React-focused job correctly scores +0 on skills
  - The matcher can actually differentiate between roles, not just filter by title

Concrete improvement: a Backend Engineer role at a React-focused company scores
40 (role) + 12 (seniority, no explicit level) + 0 (no Java/backend mentions) = 52.
The same role at a Java company scores 40 + 12 + 25 = 77.
Without descriptions, both score 52 and the distinction is invisible.

Performance trade-offs of fetching detail pages
-----------------------------------------------

┌─────────────────────┬──────────────────────────────┬───────────────────────────────┐
│ ATS                 │ Description strategy         │ Cost                          │
├─────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Greenhouse          │ Job Board API (?content=true)│ 0 extra requests per job       │
│                     │ — descriptions in list call  │ +1 HTTP call total (no browser)│
├─────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Lever               │ Public JSON API              │ 0 extra requests per job       │
│                     │ — descriptionPlain in listing│ Fields already in list payload │
├─────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Workday             │ Browser detail page per job  │ 5–15s per job (React hydration)│
│                     │ — no description in CXS API  │ 60 jobs ≈ 5–15 minutes         │
└─────────────────────┴──────────────────────────────┴───────────────────────────────┘

How ATS APIs reduce scraping overhead
--------------------------------------
Lever and Greenhouse both expose structured job data including descriptions
in their list endpoints.  This means:
  - For a 100-job company board: 1 API call, all descriptions included
  - Compare: 100 browser navigations at 5s each = 500 seconds (8+ minutes)

The Workday CXS API omits descriptions — each posting only has title,
location, and externalPath.  Getting descriptions requires loading the
React-rendered detail page via browser.

Mitigation strategy:
  1. Score all jobs by title first (CXS API list, ~7s for 60 jobs)
  2. Fetch descriptions only for top-N jobs (max_descriptions=5 by default)
  3. Re-score top-N with full text (descriptions now populated)
  4. Present the improved scores to the user

This keeps total time proportional to the number of INTERESTING jobs,
not the total number of jobs at the company.

Test targets
------------
Greenhouse — Tekion (boards.greenhouse.io/tekion)
    Uses the Greenhouse Job Board API with content=true.
    50 jobs returned; first job includes full HTML-stripped description.

Lever — Employ Inc. (jobs.lever.co/employ)
    Uses the public Lever JSON API.
    descriptionPlain is returned in the same response as the title/location.
    Lists are parsed for the "requirements" section.

Workday — Adobe (adobe.wd5.myworkdayjobs.com)
    CXS API returns 60 jobs (3 pages) WITHOUT descriptions.
    max_descriptions=1 triggers one browser navigation to the first job's
    detail page and extracts [data-automation-id="jobPostingDescription"].
"""

import sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

from browser import BrowserService
from scrapers.greenhouse_scraper import GreenhouseScraper
from scrapers.lever_scraper import LeverScraper
from scrapers.workday_scraper import WorkdayScraper
from matching.matcher import JobMatcher
from matching.profile import DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Greenhouse: Job Board API with descriptions
# ---------------------------------------------------------------------------

def test_greenhouse_description():
    """Scrape Tekion via the Greenhouse Job Board API and verify descriptions."""
    print("\n=== Greenhouse (Tekion) ===")
    print("URL:      https://job-boards.greenhouse.io/tekion")
    print("Path:     Greenhouse Job Board API with content=true")
    print()

    with BrowserService(headless=True) as browser:
        scraper = GreenhouseScraper(browser)
        t0 = time.monotonic()
        jobs = scraper.scrape("https://job-boards.greenhouse.io/tekion")
        elapsed = time.monotonic() - t0

    print(f"Elapsed:  {elapsed:.1f}s")
    print(f"Jobs:     {len(jobs)}")

    assert len(jobs) > 0, "Expected at least 1 job from Tekion"

    with_desc = [j for j in jobs if j.description]
    without_desc = [j for j in jobs if not j.description]
    with_dept = [j for j in jobs if j.department]

    print(f"With description:  {len(with_desc)}/{len(jobs)}")
    print(f"With department:   {len(with_dept)}/{len(jobs)}")
    print()

    # Show the first job with a description
    sample = next((j for j in jobs if j.description), None)
    if sample:
        print(f"Sample job:        {sample.title!r}")
        print(f"  Location:        {sample.location!r}")
        print(f"  Department:      {sample.department!r}")
        print(f"  Description len: {len(sample.description)} chars")
        print(f"  Description:     {sample.description[:300]!r}")
        if sample.description and len(sample.description) > 300:
            print(f"                   ... ({len(sample.description)} chars total)")

    # Matcher comparison: title-only vs with-description
    matcher = JobMatcher(DEFAULT_PROFILE)
    if len(jobs) >= 2:
        j = jobs[0]
        from scrapers.models import Job
        from config.constants import ATSType
        title_only = Job(
            company=j.company, title=j.title, job_url=j.job_url,
            location=j.location, source_platform=ATSType.GREENHOUSE,
        )
        score_without = matcher.match(title_only).score
        score_with = matcher.match(j).score
        print(f"\nMatcher demo (first job: {j.title!r}):")
        print(f"  Score without description: {score_without}")
        print(f"  Score with description:    {score_with}")
        if score_with != score_without:
            print(f"  Improvement:               +{score_with - score_without} pts")
        else:
            print(f"  No change (description adds no new skill signals for this job)")

    ok = len(with_desc) > 0
    print(f"\n[{'PASS' if ok else 'FAIL'}] Greenhouse description extraction")
    return ok


# ---------------------------------------------------------------------------
# Lever: descriptionPlain from public JSON API
# ---------------------------------------------------------------------------

def test_lever_description():
    """Scrape Employ Inc. via the Lever JSON API and verify description fields."""
    print("\n=== Lever (Employ Inc.) ===")
    print("URL:      https://jobs.lever.co/employ")
    print("Path:     Lever public JSON API (?mode=json)")
    print()

    with BrowserService(headless=True) as browser:
        scraper = LeverScraper(browser)
        t0 = time.monotonic()
        jobs = scraper.scrape("https://jobs.lever.co/employ")
        elapsed = time.monotonic() - t0

    print(f"Elapsed:  {elapsed:.1f}s")
    print(f"Path:     {scraper.last_path}")
    print(f"Jobs:     {len(jobs)}")

    with_desc = [j for j in jobs if j.description]
    with_req  = [j for j in jobs if j.requirements]
    with_dept = [j for j in jobs if j.department]

    print(f"With description:  {len(with_desc)}/{len(jobs)}")
    print(f"With requirements: {len(with_req)}/{len(jobs)}")
    print(f"With department:   {len(with_dept)}/{len(jobs)}")
    print()

    sample = jobs[0] if jobs else None
    if sample:
        print(f"Sample job:        {sample.title!r}")
        print(f"  Location:        {sample.location!r}")
        print(f"  Department:      {sample.department!r}")
        if sample.description:
            print(f"  Description len: {len(sample.description)} chars")
            print(f"  Description:     {sample.description[:250]!r}")
        if sample.requirements:
            print(f"  Requirements len:{len(sample.requirements)} chars")
            print(f"  Requirements:    {sample.requirements[:250]!r}")

    ok = scraper.last_path == "api" and len(with_desc) > 0
    print(f"\n[{'PASS' if ok else 'FAIL'}] Lever description extraction via API")
    return ok


# ---------------------------------------------------------------------------
# Workday: browser detail-page fetch (opt-in)
# ---------------------------------------------------------------------------

def test_workday_description():
    """
    Scrape Adobe via Workday CXS API, then fetch 1 job's description via browser.

    The CXS API returns jobs without descriptions.  Setting max_descriptions=1
    triggers a single browser navigation to the first job's detail page.
    """
    print("\n=== Workday (Adobe) ===")
    print("URL:      https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced")
    print("Path:     CXS API (list) + browser detail page (description, 1 job)")
    print()

    with BrowserService(headless=True) as browser:
        scraper = WorkdayScraper(browser, max_descriptions=1)
        t0 = time.monotonic()
        jobs = scraper.scrape(
            "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"
        )
        elapsed = time.monotonic() - t0

    print(f"Elapsed:  {elapsed:.1f}s  (includes 1 detail-page browser load)")
    print(f"Path:     {scraper.last_path}")
    print(f"Jobs:     {len(jobs)}")

    with_desc = [j for j in jobs if j.description]
    print(f"With description: {len(with_desc)}/{len(jobs)}  (max_descriptions=1)")
    print()

    if with_desc:
        sample = with_desc[0]
        print(f"Sample job:        {sample.title!r}")
        print(f"  Location:        {sample.location!r}")
        print(f"  Description len: {len(sample.description)} chars")
        print(f"  Description:     {sample.description[:300]!r}")
        if len(sample.description) > 300:
            print(f"                   ... ({len(sample.description)} chars total)")

    # Demonstrate the performance cost of N detail pages
    print()
    print("Performance note:")
    print(f"  1 detail page fetched in {elapsed:.0f}s total (CXS API + 1 browser load)")
    print(f"  For all {len(jobs)} jobs: ~{len(jobs) * elapsed / max(1, len(with_desc)):.0f}s estimated")
    print("  → Use max_descriptions=5 to enrich only top-N for the matcher")

    ok = len(with_desc) >= 1
    print(f"\n[{'PASS' if ok else 'FAIL'}] Workday description extraction")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Sprint 11.5 — Job Description Extraction ===")

    results = []

    # Greenhouse (fastest — API returns descriptions in one call)
    results.append(test_greenhouse_description())

    # Lever (also API-first, description in list response)
    results.append(test_lever_description())

    # Workday (CXS API for listing, 1 browser load for description)
    results.append(test_workday_description())

    print()
    print(f"{'='*60}")
    print(f"  Results: {sum(results)}/{len(results)} tests passed")
    print(f"  Greenhouse: {'PASS' if results[0] else 'FAIL'}")
    print(f"  Lever:      {'PASS' if results[1] else 'FAIL'}")
    print(f"  Workday:    {'PASS' if results[2] else 'FAIL'}")
    print(f"{'='*60}")
    print()


if __name__ == "__main__":
    main()
