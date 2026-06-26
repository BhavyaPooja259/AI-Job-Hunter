"""
Demo: tailor Bhavya's resume for one real Adobe job.

Pipeline
--------
  1. Open the database (auto-migrates schema if needed).
  2. Load Adobe jobs from the DB.
     If none exist → scrape Adobe's Workday board and save them.
  3. Score all Adobe jobs with the rule-based JobMatcher.
  4. Pick the highest-scoring job that has a description.
     Fall back to the highest-scoring job without a description if needed.
  5. Call ResumeTailoringAgent → generates a tailored ATS plain-text resume.
  6. Print a summary and the first section of the tailored resume.
     Both the .txt and the _analysis.json are saved under resume/tailored/.

Requirements
------------
  ANTHROPIC_API_KEY must be set in your .env or shell environment.
  Playwright must be installed (playwright install chromium) for the
  Workday scraper step.

Run
---
  python run_tailor_demo.py
"""

import logging
import sys
from pathlib import Path

# ── project root must be on the path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,       # suppress library noise
    format="%(levelname)-8s %(name)s — %(message)s",
)
# Only show INFO from our own modules so progress is visible
for ns in ("agents", "matching", "scrapers", "database"):
    logging.getLogger(ns).setLevel(logging.INFO)

from google import genai

from config import settings
from database.job_repository import JobRepository
from matching.matcher import JobMatcher
from matching.profile import DEFAULT_PROFILE
from agents.resume_tailoring_agent import ResumeTailoringAgent

ADOBE_WORKDAY_URL = "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"
BANNER = "=" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — load (or scrape) Adobe jobs
# ─────────────────────────────────────────────────────────────────────────────

def load_or_scrape_adobe_jobs():
    """
    Return a list of Adobe Job objects.

    If the DB already has Adobe jobs, return them directly.
    Otherwise scrape the Workday board, save to DB, and return the result.

    max_descriptions=5 fetches the full description for the first 5 jobs
    via a browser detail-page visit.  The CXS API (used for the remaining
    jobs) omits descriptions, so we want at least a few with descriptions
    so the tailoring agent has real content to work with.
    """
    with JobRepository() as repo:
        all_jobs = repo.get_all()
        adobe_jobs = [j for j in all_jobs if j.company == "Adobe"]

        if adobe_jobs:
            desc_count = sum(1 for j in adobe_jobs if j.description)
            print(f"  Loaded {len(adobe_jobs)} Adobe jobs from database "
                  f"({desc_count} with descriptions).")
            return adobe_jobs

        print("  No Adobe jobs in database — scraping Workday now …")
        print(f"  URL: {ADOBE_WORKDAY_URL}")
        print(f"  Fetching descriptions for the first 5 jobs (browser).")
        print()

        from browser import BrowserService
        from scrapers.workday_scraper import WorkdayScraper

        with BrowserService(headless=True) as browser:
            scraper = WorkdayScraper(browser, max_descriptions=5)
            jobs = scraper.scrape(ADOBE_WORKDAY_URL)

        if not jobs:
            print("  ERROR: WorkdayScraper returned 0 jobs. Check the URL or network.")
            sys.exit(1)

        saved, skipped = repo.save_many(jobs)
        desc_count = sum(1 for j in jobs if j.description)
        print(f"  Scraped  : {len(jobs)} jobs  (path: {scraper.last_path})")
        print(f"  Saved    : {saved} new, {skipped} already existed")
        print(f"  With desc: {desc_count} jobs have full descriptions")
        return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — pick the best job to tailor
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_job(jobs):
    """
    Score all jobs with the rule-based matcher, return (job, rule_result).

    Preference order:
      1. Highest-scoring job that HAS a description  (best content for Claude)
      2. Highest-scoring job overall                 (if none have descriptions)
    """
    matcher = JobMatcher(DEFAULT_PROFILE)
    ranked = matcher.rank(jobs)   # sorted highest score first

    # Prefer a job with a description so Claude has real content to tailor with
    for job, result in ranked:
        if job.description:
            return job, result

    # Fall back to top job even without description
    return ranked[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(BANNER)
    print("  Resume Tailoring Demo — Adobe × Bhavya L")
    print(BANNER)

    # ── verify API key up front ───────────────────────────────────────────────
    if not settings.gemini_api_key:
        print()
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Add it to your .env file or export it in your shell:")
        print("  export GEMINI_API_KEY=AIza…")
        sys.exit(1)

    # ── step 1: jobs ─────────────────────────────────────────────────────────
    print()
    print("[ 1/3 ]  Loading Adobe jobs")
    print("-" * 40)
    adobe_jobs = load_or_scrape_adobe_jobs()

    # ── step 2: pick ─────────────────────────────────────────────────────────
    print()
    print("[ 2/3 ]  Selecting the best job")
    print("-" * 40)
    job, rule_result = pick_best_job(adobe_jobs)

    print(f"  Title    : {job.title}")
    print(f"  Location : {job.location or 'Not specified'}")
    print(f"  Rule score: {rule_result.score}/100")
    print(f"  Matched  : {', '.join(rule_result.matched_skills) or 'none'}")
    print(f"  Has desc : {'yes (' + str(len(job.description)) + ' chars)' if job.description else 'no'}")
    print(f"  URL      : {job.job_url}")

    # ── step 3: tailor ───────────────────────────────────────────────────────
    print()
    print("[ 3/3 ]  Calling ResumeTailoringAgent (Gemini)")
    print("-" * 40)
    print("  Sending resume + job to Gemini … (may take 10–20 s)")

    client = genai.Client(api_key=settings.gemini_api_key)
    agent = ResumeTailoringAgent(client=client)

    result = agent.tailor(job, rule_result=rule_result)

    if result is None:
        print()
        print("ERROR: Tailoring failed. Check the logs above for details.")
        sys.exit(1)

    # ── results ──────────────────────────────────────────────────────────────
    print()
    print(BANNER)
    print("  RESULT")
    print(BANNER)
    print(f"  ATS estimate : {result.data.ats_score_estimate}%")
    print(f"  Keywords     : {', '.join(result.data.keywords_incorporated[:8])}")
    print(f"  Strengths    : {len(result.data.selected_experience)} roles selected")
    print()
    print(f"  Saved text   : {result.text_path}")
    print(f"  Saved JSON   : {result.analysis_path}")

    print()
    print(BANNER)
    print("  TAILORED RESUME (first 35 lines)")
    print(BANNER)
    lines = result.data.full_resume_text.splitlines()
    for line in lines[:35]:
        print(f"  {line}")
    if len(lines) > 35:
        print(f"  … ({len(lines) - 35} more lines — see {result.text_path.name})")

    print()
    print(BANNER)
    print("  TAILORING NOTES")
    print(BANNER)
    for note_line in result.data.tailoring_notes.splitlines():
        print(f"  {note_line}")

    print()


if __name__ == "__main__":
    main()
