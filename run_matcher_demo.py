"""
Demo: Job Matcher / Ranking pipeline.

Demonstrates the rule-based scoring pipeline — from loading jobs out of
jobs.db to a ranked list with per-job explanations — with no scraping,
tailoring, cover letters, referrals, or scheduler logic involved.

Pipeline
--------
  1. Load jobs      — open JobRepository and load every job from jobs.db.
  2. Match every job — score each job with JobMatcher against DEFAULT_PROFILE.
  3. Sort rankings  — print the top 10 jobs with scores and matched skills.
  4. Summary        — aggregate statistics and top recommended job.

Scoring dimensions (rule-based, deterministic, zero API cost)
--------------------------------------------------------------
  Role relevance   (0–40)  Title contains a preferred role phrase or keyword.
  Seniority fit    (0–20)  Seniority level aligns with SDE2 target.
  Skill signals    (0–25)  Key skills found in job description / requirements.
  Location fit     (0–15)  Remote, preferred city, or US-based.
  ──────────────────────
  Total            (0–100)

Run
---
  python run_matcher_demo.py
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
for _ns in ("matching", "database"):
    logging.getLogger(_ns).setLevel(logging.INFO)

from database.job_repository import JobRepository
from matching import DEFAULT_PROFILE, JobMatcher, MatchResult
from scrapers.models import Job

BANNER = "=" * 60
_TOP_N = 10


def main() -> None:
    print()
    print(BANNER)
    print("  Job Matcher Demo")
    print(BANNER)

    # ── 1/4 — load jobs ───────────────────────────────────────────────────────
    print()
    print("[ 1/4 ]  Load jobs")
    print("-" * 40)

    with JobRepository() as repo:
        jobs: list[Job] = repo.get_all()

    if not jobs:
        print()
        print("  No jobs found in jobs.db.")
        print("  Run run_scout_demo.py first to populate the database.")
        sys.exit(0)

    print(f"  Total jobs loaded : {len(jobs)}")

    # ── 2/4 — match every job ─────────────────────────────────────────────────
    print()
    print("[ 2/4 ]  Match every job")
    print("-" * 40)
    print(f"  Profile : {DEFAULT_PROFILE.summary()}")
    print()

    matcher = JobMatcher(DEFAULT_PROFILE)
    ranked: list[tuple[Job, MatchResult]] = matcher.rank(jobs)

    scores = [r.score for _, r in ranked]
    print(f"  Matched {len(ranked)} job(s).")

    # ── 3/4 — top 10 ──────────────────────────────────────────────────────────
    print()
    print(f"[ 3/4 ]  Sort rankings  (top {_TOP_N})")
    print("-" * 40)

    top = ranked[:_TOP_N]
    for i, (job, result) in enumerate(top, 1):
        skills_str = ", ".join(result.matched_skills) if result.matched_skills else "none"
        print()
        print(f"  {i}.")
        print(f"  {job.company}")
        print(f"  {job.title}")
        print(f"  Score: {result.score}")
        print(f"  Matched: {skills_str}")
        print("  " + "-" * 48)

    # ── 4/4 — summary ─────────────────────────────────────────────────────────
    print()
    print("[ 4/4 ]  Summary")
    print("-" * 40)

    avg_score = sum(scores) / len(scores) if scores else 0
    above_60 = sum(1 for s in scores if s >= 60)
    above_80 = sum(1 for s in scores if s >= 80)

    print(f"  Total jobs          : {len(jobs)}")
    print(f"  Average score       : {avg_score:.1f}")
    print(f"  Highest score       : {max(scores)}")
    print(f"  Lowest score        : {min(scores)}")
    print(f"  Jobs above 60       : {above_60}")
    print(f"  Jobs above 80       : {above_80}")

    # ── RESULT ────────────────────────────────────────────────────────────────
    print()
    print(BANNER)
    print("  RESULT")
    print(BANNER)

    best_job, best_result = ranked[0]
    top_reason = best_result.reasons[0] if best_result.reasons else "No reason recorded."

    print("  Top recommended job:")
    print()
    print(f"  Company : {best_job.company}")
    print(f"  Title   : {best_job.title}")
    print(f"  Score   : {best_result.score}")
    print(f"  Reason  : {top_reason}")
    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
