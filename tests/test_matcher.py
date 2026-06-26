"""
JobMatcher — demo and unit tests.

Reads every job from the SQLite database, scores each one against the
default user profile, and prints results ranked by score.

Run from the project root:
    python -m tests.test_matcher
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import JobRepository
from matching import JobMatcher, DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Inline unit tests (run without pytest)
# ---------------------------------------------------------------------------

def _run_unit_tests(matcher: JobMatcher) -> None:
    from scrapers.models import Job
    from config.constants import ATSType

    def make_job(title: str, location: str | None = None, description: str | None = None) -> Job:
        return Job(
            company="TestCo",
            title=title,
            job_url=f"https://testco.com/jobs/{title.replace(' ', '-').lower()}",
            location=location,
            description=description,
            source_platform=ATSType.GREENHOUSE,
        )

    print("--- Unit Tests ---\n")
    cases = [
        # (title, location, description, expected_minimum_score, label)
        ("Platform Engineer", "Remote", None,                     50, "preferred role + remote"),
        ("Senior Backend Engineer", "San Francisco, CA", "Java Spring Boot REST APIs", 80, "strong match with skills"),
        ("Customer Success Manager", "Anywhere in the United States", None, 0,  "non-engineering role"),
        ("Junior Java Developer", "Remote", None,                 20, "junior → seniority penalty"),
        ("Software Engineer II", "New York, NY", "Java microservices SQL", 85, "exact preferred role + skills"),
        ("Marketing Manager", None, None,                          0,  "completely off-profile"),
    ]

    all_passed = True
    for title, location, description, min_score, label in cases:
        job = make_job(title, location, description)
        result = matcher.match(job)
        status = "PASS" if result.score >= min_score else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {label}")
        print(f"         Title: '{title}' | Location: {location}")
        print(f"         Score: {result.score} (expected >= {min_score})")
        if result.matched_skills:
            print(f"         Matched: {result.matched_skills}")
        print()

    print("All unit tests passed.\n" if all_passed else "Some unit tests FAILED.\n")


# ---------------------------------------------------------------------------
# Database demo
# ---------------------------------------------------------------------------

def _run_database_demo(matcher: JobMatcher) -> None:
    print("--- Ranked Jobs from Database ---\n")

    with JobRepository() as repo:
        jobs = repo.get_all()

    if not jobs:
        print("No jobs in database. Run tests/test_database.py first.")
        return

    ranked = matcher.rank(jobs)

    print(f"{'#':<4} {'Score':<7} {'Title':<48} {'Location'}")
    print("-" * 90)

    for i, (job, result) in enumerate(ranked, start=1):
        title = job.title[:46] + ".." if len(job.title) > 48 else job.title
        loc   = (job.location or "—")[:28]
        print(f"{i:<4} {result.score:<7} {title:<48} {loc}")

    print()

    # Print full breakdown for the top 3
    print("--- Top 3 Detailed Breakdown ---\n")
    for i, (job, result) in enumerate(ranked[:3], start=1):
        print(f"#{i} [{result.score}/100] {job.title}")
        print(f"    Company:        {job.company}")
        print(f"    Location:       {job.location or '—'}")
        print(f"    Matched skills: {result.matched_skills or ['none']}")
        print(f"    Missing skills: {result.missing_skills or ['none']}")
        print("    Reasoning:")
        for reason in result.reasons:
            print(f"      • {reason}")
        print()

    print(f"Total Jobs Scored: {len(ranked)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n=== AI Job Hunter — Matcher Demo ===")
    print(f"Profile: {DEFAULT_PROFILE.summary()}\n")

    matcher = JobMatcher(DEFAULT_PROFILE)
    _run_unit_tests(matcher)
    _run_database_demo(matcher)
    print("=== Done ===\n")


if __name__ == "__main__":
    main()
