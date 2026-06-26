"""
Demo: run one complete Automation Scheduler workflow immediately.

Pipeline (same order as production)
------------------------------------
  1. SCOUT           — checks registered companies for new jobs.
                       In this demo the company registry is bypassed with a
                       no-op scout so the run is instant and network-free.
                       Remove the override below to enable real scraping.
  2. RANK            — scores every job in the database with the rule-based
                       JobMatcher and selects the top N above the threshold.
  3. NOTIFY          — builds the daily job digest.
  4. DASHBOARD       — confirms the FastAPI dashboard data is current
                       (no explicit call needed; it reads from DB on demand).
  5. COVER_LETTER    — generates cover letters for top-ranked jobs using the
                       deterministic template engine (no API key required).
  6. REFERRAL_REMINDER — lists active referral contacts in REQUEST_SENT
                         status that may need a follow-up message.

Idempotency
-----------
  first run  — cover letters are generated for any top-ranked new jobs.
  later runs — only_new_jobs=True means optional steps skip jobs that
               were already in the database before this run's scout step.
               Set only_new_jobs=False (see DEMO_CONFIG below) to force
               reprocessing of all top-ranked jobs regardless.

Run
---
  python run_scheduler_demo.py

To enable real job scraping uncomment the REAL SCOUT block and remove the
NO-OP SCOUT block below.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
for ns in ("agents", "scheduler", "matching", "database"):
    logging.getLogger(ns).setLevel(logging.INFO)

from agents.notification_agent import NotificationAgent
from agents.scout_agent import ScanResult
from agents.scheduler_agent import SchedulerAgent
from config import settings
from database.job_repository import JobRepository
from referral.referral_repository import ReferralRepository
from scheduler.scheduler import SchedulerConfig
from scheduler.workflow import WorkflowStep
from services.notification_service import NotificationService

BANNER = "=" * 60

# ── Demo configuration ────────────────────────────────────────────────────────

DEMO_CONFIG = SchedulerConfig(
    top_n=3,
    min_score=0,                   # include all jobs (demo mode)
    enable_tailoring=False,        # requires Gemini client
    enable_cover_letters=True,     # works in template mode (no API key)
    enable_referral_reminders=True,
    resume_text="",                # empty → cover letter uses template basics
    candidate_name=settings.user_name or "Bhavya L",
    candidate_email=settings.user_email or "",
    only_new_jobs=False,           # process existing top jobs (demo mode)
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(BANNER)
    print("  Automation Scheduler Demo")
    print(BANNER)
    print()
    print(f"  Candidate : {DEMO_CONFIG.candidate_name}")
    print(f"  top_n     : {DEMO_CONFIG.top_n}  |  min_score: {DEMO_CONFIG.min_score}")
    print(f"  Steps     : cover_letters={DEMO_CONFIG.enable_cover_letters}"
          f"  referrals={DEMO_CONFIG.enable_referral_reminders}")
    print(f"  Mode      : only_new_jobs={DEMO_CONFIG.only_new_jobs}")
    print()

    # ── Gemini client (optional) ──────────────────────────────────────────────
    client = None
    if settings.gemini_api_key:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        print("  AI mode   : Gemini client active")
    else:
        print("  AI mode   : template-only (GEMINI_API_KEY not set)")

    # ── Repositories ─────────────────────────────────────────────────────────
    job_repo = JobRepository()
    job_repo.initialize()

    referral_repo = ReferralRepository()
    referral_repo.initialize()

    # ── Notification agent ────────────────────────────────────────────────────
    notification_agent = NotificationAgent(service=NotificationService())

    # ── Scout: NO-OP (network-free demo) ─────────────────────────────────────
    # Replace this block with a real ScoutAgent to enable live scraping.
    noop_scout = MagicMock()
    noop_scout.run.return_value = ScanResult(
        companies_checked=0,
        jobs_scraped=0,
        jobs_saved=0,
        duplicates=0,
    )

    # ── (Alternative) REAL SCOUT — uncomment to enable live scraping:
    # from agents.scout_agent import ScoutAgent
    # noop_scout = ScoutAgent(repo=job_repo)

    # ── Scheduler agent ───────────────────────────────────────────────────────
    agent = SchedulerAgent(
        job_repo=job_repo,
        scout_agent=noop_scout,
        notification_agent=notification_agent,
        referral_repo=referral_repo,
        client=client,
    )

    print(BANNER)
    print("  Running workflow…")
    print(BANNER)
    print()

    result = agent.run(DEMO_CONFIG)

    job_repo.close()
    referral_repo.close()

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(BANNER)
    print("  WORKFLOW RESULT")
    print(BANNER)
    print()

    status = "OK" if result.success else "PARTIAL"
    print(f"  Status    : {status}")
    print(f"  Duration  : {result.total_duration_ms:.0f} ms")
    print(f"  Steps ran : {len(result.steps)}")
    print()

    for step in result.steps:
        icon = "✓" if step.success else "✗"
        line = f"  {icon}  {step.step.value:<22}  processed={step.processed}  skipped={step.skipped}"
        if step.message:
            line += f"  → {step.message}"
        print(line)

    # ── Highlight key outcomes ────────────────────────────────────────────────
    cover_step = next(
        (s for s in result.steps if s.step == WorkflowStep.COVER_LETTER), None
    )
    if cover_step and cover_step.processed > 0:
        print()
        print(f"  Cover letters saved to: storage/cover_letters/")

    referral_step = next(
        (s for s in result.steps if s.step == WorkflowStep.REFERRAL_REMINDER), None
    )
    if referral_step and referral_step.processed > 0:
        print()
        print(
            f"  {referral_step.processed} contact(s) need a follow-up — "
            "check data/referrals.db"
        )

    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
