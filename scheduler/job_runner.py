"""
JobRunner — executes each workflow step and aggregates the results.

Design
------
JobRunner is intentionally separate from SchedulerAgent so that:
  - Tests can construct a JobRunner with lightweight fakes without going
    through the full SchedulerAgent wiring.
  - SchedulerAgent stays a thin public-API wrapper with no step logic.

Idempotency
-----------
_run_scout() snapshots the set of fingerprints already in the database
BEFORE calling scout_agent.run().  After scouting, it diffs the new
full set to produce `new_fingerprints`.  Optional steps (TAILOR,
COVER_LETTER) only process jobs whose fingerprints are in
`new_fingerprints` when config.only_new_jobs=True (the default).

On a second run with no new jobs the optional steps see an empty candidate
list and report skipped=N — no files are generated, no API calls made.

Dashboard step
--------------
The FastAPI dashboard reads from the database on every request, so no
explicit refresh is needed.  The DASHBOARD step is a lightweight success
marker that records this fact and keeps the step list complete.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from matching.matcher import JobMatcher
from matching.profile import DEFAULT_PROFILE
from scheduler.scheduler import SchedulerConfig
from scheduler.workflow import StepResult, WorkflowResult, WorkflowStep

if TYPE_CHECKING:
    from agents.notification_agent import NotificationAgent
    from agents.scout_agent import ScoutAgent
    from ai.provider import AIProvider
    from database.job_repository import JobRepository
    from google import genai
    from referral.referral_repository import ReferralRepository
    from scrapers.models import Job

logger = logging.getLogger(__name__)


class JobRunner:
    """
    Executes each workflow step in order and returns a WorkflowResult.

    Parameters
    ----------
    job_repo
        Open JobRepository used to read jobs before and after scouting.
    scout_agent
        ScoutAgent instance.  Its run() method may open its own internal
        database connection — this is fine; SQLite allows concurrent reads.
    notification_agent
        NotificationAgent for building and dispatching the digest.
    referral_repo
        Open ReferralRepository for the referral-reminder step.
        Required only when config.enable_referral_reminders=True.
    client
        Gemini client for the tailoring step.
        Required only when config.enable_tailoring=True.
    """

    def __init__(
        self,
        job_repo: "JobRepository",
        scout_agent: "ScoutAgent",
        notification_agent: "NotificationAgent",
        referral_repo: "ReferralRepository | None" = None,
        client: "genai.Client | None" = None,
        provider: "AIProvider | None" = None,
    ) -> None:
        self._job_repo = job_repo
        self._scout_agent = scout_agent
        self._notification_agent = notification_agent
        self._referral_repo = referral_repo
        self._client = client
        self._provider = provider

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def run(self, config: SchedulerConfig) -> WorkflowResult:
        """
        Execute the full workflow and return a WorkflowResult.

        Steps always run: SCOUT, RANK, NOTIFY, DASHBOARD.
        Optional steps run when enabled in config and dependencies are met.
        """
        result = WorkflowResult(started_at=datetime.now())
        logger.info("workflow run started")

        # Step 1 — Scout
        scout_step, all_jobs, new_fingerprints = self._run_scout()
        result.add_step(scout_step)

        # Step 2 — Rank (filter to top N above min_score)
        rank_step, top_pairs = self._run_rank(config, all_jobs)
        result.add_step(rank_step)

        # Step 3 — Notify
        result.add_step(self._run_notify(all_jobs))

        # Step 4 — Dashboard (always a no-op success marker)
        result.add_step(self._run_dashboard(all_jobs))

        # Candidate jobs for optional steps
        if config.only_new_jobs:
            candidate_pairs = [
                (j, r) for j, r in top_pairs if j.fingerprint in new_fingerprints
            ]
        else:
            candidate_pairs = top_pairs

        # Step 5 — Tailor (optional)
        if config.enable_tailoring:
            result.add_step(self._run_tailor(config, candidate_pairs))

        # Step 6 — Cover Letter (optional)
        if config.enable_cover_letters:
            result.add_step(self._run_cover_letters(config, candidate_pairs))

        # Step 7 — Referral Reminder (optional)
        if config.enable_referral_reminders:
            result.add_step(self._run_referral_reminders())

        result.finished_at = datetime.now()
        logger.info("workflow run complete — %s", "OK" if result.success else "PARTIAL")
        return result

    # ------------------------------------------------------------------ #
    # Step implementations
    # ------------------------------------------------------------------ #

    def _run_scout(self) -> tuple[StepResult, list["Job"], set[str]]:
        """
        Scout new jobs.

        Returns the step result, all jobs now in the DB, and the set of
        fingerprints that are NEW (not present before this run).
        """
        t0 = time.monotonic()
        try:
            pre_fingerprints = {j.fingerprint for j in self._job_repo.get_all()}
            scan = self._scout_agent.run()
            all_jobs = self._job_repo.get_all()
            new_fingerprints = {j.fingerprint for j in all_jobs} - pre_fingerprints

            logger.info(
                "scout: %d companies, %d saved, %d duplicates, %d new this run",
                scan.companies_checked,
                scan.jobs_saved,
                scan.duplicates,
                len(new_fingerprints),
            )

            return (
                StepResult(
                    step=WorkflowStep.SCOUT,
                    success=scan.failure_count == 0,
                    processed=scan.jobs_saved,
                    skipped=scan.duplicates,
                    message=(
                        f"{scan.companies_checked} companies checked, "
                        f"{len(new_fingerprints)} new jobs"
                    ),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ),
                all_jobs,
                new_fingerprints,
            )

        except Exception as exc:
            logger.error("scout step failed: %s", exc)
            return (
                StepResult(
                    step=WorkflowStep.SCOUT,
                    success=False,
                    message=str(exc),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ),
                self._job_repo.get_all(),
                set(),
            )

    def _run_rank(
        self,
        config: SchedulerConfig,
        all_jobs: list["Job"],
    ) -> tuple[StepResult, list]:
        """
        Rank all jobs and return the top-N pairs above min_score.

        Returns (step_result, [(Job, MatchResult), ...]).
        """
        t0 = time.monotonic()
        try:
            matcher = JobMatcher(DEFAULT_PROFILE)
            ranked = matcher.rank(all_jobs)
            top_pairs = [
                (j, r)
                for j, r in ranked
                if r.score >= config.min_score
            ][: config.top_n]

            below = len(all_jobs) - len(top_pairs)
            logger.info(
                "rank: %d jobs total, %d meet threshold (≥%d), top %d selected",
                len(all_jobs),
                len([r for _, r in ranked if r.score >= config.min_score]),
                config.min_score,
                len(top_pairs),
            )

            return (
                StepResult(
                    step=WorkflowStep.RANK,
                    success=True,
                    processed=len(top_pairs),
                    skipped=below,
                    message=(
                        f"{len(top_pairs)} jobs meet threshold "
                        f"(min_score={config.min_score}, top_n={config.top_n})"
                    ),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ),
                top_pairs,
            )

        except Exception as exc:
            logger.error("rank step failed: %s", exc)
            return (
                StepResult(
                    step=WorkflowStep.RANK,
                    success=False,
                    message=str(exc),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ),
                [],
            )

    def _run_notify(self, all_jobs: list["Job"]) -> StepResult:
        t0 = time.monotonic()
        try:
            notification_result = self._notification_agent.notify_from_jobs(all_jobs)
            digest = notification_result.digest
            logger.info(
                "notify: digest built — %d unique, %d top jobs",
                digest.unique_count,
                len(digest.top_jobs),
            )
            return StepResult(
                step=WorkflowStep.NOTIFY,
                success=True,
                processed=digest.unique_count,
                message=f"{len(digest.top_jobs)} top jobs in digest",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.error("notify step failed: %s", exc)
            return StepResult(
                step=WorkflowStep.NOTIFY,
                success=False,
                message=str(exc),
                duration_ms=(time.monotonic() - t0) * 1000,
            )

    def _run_dashboard(self, all_jobs: list["Job"]) -> StepResult:
        """
        Dashboard step — always succeeds immediately.

        The FastAPI dashboard reads from the database on every request, so
        no explicit refresh call is needed.  This step exists to keep the
        workflow log complete and signal that the data layer is up to date.
        """
        return StepResult(
            step=WorkflowStep.DASHBOARD,
            success=True,
            processed=len(all_jobs),
            message="dashboard reads from DB on demand — data is current",
        )

    def _run_tailor(
        self,
        config: SchedulerConfig,
        candidate_pairs: list,
    ) -> StepResult:
        """
        Generate tailored resumes for candidate jobs.

        Requires config.resume_text and a Gemini client.
        Falls back gracefully if either is missing.
        """
        t0 = time.monotonic()

        if self._client is None and self._provider is None:
            return StepResult(
                step=WorkflowStep.TAILOR,
                success=True,
                skipped=len(candidate_pairs),
                message="skipped: no Gemini client (set GEMINI_API_KEY)",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        if not config.resume_text:
            return StepResult(
                step=WorkflowStep.TAILOR,
                success=True,
                skipped=len(candidate_pairs),
                message="skipped: no resume_text in SchedulerConfig",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        if not candidate_pairs:
            return StepResult(
                step=WorkflowStep.TAILOR,
                success=True,
                skipped=0,
                message="no candidate jobs to tailor",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        from agents.resume_tailoring_agent import ResumeTailoringAgent

        agent = ResumeTailoringAgent(
            client=self._client,
            provider=self._provider,
            resume_text=config.resume_text,
        )

        processed = 0
        for job, rule_result in candidate_pairs:
            tailor_result = agent.tailor(job, rule_result=rule_result)
            if tailor_result is not None:
                processed += 1
                logger.info("tailored resume for %s @ %s", job.title, job.company)

        return StepResult(
            step=WorkflowStep.TAILOR,
            success=True,
            processed=processed,
            skipped=len(candidate_pairs) - processed,
            message=f"{processed}/{len(candidate_pairs)} resumes tailored",
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    def _run_cover_letters(
        self,
        config: SchedulerConfig,
        candidate_pairs: list,
    ) -> StepResult:
        """
        Generate cover letters for candidate jobs.

        Works in template-only mode when no Gemini client is available.
        """
        t0 = time.monotonic()

        if not candidate_pairs:
            return StepResult(
                step=WorkflowStep.COVER_LETTER,
                success=True,
                skipped=0,
                message="no candidate jobs for cover letters",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        from agents.cover_letter_agent import CoverLetterAgent

        agent = CoverLetterAgent(
            resume_text=config.resume_text,
            client=self._client,
            provider=self._provider,
            candidate_name=config.candidate_name,
            candidate_email=config.candidate_email,
        )

        processed = 0
        for job, _ in candidate_pairs:
            try:
                agent.generate(job, save=True)
                processed += 1
                logger.info("cover letter generated for %s @ %s", job.title, job.company)
            except Exception as exc:
                logger.warning(
                    "cover letter failed for %s @ %s: %s", job.title, job.company, exc
                )

        return StepResult(
            step=WorkflowStep.COVER_LETTER,
            success=True,
            processed=processed,
            skipped=len(candidate_pairs) - processed,
            message=f"{processed}/{len(candidate_pairs)} cover letters generated",
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    def _run_referral_reminders(self) -> StepResult:
        """
        Identify active referral contacts that may need a follow-up.

        Contacts in REQUEST_SENT status have had an outreach sent but no
        reply yet — they are the primary candidates for a follow-up.
        """
        t0 = time.monotonic()

        if self._referral_repo is None:
            return StepResult(
                step=WorkflowStep.REFERRAL_REMINDER,
                success=True,
                message="skipped: no ReferralRepository configured",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        from referral.referral_status import ReferralStatus

        active = self._referral_repo.get_active()
        needs_followup = [
            r for r in active
            if r.status == ReferralStatus.REQUEST_SENT
        ]

        for ref in needs_followup:
            contacted = (
                ref.contacted_at.strftime("%Y-%m-%d") if ref.contacted_at else "unknown"
            )
            logger.info(
                "follow-up reminder: %s @ %s  (contacted %s)",
                ref.contact_name, ref.company, contacted,
            )

        return StepResult(
            step=WorkflowStep.REFERRAL_REMINDER,
            success=True,
            processed=len(needs_followup),
            skipped=len(active) - len(needs_followup),
            message=f"{len(needs_followup)} contacts need follow-up",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
