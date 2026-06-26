"""
NotificationAgent — daily job digest builder and dispatcher.

Responsibilities
----------------
1. Read all jobs from the repository.
2. Deduplicate using job.fingerprint (SHA-256 of company + title + URL).
   First occurrence wins, which matches DB insertion order.
3. Score each unique job with the rule-based JobMatcher.
4. Sort by score descending.
5. Split into all_jobs and top_jobs (score >= threshold).
6. Build a JobDigest and hand it to NotificationService.
7. Return a NotificationResult with delivery metadata.

Why build_digest() is a separate public method
----------------------------------------------
Separating digest construction from dispatch lets callers (and tests) inspect
the digest without triggering any delivery.  This is also useful for a
"preview" CLI command that shows the digest without sending it anywhere.

Deduplication detail
--------------------
Job.fingerprint is a SHA-256 of (company + title + job_url).  Two scrape runs
on the same job produce identical fingerprints, so the second appearance is
silently dropped.  The total_in_db / unique_count pair in JobDigest lets users
see how much deduplication occurred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from matching.digest import DigestJob, JobDigest
from matching.matcher import JobMatcher
from matching.profile import DEFAULT_PROFILE, UserProfile
from scrapers.models import Job
from services.notification_service import NotificationService

if TYPE_CHECKING:
    from database.job_repository import JobRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class NotificationResult:
    """
    Returned by NotificationAgent.notify() and notify_from_jobs().

    Attributes
    ----------
    digest
        The structured digest that was (attempted to be) delivered.
    channels_notified
        Names of channels that received the digest successfully.
    failed_channels
        Names of channels that returned False or raised an exception.
    """

    digest: JobDigest
    channels_notified: list[str]
    failed_channels: list[str]

    @property
    def success(self) -> bool:
        """True when at least one channel received the digest with no failures."""
        return bool(self.channels_notified) and not self.failed_channels

    @property
    def any_sent(self) -> bool:
        """True when at least one channel received the digest."""
        return bool(self.channels_notified)

    def __str__(self) -> str:
        status = "ok" if self.success else "partial" if self.any_sent else "failed"
        sent = ", ".join(self.channels_notified) or "none"
        return (
            f"NotificationResult [{status}]  "
            f"sent={sent}  "
            f"digest={self.digest.summary()}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class NotificationAgent:
    """
    Generates and dispatches a daily job digest.

    Usage
    -----
    from services.notification_service import NotificationService, ConsoleChannel
    from agents.notification_agent import NotificationAgent

    service = NotificationService(channels=[ConsoleChannel()])
    agent = NotificationAgent(service=service)

    # From a live repository:
    result = agent.notify(repo)

    # From an in-memory list (useful in tests):
    result = agent.notify_from_jobs(jobs)

    # Preview without sending:
    digest = agent.build_digest(jobs)
    """

    def __init__(
        self,
        service: NotificationService,
        profile: UserProfile = DEFAULT_PROFILE,
        score_threshold: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        service
            The NotificationService that routes the digest to channels.

        profile
            Candidate profile used for rule-based scoring.
            Defaults to matching.profile.DEFAULT_PROFILE.

        score_threshold
            Minimum score for a job to appear in digest.top_jobs.
            Default 60 — jobs scoring below this are still listed in
            all_jobs but are not highlighted as top matches.
        """
        self._service = service
        self._profile = profile
        self._score_threshold = score_threshold

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def notify(self, repo: "JobRepository") -> NotificationResult:
        """Read all jobs from the repository, build a digest, and dispatch it."""
        jobs = repo.get_all()
        logger.info("notification run — %d jobs from repository", len(jobs))
        return self.notify_from_jobs(jobs)

    def notify_from_jobs(self, jobs: list[Job]) -> NotificationResult:
        """Build a digest from an in-memory list of jobs and dispatch it."""
        digest = self.build_digest(jobs)
        notified, failed = self._service.send(digest)
        result = NotificationResult(
            digest=digest,
            channels_notified=notified,
            failed_channels=failed,
        )
        logger.info(
            "notification complete — %s",
            "success" if result.success else f"partial ({failed})" if result.any_sent else "no channels",
        )
        return result

    def build_digest(self, jobs: list[Job]) -> JobDigest:
        """
        Build a JobDigest without dispatching it anywhere.

        Useful for previewing the digest, testing, or building a digest
        before deciding whether to send it.
        """
        total_in_db = len(jobs)
        unique = self._deduplicate(jobs)
        scored = self._score_and_sort(unique)
        top = [dj for dj in scored if dj.score >= self._score_threshold]

        logger.info(
            "digest built — %d in DB, %d unique, %d top (score ≥ %d)",
            total_in_db, len(unique), len(top), self._score_threshold,
        )

        return JobDigest(
            all_jobs=scored,
            top_jobs=top,
            total_in_db=total_in_db,
            unique_count=len(unique),
            score_threshold=self._score_threshold,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _deduplicate(self, jobs: list[Job]) -> list[Job]:
        """Return unique jobs in first-seen order using job.fingerprint."""
        seen: dict[str, Job] = {}
        for job in jobs:
            if job.fingerprint not in seen:
                seen[job.fingerprint] = job
        duplicates = len(jobs) - len(seen)
        if duplicates:
            logger.debug("deduplication removed %d duplicate jobs", duplicates)
        return list(seen.values())

    def _score_and_sort(self, jobs: list[Job]) -> list[DigestJob]:
        """Score each job with JobMatcher and return sorted best-first."""
        matcher = JobMatcher(self._profile)
        scored: list[DigestJob] = []
        for job in jobs:
            result = matcher.match(job)
            scored.append(DigestJob(
                job=job,
                score=result.score,
                matched_skills=result.matched_skills,
                missing_skills=result.missing_skills,
            ))
        return sorted(scored, key=lambda dj: dj.score, reverse=True)
