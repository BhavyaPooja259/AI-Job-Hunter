"""
Digest data models for the NotificationAgent.

Why these live in matching/ rather than agents/
------------------------------------------------
NotificationService.send() accepts a JobDigest so it can format and deliver
it.  If JobDigest were defined in agents/notification_agent.py, the service
module would need to import from the agent module, and the agent module
imports from the service module — a circular import.

Placing the shared data types here (matching/digest.py) breaks the cycle:
both the agent and the service depend on this module; neither depends on the
other.

This follows the same pattern as:
    matching/ai_result.py      — shared output type for RankingAgent
    matching/tailor_result.py  — shared output type for ResumeTailoringAgent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from scrapers.models import Job


@dataclass
class DigestJob:
    """
    One job paired with its rule-based match assessment.

    Lightweight wrapper: the score and skill lists are copied out of
    MatchResult so callers don't need to import matching.matcher just
    to read a score.
    """

    job: Job
    score: int
    matched_skills: list[str]
    missing_skills: list[str]

    def __str__(self) -> str:
        loc = self.job.location or "Remote?"
        skills = ", ".join(self.matched_skills[:3]) or "—"
        return (
            f"[{self.score:>3}]  {self.job.title} @ {self.job.company}"
            f"  ({loc})  | {skills}"
        )


@dataclass
class JobDigest:
    """
    The full structured output of a daily notification run.

    Attributes
    ----------
    all_jobs
        Every unique job, sorted by match score descending.
    top_jobs
        Subset of all_jobs where score >= score_threshold.
    total_in_db
        Number of raw jobs read from the repository before deduplication.
    unique_count
        Number of unique jobs after fingerprint-based deduplication.
    score_threshold
        The score cutoff used to populate top_jobs.
    generated_at
        Timestamp set automatically at digest creation.
    """

    all_jobs: list[DigestJob]
    top_jobs: list[DigestJob]
    total_in_db: int
    unique_count: int
    score_threshold: int
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def by_company(self) -> dict[str, list[DigestJob]]:
        """All jobs grouped by company name, preserving score order."""
        result: dict[str, list[DigestJob]] = {}
        for dj in self.all_jobs:
            result.setdefault(dj.job.company, []).append(dj)
        return result

    def summary(self) -> str:
        """One-line human-readable digest summary."""
        return (
            f"{len(self.all_jobs)} unique jobs "
            f"({len(self.top_jobs)} top matches, score ≥ {self.score_threshold})"
        )
