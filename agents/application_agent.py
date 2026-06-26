"""
ApplicationAgent — orchestrates the job application lifecycle.

Responsibilities
----------------
1. track(job)     — create a new Application; return existing if already tracked.
2. advance(id)    — move to a new status, enforcing VALID_TRANSITIONS rules.
3. update_notes() — attach free-form notes without changing status.
4. active()       — list all applications that are not terminal.
5. stats()        — aggregate statistics across all tracked applications.

Separation of concerns
-----------------------
ApplicationRepository handles all SQL — it knows nothing about lifecycle rules.
ApplicationAgent holds lifecycle rules and duplicate-prevention logic.
This mirrors the NotificationAgent / NotificationService split: the agent
decides what to do; the repository decides how to store it.

Usage
-----
    with ApplicationRepository() as repo:
        agent = ApplicationAgent(repo=repo)
        app, is_new = agent.track(job)
        agent.advance(app.id, ApplicationStatus.READY_TO_APPLY)
        print(agent.stats().summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from application.application import Application
from application.application_repository import ApplicationRepository
from application.application_status import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    ApplicationStatus,
)
from scrapers.models import Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class ApplicationStats:
    """Aggregate statistics across all tracked applications."""

    total: int
    by_status: dict[str, int]  # status.value → count (only non-zero entries)
    active_count: int
    offer_count: int
    applied_count: int
    rejection_rate: float  # rejected / substantive (excludes SAVED and READY_TO_APPLY)

    def summary(self) -> str:
        return (
            f"Total: {self.total}  |  Active: {self.active_count}  |  "
            f"Offers: {self.offer_count}  |  "
            f"Rejection rate: {self.rejection_rate:.0%}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ApplicationAgent:
    """
    Orchestrates the job application lifecycle.

    Parameters
    ----------
    repo
        An initialized ApplicationRepository.  The agent does not manage
        the repository's connection lifecycle — the caller owns that.
    """

    def __init__(self, repo: ApplicationRepository) -> None:
        self._repo = repo

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def track(self, job: Job, notes: str = "") -> tuple[Application, bool]:
        """
        Start tracking a job application.

        Returns
        -------
        (application, is_new)
            is_new is True when the job was saved for the first time.
            is_new is False when it was already tracked; the existing
            Application is returned unchanged.
        """
        if self._repo.exists(job.fingerprint):
            existing = self._repo.get_by_fingerprint(job.fingerprint)
            logger.info(
                "already tracking %s @ %s — returning existing application",
                job.title,
                job.company,
            )
            return existing, False

        app = Application(
            company=job.company,
            title=job.title,
            job_url=job.job_url,
            job_fingerprint=job.fingerprint,
            notes=notes,
        )
        self._repo.save(app)
        logger.info(
            "started tracking %s @ %s (id=%s)", job.title, job.company, app.id
        )
        return app, True

    def advance(
        self,
        app_id: str,
        new_status: ApplicationStatus,
        notes: str = "",
    ) -> Application:
        """
        Advance an application to a new status, enforcing lifecycle rules.

        Parameters
        ----------
        app_id
            ID of the application to advance.
        new_status
            The target status.
        notes
            Optional free-form notes to attach on this transition.

        Returns
        -------
        Application
            The updated application (freshly read from the repository).

        Raises
        ------
        ValueError
            If app_id is not found, or the transition is not permitted
            by VALID_TRANSITIONS (including attempts to leave a terminal state).
        """
        app = self._repo.get_by_id(app_id)
        if app is None:
            raise ValueError(f"application {app_id!r} not found")

        valid_next = VALID_TRANSITIONS.get(app.status, frozenset())
        if new_status not in valid_next:
            if app.status in TERMINAL_STATUSES:
                raise ValueError(
                    f"cannot advance from terminal status {app.status.value!r}"
                )
            valid_str = ", ".join(s.value for s in valid_next) or "none"
            raise ValueError(
                f"invalid transition: {app.status.value} → {new_status.value}"
                f"  (valid next: {valid_str})"
            )

        self._repo.update_status(app_id, new_status)
        if notes:
            self._repo.update_notes(app_id, notes)

        updated = self._repo.get_by_id(app_id)
        logger.info(
            "advanced %s @ %s: %s → %s",
            app.title,
            app.company,
            app.status.value,
            new_status.value,
        )
        return updated

    def update_notes(self, app_id: str, notes: str) -> Application | None:
        """
        Update the free-form notes for an application without changing status.

        Returns the updated Application, or None if app_id is not found.
        """
        if not self._repo.update_notes(app_id, notes):
            return None
        return self._repo.get_by_id(app_id)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def active(self) -> list[Application]:
        """All applications that are not REJECTED or WITHDRAWN."""
        return self._repo.get_active()

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> ApplicationStats:
        """Compute aggregate statistics across all tracked applications."""
        apps = self._repo.get_all()
        total = len(apps)

        by_status: dict[str, int] = {}
        for app in apps:
            key = app.status.value
            by_status[key] = by_status.get(key, 0) + 1

        _ACTIVE = {
            ApplicationStatus.SAVED,
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.APPLIED,
            ApplicationStatus.ONLINE_ASSESSMENT,
            ApplicationStatus.PHONE_SCREEN,
            ApplicationStatus.INTERVIEW,
            ApplicationStatus.OFFER,
        }
        _INTENT_ONLY = {ApplicationStatus.SAVED, ApplicationStatus.READY_TO_APPLY}

        active_count = sum(1 for a in apps if a.status in _ACTIVE)
        offer_count = sum(1 for a in apps if a.status == ApplicationStatus.OFFER)
        applied_count = sum(1 for a in apps if a.status == ApplicationStatus.APPLIED)
        rejected_count = by_status.get(ApplicationStatus.REJECTED.value, 0)

        # Substantive = past the intent stage; gives a meaningful funnel metric
        substantive = sum(1 for a in apps if a.status not in _INTENT_ONLY)
        rejection_rate = rejected_count / substantive if substantive > 0 else 0.0

        return ApplicationStats(
            total=total,
            by_status=by_status,
            active_count=active_count,
            offer_count=offer_count,
            applied_count=applied_count,
            rejection_rate=rejection_rate,
        )
