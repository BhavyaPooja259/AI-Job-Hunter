"""
SchedulerAgent — public API for running the full job-hunting workflow.

This agent is the single entry point callers should use.  It wires all
dependencies together and delegates execution to JobRunner.

Responsibilities
----------------
1. Accept injected agents and repositories (no hard-coded defaults).
2. Create a JobRunner from those dependencies.
3. Expose a single run(config) method that returns a WorkflowResult.
4. Log the overall outcome at INFO level.

Separation of concerns
-----------------------
SchedulerAgent   — public facade; owns no step logic.
JobRunner        — owns all step logic; directly testable with fakes.
SchedulerConfig  — value object; all run-time knobs.
WorkflowResult   — value object; the complete audit trail of a run.

Usage
-----
    from agents.scheduler_agent import SchedulerAgent
    from scheduler import SchedulerConfig
    from agents.scout_agent import ScoutAgent
    from agents.notification_agent import NotificationAgent
    from database.job_repository import JobRepository
    from services.notification_service import NotificationService

    with JobRepository() as job_repo:
        scout = ScoutAgent(repo=job_repo)
        notify = NotificationAgent(service=NotificationService())
        agent = SchedulerAgent(
            job_repo=job_repo,
            scout_agent=scout,
            notification_agent=notify,
        )
        config = SchedulerConfig(enable_cover_letters=True, only_new_jobs=False)
        result = agent.run(config)
        print(result.summary())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scheduler.job_runner import JobRunner
from scheduler.scheduler import SchedulerConfig
from scheduler.workflow import WorkflowResult

if TYPE_CHECKING:
    from agents.notification_agent import NotificationAgent
    from agents.scout_agent import ScoutAgent
    from ai.provider import AIProvider
    from database.job_repository import JobRepository
    from google import genai
    from referral.referral_repository import ReferralRepository

logger = logging.getLogger(__name__)


class SchedulerAgent:
    """
    Orchestrates the full job-hunting workflow.

    Parameters
    ----------
    job_repo
        An open JobRepository.  The caller owns the connection lifecycle.
    scout_agent
        ScoutAgent that discovers and persists jobs.
    notification_agent
        NotificationAgent that builds and dispatches the daily digest.
    referral_repo
        Optional open ReferralRepository.  Required when
        config.enable_referral_reminders=True.
    client
        Optional Gemini client.  Required when config.enable_tailoring=True.
        Passed through to CoverLetterAgent for AI-enhanced letters when set.
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
        self._runner = JobRunner(
            job_repo=job_repo,
            scout_agent=scout_agent,
            notification_agent=notification_agent,
            referral_repo=referral_repo,
            client=client,
            provider=provider,
        )

    def run(self, config: SchedulerConfig | None = None) -> WorkflowResult:
        """
        Execute one full workflow run.

        Parameters
        ----------
        config
            Runtime configuration.  Defaults to SchedulerConfig() when None
            (core steps only, no optional steps, no new-jobs filter).

        Returns
        -------
        WorkflowResult
            Complete audit trail — one StepResult per step that ran.
        """
        if config is None:
            config = SchedulerConfig()

        logger.info(
            "scheduler run started — tailoring=%s  cover_letters=%s  referrals=%s  only_new=%s",
            config.enable_tailoring,
            config.enable_cover_letters,
            config.enable_referral_reminders,
            config.only_new_jobs,
        )

        result = self._runner.run(config)

        logger.info(
            "scheduler run complete — %s in %.0f ms",
            "OK" if result.success else "PARTIAL",
            result.total_duration_ms,
        )
        return result
