"""
agents package

AI agents that perform intelligent tasks across the job hunting pipeline.

    from agents import ScoutAgent
"""

from agents.scout_agent import ScoutAgent
from agents.ranking_agent import RankingAgent, RankedJob
from agents.resume_tailoring_agent import ResumeTailoringAgent, TailoringResult
from agents.notification_agent import NotificationAgent, NotificationResult
from agents.application_agent import ApplicationAgent, ApplicationStats

__all__ = [
    "ScoutAgent",
    "RankingAgent", "RankedJob",
    "ResumeTailoringAgent", "TailoringResult",
    "NotificationAgent", "NotificationResult",
    "ApplicationAgent", "ApplicationStats",
]
