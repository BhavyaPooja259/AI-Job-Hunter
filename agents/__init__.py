"""
agents package

AI agents that perform intelligent tasks across the job hunting pipeline.

    from agents import ScoutAgent
"""

from agents.scout_agent import ScoutAgent
from agents.ranking_agent import RankingAgent, RankedJob

__all__ = ["ScoutAgent", "RankingAgent", "RankedJob"]
