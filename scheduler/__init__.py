"""
scheduler package

Orchestrates the full job-hunting workflow by coordinating existing agents.

    from scheduler import SchedulerConfig, WorkflowResult
    from agents.scheduler_agent import SchedulerAgent
"""

from scheduler.scheduler import SchedulerConfig
from scheduler.workflow import StepResult, WorkflowResult, WorkflowStep

__all__ = [
    "SchedulerConfig",
    "StepResult",
    "WorkflowResult",
    "WorkflowStep",
]
