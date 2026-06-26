"""
Workflow data types — pure value objects, no I/O.

These types are the contract between JobRunner (which produces them) and
SchedulerAgent / demo scripts (which consume them).  Keeping them in a
dedicated file avoids circular imports and makes them easy to import in
tests without pulling in agent dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class WorkflowStep(str, Enum):
    """Ordered steps that make up one scheduler run."""

    SCOUT = "SCOUT"
    RANK = "RANK"
    NOTIFY = "NOTIFY"
    DASHBOARD = "DASHBOARD"
    TAILOR = "TAILOR"
    COVER_LETTER = "COVER_LETTER"
    REFERRAL_REMINDER = "REFERRAL_REMINDER"


@dataclass
class StepResult:
    """
    Outcome of a single workflow step.

    Attributes
    ----------
    step
        Which step this result belongs to.
    success
        False only on an unhandled exception — partial results still
        count as success so one failing job doesn't abort the step.
    processed
        Number of items that were fully handled (saved, generated, etc.).
    skipped
        Items that were intentionally bypassed (duplicate, below threshold,
        already processed from a prior run, optional step not enabled, …).
    message
        Human-readable one-line explanation of the outcome.
    duration_ms
        Wall-clock time for the step in milliseconds.
    """

    step: WorkflowStep
    success: bool
    processed: int = 0
    skipped: int = 0
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class WorkflowResult:
    """
    Aggregated outcome of a complete scheduler run.

    Attributes
    ----------
    started_at
        Timestamp recorded immediately before the first step.
    finished_at
        Timestamp recorded after the last step completes.  None until done.
    steps
        Ordered list of StepResult, one per step that ran.
    """

    started_at: datetime
    finished_at: datetime | None = None
    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when every step that ran reported success."""
        return bool(self.steps) and all(s.success for s in self.steps)

    @property
    def total_duration_ms(self) -> float:
        """Wall-clock duration of the full run in milliseconds."""
        if self.finished_at is not None:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return sum(s.duration_ms for s in self.steps)

    def add_step(self, result: StepResult) -> None:
        self.steps.append(result)

    def summary(self) -> str:
        """One-line-per-step human-readable summary."""
        status = "OK" if self.success else "PARTIAL"
        lines = [f"Workflow {status} — {len(self.steps)} steps ran"]
        for s in self.steps:
            icon = "✓" if s.success else "✗"
            counts = f"processed={s.processed}  skipped={s.skipped}"
            msg = f"  ({s.message})" if s.message else ""
            lines.append(f"  {icon}  {s.step.value:<22}  {counts}{msg}")
        lines.append(f"  Total: {self.total_duration_ms:.0f} ms")
        return "\n".join(lines)
