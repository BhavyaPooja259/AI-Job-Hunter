"""
Application dataclass — one tracked job application.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from application.application_status import ApplicationStatus


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Application:
    """Represents a single job application and its current lifecycle state."""

    # Required — must be supplied at construction time
    company: str
    title: str
    job_url: str
    job_fingerprint: str

    # Optional — auto-populated if not supplied
    id: str = field(default_factory=_new_id)
    status: ApplicationStatus = ApplicationStatus.SAVED
    applied_at: datetime | None = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        applied = self.applied_at.date().isoformat() if self.applied_at else "—"
        return (
            f"[{self.status.value}]  {self.title} @ {self.company}"
            f"  (applied: {applied})"
        )
