"""
application package

Tracks the lifecycle of every job application from intent to outcome.

    from application import Application, ApplicationStatus, ApplicationRepository
    from application.application_status import VALID_TRANSITIONS
"""

from application.application_status import (
    ApplicationStatus,
    VALID_TRANSITIONS,
    TERMINAL_STATUSES,
)
from application.application import Application
from application.application_repository import ApplicationRepository

__all__ = [
    "ApplicationStatus", "VALID_TRANSITIONS", "TERMINAL_STATUSES",
    "Application",
    "ApplicationRepository",
]
