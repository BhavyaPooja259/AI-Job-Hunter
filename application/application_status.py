"""
ApplicationStatus enum and lifecycle transition rules.
"""

from __future__ import annotations

from enum import Enum


class ApplicationStatus(str, Enum):
    """Lifecycle statuses for a job application, ordered from intent to outcome."""

    SAVED = "SAVED"
    READY_TO_APPLY = "READY_TO_APPLY"
    APPLIED = "APPLIED"
    ONLINE_ASSESSMENT = "ONLINE_ASSESSMENT"
    PHONE_SCREEN = "PHONE_SCREEN"
    INTERVIEW = "INTERVIEW"
    OFFER = "OFFER"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


# Allowed next statuses from each state.
# APPLIED → PHONE_SCREEN is valid because many companies skip OA.
# OFFER → REJECTED covers offer-rescinded scenarios.
VALID_TRANSITIONS: dict[ApplicationStatus, frozenset[ApplicationStatus]] = {
    ApplicationStatus.SAVED: frozenset({
        ApplicationStatus.READY_TO_APPLY,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.READY_TO_APPLY: frozenset({
        ApplicationStatus.APPLIED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.APPLIED: frozenset({
        ApplicationStatus.ONLINE_ASSESSMENT,
        ApplicationStatus.PHONE_SCREEN,
        ApplicationStatus.INTERVIEW,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.ONLINE_ASSESSMENT: frozenset({
        ApplicationStatus.PHONE_SCREEN,
        ApplicationStatus.INTERVIEW,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.PHONE_SCREEN: frozenset({
        ApplicationStatus.INTERVIEW,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.INTERVIEW: frozenset({
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.OFFER: frozenset({
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }),
    ApplicationStatus.REJECTED: frozenset(),
    ApplicationStatus.WITHDRAWN: frozenset(),
}

TERMINAL_STATUSES: frozenset[ApplicationStatus] = frozenset({
    ApplicationStatus.REJECTED,
    ApplicationStatus.WITHDRAWN,
})
