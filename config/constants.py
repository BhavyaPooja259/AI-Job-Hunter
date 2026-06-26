"""
Project-wide constants.

These values are fixed by the domain — they do not change between environments
and do not belong in .env. If a value ever needs to be user-configurable,
move it to settings.py instead.
"""

from enum import Enum


# -----------------------------------------------------------------------------
# Enumerations
# Using str mixins so values serialize naturally to/from JSON and SQLite.
# -----------------------------------------------------------------------------

class ATSType(str, Enum):
    """Supported Applicant Tracking Systems."""
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    ORACLE = "oracle"
    SUCCESSFACTORS = "successfactors"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    """Lifecycle states for a single job application."""
    NEW = "new"                    # Discovered, not yet reviewed
    BOOKMARKED = "bookmarked"      # Flagged for action
    APPLIED = "applied"            # Application submitted
    OA = "oa"                      # Online assessment received
    PHONE_SCREEN = "phone_screen"  # Recruiter screen scheduled/done
    INTERVIEW = "interview"        # Technical interviews in progress
    OFFER = "offer"                # Offer received
    REJECTED = "rejected"          # Rejected at any stage
    ARCHIVED = "archived"          # Manually dismissed


# -----------------------------------------------------------------------------
# Target Roles
# Used by AI agents when matching job titles against the user's goal.
# -----------------------------------------------------------------------------

TARGET_ROLES: list[str] = [
    "Software Engineer II",
    "Backend Engineer",
    "Platform Engineer",
    "Java Backend Developer",
    "Senior Software Engineer",
    "Software Engineer",
]


# -----------------------------------------------------------------------------
# Application metadata
# -----------------------------------------------------------------------------

APP_NAME = "AI Job Hunter"
APP_VERSION = "0.1.0"
