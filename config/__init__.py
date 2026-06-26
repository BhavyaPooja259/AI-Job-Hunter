"""
config package

Exposes the settings singleton and shared domain constants.

    from config import settings
    from config import ATSType, JobStatus, TARGET_ROLES

Company data is managed by CompanyRegistry, not stored here:

    from services.company_registry import CompanyRegistry
    registry = CompanyRegistry()
"""

from config.settings import settings
from config.constants import ATSType, JobStatus, TARGET_ROLES

__all__ = [
    "settings",
    "ATSType",
    "JobStatus",
    "TARGET_ROLES",
]
