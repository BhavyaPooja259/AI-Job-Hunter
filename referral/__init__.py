"""
referral package

Tracks recruiter / hiring-manager outreach and generates referral messages.

    from referral import Referral, ReferralMessages, ReferralStatus
    from referral import ReferralRepository
    from agents.referral_agent import ReferralAgent, ReferralStats
"""

from referral.referral import Referral, ReferralMessages, ReferralMessageData
from referral.referral_status import ReferralStatus, VALID_TRANSITIONS, TERMINAL_STATUSES
from referral.referral_repository import ReferralRepository

__all__ = [
    "Referral",
    "ReferralMessages",
    "ReferralMessageData",
    "ReferralStatus",
    "VALID_TRANSITIONS",
    "TERMINAL_STATUSES",
    "ReferralRepository",
]
