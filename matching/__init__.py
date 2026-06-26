"""
matching package

Rule-based (and later AI-powered) job matching against the user's profile.

    from matching import JobMatcher, UserProfile, MatchResult, DEFAULT_PROFILE
"""

from matching.profile import UserProfile, DEFAULT_PROFILE
from matching.matcher import MatchResult, JobMatcher
from matching.ai_result import AIMatchResult
from matching.tailor_result import TailoredResumeData, ExperienceEntry
from matching.digest import DigestJob, JobDigest

__all__ = [
    "UserProfile", "DEFAULT_PROFILE",
    "MatchResult", "JobMatcher",
    "AIMatchResult",
    "TailoredResumeData", "ExperienceEntry",
    "DigestJob", "JobDigest",
]
