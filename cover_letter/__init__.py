"""
cover_letter package

Generates personalised cover letters from a job description and resume.

    from cover_letter import CoverLetter, CoverLetterData, CoverLetterGenerator
    from agents.cover_letter_agent import CoverLetterAgent
"""

from cover_letter.cover_letter import CoverLetter
from cover_letter.cover_letter_generator import (
    CoverLetterData,
    CoverLetterGenerator,
    DEFAULT_LETTER_TEMPLATE,
)

__all__ = [
    "CoverLetter",
    "CoverLetterData",
    "CoverLetterGenerator",
    "DEFAULT_LETTER_TEMPLATE",
]
