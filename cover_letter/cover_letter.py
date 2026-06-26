"""
CoverLetter — the output type for a generated cover letter.

Mirrors the TailoringResult / RankedJob pattern: a dataclass that bundles
structured data (content, subject line, metadata) with the file paths where
the letter was saved.

txt_path and md_path are None until CoverLetterAgent.generate(save=True) is
called.  Use the is_saved property to check whether files exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CoverLetter:
    """One generated cover letter and its metadata."""

    # Required — always set at construction time
    company: str
    title: str
    candidate_name: str
    candidate_email: str
    content: str        # full letter text, ready to copy-paste
    subject_line: str   # for email submission

    # Optional — set by CoverLetterAgent._save()
    txt_path: Path | None = None
    md_path: Path | None = None

    @property
    def is_saved(self) -> bool:
        """True when both output files have been written to disk."""
        return self.txt_path is not None and self.md_path is not None

    def __str__(self) -> str:
        return self.content
