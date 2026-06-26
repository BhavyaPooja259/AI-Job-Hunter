"""
Scraper data models.

Job is the canonical output of every scraper. All ATS adapters produce a
list[Job] regardless of the source platform — the rest of the pipeline
(storage, ranking, deduplication) works exclusively with this type.

Keeping the model here rather than in a shared `models/` directory is
intentional: Job belongs to the scraping domain and will be imported
upward by the database and AI layers when they are built.
"""

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, HttpUrl, Field, computed_field
import hashlib

from config.constants import ATSType


class Job(BaseModel):
    """A single job posting discovered from a company career page."""

    # Core identity
    company: str
    title: str
    job_url: str

    # Location and type
    location: Optional[str] = None
    employment_type: Optional[str] = None  # e.g. "Full-time", "Contract"

    # Content
    description: Optional[str] = None
    requirements: Optional[str] = None   # extracted requirements / qualifications section
    department: Optional[str] = None     # team or business unit, when provided by the ATS

    # Metadata
    posted_date: Optional[date] = None
    source_platform: ATSType
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @computed_field
    @property
    def fingerprint(self) -> str:
        """
        Stable hash used for deduplication.

        Built from company + title + URL so re-crawling the same job never
        creates a duplicate record. Does not include description or dates
        because those can change between crawls for the same posting.
        """
        raw = f"{self.company.lower()}::{self.title.lower()}::{self.job_url.lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        location = f" — {self.location}" if self.location else ""
        return f"[{self.source_platform.value}] {self.company} | {self.title}{location}"
