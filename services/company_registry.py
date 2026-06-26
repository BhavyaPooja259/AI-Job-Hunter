"""
Company Registry

Loads company data from data/companies.json and exposes it through a clean,
queryable interface.

Why JSON instead of hardcoded constants:
  - Adding or disabling a company requires no code change and no redeploy
  - Non-engineers can edit the company list directly
  - The file can be version-controlled independently from source code
  - The schema is easy to extend (add `careers_url`, `priority`, `notes`)
    without touching any module that consumes companies
  - Future: swap the loader to pull from a database or API with zero
    changes to the interface callers depend on
"""

import json
from pathlib import Path
from functools import cached_property

from pydantic import BaseModel, HttpUrl, field_validator

from config.constants import ATSType


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Company(BaseModel):
    """A single target company with its ATS platform and monitoring metadata."""

    name: str
    ats: ATSType
    careers_url: HttpUrl
    priority: int  # 1 = highest priority; higher numbers = lower priority
    active: bool = True
    notes: str | None = None

    @field_validator("priority")
    @classmethod
    def priority_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("priority must be 1 or greater")
        return v

    def __str__(self) -> str:
        return f"{self.name} ({self.ats.value})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "companies.json"


class CompanyRegistry:
    """
    Loads, validates, and serves the company list from a JSON file.

    All querying goes through this class so the rest of the codebase never
    touches the raw JSON or knows where companies are stored.
    """

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._companies: list[Company] = self._load()

    def _load(self) -> list[Company]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Company data file not found: {self._path}\n"
                "Create it by copying data/companies.json from the repo."
            )
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        # Pydantic validates each entry; raises ValidationError on bad data.
        return [Company.model_validate(entry) for entry in raw]

    # -----------------------------------------------------------------------
    # Basic accessors
    # -----------------------------------------------------------------------

    def all(self) -> list[Company]:
        """Return every company, active or not."""
        return list(self._companies)

    def active(self) -> list[Company]:
        """Return only companies currently being monitored."""
        return [c for c in self._companies if c.active]

    def count(self) -> int:
        return len(self._companies)

    # -----------------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------------

    def find_by_name(self, name: str) -> Company | None:
        """Case-insensitive exact-name lookup."""
        target = name.strip().lower()
        return next((c for c in self._companies if c.name.lower() == target), None)

    def search(self, query: str) -> list[Company]:
        """Return all companies whose name contains `query` (case-insensitive)."""
        q = query.strip().lower()
        return [c for c in self._companies if q in c.name.lower()]

    # -----------------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------------

    def filter_by_ats(self, ats: ATSType) -> list[Company]:
        """Return active companies using a specific ATS platform."""
        return [c for c in self.active() if c.ats == ats]

    def high_priority(self, threshold: int = 1) -> list[Company]:
        """
        Return active companies at or above the given priority tier.

        Priority 1 is the highest. `threshold=1` returns only top-tier
        companies; `threshold=2` includes the next tier as well.
        """
        return [c for c in self.active() if c.priority <= threshold]

    def unknown_ats(self) -> list[Company]:
        """Return active companies whose ATS platform has not been identified yet."""
        return [c for c in self.active() if c.ats == ATSType.UNKNOWN]

    # -----------------------------------------------------------------------
    # Grouped views
    # -----------------------------------------------------------------------

    def grouped_by_ats(self) -> dict[ATSType, list[Company]]:
        """Return active companies grouped by their ATS platform."""
        groups: dict[ATSType, list[Company]] = {}
        for company in self.active():
            groups.setdefault(company.ats, []).append(company)
        return groups

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Company Registry — {self.count()} companies "
            f"({len(self.active())} active)",
            "",
        ]
        for ats, companies in self.grouped_by_ats().items():
            lines.append(f"  {ats.value.capitalize()} ({len(companies)})")
            for c in companies:
                priority_tag = " [HIGH]" if c.priority == 1 else ""
                lines.append(f"    • {c.name}{priority_tag}")
        return "\n".join(lines)
