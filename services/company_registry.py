"""
Company Registry

Loads company data from data/companies.json and exposes it through a clean,
queryable interface.

Why JSON instead of hardcoded constants:
  - Adding or disabling a company requires no code change and no redeploy
  - Non-engineers can edit the company list directly
  - The file can be version-controlled independently from source code
  - Richer metadata enables AI-based filtering and ranking without
    touching any module that queries companies

Why richer metadata improves AI matching:
  - `supports_remote` lets the ranking agent filter by work arrangement
    preference without parsing free-text descriptions
  - `visa_sponsorship_unknown` surfaces a critical constraint upfront
    rather than after AI generates a cover letter for a role that can't
    proceed
  - `engineering_company` lets the agent weight culture fit —
    a Java backend SDE2 fits Confluent differently than Walmart Global Tech
  - `locations` gives the agent geographic context for salary negotiation
    and relocation decisions
  - `login_required` allows the ScoutAgent to skip companies that require
    authentication before jobs are visible, preventing failed scrapes
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, HttpUrl, field_validator, Field

from config.constants import ATSType


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Company(BaseModel):
    """
    A target company with its ATS platform and rich metadata.

    All boolean fields default to the most permissive value so that
    adding a new company to the JSON with partial data never silently
    excludes it from results — missing flags are treated as "unknown /
    assume yes" until explicitly set.
    """

    # Core identity (required)
    name: str
    careers_url: HttpUrl
    ats: ATSType
    priority: int  # 1 = highest; higher numbers = lower priority

    # Monitoring control
    active: bool = True
    login_required: bool = False  # skip in automated runs if True

    # Geography
    country: str = "US"
    locations: list[str] = Field(default_factory=list)

    # Role fit signals used by the AI ranking agent
    supports_remote: Optional[bool] = None    # None = unknown
    visa_sponsorship_unknown: bool = True     # True = policy not confirmed
    engineering_company: bool = True          # False = primarily business/enterprise

    # Free-text context for AI prompts
    notes: Optional[str] = None

    @field_validator("priority")
    @classmethod
    def priority_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("priority must be 1 or greater")
        return v

    @field_validator("country")
    @classmethod
    def country_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("country must not be empty")
        return v.strip().upper()

    def __str__(self) -> str:
        remote = " (remote)" if self.supports_remote else ""
        return f"{self.name} [{self.country}]{remote} — {self.ats.value}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "companies.json"


class CompanyRegistry:
    """
    Loads, validates, and serves the company catalog from a JSON file.

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
    # Filtering — existing methods (unchanged)
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
    # Filtering — new metadata-aware methods
    # -----------------------------------------------------------------------

    def top_priority(self) -> list[Company]:
        """Return active companies at priority tier 1 only."""
        return [c for c in self.active() if c.priority == 1]

    def requires_login(self) -> list[Company]:
        """
        Return active companies whose career page requires authentication.

        ScoutAgent uses this to skip companies that cannot be scraped without
        credentials in the current session.
        """
        return [c for c in self.active() if c.login_required]

    def by_country(self, country: str) -> list[Company]:
        """
        Return active companies headquartered in `country`.

        Country codes are normalised to uppercase, so "us", "US", and "Us"
        all match "US" entries in the catalog.
        """
        target = country.strip().upper()
        return [c for c in self.active() if c.country == target]

    def engineering_companies(self) -> list[Company]:
        """
        Return active companies classified as primarily engineering organisations.

        Used by the AI ranking agent to weight culture fit — a backend SDE2
        typically gets more engineering autonomy at Confluent than at Visa.
        """
        return [c for c in self.active() if c.engineering_company]

    def supported_companies(self) -> list[Company]:
        """
        Return active companies for which a scraper is registered.

        Imports ScraperFactory lazily to avoid a circular dependency between
        the services and scrapers packages.
        """
        from scrapers import ScraperFactory
        supported_ats = set(ScraperFactory.supported_platforms())
        return [c for c in self.active() if c.ats in supported_ats]

    # -----------------------------------------------------------------------
    # Grouped views
    # -----------------------------------------------------------------------

    def grouped_by_ats(self) -> dict[ATSType, list[Company]]:
        """Return active companies grouped by their ATS platform."""
        groups: dict[ATSType, list[Company]] = {}
        for company in self.active():
            groups.setdefault(company.ats, []).append(company)
        return groups

    def grouped_by_country(self) -> dict[str, list[Company]]:
        """Return active companies grouped by country code."""
        groups: dict[str, list[Company]] = {}
        for company in self.active():
            groups.setdefault(company.country, []).append(company)
        return groups

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Company Registry — {self.count()} companies "
            f"({len(self.active())} active, {len(self.supported_companies())} scrapeable)",
            "",
        ]
        for country, companies in self.grouped_by_country().items():
            lines.append(f"  {country} ({len(companies)})")
            for c in sorted(companies, key=lambda x: x.priority):
                tags = []
                if c.priority == 1:
                    tags.append("HIGH")
                if c.supports_remote:
                    tags.append("remote")
                if c.engineering_company:
                    tags.append("eng")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"    • {c.name} ({c.ats.value}){tag_str}")
        return "\n".join(lines)
