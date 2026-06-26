"""
CompanyRegistry — unit tests for the expanded company catalog.

Tests cover model validation, all filter methods, and edge cases
introduced by the new metadata fields.

Run from the project root:
    python -m pytest tests/test_company_registry.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.constants import ATSType
from services.company_registry import Company, CompanyRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_COMPANY = {
    "name": "TestCo",
    "careers_url": "https://testco.com/jobs",
    "ats": "greenhouse",
    "priority": 1,
}

FULL_COMPANY = {
    "name": "FullCo",
    "careers_url": "https://fullco.com/jobs",
    "ats": "lever",
    "priority": 2,
    "active": True,
    "country": "IN",
    "locations": ["Bangalore, IN", "Remote"],
    "login_required": True,
    "supports_remote": True,
    "visa_sponsorship_unknown": False,
    "engineering_company": False,
    "notes": "Test company with all fields set.",
}


def _registry_from(companies: list[dict]) -> CompanyRegistry:
    """Write companies to a temp JSON file and return a registry backed by it."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(companies, f)
        f.flush()  # ensure bytes reach disk before CompanyRegistry reads the file
        return CompanyRegistry(path=Path(f.name))


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

def test_company_loads_with_minimal_fields():
    c = Company.model_validate(MINIMAL_COMPANY)
    assert c.name == "TestCo"
    assert c.ats == ATSType.GREENHOUSE
    assert c.priority == 1
    assert c.active is True
    assert c.login_required is False
    assert c.engineering_company is True
    assert c.visa_sponsorship_unknown is True
    assert c.supports_remote is None
    assert c.locations == []


def test_company_loads_with_all_fields():
    c = Company.model_validate(FULL_COMPANY)
    assert c.country == "IN"
    assert c.login_required is True
    assert c.supports_remote is True
    assert c.visa_sponsorship_unknown is False
    assert c.engineering_company is False
    assert len(c.locations) == 2


def test_priority_must_be_positive():
    with pytest.raises(Exception):
        Company.model_validate({**MINIMAL_COMPANY, "priority": 0})


def test_country_is_normalised_to_uppercase():
    c = Company.model_validate({**MINIMAL_COMPANY, "country": "us"})
    assert c.country == "US"


def test_country_cannot_be_empty():
    with pytest.raises(Exception):
        Company.model_validate({**MINIMAL_COMPANY, "country": "  "})


def test_invalid_ats_raises():
    with pytest.raises(Exception):
        Company.model_validate({**MINIMAL_COMPANY, "ats": "nonsense"})


def test_company_str_includes_name_and_country():
    c = Company.model_validate(FULL_COMPANY)
    result = str(c)
    assert "FullCo" in result
    assert "IN" in result


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def test_registry_loads_real_companies():
    registry = CompanyRegistry()
    assert registry.count() >= 5


def test_registry_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        CompanyRegistry(path=Path("/nonexistent/companies.json"))


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------

def test_all_returns_inactive_companies():
    companies = [
        {**MINIMAL_COMPANY, "name": "Active Co", "active": True},
        {**MINIMAL_COMPANY, "name": "Inactive Co", "active": False},
    ]
    registry = _registry_from(companies)
    assert registry.count() == 2
    assert len(registry.all()) == 2


def test_active_excludes_inactive():
    companies = [
        {**MINIMAL_COMPANY, "name": "Active Co", "active": True},
        {**MINIMAL_COMPANY, "name": "Inactive Co", "active": False},
    ]
    registry = _registry_from(companies)
    active = registry.active()
    assert len(active) == 1
    assert active[0].name == "Active Co"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def test_find_by_name_case_insensitive():
    registry = CompanyRegistry()
    assert registry.find_by_name("stripe") is not None
    assert registry.find_by_name("STRIPE") is not None
    assert registry.find_by_name("Stripe") is not None


def test_find_by_name_returns_none_for_unknown():
    registry = CompanyRegistry()
    assert registry.find_by_name("nonexistent_xyz") is None


def test_search_returns_partial_matches():
    registry = CompanyRegistry()
    results = registry.search("soft")  # matches "Microsoft"
    assert any(c.name == "Microsoft" for c in results)


# ---------------------------------------------------------------------------
# top_priority
# ---------------------------------------------------------------------------

def test_top_priority_returns_only_priority_1():
    registry = CompanyRegistry()
    top = registry.top_priority()
    assert all(c.priority == 1 for c in top)
    assert len(top) > 0


def test_top_priority_excludes_priority_2():
    companies = [
        {**MINIMAL_COMPANY, "name": "P1", "priority": 1},
        {**MINIMAL_COMPANY, "name": "P2", "priority": 2},
    ]
    registry = _registry_from(companies)
    top = registry.top_priority()
    assert len(top) == 1
    assert top[0].name == "P1"


# ---------------------------------------------------------------------------
# requires_login
# ---------------------------------------------------------------------------

def test_requires_login_returns_only_login_companies():
    companies = [
        {**MINIMAL_COMPANY, "name": "NeedsLogin", "login_required": True},
        {**MINIMAL_COMPANY, "name": "NoLogin", "login_required": False},
    ]
    registry = _registry_from(companies)
    result = registry.requires_login()
    assert len(result) == 1
    assert result[0].name == "NeedsLogin"


def test_requires_login_empty_for_real_catalog():
    # No company in the real catalog should require login
    registry = CompanyRegistry()
    assert registry.requires_login() == []


# ---------------------------------------------------------------------------
# by_country
# ---------------------------------------------------------------------------

def test_by_country_us_returns_us_companies():
    registry = CompanyRegistry()
    us = registry.by_country("US")
    assert all(c.country == "US" for c in us)
    assert len(us) > 0


def test_by_country_in_returns_india_companies():
    registry = CompanyRegistry()
    india = registry.by_country("IN")
    assert all(c.country == "IN" for c in india)
    names = [c.name for c in india]
    assert "PhonePe" in names
    assert "Razorpay" in names
    assert "Groww" in names


def test_by_country_is_case_insensitive():
    registry = CompanyRegistry()
    assert registry.by_country("us") == registry.by_country("US")


def test_by_country_unknown_returns_empty():
    registry = CompanyRegistry()
    assert registry.by_country("ZZ") == []


# ---------------------------------------------------------------------------
# engineering_companies
# ---------------------------------------------------------------------------

def test_engineering_companies_excludes_non_engineering():
    companies = [
        {**MINIMAL_COMPANY, "name": "TechCo", "engineering_company": True},
        {**MINIMAL_COMPANY, "name": "BizCo", "engineering_company": False},
    ]
    registry = _registry_from(companies)
    result = registry.engineering_companies()
    assert len(result) == 1
    assert result[0].name == "TechCo"


def test_engineering_companies_from_real_catalog():
    registry = CompanyRegistry()
    eng = registry.engineering_companies()
    names = [c.name for c in eng]
    # These are known engineering companies in the catalog
    assert "Stripe" in names
    assert "Databricks" in names
    assert "Confluent" in names
    # Adobe and Walmart are not engineering companies
    assert "Adobe" not in names
    assert "Walmart Global Tech" not in names


# ---------------------------------------------------------------------------
# supported_companies
# ---------------------------------------------------------------------------

def test_supported_companies_only_includes_registered_ats():
    registry = CompanyRegistry()
    supported = registry.supported_companies()
    # All returned companies must have a registered ATS scraper
    from scrapers import ScraperFactory
    valid_ats = set(ScraperFactory.supported_platforms())
    assert all(c.ats in valid_ats for c in supported)


def test_microsoft_not_in_supported_companies():
    registry = CompanyRegistry()
    supported = registry.supported_companies()
    names = [c.name for c in supported]
    assert "Microsoft" not in names  # Workday has no scraper yet


# ---------------------------------------------------------------------------
# grouped_by_country
# ---------------------------------------------------------------------------

def test_grouped_by_country_keys():
    registry = CompanyRegistry()
    groups = registry.grouped_by_country()
    assert "US" in groups
    assert "IN" in groups


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_includes_company_count():
    registry = CompanyRegistry()
    s = registry.summary()
    assert "Company Registry" in s
    assert "US" in s
    assert "IN" in s
