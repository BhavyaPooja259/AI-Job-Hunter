"""
ScraperFactory — smoke test

Verifies that:
  - The factory returns the correct concrete scraper for each ATS type
  - The factory raises ValueError for an unregistered ATS type
  - The Job model produces a consistent fingerprint
  - Runtime registration of a new scraper works correctly

Run from the project root:
    python -m pytest tests/test_scraper_factory.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.constants import ATSType
from scrapers import Job, ScraperFactory
from scrapers.greenhouse_scraper import GreenhouseScraper
from scrapers.lever_scraper import LeverScraper
from scrapers.base_scraper import BaseScraper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_browser():
    """A MagicMock standing in for BrowserService — no browser needed."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_factory_returns_greenhouse_scraper(mock_browser):
    scraper = ScraperFactory.create(ATSType.GREENHOUSE, mock_browser)
    assert isinstance(scraper, GreenhouseScraper)


def test_factory_returns_lever_scraper(mock_browser):
    scraper = ScraperFactory.create(ATSType.LEVER, mock_browser)
    assert isinstance(scraper, LeverScraper)


def test_factory_raises_for_unregistered_ats(mock_browser):
    with pytest.raises(ValueError, match="No scraper registered"):
        ScraperFactory.create(ATSType.ASHBY, mock_browser)


def test_factory_supported_platforms_includes_greenhouse_and_lever():
    platforms = ScraperFactory.supported_platforms()
    assert ATSType.GREENHOUSE in platforms
    assert ATSType.LEVER in platforms


def test_factory_runtime_registration(mock_browser):
    """A new scraper registered at runtime is immediately usable."""

    class FakeAshbyScraper(BaseScraper):
        @property
        def platform_name(self) -> str:
            return "Ashby"

        def scrape(self, careers_url: str) -> list[Job]:
            raise NotImplementedError

    ScraperFactory.register(ATSType.ASHBY, FakeAshbyScraper)
    scraper = ScraperFactory.create(ATSType.ASHBY, mock_browser)
    assert isinstance(scraper, FakeAshbyScraper)
    assert scraper.platform_name == "Ashby"

    # Clean up so other tests are not affected
    del ScraperFactory._REGISTRY[ATSType.ASHBY]


# ---------------------------------------------------------------------------
# Scraper interface tests
# ---------------------------------------------------------------------------

def test_greenhouse_scraper_navigates_via_page(mock_browser):
    """GreenhouseScraper navigates with page.goto (domcontentloaded, not networkidle)."""
    scraper = ScraperFactory.create(ATSType.GREENHOUSE, mock_browser)
    url = "https://boards.greenhouse.io/example"
    # mock page.goto / wait_for_selector / query_selector_all all return MagicMock;
    # the layout detection succeeds (no PlaywrightTimeoutError raised by mock),
    # the container loop iterates over an empty MagicMock iterator → [] returned.
    result = scraper.scrape(url)
    mock_browser.page.goto.assert_called_once_with(
        url, wait_until="domcontentloaded", timeout=25_000
    )
    assert isinstance(result, list)


def test_lever_scraper_raises_not_implemented(mock_browser):
    scraper = ScraperFactory.create(ATSType.LEVER, mock_browser)
    with pytest.raises(NotImplementedError):
        scraper.scrape("https://jobs.lever.co/example")


def test_scrapers_expose_platform_name(mock_browser):
    gh = ScraperFactory.create(ATSType.GREENHOUSE, mock_browser)
    lv = ScraperFactory.create(ATSType.LEVER, mock_browser)
    assert gh.platform_name == "Greenhouse"
    assert lv.platform_name == "Lever"


# ---------------------------------------------------------------------------
# Job model tests
# ---------------------------------------------------------------------------

def test_job_fingerprint_is_stable():
    job = Job(
        company="Stripe",
        title="Backend Engineer",
        job_url="https://stripe.com/jobs/123",
        source_platform=ATSType.GREENHOUSE,
    )
    assert job.fingerprint == job.fingerprint  # deterministic


def test_job_fingerprint_differs_for_different_urls():
    base = dict(company="Stripe", title="Backend Engineer", source_platform=ATSType.GREENHOUSE)
    job_a = Job(**base, job_url="https://stripe.com/jobs/123")
    job_b = Job(**base, job_url="https://stripe.com/jobs/456")
    assert job_a.fingerprint != job_b.fingerprint


def test_job_str_includes_company_and_title():
    job = Job(
        company="Databricks",
        title="Platform Engineer",
        job_url="https://databricks.com/jobs/1",
        location="San Francisco, CA",
        source_platform=ATSType.GREENHOUSE,
    )
    text = str(job)
    assert "Databricks" in text
    assert "Platform Engineer" in text


def test_job_optional_fields_default_to_none():
    job = Job(
        company="Uber",
        title="SDE2",
        job_url="https://uber.com/careers/1",
        source_platform=ATSType.GREENHOUSE,
    )
    assert job.location is None
    assert job.description is None
    assert job.employment_type is None
    assert job.posted_date is None
