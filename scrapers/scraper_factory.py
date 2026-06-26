"""
ScraperFactory

Maps ATSType values to their concrete scraper classes and constructs
the correct instance on demand.

Why a factory instead of a direct import + instantiation at the call site:
  - The orchestrator knows the ATS type (from CompanyRegistry) but should not
    need to know which class handles it — that's the factory's job
  - Adding a new ATS adapter requires one line here, nowhere else
  - The registry dict makes the full list of supported platforms visible
    in one place

How to add a new ATS platform:
  1. Create scrapers/<platform>_scraper.py inheriting BaseScraper
  2. Add one entry to ScraperFactory._REGISTRY below
  3. Done — the factory, orchestrator, and tests all pick it up automatically
"""

import logging

from browser import BrowserService
from config.constants import ATSType
from scrapers.base_scraper import BaseScraper
from scrapers.greenhouse_scraper import GreenhouseScraper
from scrapers.lever_scraper import LeverScraper
from scrapers.workday_scraper import WorkdayScraper

logger = logging.getLogger(__name__)


class ScraperFactory:
    """
    Returns the correct BaseScraper subclass for a given ATSType.

    Usage:
        scraper = ScraperFactory.create(ATSType.GREENHOUSE, browser)
        jobs = scraper.scrape(company.careers_url)
    """

    # Registry maps each ATS type to its implementation class.
    # The type annotation makes mypy enforce that every value is a subclass
    # of BaseScraper, catching registration mistakes at lint time.
    _REGISTRY: dict[ATSType, type[BaseScraper]] = {
        ATSType.GREENHOUSE: GreenhouseScraper,
        ATSType.LEVER: LeverScraper,
        ATSType.WORKDAY: WorkdayScraper,
    }

    @classmethod
    def create(cls, ats: ATSType, browser: BrowserService) -> BaseScraper:
        """
        Instantiate and return the scraper for `ats`.

        Args:
            ats: The ATS platform to scrape.
            browser: A started (or unstarted) BrowserService instance.
                     The scraper stores a reference; the caller owns the lifecycle.

        Raises:
            ValueError: If no scraper is registered for the given ATS type.
        """
        scraper_class = cls._REGISTRY.get(ats)

        if scraper_class is None:
            supported = [a.value for a in cls._REGISTRY]
            raise ValueError(
                f"No scraper registered for ATS '{ats.value}'. "
                f"Supported platforms: {supported}"
            )

        logger.debug("Creating %s for ATS '%s'", scraper_class.__name__, ats.value)
        return scraper_class(browser)

    @classmethod
    def register(cls, ats: ATSType, scraper_class: type[BaseScraper]) -> None:
        """
        Register a new scraper at runtime.

        Useful for plugin-style extension or for swapping implementations
        in tests without editing this file.

        Example:
            ScraperFactory.register(ATSType.ASHBY, AshbyScraper)
        """
        logger.info("Registering scraper %s for ATS '%s'", scraper_class.__name__, ats.value)
        cls._REGISTRY[ats] = scraper_class

    @classmethod
    def supported_platforms(cls) -> list[ATSType]:
        """Return the list of ATS platforms that have a registered scraper."""
        return list(cls._REGISTRY.keys())
