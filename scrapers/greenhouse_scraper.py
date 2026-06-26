"""
Greenhouse ATS scraper.

Greenhouse powers career pages for: Rippling, Stripe, Uber, Databricks,
Snowflake, Confluent, Atlassian, Airbnb, Intuit, Tekion, and others.

Typical Greenhouse URL pattern:
    https://boards.greenhouse.io/<company-slug>/jobs
    https://ats.rippling.com/<company>/jobs

HTML parsing will be implemented in Sprint 4.
"""

from browser import BrowserService
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job


class GreenhouseScraper(BaseScraper):

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)

    @property
    def platform_name(self) -> str:
        return "Greenhouse"

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Scrape all job listings from a Greenhouse-hosted careers page.

        Not yet implemented. Will:
          1. Open the careers URL in the headless browser
          2. Wait for the job list container to render
          3. Extract each job title, location, and detail URL
          4. Optionally open each detail page for the full description
          5. Return a list of Job objects

        Raises:
            NotImplementedError: Until Sprint 4 parsing is implemented.
        """
        self._logger.info("GreenhouseScraper.scrape() called for: %s", careers_url)
        raise NotImplementedError(
            "GreenhouseScraper.scrape() is not yet implemented. "
            "HTML parsing is planned for Sprint 4."
        )
