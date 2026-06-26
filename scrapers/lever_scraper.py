"""
Lever ATS scraper.

Lever powers career pages for: Razorpay and others.

Typical Lever URL pattern:
    https://jobs.lever.co/<company-slug>

Lever is notable for offering a public JSON API alongside its HTML pages:
    https://api.lever.co/v0/postings/<company-slug>?mode=json

The implementation (Sprint 4) will prefer the JSON API where available
to avoid browser overhead, and fall back to HTML scraping otherwise.
"""

from browser import BrowserService
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job


class LeverScraper(BaseScraper):

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)

    @property
    def platform_name(self) -> str:
        return "Lever"

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Scrape all job listings from a Lever-hosted careers page.

        Not yet implemented. Will:
          1. Attempt to use the Lever public JSON API (preferred — no browser needed)
          2. Fall back to browser-based HTML scraping if the API is unavailable
          3. Return a list of Job objects

        Raises:
            NotImplementedError: Until Sprint 4 parsing is implemented.
        """
        self._logger.info("LeverScraper.scrape() called for: %s", careers_url)
        raise NotImplementedError(
            "LeverScraper.scrape() is not yet implemented. "
            "HTML parsing is planned for Sprint 4."
        )
