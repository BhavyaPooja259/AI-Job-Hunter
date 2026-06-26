"""
BaseScraper — abstract interface for all ATS adapters.

Every scraper in this project inherits from this class and implements
the `scrape()` method. Nothing else in the codebase depends on a
concrete scraper type; it always works through BaseScraper.

Why inheritance here (not composition or duck typing):
  - The interface is stable and intentionally narrow: one method, one return type
  - ABC enforcement means a missing `scrape()` implementation raises at
    class-definition time, not at runtime during a crawl
  - Shared setup logic (logging, browser access) lives in __init__ once
    rather than being duplicated across every adapter
"""

import logging
from abc import ABC, abstractmethod

from browser import BrowserService
from scrapers.models import Job


class BaseScraper(ABC):
    """
    Abstract base for all ATS-specific scrapers.

    Subclasses must implement:
      - scrape(careers_url) -> list[Job]
      - platform_name (property)
    """

    def __init__(self, browser: BrowserService) -> None:
        self._browser = browser
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Human-readable name of the ATS platform (e.g. 'Greenhouse')."""
        ...

    @abstractmethod
    def scrape(self, careers_url: str) -> list[Job]:
        """
        Fetch all current job postings from `careers_url`.

        Args:
            careers_url: The company's career page URL for this ATS platform.

        Returns:
            A list of Job objects. Returns an empty list if no jobs are found.

        Raises:
            NotImplementedError: Until the subclass provides a real implementation.
            Exception: Any unrecoverable browser or parsing error.
        """
        ...
