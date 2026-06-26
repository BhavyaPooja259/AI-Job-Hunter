"""
scrapers package

Public surface of the scraping layer:

    from scrapers import Job, BaseScraper, ScraperFactory
"""

from scrapers.models import Job
from scrapers.base_scraper import BaseScraper
from scrapers.scraper_factory import ScraperFactory

__all__ = ["Job", "BaseScraper", "ScraperFactory"]
