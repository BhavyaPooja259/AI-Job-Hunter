"""
Greenhouse ATS scraper.

Greenhouse powers career pages for: Rippling, Stripe, Uber, Databricks,
Snowflake, Confluent, Atlassian, Airbnb, Intuit, Tekion, and others.

URL formats this scraper handles:
  Embed board:  https://boards.greenhouse.io/embed/job_board?for=<slug>
  Direct board: https://boards.greenhouse.io/<slug>/jobs
  New boards:   https://job-boards.greenhouse.io/<slug>/jobs

DOM structure (confirmed via inspection of live Greenhouse board):

  <td class="cell">
    <a href="https://job-boards.greenhouse.io/<slug>/jobs/<id>?gh_jid=<id>">
      <p class="body body--medium">Job Title</p>
      <p class="body body__secondary body--metadata">Location</p>
    </a>
  </td>

Selectors chosen because:
  - a[href*="/jobs/"]  — targets job links by URL pattern, not fragile class names
  - p.body--medium     — the title paragraph, identified by its modifier class
  - p.body--metadata   — the location paragraph, identified by its modifier class

Using modifier classes (.body--medium, .body--metadata) rather than positional
selectors (p:nth-child(1)) makes the code resilient to Greenhouse adding new
sibling elements inside the link in the future.
"""

import logging
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from browser import BrowserService
from config.constants import ATSType
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job

logger = logging.getLogger(__name__)

# Selector that identifies all job posting links on the page.
# Matches by URL pattern (/jobs/) rather than DOM class to survive layout changes.
_JOB_LINK_SELECTOR = 'a[href*="/jobs/"]'

# Inside each job link, these selectors identify the title and location paragraphs.
_TITLE_SELECTOR = "p.body--medium"
_LOCATION_SELECTOR = "p.body--metadata"

# How long to wait for the job list to appear before giving up (milliseconds).
_WAIT_TIMEOUT_MS = 15_000


class GreenhouseScraper(BaseScraper):

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)

    @property
    def platform_name(self) -> str:
        return "Greenhouse"

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Extract all job postings from a Greenhouse career page.

        Iterates every job link on the current page. Invalid or incomplete
        entries are skipped individually so one bad listing never aborts the run.

        Does not follow pagination — all jobs must be visible on a single page.
        Pagination support is planned for Sprint 4.

        Args:
            careers_url: A Greenhouse-hosted careers URL.

        Returns:
            A list of Job objects, one per valid listing found on the page.
            Returns an empty list if the page loads but contains no jobs.
        """
        self._logger.info("Scraping Greenhouse page: %s", careers_url)

        self._browser.open(careers_url)
        page = self._browser.page

        try:
            page.wait_for_selector(_JOB_LINK_SELECTOR, timeout=_WAIT_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            self._logger.warning(
                "No job links appeared within %dms at %s", _WAIT_TIMEOUT_MS, careers_url
            )
            return []

        links = page.query_selector_all(_JOB_LINK_SELECTOR)
        total_found = len(links)
        self._logger.info("Found %d job link(s) on page", total_found)

        company = _company_name_from_url(careers_url)
        jobs: list[Job] = []
        skipped = 0

        for link in links:
            try:
                job_url = link.get_attribute("href") or ""

                title_el = link.query_selector(_TITLE_SELECTOR)
                location_el = link.query_selector(_LOCATION_SELECTOR)

                title = title_el.inner_text().strip() if title_el else ""
                location = location_el.inner_text().strip() if location_el else None

                if not title or not job_url:
                    self._logger.debug("Skipping entry — missing title or URL")
                    skipped += 1
                    continue

                jobs.append(Job(
                    company=company,
                    title=title,
                    location=location,
                    job_url=job_url,
                    source_platform=ATSType.GREENHOUSE,
                ))

            except Exception as exc:
                self._logger.warning("Skipping entry due to error: %s", exc)
                skipped += 1

        self._logger.info(
            "Extraction complete — extracted: %d, skipped: %d, total links: %d",
            len(jobs), skipped, total_found,
        )
        return jobs


def _company_name_from_url(url: str) -> str:
    """
    Derive a human-readable company name from a Greenhouse careers URL.

    Handles two formats:
      ?for=<slug>      → embed boards (e.g. ?for=rippling → "Rippling")
      /<slug>/jobs     → direct boards (e.g. /stripe/jobs → "Stripe")
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if "for" in params:
        slug = params["for"][0]
    else:
        # Path looks like /embed/job_board or /rippling/jobs — take first real segment
        segments = [s for s in parsed.path.split("/") if s and s not in ("embed", "job_board", "jobs")]
        slug = segments[0] if segments else "unknown"

    return slug.replace("-", " ").title()
