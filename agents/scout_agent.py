"""
ScoutAgent — orchestrates the full job discovery pipeline for all companies.

Responsibility:
  Load companies → pick the right scraper → scrape jobs → persist to database.

ScoutAgent does not implement scraping logic, browser automation, or database
queries. It coordinates existing components and enforces the rule that a single
company failure must never stop the rest of the run.

Browser isolation — one browser per company:
  A single shared browser becomes unreliable across companies because some
  sites trigger redirect chains that close the Playwright page context,
  leaving subsequent navigation calls in a broken state. Creating a fresh
  BrowserService per company gives full isolation: a crash or detection on
  one site has zero impact on the next.

Why a dedicated agent class instead of a script:
  - Encapsulates the orchestration sequence so main.py stays trivial
  - Accepts injected dependencies, making it fully testable without a live browser
  - Returns a structured ScanResult so callers (schedulers, tests, future agents)
    can inspect outcomes programmatically
  - Future agents (RankingAgent, NotificationAgent) chain off ScanResult
    without re-running discovery
"""

import logging
from dataclasses import dataclass, field

from browser import BrowserService
from database import JobRepository
from scrapers import ScraperFactory
from services.company_registry import Company, CompanyRegistry

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Outcome of a single Scout Agent run."""
    companies_checked: int = 0
    jobs_scraped: int = 0
    jobs_saved: int = 0
    duplicates: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def success_count(self) -> int:
        return self.companies_checked - self.failure_count


class ScoutAgent:
    """
    Discovers and persists jobs across all active companies.

    Dependencies are injected so callers can supply mocks in tests.
    When no arguments are given, production defaults are used.

    The `browser` parameter is intended for testing only. In production,
    a fresh BrowserService is created per company for full isolation.

    Usage:
        agent = ScoutAgent()
        result = agent.run()
    """

    def __init__(
        self,
        registry: CompanyRegistry | None = None,
        repo: JobRepository | None = None,
        browser: BrowserService | None = None,
    ) -> None:
        self._registry = registry or CompanyRegistry()
        self._repo = repo or JobRepository()
        self._test_browser = browser  # non-None only in tests

    def run(self) -> ScanResult:
        """
        Execute one full scan across all active companies.

        Opens the database once for the entire run. Each company gets its
        own browser session. Failures are captured per company — the loop
        never aborts.

        Returns a ScanResult summarising the run.
        """
        result = ScanResult()
        companies = self._registry.active()
        result.companies_checked = len(companies)

        logger.info(
            "Scout run starting — %d active company/companies to check",
            result.companies_checked,
        )

        with self._repo:
            for company in companies:
                self._process_company(company, result)

        self._print_summary(result)
        return result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _process_company(self, company: Company, result: ScanResult) -> None:
        """
        Scrape and save one company using an isolated browser session.

        Any exception is caught, logged, and recorded as a failure so the
        outer loop continues uninterrupted.
        """
        logger.info("--- %s [%s] ---", company.name, company.ats.value)

        if self._test_browser is not None:
            # In tests a mock browser is injected; use it directly without
            # managing its lifecycle (the test owns it).
            self._scrape_and_save(company, self._test_browser, result)
        else:
            # Production: fresh browser per company for full isolation.
            with BrowserService(headless=True) as browser:
                self._scrape_and_save(company, browser, result)

    def _scrape_and_save(
        self,
        company: Company,
        browser: BrowserService,
        result: ScanResult,
    ) -> None:
        try:
            scraper = ScraperFactory.create(company.ats, browser)
        except ValueError as exc:
            msg = f"No scraper available: {exc}"
            logger.warning("%s — %s", company.name, msg)
            result.failures.append((company.name, msg))
            return

        try:
            jobs = scraper.scrape(str(company.careers_url))
        except Exception as exc:
            msg = f"Scrape failed: {exc}"
            logger.error("%s — %s", company.name, msg)
            result.failures.append((company.name, msg))
            return

        result.jobs_scraped += len(jobs)
        logger.info("%s — scraped %d job(s)", company.name, len(jobs))

        if not jobs:
            return

        try:
            saved, skipped = self._repo.save_many(jobs)
            result.jobs_saved += saved
            result.duplicates += skipped
            logger.info(
                "%s — saved %d, skipped %d duplicate(s)",
                company.name, saved, skipped,
            )
        except Exception as exc:
            msg = f"Database save failed: {exc}"
            logger.error("%s — %s", company.name, msg)
            result.failures.append((company.name, msg))

    def _print_summary(self, result: ScanResult) -> None:
        print("\n" + "=" * 40)
        print("  Scout Agent — Run Summary")
        print("=" * 40)
        print(f"  Companies Checked : {result.companies_checked}")
        print(f"  Successful        : {result.success_count}")
        print(f"  Jobs Scraped      : {result.jobs_scraped}")
        print(f"  Jobs Saved        : {result.jobs_saved}")
        print(f"  Duplicates        : {result.duplicates}")
        print(f"  Failures          : {result.failure_count}")
        if result.failures:
            print()
            for company_name, reason in result.failures:
                # Truncate long Playwright stack traces in the summary
                short = reason.split("\n")[0]
                print(f"    x {company_name}: {short}")
        print("=" * 40 + "\n")
