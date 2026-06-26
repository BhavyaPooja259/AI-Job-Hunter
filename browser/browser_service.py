"""
BrowserService

A thin, reusable wrapper around a Playwright Chromium instance.

Responsibilities:
  - Lifecycle management (start / close)
  - Page navigation with configurable wait strategy
  - Screenshot capture
  - Logging every action so scraper failures are traceable

What this class does NOT do:
  - Parse HTML or extract data (scrapers/ package)
  - Implement retry logic (caller's responsibility)
  - Handle authentication (future dedicated module)
  - Enforce rate limiting (scraper orchestrator's responsibility)

Design note — sync vs async:
  Playwright offers both sync and async APIs. Sync is used here because:
  - It keeps calling code straightforward (no async/await boilerplate)
  - The scraper pipeline runs one company at a time initially
  - Migrating to async later only requires changing this file
"""

import logging
from pathlib import Path

from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from config import settings

logger = logging.getLogger(__name__)


class BrowserService:
    """
    Manages a single Playwright browser session.

    Intended to be used either manually (start/close) or as a context manager:

        with BrowserService() as browser:
            browser.open("https://example.com")
            browser.screenshot("out/page.png")
    """

    def __init__(
        self,
        headless: bool | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        # Explicit constructor arguments override project settings,
        # which is useful for debugging (headless=False) or fast tests.
        self._headless: bool = headless if headless is not None else settings.headless_browser
        self._timeout_ms: int = timeout_ms or (settings.request_timeout_seconds * 1000)

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """
        Launch the Chromium browser and open a blank page.

        Must be called before open() or screenshot().
        Raises RuntimeError if the browser is already running.
        """
        if self._browser is not None:
            raise RuntimeError("BrowserService is already running. Call close() first.")

        logger.info("Starting browser (headless=%s, timeout=%dms)", self._headless, self._timeout_ms)

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self._headless)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(self._timeout_ms)
        except Exception as exc:
            logger.error("Failed to start browser: %s", exc)
            self._cleanup()
            raise

        logger.info("Browser started successfully")

    def open(self, url: str) -> None:
        """
        Navigate to `url` and wait for the network to go idle.

        networkidle is the most reliable wait strategy for dynamic career pages
        that load content via JavaScript after the initial HTML response.

        Raises:
            RuntimeError: if start() has not been called
            PlaywrightTimeoutError: if the page does not load within the timeout
        """
        self._require_started()
        logger.info("Opening: %s", url)

        try:
            self._page.goto(url, wait_until="networkidle")
        except PlaywrightTimeoutError:
            logger.error("Timed out loading %s (timeout=%dms)", url, self._timeout_ms)
            raise
        except Exception as exc:
            logger.error("Failed to open %s: %s", url, exc)
            raise

        logger.info("Page loaded: %s", self._page.title())

    def screenshot(self, path: str | Path) -> Path:
        """
        Capture a full-page screenshot and write it to `path`.

        Parent directories are created automatically.

        Returns the resolved path of the saved screenshot.

        Raises:
            RuntimeError: if start() has not been called
        """
        self._require_started()

        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._page.screenshot(path=str(dest), full_page=True)
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            raise

        logger.info("Screenshot saved: %s", dest)
        return dest

    def close(self) -> None:
        """
        Close the browser and release all Playwright resources.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._browser is None:
            logger.debug("close() called but browser is not running — skipping")
            return

        logger.info("Closing browser")
        self._cleanup()
        logger.info("Browser closed")

    @property
    def page(self) -> "Page":
        """
        The active Playwright Page object.

        Scrapers use this to call query_selector, locator, wait_for_selector,
        and other Playwright APIs directly — without re-implementing them in
        BrowserService. BrowserService manages the lifecycle; scrapers use the page.

        Raises:
            RuntimeError: if start() has not been called
        """
        self._require_started()
        return self._page

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "BrowserService":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _require_started(self) -> None:
        if self._page is None:
            raise RuntimeError(
                "BrowserService is not started. Call start() before open() or screenshot()."
            )

    def _cleanup(self) -> None:
        """Release Playwright resources without raising."""
        try:
            if self._browser:
                self._browser.close()
        except Exception as exc:
            logger.warning("Error closing browser: %s", exc)
        finally:
            self._browser = None
            self._page = None

        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            logger.warning("Error stopping Playwright: %s", exc)
        finally:
            self._playwright = None
