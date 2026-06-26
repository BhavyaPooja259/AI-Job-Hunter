"""
Greenhouse ATS scraper — multi-layout, fallback-aware.

Why Greenhouse has multiple layouts
------------------------------------
Greenhouse has shipped at least two generations of their hosted job board UI:

  Layout A — "Classic board" (pre-2019)
    URL pattern:   boards.greenhouse.io/<slug>
    Container:     <div class="opening">
    Link:          <a href="/slug/jobs/ID">
    Children:      <span class="title">Title</span>
                   <span class="location">Location</span>

  Layout B — "Modern board" (2019-present, most companies today)
    URL pattern:   boards.greenhouse.io/<slug>  OR  job-boards.greenhouse.io/<slug>
    Container:     <tr class="job-post">
    Link:          <a href="https://job-boards.greenhouse.io/slug/jobs/ID">
    Children:      <p class="body--medium">Title</p>
                   <p class="body--metadata">Location</p>

  Layout C — "Embed / custom integration" (company-owned career pages)
    A company hosts their own /careers page and embeds the Greenhouse board
    via JavaScript widget or iframe. The rendered HTML may share Layout B's
    CSS classes but wrapped in different container elements.

Why the same selector doesn't work everywhere
----------------------------------------------
Companies have been migrating from classic (A) to modern (B) boards for years,
and not all have finished. A scraper that only knows Layout B will silently
return 0 jobs from a Layout A board. Equally, a company-hosted career page
(Layout C) may redirect to `boards.greenhouse.io` or embed the board in-page —
there is no way to know without loading the page and checking.

Why fallback selectors are better than per-company scrapers
-----------------------------------------------------------
If we wrote a separate scraper for every company, adding Stripe would require
understanding Stripe's page; adding Snowflake would require Snowflake's page.
That scales as O(companies). With a layout-aware single scraper, adding a new
Greenhouse company costs O(1): just add the URL. Only if a company uses a
layout we haven't seen before do we need to extend the scraper.

The fallback chain:
  1.  .job-post  → Layout B (most common today)
  2.  .opening   → Layout A (legacy boards)
  3.  a.job__link  → some custom embed variants
  The first strategy whose selector appears on the page wins.

How this prepares us for hundreds of companies
---------------------------------------------
Our `companies.json` will grow from 19 to hundreds of entries, each with
`"ats": "greenhouse"`. The scraper doesn't need to know which company is
being scraped — it detects the layout at runtime. A URL that 404s or
returns 0 jobs is caught and logged, not silently skipped. When Greenhouse
ships Layout D, we add one entry to `_LAYOUT_STRATEGIES` and ALL existing
companies immediately benefit.
"""

import html as _html_mod
import json
import logging
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from browser import BrowserService
from config.constants import ATSType
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job

_GH_API_BASE = "https://boards-api.greenhouse.io/v1/boards"
_GH_API_TIMEOUT_S = 15
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

logger = logging.getLogger(__name__)

# Layout strategy definitions — tried in order until one succeeds.
# Each strategy specifies:
#   name       — human-readable label printed in logs
#   wait_for   — CSS selector whose presence confirms this layout
#   containers — selector that yields one element per job link
_LAYOUT_STRATEGIES = [
    {
        "name": "Modern board (.job-post)",
        "wait_for": ".job-post",
        "containers": ".job-post a",
    },
    {
        "name": "Classic board (.opening)",
        "wait_for": ".opening",
        "containers": ".opening a",
    },
    {
        "name": "Embed / custom integration (a.job__link)",
        "wait_for": "a.job__link",
        "containers": "a.job__link",
    },
]

# Per-strategy timeout when checking whether the layout selector is present.
# Kept short — we try several strategies in sequence and don't want one
# timeout to block the others.
_STRATEGY_TIMEOUT_MS = 6_000

# Time to wait after navigation for JavaScript to finish rendering the DOM.
_JS_SETTLE_MS = 2_000


class GreenhouseScraper(BaseScraper):

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)

    @property
    def platform_name(self) -> str:
        return "Greenhouse"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Extract all job postings from a Greenhouse career page.

        Preference order:
          1. Greenhouse Job Board API (boards-api.greenhouse.io) with ?content=true
             — returns all jobs with full descriptions in one HTTP request.
             Available only when the careers URL is a standard Greenhouse-hosted board
             (boards.greenhouse.io or job-boards.greenhouse.io).
          2. Browser-based HTML scraping (multi-layout detection)
             — used when the URL is a custom domain or the API is unavailable.
             Returns jobs without descriptions.

        Returns an empty list (never raises) if no jobs are found.
        """
        self._logger.info("Scraping Greenhouse page: %s", careers_url)
        company = _company_name_from_url(careers_url)

        slug = _extract_greenhouse_slug(careers_url)
        if slug:
            result = self._try_greenhouse_api(slug, company)
            if result is not None:
                self._logger.info(
                    "Greenhouse API succeeded for slug=%r — %d job(s) with descriptions",
                    slug, len(result),
                )
                return result
            self._logger.info(
                "Greenhouse API unavailable for slug=%r — falling back to browser",
                slug,
            )

        return self._scrape_browser(careers_url, company)

    # ------------------------------------------------------------------
    # Greenhouse Job Board API path
    # ------------------------------------------------------------------

    def _try_greenhouse_api(self, slug: str, company: str) -> list[Job] | None:
        """
        Call the Greenhouse Job Board API with content=true.

        The Greenhouse Job Board API is a public, undocumented-but-stable
        endpoint that Greenhouse provides for their hosted job boards.  With
        ?content=true, each job includes the full HTML description.

        Why this is better than browser scraping:
          - One HTTP request returns all jobs with descriptions (vs. N browser
            navigations for N jobs)
          - Zero Playwright/Chromium overhead
          - Structured JSON (no DOM parsing)

        Returns:
          list[Job]  — on success (empty if company has no openings)
          None       — if the API is unreachable or returns an error
        """
        url = f"{_GH_API_BASE}/{slug}/jobs?content=true"
        self._logger.info("Trying Greenhouse Job Board API: %s", url)

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=_GH_API_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            self._logger.debug("Greenhouse API HTTP %d for slug=%r", exc.code, slug)
            return None
        except urllib.error.URLError as exc:
            self._logger.debug("Greenhouse API network error for slug=%r: %s", slug, exc.reason)
            return None
        except Exception as exc:
            self._logger.debug("Greenhouse API unexpected error for slug=%r: %s", slug, exc)
            return None

        job_list = data.get("jobs", [])
        self._logger.info("Greenhouse API returned %d posting(s) for slug=%r", len(job_list), slug)

        jobs: list[Job] = []
        for posting in job_list:
            job = self._parse_greenhouse_api_job(posting, company)
            if job:
                jobs.append(job)
        return jobs

    def _parse_greenhouse_api_job(self, posting: dict, company: str) -> Job | None:
        """Parse one Greenhouse Job Board API posting into a Job."""
        title = (posting.get("title") or "").strip()
        if not title:
            return None

        job_url = (posting.get("absolute_url") or "").strip()
        if not job_url:
            return None

        location = (posting.get("location") or {}).get("name") or None

        raw_content = posting.get("content") or ""
        description = _strip_html(raw_content) if raw_content else None

        depts = posting.get("departments") or []
        department = depts[0]["name"] if depts else None

        return Job(
            company=company,
            title=title,
            job_url=job_url,
            location=location,
            description=description,
            department=department,
            source_platform=ATSType.GREENHOUSE,
        )

    # ------------------------------------------------------------------
    # Browser scraping path (original implementation)
    # ------------------------------------------------------------------

    def _scrape_browser(self, careers_url: str, company: str) -> list[Job]:
        """
        Multi-layout browser scraping — original Sprint 9 implementation.

        Used when the Greenhouse API is unavailable or the URL is a
        custom domain.  Returns jobs without descriptions.
        """
        if not self._navigate(careers_url):
            return []

        strategy = self._detect_layout()
        if strategy is None:
            self._logger.warning(
                "No Greenhouse job listing layout detected at %s — "
                "the page may require login, be blocked by a CAPTCHA, "
                "or not use a supported Greenhouse layout.",
                careers_url,
            )
            return []

        containers = self._browser.page.query_selector_all(strategy["containers"])
        self._logger.info(
            "Strategy '%s' found %d container(s)",
            strategy["name"], len(containers),
        )

        jobs: list[Job] = []
        skipped = 0

        for container in containers:
            try:
                title    = self.extract_title(container)
                job_url  = self.extract_job_url(container, careers_url)

                if not title or not job_url:
                    self._logger.debug(
                        "Skipping container — title=%r, url=%r", title, job_url
                    )
                    skipped += 1
                    continue

                location = self.extract_location(container)

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
            "Done — extracted: %d, skipped: %d",
            len(jobs), skipped,
        )
        return jobs

    # ------------------------------------------------------------------
    # Helper extraction methods
    # ------------------------------------------------------------------

    def extract_title(self, container) -> str | None:
        """
        Extract the job title from a container element.

        Strategy 1 — p.body--medium child (Greenhouse Layout B):
            <a><p class="body body--medium">Backend Engineer</p>...</a>

        Strategy 2 — .title span child (Greenhouse Layout A):
            <div class="opening"><a href="...">Title</a><span class="location">...</span></div>
            In this case the <a> itself is the title text.

        Strategy 3 — first non-empty line of the element's inner text.
            Fallback for embed/custom layouts where neither class is present.
        """
        # Strategy 1: modern board
        child = container.query_selector("p.body--medium")
        if child:
            text = child.inner_text().strip()
            if text and text != "Create a Job Alert":
                return text

        # Strategy 2: classic board — the link itself is the title
        child = container.query_selector(".title")
        if child:
            text = child.inner_text().strip()
            if text:
                return text

        # Strategy 3: first line of inner text
        raw = (container.inner_text() or "").strip()
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        return lines[0] if lines else None

    def extract_location(self, container) -> str | None:
        """
        Extract the job location from a container element.

        Strategy 1 — p.body--metadata child (Layout B):
            <p class="body body--metadata">Remote, United States</p>

        Strategy 2 — .location child (Layout A):
            <span class="location">San Francisco, CA</span>

        Strategy 3 — second non-empty line of inner text.
            Returns None rather than raising if nothing is found — location
            is optional metadata and a missing location should not prevent
            the job from being saved.
        """
        # Strategy 1: modern board
        child = container.query_selector("p.body--metadata")
        if child:
            text = child.inner_text().strip()
            return text if text else None

        # Strategy 2: classic board
        child = container.query_selector(".location")
        if child:
            text = child.inner_text().strip()
            return text if text else None

        # Strategy 3: second line of inner text
        raw = (container.inner_text() or "").strip()
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        return lines[1] if len(lines) >= 2 else None

    def extract_job_url(self, container, base_url: str) -> str | None:
        """
        Extract and normalise the job posting URL from a container element.

        Handles three cases:
          Absolute URL  — returned as-is
                          https://job-boards.greenhouse.io/tekion/jobs/123
          Root-relative — prepended with the scheme and host of the base URL
                          /job-openings/job?id=123 → https://tekion.com/job-openings/job?id=123
          Relative      — treated as invalid; caller skips this entry
        """
        href = container.get_attribute("href") or ""

        if not href:
            return None

        if href.startswith("http://") or href.startswith("https://"):
            return href

        if href.startswith("/"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{href}"

        self._logger.debug("Ignoring unrecognised URL form: %r", href)
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _navigate(self, url: str) -> bool:
        """
        Navigate to `url` using domcontentloaded (not networkidle).

        networkidle waits until all network requests have settled, which
        can take 30+ seconds on JS-heavy career pages or simply never
        complete if a tracking pixel keeps firing.  domcontentloaded is
        fast and reliable — it signals that the initial HTML is parsed and
        JavaScript has started executing.  We then add a small fixed wait
        (_JS_SETTLE_MS) to give frameworks time to render their components.

        Returns True if navigation succeeded, False otherwise.
        """
        try:
            self._browser.page.goto(url, wait_until="domcontentloaded", timeout=25_000)
            self._browser.page.wait_for_timeout(_JS_SETTLE_MS)
            return True
        except PlaywrightTimeoutError:
            self._logger.warning("Navigation timed out: %s", url)
            return False
        except Exception as exc:
            self._logger.warning("Navigation failed (%s): %s", type(exc).__name__, exc)
            return False

    def _detect_layout(self) -> dict | None:
        """
        Try each layout strategy in order and return the first one that
        finds its sentinel element within _STRATEGY_TIMEOUT_MS.

        Logs which strategy succeeded (or that all failed).
        """
        page = self._browser.page
        for strategy in _LAYOUT_STRATEGIES:
            try:
                page.wait_for_selector(strategy["wait_for"], timeout=_STRATEGY_TIMEOUT_MS)
                self._logger.info(
                    "Layout detected — strategy: '%s'", strategy["name"]
                )
                return strategy
            except PlaywrightTimeoutError:
                self._logger.debug(
                    "Strategy '%s' — selector '%s' not found within %dms",
                    strategy["name"], strategy["wait_for"], _STRATEGY_TIMEOUT_MS,
                )

        self._logger.warning("All layout strategies exhausted — no layout detected")
        return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_greenhouse_slug(url: str) -> str | None:
    """
    Extract the Greenhouse job board slug from a URL.

    Handles standard Greenhouse-hosted boards only.  Custom-domain boards
    (e.g. ats.rippling.com) cannot be resolved to a slug without additional
    configuration.

    Examples:
      https://boards.greenhouse.io/tekion               → "tekion"
      https://boards.greenhouse.io/tekion/jobs          → "tekion"
      https://job-boards.greenhouse.io/razorpay...      → "razorpay..."
      https://boards.greenhouse.io/embed/job_board?for=X → "X"
      https://ats.rippling.com/rippling/jobs            → None
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    params = parse_qs(parsed.query)
    if "for" in params:
        return params["for"][0]

    if host in ("boards.greenhouse.io", "job-boards.greenhouse.io"):
        skip = {"jobs", "embed", "job_board"}
        parts = [p for p in parsed.path.split("/") if p and p not in skip]
        return parts[0] if parts else None

    return None


def _strip_html(text: str) -> str:
    """
    Convert HTML-encoded Greenhouse job content to plain text.

    The Greenhouse API returns content as HTML-entity-encoded strings
    (e.g. &lt;p&gt; instead of <p>).  This function:
      1. Decodes HTML entities (html.unescape)
      2. Removes HTML tags via regex
      3. Collapses whitespace
    """
    text = _html_mod.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _company_name_from_url(url: str) -> str:
    """
    Derive a human-readable company name from a Greenhouse careers URL.

      boards.greenhouse.io/tekion    → "Tekion"
      ?for=rippling                  → "Rippling"
      ats.rippling.com/rippling/jobs → "Rippling"
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if "for" in params:
        slug = params["for"][0]
    else:
        segments = [s for s in parsed.path.split("/")
                    if s and s not in ("embed", "job_board", "jobs")]
        slug = segments[0] if segments else "unknown"

    return slug.replace("-", " ").title()
