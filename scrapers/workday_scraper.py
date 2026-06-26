"""
Workday ATS scraper.

Why Workday is harder than Greenhouse and Lever
------------------------------------------------

  ┌────────────────────────┬──────────────────┬───────────────┬──────────────────┐
  │ Attribute              │ Greenhouse       │ Lever         │ Workday          │
  ├────────────────────────┼──────────────────┼───────────────┼──────────────────┤
  │ Official public API    │ None             │ Yes (v0)      │ None (internal)  │
  │ Unofficial JSON API    │ None             │ N/A           │ CXS (POST)       │
  │ Browser required       │ Always           │ Optional      │ Optional         │
  │ URL scheme             │ 1 consistent     │ 1 consistent  │ 3+ variants      │
  │ Bot protection         │ None             │ None          │ Cloudflare (some)│
  │ JavaScript rendering   │ Minimal          │ Minimal       │ Heavy (React)    │
  │ Pagination             │ All on one page  │ All on one pg │ Paginated (20/pg)│
  │ Selector stability     │ Moderate         │ High          │ Low (hash names) │
  └────────────────────────┴──────────────────┴───────────────┴──────────────────┘

Workday is harder for three specific reasons:

  1. Hash-based CSS class names.
     Classes like "css-19uc56f" and "css-1q2dra3" are generated at build time.
     They change with every Workday platform release, making CSS-class selectors
     unreliable across deployments.  We use `data-automation-id` attributes
     instead — those are authored by Workday engineers and stable across releases.

  2. Paginated results with no single-page URL.
     Greenhouse and Lever show all jobs on one page.  Workday defaults to 20 per
     page and provides no way to request "all" in one call.  This scraper
     paginates the CXS API automatically up to MAX_PAGES.

  3. URL variety and per-company tenants.
     Every Workday customer gets their own subdomain:
       adobe.wd5.myworkdayjobs.com
       walmart.wd504.myworkdayjobs.com
       microsoft.wd1.myworkdayjobs.com
     The "wd<N>" shard number varies.  The "job board" path segment also
     varies per company.  There is no central registry — the URL must be
     configured per company.

Common Workday page structures
-------------------------------
Structure A — Direct Workday board (*.myworkdayjobs.com):
  The standard self-hosted Workday careers site.  Has a public-facing URL
  and loads job data from the CXS JSON API internally.

  Job card structure (HTML fallback):
    <li>
      <h3>
        <a data-automation-id="jobTitle" href="/en-US/{jobboard}/job/...">Title</a>
      </h3>
      <div data-automation-id="locations">Location Text</div>
    </li>

  CXS API structure (primary path):
    POST /wday/cxs/{tenant}/{jobboard}/jobs
    {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}
    → {"total": N, "jobPostings": [{"title": "...", "externalPath": "...", "locationsText": "..."}]}

Structure B — Branded career page (careers.<company>.com):
  Many companies run a branded career site that embeds Workday in the
  background via API calls to <tenant>.wd<N>.myworkdayjobs.com.
  The scraper detects this during HTML navigation by waiting for the
  same `data-automation-id` markers that Workday injects.

Limitations of Workday scraping
--------------------------------
1. Cloudflare protection on some tenants (Walmart wd504).  The CXS API
   bypasses Cloudflare because the POST goes to the CDN endpoint directly,
   but repeated calls may trigger rate limiting.

2. The CXS API is undocumented and not officially supported.  Workday could
   change request format or add authentication requirements at any time.

3. Pagination cap.  This scraper reads up to MAX_PAGES × 20 jobs per company.
   A company with thousands of openings (Walmart has 2000+) will be truncated.
   Raise MAX_PAGES to get more, at the cost of more API calls.

4. No `description` field.  The CXS API returns titles and locations but not
   full job descriptions.  Descriptions require a second request per job (the
   individual job detail URL).  This scraper intentionally skips descriptions
   to keep the scanning fast; descriptions can be fetched later on-demand.

Integration with the existing architecture
------------------------------------------
The WorkdayScraper plugs in identically to Greenhouse and Lever:
  1. It implements BaseScraper (scrape() → list[Job])
  2. It is registered in ScraperFactory._REGISTRY for ATSType.WORKDAY
  3. ScoutAgent calls scraper.scrape(company.careers_url) without knowing
     whether the company uses Workday, Greenhouse, or Lever
  4. The browser it receives from ScoutAgent is used only if the CXS API
     fails — the primary API path requires zero browser time

The scraper exposes `last_path: str` ("api" or "html") after each call,
matching the pattern established by LeverScraper for observability.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from typing import NamedTuple
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from browser import BrowserService
from config.constants import ATSType
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job

logger = logging.getLogger(__name__)

# CXS API pagination: read up to this many pages per company.
# 3 pages = 60 jobs, enough for the matcher to work with.
# Raise to 10+ for a full company scan.
MAX_PAGES = 3
PAGE_SIZE = 20

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_API_TIMEOUT_S = 15
_JS_SETTLE_MS = 5_000       # Workday React app takes longer to mount
_JOB_WAIT_MS = 15_000       # wait for first job card to appear


class _WorkdayConfig(NamedTuple):
    """Parsed components of a *.myworkdayjobs.com URL."""
    tenant: str      # "adobe"
    wd_num: str      # "wd5"
    jobboard: str    # "external_experienced"
    base_url: str    # "https://adobe.wd5.myworkdayjobs.com"
    lang: str        # "en-US"

    @property
    def cxs_api_url(self) -> str:
        return f"{self.base_url}/wday/cxs/{self.tenant}/{self.jobboard}/jobs"

    def full_job_url(self, external_path: str) -> str:
        """Construct the absolute job URL from the relative externalPath."""
        if external_path.startswith("http"):
            return external_path
        if not external_path.startswith("/"):
            external_path = "/" + external_path
        return f"{self.base_url}/{self.lang}/{self.jobboard}{external_path}"


class WorkdayScraper(BaseScraper):
    """
    Workday ATS scraper — CXS JSON API primary, browser HTML fallback.

    Preference order:
      1. Workday CXS internal API   — fast (no browser), paginated JSON
      2. *.myworkdayjobs.com HTML   — browser required, stable automation-id attrs
    """

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)
        self.last_path: str = "unknown"

    @property
    def platform_name(self) -> str:
        return "Workday"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Extract job postings from a Workday career page.

        Tries the CXS JSON API first if the URL is a *.myworkdayjobs.com
        domain.  Falls back to browser-based HTML scraping otherwise.

        Returns an empty list (never raises) if the page is inaccessible
        or produces no jobs.
        """
        self._logger.info("Scraping Workday page: %s", careers_url)
        company = _company_name_from_url(careers_url)

        config = _parse_workday_url(careers_url)
        if config:
            result = self._try_cxs_api(config, company)
            if result is not None:
                self._logger.info(
                    "CXS API succeeded — %d job(s) returned (tenant=%r, board=%r)",
                    len(result), config.tenant, config.jobboard,
                )
                self.last_path = "api"
                return result
            self._logger.info(
                "CXS API unavailable for %r — falling back to HTML", careers_url
            )
        else:
            self._logger.info(
                "URL is not a direct Workday domain — using HTML path directly: %s",
                careers_url,
            )

        self.last_path = "html"
        return self._scrape_html(careers_url, company, config)

    # ------------------------------------------------------------------
    # CXS API path
    # ------------------------------------------------------------------

    def _try_cxs_api(self, config: _WorkdayConfig, company: str) -> list[Job] | None:
        """
        Paginate through the Workday CXS search API.

        Returns:
          list[Job]  — on success (empty if company has no openings)
          None       — if the API is unreachable or returns an error
        """
        all_jobs: list[Job] = []
        offset = 0
        total: int | None = None

        for page_num in range(1, MAX_PAGES + 1):
            self._logger.info(
                "CXS API page %d/%d (offset=%d): %s",
                page_num, MAX_PAGES, offset, config.cxs_api_url,
            )
            data = self._cxs_post(config.cxs_api_url, offset)
            if data is None:
                if offset == 0:
                    return None   # First call failed → API unavailable
                break             # Later page failed → return what we have

            if total is None:
                total = data.get("total", 0)
                self._logger.info("CXS API total jobs on server: %d", total)

            postings = data.get("jobPostings", [])
            for posting in postings:
                job = self._parse_cxs_posting(posting, config, company)
                if job:
                    all_jobs.append(job)
                else:
                    self._logger.debug(
                        "Skipping posting — missing title or path: %r",
                        posting.get("title", ""),
                    )

            offset += PAGE_SIZE
            if total is not None and offset >= total:
                self._logger.info("CXS API: fetched all %d available jobs", total)
                break

        return all_jobs

    def _cxs_post(self, url: str, offset: int) -> dict | None:
        """POST one CXS page request; returns the parsed JSON dict or None."""
        payload = json.dumps({
            "appliedFacets": {},
            "limit": PAGE_SIZE,
            "offset": offset,
            "searchText": "",
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_API_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            self._logger.debug("CXS API HTTP %d at %s", exc.code, url)
            return None
        except urllib.error.URLError as exc:
            self._logger.debug("CXS API network error: %s", exc.reason)
            return None
        except Exception as exc:
            self._logger.debug("CXS API unexpected error: %s", exc)
            return None

    def _parse_cxs_posting(
        self,
        posting: dict,
        config: _WorkdayConfig,
        company: str,
    ) -> Job | None:
        """Convert one CXS API posting dict into a Job."""
        title = (posting.get("title") or "").strip()
        if not title:
            return None

        external_path = (posting.get("externalPath") or "").strip()
        if not external_path:
            return None

        job_url = config.full_job_url(external_path)
        location = (posting.get("locationsText") or "").strip() or None

        return Job(
            company=company,
            title=title,
            job_url=job_url,
            location=location,
            source_platform=ATSType.WORKDAY,
        )

    # ------------------------------------------------------------------
    # HTML browser fallback path
    # ------------------------------------------------------------------

    def _scrape_html(
        self,
        careers_url: str,
        company: str,
        config: _WorkdayConfig | None,
    ) -> list[Job]:
        """Browser-based scraping for Workday pages."""
        if not self._navigate(careers_url):
            return []

        page = self._browser.page
        try:
            page.wait_for_selector(
                '[data-automation-id="jobTitle"]', timeout=_JOB_WAIT_MS
            )
        except PlaywrightTimeoutError:
            self._logger.warning(
                "No Workday jobTitle elements within %dms at %s",
                _JOB_WAIT_MS, careers_url,
            )
            return []

        # Use JavaScript to extract all job cards in one pass.
        # data-automation-id attributes are stable across Workday releases;
        # CSS class names (css-XXXXXXXX) are build-time hashes and change often.
        raw_jobs: list[dict] = page.evaluate("""() => {
            const results = [];
            document.querySelectorAll('[data-automation-id="jobTitle"]').forEach(el => {
                const li = el.closest('li') || el.closest('[role="listitem"]');
                const locEl = li
                    ? (li.querySelector('[data-automation-id="locations"]')
                       || li.querySelector('[data-automation-id="jobDetails"]'))
                    : null;
                results.push({
                    title: el.textContent ? el.textContent.trim() : '',
                    href:  el.getAttribute('href') || '',
                    location: locEl ? locEl.textContent.trim() : null
                });
            });
            return results;
        }""")

        self._logger.info("HTML path found %d job card(s)", len(raw_jobs))
        jobs: list[Job] = []
        skipped = 0

        for raw in raw_jobs:
            title = self.extract_title_from_dict(raw)
            job_url = self.extract_job_url_from_dict(raw, careers_url, config)

            if not title or not job_url:
                self._logger.debug(
                    "Skipping card — title=%r, url=%r", title, job_url
                )
                skipped += 1
                continue

            location = self.extract_location_from_dict(raw)
            jobs.append(Job(
                company=company,
                title=title,
                job_url=job_url,
                location=location,
                source_platform=ATSType.WORKDAY,
            ))

        self._logger.info("HTML done — extracted: %d, skipped: %d", len(jobs), skipped)
        return jobs

    # ------------------------------------------------------------------
    # Helper extraction methods
    # ------------------------------------------------------------------

    def extract_title(self, element) -> str | None:
        """
        Extract job title from a Workday job card element.

        The `data-automation-id="jobTitle"` element IS the title link;
        its text content is the job title.

        Strategy 1 — inner_text() of the element.
        Strategy 2 — text content of inner `<h3>` if the element is a container.
        """
        try:
            text = (element.inner_text() or "").strip()
            if text:
                return text
        except Exception:
            pass

        try:
            child = element.query_selector("h3")
            if child:
                text = (child.inner_text() or "").strip()
                if text:
                    return text
        except Exception:
            pass

        return None

    def extract_title_from_dict(self, raw: dict) -> str | None:
        """Extract title from a JavaScript-evaluated job card dict."""
        text = (raw.get("title") or "").strip()
        return text if text else None

    def extract_location(self, card_element) -> str | None:
        """
        Extract location from a Workday job card element.

        Strategy 1 — [data-automation-id="locations"] child.
        Strategy 2 — [data-automation-id="jobDetails"] child (some layouts).
        Strategy 3 — None (location is optional).
        """
        for sel in [
            '[data-automation-id="locations"]',
            '[data-automation-id="jobDetails"]',
            '[data-automation-id="jobLocation"]',
        ]:
            try:
                child = card_element.query_selector(sel)
                if child:
                    text = (child.inner_text() or "").strip()
                    if text:
                        return text
            except Exception:
                continue
        return None

    def extract_location_from_dict(self, raw: dict) -> str | None:
        """Extract location from a JavaScript-evaluated job card dict."""
        loc = (raw.get("location") or "").strip()
        return loc if loc else None

    def extract_job_url(self, element, base_url: str, config: _WorkdayConfig | None) -> str | None:
        """
        Extract and normalise the job URL from a Workday job card element.

        Workday job links are root-relative on direct boards:
          /en-US/WalmartExternal/job/...
        They must be prepended with the base domain.

        On branded pages (careers.adobe.com) the href may be absolute.
        """
        try:
            href = (element.get_attribute("href") or "").strip()
            return self._normalise_href(href, base_url, config)
        except Exception:
            return None

    def extract_job_url_from_dict(
        self,
        raw: dict,
        base_url: str,
        config: _WorkdayConfig | None,
    ) -> str | None:
        """Extract URL from a JavaScript-evaluated job card dict."""
        href = (raw.get("href") or "").strip()
        return self._normalise_href(href, base_url, config)

    def _normalise_href(
        self,
        href: str,
        base_url: str,
        config: _WorkdayConfig | None,
    ) -> str | None:
        """
        Normalise a raw href to an absolute URL.

        Handles three cases:
          Absolute URL        → returned as-is
          Root-relative /...  → prepended with *.myworkdayjobs.com base
          Empty               → None
        """
        if not href:
            return None

        if href.startswith("http://") or href.startswith("https://"):
            return href

        if href.startswith("/"):
            if config:
                return f"{config.base_url}{href}"
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{href}"

        return None

    # ------------------------------------------------------------------
    # Navigation helper
    # ------------------------------------------------------------------

    def _navigate(self, url: str) -> bool:
        """Navigate using domcontentloaded + JS settle time."""
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


# ---------------------------------------------------------------------------
# URL parsing utilities
# ---------------------------------------------------------------------------

_WD_PATTERN = re.compile(
    r"^(?P<tenant>[a-zA-Z0-9-]+)\."
    r"(?P<wd_num>wd\d+)"
    r"\.myworkdayjobs\.com$",
    re.IGNORECASE,
)


def _parse_workday_url(url: str) -> _WorkdayConfig | None:
    """
    Parse a *.myworkdayjobs.com URL into a _WorkdayConfig.

    Returns None for branded career pages (careers.adobe.com) or any URL
    that does not match the Workday subdomain pattern.

    Examples:
      https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced
          → _WorkdayConfig(tenant="adobe", wd_num="wd5",
                           jobboard="external_experienced",
                           base_url="https://adobe.wd5.myworkdayjobs.com",
                           lang="en-US")

      https://careers.adobe.com/us/en/search-results
          → None  (not a direct Workday board)
    """
    parsed = urlparse(url)
    match = _WD_PATTERN.match(parsed.netloc)
    if not match:
        return None

    tenant = match.group("tenant")
    wd_num = match.group("wd_num")
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Path like /en-US/external_experienced or /en-US/external_experienced/job/...
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) < 2:
        return None

    lang = path_parts[0]       # "en-US"
    jobboard = path_parts[1]   # "external_experienced"

    return _WorkdayConfig(
        tenant=tenant,
        wd_num=wd_num,
        jobboard=jobboard,
        base_url=base_url,
        lang=lang,
    )


def _company_name_from_url(url: str) -> str:
    """
    Derive a readable company name from a Workday careers URL.

      https://adobe.wd5.myworkdayjobs.com/...  → "Adobe"
      https://careers.adobe.com/...             → "Adobe"
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    config = _parse_workday_url(url)
    if config:
        return config.tenant.replace("-", " ").title()

    # Branded career page — strip common prefixes
    for prefix in ("www.", "careers.", "jobs.", "hiring.", "work."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host.split(".")[0].replace("-", " ").title()
