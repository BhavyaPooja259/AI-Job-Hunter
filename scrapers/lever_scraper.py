"""
Lever ATS scraper.

Why Lever differs from Greenhouse
----------------------------------
Greenhouse is purely HTML-first: there is no officially documented public
JSON endpoint, and every scraper must load the page in a browser and parse
DOM nodes.  Lever was designed with a developer-friendly job board API from
the beginning, exposing all job data as structured JSON at a predictable URL.

Lever vs Greenhouse — comparison table:

  ┌─────────────────────┬──────────────────────────────┬──────────────────────────────┐
  │ Attribute           │ Greenhouse                   │ Lever                        │
  ├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Primary interface   │ HTML board (browser required)│ Public JSON API (HTTP only)  │
  │ API URL pattern     │ None                         │ api.lever.co/v0/postings/<s> │
  │ Data format         │ CSS-class-based DOM          │ Structured JSON              │
  │ Title               │ p.body--medium               │ posting["text"]              │
  │ Location            │ p.body--metadata             │ posting["categories"]["loc"] │
  │ Job URL             │ href on <a> link             │ posting["hostedUrl"]         │
  │ Dept / team         │ Not in page HTML             │ posting["categories"]["team"]│
  │ Workplace type      │ Not available                │ posting["workplaceType"]     │
  │ Salary range        │ Not available                │ posting["salaryRange"]       │
  │ Browser required?   │ Always                       │ Only if API blocked          │
  │ Parsing brittleness │ High (DOM changes break it)  │ Low (typed JSON schema)      │
  └─────────────────────┴──────────────────────────────┴──────────────────────────────┘

Why the JSON API is preferred over HTML scraping
-------------------------------------------------
1. No browser overhead — a plain HTTP request takes ~200ms; launching Playwright
   Chromium takes ~2s.  For 50 Lever companies that is the difference between
   10 seconds and 100 seconds for a full scan.

2. Zero DOM brittleness — Lever can redesign their HTML board without changing
   the JSON schema.  DOM-based scrapers silently break when CSS classes are
   renamed or elements are reordered.

3. Richer data — the API exposes fields that are absent from the HTML: salary
   range, workplace type, department, country, and the job's Lever ID.  These
   are all useful for the matcher layer (salary filtering, remote filtering, etc.)

4. Deterministic encoding — the JSON payload is UTF-8 text with no HTML
   character entities, NBSP sequences, or invisible Unicode separators.  HTML
   inner text requires extra cleaning.

Fallback strategy
-----------------
Some companies restrict the API (HTTP 403) or have disabled it.  In those cases
the scraper transparently falls back to browser-based HTML extraction using the
same `.posting` DOM structure present on every jobs.lever.co page.

The public API URL is: https://api.lever.co/v0/postings/<company-slug>?mode=json

Slug derivation:
  https://jobs.lever.co/employ               → "employ"
  https://jobs.lever.co/employ/UUID          → "employ"
  https://api.lever.co/v0/postings/employ    → "employ"
  https://razorpay.com/jobs                  → None  (non-Lever domain; skip API)

URL patterns handled:
  Standard board:   https://jobs.lever.co/<slug>
  Direct API:       https://api.lever.co/v0/postings/<slug>?mode=json
  Custom careers:   https://<company>.com/careers  (HTML fallback only)
"""

import json
import logging
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from browser import BrowserService
from config.constants import ATSType
from scrapers.base_scraper import BaseScraper
from scrapers.models import Job

logger = logging.getLogger(__name__)

_API_BASE = "https://api.lever.co/v0/postings"
_API_TIMEOUT_S = 15
_JS_SETTLE_MS = 2_000
_POSTING_WAIT_MS = 10_000

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


class LeverScraper(BaseScraper):
    """
    Lever ATS scraper with JSON API primary + HTML browser fallback.

    Preference order:
      1. Lever public JSON API   — fast, structured, no browser
      2. jobs.lever.co HTML      — browser required, DOM-based
    """

    def __init__(self, browser: BrowserService) -> None:
        super().__init__(browser)
        self.last_path: str = "unknown"   # "api" or "html" — set after each scrape()

    @property
    def platform_name(self) -> str:
        return "Lever"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape(self, careers_url: str) -> list[Job]:
        """
        Extract all job postings from a Lever-hosted careers page.

        Tries the public JSON API first.  Falls back to HTML parsing via
        Playwright if the API is blocked or unavailable.

        Returns an empty list (never raises) if no jobs are found by either
        method.
        """
        self._logger.info("Scraping Lever page: %s", careers_url)
        company = _company_name_from_url(careers_url)

        # ── 1. JSON API path ──────────────────────────────────────────
        slug = _extract_lever_slug(careers_url)
        if slug:
            result = self._try_api(slug, company)
            if result is not None:
                self._logger.info(
                    "JSON API succeeded for slug=%r — %d job(s) returned",
                    slug, len(result),
                )
                self.last_path = "api"
                return result
            self._logger.info(
                "JSON API unavailable for slug=%r — falling back to HTML",
                slug,
            )
        else:
            self._logger.info(
                "No Lever slug extractable from %r — using HTML fallback directly",
                careers_url,
            )

        # ── 2. HTML fallback ──────────────────────────────────────────
        self.last_path = "html"
        return self._scrape_html(careers_url, company)

    # ------------------------------------------------------------------
    # JSON API path
    # ------------------------------------------------------------------

    def _try_api(self, slug: str, company: str) -> list[Job] | None:
        """
        Call the Lever public JSON API for `slug`.

        Returns:
          list[Job]  — on success (may be empty if company has no open roles)
          None       — if the API is blocked, returns an error, or network fails
        """
        url = f"{_API_BASE}/{slug}?mode=json"
        self._logger.info("Trying Lever JSON API: %s", url)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_API_TIMEOUT_S) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            self._logger.debug("API HTTP %d for slug=%r: %s", exc.code, slug, exc.reason)
            return None
        except urllib.error.URLError as exc:
            self._logger.debug("API network error for slug=%r: %s", slug, exc.reason)
            return None
        except Exception as exc:
            self._logger.debug("API unexpected error for slug=%r: %s", slug, exc)
            return None

        if not isinstance(data, list):
            self._logger.debug("API returned non-list response for slug=%r: %r", slug, data)
            return None

        self._logger.info("API returned %d posting(s) for slug=%r", len(data), slug)

        jobs: list[Job] = []
        for posting in data:
            job = self._parse_api_posting(posting, company)
            if job:
                jobs.append(job)
            else:
                self._logger.debug("Skipping API posting — missing title or URL: %r", posting.get("id"))

        return jobs

    def _parse_api_posting(self, posting: dict, company: str) -> Job | None:
        """
        Parse one Lever API posting dict into a Job.

        Description fields available from the Lever API:
          descriptionPlain  — main job description, plain text, no HTML
          lists             — structured sections e.g. "What You'll Do",
                              "What We're Looking For" — HTML content
          additionalPlain   — boilerplate footer (EEO statements, legal text)

        Strategy for requirements:
          Scan `lists` for a section whose heading contains a
          requirement-related keyword.  Use that section's content (stripped
          of HTML) as `requirements`.  This gives a clean signal for the
          matcher without mixing the full description with the EEO footer.
        """
        title = (posting.get("text") or "").strip()
        if not title:
            return None

        hosted_url = (posting.get("hostedUrl") or "").strip()
        if not hosted_url:
            return None

        cats = posting.get("categories") or {}
        location = self.extract_location_from_api(cats)
        department = _extract_lever_department(cats)

        description = (posting.get("descriptionPlain") or "").strip() or None
        requirements = _extract_lever_requirements(posting.get("lists") or [])

        return Job(
            company=company,
            title=title,
            job_url=hosted_url,
            location=location,
            description=description,
            requirements=requirements,
            department=department,
            source_platform=ATSType.LEVER,
        )

    # ------------------------------------------------------------------
    # HTML fallback path
    # ------------------------------------------------------------------

    def _scrape_html(self, careers_url: str, company: str) -> list[Job]:
        """Browser-based scraping of a jobs.lever.co board."""
        if not self._navigate(careers_url):
            return []

        page = self._browser.page
        try:
            page.wait_for_selector(".posting", timeout=_POSTING_WAIT_MS)
        except PlaywrightTimeoutError:
            self._logger.warning(
                "No .posting elements within %dms at %s",
                _POSTING_WAIT_MS, careers_url,
            )
            return []

        postings = page.query_selector_all(".posting")
        self._logger.info("HTML path: found %d .posting element(s)", len(postings))

        jobs: list[Job] = []
        skipped = 0
        for posting in postings:
            try:
                title   = self.extract_title(posting)
                job_url = self.extract_job_url(posting, careers_url)

                if not title or not job_url:
                    self._logger.debug(
                        "Skipping posting — title=%r, url=%r", title, job_url
                    )
                    skipped += 1
                    continue

                location = self.extract_location(posting)
                jobs.append(Job(
                    company=company,
                    title=title,
                    job_url=job_url,
                    location=location,
                    source_platform=ATSType.LEVER,
                ))
            except Exception as exc:
                self._logger.warning("Skipping posting due to error: %s", exc)
                skipped += 1

        self._logger.info("HTML done — extracted: %d, skipped: %d", len(jobs), skipped)
        return jobs

    # ------------------------------------------------------------------
    # Helper extraction methods (used by both paths and tests)
    # ------------------------------------------------------------------

    def extract_title(self, posting_el) -> str | None:
        """
        Extract job title from a .posting HTML element.

        Strategy 1 — h5[data-qa="posting-name"]:
            <h5 data-qa="posting-name">AI Engineer - India</h5>

        Strategy 2 — h5.posting-name (older boards):
            <h5 class="posting-name">Backend Engineer</h5>

        Strategy 3 — First non-empty line of inner text.
        """
        # Strategy 1: modern boards (data-qa attribute)
        el = posting_el.query_selector('h5[data-qa="posting-name"]')
        if el:
            text = el.inner_text().strip()
            if text:
                return text

        # Strategy 2: older boards
        el = posting_el.query_selector("h5.posting-name")
        if el:
            text = el.inner_text().strip()
            if text:
                return text

        # Strategy 3: fallback
        raw = (posting_el.inner_text() or "").strip()
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        return lines[0] if lines else None

    def extract_location(self, posting_el) -> str | None:
        """
        Extract location from a .posting HTML element.

        Strategy 1 — .sort-by-location.posting-category (modern boards):
            <span class="sort-by-location posting-category ...">Bangalore</span>

        Strategy 2 — .sort-by-location alone (older variant).

        Returns None if location is absent (graceful — location is optional).
        """
        el = posting_el.query_selector(".sort-by-location.posting-category")
        if el:
            text = el.inner_text().strip()
            return text if text else None

        el = posting_el.query_selector(".sort-by-location")
        if el:
            text = el.inner_text().strip()
            return text if text else None

        return None

    def extract_job_url(self, posting_el, base_url: str) -> str | None:
        """
        Extract and normalise the job URL from a .posting HTML element.

        Lever boards use an <a class="posting-title"> that links to the
        individual job page.  The href is always absolute on Lever's own boards,
        but may be root-relative on custom-embedded implementations.

        Handles three cases:
          Absolute URL  → returned as-is
          Root-relative → prepended with scheme + host of base_url
          None / empty  → returns None (caller skips this posting)
        """
        el = posting_el.query_selector("a.posting-title")
        if not el:
            # Some older boards use a direct <a> on the posting container
            el = posting_el.query_selector('a[href*="lever.co"]')

        if not el:
            return None

        href = (el.get_attribute("href") or "").strip()
        if not href:
            return None

        if href.startswith("http://") or href.startswith("https://"):
            return href

        if href.startswith("/"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{href}"

        self._logger.debug("Ignoring unrecognised URL form: %r", href)
        return None

    def extract_location_from_api(self, categories: dict) -> str | None:
        """
        Build a location string from the Lever API `categories` dict.

        Lever API categories structure:
          {
            "location": "Bangalore",           ← primary location name
            "allLocations": ["Bangalore"],     ← list of locations for multi-site roles
            "team": "R&D Engineering",
            "commitment": "Full-time",
            "department": "Engineering"
          }

        Returns the `location` field if present, falling back to the first
        entry in `allLocations`, or None if neither exists.
        """
        loc = categories.get("location")
        if loc:
            return str(loc).strip() or None

        all_locs = categories.get("allLocations")
        if isinstance(all_locs, list) and all_locs:
            return str(all_locs[0]).strip() or None

        return None

    # ------------------------------------------------------------------
    # Navigation helper
    # ------------------------------------------------------------------

    def _navigate(self, url: str) -> bool:
        """
        Navigate to `url` using domcontentloaded (not networkidle).

        Same reasoning as GreenhouseScraper: networkidle can hang indefinitely
        on pages with background polling or chat widgets.
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _extract_lever_slug(url: str) -> str | None:
    """
    Extract the Lever company slug from a URL.

    Works for:
      https://jobs.lever.co/employ          → "employ"
      https://jobs.lever.co/employ/UUID     → "employ"
      https://api.lever.co/v0/postings/emp  → "emp"

    Returns None for non-Lever domains (company's own career pages).
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "lever.co" not in host:
        return None

    # Strip known path prefixes
    path_parts = [s for s in parsed.path.split("/") if s]
    skip = {"v0", "postings", "embed", "job_board"}
    parts = [p for p in path_parts if p not in skip]

    return parts[0] if parts else None


def _company_name_from_url(url: str) -> str:
    """
    Derive a readable company name from a careers URL.

      https://jobs.lever.co/employ   → "Employ"
      https://razorpay.com/jobs      → "Razorpay"
    """
    parsed = urlparse(url)

    # If it's a Lever board URL, use the slug
    slug = _extract_lever_slug(url)
    if slug:
        return slug.replace("-", " ").title()

    # Otherwise derive from the hostname (e.g. razorpay.com → Razorpay)
    host = parsed.netloc.lower()
    # Remove common prefixes (www., jobs., careers.)
    for prefix in ("www.", "jobs.", "careers.", "hiring."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    # Take first label before the TLD
    name = host.split(".")[0]
    return name.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Description extraction helpers
# ---------------------------------------------------------------------------

_REQUIREMENTS_KEYWORDS = frozenset([
    "require", "qualif", "looking for", "you'll need", "we need",
    "what we're looking", "what you'll need", "must have",
])


def _extract_lever_requirements(lists: list[dict]) -> str | None:
    """
    Find the requirements / qualifications section from Lever API lists.

    Lever postings are structured as a list of sections, each with a `text`
    heading and `content` HTML body.  This function finds the first section
    whose heading matches a requirements keyword, strips the HTML from its
    content, and returns it as plain text.

    Returns None if no requirements section is identified.
    """
    import html as _html_mod
    import re

    def strip(html_str: str) -> str:
        text = _html_mod.unescape(html_str or "")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for section in lists:
        heading = (section.get("text") or "").lower()
        if any(kw in heading for kw in _REQUIREMENTS_KEYWORDS):
            content = section.get("content") or ""
            cleaned = strip(content)
            if cleaned:
                return cleaned

    return None


def _extract_lever_department(categories: dict) -> str | None:
    """
    Extract the department / team from Lever API categories.

    Lever categories dict typically has both `team` (specific team name, e.g.
    "R&D Engineering") and `department` (broader grouping, e.g. "Engineering").
    We prefer `team` because it's more specific and useful for filtering.
    """
    team = (categories.get("team") or "").strip()
    if team:
        return team
    dept = (categories.get("department") or "").strip()
    return dept if dept else None
