"""
RankingAgent — AI-powered job ranking using Google Gemini.

Why AI ranking complements rule-based scoring
---------------------------------------------
The rule-based JobMatcher is fast, free, and fully deterministic.  It works
well for clear signals: "Java" appears in the description, the title says
"Senior", the location is "Remote".  But it is blind to:

  Synonyms:        "distributed systems" ≈ "microservices" — both indicate
                   the same domain competency, but only one keyword appears.

  Adjacent tech:   "Kotlin preferred, Java acceptable" — the rule scorer sees
                   only "Java" and gives full skill credit, missing the nuance
                   that Kotlin proficiency is expected.

  Buried blockers: "Must have active US security clearance" buried in paragraph
                   six — the rule scorer ignores it completely.

  Culture signals: "Startup energy, wear many hats" vs "structured team of 50+
                   engineers with dedicated oncall rotation" — neither matches
                   a rule keyword but both are meaningful.

  Holistic fit:    A job at a prestigious company with slightly mismatched tech
                   might still be worth applying to.  Rules cannot express this.

Gemini reads the full description the same way a recruiter does — it infers,
weighs, and reasons about the whole picture.  The rule score anchors the AI
(a sanity check) while the AI provides depth the rules cannot.

Why caching is important
------------------------
Gemini charges per input token.  A typical job posting costs ~500–1500 tokens.
Re-ranking 60 Workday jobs from Adobe after a minor profile edit would waste
60 × 1500 = 90 000 tokens if results were never cached.

This module caches on:
    SHA-256(job.fingerprint + sorted(profile fields) + model name)

So the cache hits when:
  - The same job is seen in a later scrape (same company + title + URL)
  - The user re-runs ranking without changing their profile or model

The cache is invalidated when:
  - The user updates their UserProfile (sorted field values change)
  - A different Gemini model is selected

The default cache is in-memory (dict), so it resets each process.  Pass an
external dict (e.g., loaded from JSON on startup, saved to JSON on exit) to
persist across runs.

Why structured JSON is safer than free-form text
-------------------------------------------------
See matching/ai_result.py for the full rationale.  Short version:

  - response_mime_type="application/json" + response_schema=AIMatchResult
    tells Gemini to produce a JSON object that matches the Pydantic model.
    The model is constrained at the API layer before the response arrives.
  - Pydantic provides a second validation layer — range, enum, and string
    constraints are all checked at parse time via model_validate_json().
  - No regex, no ad-hoc JSON parsing, no "sometimes the model forgets the
    score" bugs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from matching.ai_result import AIMatchResult
from matching.matcher import JobMatcher, MatchResult
from matching.profile import DEFAULT_PROFILE, UserProfile
from scrapers.models import Job

if TYPE_CHECKING:
    from database.job_repository import JobRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RankedJob — output type that pairs rule-based and AI scores
# ---------------------------------------------------------------------------

@dataclass
class RankedJob:
    """
    One job paired with both a rule-based and an AI match assessment.

    best_score  — AI score if available, else rule score.
                  Used for sorting; mix of the two is intentional so that
                  jobs without AI results still participate in ranking.

    recommendation — "apply"/"consider"/"skip" from the AI result if present,
                     "unknown" if the AI call failed.
    """

    job: Job
    rule_result: MatchResult
    ai_result: AIMatchResult | None

    @property
    def best_score(self) -> int:
        if self.ai_result is not None:
            return self.ai_result.score
        return self.rule_result.score

    @property
    def recommendation(self) -> str:
        if self.ai_result is not None:
            return self.ai_result.recommendation
        return "unknown"

    def __str__(self) -> str:
        tag = f"[{self.recommendation.upper()}]" if self.ai_result else "[rule-only]"
        return f"{tag}  {self.best_score:>3}  {self.job.title} @ {self.job.company}"


# ---------------------------------------------------------------------------
# RankingAgent
# ---------------------------------------------------------------------------

class RankingAgent:
    """
    Ranks jobs using Google Gemini with in-memory caching and graceful
    failure handling.

    Usage
    -----
    from google import genai
    from agents.ranking_agent import RankingAgent

    client = genai.Client(api_key="…")
    agent = RankingAgent(client=client)
    ranked = agent.rank(jobs)           # list[RankedJob], sorted best-first

    # or read directly from the DB
    ranked = agent.rank_from_repository(repo)
    """

    def __init__(
        self,
        client: genai.Client,
        profile: UserProfile = DEFAULT_PROFILE,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        max_workers: int = 1,
        max_description_chars: int = 3000,
        cache: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        client
            An initialised genai.Client() instance.  Must have a valid
            GEMINI_API_KEY (via api_key argument or GEMINI_API_KEY env var).

        profile
            The user's professional background used to build the system prompt.
            Defaults to matching.profile.DEFAULT_PROFILE.

        model
            Gemini model identifier.  Defaults to gemini-2.0-flash.

        max_tokens
            Token budget for Gemini's response.  1024 is ample for the
            structured JSON output; increase only if summaries are truncating.

        max_workers
            Number of concurrent API calls.  1 (default) = sequential.
            Increase to 3–5 for faster batch ranking (subject to your API
            rate limit tier).

        max_description_chars
            Truncate long descriptions before sending to Gemini.  Prevents
            runaway token usage for unusually verbose job postings.

        cache
            External dict to use instead of a fresh in-memory dict.  Pass a
            shared dict to persist cached results across multiple agents or
            across process restarts (after loading from / saving to JSON).
        """
        self._client = client
        self._profile = profile
        self._model = model
        self._max_tokens = max_tokens
        self._max_workers = max_workers
        self._max_description_chars = max_description_chars
        self._cache: dict[str, AIMatchResult] = {} if cache is None else cache
        self._cache_lock = threading.Lock()
        self._system_prompt = self._build_system_prompt()

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def rank(self, jobs: list[Job]) -> list[RankedJob]:
        """
        Rank a list of jobs.  Returns RankedJob objects sorted best-first.

        Rule scores are computed first (instant, free) and passed to Gemini
        as an anchor.  Jobs whose AI call fails still appear in the results
        with their rule score.
        """
        if not jobs:
            return []

        matcher = JobMatcher(self._profile)

        def do_one(job: Job) -> RankedJob:
            return self._rank_one_safe(job, matcher)

        if self._max_workers == 1:
            ranked = [do_one(j) for j in jobs]
        else:
            ranked_map: dict[str, RankedJob] = {}
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(do_one, j): j for j in jobs}
                for future in as_completed(futures):
                    result = future.result()  # _rank_one_safe never raises
                    ranked_map[result.job.fingerprint] = result
            ranked = [ranked_map[j.fingerprint] for j in jobs]

        return sorted(ranked, key=lambda r: r.best_score, reverse=True)

    def rank_one(self, job: Job, rule_score: int = 0) -> AIMatchResult | None:
        """
        Ask Gemini to rank a single job.

        Returns None if the API call fails for any reason (network error,
        validation failure, rate limit).  Callers should treat None as
        "no AI result available" and fall back to the rule score.

        Results are cached by job fingerprint + profile + model.  A second
        call with the same job and unchanged profile returns instantly from
        cache without an API round-trip.
        """
        key = self._cache_key(job)

        with self._cache_lock:
            if key in self._cache:
                logger.debug(
                    "cache hit  %s @ %s (key=%s)", job.title, job.company, key
                )
                return self._cache[key]

        logger.info(
            "ranking    %s @ %s  rule=%d", job.title, job.company, rule_score
        )

        result: AIMatchResult | None = None
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._build_user_message(job, rule_score),
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    response_mime_type="application/json",
                    response_schema=AIMatchResult,
                    max_output_tokens=self._max_tokens,
                ),
            )

            text = getattr(response, "text", None)
            if not text:
                logger.warning(
                    "empty Gemini response for %s @ %s", job.title, job.company
                )
                return None

            result = AIMatchResult.model_validate_json(text)
            logger.info(
                "result     %s @ %s  ai=%d  rec=%s  conf=%s",
                job.title, job.company,
                result.score, result.recommendation, result.confidence,
            )

        except Exception as exc:
            logger.warning(
                "AI ranking failed for %s @ %s: %s",
                job.title, job.company, exc,
            )
            return None

        # Only cache successful results — transient failures should be retried.
        with self._cache_lock:
            self._cache[key] = result

        return result

    def rank_from_repository(self, repo: "JobRepository") -> list[RankedJob]:
        """Read all jobs from the repository and rank them."""
        jobs = repo.get_all()
        logger.info("ranking %d jobs from repository", len(jobs))
        return self.rank(jobs)

    @property
    def cache_size(self) -> int:
        """Number of cached AI results."""
        with self._cache_lock:
            return len(self._cache)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _rank_one_safe(self, job: Job, matcher: JobMatcher) -> RankedJob:
        """Compute rule score then AI score; never raises."""
        rule_result = matcher.match(job)
        ai_result = self.rank_one(job, rule_result.score)
        return RankedJob(job=job, rule_result=rule_result, ai_result=ai_result)

    def _build_system_prompt(self) -> str:
        p = self._profile
        return (
            "You are an expert technical recruiter evaluating software engineering "
            "job postings for a specific candidate.\n\n"
            "Candidate profile\n"
            "-----------------\n"
            f"Years of experience:  {p.years_of_experience}\n"
            f"Primary languages:    {', '.join(p.primary_languages)}\n"
            f"Frameworks:           {', '.join(p.frameworks)}\n"
            f"Backend skills:       {', '.join(p.backend_skills)}\n"
            f"Databases:            {', '.join(p.databases)}\n"
            f"Cloud:                {', '.join(p.cloud)}\n"
            f"Target roles:         {', '.join(p.preferred_roles)}\n"
            f"Preferred locations:  {', '.join(p.preferred_locations)}\n\n"
            "Scoring rubric (0–100)\n"
            "----------------------\n"
            "90–100  Exceptional: all core skills present, ideal seniority, preferred location or remote\n"
            "70–89   Strong:      most skills align, minor gaps, acceptable seniority and location\n"
            "50–69   Moderate:    partial skill overlap; notable gaps or sub-optimal conditions\n"
            "30–49   Weak:        major skill mismatch or wrong seniority tier\n"
            "0–29    Poor:        wrong domain, incompatible requirements, or multiple dealbreakers\n\n"
            "Recommendation thresholds\n"
            "-------------------------\n"
            "apply:    score ≥ 65 and no hard dealbreaker (security clearance, relocation required, etc.)\n"
            "consider: score 45–64 or soft concerns present (some skill gaps, location unclear)\n"
            "skip:     score < 45 or a hard dealbreaker exists\n\n"
            "Confidence level\n"
            "----------------\n"
            "high:   full description and requirements provided\n"
            "medium: one major field missing\n"
            "low:    title only or extremely sparse posting\n\n"
            "A rule-based pre-score is provided for reference. Your score should be independent. "
            "If your score diverges from the pre-score by more than 30 points, verify your reasoning."
        )

    def _build_user_message(self, job: Job, rule_score: int) -> str:
        description = self._truncate(job.description)
        requirements = self._truncate(job.requirements)

        parts = [
            "Evaluate the following job posting for the candidate described in your system prompt.\n",
            f"Company:           {job.company}",
            f"Title:             {job.title}",
            f"Location:          {job.location or 'Not specified'}",
            f"Department:        {job.department or 'Not specified'}",
            f"Rule-based score:  {rule_score}/100\n",
        ]

        if description:
            parts += ["--- Description ---", description, ""]
        else:
            parts += ["Description: Not available\n"]

        if requirements:
            parts += ["--- Requirements ---", requirements, ""]

        parts.append("Respond with your structured JSON assessment for this job.")
        return "\n".join(parts)

    def _cache_key(self, job: Job) -> str:
        """
        Cache key = SHA-256 of (job fingerprint + sorted profile fields + model).

        Sorting each list field ensures the key is stable regardless of list
        ordering in the profile.  Including the model means switching models
        automatically invalidates cached assessments.
        """
        p = self._profile
        payload = json.dumps(
            {
                "fingerprint": job.fingerprint,
                "experience": p.years_of_experience,
                "languages": sorted(p.primary_languages),
                "frameworks": sorted(p.frameworks),
                "backend": sorted(p.backend_skills),
                "databases": sorted(p.databases),
                "cloud": sorted(p.cloud),
                "roles": sorted(p.preferred_roles),
                "locations": sorted(p.preferred_locations),
                "model": self._model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _truncate(self, text: str | None) -> str:
        if not text:
            return ""
        if len(text) <= self._max_description_chars:
            return text
        return text[: self._max_description_chars] + " …[truncated]"
