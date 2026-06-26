"""
JobMatcher — rule-based job relevance scorer.

Scores each job on four independent dimensions and combines them into a
0–100 integer. Every scoring decision is recorded in MatchResult.reasons
so the output is fully explainable without an LLM.

Scoring dimensions
------------------
  Role relevance   (0–40 pts)   Does the job title match the user's target roles?
  Seniority fit    (0–20 pts)   Does the seniority level align with the target?
  Skill signals    (0–25 pts)   Are the user's key skills mentioned in the posting?
  Location fit     (0–15 pts)   Is the location acceptable?

Why rule-based first (before LLM):
  - Completely deterministic: the same job always gets the same score
  - Zero cost: no API calls, no latency, no rate limits
  - Establishes a baseline to measure against once the LLM layer is added
  - Forces explicit specification of what "a good match" means, which
    becomes the evaluation rubric for the AI scorer

How the LLM will enhance this later:
  - The rule scorer will remain as a fast pre-filter (score < 20 → skip)
  - The LLM will receive the job description + profile and produce a
    richer MatchResult with nuanced reasoning, inferred skill matches,
    and culture-fit assessment that rules cannot capture
  - Both scores will be stored; the rule score acts as a sanity check
    on the AI score (large divergence = flag for human review)

How MatchResult feeds resume tailoring:
  - matched_skills → bullets in the resume to lead with
  - missing_skills → skills to de-emphasise or omit
  - reasons → context for the cover letter agent's opening paragraph
  - score → determines whether to tailor at all (threshold configurable)
"""

import logging
from pydantic import BaseModel

from scrapers.models import Job
from matching.profile import UserProfile

logger = logging.getLogger(__name__)

# Engineering role keywords that indicate the posting is in the right domain.
_ENGINEERING_KEYWORDS = frozenset([
    "engineer", "developer", "architect", "sde", "swe", "programmer",
])

# Seniority keywords and their tier (higher = more senior).
_SENIORITY_SIGNALS: dict[str, int] = {
    "principal": 3,
    "staff": 3,
    "senior": 2,
    "sr.": 2,
    " ii": 2,
    " 2": 2,
    "lead": 2,
    "junior": 0,
    "jr.": 0,
    "entry": 0,
    "associate": 0,
    "intern": 0,
}

# Location signals and how many points they award.
_US_SIGNALS = frozenset(["united states", "anywhere", "usa", "u.s."])
_REMOTE_SIGNALS = frozenset(["remote"])


class MatchResult(BaseModel):
    """The scored output of matching one Job against a UserProfile."""

    score: int               # 0–100 composite score
    matched_skills: list[str]  # user skills found in the job text
    missing_skills: list[str]  # key user skills absent from the job text
    reasons: list[str]       # human-readable explanation of the score


class JobMatcher:
    """
    Scores jobs against a UserProfile using deterministic rules.

    Usage:
        matcher = JobMatcher(DEFAULT_PROFILE)
        result  = matcher.match(job)
        ranked  = matcher.rank(jobs)   # sorted highest-score first
    """

    def __init__(self, profile: UserProfile) -> None:
        self._profile = profile
        # Pre-lowercase for O(1) comparisons during matching
        self._preferred_roles_lower = [r.lower() for r in profile.preferred_roles]
        self._all_skills_lower = [s.lower() for s in profile.all_skills()]

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def match(self, job: Job) -> MatchResult:
        """Score one job and return a fully explained MatchResult."""
        text = self._job_text(job)

        role_pts,      role_reasons      = self._score_role(job.title)
        seniority_pts, seniority_reasons = self._score_seniority(job.title, role_pts)
        skill_pts,     matched, skill_reasons = self._score_skills(text)
        location_pts,  location_reasons  = self._score_location(job.location)

        total = min(100, role_pts + seniority_pts + skill_pts + location_pts)
        missing = self._missing_key_skills(text)

        return MatchResult(
            score=total,
            matched_skills=matched,
            missing_skills=missing,
            reasons=role_reasons + seniority_reasons + skill_reasons + location_reasons,
        )

    def rank(self, jobs: list[Job]) -> list[tuple[Job, MatchResult]]:
        """Return (job, result) pairs sorted by score descending."""
        pairs = [(job, self.match(job)) for job in jobs]
        return sorted(pairs, key=lambda x: x[1].score, reverse=True)

    # -------------------------------------------------------------------------
    # Scoring dimensions
    # -------------------------------------------------------------------------

    def _score_role(self, title: str) -> tuple[int, list[str]]:
        """
        Role relevance — 0 to 40 points.

        40: title contains an exact preferred role phrase
        20: title contains an engineering keyword but not a preferred role
         0: neither
        """
        title_lower = title.lower()
        reasons: list[str] = []

        for i, role_lower in enumerate(self._preferred_roles_lower):
            if role_lower in title_lower:
                original = self._profile.preferred_roles[i]
                reasons.append(f"Preferred role '{original}' found in title (+40)")
                return 40, reasons

        for kw in _ENGINEERING_KEYWORDS:
            if kw in title_lower:
                reasons.append(f"Engineering keyword '{kw}' found in title (+20)")
                return 20, reasons

        reasons.append("Title does not match any preferred role or engineering keyword (+0)")
        return 0, reasons

    def _score_seniority(self, title: str, role_pts: int) -> tuple[int, list[str]]:
        """
        Seniority fit — 0 to 20 points.

        Points are only awarded when the role is relevant (role_pts > 0).
        A non-engineering senior role gets no seniority credit.

          tier 3 (principal/staff): 20
          tier 2 (senior/II/lead):  20   ← SDE2 target aligns with tier 2
          implied mid-level:        12
          tier 0 (junior/entry):     0
        """
        reasons: list[str] = []

        if role_pts == 0:
            reasons.append("Seniority not evaluated — role is not relevant (+0)")
            return 0, reasons

        title_lower = title.lower()

        for signal, tier in _SENIORITY_SIGNALS.items():
            if signal in title_lower:
                if tier >= 2:
                    reasons.append(
                        f"Seniority indicator '{signal.strip()}' aligns with SDE2 target (+20)"
                    )
                    return 20, reasons
                else:
                    reasons.append(
                        f"Junior/entry seniority indicator '{signal.strip()}' — below target (+0)"
                    )
                    return 0, reasons

        reasons.append("No explicit seniority indicator — mid-level assumed (+12)")
        return 12, reasons

    def _score_skills(self, text: str) -> tuple[int, list[str], list[str]]:
        """
        Skill signals — 0 to 25 points.

        5 points per key user skill found in the job text, capped at 25.
        Key skills are: Java, Spring Boot, REST, SQL, Microservices — the
        core of the user's SDE2 profile.
        """
        key_skills = ["java", "spring boot", "spring", "rest", "sql", "microservices",
                      "kafka", "redis", "postgresql", "aws", "docker", "backend"]
        matched: list[str] = []
        reasons: list[str] = []

        for skill in key_skills:
            if skill in text:
                matched.append(skill)

        if matched:
            pts = min(25, len(matched) * 5)
            reasons.append(
                f"Found {len(matched)} skill signal(s) in job text: "
                f"{', '.join(matched)} (+{pts})"
            )
        else:
            pts = 0
            reasons.append("No key skill signals found in job text (+0)")

        return pts, matched, reasons

    def _score_location(self, location: str | None) -> tuple[int, list[str]]:
        """
        Location fit — 0 to 15 points.

        15: explicitly remote OR location matches a preferred location
         8: US-based (compatible with future relocation / remote-friendly)
         0: foreign country with no remote signal
        """
        reasons: list[str] = []

        if location is None:
            reasons.append("Location not specified (+0)")
            return 0, reasons

        loc_lower = location.lower()

        for signal in _REMOTE_SIGNALS:
            if signal in loc_lower:
                reasons.append(f"Location '{location}' is remote-friendly (+15)")
                return 15, reasons

        for pref in self._profile.preferred_locations:
            if pref.lower() in loc_lower or loc_lower in pref.lower():
                reasons.append(
                    f"Location '{location}' matches preferred location '{pref}' (+15)"
                )
                return 15, reasons

        for signal in _US_SIGNALS:
            if signal in loc_lower:
                reasons.append(f"Location '{location}' is US-based (+8)")
                return 8, reasons

        reasons.append(f"Location '{location}' does not match preferences (+0)")
        return 0, reasons

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _job_text(self, job: Job) -> str:
        """
        Combine all searchable text from a job into one lowercase string.

        Why descriptions improve matching quality
        -----------------------------------------
        The job title alone rarely specifies technologies.  "Software Engineer II"
        looks the same whether the stack is Java + Spring Boot or Go + gRPC.
        Descriptions and requirements sections contain explicit skill callouts
        ("5+ years of Java", "experience with Spring Boot and Kafka") that the
        rule scorer can match against directly.

        Including `requirements` separately from `description` matters because
        some ATS platforms (Lever) separate the "what you'll do" narrative
        (description) from the "what we need" checklist (requirements).
        Concatenating both ensures skills mentioned in either section score.

        The location field is included so location-bearing terms like "Remote"
        appear in the combined text (though the dedicated location scorer is
        the primary mechanism for location fit).
        """
        parts = [
            job.title,
            job.description or "",
            job.requirements or "",
            job.location or "",
        ]
        return " ".join(parts).lower()

    def _missing_key_skills(self, text: str) -> list[str]:
        """
        Return skills from the user's core profile not found in the job text.

        Only checks a focused set of must-have skills so the list stays
        actionable rather than listing every technology the job didn't mention.
        """
        must_have = ["java", "spring boot", "rest", "sql", "microservices"]
        return [s for s in must_have if s not in text]
