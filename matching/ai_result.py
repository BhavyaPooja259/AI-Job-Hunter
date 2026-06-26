"""
AIMatchResult — structured output model for Claude's job ranking response.

Why structured JSON is safer than free-form text
-------------------------------------------------
Asking Claude to "respond with a score and explanation" produces text that
varies in format, is hard to parse reliably, and can contain well-reasoned
but machine-unreadable content:

  "I'd give this around 75 out of 100, though it depends on..."

Compare to a validated Pydantic object:

  AIMatchResult(score=75, confidence="high", recommendation="apply", ...)

The structured approach wins on every dimension:

  Reliability:   Pydantic validates type and range at parse time — a score of
                 "seventy-five" or 150 raises an error immediately rather than
                 silently producing wrong behavior downstream.

  Determinism:   The same fields are always present.  Downstream code (ranking,
                 display, storage) never has to handle a missing field.

  Testability:   Mock responses are just dicts — no regex or string parsing
                 in test setup.

  Extensibility: Adding a field to the model is one line here plus one line
                 in the tool definition.  No prompt parsing logic changes.

How this is enforced
--------------------
The RankingAgent uses Claude's tool-use API with tool_choice forced to a
single named tool ("submit_job_ranking").  Claude MUST call that tool; the
API layer rejects responses that don't match the declared input_schema.
The input dict is passed to AIMatchResult.model_validate() for a final
Pydantic check — two independent layers of validation.

Integration
-----------
This model is returned by RankingAgent.rank_one().  It is paired with
MatchResult (rule-based) in a RankedJob to give the pipeline two
complementary scores: deterministic + AI-powered.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AIMatchResult(BaseModel):
    """
    Claude's structured assessment of one job posting against a user profile.

    Scoring dimensions (0–100 composite):
      Role fit      — does the role match the candidate's target titles?
      Tech overlap  — are the candidate's skills mentioned?
      Seniority     — does the level align with the candidate's experience?
      Location fit  — remote / preferred city / acceptable market?
      Growth signal — is this role a good career step?

    Unlike the rule-based MatchResult (which scores title + skill keywords),
    AIMatchResult can understand:
      - Synonyms ("distributed systems" ≈ "microservices experience")
      - Adjacent technologies ("Kotlin" is close to "Java" but not identical)
      - Deal-breakers buried in descriptions ("US citizenship required")
      - Culture signals ("fast-paced startup" vs "large enterprise org")
    """

    score: int = Field(
        description="Overall match 0–100 (higher = better fit for the candidate)",
        ge=0,
        le=100,
    )

    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in this assessment. "
            "'high' when full description and requirements are present; "
            "'medium' when one field is missing; "
            "'low' when title-only or very sparse posting."
        )
    )

    recommendation: Literal["apply", "consider", "skip"] = Field(
        description=(
            "'apply' — strong match, candidate should prioritise this; "
            "'consider' — partial match, worth a closer read; "
            "'skip' — poor fit or a dealbreaker exists."
        )
    )

    summary: str = Field(
        description="1–2 sentence plain-English assessment of the overall fit.",
        min_length=10,
    )

    strengths: list[str] = Field(
        default_factory=list,
        description="Specific ways the candidate's profile aligns with this role.",
    )

    missing_skills: list[str] = Field(
        default_factory=list,
        description="Skills or experience required by the role that the candidate lacks.",
    )

    interview_difficulty: Literal["low", "medium", "high", "very_high"] = Field(
        description=(
            "Estimated interview bar. "
            "'low': small company, generalist role; "
            "'medium': mid-size company, standard eng process; "
            "'high': large tech company, structured rounds; "
            "'very_high': FAANG/top-tier, deep system design + coding bar."
        )
    )

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("summary must not be blank")
        return stripped
