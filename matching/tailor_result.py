"""
TailoredResumeData — structured output model for ResumeTailoringAgent.

Design mirrors AIMatchResult (matching/ai_result.py):
  Claude responds via the tool-use API with a strictly typed payload;
  Pydantic validates every field on receipt so bad output fails loudly
  rather than silently corrupting the saved file.

Never-fabricate guarantee — two layers
---------------------------------------
Layer 1 (prompt): the ResumeTailoringAgent system prompt explicitly tells
  Claude which actions are allowed (select, reorder, lightly rephrase) and
  which are forbidden (invent skills, add experiences, change numbers).

Layer 2 (model): Pydantic catches structural violations:
  - ats_score_estimate is clamped to 0–100
  - full_resume_text requires at least 100 chars (not empty)
  - selected_experience bullets are plain strings — no schema validates
    their content, but the system prompt and prompt engineering do

Integration
-----------
ResumeTailoringAgent.tailor() returns TailoringResult which wraps this
model together with the file paths where the output was saved.
"""

from pydantic import BaseModel, Field


class ExperienceEntry(BaseModel):
    """One position's selected and ordered bullets from the original resume."""

    company: str = Field(description="Employer name, e.g. 'Zoho' or 'Amazon'.")
    title: str = Field(description="Job title, e.g. 'Member Technical Staff'.")
    bullets: list[str] = Field(
        default_factory=list,
        description=(
            "Bullet points selected verbatim (or lightly rephrased) from the "
            "original resume, ordered most-relevant-first for the target job."
        ),
    )


class TailoredResumeData(BaseModel):
    """
    Claude's complete tailoring output for one job application.

    Fields
    ------
    professional_summary
        2–4 sentence summary rewritten to target this specific role and company.
        Must reference only skills and experience present in the original resume.

    highlighted_skills
        Skill-category strings from the resume (e.g. "Languages: Java, Kotlin,
        Python") reordered so the most job-relevant categories appear first.
        No new skills may be added — only reordering of existing categories.

    selected_experience
        For each position in the resume, the most relevant bullet points for
        the target role, in order of relevance.  Bullets may be lightly
        rephrased for keyword alignment but must not change the underlying facts,
        numbers, or scope of work.

    keywords_incorporated
        Job description keywords that appear naturally in the tailored content.
        Explains why the ATS score is what it is.

    ats_score_estimate
        Rough estimate of ATS keyword match rate (0–100).  Different ATS
        platforms weight keywords differently — treat this as a directional
        signal, not a precise prediction.

    tailoring_notes
        Brief plain-English explanation of the choices made: what was
        emphasised, what was de-prioritised, and why.

    full_resume_text
        The complete ATS-optimised plain-text resume, ready to paste into a
        submission form.  Section order: PROFESSIONAL SUMMARY → TECHNICAL SKILLS
        → EXPERIENCE → PROJECTS → EDUCATION → ACHIEVEMENTS.  No special
        characters, no tables, no columns — just clean plain text.
    """

    professional_summary: str = Field(
        min_length=20,
        description="Role-targeted summary using only real experience from the resume.",
    )
    highlighted_skills: list[str] = Field(
        min_length=1,
        description="Skill categories reordered by relevance to the target role.",
    )
    selected_experience: list[ExperienceEntry] = Field(
        description="Each position with its selected and ordered bullet points.",
    )
    keywords_incorporated: list[str] = Field(
        default_factory=list,
        description="Job keywords present in the selected resume content.",
    )
    ats_score_estimate: int = Field(
        ge=0,
        le=100,
        description="Estimated ATS keyword match percentage (0–100).",
    )
    tailoring_notes: str = Field(
        description="Brief explanation of tailoring choices made.",
    )
    full_resume_text: str = Field(
        min_length=100,
        description="Complete plain-text ATS-optimised resume.",
    )
