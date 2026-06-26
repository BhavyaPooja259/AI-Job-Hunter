"""
Cover letter schema and rendering layer.

CoverLetterData
    Pydantic model that describes the structured output Google Gemini returns.
    Mirrors AIMatchResult / TailoredResumeData: Pydantic validates the AI
    response before anything else touches it, so bad output fails loudly.

CoverLetterGenerator
    Turns structured data (or template variables) into a finished letter string.
    Accepts a custom template at construction time; falls back to
    DEFAULT_LETTER_TEMPLATE when none is supplied.

Template contract
-----------------
Templates are plain Python format strings.  Recognised placeholders:

    {opening_paragraph}   first paragraph
    {body_paragraph_1}    second paragraph (qualifications)
    {body_paragraph_2}    third paragraph  (specific achievement)
    {closing_paragraph}   fourth paragraph (call to action)
    {candidate_name}      signer's full name
    {candidate_email}     signer's email address
    {company}             target company name
    {title}               target job title

All placeholders are always passed to str.format(), so a custom template can
use any subset without raising a KeyError.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# AI output schema
# ---------------------------------------------------------------------------


class CoverLetterData(BaseModel):
    """
    Gemini's structured output for one cover letter.

    Using a Pydantic model (like TailoredResumeData) ensures the AI response
    always contains the expected fields and types before the agent assembles
    the final letter text.
    """

    opening_paragraph: str = Field(
        min_length=20,
        description=(
            "First paragraph that hooks the reader. Must mention the specific "
            "company and role, and explain why the candidate is excited about it."
        ),
    )
    body_paragraph_1: str = Field(
        min_length=20,
        description=(
            "Second paragraph highlighting the candidate's strongest qualification "
            "match for this specific role."
        ),
    )
    body_paragraph_2: str = Field(
        min_length=20,
        description=(
            "Third paragraph citing a specific project, achievement, or result "
            "from the resume that is relevant to the target role."
        ),
    )
    closing_paragraph: str = Field(
        min_length=20,
        description="Final paragraph with a clear, confident call to action.",
    )
    subject_line: str = Field(
        min_length=5,
        description="Email subject line, e.g. 'Application — Backend Engineer at Stripe'.",
    )
    tone_notes: str = Field(
        default="",
        description="Brief note explaining tone choices and what was emphasised.",
    )


# ---------------------------------------------------------------------------
# Default template
# ---------------------------------------------------------------------------


DEFAULT_LETTER_TEMPLATE = """\
Dear Hiring Manager,

{opening_paragraph}

{body_paragraph_1}

{body_paragraph_2}

{closing_paragraph}

Sincerely,
{candidate_name}
{candidate_email}
"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CoverLetterGenerator:
    """
    Renders a cover letter from either AI-generated paragraphs or template
    variables.

    Usage — AI path:
        generator = CoverLetterGenerator()
        text = generator.assemble(data, company="Stripe", title="SWE II",
                                  candidate_name="Ananya", candidate_email="...")

    Usage — no-AI path:
        generator = CoverLetterGenerator()
        text = generator.assemble_basic(company="Stripe", title="SWE II",
                                        candidate_name="Ananya", candidate_email="...")

    Usage — custom template:
        my_template = "Hi {company},\\n\\n{opening_paragraph}\\n\\n— {candidate_name}"
        generator = CoverLetterGenerator(template=my_template)
    """

    def __init__(self, template: str | None = None) -> None:
        self._template = template if template is not None else DEFAULT_LETTER_TEMPLATE

    @property
    def template(self) -> str:
        """The active template string."""
        return self._template

    def assemble(
        self,
        data: CoverLetterData,
        company: str,
        title: str,
        candidate_name: str,
        candidate_email: str,
    ) -> str:
        """
        Assemble a cover letter from AI-generated paragraphs.

        Passes all recognised placeholders to str.format() so the result is
        correct regardless of which subset the active template uses.
        """
        return self._template.format(
            opening_paragraph=data.opening_paragraph,
            body_paragraph_1=data.body_paragraph_1,
            body_paragraph_2=data.body_paragraph_2,
            closing_paragraph=data.closing_paragraph,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            company=company,
            title=title,
        )

    def assemble_basic(
        self,
        company: str,
        title: str,
        candidate_name: str,
        candidate_email: str,
        resume_summary: str = "",
    ) -> str:
        """
        Assemble a professional cover letter without AI, using template
        placeholders filled with structured but non-AI content.

        resume_summary, when provided, is used as the body_paragraph_2
        highlight instead of the generic fallback.
        """
        opening = (
            f"I am writing to express my strong interest in the {title} "
            f"position at {company}. Having followed {company}'s work closely, "
            f"I am excited about the opportunity to contribute to your engineering team."
        )
        body1 = (
            "My background in software engineering — including hands-on experience "
            "with Java, Spring Boot, RESTful APIs, and microservices architecture — "
            "aligns directly with the requirements of this role. I take pride in "
            "building scalable, maintainable systems and enjoy collaborating with "
            "cross-functional teams to ship high-quality software."
        )
        body2 = resume_summary or (
            "Throughout my career I have consistently delivered measurable results: "
            "optimising API latency, reducing system downtime, and mentoring junior "
            "engineers — all while maintaining a focus on code quality and engineering "
            "best practices."
        )
        closing = (
            f"I am genuinely excited about the possibility of joining {company} and "
            "would welcome the opportunity to discuss how my experience maps to your "
            "team's current priorities. Thank you for your time and consideration."
        )
        return self._template.format(
            opening_paragraph=opening,
            body_paragraph_1=body1,
            body_paragraph_2=body2,
            closing_paragraph=closing,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            company=company,
            title=title,
        )

    @staticmethod
    def to_markdown(content: str, company: str, title: str) -> str:
        """Wrap a plain-text cover letter in Markdown for editor preview."""
        return f"# Cover Letter — {title} at {company}\n\n{content}\n"
