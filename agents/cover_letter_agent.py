"""
CoverLetterAgent — generates personalised cover letters using Google Gemini.

Design overview
---------------
Mirrors ResumeTailoringAgent almost exactly:

  1. Builds a system prompt that embeds the candidate's resume text and the
     target job's details.
  2. Calls Gemini with response_mime_type="application/json" and
     response_schema=CoverLetterData so the response is a validated JSON
     payload, never free-form text.
  3. CoverLetterGenerator.assemble() renders the AI paragraphs into the
     active template.
  4. Saves two files:
       {output_dir}/cover_{company}_{timestamp}.txt  — plain text
       {output_dir}/cover_{company}_{timestamp}.md   — Markdown preview
  5. Returns a CoverLetter dataclass with the content and file paths.

No-AI mode
----------
Constructing the agent without a `client` (client=None) enables
template-only mode: all AI calls are skipped and a professional
cover letter is assembled entirely from the template and job metadata.
This is useful for:
  - Running without a Gemini API key (demo / offline mode)
  - Unit tests that don't need a mock client
  - Batch generation when AI budget is exhausted

Fallback behaviour
------------------
If the Gemini call fails for any reason — network error, empty response,
invalid JSON — the agent falls back silently to template-only mode and
still returns a usable CoverLetter.  Failure is logged but never raised.

Template support
----------------
Pass a custom template string at construction time.  See
CoverLetterGenerator and DEFAULT_LETTER_TEMPLATE in
cover_letter/cover_letter_generator.py for the placeholder contract.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ai.provider import AIProvider
from cover_letter.cover_letter import CoverLetter
from cover_letter.cover_letter_generator import CoverLetterData, CoverLetterGenerator
from scrapers.models import Job

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "storage" / "cover_letters"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CoverLetterAgent:
    """
    Generates personalised cover letters from a job description and resume.

    Parameters
    ----------
    resume_text
        The candidate's resume as plain text.  Pass this explicitly —
        unlike ResumeTailoringAgent, this agent does not read PDFs so
        that tests can pass arbitrary text without pdfplumber.

    client
        An initialised google.genai.Client instance.  Pass None to
        use template-only mode (no AI, no API key required).

    template
        Custom cover letter template string.  None = use the built-in
        DEFAULT_LETTER_TEMPLATE.  See CoverLetterGenerator for the
        full list of supported {placeholders}.

    output_dir
        Directory where .txt and .md files are saved.
        Defaults to storage/cover_letters/ relative to the project root.

    model
        Gemini model ID.  Defaults to gemini-2.0-flash.

    max_tokens
        Token budget for Gemini's response.  1500 is sufficient for a
        four-paragraph cover letter.

    candidate_name
        Full name that appears in the letter signature.

    candidate_email
        Email address that appears in the letter signature.

    max_description_chars
        Truncate long job descriptions before sending to Gemini.
    """

    def __init__(
        self,
        resume_text: str,
        client: "genai.Client | None" = None,
        provider: "AIProvider | None" = None,
        template: str | None = None,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1500,
        candidate_name: str = "",
        candidate_email: str = "",
        max_description_chars: int = 3000,
    ) -> None:
        self._resume_text = resume_text
        self._generator = CoverLetterGenerator(template=template)
        self._output_dir = output_dir
        self._model = model
        self._max_tokens = max_tokens
        self._candidate_name = candidate_name
        self._candidate_email = candidate_email
        self._max_description_chars = max_description_chars

        # Resolve provider: explicit provider > client wrapper > None (template mode)
        if provider is not None:
            self._provider: AIProvider | None = provider
        elif client is not None:
            from ai.gemini_provider import GeminiProvider
            self._provider = GeminiProvider(client=client, model=model)
        else:
            self._provider = None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def generate(self, job: Job, save: bool = True) -> CoverLetter:
        """
        Generate a cover letter for the given job.

        Parameters
        ----------
        job
            Target job (provides company, title, description, etc.).
        save
            When True, write .txt and .md files to output_dir.
            When False, return the CoverLetter without writing any files.

        Returns
        -------
        CoverLetter
            Always returns a usable letter.  Falls back to template-only
            mode if the Gemini call fails or client is None.
        """
        logger.info("generating cover letter for %s @ %s", job.title, job.company)

        if self._provider is not None:
            letter = self._generate_with_ai(job)
        else:
            letter = self._generate_from_template(job)

        if save:
            letter = self._save(letter, job)

        return letter

    # -------------------------------------------------------------------------
    # AI path
    # -------------------------------------------------------------------------

    def _generate_with_ai(self, job: Job) -> CoverLetter:
        """Use the configured AIProvider to produce personalised paragraphs, then assemble."""
        system = self._build_system_prompt()
        user_msg = self._build_user_message(job)

        try:
            text = self._provider.complete(  # type: ignore[union-attr]
                user_message=user_msg,
                system_prompt=system,
                response_schema=CoverLetterData,
                max_tokens=self._max_tokens,
            )
            data = CoverLetterData.model_validate_json(text)
            content = self._generator.assemble(
                data=data,
                company=job.company,
                title=job.title,
                candidate_name=self._candidate_name,
                candidate_email=self._candidate_email,
            )
            logger.info("cover letter generated for %s @ %s", job.title, job.company)
            return CoverLetter(
                company=job.company,
                title=job.title,
                candidate_name=self._candidate_name,
                candidate_email=self._candidate_email,
                content=content,
                subject_line=data.subject_line,
            )

        except Exception as exc:
            logger.warning(
                "cover letter AI generation failed for %s @ %s: %s — falling back to template",
                job.title, job.company, exc,
            )
            return self._generate_from_template(job)

    # -------------------------------------------------------------------------
    # Template-only path
    # -------------------------------------------------------------------------

    def _generate_from_template(self, job: Job) -> CoverLetter:
        """Generate a cover letter without AI, using the built-in template."""
        content = self._generator.assemble_basic(
            company=job.company,
            title=job.title,
            candidate_name=self._candidate_name,
            candidate_email=self._candidate_email,
        )
        name_part = self._candidate_name or "Candidate"
        subject_line = f"Application for {job.title} — {name_part}"
        return CoverLetter(
            company=job.company,
            title=job.title,
            candidate_name=self._candidate_name,
            candidate_email=self._candidate_email,
            content=content,
            subject_line=subject_line,
        )

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert career coach who writes compelling, authentic "
            "cover letters.\n\n"
            "YOUR RULES:\n"
            "1. Write in first person, professional but warm tone.\n"
            "2. Every claim must be grounded in the candidate's resume below.\n"
            "3. Do NOT invent experience, skills, metrics, or employers.\n"
            "4. Reference the company by name — show you know who they are.\n"
            "5. Keep each paragraph focused and concise (3–5 sentences).\n"
            "6. The subject_line should be brief and professional.\n"
            "7. tone_notes should explain your choices in 1–2 sentences.\n\n"
            "CANDIDATE'S RESUME:\n"
            "----------------------------------------\n"
            f"{self._resume_text}\n"
            "----------------------------------------\n"
        )

    def _build_user_message(self, job: Job) -> str:
        desc = self._truncate(job.description)
        reqs = self._truncate(job.requirements)

        parts = [
            "Write a cover letter for the following job.\n",
            "TARGET JOB",
            f"  Company:  {job.company}",
            f"  Title:    {job.title}",
            f"  Location: {job.location or 'Not specified'}",
            "",
        ]

        if desc:
            parts += ["JOB DESCRIPTION", desc, ""]
        if reqs:
            parts += ["JOB REQUIREMENTS", reqs, ""]

        parts += [
            "INSTRUCTIONS",
            "- opening_paragraph: hook that mentions the company and role specifically.",
            "- body_paragraph_1: the candidate's strongest qualification match.",
            "- body_paragraph_2: a specific achievement or project from the resume.",
            "- closing_paragraph: confident call to action.",
            "- subject_line: short, professional email subject.",
            "",
            "Respond with your complete cover letter in the required JSON format.",
        ]

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _save(self, letter: CoverLetter, job: Job) -> CoverLetter:
        """Save letter to .txt and .md files. Mutates and returns letter."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._filename_stem(job)

        txt_path = self._output_dir / f"{stem}.txt"
        md_path = self._output_dir / f"{stem}.md"

        txt_path.write_text(letter.content, encoding="utf-8")
        md_content = CoverLetterGenerator.to_markdown(
            letter.content, job.company, job.title
        )
        md_path.write_text(md_content, encoding="utf-8")

        logger.info("saved cover letter txt → %s", txt_path)
        logger.info("saved cover letter md  → %s", md_path)

        letter.txt_path = txt_path
        letter.md_path = md_path
        return letter

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _truncate(self, text: str | None) -> str:
        if not text:
            return ""
        if len(text) <= self._max_description_chars:
            return text
        return text[: self._max_description_chars] + " …[truncated]"

    @staticmethod
    def _filename_stem(job: Job) -> str:
        company = re.sub(r"[^a-z0-9]+", "_", job.company.lower()).strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cover_{company}_{ts}"
