"""
ResumeTailoringAgent — AI-powered resume tailoring using Google Gemini.

Design overview
---------------
The agent reads the candidate's resume PDF once (at construction time),
then for each job posting it:

  1. Builds a system prompt that embeds the full resume text and the
     strict never-fabricate rules.
  2. Builds a user message that describes the target job (+ optional
     AIMatchResult context from the ranking step).
  3. Calls Gemini with response_mime_type="application/json" and
     response_schema=TailoredResumeData so the response is always a
     validated JSON payload, never free-form text.
  4. Validates the payload with TailoredResumeData (Pydantic).
  5. Saves two files:
       {output_dir}/{company}_{timestamp}.txt     — plain-text ATS resume
       {output_dir}/{company}_{timestamp}_analysis.json — structured JSON
  6. Returns a TailoringResult dataclass with the data + file paths.

Never-fabricate contract
------------------------
The system prompt opens with an explicit, repeated instruction that Gemini
may only SELECT, REORDER, and LIGHTLY REPHRASE existing content.  It may
not invent skills, add employers, change metrics, or include experience
that is not in the provided resume text.

ATS optimisation strategy
--------------------------
  1. Keywords first: the professional summary is rewritten to front-load
     the keywords most frequently mentioned in the job description.
  2. Skills ordered by fit: the most relevant skill categories appear at
     the top of the Technical Skills section.
  3. Bullets ordered by impact: within each role, the most relevant
     bullets appear first so ATS parsers and human reviewers see them
     immediately.
  4. Plain text: the full_resume_text output uses ASCII-only formatting,
     standard section headers in ALL CAPS, and dash-prefixed bullets.
     No tables, no multi-column layout, no Unicode decorations that might
     confuse legacy ATS parsers.

PDF extraction
--------------
Uses pdfplumber for text extraction.  pdfplumber handles multi-column
resume layouts significantly better than pypdf (which produced
character-spaced output "B h a v y a  L" for this particular PDF).

For testing, pass resume_text directly to the constructor to bypass PDF
reading entirely — no pdfplumber dependency in unit tests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ai.provider import AIProvider
from matching.ai_result import AIMatchResult
from matching.matcher import MatchResult
from matching.tailor_result import ExperienceEntry, TailoredResumeData
from scrapers.models import Job

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)

_DEFAULT_RESUME_PATH = Path(__file__).parent.parent / "resume" / "bhavya-resume.pdf"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "resume" / "tailored"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class TailoringResult:
    """
    Returned by ResumeTailoringAgent.tailor().

    Attributes
    ----------
    job
        The job this resume was tailored for.
    data
        Gemini's structured tailoring analysis (validated TailoredResumeData).
    text_path
        Path to the saved ATS plain-text resume file.
    analysis_path
        Path to the saved JSON analysis file (full structured data).
    """

    job: Job
    data: TailoredResumeData
    text_path: Path
    analysis_path: Path

    def __str__(self) -> str:
        return (
            f"TailoringResult for {self.job.title} @ {self.job.company}\n"
            f"  ATS estimate:  {self.data.ats_score_estimate}%\n"
            f"  Keywords:      {', '.join(self.data.keywords_incorporated[:5])}\n"
            f"  Text file:     {self.text_path}\n"
            f"  Analysis file: {self.analysis_path}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResumeTailoringAgent:
    """
    Tailors the candidate's resume for a specific job using Google Gemini.

    Usage
    -----
    from google import genai
    from agents.resume_tailoring_agent import ResumeTailoringAgent

    client = genai.Client(api_key="…")
    agent = ResumeTailoringAgent(client=client)
    result = agent.tailor(job, ai_result=ranked.ai_result)
    print(result.data.full_resume_text)
    print(result.text_path)
    """

    def __init__(
        self,
        client: "genai.Client | None" = None,
        provider: "AIProvider | None" = None,
        resume_pdf_path: Path = _DEFAULT_RESUME_PATH,
        resume_text: str | None = None,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 3000,
        max_description_chars: int = 2000,
    ) -> None:
        """
        Parameters
        ----------
        client
            Initialised google.genai.Client.  Takes precedence over provider
            when both are given.  Either client or provider must be supplied.
        provider
            Any AIProvider implementation.  Used when client is None.
        resume_pdf_path
            Path to the candidate's resume PDF.  Ignored when resume_text
            is provided.  Defaults to resume/bhavya-resume.pdf.
        resume_text
            Pass resume content directly to bypass PDF extraction.
        output_dir
            Directory where tailored resumes are saved.
        model
            Model ID forwarded to GeminiProvider when wrapping a raw client.
        max_tokens
            Token budget per AI response.  3000 is safe for free-tier models
            because schema-injection stripping (see _maybe_strip_schema in
            the HTTP providers) keeps the total context well within 8192
            tokens even for real-world resumes (3600-char PDF) combined with
            full job descriptions.
        max_description_chars
            Truncate long job descriptions before sending to the provider.
        """
        # Resolve provider: explicit provider > client wrapper > error
        if provider is not None:
            self._provider: AIProvider = provider
        elif client is not None:
            from ai.gemini_provider import GeminiProvider
            self._provider = GeminiProvider(client=client, model=model)
        else:
            raise ValueError(
                "ResumeTailoringAgent requires either 'client' or 'provider'"
            )

        self._output_dir = output_dir
        self._model = model
        self._max_tokens = max_tokens
        self._max_description_chars = max_description_chars

        if resume_text is not None:
            self._resume_text = resume_text
            logger.debug("using provided resume_text (%d chars)", len(resume_text))
        else:
            self._resume_text = self._extract_pdf_text(resume_pdf_path)
            logger.info(
                "extracted %d chars from %s", len(self._resume_text), resume_pdf_path
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def tailor(
        self,
        job: Job,
        ai_result: AIMatchResult | None = None,
        rule_result: MatchResult | None = None,
    ) -> TailoringResult | None:
        """
        Tailor the resume for a specific job.

        Parameters
        ----------
        job
            The target job (provides title, company, description, etc.).

        ai_result
            Optional AIMatchResult from RankingAgent.rank_one().  When
            provided, the agent uses the strengths and missing_skills lists
            to inform which content to emphasise or de-emphasise.

        rule_result
            Optional MatchResult from JobMatcher.  When provided, its
            matched_skills list is included as additional context.

        Returns
        -------
        TailoringResult on success, None if the Gemini call fails.
        Failure is logged but never raises so a batch can continue.
        """
        logger.info("tailoring resume for %s @ %s", job.title, job.company)

        system = self._build_system_prompt()
        user_msg = self._build_user_message(job, ai_result, rule_result)

        try:
            text = self._provider.complete(
                user_message=user_msg,
                system_prompt=system,
                response_schema=TailoredResumeData,
                max_tokens=self._max_tokens,
            )
            data = TailoredResumeData.model_validate_json(text)
            logger.info(
                "tailored %s @ %s — ATS estimate %d%%, %d keywords",
                job.title, job.company,
                data.ats_score_estimate,
                len(data.keywords_incorporated),
            )

        except Exception as exc:
            logger.warning(
                "tailoring failed for %s @ %s: %s",
                job.title, job.company, exc,
            )
            return None

        return self._save(job, data)

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert ATS-optimised resume writer.\n\n"
            "YOUR ABSOLUTE RULES — NEVER VIOLATE THESE:\n"
            "1. You may ONLY use information from the candidate's resume below.\n"
            "2. You may SELECT which bullets to include (choose most relevant).\n"
            "3. You may REORDER bullets within a role (most relevant first).\n"
            "4. You may LIGHTLY REPHRASE a bullet to incorporate a job keyword, "
            "but ONLY if the underlying facts, metrics, and scope are unchanged.\n"
            "5. You may REORDER skill categories (most relevant first).\n"
            "6. You may write a new professional summary, but it must reference "
            "ONLY real skills and experience in this resume.\n"
            "7. You MUST NOT invent skills, add employers, fabricate metrics, "
            "or describe any experience not in this resume.\n"
            "8. All companies, dates, and CGPA must appear exactly as written.\n\n"
            "ATS FORMATTING RULES for full_resume_text:\n"
            "- Section headers in ALL CAPS\n"
            "- Use '- ' prefix for bullet points\n"
            "- No tables, no columns, no Unicode decorations\n"
            "- Preserve all metrics (40%, 30%, 60%, 98%, 500+) exactly\n"
            "- Contact line: name@email.com | phone\n\n"
            "CANDIDATE'S FULL RESUME:\n"
            "----------------------------------------\n"
            f"{self._resume_text}\n"
            "----------------------------------------\n"
        )

    def _build_user_message(
        self,
        job: Job,
        ai_result: AIMatchResult | None,
        rule_result: MatchResult | None,
    ) -> str:
        desc = self._truncate(job.description)
        reqs = self._truncate(job.requirements)

        parts = [
            "Tailor the candidate's resume for the following job.\n",
            f"TARGET JOB",
            f"  Company:    {job.company}",
            f"  Title:      {job.title}",
            f"  Location:   {job.location or 'Not specified'}",
            f"  Department: {job.department or 'Not specified'}",
            "",
        ]

        if desc:
            parts += ["JOB DESCRIPTION", desc, ""]
        if reqs:
            parts += ["JOB REQUIREMENTS", reqs, ""]

        if ai_result is not None:
            parts += [
                "AI MATCH ANALYSIS (from prior ranking step)",
                f"  Score:       {ai_result.score}/100",
                f"  Strengths:   {', '.join(ai_result.strengths)}",
                f"  Gaps:        {', '.join(ai_result.missing_skills)}",
                "",
            ]

        if rule_result is not None:
            if rule_result.matched_skills:
                parts += [
                    f"RULE-BASED MATCHED SKILLS: {', '.join(rule_result.matched_skills)}",
                    "",
                ]

        parts += [
            "TAILORING INSTRUCTIONS",
            "1. Incorporate the most important job keywords into the professional summary.",
            "2. Reorder skill categories so the most job-relevant appear first.",
            "3. Select and reorder bullets within each role — most relevant first.",
            "4. In full_resume_text, produce a clean plain-text ATS resume using "
            "the format described in your system prompt.",
            "5. Preserve ALL original dates, metrics, and company names exactly.",
            "",
            "Respond with your complete tailoring in the required JSON format.",
        ]

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _save(self, job: Job, data: TailoredResumeData) -> TailoringResult:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._filename_stem(job)

        text_path = self._output_dir / f"{stem}.txt"
        analysis_path = self._output_dir / f"{stem}_analysis.json"

        text_path.write_text(data.full_resume_text, encoding="utf-8")
        logger.info("saved resume text → %s", text_path)

        analysis_path.write_text(
            data.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("saved analysis    → %s", analysis_path)

        return TailoringResult(
            job=job,
            data=data,
            text_path=text_path,
            analysis_path=analysis_path,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Extract plain text from a resume PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Run: pip install pdfplumber"
            ) from exc

        with pdfplumber.open(str(path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()

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
        return f"{company}_{ts}"
