"""
Tests for Sprint 17 — Cover Letter Agent.

Coverage
--------
TestCoverLetterData        (6)  — Pydantic AI output schema validation
TestCoverLetter            (6)  — CoverLetter dataclass
TestCoverLetterGenerator   (10) — template rendering, AI and no-AI paths
TestCoverLetterAgentNoAI   (9)  — template-only mode (client=None, no mock)
TestCoverLetterAgentWithAI (12) — mocked Gemini client

The no-AI tests prove that the full pipeline works without any API key.
The AI tests use the exact same mock pattern as test_ai_ranking.py and
test_resume_tailoring.py so the style is consistent across the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.cover_letter_agent import CoverLetterAgent
from config.constants import ATSType
from cover_letter.cover_letter import CoverLetter
from cover_letter.cover_letter_generator import (
    DEFAULT_LETTER_TEMPLATE,
    CoverLetterData,
    CoverLetterGenerator,
)
from scrapers.models import Job


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RESUME_TEXT = """\
Ananya Sharma | ananya@example.com | +1-555-1234

EXPERIENCE
Software Engineer II — Zoho Corp (2022–present)
- Reduced API latency by 40% via query optimisation
- Led migration of 5 microservices to Spring Boot 3
- Mentored 3 junior engineers

Software Engineer I — StartupXYZ (2021–2022)
- Built REST API serving 500+ daily active users

SKILLS
Languages: Java, Kotlin, Python, SQL
Frameworks: Spring Boot, Hibernate, JUnit

EDUCATION
B.E. Computer Science — VIT University, 8.7 CGPA
"""

_VALID_AI_PAYLOAD = {
    "opening_paragraph": (
        "I am excited to apply for the Backend Engineer role at Stripe. "
        "Stripe's engineering culture and focus on developer experience deeply align "
        "with how I approach building software."
    ),
    "body_paragraph_1": (
        "My three years of hands-on experience with Java, Spring Boot, and "
        "microservices architecture map directly to the requirements you've outlined. "
        "At Zoho, I reduced API latency by 40% and led a five-service migration "
        "to Spring Boot 3."
    ),
    "body_paragraph_2": (
        "One project I am particularly proud of is the REST API I built at StartupXYZ "
        "that scaled to serve over 500 daily active users with less than 100ms p99 latency."
    ),
    "closing_paragraph": (
        "I would welcome the opportunity to discuss how my background aligns with "
        "Stripe's engineering team goals. Thank you for your consideration."
    ),
    "subject_line": "Application — Backend Engineer at Stripe",
    "tone_notes": "Professional but warm. Emphasised latency and scale achievements.",
}


def _make_job(
    company: str = "Stripe",
    title: str = "Backend Engineer",
    job_url: str = "https://stripe.com/jobs/1",
    description: str = "Build scalable Java backend services.",
) -> Job:
    return Job(
        company=company,
        title=title,
        job_url=job_url,
        source_platform=ATSType.GREENHOUSE,
        description=description,
    )


def _mock_client(payload: dict) -> MagicMock:
    response = MagicMock()
    response.text = json.dumps(payload)
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _empty_response_client() -> MagicMock:
    response = MagicMock()
    response.text = None
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _error_client(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = exc
    return client


# ---------------------------------------------------------------------------
# TestCoverLetterData
# ---------------------------------------------------------------------------


class TestCoverLetterData:
    def test_valid_payload_parses(self):
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        assert data.opening_paragraph.startswith("I am excited")

    def test_subject_line_stored(self):
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        assert "Stripe" in data.subject_line

    def test_tone_notes_optional(self):
        payload = {**_VALID_AI_PAYLOAD, "tone_notes": ""}
        data = CoverLetterData(**payload)
        assert data.tone_notes == ""

    def test_opening_too_short_raises(self):
        payload = {**_VALID_AI_PAYLOAD, "opening_paragraph": "Too short"}
        with pytest.raises(Exception):
            CoverLetterData(**payload)

    def test_subject_line_too_short_raises(self):
        payload = {**_VALID_AI_PAYLOAD, "subject_line": "Hi"}
        with pytest.raises(Exception):
            CoverLetterData(**payload)

    def test_all_paragraph_fields_present(self):
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        assert data.opening_paragraph
        assert data.body_paragraph_1
        assert data.body_paragraph_2
        assert data.closing_paragraph


# ---------------------------------------------------------------------------
# TestCoverLetter
# ---------------------------------------------------------------------------


class TestCoverLetter:
    def _make(self, **kwargs) -> CoverLetter:
        defaults = dict(
            company="Stripe",
            title="Backend Engineer",
            candidate_name="Ananya Sharma",
            candidate_email="ananya@example.com",
            content="Dear Hiring Manager,\n\nContent here.\n\nSincerely,\nAnanya",
            subject_line="Application — Backend Engineer at Stripe",
        )
        defaults.update(kwargs)
        return CoverLetter(**defaults)

    def test_str_returns_content(self):
        letter = self._make(content="Hello world")
        assert str(letter) == "Hello world"

    def test_is_saved_false_by_default(self):
        assert self._make().is_saved is False

    def test_is_saved_true_when_paths_set(self, tmp_path):
        letter = self._make()
        letter.txt_path = tmp_path / "letter.txt"
        letter.md_path = tmp_path / "letter.md"
        assert letter.is_saved is True

    def test_is_saved_false_with_only_txt_path(self, tmp_path):
        letter = self._make()
        letter.txt_path = tmp_path / "letter.txt"
        assert letter.is_saved is False

    def test_company_and_title_stored(self):
        letter = self._make(company="Uber", title="SRE")
        assert letter.company == "Uber"
        assert letter.title == "SRE"

    def test_paths_default_to_none(self):
        letter = self._make()
        assert letter.txt_path is None
        assert letter.md_path is None


# ---------------------------------------------------------------------------
# TestCoverLetterGenerator
# ---------------------------------------------------------------------------


class TestCoverLetterGenerator:
    def test_default_template_contains_opening_placeholder(self):
        assert "{opening_paragraph}" in DEFAULT_LETTER_TEMPLATE

    def test_default_template_contains_candidate_name_placeholder(self):
        assert "{candidate_name}" in DEFAULT_LETTER_TEMPLATE

    def test_custom_template_is_used(self):
        gen = CoverLetterGenerator(template="Hello {candidate_name} applying to {company}")
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        result = gen.assemble(data, "Stripe", "SWE", "Ananya", "a@b.com")
        assert "Hello Ananya applying to Stripe" == result

    def test_assemble_includes_company(self):
        gen = CoverLetterGenerator()
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        result = gen.assemble(data, "Stripe", "Backend Engineer", "Ananya", "a@b.com")
        assert "Stripe" in result

    def test_assemble_includes_opening_paragraph(self):
        gen = CoverLetterGenerator()
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        result = gen.assemble(data, "Stripe", "SWE", "Ananya", "a@b.com")
        assert data.opening_paragraph in result

    def test_assemble_includes_candidate_name(self):
        gen = CoverLetterGenerator()
        data = CoverLetterData(**_VALID_AI_PAYLOAD)
        result = gen.assemble(data, "Stripe", "SWE", "Ananya Sharma", "a@b.com")
        assert "Ananya Sharma" in result

    def test_assemble_basic_includes_company(self):
        gen = CoverLetterGenerator()
        result = gen.assemble_basic("Stripe", "SWE II", "Ananya", "a@b.com")
        assert "Stripe" in result

    def test_assemble_basic_includes_title(self):
        gen = CoverLetterGenerator()
        result = gen.assemble_basic("Stripe", "Backend Engineer", "Ananya", "a@b.com")
        assert "Backend Engineer" in result

    def test_assemble_basic_uses_resume_summary_when_provided(self):
        gen = CoverLetterGenerator()
        summary = "I built a payments processing system handling $1M/day."
        result = gen.assemble_basic("Stripe", "SWE", "Ananya", "a@b.com", resume_summary=summary)
        assert summary in result

    def test_to_markdown_adds_heading(self):
        md = CoverLetterGenerator.to_markdown("Content here.", "Stripe", "SWE II")
        assert md.startswith("# Cover Letter")
        assert "Stripe" in md
        assert "SWE II" in md


# ---------------------------------------------------------------------------
# TestCoverLetterAgentNoAI
# ---------------------------------------------------------------------------


class TestCoverLetterAgentNoAI:
    """Tests for client=None (template-only) mode — no mock, no API key."""

    @pytest.fixture
    def agent(self) -> CoverLetterAgent:
        return CoverLetterAgent(
            resume_text=_RESUME_TEXT,
            client=None,
            candidate_name="Ananya Sharma",
            candidate_email="ananya@example.com",
        )

    def test_generate_returns_cover_letter(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert isinstance(letter, CoverLetter)

    def test_generate_includes_company(self, agent):
        letter = agent.generate(_make_job(company="Stripe"), save=False)
        assert "Stripe" in letter.content

    def test_generate_includes_title(self, agent):
        letter = agent.generate(_make_job(title="Backend Engineer"), save=False)
        assert "Backend Engineer" in letter.content

    def test_generate_includes_candidate_name(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert "Ananya Sharma" in letter.content

    def test_generate_sets_subject_line(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert letter.subject_line
        assert len(letter.subject_line) > 5

    def test_generate_does_not_save_when_save_false(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert letter.is_saved is False

    def test_generate_saves_txt_file(self, tmp_path, agent):
        agent._output_dir = tmp_path
        letter = agent.generate(_make_job(), save=True)
        assert letter.txt_path is not None
        assert letter.txt_path.exists()

    def test_generate_saves_md_file(self, tmp_path, agent):
        agent._output_dir = tmp_path
        letter = agent.generate(_make_job(), save=True)
        assert letter.md_path is not None
        assert letter.md_path.exists()

    def test_md_file_contains_markdown_heading(self, tmp_path, agent):
        agent._output_dir = tmp_path
        letter = agent.generate(_make_job(company="Stripe"), save=True)
        md_content = letter.md_path.read_text(encoding="utf-8")
        assert md_content.startswith("# Cover Letter")
        assert "Stripe" in md_content


# ---------------------------------------------------------------------------
# TestCoverLetterAgentWithAI
# ---------------------------------------------------------------------------


class TestCoverLetterAgentWithAI:
    """Tests for the Gemini-backed path using a mocked client."""

    @pytest.fixture
    def agent(self) -> CoverLetterAgent:
        return CoverLetterAgent(
            resume_text=_RESUME_TEXT,
            client=_mock_client(_VALID_AI_PAYLOAD),
            candidate_name="Ananya Sharma",
            candidate_email="ananya@example.com",
        )

    def test_calls_generate_content_once(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(resume_text=_RESUME_TEXT, client=client)
        agent.generate(_make_job(), save=False)
        assert client.models.generate_content.call_count == 1

    def test_uses_correct_model(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(
            resume_text=_RESUME_TEXT, client=client, model="gemini-1.5-flash"
        )
        agent.generate(_make_job(), save=False)
        kwargs = client.models.generate_content.call_args
        assert kwargs[1]["model"] == "gemini-1.5-flash"

    def test_json_mode_is_enabled(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(resume_text=_RESUME_TEXT, client=client)
        agent.generate(_make_job(), save=False)
        kwargs = client.models.generate_content.call_args[1]
        assert kwargs["config"].response_mime_type == "application/json"
        assert kwargs["config"].response_schema == CoverLetterData

    def test_system_instruction_contains_resume(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(resume_text=_RESUME_TEXT, client=client)
        agent.generate(_make_job(), save=False)
        kwargs = client.models.generate_content.call_args[1]
        assert "Ananya Sharma" in kwargs["config"].system_instruction

    def test_user_message_contains_company(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(resume_text=_RESUME_TEXT, client=client)
        agent.generate(_make_job(company="Stripe"), save=False)
        kwargs = client.models.generate_content.call_args[1]
        assert "Stripe" in kwargs["contents"]

    def test_user_message_contains_title(self):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(resume_text=_RESUME_TEXT, client=client)
        agent.generate(_make_job(title="Backend Engineer"), save=False)
        kwargs = client.models.generate_content.call_args[1]
        assert "Backend Engineer" in kwargs["contents"]

    def test_returns_cover_letter_on_success(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert isinstance(letter, CoverLetter)

    def test_ai_content_in_letter(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert _VALID_AI_PAYLOAD["opening_paragraph"] in letter.content

    def test_subject_line_from_ai(self, agent):
        letter = agent.generate(_make_job(), save=False)
        assert letter.subject_line == _VALID_AI_PAYLOAD["subject_line"]

    def test_falls_back_to_template_on_empty_response(self):
        agent = CoverLetterAgent(
            resume_text=_RESUME_TEXT,
            client=_empty_response_client(),
            candidate_name="Ananya",
            candidate_email="a@b.com",
        )
        letter = agent.generate(_make_job(company="Stripe"), save=False)
        # Fallback still returns a usable letter
        assert isinstance(letter, CoverLetter)
        assert "Stripe" in letter.content

    def test_falls_back_to_template_on_exception(self):
        agent = CoverLetterAgent(
            resume_text=_RESUME_TEXT,
            client=_error_client(RuntimeError("network error")),
            candidate_name="Ananya",
            candidate_email="a@b.com",
        )
        letter = agent.generate(_make_job(company="Uber"), save=False)
        assert isinstance(letter, CoverLetter)
        assert "Uber" in letter.content

    def test_saves_files_when_save_true(self, tmp_path):
        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = CoverLetterAgent(
            resume_text=_RESUME_TEXT,
            client=client,
            output_dir=tmp_path,
        )
        letter = agent.generate(_make_job(), save=True)
        assert letter.txt_path is not None and letter.txt_path.exists()
        assert letter.md_path is not None and letter.md_path.exists()
