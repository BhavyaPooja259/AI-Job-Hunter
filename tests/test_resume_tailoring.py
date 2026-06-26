"""
Sprint 13 — ResumeTailoringAgent unit tests.

All tests use a mocked Gemini client and an injected resume_text string,
so neither a real API key nor pdfplumber is required to run the suite.

Design decisions
----------------
Mock at the boundary
    We mock genai.Client().models.generate_content, not internal helpers.
    This means the tests exercise the full call path: system prompt assembly,
    user message building, JSON parsing, Pydantic validation, and file I/O.

resume_text injection
    The ResumeTailoringAgent constructor accepts resume_text=<str> to bypass
    PDF extraction.  Tests pass SAMPLE_RESUME_TEXT (a condensed version of
    the real resume) so they run without pdfplumber and without the PDF file.

tmp_path for output
    Tests that check file saving pass pytest's tmp_path fixture as output_dir.
    This avoids polluting resume/tailored/ and makes cleanup automatic.

Gemini mock shape
    The mock returns a response where response.text = json.dumps(payload).
    This mirrors exactly what the real Gemini API returns when
    response_mime_type="application/json" is set.

Run from the project root:
    python -m pytest tests/test_resume_tailoring.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.constants import ATSType
from matching.ai_result import AIMatchResult
from matching.matcher import MatchResult
from matching.tailor_result import ExperienceEntry, TailoredResumeData
from scrapers.models import Job
from agents.resume_tailoring_agent import ResumeTailoringAgent, TailoringResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME_TEXT = """
Bhavya L
bhavya2592001@gmail.com | +91 9150313881

SUMMARY
Software Engineer with 3 years of experience building scalable enterprise
applications and delivering production-grade solutions for BFSI platforms.

TECHNICAL SKILLS
Languages : Java, Kotlin, Python
Frontend & Backend : React JS, Spring Boot, REST APIs, Microservices
System Design & Architecture : LLD, API Design, Design Patterns
Data & Caching : MySQL, PostgreSQL, MongoDB, Redis
Cloud & Engineering Practices : AWS (EC2, S3), CI/CD, JUnit, Mockito, Git

EXPERIENCE
Zoho — Member Technical Staff | June 2023–June 2026 | Chennai
- Owned end-to-end delivery of enterprise CRM features for BFSI clients.
- Optimized database performance, reducing repeated queries by 60%.
- Architected reusable and configurable platform components.

Amazon — SDE Intern | Jul 2022 – Sep 2022 | Chennai
- Developed CSV to CBE Converter, reducing data processing time by 40%.
- Developed unit tests with JUnit and Mockito, achieving 98% code coverage.

PROJECTS
Distributed Rate Limiter
- Designed Redis-based distributed rate limiter using the Token Bucket algorithm.

EDUCATION
Anna University (R.M.D.) | B.E. in ECE | 2019–2023 | CGPA: 8.9

ACHIEVEMENTS
- Solved 500+ DSA problems and earned a 4-Star rating on LeetCode.
""".strip()


VALID_TAILOR_PAYLOAD = {
    "professional_summary": (
        "Backend-focused Software Engineer with 3 years of experience building "
        "scalable Java and Spring Boot microservices for enterprise BFSI platforms. "
        "Proven track record of optimizing system performance and delivering "
        "production-grade REST API solutions end-to-end."
    ),
    "highlighted_skills": [
        "Languages: Java, Kotlin, Python",
        "Frontend & Backend: React JS, Spring Boot, REST APIs, Microservices",
        "Data & Caching: MySQL, PostgreSQL, MongoDB, Redis",
        "System Design & Architecture: LLD, API Design, Design Patterns",
        "Cloud & Engineering Practices: AWS (EC2, S3), CI/CD, JUnit, Mockito, Git",
    ],
    "selected_experience": [
        {
            "company": "Zoho",
            "title": "Member Technical Staff",
            "bullets": [
                "Optimized database performance, reducing repeated queries by 60%.",
                "Owned end-to-end delivery of enterprise CRM features for BFSI clients.",
                "Architected reusable and configurable platform components.",
            ],
        },
        {
            "company": "Amazon",
            "title": "SDE Intern",
            "bullets": [
                "Developed unit tests with JUnit and Mockito, achieving 98% code coverage.",
                "Developed CSV to CBE Converter, reducing data processing time by 40%.",
            ],
        },
    ],
    "keywords_incorporated": ["Java", "Spring Boot", "microservices", "REST APIs", "PostgreSQL"],
    "ats_score_estimate": 88,
    "tailoring_notes": (
        "Emphasized Java, Spring Boot, and microservices experience. "
        "Led with database optimization bullet to align with backend focus."
    ),
    "full_resume_text": (
        "BHAVYA L\n"
        "bhavya2592001@gmail.com | +91 9150313881\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Backend-focused Software Engineer with 3 years of experience...\n\n"
        "TECHNICAL SKILLS\n"
        "Languages: Java, Kotlin, Python\n"
        "Frontend & Backend: React JS, Spring Boot, REST APIs, Microservices\n\n"
        "EXPERIENCE\n"
        "Zoho — Member Technical Staff | June 2023–June 2026\n"
        "- Optimized database performance, reducing repeated queries by 60%.\n"
        "- Owned end-to-end delivery of enterprise CRM features.\n\n"
        "Amazon — SDE Intern | Jul 2022 – Sep 2022\n"
        "- Developed unit tests with JUnit and Mockito, achieving 98% code coverage.\n\n"
        "PROJECTS\n"
        "Distributed Rate Limiter\n"
        "- Designed Redis-based distributed rate limiter.\n\n"
        "EDUCATION\n"
        "Anna University (R.M.D.) | B.E. in ECE | 2019-2023 | CGPA: 8.9\n\n"
        "ACHIEVEMENTS\n"
        "- Solved 500+ DSA problems and earned a 4-Star rating on LeetCode.\n"
    ),
}


def _make_job(
    company: str = "Stripe",
    title: str = "Backend Engineer",
    job_url: str = "https://stripe.com/jobs/1",
    description: str = "We build scalable Java microservices with Spring Boot and REST APIs.",
    requirements: str = "5+ years Java. Spring Boot, PostgreSQL, microservices required.",
    location: str = "San Francisco, CA",
    department: str | None = "Engineering",
) -> Job:
    return Job(
        company=company,
        title=title,
        job_url=job_url,
        location=location,
        source_platform=ATSType.GREENHOUSE,
        description=description,
        requirements=requirements,
        department=department,
    )


def _make_ai_result(**overrides) -> AIMatchResult:
    defaults = {
        "score": 85,
        "confidence": "high",
        "recommendation": "apply",
        "summary": "Strong Java backend match.",
        "strengths": ["Java", "Spring Boot", "REST APIs"],
        "missing_skills": ["Kubernetes"],
        "interview_difficulty": "high",
    }
    defaults.update(overrides)
    return AIMatchResult.model_validate(defaults)


def _make_rule_result(score: int = 72) -> MatchResult:
    return MatchResult(
        score=score,
        matched_skills=["java", "spring boot", "rest", "sql"],
        missing_skills=["docker"],
        reasons=["Role match +40", "Skills +25"],
    )


def _mock_client(payload: dict) -> MagicMock:
    """Mock Gemini client that returns a JSON response with payload."""
    response = MagicMock()
    response.text = json.dumps(payload)

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _error_client(exc: Exception = RuntimeError("api error")) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = exc
    return client


def _empty_response_client() -> MagicMock:
    """Mock client whose response has no text (simulates empty/failed Gemini response)."""
    response = MagicMock()
    response.text = None

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------
# TailoredResumeData model tests
# ---------------------------------------------------------------------------

class TestTailoredResumeData:

    def test_valid_payload_parses(self):
        data = TailoredResumeData.model_validate(VALID_TAILOR_PAYLOAD)
        assert data.ats_score_estimate == 88
        assert len(data.highlighted_skills) == 5
        assert len(data.selected_experience) == 2
        assert data.selected_experience[0].company == "Zoho"
        assert len(data.full_resume_text) > 100

    def test_ats_score_below_zero_raises(self):
        with pytest.raises(Exception):
            TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "ats_score_estimate": -1})

    def test_ats_score_above_100_raises(self):
        with pytest.raises(Exception):
            TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "ats_score_estimate": 101})

    def test_ats_score_boundary_zero(self):
        data = TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "ats_score_estimate": 0})
        assert data.ats_score_estimate == 0

    def test_ats_score_boundary_100(self):
        data = TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "ats_score_estimate": 100})
        assert data.ats_score_estimate == 100

    def test_short_summary_raises(self):
        with pytest.raises(Exception):
            TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "professional_summary": "Hi"})

    def test_empty_highlighted_skills_raises(self):
        with pytest.raises(Exception):
            TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "highlighted_skills": []})

    def test_short_full_resume_text_raises(self):
        with pytest.raises(Exception):
            TailoredResumeData.model_validate({**VALID_TAILOR_PAYLOAD, "full_resume_text": "Too short"})

    def test_empty_keywords_is_valid(self):
        data = TailoredResumeData.model_validate(
            {**VALID_TAILOR_PAYLOAD, "keywords_incorporated": []}
        )
        assert data.keywords_incorporated == []

    def test_empty_selected_experience_is_valid(self):
        data = TailoredResumeData.model_validate(
            {**VALID_TAILOR_PAYLOAD, "selected_experience": []}
        )
        assert data.selected_experience == []


class TestExperienceEntry:

    def test_entry_with_bullets_parses(self):
        entry = ExperienceEntry.model_validate({
            "company": "Zoho",
            "title": "Member Technical Staff",
            "bullets": ["Built feature X.", "Optimized query performance by 60%."],
        })
        assert entry.company == "Zoho"
        assert len(entry.bullets) == 2

    def test_entry_without_bullets_defaults_empty(self):
        entry = ExperienceEntry.model_validate({
            "company": "Amazon",
            "title": "SDE Intern",
        })
        assert entry.bullets == []


# ---------------------------------------------------------------------------
# TailoringResult tests
# ---------------------------------------------------------------------------

class TestTailoringResult:

    def _make_result(self, tmp_path: Path) -> TailoringResult:
        data = TailoredResumeData.model_validate(VALID_TAILOR_PAYLOAD)
        text_path = tmp_path / "stripe_20260626_120000.txt"
        analysis_path = tmp_path / "stripe_20260626_120000_analysis.json"
        text_path.write_text(data.full_resume_text)
        analysis_path.write_text(data.model_dump_json())
        return TailoringResult(
            job=_make_job(),
            data=data,
            text_path=text_path,
            analysis_path=analysis_path,
        )

    def test_str_includes_title_and_company(self, tmp_path):
        result = self._make_result(tmp_path)
        s = str(result)
        assert "Backend Engineer" in s
        assert "Stripe" in s

    def test_str_includes_ats_estimate(self, tmp_path):
        result = self._make_result(tmp_path)
        assert "88" in str(result)

    def test_paths_point_to_real_files(self, tmp_path):
        result = self._make_result(tmp_path)
        assert result.text_path.exists()
        assert result.analysis_path.exists()


# ---------------------------------------------------------------------------
# ResumeTailoringAgent — constructor
# ---------------------------------------------------------------------------

class TestAgentConstructor:

    def test_accepts_resume_text_directly(self):
        """Constructor with resume_text= bypasses PDF extraction entirely."""
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
        )
        assert SAMPLE_RESUME_TEXT in agent._resume_text

    def test_resume_text_stored_intact(self):
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
        )
        assert agent._resume_text == SAMPLE_RESUME_TEXT


# ---------------------------------------------------------------------------
# ResumeTailoringAgent.tailor — happy path
# ---------------------------------------------------------------------------

class TestTailorHappyPath:

    def test_returns_tailoring_result(self, tmp_path):
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        result = agent.tailor(_make_job())
        assert isinstance(result, TailoringResult)

    def test_result_data_is_validated(self, tmp_path):
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        result = agent.tailor(_make_job())
        assert isinstance(result.data, TailoredResumeData)
        assert result.data.ats_score_estimate == 88

    def test_api_called_once(self, tmp_path):
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        agent.tailor(_make_job())
        client.models.generate_content.assert_called_once()

    def test_json_mode_is_enabled(self, tmp_path):
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        agent.tailor(_make_job())
        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["config"].response_mime_type == "application/json"
        assert kwargs["config"].response_schema == TailoredResumeData

    def test_model_is_passed_to_api(self, tmp_path):
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
            model="gemini-1.5-flash",
        )
        agent.tailor(_make_job())
        assert client.models.generate_content.call_args.kwargs["model"] == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# Prompt content verification
# ---------------------------------------------------------------------------

class TestPromptContent:

    def _get_system(self, tmp_path, **agent_kwargs) -> str:
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
            **agent_kwargs,
        )
        agent.tailor(_make_job())
        return client.models.generate_content.call_args.kwargs["config"].system_instruction

    def _get_user_msg(self, tmp_path, job=None, **tailor_kwargs) -> str:
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        agent.tailor(job or _make_job(), **tailor_kwargs)
        return client.models.generate_content.call_args.kwargs["contents"]

    def test_system_prompt_contains_resume_text(self, tmp_path):
        system = self._get_system(tmp_path)
        assert "SAMPLE_RESUME_TEXT" not in system  # variable name not leaked
        assert "Zoho" in system
        assert "Amazon" in system
        assert "98% code coverage" in system

    def test_system_prompt_contains_never_fabricate_rule(self, tmp_path):
        system = self._get_system(tmp_path)
        # Key rule must be present so the model knows the constraint
        assert "NOT" in system or "MUST NOT" in system
        assert "invent" in system.lower() or "fabricate" in system.lower()

    def test_user_message_contains_company(self, tmp_path):
        msg = self._get_user_msg(tmp_path, job=_make_job(company="Rippling"))
        assert "Rippling" in msg

    def test_user_message_contains_job_title(self, tmp_path):
        msg = self._get_user_msg(tmp_path, job=_make_job(title="Platform Engineer"))
        assert "Platform Engineer" in msg

    def test_user_message_contains_description(self, tmp_path):
        job = _make_job(description="We need Java and Kafka expertise for our pipeline.")
        msg = self._get_user_msg(tmp_path, job=job)
        assert "Kafka" in msg

    def test_user_message_contains_requirements(self, tmp_path):
        job = _make_job(requirements="Kubernetes and Helm chart experience required.")
        msg = self._get_user_msg(tmp_path, job=job)
        assert "Kubernetes" in msg

    def test_ai_result_strengths_in_user_message(self, tmp_path):
        ai = _make_ai_result(strengths=["Java microservices", "Spring Boot REST"])
        msg = self._get_user_msg(tmp_path, ai_result=ai)
        assert "Java microservices" in msg

    def test_ai_result_gaps_in_user_message(self, tmp_path):
        ai = _make_ai_result(missing_skills=["Kubernetes", "Helm"])
        msg = self._get_user_msg(tmp_path, ai_result=ai)
        assert "Kubernetes" in msg

    def test_no_ai_result_still_works(self, tmp_path):
        """tailor() works fine when ai_result is None."""
        msg = self._get_user_msg(tmp_path, ai_result=None)
        assert "TARGET JOB" in msg  # still produces a valid message

    def test_rule_result_matched_skills_in_user_message(self, tmp_path):
        rule = _make_rule_result()
        msg = self._get_user_msg(tmp_path, rule_result=rule)
        assert "java" in msg.lower() or "spring boot" in msg.lower()

    def test_long_description_truncated(self, tmp_path):
        long_desc = "X" * 5000
        job = _make_job(description=long_desc)
        msg = self._get_user_msg(tmp_path, job=job)
        assert "truncated" in msg
        assert long_desc not in msg

    def test_none_description_handled(self, tmp_path):
        job = _make_job(description=None, requirements=None)
        msg = self._get_user_msg(tmp_path, job=job)
        # Should not raise; description/requirements sections simply absent
        assert "TARGET JOB" in msg


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestTailorFailures:

    def test_returns_none_on_api_exception(self, tmp_path):
        agent = ResumeTailoringAgent(
            client=_error_client(RuntimeError("network timeout")),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        result = agent.tailor(_make_job())
        assert result is None

    def test_returns_none_if_response_text_is_empty(self, tmp_path):
        agent = ResumeTailoringAgent(
            client=_empty_response_client(),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        result = agent.tailor(_make_job())
        assert result is None

    def test_returns_none_on_pydantic_validation_failure(self, tmp_path):
        bad_payload = {**VALID_TAILOR_PAYLOAD, "ats_score_estimate": 999}
        agent = ResumeTailoringAgent(
            client=_mock_client(bad_payload),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        result = agent.tailor(_make_job())
        assert result is None

    def test_never_raises_on_any_exception(self, tmp_path):
        """tailor() must not propagate exceptions."""
        for exc in [RuntimeError("oops"), ValueError("bad json"), Exception("generic")]:
            agent = ResumeTailoringAgent(
                client=_error_client(exc),
                resume_text=SAMPLE_RESUME_TEXT,
                output_dir=tmp_path,
            )
            result = agent.tailor(_make_job())
            assert result is None


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestFileOutput:

    def _tailor(self, tmp_path, company="Stripe") -> TailoringResult:
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        return agent.tailor(_make_job(company=company))

    def test_text_file_is_created(self, tmp_path):
        result = self._tailor(tmp_path)
        assert result.text_path.exists()

    def test_analysis_json_is_created(self, tmp_path):
        result = self._tailor(tmp_path)
        assert result.analysis_path.exists()

    def test_text_file_contains_resume_content(self, tmp_path):
        result = self._tailor(tmp_path)
        content = result.text_path.read_text(encoding="utf-8")
        assert "BHAVYA L" in content

    def test_analysis_json_is_valid(self, tmp_path):
        result = self._tailor(tmp_path)
        parsed = json.loads(result.analysis_path.read_text(encoding="utf-8"))
        assert "ats_score_estimate" in parsed
        assert parsed["ats_score_estimate"] == 88

    def test_output_filename_includes_company(self, tmp_path):
        result = self._tailor(tmp_path, company="Rippling")
        assert "rippling" in result.text_path.name

    def test_output_dir_is_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "output"
        assert not nested.exists()
        agent = ResumeTailoringAgent(
            client=_mock_client(VALID_TAILOR_PAYLOAD),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=nested,
        )
        agent.tailor(_make_job())
        assert nested.exists()

    def test_text_and_analysis_share_same_stem(self, tmp_path):
        result = self._tailor(tmp_path)
        assert result.text_path.stem == result.analysis_path.stem.replace("_analysis", "")

    def test_text_file_encoding_is_utf8(self, tmp_path):
        result = self._tailor(tmp_path)
        # Must be readable as UTF-8 without errors
        content = result.text_path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_analysis_json_roundtrips_through_model(self, tmp_path):
        result = self._tailor(tmp_path)
        raw = result.analysis_path.read_text(encoding="utf-8")
        recovered = TailoredResumeData.model_validate_json(raw)
        assert recovered.ats_score_estimate == result.data.ats_score_estimate
        assert recovered.keywords_incorporated == result.data.keywords_incorporated

    def test_tailor_fails_gracefully_leaves_no_partial_file(self, tmp_path):
        """If Gemini fails, no file should be saved."""
        agent = ResumeTailoringAgent(
            client=_error_client(),
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        agent.tailor(_make_job())
        # Nothing written to the output directory
        assert list(tmp_path.iterdir()) == []

    def test_two_tails_for_same_job_produce_separate_files(self, tmp_path):
        """Each call to tailor() creates a new timestamped file."""
        client = _mock_client(VALID_TAILOR_PAYLOAD)
        agent = ResumeTailoringAgent(
            client=client,
            resume_text=SAMPLE_RESUME_TEXT,
            output_dir=tmp_path,
        )
        import time
        r1 = agent.tailor(_make_job())
        time.sleep(1)  # ensure different timestamp in filename
        r2 = agent.tailor(_make_job())
        # Both results are valid
        assert r1 is not None and r2 is not None
        # Files are distinct (different timestamps in name)
        assert r1.text_path != r2.text_path
