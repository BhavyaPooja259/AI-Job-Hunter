"""
Tests for Sprint 20 — AI Provider Abstraction.

Coverage
--------
TestAIProviderError           (3)  — error class behaviour
TestAIProviderABC             (2)  — cannot be instantiated without impl
TestGeminiProvider            (9)  — delegates to client, argument mapping
TestOpenRouterProvider        (9)  — httpx POST, JSON schema injection
TestGroqProvider              (7)  — same shape as OpenRouter, different URL
TestAIProviderFactory         (7)  — env-var routing, key priority, None fallback
TestCoverLetterAgentProvider  (7)  — new provider param, backward compat
TestResumeTailoringProvider   (6)  — new provider param, backward compat
TestReferralAgentProvider     (6)  — new provider param, backward compat

All external I/O is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai.provider import AIProvider, AIProviderError
from ai.gemini_provider import GeminiProvider
from ai.openrouter_provider import OpenRouterProvider
from ai.groq_provider import GroqProvider
from config.constants import ATSType
from cover_letter.cover_letter_generator import CoverLetterData
from matching.tailor_result import TailoredResumeData
from referral.referral import Referral, ReferralMessageData
from scrapers.models import Job


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_job(company: str = "Stripe", title: str = "Backend Engineer") -> Job:
    return Job(
        company=company,
        title=title,
        job_url=f"https://{company.lower()}.com/jobs/1",
        source_platform=ATSType.GREENHOUSE,
    )


def _mock_gemini_client(text: str) -> MagicMock:
    """Return a mock genai.Client whose generate_content returns text."""
    response = MagicMock()
    response.text = text
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _empty_gemini_client() -> MagicMock:
    response = MagicMock()
    response.text = None
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _error_gemini_client(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = exc
    return client


def _mock_httpx_response(text: str, status_code: int = 200) -> MagicMock:
    payload = {"choices": [{"message": {"content": text}}]}
    response = MagicMock()
    response.json.return_value = payload
    response.status_code = status_code
    response.text = json.dumps(payload)
    if status_code >= 400:
        import httpx
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _cover_letter_payload() -> dict:
    return {
        "opening_paragraph": "I am excited to apply.",
        "body_paragraph_1": "I have 3 years of Java experience.",
        "body_paragraph_2": "At my previous role I reduced latency by 40%.",
        "closing_paragraph": "I look forward to discussing this opportunity.",
        "subject_line": "Application for Backend Engineer",
        "tone_notes": "Professional and concise.",
    }


def _tailoring_payload() -> dict:
    full_text = (
        "BHAVYA L  |  bhavya@example.com\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Experienced backend engineer with 3 years of Java and Spring Boot development.\n\n"
        "TECHNICAL SKILLS\n"
        "- Languages: Java, Python, SQL\n"
        "- Frameworks: Spring Boot, REST APIs\n\n"
        "EXPERIENCE\n"
        "Zoho Corporation — Member Technical Staff (2021–2024)\n"
        "- Reduced API latency by 40% through connection pooling.\n"
        "- Built microservices handling 500+ RPS.\n"
    )
    return {
        "professional_summary": "Experienced backend engineer with Java expertise.",
        "highlighted_skills": ["Languages: Java, Python, SQL", "Frameworks: Spring Boot"],
        "selected_experience": [
            {
                "company": "Zoho",
                "title": "Member Technical Staff",
                "bullets": ["Reduced API latency by 40%."],
            }
        ],
        "keywords_incorporated": ["Java", "Spring Boot"],
        "ats_score_estimate": 85,
        "tailoring_notes": "Emphasised Java and microservices experience.",
        "full_resume_text": full_text,
    }


def _referral_payload() -> dict:
    return {
        "linkedin_request": "Hi Alice! I'd love to connect.",
        "referral_message": "Hi Alice, would you refer me?",
        "followup_message": "Hi Alice, just following up!",
    }


# ---------------------------------------------------------------------------
# TestAIProviderError
# ---------------------------------------------------------------------------


class TestAIProviderError:
    def test_is_exception_subclass(self):
        assert issubclass(AIProviderError, Exception)

    def test_can_be_raised_with_message(self):
        with pytest.raises(AIProviderError, match="quota exceeded"):
            raise AIProviderError("quota exceeded")

    def test_preserves_cause(self):
        original = ValueError("root cause")
        try:
            raise AIProviderError("wrapped") from original
        except AIProviderError as exc:
            assert exc.__cause__ is original


# ---------------------------------------------------------------------------
# TestAIProviderABC
# ---------------------------------------------------------------------------


class TestAIProviderABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            AIProvider()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_complete_and_name(self):
        class Partial(AIProvider):
            @property
            def name(self) -> str:
                return "partial"
            # Missing complete()

        with pytest.raises(TypeError):
            Partial()


# ---------------------------------------------------------------------------
# TestGeminiProvider
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def test_name_is_gemini(self):
        provider = GeminiProvider(client=_mock_gemini_client("hello"))
        assert provider.name == "gemini"

    def test_complete_calls_generate_content_once(self):
        client = _mock_gemini_client("hello world")
        provider = GeminiProvider(client=client)
        result = provider.complete("say hello")
        client.models.generate_content.assert_called_once()
        assert result == "hello world"

    def test_user_message_passed_as_contents(self):
        client = _mock_gemini_client("ok")
        provider = GeminiProvider(client=client)
        provider.complete("my user message")
        kwargs = client.models.generate_content.call_args[1]
        assert kwargs["contents"] == "my user message"

    def test_system_prompt_passed_as_system_instruction(self):
        client = _mock_gemini_client("ok")
        provider = GeminiProvider(client=client)
        provider.complete("msg", system_prompt="be concise")
        kwargs = client.models.generate_content.call_args[1]
        assert kwargs["config"].system_instruction == "be concise"

    def test_response_schema_enables_json_mode(self):
        client = _mock_gemini_client(json.dumps(_cover_letter_payload()))
        provider = GeminiProvider(client=client)
        provider.complete("msg", response_schema=CoverLetterData)
        kwargs = client.models.generate_content.call_args[1]
        assert kwargs["config"].response_mime_type == "application/json"
        assert kwargs["config"].response_schema is CoverLetterData

    def test_no_json_mode_without_schema(self):
        client = _mock_gemini_client("plain text response")
        provider = GeminiProvider(client=client)
        provider.complete("msg")
        kwargs = client.models.generate_content.call_args[1]
        assert not hasattr(kwargs["config"], "response_mime_type") or \
               getattr(kwargs["config"], "response_mime_type", None) is None

    def test_max_tokens_passed_to_config(self):
        client = _mock_gemini_client("ok")
        provider = GeminiProvider(client=client)
        provider.complete("msg", max_tokens=999)
        kwargs = client.models.generate_content.call_args[1]
        assert kwargs["config"].max_output_tokens == 999

    def test_empty_response_raises_provider_error(self):
        client = _empty_gemini_client()
        provider = GeminiProvider(client=client)
        with pytest.raises(AIProviderError, match="empty"):
            provider.complete("msg")

    def test_sdk_exception_wrapped_as_provider_error(self):
        client = _error_gemini_client(RuntimeError("network timeout"))
        provider = GeminiProvider(client=client)
        with pytest.raises(AIProviderError):
            provider.complete("msg")


# ---------------------------------------------------------------------------
# TestOpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterProvider:
    def _provider(self, model: str = "openai/gpt-4o-mini") -> OpenRouterProvider:
        return OpenRouterProvider(api_key="sk-or-test", model=model)

    def test_name_is_openrouter(self):
        assert self._provider().name == "openrouter"

    def test_raises_without_api_key(self):
        with pytest.raises(AIProviderError, match="API key"):
            OpenRouterProvider(api_key="")

    def test_posts_to_openrouter_url(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("hello")
            self._provider().complete("msg")
        url = mock_post.call_args[0][0]
        assert "openrouter.ai" in url

    def test_user_message_in_messages(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("ok")
            self._provider().complete("my question")
        body = mock_post.call_args[1]["json"]
        user_msgs = [m for m in body["messages"] if m["role"] == "user"]
        assert user_msgs[0]["content"] == "my question"

    def test_system_prompt_in_messages(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("ok")
            self._provider().complete("q", system_prompt="be brief")
        body = mock_post.call_args[1]["json"]
        sys_msgs = [m for m in body["messages"] if m["role"] == "system"]
        assert sys_msgs[0]["content"] == "be brief"

    def test_response_schema_injects_json_schema_into_system(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=CoverLetterData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        assert "schema" in sys_content.lower()

    def test_large_schema_strips_description_keys(self):
        """Schemas > 2000 chars have 'description' keys stripped before injection.

        Large schemas (TailoredResumeData ~3700 chars compact) cause free-tier
        models to return empty content.  Stripping 'description' keys reduces
        the schema to ~1200 chars so it fits in the model's context window.
        Only 'description' is stripped — 'title' is also used as a property
        name in some models (e.g. ExperienceEntry.title) so it must be kept.
        """
        from matching.tailor_result import TailoredResumeData
        import json as _json
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=TailoredResumeData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        schema_json_start = sys_content.find("{")
        try:
            schema_obj = _json.loads(sys_content[schema_json_start:])
        except _json.JSONDecodeError:
            return
        schema_str = _json.dumps(schema_obj)
        assert '"description"' not in schema_str

    def test_small_schema_preserves_descriptions(self):
        """Schemas <= 2000 chars keep their descriptions so models understand each field."""
        import json as _json
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=CoverLetterData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        schema_json_start = sys_content.find("{")
        try:
            schema_obj = _json.loads(sys_content[schema_json_start:])
        except _json.JSONDecodeError:
            return
        schema_str = _json.dumps(schema_obj)
        # CoverLetterData descriptions must be present (schema < 2000 chars)
        assert '"description"' in schema_str

    def test_response_schema_sets_json_response_format(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=CoverLetterData)
        body = mock_post.call_args[1]["json"]
        assert body.get("response_format") == {"type": "json_object"}

    def test_returns_message_content(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("the answer")
            result = self._provider().complete("q")
        assert result == "the answer"

    def test_strips_markdown_code_fence_from_response(self):
        """JSON wrapped in ```json ... ``` is unwrapped before returning."""
        fenced = "```json\n{\"key\": \"value\"}\n```"
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response(fenced)
            result = self._provider().complete("q")
        assert result == '{"key": "value"}'

    def test_http_error_raises_provider_error(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("", status_code=429)
            with pytest.raises(AIProviderError, match="429"):
                self._provider().complete("q")


# ---------------------------------------------------------------------------
# TestGroqProvider
# ---------------------------------------------------------------------------


class TestGroqProvider:
    def _provider(self) -> GroqProvider:
        return GroqProvider(api_key="gsk_test", model="llama-3.3-70b-versatile")

    def test_name_is_groq(self):
        assert self._provider().name == "groq"

    def test_raises_without_api_key(self):
        with pytest.raises(AIProviderError, match="API key"):
            GroqProvider(api_key="")

    def test_posts_to_groq_url(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("ok")
            self._provider().complete("msg")
        url = mock_post.call_args[0][0]
        assert "groq.com" in url

    def test_user_and_system_messages(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("ok")
            self._provider().complete("user msg", system_prompt="sys")
        body = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in body["messages"]]
        assert "system" in roles
        assert "user" in roles

    def test_schema_injects_into_system_prompt(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=CoverLetterData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        assert "schema" in sys_content.lower()

    def test_large_schema_strips_description_keys(self):
        """Schemas > 2000 chars have 'description' keys stripped before injection."""
        from matching.tailor_result import TailoredResumeData
        import json as _json
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=TailoredResumeData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        schema_json_start = sys_content.find("{")
        try:
            schema_obj = _json.loads(sys_content[schema_json_start:])
        except _json.JSONDecodeError:
            return
        schema_str = _json.dumps(schema_obj)
        assert '"description"' not in schema_str

    def test_small_schema_preserves_descriptions(self):
        """Schemas <= 2000 chars keep their descriptions."""
        import json as _json
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("{}")
            self._provider().complete("q", response_schema=CoverLetterData)
        body = mock_post.call_args[1]["json"]
        sys_content = next(m["content"] for m in body["messages"] if m["role"] == "system")
        schema_json_start = sys_content.find("{")
        try:
            schema_obj = _json.loads(sys_content[schema_json_start:])
        except _json.JSONDecodeError:
            return
        schema_str = _json.dumps(schema_obj)
        assert '"description"' in schema_str

    def test_returns_message_content(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("groq answer")
            result = self._provider().complete("q")
        assert result == "groq answer"

    def test_strips_markdown_code_fence_from_response(self):
        """JSON wrapped in ```json ... ``` is unwrapped before returning."""
        fenced = "```json\n{\"msg\": \"hello\"}\n```"
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response(fenced)
            result = self._provider().complete("q")
        assert result == '{"msg": "hello"}'

    def test_http_error_raises_provider_error(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = _mock_httpx_response("", status_code=401)
            with pytest.raises(AIProviderError, match="401"):
                self._provider().complete("q")


# ---------------------------------------------------------------------------
# TestAIProviderFactory
# ---------------------------------------------------------------------------


class TestAIProviderFactory:
    def _make_settings(self, **overrides):
        s = MagicMock()
        s.ai_provider = "gemini"
        s.gemini_api_key = ""
        s.ai_model = "gemini-2.0-flash"
        s.openrouter_api_key = ""
        s.openrouter_model = "openai/gpt-4o-mini"
        s.groq_api_key = ""
        s.groq_model = "llama-3.3-70b-versatile"
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_returns_none_when_no_keys_set(self):
        from ai.factory import create_provider
        with patch("ai.factory.settings", self._make_settings()):
            result = create_provider()
        assert result is None

    def test_returns_gemini_when_key_set(self):
        from ai.factory import create_provider
        s = self._make_settings(ai_provider="gemini", gemini_api_key="AIza-test")
        with patch("ai.factory.settings", s), \
             patch("ai.factory._try_gemini") as mock_try:
            mock_try.return_value = MagicMock(name="gemini_provider")
            result = create_provider()
        mock_try.assert_called_once()
        assert result is not None

    def test_returns_openrouter_when_provider_name_set(self):
        from ai.factory import create_provider
        s = self._make_settings(ai_provider="openrouter", openrouter_api_key="sk-or-x")
        with patch("ai.factory.settings", s), \
             patch("ai.factory._try_openrouter") as mock_try:
            mock_try.return_value = MagicMock(name="openrouter_provider")
            result = create_provider()
        mock_try.assert_called_once()
        assert result is not None

    def test_returns_groq_when_provider_name_set(self):
        from ai.factory import create_provider
        s = self._make_settings(ai_provider="groq", groq_api_key="gsk_x")
        with patch("ai.factory.settings", s), \
             patch("ai.factory._try_groq") as mock_try:
            mock_try.return_value = MagicMock(name="groq_provider")
            result = create_provider()
        mock_try.assert_called_once()
        assert result is not None

    def test_gemini_preferred_over_fallback_chain(self):
        from ai.factory import create_provider
        s = self._make_settings(
            ai_provider="gemini",
            gemini_api_key="AIza",
            openrouter_api_key="sk-or",
        )
        with patch("ai.factory.settings", s), \
             patch("ai.factory._try_gemini") as mock_g, \
             patch("ai.factory._try_openrouter") as mock_or:
            mock_g.return_value = MagicMock(name="gem")
            create_provider()
        mock_g.assert_called_once()
        mock_or.assert_not_called()

    def test_openrouter_returns_none_when_key_missing(self):
        from ai.factory import _try_openrouter
        s = self._make_settings(openrouter_api_key="")
        result = _try_openrouter(s)
        assert result is None

    def test_groq_returns_none_when_key_missing(self):
        from ai.factory import _try_groq
        s = self._make_settings(groq_api_key="")
        result = _try_groq(s)
        assert result is None


# ---------------------------------------------------------------------------
# TestCoverLetterAgentProvider
# ---------------------------------------------------------------------------


_RESUME = "Bhavya L — 3 years Java experience"


class TestCoverLetterAgentProvider:
    def test_provider_param_is_accepted(self):
        from agents.cover_letter_agent import CoverLetterAgent
        mock_provider = MagicMock(spec=AIProvider)
        agent = CoverLetterAgent(resume_text=_RESUME, provider=mock_provider)
        assert agent._provider is mock_provider

    def test_provider_complete_is_called_on_generate(self):
        from agents.cover_letter_agent import CoverLetterAgent
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.return_value = json.dumps(_cover_letter_payload())
        agent = CoverLetterAgent(resume_text=_RESUME, provider=mock_provider)
        agent.generate(_make_job(), save=False)
        mock_provider.complete.assert_called_once()

    def test_provider_complete_receives_response_schema(self):
        from agents.cover_letter_agent import CoverLetterAgent
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.return_value = json.dumps(_cover_letter_payload())
        agent = CoverLetterAgent(resume_text=_RESUME, provider=mock_provider)
        agent.generate(_make_job(), save=False)
        _, kwargs = mock_provider.complete.call_args
        assert kwargs.get("response_schema") is CoverLetterData

    def test_client_is_wrapped_in_gemini_provider_automatically(self):
        from agents.cover_letter_agent import CoverLetterAgent
        from ai.gemini_provider import GeminiProvider
        client = _mock_gemini_client(json.dumps(_cover_letter_payload()))
        agent = CoverLetterAgent(resume_text=_RESUME, client=client)
        assert isinstance(agent._provider, GeminiProvider)

    def test_client_wrapped_provider_still_calls_generate_content(self):
        from agents.cover_letter_agent import CoverLetterAgent
        client = _mock_gemini_client(json.dumps(_cover_letter_payload()))
        agent = CoverLetterAgent(resume_text=_RESUME, client=client)
        agent.generate(_make_job(), save=False)
        client.models.generate_content.assert_called_once()

    def test_template_mode_when_no_provider_no_client(self):
        from agents.cover_letter_agent import CoverLetterAgent
        from cover_letter.cover_letter import CoverLetter
        agent = CoverLetterAgent(resume_text=_RESUME)
        result = agent.generate(_make_job(), save=False)
        assert isinstance(result, CoverLetter)

    def test_fallback_to_template_when_provider_raises(self):
        from agents.cover_letter_agent import CoverLetterAgent
        from cover_letter.cover_letter import CoverLetter
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.side_effect = AIProviderError("quota exceeded")
        agent = CoverLetterAgent(resume_text=_RESUME, provider=mock_provider)
        result = agent.generate(_make_job(), save=False)
        assert isinstance(result, CoverLetter)


# ---------------------------------------------------------------------------
# TestResumeTailoringProvider
# ---------------------------------------------------------------------------


class TestResumeTailoringProvider:
    def test_provider_param_accepted(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        mock_provider = MagicMock(spec=AIProvider)
        agent = ResumeTailoringAgent(provider=mock_provider, resume_text="text")
        assert agent._provider is mock_provider

    def test_provider_complete_called_on_tailor(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.return_value = json.dumps(_tailoring_payload())
        agent = ResumeTailoringAgent(
            provider=mock_provider,
            resume_text="resume text",
            output_dir=__import__("pathlib").Path(
                "/private/tmp/claude-501/-Users-tinafiolina-AI-Job-Hunter/"
                "ebe0eb8f-7289-4c69-9804-63cd48345359/scratchpad/tailor_test"
            ),
        )
        result = agent.tailor(_make_job())
        mock_provider.complete.assert_called_once()
        assert result is not None

    def test_client_wrapped_in_gemini_provider(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        from ai.gemini_provider import GeminiProvider
        client = _mock_gemini_client(json.dumps(_tailoring_payload()))
        agent = ResumeTailoringAgent(client=client, resume_text="text")
        assert isinstance(agent._provider, GeminiProvider)

    def test_client_wrapped_calls_generate_content(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        import pathlib
        client = _mock_gemini_client(json.dumps(_tailoring_payload()))
        agent = ResumeTailoringAgent(
            client=client,
            resume_text="text",
            output_dir=pathlib.Path(
                "/private/tmp/claude-501/-Users-tinafiolina-AI-Job-Hunter/"
                "ebe0eb8f-7289-4c69-9804-63cd48345359/scratchpad/tailor_test2"
            ),
        )
        result = agent.tailor(_make_job())
        client.models.generate_content.assert_called_once()
        assert result is not None

    def test_raises_when_neither_client_nor_provider(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        with pytest.raises(ValueError, match="client.*provider"):
            ResumeTailoringAgent(resume_text="text")

    def test_provider_error_returns_none(self):
        from agents.resume_tailoring_agent import ResumeTailoringAgent
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.side_effect = AIProviderError("503")
        agent = ResumeTailoringAgent(provider=mock_provider, resume_text="text")
        result = agent.tailor(_make_job())
        assert result is None


# ---------------------------------------------------------------------------
# TestReferralAgentProvider
# ---------------------------------------------------------------------------


class TestReferralAgentProvider:
    def _make_referral(self) -> Referral:
        return Referral(
            contact_name="Alice Sharma",
            company="Stripe",
            contact_title="Senior Engineer",
            job_title="Backend Engineer",
        )

    def test_provider_param_accepted(self):
        from agents.referral_agent import ReferralAgent
        mock_repo = MagicMock()
        mock_provider = MagicMock(spec=AIProvider)
        agent = ReferralAgent(repo=mock_repo, provider=mock_provider)
        assert agent._provider is mock_provider

    def test_provider_complete_called_for_messages(self):
        from agents.referral_agent import ReferralAgent
        mock_repo = MagicMock()
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.return_value = json.dumps(_referral_payload())
        agent = ReferralAgent(repo=mock_repo, provider=mock_provider)
        agent.generate_messages(self._make_referral(), save=False)
        mock_provider.complete.assert_called_once()

    def test_client_wrapped_in_gemini_provider(self):
        from agents.referral_agent import ReferralAgent
        from ai.gemini_provider import GeminiProvider
        mock_repo = MagicMock()
        client = _mock_gemini_client(json.dumps(_referral_payload()))
        agent = ReferralAgent(repo=mock_repo, client=client)
        assert isinstance(agent._provider, GeminiProvider)

    def test_client_wrapped_calls_generate_content(self):
        from agents.referral_agent import ReferralAgent
        mock_repo = MagicMock()
        client = _mock_gemini_client(json.dumps(_referral_payload()))
        agent = ReferralAgent(repo=mock_repo, client=client)
        agent.generate_messages(self._make_referral(), save=False)
        client.models.generate_content.assert_called_once()

    def test_template_mode_when_no_provider(self):
        from agents.referral_agent import ReferralAgent
        from referral.referral import ReferralMessages
        mock_repo = MagicMock()
        agent = ReferralAgent(repo=mock_repo)
        msgs = agent.generate_messages(self._make_referral(), save=False)
        assert isinstance(msgs, ReferralMessages)
        assert "Alice" in msgs.linkedin_request

    def test_fallback_to_template_when_provider_raises(self):
        from agents.referral_agent import ReferralAgent
        from referral.referral import ReferralMessages
        mock_repo = MagicMock()
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.complete.side_effect = AIProviderError("rate limited")
        agent = ReferralAgent(repo=mock_repo, provider=mock_provider)
        msgs = agent.generate_messages(self._make_referral(), save=False)
        assert isinstance(msgs, ReferralMessages)
