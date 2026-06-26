"""
Sprint 12 — RankingAgent unit tests.

All tests use a mocked Anthropic client; no real API calls are made.

Design principles
-----------------
Mock at the boundary: we mock `anthropic.Anthropic().messages.create`, not
internal methods.  This means the tests are verifying the same code path that
runs in production — they exercise prompt building, tool-call parsing,
Pydantic validation, caching, and failure handling.

The mock client always returns a properly shaped tool-use response:
  response.content = [ToolUseBlock(type="tool_use", input={...})]
  response.stop_reason = "tool_use"

This mirrors exactly what the real Anthropic API returns when
tool_choice={"type": "tool", "name": "submit_job_ranking"} is set.

Run from the project root:
    python -m pytest tests/test_ai_ranking.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.constants import ATSType
from matching.ai_result import AIMatchResult
from matching.profile import DEFAULT_PROFILE, UserProfile
from scrapers.models import Job
from agents.ranking_agent import RankedJob, RankingAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(
    title: str = "Software Engineer II",
    company: str = "Stripe",
    job_url: str = "https://stripe.com/jobs/1",
    description: str = "We use Java, Spring Boot, REST APIs, and microservices.",
    requirements: str = "5+ years Java. Spring Boot required.",
    location: str = "Remote",
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


def _mock_client(tool_input: dict) -> MagicMock:
    """Return a mock Anthropic client that always returns `tool_input` as the tool call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_job_ranking"
    block.input = tool_input

    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _error_client(exc: Exception) -> MagicMock:
    """Return a mock Anthropic client that always raises `exc`."""
    client = MagicMock()
    client.messages.create.side_effect = exc
    return client


def _no_tool_client() -> MagicMock:
    """Return a mock client whose response contains no tool_use block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I think this job is a good fit."

    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = "end_turn"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# Minimal valid AI response payload.
_VALID_PAYLOAD = {
    "score": 82,
    "confidence": "high",
    "recommendation": "apply",
    "summary": "Strong Java backend match with relevant microservices experience.",
    "strengths": ["Java expertise", "Spring Boot", "REST APIs"],
    "missing_skills": ["Kubernetes"],
    "interview_difficulty": "high",
}


# ---------------------------------------------------------------------------
# AIMatchResult model tests
# ---------------------------------------------------------------------------

class TestAIMatchResult:

    def test_valid_payload_parses(self):
        result = AIMatchResult.model_validate(_VALID_PAYLOAD)
        assert result.score == 82
        assert result.confidence == "high"
        assert result.recommendation == "apply"
        assert "Java expertise" in result.strengths
        assert result.interview_difficulty == "high"

    def test_score_below_zero_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "score": -1})

    def test_score_above_100_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "score": 101})

    def test_invalid_recommendation_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "recommendation": "maybe"})

    def test_invalid_confidence_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "confidence": "uncertain"})

    def test_invalid_interview_difficulty_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "interview_difficulty": "extreme"})

    def test_blank_summary_raises(self):
        with pytest.raises(Exception):
            AIMatchResult.model_validate({**_VALID_PAYLOAD, "summary": "   "})

    def test_empty_strengths_list_is_valid(self):
        result = AIMatchResult.model_validate({**_VALID_PAYLOAD, "strengths": []})
        assert result.strengths == []

    def test_empty_missing_skills_list_is_valid(self):
        result = AIMatchResult.model_validate({**_VALID_PAYLOAD, "missing_skills": []})
        assert result.missing_skills == []

    def test_score_boundary_zero(self):
        result = AIMatchResult.model_validate({**_VALID_PAYLOAD, "score": 0})
        assert result.score == 0

    def test_score_boundary_100(self):
        result = AIMatchResult.model_validate({**_VALID_PAYLOAD, "score": 100})
        assert result.score == 100


# ---------------------------------------------------------------------------
# RankedJob tests
# ---------------------------------------------------------------------------

class TestRankedJob:

    def _make_rule_result(self, score: int = 60):
        from matching.matcher import MatchResult
        return MatchResult(
            score=score,
            matched_skills=["java"],
            missing_skills=["docker"],
            reasons=["Role match +40"],
        )

    def test_best_score_uses_ai_when_available(self):
        ai = AIMatchResult.model_validate({**_VALID_PAYLOAD, "score": 90})
        ranked = RankedJob(
            job=_make_job(),
            rule_result=self._make_rule_result(60),
            ai_result=ai,
        )
        assert ranked.best_score == 90

    def test_best_score_falls_back_to_rule_when_no_ai(self):
        ranked = RankedJob(
            job=_make_job(),
            rule_result=self._make_rule_result(55),
            ai_result=None,
        )
        assert ranked.best_score == 55

    def test_recommendation_from_ai_result(self):
        ai = AIMatchResult.model_validate({**_VALID_PAYLOAD, "recommendation": "consider"})
        ranked = RankedJob(
            job=_make_job(),
            rule_result=self._make_rule_result(50),
            ai_result=ai,
        )
        assert ranked.recommendation == "consider"

    def test_recommendation_unknown_when_no_ai(self):
        ranked = RankedJob(
            job=_make_job(),
            rule_result=self._make_rule_result(50),
            ai_result=None,
        )
        assert ranked.recommendation == "unknown"

    def test_str_includes_title_and_company(self):
        ranked = RankedJob(
            job=_make_job(title="Backend Engineer", company="Rippling"),
            rule_result=self._make_rule_result(70),
            ai_result=None,
        )
        text = str(ranked)
        assert "Backend Engineer" in text
        assert "Rippling" in text


# ---------------------------------------------------------------------------
# RankingAgent.rank_one — happy path
# ---------------------------------------------------------------------------

class TestRankOneHappyPath:

    def test_returns_ai_match_result(self):
        agent = RankingAgent(client=_mock_client(_VALID_PAYLOAD))
        result = agent.rank_one(_make_job(), rule_score=70)
        assert isinstance(result, AIMatchResult)
        assert result.score == 82

    def test_calls_messages_create_once(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        agent.rank_one(_make_job())
        client.messages.create.assert_called_once()

    def test_messages_create_uses_correct_model(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client, model="claude-haiku-4-5-20251001")
        agent.rank_one(_make_job())
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_tool_choice_forces_submit_job_ranking(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        agent.rank_one(_make_job())
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_job_ranking"}

    def test_system_prompt_contains_candidate_profile(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client, profile=DEFAULT_PROFILE)
        agent.rank_one(_make_job())
        system = client.messages.create.call_args.kwargs["system"]
        assert "Java" in system
        assert "Spring Boot" in system
        assert str(DEFAULT_PROFILE.years_of_experience) in system

    def test_user_message_contains_job_fields(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job(
            company="Databricks",
            title="Platform Engineer",
            location="San Francisco, CA",
            description="We use Spark and Scala for data processing.",
        )
        agent.rank_one(job, rule_score=55)
        msg_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Databricks" in msg_content
        assert "Platform Engineer" in msg_content
        assert "San Francisco, CA" in msg_content
        assert "55/100" in msg_content

    def test_description_is_included_in_user_message(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job(description="Must know Java and Spring Boot.")
        agent.rank_one(job)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Must know Java and Spring Boot." in content

    def test_requirements_is_included_in_user_message(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job(requirements="5+ years of Java experience required.")
        agent.rank_one(job)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "5+ years of Java experience required." in content


# ---------------------------------------------------------------------------
# RankingAgent.rank_one — failure paths
# ---------------------------------------------------------------------------

class TestRankOneFailurePaths:

    def test_returns_none_on_api_exception(self):
        agent = RankingAgent(client=_error_client(RuntimeError("connection timeout")))
        result = agent.rank_one(_make_job())
        assert result is None

    def test_returns_none_if_no_tool_block_in_response(self):
        agent = RankingAgent(client=_no_tool_client())
        result = agent.rank_one(_make_job())
        assert result is None

    def test_returns_none_on_pydantic_validation_failure(self):
        bad_payload = {**_VALID_PAYLOAD, "score": 999}  # violates ge=0 le=100
        agent = RankingAgent(client=_mock_client(bad_payload))
        result = agent.rank_one(_make_job())
        assert result is None

    def test_does_not_raise_on_any_exception(self):
        """rank_one must never propagate an exception to the caller."""
        for exc in [RuntimeError("oops"), ValueError("bad"), Exception("generic")]:
            agent = RankingAgent(client=_error_client(exc))
            result = agent.rank_one(_make_job())
            assert result is None


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:

    def test_second_call_for_same_job_uses_cache(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job()

        result1 = agent.rank_one(job)
        result2 = agent.rank_one(job)

        assert result1 == result2
        assert client.messages.create.call_count == 1  # API called once, not twice

    def test_different_jobs_produce_separate_cache_entries(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)

        job_a = _make_job(job_url="https://stripe.com/jobs/1")
        job_b = _make_job(job_url="https://stripe.com/jobs/2")

        agent.rank_one(job_a)
        agent.rank_one(job_b)

        assert client.messages.create.call_count == 2

    def test_cache_size_reflects_stored_results(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)

        assert agent.cache_size == 0
        agent.rank_one(_make_job(job_url="https://stripe.com/jobs/1"))
        assert agent.cache_size == 1
        agent.rank_one(_make_job(job_url="https://stripe.com/jobs/2"))
        assert agent.cache_size == 2

    def test_failed_call_is_not_cached(self):
        """A failed AI call must not be cached — it should be retried next time."""
        client = _error_client(RuntimeError("timeout"))
        agent = RankingAgent(client=client)
        job = _make_job()

        agent.rank_one(job)  # fails → None
        assert agent.cache_size == 0  # nothing cached

    def test_external_cache_shared_across_agents(self):
        """Two agents sharing an external cache dict reuse each other's results."""
        shared_cache: dict = {}
        client_a = _mock_client(_VALID_PAYLOAD)
        client_b = _mock_client({**_VALID_PAYLOAD, "score": 50})

        agent_a = RankingAgent(client=client_a, cache=shared_cache)
        agent_b = RankingAgent(client=client_b, cache=shared_cache)

        job = _make_job()
        result_a = agent_a.rank_one(job)
        result_b = agent_b.rank_one(job)  # should hit cache set by agent_a

        assert result_a is result_b  # exact same object from cache
        assert client_b.messages.create.call_count == 0  # agent_b never called the API

    def test_different_profile_different_cache_key(self):
        """Changing the profile invalidates the cache."""
        profile_a = DEFAULT_PROFILE
        profile_b = DEFAULT_PROFILE.model_copy(
            update={"years_of_experience": 10}
        )
        shared_cache: dict = {}
        client = _mock_client(_VALID_PAYLOAD)

        agent_a = RankingAgent(client=client, profile=profile_a, cache=shared_cache)
        agent_b = RankingAgent(client=client, profile=profile_b, cache=shared_cache)

        job = _make_job()
        agent_a.rank_one(job)
        agent_b.rank_one(job)

        assert client.messages.create.call_count == 2  # two different cache keys

    def test_different_model_different_cache_key(self):
        shared_cache: dict = {}
        client = _mock_client(_VALID_PAYLOAD)

        agent_a = RankingAgent(client=client, model="claude-sonnet-4-6", cache=shared_cache)
        agent_b = RankingAgent(client=client, model="claude-haiku-4-5-20251001", cache=shared_cache)

        job = _make_job()
        agent_a.rank_one(job)
        agent_b.rank_one(job)

        assert client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# RankingAgent.rank — batch ranking
# ---------------------------------------------------------------------------

class TestRankBatch:

    def _make_client_sequence(self, payloads: list[dict]) -> MagicMock:
        """Client that returns each payload in sequence."""
        client = MagicMock()
        client.messages.create.side_effect = [
            self._make_response(p) for p in payloads
        ]
        return client

    def _make_response(self, tool_input: dict) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.input = tool_input
        resp = MagicMock()
        resp.content = [block]
        return resp

    def test_rank_returns_ranked_job_list(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        jobs = [_make_job(job_url=f"https://example.com/jobs/{i}") for i in range(3)]
        ranked = agent.rank(jobs)
        assert len(ranked) == 3
        assert all(isinstance(r, RankedJob) for r in ranked)

    def test_rank_empty_list_returns_empty(self):
        agent = RankingAgent(client=_mock_client(_VALID_PAYLOAD))
        assert agent.rank([]) == []

    def test_rank_sorted_by_best_score_descending(self):
        payloads = [
            {**_VALID_PAYLOAD, "score": 40},
            {**_VALID_PAYLOAD, "score": 90},
            {**_VALID_PAYLOAD, "score": 65},
        ]
        client = self._make_client_sequence(payloads)
        agent = RankingAgent(client=client)
        jobs = [_make_job(job_url=f"https://example.com/jobs/{i}") for i in range(3)]
        ranked = agent.rank(jobs)
        scores = [r.best_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_continues_if_middle_job_fails(self):
        """All three jobs appear in results even when the second API call raises."""
        job_a = _make_job(job_url="https://example.com/jobs/1")
        job_b = _make_job(job_url="https://example.com/jobs/2", company="ErrorCo")
        job_c = _make_job(job_url="https://example.com/jobs/3")

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("rate limit hit")
            block = MagicMock()
            block.type = "tool_use"
            block.input = _VALID_PAYLOAD
            resp = MagicMock()
            resp.content = [block]
            return resp

        client = MagicMock()
        client.messages.create.side_effect = side_effect

        agent = RankingAgent(client=client)
        ranked = agent.rank([job_a, job_b, job_c])

        assert len(ranked) == 3
        # job_b's AI call failed — ai_result is None, but it's still in the list
        failed = next(r for r in ranked if r.job.company == "ErrorCo")
        assert failed.ai_result is None

    def test_rank_uses_cache_across_batch(self):
        """If the same job appears twice in the list, it is only ranked once."""
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job()  # same fingerprint both times
        ranked = agent.rank([job, job])
        assert client.messages.create.call_count == 1  # cached after first

    def test_rank_from_repository(self):
        """rank_from_repository reads jobs from the repo and returns RankedJob objects."""
        job = _make_job()
        mock_repo = MagicMock()
        mock_repo.get_all.return_value = [job]

        agent = RankingAgent(client=_mock_client(_VALID_PAYLOAD))
        ranked = agent.rank_from_repository(mock_repo)

        mock_repo.get_all.assert_called_once()
        assert len(ranked) == 1
        assert isinstance(ranked[0], RankedJob)


# ---------------------------------------------------------------------------
# Description truncation
# ---------------------------------------------------------------------------

class TestDescriptionTruncation:

    def test_short_description_not_truncated(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client, max_description_chars=3000)
        job = _make_job(description="Short description.")
        agent.rank_one(job)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Short description." in content
        assert "truncated" not in content

    def test_long_description_is_truncated(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client, max_description_chars=50)
        long_text = "X" * 200
        job = _make_job(description=long_text)
        agent.rank_one(job)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "truncated" in content
        assert long_text not in content

    def test_none_description_handled_gracefully(self):
        client = _mock_client(_VALID_PAYLOAD)
        agent = RankingAgent(client=client)
        job = _make_job(description=None)
        result = agent.rank_one(job)
        assert result is not None  # AI call still proceeds
