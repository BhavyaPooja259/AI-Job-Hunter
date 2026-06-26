"""
Sprint 14 — NotificationAgent, NotificationService, and ConsoleChannel tests.

No real database or API calls are made.  All tests use in-memory job lists
and mock/stub channels.

Design principles
-----------------
build_digest() tested independently from send()
    Most tests call agent.build_digest([...]) directly.  This proves that
    scoring, deduplication, and structuring work correctly without involving
    any delivery channel.

Mock channels for delivery tests
    A MockChannel captures every digest it receives and records whether send()
    was called.  A FailChannel returns False.  An ErrorChannel raises.  These
    three variants cover all paths in NotificationService.send().

capsys for ConsoleChannel
    pytest's built-in capsys fixture captures sys.stdout, letting us assert
    on the printed text without monkey-patching print().

Run from the project root:
    python -m pytest tests/test_notification_agent.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.constants import ATSType
from matching.digest import DigestJob, JobDigest
from matching.profile import DEFAULT_PROFILE
from scrapers.models import Job
from agents.notification_agent import NotificationAgent, NotificationResult
from services.notification_service import (
    ConsoleChannel,
    NotificationChannel,
    NotificationService,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_job(
    company: str = "Stripe",
    title: str = "Software Engineer II",
    job_url: str = "https://stripe.com/jobs/1",
    description: str = "We build Java microservices with Spring Boot and REST APIs.",
    location: str = "Remote",
) -> Job:
    return Job(
        company=company,
        title=title,
        job_url=job_url,
        location=location,
        source_platform=ATSType.GREENHOUSE,
        description=description,
    )


def _make_digest(
    jobs: list[DigestJob] | None = None,
    top_jobs: list[DigestJob] | None = None,
    total_in_db: int = 3,
    unique_count: int = 3,
    score_threshold: int = 60,
) -> JobDigest:
    jobs = jobs or []
    top_jobs = top_jobs or []
    return JobDigest(
        all_jobs=jobs,
        top_jobs=top_jobs,
        total_in_db=total_in_db,
        unique_count=unique_count,
        score_threshold=score_threshold,
    )


class MockChannel(NotificationChannel):
    """Captures every digest it receives; always succeeds."""

    def __init__(self, channel_name: str = "mock") -> None:
        self._name = channel_name
        self.received: list[JobDigest] = []

    @property
    def name(self) -> str:
        return self._name

    def send(self, digest: JobDigest) -> bool:
        self.received.append(digest)
        return True


class FailChannel(NotificationChannel):
    """Returns False from send() — simulates a silent delivery failure."""

    @property
    def name(self) -> str:
        return "fail"

    def send(self, digest: JobDigest) -> bool:
        return False


class ErrorChannel(NotificationChannel):
    """Raises an exception from send() — simulates a hard delivery error."""

    @property
    def name(self) -> str:
        return "error"

    def send(self, digest: JobDigest) -> bool:
        raise RuntimeError("network error")


# ---------------------------------------------------------------------------
# DigestJob tests
# ---------------------------------------------------------------------------

class TestDigestJob:

    def _make_digest_job(self, score: int = 75) -> DigestJob:
        return DigestJob(
            job=_make_job(),
            score=score,
            matched_skills=["java", "spring boot", "rest"],
            missing_skills=["kubernetes"],
        )

    def test_str_includes_title_and_company(self):
        dj = self._make_digest_job()
        text = str(dj)
        assert "Software Engineer II" in text
        assert "Stripe" in text

    def test_str_includes_score(self):
        dj = self._make_digest_job(score=82)
        assert "82" in str(dj)

    def test_str_includes_matched_skills(self):
        dj = self._make_digest_job()
        text = str(dj)
        assert "java" in text


# ---------------------------------------------------------------------------
# JobDigest tests
# ---------------------------------------------------------------------------

class TestJobDigest:

    def _make_dj(self, company: str, score: int) -> DigestJob:
        return DigestJob(
            job=_make_job(company=company, job_url=f"https://{company}.com/jobs/1"),
            score=score,
            matched_skills=["java"],
            missing_skills=[],
        )

    def test_by_company_groups_correctly(self):
        dj_a1 = self._make_dj("Adobe", 85)
        dj_a2 = DigestJob(
            job=_make_job(company="Adobe", job_url="https://adobe.com/jobs/2"),
            score=70,
            matched_skills=["java"],
            missing_skills=[],
        )
        dj_b = self._make_dj("Stripe", 90)

        digest = _make_digest(jobs=[dj_b, dj_a1, dj_a2])
        grouped = digest.by_company

        assert "Adobe" in grouped
        assert "Stripe" in grouped
        assert len(grouped["Adobe"]) == 2
        assert len(grouped["Stripe"]) == 1

    def test_by_company_preserves_score_order(self):
        dj_high = DigestJob(
            job=_make_job(company="Adobe", job_url="https://adobe.com/jobs/1"),
            score=90,
            matched_skills=[],
            missing_skills=[],
        )
        dj_low = DigestJob(
            job=_make_job(company="Adobe", job_url="https://adobe.com/jobs/2"),
            score=50,
            matched_skills=[],
            missing_skills=[],
        )
        digest = _make_digest(jobs=[dj_high, dj_low])
        grouped = digest.by_company["Adobe"]
        assert grouped[0].score >= grouped[1].score

    def test_summary_includes_unique_count(self):
        dj = self._make_dj("Stripe", 80)
        digest = _make_digest(jobs=[dj], top_jobs=[dj], unique_count=1)
        assert "1 unique" in digest.summary()

    def test_summary_includes_top_count(self):
        dj = self._make_dj("Stripe", 80)
        digest = _make_digest(jobs=[dj], top_jobs=[dj])
        assert "1 top match" in digest.summary()

    def test_summary_includes_threshold(self):
        digest = _make_digest(score_threshold=75)
        assert "75" in digest.summary()

    def test_generated_at_is_set_automatically(self):
        digest = _make_digest()
        assert digest.generated_at is not None

    def test_empty_digest_by_company_is_empty_dict(self):
        digest = _make_digest(jobs=[])
        assert digest.by_company == {}


# ---------------------------------------------------------------------------
# NotificationResult tests
# ---------------------------------------------------------------------------

class TestNotificationResult:

    def _make_result(
        self,
        notified: list[str],
        failed: list[str],
    ) -> NotificationResult:
        return NotificationResult(
            digest=_make_digest(),
            channels_notified=notified,
            failed_channels=failed,
        )

    def test_success_when_notified_and_no_failures(self):
        result = self._make_result(notified=["console"], failed=[])
        assert result.success is True

    def test_not_success_when_no_channels_notified(self):
        result = self._make_result(notified=[], failed=[])
        assert result.success is False

    def test_not_success_when_any_failure(self):
        result = self._make_result(notified=["console"], failed=["email"])
        assert result.success is False

    def test_any_sent_true_when_one_notified(self):
        result = self._make_result(notified=["console"], failed=["email"])
        assert result.any_sent is True

    def test_any_sent_false_when_nothing_notified(self):
        result = self._make_result(notified=[], failed=["email"])
        assert result.any_sent is False

    def test_str_includes_channel_names(self):
        result = self._make_result(notified=["console"], failed=[])
        assert "console" in str(result)


# ---------------------------------------------------------------------------
# NotificationAgent.build_digest tests
# ---------------------------------------------------------------------------

class TestBuildDigest:

    def _agent(self, threshold: int = 60) -> NotificationAgent:
        return NotificationAgent(
            service=NotificationService([MockChannel()]),
            score_threshold=threshold,
        )

    def test_returns_job_digest(self):
        digest = self._agent().build_digest([_make_job()])
        assert isinstance(digest, JobDigest)

    def test_empty_list_returns_empty_digest(self):
        digest = self._agent().build_digest([])
        assert digest.all_jobs == []
        assert digest.top_jobs == []
        assert digest.total_in_db == 0
        assert digest.unique_count == 0

    def test_total_in_db_reflects_raw_input(self):
        jobs = [
            _make_job(job_url="https://stripe.com/jobs/1"),
            _make_job(job_url="https://stripe.com/jobs/2"),
        ]
        digest = self._agent().build_digest(jobs)
        assert digest.total_in_db == 2

    def test_deduplicates_by_fingerprint(self):
        job = _make_job(job_url="https://stripe.com/jobs/1")
        duplicate = _make_job(job_url="https://stripe.com/jobs/1")  # identical URL
        digest = self._agent().build_digest([job, duplicate])
        assert digest.total_in_db == 2
        assert digest.unique_count == 1
        assert len(digest.all_jobs) == 1

    def test_jobs_sorted_by_score_descending(self):
        jobs = [
            _make_job(company="A", job_url="https://a.com/1", description="We need Python"),
            _make_job(company="B", job_url="https://b.com/1",
                      description="We use Java, Spring Boot, REST APIs, microservices"),
        ]
        digest = self._agent().build_digest(jobs)
        scores = [dj.score for dj in digest.all_jobs]
        assert scores == sorted(scores, reverse=True)

    def test_top_jobs_are_above_threshold(self):
        # Jobs with Java/Spring Boot descriptions score high against DEFAULT_PROFILE
        high_job = _make_job(
            job_url="https://stripe.com/jobs/1",
            description="Java Spring Boot microservices REST API SQL backend engineer",
        )
        low_job = _make_job(
            company="Other",
            job_url="https://other.com/jobs/1",
            description="Machine learning Python data science TensorFlow",
        )
        digest = self._agent(threshold=60).build_digest([high_job, low_job])
        for dj in digest.top_jobs:
            assert dj.score >= 60

    def test_all_jobs_includes_below_threshold(self):
        low_job = _make_job(
            company="Other",
            job_url="https://other.com/jobs/1",
            description="Machine learning Python data science TensorFlow",
        )
        digest = self._agent(threshold=60).build_digest([low_job])
        # low-scoring job must appear in all_jobs even if not in top_jobs
        assert len(digest.all_jobs) == 1

    def test_digest_job_has_matched_skills(self):
        job = _make_job(description="We use Java and Spring Boot")
        digest = self._agent().build_digest([job])
        dj = digest.all_jobs[0]
        assert isinstance(dj.matched_skills, list)

    def test_score_threshold_stored_in_digest(self):
        digest = self._agent(threshold=75).build_digest([])
        assert digest.score_threshold == 75


# ---------------------------------------------------------------------------
# NotificationAgent.notify_from_jobs tests
# ---------------------------------------------------------------------------

class TestNotifyFromJobs:

    def test_returns_notification_result(self):
        channel = MockChannel()
        agent = NotificationAgent(service=NotificationService([channel]))
        result = agent.notify_from_jobs([_make_job()])
        assert isinstance(result, NotificationResult)

    def test_channels_notified_contains_channel_name(self):
        channel = MockChannel("my_channel")
        agent = NotificationAgent(service=NotificationService([channel]))
        result = agent.notify_from_jobs([_make_job()])
        assert "my_channel" in result.channels_notified

    def test_channel_receives_the_digest(self):
        channel = MockChannel()
        agent = NotificationAgent(service=NotificationService([channel]))
        agent.notify_from_jobs([_make_job()])
        assert len(channel.received) == 1
        assert isinstance(channel.received[0], JobDigest)

    def test_failed_channel_in_failed_list(self):
        fail = FailChannel()
        agent = NotificationAgent(service=NotificationService([fail]))
        result = agent.notify_from_jobs([_make_job()])
        assert "fail" in result.failed_channels
        assert result.success is False

    def test_error_channel_in_failed_list(self):
        error = ErrorChannel()
        agent = NotificationAgent(service=NotificationService([error]))
        result = agent.notify_from_jobs([_make_job()])
        assert "error" in result.failed_channels


# ---------------------------------------------------------------------------
# NotificationAgent.notify (repository path)
# ---------------------------------------------------------------------------

class TestNotifyFromRepository:

    def test_reads_from_repository(self):
        channel = MockChannel()
        agent = NotificationAgent(service=NotificationService([channel]))

        mock_repo = MagicMock()
        mock_repo.get_all.return_value = [_make_job()]

        agent.notify(mock_repo)

        mock_repo.get_all.assert_called_once()

    def test_returns_notification_result(self):
        channel = MockChannel()
        agent = NotificationAgent(service=NotificationService([channel]))

        mock_repo = MagicMock()
        mock_repo.get_all.return_value = [_make_job()]

        result = agent.notify(mock_repo)
        assert isinstance(result, NotificationResult)


# ---------------------------------------------------------------------------
# ConsoleChannel tests
# ---------------------------------------------------------------------------

class TestConsoleChannel:

    def test_name_is_console(self):
        assert ConsoleChannel().name == "console"

    def test_send_returns_true(self):
        channel = ConsoleChannel()
        digest = _make_digest()
        assert channel.send(digest) is True

    def test_send_prints_to_stdout(self, capsys):
        channel = ConsoleChannel()
        digest = _make_digest()
        channel.send(digest)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_digest_word(self, capsys):
        channel = ConsoleChannel()
        digest = _make_digest()
        channel.send(digest)
        captured = capsys.readouterr()
        assert "Digest" in captured.out or "digest" in captured.out

    def test_output_contains_job_title_when_jobs_present(self, capsys):
        dj = DigestJob(
            job=_make_job(title="Backend Engineer"),
            score=82,
            matched_skills=["java"],
            missing_skills=[],
        )
        channel = ConsoleChannel()
        digest = _make_digest(jobs=[dj], top_jobs=[dj])
        channel.send(digest)
        captured = capsys.readouterr()
        assert "Backend Engineer" in captured.out

    def test_output_contains_company_when_jobs_present(self, capsys):
        dj = DigestJob(
            job=_make_job(company="Rippling"),
            score=78,
            matched_skills=["java"],
            missing_skills=[],
        )
        channel = ConsoleChannel()
        digest = _make_digest(jobs=[dj], top_jobs=[])
        channel.send(digest)
        captured = capsys.readouterr()
        assert "Rippling" in captured.out

    def test_output_contains_score_threshold(self, capsys):
        channel = ConsoleChannel()
        digest = _make_digest(score_threshold=70)
        channel.send(digest)
        captured = capsys.readouterr()
        assert "70" in captured.out

    def test_empty_digest_does_not_crash(self, capsys):
        channel = ConsoleChannel()
        channel.send(_make_digest(jobs=[], top_jobs=[]))
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# NotificationService tests
# ---------------------------------------------------------------------------

class TestNotificationService:

    def test_send_to_single_channel(self):
        channel = MockChannel()
        service = NotificationService(channels=[channel])
        notified, failed = service.send(_make_digest())
        assert "mock" in notified
        assert failed == []

    def test_send_to_multiple_channels(self):
        ch_a = MockChannel("alpha")
        ch_b = MockChannel("beta")
        service = NotificationService(channels=[ch_a, ch_b])
        notified, failed = service.send(_make_digest())
        assert "alpha" in notified
        assert "beta" in notified
        assert failed == []

    def test_fail_channel_goes_to_failed_list(self):
        fail = FailChannel()
        service = NotificationService(channels=[fail])
        notified, failed = service.send(_make_digest())
        assert notified == []
        assert "fail" in failed

    def test_error_channel_goes_to_failed_list(self):
        error = ErrorChannel()
        service = NotificationService(channels=[error])
        notified, failed = service.send(_make_digest())
        assert notified == []
        assert "error" in failed

    def test_error_channel_does_not_block_other_channels(self):
        """A failing channel must not prevent other channels from receiving the digest."""
        error = ErrorChannel()
        good = MockChannel("good")
        service = NotificationService(channels=[error, good])
        notified, failed = service.send(_make_digest())
        assert "good" in notified
        assert "error" in failed

    def test_no_channels_returns_empty_lists(self):
        service = NotificationService()
        notified, failed = service.send(_make_digest())
        assert notified == []
        assert failed == []

    def test_add_channel_registers_it(self):
        service = NotificationService()
        channel = MockChannel("dynamic")
        service.add_channel(channel)
        notified, _ = service.send(_make_digest())
        assert "dynamic" in notified

    def test_channel_names_property(self):
        ch_a = MockChannel("alpha")
        ch_b = MockChannel("beta")
        service = NotificationService(channels=[ch_a, ch_b])
        assert service.channel_names == ["alpha", "beta"]

    def test_mixed_success_and_failure(self):
        good = MockChannel("good")
        fail = FailChannel()
        service = NotificationService(channels=[good, fail])
        notified, failed = service.send(_make_digest())
        assert "good" in notified
        assert "fail" in failed
