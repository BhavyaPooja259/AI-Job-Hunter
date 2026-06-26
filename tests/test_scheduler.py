"""
Tests for Sprint 19 — Automation Scheduler.

Coverage
--------
TestWorkflowStep              (3)  — enum values
TestStepResult                (4)  — dataclass fields and defaults
TestWorkflowResult            (7)  — aggregation, success, summary
TestSchedulerConfig           (5)  — configuration defaults and overrides
TestJobRunnerScout            (5)  — scout step, new-job detection
TestJobRunnerRank             (5)  — filtering by score and top_n
TestJobRunnerNotify           (4)  — notify + dashboard steps
TestJobRunnerOptionalSteps   (11)  — tailor, cover letter, referral reminder,
                                     idempotency via only_new_jobs
TestJobRunnerFullRun          (4)  — end-to-end run() integration
TestSchedulerAgent            (5)  — public API surface

Approach
--------
All external dependencies (ScoutAgent, NotificationAgent, repos) are mocked.
JobMatcher is exercised with real Job objects to keep the rank-step tests
deterministic.  We use min_score=0 in most tests so every job passes the
threshold, decoupling test validity from scoring-weight changes.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agents.scheduler_agent import SchedulerAgent
from config.constants import ATSType
from matching.matcher import MatchResult
from scheduler.job_runner import JobRunner
from scheduler.scheduler import SchedulerConfig
from scheduler.workflow import StepResult, WorkflowResult, WorkflowStep
from scrapers.models import Job


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_job(
    company: str = "Stripe",
    title: str = "Backend Engineer",
    url: str = "https://stripe.com/1",
) -> Job:
    return Job(company=company, title=title, job_url=url, source_platform=ATSType.GREENHOUSE)


def _make_match(score: int = 75) -> MatchResult:
    return MatchResult(
        score=score, matched_skills=["java"], missing_skills=[], reasons=[]
    )


def _make_scan_result(
    companies: int = 1,
    saved: int = 1,
    duplicates: int = 0,
    failures: list | None = None,
) -> MagicMock:
    scan = MagicMock()
    scan.companies_checked = companies
    scan.jobs_saved = saved
    scan.duplicates = duplicates
    scan.failure_count = len(failures or [])
    scan.failures = failures or []
    return scan


def _mock_notification_result(unique: int = 3, top: int = 2) -> MagicMock:
    digest = MagicMock()
    digest.unique_count = unique
    digest.top_jobs = [MagicMock() for _ in range(top)]
    result = MagicMock()
    result.digest = digest
    return result


def _make_runner(
    pre_jobs: list[Job] | None = None,
    post_jobs: list[Job] | None = None,
    scan_result=None,
    notification_result=None,
    referral_repo=None,
    client=None,
) -> JobRunner:
    """Build a JobRunner backed entirely by mocks."""
    job_repo = MagicMock()
    job_repo.get_all.side_effect = [
        pre_jobs or [],
        post_jobs or [],
    ]

    scout_agent = MagicMock()
    scout_agent.run.return_value = scan_result or _make_scan_result()

    notification_agent = MagicMock()
    notification_agent.notify_from_jobs.return_value = (
        notification_result or _mock_notification_result()
    )

    return JobRunner(
        job_repo=job_repo,
        scout_agent=scout_agent,
        notification_agent=notification_agent,
        referral_repo=referral_repo,
        client=client,
    )


# ---------------------------------------------------------------------------
# TestWorkflowStep
# ---------------------------------------------------------------------------


class TestWorkflowStep:
    def test_seven_steps_exist(self):
        values = {s.value for s in WorkflowStep}
        expected = {
            "SCOUT", "RANK", "NOTIFY", "DASHBOARD",
            "TAILOR", "COVER_LETTER", "REFERRAL_REMINDER",
        }
        assert values == expected

    def test_steps_are_string_compatible(self):
        assert WorkflowStep.SCOUT == "SCOUT"
        assert WorkflowStep.COVER_LETTER == "COVER_LETTER"

    def test_step_values_are_unique(self):
        values = [s.value for s in WorkflowStep]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# TestStepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_defaults(self):
        sr = StepResult(step=WorkflowStep.SCOUT, success=True)
        assert sr.processed == 0
        assert sr.skipped == 0
        assert sr.message == ""
        assert sr.duration_ms == 0.0

    def test_success_true(self):
        sr = StepResult(step=WorkflowStep.RANK, success=True, processed=5)
        assert sr.success is True
        assert sr.processed == 5

    def test_failed_step(self):
        sr = StepResult(step=WorkflowStep.SCOUT, success=False, message="network error")
        assert sr.success is False
        assert "network" in sr.message

    def test_skipped_field(self):
        sr = StepResult(step=WorkflowStep.TAILOR, success=True, skipped=3)
        assert sr.skipped == 3


# ---------------------------------------------------------------------------
# TestWorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_success_when_all_steps_succeed(self):
        wr = WorkflowResult(started_at=datetime.now())
        wr.add_step(StepResult(step=WorkflowStep.SCOUT, success=True))
        wr.add_step(StepResult(step=WorkflowStep.RANK, success=True))
        assert wr.success is True

    def test_fails_when_any_step_fails(self):
        wr = WorkflowResult(started_at=datetime.now())
        wr.add_step(StepResult(step=WorkflowStep.SCOUT, success=True))
        wr.add_step(StepResult(step=WorkflowStep.RANK, success=False))
        assert wr.success is False

    def test_empty_steps_not_success(self):
        wr = WorkflowResult(started_at=datetime.now())
        assert wr.success is False

    def test_add_step_appends(self):
        wr = WorkflowResult(started_at=datetime.now())
        wr.add_step(StepResult(step=WorkflowStep.SCOUT, success=True))
        wr.add_step(StepResult(step=WorkflowStep.RANK, success=True))
        assert len(wr.steps) == 2

    def test_total_duration_from_timestamps(self):
        start = datetime(2024, 1, 1, 12, 0, 0)
        finish = datetime(2024, 1, 1, 12, 0, 5)  # 5 seconds
        wr = WorkflowResult(started_at=start, finished_at=finish)
        assert wr.total_duration_ms == pytest.approx(5000, abs=1)

    def test_total_duration_from_steps_when_no_finished_at(self):
        wr = WorkflowResult(started_at=datetime.now())
        wr.add_step(StepResult(step=WorkflowStep.SCOUT, success=True, duration_ms=100))
        wr.add_step(StepResult(step=WorkflowStep.RANK, success=True, duration_ms=200))
        assert wr.total_duration_ms == pytest.approx(300, abs=1)

    def test_summary_contains_step_names(self):
        wr = WorkflowResult(started_at=datetime.now(), finished_at=datetime.now())
        wr.add_step(StepResult(step=WorkflowStep.SCOUT, success=True, processed=3))
        wr.add_step(StepResult(step=WorkflowStep.RANK, success=True, skipped=1))
        summary = wr.summary()
        assert "SCOUT" in summary
        assert "RANK" in summary
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# TestSchedulerConfig
# ---------------------------------------------------------------------------


class TestSchedulerConfig:
    def test_defaults(self):
        c = SchedulerConfig()
        assert c.top_n == 5
        assert c.min_score == 60
        assert c.resume_text == ""
        assert c.candidate_name == ""

    def test_optional_steps_disabled_by_default(self):
        c = SchedulerConfig()
        assert c.enable_tailoring is False
        assert c.enable_cover_letters is False
        assert c.enable_referral_reminders is False

    def test_only_new_jobs_default_true(self):
        assert SchedulerConfig().only_new_jobs is True

    def test_custom_values(self):
        c = SchedulerConfig(top_n=3, min_score=70, enable_cover_letters=True)
        assert c.top_n == 3
        assert c.min_score == 70
        assert c.enable_cover_letters is True

    def test_only_new_jobs_can_be_disabled(self):
        c = SchedulerConfig(only_new_jobs=False)
        assert c.only_new_jobs is False


# ---------------------------------------------------------------------------
# TestJobRunnerScout
# ---------------------------------------------------------------------------


class TestJobRunnerScout:
    def test_scout_agent_called(self):
        runner = _make_runner(pre_jobs=[], post_jobs=[])
        runner._run_scout()
        runner._scout_agent.run.assert_called_once()

    def test_new_jobs_detected(self):
        job1 = _make_job(company="Stripe")
        job2 = _make_job(company="Uber", url="https://uber.com/1")
        runner = _make_runner(pre_jobs=[job1], post_jobs=[job1, job2])
        _, all_jobs, new_fps = runner._run_scout()
        assert len(all_jobs) == 2
        assert job2.fingerprint in new_fps
        assert job1.fingerprint not in new_fps

    def test_no_new_jobs_on_second_run(self):
        job1 = _make_job()
        runner = _make_runner(pre_jobs=[job1], post_jobs=[job1])
        _, _, new_fps = runner._run_scout()
        assert len(new_fps) == 0

    def test_step_result_counts_from_scan(self):
        scan = _make_scan_result(companies=3, saved=2, duplicates=1)
        runner = _make_runner(post_jobs=[], scan_result=scan)
        step, _, _ = runner._run_scout()
        assert step.processed == 2
        assert step.skipped == 1
        assert step.success is True

    def test_scout_failure_returns_failed_step(self):
        runner = _make_runner()
        runner._scout_agent.run.side_effect = RuntimeError("scraper crashed")
        step, _, _ = runner._run_scout()
        assert step.success is False
        assert "scraper crashed" in step.message


# ---------------------------------------------------------------------------
# TestJobRunnerRank
# ---------------------------------------------------------------------------


class TestJobRunnerRank:
    def test_rank_filters_by_min_score(self):
        jobs = [_make_job(title="Backend Engineer"), _make_job(company="X", url="u2")]
        runner = _make_runner()
        config = SchedulerConfig(min_score=100)  # impossible threshold
        _, top_pairs = runner._run_rank(config, jobs)
        assert len(top_pairs) == 0

    def test_rank_limits_to_top_n(self):
        jobs = [_make_job(url=f"https://s.com/{i}") for i in range(10)]
        runner = _make_runner()
        config = SchedulerConfig(min_score=0, top_n=3)
        _, top_pairs = runner._run_rank(config, jobs)
        assert len(top_pairs) <= 3

    def test_rank_step_result_success(self):
        job = _make_job()
        runner = _make_runner()
        config = SchedulerConfig(min_score=0)
        step, _ = runner._run_rank(config, [job])
        assert step.success is True
        assert step.step == WorkflowStep.RANK

    def test_rank_empty_db_returns_empty(self):
        runner = _make_runner()
        config = SchedulerConfig()
        step, top_pairs = runner._run_rank(config, [])
        assert top_pairs == []
        assert step.success is True

    def test_rank_below_threshold_reported_as_skipped(self):
        jobs = [
            _make_job(url="https://s.com/1"),
            _make_job(url="https://s.com/2"),
        ]
        runner = _make_runner()
        config = SchedulerConfig(min_score=0, top_n=5)
        step, top_pairs = runner._run_rank(config, jobs)
        assert step.processed + step.skipped == len(jobs)


# ---------------------------------------------------------------------------
# TestJobRunnerNotify
# ---------------------------------------------------------------------------


class TestJobRunnerNotify:
    def test_notify_called_with_jobs(self):
        jobs = [_make_job()]
        runner = _make_runner()
        runner._run_notify(jobs)
        runner._notification_agent.notify_from_jobs.assert_called_once_with(jobs)

    def test_notify_step_success(self):
        runner = _make_runner()
        step = runner._run_notify([_make_job()])
        assert step.success is True
        assert step.step == WorkflowStep.NOTIFY

    def test_notify_step_failure_on_exception(self):
        runner = _make_runner()
        runner._notification_agent.notify_from_jobs.side_effect = RuntimeError("fail")
        step = runner._run_notify([])
        assert step.success is False

    def test_dashboard_step_always_success(self):
        runner = _make_runner()
        step = runner._run_dashboard([_make_job(), _make_job(url="u2")])
        assert step.success is True
        assert step.step == WorkflowStep.DASHBOARD
        assert step.processed == 2


# ---------------------------------------------------------------------------
# TestJobRunnerOptionalSteps
# ---------------------------------------------------------------------------


class TestJobRunnerOptionalSteps:
    def test_tailor_skipped_when_no_client(self):
        runner = _make_runner(client=None)
        config = SchedulerConfig(resume_text="resume text")
        pairs = [(_make_job(), _make_match())]
        step = runner._run_tailor(config, pairs)
        assert step.success is True
        assert step.skipped == 1
        assert "Gemini" in step.message

    def test_tailor_skipped_when_no_resume_text(self):
        runner = _make_runner(client=MagicMock())
        config = SchedulerConfig(resume_text="")
        pairs = [(_make_job(), _make_match())]
        step = runner._run_tailor(config, pairs)
        assert step.success is True
        assert step.skipped == 1
        assert "resume_text" in step.message

    def test_tailor_skipped_when_no_candidates(self):
        runner = _make_runner(client=MagicMock())
        config = SchedulerConfig(resume_text="text")
        step = runner._run_tailor(config, [])
        assert step.success is True
        assert step.processed == 0

    def test_tailor_called_for_candidate_jobs(self):
        mock_client = MagicMock()
        runner = _make_runner(client=mock_client)
        config = SchedulerConfig(resume_text="resume content")
        job = _make_job()
        pairs = [(job, _make_match())]
        with patch("agents.resume_tailoring_agent.ResumeTailoringAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.tailor.return_value = MagicMock()
            step = runner._run_tailor(config, pairs)
        assert step.step == WorkflowStep.TAILOR

    def test_cover_letter_skipped_when_no_candidates(self):
        runner = _make_runner()
        config = SchedulerConfig(resume_text="text")
        step = runner._run_cover_letters(config, [])
        assert step.success is True
        assert step.processed == 0
        assert "no candidate" in step.message

    def test_cover_letter_works_without_client(self):
        runner = _make_runner(client=None)
        config = SchedulerConfig(resume_text="resume", candidate_name="Bhavya")
        job = _make_job()
        pairs = [(job, _make_match())]
        with patch("agents.cover_letter_agent.CoverLetterAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.generate.return_value = MagicMock()
            step = runner._run_cover_letters(config, pairs)
        assert step.step == WorkflowStep.COVER_LETTER

    def test_cover_letter_called_for_each_candidate(self):
        runner = _make_runner()
        config = SchedulerConfig(resume_text="text")
        pairs = [(_make_job(url=f"u{i}"), _make_match()) for i in range(3)]
        with patch("agents.cover_letter_agent.CoverLetterAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.generate.return_value = MagicMock()
            step = runner._run_cover_letters(config, pairs)
        assert instance.generate.call_count == 3
        assert step.processed == 3

    def test_referral_reminder_skipped_when_no_repo(self):
        runner = _make_runner(referral_repo=None)
        step = runner._run_referral_reminders()
        assert step.success is True
        assert "skipped" in step.message.lower()

    def test_referral_reminder_counts_request_sent(self):
        from referral.referral import Referral
        from referral.referral_status import ReferralStatus

        r1 = Referral(contact_name="Alice", company="Stripe")
        r1.status = ReferralStatus.REQUEST_SENT
        r2 = Referral(contact_name="Bob", company="Uber")
        r2.status = ReferralStatus.CONNECTED

        repo = MagicMock()
        repo.get_active.return_value = [r1, r2]
        runner = _make_runner(referral_repo=repo)
        step = runner._run_referral_reminders()
        assert step.processed == 1   # only REQUEST_SENT
        assert step.skipped == 1    # CONNECTED doesn't need follow-up

    def test_only_new_jobs_filters_candidate_pairs(self):
        job_old = _make_job(company="Existing", url="https://old.com")
        job_new = _make_job(company="New", url="https://new.com")
        # Simulate: old was already in DB; new was discovered this run
        runner = _make_runner(pre_jobs=[job_old], post_jobs=[job_old, job_new])
        runner._scout_agent.run.return_value = _make_scan_result(saved=1)
        runner._notification_agent.notify_from_jobs.return_value = _mock_notification_result()

        config = SchedulerConfig(
            min_score=0,
            top_n=10,
            enable_cover_letters=True,
            only_new_jobs=True,
            resume_text="text",
        )

        letter_calls = []
        with patch("agents.cover_letter_agent.CoverLetterAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.generate.side_effect = lambda j, **kw: letter_calls.append(j)
            runner.run(config)

        companies_processed = {j.company for j in letter_calls}
        assert "New" in companies_processed
        assert "Existing" not in companies_processed

    def test_cover_letter_exception_counted_as_skipped(self):
        runner = _make_runner()
        config = SchedulerConfig(resume_text="text")
        pairs = [(_make_job(), _make_match())]
        with patch("agents.cover_letter_agent.CoverLetterAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.generate.side_effect = RuntimeError("disk full")
            step = runner._run_cover_letters(config, pairs)
        assert step.success is True      # step itself didn't crash
        assert step.processed == 0
        assert step.skipped == 1


# ---------------------------------------------------------------------------
# TestJobRunnerFullRun
# ---------------------------------------------------------------------------


class TestJobRunnerFullRun:
    def _standard_runner(self, pre=None, post=None) -> JobRunner:
        return _make_runner(pre_jobs=pre or [], post_jobs=post or [])

    def test_run_returns_workflow_result(self):
        runner = self._standard_runner()
        result = runner.run(SchedulerConfig())
        assert isinstance(result, WorkflowResult)

    def test_mandatory_steps_always_present(self):
        runner = self._standard_runner()
        result = runner.run(SchedulerConfig())
        step_names = {s.step for s in result.steps}
        assert WorkflowStep.SCOUT in step_names
        assert WorkflowStep.RANK in step_names
        assert WorkflowStep.NOTIFY in step_names
        assert WorkflowStep.DASHBOARD in step_names

    def test_optional_steps_added_when_enabled(self):
        runner = self._standard_runner()
        config = SchedulerConfig(
            enable_tailoring=True,
            enable_cover_letters=True,
            enable_referral_reminders=True,
        )
        result = runner.run(config)
        step_names = {s.step for s in result.steps}
        assert WorkflowStep.TAILOR in step_names
        assert WorkflowStep.COVER_LETTER in step_names
        assert WorkflowStep.REFERRAL_REMINDER in step_names

    def test_optional_steps_absent_when_disabled(self):
        runner = self._standard_runner()
        result = runner.run(SchedulerConfig())
        step_names = {s.step for s in result.steps}
        assert WorkflowStep.TAILOR not in step_names
        assert WorkflowStep.COVER_LETTER not in step_names

    def test_finished_at_set_after_run(self):
        runner = self._standard_runner()
        result = runner.run(SchedulerConfig())
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at


# ---------------------------------------------------------------------------
# TestSchedulerAgent
# ---------------------------------------------------------------------------


class TestSchedulerAgent:
    def _make_agent(self, pre=None, post=None) -> SchedulerAgent:
        job_repo = MagicMock()
        job_repo.get_all.side_effect = [pre or [], post or []]
        scout = MagicMock()
        scout.run.return_value = _make_scan_result()
        notify = MagicMock()
        notify.notify_from_jobs.return_value = _mock_notification_result()
        return SchedulerAgent(
            job_repo=job_repo,
            scout_agent=scout,
            notification_agent=notify,
        )

    def test_agent_run_returns_workflow_result(self):
        agent = self._make_agent()
        result = agent.run(SchedulerConfig())
        assert isinstance(result, WorkflowResult)

    def test_agent_default_config_when_none_passed(self):
        agent = self._make_agent()
        result = agent.run(None)
        assert isinstance(result, WorkflowResult)

    def test_agent_has_mandatory_steps(self):
        agent = self._make_agent()
        result = agent.run(SchedulerConfig())
        assert len(result.steps) >= 4

    def test_agent_finished_at_set(self):
        agent = self._make_agent()
        result = agent.run()
        assert result.finished_at is not None

    def test_agent_summary_is_nonempty_string(self):
        agent = self._make_agent()
        result = agent.run()
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 10
