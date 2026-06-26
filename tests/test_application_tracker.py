"""
Tests for Sprint 15 — Application Tracker.

Coverage
--------
TestApplicationStatus     (5)  — enum values and transition metadata
TestApplication           (6)  — dataclass construction and __str__
TestApplicationRepository (16) — SQLite persistence: CRUD + edge cases
TestApplicationAgent      (15) — business logic: track, advance, notes, stats
"""

from __future__ import annotations

import pytest

from application.application import Application
from application.application_repository import ApplicationRepository
from application.application_status import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    ApplicationStatus,
)
from agents.application_agent import ApplicationAgent, ApplicationStats
from config.constants import ATSType
from scrapers.models import Job


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_job(
    company: str = "Stripe",
    title: str = "Software Engineer II",
    job_url: str = "https://stripe.com/jobs/1",
) -> Job:
    return Job(
        company=company,
        title=title,
        job_url=job_url,
        source_platform=ATSType.GREENHOUSE,
    )


def _make_application(
    company: str = "Stripe",
    title: str = "Software Engineer II",
    job_url: str = "https://stripe.com/jobs/1",
    fingerprint: str = "fp_stripe_swe2",
    status: ApplicationStatus = ApplicationStatus.SAVED,
) -> Application:
    return Application(
        company=company,
        title=title,
        job_url=job_url,
        job_fingerprint=fingerprint,
        status=status,
    )


# ---------------------------------------------------------------------------
# ApplicationStatus
# ---------------------------------------------------------------------------


class TestApplicationStatus:
    def test_all_nine_statuses_exist(self):
        expected = {
            "SAVED", "READY_TO_APPLY", "APPLIED",
            "ONLINE_ASSESSMENT", "PHONE_SCREEN", "INTERVIEW",
            "OFFER", "REJECTED", "WITHDRAWN",
        }
        actual = {s.value for s in ApplicationStatus}
        assert actual == expected

    def test_statuses_are_str_compatible(self):
        assert ApplicationStatus.SAVED == "SAVED"
        assert ApplicationStatus.APPLIED == "APPLIED"

    def test_terminal_statuses(self):
        assert ApplicationStatus.REJECTED in TERMINAL_STATUSES
        assert ApplicationStatus.WITHDRAWN in TERMINAL_STATUSES
        assert ApplicationStatus.OFFER not in TERMINAL_STATUSES

    def test_valid_transitions_covers_all_statuses(self):
        for status in ApplicationStatus:
            assert status in VALID_TRANSITIONS

    def test_terminal_statuses_have_empty_transitions(self):
        for terminal in TERMINAL_STATUSES:
            assert VALID_TRANSITIONS[terminal] == frozenset()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class TestApplication:
    def test_default_status_is_saved(self):
        app = _make_application()
        assert app.status == ApplicationStatus.SAVED

    def test_id_is_auto_generated(self):
        app = _make_application()
        assert app.id
        assert len(app.id) == 36  # UUID format

    def test_id_is_unique_per_instance(self):
        a = _make_application()
        b = _make_application()
        assert a.id != b.id

    def test_created_at_is_set(self):
        from datetime import datetime
        app = _make_application()
        assert isinstance(app.created_at, datetime)

    def test_str_includes_company_and_title(self):
        app = _make_application(company="Google", title="SRE")
        s = str(app)
        assert "Google" in s
        assert "SRE" in s

    def test_str_includes_status(self):
        app = _make_application(status=ApplicationStatus.INTERVIEW)
        assert "INTERVIEW" in str(app)


# ---------------------------------------------------------------------------
# ApplicationRepository
# ---------------------------------------------------------------------------


class TestApplicationRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        r = ApplicationRepository(db_path=str(tmp_path / "apps.db"))
        r.initialize()
        return r

    def test_initialize_creates_table(self, repo):
        # If initialize succeeds, the table exists; verify via a read
        assert repo.get_all() == []

    def test_save_returns_true_for_new(self, repo):
        app = _make_application()
        assert repo.save(app) is True

    def test_save_returns_false_for_duplicate_fingerprint(self, repo):
        app1 = _make_application(fingerprint="fp1")
        app2 = _make_application(fingerprint="fp1")  # same fingerprint
        assert repo.save(app1) is True
        assert repo.save(app2) is False

    def test_get_by_id_returns_application(self, repo):
        app = _make_application()
        repo.save(app)
        found = repo.get_by_id(app.id)
        assert found is not None
        assert found.id == app.id
        assert found.company == app.company

    def test_get_by_id_returns_none_for_unknown(self, repo):
        assert repo.get_by_id("nonexistent-id") is None

    def test_get_by_fingerprint_returns_application(self, repo):
        app = _make_application(fingerprint="fp_unique")
        repo.save(app)
        found = repo.get_by_fingerprint("fp_unique")
        assert found is not None
        assert found.id == app.id

    def test_get_by_company_filters_correctly(self, repo):
        repo.save(_make_application(company="Stripe", fingerprint="fp1"))
        repo.save(_make_application(company="Uber", fingerprint="fp2"))
        results = repo.get_by_company("Stripe")
        assert len(results) == 1
        assert results[0].company == "Stripe"

    def test_get_by_company_is_case_insensitive(self, repo):
        repo.save(_make_application(company="Stripe", fingerprint="fp1"))
        assert len(repo.get_by_company("stripe")) == 1
        assert len(repo.get_by_company("STRIPE")) == 1

    def test_get_by_status_filters_correctly(self, repo):
        repo.save(_make_application(status=ApplicationStatus.SAVED, fingerprint="fp1"))
        repo.save(_make_application(status=ApplicationStatus.APPLIED, fingerprint="fp2"))
        saved = repo.get_by_status(ApplicationStatus.SAVED)
        applied = repo.get_by_status(ApplicationStatus.APPLIED)
        assert len(saved) == 1
        assert len(applied) == 1

    def test_get_active_excludes_rejected(self, repo):
        repo.save(_make_application(status=ApplicationStatus.REJECTED, fingerprint="fp1"))
        repo.save(_make_application(status=ApplicationStatus.APPLIED, fingerprint="fp2"))
        active = repo.get_active()
        assert all(a.status != ApplicationStatus.REJECTED for a in active)
        assert len(active) == 1

    def test_get_active_excludes_withdrawn(self, repo):
        repo.save(_make_application(status=ApplicationStatus.WITHDRAWN, fingerprint="fp1"))
        active = repo.get_active()
        assert len(active) == 0

    def test_get_all_returns_all(self, repo):
        repo.save(_make_application(fingerprint="fp1"))
        repo.save(_make_application(fingerprint="fp2"))
        assert len(repo.get_all()) == 2

    def test_exists_returns_true_for_saved_fingerprint(self, repo):
        app = _make_application(fingerprint="fp_exists")
        repo.save(app)
        assert repo.exists("fp_exists") is True

    def test_exists_returns_false_for_unknown_fingerprint(self, repo):
        assert repo.exists("fp_does_not_exist") is False

    def test_update_status_changes_status(self, repo):
        app = _make_application()
        repo.save(app)
        repo.update_status(app.id, ApplicationStatus.READY_TO_APPLY)
        updated = repo.get_by_id(app.id)
        assert updated.status == ApplicationStatus.READY_TO_APPLY

    def test_update_status_sets_applied_at(self, repo):
        app = _make_application()
        repo.save(app)
        repo.update_status(app.id, ApplicationStatus.APPLIED)
        updated = repo.get_by_id(app.id)
        assert updated.applied_at is not None

    def test_update_status_does_not_set_applied_at_for_other_statuses(self, repo):
        app = _make_application()
        repo.save(app)
        repo.update_status(app.id, ApplicationStatus.READY_TO_APPLY)
        updated = repo.get_by_id(app.id)
        assert updated.applied_at is None

    def test_update_status_returns_false_for_unknown_id(self, repo):
        assert repo.update_status("nonexistent", ApplicationStatus.APPLIED) is False

    def test_update_notes_changes_notes(self, repo):
        app = _make_application()
        repo.save(app)
        repo.update_notes(app.id, "Great company, great role")
        updated = repo.get_by_id(app.id)
        assert updated.notes == "Great company, great role"

    def test_context_manager_usage(self, tmp_path):
        db = str(tmp_path / "ctx.db")
        with ApplicationRepository(db_path=db) as repo:
            app = _make_application()
            assert repo.save(app) is True


# ---------------------------------------------------------------------------
# ApplicationAgent
# ---------------------------------------------------------------------------


class TestApplicationAgent:
    @pytest.fixture
    def repo(self, tmp_path):
        r = ApplicationRepository(db_path=str(tmp_path / "apps.db"))
        r.initialize()
        return r

    @pytest.fixture
    def agent(self, repo):
        return ApplicationAgent(repo=repo)

    # -- track ---------------------------------------------------------------

    def test_track_saves_new_application(self, agent, repo):
        job = _make_job()
        agent.track(job)
        assert repo.exists(job.fingerprint)

    def test_track_returns_application_and_true_for_new(self, agent):
        job = _make_job()
        app, is_new = agent.track(job)
        assert is_new is True
        assert isinstance(app, Application)
        assert app.company == job.company

    def test_track_prevents_duplicate(self, agent, repo):
        job = _make_job()
        agent.track(job)
        agent.track(job)  # second call — should not insert again
        assert len(repo.get_all()) == 1

    def test_track_returns_false_for_duplicate(self, agent):
        job = _make_job()
        agent.track(job)
        _, is_new = agent.track(job)
        assert is_new is False

    def test_track_returns_existing_application_for_duplicate(self, agent):
        job = _make_job()
        app1, _ = agent.track(job)
        app2, _ = agent.track(job)
        assert app1.id == app2.id

    def test_track_with_notes(self, agent, repo):
        job = _make_job()
        app, _ = agent.track(job, notes="Top priority")
        found = repo.get_by_id(app.id)
        assert found.notes == "Top priority"

    # -- advance -------------------------------------------------------------

    def test_advance_changes_status(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        updated = agent.advance(app.id, ApplicationStatus.READY_TO_APPLY)
        assert updated.status == ApplicationStatus.READY_TO_APPLY

    def test_advance_returns_updated_application(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        result = agent.advance(app.id, ApplicationStatus.READY_TO_APPLY)
        assert isinstance(result, Application)

    def test_advance_attaches_notes_on_transition(self, agent, repo):
        job = _make_job()
        app, _ = agent.track(job)
        agent.advance(app.id, ApplicationStatus.READY_TO_APPLY, notes="Ready!")
        updated = repo.get_by_id(app.id)
        assert updated.notes == "Ready!"

    def test_advance_raises_for_invalid_transition(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        # SAVED → INTERVIEW is not a valid transition
        with pytest.raises(ValueError, match="invalid transition"):
            agent.advance(app.id, ApplicationStatus.INTERVIEW)

    def test_advance_raises_for_terminal_status(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        agent.advance(app.id, ApplicationStatus.REJECTED)
        with pytest.raises(ValueError, match="terminal"):
            agent.advance(app.id, ApplicationStatus.APPLIED)

    def test_advance_raises_for_unknown_id(self, agent):
        with pytest.raises(ValueError, match="not found"):
            agent.advance("no-such-id", ApplicationStatus.APPLIED)

    # -- update_notes --------------------------------------------------------

    def test_update_notes_returns_updated_application(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        result = agent.update_notes(app.id, "Follow up on Friday")
        assert result is not None
        assert result.notes == "Follow up on Friday"

    def test_update_notes_returns_none_for_unknown_id(self, agent):
        assert agent.update_notes("no-such-id", "Notes") is None

    # -- active --------------------------------------------------------------

    def test_active_excludes_rejected(self, agent):
        job1 = _make_job(company="A", job_url="https://a.com/1")
        job2 = _make_job(company="B", job_url="https://b.com/2")
        app1, _ = agent.track(job1)
        agent.track(job2)
        agent.advance(app1.id, ApplicationStatus.REJECTED)
        active = agent.active()
        assert all(a.status != ApplicationStatus.REJECTED for a in active)
        assert len(active) == 1

    def test_active_excludes_withdrawn(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        agent.advance(app.id, ApplicationStatus.WITHDRAWN)
        assert len(agent.active()) == 0

    # -- stats ---------------------------------------------------------------

    def test_stats_returns_application_stats(self, agent):
        job = _make_job()
        agent.track(job)
        result = agent.stats()
        assert isinstance(result, ApplicationStats)

    def test_stats_total_count(self, agent):
        agent.track(_make_job(company="A", job_url="https://a.com/1"))
        agent.track(_make_job(company="B", job_url="https://b.com/2"))
        assert agent.stats().total == 2

    def test_stats_counts_by_status(self, agent):
        job = _make_job()
        agent.track(job)
        stats = agent.stats()
        assert stats.by_status.get("SAVED", 0) == 1

    def test_stats_active_count(self, agent):
        job1 = _make_job(company="A", job_url="https://a.com/1")
        job2 = _make_job(company="B", job_url="https://b.com/2")
        app1, _ = agent.track(job1)
        agent.track(job2)
        agent.advance(app1.id, ApplicationStatus.REJECTED)
        stats = agent.stats()
        assert stats.active_count == 1

    def test_stats_offer_count(self, agent):
        job = _make_job()
        app, _ = agent.track(job)
        agent.advance(app.id, ApplicationStatus.READY_TO_APPLY)
        agent.advance(app.id, ApplicationStatus.APPLIED)
        agent.advance(app.id, ApplicationStatus.INTERVIEW)
        agent.advance(app.id, ApplicationStatus.OFFER)
        assert agent.stats().offer_count == 1

    def test_stats_rejection_rate(self, agent):
        job1 = _make_job(company="A", job_url="https://a.com/1")
        job2 = _make_job(company="B", job_url="https://b.com/2")
        app1, _ = agent.track(job1)
        app2, _ = agent.track(job2)
        # Advance both to APPLIED
        for app in (app1, app2):
            agent.advance(app.id, ApplicationStatus.READY_TO_APPLY)
            agent.advance(app.id, ApplicationStatus.APPLIED)
        # Reject one
        agent.advance(app1.id, ApplicationStatus.REJECTED)
        stats = agent.stats()
        assert stats.rejection_rate == pytest.approx(0.5)

    def test_stats_rejection_rate_zero_with_no_substantive(self, agent):
        job = _make_job()
        agent.track(job)  # SAVED only — no substantive actions
        assert agent.stats().rejection_rate == pytest.approx(0.0)

    def test_stats_summary_is_string(self, agent):
        job = _make_job()
        agent.track(job)
        assert isinstance(agent.stats().summary(), str)
