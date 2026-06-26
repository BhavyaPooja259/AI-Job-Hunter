"""
Tests for Sprint 18 — Referral Assistant.

Coverage
--------
TestReferralStatus                 (6)  — enum values and lifecycle rules
TestReferral                       (7)  — dataclass fields and fingerprint
TestReferralMessages               (5)  — three-message container + truncation
TestReferralMessageData            (4)  — Pydantic AI output schema
TestReferralRepository             (16) — SQLite CRUD, dedup, side-effects
TestReferralAgentContactManagement (12) — track, advance, notes, active
TestReferralAgentMessages          (8)  — generate_messages (template + AI mock)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.referral_agent import ReferralAgent, ReferralStats
from referral.referral import Referral, ReferralMessageData, ReferralMessages
from referral.referral_repository import ReferralRepository
from referral.referral_status import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    ReferralStatus,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_referral(
    contact_name: str = "Priya Mehta",
    company: str = "Stripe",
    job_title: str = "Backend Engineer",
    contact_title: str = "Senior Engineer",
    platform: str = "LinkedIn",
) -> Referral:
    return Referral(
        contact_name=contact_name,
        company=company,
        job_title=job_title,
        contact_title=contact_title,
        platform=platform,
    )


def _make_agent(repo: ReferralRepository, client=None) -> ReferralAgent:
    return ReferralAgent(
        repo=repo,
        client=client,
        candidate_name="Bhavya L",
        candidate_title="Software Engineer II",
        candidate_email="bhavya@example.com",
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


_VALID_AI_PAYLOAD = {
    "linkedin_request": (
        "Hi Priya! I'm Bhavya, a Software Engineer II exploring the "
        "Backend Engineer role at Stripe. Would love to connect!"
    ),
    "referral_message": (
        "Hi Priya,\n\nThank you for connecting! I'm Bhavya, a Software Engineer II "
        "with Java and Spring Boot experience. Would you be open to referring me "
        "for the Backend Engineer role at Stripe?\n\nThank you!"
    ),
    "followup_message": (
        "Hi Priya,\n\nJust following up on my earlier message about the Backend "
        "Engineer role at Stripe. Thank you for your time!"
    ),
}


# ---------------------------------------------------------------------------
# TestReferralStatus
# ---------------------------------------------------------------------------


class TestReferralStatus:
    def test_all_six_statuses_exist(self):
        statuses = {s.value for s in ReferralStatus}
        expected = {
            "NOT_CONTACTED", "REQUEST_SENT", "CONNECTED",
            "REFERRED", "DECLINED", "NO_RESPONSE",
        }
        assert statuses == expected

    def test_statuses_are_string_compatible(self):
        assert ReferralStatus.NOT_CONTACTED == "NOT_CONTACTED"
        assert ReferralStatus.REFERRED == "REFERRED"

    def test_terminal_statuses(self):
        assert ReferralStatus.REFERRED in TERMINAL_STATUSES
        assert ReferralStatus.DECLINED in TERMINAL_STATUSES
        assert ReferralStatus.NOT_CONTACTED not in TERMINAL_STATUSES

    def test_valid_transitions_covers_all_statuses(self):
        for status in ReferralStatus:
            assert status in VALID_TRANSITIONS

    def test_no_response_can_retry(self):
        assert ReferralStatus.REQUEST_SENT in VALID_TRANSITIONS[ReferralStatus.NO_RESPONSE]

    def test_terminal_statuses_have_empty_transitions(self):
        for terminal in TERMINAL_STATUSES:
            assert VALID_TRANSITIONS[terminal] == frozenset()


# ---------------------------------------------------------------------------
# TestReferral
# ---------------------------------------------------------------------------


class TestReferral:
    def test_default_status_is_not_contacted(self):
        ref = Referral(contact_name="Alice", company="Google")
        assert ref.status == ReferralStatus.NOT_CONTACTED

    def test_id_auto_generated(self):
        r1 = Referral(contact_name="Alice", company="Google")
        r2 = Referral(contact_name="Alice", company="Google")
        assert r1.id != r2.id

    def test_fingerprint_computed_from_fields(self):
        ref = _make_referral()
        assert isinstance(ref.fingerprint, str)
        assert len(ref.fingerprint) == 16

    def test_fingerprint_same_for_same_contact(self):
        r1 = Referral(contact_name="Priya Mehta", company="Stripe", job_title="Backend Engineer")
        r2 = Referral(contact_name="PRIYA MEHTA", company="STRIPE", job_title="backend engineer")
        assert r1.fingerprint == r2.fingerprint

    def test_fingerprint_different_for_different_job(self):
        r1 = Referral(contact_name="Priya", company="Stripe", job_title="Backend Engineer")
        r2 = Referral(contact_name="Priya", company="Stripe", job_title="Frontend Engineer")
        assert r1.fingerprint != r2.fingerprint

    def test_fingerprint_different_for_different_company(self):
        r1 = Referral(contact_name="Priya", company="Stripe")
        r2 = Referral(contact_name="Priya", company="Uber")
        assert r1.fingerprint != r2.fingerprint

    def test_str_includes_name_and_company(self):
        ref = _make_referral(contact_name="Priya Mehta", company="Stripe")
        result = str(ref)
        assert "Priya Mehta" in result
        assert "Stripe" in result

    def test_str_includes_status(self):
        ref = _make_referral()
        assert "NOT_CONTACTED" in str(ref)

    def test_str_includes_job_title_when_present(self):
        ref = _make_referral(job_title="Backend Engineer")
        assert "Backend Engineer" in str(ref)


# ---------------------------------------------------------------------------
# TestReferralMessages
# ---------------------------------------------------------------------------


class TestReferralMessages:
    def test_all_three_messages_present(self):
        msgs = ReferralMessages(
            linkedin_request="Hi there!",
            referral_message="Would you refer me?",
            followup_message="Just following up.",
        )
        assert msgs.linkedin_request
        assert msgs.referral_message
        assert msgs.followup_message

    def test_short_linkedin_not_truncated(self):
        short = "Hi! Would love to connect."
        msgs = ReferralMessages(
            linkedin_request=short,
            referral_message="long msg",
            followup_message="follow up",
        )
        assert msgs.linkedin_request == short

    def test_long_linkedin_truncated_to_300(self):
        too_long = "X" * 400
        msgs = ReferralMessages(
            linkedin_request=too_long,
            referral_message="body",
            followup_message="follow",
        )
        assert len(msgs.linkedin_request) == 300

    def test_truncated_linkedin_ends_with_ellipsis(self):
        msgs = ReferralMessages(
            linkedin_request="A" * 400,
            referral_message="body",
            followup_message="follow",
        )
        assert msgs.linkedin_request.endswith("...")

    def test_linkedin_exactly_300_not_truncated(self):
        exact = "B" * 300
        msgs = ReferralMessages(
            linkedin_request=exact,
            referral_message="body",
            followup_message="follow",
        )
        assert len(msgs.linkedin_request) == 300
        assert not msgs.linkedin_request.endswith("...")


# ---------------------------------------------------------------------------
# TestReferralMessageData
# ---------------------------------------------------------------------------


class TestReferralMessageData:
    def test_valid_payload_parses(self):
        data = ReferralMessageData(**_VALID_AI_PAYLOAD)
        assert "Priya" in data.linkedin_request

    def test_all_three_fields_present(self):
        data = ReferralMessageData(**_VALID_AI_PAYLOAD)
        assert data.linkedin_request
        assert data.referral_message
        assert data.followup_message

    def test_linkedin_too_long_raises(self):
        payload = {**_VALID_AI_PAYLOAD, "linkedin_request": "X" * 301}
        with pytest.raises(Exception):
            ReferralMessageData(**payload)

    def test_linkedin_too_short_raises(self):
        payload = {**_VALID_AI_PAYLOAD, "linkedin_request": "Hi"}
        with pytest.raises(Exception):
            ReferralMessageData(**payload)


# ---------------------------------------------------------------------------
# TestReferralRepository
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    r = ReferralRepository(db_path=str(tmp_path / "test.db"))
    r.initialize()
    yield r
    r.close()


class TestReferralRepository:
    def test_initialize_creates_table(self, repo):
        row = repo._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='referrals'"
        ).fetchone()
        assert row is not None

    def test_save_returns_true_for_new(self, repo):
        ref = _make_referral()
        assert repo.save(ref) is True

    def test_save_returns_false_for_duplicate(self, repo):
        ref = _make_referral()
        repo.save(ref)
        dup = _make_referral()  # same contact_name + company + job_title
        assert repo.save(dup) is False

    def test_get_by_id_returns_referral(self, repo):
        ref = _make_referral()
        repo.save(ref)
        loaded = repo.get_by_id(ref.id)
        assert loaded is not None
        assert loaded.id == ref.id
        assert loaded.contact_name == ref.contact_name

    def test_get_by_id_returns_none_for_unknown(self, repo):
        assert repo.get_by_id("does-not-exist") is None

    def test_get_by_fingerprint_returns_referral(self, repo):
        ref = _make_referral()
        repo.save(ref)
        loaded = repo.get_by_fingerprint(ref.fingerprint)
        assert loaded is not None
        assert loaded.company == ref.company

    def test_get_by_company_filters(self, repo):
        repo.save(_make_referral(contact_name="Alice", company="Stripe"))
        repo.save(_make_referral(contact_name="Bob", company="Uber", job_title="SRE"))
        stripe_refs = repo.get_by_company("Stripe")
        assert len(stripe_refs) == 1
        assert stripe_refs[0].contact_name == "Alice"

    def test_get_by_company_case_insensitive(self, repo):
        repo.save(_make_referral(contact_name="Alice", company="Stripe"))
        assert len(repo.get_by_company("stripe")) == 1
        assert len(repo.get_by_company("STRIPE")) == 1

    def test_get_by_status_filters(self, repo):
        ref = _make_referral()
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        results = repo.get_by_status(ReferralStatus.REQUEST_SENT)
        assert any(r.id == ref.id for r in results)

    def test_get_active_excludes_referred(self, repo):
        ref = _make_referral(contact_name="Alice", company="Stripe")
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        repo.update_status(ref.id, ReferralStatus.CONNECTED)
        repo.update_status(ref.id, ReferralStatus.REFERRED)
        assert not any(r.id == ref.id for r in repo.get_active())

    def test_get_active_excludes_declined(self, repo):
        ref = _make_referral(contact_name="Bob", company="Uber", job_title="SRE")
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        repo.update_status(ref.id, ReferralStatus.DECLINED)
        assert not any(r.id == ref.id for r in repo.get_active())

    def test_update_status_sets_contacted_at(self, repo):
        ref = _make_referral()
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        loaded = repo.get_by_id(ref.id)
        assert loaded.contacted_at is not None

    def test_update_status_preserves_contacted_at_on_retry(self, repo):
        ref = _make_referral()
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        first_contact = repo.get_by_id(ref.id).contacted_at
        repo.update_status(ref.id, ReferralStatus.NO_RESPONSE)
        repo.update_status(ref.id, ReferralStatus.REQUEST_SENT)
        second_contact = repo.get_by_id(ref.id).contacted_at
        assert first_contact == second_contact

    def test_update_status_sets_connected_at(self, repo):
        ref = _make_referral()
        repo.save(ref)
        repo.update_status(ref.id, ReferralStatus.CONNECTED)
        loaded = repo.get_by_id(ref.id)
        assert loaded.connected_at is not None

    def test_update_notes(self, repo):
        ref = _make_referral()
        repo.save(ref)
        repo.update_notes(ref.id, "Met at PyCon 2024")
        loaded = repo.get_by_id(ref.id)
        assert loaded.notes == "Met at PyCon 2024"

    def test_update_messages(self, repo):
        ref = _make_referral()
        repo.save(ref)
        msgs = ReferralMessages(
            linkedin_request="Hi!",
            referral_message="Would you refer me?",
            followup_message="Following up.",
        )
        repo.update_messages(ref.id, msgs)
        loaded = repo.get_by_id(ref.id)
        assert loaded.linkedin_message == "Hi!"
        assert loaded.referral_message == "Would you refer me?"
        assert loaded.followup_message == "Following up."

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with ReferralRepository(db_path=db_path) as r:
            ref = _make_referral()
            assert r.save(ref) is True
            assert r.get_by_id(ref.id) is not None


# ---------------------------------------------------------------------------
# TestReferralAgentContactManagement
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_repo(tmp_path):
    r = ReferralRepository(db_path=str(tmp_path / "agent.db"))
    r.initialize()
    yield r
    r.close()


class TestReferralAgentContactManagement:
    def test_track_saves_new_contact(self, agent_repo):
        agent = _make_agent(agent_repo)
        _, is_new = agent.track("Priya Mehta", "Stripe", job_title="Backend Engineer")
        assert is_new is True

    def test_track_returns_referral(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        assert isinstance(ref, Referral)
        assert ref.contact_name == "Priya Mehta"

    def test_track_prevents_duplicate(self, agent_repo):
        agent = _make_agent(agent_repo)
        agent.track("Priya Mehta", "Stripe", job_title="BE")
        _, is_new = agent.track("Priya Mehta", "Stripe", job_title="BE")
        assert is_new is False
        assert len(agent_repo.get_all()) == 1

    def test_track_returns_existing_on_duplicate(self, agent_repo):
        agent = _make_agent(agent_repo)
        original, _ = agent.track("Priya Mehta", "Stripe", job_title="BE")
        returned, _ = agent.track("Priya Mehta", "Stripe", job_title="BE")
        assert returned.id == original.id

    def test_advance_changes_status(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        updated = agent.advance(ref.id, ReferralStatus.REQUEST_SENT)
        assert updated.status == ReferralStatus.REQUEST_SENT

    def test_advance_returns_updated_referral(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        result = agent.advance(ref.id, ReferralStatus.REQUEST_SENT)
        assert isinstance(result, Referral)

    def test_advance_raises_for_invalid_transition(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        with pytest.raises(ValueError, match="invalid transition"):
            agent.advance(ref.id, ReferralStatus.CONNECTED)

    def test_advance_raises_for_terminal_status(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        agent.advance(ref.id, ReferralStatus.REQUEST_SENT)
        agent.advance(ref.id, ReferralStatus.DECLINED)
        with pytest.raises(ValueError, match="terminal"):
            agent.advance(ref.id, ReferralStatus.REQUEST_SENT)

    def test_advance_raises_for_unknown_id(self, agent_repo):
        agent = _make_agent(agent_repo)
        with pytest.raises(ValueError, match="not found"):
            agent.advance("nonexistent-id", ReferralStatus.REQUEST_SENT)

    def test_update_notes_returns_updated(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        updated = agent.update_notes(ref.id, "Connected at conference")
        assert updated is not None
        assert updated.notes == "Connected at conference"

    def test_update_notes_returns_none_for_unknown(self, agent_repo):
        agent = _make_agent(agent_repo)
        assert agent.update_notes("bad-id", "note") is None

    def test_active_excludes_referred(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Priya Mehta", "Stripe")
        agent.advance(ref.id, ReferralStatus.REQUEST_SENT)
        agent.advance(ref.id, ReferralStatus.CONNECTED)
        agent.advance(ref.id, ReferralStatus.REFERRED)
        assert not any(r.id == ref.id for r in agent.active())

    def test_active_excludes_declined(self, agent_repo):
        agent = _make_agent(agent_repo)
        ref, _ = agent.track("Bob Singh", "Uber", job_title="SRE")
        agent.advance(ref.id, ReferralStatus.REQUEST_SENT)
        agent.advance(ref.id, ReferralStatus.DECLINED)
        assert not any(r.id == ref.id for r in agent.active())


# ---------------------------------------------------------------------------
# TestReferralAgentStats
# ---------------------------------------------------------------------------


class TestReferralAgentStats:
    @pytest.fixture
    def populated_repo(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "stats.db"))
        repo.initialize()

        r1 = Referral(contact_name="Alice", company="Stripe", job_title="BE")
        r2 = Referral(contact_name="Bob", company="Uber", job_title="SRE")
        r3 = Referral(contact_name="Carol", company="Google", job_title="SWE")
        repo.save(r1)
        repo.save(r2)
        repo.save(r3)

        repo.update_status(r1.id, ReferralStatus.REQUEST_SENT)
        repo.update_status(r1.id, ReferralStatus.CONNECTED)
        repo.update_status(r2.id, ReferralStatus.REQUEST_SENT)
        repo.update_status(r2.id, ReferralStatus.REFERRED)

        yield repo
        repo.close()

    def test_stats_returns_referral_stats(self, populated_repo):
        agent = _make_agent(populated_repo)
        result = agent.stats()
        assert isinstance(result, ReferralStats)

    def test_stats_total_count(self, populated_repo):
        agent = _make_agent(populated_repo)
        assert agent.stats().total == 3

    def test_stats_referred_count(self, populated_repo):
        agent = _make_agent(populated_repo)
        assert agent.stats().referred_count == 1

    def test_stats_summary_is_string(self, populated_repo):
        agent = _make_agent(populated_repo)
        summary = agent.stats().summary()
        assert isinstance(summary, str)
        assert "Total" in summary


# ---------------------------------------------------------------------------
# TestReferralAgentMessages
# ---------------------------------------------------------------------------


class TestReferralAgentMessages:
    @pytest.fixture
    def repo_with_contact(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "msg.db"))
        repo.initialize()
        ref = _make_referral()
        repo.save(ref)
        yield repo, ref
        repo.close()

    def test_generate_returns_referral_messages(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert isinstance(msgs, ReferralMessages)

    def test_linkedin_within_300_chars(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert len(msgs.linkedin_request) <= 300

    def test_linkedin_includes_company(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert "Stripe" in msgs.linkedin_request

    def test_referral_message_includes_candidate_name(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert "Bhavya" in msgs.referral_message

    def test_followup_includes_company(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert "Stripe" in msgs.followup_message

    def test_save_true_persists_messages(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        agent.generate_messages(ref, save=True)
        loaded = repo.get_by_id(ref.id)
        assert loaded.linkedin_message != ""
        assert loaded.referral_message != ""
        assert loaded.followup_message != ""

    def test_save_false_does_not_persist(self, repo_with_contact):
        repo, ref = repo_with_contact
        agent = _make_agent(repo)
        agent.generate_messages(ref, save=False)
        loaded = repo.get_by_id(ref.id)
        assert loaded.linkedin_message == ""

    def test_with_mocked_client_calls_generate_content(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "ai.db"))
        repo.initialize()
        ref = _make_referral()
        repo.save(ref)

        client = _mock_client(_VALID_AI_PAYLOAD)
        agent = _make_agent(repo, client=client)
        msgs = agent.generate_messages(ref, save=False)

        assert client.models.generate_content.call_count == 1
        assert isinstance(msgs, ReferralMessages)
        repo.close()

    def test_falls_back_to_templates_on_empty_ai_response(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "fallback.db"))
        repo.initialize()
        ref = _make_referral()
        repo.save(ref)

        agent = _make_agent(repo, client=_empty_response_client())
        msgs = agent.generate_messages(ref, save=False)
        assert isinstance(msgs, ReferralMessages)
        assert "Stripe" in msgs.linkedin_request
        repo.close()

    def test_falls_back_to_templates_on_ai_exception(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "exc.db"))
        repo.initialize()
        ref = _make_referral()
        repo.save(ref)

        agent = _make_agent(repo, client=_error_client(RuntimeError("network error")))
        msgs = agent.generate_messages(ref, save=False)
        assert isinstance(msgs, ReferralMessages)
        repo.close()

    def test_template_messages_without_job_title(self, tmp_path):
        repo = ReferralRepository(db_path=str(tmp_path / "nojob.db"))
        repo.initialize()
        ref = Referral(contact_name="Raj Kumar", company="Adobe")
        repo.save(ref)

        agent = _make_agent(repo)
        msgs = agent.generate_messages(ref, save=False)
        assert len(msgs.linkedin_request) <= 300
        assert "Adobe" in msgs.linkedin_request
        repo.close()
