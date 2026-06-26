"""
ReferralAgent — manages recruiter / referrer contacts and generates outreach messages.

Responsibilities
----------------
1. track()             — add a contact; return existing if already tracked.
2. advance()           — move status, enforcing VALID_TRANSITIONS rules.
3. update_notes()      — free-form notes without changing status.
4. active()            — list all non-terminal referrals.
5. stats()             — aggregate outreach statistics.
6. generate_messages() — produce LinkedIn request, referral ask, and follow-up.

Message generation
------------------
  client=None   → deterministic templates (works offline, no API key needed).
  client=<obj>  → Gemini AI path (deferred; architecture ready to plug in).

If the AI call fails for any reason, the agent falls back to templates and
still returns a usable ReferralMessages.  The caller never sees an exception
from generate_messages().

Template design
---------------
Templates are filled from contact fields (first name, company, job_title)
and candidate fields (name, title).  When job_title is empty the messages
reference "opportunities at {company}" so they are still coherent.

Separation of concerns
----------------------
ReferralRepository — pure SQL, no lifecycle rules.
ReferralAgent      — lifecycle validation, duplicate prevention, message logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from referral.referral import Referral, ReferralMessageData, ReferralMessages
from referral.referral_repository import ReferralRepository
from referral.referral_status import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    ReferralStatus,
)

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class ReferralStats:
    """Aggregate statistics across all tracked referral contacts."""

    total: int
    by_status: dict[str, int]      # status.value → count (non-zero only)
    active_count: int
    referred_count: int
    response_rate: float            # (CONNECTED + REFERRED) / contacts_with_request

    def summary(self) -> str:
        return (
            f"Total: {self.total}  |  Active: {self.active_count}  |  "
            f"Referred: {self.referred_count}  |  "
            f"Response rate: {self.response_rate:.0%}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReferralAgent:
    """
    Manages recruiter / referrer contacts and generates outreach messages.

    Parameters
    ----------
    repo
        An initialised ReferralRepository.  The agent does not own the
        connection lifecycle — the caller manages open / close.
    client
        An initialised google.genai.Client.  Pass None (default) for
        template-only mode — fully functional without an API key.
    candidate_name
        The candidate's full name — appears in every generated message.
    candidate_title
        The candidate's job title — used in the referral request body.
    candidate_email
        Optional email — may be included in some message variants.
    model
        Gemini model ID used when client is not None.
    """

    def __init__(
        self,
        repo: ReferralRepository,
        client: "genai.Client | None" = None,
        candidate_name: str = "",
        candidate_title: str = "Software Engineer II",
        candidate_email: str = "",
        model: str = "gemini-2.0-flash",
    ) -> None:
        self._repo = repo
        self._client = client
        self._candidate_name = candidate_name
        self._candidate_title = candidate_title
        self._candidate_email = candidate_email
        self._model = model

    # ------------------------------------------------------------------ #
    # Contact management
    # ------------------------------------------------------------------ #

    def track(
        self,
        contact_name: str,
        company: str,
        contact_title: str = "",
        job_title: str = "",
        job_url: str = "",
        contact_url: str = "",
        platform: str = "LinkedIn",
        notes: str = "",
    ) -> tuple[Referral, bool]:
        """
        Add a referral contact to the pipeline.

        Returns
        -------
        (referral, is_new)
            is_new is True when the contact was saved for the first time.
            is_new is False when the same (contact_name, company, job_title)
            already exists; the existing Referral is returned unchanged.
        """
        referral = Referral(
            contact_name=contact_name,
            company=company,
            contact_title=contact_title,
            job_title=job_title,
            job_url=job_url,
            contact_url=contact_url,
            platform=platform,
            notes=notes,
        )

        if self._repo.exists(referral.fingerprint):
            existing = self._repo.get_by_fingerprint(referral.fingerprint)
            logger.info(
                "already tracking %s @ %s — returning existing referral",
                contact_name, company,
            )
            return existing, False

        self._repo.save(referral)
        logger.info("tracking new contact %s @ %s (id=%s)", contact_name, company, referral.id)
        return referral, True

    def advance(
        self,
        referral_id: str,
        new_status: ReferralStatus,
        notes: str = "",
    ) -> Referral:
        """
        Move a referral to a new status, enforcing VALID_TRANSITIONS rules.

        Raises
        ------
        ValueError
            If referral_id is not found, the transition is not in
            VALID_TRANSITIONS, or the current status is terminal.
        """
        referral = self._repo.get_by_id(referral_id)
        if referral is None:
            raise ValueError(f"referral {referral_id!r} not found")

        valid_next = VALID_TRANSITIONS.get(referral.status, frozenset())
        if new_status not in valid_next:
            if referral.status in TERMINAL_STATUSES:
                raise ValueError(
                    f"cannot advance from terminal status {referral.status.value!r}"
                )
            valid_str = ", ".join(s.value for s in valid_next) or "none"
            raise ValueError(
                f"invalid transition: {referral.status.value} → {new_status.value}"
                f"  (valid next: {valid_str})"
            )

        self._repo.update_status(referral_id, new_status)
        if notes:
            self._repo.update_notes(referral_id, notes)

        updated = self._repo.get_by_id(referral_id)
        logger.info(
            "advanced %s @ %s: %s → %s",
            referral.contact_name, referral.company,
            referral.status.value, new_status.value,
        )
        return updated

    def update_notes(self, referral_id: str, notes: str) -> Referral | None:
        """
        Update free-form notes without changing status.

        Returns the updated Referral, or None if referral_id is not found.
        """
        if not self._repo.update_notes(referral_id, notes):
            return None
        return self._repo.get_by_id(referral_id)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def active(self) -> list[Referral]:
        """All referrals that are not REFERRED or DECLINED."""
        return self._repo.get_active()

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> ReferralStats:
        """Compute aggregate outreach statistics."""
        referrals = self._repo.get_all()
        total = len(referrals)

        by_status: dict[str, int] = {}
        for r in referrals:
            key = r.status.value
            by_status[key] = by_status.get(key, 0) + 1

        _ACTIVE = {
            ReferralStatus.NOT_CONTACTED,
            ReferralStatus.REQUEST_SENT,
            ReferralStatus.CONNECTED,
            ReferralStatus.NO_RESPONSE,
        }

        active_count = sum(1 for r in referrals if r.status in _ACTIVE)
        referred_count = by_status.get(ReferralStatus.REFERRED.value, 0)
        connected_count = by_status.get(ReferralStatus.CONNECTED.value, 0)

        # Contacts where a request was sent (excluding NOT_CONTACTED)
        contacted = sum(
            1 for r in referrals if r.status != ReferralStatus.NOT_CONTACTED
        )
        responded = connected_count + referred_count
        response_rate = responded / contacted if contacted > 0 else 0.0

        return ReferralStats(
            total=total,
            by_status=by_status,
            active_count=active_count,
            referred_count=referred_count,
            response_rate=response_rate,
        )

    # ------------------------------------------------------------------ #
    # Message generation
    # ------------------------------------------------------------------ #

    def generate_messages(self, referral: Referral, save: bool = True) -> ReferralMessages:
        """
        Generate all three outreach messages for a referral contact.

        Parameters
        ----------
        referral
            The contact to generate messages for.
        save
            When True, persist the messages back to the repository so
            they are available on the Referral record later.

        Returns
        -------
        ReferralMessages
            Always returns usable messages.  Falls back to templates
            if the AI call fails or client is None.
        """
        logger.info(
            "generating messages for %s @ %s", referral.contact_name, referral.company
        )

        if self._client is not None:
            messages = self._generate_with_ai(referral)
        else:
            messages = self._generate_from_templates(referral)

        if save:
            self._repo.update_messages(referral.id, messages)

        return messages

    # ------------------------------------------------------------------ #
    # AI path
    # ------------------------------------------------------------------ #

    def _generate_with_ai(self, referral: Referral) -> ReferralMessages:
        """Call Gemini to produce personalised messages, fallback on any error."""
        from google.genai import types

        system = self._build_system_prompt()
        user_msg = self._build_user_message(referral)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=ReferralMessageData,
                    max_output_tokens=800,
                ),
            )

            text = getattr(response, "text", None)
            if not text:
                logger.warning(
                    "empty Gemini response for %s @ %s — falling back to templates",
                    referral.contact_name, referral.company,
                )
                return self._generate_from_templates(referral)

            data = ReferralMessageData.model_validate_json(text)
            return ReferralMessages(
                linkedin_request=data.linkedin_request,
                referral_message=data.referral_message,
                followup_message=data.followup_message,
            )

        except Exception as exc:
            logger.warning(
                "message AI generation failed for %s @ %s: %s — falling back to templates",
                referral.contact_name, referral.company, exc,
            )
            return self._generate_from_templates(referral)

    def _build_system_prompt(self) -> str:
        name = self._candidate_name or "the candidate"
        title = self._candidate_title
        return (
            f"You are a professional career coach helping {name}, a {title}, "
            "craft concise and genuine outreach messages.\n\n"
            "YOUR RULES:\n"
            "1. linkedin_request must be ≤300 characters — count carefully.\n"
            "2. Be warm but professional — not sycophantic.\n"
            "3. Do NOT invent experience or achievements.\n"
            "4. Address the contact by first name.\n"
            "5. referral_message is sent after connecting — assume you are now connected.\n"
            "6. followup_message is a gentle nudge — keep it brief (2–3 sentences).\n"
        )

    def _build_user_message(self, referral: Referral) -> str:
        lines = [
            "Generate outreach messages for the following contact.\n",
            "CANDIDATE",
            f"  Name  : {self._candidate_name or 'the candidate'}",
            f"  Title : {self._candidate_title}",
            "",
            "CONTACT",
            f"  Name    : {referral.contact_name}",
            f"  Title   : {referral.contact_title or 'not specified'}",
            f"  Company : {referral.company}",
        ]
        if referral.job_title:
            lines.append(f"  Role    : {referral.job_title}")
        if referral.job_url:
            lines.append(f"  URL     : {referral.job_url}")
        lines += [
            "",
            "Return all three messages in the required JSON format.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Template path
    # ------------------------------------------------------------------ #

    def _generate_from_templates(self, referral: Referral) -> ReferralMessages:
        """Build all three messages from deterministic templates (no AI)."""
        first_name = referral.contact_name.split()[0]
        company = referral.company
        job_title = referral.job_title
        candidate = self._candidate_name or "I"

        # ── LinkedIn connection request (≤ 300 chars) ─────────────────────
        if job_title:
            linkedin = (
                f"Hi {first_name}! I'm {candidate}, a {self._candidate_title} "
                f"exploring the {job_title} role at {company}. "
                f"Would love to connect — thanks!"
            )
        else:
            linkedin = (
                f"Hi {first_name}! I'm {candidate}, a {self._candidate_title} "
                f"interested in opportunities at {company}. "
                f"Would love to connect — thanks!"
            )

        # ── Referral request ──────────────────────────────────────────────
        if job_title:
            role_line = (
                f"I'm very interested in the {job_title} role at {company}"
            )
            if referral.job_url:
                role_line += f" ({referral.job_url})"
            role_line += "."
        else:
            role_line = f"I'm actively exploring opportunities at {company}."

        referral_msg = (
            f"Hi {first_name},\n\n"
            f"Thank you for connecting! I'm {candidate}, a {self._candidate_title} "
            f"with experience in Java, Spring Boot, and microservices.\n\n"
            f"{role_line} Would you be open to referring me or sharing your "
            f"experience about the team and culture?\n\n"
            f"I'd be happy to share my resume — please let me know.\n\n"
            f"Thank you,\n{candidate}"
        )

        # ── Follow-up ─────────────────────────────────────────────────────
        if job_title:
            subject = f"the {job_title} role at {company}"
        else:
            subject = f"opportunities at {company}"

        followup_msg = (
            f"Hi {first_name},\n\n"
            f"I hope you're doing well! I wanted to follow up on my earlier message "
            f"about {subject}. I remain very interested and would appreciate any "
            f"insights you might share.\n\n"
            f"Thank you for your time!\n{candidate}"
        )

        return ReferralMessages(
            linkedin_request=linkedin,
            referral_message=referral_msg,
            followup_message=followup_msg,
        )
