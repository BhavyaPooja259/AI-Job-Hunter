"""
Referral data model — contacts tracked in the outreach pipeline.

Three types are defined here:

  ReferralMessages
      The three messages generated for a referral contact.
      linkedin_request is hard-capped at 300 characters in __post_init__.

  ReferralMessageData
      Pydantic model that mirrors ReferralMessages.  Used as the
      response_schema when Gemini generates messages (AI path).  Importing
      it does not require google-genai to be installed.

  Referral
      The core contact record.  Required fields: contact_name, company.
      fingerprint is a computed property — SHA-256 of
      '{contact_name}::{company}::{job_title}' (lower-cased).  Stored
      as a UNIQUE column in SQLite for duplicate prevention.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

from referral.referral_status import ReferralStatus


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class ReferralMessages:
    """
    Three outreach messages generated for a referral contact.

    linkedin_request  ≤300 characters (LinkedIn's connection note limit).
    referral_message  longer form — sent after connecting.
    followup_message  shorter — sent when there is no initial response.
    """

    linkedin_request: str
    referral_message: str
    followup_message: str

    def __post_init__(self) -> None:
        if len(self.linkedin_request) > 300:
            self.linkedin_request = self.linkedin_request[:297] + "..."


class ReferralMessageData(BaseModel):
    """
    Pydantic schema for Gemini's JSON response when generating messages.

    This class is intentionally separate from ReferralMessages so that
    AI validation (min/max_length) does not affect the template path.
    Convert to ReferralMessages once the response is validated:

        data = ReferralMessageData.model_validate_json(response.text)
        msgs = ReferralMessages(
            linkedin_request=data.linkedin_request,
            referral_message=data.referral_message,
            followup_message=data.followup_message,
        )
    """

    linkedin_request: str = Field(
        min_length=10,
        max_length=300,
        description="Connection request note (≤300 chars).",
    )
    referral_message: str = Field(
        min_length=20,
        description="Longer referral-ask message sent after connecting.",
    )
    followup_message: str = Field(
        min_length=20,
        description="Shorter follow-up message when there is no initial response.",
    )


# ---------------------------------------------------------------------------
# Referral
# ---------------------------------------------------------------------------


@dataclass
class Referral:
    """
    A recruiter or hiring-manager contact in the outreach pipeline.

    Only contact_name and company are required.  All other fields
    default to empty strings / NOT_CONTACTED so records can be created
    with minimal information and enriched later.
    """

    contact_name: str
    company: str

    id: str = field(default_factory=_new_id)
    contact_title: str = ""       # e.g. "Senior Engineer at Stripe"
    platform: str = "LinkedIn"
    job_title: str = ""           # specific role being targeted
    job_url: str = ""
    contact_url: str = ""         # e.g. LinkedIn profile URL
    status: ReferralStatus = ReferralStatus.NOT_CONTACTED
    notes: str = ""
    linkedin_message: str = ""
    referral_message: str = ""
    followup_message: str = ""
    contacted_at: datetime | None = None
    connected_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def fingerprint(self) -> str:
        """
        16-char hex fingerprint unique to (contact_name, company, job_title).

        Two Referral objects with the same three fields produce the same
        fingerprint regardless of case, so the repository can reject
        duplicates via a UNIQUE constraint without the agent needing to
        run an existence query before every insert.
        """
        raw = (
            f"{self.contact_name.lower()}::"
            f"{self.company.lower()}::"
            f"{self.job_title.lower()}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        role_part = f" for {self.job_title}" if self.job_title else ""
        return f"[{self.status.value}]  {self.contact_name} @ {self.company}{role_part}"
