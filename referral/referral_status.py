"""
ReferralStatus — six-state lifecycle for recruiter / referrer outreach.

Transitions
-----------
  NOT_CONTACTED  ──▶  REQUEST_SENT
  REQUEST_SENT   ──▶  CONNECTED | DECLINED | NO_RESPONSE
  CONNECTED      ──▶  REFERRED  | DECLINED
  NO_RESPONSE    ──▶  REQUEST_SENT  (send a follow-up)
  REFERRED       ──   terminal (referral secured)
  DECLINED       ──   terminal (contact declined to refer)

Terminal statuses have an empty frozenset in VALID_TRANSITIONS; calling
advance() on them raises ValueError so a closed loop is always visible.
"""

from enum import Enum


class ReferralStatus(str, Enum):
    NOT_CONTACTED = "NOT_CONTACTED"
    REQUEST_SENT = "REQUEST_SENT"
    CONNECTED = "CONNECTED"
    REFERRED = "REFERRED"
    DECLINED = "DECLINED"
    NO_RESPONSE = "NO_RESPONSE"


VALID_TRANSITIONS: dict["ReferralStatus", frozenset["ReferralStatus"]] = {
    ReferralStatus.NOT_CONTACTED: frozenset({
        ReferralStatus.REQUEST_SENT,
    }),
    ReferralStatus.REQUEST_SENT: frozenset({
        ReferralStatus.CONNECTED,
        ReferralStatus.DECLINED,
        ReferralStatus.NO_RESPONSE,
    }),
    ReferralStatus.CONNECTED: frozenset({
        ReferralStatus.REFERRED,
        ReferralStatus.DECLINED,
    }),
    ReferralStatus.NO_RESPONSE: frozenset({
        ReferralStatus.REQUEST_SENT,  # send a follow-up
    }),
    ReferralStatus.REFERRED: frozenset(),
    ReferralStatus.DECLINED: frozenset(),
}

TERMINAL_STATUSES: frozenset[ReferralStatus] = frozenset({
    ReferralStatus.REFERRED,
    ReferralStatus.DECLINED,
})
