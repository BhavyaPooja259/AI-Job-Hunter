"""
ReferralRepository — SQLite-backed persistence for Referral records.

Architecture
------------
Thin data-access layer only.  No lifecycle rules live here — those are
enforced by ReferralAgent.

Usage
-----
    # Context manager (production):
    with ReferralRepository() as repo:
        repo.save(referral)

    # Explicit initialize/close (tests):
    repo = ReferralRepository(db_path=str(tmp_path / "test.db"))
    repo.initialize()
    repo.save(referral)

fingerprint has a UNIQUE constraint so save() returns False on duplicate
inserts without raising — the agent decides what to do with duplicates.

Automatic timestamps
--------------------
update_status() sets:
  • contacted_at — on first REQUEST_SENT (COALESCE preserves it on retries)
  • connected_at — on CONNECTED
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from referral.referral import Referral, ReferralMessages
from referral.referral_status import ReferralStatus

logger = logging.getLogger(__name__)


class ReferralRepository:
    """SQLite-backed persistence for Referral records."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS referrals (
            id               TEXT PRIMARY KEY,
            fingerprint      TEXT UNIQUE NOT NULL,
            contact_name     TEXT NOT NULL,
            contact_title    TEXT NOT NULL DEFAULT '',
            company          TEXT NOT NULL,
            platform         TEXT NOT NULL DEFAULT 'LinkedIn',
            job_title        TEXT NOT NULL DEFAULT '',
            job_url          TEXT NOT NULL DEFAULT '',
            contact_url      TEXT NOT NULL DEFAULT '',
            status           TEXT NOT NULL DEFAULT 'NOT_CONTACTED',
            notes            TEXT NOT NULL DEFAULT '',
            linkedin_message  TEXT NOT NULL DEFAULT '',
            referral_message  TEXT NOT NULL DEFAULT '',
            followup_message  TEXT NOT NULL DEFAULT '',
            contacted_at     TEXT,
            connected_at     TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        )
    """

    def __init__(self, db_path: str = "data/referrals.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Open the connection and create the table if it does not exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()
        logger.debug("ReferralRepository initialized at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ReferralRepository":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def save(self, referral: Referral) -> bool:
        """
        Insert a referral.

        Returns True if saved, False if fingerprint already exists.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO referrals (
                    id, fingerprint, contact_name, contact_title, company,
                    platform, job_title, job_url, contact_url, status, notes,
                    linkedin_message, referral_message, followup_message,
                    contacted_at, connected_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    referral.id,
                    referral.fingerprint,
                    referral.contact_name,
                    referral.contact_title,
                    referral.company,
                    referral.platform,
                    referral.job_title,
                    referral.job_url,
                    referral.contact_url,
                    referral.status.value,
                    referral.notes,
                    referral.linkedin_message,
                    referral.referral_message,
                    referral.followup_message,
                    referral.contacted_at.isoformat() if referral.contacted_at else None,
                    referral.connected_at.isoformat() if referral.connected_at else None,
                    referral.created_at.isoformat(),
                    referral.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
            logger.debug("saved referral %s (%s @ %s)", referral.id, referral.contact_name, referral.company)
            return True
        except sqlite3.IntegrityError:
            logger.debug("duplicate fingerprint %s — not saved", referral.fingerprint)
            return False

    def update_status(self, referral_id: str, new_status: ReferralStatus) -> bool:
        """
        Update the status of a referral.

        Side-effects:
          REQUEST_SENT → set contacted_at (COALESCE keeps the first timestamp on retries)
          CONNECTED    → set connected_at

        Returns True if found and updated, False if not found.
        """
        now = datetime.now().isoformat()

        if new_status == ReferralStatus.REQUEST_SENT:
            cursor = self._conn.execute(
                """
                UPDATE referrals
                   SET status = ?, contacted_at = COALESCE(contacted_at, ?), updated_at = ?
                 WHERE id = ?
                """,
                (new_status.value, now, now, referral_id),
            )
        elif new_status == ReferralStatus.CONNECTED:
            cursor = self._conn.execute(
                """
                UPDATE referrals
                   SET status = ?, connected_at = ?, updated_at = ?
                 WHERE id = ?
                """,
                (new_status.value, now, now, referral_id),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE referrals SET status = ?, updated_at = ? WHERE id = ?",
                (new_status.value, now, referral_id),
            )

        self._conn.commit()
        return cursor.rowcount > 0

    def update_notes(self, referral_id: str, notes: str) -> bool:
        """Update free-form notes. Returns True if found, False otherwise."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "UPDATE referrals SET notes = ?, updated_at = ? WHERE id = ?",
            (notes, now, referral_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_messages(self, referral_id: str, messages: ReferralMessages) -> bool:
        """Persist all three generated messages. Returns True if found."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """
            UPDATE referrals
               SET linkedin_message = ?, referral_message = ?, followup_message = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                messages.linkedin_request,
                messages.referral_message,
                messages.followup_message,
                now,
                referral_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def exists(self, fingerprint: str) -> bool:
        """True if a referral with this fingerprint already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM referrals WHERE fingerprint = ? LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def get_by_id(self, referral_id: str) -> Referral | None:
        row = self._conn.execute(
            "SELECT * FROM referrals WHERE id = ?", (referral_id,)
        ).fetchone()
        return self._row_to_referral(row) if row else None

    def get_by_fingerprint(self, fingerprint: str) -> Referral | None:
        row = self._conn.execute(
            "SELECT * FROM referrals WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return self._row_to_referral(row) if row else None

    def get_by_company(self, company: str) -> list[Referral]:
        """Case-insensitive company match, most-recently-updated first."""
        rows = self._conn.execute(
            "SELECT * FROM referrals"
            " WHERE lower(company) = lower(?)"
            " ORDER BY updated_at DESC",
            (company,),
        ).fetchall()
        return [self._row_to_referral(r) for r in rows]

    def get_by_status(self, status: ReferralStatus) -> list[Referral]:
        rows = self._conn.execute(
            "SELECT * FROM referrals WHERE status = ? ORDER BY updated_at DESC",
            (status.value,),
        ).fetchall()
        return [self._row_to_referral(r) for r in rows]

    def get_active(self) -> list[Referral]:
        """All referrals that are not REFERRED or DECLINED (terminal)."""
        rows = self._conn.execute(
            "SELECT * FROM referrals"
            " WHERE status NOT IN ('REFERRED', 'DECLINED')"
            " ORDER BY updated_at DESC",
        ).fetchall()
        return [self._row_to_referral(r) for r in rows]

    def get_all(self) -> list[Referral]:
        rows = self._conn.execute(
            "SELECT * FROM referrals ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_referral(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_referral(row: sqlite3.Row) -> Referral:
        return Referral(
            id=row["id"],
            contact_name=row["contact_name"],
            contact_title=row["contact_title"] or "",
            company=row["company"],
            platform=row["platform"] or "LinkedIn",
            job_title=row["job_title"] or "",
            job_url=row["job_url"] or "",
            contact_url=row["contact_url"] or "",
            status=ReferralStatus(row["status"]),
            notes=row["notes"] or "",
            linkedin_message=row["linkedin_message"] or "",
            referral_message=row["referral_message"] or "",
            followup_message=row["followup_message"] or "",
            contacted_at=datetime.fromisoformat(row["contacted_at"]) if row["contacted_at"] else None,
            connected_at=datetime.fromisoformat(row["connected_at"]) if row["connected_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
