"""
ApplicationRepository — SQLite-backed persistence for Application records.

Architecture
------------
Thin data-access layer only.  No business rules live here — duplicate
prevention and lifecycle validation are the ApplicationAgent's concern.

The repository supports two usage patterns:

    # Context manager (production):
    with ApplicationRepository() as repo:
        repo.save(app)

    # Explicit initialize/close (tests):
    repo = ApplicationRepository(db_path=str(tmp_path / "test.db"))
    repo.initialize()
    repo.save(app)

job_fingerprint has a UNIQUE constraint, so save() catches IntegrityError
and returns False on duplicate inserts without raising.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from application.application import Application
from application.application_status import ApplicationStatus

logger = logging.getLogger(__name__)


class ApplicationRepository:
    """SQLite-backed persistence for Application records."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS applications (
            id               TEXT PRIMARY KEY,
            job_fingerprint  TEXT UNIQUE NOT NULL,
            company          TEXT NOT NULL,
            title            TEXT NOT NULL,
            job_url          TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'SAVED',
            applied_at       TEXT,
            notes            TEXT NOT NULL DEFAULT '',
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        )
    """

    def __init__(self, db_path: str = "data/applications.db") -> None:
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
        logger.debug("ApplicationRepository initialized at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ApplicationRepository":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def save(self, app: Application) -> bool:
        """
        Insert an application.

        Returns True if saved, False if job_fingerprint already exists.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO applications
                    (id, job_fingerprint, company, title, job_url,
                     status, applied_at, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app.id,
                    app.job_fingerprint,
                    app.company,
                    app.title,
                    app.job_url,
                    app.status.value,
                    app.applied_at.isoformat() if app.applied_at else None,
                    app.notes,
                    app.created_at.isoformat(),
                    app.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
            logger.debug("saved application %s (%s @ %s)", app.id, app.title, app.company)
            return True
        except sqlite3.IntegrityError:
            logger.debug("duplicate fingerprint %s — not saved", app.job_fingerprint)
            return False

    def update_status(self, app_id: str, new_status: ApplicationStatus) -> bool:
        """
        Update the status of an application.

        Sets applied_at automatically when transitioning to APPLIED.
        Returns True if the record was found and updated, False if not found.
        """
        now = datetime.now().isoformat()
        if new_status == ApplicationStatus.APPLIED:
            cursor = self._conn.execute(
                "UPDATE applications"
                " SET status = ?, applied_at = ?, updated_at = ?"
                " WHERE id = ?",
                (new_status.value, now, now, app_id),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
                (new_status.value, now, app_id),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_notes(self, app_id: str, notes: str) -> bool:
        """Update free-form notes. Returns True if found, False if not found."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "UPDATE applications SET notes = ?, updated_at = ? WHERE id = ?",
            (notes, now, app_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def exists(self, job_fingerprint: str) -> bool:
        """True if an application with this job_fingerprint already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM applications WHERE job_fingerprint = ? LIMIT 1",
            (job_fingerprint,),
        ).fetchone()
        return row is not None

    def get_by_id(self, app_id: str) -> Application | None:
        row = self._conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,)
        ).fetchone()
        return self._row_to_application(row) if row else None

    def get_by_fingerprint(self, fingerprint: str) -> Application | None:
        row = self._conn.execute(
            "SELECT * FROM applications WHERE job_fingerprint = ?", (fingerprint,)
        ).fetchone()
        return self._row_to_application(row) if row else None

    def get_by_company(self, company: str) -> list[Application]:
        """Case-insensitive company name match, most-recently-updated first."""
        rows = self._conn.execute(
            "SELECT * FROM applications"
            " WHERE lower(company) = lower(?)"
            " ORDER BY updated_at DESC",
            (company,),
        ).fetchall()
        return [self._row_to_application(r) for r in rows]

    def get_by_status(self, status: ApplicationStatus) -> list[Application]:
        rows = self._conn.execute(
            "SELECT * FROM applications WHERE status = ? ORDER BY updated_at DESC",
            (status.value,),
        ).fetchall()
        return [self._row_to_application(r) for r in rows]

    def get_active(self) -> list[Application]:
        """All applications that are not REJECTED or WITHDRAWN."""
        rows = self._conn.execute(
            "SELECT * FROM applications"
            " WHERE status NOT IN ('REJECTED', 'WITHDRAWN')"
            " ORDER BY updated_at DESC",
        ).fetchall()
        return [self._row_to_application(r) for r in rows]

    def get_all(self) -> list[Application]:
        rows = self._conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_application(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_application(row: sqlite3.Row) -> Application:
        return Application(
            id=row["id"],
            job_fingerprint=row["job_fingerprint"],
            company=row["company"],
            title=row["title"],
            job_url=row["job_url"],
            status=ApplicationStatus(row["status"]),
            applied_at=datetime.fromisoformat(row["applied_at"]) if row["applied_at"] else None,
            notes=row["notes"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
