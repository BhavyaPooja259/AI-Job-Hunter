"""
JobRepository — SQLite persistence layer for Job objects.

Uses the Python standard-library `sqlite3` module directly rather than
an ORM. The schema is small and stable; an ORM would add complexity with
no benefit at this scale.

Duplicate prevention uses SQLite's INSERT OR IGNORE against the fingerprint
primary key. This is a single atomic operation — no read-before-write race
condition, no extra round trips.

The database file is created automatically on first call to initialize().
The parent directory is also created if it does not exist.
"""

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from config import settings
from config.constants import ATSType
from scrapers.models import Job

logger = logging.getLogger(__name__)

# Derive the file path from the database URL stored in settings.
# Format: sqlite:///./data/jobs.db → data/jobs.db
_DEFAULT_DB_PATH = Path(settings.database_url.replace("sqlite:///", ""))

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    fingerprint     TEXT PRIMARY KEY,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    location        TEXT,
    job_url         TEXT NOT NULL,
    source_platform TEXT NOT NULL,
    posted_date     TEXT,
    employment_type TEXT,
    discovered_at   TEXT NOT NULL,
    description     TEXT,
    requirements    TEXT,
    department      TEXT
)
"""

# Columns added in Sprint 11.5 — appended to existing tables via _migrate_schema().
_NEW_COLUMNS = ("description", "requirements", "department")

_INSERT_SQL = """
INSERT OR IGNORE INTO jobs
    (fingerprint, company, title, location, job_url,
     source_platform, posted_date, employment_type, discovered_at,
     description, requirements, department)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class JobRepository:
    """
    Read/write interface for the jobs table.

    Usage — manual lifecycle:
        repo = JobRepository()
        repo.initialize()
        repo.save(job)
        repo.close()

    Usage — context manager (preferred):
        with JobRepository() as repo:
            repo.save_many(jobs)
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Open the database connection and create the schema if needed.

        Creates the database file and any missing parent directories.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._conn is not None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Opening database: %s", self._db_path)

        self._conn = sqlite3.connect(str(self._db_path))
        # Row objects are accessible by column name, not just index.
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._migrate_schema()
        self._conn.commit()
        logger.info("Database ready")

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __enter__(self) -> "JobRepository":
        self.initialize()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------

    def save(self, job: Job) -> bool:
        """
        Persist a single Job.

        Returns True if the job was inserted, False if it already existed.
        Uses INSERT OR IGNORE so calling this with a duplicate is always safe.
        """
        self._require_open()
        cursor = self._conn.execute(_INSERT_SQL, _job_to_row(job))
        self._conn.commit()
        inserted = cursor.rowcount == 1
        if inserted:
            logger.debug("Saved: %s", job.fingerprint)
        else:
            logger.debug("Duplicate skipped: %s", job.fingerprint)
        return inserted

    def save_many(self, jobs: list[Job]) -> tuple[int, int]:
        """
        Persist a list of Jobs in a single transaction.

        Returns (saved_count, skipped_count).

        executemany with INSERT OR IGNORE is significantly faster than
        calling save() in a loop because it avoids per-row commit overhead.
        The tradeoff is that executemany does not expose per-row rowcount,
        so saved vs skipped is derived from the before/after table count.
        """
        self._require_open()
        if not jobs:
            return 0, 0

        before = self._count()
        self._conn.executemany(_INSERT_SQL, [_job_to_row(j) for j in jobs])
        self._conn.commit()
        after = self._count()

        saved = after - before
        skipped = len(jobs) - saved
        logger.info(
            "save_many: %d saved, %d duplicate(s) skipped (batch size: %d)",
            saved, skipped, len(jobs),
        )
        return saved, skipped

    # -------------------------------------------------------------------------
    # Read operations
    # -------------------------------------------------------------------------

    def exists(self, fingerprint: str) -> bool:
        """Return True if a job with this fingerprint is already stored."""
        self._require_open()
        row = self._conn.execute(
            "SELECT 1 FROM jobs WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return row is not None

    def get_all(self) -> list[Job]:
        """Return every stored job, ordered by discovery time descending."""
        self._require_open()
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY discovered_at DESC"
        ).fetchall()
        return [_row_to_job(row) for row in rows]

    def count(self) -> int:
        """Return the total number of jobs in the database."""
        self._require_open()
        return self._count()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _migrate_schema(self) -> None:
        """
        Add columns introduced after the initial schema was created.

        Uses PRAGMA table_info to check which columns already exist, then
        ALTER TABLE for any that are missing.  Safe to run on both fresh
        databases (all columns present from _CREATE_TABLE_SQL) and old ones
        (only the original nine columns exist).  Idempotent.
        """
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for col in _NEW_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
                logger.info("Migration: added column '%s' to jobs table", col)

    def _require_open(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "JobRepository is not initialized. Call initialize() or use it as a context manager."
            )

    def _count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


# ---------------------------------------------------------------------------
# Row conversion helpers (module-level, not methods — no reason to bind to self)
# ---------------------------------------------------------------------------

def _job_to_row(job: Job) -> tuple:
    return (
        job.fingerprint,
        job.company,
        job.title,
        job.location,
        job.job_url,
        job.source_platform.value,
        job.posted_date.isoformat() if job.posted_date else None,
        job.employment_type,
        job.discovered_at.isoformat(),
        job.description,
        job.requirements,
        job.department,
    )


def _row_to_job(row: sqlite3.Row) -> Job:
    keys = row.keys()
    return Job(
        company=row["company"],
        title=row["title"],
        job_url=row["job_url"],
        location=row["location"],
        source_platform=ATSType(row["source_platform"]),
        posted_date=date.fromisoformat(row["posted_date"]) if row["posted_date"] else None,
        employment_type=row["employment_type"],
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
        description=row["description"] if "description" in keys else None,
        requirements=row["requirements"] if "requirements" in keys else None,
        department=row["department"] if "department" in keys else None,
    )
