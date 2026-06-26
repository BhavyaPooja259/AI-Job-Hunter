"""
FastAPI dependency providers for database repositories.

Both providers are generator functions — FastAPI opens the repository
before the route handler runs and closes it after the response is sent.
Tests override these via app.dependency_overrides so no real SQLite
database is touched during testing.
"""

from __future__ import annotations

from typing import Generator

from application.application_repository import ApplicationRepository
from database.job_repository import JobRepository


def get_job_repo() -> Generator[JobRepository, None, None]:
    with JobRepository() as repo:
        yield repo


def get_application_repo() -> Generator[ApplicationRepository, None, None]:
    with ApplicationRepository() as repo:
        yield repo
