"""
Tests for Sprint 16 — Web Dashboard.

Strategy
--------
All database dependencies are overridden via FastAPI's dependency injection
system.  Route handlers receive mock repos that return pre-built in-memory
objects so no SQLite file is touched.  Tests verify:
  - HTTP 200 response for every page
  - Content-Type is text/html
  - Key data (company names, status values, counts) appears in rendered HTML
  - Empty-database cases render without errors
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from application.application import Application
from application.application_status import ApplicationStatus
from config.constants import ATSType
from dashboard.app import create_app
from dashboard.dependencies import get_application_repo, get_job_repo
from scrapers.models import Job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job(company: str = "Stripe", title: str = "Software Engineer II", n: int = 1) -> Job:
    return Job(
        company=company,
        title=title,
        job_url=f"https://{company.lower().replace(' ', '')}.com/jobs/{n}",
        source_platform=ATSType.GREENHOUSE,
    )


def _app(
    company: str = "Stripe",
    title: str = "Software Engineer II",
    status: ApplicationStatus = ApplicationStatus.APPLIED,
    n: int = 1,
) -> Application:
    return Application(
        company=company,
        title=title,
        job_url=f"https://{company.lower()}.com/jobs/{n}",
        job_fingerprint=f"fp_{company.lower()}_{n}",
        status=status,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MOCK_JOBS = [
    _job("Stripe", "Software Engineer II", 1),
    _job("Uber", "Backend Engineer", 2),
]

MOCK_APPS = [
    _app("Stripe", "Software Engineer II", ApplicationStatus.APPLIED, 1),
    _app("Uber", "Backend Engineer", ApplicationStatus.INTERVIEW, 2),
]


def _build_client(jobs: list[Job], apps: list[Application]) -> TestClient:
    """Construct a TestClient with the given in-memory data injected."""
    app = create_app()

    mock_job_repo = MagicMock()
    mock_job_repo.get_all.return_value = jobs
    mock_job_repo.count.return_value = len(jobs)

    mock_app_repo = MagicMock()
    mock_app_repo.get_all.return_value = apps
    mock_app_repo.get_active.return_value = [
        a for a in apps
        if a.status not in {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}
    ]
    mock_app_repo.exists.return_value = False
    mock_app_repo.get_by_fingerprint.return_value = None
    mock_app_repo.get_by_id.return_value = None

    app.dependency_overrides[get_job_repo] = lambda: mock_job_repo
    app.dependency_overrides[get_application_repo] = lambda: mock_app_repo

    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _build_client(MOCK_JOBS, MOCK_APPS)


@pytest.fixture
def empty_client() -> TestClient:
    return _build_client([], [])


# ---------------------------------------------------------------------------
# GET /  — Home
# ---------------------------------------------------------------------------


class TestHomePage:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_returns_html(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_shows_brand_name(self, client):
        assert "AI Job Hunter" in client.get("/").text

    def test_shows_total_jobs_count(self, client):
        # 2 mock jobs → "2" must appear on the page
        assert "2" in client.get("/").text

    def test_shows_nav_links(self, client):
        text = client.get("/").text
        assert 'href="/jobs"' in text
        assert 'href="/applications"' in text
        assert 'href="/stats"' in text

    def test_shows_top_jobs_section(self, client):
        assert "Top Matched Jobs" in client.get("/").text

    def test_shows_recent_applications_section(self, client):
        assert "Recent Applications" in client.get("/").text

    def test_shows_application_company(self, client):
        text = client.get("/").text
        assert "Stripe" in text

    def test_empty_db_returns_200(self, empty_client):
        assert empty_client.get("/").status_code == 200

    def test_empty_db_shows_empty_state(self, empty_client):
        text = empty_client.get("/").text
        assert "Scout Agent" in text or "No jobs" in text


# ---------------------------------------------------------------------------
# GET /jobs  — Jobs list
# ---------------------------------------------------------------------------


class TestJobsPage:
    def test_returns_200(self, client):
        assert client.get("/jobs").status_code == 200

    def test_returns_html(self, client):
        assert "text/html" in client.get("/jobs").headers["content-type"]

    def test_shows_job_companies(self, client):
        text = client.get("/jobs").text
        assert "Stripe" in text
        assert "Uber" in text

    def test_shows_job_titles(self, client):
        text = client.get("/jobs").text
        assert "Software Engineer" in text
        assert "Backend Engineer" in text

    def test_shows_total_count(self, client):
        assert "2" in client.get("/jobs").text

    def test_empty_db_returns_200(self, empty_client):
        assert empty_client.get("/jobs").status_code == 200

    def test_empty_db_shows_empty_state(self, empty_client):
        assert "Scout Agent" in empty_client.get("/jobs").text or \
               "No jobs" in empty_client.get("/jobs").text


# ---------------------------------------------------------------------------
# GET /applications  — Applications list
# ---------------------------------------------------------------------------


class TestApplicationsPage:
    def test_returns_200(self, client):
        assert client.get("/applications").status_code == 200

    def test_returns_html(self, client):
        assert "text/html" in client.get("/applications").headers["content-type"]

    def test_shows_application_companies(self, client):
        text = client.get("/applications").text
        assert "Stripe" in text
        assert "Uber" in text

    def test_shows_applied_status(self, client):
        assert "APPLIED" in client.get("/applications").text

    def test_shows_interview_status(self, client):
        assert "INTERVIEW" in client.get("/applications").text

    def test_shows_stats_cards(self, client):
        text = client.get("/applications").text
        assert "Total" in text
        assert "Active" in text

    def test_empty_db_returns_200(self, empty_client):
        assert empty_client.get("/applications").status_code == 200

    def test_empty_db_shows_empty_state(self, empty_client):
        text = empty_client.get("/applications").text
        assert "No applications" in text or "ApplicationAgent" in text


# ---------------------------------------------------------------------------
# GET /stats  — Statistics
# ---------------------------------------------------------------------------


class TestStatsPage:
    def test_returns_200(self, client):
        assert client.get("/stats").status_code == 200

    def test_returns_html(self, client):
        assert "text/html" in client.get("/stats").headers["content-type"]

    def test_shows_jobs_discovered(self, client):
        text = client.get("/stats").text
        assert "Jobs Discovered" in text

    def test_shows_job_count(self, client):
        assert "2" in client.get("/stats").text

    def test_shows_application_funnel(self, client):
        assert "Application Funnel" in client.get("/stats").text

    def test_shows_status_badges(self, client):
        text = client.get("/stats").text
        assert "APPLIED" in text or "INTERVIEW" in text

    def test_empty_db_returns_200(self, empty_client):
        assert empty_client.get("/stats").status_code == 200

    def test_empty_db_shows_empty_state(self, empty_client):
        text = empty_client.get("/stats").text
        assert "No applications" in text or "Stats will appear" in text
