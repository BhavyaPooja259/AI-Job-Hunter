"""
Dashboard route handlers.

Pages
-----
GET /               Home — headline stats, top job matches, recent applications
GET /jobs           Full sorted job list with match scores
GET /applications   All tracked applications with status badges
GET /stats          Statistics dashboard — funnel breakdown, summary cards
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from agents.application_agent import ApplicationAgent
from agents.notification_agent import NotificationAgent
from application.application_repository import ApplicationRepository
from dashboard.dependencies import get_application_repo, get_job_repo
from database.job_repository import JobRepository
from services.notification_service import NotificationService

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter()

# Stateless helper — no channels needed, we only call build_digest()
_notification_agent = NotificationAgent(service=NotificationService())


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    job_repo: JobRepository = Depends(get_job_repo),
    app_repo: ApplicationRepository = Depends(get_application_repo),
) -> HTMLResponse:
    jobs = job_repo.get_all()
    digest = _notification_agent.build_digest(jobs)

    app_agent = ApplicationAgent(repo=app_repo)
    app_stats = app_agent.stats()
    all_apps = app_repo.get_all()
    recent_apps = sorted(all_apps, key=lambda a: a.updated_at, reverse=True)[:5]

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "active_page": "home",
            "total_jobs": len(jobs),
            "top_jobs": digest.top_jobs[:5],
            "recent_applications": recent_apps,
            "app_stats": app_stats,
        },
    )


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------


@router.get("/jobs", response_class=HTMLResponse)
def jobs_page(
    request: Request,
    job_repo: JobRepository = Depends(get_job_repo),
) -> HTMLResponse:
    jobs = job_repo.get_all()
    digest = _notification_agent.build_digest(jobs)

    return templates.TemplateResponse(
        request,
        "jobs.html",
        {
            "active_page": "jobs",
            "jobs": digest.all_jobs,
            "total": len(jobs),
            "score_threshold": digest.score_threshold,
        },
    )


# ---------------------------------------------------------------------------
# GET /applications
# ---------------------------------------------------------------------------


@router.get("/applications", response_class=HTMLResponse)
def applications_page(
    request: Request,
    app_repo: ApplicationRepository = Depends(get_application_repo),
) -> HTMLResponse:
    app_agent = ApplicationAgent(repo=app_repo)
    app_stats = app_agent.stats()
    applications = sorted(
        app_repo.get_all(), key=lambda a: a.updated_at, reverse=True
    )

    return templates.TemplateResponse(
        request,
        "applications.html",
        {
            "active_page": "applications",
            "applications": applications,
            "app_stats": app_stats,
        },
    )


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------


@router.get("/stats", response_class=HTMLResponse)
def stats_page(
    request: Request,
    job_repo: JobRepository = Depends(get_job_repo),
    app_repo: ApplicationRepository = Depends(get_application_repo),
) -> HTMLResponse:
    total_jobs = job_repo.count()
    app_agent = ApplicationAgent(repo=app_repo)
    app_stats = app_agent.stats()

    # Status breakdown sorted by lifecycle order
    from application.application_status import ApplicationStatus

    status_order = list(ApplicationStatus)
    breakdown = [
        (s.value, app_stats.by_status.get(s.value, 0))
        for s in status_order
        if app_stats.by_status.get(s.value, 0) > 0
    ]

    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "active_page": "stats",
            "total_jobs": total_jobs,
            "app_stats": app_stats,
            "breakdown": breakdown,
        },
    )
