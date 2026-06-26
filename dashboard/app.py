"""
FastAPI application factory.

Usage (production):
    uvicorn dashboard.app:app --reload --port 8000

Usage (tests):
    from dashboard.app import create_app
    app = create_app()
    app.dependency_overrides[get_job_repo] = lambda: mock_repo
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app() -> FastAPI:
    """
    Build and return a configured FastAPI instance.

    Called once at module load for production and once per test session
    for testing (with dependency overrides applied before the TestClient
    is constructed).
    """
    fastapi_app = FastAPI(
        title="AI Job Hunter",
        description="Personal job hunting dashboard",
        version="1.0.0",
    )

    fastapi_app.mount(
        "/static",
        StaticFiles(directory=str(_STATIC_DIR)),
        name="static",
    )

    from dashboard.routes import router
    fastapi_app.include_router(router)

    return fastapi_app


# Module-level instance used by uvicorn:  uvicorn dashboard.app:app
app = create_app()
