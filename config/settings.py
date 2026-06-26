"""
Central configuration module.

All runtime values that change between environments or users live here.
Nothing else in the codebase should read environment variables directly —
every module imports `settings` from this module instead.

Pydantic BaseSettings is used because it:
  - Validates types at startup (fail fast, not at runtime)
  - Reads from .env automatically without extra boilerplate
  - Makes every setting self-documenting via Field descriptions
  - Allows easy overriding in tests via environment variables
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Silently ignore extra keys in .env so adding new vars never breaks old code.
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # AI Provider
    # -------------------------------------------------------------------------

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key from aistudio.google.com",
    )
    ai_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model ID used for all AI tasks",
    )
    ai_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens per AI response",
    )

    # -------------------------------------------------------------------------
    # User Profile
    # Used to personalise AI-generated resumes, cover letters, and messages.
    # -------------------------------------------------------------------------

    user_name: str = Field(default="", description="Your full name")
    user_email: str = Field(default="", description="Your contact email")
    user_years_experience: int = Field(default=3, description="Years of engineering experience")
    user_target_level: str = Field(default="SDE2", description="Target seniority level")

    # -------------------------------------------------------------------------
    # Database
    # SQLite keeps this local-first with zero infrastructure overhead.
    # Switch to a Postgres URL here when/if a hosted version is ever needed.
    # -------------------------------------------------------------------------

    database_url: str = Field(
        default="sqlite:///./data/jobs.db",
        description="SQLAlchemy-compatible database URL",
    )

    # -------------------------------------------------------------------------
    # Scraping / Browser
    # -------------------------------------------------------------------------

    scrape_delay_seconds: float = Field(
        default=2.0,
        description="Minimum pause between requests to the same domain",
    )
    request_timeout_seconds: int = Field(
        default=30,
        description="HTTP and browser page-load timeout",
    )
    max_retries: int = Field(
        default=3,
        description="Retry attempts before marking a scrape as failed",
    )
    headless_browser: bool = Field(
        default=False,
        description="Run browser in headless mode. Default is False (visible) for development; set True in production.",
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        description="User-agent string sent with all HTTP requests",
    )

    # -------------------------------------------------------------------------
    # Storage Paths
    # All paths are relative to the project root so the project stays portable.
    # -------------------------------------------------------------------------

    output_dir: Path = Field(
        default=Path("storage/output"),
        description="Root directory for all generated output files",
    )
    resumes_dir: Path = Field(
        default=Path("storage/resumes"),
        description="Tailored resume output directory",
    )
    cover_letters_dir: Path = Field(
        default=Path("storage/cover_letters"),
        description="Generated cover letter output directory",
    )
    referrals_dir: Path = Field(
        default=Path("storage/referrals"),
        description="Referral message output directory",
    )

    # -------------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------------

    notifications_enabled: bool = Field(
        default=True,
        description="Master switch for all notifications",
    )
    daily_digest_hour: int = Field(
        default=8,
        description="Hour (24h, local time) to send the daily job digest",
    )

    # -------------------------------------------------------------------------
    # Feature Flags
    # Each major module can be toggled without changing code.
    # -------------------------------------------------------------------------

    ai_ranking_enabled: bool = Field(default=True)
    resume_tailoring_enabled: bool = Field(default=True)
    cover_letter_enabled: bool = Field(default=True)
    referral_assistant_enabled: bool = Field(default=True)
    interview_prep_enabled: bool = Field(default=True)


# Single shared instance imported by all other modules.
# Constructed once at startup; Pydantic validates all fields immediately.
settings = Settings()
