"""
SchedulerConfig — all knobs for a single workflow run.

Design decisions
----------------
* SchedulerConfig is a plain dataclass with no methods or I/O so it is
  trivially serialisable (JSON / YAML) and safe to pass across boundaries.
* Optional steps (tailoring, cover letters, referral reminders) are disabled
  by default — they require extra setup (resume text, API key, referral DB).
* only_new_jobs=True (default) limits optional steps to jobs that were not
  in the database before this run's Scout step.  This is the primary
  idempotency mechanism for file-generating steps.  Set it to False in demos
  or ad-hoc runs to process existing top-ranked jobs regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    """
    Configuration for one full scheduler workflow run.

    Parameters
    ----------
    top_n
        Maximum number of jobs processed by optional steps (TAILOR,
        COVER_LETTER).  Applied after min_score filtering.
    min_score
        Minimum rule-based match score (0–100) for a job to enter the
        optional-step pipeline.  Mirrors NotificationAgent's default
        threshold of 60.
    enable_tailoring
        When True, ResumeTailoringAgent runs for eligible jobs.
        Requires a Gemini client to be passed to SchedulerAgent.
    enable_cover_letters
        When True, CoverLetterAgent runs for eligible jobs.
        Works in template-only mode if no Gemini client is available.
    enable_referral_reminders
        When True, scans the referral database for contacts in
        REQUEST_SENT status that may need a follow-up.
        Requires a ReferralRepository to be passed to SchedulerAgent.
    resume_text
        Plain-text resume used by the tailoring and cover-letter steps.
        Pass the full resume — do not truncate.
    candidate_name
        Name embedded in cover letters and referral messages.
    candidate_email
        Email embedded in cover letters.
    only_new_jobs
        When True (default), TAILOR and COVER_LETTER only run for jobs
        that were not in the database before this run's Scout step.
        Set to False to re-process all top-ranked jobs on every run.
    """

    top_n: int = 5
    min_score: int = 60
    enable_tailoring: bool = False
    enable_cover_letters: bool = False
    enable_referral_reminders: bool = False
    resume_text: str = ""
    candidate_name: str = ""
    candidate_email: str = ""
    only_new_jobs: bool = True
