"""
Demo: generate a cover letter for the highest-ranked Adobe job.

Pipeline
--------
  1. Open the jobs database and load all Adobe jobs.
  2. Score them with the rule-based JobMatcher.
  3. Pick the highest-scoring Adobe job that has a description.
  4. Extract resume text from resume/bhavya-resume.pdf via pdfplumber.
  5. Call CoverLetterAgent:
       • AI mode   — if GEMINI_API_KEY is set, Gemini generates personalised
                     paragraphs for opening, body, and closing.
       • Template  — if no API key is available, a professional letter is
                     assembled from the built-in template (no API call needed).
  6. Save cover_{company}_{timestamp}.txt and .md under storage/cover_letters/.
  7. Print the selected job, output paths, and the first 20 lines of the letter.

Requirements
------------
  • Run run_tailor_demo.py at least once first so Adobe jobs are in the DB.
  • GEMINI_API_KEY in .env is optional — the demo runs without it in template
    mode, which is still a usable (if less personalised) cover letter.
  • pdfplumber must be installed (already listed in requirements.txt).

Run
---
  python run_cover_letter_demo.py
"""

import logging
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
for ns in ("agents", "cover_letter", "database", "matching"):
    logging.getLogger(ns).setLevel(logging.INFO)

from config import settings
from database.job_repository import JobRepository
from matching.matcher import JobMatcher
from matching.profile import DEFAULT_PROFILE
from agents.cover_letter_agent import CoverLetterAgent

_RESUME_PATH = Path(__file__).parent / "resume" / "bhavya-resume.pdf"
BANNER = "=" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — load Adobe jobs from the database
# ─────────────────────────────────────────────────────────────────────────────

def load_adobe_jobs() -> list:
    """
    Return all Adobe jobs stored in the database.

    Exits with a helpful message if the database has no Adobe jobs.
    Run run_tailor_demo.py first to populate the database.
    """
    with JobRepository() as repo:
        all_jobs = repo.get_all()

    adobe_jobs = [j for j in all_jobs if j.company == "Adobe"]

    if not adobe_jobs:
        print()
        print("ERROR: No Adobe jobs found in the database.")
        print("Run the following command first to populate it:")
        print("  python run_tailor_demo.py")
        sys.exit(1)

    desc_count = sum(1 for j in adobe_jobs if j.description)
    print(f"  Loaded {len(adobe_jobs)} Adobe jobs "
          f"({desc_count} with descriptions).")
    return adobe_jobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — pick the highest-ranked job that has a description
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_job(jobs: list):
    """
    Score all jobs with the rule-based matcher and return (job, rule_result).

    Preference: highest-scoring job WITH a description so the AI has real
    content to personalise the letter with.  Falls back to the top-scored
    job overall when none have descriptions.
    """
    matcher = JobMatcher(DEFAULT_PROFILE)
    ranked = matcher.rank(jobs)   # sorted highest score first

    for job, result in ranked:
        if job.description:
            return job, result

    return ranked[0]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — extract resume text from the PDF
# ─────────────────────────────────────────────────────────────────────────────

def load_resume_text(path: Path) -> str:
    """Extract plain text from the candidate's resume PDF using pdfplumber."""
    if not path.exists():
        print()
        print(f"ERROR: Resume PDF not found at {path}")
        print("Place the resume at resume/bhavya-resume.pdf and try again.")
        sys.exit(1)

    try:
        import pdfplumber
    except ImportError:
        print()
        print("ERROR: pdfplumber is not installed.")
        print("Install it with:  pip install pdfplumber")
        sys.exit(1)

    with pdfplumber.open(str(path)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]

    text = "\n".join(pages).strip()
    if not text:
        print()
        print(f"ERROR: Could not extract text from {path}")
        print("Ensure the PDF contains selectable (not scanned) text.")
        sys.exit(1)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(BANNER)
    print("  Cover Letter Demo — Adobe × Bhavya L")
    print(BANNER)

    # ── step 1: jobs ─────────────────────────────────────────────────────────
    print()
    print("[ 1/4 ]  Loading Adobe jobs from database")
    print("-" * 40)
    adobe_jobs = load_adobe_jobs()

    # ── step 2: pick ─────────────────────────────────────────────────────────
    print()
    print("[ 2/4 ]  Selecting the best job")
    print("-" * 40)
    job, rule_result = pick_best_job(adobe_jobs)

    print(f"  Title     : {job.title}")
    print(f"  Company   : {job.company}")
    print(f"  Location  : {job.location or 'Not specified'}")
    print(f"  Score     : {rule_result.score}/100")
    print(f"  Matched   : {', '.join(rule_result.matched_skills) or 'none'}")
    print(f"  Has desc  : {'yes (' + str(len(job.description)) + ' chars)' if job.description else 'no'}")
    print(f"  URL       : {job.job_url}")

    # ── step 3: resume ───────────────────────────────────────────────────────
    print()
    print("[ 3/4 ]  Loading resume PDF")
    print("-" * 40)
    resume_text = load_resume_text(_RESUME_PATH)
    print(f"  Extracted {len(resume_text):,} characters from {_RESUME_PATH.name}")

    # ── step 4: cover letter ─────────────────────────────────────────────────
    print()
    print("[ 4/4 ]  Generating cover letter")
    print("-" * 40)

    client = None
    if settings.gemini_api_key:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        print("  Mode      : AI (Gemini) — personalised letter")
        print("  Sending resume + job description to Gemini … (may take 10–20 s)")
    else:
        print("  Mode      : Template (GEMINI_API_KEY not set)")
        print("  Assembling cover letter from built-in template …")
        print()
        print("  Tip: add GEMINI_API_KEY to your .env for a fully personalised letter.")

    agent = CoverLetterAgent(
        resume_text=resume_text,
        client=client,
        candidate_name=settings.user_name or "Bhavya L",
        candidate_email=settings.user_email or "",
    )

    letter = agent.generate(job, save=True)

    # ── results ──────────────────────────────────────────────────────────────
    print()
    print(BANNER)
    print("  RESULT")
    print(BANNER)
    print(f"  Job title  : {letter.title}")
    print(f"  Company    : {letter.company}")
    print(f"  Subject    : {letter.subject_line}")
    print()
    print(f"  Saved .txt : {letter.txt_path}")
    print(f"  Saved .md  : {letter.md_path}")

    print()
    print(BANNER)
    print("  COVER LETTER (first 20 lines)")
    print(BANNER)
    lines = letter.content.splitlines()
    for line in lines[:20]:
        print(f"  {line}")
    if len(lines) > 20:
        print(f"  … ({len(lines) - 20} more lines — see {letter.txt_path.name})")

    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
