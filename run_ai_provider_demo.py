"""
Demo: AI Provider Abstraction (Sprint 20).

What this shows
---------------
1. Which provider is active (from .env / environment variables).
2. One resume tailoring request via the provider.
3. One cover letter via the provider.
4. Graceful fallback when the provider is unavailable.
5. How to switch providers.

Switching providers
-------------------
Add ONE line to .env — no code changes needed anywhere:

    # Use Gemini (default):
    AI_PROVIDER=gemini
    GEMINI_API_KEY=AIza...

    # Use OpenRouter:
    AI_PROVIDER=openrouter
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=openai/gpt-4o-mini   # optional

    # Use Groq:
    AI_PROVIDER=groq
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama-3.3-70b-versatile    # optional

Run
---
    python run_ai_provider_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai.factory import create_provider
from ai.provider import AIProvider, AIProviderError
from config import settings

BANNER = "=" * 60
SAMPLE_RESUME = """
Bhavya L  |  bhavya@example.com  |  github.com/bhavyal

PROFESSIONAL SUMMARY
Backend Software Engineer with 3 years of experience building
scalable microservices using Java, Spring Boot, and REST APIs.

TECHNICAL SKILLS
- Languages: Java, Python, SQL
- Frameworks: Spring Boot, Hibernate, REST APIs
- Databases: MySQL, PostgreSQL
- Tools: Git, Maven, Docker, IntelliJ IDEA
- Practices: Microservices, TDD, CI/CD

EXPERIENCE
Zoho Corporation — Member Technical Staff (2021–2024)
- Reduced API latency by 40% through connection-pool tuning.
- Built payment reconciliation service processing 500K txns/day.
- Led migration of 3 legacy SOAP services to REST APIs.

EDUCATION
Vellore Institute of Technology — B.Tech Computer Science, 2021
CGPA: 9.2 / 10
""".strip()

SAMPLE_JOB_TITLE = "Software Engineer II — Payments Platform"
SAMPLE_JOB_COMPANY = "Stripe"
SAMPLE_JOB_DESC = """
We're looking for a backend engineer to join the Payments Platform team.
You will design and build APIs handling billions of dollars in transactions.
Strong Java or Go experience required. Experience with distributed systems
and high-availability architectures is a plus.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print()
    print(BANNER)
    print(f"  {title}")
    print(BANNER)


def _show_provider_info(provider: AIProvider | None) -> None:
    _section("Active AI Provider")
    if provider is None:
        print()
        print("  Provider  : None — no API key configured")
        print("  Mode      : template-only fallback will be used")
        print()
        print("  To enable AI, set one of the following in .env:")
        print("    GEMINI_API_KEY=AIza...")
        print("    OPENROUTER_API_KEY=sk-or-...")
        print("    GROQ_API_KEY=gsk_...")
    else:
        print()
        print(f"  Provider  : {provider.name}")
        print(f"  Setting   : AI_PROVIDER={settings.ai_provider}")
        if provider.name == "gemini":
            print(f"  Model     : {settings.gemini_model}")
        elif provider.name == "openrouter":
            print(f"  Model     : {settings.openrouter_model}")
        elif provider.name == "groq":
            print(f"  Model     : {settings.groq_model}")


def _demo_resume_tailoring(provider: AIProvider | None) -> None:
    _section("Resume Tailoring Request")
    print()
    print(f"  Job   : {SAMPLE_JOB_TITLE}")
    print(f"  At    : {SAMPLE_JOB_COMPANY}")
    print()

    from scrapers.models import Job
    from config.constants import ATSType

    job = Job(
        company=SAMPLE_JOB_COMPANY,
        title=SAMPLE_JOB_TITLE,
        job_url="https://stripe.com/jobs/demo",
        description=SAMPLE_JOB_DESC,
        source_platform=ATSType.GREENHOUSE,
    )

    if provider is None:
        print("  AI provider not configured — skipping tailoring.")
        print("  (Tailoring requires an AI provider; no template fallback exists.)")
        return

    from agents.resume_tailoring_agent import ResumeTailoringAgent

    output_dir = Path("/private/tmp/claude-501/-Users-tinafiolina-AI-Job-Hunter/"
                      "ebe0eb8f-7289-4c69-9804-63cd48345359/scratchpad/demo_tailor")
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = ResumeTailoringAgent(
        provider=provider,
        resume_text=SAMPLE_RESUME,
        output_dir=output_dir,
    )

    print("  Calling provider.complete() via ResumeTailoringAgent…")
    result = agent.tailor(job)

    if result is None:
        print("  Tailoring failed (provider returned an unusable response).")
        print("  This is expected if the API key is rate-limited or inactive.")
    else:
        print(f"  ATS estimate : {result.data.ats_score_estimate}%")
        print(f"  Keywords     : {', '.join(result.data.keywords_incorporated[:5])}")
        print(f"  Saved to     : {result.text_path}")


def _demo_cover_letter(provider: AIProvider | None) -> None:
    _section("Cover Letter Generation")
    print()
    print(f"  Job   : {SAMPLE_JOB_TITLE}")
    print(f"  At    : {SAMPLE_JOB_COMPANY}")
    print()

    from scrapers.models import Job
    from config.constants import ATSType

    job = Job(
        company=SAMPLE_JOB_COMPANY,
        title=SAMPLE_JOB_TITLE,
        job_url="https://stripe.com/jobs/demo",
        description=SAMPLE_JOB_DESC,
        source_platform=ATSType.GREENHOUSE,
    )

    from agents.cover_letter_agent import CoverLetterAgent

    output_dir = Path("/private/tmp/claude-501/-Users-tinafiolina-AI-Job-Hunter/"
                      "ebe0eb8f-7289-4c69-9804-63cd48345359/scratchpad/demo_cover")
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = CoverLetterAgent(
        resume_text=SAMPLE_RESUME,
        provider=provider,
        candidate_name=settings.user_name or "Bhavya L",
        candidate_email=settings.user_email or "bhavya@example.com",
        output_dir=output_dir,
    )

    mode = f"AI ({provider.name})" if provider else "template (no provider)"
    print(f"  Mode  : {mode}")
    print("  Calling CoverLetterAgent.generate()…")

    letter = agent.generate(job, save=False)
    print(f"  Subject    : {letter.subject_line}")
    preview = letter.content[:200].replace("\n", " ")
    print(f"  Preview    : {preview}…")


def _demo_graceful_fallback() -> None:
    _section("Graceful Fallback Demo")
    print()
    print("  Simulating a provider failure (AIProviderError)…")
    print()

    from unittest.mock import MagicMock
    from scrapers.models import Job
    from config.constants import ATSType
    from agents.cover_letter_agent import CoverLetterAgent

    broken_provider = MagicMock()
    broken_provider.name = "mock-broken"
    broken_provider.complete.side_effect = AIProviderError("quota exceeded — 429")

    agent = CoverLetterAgent(
        resume_text=SAMPLE_RESUME,
        provider=broken_provider,
        candidate_name="Bhavya L",
    )
    job = Job(
        company="Stripe",
        title="Backend Engineer",
        job_url="https://stripe.com/jobs/be",
        source_platform=ATSType.GREENHOUSE,
    )

    letter = agent.generate(job, save=False)
    print("  AIProviderError was raised by the mock provider.")
    print("  CoverLetterAgent caught it and used the template fallback.")
    print(f"  Result     : {type(letter).__name__} (template mode)")
    print(f"  Subject    : {letter.subject_line}")


def _show_switching_guide() -> None:
    _section("How to Switch Providers")
    print()
    print("  Edit .env — one variable selects the active provider:")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Provider   │  .env key          │  Default model   │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  Gemini     │  GEMINI_API_KEY     │  gemini-2.0-flash│")
    print("  │  OpenRouter │  OPENROUTER_API_KEY │  gpt-4o-mini     │")
    print("  │  Groq       │  GROQ_API_KEY       │  llama-3.3-70b   │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  To override the model:")
    print("    OPENROUTER_MODEL=anthropic/claude-3-haiku")
    print("    GROQ_MODEL=mixtral-8x7b-32768")
    print()
    print("  No code changes required — all agents use the provider")
    print("  injected by create_provider() from the factory.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print(BANNER)
    print("  AI Provider Abstraction Demo  (Sprint 20)")
    print(BANNER)

    provider = create_provider()

    _show_provider_info(provider)
    _demo_resume_tailoring(provider)
    _demo_cover_letter(provider)
    _demo_graceful_fallback()
    _show_switching_guide()

    print()
    print(BANNER)
    print()


if __name__ == "__main__":
    main()
