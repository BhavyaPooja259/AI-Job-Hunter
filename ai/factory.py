"""
create_provider() — build an AIProvider from the active settings.

Priority
--------
1. The AI_PROVIDER setting selects the provider by name.
2. If the required API key for that provider is missing, returns None.
3. If AI_PROVIDER is empty, falls back to whichever key is available
   (gemini → openrouter → groq).

Usage
-----
    from ai.factory import create_provider

    provider = create_provider()
    if provider is None:
        # No API key configured — use template fallback
        ...
    else:
        text = provider.complete("Hello", max_tokens=100)

Switching providers
-------------------
Add one line to .env and restart:

    AI_PROVIDER=groq
    GROQ_API_KEY=gsk_...

    AI_PROVIDER=openrouter
    OPENROUTER_API_KEY=sk-or-...

    AI_PROVIDER=gemini           # (default)
    GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings  # module-level so tests can patch ai.factory.settings

if TYPE_CHECKING:
    from ai.provider import AIProvider


def create_provider() -> "AIProvider | None":
    """
    Create the configured AIProvider, or None when no key is available.

    Never raises — missing keys produce None so callers can fall back to
    template-only mode without special-casing the configuration.
    """
    name = (settings.ai_provider or "").lower().strip()

    if name == "openrouter":
        return _try_openrouter(settings)
    if name == "groq":
        return _try_groq(settings)
    if name == "gemini" or name == "":
        return _try_gemini(settings) or _try_openrouter(settings) or _try_groq(settings)

    # Unknown provider name — try all in priority order
    return _try_gemini(settings) or _try_openrouter(settings) or _try_groq(settings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_gemini(settings=settings) -> "AIProvider | None":
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai
        from ai.gemini_provider import GeminiProvider
        client = genai.Client(api_key=settings.gemini_api_key)
        return GeminiProvider(client=client, model=settings.gemini_model)
    except Exception:
        return None


def _try_openrouter(settings=settings) -> "AIProvider | None":
    if not settings.openrouter_api_key:
        return None
    try:
        from ai.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )
    except Exception:
        return None


def _try_groq(settings=settings) -> "AIProvider | None":
    if not settings.groq_api_key:
        return None
    try:
        from ai.groq_provider import GroqProvider
        return GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
    except Exception:
        return None
