"""
ai package — provider-agnostic AI completion layer.

    from ai import AIProvider, AIProviderError, create_provider
    from ai import GeminiProvider, OpenRouterProvider, GroqProvider
"""

from ai.provider import AIProvider, AIProviderError
from ai.gemini_provider import GeminiProvider
from ai.openrouter_provider import OpenRouterProvider
from ai.groq_provider import GroqProvider
from ai.factory import create_provider

__all__ = [
    "AIProvider",
    "AIProviderError",
    "GeminiProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "create_provider",
]
