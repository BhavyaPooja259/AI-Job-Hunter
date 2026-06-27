"""
AIProvider — abstract interface for all AI completion backends.

Every concrete provider (Gemini, OpenRouter, Groq, …) must implement
complete() and expose a name property.  Business logic never imports a
concrete provider class directly — it uses the AIProvider type and
receives an implementation via dependency injection or the factory.

Contract
--------
complete()
    Returns the raw text response from the model.  When response_schema
    is given, the caller expects the text to be valid JSON conforming to
    that Pydantic model, but validation is the caller's responsibility.
    Raises AIProviderError on any failure (network, auth, quota, empty
    response).  Callers should catch Exception to enable template fallback.

AIProviderError
    Raised instead of letting provider-specific exceptions leak.  Wraps
    the original error via __cause__ so the full traceback is preserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


class AIProviderError(Exception):
    """Raised when an AI provider call cannot produce a usable response."""


class AIProvider(ABC):
    """Abstract base for all AI completion providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier shown in logs and the demo (e.g. 'gemini')."""

    @abstractmethod
    def complete(
        self,
        user_message: str,
        system_prompt: str = "",
        response_schema: "type[BaseModel] | None" = None,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a completion from the model.

        Parameters
        ----------
        user_message
            The user turn of the conversation.
        system_prompt
            System-level instructions prepended to the conversation.
        response_schema
            Optional Pydantic model class.  When supplied:
            - Gemini: uses native JSON mode with schema enforcement.
            - OpenRouter / Groq: injects the JSON Schema into the system
              prompt and requests json_object response format.
            The caller must still call model_validate_json() on the result.
        max_tokens
            Upper bound on the response length.

        Returns
        -------
        str
            Raw text from the model (JSON when response_schema was given).

        Raises
        ------
        AIProviderError
            On empty response, authentication failure, quota exceeded, or
            any other provider-level error.
        """
