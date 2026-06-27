"""
GeminiProvider — AIProvider implementation backed by google-genai.

Maps AIProvider.complete() to client.models.generate_content() using the
same keyword arguments as the original agent code so that unit tests which
mock client.models.generate_content continue to pass unchanged.

Argument mapping
----------------
complete(user_message, system_prompt, response_schema, max_tokens)
    ↓
client.models.generate_content(
    model=self._model,
    contents=user_message,
    config=GenerateContentConfig(
        system_instruction=system_prompt,       # only when non-empty
        response_mime_type="application/json",  # only when response_schema set
        response_schema=response_schema,        # only when set
        max_output_tokens=max_tokens,
    ),
)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai.provider import AIProvider, AIProviderError

if TYPE_CHECKING:
    from pydantic import BaseModel


class GeminiProvider(AIProvider):
    """
    AIProvider backed by Google Gemini via the google-genai SDK.

    Parameters
    ----------
    client
        An initialised google.genai.Client.  When None the provider reads
        GEMINI_API_KEY from settings and constructs a client automatically.
    model
        Gemini model ID.  Defaults to gemini-2.0-flash.
    """

    def __init__(
        self,
        client: "object | None" = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        if client is None:
            from config import settings
            from google import genai
            if not settings.gemini_api_key:
                raise AIProviderError("GEMINI_API_KEY is not set in settings / .env")
            client = genai.Client(api_key=settings.gemini_api_key)
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    def complete(
        self,
        user_message: str,
        system_prompt: str = "",
        response_schema: "type[BaseModel] | None" = None,
        max_tokens: int = 1000,
    ) -> str:
        from google.genai import types

        config_kwargs: dict = {"max_output_tokens": max_tokens}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_message,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = getattr(response, "text", None)
            if not text:
                raise AIProviderError("empty response from Gemini")
            return text
        except AIProviderError:
            raise
        except Exception as exc:
            raise AIProviderError(f"Gemini API error: {exc}") from exc
