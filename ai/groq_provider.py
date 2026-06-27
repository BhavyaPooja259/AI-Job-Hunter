"""
GroqProvider — AIProvider backed by api.groq.com.

Groq exposes an OpenAI-compatible REST API at a different base URL.  The
implementation mirrors OpenRouterProvider — both inject JSON schemas via
the system prompt because neither supports native Pydantic schema validation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from ai.provider import AIProvider, AIProviderError

if TYPE_CHECKING:
    from pydantic import BaseModel

_BASE_URL = "https://api.groq.com/openai/v1"


_SCHEMA_STRIP_THRESHOLD = 2000  # chars; below this, keep descriptions intact


def _strip_code_fence(text: str) -> str:
    """Remove markdown ``` fences that some free-tier models add around JSON.

    Despite response_format=json_object, certain models wrap their output in
    ```json ... ``` which breaks json.loads / model_validate_json.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    start = 1
    end = len(lines)
    if lines[-1].strip() == "```":
        end -= 1
    return "\n".join(lines[start:end]).strip()


def _maybe_strip_schema(schema: dict) -> dict:
    """Drop description/title from a JSON schema only when the schema is large.

    Small schemas (CoverLetterData ~1800 chars) keep their field descriptions
    so free-tier models understand what content to produce.

    Large schemas (TailoredResumeData ~4300 chars verbose → ~1250 stripped)
    are trimmed because verbose descriptions overflow the free-model context
    and cause `content: null` responses.  Only type, required, and constraint
    keys are needed for the model to emit valid JSON.
    """
    raw = json.dumps(schema)
    if len(raw) <= _SCHEMA_STRIP_THRESHOLD:
        return schema

    def _strip(obj: object) -> object:
        if isinstance(obj, dict):
            # Strip only "description" — "title" is also used as a property
            # name in Pydantic models (e.g. ExperienceEntry.title) so removing
            # it as a key would corrupt the schema's properties dict.
            return {k: _strip(v) for k, v in obj.items() if k != "description"}
        if isinstance(obj, list):
            return [_strip(i) for i in obj]
        return obj

    return _strip(schema)  # type: ignore[return-value]


class GroqProvider(AIProvider):
    """
    AIProvider backed by Groq.

    Parameters
    ----------
    api_key
        Groq API key from console.groq.com.  Required.
    model
        Groq model ID (e.g. "llama-3.3-70b-versatile").
    timeout
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise AIProviderError("Groq API key is required")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return "groq"

    def complete(
        self,
        user_message: str,
        system_prompt: str = "",
        response_schema: "type[BaseModel] | None" = None,
        max_tokens: int = 1000,
    ) -> str:
        effective_system = system_prompt

        if response_schema is not None:
            schema_str = json.dumps(
                _maybe_strip_schema(response_schema.model_json_schema()), indent=2
            )
            effective_system = (
                f"{effective_system}\n\n"
                f"OUTPUT FORMAT: respond with a single valid JSON object that contains "
                f"REAL data filling in ALL required fields listed below. "
                f"Do NOT return the schema definition itself.\n"
                f"Required JSON structure:\n"
                f"{schema_str}"
            ).lstrip()

        messages: list[dict] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": user_message})

        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            body["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                f"{_BASE_URL}/chat/completions",
                json=body,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            if not text:
                raise AIProviderError("empty response from Groq")
            return _strip_code_fence(text)
        except AIProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            raise AIProviderError(
                f"Groq HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise AIProviderError(f"Groq error: {exc}") from exc
