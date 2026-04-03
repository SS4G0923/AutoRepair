from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable
from google import genai

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful code debug inspection expert, you are given the "
    "following bug report, please generate a JSON bug report for the code fix "
    "planning expert, remember that the following experts will use this "
    "information to implement a bug fix, so be sure to include all the "
    "information that will be helpful to locate the bug and implement the bug "
    "fix."
)

API_KEY = "AIzaSyAVtW21YbrEb-Y1hd5nSodPYVCoNnfpRV0"


class LLMCallError(RuntimeError):
    """Raised when a Gemini call fails or the response is not valid JSON."""


def create_gemini_client(api_key: str | None = None) -> Any:
    resolved_api_key = os.getenv("GEMINI_API_KEY") or (API_KEY if api_key is None else api_key)
    if not resolved_api_key:
        raise LLMCallError("GEMINI_API_KEY is not set.")

    return genai.Client(api_key=resolved_api_key)


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    if text_parts:
        return "\n".join(text_parts)

    raise LLMCallError("Gemini response did not contain text content.")


def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    api_key: str | None = None,
    isJson: bool = True,
    stream: bool = False,
    stream_handler: Callable[[str], None] | None = None,
    tools=None,
    tool_event_handler=None,
) -> dict[str, Any] | str:
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    resolved_client = client or create_gemini_client(api_key=api_key)

    try:
        from google.genai import types
    except ImportError as exc:
        raise LLMCallError(
            "google-genai is not installed. Install the official Gemini SDK first."
        ) from exc

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json" if isJson else "text/plain",
        )
        if stream and not isJson:
            response_stream = resolved_client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=config,
            )
            response_text_parts: list[str] = []
            for chunk in response_stream:
                chunk_text = _extract_response_text(chunk)
                if not chunk_text:
                    continue
                if stream_handler is not None:
                    stream_handler(chunk_text)
                else:
                    print(chunk_text, end="", flush=True)
                response_text_parts.append(chunk_text)
            if response_text_parts and stream_handler is None:
                print(file=sys.stdout, flush=True)
            return "".join(response_text_parts).strip()

        response = resolved_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        raise LLMCallError(f"Gemini request failed: {exc}") from exc

    response_text = _extract_response_text(response).strip()

    if not isJson:
        return response_text

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMCallError(
            "Model response was not valid JSON. "
            f"Raw response: {response_text[:500]}"
        ) from exc
