from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable
from dotenv import load_dotenv
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

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


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


def _json_candidates(response_text: str) -> list[str]:
    stripped = response_text.strip()
    candidates: list[str] = []

    def add(candidate: str) -> None:
        normalized = candidate.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(stripped)

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            add("\n".join(inner_lines))

    fence_start = stripped.find("```")
    if fence_start != -1:
        fence_end = stripped.find("```", fence_start + 3)
        if fence_end != -1:
            fenced = stripped[fence_start + 3 : fence_end].lstrip()
            if fenced.startswith("json"):
                fenced = fenced[4:].lstrip()
            add(fenced)

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        add(stripped[object_start : object_end + 1])

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        add(stripped[array_start : array_end + 1])

    return candidates


def _parse_json_response(response_text: str) -> dict[str, Any]:
    for candidate in _json_candidates(response_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMCallError(
        "Model response was not valid JSON. "
        f"Raw response: {response_text[:500]}"
    )


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens))
    cached_input_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)

    if total_tokens <= 0 and input_tokens <= 0 and output_tokens <= 0:
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "token_source": "provider",
    }


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
    metadata_handler: Callable[[dict[str, Any]], None] | None = None,
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
            latest_usage: dict[str, int] | None = None
            for chunk in response_stream:
                usage = _extract_usage(chunk)
                if usage is not None:
                    latest_usage = usage
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
            response_text = "".join(response_text_parts).strip()
            if metadata_handler is not None:
                metadata_handler(
                    {
                        "raw_response_text": response_text,
                        "usage": latest_usage,
                    }
                )
            return response_text

        response = resolved_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        raise LLMCallError(f"Gemini request failed: {exc}") from exc

    response_text = _extract_response_text(response).strip()
    if metadata_handler is not None:
        metadata_handler(
            {
                "raw_response_text": response_text,
                "usage": _extract_usage(response),
            }
        )

    if not isJson:
        return response_text

    return _parse_json_response(response_text)
