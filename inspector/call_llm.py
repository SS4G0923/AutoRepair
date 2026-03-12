from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful code debug inspection expert, you are given the "
    "following bug report, please generate a JSON bug report for the code fix "
    "planning expert, remember that the following experts will use this "
    "information to implement a bug fix, so be sure to include all the "
    "information that will be helpful to locate the bug and implement the bug "
    "fix."
)

API_KEY = ""
BASE_URL = "https://api.openai.com/v1"


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or the response is not valid JSON."""


def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    resolved_api_key = API_KEY if api_key is None else api_key
    if not resolved_api_key:
        raise LLMCallError("OPENAI_API_KEY is not set.")

    resolved_base_url = BASE_URL if base_url is None else base_url
    if resolved_base_url:
        return OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    return OpenAI(api_key=resolved_api_key)


def _extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_parts: list[str] = []
        for item in message_content:
            if not hasattr(item, "type"):
                continue
            if item.type == "text" and hasattr(item, "text") and item.text:
                text_parts.append(item.text)
        if text_parts:
            return "\n".join(text_parts)

    raise LLMCallError("Model response did not contain text content.")


def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """
    Call an OpenAI chat model and parse the response as JSON.

    This helper is intentionally generic so other modules can reuse it with
    different prompts and models.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    resolved_client = client or create_openai_client(api_key=api_key, base_url=base_url)

    try:
        response = resolved_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        raise LLMCallError(f"OpenAI request failed: {exc}") from exc

    if not response.choices:
        raise LLMCallError("OpenAI response did not include any choices.")

    message = response.choices[0].message
    response_text = _extract_message_text(message.content).strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMCallError(
            "Model response was not valid JSON. "
            f"Raw response: {response_text[:500]}"
        ) from exc
