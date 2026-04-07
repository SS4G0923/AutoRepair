from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from backend.auth.store import get_db_connection


@dataclass(frozen=True)
class LLMCallContext:
    user_id: int | None = None
    history_id: int | None = None
    request_mode: str = "system"
    stage: str | None = None
    purpose: str | None = None
    source_type: str | None = None


@dataclass(frozen=True)
class TokenUsageSummary:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    token_source: str = "estimated"


def infer_provider(model: str) -> str:
    lowered = model.lower()
    if "gemini" in lowered:
        return "gemini"
    if "qwen" in lowered:
        return "qwen"
    if "gpt" in lowered or "openai" in lowered:
        return "openai"
    return "unknown"


def estimate_text_tokens(text: str) -> int:
    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def summarize_token_usage(
    *,
    prompt: str,
    system_prompt: str | None,
    response_text: str,
    provider_usage: dict[str, Any] | None,
) -> TokenUsageSummary:
    if provider_usage:
        input_tokens = int(provider_usage.get("input_tokens") or 0)
        output_tokens = int(provider_usage.get("output_tokens") or 0)
        total_tokens = int(provider_usage.get("total_tokens") or (input_tokens + output_tokens))
        cached_input_tokens = int(provider_usage.get("cached_input_tokens") or 0)
        reasoning_tokens = int(provider_usage.get("reasoning_tokens") or 0)
        if total_tokens > 0 or input_tokens > 0 or output_tokens > 0:
            return TokenUsageSummary(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or (input_tokens + output_tokens),
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
                token_source=str(provider_usage.get("token_source") or "provider"),
            )

    input_tokens = estimate_text_tokens(f"{system_prompt or ''}\n{prompt}")
    output_tokens = estimate_text_tokens(response_text)
    return TokenUsageSummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        token_source="estimated",
    )


def start_llm_request_log(
    *,
    context: LLMCallContext,
    model: str,
    provider: str,
    system_prompt: str | None,
    prompt: str,
    is_streaming: bool,
    is_json_response: bool,
) -> tuple[int, float]:
    started_monotonic = time.perf_counter()
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO llm_requests (
                    user_id,
                    history_id,
                    request_mode,
                    stage,
                    purpose,
                    provider,
                    model,
                    source_type,
                    is_streaming,
                    is_json_response,
                    request_status,
                    prompt_chars
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'started', %s)
                """,
                (
                    context.user_id,
                    context.history_id,
                    context.request_mode,
                    context.stage,
                    context.purpose,
                    provider,
                    model,
                    context.source_type,
                    1 if is_streaming else 0,
                    1 if is_json_response else 0,
                    len(prompt) + len(system_prompt or ""),
                ),
            )
            request_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO llm_request_messages (
                    request_id,
                    system_prompt,
                    prompt_text
                )
                VALUES (%s, %s, %s)
                """,
                (request_id, system_prompt, prompt),
            )
        connection.commit()
    return request_id, started_monotonic


def append_tool_event(
    *,
    request_id: int,
    status: str,
    tool_name: str,
    round_index: int | None = None,
    arguments_json: str | None = None,
    output_preview: str | None = None,
    output_truncated: bool = False,
) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO llm_request_tool_events (
                    request_id,
                    round_index,
                    status,
                    tool_name,
                    arguments_json,
                    output_preview,
                    output_truncated
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    request_id,
                    round_index,
                    status,
                    tool_name,
                    arguments_json,
                    output_preview,
                    1 if output_truncated else 0,
                ),
            )
        connection.commit()


def finish_llm_request_log(
    *,
    request_id: int,
    started_monotonic: float,
    prompt: str,
    system_prompt: str | None,
    response_text: str | None,
    parsed_response: dict[str, Any] | None,
    provider_usage: dict[str, Any] | None,
    success: bool,
    error_message: str | None = None,
) -> None:
    latency_ms = max(0, int((time.perf_counter() - started_monotonic) * 1000))
    normalized_response_text = response_text or ""
    usage = summarize_token_usage(
        prompt=prompt,
        system_prompt=system_prompt,
        response_text=normalized_response_text,
        provider_usage=provider_usage,
    )
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE llm_requests
                SET request_status = %s,
                    token_source = %s,
                    response_chars = %s,
                    latency_ms = %s,
                    error_message = %s,
                    finished_at = NOW()
                WHERE id = %s
                """,
                (
                    "completed" if success else "failed",
                    usage.token_source,
                    len(normalized_response_text),
                    latency_ms,
                    error_message,
                    request_id,
                ),
            )
            cursor.execute(
                """
                UPDATE llm_request_messages
                SET response_text = %s,
                    parsed_response_json = %s
                WHERE request_id = %s
                """,
                (
                    normalized_response_text or None,
                    json.dumps(parsed_response, ensure_ascii=False, indent=2)
                    if parsed_response is not None
                    else None,
                    request_id,
                ),
            )
            cursor.execute(
                """
                INSERT INTO llm_token_usage (
                    request_id,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    reasoning_tokens
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    input_tokens = VALUES(input_tokens),
                    output_tokens = VALUES(output_tokens),
                    total_tokens = VALUES(total_tokens),
                    cached_input_tokens = VALUES(cached_input_tokens),
                    reasoning_tokens = VALUES(reasoning_tokens)
                """,
                (
                    request_id,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_input_tokens,
                    usage.reasoning_tokens,
                ),
            )
        connection.commit()
