from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from backend.llm.agent_tools import FunctionTool
from backend.llm.call_gpt import (
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    LLMCallError,
    MAX_TOOL_EVENT_PREVIEW_CHARS,
    MAX_TOOL_ROUNDS,
    _invoke_tool,
    _parse_json_response,
)


def _resolve_chat_url(base_url: str | None) -> str:
    candidate = (base_url or "http://127.0.0.1:11434").strip()
    if not candidate:
        candidate = "http://127.0.0.1:11434"

    parsed = urllib.parse.urlparse(candidate)
    if not parsed.scheme:
        candidate = f"http://{candidate.lstrip('/')}"
        parsed = urllib.parse.urlparse(candidate)

    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    elif path.endswith("/v1/"):
        path = path[:-4]
    elif path.endswith("/api/chat"):
        path = path[: -len("/api/chat")]

    normalized_path = f"{path}/api/chat".replace("//", "/")
    return urllib.parse.urlunparse(
        (
            parsed.scheme or "http",
            parsed.netloc,
            normalized_path,
            "",
            "",
            "",
        )
    )


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _strip_thinking_content(text: str, *, thinking_enabled: bool) -> str:
    normalized = text or ""
    if thinking_enabled:
        return normalized.strip()
    if "</think>" in normalized.lower():
        closing_pattern = re.compile(r"</think>", flags=re.IGNORECASE)
        parts = closing_pattern.split(normalized)
        normalized = parts[-1]
    without_blocks = re.sub(r"<think>.*?</think>", "", normalized, flags=re.IGNORECASE | re.DOTALL)
    without_orphans = re.sub(r"</?think>", "", without_blocks, flags=re.IGNORECASE)
    return _collapse_reasoning_preface(without_orphans.strip())


def _collapse_reasoning_preface(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) < 2:
        return text.strip()

    final_paragraph = paragraphs[-1]
    prior_text = "\n\n".join(paragraphs[:-1])
    prior_lower = prior_text.lower()
    reasoning_markers = (
        "首先",
        "最后",
        "最终",
        "用户",
        "作为",
        "我需要",
        "确认",
        "检查",
        "所以",
        "let me",
        "i need to",
        "the user",
        "final answer",
        "therefore",
        "first,",
    )
    has_reasoning_marker = any(marker in prior_lower for marker in reasoning_markers)
    if not has_reasoning_marker:
        return text.strip()

    if len(final_paragraph) > 400:
        return text.strip()

    if len(final_paragraph) * 4 > max(len(prior_text), 1):
        return text.strip()

    return final_paragraph.strip()


def _extract_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    input_tokens = int(payload.get("prompt_eval_count") or 0)
    output_tokens = int(payload.get("eval_count") or 0)
    total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "token_source": "provider",
    }


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("thinking")
    return str(reasoning or "").strip()


def _encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    if body:
        return f"HTTP {exc.code}: {body}"
    return f"HTTP {exc.code}: {exc.reason}"


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: int = 180,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=_encode_payload(payload),
        headers=_build_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise LLMCallError(f"Ollama request failed: {_http_error_message(exc)}") from exc
    except urllib.error.URLError as exc:
        raise LLMCallError(f"Ollama request failed: {exc}") from exc

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMCallError(
            f"Ollama response was not valid JSON: {raw_text[:500]}",
            raw_response=raw_text,
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMCallError(
            f"Ollama response did not return a JSON object: {raw_text[:500]}",
            raw_response=raw_text,
        )
    return parsed


def _stream_chat(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    stream_handler: Callable[[str], None] | None,
    reasoning_handler: Callable[[str], None] | None,
    thinking_enabled: bool,
) -> tuple[str, str, dict[str, int] | None]:
    request = urllib.request.Request(
        url,
        data=_encode_payload(payload),
        headers=_build_headers(api_key),
        method="POST",
    )
    response_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    final_payload: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                final_payload = chunk
                message = chunk.get("message") or {}
                chunk_reasoning = _extract_reasoning_text(message)
                if chunk_reasoning:
                    reasoning_chunks.append(chunk_reasoning)
                    if reasoning_handler is not None:
                        reasoning_handler(chunk_reasoning)
                chunk_text = str(message.get("content") or "")
                if not chunk_text:
                    continue
                response_chunks.append(chunk_text)
    except urllib.error.HTTPError as exc:
        raise LLMCallError(f"Ollama streaming request failed: {_http_error_message(exc)}") from exc
    except urllib.error.URLError as exc:
        raise LLMCallError(f"Ollama streaming request failed: {exc}") from exc

    response_text = _strip_thinking_content(
        "".join(response_chunks),
        thinking_enabled=thinking_enabled,
    )
    reasoning_text = "".join(reasoning_chunks).strip()
    if response_text and stream_handler is not None:
        stream_handler(response_text)
    return response_text, reasoning_text, _extract_usage(final_payload or {})


def _as_ollama_messages(
    *,
    system_prompt: str,
    prompt: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _tool_arguments_json(raw_arguments: Any) -> str:
    if isinstance(raw_arguments, str):
        return raw_arguments
    if isinstance(raw_arguments, dict):
        return json.dumps(raw_arguments, ensure_ascii=False)
    return ""


def _call_with_tools(
    *,
    url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    isJson: bool,
    stream: bool,
    stream_handler: Callable[[str], None] | None,
    tools: list[FunctionTool],
    tool_event_handler: Callable[[str, dict[str, Any]], None] | None,
    metadata_handler: Callable[[dict[str, Any]], None] | None,
    reasoning_handler: Callable[[str], None] | None,
    api_key: str | None,
    thinking_enabled: bool,
) -> dict[str, Any] | str:
    messages = _as_ollama_messages(system_prompt=system_prompt, prompt=prompt)
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "token_source": "provider",
    }
    has_provider_usage = False
    tool_registry = {tool.name: tool for tool in tools}

    for round_index in range(MAX_TOOL_ROUNDS):
        request_payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "think": thinking_enabled,
            "messages": messages,
            "tools": [tool.as_openai_tool() for tool in tools],
        }
        response = _post_json(url, request_payload, api_key=api_key)

        usage = _extract_usage(response)
        if usage is not None:
            has_provider_usage = True
            for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
                usage_totals[key] += int(usage.get(key, 0))

        message = response.get("message") or {}
        reasoning_text = _extract_reasoning_text(message)
        if reasoning_text and reasoning_handler is not None:
            reasoning_handler(reasoning_text)
        raw_response_text = str(message.get("content") or "")
        response_text = _strip_thinking_content(
            raw_response_text,
            thinking_enabled=thinking_enabled,
        )
        tool_calls = list(message.get("tool_calls") or [])

        if not tool_calls:
            if metadata_handler is not None:
                metadata_handler(
                    {
                        "raw_response_text": response_text,
                        "reasoning_text": reasoning_text,
                        "usage": usage_totals if has_provider_usage else None,
                    }
                )
            if stream and not isJson and response_text and stream_handler is not None:
                stream_handler(response_text)
            if not isJson:
                return response_text
            return _parse_json_response(response_text)

        messages.append(
            {
                "role": "assistant",
                "content": raw_response_text,
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "")
            raw_arguments = _tool_arguments_json(function.get("arguments"))
            if tool_event_handler is not None:
                tool_event_handler(
                    "started",
                    {
                        "tool_name": tool_name,
                        "arguments": raw_arguments,
                        "round": round_index + 1,
                    },
                )
            tool_output = _invoke_tool(
                tool_registry,
                tool_name=tool_name,
                raw_arguments=raw_arguments,
            )
            if tool_event_handler is not None:
                tool_event_handler(
                    "completed",
                    {
                        "tool_name": tool_name,
                        "output_preview": tool_output[:MAX_TOOL_EVENT_PREVIEW_CHARS],
                        "output_truncated": len(tool_output) > MAX_TOOL_EVENT_PREVIEW_CHARS,
                        "round": round_index + 1,
                    },
                )
            messages.append(
                {
                    "role": "tool",
                    "content": tool_output,
                }
            )

    raise LLMCallError("Model exceeded the maximum number of tool-calling rounds.")


def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    thinking_enabled: bool = False,
    isJson: bool = True,
    stream: bool = False,
    stream_handler: Callable[[str], None] | None = None,
    reasoning_handler: Callable[[str], None] | None = None,
    tools: list[FunctionTool] | None = None,
    tool_event_handler: Callable[[str, dict[str, Any]], None] | None = None,
    metadata_handler: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | str:
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    chat_url = _resolve_chat_url(base_url)
    if tools:
        return _call_with_tools(
            url=chat_url,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            isJson=isJson,
            stream=stream,
            stream_handler=stream_handler,
            tools=tools,
            tool_event_handler=tool_event_handler,
            metadata_handler=metadata_handler,
            reasoning_handler=reasoning_handler,
            api_key=api_key,
            thinking_enabled=thinking_enabled,
        )

    request_payload: dict[str, Any] = {
        "model": model,
        "stream": bool(stream and not isJson),
        "think": thinking_enabled,
        "messages": _as_ollama_messages(system_prompt=system_prompt, prompt=prompt),
    }
    if isJson:
        request_payload["format"] = "json"

    if stream and not isJson:
        response_text, reasoning_text, usage = _stream_chat(
            chat_url,
            request_payload,
            api_key=api_key,
            stream_handler=stream_handler,
            reasoning_handler=reasoning_handler,
            thinking_enabled=thinking_enabled,
        )
        if metadata_handler is not None:
            metadata_handler(
                {
                    "raw_response_text": response_text,
                    "reasoning_text": reasoning_text,
                    "usage": usage,
                }
            )
        return response_text

    response = _post_json(chat_url, request_payload, api_key=api_key)
    message = response.get("message") or {}
    reasoning_text = _extract_reasoning_text(message)
    if reasoning_text and reasoning_handler is not None:
        reasoning_handler(reasoning_text)
    response_text = _strip_thinking_content(
        str(message.get("content") or ""),
        thinking_enabled=thinking_enabled,
    )
    if metadata_handler is not None:
        metadata_handler(
            {
                "raw_response_text": response_text,
                "reasoning_text": reasoning_text,
                "usage": _extract_usage(response),
            }
        )

    if not isJson:
        return response_text
    return _parse_json_response(response_text)
