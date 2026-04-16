from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable
from dotenv import load_dotenv

from openai import OpenAI
from backend.llm.agent_tools import FunctionTool

load_dotenv()

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful code debug inspection expert, you are given the "
    "following bug report, please generate a JSON bug report for the code fix "
    "planning expert, remember that the following experts will use this "
    "information to implement a bug fix, so be sure to include all the "
    "information that will be helpful to locate the bug and implement the bug "
    "fix."
)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_TOOL_ROUNDS = 8
MAX_TOOL_EVENT_PREVIEW_CHARS = 600


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or the response is not valid JSON."""


def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
    if not resolved_api_key:
        raise LLMCallError("No OpenAI-compatible API key is configured.")

    if base_url is not None:
        resolved_base_url = base_url.strip() or None
    elif api_key is not None:
        resolved_base_url = os.getenv("OPENAI_BASE_URL") or None
    elif os.getenv("OPENAI_API_KEY"):
        resolved_base_url = os.getenv("OPENAI_BASE_URL") or None
    else:
        resolved_base_url = (
            os.getenv("QWEN_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DASHSCOPE_BASE_URL
        )
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


def _extract_delta_text(delta_content: Any) -> str:
    if isinstance(delta_content, str):
        return delta_content

    if isinstance(delta_content, list):
        text_parts: list[str] = []
        for item in delta_content:
            item_type = getattr(item, "type", None)
            if item_type == "text" and getattr(item, "text", None):
                text_parts.append(item.text)
        return "".join(text_parts)

    return ""


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


def _extract_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None

    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached_input_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
    reasoning_tokens = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        if completion_details
        else 0
    )

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


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump(exclude_none=True)

    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", ""),
        },
    }


def _invoke_tool(
    tool_registry: dict[str, FunctionTool],
    *,
    tool_name: str,
    raw_arguments: str,
) -> str:
    tool = tool_registry.get(tool_name)
    if tool is None:
        return json.dumps(
            {
                "ok": False,
                "error": f"Unknown tool: {tool_name}",
            },
            ensure_ascii=False,
        )

    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"Invalid tool arguments JSON: {exc}",
                "raw_arguments": raw_arguments,
            },
            ensure_ascii=False,
        )

    try:
        result = tool.handler(arguments)
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
                "tool_name": tool_name,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "tool_name": tool_name,
            "result": result,
        },
        ensure_ascii=False,
    )


def _call_with_tools(
    *,
    resolved_client: OpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    isJson: bool,
    stream: bool,
    stream_handler: Callable[[str], None] | None,
    tools: list[FunctionTool],
    tool_event_handler: Callable[[str, dict[str, Any]], None] | None,
    metadata_handler: Callable[[dict[str, Any]], None] | None,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any] | str:
    tool_registry = {tool.name: tool for tool in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "token_source": "provider",
    }
    has_provider_usage = False

    for round_index in range(MAX_TOOL_ROUNDS):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": [tool.as_openai_tool() for tool in tools],
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            response = resolved_client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise LLMCallError(f"OpenAI request failed: {exc}") from exc

        if not response.choices:
            raise LLMCallError("OpenAI response did not include any choices.")

        usage = _extract_usage(getattr(response, "usage", None))
        if usage is not None:
            has_provider_usage = True
            for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
                usage_totals[key] += int(usage.get(key, 0))

        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not tool_calls:
            response_text = _extract_message_text(message.content).strip()
            if metadata_handler is not None:
                metadata_handler(
                    {
                        "raw_response_text": response_text,
                        "usage": usage_totals if has_provider_usage else None,
                    }
                )
            if stream and not isJson and response_text:
                if stream_handler is not None:
                    stream_handler(response_text)
                else:
                    print(response_text, file=sys.stdout, flush=True)
            if not isJson:
                return response_text
            return _parse_json_response(response_text)

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_serialize_tool_call(tool_call) for tool_call in tool_calls],
            }
        )

        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            tool_name = getattr(function, "name", "")
            raw_arguments = getattr(function, "arguments", "") or ""
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
                    "tool_call_id": getattr(tool_call, "id", ""),
                    "content": tool_output,
                }
            )

    raise LLMCallError("Model exceeded the maximum number of tool-calling rounds.")


def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    thinking_enabled: bool = False,
    extra_body: dict[str, Any] | None = None,
    isJson: bool = True,
    stream: bool = False,
    stream_handler: Callable[[str], None] | None = None,
    tools: list[FunctionTool] | None = None,
    tool_event_handler: Callable[[str, dict[str, Any]], None] | None = None,
    metadata_handler: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | str:
    """
    Call an OpenAI chat model and parse the response as JSON.

    This helper is intentionally generic so other modules can reuse it with
    different prompts and models.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    resolved_client = client or create_openai_client(api_key=api_key, base_url=base_url)

    if tools:
        return _call_with_tools(
            resolved_client=resolved_client,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            isJson=isJson,
            stream=stream,
            stream_handler=stream_handler,
            tools=tools,
            tool_event_handler=tool_event_handler,
            metadata_handler=metadata_handler,
            extra_body=extra_body,
        )

    try:
        resp_format = {"type": "json_object"} if isJson else {"type": "text"}
        request_kwargs = {
            "model": model,
            "response_format": resp_format,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        if stream and not isJson:
            response = resolved_client.chat.completions.create(
                **request_kwargs,
                stream=True,
            )
            response_text_parts: list[str] = []
            latest_usage: dict[str, int] | None = None
            for chunk in response:
                usage = _extract_usage(getattr(chunk, "usage", None))
                if usage is not None:
                    latest_usage = usage
                if not chunk.choices:
                    continue
                delta_text = _extract_delta_text(chunk.choices[0].delta.content)
                if not delta_text:
                    continue
                if stream_handler is not None:
                    stream_handler(delta_text)
                else:
                    print(delta_text, end="", flush=True)
                response_text_parts.append(delta_text)
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

        response = resolved_client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        raise LLMCallError(f"OpenAI request failed: {exc}") from exc

    if not response.choices:
        raise LLMCallError("OpenAI response did not include any choices.")

    message = response.choices[0].message
    response_text = _extract_message_text(message.content).strip()
    if metadata_handler is not None:
        metadata_handler(
            {
                "raw_response_text": response_text,
                "usage": _extract_usage(getattr(response, "usage", None)),
            }
        )

    if not isJson:
        return response_text

    return _parse_json_response(response_text)
