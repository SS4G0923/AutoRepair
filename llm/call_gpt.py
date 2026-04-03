from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable
from dotenv import load_dotenv

from openai import OpenAI
from llm.agent_tools import FunctionTool

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
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

API_KEY = QWEN_API_KEY
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_TOOL_ROUNDS = 8
MAX_TOOL_EVENT_PREVIEW_CHARS = 600


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or the response is not valid JSON."""


def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or API_KEY
    if not resolved_api_key:
        raise LLMCallError("OPENAI_API_KEY is not set.")

    resolved_base_url = os.getenv("OPENAI_BASE_URL") or (BASE_URL if base_url is None else base_url)
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
) -> dict[str, Any] | str:
    tool_registry = {tool.name: tool for tool in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    for round_index in range(MAX_TOOL_ROUNDS):
        try:
            response = resolved_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[tool.as_openai_tool() for tool in tools],
            )
        except Exception as exc:
            raise LLMCallError(f"OpenAI request failed: {exc}") from exc

        if not response.choices:
            raise LLMCallError("OpenAI response did not include any choices.")

        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not tool_calls:
            response_text = _extract_message_text(message.content).strip()
            if stream and not isJson and response_text:
                if stream_handler is not None:
                    stream_handler(response_text)
                else:
                    print(response_text, file=sys.stdout, flush=True)
            if not isJson:
                return response_text
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise LLMCallError(
                    "Model response was not valid JSON. "
                    f"Raw response: {response_text[:500]}"
                ) from exc

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
    isJson: bool = True,
    stream: bool = False,
    stream_handler: Callable[[str], None] | None = None,
    tools: list[FunctionTool] | None = None,
    tool_event_handler: Callable[[str, dict[str, Any]], None] | None = None,
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
        if stream and not isJson:
            response = resolved_client.chat.completions.create(
                **request_kwargs,
                stream=True,
            )
            response_text_parts: list[str] = []
            for chunk in response:
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
            return "".join(response_text_parts).strip()

        response = resolved_client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        raise LLMCallError(f"OpenAI request failed: {exc}") from exc

    if not response.choices:
        raise LLMCallError("OpenAI response did not include any choices.")

    message = response.choices[0].message
    response_text = _extract_message_text(message.content).strip()

    if not isJson:
        return response_text

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMCallError(
            "Model response was not valid JSON. "
            f"Raw response: {response_text[:500]}"
        ) from exc
