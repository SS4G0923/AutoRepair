from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from backend.llm import call_llm_for_json
from backend.llm.store import resolve_model_selection
from backend.llm.telemetry import LLMCallContext

MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 20_000

CHAT_SYSTEM_PROMPT = """You are AutoRepair Studio's coding assistant.
You help users reason about bugs, code changes, architecture, debugging steps, and implementation details.

Requirements:
- Be accurate and concise.
- Prefer practical engineering guidance.
- Use plain text only.
- Continue the conversation naturally based on the prior message history.
"""

EventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    at: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    model: str
    history_id: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("`messages` must be a non-empty array.")
        if len(raw_messages) > MAX_MESSAGES:
            raise ValueError(f"`messages` must contain at most {MAX_MESSAGES} items.")

        messages: list[ChatMessage] = []
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict):
                raise ValueError(f"`messages[{index}]` must be an object.")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"}:
                raise ValueError(f"`messages[{index}].role` must be `user` or `assistant`.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"`messages[{index}].content` must be a non-empty string.")
            if len(content) > MAX_MESSAGE_CHARS:
                raise ValueError(
                    f"`messages[{index}].content` must be at most {MAX_MESSAGE_CHARS} characters."
                )
            at = item.get("at")
            if at is not None and not isinstance(at, str):
                raise ValueError(f"`messages[{index}].at` must be a string when provided.")
            messages.append(ChatMessage(role=role, content=content.strip(), at=at))

        if messages[-1].role != "user":
            raise ValueError("The last chat message must be from the user.")

        raw_model = payload.get("model")
        if raw_model is not None and (not isinstance(raw_model, str) or not raw_model.strip()):
            raise ValueError("`model` must be a non-empty string when provided.")
        resolved_model = resolve_model_selection(
            raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None,
            purpose="chat",
        )

        raw_history_id = payload.get("history_id")
        history_id: int | None = None
        if raw_history_id is not None:
            if not isinstance(raw_history_id, int) or raw_history_id <= 0:
                raise ValueError("`history_id` must be a positive integer.")
            history_id = raw_history_id

        return cls(messages=messages, model=resolved_model.model_key, history_id=history_id)


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    transcript: list[dict[str, str]] = []
    for message in messages:
        transcript.append({"role": message.role, "content": message.content})

    return (
        "You are given the full conversation history as JSON.\n"
        "Write the next assistant reply for the most recent user message.\n"
        "Do not wrap the answer in JSON or Markdown fences.\n\n"
        f"{json.dumps(transcript, ensure_ascii=False, indent=2)}"
    )


def run_chat_pipeline(request: ChatRequest, emit: EventEmitter, *, user_id: int | None = None) -> None:
    emit(
        "accepted",
        {
            "model": request.model,
            "message_count": len(request.messages),
        },
    )

    response_parts: list[str] = []
    reasoning_parts: list[str] = []

    def on_chunk(chunk: str) -> None:
        response_parts.append(chunk)
        emit("chat_chunk", {"chunk": chunk})

    def on_reasoning_chunk(chunk: str) -> None:
        reasoning_parts.append(chunk)
        emit("chat_reasoning_chunk", {"chunk": chunk})

    response_text = call_llm_for_json(
        prompt=_build_chat_prompt(request.messages),
        system_prompt=CHAT_SYSTEM_PROMPT,
        model=request.model,
        isJson=False,
        stream=True,
        stream_handler=on_chunk,
        reasoning_handler=on_reasoning_chunk,
        audit_context=LLMCallContext(
            user_id=user_id,
            history_id=request.history_id,
            request_mode="chat",
            stage="chat",
            purpose="chat.reply",
        ),
    )

    final_text = response_text.strip() if isinstance(response_text, str) else ""
    if not final_text:
        final_text = "".join(response_parts).strip()
    if not final_text:
        raise RuntimeError("Chat model returned an empty response.")

    emit(
        "result",
        {
            "message": final_text,
            "reasoning": "".join(reasoning_parts).strip(),
            "model": request.model,
        },
    )
