from __future__ import annotations

import json

from backend.llm.telemetry import (
    LLMCallContext,
    append_tool_event,
    finish_llm_request_log,
    infer_provider,
    start_llm_request_log,
)

def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str,
    isJson: bool = True,
    stream: bool = False,
    stream_handler=None,
    tools=None,
    tool_event_handler=None,
    audit_context: LLMCallContext | None = None,
):
    if "gemini" in model.lower():
        from backend.llm.call_gemini import call_llm_for_json as provider_call
    else:
        from backend.llm.call_gpt import call_llm_for_json as provider_call

    log_request_id: int | None = None
    log_started_monotonic = 0.0
    provider_metadata: dict[str, object] = {}

    if audit_context is not None:
        log_request_id, log_started_monotonic = start_llm_request_log(
            context=audit_context,
            model=model,
            provider=infer_provider(model),
            system_prompt=system_prompt,
            prompt=prompt,
            is_streaming=stream,
            is_json_response=isJson,
        )

    def on_provider_metadata(metadata: dict[str, object]) -> None:
        provider_metadata.update(metadata)

    def wrapped_tool_event_handler(status: str, payload: dict[str, object]) -> None:
        if tool_event_handler is not None:
            tool_event_handler(status, payload)
        if log_request_id is None:
            return
        append_tool_event(
            request_id=log_request_id,
            status=status,
            tool_name=str(payload.get("tool_name") or "tool"),
            round_index=int(payload["round"]) if payload.get("round") is not None else None,
            arguments_json=str(payload["arguments"]) if payload.get("arguments") is not None else None,
            output_preview=str(payload["output_preview"]) if payload.get("output_preview") is not None else None,
            output_truncated=bool(payload.get("output_truncated", False)),
        )

    kwargs = {
        "prompt": prompt,
        "model": model,
        "isJson": isJson,
        "stream": stream,
        "stream_handler": stream_handler,
        "tools": tools,
        "tool_event_handler": wrapped_tool_event_handler,
        "metadata_handler": on_provider_metadata,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    try:
        result = provider_call(**kwargs)
    except Exception as exc:
        if log_request_id is not None:
            finish_llm_request_log(
                request_id=log_request_id,
                started_monotonic=log_started_monotonic,
                prompt=prompt,
                system_prompt=system_prompt,
                response_text=(
                    str(provider_metadata.get("raw_response_text"))
                    if provider_metadata.get("raw_response_text") is not None
                    else None
                ),
                parsed_response=None,
                provider_usage=provider_metadata.get("usage")
                if isinstance(provider_metadata.get("usage"), dict)
                else None,
                success=False,
                error_message=str(exc),
            )
        raise

    if log_request_id is not None:
        response_text = provider_metadata.get("raw_response_text")
        if response_text is None:
            response_text = json.dumps(result, ensure_ascii=False, indent=2) if isJson else str(result)
        finish_llm_request_log(
            request_id=log_request_id,
            started_monotonic=log_started_monotonic,
            prompt=prompt,
            system_prompt=system_prompt,
            response_text=str(response_text),
            parsed_response=result if isJson and isinstance(result, dict) else None,
            provider_usage=provider_metadata.get("usage")
            if isinstance(provider_metadata.get("usage"), dict)
            else None,
            success=True,
        )

    return result


__all__ = ["call_llm_for_json"]
