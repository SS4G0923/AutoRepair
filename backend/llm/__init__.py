from __future__ import annotations

import json

from backend.llm.store import (
    get_runtime_model_config,
    is_model_thinking_enabled,
    load_model_api_key,
)
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
    reasoning_handler=None,
    tools=None,
    tool_event_handler=None,
    audit_context: LLMCallContext | None = None,
):
    runtime_model = None
    try:
        runtime_model = get_runtime_model_config(model, include_disabled=True)
    except Exception:
        runtime_model = None

    provider_code = runtime_model.provider_code if runtime_model is not None else infer_provider(model)
    provider_name = runtime_model.provider_name if runtime_model is not None else provider_code
    provider_model_name = runtime_model.api_model_name if runtime_model is not None else model
    thinking_enabled = is_model_thinking_enabled(runtime_model) if runtime_model is not None else False
    is_ollama_like = bool(
        runtime_model is not None
        and (
            "ollama" in runtime_model.provider_name.strip().lower()
            or "ollama" in runtime_model.vendor_name.strip().lower()
            or (runtime_model.api_base_url or "").strip().startswith("http://127.0.0.1:11434")
            or "11434" in (runtime_model.api_base_url or "")
        )
    )

    if provider_code == "gemini":
        from backend.llm.call_gemini import call_llm_for_json as provider_call
    elif is_ollama_like:
        from backend.llm.call_ollama import call_llm_for_json as provider_call
    else:
        from backend.llm.call_gpt import call_llm_for_json as provider_call

    log_request_id: int | None = None
    log_started_monotonic = 0.0
    provider_metadata: dict[str, object] = {}

    if audit_context is not None:
        log_request_id, log_started_monotonic = start_llm_request_log(
            context=audit_context,
            model=model,
            provider=provider_name,
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
        "model": provider_model_name,
        "isJson": isJson,
        "stream": stream,
        "stream_handler": stream_handler,
        "reasoning_handler": reasoning_handler,
        "tools": tools,
        "tool_event_handler": wrapped_tool_event_handler,
        "metadata_handler": on_provider_metadata,
        "thinking_enabled": thinking_enabled,
    }
    if runtime_model is not None:
        api_key = load_model_api_key(runtime_model)
        if runtime_model.api_key_env_var and not api_key:
            raise RuntimeError(
                f"Model `{model}` requires environment variable `{runtime_model.api_key_env_var}`."
            )
        if api_key is not None:
            kwargs["api_key"] = api_key
        if provider_code != "gemini" and runtime_model.api_base_url:
            kwargs["base_url"] = runtime_model.api_base_url
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    try:
        result = provider_call(**kwargs)
    except Exception as exc:
        raw_response_text = (
            str(provider_metadata.get("raw_response_text"))
            if provider_metadata.get("raw_response_text") is not None
            else getattr(exc, "raw_response", None)
        )
        if raw_response_text:
            try:
                setattr(exc, "raw_response", raw_response_text)
            except Exception:
                pass
        if not hasattr(exc, "json_parse_failed"):
            try:
                setattr(exc, "json_parse_failed", "not valid JSON" in str(exc))
            except Exception:
                pass
        if log_request_id is not None:
            finish_llm_request_log(
                request_id=log_request_id,
                started_monotonic=log_started_monotonic,
                prompt=prompt,
                system_prompt=system_prompt,
                response_text=raw_response_text,
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
