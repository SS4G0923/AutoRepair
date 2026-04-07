from __future__ import annotations


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
):
    if "gemini" in model.lower():
        from backend.llm.call_gemini import call_llm_for_json as provider_call
    else:
        from backend.llm.call_gpt import call_llm_for_json as provider_call

    kwargs = {
        "prompt": prompt,
        "model": model,
        "isJson": isJson,
        "stream": stream,
        "stream_handler": stream_handler,
        "tools": tools,
        "tool_event_handler": tool_event_handler,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return provider_call(**kwargs)


__all__ = ["call_llm_for_json"]
