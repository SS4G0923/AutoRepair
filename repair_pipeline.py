from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Callable

from inspector.inspector_prompt import build_planner_prompt
from llm.agent_tools import RepairToolContext, build_repair_tools
from llm import call_llm_for_json
from sandbox import build_runtime_inspection_report, run_python_code_safely

DEFAULT_MODEL = "qwen3.5-flash"
MAX_CODE_CHARS = 100_000
MAX_TIMEOUT_SEC = 30

INSPECTOR_JSON_SYSTEM_PROMPT = """You are an expert bug inspection agent.
You will receive a structured local runtime report for a single uploaded Python file.
Return a compact JSON object that is directly useful for planning and fixing the bug.

Requirements:
- Focus on the most likely root cause.
- Identify the most suspicious line or code region when possible.
- Include validation ideas for the final fix.
- Stay grounded in the runtime evidence and source code.
- Use the available function tools whenever you need precise local evidence from the uploaded file or runtime outputs.
"""

PLANNER_SYSTEM_PROMPT = """You are a helpful bug fix planning expert, you will be presented with bug information from a bug inspection expert.
What you will do is to generate a bug fix plan for bug fixing expert as instructed.
Please remember that the plan you provided will be used to implement the bug fix,
so be sure to include all the information that will be helpful to locate the bug and implement the bug fix.
Use the available function tools whenever you need exact source or runtime evidence before finalizing the plan."""

INSPECT_EXPLAIN_SYSTEM_PROMPT = """你是一个优秀的代码 Bug 分析师。
请基于给定的运行结果和检查报告，输出一份第一人称 explain 报告。
报告中需要包含：我如何定位问题、我判断根因的依据、我建议重点修改的位置。
直接输出正文，不要添加开场白和结束语。"""

PLAN_EXPLAIN_SYSTEM_PROMPT = """你是一个优秀的代码 Bug 修复计划师。
请基于给定的检查报告和修复计划，输出一份第一人称 explain 报告。
报告中需要包含：我对修复目标的理解、我会如何最小化修改、我会如何验证改动。
直接输出正文，不要添加开场白和结束语。"""

CODE_SYSTEM_PROMPT = """You are a senior Python repair agent.
You will be given a failing Python file, its runtime evidence, and a repair plan.
Return ONLY a unified git diff for the original file.

Rules:
- Modify only the uploaded file.
- Keep the change minimal and correctness-focused.
- Do not wrap the diff in Markdown fences.
- Do not add explanations before or after the diff.
- Use the available function tools to inspect exact source windows before you finalize the diff.
"""

CODE_EXPLAIN_SYSTEM_PROMPT = """你是一个优秀的代码修复工程师。
请基于修复计划和最终 diff，输出一份第一人称 explain 报告。
报告中需要包含：我改了什么、为什么这样改、这份改动如何解决原始错误、还有哪些边界风险。
直接输出正文，不要添加开场白和结束语。"""

EventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class RepairRequest:
    code: str
    filename: str = "main.py"
    language: str = "python"
    timeout_sec: int = 5
    model: str = DEFAULT_MODEL

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RepairRequest":
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("`code` must be a non-empty string.")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"`code` must be at most {MAX_CODE_CHARS} characters.")

        language = payload.get("language", "python")
        if language != "python":
            raise ValueError("Only `python` is supported by the secure local runner.")

        raw_filename = payload.get("filename", "main.py")
        if not isinstance(raw_filename, str) or not raw_filename.strip():
            raise ValueError("`filename` must be a non-empty string.")
        filename = raw_filename.strip()
        pure_path = PurePath(filename)
        if pure_path.is_absolute() or ".." in pure_path.parts or len(pure_path.parts) != 1:
            raise ValueError("`filename` must be a simple relative file name.")
        if not filename.endswith(".py"):
            raise ValueError("`filename` must end with `.py`.")

        timeout_sec = payload.get("timeout_sec", 5)
        if not isinstance(timeout_sec, int):
            raise ValueError("`timeout_sec` must be an integer.")
        if timeout_sec < 1 or timeout_sec > MAX_TIMEOUT_SEC:
            raise ValueError(f"`timeout_sec` must be between 1 and {MAX_TIMEOUT_SEC}.")

        model = payload.get("model", DEFAULT_MODEL)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("`model` must be a non-empty string.")

        return cls(
            code=code,
            filename=filename,
            language=language,
            timeout_sec=timeout_sec,
            model=model.strip(),
        )


def _normalize_diff(diff_text: str) -> str:
    cleaned = diff_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _emit_stage(emit: EventEmitter, stage: str, status: str, **payload: Any) -> None:
    emit(
        "stage",
        {
            "stage": stage,
            "status": status,
            **payload,
        },
    )


def _stream_explain(
    *,
    stage: str,
    prompt: str,
    system_prompt: str,
    model: str,
    emit: EventEmitter,
) -> str:
    _emit_stage(emit, stage, "explaining")

    def on_chunk(chunk: str) -> None:
        emit("explain_chunk", {"stage": stage, "chunk": chunk})

    explain_text = call_llm_for_json(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        isJson=False,
        stream=True,
        stream_handler=on_chunk,
    )
    emit("explain_done", {"stage": stage, "text": explain_text})
    return explain_text


def _build_tool_event_handler(emit: EventEmitter, stage: str) -> Callable[[str, dict[str, Any]], None]:
    def on_tool_event(status: str, payload: dict[str, Any]) -> None:
        emit(
            "tool_event",
            {
                "stage": stage,
                "status": status,
                **payload,
            },
        )

    return on_tool_event


def run_repair_pipeline(request: RepairRequest, emit: EventEmitter) -> None:
    _emit_stage(emit, "run", "started", message="Running uploaded code in an isolated local process.")
    execution = run_python_code_safely(
        request.code,
        filename=request.filename,
        timeout_sec=request.timeout_sec,
    )
    emit(
        "run_result",
        {
            "execution": execution.to_dict(),
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        },
    )

    if execution.ok:
        _emit_stage(emit, "run", "completed", message="No runtime error detected.")
        emit(
            "result",
            {
                "status": "clean",
                "message": "No error detected.",
                "filename": request.filename,
            },
        )
        return

    runtime_report = build_runtime_inspection_report(
        code=request.code,
        filename=request.filename,
        execution=execution,
    )
    stage_tools = build_repair_tools(
        RepairToolContext(
            filename=request.filename,
            code=request.code,
            runtime_report=runtime_report,
        )
    )

    _emit_stage(emit, "inspect", "started")
    inspector_report = call_llm_for_json(
        prompt=json.dumps(runtime_report, ensure_ascii=False, indent=2),
        system_prompt=INSPECTOR_JSON_SYSTEM_PROMPT,
        model=request.model,
        isJson=True,
        tools=stage_tools,
        tool_event_handler=_build_tool_event_handler(emit, "inspect"),
    )
    emit("inspect_report", {"stage": "inspect", "report": inspector_report})
    _stream_explain(
        stage="inspect",
        prompt=json.dumps(
            {
                "runtime_report": runtime_report,
                "inspector_report": inspector_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        system_prompt=INSPECT_EXPLAIN_SYSTEM_PROMPT,
        model=request.model,
        emit=emit,
    )
    _emit_stage(emit, "inspect", "completed")

    _emit_stage(emit, "plan", "started")
    planner_prompt = build_planner_prompt(inspector_report)
    planner_report = call_llm_for_json(
        prompt=planner_prompt,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        model=request.model,
        isJson=False,
        tools=stage_tools,
        tool_event_handler=_build_tool_event_handler(emit, "plan"),
    )
    emit("plan_report", {"stage": "plan", "report": planner_report})
    _stream_explain(
        stage="plan",
        prompt=json.dumps(
            {
                "inspector_report": inspector_report,
                "planner_report": planner_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        system_prompt=PLAN_EXPLAIN_SYSTEM_PROMPT,
        model=request.model,
        emit=emit,
    )
    _emit_stage(emit, "plan", "completed")

    _emit_stage(emit, "code", "started")

    def on_diff_chunk(chunk: str) -> None:
        emit("code_diff_chunk", {"stage": "code", "chunk": chunk})

    coder_prompt = json.dumps(
        {
            "filename": request.filename,
            "runtime_report": runtime_report,
            "inspector_report": inspector_report,
            "planner_report": planner_report,
            "original_code": request.code,
        },
        ensure_ascii=False,
        indent=2,
    )
    raw_diff = call_llm_for_json(
        prompt=coder_prompt,
        system_prompt=CODE_SYSTEM_PROMPT,
        model=request.model,
        isJson=False,
        stream=True,
        stream_handler=on_diff_chunk,
        tools=stage_tools,
        tool_event_handler=_build_tool_event_handler(emit, "code"),
    )
    git_diff = _normalize_diff(raw_diff)
    if not git_diff:
        raise RuntimeError("Repair model returned an empty diff.")

    emit("code_report", {"stage": "code", "git_diff": git_diff})
    _stream_explain(
        stage="code",
        prompt=json.dumps(
            {
                "planner_report": planner_report,
                "git_diff": git_diff,
            },
            ensure_ascii=False,
            indent=2,
        ),
        system_prompt=CODE_EXPLAIN_SYSTEM_PROMPT,
        model=request.model,
        emit=emit,
    )
    _emit_stage(emit, "code", "completed")
    emit(
        "result",
        {
            "status": "repaired",
            "filename": request.filename,
            "git_diff": git_diff,
        },
    )
