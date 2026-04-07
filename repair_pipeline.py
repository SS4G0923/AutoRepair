from __future__ import annotations

import ast
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

VERIFY_JSON_SYSTEM_PROMPT = """You are a Python repair verification agent.
You will receive the original failure evidence, the repair plan, the proposed git diff, and the patched Python file.
Return a compact JSON object describing a small self-contained verification block.

Return JSON with this shape:
{
  "summary": "short verification summary",
  "verification_code": [
    "python line 1",
    "python line 2"
  ],
  "assertion_targets": [
    "what each assertion checks"
  ]
}

Rules:
- `verification_code` must be valid Python source lines, not Markdown.
- Include 2 to 6 `assert` statements somewhere in `verification_code`.
- You may include minimal setup lines, try/except blocks, and helper variables when needed.
- Use only Python stdlib and symbols already defined in the patched file.
- Do not import modules, and do not define or redefine functions/classes inside `verification_code`.
- Prefer checks that would fail on the buggy version and pass on the patched version.
- Keep the verification minimal and deterministic.
- Do not access the network, filesystem, stdin, or external packages.
- Use the available function tools when you need exact patched source context before writing assertions.
"""

VERIFY_EXPLAIN_SYSTEM_PROMPT = """你是一个优秀的代码修复验证工程师。
请基于验证计划和验证执行结果，输出一份第一人称 explain 报告。
报告中需要包含：我设计了哪些断言、为什么这些断言能覆盖原始问题、验证最终是否通过、还有什么残余风险。
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


def _apply_unified_diff_to_text(original_text: str, diff_text: str) -> str:
    normalized_original = original_text.replace("\r\n", "\n")
    normalized_diff = diff_text.replace("\r\n", "\n")
    source_lines = normalized_original.split("\n")
    diff_lines = normalized_diff.split("\n")
    had_trailing_newline = normalized_original.endswith("\n")

    diff_index = next((index for index, line in enumerate(diff_lines) if line.startswith("@@")), -1)
    if diff_index == -1:
        raise RuntimeError("Repair model returned a diff without any hunks.")

    result: list[str] = []
    source_index = 0

    while diff_index < len(diff_lines):
        header = diff_lines[diff_index]
        if not header.startswith("@@"):
            diff_index += 1
            continue

        import re

        match = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", header)
        if not match:
            raise RuntimeError(f"Invalid diff hunk header: {header}")

        target_start = max(0, int(match.group(1)) - 1)
        while source_index < target_start and source_index < len(source_lines):
            result.append(source_lines[source_index])
            source_index += 1

        diff_index += 1
        while diff_index < len(diff_lines) and not diff_lines[diff_index].startswith("@@"):
            line = diff_lines[diff_index]
            if not line or line == "\\ No newline at end of file":
                diff_index += 1
                continue

            prefix = line[0]
            value = line[1:]
            current_source = source_lines[source_index] if source_index < len(source_lines) else None

            if prefix == " ":
                if current_source != value:
                    raise RuntimeError("Diff context did not match the original code.")
                result.append(value)
                source_index += 1
            elif prefix == "-":
                if current_source != value:
                    raise RuntimeError("Diff deletion did not match the original code.")
                source_index += 1
            elif prefix == "+":
                result.append(value)
            else:
                raise RuntimeError(f"Unsupported diff line prefix: {prefix}")

            diff_index += 1

    while source_index < len(source_lines):
        result.append(source_lines[source_index])
        source_index += 1

    next_text = "\n".join(result)
    if had_trailing_newline and not next_text.endswith("\n"):
        next_text += "\n"
    return next_text


def _collect_defined_symbol_names(code: str, filename: str) -> set[str]:
    tree = ast.parse(code, filename=filename)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _normalize_verification_report(
    report: dict[str, Any],
    *,
    existing_symbol_names: set[str],
    filename: str,
) -> dict[str, Any]:
    summary = str(report.get("summary", "")).strip()
    raw_code_lines = report.get("verification_code")
    if not isinstance(raw_code_lines, list):
        raise RuntimeError("Verification model did not return `verification_code` as a list.")

    verification_blocks: list[str] = []
    for item in raw_code_lines:
        if not isinstance(item, str):
            raise RuntimeError("Verification code lines must all be strings.")
        stripped_item = item.rstrip()
        if stripped_item:
            verification_blocks.append(stripped_item)

    if not verification_blocks:
        raise RuntimeError("Verification model returned an empty verification block.")

    verification_source = "\n\n".join(verification_blocks)
    try:
        tree = ast.parse(verification_source, filename=filename)
    except SyntaxError as exc:
        raise RuntimeError(
            f"Verification code is not valid Python: {exc.msg} at line {exc.lineno}"
        ) from exc

    sanitized_body: list[ast.stmt] = []
    sanitization_notes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sanitization_notes.append(
                f"Removed verification-only import at line {getattr(node, 'lineno', '?')}."
            )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sanitization_notes.append(
                f"Removed verification-only {node.__class__.__name__} `{getattr(node, 'name', '')}` at line {getattr(node, 'lineno', '?')}."
            )
            continue
        sanitized_body.append(node)

    sanitized_module = ast.Module(body=sanitized_body, type_ignores=getattr(tree, "type_ignores", []))
    ast.fix_missing_locations(sanitized_module)

    for node in ast.walk(sanitized_module):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in existing_symbol_names:
            raise RuntimeError(
                f"Verification code must not overwrite existing symbol `{node.id}`."
            )

    assert_count = sum(1 for node in ast.walk(sanitized_module) if isinstance(node, ast.Assert))
    if assert_count < 2:
        raise RuntimeError("Verification block must contain at least two assert statements.")

    sanitized_source = ast.unparse(sanitized_module).strip()
    if not sanitized_source:
        raise RuntimeError("Verification block became empty after sanitization.")

    verification_code = sanitized_source.splitlines()

    assertion_targets = report.get("assertion_targets")
    if isinstance(assertion_targets, list):
        normalized_targets = [str(item).strip() for item in assertion_targets if str(item).strip()]
    else:
        normalized_targets = []

    return {
        "summary": summary,
        "verification_code": verification_code,
        "assertion_targets": normalized_targets,
        "assert_count": assert_count,
        "sanitization_notes": sanitization_notes,
    }


def _build_verification_script(patched_code: str, verification_code: list[str]) -> str:
    verification_lines = ["def __autorepair_verify__():"]
    for line in verification_code:
        verification_lines.append(f"    {line}" if line else "")
    verification_lines.extend(["", "__autorepair_verify__()"])

    base_code = patched_code.rstrip("\n")
    verification_block = "\n".join(verification_lines)
    return f"{base_code}\n\n{verification_block}\n"


def _is_docstring_expr(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = getattr(node, "value", None)
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _is_module_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(test.ops[0], ast.Eq)
        and isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _build_verification_base_source(patched_code: str, filename: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(patched_code, filename=filename)
    except SyntaxError as exc:
        raise RuntimeError(
            f"Patched code is not valid Python: {exc.msg} at line {exc.lineno}"
        ) from exc

    allowed_types = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    if hasattr(ast, "TypeAlias"):
        allowed_types = (*allowed_types, ast.TypeAlias)

    kept_body: list[ast.stmt] = []
    skipped_descriptions: list[str] = []
    for node in tree.body:
        if _is_docstring_expr(node):
            kept_body.append(node)
            continue
        if isinstance(node, allowed_types):
            kept_body.append(node)
            continue
        if _is_module_main_guard(node):
            skipped_descriptions.append("Skipped `if __name__ == \"__main__\"` block during verification.")
            continue
        skipped_descriptions.append(
            f"Skipped top-level {node.__class__.__name__} at line {getattr(node, 'lineno', '?')} during verification."
        )

    module = ast.Module(body=kept_body, type_ignores=getattr(tree, "type_ignores", []))
    ast.fix_missing_locations(module)
    base_source = ast.unparse(module).strip()
    if not base_source:
        raise RuntimeError("Patched code did not contain reusable definitions for verification.")
    return f"{base_source}\n", skipped_descriptions


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

    _emit_stage(emit, "verify", "started")
    patched_code = _apply_unified_diff_to_text(request.code, git_diff)
    verify_tools = build_repair_tools(
        RepairToolContext(
            filename=request.filename,
            code=patched_code,
            runtime_report=runtime_report,
        )
    )
    verify_prompt = json.dumps(
        {
            "filename": request.filename,
            "runtime_report": runtime_report,
            "inspector_report": inspector_report,
            "planner_report": planner_report,
            "git_diff": git_diff,
            "patched_code": patched_code,
        },
        ensure_ascii=False,
        indent=2,
    )
    raw_verify_report = call_llm_for_json(
        prompt=verify_prompt,
        system_prompt=VERIFY_JSON_SYSTEM_PROMPT,
        model=request.model,
        isJson=True,
        tools=verify_tools,
        tool_event_handler=_build_tool_event_handler(emit, "verify"),
    )
    verification_base_source, skipped_verification_nodes = _build_verification_base_source(
        patched_code,
        request.filename,
    )
    existing_symbol_names = _collect_defined_symbol_names(
        verification_base_source,
        request.filename,
    )
    verify_report = _normalize_verification_report(
        raw_verify_report,
        existing_symbol_names=existing_symbol_names,
        filename=request.filename,
    )
    patched_execution = run_python_code_safely(
        verification_base_source,
        filename=request.filename,
        timeout_sec=request.timeout_sec,
    )
    verification_script = _build_verification_script(
        verification_base_source,
        verify_report["verification_code"],
    )
    verify_execution = run_python_code_safely(
        verification_script,
        filename=request.filename,
        timeout_sec=request.timeout_sec,
    )
    verify_passed = patched_execution.ok and verify_execution.ok
    verify_payload = {
        **verify_report,
        "patched_execution": patched_execution.to_dict(),
        "verification_execution": verify_execution.to_dict(),
        "verification_stdout": verify_execution.stdout,
        "verification_stderr": verify_execution.stderr,
        "verification_base_mode": "definition_only",
        "skipped_top_level_nodes": skipped_verification_nodes,
        "passed": verify_passed,
    }
    emit("verify_report", {"stage": "verify", "report": verify_payload})
    _stream_explain(
        stage="verify",
        prompt=json.dumps(
            {
                "verification_report": verify_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        system_prompt=VERIFY_EXPLAIN_SYSTEM_PROMPT,
        model=request.model,
        emit=emit,
    )
    _emit_stage(emit, "verify", "completed")

    emit(
        "result",
        {
            "status": "verified" if verify_passed else "verify_failed",
            "filename": request.filename,
            "git_diff": git_diff,
            "verification_passed": verify_passed,
            "message": (
                "Verification passed. Review the patch and decide whether to accept it."
                if verify_passed
                else "Verification failed. The generated patch did not satisfy the assertion checks."
            ),
        },
    )
