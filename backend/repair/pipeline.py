from __future__ import annotations

import ast
import difflib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.inspector.inspector_prompt import build_planner_prompt
from backend.llm import call_llm_for_json
from backend.llm.agent_tools import RepairToolContext, build_repair_tools
from backend.llm.store import resolve_model_selection
from backend.llm.telemetry import LLMCallContext
from backend.repair.languages import (
    default_entrypoint_for_language,
    get_language_spec,
    normalize_language,
)
from backend.repair.workspace import (
    ProjectFileInput,
    build_project_runtime_inspection_report,
    materialize_patched_workspace,
    normalize_project_path,
    prepare_project_workspace,
)
from backend.repair.sandbox import run_project_safely

MAX_CODE_CHARS = 100_000
MAX_INPUT_CHARS = 20_000
MAX_TIMEOUT_SEC = 30
MAX_TEST_CASES = 10
MAX_USER_PROMPT_CHARS = 8_000
PATCH_GENERATION_MAX_ATTEMPTS = 3
RETRYABLE_PATCH_GENERATION_ERROR_MARKERS: tuple[str, ...] = (
    "Unified diff is missing a `+++` header.",
    "Unified diff file section did not contain any hunks.",
    "Unified diff did not contain any file sections.",
    "Encountered a patch with neither old nor new path.",
    "Repair model did not return a valid git diff or a full replacement file.",
    "Repair model returned a diff without any hunks.",
    "Invalid diff hunk header:",
    "Diff context did not match the original code.",
    "Diff deletion did not match the original code.",
    "Unsupported diff line prefix:",
)

INSPECTOR_JSON_SYSTEM_PROMPT_TEMPLATE = """You are an expert bug inspection agent.
You will receive a structured local runtime report for an uploaded {language_name} file or project.
Return EXACTLY one valid JSON object and nothing else.

Requirements:
- Focus on the most likely root cause.
- Identify the most suspicious line or code region when possible.
- Include validation ideas for the final fix.
- Stay grounded in the runtime evidence and source code.
- Use the available function tools whenever you need precise local evidence from the uploaded file or runtime outputs.
- Do not return Markdown fences, headings, bullets, commentary, or prose before/after the JSON.
- If you are uncertain, keep the uncertainty inside JSON fields instead of switching to prose.
- If the issue is caused by path/package/classpath mismatch, explicitly name the expected path and the actual path.
- If the report contains a `test_cases` block, treat any case whose `passed` is false as a concrete failing specification, even if `execution.ok` is true. Describe the observable mismatch between `actual_stdout` and `expected_stdout` in `supporting_evidence`.
- If the report contains a `user_prompt` field, treat it as the user's natural-language task description (e.g. a problem statement or intent) and use it to disambiguate the expected behaviour when evidence is ambiguous.

Return JSON with this shape:
{{
  "summary": "1-2 sentence bug summary",
  "root_cause": "most likely root cause",
  "confidence": "high|medium|low",
  "primary_location": {{
    "path": "relative/path.ext or null",
    "line": 123,
    "symbol": "function/class/method name or null",
    "reason": "why this location is suspicious"
  }},
  "supporting_evidence": [
    "specific runtime or source evidence"
  ],
  "fix_strategy": [
    "minimal repair direction"
  ],
  "validation_ideas": [
    "how to verify the fix"
  ],
  "open_questions": [
    "remaining uncertainty or missing evidence"
  ]
}}

Schema rules:
- `primary_location.path`, `primary_location.line`, and `primary_location.symbol` may be null when unknown.
- `supporting_evidence`, `fix_strategy`, `validation_ideas`, and `open_questions` must always be JSON arrays of strings.
- Every string value must be plain text, not Markdown.
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

CODE_SYSTEM_PROMPT_TEMPLATE = """You are a senior {language_name} repair agent.
You will be given a failing {language_name} file or project, its runtime evidence, and a repair plan.
Return ONLY a unified git diff for the original project.

Rules:
- Modify only files inside the uploaded project.
- Keep the change minimal and correctness-focused.
- The final response must be a patch, never a rewritten full file.
- The final response must be a valid unified git diff that can touch one or multiple files.
- Every modified file must include `--- a/<path>`, `+++ b/<path>`, and at least one `@@` hunk.
- Do not wrap the diff in Markdown fences.
- Do not add explanations, comments, JSON, or prose before or after the diff.
- If you drafted code mentally, convert it into a unified diff before responding.
- Use the available function tools to inspect exact source windows before you finalize the diff.
- If a `test_cases` block is present in the runtime report, the patched program MUST, when given each `stdin`, print the corresponding `expected_stdout` on a clean run. Derive the fix from those cases rather than hard-coding their outputs.
- If a `user_prompt` field is present, treat it as the user's natural-language intent (e.g. a LeetCode problem statement) and keep the fix consistent with it.
"""

CODE_EXPLAIN_SYSTEM_PROMPT = """你是一个优秀的代码修复工程师。
请基于修复计划和最终 diff，输出一份第一人称 explain 报告。
报告中需要包含：我改了什么、为什么这样改、这份改动如何解决原始错误、还有哪些边界风险。
直接输出正文，不要添加开场白和结束语。"""

VERIFY_JSON_SYSTEM_PROMPT = """You are a Python repair verification agent.
You will receive the original failure evidence, the repair plan, the proposed git diff, and the patched Python project.
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
报告中需要包含：我设计了哪些验证条件、为什么这些验证条件能覆盖原始问题、验证最终是否通过、还有什么残余风险。
直接输出正文，不要添加开场白和结束语。"""


def _render_inspector_system_prompt(language: str) -> str:
    return INSPECTOR_JSON_SYSTEM_PROMPT_TEMPLATE.format(
        language_name=get_language_spec(language).display_name
    )


def _render_code_system_prompt(language: str) -> str:
    return CODE_SYSTEM_PROMPT_TEMPLATE.format(
        language_name=get_language_spec(language).display_name
    )


def _first_nonempty_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"```", "```json"}:
            continue
        return line
    return ""


def _extract_root_cause_hint(raw_response: str) -> str:
    heading_match = re.search(
        r"(?is)(?:^|\n)#+\s*root cause\s*\n(?P<body>.+?)(?:\n#+\s|\Z)",
        raw_response,
    )
    if heading_match:
        body = " ".join(part.strip() for part in heading_match.group("body").splitlines() if part.strip())
        if body:
            return body
    return _first_nonempty_line(raw_response)


def _build_fallback_inspector_report(
    runtime_report: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    failure = runtime_report.get("failure")
    failure_map = failure if isinstance(failure, dict) else {}
    primary_frame = failure_map.get("primary_frame")
    primary_frame_map = primary_frame if isinstance(primary_frame, dict) else {}
    source = runtime_report.get("source")
    source_map = source if isinstance(source, dict) else {}

    raw_response = str(getattr(error, "raw_response", "") or "").strip()
    exception_type = str(failure_map.get("exception_type") or "").strip()
    exception_message = str(failure_map.get("exception_message") or "").strip()
    focus_path = str(source_map.get("focus_path") or runtime_report.get("entrypoint") or "").strip()
    line_number = primary_frame_map.get("line_number")
    symbol_name = primary_frame_map.get("function")

    supporting_evidence: list[str] = []
    runtime_failure = ": ".join(part for part in (exception_type, exception_message) if part)
    if runtime_failure:
        supporting_evidence.append(f"Runtime failure: {runtime_failure}")
    if focus_path:
        if isinstance(line_number, int):
            supporting_evidence.append(f"Primary failing location from runtime report: {focus_path}:{line_number}")
        else:
            supporting_evidence.append(f"Focused source path from runtime report: {focus_path}")
    if raw_response:
        supporting_evidence.append(
            "Inspector model returned non-JSON prose; the raw model response is preserved in `raw_inspector_response`."
        )

    validation_ideas = [
        "Re-run the original failing command and confirm the previous runtime error no longer occurs.",
    ]
    if exception_type:
        validation_ideas.append(f"Confirm `{exception_type}` is no longer raised after the patch.")

    root_cause_hint = _extract_root_cause_hint(raw_response)
    if not root_cause_hint:
        root_cause_hint = (
            "Structured inspector parsing failed. Use the preserved raw inspector response and runtime evidence to infer the root cause."
        )

    return {
        "summary": (
            "Inspector model returned non-JSON output. A fallback structured report was created so planning and repair can continue."
        ),
        "root_cause": root_cause_hint,
        "confidence": "low",
        "primary_location": {
            "path": focus_path or None,
            "line": line_number if isinstance(line_number, int) else None,
            "symbol": str(symbol_name).strip() if isinstance(symbol_name, str) and symbol_name.strip() else None,
            "reason": (
                "Derived from runtime failure evidence because the inspector response could not be parsed as JSON."
            ),
        },
        "supporting_evidence": supporting_evidence or ["See `runtime_report` for the captured execution evidence."],
        "fix_strategy": [
            "Use the runtime report and any reliable details from the raw inspector response to form a minimal repair plan.",
            "Verify the suspected root cause against exact source context before editing code.",
        ],
        "validation_ideas": validation_ideas,
        "open_questions": [
            "Some structured inspector fields may be incomplete because the model did not follow the JSON contract."
        ],
        "status": "fallback_non_json_response",
        "parse_error": str(error),
        "raw_inspector_response": raw_response,
    }

EventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PatchCandidateProfile:
    key: str
    label: str
    instructions: str


@dataclass
class PatchCandidateResult:
    index: int
    key: str
    label: str
    instructions: str
    raw_output: str = ""
    git_diff: str = ""
    modified_files: list[str] = field(default_factory=list)
    changed_file_count: int = 0
    added_lines: int = 0
    removed_lines: int = 0
    diff_line_count: int = 0
    patched_files: dict[str, str] = field(default_factory=dict)
    patched_execution: Any | None = None
    verification_execution: Any | None = None
    verification_stdout: str = ""
    verification_stderr: str = ""
    verification_report: dict[str, Any] | None = None
    verification_base_mode: str | None = None
    skipped_top_level_nodes: list[str] = field(default_factory=list)
    verify_passed: bool = False
    score: int = 0
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    rank: int = 0


PATCH_GENERATION_PROFILE = PatchCandidateProfile(
    key="repair_agent",
    label="Repair Agent",
    instructions=(
        "Generate one minimal, correctness-focused patch that directly fixes the observed failure. "
        "Avoid unnecessary refactors and keep the diff grounded in the original source context."
    ),
)


@dataclass(frozen=True)
class RepairTestCase:
    """A single user-provided I/O test case.

    The case is "passed" when the patched program exits cleanly AND its
    ``stdout`` matches ``expected_stdout`` after trailing-whitespace
    normalisation. ``name`` is optional; when absent we synthesise
    ``Case #N`` in the UI / prompts.
    """

    stdin: str
    expected_stdout: str
    name: str = ""


@dataclass(frozen=True)
class RepairRequest:
    code: str | None = None
    filename: str | None = "main.py"
    language: str = "python"
    input_text: str | None = None
    timeout_sec: int = 5
    model: str = ""
    project_files: tuple[ProjectFileInput, ...] = ()
    project_zip_base64: str | None = None
    github_repo_url: str | None = None
    github_ref: str | None = None
    project_subdir: str | None = None
    user_prompt: str | None = None
    test_cases: tuple[RepairTestCase, ...] = ()

    @property
    def source_type(self) -> str:
        if self.github_repo_url:
            return "github"
        if self.project_zip_base64:
            return "zip"
        if self.project_files:
            return "project_files"
        return "single_file"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RepairRequest":
        code = payload.get("code")
        normalized_code: str | None = None
        if code is not None:
            if not isinstance(code, str) or not code.strip():
                raise ValueError("`code` must be a non-empty string when provided.")
            if len(code) > MAX_CODE_CHARS:
                raise ValueError(f"`code` must be at most {MAX_CODE_CHARS} characters.")
            normalized_code = code

        language = normalize_language(payload.get("language", "python"))
        language_spec = get_language_spec(language)

        raw_filename = payload.get("filename")
        filename: str | None = None
        if raw_filename is not None:
            if not isinstance(raw_filename, str) or not raw_filename.strip():
                raise ValueError("`filename` must be a non-empty string when provided.")
            filename = normalize_project_path(
                raw_filename,
                required_suffixes=language_spec.source_extensions,
            )
        elif normalized_code is not None:
            filename = default_entrypoint_for_language(language)

        timeout_sec = payload.get("timeout_sec", 5)
        if not isinstance(timeout_sec, int):
            raise ValueError("`timeout_sec` must be an integer.")
        if timeout_sec < 1 or timeout_sec > MAX_TIMEOUT_SEC:
            raise ValueError(f"`timeout_sec` must be between 1 and {MAX_TIMEOUT_SEC}.")

        input_text = payload.get("input_text")
        normalized_input_text: str | None = None
        if input_text is not None:
            if not isinstance(input_text, str):
                raise ValueError("`input_text` must be a string when provided.")
            if len(input_text) > MAX_INPUT_CHARS:
                raise ValueError(f"`input_text` must be at most {MAX_INPUT_CHARS} characters.")
            normalized_input_text = input_text

        raw_model = payload.get("model")
        if raw_model is not None and (not isinstance(raw_model, str) or not raw_model.strip()):
            raise ValueError("`model` must be a non-empty string when provided.")
        resolved_model = resolve_model_selection(
            raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else None,
            purpose="repair",
        )

        raw_project_files = payload.get("project_files")
        project_files: list[ProjectFileInput] = []
        if raw_project_files is not None:
            if not isinstance(raw_project_files, list) or not raw_project_files:
                raise ValueError("`project_files` must be a non-empty array when provided.")
            for index, item in enumerate(raw_project_files):
                if not isinstance(item, dict):
                    raise ValueError(f"`project_files[{index}]` must be an object.")
                raw_path = item.get("path")
                content = item.get("content")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(f"`project_files[{index}].path` must be a non-empty string.")
                if not isinstance(content, str):
                    raise ValueError(f"`project_files[{index}].content` must be a string.")
                project_files.append(
                    ProjectFileInput(
                        path=normalize_project_path(raw_path),
                        content=content,
                    )
                )

        project_zip_base64 = payload.get("project_zip_base64")
        if project_zip_base64 is not None and (
            not isinstance(project_zip_base64, str) or not project_zip_base64.strip()
        ):
            raise ValueError("`project_zip_base64` must be a non-empty base64 string when provided.")

        github_repo_url = payload.get("github_repo_url")
        if github_repo_url is not None and (
            not isinstance(github_repo_url, str) or not github_repo_url.strip()
        ):
            raise ValueError("`github_repo_url` must be a non-empty string when provided.")

        github_ref = payload.get("github_ref")
        if github_ref is not None and not isinstance(github_ref, str):
            raise ValueError("`github_ref` must be a string when provided.")

        project_subdir = payload.get("project_subdir")
        if project_subdir is not None:
            if not isinstance(project_subdir, str) or not project_subdir.strip():
                raise ValueError("`project_subdir` must be a non-empty string when provided.")
            project_subdir = normalize_project_path(project_subdir)

        raw_user_prompt = payload.get("user_prompt")
        normalized_user_prompt: str | None = None
        if raw_user_prompt is not None:
            if not isinstance(raw_user_prompt, str):
                raise ValueError("`user_prompt` must be a string when provided.")
            trimmed_prompt = raw_user_prompt.strip()
            if trimmed_prompt:
                if len(trimmed_prompt) > MAX_USER_PROMPT_CHARS:
                    raise ValueError(
                        f"`user_prompt` must be at most {MAX_USER_PROMPT_CHARS} characters."
                    )
                normalized_user_prompt = trimmed_prompt

        raw_test_cases = payload.get("test_cases")
        test_cases: list[RepairTestCase] = []
        if raw_test_cases is not None:
            if not isinstance(raw_test_cases, list):
                raise ValueError("`test_cases` must be an array when provided.")
            if len(raw_test_cases) > MAX_TEST_CASES:
                raise ValueError(f"`test_cases` can have at most {MAX_TEST_CASES} entries.")
            for case_index, case_payload in enumerate(raw_test_cases):
                if not isinstance(case_payload, dict):
                    raise ValueError(f"`test_cases[{case_index}]` must be an object.")
                raw_stdin = case_payload.get("stdin")
                raw_expected = case_payload.get("expected_stdout")
                raw_name = case_payload.get("name")
                if raw_stdin is None:
                    raw_stdin = ""
                if raw_expected is None:
                    raw_expected = ""
                if not isinstance(raw_stdin, str):
                    raise ValueError(f"`test_cases[{case_index}].stdin` must be a string.")
                if not isinstance(raw_expected, str):
                    raise ValueError(
                        f"`test_cases[{case_index}].expected_stdout` must be a string."
                    )
                if len(raw_stdin) > MAX_INPUT_CHARS:
                    raise ValueError(
                        f"`test_cases[{case_index}].stdin` must be at most "
                        f"{MAX_INPUT_CHARS} characters."
                    )
                if len(raw_expected) > MAX_INPUT_CHARS:
                    raise ValueError(
                        f"`test_cases[{case_index}].expected_stdout` must be at most "
                        f"{MAX_INPUT_CHARS} characters."
                    )
                if raw_name is not None and not isinstance(raw_name, str):
                    raise ValueError(f"`test_cases[{case_index}].name` must be a string.")
                name_value = (raw_name or "").strip()[:80]
                # Skip empty rows where the user typed nothing at all – keeps the
                # accidental "click-and-forget" row from triggering the repair loop.
                if not raw_stdin and not raw_expected and not name_value:
                    continue
                test_cases.append(
                    RepairTestCase(
                        stdin=raw_stdin,
                        expected_stdout=raw_expected,
                        name=name_value,
                    )
                )

        source_count = sum(
            1
            for present in (
                normalized_code is not None,
                bool(project_files),
                bool(project_zip_base64),
                bool(github_repo_url),
            )
            if present
        )
        if source_count != 1:
            raise ValueError(
                "Exactly one of `code`, `project_files`, `project_zip_base64`, or `github_repo_url` must be provided."
            )

        return cls(
            code=normalized_code,
            filename=filename,
            language=language,
            input_text=normalized_input_text,
            timeout_sec=timeout_sec,
            model=resolved_model.model_key,
            project_files=tuple(project_files),
            project_zip_base64=project_zip_base64.strip() if isinstance(project_zip_base64, str) else None,
            github_repo_url=github_repo_url.strip() if isinstance(github_repo_url, str) else None,
            github_ref=github_ref.strip() if isinstance(github_ref, str) and github_ref.strip() else None,
            project_subdir=project_subdir,
            user_prompt=normalized_user_prompt,
            test_cases=tuple(test_cases),
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


def _extract_first_fenced_block(text: str) -> tuple[str, str] | None:
    match = re.search(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```", text)
    if not match:
        return None
    return match.group(1).strip().lower(), match.group(2).strip()


def _looks_like_unified_diff(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    has_headers = (
        "diff --git " in stripped
        or ("--- " in stripped and "+++ " in stripped)
    )
    has_hunk = "@@ " in stripped or "\n@@" in stripped
    return has_headers and has_hunk or has_hunk


def _extract_source_code_candidate(text: str, *, language: str) -> str | None:
    language_spec = get_language_spec(language)
    fenced = _extract_first_fenced_block(text)
    if fenced is not None:
        label, body = fenced
        if label in {"diff", "patch"}:
            return None
        if not label or label in language_spec.code_fence_labels:
            return body

    stripped = text.strip()
    if not stripped or _looks_like_unified_diff(stripped):
        return None
    if language == "python":
        try:
            ast.parse(stripped)
        except SyntaxError:
            return None
    return stripped


def _render_unified_diff(original_text: str, patched_text: str, filename: str) -> str:
    original_lines = original_text.splitlines()
    patched_lines = patched_text.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""
    return "\n".join([f"diff --git a/{filename} b/{filename}", *diff_lines]).strip()


def _normalize_patch_path(raw_path: str) -> str | None:
    value = raw_path.strip()
    if value == "/dev/null":
        return None
    if value.startswith("a/") or value.startswith("b/"):
        return value[2:]
    return value


def _parse_unified_diff_files(diff_text: str) -> list[dict[str, Any]]:
    lines = _normalize_diff(diff_text).splitlines()
    patches: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        if lines[index].startswith("diff --git "):
            index += 1
            while index < len(lines) and not lines[index].startswith("--- "):
                index += 1
        elif not lines[index].startswith("--- "):
            index += 1
            continue

        if index >= len(lines) or not lines[index].startswith("--- "):
            break
        old_path = _normalize_patch_path(lines[index][4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise RuntimeError("Unified diff is missing a `+++` header.")
        new_path = _normalize_patch_path(lines[index][4:])
        index += 1

        body_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("diff --git ") and not lines[index].startswith("--- "):
            body_lines.append(lines[index])
            index += 1
        if not any(line.startswith("@@") for line in body_lines):
            raise RuntimeError("Unified diff file section did not contain any hunks.")

        patches.append(
            {
                "old_path": old_path,
                "new_path": new_path,
                "body_lines": body_lines,
            }
        )

    return patches


def _render_project_diff(
    original_files: dict[str, str],
    patched_files: dict[str, str],
) -> str:
    diff_blocks: list[str] = []
    changed_paths = sorted(set(original_files.keys()) | set(patched_files.keys()))
    for path in changed_paths:
        original_text = original_files.get(path)
        patched_text = patched_files.get(path)
        if original_text == patched_text:
            continue

        fromfile = f"a/{path}" if original_text is not None else "/dev/null"
        tofile = f"b/{path}" if patched_text is not None else "/dev/null"
        diff_lines = list(
            difflib.unified_diff(
                [] if original_text is None else original_text.splitlines(),
                [] if patched_text is None else patched_text.splitlines(),
                fromfile=fromfile,
                tofile=tofile,
                lineterm="",
            )
        )
        if diff_lines:
            diff_blocks.extend([f"diff --git a/{path} b/{path}", *diff_lines])
    return "\n".join(diff_blocks).strip()


def _apply_unified_diff_to_project(
    original_files: dict[str, str],
    diff_text: str,
    *,
    default_path: str | None = None,
) -> dict[str, str]:
    normalized = _normalize_diff(diff_text)
    if default_path is not None and "@@" in normalized and "--- " not in normalized and "diff --git " not in normalized:
        updated = dict(original_files)
        updated[default_path] = _apply_unified_diff_to_text(original_files.get(default_path, ""), normalized)
        return updated

    patches = _parse_unified_diff_files(normalized)
    if not patches:
        raise RuntimeError("Unified diff did not contain any file sections.")

    updated_files = dict(original_files)
    for patch in patches:
        old_path = patch["old_path"]
        new_path = patch["new_path"]
        if old_path is None and new_path is None:
            raise RuntimeError("Encountered a patch with neither old nor new path.")

        original_text = updated_files.get(old_path, "") if old_path is not None else ""
        patch_text = "\n".join(
            [
                f"--- {'/dev/null' if old_path is None else f'a/{old_path}'}",
                f"+++ {'/dev/null' if new_path is None else f'b/{new_path}'}",
                *patch["body_lines"],
            ]
        )
        patched_text = _apply_unified_diff_to_text(original_text, patch_text)

        if old_path is not None and new_path is not None and old_path != new_path and old_path in updated_files:
            del updated_files[old_path]

        if new_path is None:
            if old_path is not None:
                updated_files.pop(old_path, None)
        else:
            updated_files[new_path] = patched_text

    return updated_files


def _coerce_model_output_to_diff(
    raw_output: str,
    *,
    original_files: dict[str, str],
    entrypoint: str,
    language: str,
) -> str:
    cleaned = _normalize_diff(raw_output)
    if not cleaned:
        return ""

    diff_candidate = cleaned
    fenced = _extract_first_fenced_block(raw_output)
    if fenced is not None:
        label, body = fenced
        if label in {"diff", "patch"} and body:
            diff_candidate = body

    if _looks_like_unified_diff(diff_candidate):
        patched_files = _apply_unified_diff_to_project(
            original_files,
            diff_candidate,
            default_path=entrypoint,
        )
        canonical_diff = _render_project_diff(original_files, patched_files)
        if canonical_diff:
            return canonical_diff

    source_candidate = _extract_source_code_candidate(raw_output, language=language)
    if source_candidate is not None and len(original_files) == 1:
        only_path = next(iter(original_files.keys()))
        canonical_diff = _render_unified_diff(original_files[only_path], source_candidate, only_path)
        if canonical_diff:
            return canonical_diff

    if _looks_like_unified_diff(cleaned):
        patched_files = _apply_unified_diff_to_project(
            original_files,
            cleaned,
            default_path=entrypoint,
        )
        canonical_diff = _render_project_diff(original_files, patched_files)
        if canonical_diff:
            return canonical_diff

    raise RuntimeError("Repair model did not return a valid git diff or a full replacement file.")


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
    """Collect the names of top-level symbols that verification code must not overwrite.

    Only function and class definitions are "protected": overriding one of them in the
    verification block would silently change the subject under test and make the asserts
    meaningless. Plain data variables (e.g. demo inputs like ``scores = [...]``) are
    intentionally *not* protected - it is perfectly legitimate for the verification block
    to rebind them with its own test inputs.
    """
    tree = ast.parse(code, filename=filename)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
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
                f"Verification code must not redefine function or class `{node.id}` from the patched source."
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


def _base_line_count(patched_code: str) -> int:
    """Number of source lines that precede the verification block in the script
    produced by :func:`_build_verification_script`.
    """
    stripped = patched_code.rstrip("\n")
    if not stripped:
        return 0
    return stripped.count("\n") + 1


_TRACEBACK_VERIFY_FRAME_RE = re.compile(
    r'File "[^"]*", line (\d+), in __autorepair_verify__'
)


def _detect_failed_verification_line(
    verification_execution: Any,
    *,
    base_line_count: int,
    verification_code_len: int,
) -> int | None:
    """Best-effort: parse a failed verification execution's stderr and return
    the 0-based index into ``verification_code`` of the line that raised.

    Returns ``None`` when the execution was not a failure, when the error did
    not occur inside ``__autorepair_verify__``, or when the traceback line
    cannot be mapped back into the supplied verification code.
    """
    if verification_execution is None:
        return None
    if getattr(verification_execution, "ok", False):
        return None
    stderr = getattr(verification_execution, "stderr", "") or ""
    if not stderr:
        return None

    matches = _TRACEBACK_VERIFY_FRAME_RE.findall(stderr)
    if not matches:
        return None
    try:
        script_line = int(matches[-1])
    except ValueError:
        return None

    # Line layout produced by ``_build_verification_script``:
    #   lines 1..N               -> base (patched) source
    #   line N + 1               -> blank ("\n\n")
    #   line N + 2               -> "def __autorepair_verify__():"
    #   line N + 3 + i           -> verification_code[i]
    idx = script_line - (base_line_count + 3)
    if 0 <= idx < verification_code_len:
        return idx
    return None


def _is_assert_line(code_line: str) -> bool:
    stripped = (code_line or "").strip()
    if not stripped:
        return False
    if stripped == "assert":
        return True
    return stripped.startswith("assert ") or stripped.startswith("assert(")


def _compute_assertion_statuses(
    *,
    verification_code: list[str],
    assertion_targets: list[str],
    verify_passed: bool,
    failed_line_index: int | None,
) -> list[dict[str, Any]]:
    """Pair each assertion with its natural-language target and a status.

    - ``passed`` : the assertion ran and did not raise.
    - ``failed`` : the assertion is the one that raised (AssertionError).
    - ``skipped``: a prior assertion failed / an earlier line raised, so this
      assertion never executed.
    - ``unknown``: verification failed for a non-assertion reason (NameError,
      etc.) and we cannot tell which assertion status to report.
    """
    entries: list[dict[str, Any]] = []
    assertion_ordinal = 0
    for code_index, line in enumerate(verification_code):
        if not _is_assert_line(line):
            continue
        target = (
            assertion_targets[assertion_ordinal]
            if assertion_ordinal < len(assertion_targets)
            else ""
        )

        if verify_passed:
            status = "passed"
        elif failed_line_index is None:
            status = "unknown"
        elif code_index < failed_line_index:
            status = "passed"
        elif code_index == failed_line_index:
            status = "failed"
        else:
            status = "skipped"

        entries.append(
            {
                "index": assertion_ordinal + 1,
                "code": line.strip(),
                "target": target,
                "status": status,
            }
        )
        assertion_ordinal += 1
    return entries


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


def _normalize_output_for_match(text: str) -> str:
    """Normalise a stdout blob before comparing it against the expected output.

    We do not require byte-for-byte equality because that is too strict for
    LeetCode-style problems: the expected output is usually handwritten and
    the buggy program is often dumping the answer via ``print``. The rules:

    * strip trailing whitespace on every line (catches stray spaces)
    * collapse every consecutive run of ``\\n`` into exactly one ``\\n``
      (catches stray blank lines the buggy program sometimes prints)
    * strip leading/trailing whitespace from the whole blob
    """

    if not text:
        return ""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    collapsed: list[str] = []
    blank_pending = False
    for line in lines:
        if line == "":
            blank_pending = True
            continue
        if blank_pending and collapsed:
            collapsed.append("")
        collapsed.append(line)
        blank_pending = False
    return "\n".join(collapsed).strip()


def _case_display_name(index: int, case: "RepairTestCase") -> str:
    if case.name:
        return case.name
    return f"Case #{index + 1}"


def _run_one_test_case(
    *,
    workspace: Any,
    file_map: dict[str, str] | None,
    case: "RepairTestCase",
    language: str,
    timeout_sec: int,
    workspace_root: Path | None = None,
    index: int,
) -> dict[str, Any]:
    """Run a single user-provided test case against either the current
    workspace-on-disk or a materialised patched workspace.

    ``file_map`` is provided in the patched-rerun path so we can materialise
    the patched files into a fresh tempdir; otherwise we run against the
    workspace root already on disk.
    """

    display_name = _case_display_name(index, case)
    if file_map is not None:
        with materialize_patched_workspace(
            workspace.root_dir,
            file_map,
            set(workspace.file_map.keys()),
        ) as rerun_root:
            execution = run_project_safely(
                rerun_root,
                filename=workspace.entrypoint,
                language=language,
                input_text=case.stdin,
                timeout_sec=timeout_sec,
            )
    else:
        target_root = workspace_root or workspace.root_dir
        execution = run_project_safely(
            target_root,
            filename=workspace.entrypoint,
            language=language,
            input_text=case.stdin,
            timeout_sec=timeout_sec,
        )

    actual = execution.stdout
    expected = case.expected_stdout
    expected_provided = bool(expected.strip())
    matched_output = _normalize_output_for_match(actual) == _normalize_output_for_match(expected)
    passed = execution.ok and (matched_output if expected_provided else True)

    return {
        "index": index,
        "name": display_name,
        "stdin": case.stdin,
        "expected_stdout": expected,
        "expected_provided": expected_provided,
        "stdout": actual,
        "stderr": execution.stderr,
        "returncode": execution.returncode,
        "timed_out": execution.timed_out,
        "duration_sec": execution.duration_sec,
        "runtime_ok": execution.ok,
        "matched_output": matched_output if expected_provided else None,
        "passed": passed,
    }


def _run_all_test_cases(
    *,
    workspace: Any,
    cases: tuple["RepairTestCase", ...],
    language: str,
    timeout_sec: int,
    file_map: dict[str, str] | None = None,
    workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        results.append(
            _run_one_test_case(
                workspace=workspace,
                file_map=file_map,
                case=case,
                language=language,
                timeout_sec=timeout_sec,
                workspace_root=workspace_root,
                index=index,
            )
        )
    return results


def _summarise_test_case_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "provided": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "all_passed": True,
        }
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    return {
        "provided": True,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
    }


def _build_runtime_only_verification_report(
    *,
    language: str,
    patched_execution: Any,
    modified_files: list[str],
) -> dict[str, Any]:
    language_name = get_language_spec(language).display_name
    passed = bool(patched_execution is not None and patched_execution.ok)
    if passed:
        summary = f"Patched {language_name} project re-ran successfully with a clean exit status."
    elif patched_execution is not None and patched_execution.timed_out:
        summary = f"Patched {language_name} project timed out during verification rerun."
    else:
        summary = f"Patched {language_name} project still failed during verification rerun."

    return {
        "summary": summary,
        "verification_strategy": "runtime_rerun_only",
        "assertion_targets": [
            "program exits successfully after applying the patch",
            "stderr no longer contains the original failure signal",
        ],
        "assert_count": 0,
        "modified_files": modified_files,
        "passed": passed,
    }


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
    audit_context: LLMCallContext | None = None,
) -> str:
    _emit_stage(emit, stage, "explaining")

    def on_chunk(chunk: str) -> None:
        emit("explain_chunk", {"stage": stage, "chunk": chunk})

    def on_reasoning_chunk(chunk: str) -> None:
        emit("stage_reasoning_chunk", {"stage": stage, "chunk": chunk})

    explain_text = call_llm_for_json(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        isJson=False,
        stream=True,
        stream_handler=on_chunk,
        reasoning_handler=on_reasoning_chunk,
        audit_context=audit_context,
    )
    emit("explain_done", {"stage": stage, "text": explain_text})
    return explain_text


def _stream_explain_async(
    *,
    stage: str,
    prompt: str,
    system_prompt: str,
    model: str,
    emit: EventEmitter,
    audit_context: LLMCallContext | None = None,
) -> threading.Thread:
    """Run `_stream_explain` in a background thread so the next stage can start sooner.

    The explain call produces a user-facing first-person summary that is not consumed by any
    downstream stage, so pipelining it with the next stage's primary LLM call substantially
    reduces the end-to-end wait time without changing functional behavior.

    The worker always emits `stage/completed` for the stage when done, even on failure, so the
    UI does not get stuck in an `explaining` state.
    """

    def worker() -> None:
        try:
            _stream_explain(
                stage=stage,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                emit=emit,
                audit_context=audit_context,
            )
        except Exception as exc:
            emit(
                "explain_chunk",
                {"stage": stage, "chunk": f"\n[explain skipped: {exc}]"},
            )
        finally:
            _emit_stage(emit, stage, "completed")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def _build_tool_event_handler(
    emit: EventEmitter,
    stage: str,
    *,
    candidate_label: str | None = None,
) -> Callable[[str, dict[str, Any]], None]:
    def on_tool_event(status: str, payload: dict[str, Any]) -> None:
        next_payload = dict(payload)
        if candidate_label:
            tool_name = str(next_payload.get("tool_name") or "tool")
            next_payload["tool_name"] = f"{candidate_label}: {tool_name}"
        emit(
            "tool_event",
            {
                "stage": stage,
                "status": status,
                **next_payload,
            },
        )

    return on_tool_event


def _build_reasoning_event_handler(
    emit: EventEmitter,
    stage: str,
    *,
    candidate_label: str | None = None,
) -> Callable[[str], None]:
    did_emit_label = False

    def on_reasoning_chunk(chunk: str) -> None:
        nonlocal did_emit_label
        next_chunk = chunk
        if candidate_label and not did_emit_label:
            next_chunk = f"[{candidate_label}]\n{chunk}"
            did_emit_label = True
        emit(
            "stage_reasoning_chunk",
            {
                "stage": stage,
                "chunk": next_chunk,
            },
        )

    return on_reasoning_chunk


def _count_diff_lines(git_diff: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in git_diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _summarize_candidate(candidate: PatchCandidateResult, *, include_details: bool = False) -> dict[str, Any]:
    summary = {
        "rank": candidate.rank,
        "candidate_key": candidate.key,
        "candidate_label": candidate.label,
        "strategy": candidate.instructions,
        "score": candidate.score,
        "verify_passed": candidate.verify_passed,
        "changed_file_count": candidate.changed_file_count,
        "modified_files": candidate.modified_files,
        "added_lines": candidate.added_lines,
        "removed_lines": candidate.removed_lines,
        "diff_line_count": candidate.diff_line_count,
        "verification_summary": (
            candidate.verification_report.get("summary")
            if isinstance(candidate.verification_report, dict)
            else None
        ),
        "assert_count": (
            int(candidate.verification_report.get("assert_count") or 0)
            if isinstance(candidate.verification_report, dict)
            else 0
        ),
        "error_message": candidate.error_message,
    }
    if include_details:
        summary["git_diff"] = candidate.git_diff
        summary["verification_report"] = candidate.verification_report
        summary["score_breakdown"] = candidate.score_breakdown
        summary["patched_execution"] = (
            candidate.patched_execution.to_dict() if candidate.patched_execution is not None else None
        )
        summary["verification_execution"] = (
            candidate.verification_execution.to_dict()
            if candidate.verification_execution is not None
            else None
        )
        summary["verification_stdout"] = candidate.verification_stdout
        summary["verification_stderr"] = candidate.verification_stderr
        summary["verification_base_mode"] = candidate.verification_base_mode
        summary["skipped_top_level_nodes"] = candidate.skipped_top_level_nodes
    return summary


def _build_candidate_generation_report(
    candidates: list[PatchCandidateResult],
    *,
    provisional_leader: PatchCandidateResult | None,
) -> str:
    single_candidate_mode = len(candidates) == 1
    payload = {
        "collaboration_mode": (
            "single_patch_generation"
            if single_candidate_mode
            else "multi_candidate_patch_committee"
        ),
        "selection_policy": (
            "A single repair agent generated one patch candidate. "
            "If the diff does not apply cleanly to the original project, generation is retried before verification."
            if single_candidate_mode
            else (
                "Specialized coder agents generated multiple patch candidates. "
                "A later verification stage will execute, score, rank, and auto-select the best candidate."
            )
        ),
        "candidate_count": len(candidates),
        "provisional_leader": (
            {
                "candidate_key": provisional_leader.key,
                "candidate_label": provisional_leader.label,
            }
            if provisional_leader is not None
            else None
        ),
        "candidates": [
            {
                "candidate_key": candidate.key,
                "candidate_label": candidate.label,
                "strategy": candidate.instructions,
                "generated_diff": bool(candidate.git_diff),
                "changed_file_count": candidate.changed_file_count,
                "modified_files": candidate.modified_files,
                "added_lines": candidate.added_lines,
                "removed_lines": candidate.removed_lines,
                "error_message": candidate.error_message,
            }
            for candidate in candidates
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _score_candidate_with_breakdown(
    candidate: PatchCandidateResult,
) -> tuple[int, list[dict[str, Any]]]:
    """Compute the candidate's score together with a per-rule breakdown.

    Each breakdown entry captures (`code`, `delta`, `note`) so the UI can render
    a transparent "why did we score it this way" table without re-implementing
    the same heuristics on the frontend.
    """

    breakdown: list[dict[str, Any]] = []
    score = 0

    def add(code: str, delta: int, note: str = "") -> None:
        nonlocal score
        if delta == 0:
            return
        entry: dict[str, Any] = {"code": code, "delta": delta}
        if note:
            entry["note"] = note
        breakdown.append(entry)
        score += delta

    if candidate.git_diff:
        add("diff_generated", 20, "Patch produced a valid unified diff")
    if candidate.patched_execution is not None:
        if candidate.patched_execution.ok:
            add(
                "patched_runtime_ok",
                40,
                "Patched project ran to completion",
            )
        elif candidate.patched_execution.timed_out:
            add(
                "patched_runtime_timeout",
                -18,
                "Patched project timed out during rerun",
            )
        else:
            add(
                "patched_runtime_failed",
                -8,
                "Patched project still exited with a non-zero status",
            )
    if candidate.verification_report is not None:
        assert_count = int(candidate.verification_report.get("assert_count") or 0)
        assert_bonus = min(assert_count, 6)
        if assert_bonus > 0:
            add(
                "assertion_coverage",
                assert_bonus,
                f"{assert_count} assertion{'s' if assert_count != 1 else ''} "
                f"(counted up to 6)",
            )
    if candidate.verification_execution is not None:
        if candidate.verification_execution.ok:
            add(
                "verification_ok",
                34,
                "Verification block (asserts + rerun) passed cleanly",
            )
        elif candidate.verification_execution.timed_out:
            add(
                "verification_timeout",
                -18,
                "Verification execution timed out",
            )
        else:
            add(
                "verification_failed",
                -10,
                "Verification execution raised or exited non-zero",
            )
    if candidate.verify_passed:
        add(
            "verify_passed",
            25,
            "Both the rerun and the assertions passed",
        )

    files_penalty = max(0, candidate.changed_file_count - 1) * 3
    if files_penalty > 0:
        add(
            "extra_files_penalty",
            -files_penalty,
            f"Patch touched {candidate.changed_file_count} files (−3 per extra file)",
        )
    lines_penalty = min(candidate.diff_line_count, 120) // 12
    if lines_penalty > 0:
        add(
            "diff_size_penalty",
            -lines_penalty,
            f"{candidate.diff_line_count} changed lines (−1 per 12 lines, capped)",
        )
    if candidate.error_message:
        truncated = (candidate.error_message or "").strip().replace("\n", " ")
        if len(truncated) > 140:
            truncated = truncated[:137] + "..."
        add("error_reported", -20, truncated or "An error was reported")

    return score, breakdown


def _score_candidate(candidate: PatchCandidateResult) -> int:
    score, breakdown = _score_candidate_with_breakdown(candidate)
    candidate.score_breakdown = breakdown
    return score


def _rank_candidates(candidates: list[PatchCandidateResult]) -> list[PatchCandidateResult]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.verify_passed,
            bool(candidate.verification_execution and candidate.verification_execution.ok),
            bool(candidate.patched_execution and candidate.patched_execution.ok),
            candidate.score,
            -candidate.changed_file_count,
            -candidate.diff_line_count,
            -candidate.index,
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank
    return ranked


def _build_candidate_failure_message(candidates: list[PatchCandidateResult]) -> str:
    details: list[str] = []
    for candidate in candidates:
        reason = candidate.error_message or "did not return a valid unified diff"
        details.append(f"{candidate.label}: {reason}")
    if len(candidates) == 1:
        if not details:
            return "The repair agent failed to produce a valid patch."
        return "The repair agent failed to produce a valid patch. " + " | ".join(details[:3])
    if not details:
        return "All coder agents failed to produce a valid patch candidate."
    return "All coder agents failed to produce a valid patch candidate. " + " | ".join(details[:6])


def _reset_candidate_patch_state(candidate: PatchCandidateResult) -> None:
    candidate.git_diff = ""
    candidate.modified_files = []
    candidate.changed_file_count = 0
    candidate.added_lines = 0
    candidate.removed_lines = 0
    candidate.diff_line_count = 0
    candidate.patched_files = {}


def _populate_candidate_patch_state(
    candidate: PatchCandidateResult,
    *,
    original_files: dict[str, str],
    entrypoint: str,
) -> None:
    candidate.patched_files = _apply_unified_diff_to_project(
        original_files,
        candidate.git_diff,
        default_path=entrypoint,
    )
    candidate.modified_files = sorted(
        path
        for path in set(original_files.keys()) | set(candidate.patched_files.keys())
        if original_files.get(path) != candidate.patched_files.get(path)
    )
    candidate.changed_file_count = len(candidate.modified_files)
    candidate.added_lines, candidate.removed_lines = _count_diff_lines(candidate.git_diff)
    candidate.diff_line_count = candidate.added_lines + candidate.removed_lines


def _should_retry_patch_generation(error_message: str) -> bool:
    return any(marker in error_message for marker in RETRYABLE_PATCH_GENERATION_ERROR_MARKERS)


def _build_patch_retry_instruction(error_message: str) -> str:
    if "did not match the original code" in error_message:
        return (
            "The previous diff did not apply cleanly to the original project. Re-read the exact source "
            "context and regenerate a fresh unified diff from the original files instead of editing the "
            "failed diff."
        )
    if "Diff deletion did not match the original code." in error_message:
        return (
            "The previous diff deleted lines that do not exactly match the original project. Re-read the "
            "source and regenerate a fresh unified diff against the original files."
        )
    return (
        "The previous response was not a valid applicable unified diff. Return a fresh patch that follows "
        "the required `diff --git`, `---`, `+++`, and `@@` structure against the original files only."
    )


def _generate_patch_candidate(
    candidate: PatchCandidateResult,
    *,
    coder_prompt_payload: dict[str, Any],
    original_files: dict[str, str],
    entrypoint: str,
    language: str,
    model: str,
    stage_tools: list[Any],
    emit: EventEmitter,
    make_llm_context: Callable[[str, str], LLMCallContext],
) -> None:
    previous_error = ""
    previous_raw_output = ""

    def on_diff_chunk(chunk: str) -> None:
        emit("code_diff_chunk", {"stage": "code", "chunk": chunk})

    for attempt in range(1, PATCH_GENERATION_MAX_ATTEMPTS + 1):
        _reset_candidate_patch_state(candidate)
        candidate.error_message = None
        emit(
            "candidate_status",
            {
                "stage": "code",
                "candidate_key": candidate.key,
                "candidate_label": candidate.label,
                "status": "started" if attempt == 1 else "retrying",
                "attempt": attempt,
                "max_attempts": PATCH_GENERATION_MAX_ATTEMPTS,
            },
        )

        prompt_payload = {
            **coder_prompt_payload,
            "collaboration_context": {
                "mode": "single_patch_generation",
                "candidate_key": candidate.key,
                "candidate_label": candidate.label,
                "candidate_instructions": candidate.instructions,
                "attempt": attempt,
                "max_attempts": PATCH_GENERATION_MAX_ATTEMPTS,
            },
        }
        if previous_error:
            prompt_payload["retry_context"] = {
                "previous_error": previous_error,
                "instruction": _build_patch_retry_instruction(previous_error),
                "previous_failed_output_excerpt": previous_raw_output[:4000],
            }

        raw_diff = ""
        try:
            raw_diff = call_llm_for_json(
                prompt=json.dumps(
                    prompt_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                system_prompt=_render_code_system_prompt(language),
                model=model,
                isJson=False,
                stream=attempt == 1,
                stream_handler=on_diff_chunk if attempt == 1 else None,
                tools=stage_tools,
                reasoning_handler=_build_reasoning_event_handler(
                    emit,
                    "code",
                    candidate_label=candidate.label,
                ),
                tool_event_handler=_build_tool_event_handler(
                    emit,
                    "code",
                    candidate_label=candidate.label,
                ),
                audit_context=make_llm_context("code", f"code.diff.{candidate.key}.attempt_{attempt}"),
            )
            candidate.raw_output = raw_diff
            candidate.git_diff = _coerce_model_output_to_diff(
                raw_diff,
                original_files=original_files,
                entrypoint=entrypoint,
                language=language,
            )
            _populate_candidate_patch_state(
                candidate,
                original_files=original_files,
                entrypoint=entrypoint,
            )
            emit(
                "candidate_status",
                {
                    "stage": "code",
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "status": "generated",
                    "attempt": attempt,
                    "changed_file_count": candidate.changed_file_count,
                    "added_lines": candidate.added_lines,
                    "removed_lines": candidate.removed_lines,
                },
            )
            return
        except Exception as exc:
            candidate.raw_output = raw_diff
            candidate.error_message = str(exc)
            previous_error = candidate.error_message
            previous_raw_output = raw_diff
            _reset_candidate_patch_state(candidate)
            will_retry = attempt < PATCH_GENERATION_MAX_ATTEMPTS and _should_retry_patch_generation(
                candidate.error_message
            )
            emit(
                "candidate_status",
                {
                    "stage": "code",
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "status": "failed",
                    "attempt": attempt,
                    "max_attempts": PATCH_GENERATION_MAX_ATTEMPTS,
                    "error_message": candidate.error_message,
                    "will_retry": will_retry,
                },
            )
            if not will_retry:
                return


def run_repair_pipeline(
    request: RepairRequest,
    emit: EventEmitter,
    *,
    user_id: int | None = None,
) -> None:
    with prepare_project_workspace(
        code=request.code,
        filename=request.filename,
        language=request.language,
        project_files=request.project_files,
        project_zip_base64=request.project_zip_base64,
        github_repo_url=request.github_repo_url,
        github_ref=request.github_ref,
        project_subdir=request.project_subdir,
    ) as workspace:
        def make_llm_context(stage: str, purpose: str) -> LLMCallContext:
            return LLMCallContext(
                user_id=user_id,
                request_mode="repair",
                stage=stage,
                purpose=purpose,
                source_type=request.source_type,
            )

        _emit_stage(
            emit,
            "run",
            "started",
            message=(
                f"Running uploaded {get_language_spec(request.language).display_name} project "
                "in an isolated local process."
            ),
            entrypoint=workspace.entrypoint,
            source_type=workspace.source_type,
        )
        execution = run_project_safely(
            workspace.root_dir,
            filename=workspace.entrypoint,
            language=request.language,
            input_text=request.input_text,
            timeout_sec=request.timeout_sec,
        )

        # Optional: user-provided I/O test cases. When present they *replace*
        # the "no error = done" early-return: even a runtime-clean program
        # must also satisfy every case before we skip the repair pipeline.
        initial_case_results: list[dict[str, Any]] = []
        if request.test_cases:
            initial_case_results = _run_all_test_cases(
                workspace=workspace,
                cases=request.test_cases,
                language=request.language,
                timeout_sec=request.timeout_sec,
            )
        case_summary = _summarise_test_case_results(initial_case_results)

        emit(
            "run_result",
            {
                "execution": execution.to_dict(),
                "input_text": request.input_text or "",
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "entrypoint": workspace.entrypoint,
                "entrypoint_code": workspace.entrypoint_code,
                "source_type": workspace.source_type,
                "file_count": len(workspace.file_map),
                "user_prompt": request.user_prompt or "",
                "test_cases_summary": case_summary,
                "test_case_results": initial_case_results,
            },
        )

        # Decide whether to short-circuit as "clean" (no repair needed).
        #
        #   * No test cases provided  -> legacy behaviour: clean iff default
        #     execution succeeded.
        #   * Test cases provided     -> test cases are authoritative. The
        #     default execution is often NOT usable as a signal here, because
        #     the program may need stdin that only appears inside the cases
        #     themselves (e.g. LeetCode-style "read from stdin, print answer").
        #     So we declare clean iff every provided case passed.
        clean = (
            case_summary["all_passed"]
            if request.test_cases
            else execution.ok
        )
        if clean:
            _emit_stage(emit, "run", "completed", message="No runtime error detected.")
            emit(
                "result",
                {
                    "status": "clean",
                    "message": (
                        "No error detected."
                        if not request.test_cases
                        else (
                            "No runtime error and all "
                            f"{case_summary['total']} test case(s) passed."
                        )
                    ),
                    "filename": workspace.entrypoint,
                    "test_cases_summary": case_summary,
                    "test_case_results": initial_case_results,
                },
            )
            return

        runtime_report = build_project_runtime_inspection_report(
            workspace,
            request.input_text,
            execution,
        )
        # Surface user-provided context to the Inspector / Planner / Coder.
        # The test-case block is the *reason* we are repairing when the program
        # did not raise on the default stdin; the user-prompt is the free-form
        # hint (e.g. LeetCode problem statement).
        if request.user_prompt:
            runtime_report["user_prompt"] = request.user_prompt
        if request.test_cases:
            failing_cases = [r for r in initial_case_results if not r.get("passed")]
            runtime_report["test_cases"] = {
                "summary": case_summary,
                "cases": [
                    {
                        "index": r["index"],
                        "name": r["name"],
                        "stdin": r["stdin"],
                        "expected_stdout": r["expected_stdout"],
                        "actual_stdout": r["stdout"],
                        "actual_stderr_tail": (r.get("stderr") or "")[-1000:],
                        "passed": r["passed"],
                        "matched_output": r.get("matched_output"),
                        "runtime_ok": r.get("runtime_ok"),
                        "timed_out": r.get("timed_out"),
                    }
                    for r in initial_case_results
                ],
                "failing_cases": [r["name"] for r in failing_cases],
            }
        stage_tools = build_repair_tools(
            RepairToolContext(
                language=request.language,
                entrypoint=workspace.entrypoint,
                file_map=workspace.file_map,
                runtime_report=runtime_report,
                dependency_graph=workspace.dependency_graph,
                reverse_dependency_graph=workspace.reverse_dependency_graph,
            )
        )

        _emit_stage(emit, "inspect", "started")
        try:
            inspector_report = call_llm_for_json(
                prompt=json.dumps(runtime_report, ensure_ascii=False, indent=2),
                system_prompt=_render_inspector_system_prompt(request.language),
                model=request.model,
                isJson=True,
                tools=stage_tools,
                reasoning_handler=_build_reasoning_event_handler(emit, "inspect"),
                tool_event_handler=_build_tool_event_handler(emit, "inspect"),
                audit_context=make_llm_context("inspect", "inspect.report"),
            )
        except Exception as exc:
            if not getattr(exc, "json_parse_failed", False):
                raise
            inspector_report = _build_fallback_inspector_report(runtime_report, exc)
        # Propagate user-supplied hints into the Planner's context as well.
        # The Planner prompt serialises `inspector_report` in full, so attaching
        # these fields here makes them visible to the Planner without having to
        # modify the Planner prompt template.
        if isinstance(inspector_report, dict):
            if request.user_prompt and "user_prompt" not in inspector_report:
                inspector_report["user_prompt"] = request.user_prompt
            if request.test_cases and "test_cases" in runtime_report:
                inspector_report["test_cases"] = runtime_report["test_cases"]
        emit("inspect_report", {"stage": "inspect", "report": inspector_report})
        # Kick off the inspect first-person summary in the background so the plan stage's
        # primary LLM call can start immediately. The worker will emit `stage/completed`
        # for inspect when the streaming summary finishes.
        background_threads: list[threading.Thread] = []
        background_threads.append(
            _stream_explain_async(
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
                audit_context=make_llm_context("inspect", "inspect.explain"),
            )
        )

        _emit_stage(emit, "plan", "started")
        planner_prompt = build_planner_prompt(inspector_report)
        planner_report = call_llm_for_json(
            prompt=planner_prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model=request.model,
            isJson=False,
            tools=stage_tools,
            reasoning_handler=_build_reasoning_event_handler(emit, "plan"),
            tool_event_handler=_build_tool_event_handler(emit, "plan"),
            audit_context=make_llm_context("plan", "plan.report"),
        )
        emit("plan_report", {"stage": "plan", "report": planner_report})
        background_threads.append(
            _stream_explain_async(
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
                audit_context=make_llm_context("plan", "plan.explain"),
            )
        )

        _emit_stage(emit, "code", "started")

        coder_prompt_payload = {
            "language": request.language,
            "language_display_name": get_language_spec(request.language).display_name,
            "entrypoint": workspace.entrypoint,
            "project_summary": workspace.to_summary(),
            "runtime_report": runtime_report,
            "inspector_report": inspector_report,
            "planner_report": planner_report,
            "entrypoint_code": workspace.entrypoint_code,
            "output_contract": {
                "format": "unified git diff",
                "project_scope": "uploaded project only",
                "required_structure": [
                    "diff --git a/<path> b/<path>",
                    "--- a/<path>",
                    "+++ b/<path>",
                    "@@ ... @@",
                ],
                "forbidden_outputs": [
                    "full rewritten file",
                    "markdown code fences",
                    "explanatory prose",
                    "json",
                ],
            },
        }

        candidate = PatchCandidateResult(
            index=1,
            key=PATCH_GENERATION_PROFILE.key,
            label=PATCH_GENERATION_PROFILE.label,
            instructions=PATCH_GENERATION_PROFILE.instructions,
        )
        patch_candidates = [candidate]
        _generate_patch_candidate(
            candidate,
            coder_prompt_payload=coder_prompt_payload,
            original_files=workspace.file_map,
            entrypoint=workspace.entrypoint,
            language=request.language,
            model=request.model,
            stage_tools=stage_tools,
            emit=emit,
            make_llm_context=make_llm_context,
        )
        valid_candidates = [candidate for candidate in patch_candidates if candidate.git_diff]
        if not valid_candidates:
            raise RuntimeError(_build_candidate_failure_message(patch_candidates))

        provisional_leader = min(
            valid_candidates,
            key=lambda candidate: (candidate.changed_file_count, candidate.diff_line_count, candidate.index),
        )

        emit(
            "code_report",
            {
                "stage": "code",
                "git_diff": provisional_leader.git_diff,
                "report": _build_candidate_generation_report(
                    patch_candidates,
                    provisional_leader=provisional_leader,
                ),
            },
        )
        background_threads.append(
            _stream_explain_async(
                stage="code",
                prompt=json.dumps(
                    {
                        "planner_report": planner_report,
                        "language": request.language,
                        "candidate_generation_report": [
                            _summarize_candidate(candidate)
                            for candidate in patch_candidates
                        ],
                        "provisional_leader": {
                            "candidate_key": provisional_leader.key,
                            "candidate_label": provisional_leader.label,
                            "git_diff": provisional_leader.git_diff,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                system_prompt=CODE_EXPLAIN_SYSTEM_PROMPT,
                model=request.model,
                emit=emit,
                audit_context=make_llm_context("code", "code.explain"),
            )
        )

        _emit_stage(emit, "verify", "started")

        for candidate in patch_candidates:
            if not candidate.git_diff or not candidate.patched_files:
                candidate.score = _score_candidate(candidate)
                continue

            emit(
                "candidate_status",
                {
                    "stage": "verify",
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "status": "started",
                },
            )
            try:
                verify_tools = build_repair_tools(
                    RepairToolContext(
                        language=request.language,
                        entrypoint=workspace.entrypoint,
                        file_map=candidate.patched_files,
                        runtime_report=runtime_report,
                        dependency_graph=workspace.dependency_graph,
                        reverse_dependency_graph=workspace.reverse_dependency_graph,
                    )
                )
                verify_prompt = json.dumps(
                    {
                        "entrypoint": workspace.entrypoint,
                        "project_summary": workspace.to_summary(),
                        "runtime_report": runtime_report,
                        "inspector_report": inspector_report,
                        "planner_report": planner_report,
                        "git_diff": candidate.git_diff,
                        "candidate_profile": {
                            "candidate_key": candidate.key,
                            "candidate_label": candidate.label,
                            "candidate_instructions": candidate.instructions,
                        },
                        "patched_entrypoint_code": candidate.patched_files.get(workspace.entrypoint, ""),
                        "modified_files": candidate.modified_files,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                if request.language == "python":
                    # Run the patched project reproduction and the verify LLM call in
                    # parallel: neither depends on the other, so pipelining them saves a
                    # roughly rerun-duration-long wait on every Python repair.
                    patched_rerun_holder: dict[str, Any] = {}

                    def _run_patched_rerun() -> None:
                        try:
                            with materialize_patched_workspace(
                                workspace.root_dir,
                                candidate.patched_files,
                                set(workspace.file_map.keys()),
                            ) as patched_root:
                                patched_rerun_holder["execution"] = run_project_safely(
                                    patched_root,
                                    filename=workspace.entrypoint,
                                    language=request.language,
                                    input_text=request.input_text,
                                    timeout_sec=request.timeout_sec,
                                )
                        except Exception as rerun_exc:  # noqa: BLE001
                            patched_rerun_holder["error"] = rerun_exc

                    patched_rerun_thread = threading.Thread(
                        target=_run_patched_rerun,
                        daemon=True,
                    )
                    patched_rerun_thread.start()
                    try:
                        raw_verify_report = call_llm_for_json(
                            prompt=verify_prompt,
                            system_prompt=VERIFY_JSON_SYSTEM_PROMPT,
                            model=request.model,
                            isJson=True,
                            tools=verify_tools,
                            reasoning_handler=_build_reasoning_event_handler(
                                emit,
                                "verify",
                                candidate_label=candidate.label,
                            ),
                            tool_event_handler=_build_tool_event_handler(
                                emit,
                                "verify",
                                candidate_label=candidate.label,
                            ),
                            audit_context=make_llm_context("verify", f"verify.report.{candidate.key}"),
                        )
                    finally:
                        patched_rerun_thread.join()
                    if "error" in patched_rerun_holder:
                        raise patched_rerun_holder["error"]
                    candidate.patched_execution = patched_rerun_holder["execution"]
                else:
                    with materialize_patched_workspace(
                        workspace.root_dir,
                        candidate.patched_files,
                        set(workspace.file_map.keys()),
                    ) as patched_root:
                        candidate.patched_execution = run_project_safely(
                            patched_root,
                            filename=workspace.entrypoint,
                            language=request.language,
                            input_text=request.input_text,
                            timeout_sec=request.timeout_sec,
                        )
                if request.language == "python":
                    verification_base_source, skipped_verification_nodes = _build_verification_base_source(
                        candidate.patched_files.get(workspace.entrypoint, ""),
                        workspace.entrypoint,
                    )
                    existing_symbol_names = _collect_defined_symbol_names(
                        verification_base_source,
                        workspace.entrypoint,
                    )
                    verify_report = _normalize_verification_report(
                        raw_verify_report,
                        existing_symbol_names=existing_symbol_names,
                        filename=workspace.entrypoint,
                    )
                    verification_script = _build_verification_script(
                        verification_base_source,
                        verify_report["verification_code"],
                    )
                    with materialize_patched_workspace(
                        workspace.root_dir,
                        candidate.patched_files,
                        set(workspace.file_map.keys()),
                    ) as verify_root:
                        verify_target = verify_root / workspace.entrypoint
                        verify_target.parent.mkdir(parents=True, exist_ok=True)
                        verify_target.write_text(verification_script, encoding="utf-8")
                        candidate.verification_execution = run_project_safely(
                            verify_root,
                            filename=workspace.entrypoint,
                            language=request.language,
                            input_text=request.input_text,
                            timeout_sec=request.timeout_sec,
                        )

                    # If the user supplied test cases, re-run them against the
                    # patched workspace and require every one to pass before we
                    # declare the candidate "verified". This is the scenario
                    # behind a LeetCode-style submission: assertions alone are
                    # not enough — the actual program must print the expected
                    # output for every hidden case too.
                    patched_case_results: list[dict[str, Any]] = []
                    if request.test_cases:
                        patched_case_results = _run_all_test_cases(
                            workspace=workspace,
                            cases=request.test_cases,
                            language=request.language,
                            timeout_sec=request.timeout_sec,
                            file_map=candidate.patched_files,
                        )
                    patched_case_summary = _summarise_test_case_results(patched_case_results)

                    candidate.verify_passed = bool(
                        candidate.patched_execution.ok
                        and candidate.verification_execution.ok
                        and (not request.test_cases or patched_case_summary["all_passed"])
                    )
                    failed_line_index = _detect_failed_verification_line(
                        candidate.verification_execution,
                        base_line_count=_base_line_count(verification_base_source),
                        verification_code_len=len(verify_report["verification_code"]),
                    )
                    assertion_statuses = _compute_assertion_statuses(
                        verification_code=verify_report["verification_code"],
                        assertion_targets=verify_report.get("assertion_targets") or [],
                        verify_passed=candidate.verify_passed,
                        failed_line_index=failed_line_index,
                    )
                    candidate.verification_report = {
                        **verify_report,
                        "modified_files": candidate.modified_files,
                        "passed": candidate.verify_passed,
                        "assertion_statuses": assertion_statuses,
                        "failed_assertion_index": (
                            failed_line_index if failed_line_index is not None else None
                        ),
                        "test_cases_summary": patched_case_summary,
                        "test_case_results": patched_case_results,
                    }
                    candidate.verification_stdout = candidate.verification_execution.stdout
                    candidate.verification_stderr = candidate.verification_execution.stderr
                    candidate.verification_base_mode = "definition_only_entrypoint_context"
                    candidate.skipped_top_level_nodes = skipped_verification_nodes
                else:
                    patched_case_results_non_py: list[dict[str, Any]] = []
                    if request.test_cases:
                        patched_case_results_non_py = _run_all_test_cases(
                            workspace=workspace,
                            cases=request.test_cases,
                            language=request.language,
                            timeout_sec=request.timeout_sec,
                            file_map=candidate.patched_files,
                        )
                    patched_case_summary_non_py = _summarise_test_case_results(
                        patched_case_results_non_py
                    )
                    candidate.verification_execution = candidate.patched_execution
                    candidate.verify_passed = bool(
                        candidate.patched_execution.ok
                        and (
                            not request.test_cases
                            or patched_case_summary_non_py["all_passed"]
                        )
                    )
                    candidate.verification_report = {
                        **_build_runtime_only_verification_report(
                            language=request.language,
                            patched_execution=candidate.patched_execution,
                            modified_files=candidate.modified_files,
                        ),
                        "test_cases_summary": patched_case_summary_non_py,
                        "test_case_results": patched_case_results_non_py,
                    }
                    candidate.verification_stdout = candidate.patched_execution.stdout
                    candidate.verification_stderr = candidate.patched_execution.stderr
                    candidate.verification_base_mode = "runtime_rerun_only"
            except Exception as exc:
                candidate.error_message = str(exc)

            candidate.score = _score_candidate(candidate)
            emit(
                "candidate_status",
                {
                    "stage": "verify",
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "status": "completed",
                    "passed": candidate.verify_passed,
                    "score": candidate.score,
                    "error_message": candidate.error_message,
                },
            )

        ranked_candidates = _rank_candidates(patch_candidates)
        selected_candidate = ranked_candidates[0]
        single_candidate_mode = len(ranked_candidates) == 1
        verify_payload = {
            "collaboration_mode": (
                "single_patch_generation"
                if single_candidate_mode
                else "multi_candidate_patch_committee"
            ),
            "selection_policy": (
                (
                    "A single repair agent generated one patch, and verification re-ran that patch before "
                    "keeping the result."
                )
                if single_candidate_mode
                else (
                    "Candidates are ranked by patched runtime success, verification success, and patch size. "
                    + (
                        "Python candidates also receive additional credit for assertion coverage."
                        if request.language == "python"
                        else "For non-Python languages the verification step uses patched runtime reruns."
                    )
                )
            ),
            "candidate_count": len(ranked_candidates),
            "selected_candidate": _summarize_candidate(
                selected_candidate,
                include_details=True,
            ),
            "ranked_candidates": [
                _summarize_candidate(candidate)
                for candidate in ranked_candidates
            ],
            "passed": selected_candidate.verify_passed,
        }
        emit("verify_report", {"stage": "verify", "report": verify_payload})
        # Stream the verify summary in the background too, so the user-facing "result" event
        # (with the diff, ready for apply) arrives as soon as verification data is available.
        background_threads.append(
            _stream_explain_async(
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
                audit_context=make_llm_context("verify", "verify.explain"),
            )
        )

        emit(
            "result",
            {
                "status": "verified" if selected_candidate.verify_passed else "verify_failed",
                "filename": workspace.entrypoint,
                "git_diff": selected_candidate.git_diff,
                "verification_passed": selected_candidate.verify_passed,
                "selection_summary": (
                    f"Verified the patch generated by {selected_candidate.label} with score "
                    f"{selected_candidate.score}."
                    if len(ranked_candidates) == 1
                    else (
                        f"Auto-selected {selected_candidate.label} from {len(ranked_candidates)} patch "
                        f"candidates with score {selected_candidate.score}."
                    )
                ),
                "message": (
                    "Verification passed. Review the patch and decide whether to accept it."
                    if selected_candidate.verify_passed
                    else (
                        "Verification failed. The generated patch did not satisfy the assertion checks."
                        if request.language == "python"
                        else "Verification failed. The generated patch still did not run cleanly after rerun."
                    )
                ),
            },
        )

        # Wait for every background explain thread to flush before the SSE stream closes,
        # otherwise the sentinel in the API layer could be pushed while chunks are still
        # being emitted to the queue.
        for thread in background_threads:
            thread.join()
