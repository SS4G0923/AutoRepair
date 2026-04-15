from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.inspector.inspector_prompt import build_planner_prompt
from backend.llm import call_llm_for_json
from backend.llm.agent_tools import RepairToolContext, build_repair_tools
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

DEFAULT_MODEL = "qwen3.5-flash"
MAX_CODE_CHARS = 100_000
MAX_TIMEOUT_SEC = 30

INSPECTOR_JSON_SYSTEM_PROMPT_TEMPLATE = """You are an expert bug inspection agent.
You will receive a structured local runtime report for an uploaded {language_name} file or project.
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
    error_message: str | None = None
    rank: int = 0


PATCH_CANDIDATE_PROFILES: tuple[PatchCandidateProfile, ...] = (
    PatchCandidateProfile(
        key="minimal_hotfix",
        label="Minimal Hotfix Agent",
        instructions=(
            "Prioritize the smallest safe change that directly fixes the observed failure. "
            "Avoid refactors and touch as few lines and files as possible."
        ),
    ),
    PatchCandidateProfile(
        key="root_cause_agent",
        label="Root Cause Agent",
        instructions=(
            "Prioritize a durable fix for the underlying root cause, even if it requires a small helper change "
            "or a narrowly scoped cross-file edit."
        ),
    ),
    PatchCandidateProfile(
        key="defensive_guard_agent",
        label="Defensive Guard Agent",
        instructions=(
            "Prioritize resilience and edge-case handling around the failure path while keeping behavior changes small "
            "and backwards compatible."
        ),
    ),
)


@dataclass(frozen=True)
class RepairRequest:
    code: str | None = None
    filename: str | None = "main.py"
    language: str = "python"
    timeout_sec: int = 5
    model: str = DEFAULT_MODEL
    project_files: tuple[ProjectFileInput, ...] = ()
    project_zip_base64: str | None = None
    github_repo_url: str | None = None
    github_ref: str | None = None
    project_subdir: str | None = None

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

        model = payload.get("model", DEFAULT_MODEL)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("`model` must be a non-empty string.")

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
            timeout_sec=timeout_sec,
            model=model.strip(),
            project_files=tuple(project_files),
            project_zip_base64=project_zip_base64.strip() if isinstance(project_zip_base64, str) else None,
            github_repo_url=github_repo_url.strip() if isinstance(github_repo_url, str) else None,
            github_ref=github_ref.strip() if isinstance(github_ref, str) and github_ref.strip() else None,
            project_subdir=project_subdir,
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

    explain_text = call_llm_for_json(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        isJson=False,
        stream=True,
        stream_handler=on_chunk,
        audit_context=audit_context,
    )
    emit("explain_done", {"stage": stage, "text": explain_text})
    return explain_text


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
    payload = {
        "collaboration_mode": "multi_candidate_patch_committee",
        "selection_policy": (
            "Specialized coder agents generated multiple patch candidates. "
            "A later verification stage will execute, score, rank, and auto-select the best candidate."
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


def _score_candidate(candidate: PatchCandidateResult) -> int:
    score = 0
    if candidate.git_diff:
        score += 20
    if candidate.patched_execution is not None:
        if candidate.patched_execution.ok:
            score += 40
        elif candidate.patched_execution.timed_out:
            score -= 18
        else:
            score -= 8
    if candidate.verification_report is not None:
        score += min(int(candidate.verification_report.get("assert_count") or 0), 6)
    if candidate.verification_execution is not None:
        if candidate.verification_execution.ok:
            score += 34
        elif candidate.verification_execution.timed_out:
            score -= 18
        else:
            score -= 10
    if candidate.verify_passed:
        score += 25
    score -= max(0, candidate.changed_file_count - 1) * 3
    score -= min(candidate.diff_line_count, 120) // 12
    if candidate.error_message:
        score -= 20
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
    if not details:
        return "All coder agents failed to produce a valid patch candidate."
    return "All coder agents failed to produce a valid patch candidate. " + " | ".join(details[:6])


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
            timeout_sec=request.timeout_sec,
        )
        emit(
            "run_result",
            {
                "execution": execution.to_dict(),
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "entrypoint": workspace.entrypoint,
                "source_type": workspace.source_type,
                "file_count": len(workspace.file_map),
            },
        )

        if execution.ok:
            _emit_stage(emit, "run", "completed", message="No runtime error detected.")
            emit(
                "result",
                {
                    "status": "clean",
                    "message": "No error detected.",
                    "filename": workspace.entrypoint,
                },
            )
            return

        runtime_report = build_project_runtime_inspection_report(workspace, execution)
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
        inspector_report = call_llm_for_json(
            prompt=json.dumps(runtime_report, ensure_ascii=False, indent=2),
            system_prompt=_render_inspector_system_prompt(request.language),
            model=request.model,
            isJson=True,
            tools=stage_tools,
            tool_event_handler=_build_tool_event_handler(emit, "inspect"),
            audit_context=make_llm_context("inspect", "inspect.report"),
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
            audit_context=make_llm_context("inspect", "inspect.explain"),
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
            audit_context=make_llm_context("plan", "plan.report"),
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
            audit_context=make_llm_context("plan", "plan.explain"),
        )
        _emit_stage(emit, "plan", "completed")

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

        patch_candidates: list[PatchCandidateResult] = []
        for index, profile in enumerate(PATCH_CANDIDATE_PROFILES, start=1):
            candidate = PatchCandidateResult(
                index=index,
                key=profile.key,
                label=profile.label,
                instructions=profile.instructions,
            )
            patch_candidates.append(candidate)
            emit(
                "candidate_status",
                {
                    "stage": "code",
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "status": "started",
                },
            )
            try:
                raw_diff = call_llm_for_json(
                    prompt=json.dumps(
                        {
                            **coder_prompt_payload,
                            "collaboration_context": {
                                "mode": "multi_candidate_patch_committee",
                                "candidate_key": candidate.key,
                                "candidate_label": candidate.label,
                                "candidate_instructions": candidate.instructions,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    system_prompt=_render_code_system_prompt(request.language),
                    model=request.model,
                    isJson=False,
                    stream=False,
                    tools=stage_tools,
                    tool_event_handler=_build_tool_event_handler(
                        emit,
                        "code",
                        candidate_label=candidate.label,
                    ),
                    audit_context=make_llm_context("code", f"code.diff.{candidate.key}"),
                )
                candidate.raw_output = raw_diff
                candidate.git_diff = _coerce_model_output_to_diff(
                    raw_diff,
                    original_files=workspace.file_map,
                    entrypoint=workspace.entrypoint,
                    language=request.language,
                )
                candidate.patched_files = _apply_unified_diff_to_project(
                    workspace.file_map,
                    candidate.git_diff,
                    default_path=workspace.entrypoint,
                )
                candidate.modified_files = sorted(
                    path
                    for path in set(workspace.file_map.keys()) | set(candidate.patched_files.keys())
                    if workspace.file_map.get(path) != candidate.patched_files.get(path)
                )
                candidate.changed_file_count = len(candidate.modified_files)
                candidate.added_lines, candidate.removed_lines = _count_diff_lines(candidate.git_diff)
                candidate.diff_line_count = candidate.added_lines + candidate.removed_lines
                emit(
                    "candidate_status",
                    {
                        "stage": "code",
                        "candidate_key": candidate.key,
                        "candidate_label": candidate.label,
                        "status": "generated",
                        "changed_file_count": candidate.changed_file_count,
                        "added_lines": candidate.added_lines,
                        "removed_lines": candidate.removed_lines,
                    },
                )
            except Exception as exc:
                candidate.error_message = str(exc)
                emit(
                    "candidate_status",
                    {
                        "stage": "code",
                        "candidate_key": candidate.key,
                        "candidate_label": candidate.label,
                        "status": "failed",
                        "error_message": candidate.error_message,
                    },
                )

        valid_candidates = [candidate for candidate in patch_candidates if candidate.git_diff]
        if not valid_candidates:
            fallback_candidate = PatchCandidateResult(
                index=len(patch_candidates) + 1,
                key="single_agent_fallback",
                label="Single Agent Fallback",
                instructions=(
                    "Retry patch generation with the original single-agent streaming path so the "
                    "workflow can recover even if all specialized candidates fail."
                ),
            )
            patch_candidates.append(fallback_candidate)
            emit(
                "candidate_status",
                {
                    "stage": "code",
                    "candidate_key": fallback_candidate.key,
                    "candidate_label": fallback_candidate.label,
                    "status": "started",
                },
            )

            def on_fallback_diff_chunk(chunk: str) -> None:
                emit("code_diff_chunk", {"stage": "code", "chunk": chunk})

            try:
                fallback_raw_diff = call_llm_for_json(
                    prompt=json.dumps(
                        coder_prompt_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    system_prompt=_render_code_system_prompt(request.language),
                    model=request.model,
                    isJson=False,
                    stream=True,
                    stream_handler=on_fallback_diff_chunk,
                    tools=stage_tools,
                    tool_event_handler=_build_tool_event_handler(
                        emit,
                        "code",
                        candidate_label=fallback_candidate.label,
                    ),
                    audit_context=make_llm_context("code", "code.diff.single_agent_fallback"),
                )
                fallback_candidate.raw_output = fallback_raw_diff
                fallback_candidate.git_diff = _coerce_model_output_to_diff(
                    fallback_raw_diff,
                    original_files=workspace.file_map,
                    entrypoint=workspace.entrypoint,
                    language=request.language,
                )
                fallback_candidate.patched_files = _apply_unified_diff_to_project(
                    workspace.file_map,
                    fallback_candidate.git_diff,
                    default_path=workspace.entrypoint,
                )
                fallback_candidate.modified_files = sorted(
                    path
                    for path in set(workspace.file_map.keys()) | set(fallback_candidate.patched_files.keys())
                    if workspace.file_map.get(path) != fallback_candidate.patched_files.get(path)
                )
                fallback_candidate.changed_file_count = len(fallback_candidate.modified_files)
                fallback_candidate.added_lines, fallback_candidate.removed_lines = _count_diff_lines(
                    fallback_candidate.git_diff
                )
                fallback_candidate.diff_line_count = (
                    fallback_candidate.added_lines + fallback_candidate.removed_lines
                )
                emit(
                    "candidate_status",
                    {
                        "stage": "code",
                        "candidate_key": fallback_candidate.key,
                        "candidate_label": fallback_candidate.label,
                        "status": "generated",
                        "changed_file_count": fallback_candidate.changed_file_count,
                        "added_lines": fallback_candidate.added_lines,
                        "removed_lines": fallback_candidate.removed_lines,
                    },
                )
            except Exception as exc:
                fallback_candidate.error_message = str(exc)
                emit(
                    "candidate_status",
                    {
                        "stage": "code",
                        "candidate_key": fallback_candidate.key,
                        "candidate_label": fallback_candidate.label,
                        "status": "failed",
                        "error_message": fallback_candidate.error_message,
                    },
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
        _stream_explain(
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
        _emit_stage(emit, "code", "completed")

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
                with materialize_patched_workspace(
                    workspace.root_dir,
                    candidate.patched_files,
                    set(workspace.file_map.keys()),
                ) as patched_root:
                    candidate.patched_execution = run_project_safely(
                        patched_root,
                        filename=workspace.entrypoint,
                        language=request.language,
                        timeout_sec=request.timeout_sec,
                    )
                if request.language == "python":
                    raw_verify_report = call_llm_for_json(
                        prompt=verify_prompt,
                        system_prompt=VERIFY_JSON_SYSTEM_PROMPT,
                        model=request.model,
                        isJson=True,
                        tools=verify_tools,
                        tool_event_handler=_build_tool_event_handler(
                            emit,
                            "verify",
                            candidate_label=candidate.label,
                        ),
                        audit_context=make_llm_context("verify", f"verify.report.{candidate.key}"),
                    )
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
                            timeout_sec=request.timeout_sec,
                        )

                    candidate.verify_passed = bool(
                        candidate.patched_execution.ok
                        and candidate.verification_execution.ok
                    )
                    candidate.verification_report = {
                        **verify_report,
                        "modified_files": candidate.modified_files,
                        "passed": candidate.verify_passed,
                    }
                    candidate.verification_stdout = candidate.verification_execution.stdout
                    candidate.verification_stderr = candidate.verification_execution.stderr
                    candidate.verification_base_mode = "definition_only_entrypoint_context"
                    candidate.skipped_top_level_nodes = skipped_verification_nodes
                else:
                    candidate.verification_execution = candidate.patched_execution
                    candidate.verify_passed = bool(candidate.patched_execution.ok)
                    candidate.verification_report = _build_runtime_only_verification_report(
                        language=request.language,
                        patched_execution=candidate.patched_execution,
                        modified_files=candidate.modified_files,
                    )
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
        verify_payload = {
            "collaboration_mode": "multi_candidate_patch_committee",
            "selection_policy": (
                "Candidates are ranked by patched runtime success, verification success, and patch size. "
                + (
                    "Python candidates also receive additional credit for assertion coverage."
                    if request.language == "python"
                    else "For non-Python languages the verification step uses patched runtime reruns."
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
            audit_context=make_llm_context("verify", "verify.explain"),
        )
        _emit_stage(emit, "verify", "completed")

        emit(
            "result",
            {
                "status": "verified" if selected_candidate.verify_passed else "verify_failed",
                "filename": workspace.entrypoint,
                "git_diff": selected_candidate.git_diff,
                "verification_passed": selected_candidate.verify_passed,
                "selection_summary": (
                    f"Auto-selected {selected_candidate.label} from {len(ranked_candidates)} patch candidates "
                    f"with score {selected_candidate.score}."
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
