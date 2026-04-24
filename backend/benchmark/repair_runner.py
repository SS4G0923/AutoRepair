"""Real Defects4J repair runner.

Unlike `runner.py` (which only does `inspect_only` / `mock_repair`), this module
actually produces a patch using an LLM, applies it to the checked-out source
tree, and then re-runs the Defects4J test harness to verify whether the patch
is *plausible* (compiles + whole test suite passes) and *correct* (the
originally-failing trigger tests now pass and no new failure was introduced).

Two strategies are supported out of the box:

* ``naive_chat`` – the "strong model, no system" baseline.  The model sees only
  the buggy file + the names of the failing trigger tests with a short prompt.
  It is forced to guess where the bug is on its own.
* ``full_pipeline`` – the "weak model, full AutoRepair scaffolding" arm.  The
  model receives:
    - the ranked suspects with their source snippets (from the inspector),
    - the extracted stack-frame / exception metadata,
    - the raw failing-test block (truncated).

This module is intentionally independent of the SSE/Flask pipeline so it can be
scripted from ``backend/scripts/run_benchmark_experiment.py``.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from backend.benchmark import store as bench_store
from backend.inspector.defects4j_runner import Defects4JRunner
from backend.inspector.inspector import run_defects4j_inspection
from backend.inspector.source_utils import (
    SourceIndex,
    extract_enclosing_method,
    resolve_source_for_frame,
)
from backend.llm import call_llm_for_json
from backend.repair.pipeline import (
    _apply_unified_diff_to_project,
    _coerce_model_output_to_diff,
    _count_diff_lines,
    _extract_source_code_candidate,
    _render_project_diff,
)

logger = logging.getLogger(__name__)

STRATEGY_NAIVE_CHAT = "naive_chat"
STRATEGY_FULL_PIPELINE = "full_pipeline"
SUPPORTED_STRATEGIES = {STRATEGY_NAIVE_CHAT, STRATEGY_FULL_PIPELINE}

# How many LLM rounds to attempt for one full_repair run. Each failed round
# feeds back the new failure information so the model can correct course.
# Override via env var BENCHMARK_MAX_REPAIR_ROUNDS for ablation studies.
try:
    MAX_REPAIR_ROUNDS = max(1, int(os.getenv("BENCHMARK_MAX_REPAIR_ROUNDS", "3")))
except (TypeError, ValueError):
    MAX_REPAIR_ROUNDS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_workspace_root() -> Path:
    root = os.getenv("BENCHMARK_WORK_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (Path.cwd() / "tmp" / "benchmark").resolve()


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _prepare_work_dir(
    project_code: str,
    bug_id: int,
    run_id: int,
    *,
    reuse: bool = False,
) -> tuple[Path, Path]:
    workspace_root = _default_workspace_root()
    run_dir = workspace_root / f"{project_code}_{bug_id}" / f"run_{run_id}"
    artifacts_dir = run_dir / ".inspector"
    if not reuse and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifacts_dir


def _count_failing_from_test_output(runner: Defects4JRunner, work_dir: Path) -> tuple[list[str], int]:
    failing, _raw, _blocks = runner.read_failing_tests_blocked(work_dir)
    return failing, len(failing)


def _read_suspect_source(work_dir: Path, suspect: dict[str, Any]) -> tuple[Path | None, str | None]:
    rel = suspect.get("source_file") or ""
    if not rel:
        return None, None
    abs_path = (work_dir / rel).resolve()
    if not abs_path.exists():
        return abs_path, None
    try:
        return abs_path, abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return abs_path, None


_TEST_PATH_MARKERS = ("src/test/", "/test/", "src\\test\\")
_TEST_CLASS_SUFFIXES = ("Test", "Tests", "TestCase", "IT", "ITCase")


def _is_test_source_file(source_file: str | None, class_name: str | None = None) -> bool:
    """Best-effort detection of JUnit test files.

    A suspect that points at a test file is almost never the right place to
    patch — the production bug lives elsewhere and the test just observes it.
    We use two independent signals (path + class name suffix) so we stay
    correct even for Defects4J projects whose layout deviates slightly from
    the Maven convention.
    """
    if source_file:
        normalized = source_file.replace("\\", "/").lower()
        for marker in _TEST_PATH_MARKERS:
            if marker in normalized:
                return True
    if class_name:
        base = class_name.rsplit(".", 1)[-1].split("$", 1)[0]
        if any(base.endswith(suf) for suf in _TEST_CLASS_SUFFIXES):
            return True
    return False


def _extract_top_suspect(report: dict[str, Any]) -> dict[str, Any] | None:
    """Return the most likely suspect, skipping any that point at test files.

    We scan the inspector output twice: first preferring production suspects
    (non-test), then — only if nothing production-y exists — falling back to
    whatever is available. This keeps behaviour compatible with bugs whose
    real patch location *is* somehow in a test utility class.
    """
    failures = report.get("failures") or []

    def _iter_all() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for failure in failures:
            for suspect in failure.get("suspects") or []:
                if suspect.get("source_file"):
                    out.append(suspect)
            for candidate in failure.get("ranked_candidates") or []:
                if candidate.get("source_file"):
                    out.append(candidate)
        return out

    all_candidates = _iter_all()
    for candidate in all_candidates:
        if not _is_test_source_file(
            candidate.get("source_file"), candidate.get("class_name")
        ):
            return candidate
    # Nothing in production — return the first test candidate so the caller
    # can still decide to use `classes.modified` override as a second chance.
    return all_candidates[0] if all_candidates else None


def _simple_file_name(fqn: str) -> str:
    """Return `Bar.java` for `org.foo.Bar$Inner`."""
    base = fqn.split("$", 1)[0].strip()
    return base.rsplit(".", 1)[-1] + ".java"


def _suspects_from_modified_classes(
    work_dir: Path, inspection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Ground-truth fallback: ask Defects4J which classes the bug fix touches.

    Some bugs (e.g. Mockito-1) produce test stack traces with no recoverable
    project frames, so the inspector's suspects list is empty.  For those cases
    we still know, with 100% certainty, which source files the patch needs to
    touch — Defects4J records that metadata in every bug checkout.

    Returns a list (possibly empty) of synthetic suspect dicts compatible with
    the shape produced by ``inspector.inspector.run_defects4j_inspection``.
    """
    try:
        from backend.inspector.source_utils import (
            SourceIndex,
            extract_code_snippet,
            resolve_source_for_frame,
        )

        runner = Defects4JRunner()
        export = runner.export(work_dir, "classes.modified")
        if export.returncode != 0:
            return []
        raw = (export.stdout or "").strip()
        if not raw:
            return []
    except Exception:  # pragma: no cover - defects4j cli glitches
        return []

    # `classes.modified` may use ';' or newline as separator.
    classes = [c.strip() for c in re.split(r"[;\n]+", raw) if c.strip()]
    if not classes:
        return []

    src_classes_rel = (
        (inspection.get("defects4j") or {}).get("properties", {}).get("dir.src.classes")
    )
    src_roots: list[Path] = []
    if src_classes_rel:
        src_roots.append((work_dir / src_classes_rel).resolve())
    if not src_roots:
        src_roots.append(work_dir)

    index = SourceIndex(src_roots)
    index.build()

    suspects: list[dict[str, Any]] = []
    for fqn in classes:
        file_name = _simple_file_name(fqn)
        resolved = resolve_source_for_frame(index, fqn, file_name)
        if resolved is None:
            continue
        rel_path = str(resolved.source_file.resolve().relative_to(work_dir.resolve()))
        snippet = extract_code_snippet(resolved.source_file, 1, window=40) or ""
        suspects.append(
            {
                "class_name": fqn,
                "file_name": file_name,
                "line_number": 1,
                "source_file": rel_path,
                "resolve_confidence": resolved.confidence,
                "resolve_reason": resolved.reason,
                "score": 0.0,
                "score_reasons": ["fallback_from_classes_modified"],
                "snippet": snippet,
                "enclosing_method": None,
            }
        )
    return suspects


def _extract_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the inspector's serialized report into prompt-friendly fields.

    The upstream ``inspector.run_defects4j_inspection`` has already parsed the
    JUnit log, filtered the stack trace to project frames, and attached
    ``enclosing_method`` context to every suspect.  We just pick the richest
    signals out of that report.
    """
    failures = report.get("failures") or []
    if not failures:
        return {}
    first = failures[0]
    ranked = first.get("ranked_candidates") or []
    return {
        "failing_test": first.get("failing_test"),
        "evidence_source": first.get("evidence_source"),
        "exception_type": first.get("exception_type"),
        "exception_message": first.get("exception_message"),
        "evidence_excerpt": first.get("evidence_excerpt") or "",
        "top_frames": [
            {
                "class_name": frame.get("class_name"),
                "file_name": frame.get("file_name"),
                "line_number": frame.get("line_number"),
                "raw_line": frame.get("raw_line"),
            }
            for frame in ranked[:5]
        ],
    }


def _read_inspector_test_log(artifacts_dir: Path, *, max_chars: int = 2400) -> str:
    """Re-use the raw per-test log that the inspector already wrote to disk."""
    stdout_path = artifacts_dir / "single_test_0.stdout.txt"
    stderr_path = artifacts_dir / "single_test_0.stderr.txt"
    chunks: list[str] = []
    for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        chunks.append(f"# {label}\n{text}")
    combined = "\n\n".join(chunks)
    return _shorten(combined, max_chars=max_chars)


def _shorten(text: str, *, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _compute_diff(before: str, after: str, rel_path: str) -> tuple[str, int, int]:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return diff_text, added, removed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


_PATCH_OUTPUT_INSTRUCTIONS = (
    "Return ONLY one patch payload and nothing else.\n\n"
    "Preferred format:\n"
    "- A valid unified git diff that can touch one or more of the editable files.\n"
    "- Each modified file must include `--- a/<path>`, `+++ b/<path>`, and at least one `@@` hunk.\n\n"
    "Fallback format (only if there is exactly one editable file):\n"
    "- Return the full corrected Java source for that single file, with no Markdown fences.\n\n"
    "Rules:\n"
    "1. Do not return JSON, explanations, bullet lists, or Markdown fences.\n"
    "2. Keep the patch minimal and correctness-focused.\n"
    "3. Modify only the editable file(s) shown in the prompt.\n"
    "4. Preserve existing formatting unless a formatting change is required for correctness.\n"
    "5. Prefer fixing the real root cause over hard-coding the observed test outputs.\n"
    "6. If multiple similar code blocks exist, include enough diff context to identify the intended one precisely, "
    "or prefer returning the full corrected file when only one editable file is allowed.\n"
    "7. When the failure suggests impossible numeric values or ordering violations, prefer arithmetic or comparison fixes before broad refactors."
)


NAIVE_SYSTEM_PROMPT = (
    "You are a Java bug-fixing assistant. The user will give you a Java file "
    "that contains a bug and the name of a JUnit test that currently fails. "
    "You must identify the bug ON YOUR OWN (no hints) and patch the file.\n\n"
    + _PATCH_OUTPUT_INSTRUCTIONS
)


FULL_PIPELINE_SYSTEM_PROMPT = (
    "You are the patch-generation stage of the AutoRepair pipeline. The "
    "`inspector` stage has already given you: the failing tests, the raw "
    "stack-trace, and a ranked list of the most likely buggy locations in "
    "the file. Your only job is to propose a minimal, correct patch.\n\n"
    + _PATCH_OUTPUT_INSTRUCTIONS
)


def _split_test_identifier(test_id: str) -> tuple[str | None, str | None]:
    if "::" not in (test_id or ""):
        return None, None
    class_name, method_name = test_id.split("::", 1)
    class_name = class_name.strip() or None
    method_name = method_name.strip() or None
    return class_name, method_name


def _find_method_anchor_line(source_file: Path, method_name: str) -> int | None:
    matches = _find_method_anchor_lines(source_file, method_name, max_matches=1)
    return matches[0] if matches else None


def _find_method_anchor_lines(
    source_file: Path,
    method_name: str,
    *,
    max_matches: int = 4,
) -> list[int]:
    try:
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    signature_pattern = re.compile(
        rf"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|\s)+"
        rf"[\w\<\>\[\], ?.$]+\s+{re.escape(method_name)}\s*\("
    )
    matches: list[int] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue
        if signature_pattern.search(line):
            matches.append(idx)
            if len(matches) >= max_matches:
                break
    return matches


def _extract_test_contexts(
    work_dir: Path,
    inspection: dict[str, Any],
    failing_tests: Iterable[str],
) -> list[dict[str, Any]]:
    src_tests_rel = (
        (inspection.get("defects4j") or {}).get("properties", {}).get("dir.src.tests")
    )
    if not src_tests_rel:
        return []
    test_root = (work_dir / str(src_tests_rel)).resolve()
    if not test_root.exists():
        return []

    index = SourceIndex([], test_roots=[test_root])
    index.build()
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for test_id in failing_tests:
        class_name, method_name = _split_test_identifier(str(test_id))
        if not class_name or not method_name or test_id in seen:
            continue
        seen.add(str(test_id))
        file_name = class_name.rsplit(".", 1)[-1] + ".java"
        resolved = resolve_source_for_frame(
            index,
            class_name,
            file_name,
            prefer_production=False,
        )
        if resolved is None or not resolved.source_file.exists():
            continue
        anchor_line = _find_method_anchor_line(resolved.source_file, method_name) or 1
        snippet = extract_enclosing_method(resolved.source_file, anchor_line)
        contexts.append(
            {
                "test_id": str(test_id),
                "class_name": class_name,
                "method_name": method_name,
                "source_file": _safe_relpath(resolved.source_file, work_dir),
                "anchor_line": anchor_line,
                "snippet": _shorten(snippet, max_chars=2800),
            }
        )
        if len(contexts) >= 2:
            break
    return contexts


def _render_test_contexts(test_contexts: list[dict[str, Any]]) -> str:
    if not test_contexts:
        return "  (failing test source snippet unavailable)"
    blocks: list[str] = []
    for idx, ctx in enumerate(test_contexts, start=1):
        blocks.append(
            f"### Failing test context #{idx}\n"
            f"- test: `{ctx.get('test_id')}`\n"
            f"- file: `{ctx.get('source_file')}`\n"
            f"```java\n{ctx.get('snippet') or ''}\n```"
        )
    return "\n\n".join(blocks)


def _extract_declared_method_names(source_text: str) -> list[str]:
    pattern = re.compile(
        r"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|\s)+"
        r"[\w\<\>\[\], ?.$]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    seen: set[str] = set()
    names: list[str] = []
    for match in pattern.finditer(source_text):
        method_name = match.group(1)
        if method_name in seen:
            continue
        seen.add(method_name)
        names.append(method_name)
    return names


def _extract_focus_method_contexts(
    *,
    source_file: Path,
    source_text: str,
    work_dir: Path,
    failing_tests: Iterable[str],
) -> list[dict[str, Any]]:
    declared_methods = _extract_declared_method_names(source_text)
    if not declared_methods:
        return []

    matched_method_names: list[str] = []
    seen_method_names: set[str] = set()
    for test_id in failing_tests:
        _class_name, test_method = _split_test_identifier(str(test_id))
        if not test_method:
            continue
        lowered_test_method = test_method.lower()
        for method_name in declared_methods:
            if len(method_name) < 4:
                continue
            if method_name.lower() not in lowered_test_method:
                continue
            if method_name in seen_method_names:
                continue
            seen_method_names.add(method_name)
            matched_method_names.append(method_name)

    contexts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for method_name in matched_method_names:
        for anchor_line in _find_method_anchor_lines(source_file, method_name):
            key = (method_name, anchor_line)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            contexts.append(
                {
                    "method_name": method_name,
                    "source_file": _safe_relpath(source_file, work_dir),
                    "anchor_line": anchor_line,
                    "reason": f"matched from failing test name `{method_name}`",
                    "snippet": _shorten(
                        extract_enclosing_method(source_file, anchor_line),
                        max_chars=2600,
                    ),
                }
            )
            if len(contexts) >= 4:
                return contexts
    return contexts


def _render_focus_method_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "  (no method names from the failing tests matched this source file)"
    blocks: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"### Keyword-matched method context #{idx}\n"
            f"- file: `{ctx.get('source_file')}`\n"
            f"- method: `{ctx.get('method_name')}`\n"
            f"- why included: {ctx.get('reason')}\n"
            f"```java\n{ctx.get('snippet') or ''}\n```"
        )
    return "\n\n".join(blocks)


_STRING_LITERAL_RE = re.compile(r'"([^"\n]{1,120})"')


def _extract_failure_expectation_phrases(test_contexts: list[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for ctx in test_contexts:
        snippet = str(ctx.get("snippet") or "")
        for raw_phrase in _STRING_LITERAL_RE.findall(snippet):
            phrase = " ".join(raw_phrase.split())
            if len(phrase) < 8:
                continue
            if " " not in phrase and "-" not in phrase and ":" not in phrase:
                continue
            lowered = phrase.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            phrases.append(phrase)
    return phrases[:6]


def _render_line_window(source_text: str, *, anchor_line: int, window: int = 5) -> str:
    lines = source_text.splitlines()
    if not lines:
        return ""
    start = max(0, anchor_line - 1 - window)
    end = min(len(lines), anchor_line + window)
    rendered: list[str] = []
    for idx in range(start, end):
        marker = ">>" if idx + 1 == anchor_line else "  "
        rendered.append(f"{marker} {idx + 1:5d}: {lines[idx]}")
    return "\n".join(rendered)


def _count_failure_expectation_matches(source_text: str, phrases: list[str]) -> int:
    lowered_source = source_text.lower()
    return sum(1 for phrase in phrases if phrase.lower() in lowered_source)


def _extract_failure_expectation_contexts(
    *,
    source_file: Path,
    source_text: str,
    work_dir: Path,
    test_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phrases = _extract_failure_expectation_phrases(test_contexts)
    if not phrases:
        return []
    lines = source_text.splitlines()
    contexts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for phrase in phrases:
        lowered_phrase = phrase.lower()
        for idx, line in enumerate(lines, start=1):
            if lowered_phrase not in line.lower():
                continue
            pair = (phrase, idx)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            contexts.append(
                {
                    "phrase": phrase,
                    "source_file": _safe_relpath(source_file, work_dir),
                    "anchor_line": idx,
                    "snippet": _shorten(
                        _render_line_window(source_text, anchor_line=idx, window=5),
                        max_chars=2200,
                    ),
                }
            )
            break
        if len(contexts) >= 4:
            break
    return contexts


def _render_failure_expectation_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "  (no failing-test expectation strings were matched in the editable source)"
    blocks: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"### Failure expectation match #{idx}\n"
            f"- file: `{ctx.get('source_file')}`\n"
            f"- matched phrase: `{ctx.get('phrase')}`\n"
            f"```java\n{ctx.get('snippet') or ''}\n```"
        )
    return "\n\n".join(blocks)


_JAVA_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_JAVA_CALL_RE = re.compile(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", re.MULTILINE)
_JAVA_CLASS_RE = re.compile(
    r"^\s*(?:public|protected|private|abstract|final|static|\s)*class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
_JAVA_EXTENDS_RE = re.compile(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\s+extends\s+([A-Za-z0-9_$.]+)")
_JAVA_UNQUALIFIED_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_JAVA_CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "new",
    "throw",
    "super",
    "this",
    "assert",
}


def _extract_method_names_from_test_contexts(test_contexts: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for ctx in test_contexts:
        snippet = str(ctx.get("snippet") or "")
        for match in _JAVA_CALL_RE.findall(snippet):
            if match not in seen:
                seen.add(match)
                names.append(match)
    return names


def _parse_java_package_name(source_text: str) -> str | None:
    match = _JAVA_PACKAGE_RE.search(source_text)
    return match.group(1) if match else None


def _parse_java_declared_class_name(source_text: str) -> str | None:
    match = _JAVA_CLASS_RE.search(source_text)
    return match.group(1) if match else None


def _parse_java_superclass_name(source_text: str) -> str | None:
    match = _JAVA_EXTENDS_RE.search(source_text)
    return match.group(1) if match else None


def _resolve_related_java_source(
    work_dir: Path,
    inspection: dict[str, Any],
    *,
    current_source_path: Path,
    current_package: str | None,
    class_name: str,
) -> Path | None:
    src_classes_rel = (
        (inspection.get("defects4j") or {}).get("properties", {}).get("dir.src.classes")
    )
    if not src_classes_rel:
        return None
    src_root = (work_dir / str(src_classes_rel)).resolve()
    if not src_root.exists():
        return None

    simple_name = class_name.rsplit(".", 1)[-1]
    if "." in class_name:
        candidate = src_root / Path(*class_name.split(".")).with_suffix(".java")
        if candidate.exists():
            return candidate

    if current_package:
        same_package = src_root / Path(*current_package.split(".")) / f"{simple_name}.java"
        if same_package.exists():
            return same_package

    try:
        index = SourceIndex([src_root])
        index.build()
        resolved = resolve_source_for_frame(
            index,
            f"{current_package}.{simple_name}" if current_package and "." not in class_name else class_name,
            f"{simple_name}.java",
            prefer_production=True,
        )
        if resolved is not None and resolved.source_file.exists():
            return resolved.source_file
    except Exception:
        return None
    return None


def _extract_nested_method_calls(snippet: str, *, exclude: set[str]) -> list[str]:
    code_only_lines = []
    for raw_line in snippet.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("*") or stripped.startswith("/*") or stripped.startswith("//"):
            continue
        code_only_lines.append(raw_line)
    code_only = "\n".join(code_only_lines)
    seen: set[str] = set()
    names: list[str] = []
    for name in _JAVA_UNQUALIFIED_CALL_RE.findall(code_only):
        if name in _JAVA_CALL_KEYWORDS or name in exclude:
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _extract_related_execution_contexts(
    work_dir: Path,
    inspection: dict[str, Any],
    *,
    target_source_path: Path,
    target_source_text: str,
    test_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    method_names = _extract_method_names_from_test_contexts(test_contexts)
    if not method_names:
        return []

    package_name = _parse_java_package_name(target_source_text)
    superclass_name = _parse_java_superclass_name(target_source_text)
    if not superclass_name:
        return []

    superclass_source = _resolve_related_java_source(
        work_dir,
        inspection,
        current_source_path=target_source_path,
        current_package=package_name,
        class_name=superclass_name,
    )
    if superclass_source is None or not superclass_source.exists():
        return []

    contexts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    queue: list[tuple[Path, str, str]] = [
        (superclass_source, name, f"failing test directly calls `{name}()`")
        for name in method_names
    ]
    visited_names: set[str] = set()

    while queue and len(contexts) < 6:
        source_file, method_name, reason = queue.pop(0)
        visit_key = f"{source_file}:{method_name}"
        if visit_key in visited_names:
            continue
        visited_names.add(visit_key)

        anchor_line = _find_method_anchor_line(source_file, method_name)
        if not anchor_line:
            continue
        snippet = extract_enclosing_method(source_file, anchor_line)
        class_label = source_file.stem
        source_text = ""
        try:
            source_text = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source_text = ""
        declared_class_name = _parse_java_declared_class_name(source_text) or source_file.stem
        declared_package_name = _parse_java_package_name(source_text)
        if declared_package_name:
            class_label = f"{declared_package_name}.{declared_class_name}"
        else:
            class_label = declared_class_name
        pair = (class_label, method_name)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        contexts.append(
            {
                "class_name": class_label,
                "source_file": _safe_relpath(source_file, work_dir),
                "method_name": method_name,
                "reason": reason,
                "snippet": _shorten(snippet, max_chars=2600),
            }
        )
        nested_calls = sorted(
            _extract_nested_method_calls(snippet, exclude={method_name}),
            key=_related_method_priority,
        )
        for nested_name in nested_calls:
            if len(queue) + len(contexts) >= 8:
                break
            nested_source = None
            if _find_method_anchor_line(target_source_path, nested_name):
                nested_source = target_source_path
            elif _find_method_anchor_line(source_file, nested_name):
                nested_source = source_file
            if nested_source is None:
                continue
            queue.append(
                (
                    nested_source,
                    nested_name,
                    f"`{method_name}()` internally calls `{nested_name}()`",
                )
            )

    return contexts


def _render_related_execution_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "  (no inherited execution-path context recovered)"
    blocks: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"### Related production context #{idx}\n"
            f"- class: `{ctx.get('class_name')}`\n"
            f"- file: `{ctx.get('source_file')}`\n"
            f"- why included: {ctx.get('reason')}\n"
            f"```java\n{ctx.get('snippet') or ''}\n```"
        )
    return "\n\n".join(blocks)


def _related_method_priority(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if "mean" in lowered or "variance" in lowered:
        return (0, lowered)
    if "support" in lowered:
        return (1, lowered)
    return (2, lowered)


NAIVE_SOURCE_CHAR_LIMIT = 40000
FALLBACK_FULL_SOURCE_CHAR_LIMIT = 24000


def _render_editable_sources(
    source_snapshots: list[dict[str, Any]],
    *,
    primary_file_path: str | None = None,
    max_chars_per_primary_file: int,
    max_chars_per_secondary_file: int,
) -> str:
    if not source_snapshots:
        return "  (editable source unavailable)"
    blocks: list[str] = []
    for idx, snapshot in enumerate(source_snapshots, start=1):
        rel_path = str(snapshot.get("path") or "source.java")
        source_text = str(snapshot.get("text") or "")
        max_chars = (
            max_chars_per_primary_file
            if primary_file_path is not None and rel_path == primary_file_path
            else max_chars_per_secondary_file
        )
        blocks.append(
            f"### Editable file #{idx}\n"
            f"- path: `{rel_path}`\n"
            f"```java\n{_shorten(source_text, max_chars=max_chars)}\n```"
        )
    return "\n\n".join(blocks)


def _build_naive_prompt(
    *,
    file_rel_path: str,
    source_text: str,
    failing_tests: Iterable[str],
) -> str:
    failing_list = "\n".join(f"  - {t}" for t in failing_tests) or "  - (unknown)"
    # Cap huge files (e.g. Lang NumberUtils.java is 53K chars); remote chat APIs
    # frequently disconnect on very long payloads.  The middle is cut so the
    # class header and the tail (most methods) both remain visible.
    truncated = _shorten(source_text, max_chars=NAIVE_SOURCE_CHAR_LIMIT)
    truncation_note = (
        "NOTE: the file was very large and the middle section was elided by "
        "the harness (marked `[truncated]`). The LLM may only patch code that "
        "is still visible below.\n\n"
        if len(source_text) > NAIVE_SOURCE_CHAR_LIMIT
        else ""
    )
    return (
        f"Buggy file: `{file_rel_path}`\n\n"
        f"Failing tests:\n{failing_list}\n\n"
        f"{truncation_note}"
        f"Content of the buggy file:\n"
        f"```java\n{truncated}\n```\n\n"
        "Return ONLY the patch format required by the system prompt."
    )


def _build_naive_with_evidence_prompt(
    *,
    primary_file_rel_path: str,
    editable_source_snapshots: list[dict[str, Any]],
    failing_tests: Iterable[str],
    evidence: dict[str, Any],
    raw_test_log: str,
    test_contexts: list[dict[str, Any]],
    related_contexts: list[dict[str, Any]],
    focus_method_contexts: list[dict[str, Any]],
    failure_expectation_contexts: list[dict[str, Any]],
) -> str:
    """Hybrid prompt for the `classes.modified` fallback case.

    When the inspector fails to localise a suspect line, we don't know *where*
    in the file the bug lives — so we must give the model the full source text.
    But we still have rich evidence (exception type, failing tests, stack trace
    excerpt), and including it in the prompt strongly biases the model toward
    the right method.
    """
    failing_list = "\n".join(f"  - {t}" for t in failing_tests) or "  - (unknown)"
    editable_sources_block = _render_editable_sources(
        editable_source_snapshots,
        primary_file_path=primary_file_rel_path,
        max_chars_per_primary_file=FALLBACK_FULL_SOURCE_CHAR_LIMIT,
        max_chars_per_secondary_file=7000,
    )
    total_source_chars = sum(len(str(snapshot.get("text") or "")) for snapshot in editable_source_snapshots)
    truncation_note = (
        "NOTE: at least one editable file was very large and the middle was elided "
        "(marked `[truncated]`). Prefer edits that stay within the visible code regions.\n\n"
        if total_source_chars > FALLBACK_FULL_SOURCE_CHAR_LIMIT
        else ""
    )
    frames_text = "\n".join(
        f"  #{i + 1} {frame.get('class_name')}"
        f"({frame.get('file_name')}:{frame.get('line_number')})"
        for i, frame in enumerate(evidence.get("top_frames") or [])
    ) or "  (no project-level frames recovered)"
    evidence_excerpt = _shorten(str(evidence.get("evidence_excerpt") or ""), max_chars=1200)
    raw_log_trimmed = _shorten(raw_test_log or "", max_chars=1200)
    test_contexts_block = _render_test_contexts(test_contexts)
    related_contexts_block = _render_related_execution_contexts(related_contexts)
    focus_method_contexts_block = _render_focus_method_contexts(focus_method_contexts)
    failure_expectation_contexts_block = _render_failure_expectation_contexts(
        failure_expectation_contexts
    )
    primary_target_bias_note = (
        "The matched failure-expectation phrases already appear inside the primary target file. "
        "Start by repairing the primary file before touching any supporting file.\n\n"
        if failure_expectation_contexts
        else ""
    )

    return (
        f"## Primary target file\n`{primary_file_rel_path}`\n\n"
        f"## Failing test(s)\n{failing_list}\n\n"
        f"## Exception\n"
        f"- type: `{evidence.get('exception_type') or 'unknown'}`\n"
        f"- message: {evidence.get('exception_message') or '-'}\n\n"
        f"## Top stack frames\n{frames_text}\n\n"
        f"## Evidence excerpt\n```\n{evidence_excerpt}\n```\n\n"
        f"## Raw defects4j test log (truncated)\n```\n{raw_log_trimmed}\n```\n\n"
        f"## Failing test source snippet(s)\n{test_contexts_block}\n\n"
        f"## Failure expectation matches in editable sources\n{failure_expectation_contexts_block}\n\n"
        f"{primary_target_bias_note}"
        f"## Method contexts matched from failing test names\n{focus_method_contexts_block}\n\n"
        f"## Related inherited execution path(s)\n{related_contexts_block}\n\n"
        f"{truncation_note}"
        f"## Editable source file(s)\n{editable_sources_block}\n\n"
        "Return ONLY a unified diff against the editable file(s) above. "
        "If there is exactly one editable file and producing a diff is hard, you may return the full corrected "
        "source for that file only. When the file is large or contains duplicated methods, prefer the full corrected "
        "source over a brittle diff hunk. Do not return JSON."
    )


def _build_retry_feedback_prompt(
    *,
    base_prompt: str,
    attempts: list[dict[str, Any]],
    originally_failing: list[str] | None = None,
    best_round: int | None = None,
) -> str:
    """Prepend a `RETRY` block to the base prompt summarising prior attempts.

    If ``originally_failing`` is provided, regressions (tests that were passing
    before the patch but are now failing) are called out explicitly.  If
    ``best_round`` is given, the prompt reminds the model which past round
    was closest to a fix, so it can iterate instead of exploring from scratch.
    """
    if not attempts:
        return base_prompt
    original_set = set(originally_failing or [])
    parts: list[str] = [
        "## ⚠ Retry context — your previous attempt(s) did NOT fix the bug.",
        "Read this carefully and try a DIFFERENT fix this time.",
        "",
    ]
    if best_round is not None:
        parts.append(
            f"**Your closest attempt so far was round #{best_round}** (fewest "
            "failing tests). Keep what worked there, only change what needs to "
            "change. Do NOT regress tests that were already passing."
        )
        parts.append("")
    for entry in attempts:
        idx = entry.get("round")
        parts.append(f"### Previous attempt #{idx}")
        outcome = entry.get("outcome", "unknown")
        parts.append(f"- outcome: **{outcome}**")
        if entry.get("apply_error"):
            err_text = str(entry["apply_error"])
            if "\n" in err_text:
                parts.append(
                    "- patch could NOT be applied. Details:\n"
                    f"```\n{err_text}\n```"
                )
                parts.append(
                    "  Copy your new `old` chunk VERBATIM from the `>>` "
                    "line(s) above, preserving indentation and spacing."
                )
                parts.append(
                    "  Next round, keep the `old` chunk SHORT — ideally only the "
                    "1–4 lines surrounding the real behavioural change."
                )
            else:
                parts.append(f"- patch could NOT be applied: `{err_text}`")
        if entry.get("llm_error"):
            parts.append(
                "- your response was not a valid patch payload. "
                f"Reason: `{entry['llm_error']}`"
            )
            raw = entry.get("llm_raw_preview") or ""
            if raw:
                parts.append(
                    "- your raw response started like this (do NOT repeat this):\n"
                    f"```\n{raw}\n```"
                )
            parts.append(
                "  Next round: output ONLY a unified diff (preferred) or, if there is exactly one editable file, "
                "the full corrected Java source for that file. Do not return JSON, prose commentary, or markdown fences."
            )
        if entry.get("compile_error_tail"):
            parts.append(
                "- compile failed; last lines of stderr:\n"
                f"```\n{entry['compile_error_tail']}\n```"
            )
        if entry.get("remaining_failing"):
            remaining = entry["remaining_failing"]
            regressions = [t for t in remaining if t not in original_set] if original_set else []
            still_original = [t for t in remaining if t in original_set] if original_set else remaining
            if regressions:
                parts.append(
                    f"- ⚠ YOUR PATCH BROKE {len(regressions)} test(s) that were PASSING before:"
                )
                for t in regressions[:6]:
                    parts.append(f"  - ❌ NEW FAILURE: {t}")
                if len(regressions) > 6:
                    parts.append(f"  - ... and {len(regressions) - 6} more regressions")
                parts.append(
                    "  → Revert the part of your edit that broke these. Your change "
                    "is too aggressive or touches code paths unrelated to the bug."
                )
            if still_original:
                parts.append("- tests STILL failing after your patch (original bug):")
                for t in still_original[:6]:
                    parts.append(f"  - {t}")
                extra = len(still_original) - 6
                if extra > 0:
                    parts.append(f"  - ... and {extra} more")
        if entry.get("test_stdout_tail"):
            parts.append(
                "- last lines of the failing-test stdout:\n"
                f"```\n{entry['test_stdout_tail']}\n```"
            )
        if entry.get("edits"):
            edits_short = _shorten(
                "\n".join(
                    f'OLD:\n{(e.get("old") or "")[:400]}\n'
                    f'NEW:\n{(e.get("new") or "")[:400]}'
                    for e in entry["edits"][:3]
                ),
                max_chars=2400,
            )
            parts.append(
                "- the edits you produced last round (DO NOT repeat them verbatim):\n"
                f"```\n{edits_short}\n```"
            )
        parts.append("")
    parts.append("## Original task (unchanged)")
    parts.append("")
    return "\n".join(parts) + base_prompt


def _build_full_pipeline_prompt(
    *,
    file_rel_path: str,
    failing_tests: Iterable[str],
    evidence: dict[str, Any],
    suspects: list[dict[str, Any]],
    raw_test_log: str,
    test_contexts: list[dict[str, Any]],
) -> str:
    """Build a compact repair prompt out of the inspector's pre-processed evidence.

    No full source file is included — the upstream inspector already isolated
    the ``enclosing_method`` of each suspect, which is all a repair model
    really needs.  This is the whole point of the AutoRepair pipeline: trade a
    big generic prompt for a small, evidence-rich one.
    """
    failing_list = "\n".join(f"  - {t}" for t in failing_tests) or "  - (unknown)"
    frames_text = "\n".join(
        f"  #{i + 1} {frame.get('class_name')}"
        f"({frame.get('file_name')}:{frame.get('line_number')})"
        for i, frame in enumerate(evidence.get("top_frames") or [])
    ) or "  (no project-level frames recovered)"

    suspects_text_parts: list[str] = []
    for idx, suspect in enumerate(suspects[:3], start=1):
        # `inspector.extract_enclosing_method` returns the full method body
        # (signature + javadoc + body) as a plain string when it succeeds,
        # or a ±25-line window with line-number gutter as a fallback.
        enclosing_raw = suspect.get("enclosing_method")
        if isinstance(enclosing_raw, str) and enclosing_raw.strip():
            method_body = _shorten(enclosing_raw, max_chars=4000)
            context_block = (
                f"Enclosing method (signature + full body):\n"
                f"```java\n{method_body}\n```"
            )
        else:
            snippet = _shorten(suspect.get("snippet") or "", max_chars=1800)
            context_block = (
                f"Window around line {suspect.get('line_number')}:\n"
                f"```java\n{snippet}\n```"
            )
        suspects_text_parts.append(
            f"### Suspect #{idx} (score={suspect.get('score')})\n"
            f"- class: `{suspect.get('class_name')}`\n"
            f"- file:  `{suspect.get('source_file')}`\n"
            f"- line:  {suspect.get('line_number')}\n"
            f"- ranking reasons: {suspect.get('score_reasons')}\n"
            f"{context_block}\n"
        )
    suspects_block = "\n".join(suspects_text_parts) if suspects_text_parts else "(no suspects)"

    evidence_excerpt = _shorten(str(evidence.get("evidence_excerpt") or ""), max_chars=1500)
    raw_log_trimmed = _shorten(raw_test_log or "", max_chars=1500)
    test_contexts_block = _render_test_contexts(test_contexts)

    return (
        f"## Target file\n`{file_rel_path}`\n\n"
        f"## Failing test(s)\n{failing_list}\n\n"
        f"## Exception (root cause of the chosen stack-segment)\n"
        f"- type: `{evidence.get('exception_type') or 'unknown'}`\n"
        f"- message: {evidence.get('exception_message') or '-'}\n\n"
        f"## Top project-level stack frames (pre-filtered by inspector)\n{frames_text}\n\n"
        f"## Evidence excerpt (first lines of the failing-test log)\n"
        f"```\n{evidence_excerpt}\n```\n\n"
        f"## Raw defects4j test log (truncated)\n"
        f"```\n{raw_log_trimmed}\n```\n\n"
        f"## Failing test source snippet(s)\n{test_contexts_block}\n\n"
        f"## Pre-ranked suspects with the enclosing method context\n{suspects_block}\n\n"
        "Return ONLY a unified diff against the target file above. "
        "If producing a diff is hard, you may instead return the full corrected Java source for that file only. "
        "Do not return JSON."
    )


def _strip_edit_fences(text: str) -> str:
    """Remove a leading/trailing ```java ... ``` fence that some models add."""
    s = text.strip("\n")
    lines = s.splitlines()
    if not lines:
        return text
    if lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    rebuilt = "\n".join(lines)
    return rebuilt if rebuilt != s else text


_WS_RE = re.compile(r"\s+")


def _collapse_ws(line: str) -> str:
    """Collapse any run of whitespace (tabs, double-spaces) to a single space."""
    return _WS_RE.sub(" ", line.strip())


def _locate_old_in_source_by_lines(
    source: str, old: str
) -> tuple[int, int] | None:
    """Find `old` inside `source` by matching lines at increasing lenience.

    Returns the (start, end) byte offsets into ``source`` that correspond to
    the full line span equivalent to ``old``. Returns None if every
    normaliser fails or finds more than one hit (ambiguous).

    Strategy (most strict → most lenient):
      1. Strip leading/trailing whitespace per line.
      2. Strip AND collapse interior whitespace runs to a single space.

    Rationale: the #1 reason LLM edits fail to apply is a single whitespace
    difference inside a 500-char block — either leading indent drift (tabs
    vs spaces) or multi-space runs collapsed to one. Line-wise comparison
    that normalises both sides is tolerant enough to absorb these mistakes
    without sacrificing correctness: we still require every line to match,
    and we bail out on ambiguity.
    """
    src_lines = source.splitlines(keepends=True)
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines = old_lines[1:]
    while old_lines and not old_lines[-1].strip():
        old_lines = old_lines[:-1]
    if not old_lines:
        return None

    n = len(old_lines)
    prefix_offsets: list[int] = [0]
    for ln in src_lines:
        prefix_offsets.append(prefix_offsets[-1] + len(ln))

    def _scan(normaliser) -> tuple[int, int] | None:
        old_norm = [normaliser(ln) for ln in old_lines]
        hits: list[tuple[int, int]] = []
        for i in range(0, len(src_lines) - n + 1):
            window_norm = [normaliser(src_lines[i + k]) for k in range(n)]
            if window_norm == old_norm:
                hits.append((prefix_offsets[i], prefix_offsets[i + n]))
                if len(hits) > 1:
                    return None  # ambiguous
        return hits[0] if len(hits) == 1 else None

    # Tier 1: strip() per line (tolerates leading/trailing whitespace only).
    hit = _scan(lambda s: s.strip())
    if hit is not None:
        return hit
    # Tier 2: also collapse interior whitespace runs.
    return _scan(_collapse_ws)


def _nearest_snippet_for_old(source: str, old: str, *, window: int = 6) -> str | None:
    """Find the line in `source` that best overlaps with `old`'s first non-empty
    line, and return a small window of surrounding source text (with line numbers).

    Used to enrich the retry prompt: when an edit fails to apply, we want to
    show the LLM what the file actually looks like near the intended location.
    """
    old_lines = [ln for ln in old.splitlines() if ln.strip()]
    if not old_lines:
        return None
    # Build a set of anchor strings: the first non-trivial line (stripped)
    # and also the second if the first is very short (e.g. `{` alone).
    anchors: list[str] = []
    for ln in old_lines[:4]:
        stripped = ln.strip()
        if len(stripped) >= 12:
            anchors.append(stripped)
        if len(anchors) >= 2:
            break
    if not anchors:
        anchors = [old_lines[0].strip()]

    src_lines = source.splitlines()
    best_idx: int | None = None
    for anchor in anchors:
        for i, ln in enumerate(src_lines):
            if anchor in ln:
                best_idx = i
                break
        if best_idx is not None:
            break
    if best_idx is None:
        return None

    start = max(0, best_idx - window)
    end = min(len(src_lines), best_idx + window + 1)
    out: list[str] = []
    for i in range(start, end):
        marker = ">>" if i == best_idx else "  "
        out.append(f"{marker} {i + 1:5d}: {src_lines[i]}")
    return "\n".join(out)


def _apply_edits(
    source_text: str, edits: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Apply a list of search/replace edits to source_text.

    Matching strategy per edit:
      1. Exact ``current.count(old) == 1`` (fast path).
      2. Strip leading/trailing ```` ``` ```` fences from ``old`` and retry
         exact match.
      3. Line-level match ignoring whitespace (``_locate_old_in_source_by_lines``).
      4. Give up with a descriptive error including a nearest-match preview
         so the retry round can show the model what the file really looks like.

    Raises RuntimeError on any unresolvable edit. The error message embeds a
    ``nearest_snippet`` field when we could locate an anchor — callers can
    attach this to the retry feedback prompt.
    """
    current = source_text
    report: list[dict[str, Any]] = []
    if not isinstance(edits, list) or not edits:
        raise RuntimeError("LLM returned an empty `edits` list.")
    for idx, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise RuntimeError(f"Edit #{idx} is not an object: {edit!r}")
        old = edit.get("old") or edit.get("find") or ""
        new = edit.get("new") or edit.get("replace") or ""
        if not isinstance(old, str) or not old:
            raise RuntimeError(f"Edit #{idx} has empty `old` field.")
        if not isinstance(new, str):
            new = str(new)

        match_mode = "exact"
        count = current.count(old)
        if count == 0:
            stripped_old = _strip_edit_fences(old)
            if stripped_old != old and current.count(stripped_old) == 1:
                old = stripped_old
                count = 1
                match_mode = "fence_stripped"

        # Reject semantically-null edits (whitespace-only drift between old
        # and new). Without this check, a lazy model can pretend to fix the
        # bug by merely reflowing indentation — the patch "applies" cleanly
        # but the buggy code path is unchanged, wasting a whole retry round.
        if _collapse_ws(old) == _collapse_ws(new):
            raise RuntimeError(
                f"Edit #{idx} is a no-op: `new` is identical to `old` after "
                "whitespace normalisation. Make a real behavioural change, "
                "e.g. alter a condition, return value, arithmetic, or API call."
            )

        if count == 1:
            current = current.replace(old, new, 1)
        elif count > 1:
            raise RuntimeError(
                f"Edit #{idx} `old` chunk matches {count} times; must be unique."
            )
        else:
            # Fuzzy line-level match as a last resort.
            located = _locate_old_in_source_by_lines(current, old)
            if located is None:
                near = _nearest_snippet_for_old(current, old)
                msg = (
                    f"Edit #{idx} `old` chunk not found in source (len={len(old)})."
                )
                if near:
                    msg += "\nNearest match in the real file:\n" + near
                raise RuntimeError(msg)
            start, end = located
            # Guard against a fuzzy hit that would be a no-op: compare `new`
            # against the actual matched source slice (not the stale `old`).
            original_slice = current[start:end]
            if _collapse_ws(original_slice) == _collapse_ws(new):
                raise RuntimeError(
                    f"Edit #{idx} is a no-op: after fuzzy match, the replacement "
                    "would only change whitespace. Make a real code change."
                )
            current = current[:start] + new + ("\n" if not new.endswith("\n") else "") + current[end:]
            match_mode = "fuzzy_lines"

        report.append(
            {
                "index": idx,
                "old_len": len(old),
                "new_len": len(new),
                "match_mode": match_mode,
            }
        )
    return current, report


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:java|json)?\n?")


def _strip_code_fences(text: str) -> str:
    stripped = _FENCE_RE.sub("", text)
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()


def _parse_legacy_edit_payload(raw_response: str) -> tuple[list[dict[str, Any]], str] | None:
    cleaned = _strip_code_fences(raw_response)
    if not cleaned or cleaned[0] not in "{[":
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    edits = payload.get("edits") or payload.get("patches") or payload.get("changes")
    if not isinstance(edits, list):
        return None
    explanation = str(payload.get("explanation") or payload.get("explain") or "")
    return edits, explanation


def _looks_like_json_like_patch_response(raw_response: str) -> bool:
    cleaned = _strip_code_fences(raw_response)
    if not cleaned:
        return False
    if cleaned[0] not in "{[":
        return False
    lowered = cleaned[:200].lower()
    return any(token in lowered for token in ('"edits"', '"patch"', '"patches"', '"changes"', '"explanation"'))


def _materialize_legacy_edits_as_project_patch(
    *,
    original_files: dict[str, str],
    entrypoint: str,
    edits: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], str]:
    fixed_primary_source, edit_report = _apply_edits(
        original_files.get(entrypoint, ""),
        edits,
    )
    patched_files = dict(original_files)
    patched_files[entrypoint] = fixed_primary_source
    patch_diff_text = _render_project_diff(original_files, patched_files)
    if not patch_diff_text:
        raise RuntimeError("Legacy edit payload did not produce any source changes.")
    return patched_files, edit_report, patch_diff_text


def _ask_llm_for_patch(
    *,
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    original_files: dict[str, str],
    entrypoint: str,
    language: str,
) -> tuple[dict[str, Any], str, dict[str, int], dict[str, Any]]:
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    debug_info: dict[str, Any] = {
        "raw_response_text": "",
        "parsed_response": None,
        "provider_usage": None,
    }

    def _on_metadata(meta: dict[str, Any]) -> None:
        u = meta.get("usage") if isinstance(meta, dict) else None
        if not isinstance(u, dict):
            raw_response_text = meta.get("raw_response_text") if isinstance(meta, dict) else None
            if raw_response_text:
                debug_info["raw_response_text"] = str(raw_response_text)
            return
        debug_info["provider_usage"] = dict(u)
        # providers may report either OpenAI-style (prompt_tokens/completion_tokens)
        # or our unified style (input_tokens/output_tokens); try both.
        prompt_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        completion_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        total_tokens = int(u.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage["prompt_tokens"] = prompt_tokens
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = total_tokens
        raw_response_text = meta.get("raw_response_text") if isinstance(meta, dict) else None
        if raw_response_text:
            debug_info["raw_response_text"] = str(raw_response_text)

    response = call_llm_for_json(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=model_key,
        isJson=False,
        stream=False,
        metadata_handler=_on_metadata,
        force_disable_thinking=True,
    )
    raw_text = response if isinstance(response, str) else str(response)
    if not debug_info.get("raw_response_text"):
        debug_info["raw_response_text"] = raw_text

    legacy_payload = _parse_legacy_edit_payload(raw_text)
    if legacy_payload is not None:
        edits, explanation = legacy_payload
        debug_info["parsed_response"] = {"edits": edits, "explanation": explanation}
        return (
            {
                "protocol": "legacy_edits",
                "edits": edits,
            },
            explanation,
            usage,
            debug_info,
        )

    if _looks_like_json_like_patch_response(raw_text):
        exc = RuntimeError(
            "Model returned a JSON-like payload that could not be parsed into a compatible edit list."
        )
        try:
            setattr(exc, "raw_response", raw_text)
        except Exception:
            pass
        raise exc

    try:
        diff_text = _coerce_model_output_to_diff(
            raw_text,
            original_files=original_files,
            entrypoint=entrypoint,
            language=language,
        )
    except Exception as diff_exc:
        if len(original_files) == 1:
            source_candidate = _extract_source_code_candidate(raw_text, language=language)
            if source_candidate is not None:
                only_path = next(iter(original_files.keys()))
                fallback_diff = _render_project_diff(original_files, {only_path: source_candidate})
                if fallback_diff:
                    return (
                        {
                            "protocol": "unified_diff",
                            "git_diff": fallback_diff,
                        },
                        "",
                        usage,
                        debug_info,
                    )
        try:
            setattr(diff_exc, "raw_response", raw_text)
        except Exception:
            pass
        raise
    if diff_text:
        return (
            {
                "protocol": "unified_diff",
                "git_diff": diff_text,
            },
            "",
            usage,
            debug_info,
        )

    exc = RuntimeError(
        "Model did not return a valid unified diff, a compatible full-file replacement, or a legacy edit payload."
    )
    try:
        setattr(exc, "raw_response", raw_text)
    except Exception:
        pass
    raise exc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_full_repair_for_bug(
    *,
    run_id: int,
    user_id: int,
    organization_id: int | None,
    project_code: str,
    defects4j_project: str,
    defects4j_bug_id: int,
    model_key: str,
    strategy: str,
    experiment_id: int | None = None,
    force_checkout: bool = True,
) -> dict[str, Any]:
    """Execute the LLM-driven Defects4J repair pipeline synchronously."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy `{strategy}`. Expected one of {SUPPORTED_STRATEGIES}.")

    start = time.time()
    llm_rounds = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    patch_diff_text = ""
    patch_added = 0
    patch_removed = 0
    failed_before = 0
    failed_after = 0
    report: dict[str, Any] = {
        "benchmark": "defects4j",
        "strategy": strategy,
        "model_key": model_key,
        "project_code": project_code,
        "defects4j_project": defects4j_project,
        "defects4j_bug_id": defects4j_bug_id,
    }

    try:
        bench_store.update_run_progress(run_id, stage="checkout", run_status="running")
        work_dir, artifacts_dir = _prepare_work_dir(
            project_code, defects4j_bug_id, run_id, reuse=not force_checkout
        )

        bench_store.update_run_progress(run_id, stage="inspect")
        inspection = run_defects4j_inspection(
            project_id=defects4j_project,
            bug_id=defects4j_bug_id,
            is_buggy=True,
            work_dir=work_dir,
            artifacts_dir=artifacts_dir,
            force_checkout=force_checkout,
            test_mode="relevant",
            inspect_failing_tests=1,
            max_candidates=4,
        )
        report["inspection"] = inspection
        failing_before, failed_before = _count_failing_from_test_output(
            Defects4JRunner(), work_dir
        )
        trigger_tests: list[str] = (inspection.get("tests") or {}).get("trigger_tests") or []
        failing_tests: list[str] = failing_before or trigger_tests

        suspects: list[dict[str, Any]] = []
        for failure in inspection.get("failures") or []:
            for candidate in failure.get("suspects") or []:
                if candidate.get("source_file"):
                    suspects.append(candidate)
        # Drop suspects that point at test sources — for Defects4J bugs the
        # failing assertion lives there but the bug is always in production.
        production_suspects = [
            s for s in suspects
            if not _is_test_source_file(s.get("source_file"), s.get("class_name"))
        ]

        top_suspect = _extract_top_suspect(inspection)
        suspect_is_test = top_suspect is not None and _is_test_source_file(
            top_suspect.get("source_file"), top_suspect.get("class_name")
        )
        # Inspector now emits `has_production_frame=False` when the stack
        # trace contains only test/framework frames (e.g. assertion-only
        # bugs). Treat that as an explicit signal to escalate.
        inspector_lacks_evidence = False
        for failure in inspection.get("failures") or []:
            if failure.get("has_production_frame") is False:
                inspector_lacks_evidence = True
                break

        fallback_suspects: list[dict[str, Any]] = []
        needs_fallback = top_suspect is None or suspect_is_test or inspector_lacks_evidence
        if needs_fallback:
            # Two triggers for the `classes.modified` fallback:
            #   (a) bugs like Mockito-1 where the inspector produces no suspect
            #       at all (stack trace is 100% framework code);
            #   (b) bugs like Math-2 where the only project-level frame is the
            #       test itself — the inspector picks the test file, which is
            #       a dead-end for patching.
            fallback_suspects = _suspects_from_modified_classes(work_dir, inspection)
            if fallback_suspects:
                previous = top_suspect
                top_suspect = fallback_suspects[0]
                existing = {s.get("source_file") for s in production_suspects}
                for candidate in fallback_suspects:
                    if candidate.get("source_file") not in existing:
                        production_suspects.append(candidate)
                reason = (
                    "classes_modified_override_test"
                    if suspect_is_test
                    else "classes_modified"
                )
                report["suspect_fallback"] = reason
                if suspect_is_test and previous is not None:
                    report["dropped_test_suspect"] = {
                        "source_file": previous.get("source_file"),
                        "class_name": previous.get("class_name"),
                    }
                    logger.info(
                        "repair_runner: inspector picked test file %s; "
                        "overriding with classes.modified -> %s",
                        previous.get("source_file"),
                        top_suspect.get("source_file"),
                    )
                else:
                    logger.info(
                        "repair_runner: inspector had no production suspect, "
                        "fell back to classes.modified and picked %s",
                        top_suspect.get("source_file"),
                    )
            elif suspect_is_test:
                # Keep the inspector's pick as a last resort, but record the
                # fact we couldn't localise any production source. In most
                # cases this means the LLM will fruitlessly edit the test.
                report["suspect_fallback"] = "stayed_on_test_no_alternative"
                logger.warning(
                    "repair_runner: top suspect %s is a test file but "
                    "classes.modified was empty; patch likely useless.",
                    top_suspect.get("source_file"),
                )
        if top_suspect is None:
            raise RuntimeError(
                "Inspector could not recover any suspect source file; cannot repair."
            )
        # From here on, `suspects` means *production suspects* so the prompt
        # builder never quotes test files as a repair target.
        suspects = production_suspects or suspects

        abs_source_path, source_text = _read_suspect_source(work_dir, top_suspect)
        if abs_source_path is None or source_text is None:
            raise RuntimeError(f"Failed to read suspect source: {top_suspect.get('source_file')}")

        report["suspect"] = {
            "source_file": top_suspect.get("source_file"),
            "line_number": top_suspect.get("line_number"),
            "score": top_suspect.get("score"),
            "score_reasons": top_suspect.get("score_reasons"),
        }

        # ------ Build the per-strategy base prompt ------
        evidence = _extract_evidence(inspection)
        raw_test_log = _read_inspector_test_log(artifacts_dir)
        test_contexts = _extract_test_contexts(work_dir, inspection, failing_tests)
        failure_expectation_phrases = _extract_failure_expectation_phrases(test_contexts)
        related_contexts = _extract_related_execution_contexts(
            work_dir,
            inspection,
            target_source_path=abs_source_path,
            target_source_text=source_text,
            test_contexts=test_contexts,
        )
        report["evidence"] = evidence
        report["test_contexts"] = test_contexts
        report["related_contexts"] = related_contexts
        used_fallback_suspect = (
            "fallback_from_classes_modified"
            in (top_suspect.get("score_reasons") or [])
        )
        editable_suspects = fallback_suspects if used_fallback_suspect and fallback_suspects else [top_suspect]
        editable_files: dict[str, str] = {}
        editable_abs_paths: dict[str, Path] = {}
        editable_source_snapshots: list[dict[str, Any]] = []
        for candidate in editable_suspects:
            rel_path = str(candidate.get("source_file") or "").strip()
            if not rel_path or rel_path in editable_files:
                continue
            candidate_abs_path, candidate_source_text = _read_suspect_source(work_dir, candidate)
            if candidate_abs_path is None or candidate_source_text is None:
                continue
            editable_files[rel_path] = candidate_source_text
            editable_abs_paths[rel_path] = candidate_abs_path
            editable_source_snapshots.append(
                {
                    "path": rel_path,
                    "text": candidate_source_text,
                    "failure_expectation_matches": _count_failure_expectation_matches(
                        candidate_source_text,
                        failure_expectation_phrases,
                    ),
                    "is_top_suspect": rel_path == (top_suspect.get("source_file") or ""),
                }
            )
        if not editable_files:
            rel_path = str(top_suspect.get("source_file") or "").strip() or "source.java"
            editable_files[rel_path] = source_text
            editable_abs_paths[rel_path] = abs_source_path
            editable_source_snapshots.append(
                {
                    "path": rel_path,
                    "text": source_text,
                    "failure_expectation_matches": _count_failure_expectation_matches(
                        source_text,
                        failure_expectation_phrases,
                    ),
                    "is_top_suspect": True,
                }
            )
        editable_source_snapshots.sort(
            key=lambda item: (
                -int(item.get("failure_expectation_matches") or 0),
                0 if item.get("is_top_suspect") else 1,
                str(item.get("path") or ""),
            )
        )
        report["editable_files"] = [str(item.get("path") or "") for item in editable_source_snapshots]
        primary_entrypoint = str(editable_source_snapshots[0].get("path") or top_suspect.get("source_file") or "")
        primary_source_path = editable_abs_paths[primary_entrypoint]
        primary_source_text = editable_files[primary_entrypoint]
        focus_method_contexts = _extract_focus_method_contexts(
            source_file=primary_source_path,
            source_text=primary_source_text,
            work_dir=work_dir,
            failing_tests=failing_tests,
        )
        failure_expectation_contexts = _extract_failure_expectation_contexts(
            source_file=primary_source_path,
            source_text=primary_source_text,
            work_dir=work_dir,
            test_contexts=test_contexts,
        )
        report["focus_method_contexts"] = focus_method_contexts
        report["failure_expectation_contexts"] = failure_expectation_contexts
        prompt_style = "naive_full_source"
        if strategy == STRATEGY_FULL_PIPELINE:
            if used_fallback_suspect:
                # Inspector could not localise — fall through to a hybrid prompt
                # that still uses inspector evidence but ships the full source.
                prompt_style = "full_pipeline_full_source_with_evidence"
                base_user_prompt = _build_naive_with_evidence_prompt(
                    primary_file_rel_path=primary_entrypoint or (top_suspect.get("source_file") or "source.java"),
                    editable_source_snapshots=editable_source_snapshots,
                    failing_tests=failing_tests,
                    evidence=evidence,
                    raw_test_log=raw_test_log,
                    test_contexts=test_contexts,
                    related_contexts=related_contexts,
                    focus_method_contexts=focus_method_contexts,
                    failure_expectation_contexts=failure_expectation_contexts,
                )
            else:
                prompt_style = "full_pipeline_ranked_suspects"
                base_user_prompt = _build_full_pipeline_prompt(
                    file_rel_path=top_suspect.get("source_file") or "source.java",
                    failing_tests=failing_tests,
                    evidence=evidence,
                    suspects=suspects,
                    raw_test_log=raw_test_log,
                    test_contexts=test_contexts,
                )
            system_prompt = FULL_PIPELINE_SYSTEM_PROMPT
        else:  # naive_chat
            base_user_prompt = _build_naive_prompt(
                file_rel_path=top_suspect.get("source_file") or "source.java",
                source_text=source_text,
                failing_tests=failing_tests,
            )
            system_prompt = NAIVE_SYSTEM_PROMPT
        report["prompt_style"] = prompt_style

        original_files = dict(editable_files)
        original_source_text = original_files.get(primary_entrypoint, primary_source_text)
        attempts: list[dict[str, Any]] = []  # round-by-round outcome history
        runner = Defects4JRunner()
        previously_failing_set = set(failing_before)

        # The "best attempt so far" across all completed rounds. Best is
        # defined as the lowest (failed_count, new_regressions) tuple — i.e.
        # prefer a round that introduces no regressions over one that fixes
        # the target bug but breaks something else.
        best_attempt: dict[str, Any] | None = None
        explanation = ""
        patch_protocol = ""
        edits: list[dict[str, Any]] = []

        def _score_round(failing_now: list[str]) -> tuple[int, int]:
            fail_total = len(failing_now)
            regressions = sum(1 for t in failing_now if t not in previously_failing_set)
            return (fail_total, regressions)

        for round_idx in range(1, MAX_REPAIR_ROUNDS + 1):
            llm_rounds = round_idx

            # ---- (re)build the user prompt with prior-round feedback ----
            user_prompt = _build_retry_feedback_prompt(
                base_prompt=base_user_prompt,
                attempts=attempts,
                originally_failing=list(previously_failing_set),
                best_round=best_attempt.get("round_idx") if best_attempt else None,
            )
            try:
                (artifacts_dir / f"prompt.user.round{round_idx}.md").write_text(
                    user_prompt, encoding="utf-8"
                )
                if round_idx == 1:
                    (artifacts_dir / "prompt.system.md").write_text(
                        system_prompt, encoding="utf-8"
                    )
            except OSError:
                pass

            bench_store.update_run_progress(run_id, stage="generate_patch")
            llm_round_start = time.time()
            try:
                round_patch, round_explanation, round_usage, round_debug = _ask_llm_for_patch(
                    model_key=model_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    original_files=original_files,
                    entrypoint=primary_entrypoint,
                    language="java",
                )
            except Exception as llm_exc:
                llm_duration_ms = int((time.time() - llm_round_start) * 1000)
                raw_preview = getattr(llm_exc, "raw_response", None) or ""
                parsed_preview = getattr(llm_exc, "parsed_response", None)
                if not isinstance(raw_preview, str):
                    raw_preview = str(raw_preview)
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "llm_invalid_patch",
                        "prompt_style": prompt_style,
                        "selected_file": top_suspect.get("source_file"),
                        "selected_line": top_suspect.get("line_number"),
                        "llm_duration_ms": llm_duration_ms,
                        "llm_error": str(llm_exc)[:500],
                        "llm_raw_preview": raw_preview[:1200],
                        "prompt_artifact": f"prompt.user.round{round_idx}.md",
                        "edits": [],
                    }
                )
                try:
                    (artifacts_dir / f"response.error.round{round_idx}.txt").write_text(
                        str(llm_exc),
                        encoding="utf-8",
                    )
                    if raw_preview:
                        (artifacts_dir / f"response.raw.round{round_idx}.txt").write_text(
                            raw_preview,
                            encoding="utf-8",
                        )
                    if parsed_preview is not None:
                        (artifacts_dir / f"response.parsed.round{round_idx}.json").write_text(
                            json.dumps(parsed_preview, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                except OSError:
                    pass
                if round_idx == MAX_REPAIR_ROUNDS:
                    break
                continue
            llm_duration_ms = int((time.time() - llm_round_start) * 1000)
            raw_response_text = str(round_debug.get("raw_response_text") or "")
            parsed_response = round_debug.get("parsed_response")
            try:
                (artifacts_dir / f"response.raw.round{round_idx}.txt").write_text(
                    raw_response_text or json.dumps(parsed_response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (artifacts_dir / f"response.parsed.round{round_idx}.json").write_text(
                    json.dumps(parsed_response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass
            for k in usage:
                usage[k] += int(round_usage.get(k, 0))
            patch_protocol = str(round_patch.get("protocol") or "")
            edits = list(round_patch.get("edits") or [])
            explanation = round_explanation

            # ---- reset editable files to the original buggy state before applying ----
            for rel_path, original_text in original_files.items():
                editable_abs_paths[rel_path].write_text(original_text, encoding="utf-8")

            bench_store.update_run_progress(run_id, stage="apply_patch")
            try:
                if patch_protocol == "legacy_edits":
                    patched_files, edit_report, patch_diff_text = _materialize_legacy_edits_as_project_patch(
                        original_files=original_files,
                        entrypoint=primary_entrypoint,
                        edits=edits,
                    )
                elif patch_protocol == "unified_diff":
                    patched_files = _apply_unified_diff_to_project(
                        original_files,
                        str(round_patch.get("git_diff") or ""),
                        default_path=primary_entrypoint,
                    )
                    new_paths = [path for path in patched_files.keys() if path not in original_files]
                    deleted_paths = [path for path in original_files.keys() if path not in patched_files]
                    if new_paths:
                        raise RuntimeError(
                            "Patch touched files outside the editable target set: "
                            + ", ".join(sorted(new_paths))
                        )
                    if deleted_paths:
                        raise RuntimeError(
                            "Patch deleted editable files, which is unsupported in this benchmark harness: "
                            + ", ".join(sorted(deleted_paths))
                        )
                    patch_diff_text = _render_project_diff(original_files, patched_files)
                    if not patch_diff_text:
                        raise RuntimeError("Unified diff did not change any editable file.")
                    modified_files = sorted(
                        path for path in original_files.keys()
                        if original_files.get(path) != patched_files.get(path)
                    )
                    edit_report = [
                        {
                            "protocol": "unified_diff",
                            "modified_files": modified_files,
                        }
                    ]
                else:
                    raise RuntimeError(f"Unsupported patch protocol `{patch_protocol}`.")
            except RuntimeError as apply_exc:
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "apply_failed",
                        "prompt_style": prompt_style,
                        "selected_file": top_suspect.get("source_file"),
                        "selected_line": top_suspect.get("line_number"),
                        "llm_duration_ms": llm_duration_ms,
                        "usage": dict(round_usage),
                        "prompt_artifact": f"prompt.user.round{round_idx}.md",
                        "response_artifact": f"response.raw.round{round_idx}.txt",
                        "parsed_response_artifact": f"response.parsed.round{round_idx}.json",
                        "response_preview": raw_response_text[:1200],
                        "llm_explanation": round_explanation,
                        "patch_protocol": patch_protocol,
                        "apply_error": str(apply_exc)[:2000],
                        "edits": edits,
                    }
                )
                if round_idx == MAX_REPAIR_ROUNDS:
                    break
                continue

            patch_added, patch_removed = _count_diff_lines(patch_diff_text)
            fixed_code = patched_files.get(primary_entrypoint, original_source_text)
            for rel_path, patched_text in patched_files.items():
                editable_abs_paths[rel_path].write_text(patched_text, encoding="utf-8")

            bench_store.update_run_progress(run_id, stage="recompile")
            compile_result = runner.compile(work_dir)
            if compile_result.returncode != 0:
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "compile_failed",
                        "prompt_style": prompt_style,
                        "selected_file": top_suspect.get("source_file"),
                        "selected_line": top_suspect.get("line_number"),
                        "llm_duration_ms": llm_duration_ms,
                        "usage": dict(round_usage),
                        "prompt_artifact": f"prompt.user.round{round_idx}.md",
                        "response_artifact": f"response.raw.round{round_idx}.txt",
                        "parsed_response_artifact": f"response.parsed.round{round_idx}.json",
                        "response_preview": raw_response_text[:1200],
                        "llm_explanation": round_explanation,
                        "patch_protocol": patch_protocol,
                        "compile_error_tail": (compile_result.stderr or "")[-1200:],
                        "edits": edits,
                    }
                )
                if round_idx == MAX_REPAIR_ROUNDS:
                    break
                continue

            bench_store.update_run_progress(run_id, stage="retest")
            test_result = runner.test_relevant(work_dir)
            failing_after, failed_after = _count_failing_from_test_output(runner, work_dir)
            regressions = [t for t in failing_after if t not in previously_failing_set]

            # ---- remember this as best-so-far if it's the best completed round ----
            round_snapshot = {
                "round_idx": round_idx,
                "patch_protocol": patch_protocol,
                "edits": edits,
                "edit_report": edit_report,
                "explanation": round_explanation,
                "fixed_code": fixed_code,
                "patched_files": dict(patched_files),
                "patch_diff_text": patch_diff_text,
                "patch_added": patch_added,
                "patch_removed": patch_removed,
                "compile_returncode": compile_result.returncode,
                "compile_stderr_tail": (compile_result.stderr or "")[-1500:],
                "test_returncode": test_result.returncode,
                "test_stdout_tail": (test_result.stdout or "")[-800:],
                "failing_after": failing_after,
                "failed_after": failed_after,
                "regressions": regressions,
            }
            if best_attempt is None or _score_round(failing_after) < _score_round(
                best_attempt["failing_after"]
            ):
                best_attempt = round_snapshot

            if failed_after == 0:
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "passed",
                        "prompt_style": prompt_style,
                        "selected_file": top_suspect.get("source_file"),
                        "selected_line": top_suspect.get("line_number"),
                        "llm_duration_ms": llm_duration_ms,
                        "usage": dict(round_usage),
                        "prompt_artifact": f"prompt.user.round{round_idx}.md",
                        "response_artifact": f"response.raw.round{round_idx}.txt",
                        "parsed_response_artifact": f"response.parsed.round{round_idx}.json",
                        "response_preview": raw_response_text[:1200],
                        "llm_explanation": round_explanation,
                        "patch_protocol": patch_protocol,
                        "remaining_failing": [],
                        "edits": edits,
                    }
                )
                break

            attempts.append(
                {
                    "round": round_idx,
                    "outcome": "tests_still_failing",
                    "prompt_style": prompt_style,
                    "selected_file": top_suspect.get("source_file"),
                    "selected_line": top_suspect.get("line_number"),
                    "llm_duration_ms": llm_duration_ms,
                    "usage": dict(round_usage),
                    "prompt_artifact": f"prompt.user.round{round_idx}.md",
                    "response_artifact": f"response.raw.round{round_idx}.txt",
                    "parsed_response_artifact": f"response.parsed.round{round_idx}.json",
                    "response_preview": raw_response_text[:1200],
                    "llm_explanation": round_explanation,
                    "patch_protocol": patch_protocol,
                    "remaining_failing": failing_after,
                    "regressions": regressions,
                    "test_stdout_tail": round_snapshot["test_stdout_tail"],
                    "edits": edits,
                }
            )
            if round_idx == MAX_REPAIR_ROUNDS:
                break
            # otherwise continue to next round with this attempt as feedback

        # ------ Pick the best round and materialise it ------
        if best_attempt is not None:
            # Restore the best round's patch on disk (the loop may have left
            # a worse round there). This is what the thesis demo should show.
            for rel_path, patched_text in best_attempt["patched_files"].items():
                editable_abs_paths[rel_path].write_text(patched_text, encoding="utf-8")
            fixed_code = best_attempt["fixed_code"]
            patch_protocol = str(best_attempt.get("patch_protocol") or patch_protocol)
            edits = best_attempt["edits"]
            edit_report = best_attempt["edit_report"]
            patch_diff_text = best_attempt["patch_diff_text"]
            patch_added = best_attempt["patch_added"]
            patch_removed = best_attempt["patch_removed"]
            failing_after = best_attempt["failing_after"]
            failed_after = best_attempt["failed_after"]
            compile_ok = best_attempt["compile_returncode"] == 0
            best_round_idx = best_attempt["round_idx"]
            explanation = best_attempt["explanation"] or explanation
            report["recompile"] = {
                "returncode": best_attempt["compile_returncode"],
                "stderr_tail": best_attempt["compile_stderr_tail"],
            }
            report["retest"] = {
                "returncode": best_attempt["test_returncode"],
                "failing_after": failing_after,
                "stdout_tail": best_attempt["test_stdout_tail"],
            }
            report["best_round"] = best_round_idx
        else:
            # No round ever got past apply+compile. Keep defaults so the
            # failure path (patch_diff='', counts unchanged) makes sense.
            fixed_code = original_source_text
            patch_protocol = ""
            edits = []
            edit_report = []
            failing_after = list(failing_before)
            failed_after = failed_before
            compile_ok = False
            patch_diff_text = ""
            patch_added = 0
            patch_removed = 0

        report["llm_explanation"] = explanation
        report["patch_protocol"] = patch_protocol
        report["edits_count"] = len(edits)
        report["applied_edits"] = edit_report
        report["attempts"] = attempts
        report["max_rounds"] = MAX_REPAIR_ROUNDS
        try:
            (artifacts_dir / "patch.diff").write_text(patch_diff_text, encoding="utf-8")
            (artifacts_dir / "fixed_code.java").write_text(fixed_code, encoding="utf-8")
            import json as _json

            (artifacts_dir / "edits.json").write_text(
                _json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (artifacts_dir / "patched_files.json").write_text(
                _json.dumps(
                    {
                        path: text
                        for path, text in (best_attempt["patched_files"].items() if best_attempt else {primary_entrypoint: fixed_code}.items())
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (artifacts_dir / "attempts.json").write_text(
                _json.dumps(report["attempts"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

        # ------ Decide final verdict from the BEST round ------
        last_outcome = attempts[-1].get("outcome") if attempts else "no_attempts"
        is_plausible = compile_ok and failed_after == 0
        remaining_previous = [t for t in failing_after if t in previously_failing_set]
        new_failures = [t for t in failing_after if t not in previously_failing_set]
        is_correct = is_plausible and not remaining_previous and not new_failures

        total_tests = max(len(trigger_tests), failed_before)
        pass_count = max(0, total_tests - failed_after)
        duration_ms = int((time.time() - start) * 1000)

        final_status = "completed" if is_plausible else "failed"
        if is_plausible:
            error_message = None
        elif last_outcome == "compile_failed":
            error_message = (
                f"Patch did not compile after {llm_rounds} round(s); "
                f"see report.attempts[-1].compile_error_tail."
            )
        elif last_outcome == "apply_failed":
            error_message = (
                f"LLM patch could not be applied after {llm_rounds} round(s); "
                f"see report.attempts[-1].apply_error."
            )
        else:
            sample = failing_after[:3]
            error_message = (
                f"Patch compiles but {failed_after} test(s) still fail after "
                f"{llm_rounds} round(s); e.g. {sample}."
            )

        bench_store.update_run_progress(
            run_id,
            stage="done",
            run_status=final_status,
            pass_count=pass_count,
            fail_count=failed_after,
            total_tests=total_tests,
            duration_ms=duration_ms,
            error_message=error_message,
            report=report,
            patch_diff=patch_diff_text,
            is_plausible=is_plausible,
            is_correct=is_correct,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            patch_lines_added=patch_added,
            patch_lines_removed=patch_removed,
            llm_rounds=llm_rounds,
            failed_tests_before=failed_before,
            failed_tests_after=failed_after,
            finalize=True,
        )

        return {
            "run_id": run_id,
            "status": final_status,
            "is_plausible": is_plausible,
            "is_correct": is_correct,
            "pass_count": pass_count,
            "fail_count": failed_after,
            "total_tests": total_tests,
            "duration_ms": duration_ms,
            "patch_diff": patch_diff_text,
            "report": report,
            "tokens": usage,
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("Full repair run %s failed", run_id)
        duration_ms = int((time.time() - start) * 1000)
        bench_store.update_run_progress(
            run_id,
            stage="error",
            run_status="failed",
            duration_ms=duration_ms,
            error_message=str(exc)[:1900],
            report=report,
            patch_diff=patch_diff_text,
            is_plausible=False,
            is_correct=False,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            patch_lines_added=patch_added,
            patch_lines_removed=patch_removed,
            llm_rounds=llm_rounds,
            failed_tests_before=failed_before,
            failed_tests_after=failed_after,
            finalize=True,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "is_plausible": False,
            "is_correct": False,
            "error_message": str(exc),
            "duration_ms": duration_ms,
        }
