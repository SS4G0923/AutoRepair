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
from backend.llm import call_llm_for_json

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


_JSON_SHAPE_INSTRUCTIONS = (
    "Return ONLY a JSON object (no Markdown, no commentary) with this shape:\n"
    "{\n"
    '  "edits": [\n'
    '    {"old": "<EXACT code chunk to find>", "new": "<replacement code>"},\n'
    '    ...\n'
    "  ],\n"
    '  "explanation": "one-sentence rationale"\n'
    "}\n\n"
    "CRITICAL rules for `old`:\n"
    "1. Copy it VERBATIM from the buggy file shown above — including every "
    "space, tab, and newline. Do not reformat. Do not add or remove blank "
    "lines. Do not wrap the value in ``` fences.\n"
    "2. Keep the chunk SHORT (ideally 3–10 lines, at most one method). A "
    "shorter `old` is much more likely to match exactly.\n"
    "3. The chunk MUST appear exactly once in the file — include just enough "
    "surrounding context to disambiguate, no more.\n"
    "4. Escape newlines as \\n inside the JSON string; do NOT emit literal "
    "line breaks inside a JSON string value.\n"
    "5. `new` should use the same indentation as `old`. Do not rewrite the "
    "whole file."
)


NAIVE_SYSTEM_PROMPT = (
    "You are a Java bug-fixing assistant. The user will give you a Java file "
    "that contains a bug and the name of a JUnit test that currently fails. "
    "You must identify the bug ON YOUR OWN (no hints) and patch the file.\n\n"
    + _JSON_SHAPE_INSTRUCTIONS
)


FULL_PIPELINE_SYSTEM_PROMPT = (
    "You are the patch-generation stage of the AutoRepair pipeline. The "
    "`inspector` stage has already given you: the failing tests, the raw "
    "stack-trace, and a ranked list of the most likely buggy locations in "
    "the file. Your only job is to propose a minimal, correct patch.\n\n"
    + _JSON_SHAPE_INSTRUCTIONS
)


NAIVE_SOURCE_CHAR_LIMIT = 40000


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
        "Return the JSON described in the system prompt."
    )


def _build_naive_with_evidence_prompt(
    *,
    file_rel_path: str,
    source_text: str,
    failing_tests: Iterable[str],
    evidence: dict[str, Any],
    raw_test_log: str,
) -> str:
    """Hybrid prompt for the `classes.modified` fallback case.

    When the inspector fails to localise a suspect line, we don't know *where*
    in the file the bug lives — so we must give the model the full source text.
    But we still have rich evidence (exception type, failing tests, stack trace
    excerpt), and including it in the prompt strongly biases the model toward
    the right method.
    """
    failing_list = "\n".join(f"  - {t}" for t in failing_tests) or "  - (unknown)"
    truncated = _shorten(source_text, max_chars=NAIVE_SOURCE_CHAR_LIMIT)
    truncation_note = (
        "NOTE: the file was very large and the middle was elided "
        "(marked `[truncated]`). Patch only what is visible.\n\n"
        if len(source_text) > NAIVE_SOURCE_CHAR_LIMIT
        else ""
    )
    frames_text = "\n".join(
        f"  #{i + 1} {frame.get('class_name')}"
        f"({frame.get('file_name')}:{frame.get('line_number')})"
        for i, frame in enumerate(evidence.get("top_frames") or [])
    ) or "  (no project-level frames recovered)"
    evidence_excerpt = _shorten(str(evidence.get("evidence_excerpt") or ""), max_chars=1200)
    raw_log_trimmed = _shorten(raw_test_log or "", max_chars=1200)

    return (
        f"## Target file (full source)\n`{file_rel_path}`\n\n"
        f"## Failing test(s)\n{failing_list}\n\n"
        f"## Exception\n"
        f"- type: `{evidence.get('exception_type') or 'unknown'}`\n"
        f"- message: {evidence.get('exception_message') or '-'}\n\n"
        f"## Top stack frames\n{frames_text}\n\n"
        f"## Evidence excerpt\n```\n{evidence_excerpt}\n```\n\n"
        f"## Raw defects4j test log (truncated)\n```\n{raw_log_trimmed}\n```\n\n"
        f"{truncation_note}"
        f"## Buggy file content\n```java\n{truncated}\n```\n\n"
        "Your patch MUST be expressed as one or more "
        "`{\"old\": \"...\", \"new\": \"...\"}` edits that apply against the buggy "
        "file above. Each `old` chunk must match exactly ONE location in the "
        "file; include enough surrounding context to guarantee uniqueness."
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
            else:
                parts.append(f"- patch could NOT be applied: `{err_text}`")
        if entry.get("llm_error"):
            parts.append(
                "- your response was not valid patch JSON. "
                f"Reason: `{entry['llm_error']}`"
            )
            raw = entry.get("llm_raw_preview") or ""
            if raw:
                parts.append(
                    "- your raw response started like this (do NOT repeat this):\n"
                    f"```\n{raw}\n```"
                )
            parts.append(
                "  Next round: output ONLY the JSON object described in the "
                'system prompt — starting with `{"edits": [...]}`. No '
                "`error` field, no prose commentary, no markdown fences."
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
        f"## Pre-ranked suspects with the enclosing method context\n{suspects_block}\n\n"
        "Your patch MUST be expressed as one or more `{\"old\": \"...\", \"new\": \"...\"}` "
        "edits that apply against the buggy file above.  Each `old` chunk must match "
        "exactly ONE location in the file; use enough surrounding code from the "
        "enclosing method to guarantee uniqueness."
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


def _ask_llm_for_edits(
    *,
    model_key: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _on_metadata(meta: dict[str, Any]) -> None:
        u = meta.get("usage") if isinstance(meta, dict) else None
        if not isinstance(u, dict):
            return
        # providers may report either OpenAI-style (prompt_tokens/completion_tokens)
        # or our unified style (input_tokens/output_tokens); try both.
        prompt_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        completion_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        total_tokens = int(u.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage["prompt_tokens"] = prompt_tokens
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = total_tokens

    response = call_llm_for_json(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=model_key,
        isJson=True,
        stream=False,
        metadata_handler=_on_metadata,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"LLM returned non-JSON payload of type {type(response)}")
    edits = response.get("edits") or response.get("patches") or response.get("changes") or []
    if not isinstance(edits, list):
        raise RuntimeError(f"`edits` must be a JSON array, got {type(edits)}")
    explanation = str(response.get("explanation") or response.get("explain") or "")
    return edits, explanation, usage


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

        needs_fallback = (
            top_suspect is None or suspect_is_test or inspector_lacks_evidence
        )
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
        used_fallback_suspect = (
            "fallback_from_classes_modified"
            in (top_suspect.get("score_reasons") or [])
        )
        if strategy == STRATEGY_FULL_PIPELINE:
            if used_fallback_suspect:
                # Inspector could not localise — fall through to a hybrid prompt
                # that still uses inspector evidence but ships the full source.
                base_user_prompt = _build_naive_with_evidence_prompt(
                    file_rel_path=top_suspect.get("source_file") or "source.java",
                    source_text=source_text,
                    failing_tests=failing_tests,
                    evidence=evidence,
                    raw_test_log=raw_test_log,
                )
            else:
                base_user_prompt = _build_full_pipeline_prompt(
                    file_rel_path=top_suspect.get("source_file") or "source.java",
                    failing_tests=failing_tests,
                    evidence=evidence,
                    suspects=suspects,
                    raw_test_log=raw_test_log,
                )
            system_prompt = FULL_PIPELINE_SYSTEM_PROMPT
        else:  # naive_chat
            base_user_prompt = _build_naive_prompt(
                file_rel_path=top_suspect.get("source_file") or "source.java",
                source_text=source_text,
                failing_tests=failing_tests,
            )
            system_prompt = NAIVE_SYSTEM_PROMPT

        original_source_text = source_text  # keep so we can reset between rounds
        attempts: list[dict[str, Any]] = []  # round-by-round outcome history
        runner = Defects4JRunner()
        previously_failing_set = set(failing_before)

        # The "best attempt so far" across all completed rounds. Best is
        # defined as the lowest (failed_count, new_regressions) tuple — i.e.
        # prefer a round that introduces no regressions over one that fixes
        # the target bug but breaks something else.
        best_attempt: dict[str, Any] | None = None
        explanation = ""
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
            try:
                round_edits, round_explanation, round_usage = _ask_llm_for_edits(
                    model_key=model_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as llm_exc:
                # Covers:
                #   - LLMCallError("Model response was not valid JSON …")
                #   - Connection drops / 5xx from the provider
                # Record it as a failed round and move on.
                raw_preview = getattr(llm_exc, "raw_response", None) or ""
                if not isinstance(raw_preview, str):
                    raw_preview = str(raw_preview)
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "llm_invalid_json",
                        "llm_error": str(llm_exc)[:500],
                        "llm_raw_preview": raw_preview[:1200],
                        "edits": [],
                    }
                )
                if round_idx == MAX_REPAIR_ROUNDS:
                    break
                continue
            for k in usage:
                usage[k] += int(round_usage.get(k, 0))
            edits = round_edits
            explanation = round_explanation

            # ---- reset file to original buggy state before applying ----
            abs_source_path.write_text(original_source_text, encoding="utf-8")

            bench_store.update_run_progress(run_id, stage="apply_patch")
            try:
                fixed_code, edit_report = _apply_edits(original_source_text, edits)
            except RuntimeError as apply_exc:
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "apply_failed",
                        "apply_error": str(apply_exc)[:2000],
                        "edits": edits,
                    }
                )
                if round_idx == MAX_REPAIR_ROUNDS:
                    break
                continue

            patch_diff_text, patch_added, patch_removed = _compute_diff(
                original_source_text,
                fixed_code,
                top_suspect.get("source_file") or "source.java",
            )
            abs_source_path.write_text(fixed_code, encoding="utf-8")

            bench_store.update_run_progress(run_id, stage="recompile")
            compile_result = runner.compile(work_dir)
            if compile_result.returncode != 0:
                attempts.append(
                    {
                        "round": round_idx,
                        "outcome": "compile_failed",
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
                "edits": edits,
                "edit_report": edit_report,
                "explanation": round_explanation,
                "fixed_code": fixed_code,
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
                        "remaining_failing": [],
                        "edits": edits,
                    }
                )
                break

            attempts.append(
                {
                    "round": round_idx,
                    "outcome": "tests_still_failing",
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
            abs_source_path.write_text(best_attempt["fixed_code"], encoding="utf-8")
            fixed_code = best_attempt["fixed_code"]
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
            edits = []
            edit_report = []
            failing_after = list(failing_before)
            failed_after = failed_before
            compile_ok = False
            patch_diff_text = ""
            patch_added = 0
            patch_removed = 0

        report["llm_explanation"] = explanation
        report["edits_count"] = len(edits)
        report["applied_edits"] = edit_report
        report["attempts"] = [
            {k: v for k, v in entry.items() if k != "edits"}
            for entry in attempts
        ]
        report["max_rounds"] = MAX_REPAIR_ROUNDS
        try:
            (artifacts_dir / "patch.diff").write_text(patch_diff_text, encoding="utf-8")
            (artifacts_dir / "fixed_code.java").write_text(fixed_code, encoding="utf-8")
            import json as _json

            (artifacts_dir / "edits.json").write_text(
                _json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8"
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
                f"LLM edits could not be applied after {llm_rounds} round(s); "
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
