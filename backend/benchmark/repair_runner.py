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


def _extract_top_suspect(report: dict[str, Any]) -> dict[str, Any] | None:
    failures = report.get("failures") or []
    for failure in failures:
        for suspect in failure.get("suspects") or []:
            if suspect.get("source_file"):
                return suspect
        for candidate in failure.get("ranked_candidates") or []:
            if candidate.get("source_file"):
                return candidate
    return None


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
    "}\n"
    "Each `old` chunk MUST appear exactly once in the current buggy file. "
    "Prefer small edits (one contiguous chunk per bug). Copy whitespace and "
    "indentation verbatim. Do NOT rewrite the whole file."
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


def _apply_edits(source_text: str, edits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply a list of search/replace edits to source_text.

    Returns (new_text, per_edit_report). Raises RuntimeError if any edit cannot
    be applied unambiguously.
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
        count = current.count(old)
        if count == 0:
            raise RuntimeError(
                f"Edit #{idx} `old` chunk not found in source (len={len(old)})."
            )
        if count > 1:
            raise RuntimeError(
                f"Edit #{idx} `old` chunk matches {count} times; must be unique."
            )
        current = current.replace(old, new, 1)
        report.append(
            {
                "index": idx,
                "old_len": len(old),
                "new_len": len(new) if isinstance(new, str) else 0,
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
        top_suspect = _extract_top_suspect(inspection)
        if top_suspect is None:
            raise RuntimeError("Inspector could not recover any suspect source file; cannot repair.")

        abs_source_path, source_text = _read_suspect_source(work_dir, top_suspect)
        if abs_source_path is None or source_text is None:
            raise RuntimeError(f"Failed to read suspect source: {top_suspect.get('source_file')}")

        report["suspect"] = {
            "source_file": top_suspect.get("source_file"),
            "line_number": top_suspect.get("line_number"),
            "score": top_suspect.get("score"),
            "score_reasons": top_suspect.get("score_reasons"),
        }

        # ------ Build prompt per strategy ------
        bench_store.update_run_progress(run_id, stage="generate_patch")
        if strategy == STRATEGY_FULL_PIPELINE:
            evidence = _extract_evidence(inspection)
            raw_test_log = _read_inspector_test_log(artifacts_dir)
            user_prompt = _build_full_pipeline_prompt(
                file_rel_path=top_suspect.get("source_file") or "source.java",
                failing_tests=failing_tests,
                evidence=evidence,
                suspects=suspects,
                raw_test_log=raw_test_log,
            )
            system_prompt = FULL_PIPELINE_SYSTEM_PROMPT
        else:  # naive_chat
            user_prompt = _build_naive_prompt(
                file_rel_path=top_suspect.get("source_file") or "source.java",
                source_text=source_text,
                failing_tests=failing_tests,
            )
            system_prompt = NAIVE_SYSTEM_PROMPT

        # Save prompt artifacts for later inspection
        try:
            (artifacts_dir / "prompt.user.md").write_text(user_prompt, encoding="utf-8")
            (artifacts_dir / "prompt.system.md").write_text(system_prompt, encoding="utf-8")
        except OSError:
            pass

        # ------ Call LLM ------
        llm_rounds = 1
        edits, explanation, usage = _ask_llm_for_edits(
            model_key=model_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        report["llm_explanation"] = explanation
        report["edits_count"] = len(edits)

        # ------ Apply patch ------
        bench_store.update_run_progress(run_id, stage="apply_patch")
        try:
            fixed_code, edit_report = _apply_edits(source_text, edits)
        except RuntimeError as apply_exc:
            bench_store.update_run_progress(
                run_id,
                stage="error",
                run_status="failed",
                error_message=f"Patch application failed: {apply_exc}",
                duration_ms=int((time.time() - start) * 1000),
                report=report,
                is_plausible=False,
                is_correct=False,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                llm_rounds=llm_rounds,
                failed_tests_before=failed_before,
                failed_tests_after=failed_before,
                finalize=True,
            )
            return {
                "run_id": run_id,
                "status": "failed",
                "is_plausible": False,
                "is_correct": False,
                "stage": "apply_patch",
                "error_message": str(apply_exc),
            }
        report["applied_edits"] = edit_report
        patch_diff_text, patch_added, patch_removed = _compute_diff(
            source_text,
            fixed_code,
            top_suspect.get("source_file") or "source.java",
        )
        try:
            (artifacts_dir / "patch.diff").write_text(patch_diff_text, encoding="utf-8")
            (artifacts_dir / "fixed_code.java").write_text(fixed_code, encoding="utf-8")
            import json as _json
            (artifacts_dir / "edits.json").write_text(
                _json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        abs_source_path.write_text(fixed_code, encoding="utf-8")

        # ------ Recompile ------
        bench_store.update_run_progress(run_id, stage="recompile")
        runner = Defects4JRunner()
        compile_result = runner.compile(work_dir)
        report["recompile"] = {
            "returncode": compile_result.returncode,
            "stderr_tail": (compile_result.stderr or "")[-1500:],
        }
        if compile_result.returncode != 0:
            bench_store.update_run_progress(
                run_id,
                stage="error",
                run_status="failed",
                error_message="Patch failed to compile.",
                duration_ms=int((time.time() - start) * 1000),
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
                failed_tests_after=failed_before,
                finalize=True,
            )
            return {
                "run_id": run_id,
                "status": "failed",
                "is_plausible": False,
                "is_correct": False,
                "stage": "recompile",
                "error_message": "Patch failed to compile.",
            }

        # ------ Retest ------
        bench_store.update_run_progress(run_id, stage="retest")
        test_result = runner.test_relevant(work_dir)
        failing_after, failed_after = _count_failing_from_test_output(runner, work_dir)
        report["retest"] = {
            "returncode": test_result.returncode,
            "failing_after": failing_after,
            "stdout_tail": (test_result.stdout or "")[-800:],
        }

        is_plausible = failed_after == 0 and compile_result.returncode == 0
        previously_failing_set = set(failing_before)
        remaining_previous = [t for t in failing_after if t in previously_failing_set]
        new_failures = [t for t in failing_after if t not in previously_failing_set]
        is_correct = is_plausible and not remaining_previous and not new_failures

        total_tests = max(len(trigger_tests), failed_before)
        pass_count = max(0, total_tests - failed_after)
        duration_ms = int((time.time() - start) * 1000)

        final_status = "completed" if is_plausible else "failed"
        error_message = None if is_plausible else (
            f"Patch applied but tests still failing: {failing_after[:3]}"
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
