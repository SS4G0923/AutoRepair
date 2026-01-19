# inspector/inspector.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from defects4j_runner import Defects4JRunner, Defects4JError
from stacktrace_filter import (
    build_evidence_from_log,
    choose_best_frame,
    filter_and_rank_frames,
)
from source_utils import (
    SourceIndex,
    resolve_source_for_frame,
    extract_code_snippet,
    extract_enclosing_method,
)
from inspector_prompt import build_planner_prompt


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", errors="replace")


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _is_valid_test_id(test_id: str) -> bool:
    if test_id.startswith("--- "):
        return False
    if "::" not in test_id:
        return False
    return not any(ch.isspace() for ch in test_id)


def _has_exception_or_stacktrace(log_text: str) -> bool:
    evidence = build_evidence_from_log(log_text)
    if evidence.exception.exception_type != "UnknownException":
        return True
    return bool(evidence.frames)


def _build_evidence_excerpt(raw_log: str, max_lines: int = 20) -> str:
    lines = raw_log.splitlines()
    return "\n".join(lines[:max_lines]).rstrip()


def run_defects4j_inspection(
    project_id: str,
    bug_id: int,
    is_buggy: bool,
    work_dir: Path,
    artifacts_dir: Path,
    force_checkout: bool = False,
    test_mode: str = "relevant",  # relevant | all
    inspect_failing_tests: int = 1,
    max_candidates: int = 8,
) -> dict[str, Any]:
    runner = Defects4JRunner()

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) checkout (optional)
    checkout_result = None
    if force_checkout or not work_dir.exists() or not any(work_dir.iterdir()):
        checkout_result = runner.checkout(
            project_id=project_id,
            bug_id=bug_id,
            is_buggy=is_buggy,
            work_dir=work_dir,
            force=force_checkout,
        )

    # 2) export properties needed for source resolution
    # dir.src.classes / dir.src.tests are relative to work_dir
    try:
        src_classes_rel = runner.export_value(work_dir, "dir.src.classes")
        src_tests_rel = runner.export_value(work_dir, "dir.src.tests")
    except Defects4JError as e:
        raise

    src_roots = []
    if src_classes_rel:
        src_roots.append((work_dir / src_classes_rel).resolve())
    if src_tests_rel:
        src_roots.append((work_dir / src_tests_rel).resolve())

    # Build file index (fast resolution by file name)
    index = SourceIndex(src_roots)
    index.build()

    # 3) compile
    compile_result = runner.compile(work_dir)
    _write_text(artifacts_dir / "compile.stdout.txt", compile_result.stdout)
    _write_text(artifacts_dir / "compile.stderr.txt", compile_result.stderr)

    # 4) test
    if test_mode == "all":
        test_result = runner.test_all(work_dir)
    else:
        test_result = runner.test_relevant(work_dir)

    _write_text(artifacts_dir / "test.stdout.txt", test_result.stdout)
    _write_text(artifacts_dir / "test.stderr.txt", test_result.stderr)

    failing_ids, failing_raw, failing_blocks = runner.read_failing_tests_blocked(work_dir)
    _write_text(artifacts_dir / "failing_tests.raw.txt", failing_raw)

    # 5) export triggers
    trigger_export = runner.export(work_dir, "tests.trigger")
    trigger_tests = runner.parse_tests_trigger(trigger_export.stdout)

    _write_text(artifacts_dir / "tests.trigger.txt", trigger_export.stdout)
    _write_text(artifacts_dir / "tests.trigger.stderr.txt", trigger_export.stderr)

    # 6) pick N failing tests to inspect in detail
    detailed_failures: list[dict[str, Any]] = []
    inspection_targets = trigger_tests if trigger_tests else failing_ids
    for i, t in enumerate(inspection_targets[: max(0, inspect_failing_tests)]):
        if not _is_valid_test_id(t):
            _write_text(artifacts_dir / f"single_test_{i}.stdout.txt", "")
            _write_text(artifacts_dir / f"single_test_{i}.stderr.txt", "")
            detailed_failures.append(
                {
                    "failing_test": t,
                    "raw_log_artifact": f"single_test_{i}.stdout.txt + single_test_{i}.stderr.txt",
                    "inspector_error": f"Invalid test id format: {t}",
                    "ranked_candidates": [],
                    "selected_frame": None,
                    "suspects": [],
                }
            )
            continue

        single = runner.test_single(work_dir, t)
        _write_text(artifacts_dir / f"single_test_{i}.stdout.txt", single.stdout)
        _write_text(artifacts_dir / f"single_test_{i}.stderr.txt", single.stderr)

        # Evidence is best extracted from stderr+stdout combined (JUnit output varies)
        raw_log = (single.stdout or "") + "\n" + (single.stderr or "")
        evidence_source = "single_test_run"
        if not raw_log.strip() or not _has_exception_or_stacktrace(raw_log):
            fallback_log = failing_blocks.get(t)
            if fallback_log is not None:
                raw_log = fallback_log
                evidence_source = "failing_tests_file_block"
        evidence = build_evidence_from_log(raw_log)
        evidence_excerpt = _build_evidence_excerpt(raw_log)

        def source_exists_fn(frame) -> bool:
            resolved = resolve_source_for_frame(index, frame.class_name, frame.file_name)
            return bool(resolved and resolved.source_file.exists())

        ranked = filter_and_rank_frames(evidence, source_exists_fn=source_exists_fn, max_candidates=max_candidates)
        best = choose_best_frame(ranked)

        suspects: list[dict[str, Any]] = []
        for s in ranked[: min(len(ranked), 3)]:
            res = resolve_source_for_frame(index, s.frame.class_name, s.frame.file_name)
            if not res:
                continue
            snippet = extract_code_snippet(res.source_file, s.frame.line_number, window=8)
            method_ctx = extract_enclosing_method(res.source_file, s.frame.line_number)
            suspects.append(
                {
                    "class_name": s.frame.class_name,
                    "file_name": s.frame.file_name,
                    "line_number": s.frame.line_number,
                    "source_file": _safe_relpath(res.source_file, work_dir),
                    "resolve_confidence": res.confidence,
                    "resolve_reason": res.reason,
                    "score": s.score,
                    "score_reasons": s.reasons,
                    "snippet": snippet,
                    "enclosing_method": method_ctx,
                }
            )

        detailed_failures.append(
            {
                "failing_test": t,
                "exception_type": evidence.exception.exception_type,
                "exception_message": evidence.exception.message,
                "raw_log_artifact": f"single_test_{i}.stdout.txt + single_test_{i}.stderr.txt",
                "evidence_source": evidence_source,
                "evidence_excerpt": evidence_excerpt,
                "ranked_candidates": [
                    {
                        "class_name": s.frame.class_name,
                        "file_name": s.frame.file_name,
                        "line_number": s.frame.line_number,
                        "index_in_trace": s.frame.index_in_trace,
                        "raw_line": s.frame.raw_line,
                        "score": s.score,
                        "score_reasons": s.reasons,
                    }
                    for s in ranked
                ],
                "selected_frame": (
                    {
                        "class_name": best.frame.class_name,
                        "file_name": best.frame.file_name,
                        "line_number": best.frame.line_number,
                        "index_in_trace": best.frame.index_in_trace,
                        "raw_line": best.frame.raw_line,
                        "score": best.score,
                        "score_reasons": best.reasons,
                    }
                    if best
                    else None
                ),
                "suspects": suspects,
            }
        )

    # 7) Build report
    report: dict[str, Any] = {
        "benchmark": "defects4j",
        "project_id": project_id,
        "bug_id": bug_id,
        "version": "b" if is_buggy else "f",
        "work_dir": str(work_dir.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "defects4j": {
            "checkout": asdict(checkout_result) if checkout_result else None,
            "compile": asdict(compile_result),
            "test": asdict(test_result),
            "export_tests_trigger": asdict(trigger_export),
            "properties": {
                "dir.src.classes": src_classes_rel,
                "dir.src.tests": src_tests_rel,
            },
        },
        "tests": {
            "mode": test_mode,
            "failing_tests": failing_ids,
            "failing_tests_raw_artifact": "failing_tests.raw.txt",
            "failing_test_blocks_present": len(failing_blocks),
            "trigger_tests": trigger_tests,
        },
        "failures": detailed_failures,
    }

    # Optional: embed planner prompt for convenience
    report["planner_prompt"] = build_planner_prompt(report)

    return report


def main() -> None:

    project_id = "Lang"
    bug_id = 1
    version = "b"
    is_buggy = version == "b"
    work_dir = "../tmp/d4j/" + project_id + "_" + str(bug_id) + version
    work_dir = Path(work_dir).resolve()
    artifacts_dir = (work_dir / ".inspector").resolve()
    force_checkout = True
    test_mode = "relevant"
    inspect_failing_tests = 1
    max_candidates = 3

    report = run_defects4j_inspection(
        project_id=project_id,
        bug_id=bug_id,
        is_buggy=is_buggy,
        work_dir=work_dir,
        artifacts_dir=artifacts_dir,
        force_checkout=force_checkout,
        test_mode=test_mode,
        inspect_failing_tests=inspect_failing_tests,
        max_candidates=max_candidates,
    )

    with open(project_id + str(bug_id) + version + "_inspector_output.json", 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[Inspector] Artifacts dir: {artifacts_dir}")


if __name__ == "__main__":
    main()
