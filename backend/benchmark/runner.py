"""Benchmark runner that bridges the Defects4J CLI wrapper to the persistent
benchmark schema.

Each run moves through a pipeline:
  1. `checkout`   – fetch the buggy snapshot (or reuse an existing checkout)
  2. `compile`    – compile the buggy snapshot
  3. `test`       – run developer tests to reproduce the failing assertions
  4. `inspect`    – build an LLM-ready inspection report
  5. `repair`     – (planned, for `full_repair` runs) apply a model-generated patch
  6. `revalidate` – (planned) re-run tests after applying the patch

For the MVP we currently support two run modes:
  - `inspect_only`: runs checkout + compile + test + inspect, writes results
    to the DB and returns the structured inspection report.
  - `mock_repair`: same pipeline but marks the run as "completed" when all
    relevant tests pass on the buggy snapshot, otherwise "failed".
    No real patch application is performed yet.

This module is intentionally self-contained so the integration test under
`backend/scripts/benchmark_smoke.py` can exercise it without going through
the Flask/SSE layer.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from backend.benchmark import store as bench_store
from backend.inspector.inspector import run_defects4j_inspection

logger = logging.getLogger(__name__)


def _default_workspace_root() -> Path:
    root = os.getenv("BENCHMARK_WORK_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (Path.cwd() / "tmp" / "benchmark").resolve()


def _prepare_work_dir(project_code: str, bug_id: int, run_id: int, *, reuse: bool = False) -> tuple[Path, Path]:
    workspace_root = _default_workspace_root()
    run_dir = workspace_root / f"{project_code}_{bug_id}" / f"run_{run_id}"
    artifacts_dir = run_dir / ".inspector"
    if not reuse and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifacts_dir


def _derive_test_stats(report: dict[str, Any]) -> tuple[int, int, int]:
    tests = report.get("tests") or {}
    failing_tests = tests.get("failing_tests") or []
    trigger_tests = tests.get("trigger_tests") or []
    fail_count = len(failing_tests)
    total_tests = max(fail_count, len(trigger_tests))
    pass_count = max(0, total_tests - fail_count)
    return pass_count, fail_count, total_tests


def run_benchmark_for_bug(
    *,
    run_id: int,
    user_id: int,
    organization_id: int | None,
    project_code: str,
    defects4j_project: str,
    defects4j_bug_id: int,
    model_key: str,
    run_mode: str,
    force_checkout: bool = True,
) -> dict[str, Any]:
    """Execute the benchmark pipeline synchronously, persisting progress to DB."""
    start = time.time()
    try:
        bench_store.update_run_progress(run_id, stage="checkout", run_status="running")
        work_dir, artifacts_dir = _prepare_work_dir(project_code, defects4j_bug_id, run_id, reuse=not force_checkout)

        bench_store.update_run_progress(run_id, stage="inspect")
        report = run_defects4j_inspection(
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

        pass_count, fail_count, total_tests = _derive_test_stats(report)
        duration_ms = int((time.time() - start) * 1000)

        # `inspect_only` always ends as "completed" – what matters is that we
        # reliably surface the failing-test story. `mock_repair` is marked
        # failed if the buggy snapshot still has failing tests (expected), or
        # completed if the snapshot happens to be clean.
        if run_mode == "mock_repair":
            final_status = "completed" if fail_count == 0 else "failed"
            error_message = None if final_status == "completed" else (
                f"Buggy snapshot still has {fail_count} failing test(s); real patch generation not wired yet."
            )
        else:
            final_status = "completed"
            error_message = None

        bench_store.update_run_progress(
            run_id,
            stage="done",
            run_status=final_status,
            pass_count=pass_count,
            fail_count=fail_count,
            total_tests=total_tests,
            duration_ms=duration_ms,
            error_message=error_message,
            report=report,
            finalize=True,
        )

        return {
            "run_id": run_id,
            "status": final_status,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total_tests": total_tests,
            "duration_ms": duration_ms,
            "report": report,
        }
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.exception("Benchmark run %s failed", run_id)
        duration_ms = int((time.time() - start) * 1000)
        bench_store.update_run_progress(
            run_id,
            stage="error",
            run_status="failed",
            duration_ms=duration_ms,
            error_message=str(exc)[:1900],
            finalize=True,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": str(exc),
            "duration_ms": duration_ms,
        }


def run_benchmark_in_background(**kwargs: Any) -> threading.Thread:
    """Launch the synchronous runner in a daemon thread, useful for API endpoints."""
    thread = threading.Thread(
        target=run_benchmark_for_bug,
        kwargs=kwargs,
        daemon=True,
        name=f"benchmark-run-{kwargs.get('run_id')}",
    )
    thread.start()
    return thread
