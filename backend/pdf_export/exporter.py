"""High-level helpers that assemble `PdfDocument` instances for repair
reports and benchmark runs, then persist an audit row in `user_pdf_exports`."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from backend.auth.store import get_db_connection
from backend.history.store import get_history_for_user
from backend.pdf_export.pdf_writer import (
    PdfDocument,
    build_pdf_bytes,
    document_sha256,
)
from backend.benchmark import store as bench_store


def _truncate_block(text: str | None, *, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated for PDF export)"


def _record_export(
    *,
    user_id: int,
    history_id: int | None,
    benchmark_run_id: int | None,
    export_type: str,
    file_bytes: int,
    file_sha256: str,
) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_pdf_exports
                    (user_id, history_id, benchmark_run_id, export_type, file_bytes, file_sha256)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, history_id, benchmark_run_id, export_type, file_bytes, file_sha256),
            )
        connection.commit()


def build_repair_report_pdf(*, user_id: int, history_id: int) -> tuple[bytes, str]:
    history = get_history_for_user(user_id, history_id)
    if history is None:
        raise ValueError("History record was not found.")
    if history.get("mode") != "agent":
        raise ValueError("PDF export is only available for agent-mode histories.")

    snapshot = history.get("snapshot") or {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except json.JSONDecodeError:
            snapshot = {}

    document = PdfDocument(title="AutoRepair — Repair Report")
    document.add_heading("AutoRepair Repair Report")
    document.add_meta("Title", str(history.get("title") or ""))
    document.add_meta("Model", str(snapshot.get("model") or history.get("model") or "-"))
    document.add_meta("Language", str(snapshot.get("language") or history.get("language") or "-"))
    document.add_meta("Filename", str(snapshot.get("filename") or "-"))
    document.add_meta("Created", str(history.get("created_at") or ""))
    document.add_meta("Updated", str(history.get("updated_at") or ""))
    document.add_meta("Exported", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

    final_status = snapshot.get("final_status") or ""
    if final_status:
        document.add_paragraph(f"Pipeline status: {final_status}")

    # Input code
    document.add_code_block(_truncate_block(str(snapshot.get("code") or "")), title="Original code")

    # Run result
    run_result = snapshot.get("run_result") or {}
    if isinstance(run_result, dict):
        stdout_text = _truncate_block(str(run_result.get("stdout") or ""), max_chars=2500)
        stderr_text = _truncate_block(str(run_result.get("stderr") or ""), max_chars=2500)
        if stdout_text:
            document.add_code_block(stdout_text, title="Run output · stdout")
        if stderr_text:
            document.add_code_block(stderr_text, title="Run output · stderr")

    stages = snapshot.get("stages") or {}
    if isinstance(stages, dict):
        for stage_name in ("run", "inspect", "plan", "code", "verify"):
            stage = stages.get(stage_name)
            if not isinstance(stage, dict):
                continue
            status = str(stage.get("status") or "idle")
            document.add_subheading(f"Stage · {stage_name}  [{status}]")
            explain = str(stage.get("explain") or "").strip()
            if explain:
                document.add_paragraph(explain[:1200])
            report = str(stage.get("report") or "").strip()
            if report:
                document.add_code_block(_truncate_block(report, max_chars=2500), title=f"{stage_name} · report")

    final_diff = snapshot.get("final_diff") or ""
    if final_diff:
        document.add_code_block(_truncate_block(str(final_diff), max_chars=5000), title="Final unified diff")

    error_message = snapshot.get("error_message") or ""
    if error_message:
        document.add_code_block(_truncate_block(str(error_message), max_chars=1500), title="Error message")

    pdf_bytes = build_pdf_bytes(document)
    sha = document_sha256(pdf_bytes)
    _record_export(
        user_id=user_id,
        history_id=history_id,
        benchmark_run_id=None,
        export_type="repair_report",
        file_bytes=len(pdf_bytes),
        file_sha256=sha,
    )
    safe_title = (history.get("title") or f"history_{history_id}").replace("/", "-")[:40]
    filename = f"repair_report_{history_id}_{safe_title}.pdf"
    return pdf_bytes, filename


def build_benchmark_report_pdf(*, user_id: int, run_id: int) -> tuple[bytes, str]:
    run = bench_store.get_run(run_id, include_heavy=True)
    if run is None:
        raise ValueError("Benchmark run was not found.")
    if int(run.get("user_id") or 0) != user_id:
        raise PermissionError("This benchmark run does not belong to the current user.")

    document = PdfDocument(title="AutoRepair — Benchmark Report")
    document.add_heading("AutoRepair Benchmark Report")
    document.add_meta("Run ID", str(run["id"]))
    document.add_meta("Benchmark", f"{run.get('project_code')} · {run.get('bug_key')}")
    document.add_meta(
        "Defects4J",
        f"project={run.get('defects4j_project')}, bug_id={run.get('defects4j_bug_id')}",
    )
    document.add_meta("Model", str(run.get("model_key")))
    document.add_meta("Mode", str(run.get("run_mode")))
    document.add_meta("Status", str(run.get("run_status")))
    document.add_meta(
        "Tests",
        f"pass={run.get('pass_count')}, fail={run.get('fail_count')}, total={run.get('total_tests')}",
    )
    document.add_meta("Duration (ms)", str(run.get("duration_ms")))
    document.add_meta("Credits spent", str(run.get("credits_spent")))
    document.add_meta("Started", str(run.get("started_at") or "-"))
    document.add_meta("Finished", str(run.get("finished_at") or "-"))

    error_message = run.get("error_message") or ""
    if error_message:
        document.add_code_block(_truncate_block(str(error_message)), title="Error message")

    report = run.get("report") or {}
    if isinstance(report, dict):
        failures = report.get("failures") or []
        for idx, failure in enumerate(failures[:3]):
            document.add_subheading(f"Failure #{idx + 1}")
            document.add_meta("Test", str(failure.get("failing_test") or ""))
            document.add_meta("Exception", str(failure.get("exception_type") or ""))
            document.add_paragraph(str(failure.get("exception_message") or "")[:800])
            evidence = str(failure.get("evidence_excerpt") or "")
            if evidence:
                document.add_code_block(_truncate_block(evidence, max_chars=1500), title="Evidence excerpt")
            suspects = failure.get("suspects") or []
            for suspect_idx, suspect in enumerate(suspects[:2]):
                document.add_meta(
                    f"Suspect #{suspect_idx + 1}",
                    f"{suspect.get('class_name')}::{suspect.get('line_number')} (score={suspect.get('score')})",
                )
                snippet = str(suspect.get("snippet") or "")
                if snippet:
                    document.add_code_block(_truncate_block(snippet, max_chars=1200), title="Source snippet")

    patch_diff = run.get("patch_diff") or ""
    if patch_diff:
        document.add_code_block(_truncate_block(str(patch_diff), max_chars=4000), title="Applied diff")

    pdf_bytes = build_pdf_bytes(document)
    sha = document_sha256(pdf_bytes)
    _record_export(
        user_id=user_id,
        history_id=None,
        benchmark_run_id=run_id,
        export_type="benchmark_report",
        file_bytes=len(pdf_bytes),
        file_sha256=sha,
    )
    filename = f"benchmark_run_{run_id}_{run.get('project_code')}_{run.get('bug_key')}.pdf"
    return pdf_bytes, filename
