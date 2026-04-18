from __future__ import annotations

import json
from typing import Any

from backend.auth.store import get_db_connection


def _serialize_project(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "project_code": row["project_code"],
        "display_name": row["display_name"],
        "source_type": row["source_type"],
        "language": row["language"],
        "description": row.get("description"),
        "tags": [tag.strip() for tag in (row.get("tags") or "").split(",") if tag.strip()],
        "is_active": bool(row.get("is_active", 1)),
        "sort_order": int(row.get("sort_order") or 0),
        "bug_count": int(row.get("bug_count") or 0),
        "last_run_at": str(row["last_run_at"]) if row.get("last_run_at") else None,
    }


def _serialize_bug(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "project_id": int(row["project_id"]),
        "bug_key": row["bug_key"],
        "title": row["title"],
        "severity": row["severity"],
        "defects4j_project": row.get("defects4j_project"),
        "defects4j_bug_id": int(row["defects4j_bug_id"]) if row.get("defects4j_bug_id") is not None else None,
        "description": row.get("description"),
        "tags": [tag.strip() for tag in (row.get("tags") or "").split(",") if tag.strip()],
        "is_active": bool(row.get("is_active", 1)),
    }


def _serialize_run(row: dict[str, Any], *, include_heavy: bool = False) -> dict[str, Any]:
    payload = {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "organization_id": int(row["organization_id"]) if row.get("organization_id") is not None else None,
        "project_id": int(row["project_id"]),
        "bug_id": int(row["bug_id"]),
        "project_code": row.get("project_code"),
        "bug_key": row.get("bug_key"),
        "defects4j_project": row.get("defects4j_project"),
        "defects4j_bug_id": int(row["defects4j_bug_id"]) if row.get("defects4j_bug_id") is not None else None,
        "model_key": row["model_key"],
        "run_mode": row["run_mode"],
        "run_status": row["run_status"],
        "stage": row.get("stage"),
        "strategy": row.get("strategy"),
        "experiment_id": int(row["experiment_id"]) if row.get("experiment_id") is not None else None,
        "pass_count": int(row["pass_count"] or 0),
        "fail_count": int(row["fail_count"] or 0),
        "total_tests": int(row["total_tests"] or 0),
        "duration_ms": int(row["duration_ms"] or 0),
        "credits_spent": int(row["credits_spent"] or 0),
        "is_plausible": bool(row.get("is_plausible") or 0),
        "is_correct": bool(row.get("is_correct") or 0),
        "prompt_tokens": int(row.get("prompt_tokens") or 0),
        "completion_tokens": int(row.get("completion_tokens") or 0),
        "total_tokens": int(row.get("total_tokens") or 0),
        "patch_lines_added": int(row.get("patch_lines_added") or 0),
        "patch_lines_removed": int(row.get("patch_lines_removed") or 0),
        "llm_rounds": int(row.get("llm_rounds") or 0),
        "failed_tests_before": int(row.get("failed_tests_before") or 0),
        "failed_tests_after": int(row.get("failed_tests_after") or 0),
        "error_message": row.get("error_message"),
        "started_at": str(row["started_at"]) if row.get("started_at") else None,
        "finished_at": str(row["finished_at"]) if row.get("finished_at") else None,
    }
    if include_heavy:
        raw_report = row.get("report_json")
        if raw_report:
            try:
                payload["report"] = json.loads(raw_report)
            except json.JSONDecodeError:
                payload["report"] = {"raw": raw_report}
        else:
            payload["report"] = None
        payload["patch_diff"] = row.get("patch_diff") or ""
    return payload


def list_benchmark_projects() -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.project_code, p.display_name, p.source_type, p.language,
                       p.description, p.tags, p.is_active, p.sort_order,
                       (SELECT COUNT(*) FROM benchmark_bugs b WHERE b.project_id = p.id AND b.is_active = 1) AS bug_count,
                       (SELECT MAX(r.started_at) FROM benchmark_runs r WHERE r.project_id = p.id) AS last_run_at
                FROM benchmark_projects p
                WHERE p.is_active = 1
                ORDER BY p.sort_order, p.project_code
                """
            )
            rows = cursor.fetchall() or []
    return [_serialize_project(row) for row in rows]


def list_benchmark_bugs(project_id: int | None = None) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            if project_id is None:
                cursor.execute(
                    """
                    SELECT id, project_id, bug_key, title, severity, defects4j_project,
                           defects4j_bug_id, description, tags, is_active
                    FROM benchmark_bugs
                    WHERE is_active = 1
                    ORDER BY project_id, id
                    """
                )
            else:
                cursor.execute(
                    """
                    SELECT id, project_id, bug_key, title, severity, defects4j_project,
                           defects4j_bug_id, description, tags, is_active
                    FROM benchmark_bugs
                    WHERE is_active = 1 AND project_id = %s
                    ORDER BY id
                    """,
                    (project_id,),
                )
            rows = cursor.fetchall() or []
    return [_serialize_bug(row) for row in rows]


def get_bug_with_project(bug_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT b.id AS bug_id, b.project_id, b.bug_key, b.title, b.severity,
                       b.defects4j_project, b.defects4j_bug_id, b.description, b.tags AS bug_tags,
                       p.project_code, p.display_name, p.language, p.source_type
                FROM benchmark_bugs b
                INNER JOIN benchmark_projects p ON p.id = b.project_id
                WHERE b.id = %s AND b.is_active = 1
                """,
                (bug_id,),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return {
        "bug": {
            "id": int(row["bug_id"]),
            "project_id": int(row["project_id"]),
            "bug_key": row["bug_key"],
            "title": row["title"],
            "severity": row["severity"],
            "defects4j_project": row.get("defects4j_project"),
            "defects4j_bug_id": int(row["defects4j_bug_id"]) if row.get("defects4j_bug_id") is not None else None,
            "description": row.get("description"),
            "tags": [tag.strip() for tag in (row.get("bug_tags") or "").split(",") if tag.strip()],
        },
        "project": {
            "id": int(row["project_id"]),
            "project_code": row["project_code"],
            "display_name": row["display_name"],
            "language": row["language"],
            "source_type": row["source_type"],
        },
    }


def create_run(
    *,
    user_id: int,
    organization_id: int | None,
    project_id: int,
    bug_id: int,
    model_key: str,
    run_mode: str,
    credits_spent: int,
    strategy: str | None = None,
    experiment_id: int | None = None,
) -> int:
    strategy_value = strategy or run_mode
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO benchmark_runs
                    (user_id, organization_id, project_id, bug_id, model_key,
                     run_mode, run_status, credits_spent, strategy, experiment_id)
                VALUES (%s, %s, %s, %s, %s, %s, 'running', %s, %s, %s)
                """,
                (
                    user_id,
                    organization_id,
                    project_id,
                    bug_id,
                    model_key,
                    run_mode,
                    credits_spent,
                    strategy_value,
                    experiment_id,
                ),
            )
            run_id = int(cursor.lastrowid)
        connection.commit()
    return run_id


def update_run_progress(
    run_id: int,
    *,
    stage: str | None = None,
    run_status: str | None = None,
    pass_count: int | None = None,
    fail_count: int | None = None,
    total_tests: int | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    report: dict[str, Any] | None = None,
    patch_diff: str | None = None,
    is_plausible: bool | None = None,
    is_correct: bool | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    patch_lines_added: int | None = None,
    patch_lines_removed: int | None = None,
    llm_rounds: int | None = None,
    failed_tests_before: int | None = None,
    failed_tests_after: int | None = None,
    finalize: bool = False,
) -> None:
    updates: list[str] = []
    values: list[Any] = []
    if stage is not None:
        updates.append("stage = %s")
        values.append(stage)
    if run_status is not None:
        updates.append("run_status = %s")
        values.append(run_status)
    if pass_count is not None:
        updates.append("pass_count = %s")
        values.append(int(pass_count))
    if fail_count is not None:
        updates.append("fail_count = %s")
        values.append(int(fail_count))
    if total_tests is not None:
        updates.append("total_tests = %s")
        values.append(int(total_tests))
    if duration_ms is not None:
        updates.append("duration_ms = %s")
        values.append(int(duration_ms))
    if error_message is not None:
        updates.append("error_message = %s")
        values.append(error_message[:2000])
    if report is not None:
        updates.append("report_json = %s")
        values.append(json.dumps(report, ensure_ascii=False))
    if patch_diff is not None:
        updates.append("patch_diff = %s")
        values.append(patch_diff)
    if is_plausible is not None:
        updates.append("is_plausible = %s")
        values.append(1 if is_plausible else 0)
    if is_correct is not None:
        updates.append("is_correct = %s")
        values.append(1 if is_correct else 0)
    if prompt_tokens is not None:
        updates.append("prompt_tokens = %s")
        values.append(int(prompt_tokens))
    if completion_tokens is not None:
        updates.append("completion_tokens = %s")
        values.append(int(completion_tokens))
    if total_tokens is not None:
        updates.append("total_tokens = %s")
        values.append(int(total_tokens))
    if patch_lines_added is not None:
        updates.append("patch_lines_added = %s")
        values.append(int(patch_lines_added))
    if patch_lines_removed is not None:
        updates.append("patch_lines_removed = %s")
        values.append(int(patch_lines_removed))
    if llm_rounds is not None:
        updates.append("llm_rounds = %s")
        values.append(int(llm_rounds))
    if failed_tests_before is not None:
        updates.append("failed_tests_before = %s")
        values.append(int(failed_tests_before))
    if failed_tests_after is not None:
        updates.append("failed_tests_after = %s")
        values.append(int(failed_tests_after))
    if finalize:
        updates.append("finished_at = NOW()")
    if not updates:
        return
    values.append(run_id)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE benchmark_runs SET {', '.join(updates)} WHERE id = %s",
                values,
            )
        connection.commit()


def get_run(run_id: int, *, include_heavy: bool = True) -> dict[str, Any] | None:
    columns = [
        "r.id",
        "r.user_id",
        "r.organization_id",
        "r.project_id",
        "r.bug_id",
        "r.model_key",
        "r.run_mode",
        "r.run_status",
        "r.stage",
        "r.strategy",
        "r.experiment_id",
        "r.pass_count",
        "r.fail_count",
        "r.total_tests",
        "r.duration_ms",
        "r.credits_spent",
        "r.is_plausible",
        "r.is_correct",
        "r.prompt_tokens",
        "r.completion_tokens",
        "r.total_tokens",
        "r.patch_lines_added",
        "r.patch_lines_removed",
        "r.llm_rounds",
        "r.failed_tests_before",
        "r.failed_tests_after",
        "r.error_message",
        "r.started_at",
        "r.finished_at",
        "p.project_code",
        "p.display_name AS project_display_name",
        "b.bug_key",
        "b.title AS bug_title",
        "b.defects4j_project",
        "b.defects4j_bug_id",
    ]
    if include_heavy:
        columns.extend(["r.report_json", "r.patch_diff"])

    sql = (
        f"SELECT {', '.join(columns)} FROM benchmark_runs r "
        "INNER JOIN benchmark_projects p ON p.id = r.project_id "
        "INNER JOIN benchmark_bugs b ON b.id = r.bug_id WHERE r.id = %s"
    )
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (run_id,))
            row = cursor.fetchone()
    if row is None:
        return None
    return _serialize_run(row, include_heavy=include_heavy)


def list_runs_for_user(
    user_id: int,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM benchmark_runs WHERE user_id = %s",
                (user_id,),
            )
            total = int((cursor.fetchone() or {}).get("total", 0))
            cursor.execute(
                """
                SELECT r.id, r.user_id, r.organization_id, r.project_id, r.bug_id,
                       r.model_key, r.run_mode, r.run_status, r.stage,
                       r.strategy, r.experiment_id,
                       r.pass_count, r.fail_count, r.total_tests,
                       r.duration_ms, r.credits_spent,
                       r.is_plausible, r.is_correct,
                       r.prompt_tokens, r.completion_tokens, r.total_tokens,
                       r.patch_lines_added, r.patch_lines_removed, r.llm_rounds,
                       r.failed_tests_before, r.failed_tests_after,
                       r.error_message, r.started_at, r.finished_at,
                       p.project_code, b.bug_key, b.defects4j_project, b.defects4j_bug_id
                FROM benchmark_runs r
                INNER JOIN benchmark_projects p ON p.id = r.project_id
                INNER JOIN benchmark_bugs b ON b.id = r.bug_id
                WHERE r.user_id = %s
                ORDER BY r.id DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            )
            rows = cursor.fetchall() or []
    return {
        "items": [_serialize_run(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def list_runs_for_admin(*, limit: int = 100) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.id, r.user_id, r.organization_id, r.project_id, r.bug_id,
                       r.model_key, r.run_mode, r.run_status, r.stage,
                       r.strategy, r.experiment_id,
                       r.pass_count, r.fail_count, r.total_tests,
                       r.duration_ms, r.credits_spent,
                       r.is_plausible, r.is_correct,
                       r.prompt_tokens, r.completion_tokens, r.total_tokens,
                       r.patch_lines_added, r.patch_lines_removed, r.llm_rounds,
                       r.failed_tests_before, r.failed_tests_after,
                       r.error_message, r.started_at, r.finished_at,
                       p.project_code, b.bug_key, b.defects4j_project, b.defects4j_bug_id,
                       u.email AS user_email, u.display_name AS user_display_name
                FROM benchmark_runs r
                INNER JOIN benchmark_projects p ON p.id = r.project_id
                INNER JOIN benchmark_bugs b ON b.id = r.bug_id
                LEFT JOIN users u ON u.id = r.user_id
                ORDER BY r.id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall() or []
    return [
        {
            **_serialize_run(row),
            "user_email": row.get("user_email"),
            "user_display_name": row.get("user_display_name"),
        }
        for row in rows
    ]


def recompute_leaderboard() -> None:
    """Aggregate completed benchmark runs into the leaderboard cache."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM benchmark_leaderboard")
            cursor.execute(
                """
                INSERT INTO benchmark_leaderboard
                    (project_id, model_key, sample_count, success_count, pass_rate,
                     avg_duration_ms, last_run_at)
                SELECT
                    r.project_id,
                    r.model_key,
                    COUNT(*) AS sample_count,
                    SUM(CASE WHEN r.run_status = 'completed' AND r.fail_count = 0 AND r.total_tests > 0 THEN 1 ELSE 0 END) AS success_count,
                    CASE WHEN COUNT(*) = 0 THEN 0
                         ELSE SUM(CASE WHEN r.run_status = 'completed' AND r.fail_count = 0 AND r.total_tests > 0 THEN 1 ELSE 0 END) / COUNT(*)
                    END AS pass_rate,
                    ROUND(AVG(r.duration_ms)) AS avg_duration_ms,
                    MAX(r.started_at) AS last_run_at
                FROM benchmark_runs r
                WHERE r.run_status IN ('completed', 'failed')
                GROUP BY r.project_id, r.model_key
                """
            )
        connection.commit()


def get_leaderboard() -> list[dict[str, Any]]:
    recompute_leaderboard()
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT l.project_id, l.model_key, l.sample_count, l.success_count,
                       l.pass_rate, l.avg_duration_ms, l.last_run_at,
                       p.project_code, p.display_name
                FROM benchmark_leaderboard l
                INNER JOIN benchmark_projects p ON p.id = l.project_id
                ORDER BY p.sort_order, l.pass_rate DESC, l.sample_count DESC
                """
            )
            rows = cursor.fetchall() or []
    return [
        {
            "project_id": int(r["project_id"]),
            "project_code": r["project_code"],
            "project_display_name": r["display_name"],
            "model_key": r["model_key"],
            "sample_count": int(r["sample_count"] or 0),
            "success_count": int(r["success_count"] or 0),
            "pass_rate": float(r["pass_rate"] or 0),
            "avg_duration_ms": int(r["avg_duration_ms"] or 0),
            "last_run_at": str(r["last_run_at"]) if r.get("last_run_at") else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Experiment helpers (used by the comparison-study runner)
# ---------------------------------------------------------------------------


def get_or_create_experiment(
    experiment_code: str,
    *,
    title: str | None = None,
    description: str | None = None,
    hypothesis: str | None = None,
    config: dict[str, Any] | None = None,
    created_by_user_id: int | None = None,
) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM benchmark_experiments WHERE experiment_code = %s",
                (experiment_code,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO benchmark_experiments
                        (experiment_code, title, description, hypothesis,
                         config_json, created_by_user_id, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                    """,
                    (
                        experiment_code,
                        title or experiment_code,
                        description,
                        hypothesis,
                        json.dumps(config or {}, ensure_ascii=False),
                        created_by_user_id,
                    ),
                )
                connection.commit()
                cursor.execute(
                    "SELECT * FROM benchmark_experiments WHERE experiment_code = %s",
                    (experiment_code,),
                )
                row = cursor.fetchone()
    return dict(row) if row else {}


def update_experiment_status(
    experiment_id: int,
    *,
    status: str | None = None,
    mark_started: bool = False,
    mark_finished: bool = False,
    config: dict[str, Any] | None = None,
) -> None:
    updates: list[str] = []
    values: list[Any] = []
    if status is not None:
        updates.append("status = %s")
        values.append(status)
    if mark_started:
        updates.append("started_at = COALESCE(started_at, NOW())")
    if mark_finished:
        updates.append("finished_at = NOW()")
    if config is not None:
        updates.append("config_json = %s")
        values.append(json.dumps(config, ensure_ascii=False))
    if not updates:
        return
    values.append(experiment_id)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE benchmark_experiments SET {', '.join(updates)} WHERE id = %s",
                values,
            )
        connection.commit()


def recount_experiment(experiment_id: int) -> None:
    """Refresh the cached aggregate counts on the experiment row."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE benchmark_experiments e
                SET
                    total_runs = (SELECT COUNT(*) FROM benchmark_runs r WHERE r.experiment_id = e.id),
                    completed_runs = (SELECT COUNT(*) FROM benchmark_runs r
                                      WHERE r.experiment_id = e.id AND r.run_status = 'completed'),
                    failed_runs = (SELECT COUNT(*) FROM benchmark_runs r
                                   WHERE r.experiment_id = e.id AND r.run_status = 'failed')
                WHERE e.id = %s
                """,
                (experiment_id,),
            )
        connection.commit()


def _serialize_experiment(row: dict[str, Any]) -> dict[str, Any]:
    config: Any = row.get("config_json")
    if isinstance(config, (bytes, bytearray)):
        config = config.decode("utf-8", errors="replace")
    if isinstance(config, str) and config:
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {"raw": config}
    return {
        "id": int(row["id"]),
        "experiment_code": row.get("experiment_code"),
        "title": row.get("title"),
        "description": row.get("description"),
        "hypothesis": row.get("hypothesis"),
        "status": row.get("status"),
        "total_runs": int(row.get("total_runs") or 0),
        "completed_runs": int(row.get("completed_runs") or 0),
        "failed_runs": int(row.get("failed_runs") or 0),
        "config": config or {},
        "created_by_user_id": (
            int(row["created_by_user_id"]) if row.get("created_by_user_id") is not None else None
        ),
        "created_at": str(row["created_at"]) if row.get("created_at") else None,
        "started_at": str(row["started_at"]) if row.get("started_at") else None,
        "finished_at": str(row["finished_at"]) if row.get("finished_at") else None,
    }


def list_experiments(*, limit: int = 100) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM benchmark_experiments ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cursor.fetchall() or []
    return [_serialize_experiment(row) for row in rows]


def get_experiment_results(experiment_id: int) -> dict[str, Any]:
    """Aggregated per-arm results for an experiment: pass/plausible/correct rates."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM benchmark_experiments WHERE id = %s",
                (experiment_id,),
            )
            experiment = cursor.fetchone()
            if experiment is None:
                return {}
            cursor.execute(
                """
                SELECT strategy, model_key,
                       COUNT(*) AS total,
                       SUM(CASE WHEN run_status = 'completed' THEN 1 ELSE 0 END) AS completed,
                       SUM(is_plausible) AS plausible,
                       SUM(is_correct) AS correct,
                       ROUND(AVG(duration_ms)) AS avg_duration_ms,
                       ROUND(AVG(total_tokens)) AS avg_tokens
                FROM benchmark_runs
                WHERE experiment_id = %s
                GROUP BY strategy, model_key
                ORDER BY strategy, model_key
                """,
                (experiment_id,),
            )
            arms = cursor.fetchall() or []
            cursor.execute(
                """
                SELECT b.bug_key, r.strategy, r.model_key, r.run_status,
                       r.is_plausible, r.is_correct, r.duration_ms, r.total_tokens,
                       r.error_message
                FROM benchmark_runs r
                INNER JOIN benchmark_bugs b ON b.id = r.bug_id
                WHERE r.experiment_id = %s
                ORDER BY b.bug_key, r.strategy, r.model_key
                """,
                (experiment_id,),
            )
            per_bug = cursor.fetchall() or []
    return {
        "experiment": _serialize_experiment(dict(experiment)),
        "arms": [
            {
                "strategy": row["strategy"],
                "model_key": row["model_key"],
                "total": int(row["total"] or 0),
                "completed": int(row["completed"] or 0),
                "plausible": int(row["plausible"] or 0),
                "correct": int(row["correct"] or 0),
                "plausible_rate": (float(row["plausible"] or 0) / int(row["total"])) if row["total"] else 0.0,
                "correct_rate": (float(row["correct"] or 0) / int(row["total"])) if row["total"] else 0.0,
                "avg_duration_ms": int(row["avg_duration_ms"] or 0),
                "avg_tokens": int(row["avg_tokens"] or 0),
            }
            for row in arms
        ],
        "per_bug": [
            {
                "bug_key": row["bug_key"],
                "strategy": row["strategy"],
                "model_key": row["model_key"],
                "run_status": row["run_status"],
                "is_plausible": bool(row["is_plausible"]),
                "is_correct": bool(row["is_correct"]),
                "duration_ms": int(row["duration_ms"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "error_message": row.get("error_message"),
            }
            for row in per_bug
        ],
    }


# ---------------------------------------------------------------------------
# Bulk bug import (used by scripts/import_defects4j_bugs.py)
# ---------------------------------------------------------------------------


def upsert_project(
    *,
    project_code: str,
    display_name: str,
    source_type: str,
    language: str,
    description: str | None = None,
    tags: str | None = None,
) -> int:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM benchmark_projects WHERE project_code = %s",
                (project_code,),
            )
            row = cursor.fetchone()
            if row:
                return int(row["id"])
            cursor.execute(
                """
                INSERT INTO benchmark_projects
                    (project_code, display_name, source_type, language,
                     description, tags, is_active, sort_order)
                VALUES (%s, %s, %s, %s, %s, %s, 1, 100)
                """,
                (project_code, display_name, source_type, language, description, tags),
            )
            project_id = int(cursor.lastrowid)
        connection.commit()
    return project_id


def upsert_bug(
    *,
    project_id: int,
    bug_key: str,
    title: str,
    severity: str = "normal",
    defects4j_project: str | None = None,
    defects4j_bug_id: int | None = None,
    description: str | None = None,
    tags: str | None = None,
) -> int:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM benchmark_bugs WHERE project_id = %s AND bug_key = %s",
                (project_id, bug_key),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    """
                    UPDATE benchmark_bugs
                    SET title = %s, description = %s, tags = %s,
                        defects4j_project = %s, defects4j_bug_id = %s,
                        severity = %s, is_active = 1
                    WHERE id = %s
                    """,
                    (title, description, tags, defects4j_project, defects4j_bug_id, severity, row["id"]),
                )
                connection.commit()
                return int(row["id"])
            cursor.execute(
                """
                INSERT INTO benchmark_bugs
                    (project_id, bug_key, title, severity, defects4j_project,
                     defects4j_bug_id, description, tags, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
                """,
                (
                    project_id,
                    bug_key,
                    title,
                    severity,
                    defects4j_project,
                    defects4j_bug_id,
                    description,
                    tags,
                ),
            )
            bug_id = int(cursor.lastrowid)
        connection.commit()
    return bug_id


def count_bugs_for_project(project_id: int) -> int:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS c FROM benchmark_bugs WHERE project_id = %s AND is_active = 1",
                (project_id,),
            )
            row = cursor.fetchone() or {}
    return int(row.get("c") or 0)


def get_benchmark_summary() -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total_projects FROM benchmark_projects WHERE is_active = 1"
            )
            total_projects = int((cursor.fetchone() or {}).get("total_projects", 0))
            cursor.execute("SELECT COUNT(*) AS total_bugs FROM benchmark_bugs WHERE is_active = 1")
            total_bugs = int((cursor.fetchone() or {}).get("total_bugs", 0))
            cursor.execute(
                """
                SELECT COUNT(*) AS total_runs,
                       SUM(CASE WHEN run_status = 'completed' THEN 1 ELSE 0 END) AS completed_runs,
                       SUM(CASE WHEN run_status = 'failed' THEN 1 ELSE 0 END) AS failed_runs
                FROM benchmark_runs
                """
            )
            run_summary = cursor.fetchone() or {}
    return {
        "total_projects": total_projects,
        "total_bugs": total_bugs,
        "total_runs": int(run_summary.get("total_runs") or 0),
        "completed_runs": int(run_summary.get("completed_runs") or 0),
        "failed_runs": int(run_summary.get("failed_runs") or 0),
    }
