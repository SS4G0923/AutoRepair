"""Run a benchmark comparison experiment and persist the results to MySQL.

A single experiment is parameterised by:

* ``bugs`` – the set of Defects4J bugs to attempt.
* ``arms`` – one or more (strategy, model) pairs.  Each arm represents one
  hypothesis we want to compare (e.g. "strong model, naive chat" vs
  "weak model, full AutoRepair pipeline").

Results are written to two places:

1. One row per (bug × arm) in ``benchmark_runs`` (with ``experiment_id``
   populated), so you can drill into per-bug data.
2. Aggregate counts on ``benchmark_experiments`` via ``recount_experiment``.

Example:

    ./.venv/bin/python -m backend.scripts.run_benchmark_experiment \\
        --code weak-vs-strong-v1 \\
        --arm "naive_chat:gemini-1.5-pro" \\
        --arm "full_pipeline:gemini-1.5-flash" \\
        --bugs Lang-1 Lang-3 Math-2 Math-5 \\
        --user-id 1

Or, skip ``--bugs`` and use ``--bugs-from-project Lang --limit 10`` to pick
the first 10 imported Lang bugs automatically.

The script prints a summary table on stdout and writes
``tmp/benchmark/<experiment_code>.report.json`` for downstream analysis
(thesis figures, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from backend.auth.store import get_db_connection
from backend.benchmark import store as bench_store
from backend.benchmark.repair_runner import (
    SUPPORTED_STRATEGIES,
    run_full_repair_for_bug,
)


def _parse_arm(raw: str) -> tuple[str, str]:
    """Parse ``strategy:model`` CLI argument."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"Invalid --arm `{raw}`; expected `strategy:model`.")
    strategy, model = raw.split(":", 1)
    strategy = strategy.strip()
    model = model.strip()
    if strategy not in SUPPORTED_STRATEGIES:
        raise argparse.ArgumentTypeError(
            f"Strategy `{strategy}` is not supported. Known: {sorted(SUPPORTED_STRATEGIES)}."
        )
    if not model:
        raise argparse.ArgumentTypeError(f"Model is empty in arm `{raw}`.")
    return strategy, model


def _resolve_bug(bug_key: str) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT b.id AS bug_id, b.bug_key, b.defects4j_project, b.defects4j_bug_id,
                       p.id AS project_id, p.project_code
                FROM benchmark_bugs b
                INNER JOIN benchmark_projects p ON p.id = b.project_id
                WHERE b.bug_key = %s OR b.id = %s
                LIMIT 1
                """,
                (bug_key, bug_key if bug_key.isdigit() else -1),
            )
            row = cursor.fetchone()
    if not row:
        raise SystemExit(f"Bug `{bug_key}` not found. Run the importer first.")
    return dict(row)


def _bugs_from_project(project_code: str, limit: int) -> list[str]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT b.bug_key FROM benchmark_bugs b
                INNER JOIN benchmark_projects p ON p.id = b.project_id
                WHERE p.project_code = %s AND b.is_active = 1
                ORDER BY b.defects4j_bug_id
                LIMIT %s
                """,
                (project_code, limit),
            )
            rows = cursor.fetchall() or []
    return [row["bug_key"] for row in rows]


def _print_summary(experiment_id: int) -> dict[str, Any]:
    data = bench_store.get_experiment_results(experiment_id)
    if not data:
        return {}
    exp = data["experiment"]
    arms = data["arms"]
    print("\n================ Experiment Summary ================")
    print(
        f"[{exp.get('experiment_code')}] {exp.get('title')}"
        f" — status={exp.get('status')} total_runs={exp.get('total_runs')}"
    )
    print(
        f"{'strategy':<18}{'model':<36}"
        f"{'runs':>6}{'ok':>6}{'plaus':>8}{'corr':>8}"
        f"{'plaus%':>9}{'corr%':>9}{'avg_ms':>10}{'avg_tok':>10}"
    )
    print("-" * 120)
    for arm in arms:
        print(
            f"{arm['strategy']:<18}{arm['model_key']:<36}"
            f"{arm['total']:>6}{arm['completed']:>6}"
            f"{arm['plausible']:>8}{arm['correct']:>8}"
            f"{arm['plausible_rate'] * 100:>8.1f}%{arm['correct_rate'] * 100:>8.1f}%"
            f"{arm['avg_duration_ms']:>10}{arm['avg_tokens']:>10}"
        )
    print("-" * 120)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a comparison benchmark experiment against Defects4J bugs."
    )
    parser.add_argument("--code", required=True, help="Unique experiment code / slug.")
    parser.add_argument("--title", default=None, help="Human-friendly title.")
    parser.add_argument("--description", default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        required=True,
        metavar="STRATEGY:MODEL",
        help="One arm of the experiment, e.g. naive_chat:gemini-1.5-pro.",
    )
    parser.add_argument(
        "--bugs",
        nargs="*",
        default=[],
        help="Space-separated bug keys (e.g. Lang-1 Math-2).",
    )
    parser.add_argument(
        "--bugs-from-project",
        default=None,
        help="Alternatively: pick the first N bugs from this project.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Used together with --bugs-from-project.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Attribute runs to this user (default: first admin).",
    )
    parser.add_argument(
        "--skip-failed-checkout",
        action="store_true",
        help="Continue the experiment even if a bug fails during checkout/inspect.",
    )
    args = parser.parse_args()

    if not args.bugs and not args.bugs_from_project:
        raise SystemExit("You must provide either --bugs or --bugs-from-project.")

    bug_keys: list[str]
    if args.bugs_from_project:
        bug_keys = _bugs_from_project(args.bugs_from_project, args.limit)
    else:
        bug_keys = list(args.bugs)
    if not bug_keys:
        raise SystemExit("No bugs resolved for the experiment.")

    bugs = [_resolve_bug(b) for b in bug_keys]

    # ---------------------------------------------------------------
    # Resolve a user to attribute runs to (needed for FK constraint).
    # ---------------------------------------------------------------
    user_id = args.user_id
    if user_id is None:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM users ORDER BY CASE role WHEN 'admin' THEN 0 ELSE 1 END, id LIMIT 1"
                )
                row = cursor.fetchone()
        if not row:
            raise SystemExit("No users exist; pass --user-id explicitly.")
        user_id = int(row["id"])

    config = {
        "arms": [{"strategy": s, "model_key": m} for s, m in args.arm],
        "bug_keys": bug_keys,
    }

    experiment = bench_store.get_or_create_experiment(
        experiment_code=args.code,
        title=args.title or args.code,
        description=args.description,
        hypothesis=args.hypothesis,
        config=config,
        created_by_user_id=user_id,
    )
    experiment_id = int(experiment["id"])
    bench_store.update_experiment_status(
        experiment_id,
        status="running",
        mark_started=True,
        config=config,
    )

    print(f"[experiment] id={experiment_id} code={args.code}")
    print(f"[experiment] bugs={bug_keys}")
    print(f"[experiment] arms={args.arm}")

    total_started = time.time()
    for bug in bugs:
        for strategy, model_key in args.arm:
            print(
                f"\n---- bug={bug['bug_key']} strategy={strategy} model={model_key} ----"
            )
            run_id = bench_store.create_run(
                user_id=user_id,
                organization_id=None,
                project_id=int(bug["project_id"]),
                bug_id=int(bug["bug_id"]),
                model_key=model_key,
                run_mode="full_repair",
                credits_spent=0,
                strategy=strategy,
                experiment_id=experiment_id,
            )
            try:
                result = run_full_repair_for_bug(
                    run_id=run_id,
                    user_id=user_id,
                    organization_id=None,
                    project_code=str(bug["project_code"]),
                    defects4j_project=str(bug["defects4j_project"]),
                    defects4j_bug_id=int(bug["defects4j_bug_id"]),
                    model_key=model_key,
                    strategy=strategy,
                    experiment_id=experiment_id,
                    force_checkout=True,
                )
                print(
                    f"    → status={result.get('status')} "
                    f"plausible={result.get('is_plausible')} "
                    f"correct={result.get('is_correct')} "
                    f"duration_ms={result.get('duration_ms')}"
                )
            except Exception as exc:  # pragma: no cover
                print(f"    !! run failed with unhandled exception: {exc}", file=sys.stderr)
                if not args.skip_failed_checkout:
                    raise

    bench_store.recount_experiment(experiment_id)
    bench_store.update_experiment_status(
        experiment_id,
        status="completed",
        mark_finished=True,
    )

    report = _print_summary(experiment_id)
    out_path = Path("tmp/benchmark") / f"{args.code}.report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    elapsed = time.time() - total_started
    print(
        f"\n[experiment] finished in {elapsed:.1f}s, summary written to {out_path}"
    )


if __name__ == "__main__":
    main()
