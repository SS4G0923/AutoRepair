"""Bulk-import every active Defects4J bug into the benchmark tables.

Usage:

    # Import all projects (Chart, Closure, Lang, Math, Time, ...)
    ./.venv/bin/python -m backend.scripts.import_defects4j_bugs

    # Import only a subset
    ./.venv/bin/python -m backend.scripts.import_defects4j_bugs --projects Lang Math

    # Limit bugs per project (useful for smoke-testing)
    ./.venv/bin/python -m backend.scripts.import_defects4j_bugs --limit-per-project 5

The script reads each project's ``active-bugs.csv`` under
``$D4J_HOME/framework/projects/<PID>/``, and upserts rows into
``benchmark_projects`` / ``benchmark_bugs``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from backend.benchmark import store as bench_store


PROJECT_DISPLAY_NAMES: dict[str, str] = {
    "Chart": "JFreeChart",
    "Cli": "Apache Commons CLI",
    "Closure": "Google Closure Compiler",
    "Codec": "Apache Commons Codec",
    "Collections": "Apache Commons Collections",
    "Compress": "Apache Commons Compress",
    "Csv": "Apache Commons CSV",
    "Gson": "Google Gson",
    "JacksonCore": "Jackson Core",
    "JacksonDatabind": "Jackson Databind",
    "JacksonXml": "Jackson XML",
    "Jsoup": "Jsoup",
    "JxPath": "Apache Commons JxPath",
    "Lang": "Apache Commons Lang",
    "Math": "Apache Commons Math",
    "Mockito": "Mockito",
    "Time": "Joda-Time",
}


def _resolve_defects4j_home(explicit: str | None) -> Path:
    candidate = explicit or os.getenv("D4J_HOME")
    if not candidate:
        raise SystemExit(
            "Defects4J installation not found. Pass --d4j-home or set $D4J_HOME."
        )
    path = Path(candidate).expanduser().resolve()
    if not (path / "framework" / "bin" / "defects4j").exists():
        raise SystemExit(f"{path} does not look like a Defects4J checkout.")
    return path


def _iter_active_bugs(project_dir: Path) -> list[dict[str, str]]:
    csv_path = project_dir / "active-bugs.csv"
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _load_modified_classes(project_dir: Path, bug_id: str) -> str | None:
    """Grab the modified classes list (first one or two) as a hint for the title."""
    mc_file = project_dir / "modified_classes" / f"{bug_id}.src"
    if not mc_file.exists():
        return None
    try:
        lines = [line.strip() for line in mc_file.read_text().splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    return f"{lines[0]} (+{len(lines) - 1} more)"


def _import_project(
    *,
    d4j_home: Path,
    project_id: str,
    limit_per_project: int | None,
) -> tuple[int, int]:
    project_dir = d4j_home / "framework" / "projects" / project_id
    if not project_dir.exists():
        print(f"  [skip] {project_id}: project dir not found", file=sys.stderr)
        return 0, 0

    display_name = PROJECT_DISPLAY_NAMES.get(project_id, project_id)
    project_row_id = bench_store.upsert_project(
        project_code=project_id,
        display_name=display_name,
        source_type="defects4j",
        language="java",
        description=f"Defects4J project {project_id} ({display_name}).",
        tags="defects4j",
    )

    bugs = _iter_active_bugs(project_dir)
    if not bugs:
        return 0, 0

    seen_before = bench_store.count_bugs_for_project(project_row_id)
    imported = 0
    for index, row in enumerate(bugs, start=1):
        if limit_per_project is not None and imported >= limit_per_project:
            break
        bug_number = row.get("bug.id") or row.get("bug_id")
        if not bug_number or not bug_number.isdigit():
            continue
        bug_number_int = int(bug_number)
        report_id = row.get("report.id") or row.get("report_id") or f"{project_id}-{bug_number}"
        modified = _load_modified_classes(project_dir, bug_number)
        title = f"{project_id}-{bug_number_int} ({report_id})"
        description = (
            f"Defects4J active bug {project_id}-{bug_number_int}. "
            f"Upstream ticket: {report_id}. "
            f"Modified: {modified or 'unknown'}. "
            f"Buggy commit: {row.get('revision.id.buggy')}. "
            f"Fixed commit:  {row.get('revision.id.fixed')}."
        )
        bench_store.upsert_bug(
            project_id=project_row_id,
            bug_key=f"{project_id}-{bug_number_int}",
            title=title,
            severity="normal",
            defects4j_project=project_id,
            defects4j_bug_id=bug_number_int,
            description=description,
            tags="defects4j,active",
        )
        imported += 1
    seen_after = bench_store.count_bugs_for_project(project_row_id)
    return imported, max(0, seen_after - seen_before)


def import_all(
    *,
    d4j_home: str | Path | None = None,
    projects: list[str] | None = None,
    limit_per_project: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Programmatic entry point. Returns a summary dict suitable for API responses."""
    resolved_home = _resolve_defects4j_home(str(d4j_home) if d4j_home else None)
    targets = list(projects) if projects else sorted(PROJECT_DISPLAY_NAMES.keys())

    per_project: list[dict[str, object]] = []
    total_imported = 0
    total_new = 0
    for project_id in targets:
        if verbose:
            print(f"  -> {project_id} ... ", end="", flush=True)
        imported, new = _import_project(
            d4j_home=resolved_home,
            project_id=project_id,
            limit_per_project=limit_per_project,
        )
        total_imported += imported
        total_new += new
        per_project.append(
            {"project_code": project_id, "upserted": imported, "new": new}
        )
        if verbose:
            print(f"upserted {imported} bugs (+{new} new).")

    return {
        "d4j_home": str(resolved_home),
        "projects": per_project,
        "total_imported": total_imported,
        "total_new": total_new,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-import Defects4J bugs.")
    parser.add_argument("--d4j-home", default=None, help="Path to Defects4J checkout.")
    parser.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="Subset of Defects4J project IDs. Default: all known projects.",
    )
    parser.add_argument(
        "--limit-per-project",
        type=int,
        default=None,
        help="Import at most N bugs per project (useful for smoke-testing).",
    )
    args = parser.parse_args()

    print(f"[import] Using Defects4J at {_resolve_defects4j_home(args.d4j_home)}")
    summary = import_all(
        d4j_home=args.d4j_home,
        projects=args.projects,
        limit_per_project=args.limit_per_project,
        verbose=True,
    )
    print(
        f"[import] Done. Upserted {summary['total_imported']} bugs total "
        f"(+{summary['total_new']} new)."
    )


if __name__ == "__main__":
    main()
