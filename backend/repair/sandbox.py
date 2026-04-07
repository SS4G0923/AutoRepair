from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRACEBACK_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$'
)


@dataclass(frozen=True)
class SandboxedExecutionResult:
    filename: str
    command: list[str]
    work_dir: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool
    output_truncated: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.has_stderr_output

    @property
    def has_stderr_output(self) -> bool:
        return bool(self.stderr.strip())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["has_stderr_output"] = self.has_stderr_output
        return data


def _build_snippet(code: str, line_number: int | None, window: int = 4) -> str:
    if line_number is None:
        return ""

    lines = code.splitlines()
    start = max(0, line_number - window - 1)
    end = min(len(lines), line_number + window)
    snippet_lines: list[str] = []
    for idx in range(start, end):
        prefix = ">>" if idx + 1 == line_number else "  "
        snippet_lines.append(f"{prefix} {idx + 1:4d}: {lines[idx]}")
    return "\n".join(snippet_lines)


def _extract_traceback_frames(stderr: str, filename: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for raw_line in stderr.splitlines():
        match = TRACEBACK_FRAME_RE.match(raw_line)
        if not match:
            continue
        file_name = match.group("file")
        if Path(file_name).name != Path(filename).name:
            continue
        frames.append(
            {
                "file": file_name,
                "line_number": int(match.group("line")),
                "function": match.group("func").strip(),
                "raw_line": raw_line.strip(),
            }
        )
    return frames


def _extract_project_traceback_frames(stderr: str, project_root: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    project_root_resolved = project_root.resolve()
    for raw_line in stderr.splitlines():
        match = TRACEBACK_FRAME_RE.match(raw_line)
        if not match:
            continue
        file_name = match.group("file")
        file_path = Path(file_name)
        try:
            relative_path = file_path.resolve().relative_to(project_root_resolved).as_posix()
        except Exception:
            continue
        if relative_path == "__autorepair_launcher__.py":
            continue
        frames.append(
            {
                "file": file_name,
                "path": relative_path,
                "line_number": int(match.group("line")),
                "function": match.group("func").strip(),
                "raw_line": raw_line.strip(),
            }
        )
    return frames


def _extract_exception(stderr: str) -> tuple[str | None, str | None]:
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            exc_type, message = stripped.split(":", 1)
            if exc_type.endswith("Error") or exc_type.endswith("Exception"):
                return exc_type.strip(), message.strip()
        if stripped.endswith("Error") or stripped.endswith("Exception"):
            return stripped, ""
        break
    return None, None


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _limit_resources(memory_limit_mb: int, cpu_time_sec: int, file_limit_bytes: int) -> None:
    import resource

    os.setsid()

    limits = [
        (resource.RLIMIT_CPU, (cpu_time_sec, cpu_time_sec + 1)),
        (resource.RLIMIT_AS, (memory_limit_mb * 1024 * 1024, memory_limit_mb * 1024 * 1024)),
        (resource.RLIMIT_FSIZE, (file_limit_bytes, file_limit_bytes)),
        (resource.RLIMIT_CORE, (0, 0)),
        (resource.RLIMIT_NOFILE, (32, 32)),
    ]
    for limit_name, limit_value in limits:
        try:
            resource.setrlimit(limit_name, limit_value)
        except (OSError, ValueError):
            continue


def _run_python_entrypoint_safely(
    *,
    work_dir: Path,
    filename: str,
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> SandboxedExecutionResult:
    start = time.time()
    timed_out = False
    truncated = False
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"
    command = ["python3", "-I", "-B", "-S", filename]
    env = {
        "HOME": str(work_dir),
        "PATH": os.getenv("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            text=True,
            preexec_fn=lambda: _limit_resources(memory_limit_mb, timeout_sec, max_output_chars),
        )
        try:
            returncode = process.wait(timeout=timeout_sec + 1)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            returncode = -signal.SIGKILL

    stdout_text, stdout_truncated = _truncate_text(
        stdout_path.read_text(encoding="utf-8", errors="replace"),
        max_output_chars,
    )
    stderr_text, stderr_truncated = _truncate_text(
        stderr_path.read_text(encoding="utf-8", errors="replace"),
        max_output_chars,
    )
    truncated = stdout_truncated or stderr_truncated
    end = time.time()
    return SandboxedExecutionResult(
        filename=filename,
        command=command,
        work_dir=str(work_dir),
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        duration_sec=end - start,
        timed_out=timed_out,
        output_truncated=truncated,
    )


def _run_python_entrypoint_via_launcher(
    *,
    work_dir: Path,
    filename: str,
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> SandboxedExecutionResult:
    launcher_name = "__autorepair_launcher__.py"
    launcher_path = work_dir / launcher_name
    launcher_code = (
        "import runpy\n"
        "import sys\n\n"
        "sys.path.insert(0, '')\n"
        f"runpy.run_path({filename!r}, run_name='__main__')\n"
    )
    launcher_path.write_text(launcher_code, encoding="utf-8")
    result = _run_python_entrypoint_safely(
        work_dir=work_dir,
        filename=launcher_name,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
    )
    return SandboxedExecutionResult(
        filename=filename,
        command=result.command,
        work_dir=result.work_dir,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_sec=result.duration_sec,
        timed_out=result.timed_out,
        output_truncated=result.output_truncated,
    )


def run_python_code_safely(
    code: str,
    *,
    filename: str = "main.py",
    timeout_sec: int = 5,
    memory_limit_mb: int = 256,
    max_output_chars: int = 20_000,
) -> SandboxedExecutionResult:
    with tempfile.TemporaryDirectory(prefix="autorepair-", dir="/tmp") as tmp_dir:
        work_dir = Path(tmp_dir)
        entrypoint = work_dir / filename
        entrypoint.write_text(code, encoding="utf-8")
        return _run_python_entrypoint_safely(
            work_dir=work_dir,
            filename=filename,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
        )


def run_python_project_safely(
    project_root: str | Path,
    *,
    filename: str,
    timeout_sec: int = 5,
    memory_limit_mb: int = 256,
    max_output_chars: int = 20_000,
) -> SandboxedExecutionResult:
    work_dir = Path(project_root)
    return _run_python_entrypoint_via_launcher(
        work_dir=work_dir,
        filename=filename,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
    )


def build_runtime_inspection_report(
    *,
    code: str,
    filename: str,
    execution: SandboxedExecutionResult,
) -> dict[str, Any]:
    traceback_frames = _extract_traceback_frames(execution.stderr, filename)
    exception_type, exception_message = _extract_exception(execution.stderr)
    primary_frame = traceback_frames[-1] if traceback_frames else None
    primary_line = primary_frame["line_number"] if primary_frame else None
    return {
        "language": "python",
        "filename": filename,
        "execution": execution.to_dict(),
        "failure": {
            "exception_type": exception_type,
            "exception_message": exception_message,
            "timed_out": execution.timed_out,
            "primary_frame": primary_frame,
        },
        "traceback_frames": traceback_frames,
        "source": {
            "code": code,
            "focus_snippet": _build_snippet(code, primary_line),
        },
    }


def build_project_runtime_inspection_report(
    *,
    file_map: dict[str, str],
    entrypoint: str,
    project_root: Path,
    dependency_graph: dict[str, list[str]],
    reverse_dependency_graph: dict[str, list[str]],
    execution: SandboxedExecutionResult,
    source_type: str,
    source_label: str,
) -> dict[str, Any]:
    traceback_frames = _extract_project_traceback_frames(execution.stderr, project_root)
    exception_type, exception_message = _extract_exception(execution.stderr)
    primary_frame = traceback_frames[-1] if traceback_frames else None
    focus_path = primary_frame["path"] if primary_frame else entrypoint
    focus_code = file_map.get(focus_path, "")
    primary_line = primary_frame["line_number"] if primary_frame else None

    return {
        "language": "python",
        "entrypoint": entrypoint,
        "execution": execution.to_dict(),
        "failure": {
            "exception_type": exception_type,
            "exception_message": exception_message,
            "timed_out": execution.timed_out,
            "primary_frame": primary_frame,
        },
        "traceback_frames": traceback_frames,
        "source": {
            "entrypoint_code": file_map.get(entrypoint, ""),
            "focus_path": focus_path,
            "focus_snippet": _build_snippet(focus_code, primary_line),
        },
        "project": {
            "source_type": source_type,
            "source_label": source_label,
            "file_count": len(file_map),
            "files": sorted(file_map.keys())[:120],
            "dependencies": dependency_graph.get(focus_path, []),
            "reverse_dependencies": reverse_dependency_graph.get(focus_path, []),
        },
    }
