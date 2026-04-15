from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.repair.languages import get_language_spec


TRACEBACK_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$'
)
GENERIC_FRAME_RE = re.compile(
    r"(?P<file>[^:\s][^:\n]*?\.(?:py|js|mjs|cjs|ts|mts|cts|java|go|c|cc|cpp|cxx|c\+\+)):(?P<line>\d+)(?::(?P<column>\d+))?"
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


def _extract_generic_project_frames(stderr: str, project_root: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    project_root_resolved = project_root.resolve()
    for raw_line in stderr.splitlines():
        for match in GENERIC_FRAME_RE.finditer(raw_line):
            file_name = match.group("file")
            file_path = Path(file_name)
            if not file_path.is_absolute():
                file_path = (project_root / file_path).resolve()
            try:
                relative_path = file_path.resolve().relative_to(project_root_resolved).as_posix()
            except Exception:
                continue
            frames.append(
                {
                    "file": str(file_path),
                    "path": relative_path,
                    "line_number": int(match.group("line")),
                    "column_number": int(match.group("column")) if match.group("column") else None,
                    "function": None,
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
        (resource.RLIMIT_NOFILE, (256, 256)),
    ]
    for limit_name, limit_value in limits:
        try:
            resource.setrlimit(limit_name, limit_value)
        except (OSError, ValueError):
            continue


def _run_command_safely(
    *,
    work_dir: Path,
    filename: str,
    command: list[str],
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
    extra_env: dict[str, str] | None = None,
    file_limit_bytes: int | None = None,
) -> SandboxedExecutionResult:
    start = time.time()
    timed_out = False
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"
    env = {
        "HOME": str(work_dir),
        "PATH": os.getenv("PATH", ""),
        **(extra_env or {}),
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
            preexec_fn=lambda: _limit_resources(
                memory_limit_mb,
                timeout_sec,
                file_limit_bytes if file_limit_bytes is not None else max_output_chars,
            ),
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
        output_truncated=stdout_truncated or stderr_truncated,
    )


def _require_command(command_name: str) -> str:
    resolved = shutil.which(command_name)
    if resolved:
        return resolved
    raise RuntimeError(f"Required runtime command `{command_name}` is not installed or not on PATH.")


def _build_python_command(filename: str) -> tuple[list[str], dict[str, str]]:
    return (
        ["python3", "-I", "-B", "-S", filename],
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )


def _build_javascript_command(filename: str) -> tuple[list[str], dict[str, str]]:
    return ([_require_command("node"), filename], {})


def _build_typescript_command(filename: str) -> tuple[list[str], dict[str, str]]:
    ts_node = shutil.which("ts-node")
    if ts_node:
        return (
            [ts_node, "--transpile-only", filename],
            {"TS_NODE_COMPILER_OPTIONS": '{"module":"commonjs","moduleResolution":"node"}'},
        )
    return (
        [_require_command("npx"), "ts-node", "--transpile-only", filename],
        {"TS_NODE_COMPILER_OPTIONS": '{"module":"commonjs","moduleResolution":"node"}'},
    )


def _detect_java_main_class(work_dir: Path, filename: str) -> str:
    entrypoint = work_dir / filename
    content = entrypoint.read_text(encoding="utf-8", errors="replace")
    package_match = re.search(r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;", content, flags=re.MULTILINE)
    class_name = Path(filename).stem
    if package_match:
        return f"{package_match.group(1)}.{class_name}"
    return class_name


def _compile_and_run_java_project(
    *,
    work_dir: Path,
    filename: str,
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> SandboxedExecutionResult:
    compile_memory_limit_mb = max(memory_limit_mb, 1024)
    compile_file_limit_bytes = 100 * 1024 * 1024
    javac = _require_command("javac")
    java = _require_command("java")
    java_sources = sorted(str(path.relative_to(work_dir)) for path in work_dir.rglob("*.java"))
    if not java_sources:
        raise RuntimeError("No Java source files were found to compile.")
    compile_result = _run_command_safely(
        work_dir=work_dir,
        filename=filename,
        command=[javac, *java_sources],
        timeout_sec=timeout_sec,
        memory_limit_mb=compile_memory_limit_mb,
        max_output_chars=max_output_chars,
        file_limit_bytes=compile_file_limit_bytes,
    )
    if compile_result.returncode != 0:
        return compile_result
    main_class = _detect_java_main_class(work_dir, filename)
    return _run_command_safely(
        work_dir=work_dir,
        filename=filename,
        command=[java, "-cp", str(work_dir), main_class],
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
    )


def _compile_and_run_c_family_project(
    *,
    work_dir: Path,
    filename: str,
    compiler_name: str,
    source_suffixes: tuple[str, ...],
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> SandboxedExecutionResult:
    compile_memory_limit_mb = max(memory_limit_mb, 1024)
    compile_file_limit_bytes = 100 * 1024 * 1024
    compiler = _require_command(compiler_name)
    sources = sorted(
        str(path.relative_to(work_dir))
        for path in work_dir.rglob("*")
        if path.suffix.lower() in source_suffixes
    )
    if not sources:
        raise RuntimeError("No source files were found to compile.")
    binary_name = "__autorepair_binary__"
    compile_result = _run_command_safely(
        work_dir=work_dir,
        filename=filename,
        command=[compiler, "-O0", "-g", *sources, "-o", binary_name],
        timeout_sec=timeout_sec,
        memory_limit_mb=compile_memory_limit_mb,
        max_output_chars=max_output_chars,
        file_limit_bytes=compile_file_limit_bytes,
    )
    if compile_result.returncode != 0:
        return compile_result
    return _run_command_safely(
        work_dir=work_dir,
        filename=filename,
        command=[str(work_dir / binary_name)],
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
    )


def _build_go_command(work_dir: Path, filename: str) -> tuple[list[str], dict[str, str]]:
    go = _require_command("go")
    entrypoint_path = Path(filename)
    entrypoint_dir = (work_dir / entrypoint_path).parent
    sibling_sources = sorted(
        str(path.relative_to(work_dir))
        for path in entrypoint_dir.glob("*.go")
        if not path.name.endswith("_test.go")
    )
    if sibling_sources:
        return ([go, "run", *sibling_sources], {"GO111MODULE": "off"})
    return ([go, "run", filename], {})


def _run_python_entrypoint_safely(
    *,
    work_dir: Path,
    filename: str,
    timeout_sec: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> SandboxedExecutionResult:
    command, extra_env = _build_python_command(filename)
    return _run_command_safely(
        work_dir=work_dir,
        filename=filename,
        command=command,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
        extra_env=extra_env,
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


def run_project_safely(
    project_root: str | Path,
    *,
    filename: str,
    language: str,
    timeout_sec: int = 5,
    memory_limit_mb: int = 256,
    max_output_chars: int = 20_000,
) -> SandboxedExecutionResult:
    work_dir = Path(project_root)
    language_key = language.strip().lower()
    if language_key == "python":
        return run_python_project_safely(
            work_dir,
            filename=filename,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
        )
    if language_key == "javascript":
        command, extra_env = _build_javascript_command(filename)
        return _run_command_safely(
            work_dir=work_dir,
            filename=filename,
            command=command,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
            extra_env=extra_env,
        )
    if language_key == "typescript":
        command, extra_env = _build_typescript_command(filename)
        return _run_command_safely(
            work_dir=work_dir,
            filename=filename,
            command=command,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
            extra_env=extra_env,
        )
    if language_key == "java":
        return _compile_and_run_java_project(
            work_dir=work_dir,
            filename=filename,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
        )
    if language_key == "go":
        command, extra_env = _build_go_command(work_dir, filename)
        return _run_command_safely(
            work_dir=work_dir,
            filename=filename,
            command=command,
            timeout_sec=timeout_sec,
            memory_limit_mb=max(memory_limit_mb, 1024),
            max_output_chars=max_output_chars,
            extra_env=extra_env,
            file_limit_bytes=100 * 1024 * 1024,
        )
    if language_key == "c":
        return _compile_and_run_c_family_project(
            work_dir=work_dir,
            filename=filename,
            compiler_name="gcc",
            source_suffixes=(".c",),
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
        )
    if language_key == "cpp":
        return _compile_and_run_c_family_project(
            work_dir=work_dir,
            filename=filename,
            compiler_name="g++",
            source_suffixes=(".cpp", ".cc", ".cxx", ".c++"),
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            max_output_chars=max_output_chars,
        )
    raise RuntimeError(f"Unsupported runtime language: {language_key}")


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
    language: str,
    entrypoint: str,
    project_root: Path,
    dependency_graph: dict[str, list[str]],
    reverse_dependency_graph: dict[str, list[str]],
    execution: SandboxedExecutionResult,
    source_type: str,
    source_label: str,
) -> dict[str, Any]:
    if language == "python":
        traceback_frames = _extract_project_traceback_frames(execution.stderr, project_root)
    else:
        traceback_frames = _extract_generic_project_frames(execution.stderr, project_root)
    exception_type, exception_message = _extract_exception(execution.stderr)
    primary_frame = traceback_frames[-1] if traceback_frames else None
    focus_path = primary_frame["path"] if primary_frame else entrypoint
    focus_code = file_map.get(focus_path, "")
    primary_line = primary_frame["line_number"] if primary_frame else None

    return {
        "language": language,
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
            "language_display_name": get_language_spec(language).display_name,
            "file_count": len(file_map),
            "files": sorted(file_map.keys())[:120],
            "dependencies": dependency_graph.get(focus_path, []),
            "reverse_dependencies": reverse_dependency_graph.get(focus_path, []),
        },
    }
