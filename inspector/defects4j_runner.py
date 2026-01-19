# inspector/defects4j_runner.py
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Defects4JError(RuntimeError):
    pass


def _resolve_defects4j_executable() -> str:
    """
    Resolve defects4j CLI location:
      - If $D4J_HOME is set: $D4J_HOME/framework/bin/defects4j
      - Else: rely on PATH
    """
    which = shutil.which("defects4j")
    if which:
        return which
    else:
        return "perl "+str(Path("..\\..\\defects4j\\framework\\bin\\defects4j").resolve())

    raise Defects4JError(
        "Cannot find defects4j executable. Set D4J_HOME or add defects4j to PATH."
    )


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    timeout_sec: Optional[int] = None,
) -> CommandResult:
    start = time.time()
    p = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    end = time.time()
    return CommandResult(
        cmd=list(cmd),
        cwd=str(cwd) if cwd else os.getcwd(),
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        duration_sec=end - start,
    )


class Defects4JRunner:
    """
    Thin wrapper around Defects4J CLI.
    This module does *not* depend on any LLM. It is benchmark plumbing.
    """

    def __init__(self, timeout_compile_sec: int = 600, timeout_test_sec: int = 1200):
        self.defects4j = _resolve_defects4j_executable()
        self.timeout_compile_sec = timeout_compile_sec
        self.timeout_test_sec = timeout_test_sec

    def checkout(
        self,
        project_id: str,
        bug_id: int,
        is_buggy: bool,
        work_dir: Path,
        force: bool = False,
    ) -> CommandResult:
        """
        checkout -p <pid> -v <bid><b|f> -w <work_dir>
        """
        work_dir = work_dir.resolve()
        if force and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

        work_dir.parent.mkdir(parents=True, exist_ok=True)

        version = f"{bug_id}{'b' if is_buggy else 'f'}"
        cmd = [self.defects4j, "checkout", "-p", project_id, "-v", version, "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=self.timeout_compile_sec)

    def compile(self, work_dir: Path) -> CommandResult:
        cmd = [self.defects4j, "compile", "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=self.timeout_compile_sec)

    def test_relevant(self, work_dir: Path) -> CommandResult:
        """
        defects4j test -r: executes only relevant developer-written tests.
        Always writes failing tests to work_dir/failing_tests.
        """
        cmd = [self.defects4j, "test", "-r", "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=self.timeout_test_sec)

    def test_all(self, work_dir: Path) -> CommandResult:
        cmd = [self.defects4j, "test", "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=self.timeout_test_sec)

    def test_single(self, work_dir: Path, test_method: str) -> CommandResult:
        """
        defects4j test -t <test_class>::<test_method>
        """
        cmd = [self.defects4j, "test", "-t", test_method, "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=self.timeout_test_sec)

    def export(self, work_dir: Path, prop: str) -> CommandResult:
        cmd = [self.defects4j, "export", "-p", prop, "-w", str(work_dir)]
        return run_cmd(cmd, timeout_sec=120)

    def export_value(self, work_dir: Path, prop: str) -> str:
        r = self.export(work_dir, prop)
        if r.returncode != 0:
            raise Defects4JError(f"defects4j export failed for {prop}: {r.stderr}")
        # Defects4J prints values to stdout; may contain trailing newline.
        return r.stdout.strip()

    @staticmethod
    def read_failing_tests(work_dir: Path) -> list[str]:
        """
        Defects4J writes failing tests to file named 'failing_tests' in work_dir.
        Format varies slightly by project; typically:
          <fully.qualified.TestClass>::testMethod
        """
        f = work_dir / "failing_tests"
        if not f.exists():
            return []
        lines = []
        for raw in f.read_text(errors="replace").splitlines():
            s = raw.strip()
            if not s:
                continue
            lines.append(s)
        return lines

    @staticmethod
    def parse_tests_trigger(export_stdout: str) -> list[str]:
        """
        defects4j export -p tests.trigger returns semicolon-separated test methods.
        """
        s = export_stdout.strip()
        if not s:
            return []
        # tests.trigger are separated by semicolons ';'
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts
