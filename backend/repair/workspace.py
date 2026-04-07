from __future__ import annotations

import ast
import base64
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MAX_PROJECT_FILES = 400
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_ANALYZED_TEXT_FILES = 300
MAX_ANALYZED_FILE_BYTES = 200_000
MAX_ANALYZED_TOTAL_BYTES = 1_500_000
ENTRYPOINT_CANDIDATES = (
    "main.py",
    "app.py",
    "run.py",
    "manage.py",
    "__main__.py",
    "src/main.py",
)
TEXT_FILE_SUFFIXES = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".env",
    ".rst",
}


@dataclass(frozen=True)
class ProjectFileInput:
    path: str
    content: str


@dataclass(frozen=True)
class PreparedProjectWorkspace:
    source_type: str
    root_dir: Path
    entrypoint: str
    file_map: dict[str, str]
    python_files: dict[str, str]
    module_to_path: dict[str, str]
    dependency_graph: dict[str, list[str]]
    reverse_dependency_graph: dict[str, list[str]]
    source_label: str

    @property
    def entrypoint_code(self) -> str:
        return self.file_map[self.entrypoint]

    @property
    def all_paths(self) -> list[str]:
        return sorted(self.file_map.keys())

    def to_summary(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_label": self.source_label,
            "entrypoint": self.entrypoint,
            "file_count": len(self.file_map),
            "python_file_count": len(self.python_files),
            "files": self.all_paths[:120],
            "entrypoint_dependencies": self.dependency_graph.get(self.entrypoint, []),
            "entrypoint_reverse_dependencies": self.reverse_dependency_graph.get(self.entrypoint, []),
        }


def normalize_project_path(raw_path: str, *, require_python: bool = False) -> str:
    value = raw_path.strip().replace("\\", "/")
    if not value:
        raise ValueError("Path must be a non-empty string.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Path must be a safe relative path.")
    if not pure.name:
        raise ValueError("Path must point to a file.")
    normalized = pure.as_posix()
    if require_python and not normalized.endswith(".py"):
        raise ValueError("Entrypoint must be a `.py` file.")
    return normalized


def _decode_zip_payload(zip_base64: str) -> bytes:
    try:
        archive_bytes = base64.b64decode(zip_base64, validate=True)
    except Exception as exc:
        raise ValueError("`project_zip_base64` must be a valid base64 ZIP archive.") from exc
    if not archive_bytes:
        raise ValueError("`project_zip_base64` decoded to an empty archive.")
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"ZIP archive must be at most {MAX_ARCHIVE_BYTES} bytes.")
    return archive_bytes


def _safe_target_path(root_dir: Path, relative_path: str) -> Path:
    normalized = normalize_project_path(relative_path)
    target = (root_dir / normalized).resolve()
    root_resolved = root_dir.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError("Path escapes the prepared project root.")
    return target


def _write_project_file(root_dir: Path, relative_path: str, content: str) -> None:
    target = _safe_target_path(root_dir, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _extract_zip_into(root_dir: Path, archive_bytes: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) > MAX_PROJECT_FILES:
            raise ValueError(f"ZIP archive may contain at most {MAX_PROJECT_FILES} files.")
        for member in members:
            relative_path = normalize_project_path(member.filename)
            target = _safe_target_path(root_dir, relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _clone_github_repo_into(root_dir: Path, repo_url: str, repo_ref: str | None) -> Path:
    clone_dir = root_dir / "repo"
    command = ["git", "clone", "--depth", "1"]
    if repo_ref:
        command.extend(["--branch", repo_ref])
    command.extend([repo_url, str(clone_dir)])

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Failed to clone GitHub repository: {message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to clone GitHub repository: {exc}") from exc

    return clone_dir


def _is_probably_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_FILE_SUFFIXES:
        return True
    if path.suffix == "":
        return path.name in {"requirements.txt", "Dockerfile", "Makefile"}
    return False


def _read_analyzable_files(source_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    file_map: dict[str, str] = {}
    python_files: dict[str, str] = {}
    total_bytes = 0
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and not any(part.startswith(".git") for part in path.parts)
    )
    if len(paths) > MAX_PROJECT_FILES:
        raise ValueError(f"Project may contain at most {MAX_PROJECT_FILES} files.")

    for path in paths:
        if len(file_map) >= MAX_ANALYZED_TEXT_FILES:
            break
        if not _is_probably_text_file(path):
            continue
        try:
            file_size = path.stat().st_size
        except OSError:
            continue
        if file_size > MAX_ANALYZED_FILE_BYTES:
            continue
        if total_bytes + file_size > MAX_ANALYZED_TOTAL_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative_path = path.relative_to(source_root).as_posix()
        file_map[relative_path] = content
        total_bytes += file_size
        if path.suffix.lower() == ".py":
            python_files[relative_path] = content

    return file_map, python_files


def _build_module_map(python_files: dict[str, str]) -> dict[str, str]:
    module_map: dict[str, str] = {}
    for relative_path in python_files:
        pure = PurePosixPath(relative_path)
        parts = list(pure.parts)
        if parts[-1] == "__init__.py":
            module_parts = parts[:-1]
        else:
            module_parts = parts[:-1] + [pure.stem]
        module_name = ".".join(part for part in module_parts if part)
        if module_name:
            module_map[module_name] = relative_path
    return module_map


def _resolve_import_target(
    *,
    current_path: str,
    module_name: str | None,
    level: int,
    module_to_path: dict[str, str],
) -> str | None:
    current_pure = PurePosixPath(current_path)
    current_parts = list(current_pure.parts)
    if current_parts[-1] == "__init__.py":
        package_parts = current_parts[:-1]
    else:
        package_parts = current_parts[:-1]

    if level > 0:
        if level > len(package_parts) + 1:
            return None
        base_parts = package_parts[: len(package_parts) - (level - 1)]
        absolute_parts = base_parts + ([part for part in (module_name or "").split(".") if part])
    else:
        absolute_parts = [part for part in (module_name or "").split(".") if part]

    if not absolute_parts:
        return None

    absolute_name = ".".join(absolute_parts)
    if absolute_name in module_to_path:
        return module_to_path[absolute_name]

    for candidate_name, candidate_path in module_to_path.items():
        if candidate_name == absolute_name or candidate_name.startswith(f"{absolute_name}."):
            return candidate_path
    return None


def _build_dependency_graph(
    python_files: dict[str, str],
    module_to_path: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    dependency_graph: dict[str, list[str]] = {path: [] for path in python_files}
    reverse_graph: dict[str, set[str]] = {path: set() for path in python_files}

    for current_path, content in python_files.items():
        try:
            tree = ast.parse(content, filename=current_path)
        except SyntaxError:
            continue
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _resolve_import_target(
                        current_path=current_path,
                        module_name=alias.name,
                        level=0,
                        module_to_path=module_to_path,
                    )
                    if resolved:
                        dependencies.add(resolved)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                resolved = _resolve_import_target(
                    current_path=current_path,
                    module_name=module_name,
                    level=node.level,
                    module_to_path=module_to_path,
                )
                if resolved:
                    dependencies.add(resolved)

        dependency_graph[current_path] = sorted(dependencies)
        for dependency in dependencies:
            reverse_graph.setdefault(dependency, set()).add(current_path)

    reverse_dependency_graph = {
        path: sorted(reverse_graph.get(path, set()))
        for path in python_files
    }
    return dependency_graph, reverse_dependency_graph


def _guess_entrypoint(file_map: dict[str, str], requested_entrypoint: str | None) -> str:
    if requested_entrypoint:
        normalized = normalize_project_path(requested_entrypoint, require_python=True)
        if normalized in file_map:
            return normalized
        raise ValueError(f"Entrypoint `{normalized}` was not found in the prepared project.")

    for candidate in ENTRYPOINT_CANDIDATES:
        if candidate in file_map:
            return candidate

    python_candidates = sorted(path for path in file_map if path.endswith(".py"))
    if python_candidates:
        return python_candidates[0]
    raise ValueError("Could not determine a Python entrypoint from the provided project.")


def build_project_runtime_inspection_report(
    workspace: PreparedProjectWorkspace,
    execution: dict[str, Any],
) -> dict[str, Any]:
    from backend.repair.sandbox import build_project_runtime_inspection_report as build_report

    return build_report(
        file_map=workspace.file_map,
        entrypoint=workspace.entrypoint,
        project_root=workspace.root_dir,
        dependency_graph=workspace.dependency_graph,
        reverse_dependency_graph=workspace.reverse_dependency_graph,
        execution=execution,
        source_type=workspace.source_type,
        source_label=workspace.source_label,
    )


@contextmanager
def prepare_project_workspace(
    *,
    code: str | None,
    filename: str | None,
    project_files: tuple[ProjectFileInput, ...],
    project_zip_base64: str | None,
    github_repo_url: str | None,
    github_ref: str | None,
    project_subdir: str | None,
) -> Iterator[PreparedProjectWorkspace]:
    source_count = sum(
        1
        for item in (code is not None, bool(project_files), bool(project_zip_base64), bool(github_repo_url))
        if item
    )
    if source_count != 1:
        raise ValueError(
            "Exactly one of `code`, `project_files`, `project_zip_base64`, or `github_repo_url` must be provided."
        )

    with tempfile.TemporaryDirectory(prefix="autorepair-project-", dir="/tmp") as tmp_dir:
        root_dir = Path(tmp_dir)
        source_root = root_dir
        source_type = "single_file"
        source_label = "inline-code"

        if code is not None:
            entrypoint = normalize_project_path(filename or "main.py", require_python=True)
            _write_project_file(source_root, entrypoint, code)
        elif project_files:
            source_type = "project_files"
            source_label = "inline-project-files"
            if len(project_files) > MAX_PROJECT_FILES:
                raise ValueError(f"`project_files` may contain at most {MAX_PROJECT_FILES} files.")
            for project_file in project_files:
                _write_project_file(source_root, project_file.path, project_file.content)
            entrypoint = filename or None
        elif project_zip_base64:
            source_type = "zip"
            source_label = "zip-upload"
            archive_bytes = _decode_zip_payload(project_zip_base64)
            _extract_zip_into(source_root, archive_bytes)
            entrypoint = filename or None
        else:
            source_type = "github"
            source_label = github_repo_url or "github-repo"
            source_root = _clone_github_repo_into(root_dir, github_repo_url or "", github_ref)
            entrypoint = filename or None

        if project_subdir:
            normalized_subdir = normalize_project_path(project_subdir)
            source_root = (source_root / normalized_subdir).resolve()
            if not source_root.is_dir():
                raise ValueError(f"`project_subdir` was not found: {normalized_subdir}")

        file_map, python_files = _read_analyzable_files(source_root)
        if not file_map:
            raise ValueError("No analyzable text files were found in the provided project.")

        resolved_entrypoint = _guess_entrypoint(file_map, entrypoint)
        module_to_path = _build_module_map(python_files)
        dependency_graph, reverse_dependency_graph = _build_dependency_graph(
            python_files,
            module_to_path,
        )

        yield PreparedProjectWorkspace(
            source_type=source_type,
            root_dir=source_root,
            entrypoint=resolved_entrypoint,
            file_map=file_map,
            python_files=python_files,
            module_to_path=module_to_path,
            dependency_graph=dependency_graph,
            reverse_dependency_graph=reverse_dependency_graph,
            source_label=source_label,
        )


@contextmanager
def materialize_patched_workspace(
    original_root: Path,
    patched_files: dict[str, str],
    original_known_files: set[str],
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="autorepair-patched-", dir="/tmp") as tmp_dir:
        patched_root = Path(tmp_dir) / "workspace"
        shutil.copytree(original_root, patched_root)

        for relative_path in original_known_files:
            if relative_path not in patched_files:
                target = patched_root / relative_path
                if target.exists():
                    target.unlink()

        for relative_path, content in patched_files.items():
            target = patched_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        yield patched_root
