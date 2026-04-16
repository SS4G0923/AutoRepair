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

from backend.repair.languages import (
    default_entrypoint_for_language,
    get_language_spec,
    infer_language_from_path,
)


MAX_PROJECT_FILES = 400
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_ANALYZED_TEXT_FILES = 300
MAX_ANALYZED_FILE_BYTES = 200_000
MAX_ANALYZED_TOTAL_BYTES = 1_500_000
TEXT_FILE_SUFFIXES = {
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".mts",
    ".cts",
    ".java",
    ".go",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
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
    source_root: Path
    language: str
    entrypoint: str
    file_map: dict[str, str]
    language_files: dict[str, str]
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
            "language": self.language,
            "entrypoint": self.entrypoint,
            "file_count": len(self.file_map),
            "language_file_count": len(self.language_files),
            "files": self.all_paths[:120],
            "entrypoint_dependencies": self.dependency_graph.get(self.entrypoint, []),
            "entrypoint_reverse_dependencies": self.reverse_dependency_graph.get(self.entrypoint, []),
        }


@dataclass(frozen=True)
class ProjectSourcePreview:
    source_type: str
    root_dir: Path
    source_root: Path
    source_label: str
    file_map: dict[str, str]


def normalize_project_path(raw_path: str, *, required_suffixes: tuple[str, ...] | None = None) -> str:
    value = raw_path.strip().replace("\\", "/")
    if not value:
        raise ValueError("Path must be a non-empty string.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Path must be a safe relative path.")
    if not pure.name:
        raise ValueError("Path must point to a file.")
    normalized = pure.as_posix()
    if required_suffixes and PurePosixPath(normalized).suffix.lower() not in required_suffixes:
        allowed = ", ".join(required_suffixes)
        raise ValueError(f"Entrypoint must use one of these suffixes: {allowed}.")
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


def _read_analyzable_files(
    source_root: Path,
    *,
    language: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    file_map: dict[str, str] = {}
    language_files: dict[str, str] = {}
    total_bytes = 0
    language_suffixes = get_language_spec(language).source_extensions if language is not None else None
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
        if language_suffixes is None:
            if infer_language_from_path(relative_path) is not None:
                language_files[relative_path] = content
        elif path.suffix.lower() in language_suffixes:
            language_files[relative_path] = content

    return file_map, language_files


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


def _guess_entrypoint(
    file_map: dict[str, str],
    requested_entrypoint: str | None,
    *,
    language: str,
) -> str:
    language_spec = get_language_spec(language)
    if requested_entrypoint:
        normalized = normalize_project_path(
            requested_entrypoint,
            required_suffixes=language_spec.source_extensions,
        )
        if normalized in file_map:
            return normalized
        raise ValueError(f"Entrypoint `{normalized}` was not found in the prepared project.")

    for candidate in language_spec.entrypoint_candidates:
        if candidate in file_map:
            return candidate

    language_candidates = sorted(
        path for path in file_map if PurePosixPath(path).suffix.lower() in language_spec.source_extensions
    )
    if language_candidates:
        return language_candidates[0]
    raise ValueError(
        f"Could not determine a {language_spec.display_name} entrypoint from the provided project."
    )


def build_project_runtime_inspection_report(
    workspace: PreparedProjectWorkspace,
    input_text: str | None,
    execution: dict[str, Any],
) -> dict[str, Any]:
    from backend.repair.sandbox import build_project_runtime_inspection_report as build_report

    return build_report(
        file_map=workspace.file_map,
        language=workspace.language,
        entrypoint=workspace.entrypoint,
        input_text=input_text,
        project_root=workspace.root_dir,
        dependency_graph=workspace.dependency_graph,
        reverse_dependency_graph=workspace.reverse_dependency_graph,
        execution=execution,
        source_type=workspace.source_type,
        source_label=workspace.source_label,
    )


@contextmanager
def _prepare_project_source(
    *,
    code: str | None,
    filename: str | None,
    language: str | None,
    project_files: tuple[ProjectFileInput, ...],
    project_zip_base64: str | None,
    github_repo_url: str | None,
    github_ref: str | None,
    project_subdir: str | None,
) -> Iterator[ProjectSourcePreview]:
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
        project_root = source_root
        source_type = "single_file"
        source_label = "inline-code"

        if code is not None:
            if language is None:
                raise ValueError("`language` is required when preparing inline code.")
            language_spec = get_language_spec(language)
            entrypoint = normalize_project_path(
                filename or default_entrypoint_for_language(language),
                required_suffixes=language_spec.source_extensions,
            )
            _write_project_file(project_root, entrypoint, code)
        elif project_files:
            source_type = "project_files"
            source_label = "inline-project-files"
            if len(project_files) > MAX_PROJECT_FILES:
                raise ValueError(f"`project_files` may contain at most {MAX_PROJECT_FILES} files.")
            for project_file in project_files:
                _write_project_file(project_root, project_file.path, project_file.content)
        elif project_zip_base64:
            source_type = "zip"
            source_label = "zip-upload"
            archive_bytes = _decode_zip_payload(project_zip_base64)
            _extract_zip_into(project_root, archive_bytes)
        else:
            source_type = "github"
            source_label = github_repo_url or "github-repo"
            source_root = _clone_github_repo_into(root_dir, github_repo_url or "", github_ref)
            project_root = source_root

        if project_subdir:
            normalized_subdir = normalize_project_path(project_subdir)
            project_root = (source_root / normalized_subdir).resolve()
            if not project_root.is_dir():
                raise ValueError(f"`project_subdir` was not found: {normalized_subdir}")

        file_map, _ = _read_analyzable_files(project_root, language=language)
        if not file_map:
            raise ValueError("No analyzable text files were found in the provided project.")

        yield ProjectSourcePreview(
            source_type=source_type,
            root_dir=project_root,
            source_root=source_root,
            source_label=source_label,
            file_map=file_map,
        )


def list_project_entrypoint_options(
    *,
    project_files: tuple[ProjectFileInput, ...] = (),
    project_zip_base64: str | None = None,
    github_repo_url: str | None = None,
    github_ref: str | None = None,
    project_subdir: str | None = None,
    preview_path: str | None = None,
) -> dict[str, Any]:
    with _prepare_project_source(
        code=None,
        filename=None,
        language=None,
        project_files=project_files,
        project_zip_base64=project_zip_base64,
        github_repo_url=github_repo_url,
        github_ref=github_ref,
        project_subdir=project_subdir,
    ) as preview:
        options = [
            {
                "path": path,
                "language": infer_language_from_path(path),
            }
            for path in sorted(preview.file_map.keys())
            if infer_language_from_path(path) is not None
        ]
        selected_preview_path = ""
        preview_content = ""
        if options:
            try:
                normalized_preview_path = (
                    normalize_project_path(preview_path)
                    if isinstance(preview_path, str) and preview_path.strip()
                    else ""
                )
            except ValueError:
                normalized_preview_path = ""
            available_paths = {str(item["path"]) for item in options}
            if normalized_preview_path in available_paths:
                selected_preview_path = normalized_preview_path
            else:
                selected_preview_path = str(options[0]["path"])
            preview_content = preview.file_map.get(selected_preview_path, "")
        return {
            "source_type": preview.source_type,
            "source_label": preview.source_label,
            "file_count": len(preview.file_map),
            "entrypoint_options": options,
            "preview_path": selected_preview_path,
            "preview_content": preview_content,
        }


@contextmanager
def prepare_project_workspace(
    *,
    code: str | None,
    filename: str | None,
    language: str,
    project_files: tuple[ProjectFileInput, ...],
    project_zip_base64: str | None,
    github_repo_url: str | None,
    github_ref: str | None,
    project_subdir: str | None,
) -> Iterator[PreparedProjectWorkspace]:
    with _prepare_project_source(
        code=code,
        filename=filename,
        language=language,
        project_files=project_files,
        project_zip_base64=project_zip_base64,
        github_repo_url=github_repo_url,
        github_ref=github_ref,
        project_subdir=project_subdir,
    ) as preview:
        file_map, language_files = _read_analyzable_files(preview.root_dir, language=language)
        if not file_map:
            raise ValueError("No analyzable text files were found in the provided project.")

        resolved_entrypoint = _guess_entrypoint(file_map, filename or None, language=language)
        if language == "python":
            module_to_path = _build_module_map(language_files)
            dependency_graph, reverse_dependency_graph = _build_dependency_graph(
                language_files,
                module_to_path,
            )
        else:
            module_to_path = {}
            dependency_graph = {}
            reverse_dependency_graph = {}

        yield PreparedProjectWorkspace(
            source_type=preview.source_type,
            root_dir=preview.root_dir,
            source_root=preview.source_root,
            language=language,
            entrypoint=resolved_entrypoint,
            file_map=file_map,
            language_files=language_files,
            module_to_path=module_to_path,
            dependency_graph=dependency_graph,
            reverse_dependency_graph=reverse_dependency_graph,
            source_label=preview.source_label,
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
