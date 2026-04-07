from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable


MAX_WINDOW_LINES = 80
MAX_SEARCH_RESULTS = 30
MAX_OUTPUT_CHARS = 4_000
MAX_LISTED_PROJECT_FILES = 200


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class RepairToolContext:
    entrypoint: str
    file_map: dict[str, str]
    runtime_report: dict[str, Any] | None = None
    dependency_graph: dict[str, list[str]] | None = None
    reverse_dependency_graph: dict[str, list[str]] | None = None


def _line_window(lines: list[str], start_line: int, end_line: int) -> list[dict[str, Any]]:
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), end_line)
    return [
        {"line_number": index + 1, "content": lines[index]}
        for index in range(start_index, end_index)
    ]


def _node_end_lineno(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if isinstance(end_lineno, int) and end_lineno >= getattr(node, "lineno", 1):
        return end_lineno
    return getattr(node, "lineno", 1)


def _build_symbol_index_per_file(
    file_map: dict[str, str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    symbol_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    parse_errors: dict[str, str] = {}

    for path, code in file_map.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(code, filename=path)
        except SyntaxError as exc:
            parse_errors[path] = f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"
            continue

        symbol_index: dict[str, dict[str, Any]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_index[node.name] = {
                    "name": node.name,
                    "kind": "function",
                    "line_start": node.lineno,
                    "line_end": _node_end_lineno(node),
                }
                continue

            if isinstance(node, ast.ClassDef):
                method_names: list[str] = []
                symbol_index[node.name] = {
                    "name": node.name,
                    "kind": "class",
                    "line_start": node.lineno,
                    "line_end": _node_end_lineno(node),
                    "methods": method_names,
                }
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    qualified_name = f"{node.name}.{item.name}"
                    method_names.append(item.name)
                    symbol_index[qualified_name] = {
                        "name": qualified_name,
                        "kind": "method",
                        "parent": node.name,
                        "line_start": item.lineno,
                        "line_end": _node_end_lineno(item),
                    }

        symbol_indexes[path] = symbol_index

    return symbol_indexes, parse_errors


def _list_imports(code: str, filename: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as exc:
        return [], f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"

    imports: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(
                {
                    "kind": "import",
                    "line_number": node.lineno,
                    "modules": [alias.name for alias in node.names],
                }
            )
            continue
        if isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "kind": "from_import",
                    "line_number": node.lineno,
                    "module": node.module,
                    "level": node.level,
                    "names": [alias.name for alias in node.names],
                }
            )

    return imports, None


def build_repair_tools(context: RepairToolContext) -> list[FunctionTool]:
    file_map = dict(context.file_map)
    runtime_report = context.runtime_report or {}
    dependency_graph = context.dependency_graph or {}
    reverse_dependency_graph = context.reverse_dependency_graph or {}
    symbol_indexes, parse_errors = _build_symbol_index_per_file(file_map)
    import_indexes = {
        path: _list_imports(content, path)
        for path, content in file_map.items()
        if path.endswith(".py")
    }

    def resolve_path(arguments: dict[str, Any], *, default_to_entrypoint: bool = True) -> str:
        requested_path = arguments.get("path")
        if requested_path is None:
            if default_to_entrypoint:
                return context.entrypoint
            raise ValueError("`path` is required.")
        normalized_path = str(requested_path).strip().replace("\\", "/")
        if normalized_path not in file_map:
            raise ValueError(f"File not found in project: {normalized_path}")
        return normalized_path

    def get_failure_summary(_: dict[str, Any]) -> dict[str, Any]:
        execution = runtime_report.get("execution") or {}
        failure = runtime_report.get("failure") or {}
        return {
            "entrypoint": context.entrypoint,
            "returncode": execution.get("returncode"),
            "timed_out": execution.get("timed_out"),
            "has_stderr_output": execution.get("has_stderr_output"),
            "exception_type": failure.get("exception_type"),
            "exception_message": failure.get("exception_message"),
            "primary_frame": failure.get("primary_frame"),
            "traceback_frames": runtime_report.get("traceback_frames") or [],
            "focus_snippet": (runtime_report.get("source") or {}).get("focus_snippet", ""),
            "project": runtime_report.get("project") or {},
        }

    def get_runtime_output(arguments: dict[str, Any]) -> dict[str, Any]:
        stream_name = str(arguments.get("stream", "stderr")).strip().lower()
        if stream_name not in {"stdout", "stderr"}:
            raise ValueError("`stream` must be `stdout` or `stderr`.")
        execution = runtime_report.get("execution") or {}
        text = str(execution.get(stream_name, ""))
        max_chars = int(arguments.get("max_chars", MAX_OUTPUT_CHARS))
        max_chars = max(1, min(MAX_OUTPUT_CHARS, max_chars))
        return {
            "stream": stream_name,
            "truncated": len(text) > max_chars,
            "content": text[:max_chars],
        }

    def list_project_files(arguments: dict[str, Any]) -> dict[str, Any]:
        include_only_python = bool(arguments.get("python_only", False))
        selected_paths = [
            path for path in sorted(file_map.keys())
            if not include_only_python or path.endswith(".py")
        ]
        limited_paths = selected_paths[:MAX_LISTED_PROJECT_FILES]
        return {
            "entrypoint": context.entrypoint,
            "returned_file_count": len(limited_paths),
            "total_file_count": len(selected_paths),
            "files": [
                {
                    "path": path,
                    "line_count": len(file_map[path].splitlines()),
                }
                for path in limited_paths
            ],
        }

    def read_source_window(arguments: dict[str, Any]) -> dict[str, Any]:
        path = resolve_path(arguments)
        start_line = int(arguments["start_line"])
        end_line = int(arguments["end_line"])
        if start_line < 1 or end_line < start_line:
            raise ValueError("Invalid line range.")
        if end_line - start_line + 1 > MAX_WINDOW_LINES:
            raise ValueError(f"Line window must be at most {MAX_WINDOW_LINES} lines.")
        lines = file_map[path].splitlines()
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "lines": _line_window(lines, start_line, end_line),
        }

    def search_source(arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("`pattern` must be a non-empty string.")
        case_sensitive = bool(arguments.get("case_sensitive", False))
        max_results = int(arguments.get("max_results", MAX_SEARCH_RESULTS))
        max_results = max(1, min(MAX_SEARCH_RESULTS, max_results))

        target_paths: list[str]
        if "path" in arguments and arguments.get("path") is not None:
            target_paths = [resolve_path(arguments)]
        else:
            python_only = bool(arguments.get("python_only", False))
            target_paths = [
                path for path in sorted(file_map.keys())
                if not python_only or path.endswith(".py")
            ]

        needle = pattern if case_sensitive else pattern.lower()
        matches: list[dict[str, Any]] = []
        total_matches = 0
        for path in target_paths:
            for index, line in enumerate(file_map[path].splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if needle not in haystack:
                    continue
                total_matches += 1
                if len(matches) < max_results:
                    matches.append(
                        {
                            "path": path,
                            "line_number": index,
                            "content": line,
                        }
                    )

        return {
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "searched_file_count": len(target_paths),
            "total_matches": total_matches,
            "returned_matches": matches,
        }

    def list_symbols(arguments: dict[str, Any]) -> dict[str, Any]:
        if bool(arguments.get("project_scope", False)):
            files: list[dict[str, Any]] = []
            for path in sorted(symbol_indexes.keys()):
                imports, import_error = import_indexes.get(path, ([], None))
                files.append(
                    {
                        "path": path,
                        "parse_error": parse_errors.get(path),
                        "imports_parse_error": import_error,
                        "imports": imports,
                        "symbols": [value for key, value in symbol_indexes[path].items() if "." not in key],
                    }
                )
            return {"entrypoint": context.entrypoint, "files": files}

        path = resolve_path(arguments)
        if not path.endswith(".py"):
            raise ValueError("`list_symbols` only works for Python files.")
        imports, import_error = import_indexes.get(path, ([], None))
        return {
            "path": path,
            "parse_error": parse_errors.get(path),
            "imports_parse_error": import_error,
            "imports": imports,
            "symbols": [value for key, value in symbol_indexes.get(path, {}).items() if "." not in key],
        }

    def get_symbol_source(arguments: dict[str, Any]) -> dict[str, Any]:
        symbol_name = str(arguments.get("symbol_name", "")).strip()
        if not symbol_name:
            raise ValueError("`symbol_name` must be a non-empty string.")

        target_path = str(arguments.get("path", "")).strip().replace("\\", "/")
        search_paths = [target_path] if target_path else sorted(symbol_indexes.keys())
        matches: list[dict[str, Any]] = []
        for path in search_paths:
            if path and path not in symbol_indexes:
                continue
            if path and path not in file_map:
                continue
            candidates = symbol_indexes.get(path, {}) if path else None
            if candidates is not None:
                symbol = candidates.get(symbol_name)
                if symbol is None:
                    continue
                lines = file_map[path].splitlines()
                matches.append(
                    {
                        "path": path,
                        "kind": symbol["kind"],
                        "line_start": int(symbol["line_start"]),
                        "line_end": int(symbol["line_end"]),
                        "lines": _line_window(lines, int(symbol["line_start"]), int(symbol["line_end"])),
                    }
                )
                continue
            for project_path, project_symbols in symbol_indexes.items():
                symbol = project_symbols.get(symbol_name)
                if symbol is None:
                    continue
                lines = file_map[project_path].splitlines()
                matches.append(
                    {
                        "path": project_path,
                        "kind": symbol["kind"],
                        "line_start": int(symbol["line_start"]),
                        "line_end": int(symbol["line_end"]),
                        "lines": _line_window(lines, int(symbol["line_start"]), int(symbol["line_end"])),
                    }
                )

        if not matches:
            return {
                "symbol_name": symbol_name,
                "found": False,
                "available_files": sorted(symbol_indexes.keys()),
            }
        if len(matches) == 1:
            return {
                "symbol_name": symbol_name,
                "found": True,
                **matches[0],
            }
        return {
            "symbol_name": symbol_name,
            "found": True,
            "match_count": len(matches),
            "matches": matches[:10],
        }

    def get_file_dependencies(arguments: dict[str, Any]) -> dict[str, Any]:
        path = resolve_path(arguments)
        return {
            "path": path,
            "dependencies": dependency_graph.get(path, []),
            "reverse_dependencies": reverse_dependency_graph.get(path, []),
        }

    return [
        FunctionTool(
            name="get_failure_summary",
            description="Return the structured runtime failure summary, traceback frames, and project focus snippet.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=get_failure_summary,
        ),
        FunctionTool(
            name="get_runtime_output",
            description="Read stdout or stderr from the isolated local execution result.",
            parameters={
                "type": "object",
                "properties": {
                    "stream": {"type": "string", "enum": ["stdout", "stderr"]},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_OUTPUT_CHARS},
                },
                "required": ["stream"],
                "additionalProperties": False,
            },
            handler=get_runtime_output,
        ),
        FunctionTool(
            name="list_project_files",
            description="List analyzable files in the uploaded project.",
            parameters={
                "type": "object",
                "properties": {
                    "python_only": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            handler=list_project_files,
        ),
        FunctionTool(
            name="read_source_window",
            description="Read an exact source line window from any file in the uploaded project.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["start_line", "end_line"],
                "additionalProperties": False,
            },
            handler=read_source_window,
        ),
        FunctionTool(
            name="search_source",
            description="Search one file or the whole uploaded project by substring and return matching lines.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "python_only": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=search_source,
        ),
        FunctionTool(
            name="list_symbols",
            description="List top-level imports, functions, and classes in one Python file or across the whole project.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "project_scope": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            handler=list_symbols,
        ),
        FunctionTool(
            name="get_symbol_source",
            description="Return the exact source for a top-level function, class, or class method by name across the project.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["symbol_name"],
                "additionalProperties": False,
            },
            handler=get_symbol_source,
        ),
        FunctionTool(
            name="get_file_dependencies",
            description="Return direct dependency and reverse-dependency files for a Python file in the project.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=get_file_dependencies,
        ),
    ]
