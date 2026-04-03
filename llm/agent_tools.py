from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable


MAX_WINDOW_LINES = 80
MAX_SEARCH_RESULTS = 20
MAX_OUTPUT_CHARS = 4_000


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
    filename: str
    code: str
    runtime_report: dict[str, Any] | None = None


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


def _build_symbol_index(code: str, filename: str) -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as exc:
        return {}, f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"

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

    return symbol_index, None


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
                    "names": [alias.name for alias in node.names],
                }
            )

    return imports, None


def build_repair_tools(context: RepairToolContext) -> list[FunctionTool]:
    lines = context.code.splitlines()
    symbol_index, symbol_error = _build_symbol_index(context.code, context.filename)
    imports, import_error = _list_imports(context.code, context.filename)
    runtime_report = context.runtime_report or {}

    def get_failure_summary(_: dict[str, Any]) -> dict[str, Any]:
        execution = runtime_report.get("execution") or {}
        failure = runtime_report.get("failure") or {}
        return {
            "filename": context.filename,
            "returncode": execution.get("returncode"),
            "timed_out": execution.get("timed_out"),
            "has_stderr_output": execution.get("has_stderr_output"),
            "exception_type": failure.get("exception_type"),
            "exception_message": failure.get("exception_message"),
            "primary_frame": failure.get("primary_frame"),
            "traceback_frames": runtime_report.get("traceback_frames") or [],
            "focus_snippet": (runtime_report.get("source") or {}).get("focus_snippet", ""),
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

    def read_source_window(arguments: dict[str, Any]) -> dict[str, Any]:
        start_line = int(arguments["start_line"])
        end_line = int(arguments["end_line"])
        if start_line < 1 or end_line < start_line:
            raise ValueError("Invalid line range.")
        if end_line - start_line + 1 > MAX_WINDOW_LINES:
            raise ValueError(f"Line window must be at most {MAX_WINDOW_LINES} lines.")
        return {
            "filename": context.filename,
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
        needle = pattern if case_sensitive else pattern.lower()
        matches: list[dict[str, Any]] = []
        total_matches = 0
        for index, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.lower()
            if needle not in haystack:
                continue
            total_matches += 1
            if len(matches) < max_results:
                matches.append({"line_number": index, "content": line})
        return {
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "total_matches": total_matches,
            "returned_matches": matches,
        }

    def list_symbols(_: dict[str, Any]) -> dict[str, Any]:
        top_level_symbols = [
            value
            for key, value in symbol_index.items()
            if "." not in key
        ]
        return {
            "filename": context.filename,
            "parse_error": symbol_error,
            "imports_parse_error": import_error,
            "imports": imports,
            "symbols": top_level_symbols,
        }

    def get_symbol_source(arguments: dict[str, Any]) -> dict[str, Any]:
        symbol_name = str(arguments.get("symbol_name", "")).strip()
        if not symbol_name:
            raise ValueError("`symbol_name` must be a non-empty string.")
        symbol = symbol_index.get(symbol_name)
        if symbol is None:
            return {
                "symbol_name": symbol_name,
                "found": False,
                "available_symbols": sorted(symbol_index.keys()),
            }
        start_line = int(symbol["line_start"])
        end_line = int(symbol["line_end"])
        return {
            "symbol_name": symbol_name,
            "found": True,
            "kind": symbol["kind"],
            "line_start": start_line,
            "line_end": end_line,
            "lines": _line_window(lines, start_line, end_line),
        }

    return [
        FunctionTool(
            name="get_failure_summary",
            description="Return the structured runtime failure summary, traceback frames, and focus snippet.",
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
            name="read_source_window",
            description="Read an exact source line window from the uploaded file.",
            parameters={
                "type": "object",
                "properties": {
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
            description="Search the uploaded source code by substring and return matching lines with numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=search_source,
        ),
        FunctionTool(
            name="list_symbols",
            description="List top-level imports, functions, and classes parsed from the uploaded file.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=list_symbols,
        ),
        FunctionTool(
            name="get_symbol_source",
            description="Return the exact source for a top-level function, class, or class method by name.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string"},
                },
                "required": ["symbol_name"],
                "additionalProperties": False,
            },
            handler=get_symbol_source,
        ),
    ]
