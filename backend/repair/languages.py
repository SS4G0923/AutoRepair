from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class LanguageSpec:
    key: str
    display_name: str
    primary_extension: str
    source_extensions: tuple[str, ...]
    entrypoint_candidates: tuple[str, ...]
    code_fence_labels: tuple[str, ...]


SUPPORTED_LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        key="python",
        display_name="Python",
        primary_extension="py",
        source_extensions=(".py",),
        entrypoint_candidates=("main.py", "app.py", "run.py", "manage.py", "__main__.py", "src/main.py"),
        code_fence_labels=("python", "py"),
    ),
    "javascript": LanguageSpec(
        key="javascript",
        display_name="JavaScript",
        primary_extension="js",
        source_extensions=(".js", ".mjs", ".cjs"),
        entrypoint_candidates=("index.js", "main.js", "app.js", "src/index.js", "src/main.js"),
        code_fence_labels=("javascript", "js", "node"),
    ),
    "typescript": LanguageSpec(
        key="typescript",
        display_name="TypeScript",
        primary_extension="ts",
        source_extensions=(".ts", ".mts", ".cts"),
        entrypoint_candidates=("index.ts", "main.ts", "app.ts", "src/index.ts", "src/main.ts"),
        code_fence_labels=("typescript", "ts"),
    ),
    "java": LanguageSpec(
        key="java",
        display_name="Java",
        primary_extension="java",
        source_extensions=(".java",),
        entrypoint_candidates=("Main.java", "App.java", "src/Main.java", "src/main/java/Main.java"),
        code_fence_labels=("java",),
    ),
    "go": LanguageSpec(
        key="go",
        display_name="Go",
        primary_extension="go",
        source_extensions=(".go",),
        entrypoint_candidates=("main.go", "cmd/main.go", "cmd/app/main.go"),
        code_fence_labels=("go", "golang"),
    ),
    "c": LanguageSpec(
        key="c",
        display_name="C",
        primary_extension="c",
        source_extensions=(".c",),
        entrypoint_candidates=("main.c", "src/main.c"),
        code_fence_labels=("c",),
    ),
    "cpp": LanguageSpec(
        key="cpp",
        display_name="C++",
        primary_extension="cpp",
        source_extensions=(".cpp", ".cc", ".cxx", ".c++"),
        entrypoint_candidates=("main.cpp", "main.cc", "src/main.cpp", "src/main.cc"),
        code_fence_labels=("cpp", "c++", "cc", "cxx"),
    ),
}


def normalize_language(value: str | None) -> str:
    candidate = (value or "python").strip().lower()
    if candidate == "c++":
        candidate = "cpp"
    if candidate not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language `{candidate}`. Supported values: {supported}.")
    return candidate


def get_language_spec(value: str | None) -> LanguageSpec:
    return SUPPORTED_LANGUAGES[normalize_language(value)]


def infer_language_from_path(path: str | None) -> str | None:
    if not path:
        return None
    suffix = PurePosixPath(path).suffix.lower()
    for language, spec in SUPPORTED_LANGUAGES.items():
        if suffix in spec.source_extensions:
            return language
    return None


def default_entrypoint_for_language(language: str) -> str:
    return f"main.{get_language_spec(language).primary_extension}"

