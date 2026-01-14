# inspector/source_utils.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_METHOD_DECL_RE = re.compile(
    r"""
    ^\s*
    (public|protected|private)?\s*
    (static\s+)?(final\s+)?(synchronized\s+)?(abstract\s+)?(native\s+)?\s*
    ([\w\<\>\[\], ?]+)\s+          # return type
    ([\w$]+)\s*                    # method name
    \([^;]*\)\s*                   # args
    (throws\s+[\w\.,\s]+)?\s*
    \{?\s*$
    """,
    re.VERBOSE,
)


@dataclass
class ResolvedSource:
    source_file: Path
    confidence: float
    reason: str


class SourceIndex:
    """
    Index source roots for fast resolution by file name.
    Supports multiple src roots (dir.src.classes + dir.src.tests, etc.).
    """

    def __init__(self, src_roots: Iterable[Path]):
        self.src_roots = [p.resolve() for p in src_roots]
        self._by_filename: dict[str, list[Path]] = {}

    def build(self) -> None:
        by_fn: dict[str, list[Path]] = {}
        for root in self.src_roots:
            if not root.exists():
                continue
            for p in root.rglob("*.java"):
                by_fn.setdefault(p.name, []).append(p.resolve())
        self._by_filename = by_fn

    def candidates_for_filename(self, file_name: str) -> list[Path]:
        return list(self._by_filename.get(file_name, []))


def _package_path_tokens(class_name: str) -> list[str]:
    """
    org.foo.Bar$Inner -> ["org","foo","Bar.java"] is not always mapped,
    but we can at least use package tokens for matching directory prefixes.
    """
    # Remove inner class
    base = class_name.split("$", 1)[0]
    parts = base.split(".")
    if len(parts) <= 1:
        return []
    # package tokens only
    return parts[:-1]


def resolve_source_for_frame(
    index: SourceIndex,
    frame_class_name: str,
    file_name: str,
) -> Optional[ResolvedSource]:
    """
    Resolve source file path based on file name + class name package hint.
    If multiple matches exist, pick best by matching package tokens with path.
    """
    cands = index.candidates_for_filename(file_name)
    if not cands:
        return None
    if len(cands) == 1:
        return ResolvedSource(cands[0], confidence=0.9, reason="unique_filename_match")

    pkg_tokens = _package_path_tokens(frame_class_name)
    if not pkg_tokens:
        return ResolvedSource(cands[0], confidence=0.5, reason="multiple_matches_no_package_hint")

    best = None
    best_score = -1
    for p in cands:
        parts = [x for x in p.parts]
        # score how many package tokens appear in-order in the path
        score = 0
        for tok in pkg_tokens:
            if tok in parts:
                score += 1
        if score > best_score:
            best_score = score
            best = p

    if best is None:
        return ResolvedSource(cands[0], confidence=0.4, reason="fallback_first_match")

    conf = 0.6 + min(0.3, 0.03 * best_score)
    return ResolvedSource(best, confidence=conf, reason=f"package_path_match(score={best_score})")


def extract_code_snippet(
    source_file: Path,
    line_number: int,
    window: int = 8,
) -> str:
    lines = source_file.read_text(errors="replace").splitlines()
    # line_number is 1-based
    start = max(0, line_number - window - 1)
    end = min(len(lines), line_number + window)
    out = []
    for i in range(start, end):
        prefix = ">>" if (i + 1) == line_number else "  "
        out.append(f"{prefix} {i+1:5d}: {lines[i]}")
    return "\n".join(out)


def extract_enclosing_method(
    source_file: Path,
    line_number: int,
    max_scan_up: int = 200,
    max_scan_down: int = 400,
) -> str:
    """
    Heuristic method extraction:
      - scan upward to find a likely method declaration
      - then scan downward counting braces to find end
    Not a full Java parser, but works well enough for localized context.
    """
    lines = source_file.read_text(errors="replace").splitlines()
    target_idx = max(0, min(len(lines) - 1, line_number - 1))

    # 1) find start
    start_idx = None
    for i in range(target_idx, max(-1, target_idx - max_scan_up), -1):
        if _METHOD_DECL_RE.match(lines[i]):
            start_idx = i
            break
        # handle constructors (no return type): "public Foo(...) {"
        if re.match(r"^\s*(public|protected|private)\s+[\w$]+\s*\([^;]*\)\s*\{?\s*$", lines[i]):
            start_idx = i
            break

    if start_idx is None:
        # fallback: return a wider snippet around the line
        return extract_code_snippet(source_file, line_number, window=20)

    # 2) find end by brace matching
    brace_balance = 0
    end_idx = start_idx
    opened = False

    for j in range(start_idx, min(len(lines), start_idx + max_scan_down)):
        line = lines[j]
        # naive brace count (strings/comments ignored)
        brace_balance += line.count("{")
        brace_balance -= line.count("}")
        if line.count("{") > 0:
            opened = True
        end_idx = j
        if opened and brace_balance <= 0 and j > start_idx:
            break

    # 3) render block
    block = []
    for k in range(start_idx, end_idx + 1):
        prefix = ">>" if k == target_idx else "  "
        block.append(f"{prefix} {k+1:5d}: {lines[k]}")
    return "\n".join(block)
