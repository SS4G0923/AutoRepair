# inspector/source_utils.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

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


# A fairly permissive method/ctor “header” detector for the first line of a signature.
_METHOD_HEAD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|\s)+"
    r"[\w\<\>\[\], ?.$]+?\s+[\w$]+\s*\(.*$"
)
_CTOR_HEAD_RE = re.compile(
    r"^\s*(?:public|protected|private)\s+[\w$]+\s*\(.*$"
)

_CONTROL_KEYWORDS = ("if", "for", "while", "switch", "catch", "do", "try", "synchronized")


@dataclass
class _LexState:
    in_block_comment: bool = False
    in_string: bool = False
    in_char: bool = False
    escape: bool = False


def _strip_java_line_for_braces(line: str, st: _LexState) -> str:
    """
    Remove content inside strings/chars/comments so brace counting is accurate.
    Maintains state across lines for block comments and strings.
    """
    out = []
    i = 0
    n = len(line)

    while i < n:
        c = line[i]
        nxt = line[i + 1] if i + 1 < n else ""

        # Block comment
        if st.in_block_comment:
            if c == "*" and nxt == "/":
                st.in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        # Line comment
        if not st.in_string and not st.in_char and c == "/" and nxt == "/":
            break

        # Start block comment
        if not st.in_string and not st.in_char and c == "/" and nxt == "*":
            st.in_block_comment = True
            i += 2
            continue

        # String literal
        if st.in_string:
            if st.escape:
                st.escape = False
                i += 1
                continue
            if c == "\\":
                st.escape = True
                i += 1
                continue
            if c == '"':
                st.in_string = False
            i += 1
            continue

        # Char literal
        if st.in_char:
            if st.escape:
                st.escape = False
                i += 1
                continue
            if c == "\\":
                st.escape = True
                i += 1
                continue
            if c == "'":
                st.in_char = False
            i += 1
            continue

        # Enter string/char
        if c == '"':
            st.in_string = True
            i += 1
            continue
        if c == "'":
            st.in_char = True
            i += 1
            continue

        # Keep non-comment, non-string characters
        out.append(c)
        i += 1

    return "".join(out)


def _looks_like_signature_start(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Avoid grabbing control statements
    for kw in _CONTROL_KEYWORDS:
        if s.startswith(kw + " " ) or s.startswith(kw + "("):
            return False
    if _METHOD_HEAD_RE.match(line) or _CTOR_HEAD_RE.match(line):
        return True
    return False


def _find_signature_span(lines: list[str], anchor_idx: int, max_scan_up: int) -> Optional[Tuple[int, int]]:
    """
    Find a (start_idx, end_idx) span of the method signature (may be multiline),
    where end_idx is the last signature line (may contain '{' or ')', etc).
    """
    start_idx = None
    for i in range(anchor_idx, max(-1, anchor_idx - max_scan_up), -1):
        if _looks_like_signature_start(lines[i]):
            start_idx = i
            break
    if start_idx is None:
        return None

    # Extend downward until parentheses balance closes (signature may span multiple lines).
    paren = 0
    st = _LexState()
    end_idx = start_idx
    saw_paren = False

    for j in range(start_idx, min(len(lines), start_idx + 80)):
        clean = _strip_java_line_for_braces(lines[j], st)  # also strips comments/strings
        for ch in clean:
            if ch == "(":
                paren += 1
                saw_paren = True
            elif ch == ")":
                paren -= 1
        end_idx = j
        if saw_paren and paren <= 0:
            break

    return (start_idx, end_idx)


def _include_leading_javadoc_and_annotations(lines: list[str], sig_start: int) -> int:
    """
    Pull in contiguous annotations and Javadoc immediately above the signature.
    """
    i = sig_start - 1
    while i >= 0:
        s = lines[i].strip()
        if not s:
            # stop at blank line boundary
            break
        if s.startswith("@"):
            i -= 1
            continue
        # Javadoc block
        if s.startswith("*/") or s.startswith("*") or s.startswith("/**") or s.startswith("/*"):
            i -= 1
            continue
        break
    return i + 1


def extract_enclosing_method(
    source_file: Path,
    line_number: int,
    max_scan_up: int = 250,
    max_scan_down: int = 800,
) -> str:
    """
    Extract the enclosing Java method block (signature + body) around line_number.

    Precision improvements:
    - multiline signature support
    - include Javadoc/annotations
    - brace matching ignores braces in strings/comments
    """
    lines = source_file.read_text(errors="replace").splitlines()
    if not lines:
        return ""

    target_idx = max(0, min(len(lines) - 1, line_number - 1))

    span = _find_signature_span(lines, target_idx, max_scan_up=max_scan_up)
    if span is None:
        # fallback: return a wide snippet
        start = max(0, target_idx - 25)
        end = min(len(lines), target_idx + 25)
        return "\n".join(f"{'>>' if i==target_idx else '  '} {i+1:5d}: {lines[i]}" for i in range(start, end))

    sig_start, sig_end = span
    sig_start = _include_leading_javadoc_and_annotations(lines, sig_start)

    # Find the opening brace '{' that starts the method body
    st = _LexState()
    open_line = None
    open_pos = None

    for j in range(sig_end, min(len(lines), sig_end + 40)):
        clean = _strip_java_line_for_braces(lines[j], st)
        pos = clean.find("{")
        if pos != -1:
            open_line = j
            open_pos = pos
            break
        # Interface/abstract method could end with ';'
        if ";" in clean and "(" in clean and ")" in clean:
            # No body
            start = max(0, target_idx - 25)
            end = min(len(lines), target_idx + 25)
            return "\n".join(f"{'>>' if i==target_idx else '  '} {i+1:5d}: {lines[i]}" for i in range(start, end))

    if open_line is None:
        # fallback
        start = max(0, target_idx - 25)
        end = min(len(lines), target_idx + 25)
        return "\n".join(f"{'>>' if i==target_idx else '  '} {i+1:5d}: {lines[i]}" for i in range(start, end))

    # Brace match from the opening brace
    brace = 0
    st2 = _LexState()
    end_idx = open_line

    for k in range(open_line, min(len(lines), open_line + max_scan_down)):
        clean = _strip_java_line_for_braces(lines[k], st2)
        # Count braces
        for ch in clean:
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
        end_idx = k
        if brace == 0 and k > open_line:
            break

    # Render block
    block = []
    for i in range(sig_start, end_idx + 1):
        prefix = ">>" if i == target_idx else "  "
        block.append(f"{prefix} {i+1:5d}: {lines[i]}")
    return "\n".join(block)
