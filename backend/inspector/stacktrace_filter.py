# inspector/stacktrace_filter.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class StackFrame:
    class_name: str          # e.g., org.apache.commons.lang3.math.NumberUtils
    method_name: str         # e.g., createInteger
    file_name: str           # e.g., NumberUtils.java
    line_number: int         # e.g., 684
    raw_line: str            # original "at ..." line
    index_in_trace: int      # 0 = topmost frame within the selected cause segment


@dataclass(frozen=True)
class ExceptionInfo:
    exception_type: str      # e.g., java.lang.IllegalArgumentException
    message: str             # e.g., "Foo must not be null"


@dataclass(frozen=True)
class StackTraceEvidence:
    """
    Evidence for a *single* selected exception cause segment (preferably the root cause).
    """
    exception: ExceptionInfo
    frames: list[StackFrame]
    raw_stack_text: str
    exception_chain: tuple[ExceptionInfo, ...] = ()


# ----------------------------
# Regex
# ----------------------------
# Accept optional module prefix:  at java.base/java.lang.Integer.parseInt(Integer.java:652)
# Capture: class + method + file + line
STACK_FRAME_RE = re.compile(
    r"^\s*at\s+(?:[\w.]+/)?([\w.$]+)\.([\w$<>]+)\(([^:()]+):(\d+)\)\s*$"
)

# Match both:
#   java.lang.IllegalArgumentException: msg
#   Caused by: java.lang.NullPointerException
EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:Caused by:\s*)?([\w.$]+(?:Exception|Error))(?::\s*(.*))?\s*$"
)


ARGUMENT_EXCEPTIONS = {
    "java.lang.IllegalArgumentException",
    "java.lang.NullPointerException",
    "java.lang.IndexOutOfBoundsException",
    "java.lang.ArrayIndexOutOfBoundsException",
    "java.lang.NumberFormatException",
}


# ----------------------------
# Frame classification
# ----------------------------
def is_test_frame(frame: StackFrame) -> bool:
    """
    Conservative: treat common test naming + junit packages as tests.
    Better accuracy comes from keeping class_name as "class only".
    """
    n = frame.class_name.lower()
    return (
        ".test." in n
        or n.endswith("test")
        or n.endswith("tests")
        or n.endswith("testcase")
        or n.startswith("org.junit.")
        or n.startswith("junit.")
    )


def is_standard_lib(frame: StackFrame) -> bool:
    return frame.class_name.startswith(("java.", "javax.", "sun.", "jdk.", "org.junit.", "junit."))


def is_build_or_runner_frame(frame: StackFrame) -> bool:
    """
    Defects4J often runs via Ant/JUnit runners; these frames are not helpful for localization.
    """
    n = frame.class_name
    return (
        n.startswith("org.apache.tools.ant.")
        or n.startswith("org.junit.runners.")
        or n.startswith("org.junit.internal.")
        or n.startswith("junit.framework.")
    )


def is_reflection_or_proxy(frame: StackFrame) -> bool:
    n = frame.class_name
    return (
        n.startswith("java.lang.reflect.")
        or n.startswith("jdk.internal.reflect.")
        or "CGLIB" in n
        or "ByteBuddy" in n
        or n.startswith("org.springframework.")
    )


def is_utility_class(frame: StackFrame) -> bool:
    name = frame.class_name.lower()
    return any(k in name for k in ["util", "utils", "utility", "helper", "common"])


def frame_kind(frame: StackFrame) -> str:
    """
    Coarse categories used for adjacency bonuses.
    """
    if is_standard_lib(frame):
        return "stdlib"
    if is_build_or_runner_frame(frame):
        return "runner"
    if is_reflection_or_proxy(frame):
        return "reflect"
    if is_test_frame(frame):
        return "test"
    return "project"


# ----------------------------
# Parsing: select root cause segment
# ----------------------------
def _parse_exception_chain(lines: list[str]) -> list[ExceptionInfo]:
    chain: list[ExceptionInfo] = []
    for line in lines:
        m = EXCEPTION_LINE_RE.match(line.strip())
        if m:
            exc_type, msg = m.groups()
            chain.append(ExceptionInfo(exception_type=exc_type, message=(msg or "").strip()))
    return chain


def _parse_frames_from_lines(lines: list[str]) -> list[StackFrame]:
    frames: list[StackFrame] = []
    idx = 0
    for line in lines:
        m = STACK_FRAME_RE.match(line)
        if not m:
            continue
        cls, method, file, line_no = m.groups()
        if not file.endswith(".java"):
            continue
        try:
            ln = int(line_no)
        except ValueError:
            continue
        frames.append(
            StackFrame(
                class_name=cls,
                method_name=method,
                file_name=file,
                line_number=ln,
                raw_line=line.rstrip("\n"),
                index_in_trace=idx,
            )
        )
        idx += 1
    return frames


def build_evidence_from_log(test_log: str) -> StackTraceEvidence:
    """
    More precise than "first exception line":
    - Parse exception chain
    - Split into segments by "Caused by:" (root cause = last cause segment)
    - Frames are collected per segment (index_in_trace resets per segment)
    """
    lines = test_log.splitlines()
    chain = _parse_exception_chain(lines)

    # Find all segment starts (exception lines). We treat each match as a new segment.
    segment_starts: list[int] = []
    for i, line in enumerate(lines):
        if EXCEPTION_LINE_RE.match(line.strip()):
            segment_starts.append(i)

    # If no exception line is found, treat the whole log as one segment.
    if not segment_starts:
        frames = _parse_frames_from_lines(lines)
        exc = ExceptionInfo("UnknownException", "")
        return StackTraceEvidence(exception=exc, frames=frames, raw_stack_text=test_log, exception_chain=tuple(chain))

    # Build segments [exc_line_i .. exc_line_{i+1}) and take the *last* (deepest) one.
    # This approximates "root cause" behavior.
    segments: list[tuple[ExceptionInfo, list[str]]] = []
    for si, start in enumerate(segment_starts):
        end = segment_starts[si + 1] if si + 1 < len(segment_starts) else len(lines)
        exc_line = lines[start].strip()
        m = EXCEPTION_LINE_RE.match(exc_line)
        if not m:
            continue
        exc_type, msg = m.groups()
        seg_exc = ExceptionInfo(exc_type, (msg or "").strip())
        seg_lines = lines[start:end]
        segments.append((seg_exc, seg_lines))

    # Choose deepest segment that actually has frames; otherwise fallback to last.
    chosen_exc, chosen_lines = segments[-1]
    chosen_frames = _parse_frames_from_lines(chosen_lines)
    if not chosen_frames:
        # fallback: try earlier segments
        for exc, seg_lines in reversed(segments[:-1]):
            fs = _parse_frames_from_lines(seg_lines)
            if fs:
                chosen_exc, chosen_lines, chosen_frames = exc, seg_lines, fs
                break

    return StackTraceEvidence(
        exception=chosen_exc,
        frames=chosen_frames,
        raw_stack_text=test_log,
        exception_chain=tuple(chain),
    )


# ----------------------------
# Ranking
# ----------------------------
@dataclass(frozen=True)
class FrameScore:
    frame: StackFrame
    score: float
    reasons: list[str]


def score_frame(
    frame: StackFrame,
    exception_type: str,
    source_file_exists: bool,
    prev_frame: Optional[StackFrame],
) -> FrameScore:
    """
    Improvements over your current scoring:
    - Still rewards early frames and source existence
    - Adds 'stdlib_boundary' bonus: if previous frame is stdlib/reflect/runner and current is project
      => often the correct repair site (especially for NumberFormatException/IllegalArgumentException)
    - Strongly drops test/runner frames
    """
    score = 0.0
    reasons: list[str] = []

    kind = frame_kind(frame)

    # Base: source existence is a strong signal of project relevance.
    if source_file_exists:
        score += 7.0
        reasons.append("source_exists(+7)")
    else:
        score -= 3.0
        reasons.append("source_missing(-3)")

    # Early frames within the selected cause segment tend to be closer to throw/cause.
    early_bonus = max(0.0, 7.0 - float(frame.index_in_trace))
    score += early_bonus
    if early_bonus > 0:
        reasons.append(f"early_frame(+{early_bonus:.1f})")

    # Penalize non-project categories heavily (we usually filter them out anyway)
    if kind == "stdlib":
        score -= 10.0
        reasons.append("standard_lib(-10)")
    elif kind == "test":
        score -= 9.0
        reasons.append("test_frame(-9)")
    elif kind == "runner":
        score -= 8.0
        reasons.append("runner_frame(-8)")
    elif kind == "reflect":
        score -= 4.0
        reasons.append("reflection_or_proxy(-4)")

    # Root-cause boundary heuristic:
    # If exception thrown inside stdlib/reflect/runner and we are the first project caller,
    # that project caller is often the bug site (bad input, wrong bounds, wrong type selection, etc.)
    if prev_frame is not None:
        prev_kind = frame_kind(prev_frame)
        if kind == "project" and prev_kind in {"stdlib", "reflect", "runner"}:
            bonus = 4.0 if exception_type in ARGUMENT_EXCEPTIONS else 2.5
            score += bonus
            reasons.append(f"boundary_after_{prev_kind}(+{bonus:.1f})")

    # Utility downweighting for argument-related exceptions (common overfitting location)
    if kind == "project" and is_utility_class(frame) and exception_type in ARGUMENT_EXCEPTIONS:
        score -= 2.5
        reasons.append("utility_and_argument_exception(-2.5)")

    # Slightly penalize inner classes unless it's the only candidate.
    if "$" in frame.class_name and kind == "project":
        score -= 0.5
        reasons.append("inner_class(-0.5)")

    return FrameScore(frame=frame, score=score, reasons=reasons)


def filter_and_rank_frames(
    evidence: StackTraceEvidence,
    source_exists_fn: Callable[[StackFrame], bool],
    max_candidates: int = 10,
) -> list[FrameScore]:
    """
    Main improvements:
    - We keep frames in evidence (including stdlib/test/runner) to compute adjacency bonuses.
    - Candidates are restricted to likely project frames.
    """
    exc_type = evidence.exception.exception_type
    frames = evidence.frames

    scored: list[FrameScore] = []
    for i, f in enumerate(frames):
        prev_f = frames[i - 1] if i > 0 else None
        exists = bool(source_exists_fn(f))

        # Candidate gate: keep only likely project repair targets.
        # (You can relax this if you later want to include test oracle extraction separately.)
        if is_standard_lib(f) or is_build_or_runner_frame(f) or is_reflection_or_proxy(f):
            continue
        if is_test_frame(f):
            continue

        scored.append(score_frame(f, exc_type, exists, prev_f))

    # Fallback: if we filtered everything, score all frames (still sorted)
    if not scored:
        for i, f in enumerate(frames[: max_candidates * 2]):
            prev_f = frames[i - 1] if i > 0 else None
            exists = bool(source_exists_fn(f))
            scored.append(score_frame(f, exc_type, exists, prev_f))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:max_candidates]


def choose_best_frame(ranked: list[FrameScore]) -> Optional[FrameScore]:
    return ranked[0] if ranked else None
