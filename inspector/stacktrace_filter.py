# inspector/stacktrace_filter.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StackFrame:
    class_name: str          # e.g., org.apache.commons.lang3.StringUtils
    file_name: str           # e.g., StringUtils.java
    line_number: int         # e.g., 123
    raw_line: str            # original "at ..." line
    index_in_trace: int      # 0 = topmost frame in the printed trace


@dataclass(frozen=True)
class ExceptionInfo:
    exception_type: str      # e.g., java.lang.IllegalArgumentException
    message: str             # e.g., "Foo must not be null"


@dataclass(frozen=True)
class StackTraceEvidence:
    exception: ExceptionInfo
    frames: list[StackFrame]
    raw_stack_text: str


# Typical "at ..." line
STACK_FRAME_RE = re.compile(r"^\s*at\s+([\w.$]+)\(([^:()]+):(\d+)\)\s*$")

# "Caused by: ..." and also plain exception lines
# Examples:
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
}


def parse_stack_frames(test_log: str) -> list[StackFrame]:
    frames: list[StackFrame] = []
    idx = 0
    for line in test_log.splitlines():
        m = STACK_FRAME_RE.match(line)
        if not m:
            continue
        cls, file, line_no = m.groups()
        # Some frames may have "Unknown Source" etc.; ignore those
        if not file.endswith(".java"):
            continue
        try:
            ln = int(line_no)
        except ValueError:
            continue
        frames.append(
            StackFrame(
                class_name=cls,
                file_name=file,
                line_number=ln,
                raw_line=line.rstrip("\n"),
                index_in_trace=idx,
            )
        )
        idx += 1
    return frames


def extract_exception_info(test_log: str) -> ExceptionInfo:
    """
    Heuristic: pick the first exception-like line in the log (topmost).
    If multiple, the first usually corresponds to the observed failure.
    """
    for line in test_log.splitlines():
        m = EXCEPTION_LINE_RE.match(line.strip())
        if not m:
            continue
        exc_type, msg = m.groups()
        return ExceptionInfo(exception_type=exc_type, message=(msg or "").strip())
    return ExceptionInfo(exception_type="UnknownException", message="")


def is_test_frame(frame: StackFrame) -> bool:
    n = frame.class_name.lower()
    return (
        ".test." in n
        or n.endswith("test")
        or n.endswith("tests")
        or ".junit." in n
        or n.startswith("org.junit.")
    )


def is_standard_lib(frame: StackFrame) -> bool:
    return frame.class_name.startswith(("java.", "javax.", "sun.", "jdk.", "org.junit.", "junit."))


def is_reflection_or_proxy(frame: StackFrame) -> bool:
    n = frame.class_name
    return (
        n.startswith("java.lang.reflect.")
        or "CGLIB" in n
        or "ByteBuddy" in n
        or n.startswith("org.springframework.")
    )


def is_utility_class(frame: StackFrame) -> bool:
    name = frame.class_name.lower()
    return any(k in name for k in ["util", "utils", "utility", "helper", "common"])


@dataclass(frozen=True)
class FrameScore:
    frame: StackFrame
    score: float
    reasons: list[str]


def score_frame(
    frame: StackFrame,
    exception_type: str,
    source_file_exists: bool,
) -> FrameScore:
    """
    Deterministic scoring; higher is better.

    You can tune weights. The point is:
      - prefer project frames that map to a real source file
      - downweight test/lib/reflective noise
      - prefer earlier frames in the stack
    """
    score = 0.0
    reasons: list[str] = []

    # Early frames are usually closer to the throw site
    early_bonus = max(0.0, 6.0 - float(frame.index_in_trace))
    score += early_bonus
    if early_bonus > 0:
        reasons.append(f"early_frame(+{early_bonus:.1f})")

    if source_file_exists:
        score += 6.0
        reasons.append("source_exists(+6)")
    else:
        score -= 2.0
        reasons.append("source_missing(-2)")

    if is_standard_lib(frame):
        score -= 8.0
        reasons.append("standard_lib(-8)")

    if is_test_frame(frame):
        score -= 6.0
        reasons.append("test_frame(-6)")

    if is_reflection_or_proxy(frame):
        score -= 2.0
        reasons.append("reflection_or_proxy(-2)")

    # Utility downweighting for argument-related exceptions (common overfitting fix site)
    if is_utility_class(frame) and exception_type in ARGUMENT_EXCEPTIONS:
        score -= 3.0
        reasons.append("utility_and_argument_exception(-3)")

    # Prefer non-inner synthetic classes slightly less
    if "$" in frame.class_name:
        score -= 0.5
        reasons.append("inner_class(-0.5)")

    return FrameScore(frame=frame, score=score, reasons=reasons)


def filter_and_rank_frames(
    evidence: StackTraceEvidence,
    source_exists_fn,
    max_candidates: int = 10,
) -> list[FrameScore]:
    """
    Produce ranked candidate frames with scores.

    source_exists_fn: Callable[[StackFrame], bool]
      Provided by source_utils via source indexing / resolution.
    """
    exc_type = evidence.exception.exception_type
    scored: list[FrameScore] = []
    for f in evidence.frames:
        # Cheap elimination first:
        if is_standard_lib(f):
            continue
        # keep tests for reference? default drop:
        if is_test_frame(f):
            continue

        exists = bool(source_exists_fn(f))
        scored.append(score_frame(f, exc_type, exists))

    # If everything got filtered, fall back to raw frames with lower confidence
    if not scored:
        for f in evidence.frames[: max_candidates * 2]:
            exists = bool(source_exists_fn(f))
            scored.append(score_frame(f, exc_type, exists))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:max_candidates]


def choose_best_frame(
    ranked: list[FrameScore],
) -> Optional[FrameScore]:
    if not ranked:
        return None
    return ranked[0]


def build_evidence_from_log(test_log: str) -> StackTraceEvidence:
    exc = extract_exception_info(test_log)
    frames = parse_stack_frames(test_log)
    return StackTraceEvidence(exception=exc, frames=frames, raw_stack_text=test_log)
