# inspector/inspector_prompt.py
from __future__ import annotations

import json
from typing import Any


PLANNER_SYSTEM_INSTRUCTIONS = """You are the Planner Agent for Automated Program Repair.

Rules:
- Do NOT write code.
- Produce a minimal, actionable repair plan grounded in the Inspector evidence.
- If evidence is insufficient, request exactly what additional evidence is needed.
- Focus on correctness, minimal changes, and avoiding test overfitting.
- Provide side-effect analysis: what could break and why.
"""


def build_planner_prompt(inspector_report: dict[str, Any]) -> str:
    """
    Build a single prompt string for Planner Agent.
    The Planner consumes: evidence JSON + curated snippets.
    """
    pretty = json.dumps(inspector_report, indent=2, ensure_ascii=False)

    return f"""{PLANNER_SYSTEM_INSTRUCTIONS}

=== INSPECTOR REPORT (JSON) ===
{pretty}

=== TASK ===
1) Identify the most likely root cause.
2) Propose a minimal repair plan (NO CODE).
3) State risks / side effects and how to validate.
4) If multiple hypotheses exist, list them and choose the best supported by evidence.
"""
