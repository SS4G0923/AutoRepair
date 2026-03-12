from __future__ import annotations

import json
from pathlib import Path
from inspector.call_llm import call_llm_for_json
from inspector.inspector_prompt import build_planner_prompt
from inspector.inspector import run_defects4j_inspection

def main() -> None:

    project_id = "Lang"
    bug_id = 1
    version = "b"
    is_buggy = version == "b"
    work_dir = "./tmp/d4j/" + project_id + "_" + str(bug_id) + version
    work_dir = Path(work_dir).resolve()
    artifacts_dir = (work_dir / ".inspector").resolve()
    force_checkout = True
    test_mode = "relevant"
    inspect_failing_tests = 1
    max_candidates = 3

    report = run_defects4j_inspection(
        project_id=project_id,
        bug_id=bug_id,
        is_buggy=is_buggy,
        work_dir=work_dir,
        artifacts_dir=artifacts_dir,
        force_checkout=force_checkout,
        test_mode=test_mode,
        inspect_failing_tests=inspect_failing_tests,
        max_candidates=max_candidates,
    )

    with open(project_id + str(bug_id) + version + "_inspector_output.json", 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[Inspector] Artifacts dir: {artifacts_dir}")

    inspector_report = call_llm_for_json(
        json.dumps(report, indent=2),
        model="gpt-5.2",
    )
    print(inspector_report)

    planner_prompt = build_planner_prompt(inspector_report)

    planner_report = call_llm_for_json(
        prompt=planner_prompt,
        system_prompt="""You are a helpful bug fix planning expert, you will be presented with bug information from a bug inspection expert. 
        What you will do is to generate a bug fix plan for bug fixing expert as instructed. 
        Please remember that the plan you provided will be used to implement the bug fix, 
        so be sure to include all the information that will be helpful to locate the bug and implement the bug fix.""",
        model="gpt-5.2",
    )
    print(planner_report)





if __name__ == "__main__":
    main()
