from __future__ import annotations

import json
from pathlib import Path

from llm.call_gemini import call_llm_for_json as call_gemini_for_json
from llm.call_gpt import call_llm_for_json as call_gpt_for_json
from inspector.inspector_prompt import build_planner_prompt
from inspector.inspector import run_defects4j_inspection


def call_llm_for_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str,
    isJson: bool = True,
    stream: bool = False,
):
    llm_caller = call_gemini_for_json if "gemini" in model.lower() else call_gpt_for_json
    kwargs = {"prompt": prompt, "model": model, "isJson": isJson, "stream": stream}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return llm_caller(**kwargs)

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
        model="qwen3.5-plus",
    )
    print(inspector_report)

    print("[Inspector Explain]")
    inspector_explain = call_llm_for_json(
        system_prompt="""你是一个优秀的代码Bug分析师，你的任务是分析代码中的Bug并提供详细的分析报告。
        请根据以下信息生成一个以第一人称描述的分析报告，包括可能的Bug产生的原因，位置，以及修复建议。
        请确保分析报告清晰、详细，并且包含足够的信息以帮助开发人员理解和修复Bug。
        直接输出报告，不要有开头的引导语和收尾语。尽量详细简练，控制在100字以内。""",
        prompt=json.dumps(inspector_report),
        model="qwen3.5-plus",
        isJson=False,
        stream=True,
    )

    planner_prompt = build_planner_prompt(inspector_report)

    print("[Planner Report]")
    planner_report = call_llm_for_json(
        prompt=planner_prompt,
        system_prompt="""You are a helpful bug fix planning expert, you will be presented with bug information from a bug inspection expert. 
        What you will do is to generate a bug fix plan for bug fixing expert as instructed. 
        Please remember that the plan you provided will be used to implement the bug fix, 
        so be sure to include all the information that will be helpful to locate the bug and implement the bug fix.""",
        model="qwen3.5-plus",
        isJson=False,
        stream=False,
    )
    print(planner_report)

    print("[Planner Explain]")
    planner_explain = call_llm_for_json(
        system_prompt="""你是一个优秀的代码Bug修复计划师，你的任务是分析代码中的Bug修复计划并提供详细的分析报告。
        请根据以下信息生成一个以第一人称描述的分析报告，包括对修复计划的理解，可能的修复方案，以及实施修复计划的建议。
        直接输出报告，不要有开头的引导语和收尾语。尽量详细简练，控制在100字以内。""",
        prompt=json.dumps(inspector_report) + "\n\n" + planner_report,
        model="qwen3.5-plus",
        isJson=False,
        stream=True,
    )


    print("[Coder Report]")
    coder_report = call_llm_for_json(
        prompt=json.dumps(planner_report),
        system_prompt="""You are a coding expert, now there is a bug you need to fix, you will be presented with a bug fix plan, 
        please fix the bug as instructed. Make sure to output the original code and fixed code as git diff format.""",
        model="qwen3.5-plus",
        isJson=False,
        stream=True,
    )

if __name__ == "__main__":
    main()
