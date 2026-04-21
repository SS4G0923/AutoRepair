import { CodeEditor } from "../CodeEditor";
import { CollapsibleSection } from "../CollapsibleSection";
import { DiffView, computeDiffStats } from "../DiffView";
import { StageCard } from "../StageCard";
import { Dropdown } from "../Dropdown";
import { AppCopy, languageOptions, stageOrder } from "../../i18n";
import type {
  AgentSourceType,
  CodeLanguage,
  ModelCatalogItem,
  ModelOptionValue,
  ProjectEntrypointOption,
  RepairTestCase,
  RunResult,
  SessionStatus,
  StageName,
  StageState,
  TestCaseResult,
  TestCasesSummary,
  UiLocale,
} from "../../types";

function formatProgressLabel(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    const value = values[key];
    return value === undefined || value === null ? "" : String(value);
  });
}

function TestCaseResultsPanel({
  copy,
  title,
  summary,
  results,
}: {
  copy: AppCopy;
  title: string;
  summary?: TestCasesSummary | null;
  results?: TestCaseResult[] | null;
}) {
  if (!results || results.length === 0) return null;
  const total = summary?.total ?? results.length;
  const passed = summary?.passed ?? results.filter((r) => r.passed).length;
  const failed = total - passed;
  const allPassed = summary?.all_passed ?? failed === 0;
  return (
    <section className="mt-4 rounded-2xl border border-black/5 bg-white/60 p-3 dark:border-white/10 dark:bg-white/[0.03]">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/55">
          {title}
        </div>
        <div
          className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${
            allPassed
              ? "bg-emerald-500/10 text-emerald-700 ring-emerald-500/30 dark:text-emerald-300"
              : "bg-rose-500/10 text-rose-700 ring-rose-500/30 dark:text-rose-300"
          }`}
        >
          {formatProgressLabel(
            allPassed ? copy.testCasesSummaryPassed : copy.testCasesSummaryFailed,
            { passed, failed, total },
          )}
        </div>
      </div>
      <ul className="mt-2 space-y-2">
        {results.map((result) => {
          const statusLabel = result.passed
            ? copy.testCasePassed
            : result.timed_out
              ? copy.testCaseTimedOut
              : !result.runtime_ok
                ? copy.testCaseRuntimeError
                : copy.testCaseFailed;
          const statusClass = result.passed
            ? "bg-emerald-500/10 text-emerald-700 ring-emerald-500/30 dark:text-emerald-300"
            : "bg-rose-500/10 text-rose-700 ring-rose-500/30 dark:text-rose-300";
          return (
            <li
              key={result.index}
              className={`rounded-xl border p-2.5 ${
                result.passed
                  ? "border-emerald-400/30 bg-emerald-50/60 dark:border-emerald-300/20 dark:bg-emerald-500/5"
                  : "border-rose-400/30 bg-rose-50/60 dark:border-rose-300/20 dark:bg-rose-500/5"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-black/[0.06] px-2 py-0.5 text-[11px] font-semibold text-slate-700 dark:bg-white/10 dark:text-white/80">
                  #{result.index + 1} · {result.name}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${statusClass}`}
                >
                  {result.passed ? "✓" : "✗"} {statusLabel}
                </span>
                <span className="text-[11px] text-slate-500 dark:text-white/45">
                  rc={result.returncode} · {result.duration_sec.toFixed(2)}s
                </span>
              </div>
              {result.stdin.trim() ? (
                <div className="mt-1.5">
                  <div className="text-[11px] font-semibold text-slate-500 dark:text-white/50">
                    {copy.testCaseStdin}
                  </div>
                  <pre className="mt-0.5 max-h-16 overflow-auto rounded-lg bg-black/[0.05] p-2 font-mono text-[11px] leading-4 text-slate-700 dark:bg-black/30 dark:text-white/80">
                    {result.stdin}
                  </pre>
                </div>
              ) : null}
              {result.expected_provided ? (
                <div className="mt-1.5 grid gap-2 md:grid-cols-2">
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 dark:text-white/50">
                      {copy.testCaseExpectedOutput}
                    </div>
                    <pre className="mt-0.5 max-h-28 overflow-auto rounded-lg bg-black/[0.05] p-2 font-mono text-[11px] leading-4 text-slate-700 dark:bg-black/30 dark:text-white/80">
                      {result.expected_stdout || "(empty)"}
                    </pre>
                  </div>
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 dark:text-white/50">
                      {copy.testCaseActualOutput}
                    </div>
                    <pre className="mt-0.5 max-h-28 overflow-auto rounded-lg bg-black/[0.05] p-2 font-mono text-[11px] leading-4 text-slate-700 dark:bg-black/30 dark:text-white/80">
                      {result.stdout || "(empty)"}
                    </pre>
                  </div>
                </div>
              ) : null}
              {result.stderr.trim() ? (
                <div className="mt-1.5">
                  <div className="text-[11px] font-semibold text-slate-500 dark:text-white/50">
                    {copy.testCaseStderrTail}
                  </div>
                  <pre className="mt-0.5 max-h-20 overflow-auto rounded-lg bg-black/[0.05] p-2 font-mono text-[11px] leading-4 text-slate-700 dark:bg-black/30 dark:text-white/80">
                    {result.stderr.length > 1200 ? result.stderr.slice(-1200) : result.stderr}
                  </pre>
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function extractVerifyTestCaseResults(
  report: string | null | undefined,
): { summary?: TestCasesSummary; results?: TestCaseResult[] } {
  if (!report) return {};
  try {
    const parsed = JSON.parse(report) as Record<string, unknown>;
    const selected = parsed.selected_candidate as Record<string, unknown> | undefined;
    const verificationReport = selected?.verification_report as Record<string, unknown> | undefined;
    const results = verificationReport?.test_case_results;
    const summary = verificationReport?.test_cases_summary;
    if (Array.isArray(results)) {
      return {
        summary: summary as TestCasesSummary | undefined,
        results: results as TestCaseResult[],
      };
    }
  } catch {
    return {};
  }
  return {};
}

interface StageProgressBarProps {
  copy: AppCopy;
  stages: Record<StageName, StageState>;
}

function StageProgressBar({ copy, stages }: StageProgressBarProps) {
  const total = stageOrder.length;
  const doneCount = stageOrder.filter((stage) => stages[stage].status === "completed").length;
  const activeStage = stageOrder.find(
    (stage) => stages[stage].status === "started" || stages[stage].status === "explaining",
  );
  const hasActivity = doneCount > 0 || Boolean(activeStage);
  if (!hasActivity) {
    return null;
  }
  const progressPct = Math.round((doneCount / total) * 100);

  return (
    <section className="rounded-[20px] border border-black/5 bg-white/60 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.04]">
      <div className="flex items-center justify-between gap-3">
        <div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 dark:text-white/55">
          {copy.stageProgressLabel}
        </div>
        <div className="text-[11px] text-slate-500 dark:text-white/55">
          {formatProgressLabel(copy.stageProgressFormat, { done: doneCount, total })}
        </div>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-black/[0.06] dark:bg-white/10">
        <div
          className="h-full rounded-full bg-gradient-to-r from-sky-400 via-indigo-400 to-emerald-400 transition-all duration-500"
          style={{ width: `${progressPct}%` }}
        />
      </div>
    </section>
  );
}

interface AgentWorkspaceProps {
  agentSourceType: AgentSourceType;
  code: string;
  copy: AppCopy;
  diffApplied: boolean;
  diffDecisionMessage: string;
  entrypointPath: string;
  errorMessage: string;
  finalDiff: string;
  finalMessage: string;
  githubRef: string;
  githubRepoUrl: string;
  inputText: string;
  userPrompt: string;
  testCases: RepairTestCase[];
  language: CodeLanguage;
  locale: UiLocale;
  model: ModelOptionValue;
  modelOptions: ModelCatalogItem[];
  projectActionLoading: boolean;
  projectSubdir: string;
  projectEntrypointOptions: ProjectEntrypointOption[];
  projectFilesLoading: boolean;
  languageSupported: boolean;
  runResult: RunResult | null;
  stages: Record<StageName, StageState>;
  status: SessionStatus;
  statusText: string;
  verificationPassed: boolean | null;
  workspaceMainClass: string;
  zipFileName: string;
  onApplyDiff: () => void;
  onCodeChange: (value: string) => void;
  onEntrypointChange: (value: string) => void;
  onGithubRefChange: (value: string) => void;
  onGithubRepoUrlChange: (value: string) => void;
  onInputTextChange: (value: string) => void;
  onUserPromptChange: (value: string) => void;
  onTestCasesChange: (value: RepairTestCase[]) => void;
  onLanguageChange: (value: CodeLanguage) => void;
  onModelChange: (value: ModelOptionValue) => void;
  onProjectSubdirChange: (value: string) => void;
  onReset: () => void;
  onSend: () => void;
  onSkipDiff: () => void;
  onSourceTypeChange: (value: AgentSourceType) => void;
  onStop: () => void;
  onZipSelected: (file: File | null) => void | Promise<void>;
}

export function AgentWorkspace({
  agentSourceType,
  code,
  copy,
  diffApplied,
  diffDecisionMessage,
  entrypointPath,
  errorMessage,
  finalDiff,
  finalMessage,
  githubRef,
  githubRepoUrl,
  inputText,
  userPrompt,
  testCases,
  language,
  locale,
  model,
  modelOptions,
  projectActionLoading,
  projectSubdir,
  projectEntrypointOptions,
  projectFilesLoading,
  languageSupported,
  runResult,
  stages,
  status,
  statusText,
  verificationPassed,
  workspaceMainClass,
  zipFileName,
  onApplyDiff,
  onCodeChange,
  onEntrypointChange,
  onGithubRefChange,
  onGithubRepoUrlChange,
  onInputTextChange,
  onUserPromptChange,
  onTestCasesChange,
  onLanguageChange,
  onModelChange,
  onProjectSubdirChange,
  onReset,
  onSend,
  onSkipDiff,
  onSourceTypeChange,
  onStop,
  onZipSelected,
}: AgentWorkspaceProps) {
  const hasVisibleText = (value?: string | null) => Boolean(value && value.trim().length > 0);
  const isProjectMode = agentSourceType !== "single_file";
  const projectEditorTitle = entrypointPath.trim() || copy.entrypoint;
  const projectApplyPrompt = agentSourceType === "zip" ? copy.applyProjectZipPrompt : copy.applyProjectGithubPrompt;
  const projectApplyLabel = agentSourceType === "zip" ? copy.applyZipAccept : copy.applyGithubAccept;

  const runOutputSections: Array<{ key: string; label: string; content: string }> = [];

  if (runResult) {
    if (hasVisibleText(runResult.input_text)) {
      runOutputSections.push({
        key: "stdin",
        label: copy.stdin,
        content: runResult.input_text!,
      });
    }
    if (hasVisibleText(runResult.stdout)) {
      runOutputSections.push({
        key: "stdout",
        label: copy.stdout,
        content: runResult.stdout,
      });
    }
    if (hasVisibleText(runResult.stderr)) {
      runOutputSections.push({
        key: "stderr",
        label: copy.stderr,
        content: runResult.stderr,
      });
    }
  }

  const runOutputGridClassName =
    runOutputSections.length <= 1
      ? "mt-4 grid gap-4"
      : runOutputSections.length === 2
        ? "mt-4 grid gap-4 md:grid-cols-2"
        : "mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3";

  return (
    <div className={workspaceMainClass}>
      <div className="grid h-full min-h-0 items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="flex h-full min-h-0 min-w-0 flex-col overflow-y-auto pr-1">
          <div className="flex min-h-0 flex-1 flex-col rounded-[24px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
            <div className="shrink-0 flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div className="flex shrink-0 items-center gap-1.5 rounded-full border border-black/5 bg-black/[0.03] p-1 dark:border-white/10 dark:bg-white/[0.03]">
                  {(
                    [
                      ["single_file", copy.sourceSingle],
                      ["zip", copy.sourceZip],
                      ["github", copy.sourceGithub],
                    ] as const
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      onClick={() => onSourceTypeChange(value)}
                      className={`whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition ${
                        agentSourceType === value
                          ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                          : "text-slate-600 hover:text-slate-900 dark:text-white/65 dark:hover:text-white"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                <div className="flex min-w-0 items-center gap-2">
                  {agentSourceType === "single_file" ? (
                    <Dropdown
                      value={language}
                      options={languageOptions}
                      onChange={(val) => onLanguageChange(val as CodeLanguage)}
                      className="min-w-0"
                    />
                  ) : (
                    <div className="min-w-0 truncate rounded-full border border-black/10 bg-white/50 px-3 py-1.5 text-sm text-slate-900 dark:border-white/10 dark:bg-white/5 dark:text-white">
                      {languageOptions.find((option) => option.value === language)?.label ?? language}
                    </div>
                  )}
                  <Dropdown
                    value={model}
                    options={modelOptions}
                    onChange={(val) => onModelChange(val as ModelOptionValue)}
                    disabled={modelOptions.length === 0}
                    placeholder={modelOptions.length === 0 ? copy.modelEmpty : undefined}
                    className="w-[clamp(8rem,12vw,9rem)] min-w-0"
                    menuClassName="max-w-[min(14rem,calc(100vw-2rem))]"
                    triggerLabelClassName="min-w-0 flex-1 whitespace-nowrap"
                    optionLabelClassName="whitespace-nowrap"
                  />
                </div>
              </div>

              {agentSourceType === "single_file" ? null : agentSourceType === "zip" ? (
                <div className="space-y-3">
                  <div className="grid gap-3 xl:grid-cols-2">
                    <label className="block">
                      <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.entrypoint}
                      </div>
                      <select
                        value={entrypointPath}
                        onChange={(event) => onEntrypointChange(event.target.value)}
                        className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                        disabled={projectFilesLoading || projectEntrypointOptions.length === 0}
                      >
                        {projectFilesLoading ? (
                          <option value="">{copy.projectFilesLoading}</option>
                        ) : projectEntrypointOptions.length > 0 ? (
                          projectEntrypointOptions.map((option) => (
                            <option key={option.path} value={option.path}>
                              {option.path}
                            </option>
                          ))
                        ) : (
                          <option value="">{copy.projectFilesEmpty}</option>
                        )}
                      </select>
                    </label>
                    <label className="block">
                      <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.projectSubdir}
                      </div>
                      <input
                        value={projectSubdir}
                        onChange={(event) => onProjectSubdirChange(event.target.value)}
                        placeholder="src"
                        className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                      />
                    </label>
                  </div>

                  <div className="rounded-[22px] border border-dashed border-black/10 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/[0.03]">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-900 dark:text-white">
                          {copy.zipUpload}
                        </div>
                        <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                          {zipFileName ? `${copy.zipSelected}: ${zipFileName}` : copy.sourceZipHint}
                        </div>
                      </div>
                      <label className="inline-flex cursor-pointer items-center rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85">
                        <span>{zipFileName ? copy.zipReplace : copy.zipChoose}</span>
                        <input
                          type="file"
                          accept=".zip,application/zip"
                          className="hidden"
                          onChange={(event) => {
                            const file = event.target.files?.[0] ?? null;
                            void onZipSelected(file);
                          }}
                        />
                      </label>
                    </div>
                  </div>
                </div>
              ) : null}

              {agentSourceType === "github" ? (
                <div className="grid gap-3 xl:grid-cols-2">
                  <label className="block xl:col-span-2">
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.githubRepoUrl}
                    </div>
                    <input
                      value={githubRepoUrl}
                      onChange={(event) => onGithubRepoUrlChange(event.target.value)}
                      placeholder="https://github.com/owner/repo"
                      className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                    />
                  </label>
                  <label className="block">
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.githubRef}
                    </div>
                    <input
                      value={githubRef}
                      onChange={(event) => onGithubRefChange(event.target.value)}
                      placeholder="main"
                      className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                    />
                  </label>
                  <label className="block">
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.entrypoint}
                    </div>
                    <select
                      value={entrypointPath}
                      onChange={(event) => onEntrypointChange(event.target.value)}
                      className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                      disabled={projectFilesLoading || projectEntrypointOptions.length === 0}
                    >
                      {projectFilesLoading ? (
                        <option value="">{copy.projectFilesLoading}</option>
                      ) : projectEntrypointOptions.length > 0 ? (
                        projectEntrypointOptions.map((option) => (
                          <option key={option.path} value={option.path}>
                            {option.path}
                          </option>
                        ))
                      ) : (
                        <option value="">{copy.projectFilesEmpty}</option>
                      )}
                    </select>
                  </label>
                </div>
              ) : null}

              <CollapsibleSection
                title={copy.userPrompt}
                defaultOpen={userPrompt.trim().length > 0}
                meta={
                  userPrompt.trim().length > 0
                    ? `${userPrompt.trim().length} ch`
                    : undefined
                }
              >
                <div className="text-[11px] text-slate-500 dark:text-white/50">
                  {copy.userPromptHint}
                </div>
                <textarea
                  value={userPrompt}
                  onChange={(event) => onUserPromptChange(event.target.value)}
                  placeholder={copy.userPromptPlaceholder}
                  rows={5}
                  className="mt-2 w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                />
              </CollapsibleSection>

              <CollapsibleSection
                title={copy.programInput}
                defaultOpen={inputText.trim().length > 0}
                meta={
                  inputText.trim().length > 0
                    ? `${inputText.trim().length} ch`
                    : undefined
                }
              >
                <div className="text-[11px] text-slate-500 dark:text-white/50">
                  {copy.programInputHint}
                </div>
                <textarea
                  value={inputText}
                  onChange={(event) => onInputTextChange(event.target.value)}
                  placeholder={copy.programInputHint}
                  rows={agentSourceType === "single_file" ? 4 : 3}
                  className="mt-2 w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                />
              </CollapsibleSection>

              <CollapsibleSection
                title={copy.testCases}
                defaultOpen={testCases.length > 0}
                meta={
                  testCases.length > 0 ? `${testCases.length}` : undefined
                }
              >
                <div className="text-[11px] text-slate-500 dark:text-white/50">
                  {copy.testCasesHint}
                </div>
                {testCases.length === 0 ? (
                  <div className="mt-2 rounded-2xl border border-dashed border-black/10 bg-black/[0.02] px-3 py-4 text-center text-xs text-slate-500 dark:border-white/10 dark:bg-white/[0.02] dark:text-white/45">
                    {copy.testCaseEmpty}
                  </div>
                ) : (
                  <ul className="mt-2 space-y-3">
                    {testCases.map((testCase, index) => {
                      const updateCase = (next: Partial<RepairTestCase>) => {
                        onTestCasesChange(
                          testCases.map((c, i) => (i === index ? { ...c, ...next } : c)),
                        );
                      };
                      const removeCase = () => {
                        onTestCasesChange(testCases.filter((_, i) => i !== index));
                      };
                      return (
                        <li
                          key={index}
                          className="rounded-2xl border border-black/5 bg-white/70 p-3 dark:border-white/10 dark:bg-white/[0.03]"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <input
                              value={testCase.name ?? ""}
                              onChange={(event) => updateCase({ name: event.target.value })}
                              placeholder={copy.testCaseNamePlaceholder}
                              className="min-w-0 flex-1 rounded-full border border-black/10 bg-white/60 px-3 py-1.5 text-xs text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/30"
                            />
                            <button
                              type="button"
                              onClick={removeCase}
                              className="shrink-0 rounded-full border border-black/10 bg-white/40 px-3 py-1.5 text-[11px] text-slate-600 transition hover:border-rose-400 hover:text-rose-600 dark:border-white/10 dark:bg-white/5 dark:text-white/60 dark:hover:border-rose-400 dark:hover:text-rose-300"
                            >
                              {copy.testCaseRemove}
                            </button>
                          </div>
                          <div className="mt-2 grid gap-2 md:grid-cols-2">
                            <label className="block">
                              <div className="mb-1 text-[11px] uppercase tracking-[0.22em] text-slate-500 dark:text-white/45">
                                {copy.testCaseStdin}
                              </div>
                              <textarea
                                value={testCase.stdin}
                                onChange={(event) => updateCase({ stdin: event.target.value })}
                                placeholder={copy.testCaseStdinPlaceholder}
                                rows={3}
                                className="w-full rounded-xl border border-black/10 bg-white/60 px-3 py-2 font-mono text-[12px] leading-5 text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/30"
                              />
                            </label>
                            <label className="block">
                              <div className="mb-1 text-[11px] uppercase tracking-[0.22em] text-slate-500 dark:text-white/45">
                                {copy.testCaseExpected}
                              </div>
                              <textarea
                                value={testCase.expected_stdout}
                                onChange={(event) =>
                                  updateCase({ expected_stdout: event.target.value })
                                }
                                placeholder={copy.testCaseExpectedPlaceholder}
                                rows={3}
                                className="w-full rounded-xl border border-black/10 bg-white/60 px-3 py-2 font-mono text-[12px] leading-5 text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/30"
                              />
                            </label>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
                <button
                  type="button"
                  onClick={() =>
                    onTestCasesChange([
                      ...testCases,
                      { stdin: "", expected_stdout: "", name: "" },
                    ])
                  }
                  className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-black/10 bg-white/60 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/80 dark:hover:border-white/40"
                >
                  + {copy.testCaseAdd}
                </button>
              </CollapsibleSection>
            </div>

            {!languageSupported ? (
              <div className="mt-4 shrink-0 rounded-3xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-200">
                {copy.unsupported}
              </div>
            ) : null}

            <div className="mt-3 min-h-0 flex flex-1 flex-col">
              {!isProjectMode ? (
                <CodeEditor value={code} onChange={onCodeChange} placeholder={copy.placeholder} />
              ) : (
                <CodeEditor
                  value={code}
                  onChange={onCodeChange}
                  placeholder={projectFilesLoading ? copy.projectFilesLoading : copy.projectFilesEmpty}
                  readOnly
                  title={projectEditorTitle}
                />
              )}
            </div>

            <div className="mt-3 shrink-0 flex flex-wrap items-center gap-2">
              <button
                onClick={onSend}
                disabled={status === "streaming" || !languageSupported || modelOptions.length === 0}
                className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-45 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
              >
                {copy.send}
              </button>
              <button
                onClick={onStop}
                disabled={status !== "streaming"}
                className="rounded-full border border-black/10 bg-white/50 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
              >
                {copy.stop}
              </button>
              <button
                onClick={onReset}
                className="rounded-full border border-black/10 bg-white/50 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
              >
                {copy.reset}
              </button>
                <div className="rounded-full border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-white/70">
                  {statusText}
                </div>
              </div>
          </div>
        </section>

        <section className="h-full min-h-0 min-w-0 space-y-3 overflow-y-auto pr-1">
          {(runResult || errorMessage || finalMessage || finalDiff) ? (
            <section className="rounded-[20px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
              <div className="flex items-center justify-between gap-4">
                <div className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/80">
                  {copy.runOutput}
                </div>
                <div className="text-right text-xs text-slate-500 dark:text-white/75">
                  {runResult?.entrypoint ? <div>{runResult.entrypoint}</div> : null}
                  {runResult?.execution ? (
                    <div>
                      rc={runResult.execution.returncode} · {runResult.execution.duration_sec.toFixed(2)}s
                      {typeof runResult.file_count === "number" ? ` · ${runResult.file_count} files` : ""}
                    </div>
                  ) : null}
                </div>
              </div>

              {runOutputSections.length > 0 ? (
                <div className={runOutputGridClassName}>
                  {runOutputSections.map((section) => (
                    <div key={section.key}>
                      <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/80">
                        {section.label}
                      </div>
                      <pre className="min-h-[120px] whitespace-pre-wrap break-words rounded-3xl bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:bg-slate-950/95 dark:text-white">
                        {section.content}
                      </pre>
                    </div>
                  ))}
                </div>
              ) : null}

              <TestCaseResultsPanel
                copy={copy}
                title={copy.testCasesInitialRunLabel}
                summary={runResult?.test_cases_summary}
                results={runResult?.test_case_results}
              />

              {(() => {
                const { summary: verifySummary, results: verifyResults } =
                  extractVerifyTestCaseResults(stages.verify?.report);
                return (
                  <TestCaseResultsPanel
                    copy={copy}
                    title={copy.testCasesVerifyRunLabel}
                    summary={verifySummary}
                    results={verifyResults}
                  />
                );
              })()}

              {errorMessage ? (
                <div className="mt-4 rounded-3xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                  {errorMessage}
                </div>
              ) : null}

              {finalMessage ? (
                <div className="mt-4 rounded-3xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-200">
                  {finalMessage}
                </div>
              ) : null}

              {finalDiff ? (
                <div className="mt-4">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.finalDiff}
                    </div>
                    {(() => {
                      const { added, removed, files } = computeDiffStats(finalDiff);
                      if (added + removed + files === 0) return null;
                      return (
                        <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-white/50">
                          <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-emerald-700 dark:text-emerald-200">
                            +{added}
                          </span>
                          <span className="rounded-full border border-rose-400/30 bg-rose-400/10 px-2 py-0.5 text-rose-700 dark:text-rose-200">
                            -{removed}
                          </span>
                          <span>
                            {files} file{files === 1 ? "" : "s"}
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                  <DiffView content={finalDiff} maxHeight="32rem" />

                  {verificationPassed && !isProjectMode ? (
                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <div className="text-sm text-slate-600 dark:text-white/70">{copy.applyPrompt}</div>
                      <button
                        onClick={() => {
                          void onApplyDiff();
                        }}
                        disabled={diffApplied}
                        className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      >
                        {copy.applyAccept}
                      </button>
                      <button
                        onClick={onSkipDiff}
                        className="rounded-full border border-black/10 bg-white/50 px-4 py-2 text-sm text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                      >
                        {copy.applyDecline}
                      </button>
                    </div>
                  ) : null}

                  {verificationPassed && isProjectMode ? (
                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <div className="text-sm text-slate-600 dark:text-white/70">{projectApplyPrompt}</div>
                      <button
                        onClick={() => {
                          void onApplyDiff();
                        }}
                        disabled={diffApplied || projectActionLoading}
                        className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      >
                        {projectActionLoading ? copy.applyProjectActionRunning : projectApplyLabel}
                      </button>
                      <button
                        onClick={onSkipDiff}
                        disabled={projectActionLoading}
                        className="rounded-full border border-black/10 bg-white/50 px-4 py-2 text-sm text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                      >
                        {copy.applyDecline}
                      </button>
                    </div>
                  ) : null}

                  {diffDecisionMessage ? (
                    <div className="mt-4 rounded-3xl border border-black/10 bg-black/[0.03] px-4 py-3 text-sm text-slate-700 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/75">
                      {diffDecisionMessage}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : null}

          <StageProgressBar copy={copy} stages={stages} />

          {stageOrder.map((stage) => (
            <StageCard key={stage} locale={locale} stage={stage} state={stages[stage]} copy={copy} />
          ))}
        </section>
      </div>
    </div>
  );
}
