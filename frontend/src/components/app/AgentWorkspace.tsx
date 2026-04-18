import { CodeEditor } from "../CodeEditor";
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
  RunResult,
  SessionStatus,
  StageName,
  StageState,
  UiLocale,
} from "../../types";

function formatProgressLabel(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    const value = values[key];
    return value === undefined || value === null ? "" : String(value);
  });
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

              <label className="block">
                <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                  {copy.programInput}
                </div>
                <textarea
                  value={inputText}
                  onChange={(event) => onInputTextChange(event.target.value)}
                  placeholder={copy.programInputHint}
                  rows={agentSourceType === "single_file" ? 4 : 3}
                  className="w-full rounded-[18px] border border-black/10 bg-white/50 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                />
              </label>
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
