import { CodeEditor } from "../CodeEditor";
import { StageCard } from "../StageCard";
import { AppCopy, languageOptions, modelOptions, stageOrder } from "../../i18n";
import type {
  AgentSourceType,
  CodeLanguage,
  ModelOptionValue,
  RunResult,
  SessionStatus,
  StageName,
  StageState,
  UiLocale,
} from "../../types";

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
  language: CodeLanguage;
  locale: UiLocale;
  model: ModelOptionValue;
  projectSubdir: string;
  pythonSupported: boolean;
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
  language,
  locale,
  model,
  projectSubdir,
  pythonSupported,
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
  return (
    <div className={workspaceMainClass}>
      <div className="grid h-full min-h-0 items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden pr-1">
          <div className="flex min-h-0 flex-1 flex-col rounded-[24px] border border-black/5 bg-white/72 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
            <div className="shrink-0 flex flex-col gap-4">
              <div className={`flex gap-3 ${agentSourceType === "single_file" ? "items-start" : "items-center"}`}>
                <div className="flex items-center gap-1.5 rounded-full border border-black/5 bg-black/[0.03] p-1 dark:border-white/10 dark:bg-white/[0.03]">
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
                          ? "bg-slate-900 text-white shadow-sm dark:bg-white dark:text-slate-950"
                          : "text-slate-600 hover:text-slate-900 dark:text-white/65 dark:hover:text-white"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {agentSourceType === "single_file" ? (
                  <div className="flex min-w-[180px] flex-col gap-2">
                    <select
                      value={language}
                      onChange={(event) => onLanguageChange(event.target.value as CodeLanguage)}
                      className="rounded-full border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                    >
                      {languageOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <select
                      value={model}
                      onChange={(event) => onModelChange(event.target.value as ModelOptionValue)}
                      className="rounded-full border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                    >
                      {modelOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {copy.model}: {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <select
                    value={model}
                    onChange={(event) => onModelChange(event.target.value as ModelOptionValue)}
                    className="rounded-full border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                  >
                    {modelOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {copy.model}: {option.label}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {agentSourceType === "single_file" ? null : agentSourceType === "zip" ? (
                <div className="space-y-3">
                  <div className="grid gap-3 xl:grid-cols-2">
                    <label className="block">
                      <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.entrypoint}
                      </div>
                      <input
                        value={entrypointPath}
                        onChange={(event) => onEntrypointChange(event.target.value)}
                        placeholder="app/main.py"
                        className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                      />
                    </label>
                    <label className="block">
                      <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.projectSubdir}
                      </div>
                      <input
                        value={projectSubdir}
                        onChange={(event) => onProjectSubdirChange(event.target.value)}
                        placeholder="src"
                        className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
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
                      className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
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
                      className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                    />
                  </label>
                  <label className="block">
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.entrypoint}
                    </div>
                    <input
                      value={entrypointPath}
                      onChange={(event) => onEntrypointChange(event.target.value)}
                      placeholder="app/main.py"
                      className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                    />
                  </label>
                  <label className="block">
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                      {copy.projectSubdir}
                    </div>
                    <input
                      value={projectSubdir}
                      onChange={(event) => onProjectSubdirChange(event.target.value)}
                      placeholder="packages/api"
                      className="w-full rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
                    />
                  </label>
                </div>
              ) : null}
            </div>

            {!pythonSupported ? (
              <div className="mt-4 shrink-0 rounded-3xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-200">
                {copy.unsupported}
              </div>
            ) : null}

            <div className="mt-3 min-h-0 flex flex-1 flex-col">
              {agentSourceType === "single_file" ? (
                <CodeEditor value={code} onChange={onCodeChange} placeholder={copy.placeholder} />
              ) : (
                <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[28px] border border-black/5 bg-black/[0.03] p-5 dark:border-white/10 dark:bg-white/[0.03]">
                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-[22px] border border-black/5 bg-white/70 p-4 dark:border-white/10 dark:bg-slate-950/70">
                      <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.entrypoint}
                      </div>
                      <div className="mt-2 font-mono text-sm text-slate-800 dark:text-white">
                        {entrypointPath.trim() || "main.py"}
                      </div>
                    </div>
                    <div className="rounded-[22px] border border-black/5 bg-white/70 p-4 dark:border-white/10 dark:bg-slate-950/70">
                      <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {copy.projectSubdir}
                      </div>
                      <div className="mt-2 font-mono text-sm text-slate-800 dark:text-white">
                        {projectSubdir.trim() || "∅"}
                      </div>
                    </div>
                    <div className="rounded-[22px] border border-black/5 bg-white/70 p-4 dark:border-white/10 dark:bg-slate-950/70 lg:col-span-2">
                      <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                        {agentSourceType === "zip" ? copy.zipUpload : copy.githubRepoUrl}
                      </div>
                      <div className="mt-2 whitespace-pre-wrap break-words font-mono text-sm text-slate-800 dark:text-white">
                        {agentSourceType === "zip"
                          ? zipFileName || copy.zipRequired
                          : githubRepoUrl.trim() || copy.githubRepoRequired}
                      </div>
                    </div>
                    {agentSourceType === "github" ? (
                      <div className="rounded-[22px] border border-black/5 bg-white/70 p-4 dark:border-white/10 dark:bg-slate-950/70 lg:col-span-2">
                        <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                          {copy.githubRef}
                        </div>
                        <div className="mt-2 font-mono text-sm text-slate-800 dark:text-white">
                          {githubRef.trim() || "default"}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}
            </div>

            <div className="mt-3 shrink-0 flex flex-wrap items-center gap-2">
              <button
                onClick={onSend}
                disabled={status === "streaming" || !pythonSupported}
                className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-45 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
              >
                {copy.send}
              </button>
              <button
                onClick={onStop}
                disabled={status !== "streaming"}
                className="rounded-full border border-black/10 bg-white/70 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
              >
                {copy.stop}
              </button>
              <button
                onClick={onReset}
                className="rounded-full border border-black/10 bg-white/70 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
              >
                {copy.reset}
              </button>
              <div className="rounded-full border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-white/70">
                {statusText}
              </div>
            </div>
          </div>
        </section>

        <section className="h-full min-h-0 min-w-0 space-y-3 overflow-y-auto pr-1">
          {(runResult || errorMessage || finalMessage || finalDiff) ? (
            <section className="rounded-[20px] border border-black/5 bg-white/75 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
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

              {runResult ? (
                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <div>
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/80">
                      {copy.stdout}
                    </div>
                    <pre className="min-h-[120px] whitespace-pre-wrap break-words rounded-3xl bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:bg-slate-950/95 dark:text-white">
                      {runResult.stdout || "∅"}
                    </pre>
                  </div>
                  <div>
                    <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/80">
                      {copy.stderr}
                    </div>
                    <pre className="min-h-[120px] whitespace-pre-wrap break-words rounded-3xl bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:bg-slate-950/95 dark:text-white">
                      {runResult.stderr || "∅"}
                    </pre>
                  </div>
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
                  <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                    {copy.finalDiff}
                  </div>
                  <pre className="overflow-y-auto whitespace-pre-wrap break-words rounded-3xl bg-slate-950 p-5 font-mono text-xs leading-6 text-slate-100 [overflow-wrap:anywhere] dark:bg-ink-900">
                    {finalDiff}
                  </pre>

                  {verificationPassed && agentSourceType === "single_file" ? (
                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <div className="text-sm text-slate-600 dark:text-white/70">{copy.applyPrompt}</div>
                      <button
                        onClick={onApplyDiff}
                        disabled={diffApplied}
                        className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      >
                        {copy.applyAccept}
                      </button>
                      <button
                        onClick={onSkipDiff}
                        className="rounded-full border border-black/10 bg-white/70 px-4 py-2 text-sm text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                      >
                        {copy.applyDecline}
                      </button>
                    </div>
                  ) : null}

                  {verificationPassed && agentSourceType !== "single_file" ? (
                    <div className="mt-4 rounded-3xl border border-black/10 bg-black/[0.03] px-4 py-3 text-sm text-slate-700 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/75">
                      {copy.applyProjectOnly}
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

          {stageOrder.map((stage) => (
            <StageCard key={stage} locale={locale} stage={stage} state={stages[stage]} copy={copy} />
          ))}
        </section>
      </div>
    </div>
  );
}
