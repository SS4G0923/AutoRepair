import {
  stageActiveHints,
  stageExplainHints,
  stageLabels,
  stageOrder,
  stageSubtitles,
} from "../i18n";
import type { StageName, StageState, UiLocale } from "../types";
import { CandidateRankingPanel, isCandidateReportContent } from "./CandidateRankingPanel";
import { ReasoningPanel } from "./ReasoningPanel";
import { ThinkingDots } from "./ThinkingDots";

interface StageCardProps {
  locale: UiLocale;
  stage: StageName;
  state: StageState;
  copy: {
    thinking: string;
    reasoningLabel: string;
    reasoningShow: string;
    reasoningHide: string;
    noContent: string;
    reportLabel: string;
    toolCallsLabel: string;
    toolCallsEmpty: string;
    toolArguments: string;
    toolOutput: string;
    toolStarted: string;
    toolCompleted: string;
    stageWaiting: string;
    stageActive: string;
    stageExplaining: string;
    stageDone: string;
    stageRetryingLabel: string;
  };
}

function formatTemplate(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    const value = values[key];
    return value === undefined || value === null ? "" : String(value);
  });
}

function getStageIndex(stage: StageName): number {
  return Math.max(0, stageOrder.indexOf(stage));
}

export function StageCard({ locale, stage, state, copy }: StageCardProps) {
  const label = stageLabels[locale][stage];
  const subtitle = stageSubtitles[locale][stage];
  const activeHint = stageActiveHints[locale][stage];
  const explainHint = stageExplainHints[locale][stage];
  const reportContent = state.report || state.diff;
  const hasStructuredPatchReport = reportContent ? isCandidateReportContent(reportContent) : false;
  const isIdle = state.status === "idle";
  const isActive = state.status === "started";
  const isExplaining = state.status === "explaining";
  const isCompleted = state.status === "completed";
  const hasContent = Boolean(state.reasoning || state.explain || reportContent || state.toolEvents.length > 0);

  const statusText = isCompleted
    ? copy.stageDone
    : isExplaining
      ? copy.stageExplaining
      : isActive
        ? copy.stageActive
        : copy.stageWaiting;

  const statusBadgeClass = isCompleted
    ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-700 dark:border-emerald-300/25 dark:text-emerald-200"
    : isExplaining
      ? "border-sky-400/30 bg-sky-400/10 text-sky-700 dark:border-sky-300/25 dark:text-sky-200"
      : isActive
        ? "border-amber-400/40 bg-amber-400/10 text-amber-700 dark:border-amber-300/30 dark:text-amber-200"
        : "border-black/10 bg-transparent text-slate-500 dark:border-white/10 dark:text-white/45";

  const activeHintText = isExplaining ? explainHint : isActive ? activeHint : "";

  const retryBadge =
    state.retryAttempt && state.retryAttempt >= 1
      ? formatTemplate(copy.stageRetryingLabel, {
          attempt: state.retryAttempt,
          max: state.retryMax ?? state.retryAttempt,
        })
      : "";

  const stepLabel = `${getStageIndex(stage) + 1}/${stageOrder.length}`;

  if (isIdle && !hasContent) {
    return (
      <section className="flex min-w-0 items-center justify-between gap-3 rounded-2xl border border-black/5 bg-white/60 px-4 py-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.03]">
        <div className="flex min-w-0 items-center gap-3">
          <span className="rounded-full border border-black/10 px-2 py-0.5 font-mono text-[10px] text-slate-500 dark:border-white/15 dark:text-white/50">
            {stepLabel}
          </span>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-slate-600 dark:text-white/65">{label}</div>
            <div className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-white/40">{subtitle}</div>
          </div>
        </div>
        <span
          className={`shrink-0 rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-[0.22em] ${statusBadgeClass}`}
        >
          {statusText}
        </span>
      </section>
    );
  }

  return (
    <section className="min-w-0 rounded-[24px] border border-black/5 bg-white/50 p-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-black/10 px-2 py-0.5 font-mono text-[10px] text-slate-500 dark:border-white/15 dark:text-white/50">
              {stepLabel}
            </span>
            <span className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/55">
              {label}
            </span>
            {(isActive || isExplaining) ? <ThinkingDots /> : null}
            {retryBadge ? (
              <span className="rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-300/30 dark:text-amber-200">
                {retryBadge}
              </span>
            ) : null}
          </div>
          <div className="mt-1 text-[11px] text-slate-500 dark:text-white/45">{subtitle}</div>
          {activeHintText ? (
            <div className="mt-2 text-[11px] italic text-slate-500 dark:text-white/55">{activeHintText}</div>
          ) : null}
        </div>
        <span
          className={`shrink-0 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.22em] ${statusBadgeClass}`}
        >
          {statusText}
        </span>
      </div>

      {state.explain ? (
        <div className="mt-3 rounded-2xl border border-black/5 bg-slate-950 px-4 py-3 font-mono text-sm leading-7 text-slate-100 dark:border-white/10 dark:bg-ink-900">
          <pre className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{state.explain}</pre>
        </div>
      ) : null}

      {state.reasoning ? (
        <div className="mt-3">
          <ReasoningPanel
            label={copy.reasoningLabel}
            showLabel={copy.reasoningShow}
            hideLabel={copy.reasoningHide}
            content={state.reasoning}
          />
        </div>
      ) : null}

      {reportContent ? (
        <div className="mt-3">
          <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
            {copy.reportLabel}
          </div>
          <CandidateRankingPanel locale={locale} reportContent={reportContent} stage={stage} />
          {!hasStructuredPatchReport ? (
            <pre className="overflow-y-auto whitespace-pre-wrap break-words rounded-2xl border border-black/5 bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/65">
              {reportContent}
            </pre>
          ) : null}
        </div>
      ) : null}

      {state.toolEvents.length > 0 ? (
        <div className="mt-3">
          <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
            {copy.toolCallsLabel}
          </div>
          <div className="max-h-60 space-y-3 overflow-y-auto rounded-2xl border border-black/5 bg-black/[0.03] p-3 dark:border-white/10 dark:bg-white/[0.03]">
            {state.toolEvents.map((item) => (
              <div
                key={item.id}
                className="rounded-[18px] border border-black/5 bg-white/50 p-3 dark:border-white/10 dark:bg-slate-950/70"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate font-mono text-xs text-slate-900 dark:text-white">
                      {item.tool_name}
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500 dark:text-white/40">
                      {item.round ? `round ${item.round}` : null}
                      {item.round && item.at ? " · " : null}
                      {item.at}
                    </div>
                  </div>
                  <div className="rounded-full border border-black/10 px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-600 dark:border-white/10 dark:text-white/60">
                    {item.status === "completed" ? copy.toolCompleted : copy.toolStarted}
                  </div>
                </div>

                {item.arguments ? (
                  <div className="mt-3">
                    <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
                      {copy.toolArguments}
                    </div>
                    <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/72">
                      {item.arguments}
                    </pre>
                  </div>
                ) : null}

                {item.output_preview ? (
                  <div className="mt-3">
                    <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
                      {copy.toolOutput}
                    </div>
                    <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/72">
                      {item.output_preview}
                      {item.output_truncated ? "\n..." : ""}
                    </pre>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
