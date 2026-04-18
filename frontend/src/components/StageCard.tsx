import { useMemo } from "react";
import {
  stageActiveHints,
  stageExplainHints,
  stageLabels,
  stageOrder,
  stageSubtitles,
} from "../i18n";
import type { StageName, StageState, UiLocale } from "../types";
import { CandidateRankingPanel, isCandidateReportContent } from "./CandidateRankingPanel";
import { CollapsibleSection } from "./CollapsibleSection";
import { DiffView, computeDiffStats } from "./DiffView";
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
    explainLabel?: string;
    finalDiff?: string;
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

function parseReportForMeta(stage: StageName, content: string): string {
  if (!content) return "";
  if (!isCandidateReportContent(content)) {
    return `${content.length.toLocaleString()} chars`;
  }
  try {
    const parsed = JSON.parse(content) as {
      selected_candidate?: {
        added_lines?: number;
        removed_lines?: number;
        changed_file_count?: number;
        verify_passed?: boolean;
      };
    };
    const sel = parsed.selected_candidate;
    if (!sel) return "";
    if (stage === "code") {
      const added = sel.added_lines ?? 0;
      const removed = sel.removed_lines ?? 0;
      const files = sel.changed_file_count ?? 0;
      return `+${added} / -${removed} · ${files} file${files === 1 ? "" : "s"}`;
    }
    if (stage === "verify") {
      if (sel.verify_passed === true) return "✓ passed";
      if (sel.verify_passed === false) return "× failed";
    }
  } catch {
    /* fall through */
  }
  return "";
}

export function StageCard({ locale, stage, state, copy }: StageCardProps) {
  const label = stageLabels[locale][stage];
  const subtitle = stageSubtitles[locale][stage];
  const activeHint = stageActiveHints[locale][stage];
  const explainHint = stageExplainHints[locale][stage];
  const reportContent = state.report || state.diff;
  const hasStructuredPatchReport = reportContent ? isCandidateReportContent(reportContent) : false;
  const showRawDiff = stage === "code" && Boolean(state.diff);
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

  const liveHint = isExplaining ? explainHint : isActive ? activeHint : "";

  const retryBadge =
    state.retryAttempt && state.retryAttempt >= 1
      ? formatTemplate(copy.stageRetryingLabel, {
          attempt: state.retryAttempt,
          max: state.retryMax ?? state.retryAttempt,
        })
      : "";

  const stepLabel = `${getStageIndex(stage) + 1}/${stageOrder.length}`;

  const reportMeta = useMemo(() => {
    if (showRawDiff) {
      const { added, removed, files } = computeDiffStats(state.diff);
      if (added + removed + files === 0) return "";
      return `+${added} / -${removed} · ${files} file${files === 1 ? "" : "s"}`;
    }
    return parseReportForMeta(stage, reportContent);
  }, [showRawDiff, state.diff, stage, reportContent]);

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

  const explainOpen = true;
  const reasoningOpen = false;
  const reportOpen = true;
  const toolCallsOpen = false;

  const explainCharCount = state.explain.length;
  const reasoningCharCount = state.reasoning.length;
  const toolCallsMeta = state.toolEvents.length
    ? `${state.toolEvents.length} call${state.toolEvents.length === 1 ? "" : "s"}`
    : "";

  return (
    <section className="min-w-0 rounded-[24px] border border-black/5 bg-white/50 p-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
      <header className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="rounded-full border border-black/10 px-2 py-0.5 font-mono text-[10px] text-slate-500 dark:border-white/15 dark:text-white/50">
            {stepLabel}
          </span>
          <span className="truncate text-sm font-medium text-slate-900 dark:text-white">{label}</span>
          {(isActive || isExplaining) ? <ThinkingDots /> : null}
          {retryBadge ? (
            <span className="rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-300/30 dark:text-amber-200">
              {retryBadge}
            </span>
          ) : null}
        </div>
        <span
          className={`shrink-0 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.22em] ${statusBadgeClass}`}
        >
          {statusText}
        </span>
      </header>

      {liveHint ? (
        <div className="mt-2 text-[11px] italic text-slate-500 dark:text-white/55">{liveHint}</div>
      ) : null}

      <div className="mt-3 space-y-2">
        {state.explain ? (
          <CollapsibleSection
            title={copy.explainLabel ?? "Summary"}
            meta={explainCharCount > 0 ? `${explainCharCount.toLocaleString()} chars` : ""}
            defaultOpen={explainOpen}
            tone="accent"
          >
            <div className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-700 [overflow-wrap:anywhere] dark:text-white/80">
              {state.explain}
            </div>
          </CollapsibleSection>
        ) : null}

        {state.reasoning ? (
          <CollapsibleSection
            title={copy.reasoningLabel}
            meta={reasoningCharCount > 0 ? `${reasoningCharCount.toLocaleString()} chars` : ""}
            defaultOpen={reasoningOpen}
            tone="warning"
          >
            <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/75">
              {state.reasoning}
            </pre>
          </CollapsibleSection>
        ) : null}

        {showRawDiff ? (
          <CollapsibleSection
            title={copy.finalDiff ?? copy.reportLabel}
            meta={reportMeta}
            defaultOpen={reportOpen}
            bodyPadding="none"
          >
            <DiffView content={state.diff} bare />
          </CollapsibleSection>
        ) : reportContent ? (
          <CollapsibleSection
            title={copy.reportLabel}
            meta={reportMeta}
            defaultOpen={reportOpen}
            bodyPadding={hasStructuredPatchReport ? "tight" : "default"}
          >
            {hasStructuredPatchReport ? (
              <CandidateRankingPanel locale={locale} reportContent={reportContent} stage={stage} compact />
            ) : (
              <pre className="overflow-y-auto whitespace-pre-wrap break-words font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/65">
                {reportContent}
              </pre>
            )}
          </CollapsibleSection>
        ) : null}

        {state.toolEvents.length > 0 ? (
          <CollapsibleSection
            title={copy.toolCallsLabel}
            meta={toolCallsMeta}
            defaultOpen={toolCallsOpen}
            bodyPadding="tight"
          >
            <div className="max-h-72 space-y-2 overflow-y-auto">
              {state.toolEvents.map((item) => (
                <CollapsibleSection
                  key={item.id}
                  title={
                    <span className="font-mono text-[11px] normal-case tracking-normal text-slate-800 dark:text-white/80">
                      {item.tool_name}
                    </span>
                  }
                  meta={
                    <div className="flex items-center gap-2">
                      {item.round ? (
                        <span className="text-[10px] text-slate-500 dark:text-white/40">
                          round {item.round}
                        </span>
                      ) : null}
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] ${
                          item.status === "completed"
                            ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-700 dark:border-emerald-300/25 dark:text-emerald-200"
                            : "border-amber-400/25 bg-amber-400/10 text-amber-700 dark:border-amber-300/25 dark:text-amber-200"
                        }`}
                      >
                        {item.status === "completed" ? copy.toolCompleted : copy.toolStarted}
                      </span>
                    </div>
                  }
                  defaultOpen={false}
                  bodyPadding="tight"
                >
                  {item.arguments ? (
                    <div>
                      <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
                        {copy.toolArguments}
                      </div>
                      <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/72">
                        {item.arguments}
                      </pre>
                    </div>
                  ) : null}
                  {item.output_preview ? (
                    <div className={item.arguments ? "mt-2" : ""}>
                      <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
                        {copy.toolOutput}
                      </div>
                      <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/72">
                        {item.output_preview}
                        {item.output_truncated ? "\n..." : ""}
                      </pre>
                    </div>
                  ) : null}
                  {!item.arguments && !item.output_preview ? (
                    <div className="text-[11px] text-slate-500 dark:text-white/45">
                      {item.at}
                    </div>
                  ) : null}
                </CollapsibleSection>
              ))}
            </div>
          </CollapsibleSection>
        ) : null}
      </div>
    </section>
  );
}
