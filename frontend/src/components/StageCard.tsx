import { stageLabels } from "../i18n";
import type { StageName, StageState, UiLocale } from "../types";
import { CandidateRankingPanel } from "./CandidateRankingPanel";
import { ThinkingDots } from "./ThinkingDots";

interface StageCardProps {
  locale: UiLocale;
  stage: StageName;
  state: StageState;
  copy: {
    thinking: string;
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
    stageDone: string;
  };
}

export function StageCard({ locale, stage, state, copy }: StageCardProps) {
  const label = stageLabels[locale][stage];
  const reportContent = state.report || state.diff;
  const statusText =
    state.status === "completed"
      ? copy.stageDone
      : state.status === "explaining"
        ? copy.stageActive
        : copy.stageWaiting;

  return (
    <section className="min-w-0 rounded-[28px] border border-black/5 bg-white/75 p-5 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">{label}</div>
          <h3 className="mt-2 font-display text-xl text-slate-900 dark:text-white">{statusText}</h3>
        </div>
        <div className="rounded-full border border-slate-900/10 px-3 py-1 text-xs uppercase tracking-[0.22em] text-slate-600 dark:border-white/10 dark:text-white/50">
          {state.status}
        </div>
      </div>

      <div className="mt-4 rounded-3xl border border-black/5 bg-slate-950 px-4 py-4 font-mono text-sm leading-7 text-slate-100 dark:border-white/10 dark:bg-ink-900">
        {state.explain ? (
          <pre className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{state.explain}</pre>
        ) : state.status === "started" || state.status === "explaining" ? (
          <div className="flex items-center gap-3">
            <span className="uppercase tracking-[0.18em] text-white/50">{copy.thinking}</span>
            <ThinkingDots />
          </div>
        ) : (
          <div className="text-white/45">{copy.noContent}</div>
        )}
      </div>

      {reportContent ? (
        <div className="mt-4">
          <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
            {copy.reportLabel}
          </div>
          <CandidateRankingPanel locale={locale} reportContent={reportContent} stage={stage} />
          {!reportContent.includes("\"collaboration_mode\": \"multi_candidate_patch_committee\"") ? (
            <pre className="overflow-y-auto whitespace-pre-wrap break-words rounded-3xl border border-black/5 bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/65">
              {reportContent}
            </pre>
          ) : null}
        </div>
      ) : null}

      <div className="mt-4">
        <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
          {copy.toolCallsLabel}
        </div>
        {state.toolEvents.length > 0 ? (
          <div className="max-h-60 space-y-3 overflow-y-auto rounded-3xl border border-black/5 bg-black/[0.03] p-4 dark:border-white/10 dark:bg-white/[0.03]">
            {state.toolEvents.map((item) => (
              <div
                key={item.id}
                className="rounded-[22px] border border-black/5 bg-white/70 p-3 dark:border-white/10 dark:bg-slate-950/70"
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
        ) : (
          <div className="rounded-3xl border border-black/5 bg-black/[0.03] px-4 py-3 text-sm text-slate-500 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/45">
            {copy.toolCallsEmpty}
          </div>
        )}
      </div>
    </section>
  );
}
