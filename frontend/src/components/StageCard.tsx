import { stageLabels } from "../i18n";
import type { StageName, StageState, UiLocale } from "../types";
import { ThinkingDots } from "./ThinkingDots";

interface StageCardProps {
  locale: UiLocale;
  stage: StageName;
  state: StageState;
  copy: {
    thinking: string;
    noContent: string;
    reportLabel: string;
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
          <pre className="overflow-y-auto whitespace-pre-wrap break-words rounded-3xl border border-black/5 bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/65">
            {reportContent}
          </pre>
        </div>
      ) : null}
    </section>
  );
}
