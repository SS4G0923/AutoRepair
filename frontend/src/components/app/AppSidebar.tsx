import type { AppCopy } from "../../i18n";
import type { HistorySummary, WorkspaceMode, UiLocale } from "../../types";
import {
  AgentIcon,
  BenchmarkIcon,
  ChatIcon,
  TrashIcon,
} from "./AppIcons";

function formatRelativeTime(dateString: string, locale: UiLocale): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.max(0, Math.floor((now.getTime() - date.getTime()) / 1000));

  const rtf = new Intl.RelativeTimeFormat(locale === "zh" ? "zh-CN" : "en-US", { numeric: "auto" });

  if (diffInSeconds < 60) {
    return rtf.format(-diffInSeconds || 0, "second");
  }
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return rtf.format(-diffInMinutes, "minute");
  }
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return rtf.format(-diffInHours, "hour");
  }
  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 7) {
    return rtf.format(-diffInDays, "day");
  }
  const diffInWeeks = Math.floor(diffInDays / 7);
  if (diffInWeeks < 4) {
    return rtf.format(-diffInWeeks, "week");
  }
  const diffInMonths = Math.floor(diffInDays / 30);
  if (diffInMonths < 12) {
    return rtf.format(-diffInMonths, "month");
  }
  const diffInYears = Math.floor(diffInDays / 365);
  return rtf.format(-diffInYears, "year");
}

interface AppSidebarProps {
  copy: AppCopy;
  deletingHistoryId: number | null;
  historyItems: HistorySummary[];
  historyLoading: boolean;
  isDesktopLayout: boolean;
  locale: UiLocale;
  selectedHistoryId: number | null;
  workspaceMode: WorkspaceMode;
  onDeleteHistory: (historyId: number) => void;
  onOpenHistory: (historyId: number) => void;
  onStartNewAgentSession: () => void;
  onStartNewChatSession: () => void;
  onStartNewBenchmarkSession: () => void;
}

export function AppSidebar({
  copy,
  deletingHistoryId,
  historyItems,
  historyLoading,
  isDesktopLayout,
  locale,
  selectedHistoryId,
  workspaceMode,
  onDeleteHistory,
  onOpenHistory,
  onStartNewAgentSession,
  onStartNewChatSession,
  onStartNewBenchmarkSession,
}: AppSidebarProps) {
  return (
    <aside
      className={`flex h-full min-h-0 flex-col gap-2 overflow-hidden pr-1 ${!isDesktopLayout ? "shrink-0" : ""}`}
    >
      <section className="shrink-0 rounded-[22px] border border-black/5 bg-white/50 p-2.5 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
        <div className="px-2 text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">
          {copy.sidebarWorkspace}
        </div>
        <div className="mt-2 space-y-1.5">
          <button
            onClick={onStartNewAgentSession}
            className={`flex w-full items-center gap-2 rounded-[16px] px-2.5 py-2.5 text-left transition ${
              workspaceMode === "agent"
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
            }`}
          >
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
              <AgentIcon />
            </div>
            <div className="min-w-0">
              <div className="font-medium">{copy.modeAgent}</div>
              <div
                className={`text-xs ${
                  workspaceMode === "agent"
                    ? "text-white/70 dark:text-slate-950/70"
                    : "text-slate-500 dark:text-white/40"
                }`}
              >
                {copy.modeAgentHint}
              </div>
            </div>
          </button>

          <button
            onClick={onStartNewChatSession}
            className={`flex w-full items-center gap-2 rounded-[16px] px-2.5 py-2.5 text-left transition ${
              workspaceMode === "chat"
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
            }`}
          >
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
              <ChatIcon />
            </div>
            <div className="min-w-0">
              <div className="font-medium">{copy.modeChat}</div>
              <div
                className={`text-xs ${
                  workspaceMode === "chat"
                    ? "text-white/70 dark:text-slate-950/70"
                    : "text-slate-500 dark:text-white/40"
                }`}
              >
                {copy.modeChatHint}
              </div>
            </div>
          </button>

          <button
            onClick={onStartNewBenchmarkSession}
            className={`flex w-full items-center gap-2 rounded-[16px] px-2.5 py-2.5 text-left transition ${
              workspaceMode === "benchmark"
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
            }`}
          >
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
              <BenchmarkIcon />
            </div>
            <div className="min-w-0">
              <div className="font-medium">{copy.modeBenchmark}</div>
              <div
                className={`text-xs ${
                  workspaceMode === "benchmark"
                    ? "text-white/70 dark:text-slate-950/70"
                    : "text-slate-500 dark:text-white/40"
                }`}
              >
                {copy.modeBenchmarkHint}
              </div>
            </div>
          </button>
        </div>
      </section>

      <section className="flex min-h-0 flex-1 flex-col rounded-[22px] border border-black/5 bg-white/50 p-2.5 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
        <div className="shrink-0 px-2 text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">
          {copy.sidebarHistory}
        </div>

        <div className="mt-2 min-h-0 flex-1 space-y-1.5 overflow-y-auto">
          {historyLoading && historyItems.length === 0 ? (
            <div className="rounded-[22px] bg-black/[0.03] px-4 py-4 text-sm text-slate-500 dark:bg-white/[0.03] dark:text-white/45">
              {copy.historyLoading}
            </div>
          ) : historyItems.length === 0 ? (
            <div className="rounded-[22px] bg-black/[0.03] px-4 py-4 text-sm text-slate-500 dark:bg-white/[0.03] dark:text-white/45">
              {copy.historyEmpty}
            </div>
          ) : (
            <div className="space-y-1.5 pr-1">
              {historyLoading ? (
                <div className="px-2 pb-1 text-[11px] uppercase tracking-[0.18em] text-slate-400 dark:text-white/28">
                  {copy.historyLoading}
                </div>
              ) : null}

              {historyItems.map((item) => (
                <div
                  key={item.id}
                  className={`app-slide-down flex items-start gap-2 rounded-[16px] px-2 py-2 transition ${
                    selectedHistoryId === item.id
                      ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                      : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
                  }`}
                >
                  <button
                    onClick={() => onOpenHistory(item.id)}
                    className="flex min-w-0 flex-1 items-start gap-2 text-left"
                  >
                    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
                      {item.mode === "agent" ? <AgentIcon /> : <ChatIcon />}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{item.title}</div>
                      <div
                        className={`mt-1 flex items-center gap-2 text-[11px] uppercase tracking-[0.05em] ${
                          selectedHistoryId === item.id
                            ? "text-white/60 dark:text-slate-950/60"
                            : "text-slate-400 dark:text-white/40"
                        }`}
                      >
                        {item.model ? <span className="truncate">{item.model}</span> : null}
                        {item.model ? <span className="opacity-50">•</span> : null}
                        <span className="shrink-0">{formatRelativeTime(item.updated_at, locale)}</span>
                      </div>
                    </div>
                  </button>

                  <button
                    onClick={() => onDeleteHistory(item.id)}
                    disabled={deletingHistoryId === item.id}
                    aria-label={copy.historyDelete}
                    title={copy.historyDelete}
                    className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg transition ${
                      selectedHistoryId === item.id
                        ? "text-white/75 hover:bg-white/10 dark:text-slate-950/70 dark:hover:bg-slate-950/10"
                        : "text-slate-400 hover:bg-black/[0.05] hover:text-rose-500 dark:text-white/35 dark:hover:bg-white/[0.06] dark:hover:text-rose-300"
                    } disabled:cursor-not-allowed disabled:opacity-40`}
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </aside>
  );
}
