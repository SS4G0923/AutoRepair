import type { EventEntry } from "../types";

interface EventFeedProps {
  events: EventEntry[];
  title: string;
  emptyText: string;
}

export function EventFeed({ events, title, emptyText }: EventFeedProps) {
  return (
    <section className="min-w-0 rounded-[28px] border border-black/5 bg-white/75 p-5 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
      <div className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">{title}</div>
      <div className="mt-4 max-h-[300px] space-y-3 overflow-auto pr-1">
        {events.length === 0 ? (
          <div className="rounded-3xl border border-dashed border-slate-300/70 px-4 py-6 text-sm text-slate-500 dark:border-white/10 dark:text-white/45">
            {emptyText}
          </div>
        ) : (
          events.map((event) => (
            <div
              key={event.id}
              className="rounded-3xl border border-black/5 bg-black/[0.03] px-4 py-3 dark:border-white/10 dark:bg-white/[0.03]"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="font-mono text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/45">
                  {event.event}
                </div>
                <div className="text-xs text-slate-400 dark:text-white/30">{event.at}</div>
              </div>
              <div className="mt-2 break-words text-sm text-slate-700 [overflow-wrap:anywhere] dark:text-white/75">
                {event.summary}
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
