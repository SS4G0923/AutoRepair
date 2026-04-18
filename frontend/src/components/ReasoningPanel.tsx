interface ReasoningPanelProps {
  label: string;
  showLabel: string;
  hideLabel: string;
  content: string;
  defaultOpen?: boolean;
  compact?: boolean;
}

export function ReasoningPanel({
  label,
  showLabel,
  hideLabel,
  content,
  defaultOpen = false,
  compact = false,
}: ReasoningPanelProps) {
  if (!content.trim()) {
    return null;
  }

  return (
    <details
      className="rounded-2xl border border-amber-500/20 bg-amber-500/5"
      open={defaultOpen}
    >
      <summary className="cursor-pointer list-none px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs uppercase tracking-[0.22em] text-amber-700 dark:text-amber-200/80">
            {label}
          </div>
          <div className="text-[11px] text-amber-700/80 dark:text-amber-200/70">
            {defaultOpen ? hideLabel : showLabel}
          </div>
        </div>
      </summary>
      <div className="border-t border-amber-500/15 px-4 py-3">
        <pre
          className={`whitespace-pre-wrap break-words font-mono text-slate-700 [overflow-wrap:anywhere] dark:text-white/80 ${
            compact ? "text-[11px] leading-6" : "text-xs leading-6"
          }`}
        >
          {content}
        </pre>
      </div>
    </details>
  );
}
