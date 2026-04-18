import { useCallback, useState } from "react";
import { Collapse } from "./Collapse";

interface ReasoningPanelProps {
  label: string;
  showLabel: string;
  hideLabel: string;
  content: string;
  defaultOpen?: boolean;
  compact?: boolean;
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 12 12"
      width={10}
      height={10}
      className={`shrink-0 transition-transform duration-300 ease-out ${open ? "rotate-90" : ""}`}
      aria-hidden
    >
      <path
        d="M4 2l4 4-4 4"
        stroke="currentColor"
        strokeWidth={1.5}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ReasoningPanel({
  label,
  showLabel,
  hideLabel,
  content,
  defaultOpen = false,
  compact = false,
}: ReasoningPanelProps) {
  const [open, setOpen] = useState<boolean>(defaultOpen);
  const toggle = useCallback(() => setOpen((prev) => !prev), []);

  if (!content.trim()) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-amber-700 dark:text-amber-200/80">
          <Chevron open={open} />
          {label}
        </div>
        <div className="text-[11px] text-amber-700/80 dark:text-amber-200/70">
          {open ? hideLabel : showLabel}
        </div>
      </button>
      <Collapse open={open} duration={280}>
        <div className="border-t border-amber-500/15 px-4 py-3">
          <pre
            className={`whitespace-pre-wrap break-words font-mono text-slate-700 [overflow-wrap:anywhere] dark:text-white/80 ${
              compact ? "text-[11px] leading-6" : "text-xs leading-6"
            }`}
          >
            {content}
          </pre>
        </div>
      </Collapse>
    </div>
  );
}
