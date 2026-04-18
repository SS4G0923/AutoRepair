import { useCallback, useState, type ReactNode } from "react";
import { Collapse } from "./Collapse";

interface CollapsibleSectionProps {
  title: ReactNode;
  meta?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  /**
   * Visual tone for the container border/background.
   * - `default`: neutral slate surface, used for most stage sections
   * - `accent`: used for the main explain summary to stand out slightly
   * - `warning`: used for reasoning / retry content
   */
  tone?: "default" | "accent" | "warning";
  /**
   * Body padding. When the inner content renders its own card styling (e.g. the
   * candidate ranking panel) pass `"none"` so we do not add duplicate padding.
   */
  bodyPadding?: "default" | "tight" | "none";
  className?: string;
}

const TONE_CLASS: Record<NonNullable<CollapsibleSectionProps["tone"]>, string> = {
  default:
    "border-black/5 bg-white/60 dark:border-white/10 dark:bg-white/[0.03]",
  accent:
    "border-sky-400/20 bg-sky-400/5 dark:border-sky-300/20 dark:bg-sky-400/[0.06]",
  warning:
    "border-amber-500/20 bg-amber-500/5 dark:border-amber-300/20 dark:bg-amber-400/[0.06]",
};

const BODY_PADDING_CLASS: Record<NonNullable<CollapsibleSectionProps["bodyPadding"]>, string> = {
  default: "px-4 py-3",
  tight: "px-3 py-2",
  none: "",
};

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

export function CollapsibleSection({
  title,
  meta,
  children,
  defaultOpen = false,
  tone = "default",
  bodyPadding = "default",
  className = "",
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState<boolean>(defaultOpen);
  const toggle = useCallback(() => setOpen((prev) => !prev), []);

  return (
    <div className={`rounded-2xl border ${TONE_CLASS[tone]} ${className}`}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left"
      >
        <div className="flex min-w-0 items-center gap-2 text-xs uppercase tracking-[0.22em] text-slate-600 dark:text-white/55">
          <Chevron open={open} />
          <span className="truncate">{title}</span>
        </div>
        {meta !== undefined && meta !== null && meta !== "" ? (
          <div className="shrink-0 text-[11px] text-slate-500 dark:text-white/50">{meta}</div>
        ) : null}
      </button>

      <Collapse open={open} duration={300}>
        <div
          className={`${bodyPadding === "none" ? "" : "border-t border-black/5 dark:border-white/10"} ${BODY_PADDING_CLASS[bodyPadding]}`}
        >
          {children}
        </div>
      </Collapse>
    </div>
  );
}
