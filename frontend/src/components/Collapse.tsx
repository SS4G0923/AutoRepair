import type { ReactNode } from "react";

interface CollapseProps {
  open: boolean;
  children: ReactNode;
  /**
   * Transition duration in milliseconds. Defaults to a value that feels natural
   * without being draggy (roughly aligned with Material / Tailwind defaults).
   */
  duration?: number;
  className?: string;
}

/**
 * Low-level open/close animation primitive using the CSS grid-template-rows
 * 0fr → 1fr trick. Unlike a classic `<details>` element this keeps children
 * mounted but visually collapses them, so:
 *
 * - Content does not jump in/out of the DOM every toggle (form state, tool
 *   event lists, etc. stay stable).
 * - The animation works without JS-measured heights and smoothly handles
 *   content whose height changes while expanded.
 *
 * Pair with any external trigger (button, row click, etc.) that controls the
 * `open` prop. `aria-hidden` tracks the collapsed state so assistive tech
 * skips hidden content.
 */
export function Collapse({ open, children, duration = 260, className = "" }: CollapseProps) {
  return (
    <div
      className={`grid overflow-hidden transition-[grid-template-rows,opacity] ease-out ${
        open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
      } ${className}`}
      style={{ transitionDuration: `${duration}ms` }}
      aria-hidden={!open}
    >
      <div className="min-h-0 overflow-hidden">{children}</div>
    </div>
  );
}
