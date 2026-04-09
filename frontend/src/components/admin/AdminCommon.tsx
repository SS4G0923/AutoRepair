import type { PropsWithChildren, ReactNode } from "react";

type Tone = "slate" | "sky" | "emerald" | "amber" | "rose";

function toneClasses(tone: Tone) {
  switch (tone) {
    case "sky":
      return "border-sky-400/20 bg-sky-500/10 text-sky-700 dark:text-sky-200";
    case "emerald":
      return "border-emerald-400/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200";
    case "amber":
      return "border-amber-400/20 bg-amber-500/10 text-amber-700 dark:text-amber-200";
    case "rose":
      return "border-rose-400/20 bg-rose-500/10 text-rose-700 dark:text-rose-200";
    default:
      return "border-black/10 bg-black/[0.03] text-slate-700 dark:border-white/10 dark:bg-white/[0.04] dark:text-white/75";
  }
}

export function formatAdminNumber(value: number | null | undefined) {
  return new Intl.NumberFormat().format(Number(value ?? 0));
}

export function formatAdminDateTime(value: string | null | undefined) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function formatAdminDurationMs(value: number | null | undefined) {
  const safeValue = Math.max(0, Number(value ?? 0));
  if (safeValue < 1000) {
    return `${safeValue} ms`;
  }
  if (safeValue < 60_000) {
    return `${(safeValue / 1000).toFixed(safeValue < 10_000 ? 1 : 0)} s`;
  }
  return `${(safeValue / 60_000).toFixed(1)} min`;
}

export function toStatusTone(status: string | null | undefined): Tone {
  const normalized = (status ?? "").toLowerCase();
  if (normalized.includes("fail") || normalized.includes("error") || normalized.includes("suspend")) {
    return "rose";
  }
  if (normalized.includes("admin")) {
    return "sky";
  }
  if (normalized.includes("advanced")) {
    return "amber";
  }
  if (normalized.includes("success") || normalized.includes("completed") || normalized.includes("active")) {
    return "emerald";
  }
  if (normalized.includes("started") || normalized.includes("stream")) {
    return "amber";
  }
  return "slate";
}

export function getNameInitials(name: string, fallback = "") {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return fallback.slice(0, 2).toUpperCase() || "NA";
  }
  return parts
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

interface AdminSurfaceProps extends PropsWithChildren {
  className?: string;
}

export function AdminSurface({ children, className = "" }: AdminSurfaceProps) {
  return (
    <section
      className={`rounded-[24px] border border-black/5 bg-white/70 p-4 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.04] dark:shadow-glow ${className}`.trim()}
    >
      {children}
    </section>
  );
}

interface AdminSectionTitleProps {
  eyebrow?: string;
  title: string;
  hint?: string;
  actions?: ReactNode;
}

export function AdminSectionTitle({
  eyebrow,
  title,
  hint,
  actions,
}: AdminSectionTitleProps) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div>
        {eyebrow ? (
          <div className="text-[11px] uppercase tracking-[0.28em] text-slate-500 dark:text-white/35">
            {eyebrow}
          </div>
        ) : null}
        <div className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">{title}</div>
        {hint ? <div className="mt-1 text-sm text-slate-600 dark:text-white/60">{hint}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}

interface AdminMetricCardProps {
  label: string;
  value: string;
  caption?: string;
  tone?: Tone;
}

export function AdminMetricCard({
  label,
  value,
  caption,
  tone = "slate",
}: AdminMetricCardProps) {
  return (
    <div className={`rounded-[22px] border p-4 ${toneClasses(tone)}`}>
      <div className="text-[11px] uppercase tracking-[0.24em] opacity-70">{label}</div>
      <div className="mt-3 text-2xl font-semibold tracking-tight">{value}</div>
      {caption ? <div className="mt-2 text-sm opacity-75">{caption}</div> : null}
    </div>
  );
}

interface AdminBadgeProps {
  label: string;
  tone?: Tone;
}

export function AdminBadge({ label, tone = "slate" }: AdminBadgeProps) {
  return (
    <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-medium ${toneClasses(tone)}`}>
      {label}
    </span>
  );
}

interface AdminEmptyStateProps {
  message: string;
}

export function AdminEmptyState({ message }: AdminEmptyStateProps) {
  return (
    <div className="grid min-h-[12rem] place-items-center rounded-[24px] border border-dashed border-black/10 bg-black/[0.02] px-6 py-8 text-center text-sm text-slate-500 dark:border-white/10 dark:bg-white/[0.02] dark:text-white/45">
      {message}
    </div>
  );
}

interface AdminCodeBlockProps {
  title: string;
  content: string;
}

export function AdminCodeBlock({ title, content }: AdminCodeBlockProps) {
  return (
    <div className="rounded-[22px] border border-black/5 bg-black/[0.025] dark:border-white/10 dark:bg-white/[0.025]">
      <div className="border-b border-black/5 px-4 py-3 text-xs uppercase tracking-[0.24em] text-slate-500 dark:border-white/10 dark:text-white/35">
        {title}
      </div>
      <pre className="max-h-[18rem] overflow-auto px-4 py-4 text-xs leading-6 text-slate-700 dark:text-white/75">
        {content.trim() || "-"}
      </pre>
    </div>
  );
}
