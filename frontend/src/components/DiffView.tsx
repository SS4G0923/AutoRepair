import { useMemo } from "react";

interface DiffViewProps {
  content: string;
  /**
   * Optional max height for the scroll container. Defaults to a reasonable
   * in-card height; pass `"none"` to let it grow with content.
   */
  maxHeight?: string | "none";
  /**
   * When true, the view drops its own rounded border so it can be embedded
   * flush inside a parent container that already provides those styles.
   */
  bare?: boolean;
  className?: string;
}

type LineKind = "add" | "remove" | "hunk" | "file" | "meta" | "context";

interface DiffLine {
  kind: LineKind;
  text: string;
}

function classifyLine(line: string): LineKind {
  if (line.startsWith("@@")) return "hunk";
  if (
    line.startsWith("diff --git ") ||
    line.startsWith("index ") ||
    line.startsWith("new file mode") ||
    line.startsWith("deleted file mode") ||
    line.startsWith("similarity index") ||
    line.startsWith("rename from") ||
    line.startsWith("rename to")
  ) {
    return "meta";
  }
  if (line.startsWith("+++") || line.startsWith("---")) return "file";
  if (line.startsWith("+")) return "add";
  if (line.startsWith("-")) return "remove";
  return "context";
}

const LINE_CLASS: Record<LineKind, string> = {
  add: "bg-emerald-500/10 text-emerald-800 dark:text-emerald-200",
  remove: "bg-rose-500/10 text-rose-800 dark:text-rose-200",
  hunk: "bg-sky-500/10 text-sky-700 dark:text-sky-200",
  file: "bg-slate-500/10 text-slate-600 dark:text-white/65",
  meta: "text-slate-400 dark:text-white/35",
  context: "text-slate-700 dark:text-white/70",
};

const PREFIX_CLASS: Record<LineKind, string> = {
  add: "text-emerald-600 dark:text-emerald-300",
  remove: "text-rose-600 dark:text-rose-300",
  hunk: "text-sky-600 dark:text-sky-300",
  file: "text-slate-500 dark:text-white/45",
  meta: "text-slate-400 dark:text-white/30",
  context: "text-slate-400 dark:text-white/35",
};

export function DiffView({
  content,
  maxHeight = "28rem",
  bare = false,
  className = "",
}: DiffViewProps) {
  const lines = useMemo<DiffLine[]>(() => {
    if (!content) return [];
    const raw = content.endsWith("\n") ? content.slice(0, -1) : content;
    return raw.split("\n").map((text) => ({ kind: classifyLine(text), text }));
  }, [content]);

  if (lines.length === 0) {
    return null;
  }

  const scrollStyle =
    maxHeight === "none"
      ? undefined
      : { maxHeight, overflowY: "auto" as const };

  const shellClass = bare
    ? "bg-transparent font-mono text-[12px] leading-6"
    : "rounded-2xl border border-black/5 bg-black/[0.02] font-mono text-[12px] leading-6 dark:border-white/10 dark:bg-white/[0.02]";

  return (
    <div className={`${shellClass} ${className}`} style={scrollStyle}>
      <div className="py-1">
        {lines.map((line, idx) => {
          const prefix = line.text.charAt(0);
          const body = line.text.length > 0 ? line.text.slice(1) : "";
          const showPrefix = line.kind === "add" || line.kind === "remove" || line.kind === "context";
          return (
            <div
              key={idx}
              className={`flex min-h-[1.5rem] items-start gap-3 px-3 ${LINE_CLASS[line.kind]}`}
            >
              <span
                className={`select-none tabular-nums text-right ${PREFIX_CLASS[line.kind]}`}
                style={{ minWidth: "2ch" }}
              >
                {showPrefix ? prefix : ""}
              </span>
              <span className="min-w-0 flex-1 whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
                {showPrefix ? body : line.text || "\u00a0"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function computeDiffStats(diff: string): {
  added: number;
  removed: number;
  files: number;
} {
  if (!diff) return { added: 0, removed: 0, files: 0 };
  let added = 0;
  let removed = 0;
  const files = new Set<string>();
  for (const rawLine of diff.split("\n")) {
    if (rawLine.startsWith("+++ ") || rawLine.startsWith("--- ")) {
      const path = rawLine.slice(4).replace(/^[ab]\//, "").trim();
      if (path && path !== "/dev/null") {
        files.add(path);
      }
      continue;
    }
    if (rawLine.startsWith("+") && !rawLine.startsWith("+++")) {
      added += 1;
    } else if (rawLine.startsWith("-") && !rawLine.startsWith("---")) {
      removed += 1;
    }
  }
  return { added, removed, files: files.size };
}
