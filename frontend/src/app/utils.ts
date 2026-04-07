import type { AuthenticatedUser, StageName, StageState } from "../types";

export const SIDEBAR_WIDTH_STORAGE_KEY = "autorepair-sidebar-width";
export const DEFAULT_SIDEBAR_WIDTH = 200;
export const MIN_SIDEBAR_WIDTH = 160;
export const MAX_SIDEBAR_WIDTH = 560;

export const STAGE_NAMES: StageName[] = ["run", "inspect", "plan", "code", "verify"];

export const createStageState = (): Record<StageName, StageState> => ({
  run: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  inspect: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  plan: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  code: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  verify: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
});

export function formatTimestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function fileToBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const raw = typeof reader.result === "string" ? reader.result : "";
      const commaIndex = raw.indexOf(",");
      if (commaIndex === -1) {
        reject(new Error("Failed to read ZIP archive."));
        return;
      }
      resolve(raw.slice(commaIndex + 1));
    };
    reader.onerror = () => reject(new Error("Failed to read ZIP archive."));
    reader.readAsDataURL(file);
  });
}

export function buildSummary(event: string, data: Record<string, unknown>) {
  if (event === "stage") {
    return `${String(data.stage)} · ${String(data.status)}`;
  }
  if (event === "tool_event") {
    return `${String(data.stage)} · ${String(data.tool_name)} · ${String(data.status)}`;
  }
  if (event === "error") {
    return "error";
  }
  if (event === "result") {
    return "result";
  }
  return JSON.stringify(data).slice(0, 140);
}

export function parseSseBlock(block: string) {
  const lines = block.split("\n");
  let event = "message";
  const dataParts: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataParts.push(line.slice(5).trimStart());
    }
  }

  return {
    event,
    data: dataParts.length > 0 ? (JSON.parse(dataParts.join("\n")) as Record<string, unknown>) : {},
  };
}

export function isStageName(value: unknown): value is StageName {
  return typeof value === "string" && STAGE_NAMES.includes(value as StageName);
}

export function normalizeStageMap(
  incoming?: Partial<Record<StageName, Partial<StageState>>>,
): Record<StageName, StageState> {
  const base = createStageState();
  if (!incoming) {
    return base;
  }

  for (const stage of STAGE_NAMES) {
    const next = incoming[stage];
    if (!next) {
      continue;
    }
    base[stage] = {
      status: next.status ?? base[stage].status,
      explain: next.explain ?? base[stage].explain,
      report: next.report ?? base[stage].report,
      diff: next.diff ?? base[stage].diff,
      toolEvents: Array.isArray(next.toolEvents)
        ? next.toolEvents.map((item, index) => ({
            id:
              typeof item.id === "string" && item.id
                ? item.id
                : `history-tool-${stage}-${index}`,
            tool_name:
              typeof item.tool_name === "string" && item.tool_name ? item.tool_name : "tool",
            status: item.status === "completed" ? "completed" : "started",
            round: typeof item.round === "number" ? item.round : undefined,
            arguments: typeof item.arguments === "string" ? item.arguments : undefined,
            output_preview:
              typeof item.output_preview === "string" ? item.output_preview : undefined,
            output_truncated: Boolean(item.output_truncated ?? false),
            at: typeof item.at === "string" ? item.at : "",
          }))
        : [],
    };
  }

  return base;
}

export function applyUnifiedDiffToText(originalText: string, diffText: string) {
  const normalizedOriginal = originalText.replace(/\r\n/g, "\n");
  const normalizedDiff = diffText.replace(/\r\n/g, "\n");
  const sourceLines = normalizedOriginal.split("\n");
  const diffLines = normalizedDiff.split("\n");
  const hadTrailingNewline = normalizedOriginal.endsWith("\n");

  let diffIndex = diffLines.findIndex((line) => line.startsWith("@@"));
  if (diffIndex === -1) {
    throw new Error("Missing hunk header.");
  }

  const result: string[] = [];
  let sourceIndex = 0;

  while (diffIndex < diffLines.length) {
    const header = diffLines[diffIndex];
    if (!header.startsWith("@@")) {
      diffIndex += 1;
      continue;
    }

    const match = header.match(/^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@/);
    if (!match) {
      throw new Error(`Invalid hunk header: ${header}`);
    }

    const targetStart = Math.max(0, Number(match[1]) - 1);
    while (sourceIndex < targetStart && sourceIndex < sourceLines.length) {
      result.push(sourceLines[sourceIndex]);
      sourceIndex += 1;
    }

    diffIndex += 1;
    while (diffIndex < diffLines.length && !diffLines[diffIndex].startsWith("@@")) {
      const line = diffLines[diffIndex];
      if (!line || line === "\\ No newline at end of file") {
        diffIndex += 1;
        continue;
      }
      const prefix = line[0];
      const value = line.slice(1);
      if (prefix === " ") {
        result.push(value);
        sourceIndex += 1;
      } else if (prefix === "-") {
        sourceIndex += 1;
      } else if (prefix === "+") {
        result.push(value);
      }
      diffIndex += 1;
    }
  }

  while (sourceIndex < sourceLines.length) {
    result.push(sourceLines[sourceIndex]);
    sourceIndex += 1;
  }

  let nextText = result.join("\n");
  if (hadTrailingNewline && !nextText.endsWith("\n")) {
    nextText += "\n";
  }
  return nextText;
}

export function getUserInitials(user: AuthenticatedUser) {
  const parts = user.display_name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return user.email.slice(0, 2).toUpperCase();
  }
  return parts
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}
