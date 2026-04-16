import type { StageName, UiLocale } from "../types";

interface CandidateSummary {
  rank?: number;
  candidate_key?: string;
  candidate_label?: string;
  strategy?: string;
  score?: number;
  verify_passed?: boolean;
  changed_file_count?: number;
  modified_files?: string[];
  added_lines?: number;
  removed_lines?: number;
  diff_line_count?: number;
  verification_summary?: string | null;
  assert_count?: number;
  error_message?: string | null;
}

interface CandidateReportPayload {
  collaboration_mode?: string;
  selection_policy?: string;
  candidate_count?: number;
  provisional_leader?: {
    candidate_key?: string;
    candidate_label?: string;
  } | null;
  selected_candidate?: CandidateSummary | null;
  candidates?: CandidateSummary[];
  ranked_candidates?: CandidateSummary[];
}

interface CandidateRankingPanelProps {
  locale: UiLocale;
  reportContent: string;
  stage: StageName;
}

const uiCopy = {
  en: {
    headingCode: "Patch Committee",
    headingVerify: "Patch Ranking",
    subheadingCode: "Multiple coder agents generated candidate patches before verification.",
    subheadingVerify: "Candidates were verified, scored, ranked, and auto-selected.",
    selected: "Selected",
    provisional: "Provisional Leader",
    score: "Score",
    files: "Files",
    lines: "Lines",
    assertions: "Assertions",
    summary: "Verification",
    error: "Error",
    modifiedFiles: "Modified files",
    rawReport: "Raw structured report",
    passed: "Passed",
    failed: "Failed",
    generated: "Generated",
  },
  zh: {
    headingCode: "候选 Patch 委员会",
    headingVerify: "候选 Patch 排名",
    subheadingCode: "多个 coder agent 先生成候选 patch，再进入自动验证。",
    subheadingVerify: "系统已自动验证、打分、排序，并选出最优方案。",
    selected: "已选中",
    provisional: "临时领先",
    score: "得分",
    files: "文件数",
    lines: "改动行",
    assertions: "断言数",
    summary: "验证摘要",
    error: "错误",
    modifiedFiles: "修改文件",
    rawReport: "原始结构化报告",
    passed: "通过",
    failed: "失败",
    generated: "已生成",
  },
} as const;

function parseCandidateReport(reportContent: string): CandidateReportPayload | null {
  try {
    const parsed = JSON.parse(reportContent) as CandidateReportPayload;
    if (
      parsed &&
      parsed.collaboration_mode === "multi_candidate_patch_committee" &&
      (Array.isArray(parsed.candidates) || Array.isArray(parsed.ranked_candidates))
    ) {
      return parsed;
    }
  } catch {
    return null;
  }
  return null;
}

function badgeClass(kind: "selected" | "passed" | "failed" | "default") {
  if (kind === "selected") {
    return "border-sky-400/20 bg-sky-500/10 text-sky-700 dark:text-sky-200";
  }
  if (kind === "passed") {
    return "border-emerald-400/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200";
  }
  if (kind === "failed") {
    return "border-rose-400/20 bg-rose-500/10 text-rose-700 dark:text-rose-200";
  }
  return "border-black/10 bg-black/[0.03] text-slate-700 dark:border-white/10 dark:bg-white/[0.04] dark:text-white/75";
}

function formatMetric(value: number | string | null | undefined) {
  if (value == null || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return new Intl.NumberFormat().format(value);
  }
  return value;
}

export function CandidateRankingPanel({
  locale,
  reportContent,
  stage,
}: CandidateRankingPanelProps) {
  const report = parseCandidateReport(reportContent);
  if (!report) {
    return null;
  }

  const dict = uiCopy[locale];
  const cards = (stage === "verify" ? report.ranked_candidates : report.candidates) ?? [];
  const highlightKey =
    stage === "verify"
      ? report.selected_candidate?.candidate_key
      : report.provisional_leader?.candidate_key;

  return (
    <div className="space-y-3">
      <div className="rounded-[24px] border border-black/5 bg-white/50 p-4 dark:border-white/10 dark:bg-white/[0.03]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white">
              {stage === "verify" ? dict.headingVerify : dict.headingCode}
            </div>
            <div className="mt-1 text-sm text-slate-600 dark:text-white/60">
              {stage === "verify" ? dict.subheadingVerify : dict.subheadingCode}
            </div>
            {report.selection_policy ? (
              <div className="mt-2 text-xs leading-6 text-slate-500 dark:text-white/45">
                {report.selection_policy}
              </div>
            ) : null}
          </div>
          {highlightKey ? (
            <span className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-medium ${badgeClass("selected")}`}>
              {stage === "verify" ? dict.selected : dict.provisional}
            </span>
          ) : null}
        </div>
      </div>

      <div className="grid gap-3">
        {cards.map((candidate, index) => {
          const candidateKey = candidate.candidate_key || `candidate-${index + 1}`;
          const highlighted = candidateKey === highlightKey;
          const statusLabel =
            candidate.verify_passed === true
              ? dict.passed
              : candidate.error_message
                ? dict.failed
                : dict.generated;
          const statusKind =
            highlighted
              ? "selected"
              : candidate.verify_passed === true
                ? "passed"
                : candidate.error_message
                  ? "failed"
                  : "default";

          return (
            <div
              key={candidateKey}
              className={`rounded-[24px] border p-4 ${
                highlighted
                  ? "border-slate-900/15 bg-slate-900/[0.04] dark:border-white/15 dark:bg-white/[0.05]"
                  : "border-black/5 bg-black/[0.02] dark:border-white/10 dark:bg-white/[0.02]"
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    {candidate.rank ? (
                      <span className="rounded-full border border-black/10 px-2.5 py-1 text-[11px] font-medium text-slate-600 dark:border-white/10 dark:text-white/65">
                        #{candidate.rank}
                      </span>
                    ) : null}
                    <div className="font-medium text-slate-900 dark:text-white">
                      {candidate.candidate_label || candidate.candidate_key || candidateKey}
                    </div>
                  </div>
                  {candidate.strategy ? (
                    <div className="mt-2 text-sm text-slate-600 dark:text-white/60">
                      {candidate.strategy}
                    </div>
                  ) : null}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <span className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-medium ${badgeClass(statusKind)}`}>
                    {statusLabel}
                  </span>
                  <span className="rounded-full border border-black/10 px-3 py-1 text-[11px] font-medium text-slate-600 dark:border-white/10 dark:text-white/65">
                    {dict.score}: {formatMetric(candidate.score)}
                  </span>
                </div>
              </div>

              <div className="mt-4 grid gap-2 text-xs text-slate-500 dark:text-white/45 sm:grid-cols-4">
                <div>{dict.files}: {formatMetric(candidate.changed_file_count)}</div>
                <div>
                  {dict.lines}: +{formatMetric(candidate.added_lines)} / -{formatMetric(candidate.removed_lines)}
                </div>
                <div>{dict.assertions}: {formatMetric(candidate.assert_count)}</div>
                <div>{dict.summary}: {candidate.verification_summary || "-"}</div>
              </div>

              {Array.isArray(candidate.modified_files) && candidate.modified_files.length > 0 ? (
                <div className="mt-3">
                  <div className="mb-1 text-[11px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
                    {dict.modifiedFiles}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {candidate.modified_files.slice(0, 6).map((path) => (
                      <span
                        key={path}
                        className="rounded-full border border-black/10 bg-white/50 px-2.5 py-1 font-mono text-[11px] text-slate-700 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/70"
                      >
                        {path}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {candidate.error_message ? (
                <div className="mt-3 rounded-[18px] border border-rose-500/25 bg-rose-500/10 px-3 py-3 text-sm text-rose-700 dark:text-rose-200">
                  {dict.error}: {candidate.error_message}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <details className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-3 dark:border-white/10 dark:bg-white/[0.02]">
        <summary className="cursor-pointer text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
          {dict.rawReport}
        </summary>
        <pre className="mt-3 overflow-y-auto whitespace-pre-wrap break-words font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/65">
          {reportContent}
        </pre>
      </details>
    </div>
  );
}
