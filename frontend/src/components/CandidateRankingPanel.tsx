import type { StageName, UiLocale } from "../types";
import { CollapsibleSection } from "./CollapsibleSection";

type AssertionStatus = "passed" | "failed" | "skipped" | "unknown";

interface AssertionStatusEntry {
  index: number;
  code: string;
  target: string;
  status: AssertionStatus;
}

interface ScoreBreakdownEntry {
  code: string;
  delta: number;
  note?: string;
}

interface VerificationReport {
  summary?: string;
  verification_code?: string[];
  assertion_targets?: string[];
  assert_count?: number;
  assertion_statuses?: AssertionStatusEntry[];
  failed_assertion_index?: number | null;
  verification_strategy?: string;
  sanitization_notes?: string[];
  passed?: boolean;
  modified_files?: string[];
}

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
  // Present only on the winning candidate (stage=verify, selected_candidate).
  verification_report?: VerificationReport | null;
  score_breakdown?: ScoreBreakdownEntry[];
  verification_stderr?: string | null;
}

interface CandidateReportPayload {
  collaboration_mode?: string;
  selection_policy?: string;
  candidate_count?: number;
  provisional_leader?: {
    candidate_key?: string;
    candidate_label?: string;
  } | null;
  selected_candidate?: (CandidateSummary & {
    verification_report?: VerificationReport | null;
    score_breakdown?: ScoreBreakdownEntry[];
    verification_stderr?: string | null;
  }) | null;
  candidates?: CandidateSummary[];
  ranked_candidates?: CandidateSummary[];
}

interface CandidateRankingPanelProps {
  locale: UiLocale;
  reportContent: string;
  stage: StageName;
  /**
   * When true the panel skips the outer heading/subheading wrapper (because
   * the parent collapsible section already labels the report) and hides the
   * raw JSON toggle to reduce visual noise.
   */
  compact?: boolean;
}

const uiCopy = {
  en: {
    headingCodeMulti: "Patch Committee",
    headingVerifyMulti: "Patch Ranking",
    subheadingCodeMulti: "Multiple coder agents generated candidate patches before verification.",
    subheadingVerifyMulti: "Candidates were verified, scored, ranked, and auto-selected.",
    headingCodeSingle: "Generated Patch",
    headingVerifySingle: "Patch Verification",
    subheadingCodeSingle: "A single repair agent generated the current patch.",
    subheadingVerifySingle: "The generated patch was verified and kept as the final result.",
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
    assertionsTitle: "Assertions",
    assertionsMeta: (total: number, passed: number, failed: number) =>
      `${passed}/${total} passed${failed > 0 ? `, ${failed} failed` : ""}`,
    assertionPassed: "passed",
    assertionFailed: "failed",
    assertionSkipped: "not run",
    assertionUnknown: "unclear",
    assertionNoTarget: "(no description)",
    assertionRuntimeOnly:
      "This run used runtime rerun only — no custom assert statements were generated.",
    verifyStderrTitle: "Verification stderr",
    scoreBreakdownTitle: "Score breakdown",
    scoreBreakdownMeta: (total: number) => `total ${total}`,
    scoreRuleCopy: {
      diff_generated: "Produced a valid unified diff",
      patched_runtime_ok: "Patched project ran to completion",
      patched_runtime_timeout: "Patched project timed out during rerun",
      patched_runtime_failed: "Patched project still exited non-zero",
      assertion_coverage: "Assertion coverage",
      verification_ok: "Verification block passed cleanly",
      verification_timeout: "Verification execution timed out",
      verification_failed: "Verification execution failed",
      verify_passed: "Rerun and assertions both passed",
      extra_files_penalty: "Patch touched multiple files",
      diff_size_penalty: "Large patch size",
      error_reported: "Error reported during verification",
    } as Record<string, string>,
  },
  zh: {
    headingCodeMulti: "候选 Patch 委员会",
    headingVerifyMulti: "候选 Patch 排名",
    subheadingCodeMulti: "多个 coder agent 先生成候选 patch，再进入自动验证。",
    subheadingVerifyMulti: "系统已自动验证、打分、排序，并选出最优方案。",
    headingCodeSingle: "Patch 生成结果",
    headingVerifySingle: "Patch 验证结果",
    subheadingCodeSingle: "系统仅使用一个 repair agent 生成当前 patch。",
    subheadingVerifySingle: "系统已对当前 patch 完成自动验证。",
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
    assertionsTitle: "断言检查",
    assertionsMeta: (total: number, passed: number, failed: number) =>
      `${passed}/${total} 通过${failed > 0 ? `，${failed} 失败` : ""}`,
    assertionPassed: "通过",
    assertionFailed: "失败",
    assertionSkipped: "未执行",
    assertionUnknown: "未知",
    assertionNoTarget: "（未提供说明）",
    assertionRuntimeOnly: "本次仅做了运行时重跑，没有生成自定义 assert。",
    verifyStderrTitle: "验证错误输出",
    scoreBreakdownTitle: "得分依据",
    scoreBreakdownMeta: (total: number) => `合计 ${total}`,
    scoreRuleCopy: {
      diff_generated: "生成了合法的 unified diff",
      patched_runtime_ok: "打过补丁的项目可以正常跑通",
      patched_runtime_timeout: "打过补丁的项目运行超时",
      patched_runtime_failed: "打过补丁的项目仍以非 0 退出",
      assertion_coverage: "断言覆盖度",
      verification_ok: "验证脚本（assert + 重跑）顺利通过",
      verification_timeout: "验证脚本执行超时",
      verification_failed: "验证脚本执行失败",
      verify_passed: "重跑与所有断言同时通过",
      extra_files_penalty: "补丁改动了多个文件",
      diff_size_penalty: "补丁体量较大",
      error_reported: "验证期间报告了错误",
    } as Record<string, string>,
  },
} as const;

function parseCandidateReport(reportContent: string): CandidateReportPayload | null {
  try {
    const parsed = JSON.parse(reportContent) as CandidateReportPayload;
    if (
      parsed &&
      (parsed.collaboration_mode === "multi_candidate_patch_committee" ||
        parsed.collaboration_mode === "single_patch_generation") &&
      (Array.isArray(parsed.candidates) || Array.isArray(parsed.ranked_candidates))
    ) {
      return parsed;
    }
  } catch {
    return null;
  }
  return null;
}

export function isCandidateReportContent(reportContent: string) {
  return parseCandidateReport(reportContent) !== null;
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
  compact = false,
}: CandidateRankingPanelProps) {
  const report = parseCandidateReport(reportContent);
  if (!report) {
    return null;
  }

  const dict = uiCopy[locale];
  const isMultiCandidateMode = report.collaboration_mode === "multi_candidate_patch_committee";
  const cards = (stage === "verify" ? report.ranked_candidates : report.candidates) ?? [];
  const highlightKey =
    stage === "verify"
      ? report.selected_candidate?.candidate_key
      : report.provisional_leader?.candidate_key;
  const heading =
    stage === "verify"
      ? isMultiCandidateMode
        ? dict.headingVerifyMulti
        : dict.headingVerifySingle
      : isMultiCandidateMode
        ? dict.headingCodeMulti
        : dict.headingCodeSingle;
  const subheading =
    stage === "verify"
      ? isMultiCandidateMode
        ? dict.subheadingVerifyMulti
        : dict.subheadingVerifySingle
      : isMultiCandidateMode
        ? dict.subheadingCodeMulti
        : dict.subheadingCodeSingle;

  return (
    <div className="space-y-3">
      {!compact ? (
        <div className="rounded-[24px] border border-black/5 bg-white/50 p-4 dark:border-white/10 dark:bg-white/[0.03]">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-slate-900 dark:text-white">
                {heading}
              </div>
              <div className="mt-1 text-sm text-slate-600 dark:text-white/60">
                {subheading}
              </div>
              {report.selection_policy ? (
                <div className="mt-2 text-xs leading-6 text-slate-500 dark:text-white/45">
                  {report.selection_policy}
                </div>
              ) : null}
            </div>
            {isMultiCandidateMode && highlightKey ? (
              <span className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-medium ${badgeClass("selected")}`}>
                {stage === "verify" ? dict.selected : dict.provisional}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="grid gap-3">
        {cards.map((candidate, index) => {
          const candidateKey = candidate.candidate_key || `candidate-${index + 1}`;
          const highlighted = candidateKey === highlightKey;
          const useSelectedStyling = isMultiCandidateMode && highlighted;
          const statusLabel =
            candidate.verify_passed === true
              ? dict.passed
              : candidate.error_message
                ? dict.failed
                : dict.generated;
          const statusKind =
            useSelectedStyling
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
                useSelectedStyling
                  ? "border-slate-900/15 bg-slate-900/[0.04] dark:border-white/15 dark:bg-white/[0.05]"
                  : "border-black/5 bg-black/[0.02] dark:border-white/10 dark:bg-white/[0.02]"
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    {isMultiCandidateMode && candidate.rank ? (
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

              {(() => {
                const metrics: { label: string; value: string }[] = [];
                if ((candidate.changed_file_count ?? 0) > 0) {
                  metrics.push({
                    label: dict.files,
                    value: formatMetric(candidate.changed_file_count),
                  });
                }
                if ((candidate.added_lines ?? 0) > 0 || (candidate.removed_lines ?? 0) > 0) {
                  metrics.push({
                    label: dict.lines,
                    value: `+${formatMetric(candidate.added_lines)} / -${formatMetric(candidate.removed_lines)}`,
                  });
                }
                if ((candidate.assert_count ?? 0) > 0) {
                  metrics.push({
                    label: dict.assertions,
                    value: formatMetric(candidate.assert_count),
                  });
                }
                if (candidate.verification_summary) {
                  metrics.push({
                    label: dict.summary,
                    value: candidate.verification_summary,
                  });
                }
                if (metrics.length === 0) return null;
                return (
                  <div className="mt-4 grid gap-2 text-xs text-slate-500 dark:text-white/45 sm:grid-cols-2 lg:grid-cols-4">
                    {metrics.map((m) => (
                      <div key={m.label} className="min-w-0">
                        <span className="text-slate-500 dark:text-white/40">{m.label}: </span>
                        <span className="text-slate-700 dark:text-white/70">{m.value}</span>
                      </div>
                    ))}
                  </div>
                );
              })()}

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

      {stage === "verify" && report.selected_candidate ? (
        <VerifyDetails
          dict={dict}
          selected={report.selected_candidate}
        />
      ) : null}

      {!compact ? (
        <details className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-3 dark:border-white/10 dark:bg-white/[0.02]">
          <summary className="cursor-pointer text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
            {dict.rawReport}
          </summary>
          <pre className="mt-3 overflow-y-auto whitespace-pre-wrap break-words font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:text-white/65">
            {reportContent}
          </pre>
        </details>
      ) : null}
    </div>
  );
}

const STATUS_STYLE: Record<AssertionStatus, { dot: string; chip: string; row: string }> = {
  passed: {
    dot: "bg-emerald-500",
    chip: "border-emerald-400/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200",
    row: "border-emerald-400/20 bg-emerald-500/[0.04] dark:bg-emerald-500/[0.06]",
  },
  failed: {
    dot: "bg-rose-500",
    chip: "border-rose-400/25 bg-rose-500/10 text-rose-700 dark:text-rose-200",
    row: "border-rose-400/25 bg-rose-500/[0.06] dark:bg-rose-500/[0.08]",
  },
  skipped: {
    dot: "bg-slate-400",
    chip: "border-black/10 bg-black/[0.04] text-slate-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-white/60",
    row: "border-black/5 bg-black/[0.015] dark:border-white/10 dark:bg-white/[0.02]",
  },
  unknown: {
    dot: "bg-amber-400",
    chip: "border-amber-400/30 bg-amber-400/10 text-amber-700 dark:text-amber-200",
    row: "border-amber-400/20 bg-amber-400/[0.06]",
  },
};

function statusLabel(status: AssertionStatus, dict: (typeof uiCopy)["en"] | (typeof uiCopy)["zh"]): string {
  switch (status) {
    case "passed":
      return dict.assertionPassed;
    case "failed":
      return dict.assertionFailed;
    case "skipped":
      return dict.assertionSkipped;
    default:
      return dict.assertionUnknown;
  }
}

interface VerifyDetailsProps {
  dict: (typeof uiCopy)["en"] | (typeof uiCopy)["zh"];
  selected: NonNullable<CandidateReportPayload["selected_candidate"]>;
}

function VerifyDetails({ dict, selected }: VerifyDetailsProps) {
  const report = selected.verification_report ?? null;
  const statuses = report?.assertion_statuses ?? [];
  const breakdown = selected.score_breakdown ?? [];
  const runtimeOnly = report?.verification_strategy === "runtime_rerun_only";

  const assertionsTotal = statuses.length;
  const assertionsPassed = statuses.filter((s) => s.status === "passed").length;
  const assertionsFailed = statuses.filter((s) => s.status === "failed").length;

  const showAssertionsSection = Boolean(report) && (runtimeOnly || assertionsTotal > 0);
  const showBreakdownSection = breakdown.length > 0;

  if (!showAssertionsSection && !showBreakdownSection) {
    return null;
  }

  return (
    <div className="space-y-2">
      {showAssertionsSection ? (
        <CollapsibleSection
          title={dict.assertionsTitle}
          meta={
            runtimeOnly
              ? undefined
              : dict.assertionsMeta(assertionsTotal, assertionsPassed, assertionsFailed)
          }
          defaultOpen
        >
          {runtimeOnly ? (
            <div className="text-xs text-slate-600 dark:text-white/60">
              {dict.assertionRuntimeOnly}
            </div>
          ) : (
            <div className="space-y-2">
              {statuses.map((entry) => {
                const style = STATUS_STYLE[entry.status];
                return (
                  <div
                    key={`${entry.index}-${entry.code}`}
                    className={`rounded-xl border px-3 py-2.5 ${style.row}`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
                        <span className="text-[11px] font-mono text-slate-500 dark:text-white/45">
                          #{entry.index}
                        </span>
                        <span className="truncate text-sm text-slate-800 dark:text-white/85">
                          {entry.target.trim() || dict.assertionNoTarget}
                        </span>
                      </div>
                      <span
                        className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] ${style.chip}`}
                      >
                        {statusLabel(entry.status, dict)}
                      </span>
                    </div>
                    {entry.code ? (
                      <pre className="mt-1.5 whitespace-pre-wrap break-words rounded-lg border border-black/5 bg-black/[0.03] px-2 py-1 font-mono text-[11px] leading-5 text-slate-700 [overflow-wrap:anywhere] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/75">
                        {entry.code}
                      </pre>
                    ) : null}
                  </div>
                );
              })}

              {selected.verification_stderr && (assertionsFailed > 0 || assertionsTotal === 0) ? (
                <CollapsibleSection
                  title={dict.verifyStderrTitle}
                  meta={`${selected.verification_stderr.length.toLocaleString()} chars`}
                  tone="warning"
                >
                  <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-slate-700 [overflow-wrap:anywhere] dark:text-white/70">
                    {selected.verification_stderr}
                  </pre>
                </CollapsibleSection>
              ) : null}
            </div>
          )}
        </CollapsibleSection>
      ) : null}

      {showBreakdownSection ? (
        <CollapsibleSection
          title={dict.scoreBreakdownTitle}
          meta={dict.scoreBreakdownMeta(typeof selected.score === "number" ? selected.score : 0)}
          defaultOpen
        >
          <div className="space-y-1.5">
            {breakdown.map((entry, idx) => {
              const isBonus = entry.delta > 0;
              const label =
                dict.scoreRuleCopy[entry.code] || entry.code.replace(/_/g, " ");
              return (
                <div
                  key={`${entry.code}-${idx}`}
                  className="flex items-start justify-between gap-3 rounded-xl border border-black/5 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.02]"
                >
                  <div className="min-w-0">
                    <div className="text-sm text-slate-800 dark:text-white/85">
                      {label}
                    </div>
                    {entry.note ? (
                      <div className="mt-0.5 text-[11px] text-slate-500 dark:text-white/50">
                        {entry.note}
                      </div>
                    ) : null}
                  </div>
                  <span
                    className={`shrink-0 rounded-full border px-2.5 py-0.5 font-mono text-[11px] ${
                      isBonus
                        ? "border-emerald-400/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200"
                        : "border-rose-400/25 bg-rose-500/10 text-rose-700 dark:text-rose-200"
                    }`}
                  >
                    {isBonus ? "+" : ""}
                    {entry.delta}
                  </span>
                </div>
              );
            })}
          </div>
        </CollapsibleSection>
      ) : null}
    </div>
  );
}
