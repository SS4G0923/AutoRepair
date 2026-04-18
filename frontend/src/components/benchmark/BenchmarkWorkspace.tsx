import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppCopy } from "../../i18n";
import type {
  BenchmarkBug,
  BenchmarkExperimentDetail,
  BenchmarkExperimentSummary,
  BenchmarkLeaderboardItem,
  BenchmarkPage,
  BenchmarkProject,
  BenchmarkRunDetail,
  BenchmarkRunMode,
  BenchmarkRunSummary,
  BenchmarkStrategy,
  BenchmarkSummary,
  ModelCatalogItem,
} from "../../types";
import {
  AdminBadge,
  AdminCodeBlock,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminDurationMs,
  formatAdminNumber,
  toStatusTone,
} from "../admin/AdminCommon";

interface BenchmarkWorkspaceProps {
  apiBaseUrl: string;
  copy: AppCopy;
  page: BenchmarkPage;
  modelOptions: ModelCatalogItem[];
  model: string;
  onModelChange: (value: string) => void;
  onPageChange: (page: BenchmarkPage) => void;
  workspaceMainClass: string;
}

function severityTone(severity: string): "rose" | "amber" | "emerald" | "slate" {
  const normalized = severity.toLowerCase();
  if (normalized === "critical" || normalized === "high" || normalized === "hard") return "rose";
  if (normalized === "medium" || normalized === "normal") return "amber";
  if (normalized === "low" || normalized === "easy") return "emerald";
  return "slate";
}

function safeLog(report: Record<string, unknown> | null | undefined, key: string): string {
  if (!report) return "";
  const value = report[key];
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }
  return "";
}

export function BenchmarkWorkspace({
  apiBaseUrl,
  copy,
  page,
  modelOptions,
  model,
  onModelChange,
  onPageChange,
  workspaceMainClass,
}: BenchmarkWorkspaceProps) {
  const [projects, setProjects] = useState<BenchmarkProject[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [bugs, setBugs] = useState<BenchmarkBug[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [selectedBugId, setSelectedBugId] = useState<number | null>(null);
  const [runMode, setRunMode] = useState<BenchmarkRunMode>("full_repair");
  const [strategy, setStrategy] = useState<BenchmarkStrategy>("full_pipeline");
  const [bugQuery, setBugQuery] = useState("");

  const [experiments, setExperiments] = useState<BenchmarkExperimentSummary[]>([]);
  const [activeExperimentId, setActiveExperimentId] = useState<number | null>(null);
  const [experimentDetail, setExperimentDetail] = useState<BenchmarkExperimentDetail | null>(null);
  const [expFormOpen, setExpFormOpen] = useState(false);
  const [expCode, setExpCode] = useState("");
  const [expTitle, setExpTitle] = useState("");
  const [expDescription, setExpDescription] = useState("");
  const [expArms, setExpArms] = useState<Array<{ strategy: BenchmarkStrategy; model_key: string }>>(
    [
      { strategy: "naive_chat", model_key: "" },
      { strategy: "full_pipeline", model_key: "" },
    ],
  );
  const [expProjectCode, setExpProjectCode] = useState("");
  const [expLimit, setExpLimit] = useState(3);
  const [expSubmitting, setExpSubmitting] = useState(false);
  const experimentPollerRef = useRef<number | null>(null);
  const [runs, setRuns] = useState<BenchmarkRunSummary[]>([]);
  const [leaderboard, setLeaderboard] = useState<BenchmarkLeaderboardItem[]>([]);
  const [activeRunId, setActiveRunId] = useState<number | null>(null);
  const [runDetail, setRunDetail] = useState<BenchmarkRunDetail | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [loading, setLoading] = useState(false);

  const pollerRef = useRef<number | null>(null);

  const fetchProjects = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/benchmark/projects`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const items: BenchmarkProject[] = data.items ?? [];
      setProjects(items);
      setSummary(data.summary ?? null);
      setSelectedProjectId((current) => current ?? items[0]?.id ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  const fetchBugs = useCallback(
    async (projectId: number) => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/benchmark/projects/${projectId}/bugs`, {
          credentials: "include",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const items: BenchmarkBug[] = data.items ?? [];
        setBugs(items);
        setSelectedBugId((current) => {
          if (current && items.some((item) => item.id === current)) return current;
          return items[0]?.id ?? null;
        });
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    },
    [apiBaseUrl],
  );

  const fetchRuns = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/api/benchmark/runs?limit=100`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setRuns((data.items ?? []) as BenchmarkRunSummary[]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  const fetchLeaderboard = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/benchmark/leaderboard`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setLeaderboard((data.items ?? []) as BenchmarkLeaderboardItem[]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  const fetchRunDetail = useCallback(
    async (runId: number) => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/benchmark/runs/${runId}`, {
          credentials: "include",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const detail = (await response.json()) as BenchmarkRunDetail;
        setRunDetail(detail);
        return detail;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
        return null;
      }
    },
    [apiBaseUrl],
  );

  const fetchExperiments = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/benchmark/experiments?limit=50`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setExperiments((data.items ?? []) as BenchmarkExperimentSummary[]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  const fetchExperimentDetail = useCallback(
    async (experimentId: number) => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/benchmark/experiments/${experimentId}`,
          { credentials: "include" },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const detail = (await response.json()) as BenchmarkExperimentDetail;
        setExperimentDetail(detail);
        return detail;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
        return null;
      }
    },
    [apiBaseUrl],
  );

  useEffect(() => {
    void fetchProjects();
    void fetchLeaderboard();
  }, [fetchProjects, fetchLeaderboard]);

  useEffect(() => {
    if (page === "experiments") {
      void fetchExperiments();
    }
  }, [page, fetchExperiments]);

  useEffect(() => {
    if (activeExperimentId) {
      void fetchExperimentDetail(activeExperimentId);
    } else {
      setExperimentDetail(null);
    }
  }, [activeExperimentId, fetchExperimentDetail]);

  useEffect(() => {
    if (experimentPollerRef.current) {
      window.clearInterval(experimentPollerRef.current);
      experimentPollerRef.current = null;
    }
    if (page !== "experiments") return;
    const anyRunning = experiments.some(
      (item) => item.status === "queued" || item.status === "running",
    );
    if (!anyRunning) return;
    experimentPollerRef.current = window.setInterval(() => {
      void fetchExperiments();
      if (activeExperimentId) void fetchExperimentDetail(activeExperimentId);
    }, 4000);
    return () => {
      if (experimentPollerRef.current) {
        window.clearInterval(experimentPollerRef.current);
        experimentPollerRef.current = null;
      }
    };
  }, [page, experiments, fetchExperiments, fetchExperimentDetail, activeExperimentId]);

  useEffect(() => {
    if (selectedProjectId) {
      void fetchBugs(selectedProjectId);
    } else {
      setBugs([]);
    }
  }, [selectedProjectId, fetchBugs]);

  useEffect(() => {
    if (page === "runs" || page === "projects") {
      void fetchRuns();
    }
  }, [page, fetchRuns]);

  useEffect(() => {
    if (activeRunId) {
      void fetchRunDetail(activeRunId);
    }
  }, [activeRunId, fetchRunDetail]);

  useEffect(() => {
    if (pollerRef.current) {
      window.clearInterval(pollerRef.current);
      pollerRef.current = null;
    }
    const running = runs.some(
      (item) =>
        item.run_status === "queued" ||
        item.run_status === "running" ||
        item.stage === "checkout" ||
        item.stage === "inspect",
    );
    if (!running) return;
    pollerRef.current = window.setInterval(() => {
      void fetchRuns();
      if (activeRunId) void fetchRunDetail(activeRunId);
    }, 3000);
    return () => {
      if (pollerRef.current) {
        window.clearInterval(pollerRef.current);
        pollerRef.current = null;
      }
    };
  }, [runs, fetchRuns, fetchRunDetail, activeRunId]);

  const selectedBug = useMemo(
    () => bugs.find((bug) => bug.id === selectedBugId) ?? null,
    [bugs, selectedBugId],
  );

  const enabledProjects = useMemo(
    () => projects.filter((project) => project.is_active && project.bug_count > 0),
    [projects],
  );

  useEffect(() => {
    setBugQuery("");
  }, [selectedProjectId]);

  const filteredBugs = useMemo(() => {
    const q = bugQuery.trim().toLowerCase();
    if (!q) return bugs;
    return bugs.filter((bug) => {
      return (
        bug.bug_key.toLowerCase().includes(q) ||
        (bug.title ?? "").toLowerCase().includes(q) ||
        (bug.description ?? "").toLowerCase().includes(q) ||
        (bug.defects4j_project ?? "").toLowerCase().includes(q) ||
        String(bug.defects4j_bug_id ?? "").includes(q)
      );
    });
  }, [bugs, bugQuery]);

  const bugsToRender = useMemo(() => filteredBugs.slice(0, 120), [filteredBugs]);

  async function handleStartRun() {
    if (!selectedBugId) {
      setError(copy.benchmarkSelectBug);
      return;
    }
    if (!model) return;
    setSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/benchmark/runs`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bug_id: selectedBugId,
          model_key: model,
          run_mode: runMode,
          ...(runMode === "full_repair" ? { strategy } : {}),
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      const payload = await response.json();
      setActiveRunId(payload.run_id);
      onPageChange("runs");
      await fetchRuns();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubmitExperiment() {
    if (!expCode.trim()) {
      setError(copy.benchmarkExperimentCode);
      return;
    }
    const validArms = expArms.filter((arm) => arm.model_key.trim());
    if (validArms.length === 0) {
      setError(copy.benchmarkExperimentArms);
      return;
    }
    if (!expProjectCode.trim()) {
      setError(copy.benchmarkExperimentBugsFromProject);
      return;
    }
    setExpSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/benchmark/experiments`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_code: expCode.trim(),
          title: expTitle.trim() || expCode.trim(),
          description: expDescription.trim() || null,
          arms: validArms,
          project_code: expProjectCode.trim(),
          limit: Math.max(1, expLimit),
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      const payload = await response.json();
      setActiveExperimentId(payload.experiment_id);
      setExpFormOpen(false);
      await fetchExperiments();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setExpSubmitting(false);
    }
  }

  async function handleExportPdf(runId: number) {
    const response = await fetch(`${apiBaseUrl}/api/pdf/benchmark/${runId}`, {
      credentials: "include",
    });
    if (!response.ok) return;
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `benchmark-run-${runId}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 ${workspaceMainClass}`}
    >
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.benchmarkTitle}
          </div>
          <div className="hidden text-xs text-slate-500 dark:text-white/45 sm:block">
            {copy.benchmarkHint}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {(["projects", "runs", "experiments", "leaderboard"] as BenchmarkPage[]).map((tab) => (
            <button
              key={tab}
              onClick={() => onPageChange(tab)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                page === tab
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                  : "bg-black/[0.04] text-slate-600 hover:bg-black/[0.07] dark:bg-white/[0.05] dark:text-white/70 dark:hover:bg-white/[0.09]"
              }`}
            >
              {tab === "projects"
                ? copy.benchmarkProjects
                : tab === "runs"
                  ? copy.benchmarkRuns
                  : tab === "experiments"
                    ? copy.benchmarkExperiments
                    : copy.benchmarkLeaderboard}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <div className="mt-2 rounded-2xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto app-fade-in">
        {page === "projects" ? (
          <div className="grid gap-2 lg:grid-cols-[0.9fr,1.1fr]">
            <AdminSurface>
              <AdminSectionTitle
                title={copy.benchmarkProjects}
                actions={
                  summary ? (
                    <span className="text-xs text-slate-500 dark:text-white/45">
                      {summary.total_projects} · {summary.total_bugs}{" "}
                      {copy.benchmarkBugs.toLowerCase()} · {summary.completed_runs}/{summary.total_runs}{" "}
                      runs
                    </span>
                  ) : null
                }
              />
              {enabledProjects.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.benchmarkNoProjects} />
                </div>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {enabledProjects.map((project) => (
                    <button
                      key={project.id}
                      onClick={() => setSelectedProjectId(project.id)}
                      className={`flex w-full items-start justify-between gap-3 rounded-[16px] border px-3 py-2.5 text-left transition ${
                        selectedProjectId === project.id
                          ? "border-slate-900 bg-slate-900 text-white dark:border-white dark:bg-white dark:text-slate-950"
                          : "border-black/5 bg-black/[0.02] text-slate-700 hover:bg-black/[0.05] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold">{project.display_name}</div>
                        <div
                          className={`truncate text-xs ${
                            selectedProjectId === project.id
                              ? "text-white/70 dark:text-slate-950/70"
                              : "text-slate-500 dark:text-white/45"
                          }`}
                        >
                          {project.description ?? project.project_code}
                        </div>
                      </div>
                      <span className="shrink-0 rounded-full bg-black/10 px-2 py-0.5 text-[11px] font-medium dark:bg-white/10">
                        {project.bug_count}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </AdminSurface>

            <AdminSurface>
              <AdminSectionTitle
                title={copy.benchmarkBugs}
                hint={copy.benchmarkSelectBug}
                actions={
                  <span className="text-xs text-slate-500 dark:text-white/45">
                    {bugQuery
                      ? `${filteredBugs.length} / ${bugs.length}`
                      : `${bugs.length}`}
                  </span>
                }
              />
              {bugs.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.benchmarkNoProjects} />
                </div>
              ) : (
                <>
                  <input
                    type="search"
                    value={bugQuery}
                    onChange={(event) => setBugQuery(event.target.value)}
                    placeholder={copy.benchmarkBugSearchPlaceholder}
                    className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm placeholder:text-slate-400 dark:border-white/10 dark:bg-slate-900 dark:placeholder:text-white/30"
                  />
                  {filteredBugs.length === 0 ? (
                    <div className="mt-2">
                      <AdminEmptyState message={copy.benchmarkNoMatchingBugs} />
                    </div>
                  ) : (
                    <div className="mt-2 grid gap-1.5 sm:grid-cols-2">
                      {bugsToRender.map((bug) => (
                        <button
                          key={bug.id}
                          onClick={() => setSelectedBugId(bug.id)}
                          className={`rounded-2xl border px-3 py-2 text-left transition ${
                            selectedBugId === bug.id
                              ? "border-sky-500/70 bg-sky-500/10 dark:border-sky-400/50"
                              : "border-black/5 bg-black/[0.02] hover:bg-black/[0.05] dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 text-xs font-semibold">
                            <span className="truncate">{bug.bug_key}</span>
                            <AdminBadge label={bug.severity} tone={severityTone(bug.severity)} />
                          </div>
                          <div className="mt-1 line-clamp-2 text-xs text-slate-500 dark:text-white/45">
                            {bug.title || bug.description || bug.bug_key}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                  {filteredBugs.length > bugsToRender.length ? (
                    <div className="mt-2 rounded-xl bg-black/[0.03] px-3 py-1.5 text-[11px] text-slate-500 dark:bg-white/[0.04] dark:text-white/50">
                      {copy.benchmarkMoreResultsHint
                        .replace("{shown}", String(bugsToRender.length))
                        .replace("{total}", String(filteredBugs.length))}
                    </div>
                  ) : null}
                </>
              )}

              <div className="mt-3 space-y-2 rounded-2xl border border-black/5 bg-black/[0.02] p-2.5 dark:border-white/10 dark:bg-white/[0.03]">
                <div
                  className={`grid gap-2 ${
                    runMode === "full_repair" ? "sm:grid-cols-4" : "sm:grid-cols-3"
                  }`}
                >
                  <label className="flex flex-col gap-1 text-xs">
                    <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                      {copy.model}
                    </span>
                    <select
                      value={model}
                      onChange={(event) => onModelChange(event.target.value)}
                      className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                    >
                      {modelOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1 text-xs">
                    <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                      {copy.benchmarkRunMode}
                    </span>
                    <select
                      value={runMode}
                      onChange={(event) => setRunMode(event.target.value as BenchmarkRunMode)}
                      className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                    >
                      <option value="inspect_only">{copy.benchmarkInspectOnly}</option>
                      <option value="full_repair">{copy.benchmarkFullRepair}</option>
                    </select>
                  </label>
                  {runMode === "full_repair" ? (
                    <label className="flex flex-col gap-1 text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkStrategy}
                      </span>
                      <select
                        value={strategy}
                        onChange={(event) => setStrategy(event.target.value as BenchmarkStrategy)}
                        className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                      >
                        <option value="full_pipeline">{copy.benchmarkStrategyFullPipeline}</option>
                        <option value="naive_chat">{copy.benchmarkStrategyNaiveChat}</option>
                      </select>
                    </label>
                  ) : null}
                  <div className="flex items-end">
                    <button
                      onClick={handleStartRun}
                      disabled={submitting || !selectedBugId || !model}
                      className={`w-full rounded-xl px-3 py-2 text-sm font-semibold transition ${
                        submitting || !selectedBugId || !model
                          ? "cursor-not-allowed bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-white/50"
                          : "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      }`}
                    >
                      {submitting ? copy.benchmarkRunning : copy.benchmarkStart}
                    </button>
                  </div>
                </div>
                {runMode === "full_repair" ? (
                  <div className="rounded-xl bg-sky-500/5 px-2.5 py-1.5 text-[11px] text-slate-600 dark:bg-sky-400/10 dark:text-white/65">
                    {copy.benchmarkFullRepairHint}
                  </div>
                ) : null}
                {selectedBug ? (
                  <div className="rounded-xl bg-black/[0.03] px-2.5 py-2 text-xs text-slate-600 dark:bg-white/[0.04] dark:text-white/60">
                    <div className="font-medium text-slate-800 dark:text-white/80">
                      {selectedBug.bug_key} · {selectedBug.title}
                    </div>
                    {selectedBug.description ? (
                      <div className="mt-0.5">{selectedBug.description}</div>
                    ) : null}
                    {selectedBug.defects4j_project ? (
                      <div className="mt-1 font-mono text-[11px] text-slate-500 dark:text-white/45">
                        defects4j · {selectedBug.defects4j_project}-{selectedBug.defects4j_bug_id}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </AdminSurface>
          </div>
        ) : null}

        {page === "runs" ? (
          <div className="grid gap-2 lg:grid-cols-[1fr,1.2fr]">
            <AdminSurface>
              <AdminSectionTitle
                title={copy.benchmarkRuns}
                actions={
                  <button
                    onClick={fetchRuns}
                    className="rounded-full bg-black/[0.04] px-3 py-1 text-xs dark:bg-white/[0.06]"
                  >
                    {copy.adminRefresh}
                  </button>
                }
              />
              {loading && runs.length === 0 ? (
                <div className="mt-2 text-sm text-slate-500 dark:text-white/40">
                  {copy.adminRefresh}…
                </div>
              ) : runs.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.benchmarkNoRuns} />
                </div>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {runs.map((run) => (
                    <button
                      key={run.id}
                      onClick={() => setActiveRunId(run.id)}
                      className={`flex w-full items-start justify-between gap-2 rounded-2xl border px-3 py-2 text-left transition ${
                        activeRunId === run.id
                          ? "border-slate-900 bg-slate-900 text-white dark:border-white dark:bg-white dark:text-slate-950"
                          : "border-black/5 bg-black/[0.02] hover:bg-black/[0.05] dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-xs font-semibold">
                          <span className="truncate">
                            {(run.project_code ?? `#${run.project_id}`)} / {run.bug_key ?? `#${run.bug_id}`}
                          </span>
                          <AdminBadge label={run.run_status} tone={toStatusTone(run.run_status)} />
                          {run.is_correct ? (
                            <AdminBadge label={copy.benchmarkCorrect} tone="emerald" />
                          ) : run.is_plausible ? (
                            <AdminBadge label={copy.benchmarkPlausible} tone="amber" />
                          ) : null}
                        </div>
                        <div
                          className={`mt-0.5 truncate text-xs ${
                            activeRunId === run.id
                              ? "text-white/70 dark:text-slate-950/70"
                              : "text-slate-500 dark:text-white/45"
                          }`}
                        >
                          {run.model_key} · {run.run_mode}
                          {run.strategy && run.run_mode === "full_repair" ? ` · ${run.strategy}` : ""} ·{" "}
                          {formatAdminDurationMs(run.duration_ms)} · {run.pass_count}/{run.total_tests || 0}
                        </div>
                      </div>
                      <div className="shrink-0 text-right text-[11px] opacity-70">
                        <div>{run.stage ?? "—"}</div>
                        <div>{formatAdminDateTime(run.started_at)}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </AdminSurface>

            <AdminSurface>
              {runDetail ? (
                <>
                  <AdminSectionTitle
                    title={`${runDetail.project_code ?? "—"} / ${runDetail.bug_key ?? "—"}`}
                    actions={
                      <>
                        <AdminBadge label={runDetail.run_status} tone={toStatusTone(runDetail.run_status)} />
                        {runDetail.is_correct ? (
                          <AdminBadge label={copy.benchmarkCorrect} tone="emerald" />
                        ) : runDetail.is_plausible ? (
                          <AdminBadge label={copy.benchmarkPlausible} tone="amber" />
                        ) : null}
                        {runDetail.strategy && runDetail.run_mode === "full_repair" ? (
                          <AdminBadge label={runDetail.strategy} tone="slate" />
                        ) : null}
                        <button
                          onClick={() => handleExportPdf(runDetail.id)}
                          className="rounded-full bg-black/[0.05] px-3 py-1 text-xs dark:bg-white/[0.07]"
                        >
                          {copy.pdfExport}
                        </button>
                      </>
                    }
                  />
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <AdminMetricCard
                      label={copy.benchmarkStage}
                      value={runDetail.stage ?? "-"}
                    />
                    <AdminMetricCard
                      label={copy.benchmarkDuration}
                      value={formatAdminDurationMs(runDetail.duration_ms)}
                    />
                    <AdminMetricCard
                      label="Pass"
                      value={`${runDetail.pass_count}/${runDetail.total_tests || 0}`}
                      tone={runDetail.fail_count === 0 && runDetail.total_tests > 0 ? "emerald" : "slate"}
                    />
                    <AdminMetricCard
                      label="Fail"
                      value={formatAdminNumber(runDetail.fail_count)}
                      tone={runDetail.fail_count > 0 ? "rose" : "slate"}
                    />
                  </div>
                  {runDetail.run_mode === "full_repair" ? (
                    <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                      <AdminMetricCard
                        label={copy.benchmarkTokens}
                        value={formatAdminNumber(runDetail.total_tokens ?? 0)}
                      />
                      <AdminMetricCard
                        label={copy.benchmarkLLMRounds}
                        value={formatAdminNumber(runDetail.llm_rounds ?? 0)}
                      />
                      <AdminMetricCard
                        label={copy.benchmarkPatchLines}
                        value={`+${runDetail.patch_lines_added ?? 0} / -${
                          runDetail.patch_lines_removed ?? 0
                        }`}
                      />
                      <AdminMetricCard
                        label={copy.benchmarkFailedBeforeAfter}
                        value={`${runDetail.failed_tests_before ?? 0} → ${
                          runDetail.failed_tests_after ?? 0
                        }`}
                        tone={
                          (runDetail.failed_tests_after ?? 0) === 0 &&
                          (runDetail.failed_tests_before ?? 0) > 0
                            ? "emerald"
                            : "slate"
                        }
                      />
                    </div>
                  ) : null}
                  {runDetail.error_message ? (
                    <div className="mt-2 rounded-2xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-xs text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
                      {runDetail.error_message}
                    </div>
                  ) : null}
                  <div className="mt-3 space-y-2">
                    <AdminCodeBlock
                      title={copy.benchmarkCheckoutLog}
                      content={safeLog(runDetail.report, "checkout")}
                    />
                    <AdminCodeBlock
                      title={copy.benchmarkCompileLog}
                      content={safeLog(runDetail.report, "compile")}
                    />
                    <AdminCodeBlock
                      title={copy.benchmarkTestLog}
                      content={safeLog(runDetail.report, "tests")}
                    />
                    {runDetail.patch_diff ? (
                      <AdminCodeBlock title={copy.benchmarkPatch} content={runDetail.patch_diff} />
                    ) : null}
                  </div>
                </>
              ) : (
                <AdminEmptyState message={copy.adminNoSelection} />
              )}
            </AdminSurface>
          </div>
        ) : null}

        {page === "leaderboard" ? (
          <AdminSurface>
            <AdminSectionTitle
              title={copy.benchmarkLeaderboard}
              actions={
                <button
                  onClick={fetchLeaderboard}
                  className="rounded-full bg-black/[0.04] px-3 py-1 text-xs dark:bg-white/[0.06]"
                >
                  {copy.adminRefresh}
                </button>
              }
            />
            {leaderboard.length === 0 ? (
              <div className="mt-2">
                <AdminEmptyState message={copy.benchmarkNoLeaderboard} />
              </div>
            ) : (
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                    <tr>
                      <th className="py-1.5 pr-2">#</th>
                      <th className="py-1.5 pr-2">Model</th>
                      <th className="py-1.5 pr-2">Project</th>
                      <th className="py-1.5 pr-2 text-right">Samples</th>
                      <th className="py-1.5 pr-2 text-right">Success</th>
                      <th className="py-1.5 pr-2 text-right">Pass rate</th>
                      <th className="py-1.5 pr-2 text-right">Avg</th>
                      <th className="py-1.5 pr-2 text-right">Last</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((item, index) => (
                      <tr
                        key={`${item.project_id}-${item.model_key}`}
                        className="border-t border-black/5 text-xs text-slate-700 transition hover:bg-black/[0.04] dark:border-white/10 dark:text-white/75 dark:hover:bg-white/[0.06]"
                      >
                        <td className="py-1.5 pr-2 font-semibold">{index + 1}</td>
                        <td className="py-1.5 pr-2 font-medium">{item.model_key}</td>
                        <td className="py-1.5 pr-2">
                          {item.project_display_name ?? item.project_code ?? `#${item.project_id}`}
                        </td>
                        <td className="py-1.5 pr-2 text-right">
                          {formatAdminNumber(item.sample_count)}
                        </td>
                        <td className="py-1.5 pr-2 text-right">
                          {formatAdminNumber(item.success_count)}
                        </td>
                        <td className="py-1.5 pr-2 text-right">
                          {(item.pass_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-1.5 pr-2 text-right">
                          {formatAdminDurationMs(item.avg_duration_ms)}
                        </td>
                        <td className="py-1.5 pr-2 text-right text-[11px] text-slate-500 dark:text-white/45">
                          {formatAdminDateTime(item.last_run_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </AdminSurface>
        ) : null}

        {page === "experiments" ? (
          <div className="grid gap-2 lg:grid-cols-[0.9fr,1.6fr]">
            <AdminSurface>
              <AdminSectionTitle
                title={copy.benchmarkExperiments}
                hint={copy.benchmarkExperimentsHint}
                actions={
                  <button
                    onClick={() => setExpFormOpen((current) => !current)}
                    className="rounded-full bg-sky-500 px-3 py-1 text-xs font-semibold text-white hover:bg-sky-600"
                  >
                    {expFormOpen ? "×" : copy.benchmarkExperimentNew}
                  </button>
                }
              />
              {expFormOpen ? (
                <div className="mt-2 space-y-2 rounded-2xl border border-sky-500/20 bg-sky-500/5 p-3 dark:border-sky-400/20 dark:bg-sky-400/10">
                  <div className="grid gap-2 sm:grid-cols-2">
                    <label className="flex flex-col gap-1 text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentCode}
                      </span>
                      <input
                        value={expCode}
                        onChange={(event) => setExpCode(event.target.value)}
                        placeholder="weak-vs-strong-v2"
                        className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentTitle}
                      </span>
                      <input
                        value={expTitle}
                        onChange={(event) => setExpTitle(event.target.value)}
                        className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                      />
                    </label>
                  </div>
                  <label className="flex flex-col gap-1 text-xs">
                    <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                      {copy.benchmarkExperimentDescription}
                    </span>
                    <textarea
                      value={expDescription}
                      onChange={(event) => setExpDescription(event.target.value)}
                      rows={2}
                      className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                    />
                  </label>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentArms}
                      </span>
                      <button
                        onClick={() =>
                          setExpArms((arms) => [
                            ...arms,
                            { strategy: "full_pipeline", model_key: "" },
                          ])
                        }
                        className="rounded-full bg-black/[0.05] px-2.5 py-0.5 text-[11px] font-semibold dark:bg-white/[0.07]"
                      >
                        + {copy.benchmarkExperimentAddArm}
                      </button>
                    </div>
                    <div className="space-y-1.5">
                      {expArms.map((arm, idx) => (
                        <div key={idx} className="grid grid-cols-[auto,1fr,auto] gap-1.5">
                          <select
                            value={arm.strategy}
                            onChange={(event) =>
                              setExpArms((arms) =>
                                arms.map((a, i) =>
                                  i === idx
                                    ? { ...a, strategy: event.target.value as BenchmarkStrategy }
                                    : a,
                                ),
                              )
                            }
                            className="rounded-lg border border-black/10 bg-white px-2 py-1 text-xs dark:border-white/10 dark:bg-slate-900"
                          >
                            <option value="full_pipeline">full_pipeline</option>
                            <option value="naive_chat">naive_chat</option>
                          </select>
                          <select
                            value={arm.model_key}
                            onChange={(event) =>
                              setExpArms((arms) =>
                                arms.map((a, i) =>
                                  i === idx ? { ...a, model_key: event.target.value } : a,
                                ),
                              )
                            }
                            className="rounded-lg border border-black/10 bg-white px-2 py-1 text-xs dark:border-white/10 dark:bg-slate-900"
                          >
                            <option value="">— {copy.model} —</option>
                            {modelOptions.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() =>
                              setExpArms((arms) => arms.filter((_, i) => i !== idx))
                            }
                            className="rounded-lg bg-rose-500/10 px-2 text-xs text-rose-600 hover:bg-rose-500/20 dark:text-rose-300"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-[1fr,auto,auto]">
                    <label className="flex flex-col gap-1 text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentBugsFromProject}
                      </span>
                      <select
                        value={expProjectCode}
                        onChange={(event) => setExpProjectCode(event.target.value)}
                        className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                      >
                        <option value="">— {copy.benchmarkProjects} —</option>
                        {enabledProjects.map((project) => (
                          <option key={project.id} value={project.project_code}>
                            {project.display_name} ({project.bug_count})
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="flex flex-col gap-1 text-xs">
                      <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentLimit}
                      </span>
                      <input
                        type="number"
                        min={1}
                        max={50}
                        value={expLimit}
                        onChange={(event) => setExpLimit(parseInt(event.target.value, 10) || 1)}
                        className="w-24 rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                      />
                    </label>
                    <div className="flex items-end">
                      <button
                        onClick={handleSubmitExperiment}
                        disabled={expSubmitting}
                        className={`rounded-xl px-3 py-1.5 text-sm font-semibold transition ${
                          expSubmitting
                            ? "cursor-not-allowed bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-white/50"
                            : "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                        }`}
                      >
                        {expSubmitting ? copy.benchmarkRunning : copy.benchmarkExperimentSubmit}
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
              {experiments.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.benchmarkExperimentEmpty} />
                </div>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {experiments.map((exp) => (
                    <button
                      key={exp.id}
                      onClick={() => setActiveExperimentId(exp.id)}
                      className={`flex w-full items-start justify-between gap-2 rounded-2xl border px-3 py-2 text-left transition ${
                        activeExperimentId === exp.id
                          ? "border-slate-900 bg-slate-900 text-white dark:border-white dark:bg-white dark:text-slate-950"
                          : "border-black/5 bg-black/[0.02] hover:bg-black/[0.05] dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-xs font-semibold">
                          <span className="truncate">{exp.experiment_code}</span>
                          <AdminBadge
                            label={exp.status}
                            tone={
                              exp.status === "completed"
                                ? "emerald"
                                : exp.status === "running" || exp.status === "queued"
                                  ? "amber"
                                  : exp.status === "failed"
                                    ? "rose"
                                    : "slate"
                            }
                          />
                        </div>
                        <div
                          className={`mt-0.5 truncate text-xs ${
                            activeExperimentId === exp.id
                              ? "text-white/70 dark:text-slate-950/70"
                              : "text-slate-500 dark:text-white/45"
                          }`}
                        >
                          {exp.title || exp.experiment_code} · {exp.completed_runs}/
                          {exp.total_runs} runs
                        </div>
                      </div>
                      <div className="shrink-0 text-[11px] opacity-70">
                        {formatAdminDateTime(exp.created_at)}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </AdminSurface>

            <AdminSurface>
              {experimentDetail ? (
                <>
                  <AdminSectionTitle
                    title={
                      experimentDetail.experiment.title ||
                      experimentDetail.experiment.experiment_code
                    }
                    actions={
                      <AdminBadge
                        label={experimentDetail.experiment.status}
                        tone={
                          experimentDetail.experiment.status === "completed"
                            ? "emerald"
                            : experimentDetail.experiment.status === "running" ||
                                experimentDetail.experiment.status === "queued"
                              ? "amber"
                              : experimentDetail.experiment.status === "failed"
                                ? "rose"
                                : "slate"
                        }
                      />
                    }
                  />
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <AdminMetricCard
                      label={copy.benchmarkExperimentRuns}
                      value={`${experimentDetail.experiment.completed_runs}/${experimentDetail.experiment.total_runs}`}
                    />
                    <AdminMetricCard
                      label={copy.benchmarkExperimentPlaus}
                      value={formatAdminNumber(
                        experimentDetail.arms.reduce((sum, a) => sum + a.plausible, 0),
                      )}
                      tone="amber"
                    />
                    <AdminMetricCard
                      label={copy.benchmarkExperimentCorr}
                      value={formatAdminNumber(
                        experimentDetail.arms.reduce((sum, a) => sum + a.correct, 0),
                      )}
                      tone="emerald"
                    />
                    <AdminMetricCard
                      label={copy.benchmarkExperimentAvgTokens}
                      value={formatAdminNumber(
                        experimentDetail.arms.reduce((sum, a) => sum + a.avg_tokens, 0) /
                          Math.max(1, experimentDetail.arms.length),
                      )}
                    />
                  </div>
                  {experimentDetail.experiment.description ? (
                    <div className="mt-2 rounded-xl bg-black/[0.03] px-3 py-2 text-xs text-slate-600 dark:bg-white/[0.04] dark:text-white/60">
                      {experimentDetail.experiment.description}
                    </div>
                  ) : null}
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        <tr>
                          <th className="py-1.5 pr-2">Strategy</th>
                          <th className="py-1.5 pr-2">Model</th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentRuns}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentPlaus}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentCorr}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentPlausRate}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentCorrRate}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentAvgMs}
                          </th>
                          <th className="py-1.5 pr-2 text-right">
                            {copy.benchmarkExperimentAvgTokens}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {experimentDetail.arms.map((arm) => (
                          <tr
                            key={`${arm.strategy}-${arm.model_key}`}
                            className="border-t border-black/5 text-xs dark:border-white/10"
                          >
                            <td className="py-1.5 pr-2 font-semibold">{arm.strategy}</td>
                            <td className="py-1.5 pr-2">{arm.model_key}</td>
                            <td className="py-1.5 pr-2 text-right">
                              {arm.completed}/{arm.total}
                            </td>
                            <td className="py-1.5 pr-2 text-right">{arm.plausible}</td>
                            <td className="py-1.5 pr-2 text-right font-semibold text-emerald-600 dark:text-emerald-300">
                              {arm.correct}
                            </td>
                            <td className="py-1.5 pr-2 text-right">
                              {(arm.plausible_rate * 100).toFixed(1)}%
                            </td>
                            <td className="py-1.5 pr-2 text-right font-semibold">
                              {(arm.correct_rate * 100).toFixed(1)}%
                            </td>
                            <td className="py-1.5 pr-2 text-right">
                              {formatAdminDurationMs(arm.avg_duration_ms)}
                            </td>
                            <td className="py-1.5 pr-2 text-right">
                              {formatAdminNumber(arm.avg_tokens)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {experimentDetail.per_bug.length > 0 ? (
                    <div className="mt-3">
                      <div className="mb-1 text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        {copy.benchmarkExperimentPerBug}
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead className="text-[11px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                            <tr>
                              <th className="py-1 pr-2">Bug</th>
                              <th className="py-1 pr-2">
                                {copy.benchmarkExperimentArmColumn}
                              </th>
                              <th className="py-1 pr-2">{copy.benchmarkStatus}</th>
                              <th className="py-1 pr-2 text-right">Plaus</th>
                              <th className="py-1 pr-2 text-right">Corr</th>
                              <th className="py-1 pr-2 text-right">ms</th>
                              <th className="py-1 pr-2 text-right">tok</th>
                            </tr>
                          </thead>
                          <tbody>
                            {experimentDetail.per_bug.map((row, idx) => (
                              <tr
                                key={idx}
                                className="border-t border-black/5 dark:border-white/10"
                              >
                                <td className="py-1 pr-2 font-mono">{row.bug_key}</td>
                                <td className="py-1 pr-2">
                                  {row.strategy} · {row.model_key}
                                </td>
                                <td className="py-1 pr-2">
                                  <AdminBadge
                                    label={row.run_status}
                                    tone={toStatusTone(row.run_status)}
                                  />
                                </td>
                                <td className="py-1 pr-2 text-right">
                                  {row.is_plausible ? "✓" : "—"}
                                </td>
                                <td className="py-1 pr-2 text-right font-semibold text-emerald-600 dark:text-emerald-300">
                                  {row.is_correct ? "✓" : "—"}
                                </td>
                                <td className="py-1 pr-2 text-right">
                                  {formatAdminDurationMs(row.duration_ms)}
                                </td>
                                <td className="py-1 pr-2 text-right">
                                  {formatAdminNumber(row.total_tokens)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                </>
              ) : (
                <AdminEmptyState message={copy.benchmarkExperimentNoneSelected} />
              )}
            </AdminSurface>
          </div>
        ) : null}
      </div>
    </section>
  );
}
