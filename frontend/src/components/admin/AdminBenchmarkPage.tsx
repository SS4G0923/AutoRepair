import { useCallback, useEffect, useState } from "react";
import type { AppCopy } from "../../i18n";
import type { AdminBenchmarkRefreshResult, AdminBenchmarkRun } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminDurationMs,
  formatAdminNumber,
  toStatusTone,
} from "./AdminCommon";

interface AdminBenchmarkPageProps {
  apiBaseUrl: string;
  copy: AppCopy;
}

export function AdminBenchmarkPage({ apiBaseUrl, copy }: AdminBenchmarkPageProps) {
  const [runs, setRuns] = useState<AdminBenchmarkRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshResult, setRefreshResult] = useState<AdminBenchmarkRefreshResult | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchRuns = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/api/admin/benchmark/runs?limit=200`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setRuns((data.items ?? []) as AdminBenchmarkRun[]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void fetchRuns();
  }, [fetchRuns]);

  async function handleRefreshDefects4j() {
    setRefreshing(true);
    setError("");
    setRefreshResult(null);
    try {
      const response = await fetch(`${apiBaseUrl}/api/admin/benchmark/refresh-defects4j`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      const payload = await response.json();
      setRefreshResult(payload.summary as AdminBenchmarkRefreshResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <div className="space-y-2">
      <AdminSurface>
        <AdminSectionTitle
          title={copy.adminBenchmarkTitle}
          hint={copy.adminBenchmarkHint}
          actions={
            <button
              onClick={handleRefreshDefects4j}
              disabled={refreshing}
              className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                refreshing
                  ? "cursor-wait bg-slate-300 text-slate-600 dark:bg-white/20 dark:text-white/70"
                  : "bg-sky-500 text-white hover:bg-sky-600"
              }`}
            >
              {refreshing ? copy.adminRefreshDefects4jRunning : copy.adminRefreshDefects4j}
            </button>
          }
        />
        {error ? (
          <div className="mt-2 rounded-xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
            {error}
          </div>
        ) : null}
        {refreshResult ? (
          <div className="mt-2 space-y-1.5 rounded-2xl border border-emerald-500/20 bg-emerald-50 p-3 text-xs dark:border-emerald-400/20 dark:bg-emerald-500/10">
            <div className="font-semibold text-emerald-700 dark:text-emerald-200">
              {copy.adminRefreshDefects4jDone
                .replace("{total}", String(refreshResult.total_imported))
                .replace("{new_}", String(refreshResult.total_new))}
            </div>
            <div className="font-mono text-[11px] text-slate-600 dark:text-white/60">
              D4J_HOME: {refreshResult.d4j_home}
            </div>
            <div className="grid gap-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
              {refreshResult.projects.map((project) => (
                <div
                  key={project.project_code}
                  className="rounded-lg bg-black/[0.03] px-2 py-1 text-[11px] text-slate-700 dark:bg-white/[0.05] dark:text-white/75"
                >
                  <span className="font-semibold">{project.project_code}</span>
                  : {project.upserted}{" "}
                  {project.new > 0 ? (
                    <span className="text-emerald-600 dark:text-emerald-300">
                      (+{project.new})
                    </span>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </AdminSurface>

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
            <AdminEmptyState message={copy.adminBenchmarkNoRuns} />
          </div>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                <tr>
                  <th className="py-1.5 pr-2">#</th>
                  <th className="py-1.5 pr-2">{copy.adminBenchmarkColUser}</th>
                  <th className="py-1.5 pr-2">{copy.adminBenchmarkColRun}</th>
                  <th className="py-1.5 pr-2">{copy.adminBenchmarkColStrategy}</th>
                  <th className="py-1.5 pr-2">Model</th>
                  <th className="py-1.5 pr-2">{copy.adminBenchmarkColStatus}</th>
                  <th className="py-1.5 pr-2 text-right">
                    {copy.adminBenchmarkColTokens}
                  </th>
                  <th className="py-1.5 pr-2 text-right">
                    {copy.benchmarkDuration}
                  </th>
                  <th className="py-1.5 pr-2 text-right">
                    {copy.adminBenchmarkColWhen}
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.id}
                    className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                  >
                    <td className="py-1.5 pr-2 font-mono">{run.id}</td>
                    <td className="py-1.5 pr-2 truncate">
                      {run.user_display_name || run.user_email || `#${run.user_id}`}
                    </td>
                    <td className="py-1.5 pr-2">
                      <span className="font-mono">
                        {run.project_code ?? "—"}/{run.bug_key ?? run.bug_id}
                      </span>
                    </td>
                    <td className="py-1.5 pr-2">
                      {run.strategy ? (
                        <AdminBadge label={String(run.strategy)} tone="slate" />
                      ) : (
                        <span className="text-slate-400 dark:text-white/30">{run.run_mode}</span>
                      )}
                    </td>
                    <td className="py-1.5 pr-2 truncate">{run.model_key}</td>
                    <td className="py-1.5 pr-2">
                      <div className="flex items-center gap-1">
                        <AdminBadge
                          label={run.run_status}
                          tone={toStatusTone(run.run_status)}
                        />
                        {run.is_correct ? (
                          <AdminBadge label={copy.benchmarkCorrect} tone="emerald" />
                        ) : run.is_plausible ? (
                          <AdminBadge label={copy.benchmarkPlausible} tone="amber" />
                        ) : null}
                      </div>
                    </td>
                    <td className="py-1.5 pr-2 text-right">
                      {formatAdminNumber(run.total_tokens ?? 0)}
                    </td>
                    <td className="py-1.5 pr-2 text-right">
                      {formatAdminDurationMs(run.duration_ms)}
                    </td>
                    <td className="py-1.5 pr-2 text-right text-[11px] text-slate-500 dark:text-white/45">
                      {formatAdminDateTime(run.started_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </AdminSurface>
    </div>
  );
}
