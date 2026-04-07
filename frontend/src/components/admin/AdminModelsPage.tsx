import type { AppCopy } from "../../i18n";
import type { AdminModelUsageReport } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminDurationMs,
  formatAdminNumber,
} from "./AdminCommon";

interface AdminModelsPageProps {
  copy: AppCopy;
  days: number;
  modelUsage: AdminModelUsageReport | null;
  onDaysChange: (days: number) => void;
}

export function AdminModelsPage({
  copy,
  days,
  modelUsage,
  onDaysChange,
}: AdminModelsPageProps) {
  const items = modelUsage?.items ?? [];
  const totalRequests = items.reduce((sum, item) => sum + item.request_count, 0);
  const totalTokens = items.reduce((sum, item) => sum + item.total_tokens, 0);
  const averageLatency =
    totalRequests > 0
      ? items.reduce((sum, item) => sum + item.avg_latency_ms * item.request_count, 0) / totalRequests
      : 0;
  const dailySeries = modelUsage?.daily_series ?? [];
  const maxSeriesValue = dailySeries.reduce((current, item) => Math.max(current, item.total_tokens), 0);

  return (
    <div className="space-y-3">
      <AdminSurface>
        <AdminSectionTitle
          eyebrow={copy.adminModels}
          title={copy.adminModelsTitle}
          hint={copy.adminModelsHint}
          actions={
            <>
              {[7, 30, 90].map((value) => (
                <button
                  key={value}
                  onClick={() => onDaysChange(value)}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                    days === value
                      ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                      : "border border-black/10 text-slate-600 dark:border-white/10 dark:text-white/65"
                  }`}
                >
                  {value}d
                </button>
              ))}
            </>
          }
        />
      </AdminSurface>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminModels} value={formatAdminNumber(items.length)} tone="sky" />
        <AdminMetricCard label={copy.adminRequestCount} value={formatAdminNumber(totalRequests)} tone="emerald" />
        <AdminMetricCard label={copy.adminTokens} value={formatAdminNumber(totalTokens)} tone="amber" />
        <AdminMetricCard label={copy.adminLatency} value={formatAdminDurationMs(averageLatency)} tone="rose" />
      </div>

      <div className="grid gap-3 xl:grid-cols-[0.9fr_1.1fr]">
        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminModels}
            title={copy.adminDailyModelUsage}
            hint={`${days}d`}
          />
          <div className="mt-4 space-y-3">
            {dailySeries.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              dailySeries.map((item, index) => (
                <div key={`${item.day}-${item.model}-${index}`}>
                  <div className="mb-1 flex items-center justify-between gap-3 text-xs text-slate-500 dark:text-white/45">
                    <span className="truncate">{item.day} · {item.model}</span>
                    <span>{formatAdminNumber(item.total_tokens)} {copy.adminTokens}</span>
                  </div>
                  <div className="h-3 rounded-full bg-black/[0.05] dark:bg-white/[0.06]">
                    <div
                      className="h-full rounded-full bg-slate-900 dark:bg-white"
                      style={{
                        width: `${maxSeriesValue > 0 ? Math.max(8, (item.total_tokens / maxSeriesValue) * 100) : 8}%`,
                      }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminTopModels}
            title={copy.adminModels}
            hint={`${copy.adminFilterAll}: ${formatAdminNumber(items.length)}`}
          />
          <div className="mt-4 overflow-x-auto">
            {items.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              <table className="min-w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/35">
                  <tr>
                    <th className="px-3 py-3">{copy.model}</th>
                    <th className="px-3 py-3">{copy.adminRequestProvider}</th>
                    <th className="px-3 py-3">{copy.adminRequestCount}</th>
                    <th className="px-3 py-3">{copy.adminInputTokens}</th>
                    <th className="px-3 py-3">{copy.adminOutputTokens}</th>
                    <th className="px-3 py-3">{copy.adminTokens}</th>
                    <th className="px-3 py-3">{copy.adminLatency}</th>
                    <th className="px-3 py-3">{copy.adminLastLogin}</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr
                      key={`${item.provider}-${item.model}`}
                      className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                    >
                      <td className="px-3 py-3">
                        <div className="font-medium text-slate-900 dark:text-white">{item.model}</div>
                      </td>
                      <td className="px-3 py-3">
                        <AdminBadge label={item.provider} tone="slate" />
                      </td>
                      <td className="px-3 py-3">{formatAdminNumber(item.request_count)}</td>
                      <td className="px-3 py-3">{formatAdminNumber(item.input_tokens)}</td>
                      <td className="px-3 py-3">{formatAdminNumber(item.output_tokens)}</td>
                      <td className="px-3 py-3">{formatAdminNumber(item.total_tokens)}</td>
                      <td className="px-3 py-3">{formatAdminDurationMs(item.avg_latency_ms)}</td>
                      <td className="px-3 py-3">{formatAdminDateTime(item.last_used_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </AdminSurface>
      </div>
    </div>
  );
}
