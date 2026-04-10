import type { AppCopy } from "../../i18n";
import type { AdminLoginEventList } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminNumber,
  toStatusTone,
} from "./AdminCommon";

interface AdminActivityPageProps {
  activityPage: number;
  copy: AppCopy;
  loginEvents: AdminLoginEventList | null;
  onPageChange: (page: number) => void;
}

export function AdminActivityPage({
  activityPage,
  copy,
  loginEvents,
  onPageChange,
}: AdminActivityPageProps) {
  const items = loginEvents?.items ?? [];
  const successCount = items.filter((item) => item.login_status === "success").length;
  const failedCount = items.filter((item) => item.login_status !== "success").length;
  const distinctUsers = new Set(
    items.map((item) => item.user_email || item.email_attempt).filter(Boolean),
  ).size;
  const canGoPrevious = activityPage > 1;
  const canGoNext = Boolean(loginEvents && loginEvents.page * loginEvents.page_size < loginEvents.total);

  return (
    <div className="space-y-2">
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminRecentLoginAttempts} value={formatAdminNumber(items.length)} tone="sky" />
        <AdminMetricCard label={copy.adminFilterCompleted} value={formatAdminNumber(successCount)} tone="emerald" />
        <AdminMetricCard label={copy.adminFilterFailed} value={formatAdminNumber(failedCount)} tone="rose" />
        <AdminMetricCard label={copy.adminUsers} value={formatAdminNumber(distinctUsers)} tone="amber" />
      </div>

      <AdminSurface>
        <div className="overflow-x-auto">
          {items.length === 0 ? (
            <AdminEmptyState message={copy.adminNoData} />
          ) : (
            <table className="min-w-full text-left text-xs">
              <thead className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                <tr>
                  <th className="px-2 py-2">{copy.adminUserLabel}</th>
                  <th className="px-2 py-2">{copy.adminLoginMethod}</th>
                  <th className="px-2 py-2">{copy.adminLoginStatus}</th>
                  <th className="px-2 py-2">{copy.adminFailureReason}</th>
                  <th className="px-2 py-2">{copy.adminIpAddress}</th>
                  <th className="px-2 py-2">{copy.adminUserAgent}</th>
                  <th className="px-2 py-2">{copy.adminCreatedAt}</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.id}
                    className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                  >
                    <td className="px-2 py-2">
                      <div className="min-w-[11rem]">
                        <div className="font-medium text-slate-900 dark:text-white">
                          {item.user_display_name || item.user_email || item.email_attempt || "-"}
                        </div>
                        <div className="text-[11px] text-slate-500 dark:text-white/45">
                          {item.user_email || item.email_attempt || "-"}
                        </div>
                      </div>
                    </td>
                    <td className="px-2 py-2">{item.login_method}</td>
                    <td className="px-2 py-2">
                      <AdminBadge label={item.login_status} tone={toStatusTone(item.login_status)} />
                    </td>
                    <td className="px-2 py-2">{item.failure_reason || "-"}</td>
                    <td className="px-2 py-2">{item.ip_address || "-"}</td>
                    <td className="px-2 py-2">
                      <div className="max-w-[14rem] truncate text-[11px] text-slate-500 dark:text-white/45">
                        {item.user_agent || "-"}
                      </div>
                    </td>
                    <td className="px-2 py-2">{formatAdminDateTime(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 dark:text-white/60">
          <div>
            {copy.adminShowing} {items.length} {copy.adminOf} {formatAdminNumber(loginEvents?.total ?? 0)}
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => onPageChange(Math.max(1, activityPage - 1))}
              disabled={!canGoPrevious}
              className="rounded-full border border-black/10 px-3 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminPrevious}
            </button>
            <button
              onClick={() => onPageChange(activityPage + 1)}
              disabled={!canGoNext}
              className="rounded-full border border-black/10 px-3 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminNext}
            </button>
          </div>
        </div>
      </AdminSurface>
    </div>
  );
}
