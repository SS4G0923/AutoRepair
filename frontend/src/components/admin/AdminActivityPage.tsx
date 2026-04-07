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
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminRecentLoginAttempts} value={formatAdminNumber(items.length)} tone="sky" />
        <AdminMetricCard label={copy.adminFilterCompleted} value={formatAdminNumber(successCount)} tone="emerald" />
        <AdminMetricCard label={copy.adminFilterFailed} value={formatAdminNumber(failedCount)} tone="rose" />
        <AdminMetricCard label={copy.adminUsers} value={formatAdminNumber(distinctUsers)} tone="amber" />
      </div>

      <AdminSurface>
        <AdminSectionTitle
          eyebrow={copy.adminActivity}
          title={copy.adminActivityTitle}
          hint={copy.adminActivityHint}
        />
        <div className="mt-4 overflow-x-auto">
          {items.length === 0 ? (
            <AdminEmptyState message={copy.adminNoData} />
          ) : (
            <table className="min-w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/35">
                <tr>
                  <th className="px-3 py-3">{copy.adminUserLabel}</th>
                  <th className="px-3 py-3">{copy.adminLoginMethod}</th>
                  <th className="px-3 py-3">{copy.adminLoginStatus}</th>
                  <th className="px-3 py-3">{copy.adminFailureReason}</th>
                  <th className="px-3 py-3">{copy.adminIpAddress}</th>
                  <th className="px-3 py-3">{copy.adminUserAgent}</th>
                  <th className="px-3 py-3">{copy.adminCreatedAt}</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.id}
                    className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                  >
                    <td className="px-3 py-3">
                      <div className="min-w-[14rem]">
                        <div className="font-medium text-slate-900 dark:text-white">
                          {item.user_display_name || item.user_email || item.email_attempt || "-"}
                        </div>
                        <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                          {item.user_email || item.email_attempt || "-"}
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3">{item.login_method}</td>
                    <td className="px-3 py-3">
                      <AdminBadge label={item.login_status} tone={toStatusTone(item.login_status)} />
                    </td>
                    <td className="px-3 py-3">{item.failure_reason || "-"}</td>
                    <td className="px-3 py-3">{item.ip_address || "-"}</td>
                    <td className="px-3 py-3">
                      <div className="max-w-[18rem] break-words text-xs text-slate-500 dark:text-white/45">
                        {item.user_agent || "-"}
                      </div>
                    </td>
                    <td className="px-3 py-3">{formatAdminDateTime(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600 dark:text-white/60">
          <div>
            {copy.adminShowing} {items.length} {copy.adminOf} {formatAdminNumber(loginEvents?.total ?? 0)}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(Math.max(1, activityPage - 1))}
              disabled={!canGoPrevious}
              className="rounded-full border border-black/10 px-4 py-2 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminPrevious}
            </button>
            <button
              onClick={() => onPageChange(activityPage + 1)}
              disabled={!canGoNext}
              className="rounded-full border border-black/10 px-4 py-2 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminNext}
            </button>
          </div>
        </div>
      </AdminSurface>
    </div>
  );
}
