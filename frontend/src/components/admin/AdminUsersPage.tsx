import type { AppCopy } from "../../i18n";
import type { AdminUserItem } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminNumber,
  getNameInitials,
  toStatusTone,
} from "./AdminCommon";

interface AdminUsersPageProps {
  copy: AppCopy;
  users: AdminUserItem[];
}

export function AdminUsersPage({ copy, users }: AdminUsersPageProps) {
  const adminCount = users.filter((item) => item.role === "admin").length;
  const totalRequests = users.reduce((sum, item) => sum + item.llm_request_count, 0);
  const totalTokens = users.reduce((sum, item) => sum + item.total_tokens, 0);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminUsers} value={formatAdminNumber(users.length)} tone="sky" />
        <AdminMetricCard label={copy.adminDashboardAdminUsers} value={formatAdminNumber(adminCount)} tone="emerald" />
        <AdminMetricCard label={copy.adminRequestCount} value={formatAdminNumber(totalRequests)} tone="amber" />
        <AdminMetricCard label={copy.adminTokens} value={formatAdminNumber(totalTokens)} tone="rose" />
      </div>

      <AdminSurface>
        <AdminSectionTitle
          eyebrow={copy.adminUsers}
          title={copy.adminUsersTitle}
          hint={copy.adminUsersHint}
        />
        <div className="mt-4 overflow-x-auto">
          {users.length === 0 ? (
            <AdminEmptyState message={copy.adminNoData} />
          ) : (
            <table className="min-w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/35">
                <tr>
                  <th className="px-3 py-3">{copy.nameLabel}</th>
                  <th className="px-3 py-3">{copy.adminRole}</th>
                  <th className="px-3 py-3">{copy.adminAccountStatus}</th>
                  <th className="px-3 py-3">{copy.adminAuthSource}</th>
                  <th className="px-3 py-3">{copy.adminCreatedAt}</th>
                  <th className="px-3 py-3">{copy.adminLastLogin}</th>
                  <th className="px-3 py-3">{copy.adminHistoryCount}</th>
                  <th className="px-3 py-3">{copy.adminRequestCount}</th>
                  <th className="px-3 py-3">{copy.adminTokens}</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr
                    key={user.id}
                    className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                  >
                    <td className="px-3 py-3">
                      <div className="flex min-w-[16rem] items-center gap-3">
                        {user.avatar_url ? (
                          <img
                            src={user.avatar_url}
                            alt={user.display_name}
                            className="h-10 w-10 rounded-full object-cover"
                          />
                        ) : (
                          <div className="grid h-10 w-10 place-items-center rounded-full bg-slate-900 text-xs font-semibold text-white dark:bg-white dark:text-slate-950">
                            {getNameInitials(user.display_name, user.email)}
                          </div>
                        )}
                        <div className="min-w-0">
                          <div className="truncate font-medium text-slate-900 dark:text-white">
                            {user.display_name}
                          </div>
                          <div className="truncate text-xs text-slate-500 dark:text-white/45">
                            {user.email}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <AdminBadge
                        label={user.role === "admin" ? copy.adminRoleAdmin : copy.adminRoleBasic}
                        tone={toStatusTone(user.role)}
                      />
                    </td>
                    <td className="px-3 py-3">
                      <AdminBadge
                        label={
                          user.account_status === "active"
                            ? copy.adminAccountActive
                            : copy.adminAccountSuspended
                        }
                        tone={toStatusTone(user.account_status)}
                      />
                    </td>
                    <td className="px-3 py-3">{user.auth_source}</td>
                    <td className="px-3 py-3">{formatAdminDateTime(user.created_at)}</td>
                    <td className="px-3 py-3">{formatAdminDateTime(user.last_login_at)}</td>
                    <td className="px-3 py-3">{formatAdminNumber(user.history_count)}</td>
                    <td className="px-3 py-3">{formatAdminNumber(user.llm_request_count)}</td>
                    <td className="px-3 py-3">{formatAdminNumber(user.total_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </AdminSurface>
    </div>
  );
}
