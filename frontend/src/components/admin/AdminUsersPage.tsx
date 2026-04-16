import { useEffect, useState } from "react";
import { getUserRoleLabel } from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type { AdminUserItem, UserRole } from "../../types";
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
  updatingUserId: number | null;
  onUpdateUserRole: (userId: number, role: UserRole) => void;
}

export function AdminUsersPage({
  copy,
  users,
  updatingUserId,
  onUpdateUserRole,
}: AdminUsersPageProps) {
  const [roleDrafts, setRoleDrafts] = useState<Record<number, UserRole>>({});
  const adminCount = users.filter((item) => item.role === "admin").length;
  const advancedCount = users.filter((item) => item.role === "advanced").length;
  const totalTokens = users.reduce((sum, item) => sum + item.total_tokens, 0);

  useEffect(() => {
    setRoleDrafts(
      users.reduce<Record<number, UserRole>>((accumulator, user) => {
        accumulator[user.id] = user.role;
        return accumulator;
      }, {}),
    );
  }, [users]);

  return (
    <div className="space-y-2">
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminUsers} value={formatAdminNumber(users.length)} tone="sky" />
        <AdminMetricCard label={copy.adminDashboardAdminUsers} value={formatAdminNumber(adminCount)} tone="emerald" />
        <AdminMetricCard label={copy.adminRoleAdvanced} value={formatAdminNumber(advancedCount)} tone="amber" />
        <AdminMetricCard label={copy.adminTokens} value={formatAdminNumber(totalTokens)} tone="rose" />
      </div>

      <AdminSurface>
        <div className="overflow-x-auto">
          {users.length === 0 ? (
            <AdminEmptyState message={copy.adminNoData} />
          ) : (
            <table className="min-w-full text-left text-xs">
              <thead className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                <tr>
                  <th className="px-2 py-2">{copy.nameLabel}</th>
                  <th className="px-2 py-2">{copy.adminRole}</th>
                  <th className="px-2 py-2">{copy.adminAccountStatus}</th>
                  <th className="px-2 py-2">{copy.adminAuthSource}</th>
                  <th className="px-2 py-2">{copy.adminCreatedAt}</th>
                  <th className="px-2 py-2">{copy.adminLastLogin}</th>
                  <th className="px-2 py-2">{copy.adminHistoryCount}</th>
                  <th className="px-2 py-2">{copy.adminRequestCount}</th>
                  <th className="px-2 py-2">{copy.adminPaymentOrderCount}</th>
                  <th className="px-2 py-2">{copy.adminActivePlan}</th>
                  <th className="px-2 py-2">{copy.adminTokens}</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr
                    key={user.id}
                    className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                  >
                    <td className="px-2 py-2">
                      <div className="flex min-w-[13rem] items-center gap-2">
                        {user.avatar_url ? (
                          <img
                            src={user.avatar_url}
                            alt={user.display_name}
                            className="h-7 w-7 rounded-full object-cover"
                          />
                        ) : (
                          <div className="grid h-7 w-7 place-items-center rounded-full bg-slate-900 text-[10px] font-semibold text-white dark:bg-white dark:text-slate-950">
                            {getNameInitials(user.display_name, user.email)}
                          </div>
                        )}
                        <div className="min-w-0">
                          <div className="truncate font-medium text-slate-900 dark:text-white">
                            {user.display_name}
                          </div>
                          <div className="truncate text-[11px] text-slate-500 dark:text-white/45">
                            {user.email}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex min-w-[10rem] flex-col gap-1.5">
                        <AdminBadge label={getUserRoleLabel(copy, user.role)} tone={toStatusTone(user.role)} />
                        <div className="flex items-center gap-1.5">
                          <select
                            value={roleDrafts[user.id] ?? user.role}
                            onChange={(event) =>
                              setRoleDrafts((current) => ({
                                ...current,
                                [user.id]: event.target.value as UserRole,
                              }))
                            }
                            className="min-w-0 flex-1 rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-[11px] text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                          >
                            <option value="basic">{copy.adminRoleBasic}</option>
                            <option value="advanced">{copy.adminRoleAdvanced}</option>
                            <option value="admin">{copy.adminRoleAdmin}</option>
                          </select>
                          <button
                            onClick={() => onUpdateUserRole(user.id, roleDrafts[user.id] ?? user.role)}
                            disabled={updatingUserId === user.id || (roleDrafts[user.id] ?? user.role) === user.role}
                            className="rounded-full bg-slate-900 px-2.5 py-1.5 text-[11px] font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                          >
                            {updatingUserId === user.id ? copy.adminUpdatingRole : copy.adminSaveRole}
                          </button>
                        </div>
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <AdminBadge
                        label={
                          user.account_status === "active"
                            ? copy.adminAccountActive
                            : copy.adminAccountSuspended
                        }
                        tone={toStatusTone(user.account_status)}
                      />
                    </td>
                    <td className="px-2 py-2">{user.auth_source}</td>
                    <td className="px-2 py-2">{formatAdminDateTime(user.created_at)}</td>
                    <td className="px-2 py-2">{formatAdminDateTime(user.last_login_at)}</td>
                    <td className="px-2 py-2">{formatAdminNumber(user.history_count)}</td>
                    <td className="px-2 py-2">{formatAdminNumber(user.llm_request_count)}</td>
                    <td className="px-2 py-2">{formatAdminNumber(user.payment_order_count)}</td>
                    <td className="px-2 py-2">{user.active_subscription_plan || "-"}</td>
                    <td className="px-2 py-2">{formatAdminNumber(user.total_tokens)}</td>
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
