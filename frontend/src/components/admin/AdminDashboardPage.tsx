import { formatCurrencyAmount, getPaymentMethodLabel } from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type { AdminDashboardData } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminDurationMs,
  formatAdminNumber,
  toStatusTone,
} from "./AdminCommon";

interface AdminDashboardPageProps {
  copy: AppCopy;
  data: AdminDashboardData | null;
}

function DashboardBarList({
  items,
  emptyMessage,
}: {
  items: Array<{ label: string; value: number; meta: string }>;
  emptyMessage: string;
}) {
  const maxValue = items.reduce((current, item) => Math.max(current, item.value), 0);

  if (items.length === 0) {
    return <AdminEmptyState message={emptyMessage} />;
  }

  return (
    <div className="space-y-3">
      {items.map((item) => (
        <div key={item.label}>
          <div className="mb-1 flex items-center justify-between gap-3 text-xs text-slate-500 dark:text-white/45">
            <span>{item.label}</span>
            <span>{item.meta}</span>
          </div>
          <div className="h-3 rounded-full bg-black/[0.05] dark:bg-white/[0.06]">
            <div
              className="h-full rounded-full bg-slate-900 dark:bg-white"
              style={{
                width: `${maxValue > 0 ? Math.max(8, (item.value / maxValue) * 100) : 8}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function AdminDashboardPage({ copy, data }: AdminDashboardPageProps) {
  if (!data) {
    return <AdminEmptyState message={copy.adminNoData} />;
  }

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard
          label={copy.adminDashboardTotalUsers}
          value={formatAdminNumber(data.summary.total_users)}
          caption={`${copy.adminDashboardAdminUsers}: ${formatAdminNumber(data.summary.admin_users)} · ${copy.adminRoleAdvanced}: ${formatAdminNumber(data.summary.advanced_users)}`}
          tone="sky"
        />
        <AdminMetricCard
          label={copy.adminDashboardNewUsers7d}
          value={formatAdminNumber(data.summary.new_users_7d)}
          caption={`${copy.adminUsers}: ${formatAdminNumber(data.summary.total_users)}`}
          tone="emerald"
        />
        <AdminMetricCard
          label={copy.adminDashboardRequests7d}
          value={formatAdminNumber(data.summary.llm_requests_7d)}
          caption={`${copy.adminFilterChat}: ${formatAdminNumber(data.summary.chat_requests_7d)} / ${copy.adminFilterRepair}: ${formatAdminNumber(data.summary.repair_requests_7d)}`}
          tone="amber"
        />
        <AdminMetricCard
          label={copy.adminDashboardTokens7d}
          value={formatAdminNumber(data.summary.total_tokens_7d)}
          caption={`${copy.adminDashboardFailed7d}: ${formatAdminNumber(data.summary.failed_requests_7d)}`}
          tone="rose"
        />
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <AdminMetricCard
          label={copy.billingPaidOrders}
          value={formatAdminNumber(data.summary.paid_orders_30d)}
          caption={copy.adminPaymentsTitle}
          tone="emerald"
        />
        <AdminMetricCard
          label={copy.adminRevenue30d}
          value={formatCurrencyAmount(data.summary.paid_amount_cents_30d)}
          caption={copy.adminPaymentsHint}
          tone="amber"
        />
      </div>

      <div className="grid gap-3 xl:grid-cols-[0.95fr_1.05fr]">
        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminDashboard}
            title={copy.adminDailyTokens}
            hint={copy.adminDashboardHint}
          />
          <div className="mt-4">
            <DashboardBarList
              emptyMessage={copy.adminNoData}
              items={data.daily_token_usage.map((item) => ({
                label: item.day,
                value: item.total_tokens,
                meta: `${formatAdminNumber(item.total_tokens)} ${copy.adminTokens} · ${formatAdminNumber(item.request_count)} req`,
              }))}
            />
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminUsers}
            title={copy.adminUserGrowth}
            hint={copy.adminUsersHint}
          />
          <div className="mt-4">
            <DashboardBarList
              emptyMessage={copy.adminNoData}
              items={data.daily_user_growth.map((item) => ({
                label: item.day,
                value: item.cumulative_users,
                meta: `+${formatAdminNumber(item.new_users)} / ${formatAdminNumber(item.cumulative_users)}`,
              }))}
            />
          </div>
        </AdminSurface>
      </div>

      <div className="grid gap-3 xl:grid-cols-[0.88fr_1.12fr]">
        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminModels}
            title={copy.adminTopModels}
            hint={copy.adminModelsHint}
          />
          <div className="mt-4 space-y-3">
            {data.model_usage.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.model_usage.map((item) => (
                <div
                  key={`${item.provider}-${item.model}`}
                  className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-3 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-medium text-slate-900 dark:text-white">
                        {item.model}
                      </div>
                      <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                        {item.provider} · {formatAdminNumber(item.request_count)} req
                      </div>
                    </div>
                    <AdminBadge label={formatAdminNumber(item.total_tokens)} tone="sky" />
                  </div>
                  <div className="mt-3 grid gap-2 text-xs text-slate-500 dark:text-white/45 sm:grid-cols-2">
                    <div>{copy.adminLatency}: {formatAdminDurationMs(item.avg_latency_ms)}</div>
                    <div>{copy.adminLastLogin}: {formatAdminDateTime(item.last_used_at)}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminRequests}
            title={copy.adminLatestRequests}
            hint={`${copy.adminDashboardRequests7d}: ${formatAdminNumber(data.summary.llm_requests_7d)} · ${copy.adminDashboardTokens7d}: ${formatAdminNumber(data.summary.total_tokens_7d)}`}
          />
          <div className="mt-4 space-y-3">
            {data.latest_requests.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.latest_requests.map((item) => (
                <div
                  key={item.id}
                  className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-3 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="font-medium text-slate-900 dark:text-white">
                          #{item.id} · {item.model}
                        </div>
                        <AdminBadge label={item.request_status} tone={toStatusTone(item.request_status)} />
                      </div>
                      <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                        {item.request_mode} · {item.stage || "-"} · {item.purpose || "-"}
                      </div>
                    </div>
                    <div className="text-right text-xs text-slate-500 dark:text-white/45">
                      <div>{formatAdminNumber(item.total_tokens)} {copy.adminTokens}</div>
                      <div>{formatAdminDateTime(item.started_at)}</div>
                    </div>
                  </div>
                  <div className="mt-3 grid gap-2 text-xs text-slate-500 dark:text-white/45 sm:grid-cols-2">
                    <div>{copy.adminUserLabel}: {item.user_display_name || item.user_email || "-"}</div>
                    <div>{copy.adminRequestProvider}: {item.provider}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>
      </div>

      <div className="grid gap-3 xl:grid-cols-[0.95fr_1.05fr]">
        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminPayments}
            title={copy.adminDailyPayments}
            hint={copy.adminPaymentsHint}
          />
          <div className="mt-4">
            <DashboardBarList
              emptyMessage={copy.adminNoData}
              items={data.daily_payment_volume.map((item) => ({
                label: item.day,
                value: item.paid_amount_cents,
                meta: `${formatCurrencyAmount(item.paid_amount_cents)} · ${formatAdminNumber(item.paid_orders)} ${copy.billingPaidOrders}`,
              }))}
            />
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle
            eyebrow={copy.adminPayments}
            title={copy.adminPaymentMethodMix}
            hint={copy.adminRevenue30d}
          />
          <div className="mt-4 space-y-3">
            {data.payment_method_usage.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.payment_method_usage.map((item) => (
                <div
                  key={item.payment_method}
                  className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-3 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-medium text-slate-900 dark:text-white">
                      {getPaymentMethodLabel(copy, item.payment_method)}
                    </div>
                    <AdminBadge label={formatCurrencyAmount(item.paid_amount_cents)} tone="sky" />
                  </div>
                  <div className="mt-2 text-xs text-slate-500 dark:text-white/45">
                    {formatAdminNumber(item.paid_orders)} {copy.billingPaidOrders}
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>
      </div>
    </div>
  );
}
