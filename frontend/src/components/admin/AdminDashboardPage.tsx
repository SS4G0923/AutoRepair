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
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.label}>
          <div className="mb-0.5 flex items-center justify-between gap-3 text-[11px] text-slate-500 dark:text-white/45">
            <span>{item.label}</span>
            <span>{item.meta}</span>
          </div>
          <div className="h-2.5 rounded-full bg-black/[0.05] dark:bg-white/[0.06]">
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
    <div className="space-y-2">
      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        <AdminMetricCard
          label={copy.adminDashboardTotalUsers}
          value={formatAdminNumber(data.summary.total_users)}
          caption={`${copy.adminDashboardAdminUsers}: ${formatAdminNumber(data.summary.admin_users)}`}
          tone="sky"
        />
        <AdminMetricCard
          label={copy.adminDashboardNewUsers7d}
          value={formatAdminNumber(data.summary.new_users_7d)}
          tone="emerald"
        />
        <AdminMetricCard
          label={copy.adminDashboardRequests7d}
          value={formatAdminNumber(data.summary.llm_requests_7d)}
          caption={`${copy.adminFilterChat} ${formatAdminNumber(data.summary.chat_requests_7d)} / ${copy.adminFilterRepair} ${formatAdminNumber(data.summary.repair_requests_7d)}`}
          tone="amber"
        />
        <AdminMetricCard
          label={copy.adminDashboardTokens7d}
          value={formatAdminNumber(data.summary.total_tokens_7d)}
          tone="rose"
        />
        <AdminMetricCard
          label={copy.billingPaidOrders}
          value={formatAdminNumber(data.summary.paid_orders_30d)}
          tone="emerald"
        />
        <AdminMetricCard
          label={copy.adminRevenue30d}
          value={formatCurrencyAmount(data.summary.paid_amount_cents_30d)}
          tone="amber"
        />
      </div>

      <div className="grid gap-2 xl:grid-cols-2">
        <AdminSurface>
          <AdminSectionTitle title={copy.adminDailyTokens} />
          <div className="mt-2">
            <DashboardBarList
              emptyMessage={copy.adminNoData}
              items={data.daily_token_usage.map((item) => ({
                label: item.day,
                value: item.total_tokens,
                meta: `${formatAdminNumber(item.total_tokens)} tk · ${formatAdminNumber(item.request_count)} req`,
              }))}
            />
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle title={copy.adminUserGrowth} />
          <div className="mt-2">
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

      <div className="grid gap-2 xl:grid-cols-[0.88fr_1.12fr]">
        <AdminSurface>
          <AdminSectionTitle title={copy.adminTopModels} />
          <div className="mt-2 space-y-2">
            {data.model_usage.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.model_usage.map((item) => (
                <div
                  key={`${item.provider}-${item.model}`}
                  className="flex items-center justify-between gap-2 rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2.5 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-900 dark:text-white">
                      {item.model}
                    </div>
                    <div className="text-[11px] text-slate-500 dark:text-white/45">
                      {item.provider} · {formatAdminNumber(item.request_count)} req · {formatAdminDurationMs(item.avg_latency_ms)}
                    </div>
                  </div>
                  <AdminBadge label={formatAdminNumber(item.total_tokens)} tone="sky" />
                </div>
              ))
            )}
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle title={copy.adminLatestRequests} />
          <div className="mt-2 space-y-2">
            {data.latest_requests.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.latest_requests.map((item) => (
                <div
                  key={item.id}
                  className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2.5 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                      <div className="text-sm font-medium text-slate-900 dark:text-white">
                        #{item.id} · {item.model}
                      </div>
                      <AdminBadge label={item.request_status} tone={toStatusTone(item.request_status)} />
                    </div>
                    <div className="text-right text-[11px] text-slate-500 dark:text-white/45">
                      {formatAdminNumber(item.total_tokens)} tk · {formatAdminDateTime(item.started_at)}
                    </div>
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500 dark:text-white/45">
                    {item.request_mode} · {item.stage || "-"} · {item.user_display_name || item.user_email || "-"} · {item.provider}
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>
      </div>

      <div className="grid gap-2 xl:grid-cols-2">
        <AdminSurface>
          <AdminSectionTitle title={copy.adminDailyPayments} />
          <div className="mt-2">
            <DashboardBarList
              emptyMessage={copy.adminNoData}
              items={data.daily_payment_volume.map((item) => ({
                label: item.day,
                value: item.paid_amount_cents,
                meta: `${formatCurrencyAmount(item.paid_amount_cents)} · ${formatAdminNumber(item.paid_orders)} paid`,
              }))}
            />
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle title={copy.adminPaymentMethodMix} />
          <div className="mt-2 space-y-2">
            {data.payment_method_usage.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              data.payment_method_usage.map((item) => (
                <div
                  key={item.payment_method}
                  className="flex items-center justify-between gap-2 rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2.5 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div>
                    <div className="text-sm font-medium text-slate-900 dark:text-white">
                      {getPaymentMethodLabel(copy, item.payment_method)}
                    </div>
                    <div className="text-[11px] text-slate-500 dark:text-white/45">
                      {formatAdminNumber(item.paid_orders)} {copy.billingPaidOrders}
                    </div>
                  </div>
                  <AdminBadge label={formatCurrencyAmount(item.paid_amount_cents)} tone="sky" />
                </div>
              ))
            )}
          </div>
        </AdminSurface>
      </div>
    </div>
  );
}
