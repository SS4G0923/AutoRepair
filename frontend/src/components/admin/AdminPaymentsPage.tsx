import type { Dispatch, SetStateAction } from "react";
import type { AdminPaymentFilters } from "../../app/useAdminConsole";
import { formatCurrencyAmount, getPaymentMethodLabel, getPaymentStatusLabel } from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type { AdminPaymentOrderList } from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminNumber,
} from "./AdminCommon";

interface AdminPaymentsPageProps {
  actingOrderId: number | null;
  copy: AppCopy;
  filters: AdminPaymentFilters;
  orders: AdminPaymentOrderList | null;
  setFilters: Dispatch<SetStateAction<AdminPaymentFilters>>;
  onApprove: (orderId: number) => void;
  onReject: (orderId: number) => void;
}

export function AdminPaymentsPage({
  actingOrderId,
  copy,
  filters,
  orders,
  setFilters,
  onApprove,
  onReject,
}: AdminPaymentsPageProps) {
  const items = orders?.items ?? [];
  const summary = orders?.summary;
  const startIndex = orders ? (orders.page - 1) * orders.page_size + 1 : 0;
  const endIndex = orders ? Math.min(orders.total, startIndex + items.length - 1) : 0;
  const canGoPrevious = Boolean(orders && orders.page > 1);
  const canGoNext = Boolean(orders && orders.page * orders.page_size < orders.total);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard
          label={copy.adminPayments}
          value={formatAdminNumber(summary?.total_orders ?? 0)}
          tone="sky"
        />
        <AdminMetricCard
          label={copy.billingPendingOrders}
          value={formatAdminNumber(summary?.pending_orders ?? 0)}
          tone="amber"
        />
        <AdminMetricCard
          label={copy.billingPaidOrders}
          value={formatAdminNumber(summary?.paid_orders ?? 0)}
          tone="emerald"
        />
        <AdminMetricCard
          label={copy.adminRevenue30d}
          value={formatCurrencyAmount(summary?.paid_amount_cents ?? 0)}
          tone="rose"
        />
      </div>

      <AdminSurface>
        <AdminSectionTitle
          eyebrow={copy.adminPayments}
          title={copy.adminPaymentsTitle}
          hint={copy.adminPaymentsHint}
        />

        <div className="mt-4 grid gap-3 xl:grid-cols-[1.35fr_0.8fr_0.8fr]">
          <input
            value={filters.q}
            onChange={(event) =>
              setFilters((current) => ({
                ...current,
                page: 1,
                q: event.target.value,
              }))
            }
            placeholder={copy.adminSearchPlaceholder}
            className="rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
          />
          <select
            value={filters.status}
            onChange={(event) =>
              setFilters((current) => ({
                ...current,
                page: 1,
                status: event.target.value,
              }))
            }
            className="rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
          >
            <option value="">{copy.adminFilterStatus}: {copy.adminFilterAll}</option>
            <option value="pending">{copy.billingStatusPending}</option>
            <option value="paid">{copy.billingStatusPaid}</option>
            <option value="rejected">{copy.billingStatusRejected}</option>
          </select>
          <select
            value={filters.paymentMethod}
            onChange={(event) =>
              setFilters((current) => ({
                ...current,
                page: 1,
                paymentMethod: event.target.value,
              }))
            }
            className="rounded-[18px] border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
          >
            <option value="">{copy.billingPaymentMethod}: {copy.adminFilterAll}</option>
            <option value="card">{copy.billingProviderCard}</option>
            <option value="paypal">{copy.billingProviderPaypal}</option>
            <option value="wechat">{copy.billingProviderWechat}</option>
            <option value="alipay">{copy.billingProviderAlipay}</option>
          </select>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600 dark:text-white/60">
          <div>
            {copy.adminShowing} {orders?.total ? startIndex : 0}-{endIndex} {copy.adminOf} {formatAdminNumber(orders?.total ?? 0)}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() =>
                setFilters((current) => ({
                  ...current,
                  page: Math.max(1, current.page - 1),
                }))
              }
              disabled={!canGoPrevious}
              className="rounded-full border border-black/10 px-4 py-2 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminPrevious}
            </button>
            <button
              onClick={() =>
                setFilters((current) => ({
                  ...current,
                  page: current.page + 1,
                }))
              }
              disabled={!canGoNext}
              className="rounded-full border border-black/10 px-4 py-2 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminNext}
            </button>
          </div>
        </div>
      </AdminSurface>

      <AdminSurface>
        <div className="space-y-3">
          {items.length === 0 ? (
            <AdminEmptyState message={copy.adminNoData} />
          ) : (
            items.map((order) => {
              const actionDisabled = actingOrderId === order.id;
              return (
                <div
                  key={order.id}
                  className="rounded-[22px] border border-black/5 bg-black/[0.02] px-4 py-4 dark:border-white/10 dark:bg-white/[0.03]"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="font-medium text-slate-900 dark:text-white">
                          #{order.order_no} · {order.plan_name}
                        </div>
                        <AdminBadge label={getPaymentStatusLabel(copy, order.order_status)} tone={
                          order.order_status === "paid"
                            ? "emerald"
                            : order.order_status === "rejected" || order.order_status === "failed"
                              ? "rose"
                              : "amber"
                        } />
                      </div>
                      <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                        {(order.user_display_name || order.user_email || "-")} · {getPaymentMethodLabel(copy, order.payment_method)}
                      </div>
                    </div>
                    <div className="text-right text-sm text-slate-600 dark:text-white/60">
                      <div>{formatCurrencyAmount(order.amount_cents, order.currency)}</div>
                      <div className="mt-1 text-xs">{formatAdminDateTime(order.created_at)}</div>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 text-xs text-slate-500 dark:text-white/45 sm:grid-cols-2 xl:grid-cols-5">
                    <div>{copy.billingPaymentMethod}: {getPaymentMethodLabel(copy, order.payment_method)}</div>
                    <div>{copy.billingPaymentMode}: {order.checkout_action}</div>
                    <div>{copy.billingStatusPaid}: {order.paid_at ? formatAdminDateTime(order.paid_at) : "-"}</div>
                    <div>{copy.adminUserLabel}: {order.user_email || "-"}</div>
                    <div>{copy.adminRole}: {order.target_role}</div>
                  </div>

                  {order.instructions ? (
                    <div className="mt-3 rounded-[16px] border border-dashed border-black/10 px-3 py-3 text-xs text-slate-600 dark:border-white/10 dark:text-white/55">
                      {order.instructions}
                    </div>
                  ) : null}

                  {order.order_status === "pending" ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      <button
                        onClick={() => onApprove(order.id)}
                        disabled={actionDisabled}
                        className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      >
                        {actionDisabled ? copy.adminRefresh : copy.adminApprove}
                      </button>
                      <button
                        onClick={() => onReject(order.id)}
                        disabled={actionDisabled}
                        className="rounded-full border border-black/10 px-4 py-2 text-sm font-semibold transition hover:bg-black/[0.03] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:hover:bg-white/[0.05]"
                      >
                        {copy.adminReject}
                      </button>
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </AdminSurface>
    </div>
  );
}
