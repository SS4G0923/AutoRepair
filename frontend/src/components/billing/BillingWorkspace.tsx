import { useEffect, useMemo, useState } from "react";
import { formatCurrencyAmount, formatTimestamp, getPaymentMethodLabel, getPaymentStatusLabel, getUserRoleLabel } from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type {
  AuthenticatedUser,
  BillingOrderItem,
  BillingSummaryData,
  PaymentMethodCode,
} from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
} from "../admin/AdminCommon";

interface BillingWorkspaceProps {
  activeOrderId: number | null;
  billingActing: boolean;
  billingData: BillingSummaryData | null;
  billingError: string;
  billingLoading: boolean;
  copy: AppCopy;
  currentUser: AuthenticatedUser;
  workspaceMainClass: string;
  onCompleteSandboxOrder: (orderId: number) => Promise<void>;
  onCreateOrder: (planCode: string, paymentMethod: PaymentMethodCode) => Promise<unknown>;
  onRefresh: () => void;
  onSelectOrder: (orderId: number) => void;
}

export function BillingWorkspace({
  activeOrderId,
  billingActing,
  billingData,
  billingError,
  billingLoading,
  copy,
  currentUser,
  workspaceMainClass,
  onCompleteSandboxOrder,
  onCreateOrder,
  onRefresh,
  onSelectOrder,
}: BillingWorkspaceProps) {
  const [selectedPlanCode, setSelectedPlanCode] = useState("");
  const [selectedMethod, setSelectedMethod] = useState<PaymentMethodCode>("card");

  useEffect(() => {
    if (!billingData) {
      return;
    }
    if (!selectedPlanCode) {
      setSelectedPlanCode(billingData.plans[0]?.plan_code ?? "");
    }
  }, [billingData, selectedPlanCode]);

  const activeOrder = useMemo(() => {
    if (!billingData) {
      return null;
    }
    return billingData.orders.find((item) => item.id === activeOrderId) ?? billingData.orders[0] ?? null;
  }, [activeOrderId, billingData]);

  const canCreateOrder = Boolean(selectedPlanCode) && currentUser.role === "basic" && !billingActing;
  const isUpgraded = currentUser.role === "advanced" || currentUser.role === "admin";

  async function handleCreateOrder() {
    if (!selectedPlanCode) {
      return;
    }
    await onCreateOrder(selectedPlanCode, selectedMethod);
  }

  async function handleSandboxComplete(order: BillingOrderItem) {
    await onCompleteSandboxOrder(order.id);
  }

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/72 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow ${workspaceMainClass}`}
    >
      <div className="flex shrink-0 flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.32em] text-slate-500 dark:text-white/40">
            {copy.billingOpen}
          </div>
          <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.billingTitle}
          </div>
          <div className="mt-2 max-w-4xl text-sm text-slate-600 dark:text-white/65">
            {copy.billingHint}
          </div>
        </div>

        <button
          onClick={onRefresh}
          className={`rounded-full px-5 py-3 text-sm font-semibold transition ${
            billingLoading
              ? "cursor-wait bg-slate-300 text-slate-700 dark:bg-white/20 dark:text-white/75"
              : "bg-slate-900 text-white hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
          }`}
        >
          {copy.adminRefresh}
        </button>
      </div>

      {billingError ? (
        <div className="mt-3 rounded-[20px] border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
          {billingError}
        </div>
      ) : null}

      <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
        {billingLoading && !billingData ? (
          <div className="grid min-h-[18rem] place-items-center text-sm text-slate-500 dark:text-white/45">
            {copy.adminRefresh}...
          </div>
        ) : !billingData ? (
          <AdminEmptyState message={copy.adminNoData} />
        ) : (
          <div className="space-y-3">
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <AdminMetricCard
                label={copy.billingCurrentRole}
                value={getUserRoleLabel(copy, currentUser.role)}
                caption={currentUser.email}
                tone="sky"
              />
              <AdminMetricCard
                label={copy.billingCurrentSubscription}
                value={billingData.current_subscription?.plan_name ?? copy.billingNoSubscription}
                caption={billingData.current_subscription?.started_at ? formatAdminDateTime(billingData.current_subscription.started_at) : "-"}
                tone="emerald"
              />
              <AdminMetricCard
                label={copy.billingPaidOrders}
                value={String(billingData.order_summary.paid_orders)}
                caption={copy.billingHistoryTitle}
                tone="amber"
              />
              <AdminMetricCard
                label={copy.billingRevenue}
                value={formatCurrencyAmount(billingData.order_summary.paid_amount_cents)}
                caption={`${copy.billingPaymentMode}: ${
                  billingData.payment_mode === "live"
                    ? copy.billingModeLive
                    : billingData.payment_mode === "manual"
                      ? copy.billingModeManual
                      : copy.billingModeSandbox
                }`}
                tone="rose"
              />
            </div>

            {isUpgraded ? (
              <div className="rounded-[22px] border border-emerald-500/20 bg-emerald-500/10 px-4 py-4 text-sm text-emerald-700 dark:text-emerald-200">
                <div className="font-semibold">{copy.billingSuccessTitle}</div>
                <div className="mt-1">{copy.billingSuccessHint}</div>
              </div>
            ) : null}

            <div className="grid gap-3 xl:grid-cols-[1.1fr_0.9fr]">
              <AdminSurface>
                <AdminSectionTitle
                  eyebrow={copy.billingOpen}
                  title={copy.billingPlanSectionTitle}
                  hint={copy.billingPlanSectionHint}
                />
                <div className="mt-4 space-y-3">
                  {billingData.plans.map((plan) => {
                    const active = selectedPlanCode === plan.plan_code;
                    return (
                      <button
                        key={plan.plan_code}
                        onClick={() => setSelectedPlanCode(plan.plan_code)}
                        className={`w-full rounded-[22px] border px-4 py-4 text-left transition ${
                          active
                            ? "border-slate-900 bg-slate-900 text-white shadow-lg dark:border-white dark:bg-white dark:text-slate-950"
                            : "border-black/5 bg-black/[0.02] hover:bg-black/[0.04] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:bg-white/[0.05]"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-base font-semibold">{plan.plan_name}</div>
                            <div
                              className={`mt-1 text-sm ${
                                active ? "text-white/75 dark:text-slate-950/75" : "text-slate-500 dark:text-white/50"
                              }`}
                            >
                              {plan.description}
                            </div>
                          </div>
                          <AdminBadge label={getUserRoleLabel(copy, plan.role_granted)} tone="sky" />
                        </div>
                        <div
                          className={`mt-4 flex items-center justify-between text-sm ${
                            active ? "text-white/75 dark:text-slate-950/75" : "text-slate-600 dark:text-white/55"
                          }`}
                        >
                          <span>{plan.billing_cycle}</span>
                          <span className="text-lg font-semibold">
                            {formatCurrencyAmount(plan.amount_cents, plan.currency)}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </AdminSurface>

              <AdminSurface>
                <AdminSectionTitle
                  eyebrow={copy.billingPaymentMethod}
                  title={copy.billingMethodSectionTitle}
                  hint={copy.billingMethodSectionHint}
                />
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {billingData.payment_methods.map((method) => {
                    const active = selectedMethod === method.code;
                    return (
                      <button
                        key={method.code}
                        onClick={() => setSelectedMethod(method.code)}
                        className={`rounded-[22px] border px-4 py-4 text-left transition ${
                          active
                            ? "border-slate-900 bg-slate-900 text-white shadow-lg dark:border-white dark:bg-white dark:text-slate-950"
                            : "border-black/5 bg-black/[0.02] hover:bg-black/[0.04] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:bg-white/[0.05]"
                        }`}
                      >
                        <div className="font-medium">{getPaymentMethodLabel(copy, method.code)}</div>
                        <div
                          className={`mt-2 text-xs ${
                            active ? "text-white/75 dark:text-slate-950/75" : "text-slate-500 dark:text-white/45"
                          }`}
                        >
                          {copy.billingPaymentMode}:{" "}
                          {method.mode === "live"
                            ? copy.billingModeLive
                            : method.mode === "manual"
                              ? copy.billingModeManual
                              : copy.billingModeSandbox}
                        </div>
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => void handleCreateOrder()}
                  disabled={!canCreateOrder}
                  className={`mt-4 w-full rounded-[18px] px-4 py-3 text-sm font-semibold transition ${
                    canCreateOrder
                      ? "bg-slate-900 text-white hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      : "cursor-not-allowed bg-slate-300 text-slate-600 dark:bg-white/15 dark:text-white/45"
                  }`}
                >
                  {billingActing ? copy.billingProcessing : copy.billingCreateOrder}
                </button>

                <div className="mt-3 text-xs text-slate-500 dark:text-white/40">{formatTimestamp()}</div>
              </AdminSurface>
            </div>

            <div className="grid gap-3 xl:grid-cols-[0.85fr_1.15fr]">
              <AdminSurface>
                <AdminSectionTitle
                  eyebrow={copy.billingOpen}
                  title={copy.billingPaymentInstructions}
                  hint={activeOrder ? `#${activeOrder.order_no}` : copy.billingNoOrders}
                />
                <div className="mt-4">
                  {!activeOrder ? (
                    <AdminEmptyState message={copy.billingNoOrders} />
                  ) : (
                    <div className="space-y-3">
                      <div className="rounded-[20px] border border-black/5 bg-black/[0.02] px-4 py-4 dark:border-white/10 dark:bg-white/[0.03]">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <div className="font-medium text-slate-900 dark:text-white">
                              {activeOrder.plan_name}
                            </div>
                            <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                              {copy.billingOrderNo}: {activeOrder.order_no}
                            </div>
                          </div>
                          <AdminBadge
                            label={getPaymentStatusLabel(copy, activeOrder.order_status)}
                            tone={
                              activeOrder.order_status === "paid"
                                ? "emerald"
                                : activeOrder.order_status === "rejected" || activeOrder.order_status === "failed"
                                  ? "rose"
                                  : "amber"
                            }
                          />
                        </div>

                        <div className="mt-4 space-y-2 text-sm text-slate-600 dark:text-white/60">
                          <div>
                            {copy.billingAmount}: {formatCurrencyAmount(activeOrder.amount_cents, activeOrder.currency)}
                          </div>
                          <div>
                            {copy.billingPaymentMethod}: {getPaymentMethodLabel(copy, activeOrder.payment_method)}
                          </div>
                          <div>{activeOrder.instructions || "-"}</div>
                          {activeOrder.qr_code_text ? (
                            <div className="rounded-[16px] border border-dashed border-black/10 px-3 py-3 font-mono text-xs dark:border-white/10">
                              {activeOrder.qr_code_text}
                            </div>
                          ) : null}
                        </div>
                      </div>

                      {activeOrder.checkout_url ? (
                        <a
                          href={activeOrder.checkout_url}
                          target="_blank"
                          rel="noreferrer"
                          className="flex w-full items-center justify-center rounded-[18px] bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                        >
                          {copy.billingOpenCheckout}
                        </a>
                      ) : null}

                      {activeOrder.checkout_action === "sandbox" && activeOrder.order_status !== "paid" ? (
                        <button
                          onClick={() => void handleSandboxComplete(activeOrder)}
                          disabled={billingActing}
                          className="flex w-full items-center justify-center rounded-[18px] border border-black/10 px-4 py-3 text-sm font-semibold transition hover:bg-black/[0.03] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:hover:bg-white/[0.05]"
                        >
                          {copy.billingSandboxComplete}
                        </button>
                      ) : null}
                    </div>
                  )}
                </div>
              </AdminSurface>

              <AdminSurface>
                <AdminSectionTitle
                  eyebrow={copy.billingOpen}
                  title={copy.billingHistoryTitle}
                  hint={copy.billingHistoryHint}
                />
                <div className="mt-4 space-y-3">
                  {billingData.orders.length === 0 ? (
                    <AdminEmptyState message={copy.billingNoOrders} />
                  ) : (
                    billingData.orders.map((order) => {
                      const active = order.id === activeOrder?.id;
                      return (
                        <button
                          key={order.id}
                          onClick={() => onSelectOrder(order.id)}
                          className={`w-full rounded-[22px] border px-4 py-4 text-left transition ${
                            active
                              ? "border-slate-900 bg-slate-900 text-white shadow-lg dark:border-white dark:bg-white dark:text-slate-950"
                              : "border-black/5 bg-black/[0.02] hover:bg-black/[0.04] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:bg-white/[0.05]"
                          }`}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate font-medium">{order.plan_name}</div>
                              <div
                                className={`mt-1 text-xs ${
                                  active ? "text-white/75 dark:text-slate-950/75" : "text-slate-500 dark:text-white/45"
                                }`}
                              >
                                {copy.billingOrderNo}: {order.order_no}
                              </div>
                            </div>
                            <AdminBadge
                              label={getPaymentStatusLabel(copy, order.order_status)}
                              tone={
                                order.order_status === "paid"
                                  ? "emerald"
                                  : order.order_status === "rejected" || order.order_status === "failed"
                                    ? "rose"
                                    : "amber"
                              }
                            />
                          </div>

                          <div
                            className={`mt-3 grid gap-2 text-xs sm:grid-cols-2 ${
                              active ? "text-white/75 dark:text-slate-950/75" : "text-slate-500 dark:text-white/45"
                            }`}
                          >
                            <div>{formatCurrencyAmount(order.amount_cents, order.currency)}</div>
                            <div>{getPaymentMethodLabel(copy, order.payment_method)}</div>
                            <div>{formatAdminDateTime(order.created_at)}</div>
                            <div>{order.paid_at ? formatAdminDateTime(order.paid_at) : "-"}</div>
                          </div>
                        </button>
                      );
                    })
                  )}
                </div>
              </AdminSurface>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
