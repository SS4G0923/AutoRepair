import { useEffect, useMemo, useState } from "react";
import {
  formatCurrencyAmount,
  getPaymentMethodLabel,
  getPaymentStatusLabel,
  getUserRoleLabel,
} from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type {
  AuthenticatedUser,
  BillingOrderSession,
  BillingSummaryData,
  PaymentMethodCode,
} from "../../types";
import { AdminBadge, AdminEmptyState, formatAdminDateTime } from "../admin/AdminCommon";

interface BillingWorkspaceProps {
  activeOrderId: number | null;
  activeOrderSession: BillingOrderSession | null;
  billingActing: boolean;
  billingData: BillingSummaryData | null;
  billingError: string;
  billingLoading: boolean;
  copy: AppCopy;
  currentUser: AuthenticatedUser;
  workspaceMainClass: string;
  onCreateOrder: (planCode: string, paymentMethod: PaymentMethodCode) => Promise<unknown>;
  onRefresh: () => void;
  onRefreshOrderSession: (orderId?: number) => Promise<void>;
  onSelectOrder: (orderId: number) => void;
}

export function BillingWorkspace({
  activeOrderId,
  activeOrderSession,
  billingActing,
  billingData,
  billingError,
  billingLoading,
  copy,
  currentUser,
  workspaceMainClass,
  onCreateOrder,
  onRefresh,
  onRefreshOrderSession,
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

  const sessionView = activeOrderSession?.session ?? null;
  const canCreateOrder = Boolean(selectedPlanCode) && currentUser.role === "basic" && !billingActing;
  const isUpgraded = currentUser.role === "advanced" || currentUser.role === "admin";

  async function handleCreateOrder() {
    if (!selectedPlanCode) {
      return;
    }
    await onCreateOrder(selectedPlanCode, selectedMethod);
  }

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/72 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 ${workspaceMainClass}`}
    >
      {/* ── header ── */}
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.billingTitle}
          </div>
          <span className="rounded-full bg-indigo-500/10 px-2.5 py-0.5 text-xs font-medium text-indigo-600 dark:bg-indigo-400/15 dark:text-indigo-300">
            {getUserRoleLabel(copy, currentUser.role)}
          </span>
          {billingData?.current_subscription ? (
            <span className="hidden text-xs text-slate-500 dark:text-white/45 sm:inline">
              {billingData.current_subscription.plan_name}
            </span>
          ) : null}
        </div>
        <button
          onClick={onRefresh}
          className={`rounded-full px-4 py-1.5 text-xs font-medium transition ${
            billingLoading
              ? "cursor-wait bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-white/50"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-white/10 dark:text-white/70 dark:hover:bg-white/15"
          }`}
        >
          {copy.adminRefresh}
        </button>
      </div>

      {billingError ? (
        <div className="mt-2 rounded-2xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
          {billingError}
        </div>
      ) : null}

      {/* ── body ── */}
      <div className="mt-2 min-h-0 flex-1 overflow-y-auto">
        {billingLoading && !billingData ? (
          <div className="grid min-h-[10rem] place-items-center text-sm text-slate-400 dark:text-white/35">
            {copy.adminRefresh}…
          </div>
        ) : !billingData ? (
          <AdminEmptyState message={copy.adminNoData} />
        ) : (
          <div className="space-y-3">
            {/* upgraded banner */}
            {isUpgraded ? (
              <div className="flex items-center gap-2.5 rounded-2xl bg-gradient-to-r from-emerald-50 to-teal-50 px-4 py-3 dark:from-emerald-500/10 dark:to-teal-500/10">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <div className="text-sm font-medium text-emerald-700 dark:text-emerald-300">{copy.billingSuccessTitle}</div>
                  <div className="text-xs text-emerald-600/70 dark:text-emerald-400/60">{copy.billingSuccessHint}</div>
                </div>
              </div>
            ) : null}

            {/* ── plans + method + CTA ── */}
            <div className="grid gap-3 xl:grid-cols-[1.1fr_0.9fr]">
              {/* plans */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-white/35">
                  {copy.billingPlanSectionTitle}
                </div>
                {billingData.plans.map((plan) => {
                  const active = selectedPlanCode === plan.plan_code;
                  return (
                    <button
                      key={plan.plan_code}
                      onClick={() => setSelectedPlanCode(plan.plan_code)}
                      className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                        active
                          ? "border-indigo-500 bg-indigo-500 text-white shadow-md shadow-indigo-500/20 dark:border-indigo-400 dark:bg-indigo-500 dark:shadow-indigo-500/15"
                          : "border-black/5 bg-white/60 hover:border-indigo-300 hover:bg-indigo-50/50 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:border-indigo-400/30 dark:hover:bg-indigo-500/5"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-sm font-semibold">{plan.plan_name}</div>
                          <div className={`mt-0.5 text-xs ${active ? "text-white/70" : "text-slate-500 dark:text-white/45"}`}>
                            {plan.description}
                          </div>
                        </div>
                        <div className={`text-lg font-bold tabular-nums ${active ? "text-white" : "text-indigo-600 dark:text-indigo-400"}`}>
                          {formatCurrencyAmount(plan.amount_cents, plan.currency)}
                        </div>
                      </div>
                      <div className={`mt-1.5 flex items-center gap-2 text-[11px] ${active ? "text-white/60" : "text-slate-400 dark:text-white/30"}`}>
                        <span>{plan.billing_cycle}</span>
                        <span>·</span>
                        <span>{getUserRoleLabel(copy, plan.role_granted)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* method + CTA */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-white/35">
                  {copy.billingMethodSectionTitle}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {billingData.payment_methods.map((method) => {
                    const active = selectedMethod === method.code;
                    return (
                      <button
                        key={method.code}
                        onClick={() => setSelectedMethod(method.code)}
                        className={`rounded-xl border px-3 py-2.5 text-left transition ${
                          active
                            ? "border-indigo-500 bg-indigo-500 text-white shadow-sm shadow-indigo-500/15 dark:border-indigo-400 dark:bg-indigo-500"
                            : "border-black/5 bg-white/60 hover:border-indigo-300 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:border-indigo-400/30"
                        }`}
                      >
                        <div className="text-sm font-medium">{getPaymentMethodLabel(copy, method.code)}</div>
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => void handleCreateOrder()}
                  disabled={!canCreateOrder}
                  className={`mt-1 w-full rounded-xl px-4 py-2.5 text-sm font-semibold transition ${
                    canCreateOrder
                      ? "bg-gradient-to-r from-indigo-500 to-violet-500 text-white shadow-md shadow-indigo-500/25 hover:shadow-lg hover:shadow-indigo-500/30 dark:from-indigo-500 dark:to-violet-500"
                      : "cursor-not-allowed bg-slate-100 text-slate-400 dark:bg-white/5 dark:text-white/25"
                  }`}
                >
                  {billingActing ? copy.billingProcessing : copy.billingCreateOrder}
                </button>

                {isUpgraded && !canCreateOrder ? (
                  <div className="text-center text-[11px] text-slate-400 dark:text-white/30">
                    {copy.billingSuccessHint}
                  </div>
                ) : null}
              </div>
            </div>

            {/* ── active order + session details ── */}
            {activeOrder ? (
              <div className="rounded-2xl border border-black/5 bg-white/50 p-3 dark:border-white/10 dark:bg-white/[0.02]">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-slate-900 dark:text-white">{activeOrder.plan_name}</span>
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
                  <button
                    onClick={() => void onRefreshOrderSession(activeOrder.id)}
                    className="rounded-full px-3 py-1 text-[11px] text-slate-500 transition hover:bg-slate-100 dark:text-white/45 dark:hover:bg-white/10"
                  >
                    {copy.adminRefresh}
                  </button>
                </div>
                <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-white/45">
                  <span>#{activeOrder.order_no}</span>
                  <span>{formatCurrencyAmount(activeOrder.amount_cents, activeOrder.currency)}</span>
                  <span>{getPaymentMethodLabel(copy, activeOrder.payment_method)}</span>
                  <span>{formatAdminDateTime(activeOrder.created_at)}</span>
                </div>
                {activeOrder.instructions ? (
                  <div className="mt-2 rounded-xl border border-dashed border-slate-200 px-3 py-2 text-xs text-slate-600 dark:border-white/10 dark:text-white/50">
                    {activeOrder.instructions}
                  </div>
                ) : null}

                {sessionView ? (
                  <div className="mt-3 space-y-2">
                    {sessionView.missing_config.length > 0 ? (
                      <div className="rounded-xl bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-300">
                        {copy.billingConfigMissing}: {sessionView.missing_config.join(", ")}
                      </div>
                    ) : null}

                    {sessionView.display_mode === "card_form" ? (
                      <div className="space-y-2">
                        <div className="text-xs font-medium text-slate-700 dark:text-white/70">{copy.billingCardFormTitle}</div>
                        <div className="grid gap-2">
                          <div id="stripe-card-number-element" className="rounded-xl border border-dashed border-slate-200 px-3 py-2.5 text-xs text-slate-400 dark:border-white/10 dark:text-white/30">
                            Card Number
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div id="stripe-card-expiry-element" className="rounded-xl border border-dashed border-slate-200 px-3 py-2.5 text-xs text-slate-400 dark:border-white/10 dark:text-white/30">
                              Expiry
                            </div>
                            <div id="stripe-card-cvc-element" className="rounded-xl border border-dashed border-slate-200 px-3 py-2.5 text-xs text-slate-400 dark:border-white/10 dark:text-white/30">
                              CVC
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : sessionView.display_mode === "paypal_buttons" ? (
                      <div className="space-y-2">
                        <div className="text-xs font-medium text-slate-700 dark:text-white/70">{copy.billingPaypalTitle}</div>
                        <div
                          id="paypal-button-container"
                          className="rounded-xl border border-dashed border-slate-200 px-4 py-5 text-center text-xs text-slate-400 dark:border-white/10 dark:text-white/30"
                        >
                          PayPal Buttons
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="text-xs font-medium text-slate-700 dark:text-white/70">{copy.billingQrTitle}</div>
                        <div className="grid gap-3 sm:grid-cols-[160px_minmax(0,1fr)] sm:items-start">
                          <div
                            id={`${activeOrder.payment_method}-qr-container`}
                            className="grid aspect-square place-items-center rounded-xl border border-dashed border-slate-200 bg-white text-xs text-slate-400 dark:border-white/10 dark:bg-white/[0.02] dark:text-white/30"
                          >
                            QR
                          </div>
                          <div className="break-all rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-[11px] text-slate-500 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/45">
                            {sessionView.qr_code_url || sessionView.qr_code_text || copy.billingQrPending}
                          </div>
                        </div>
                      </div>
                    )}

                    {sessionView.script_url ? (
                      <div className="truncate rounded-lg bg-slate-50 px-2.5 py-1.5 text-[10px] text-slate-400 dark:bg-white/[0.03] dark:text-white/30">
                        SDK: {sessionView.script_url}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}

            {/* ── compact order history ── */}
            {billingData.orders.length > 0 ? (
              <div>
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-white/35">
                  {copy.billingHistoryTitle}
                </div>
                <div className="space-y-1">
                  {billingData.orders.map((order) => {
                    const active = order.id === activeOrder?.id;
                    return (
                      <button
                        key={order.id}
                        onClick={() => onSelectOrder(order.id)}
                        className={`flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left transition ${
                          active
                            ? "bg-indigo-50 dark:bg-indigo-500/10"
                            : "hover:bg-slate-50 dark:hover:bg-white/[0.03]"
                        }`}
                      >
                        <div className="flex min-w-0 items-center gap-2">
                          <div className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                            order.order_status === "paid" ? "bg-emerald-500" :
                            order.order_status === "pending" ? "bg-amber-500" : "bg-rose-500"
                          }`} />
                          <span className={`truncate text-sm ${active ? "font-medium text-indigo-700 dark:text-indigo-300" : "text-slate-700 dark:text-white/70"}`}>
                            {order.plan_name}
                          </span>
                          <span className="shrink-0 text-[11px] text-slate-400 dark:text-white/30">
                            #{order.order_no}
                          </span>
                        </div>
                        <div className="flex shrink-0 items-center gap-2.5 text-xs text-slate-500 dark:text-white/45">
                          <span>{formatCurrencyAmount(order.amount_cents, order.currency)}</span>
                          <span className="hidden sm:inline">{formatAdminDateTime(order.created_at)}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </section>
  );
}
