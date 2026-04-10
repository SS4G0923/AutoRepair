import { useEffect, useState } from "react";
import type {
  BillingOrderItem,
  BillingOrderSession,
  BillingSummaryData,
  PaymentMethodCode,
} from "../types";

interface UseBillingCenterOptions {
  apiBaseUrl: string;
  enabled: boolean;
  refreshSession?: () => Promise<void>;
}

async function fetchBillingJson<T>(
  apiBaseUrl: string,
  path: string,
  init?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    credentials: "include",
    ...init,
    signal,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function useBillingCenter({
  apiBaseUrl,
  enabled,
  refreshSession,
}: UseBillingCenterOptions) {
  const [billingLoading, setBillingLoading] = useState(false);
  const [billingActing, setBillingActing] = useState(false);
  const [billingError, setBillingError] = useState("");
  const [billingData, setBillingData] = useState<BillingSummaryData | null>(null);
  const [activeOrderId, setActiveOrderId] = useState<number | null>(null);
  const [activeOrderSession, setActiveOrderSession] = useState<BillingOrderSession | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setBillingError("");
      setBillingData(null);
      setActiveOrderId(null);
      setActiveOrderSession(null);
      return;
    }

    const controller = new AbortController();
    setBillingLoading(true);
    setBillingError("");

    void fetchBillingJson<BillingSummaryData>(
      apiBaseUrl,
      "/api/billing/summary",
      undefined,
      controller.signal,
    )
      .then((data) => {
        setBillingData(data);
        const nextActiveOrderId = data.orders[0]?.id ?? null;
        setActiveOrderId((current) => current ?? nextActiveOrderId);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setBillingError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setBillingLoading(false);
        }
      });

    return () => controller.abort();
  }, [apiBaseUrl, enabled, refreshNonce]);

  useEffect(() => {
    if (!enabled || activeOrderId == null) {
      setActiveOrderSession(null);
      return;
    }

    const controller = new AbortController();
    void fetchBillingJson<BillingOrderSession>(
      apiBaseUrl,
      `/api/billing/orders/${activeOrderId}/session`,
      undefined,
      controller.signal,
    )
      .then((payload) => setActiveOrderSession(payload))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setBillingError(error instanceof Error ? error.message : String(error));
      });

    return () => controller.abort();
  }, [activeOrderId, apiBaseUrl, enabled]);

  async function refreshBillingData() {
    setRefreshNonce((current) => current + 1);
  }

  async function createOrder(planCode: string, paymentMethod: PaymentMethodCode) {
    setBillingActing(true);
    setBillingError("");
    try {
      const payload = await fetchBillingJson<{ order: BillingOrderItem }>(
        apiBaseUrl,
        "/api/billing/orders",
        {
          method: "POST",
          body: JSON.stringify({
            plan_code: planCode,
            payment_method: paymentMethod,
          }),
        },
      );
      setActiveOrderId(payload.order.id);
      await refreshBillingData();
      return payload.order;
    } catch (error) {
      setBillingError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      setBillingActing(false);
    }
  }

  async function refreshOrderSession(orderId?: number) {
    const targetOrderId = orderId ?? activeOrderId;
    if (targetOrderId == null) {
      return;
    }
    setBillingActing(true);
    setBillingError("");
    try {
      const payload = await fetchBillingJson<BillingOrderSession>(
        apiBaseUrl,
        `/api/billing/orders/${targetOrderId}/session`,
      );
      setActiveOrderSession(payload);
      if (refreshSession) {
        await refreshSession();
      }
    } catch (error) {
      setBillingError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      setBillingActing(false);
    }
  }

  return {
    billingLoading,
    billingActing,
    billingError,
    billingData,
    activeOrderId,
    activeOrderSession,
    setActiveOrderId,
    refreshBillingData,
    refreshOrderSession,
    createOrder,
  };
}
