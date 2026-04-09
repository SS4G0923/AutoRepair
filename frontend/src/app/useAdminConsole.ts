import { useEffect, useState } from "react";
import type {
  AdminDashboardData,
  AdminLoginEventList,
  AdminLlmRequestDetail,
  AdminLlmRequestList,
  AdminModelUsageReport,
  AdminPage,
  AdminPaymentOrderList,
  AdminUserItem,
  UserRole,
} from "../types";

interface UseAdminConsoleOptions {
  apiBaseUrl: string;
  enabled: boolean;
  refreshSession?: () => Promise<void>;
}

export interface AdminRequestFilters {
  page: number;
  pageSize: number;
  q: string;
  model: string;
  status: string;
  requestMode: string;
}

export interface AdminPaymentFilters {
  page: number;
  pageSize: number;
  q: string;
  status: string;
  paymentMethod: string;
}

async function fetchAdminJson<T>(apiBaseUrl: string, path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    credentials: "include",
    signal,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

async function postAdminJson<T>(
  apiBaseUrl: string,
  path: string,
  body?: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function useAdminConsole({
  apiBaseUrl,
  enabled,
  refreshSession,
}: UseAdminConsoleOptions) {
  const [adminPage, setAdminPage] = useState<AdminPage>("dashboard");
  const [adminLoading, setAdminLoading] = useState(false);
  const [adminError, setAdminError] = useState("");
  const [adminUserRoleUpdatingId, setAdminUserRoleUpdatingId] = useState<number | null>(null);
  const [adminPaymentActingOrderId, setAdminPaymentActingOrderId] = useState<number | null>(null);
  const [dashboardData, setDashboardData] = useState<AdminDashboardData | null>(null);
  const [users, setUsers] = useState<AdminUserItem[]>([]);
  const [requests, setRequests] = useState<AdminLlmRequestList | null>(null);
  const [requestDetail, setRequestDetail] = useState<AdminLlmRequestDetail | null>(null);
  const [requestDetailLoading, setRequestDetailLoading] = useState(false);
  const [selectedRequestId, setSelectedRequestId] = useState<number | null>(null);
  const [modelUsage, setModelUsage] = useState<AdminModelUsageReport | null>(null);
  const [loginEvents, setLoginEvents] = useState<AdminLoginEventList | null>(null);
  const [paymentOrders, setPaymentOrders] = useState<AdminPaymentOrderList | null>(null);
  const [requestFilters, setRequestFilters] = useState<AdminRequestFilters>({
    page: 1,
    pageSize: 25,
    q: "",
    model: "",
    status: "",
    requestMode: "",
  });
  const [paymentFilters, setPaymentFilters] = useState<AdminPaymentFilters>({
    page: 1,
    pageSize: 25,
    q: "",
    status: "",
    paymentMethod: "",
  });
  const [modelUsageDays, setModelUsageDays] = useState(30);
  const [activityPage, setActivityPage] = useState(1);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setAdminError("");
      setDashboardData(null);
      setUsers([]);
      setRequests(null);
      setModelUsage(null);
      setLoginEvents(null);
      setPaymentOrders(null);
      setRequestDetail(null);
      setSelectedRequestId(null);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const controller = new AbortController();
    setAdminLoading(true);
    setAdminError("");

    const load = async () => {
      if (adminPage === "dashboard") {
        setDashboardData(
          await fetchAdminJson<AdminDashboardData>(apiBaseUrl, "/api/admin/dashboard", controller.signal),
        );
        return;
      }

      if (adminPage === "users") {
        const data = await fetchAdminJson<{ items: AdminUserItem[] }>(
          apiBaseUrl,
          "/api/admin/users?limit=300",
          controller.signal,
        );
        setUsers(data.items ?? []);
        return;
      }

      if (adminPage === "requests") {
        const params = new URLSearchParams({
          page: String(requestFilters.page),
          page_size: String(requestFilters.pageSize),
        });
        if (requestFilters.q.trim()) {
          params.set("q", requestFilters.q.trim());
        }
        if (requestFilters.model.trim()) {
          params.set("model", requestFilters.model.trim());
        }
        if (requestFilters.status.trim()) {
          params.set("status", requestFilters.status.trim());
        }
        if (requestFilters.requestMode.trim()) {
          params.set("request_mode", requestFilters.requestMode.trim());
        }
        const data = await fetchAdminJson<AdminLlmRequestList>(
          apiBaseUrl,
          `/api/admin/llm-requests?${params.toString()}`,
          controller.signal,
        );
        setRequests(data);
        setSelectedRequestId((current) => {
          if (data.items.length === 0) {
            return null;
          }
          if (current == null) {
            return data.items[0].id;
          }
          return data.items.some((item) => item.id === current) ? current : data.items[0].id;
        });
        return;
      }

      if (adminPage === "models") {
        setModelUsage(
          await fetchAdminJson<AdminModelUsageReport>(
            apiBaseUrl,
            `/api/admin/model-usage?days=${modelUsageDays}`,
            controller.signal,
          ),
        );
        return;
      }

      if (adminPage === "payments") {
        const params = new URLSearchParams({
          page: String(paymentFilters.page),
          page_size: String(paymentFilters.pageSize),
        });
        if (paymentFilters.q.trim()) {
          params.set("q", paymentFilters.q.trim());
        }
        if (paymentFilters.status.trim()) {
          params.set("status", paymentFilters.status.trim());
        }
        if (paymentFilters.paymentMethod.trim()) {
          params.set("payment_method", paymentFilters.paymentMethod.trim());
        }
        setPaymentOrders(
          await fetchAdminJson<AdminPaymentOrderList>(
            apiBaseUrl,
            `/api/admin/payment-orders?${params.toString()}`,
            controller.signal,
          ),
        );
        return;
      }

      setLoginEvents(
        await fetchAdminJson<AdminLoginEventList>(
          apiBaseUrl,
          `/api/admin/login-events?page=${activityPage}&page_size=50`,
          controller.signal,
        ),
      );
    };

    void load()
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setAdminError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setAdminLoading(false);
        }
      });

    return () => controller.abort();
  }, [
    activityPage,
    adminPage,
    apiBaseUrl,
    enabled,
    modelUsageDays,
    paymentFilters.page,
    paymentFilters.pageSize,
    paymentFilters.paymentMethod,
    paymentFilters.q,
    paymentFilters.status,
    refreshNonce,
    requestFilters.model,
    requestFilters.page,
    requestFilters.pageSize,
    requestFilters.q,
    requestFilters.requestMode,
    requestFilters.status,
  ]);

  useEffect(() => {
    if (!enabled || adminPage !== "requests" || selectedRequestId == null) {
      setRequestDetail(null);
      return;
    }

    const controller = new AbortController();
    setRequestDetailLoading(true);

    void fetchAdminJson<AdminLlmRequestDetail>(
      apiBaseUrl,
      `/api/admin/llm-requests/${selectedRequestId}`,
      controller.signal,
    )
      .then((data) => setRequestDetail(data))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setAdminError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setRequestDetailLoading(false);
        }
      });

    return () => controller.abort();
  }, [adminPage, apiBaseUrl, enabled, selectedRequestId]);

  function resetAdminState() {
    setAdminPage("dashboard");
    setAdminError("");
    setDashboardData(null);
    setUsers([]);
    setRequests(null);
    setRequestDetail(null);
    setSelectedRequestId(null);
    setModelUsage(null);
    setLoginEvents(null);
    setPaymentOrders(null);
    setRequestFilters({
      page: 1,
      pageSize: 25,
      q: "",
      model: "",
      status: "",
      requestMode: "",
    });
    setPaymentFilters({
      page: 1,
      pageSize: 25,
      q: "",
      status: "",
      paymentMethod: "",
    });
    setModelUsageDays(30);
    setActivityPage(1);
  }

  function refreshAdminData() {
    setRefreshNonce((current) => current + 1);
  }

  async function updateUserRole(userId: number, role: UserRole) {
    setAdminUserRoleUpdatingId(userId);
    setAdminError("");
    try {
      await postAdminJson(apiBaseUrl, `/api/admin/users/${userId}/role`, { role });
      if (refreshSession) {
        await refreshSession();
      }
      refreshAdminData();
    } catch (error) {
      setAdminError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      setAdminUserRoleUpdatingId(null);
    }
  }

  async function actOnPaymentOrder(orderId: number, approve: boolean) {
    setAdminPaymentActingOrderId(orderId);
    setAdminError("");
    try {
      await postAdminJson(
        apiBaseUrl,
        `/api/admin/payment-orders/${orderId}/${approve ? "approve" : "reject"}`,
      );
      refreshAdminData();
    } catch (error) {
      setAdminError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      setAdminPaymentActingOrderId(null);
    }
  }

  return {
    adminPage,
    adminLoading,
    adminError,
    adminUserRoleUpdatingId,
    adminPaymentActingOrderId,
    dashboardData,
    users,
    requests,
    requestDetail,
    requestDetailLoading,
    selectedRequestId,
    modelUsage,
    loginEvents,
    paymentOrders,
    requestFilters,
    paymentFilters,
    modelUsageDays,
    activityPage,
    setAdminPage,
    setSelectedRequestId,
    setRequestFilters,
    setPaymentFilters,
    setModelUsageDays,
    setActivityPage,
    refreshAdminData,
    resetAdminState,
    updateUserRole,
    approvePaymentOrder: (orderId: number) => actOnPaymentOrder(orderId, true),
    rejectPaymentOrder: (orderId: number) => actOnPaymentOrder(orderId, false),
  };
}
