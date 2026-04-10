import type { Dispatch, SetStateAction } from "react";
import type { AdminPaymentFilters, AdminRequestFilters } from "../../app/useAdminConsole";
import type { AppCopy } from "../../i18n";
import type {
  AdminDashboardData,
  AdminLoginEventList,
  AdminLlmRequestDetail,
  AdminLlmRequestList,
  AdminModelUsageReport,
  AdminPaymentOrderList,
  AdminPage,
  AdminUserItem,
} from "../../types";
import { AdminActivityPage } from "./AdminActivityPage";
import { AdminDashboardPage } from "./AdminDashboardPage";
import { AdminModelsPage } from "./AdminModelsPage";
import { AdminPaymentsPage } from "./AdminPaymentsPage";
import { AdminRequestsPage } from "./AdminRequestsPage";
import { AdminUsersPage } from "./AdminUsersPage";

interface AdminWorkspaceProps {
  activityPage: number;
  adminError: string;
  adminLoading: boolean;
  adminPage: AdminPage;
  adminPaymentActingOrderId: number | null;
  adminUserRoleUpdatingId: number | null;
  copy: AppCopy;
  dashboardData: AdminDashboardData | null;
  loginEvents: AdminLoginEventList | null;
  modelUsage: AdminModelUsageReport | null;
  modelUsageDays: number;
  paymentFilters: AdminPaymentFilters;
  paymentOrders: AdminPaymentOrderList | null;
  requestDetail: AdminLlmRequestDetail | null;
  requestDetailLoading: boolean;
  requestFilters: AdminRequestFilters;
  requests: AdminLlmRequestList | null;
  selectedRequestId: number | null;
  users: AdminUserItem[];
  workspaceMainClass: string;
  onActivityPageChange: (page: number) => void;
  onModelUsageDaysChange: (days: number) => void;
  onApprovePaymentOrder: (orderId: number) => void;
  onPaymentFiltersChange: Dispatch<SetStateAction<AdminPaymentFilters>>;
  onRefresh: () => void;
  onRequestFiltersChange: Dispatch<SetStateAction<AdminRequestFilters>>;
  onSelectRequest: (requestId: number) => void;
  onUpdateUserRole: (userId: number, role: AdminUserItem["role"]) => void;
  onRejectPaymentOrder: (orderId: number) => void;
}

export function AdminWorkspace({
  activityPage,
  adminError,
  adminLoading,
  adminPage,
  adminPaymentActingOrderId,
  adminUserRoleUpdatingId,
  copy,
  dashboardData,
  loginEvents,
  modelUsage,
  modelUsageDays,
  paymentFilters,
  paymentOrders,
  requestDetail,
  requestDetailLoading,
  requestFilters,
  requests,
  selectedRequestId,
  users,
  workspaceMainClass,
  onActivityPageChange,
  onModelUsageDaysChange,
  onApprovePaymentOrder,
  onPaymentFiltersChange,
  onRefresh,
  onRequestFiltersChange,
  onSelectRequest,
  onUpdateUserRole,
  onRejectPaymentOrder,
}: AdminWorkspaceProps) {
  const pageMeta: Record<AdminPage, { eyebrow: string; title: string; hint: string }> = {
    dashboard: {
      eyebrow: copy.adminDashboard,
      title: copy.adminDashboardTitle,
      hint: copy.adminDashboardHint,
    },
    users: {
      eyebrow: copy.adminUsers,
      title: copy.adminUsersTitle,
      hint: copy.adminUsersHint,
    },
    requests: {
      eyebrow: copy.adminRequests,
      title: copy.adminRequestsTitle,
      hint: copy.adminRequestsHint,
    },
    models: {
      eyebrow: copy.adminModels,
      title: copy.adminModelsTitle,
      hint: copy.adminModelsHint,
    },
    activity: {
      eyebrow: copy.adminActivity,
      title: copy.adminActivityTitle,
      hint: copy.adminActivityHint,
    },
    payments: {
      eyebrow: copy.adminPayments,
      title: copy.adminPaymentsTitle,
      hint: copy.adminPaymentsHint,
    },
  };

  const activeMeta = pageMeta[adminPage];

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/72 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow ${workspaceMainClass}`}
    >
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            {activeMeta.title}
          </div>
        </div>
        <button
          onClick={onRefresh}
          className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
            adminLoading
              ? "cursor-wait bg-slate-300 text-slate-700 dark:bg-white/20 dark:text-white/75"
              : "bg-slate-900 text-white hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
          }`}
        >
          {copy.adminRefresh}
        </button>
      </div>

      {adminError ? (
        <div className="mt-3 rounded-[20px] border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
          {adminError}
        </div>
      ) : null}

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto pr-1">
        {adminPage === "dashboard" ? (
          <AdminDashboardPage copy={copy} data={dashboardData} />
        ) : adminPage === "users" ? (
          <AdminUsersPage
            copy={copy}
            users={users}
            updatingUserId={adminUserRoleUpdatingId}
            onUpdateUserRole={onUpdateUserRole}
          />
        ) : adminPage === "requests" ? (
          <AdminRequestsPage
            copy={copy}
            requestDetail={requestDetail}
            requestDetailLoading={requestDetailLoading}
            requestFilters={requestFilters}
            requests={requests}
            selectedRequestId={selectedRequestId}
            setRequestFilters={onRequestFiltersChange}
            onSelectRequest={onSelectRequest}
          />
        ) : adminPage === "models" ? (
          <AdminModelsPage
            copy={copy}
            days={modelUsageDays}
            modelUsage={modelUsage}
            onDaysChange={onModelUsageDaysChange}
          />
        ) : adminPage === "payments" ? (
          <AdminPaymentsPage
            actingOrderId={adminPaymentActingOrderId}
            copy={copy}
            filters={paymentFilters}
            orders={paymentOrders}
            setFilters={onPaymentFiltersChange}
            onApprove={onApprovePaymentOrder}
            onReject={onRejectPaymentOrder}
          />
        ) : (
          <AdminActivityPage
            activityPage={activityPage}
            copy={copy}
            loginEvents={loginEvents}
            onPageChange={onActivityPageChange}
          />
        )}
      </div>
    </section>
  );
}
