import type { Dispatch, SetStateAction } from "react";
import type { AdminPaymentFilters, AdminRequestFilters } from "../../app/useAdminConsole";
import { getUserInitials } from "../../app/utils";
import type { AppCopy } from "../../i18n";
import type {
  AuthenticatedUser,
  AdminDashboardData,
  AdminLoginEventList,
  AdminModelConfigItem,
  AdminModelConfigPayload,
  AdminLlmRequestDetail,
  AdminLlmRequestList,
  AdminModelUsageReport,
  AdminPaymentOrderList,
  AdminPage,
  AdminUserItem,
} from "../../types";
import {
  ActivityIcon,
  AdminIcon,
  BenchmarkIcon,
  DashboardIcon,
  ModelsIcon,
  PaymentIcon,
  RefreshIcon,
  RequestsIcon,
  UsersIcon,
} from "../app/AppIcons";
import { AdminActivityPage } from "./AdminActivityPage";
import { AdminBenchmarkPage } from "./AdminBenchmarkPage";
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
  apiBaseUrl: string;
  adminModelMutatingId: number | "create" | null;
  adminPaymentActingOrderId: number | null;
  adminUserRoleUpdatingId: number | null;
  copy: AppCopy;
  currentUser: AuthenticatedUser;
  dashboardData: AdminDashboardData | null;
  loginEvents: AdminLoginEventList | null;
  modelConfigs: AdminModelConfigItem[];
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
  onActivityPageChange: (page: number) => void;
  onModelUsageDaysChange: (days: number) => void;
  onCreateModelConfig: (payload: AdminModelConfigPayload) => Promise<void>;
  onDeleteModelConfig: (modelConfigId: number) => Promise<void>;
  onApprovePaymentOrder: (orderId: number) => void;
  onPaymentFiltersChange: Dispatch<SetStateAction<AdminPaymentFilters>>;
  onRefresh: () => void;
  onRequestFiltersChange: Dispatch<SetStateAction<AdminRequestFilters>>;
  onExitAdmin: () => void;
  onSelectAdminPage: (page: AdminPage) => void;
  onSelectRequest: (requestId: number | null) => void;
  onUpdateModelConfig: (modelConfigId: number, payload: AdminModelConfigPayload) => Promise<void>;
  onUpdateUserRole: (userId: number, role: AdminUserItem["role"]) => void;
  onRejectPaymentOrder: (orderId: number) => void;
}

export function AdminWorkspace({
  activityPage,
  adminError,
  adminLoading,
  adminPage,
  adminModelMutatingId,
  adminPaymentActingOrderId,
  adminUserRoleUpdatingId,
  apiBaseUrl,
  copy,
  currentUser,
  dashboardData,
  loginEvents,
  modelConfigs,
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
  onActivityPageChange,
  onCreateModelConfig,
  onDeleteModelConfig,
  onModelUsageDaysChange,
  onApprovePaymentOrder,
  onPaymentFiltersChange,
  onRefresh,
  onRequestFiltersChange,
  onExitAdmin,
  onSelectAdminPage,
  onSelectRequest,
  onUpdateModelConfig,
  onUpdateUserRole,
  onRejectPaymentOrder,
}: AdminWorkspaceProps) {
  const navItems: Array<{ page: AdminPage; label: string; icon: typeof DashboardIcon }> = [
    { page: "dashboard", label: copy.adminDashboard, icon: DashboardIcon },
    { page: "users", label: copy.adminUsers, icon: UsersIcon },
    { page: "requests", label: copy.adminRequests, icon: RequestsIcon },
    { page: "models", label: copy.adminModels, icon: ModelsIcon },
    { page: "activity", label: copy.adminActivity, icon: ActivityIcon },
    { page: "payments", label: copy.adminPayments, icon: PaymentIcon },
    { page: "benchmark", label: copy.adminBenchmark, icon: BenchmarkIcon },
  ];

  const pageMeta: Record<AdminPage, { title: string; hint: string }> = {
    dashboard: { title: copy.adminDashboardTitle, hint: copy.adminDashboardHint },
    users: { title: copy.adminUsersTitle, hint: copy.adminUsersHint },
    requests: { title: copy.adminRequestsTitle, hint: copy.adminRequestsHint },
    models: { title: copy.adminModelsTitle, hint: copy.adminModelsHint },
    activity: { title: copy.adminActivityTitle, hint: copy.adminActivityHint },
    payments: { title: copy.adminPaymentsTitle, hint: copy.adminPaymentsHint },
    benchmark: { title: copy.adminBenchmarkTitle, hint: copy.adminBenchmarkHint },
  };

  const activeMeta = pageMeta[adminPage];

  return (
    <section className="flex h-full w-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-none bg-slate-100 text-slate-900 dark:bg-[#0b1020] dark:text-white lg:flex-row">
      <aside className="flex w-full shrink-0 flex-col gap-4 border-b border-black/5 bg-white/78 px-4 py-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.04] lg:w-[228px] lg:border-b-0 lg:border-r">
        <div className="rounded-[28px] bg-slate-950 px-4 py-4 text-white dark:bg-white dark:text-slate-950">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-white/10 dark:bg-slate-950/10">
              <AdminIcon />
            </div>
            <div className="min-w-0">
              <div className="text-[11px] uppercase tracking-[0.18em] text-white/60 dark:text-slate-950/55">
                {copy.adminOpen}
              </div>
              <div className="truncate text-lg font-semibold">{copy.title}</div>
            </div>
          </div>
        </div>

        <div className="flex gap-2 overflow-x-auto pb-1 lg:flex-1 lg:flex-col lg:overflow-y-auto lg:pb-0">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = adminPage === item.page;
            return (
              <button
                key={item.page}
                onClick={() => onSelectAdminPage(item.page)}
                className={`flex h-11 w-[136px] shrink-0 items-center gap-3 rounded-2xl border px-3 text-left text-sm font-medium transition lg:w-full ${
                  active
                    ? "border-slate-900 bg-slate-900 text-white dark:border-white dark:bg-white dark:text-slate-950"
                    : "border-black/5 bg-white/70 text-slate-600 hover:border-black/10 hover:bg-white hover:text-slate-900 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/65 dark:hover:bg-white/[0.06] dark:hover:text-white"
                }`}
              >
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-xl bg-black/[0.05] dark:bg-white/[0.06]">
                  <Icon />
                </span>
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="hidden rounded-[24px] border border-black/5 bg-white/70 p-3 dark:border-white/10 dark:bg-white/[0.03] lg:block">
          <div className="flex items-center gap-3">
            {currentUser.avatar_url ? (
              <img
                src={currentUser.avatar_url}
                alt={currentUser.display_name}
                className="h-10 w-10 rounded-full object-cover"
              />
            ) : (
              <div className="grid h-10 w-10 place-items-center rounded-full bg-slate-900 text-sm font-semibold text-white dark:bg-white dark:text-slate-950">
                {getUserInitials(currentUser)}
              </div>
            )}
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-slate-900 dark:text-white">
                {currentUser.display_name}
              </div>
              <div className="truncate text-xs text-slate-500 dark:text-white/45">{currentUser.email}</div>
            </div>
          </div>
        </div>
      </aside>

      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 flex-col gap-3 border-b border-black/5 bg-white/75 px-4 py-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.03] lg:flex-row lg:items-center lg:justify-between lg:px-6">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400 dark:text-white/35">
              {copy.adminOpen}
            </div>
            <div className="mt-1 truncate text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">
              {activeMeta.title}
            </div>
            <div className="mt-1 text-sm text-slate-500 dark:text-white/45">{activeMeta.hint}</div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onExitAdmin}
              className="rounded-full border border-black/10 bg-white/80 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-slate-950 dark:border-white/10 dark:bg-white/[0.04] dark:text-white/75 dark:hover:text-white"
            >
              {copy.adminBackToWorkspace}
            </button>
            <button
              onClick={onRefresh}
              disabled={adminLoading}
              aria-label={copy.adminRefresh}
              title={copy.adminRefresh}
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-black/10 bg-slate-900 text-white transition hover:bg-slate-700 dark:border-white/10 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85 ${
                adminLoading ? "cursor-wait opacity-60" : ""
              }`}
            >
              <span className={adminLoading ? "animate-spin" : ""}>
                <RefreshIcon />
              </span>
            </button>
          </div>
        </header>

        {adminError ? (
          <div className="mx-4 mt-4 rounded-[20px] border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200 lg:mx-6">
            {adminError}
          </div>
        ) : null}

        <div className="min-h-0 w-full min-w-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
          <div key={adminPage} className="app-fade-in w-full min-w-0">
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
                modelConfigs={modelConfigs}
                modelUsage={modelUsage}
                mutatingModelId={adminModelMutatingId}
                onCreateModelConfig={onCreateModelConfig}
                onDeleteModelConfig={onDeleteModelConfig}
                onDaysChange={onModelUsageDaysChange}
                onUpdateModelConfig={onUpdateModelConfig}
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
            ) : adminPage === "benchmark" ? (
              <AdminBenchmarkPage apiBaseUrl={apiBaseUrl} copy={copy} />
            ) : (
              <AdminActivityPage
                activityPage={activityPage}
                copy={copy}
                loginEvents={loginEvents}
                onPageChange={onActivityPageChange}
              />
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
