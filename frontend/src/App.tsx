import { useEffect, useRef, useState } from "react";
import { useAdminConsole } from "./app/useAdminConsole";
import { useBillingCenter } from "./app/useBillingCenter";
import { useChatSession } from "./app/useChatSession";
import { useRepairSession } from "./app/useRepairSession";
import { useSidebarLayout } from "./app/useSidebarLayout";
import { MAX_SIDEBAR_WIDTH, MIN_SIDEBAR_WIDTH } from "./app/utils";
import { AuthPage } from "./components/AuthPage";
import { AdminWorkspace } from "./components/admin/AdminWorkspace";
import { AgentWorkspace } from "./components/app/AgentWorkspace";
import { AppHeader } from "./components/app/AppHeader";
import { AppSidebar } from "./components/app/AppSidebar";
import { ChatWorkspace } from "./components/app/ChatWorkspace";
import { SiteMapWidget } from "./components/app/SiteMapWidget";
import { BenchmarkWorkspace } from "./components/benchmark/BenchmarkWorkspace";
import { BillingWorkspace } from "./components/billing/BillingWorkspace";
import { ProfileWorkspace } from "./components/profile/ProfileWorkspace";
import { TeamsWorkspace } from "./components/teams/TeamsWorkspace";
import { copy } from "./i18n";
import type {
  AdminPage,
  AgentHistorySnapshot,
  AuthenticatedUser,
  BenchmarkPage,
  ChatHistorySnapshot,
  HistoryDetail,
  HistorySummary,
  ModelCatalogItem,
  ModelOptionValue,
  OAuthProvider,
  ProfilePage,
  PublicModelCatalog,
  TeamsPage,
  ThemeMode,
  UiLocale,
  WorkspaceMode,
} from "./types";

function App() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
  const [locale, setLocale] = useState<UiLocale>("zh");
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") {
      return "dark";
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [currentUser, setCurrentUser] = useState<AuthenticatedUser | null>(null);
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([]);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [authError, setAuthError] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("agent");
  const [historyItems, setHistoryItems] = useState<HistorySummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deletingHistoryId, setDeletingHistoryId] = useState<number | null>(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState<number | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelCatalogItem[]>([]);
  const [agentModel, setAgentModel] = useState<ModelOptionValue>("");
  const [chatModel, setChatModel] = useState<ModelOptionValue>("");
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [benchmarkPage, setBenchmarkPage] = useState<BenchmarkPage>("projects");
  const [profilePage, setProfilePage] = useState<ProfilePage>("overview");
  const [teamsPage, setTeamsPage] = useState<TeamsPage>("organizations");

  const oauthReturnRef = useRef(false);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const historyRequestRef = useRef(0);

  const {
    sidebarWidthPx,
    sidebarResizing,
    isDesktopLayout,
    handleSidebarResizePointerDown,
    handleSidebarResizeReset,
  } = useSidebarLayout();

  const dict = copy[locale];
  const activeChatModel = availableModels.find((item) => item.value === chatModel) ?? availableModels[0];
  const canAccessAdmin = currentUser?.role === "admin";
  const showUpgrade = currentUser?.role === "basic";

  const repair = useRepairSession({
    apiBaseUrl,
    dict,
    model: agentModel,
    refreshHistoryList: fetchHistoryList,
    selectHistory: setSelectedHistoryId,
    upsertHistoryItem,
  });

  const chat = useChatSession({
    apiBaseUrl,
    model: chatModel,
    refreshHistoryList: fetchHistoryList,
    selectHistory: setSelectedHistoryId,
    upsertHistoryItem,
  });

  const admin = useAdminConsole({
    apiBaseUrl,
    enabled: Boolean(currentUser && canAccessAdmin && workspaceMode === "admin"),
    refreshSession: refreshSessionState,
    refreshModels: fetchModelCatalog,
  });

  const billing = useBillingCenter({
    apiBaseUrl,
    enabled: Boolean(currentUser && workspaceMode === "billing"),
    refreshSession: refreshSessionState,
  });

  function upsertHistoryItem(summary: HistorySummary) {
    setHistoryItems((current) => {
      const filtered = current.filter((item) => item.id !== summary.id);
      return [summary, ...filtered];
    });
  }

  function pickModelValue(
    requested: string | null | undefined,
    items: ModelCatalogItem[],
    fallback: string | null | undefined,
  ) {
    if (requested && items.some((item) => item.value === requested)) {
      return requested;
    }
    if (fallback && items.some((item) => item.value === fallback)) {
      return fallback;
    }
    return items[0]?.value ?? "";
  }

  async function fetchModelCatalog() {
    const response = await fetch(`${apiBaseUrl}/api/models`, {
      credentials: "include",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = (await response.json()) as PublicModelCatalog;
    const items = Array.isArray(data.items) ? data.items : [];
    setAvailableModels(items);
    setAgentModel((current) => pickModelValue(current, items, data.default_repair_model));
    setChatModel((current) => pickModelValue(current, items, data.default_chat_model));
    return data;
  }

  async function fetchSession() {
    const response = await fetch(`${apiBaseUrl}/api/auth/session`, {
      credentials: "include",
    });
    const data = await response.json();
    setCurrentUser(data.authenticated ? (data.user as AuthenticatedUser) : null);
    setOauthProviders((data.oauth_providers ?? []) as OAuthProvider[]);
    return Boolean(data.authenticated);
  }

  async function refreshSessionState() {
    try {
      await fetchSession();
    } catch {
      return;
    }
  }

  async function fetchHistoryList() {
    if (!currentUser) {
      setHistoryItems([]);
      setHistoryLoading(false);
      return;
    }

    const requestId = historyRequestRef.current + 1;
    historyRequestRef.current = requestId;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 8000);
    setHistoryLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/history`, {
        credentials: "include",
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (historyRequestRef.current === requestId) {
        setHistoryItems((data.items ?? []) as HistorySummary[]);
      }
    } catch {
      if (historyRequestRef.current === requestId) {
        setHistoryItems((current) => current);
      }
    } finally {
      window.clearTimeout(timeoutId);
      if (historyRequestRef.current === requestId) {
        setHistoryLoading(false);
      }
    }
  }

  useEffect(() => {
    const storedLocale = window.localStorage.getItem("autorepair-locale") as UiLocale | null;
    setLocale(storedLocale ?? "zh");
  }, []);

  useEffect(() => {
    void fetchModelCatalog().catch(() => {
      setAvailableModels([]);
      setAgentModel("");
      setChatModel("");
    });
  }, [apiBaseUrl]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applySystemTheme = () => {
      setTheme(mediaQuery.matches ? "dark" : "light");
    };

    applySystemTheme();
    mediaQuery.addEventListener("change", applySystemTheme);
    return () => mediaQuery.removeEventListener("change", applySystemTheme);
  }, []);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!userMenuRef.current) {
        return;
      }
      if (!userMenuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false);
      }
    }

    if (userMenuOpen) {
      document.addEventListener("mousedown", handlePointerDown);
    }
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [userMenuOpen]);

  useEffect(() => {
    window.localStorage.setItem("autorepair-locale", locale);
  }, [locale]);

  useEffect(() => {
    if (currentUser) {
      void fetchHistoryList();
      return;
    }
    setHistoryItems([]);
    setSelectedHistoryId(null);
    chat.clearActiveChatHistoryId();
  }, [currentUser]);

  useEffect(() => {
    if (currentUser?.role === "admin") {
      return;
    }
    if (workspaceMode === "admin") {
      setWorkspaceMode("agent");
    }
    admin.resetAdminState();
  }, [currentUser?.role]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    oauthReturnRef.current = params.get("auth_success") === "1";
    if (params.get("auth_error") === "provider_unavailable") {
      setAuthError(dict.oauthUnavailable);
    } else if (params.get("auth_error")) {
      setAuthError(dict.oauthFailed);
    } else if (params.get("auth_success")) {
      setAuthMessage(dict.oauthSuccess);
    }

    if (params.has("auth_error") || params.has("auth_success")) {
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [dict.oauthFailed, dict.oauthSuccess, dict.oauthUnavailable]);

  useEffect(() => {
    let active = true;

    async function loadSession() {
      const isOAuthReturn = oauthReturnRef.current;
      try {
        let authenticated = await fetchSession();
        if (!authenticated && isOAuthReturn) {
          for (let attempt = 0; attempt < 3 && !authenticated; attempt += 1) {
            await new Promise((resolve) => window.setTimeout(resolve, 350));
            authenticated = await fetchSession();
          }
        }
        if (!active) {
          return;
        }
        if (!authenticated && isOAuthReturn) {
          setAuthError(`${dict.oauthFailed} Session was not established.`);
        }
      } catch {
        if (!active) {
          return;
        }
        setCurrentUser(null);
        setOauthProviders([]);
        if (isOAuthReturn) {
          setAuthError(`${dict.oauthFailed} Session check failed.`);
        }
      } finally {
        if (active) {
          setSessionLoading(false);
        }
      }
    }

    void loadSession();
    return () => {
      active = false;
    };
  }, [apiBaseUrl, dict.oauthFailed]);

  async function handleLogout() {
    await fetch(`${apiBaseUrl}/api/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
    setUserMenuOpen(false);
    setCurrentUser(null);
    setWorkspaceMode("agent");
    setHistoryItems([]);
    setSelectedHistoryId(null);
    repair.resetRepairState();
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
    admin.resetAdminState();
  }

  function handleReset() {
    repair.resetRepairState();
    setSelectedHistoryId(null);
  }

  function startNewAgentSession() {
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
    setWorkspaceMode("agent");
    setSelectedHistoryId(null);
    repair.startNewAgentSession();
  }

  function startNewChatSession() {
    setWorkspaceMode("chat");
    setSelectedHistoryId(null);
    repair.resetRepairState();
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
  }

  function openAdminConsole() {
    if (!canAccessAdmin) {
      return;
    }
    setUserMenuOpen(false);
    admin.setAdminPage("dashboard");
    setWorkspaceMode("admin");
    admin.refreshAdminData();
  }

  function openBillingCenter() {
    if (!currentUser) {
      return;
    }
    setUserMenuOpen(false);
    setWorkspaceMode("billing");
    void billing.refreshBillingData();
  }

  function startNewBenchmarkSession() {
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
    repair.resetRepairState();
    setSelectedHistoryId(null);
    setWorkspaceMode("benchmark");
  }

  function startNewProfileSession() {
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
    repair.resetRepairState();
    setSelectedHistoryId(null);
    setWorkspaceMode("profile");
  }

  function startNewTeamsSession() {
    chat.resetChatState({ abort: true, clearActiveHistoryId: true });
    repair.resetRepairState();
    setSelectedHistoryId(null);
    setWorkspaceMode("teams");
  }

  function selectAdminPage(page: AdminPage) {
    if (!canAccessAdmin) {
      return;
    }
    setWorkspaceMode("admin");
    admin.setAdminPage(page);
  }

  function handleAuthenticated(user: AuthenticatedUser, providers: OAuthProvider[]) {
    setCurrentUser(user);
    setOauthProviders(providers);
    setAuthError("");
    setAuthMessage("");
    setUserMenuOpen(false);
  }

  async function handleHistoryOpen(historyId: number) {
    try {
      const response = await fetch(`${apiBaseUrl}/api/history/${historyId}`, {
        credentials: "include",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const detail = (await response.json()) as HistoryDetail;
      setSelectedHistoryId(detail.id);

      if (detail.mode === "chat") {
        const snapshot = detail.snapshot as ChatHistorySnapshot;
        setWorkspaceMode("chat");
        if (detail.model) {
          setChatModel((current) => pickModelValue(detail.model, availableModels, current));
        }
        chat.loadHistorySnapshot(detail.id, snapshot);
        repair.prepareForChatHistoryView();
        return;
      }

      const snapshot = detail.snapshot as AgentHistorySnapshot;
      setWorkspaceMode("agent");
      chat.resetChatState({ clearActiveHistoryId: true });
      if (snapshot.model) {
        setAgentModel((current) => pickModelValue(snapshot.model, availableModels, current));
      }
      repair.loadHistorySnapshot(snapshot);
    } catch (error) {
      repair.setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleDeleteHistory(historyId: number) {
    if (!window.confirm(dict.historyDeleteConfirm)) {
      return;
    }

    setDeletingHistoryId(historyId);
    try {
      const response = await fetch(`${apiBaseUrl}/api/history/${historyId}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!response.ok) {
        const payload = await response.text();
        throw new Error(payload || `HTTP ${response.status}`);
      }

      setHistoryItems((current) => current.filter((item) => item.id !== historyId));
      if (selectedHistoryId === historyId) {
        setSelectedHistoryId(null);
      }
      if (chat.activeChatHistoryId === historyId) {
        chat.clearActiveChatHistoryId();
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : dict.historyDeleteFailed;
      repair.setErrorMessage(message);
      chat.setChatError(message);
    } finally {
      setDeletingHistoryId(null);
    }
  }

  const workspaceMainClass = `min-h-0 min-w-0 h-full app-fade-in ${!isDesktopLayout ? "flex-1" : ""}`.trim();

  const statusText =
    repair.status === "streaming"
      ? dict.statusStreaming
      : repair.status === "done"
        ? dict.statusDone
        : repair.status === "error"
          ? dict.statusError
          : dict.statusIdle;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(255,133,82,0.2),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(39,111,191,0.22),_transparent_30%),linear-gradient(180deg,#f7f3ea_0%,#efe7d8_100%)] text-slate-900 transition-colors dark:bg-[radial-gradient(circle_at_top_left,_rgba(255,133,82,0.12),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(39,111,191,0.18),_transparent_30%),linear-gradient(180deg,#171411_0%,#0f1118_100%)] dark:text-white">
      <div
        className={`mx-auto max-w-[1500px] px-3 pt-3 sm:px-4 lg:px-6 ${
          !sessionLoading && !currentUser
            ? "h-screen overflow-hidden pb-3"
            : !sessionLoading && currentUser
              ? "flex h-[100dvh] max-h-[100dvh] min-h-0 flex-col overflow-hidden pb-1.5"
              : "pb-4"
        }`}
      >
        <AppHeader
          canAccessAdmin={Boolean(canAccessAdmin)}
          currentUser={currentUser}
          copy={dict}
          showUpgrade={Boolean(showUpgrade)}
          theme={theme}
          userMenuOpen={userMenuOpen}
          userMenuRef={userMenuRef}
          onLogout={handleLogout}
          onOpenAdmin={openAdminConsole}
          onOpenBilling={openBillingCenter}
          onToggleLocale={() => setLocale((current) => (current === "zh" ? "en" : "zh"))}
          onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          onToggleUserMenu={() => setUserMenuOpen((current) => !current)}
        />

        {sessionLoading ? (
          <div className="grid min-h-[82vh] place-items-center">
            <div className="rounded-[32px] border border-black/5 bg-white/50 px-8 py-6 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
              <div className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-white/45">
                {dict.authLoading}
              </div>
            </div>
          </div>
        ) : null}

        {!sessionLoading && !currentUser ? (
          <>
            <AuthPage
              apiBaseUrl={apiBaseUrl}
              locale={locale}
              oauthProviders={oauthProviders}
              onAuthenticated={handleAuthenticated}
              initialError={authError}
              initialMessage={authMessage}
              onResetNotice={() => {
                setAuthError("");
                setAuthMessage("");
              }}
              copy={dict}
            />
            <SiteMapWidget apiBaseUrl={apiBaseUrl} copy={dict} />
          </>
        ) : null}

        {!sessionLoading && currentUser ? (
          <>
            <main
              className={
                isDesktopLayout
                  ? "mt-2 grid min-h-0 flex-1 items-stretch gap-0 overflow-hidden"
                  : "mt-2 flex min-h-0 flex-1 flex-col gap-3 overflow-hidden"
              }
              style={
                isDesktopLayout
                  ? { gridTemplateColumns: `${sidebarWidthPx}px 6px minmax(0,1fr)` }
                  : undefined
              }
            >
              <AppSidebar
                adminPage={admin.adminPage}
                canAccessAdmin={Boolean(canAccessAdmin)}
                copy={dict}
                deletingHistoryId={deletingHistoryId}
                historyItems={historyItems}
                historyLoading={historyLoading}
                isDesktopLayout={isDesktopLayout}
                locale={locale}
                selectedHistoryId={selectedHistoryId}
                workspaceMode={workspaceMode}
                onDeleteHistory={(historyId) => {
                  void handleDeleteHistory(historyId);
                }}
                onOpenHistory={(historyId) => {
                  void handleHistoryOpen(historyId);
                }}
                onSelectAdminPage={selectAdminPage}
                onStartNewAgentSession={startNewAgentSession}
                onStartNewAdminSession={openAdminConsole}
                onStartNewChatSession={startNewChatSession}
                onStartNewBenchmarkSession={startNewBenchmarkSession}
                onStartNewProfileSession={startNewProfileSession}
                onStartNewTeamsSession={startNewTeamsSession}
              />

              {isDesktopLayout ? (
                <div
                  role="separator"
                  aria-orientation="vertical"
                  aria-valuemin={MIN_SIDEBAR_WIDTH}
                  aria-valuemax={MAX_SIDEBAR_WIDTH}
                  aria-valuenow={Math.round(sidebarWidthPx)}
                  aria-label={
                    locale === "zh"
                      ? "拖动调整侧栏宽度，双击恢复默认"
                      : "Drag to resize sidebar; double-click to reset"
                  }
                  onPointerDown={handleSidebarResizePointerDown}
                  onDoubleClick={handleSidebarResizeReset}
                  className={`relative z-20 w-[6px] shrink-0 cursor-col-resize touch-none select-none ${
                    sidebarResizing ? "bg-black/[0.06] dark:bg-white/[0.08]" : ""
                  }`}
                >
                  <div className="pointer-events-none absolute inset-y-3 left-1/2 w-px -translate-x-1/2 rounded-full bg-black/15 dark:bg-white/20" />
                </div>
              ) : null}

              {workspaceMode === "admin" ? (
                <AdminWorkspace
                  activityPage={admin.activityPage}
                  adminError={admin.adminError}
                  adminLoading={admin.adminLoading}
                  adminModelMutatingId={admin.adminModelMutatingId}
                  adminPage={admin.adminPage}
                  adminPaymentActingOrderId={admin.adminPaymentActingOrderId}
                  adminUserRoleUpdatingId={admin.adminUserRoleUpdatingId}
                  apiBaseUrl={apiBaseUrl}
                  copy={dict}
                  dashboardData={admin.dashboardData}
                  loginEvents={admin.loginEvents}
                  modelConfigs={admin.modelConfigs}
                  modelUsage={admin.modelUsage}
                  modelUsageDays={admin.modelUsageDays}
                  paymentFilters={admin.paymentFilters}
                  paymentOrders={admin.paymentOrders}
                  requestDetail={admin.requestDetail}
                  requestDetailLoading={admin.requestDetailLoading}
                  requestFilters={admin.requestFilters}
                  requests={admin.requests}
                  selectedRequestId={admin.selectedRequestId}
                  users={admin.users}
                  workspaceMainClass={workspaceMainClass}
                  onActivityPageChange={admin.setActivityPage}
                  onCreateModelConfig={admin.createAdminModelConfig}
                  onDeleteModelConfig={admin.deleteAdminModelConfig}
                  onModelUsageDaysChange={admin.setModelUsageDays}
                  onApprovePaymentOrder={(orderId) => {
                    void admin.approvePaymentOrder(orderId);
                  }}
                  onPaymentFiltersChange={admin.setPaymentFilters}
                  onRefresh={admin.refreshAdminData}
                  onRequestFiltersChange={admin.setRequestFilters}
                  onSelectRequest={admin.setSelectedRequestId}
                  onUpdateModelConfig={admin.updateAdminModelConfig}
                  onUpdateUserRole={(userId, role) => {
                    void admin.updateUserRole(userId, role);
                  }}
                  onRejectPaymentOrder={(orderId) => {
                    void admin.rejectPaymentOrder(orderId);
                  }}
                />
              ) : workspaceMode === "billing" ? (
                <BillingWorkspace
                  activeOrderId={billing.activeOrderId}
                  activeOrderSession={billing.activeOrderSession}
                  billingActing={billing.billingActing}
                  billingData={billing.billingData}
                  billingError={billing.billingError}
                  billingLoading={billing.billingLoading}
                  copy={dict}
                  currentUser={currentUser}
                  workspaceMainClass={workspaceMainClass}
                  onCreateOrder={billing.createOrder}
                  onRefresh={() => {
                    void billing.refreshBillingData();
                  }}
                  onRefreshOrderSession={billing.refreshOrderSession}
                  onSelectOrder={billing.setActiveOrderId}
                />
              ) : workspaceMode === "benchmark" ? (
                <BenchmarkWorkspace
                  apiBaseUrl={apiBaseUrl}
                  copy={dict}
                  page={benchmarkPage}
                  modelOptions={availableModels}
                  model={agentModel}
                  onModelChange={setAgentModel}
                  onPageChange={setBenchmarkPage}
                  workspaceMainClass={workspaceMainClass}
                />
              ) : workspaceMode === "profile" ? (
                <ProfileWorkspace
                  apiBaseUrl={apiBaseUrl}
                  copy={dict}
                  currentUser={currentUser}
                  modelOptions={availableModels}
                  page={profilePage}
                  onPageChange={setProfilePage}
                  workspaceMainClass={workspaceMainClass}
                />
              ) : workspaceMode === "teams" ? (
                <TeamsWorkspace
                  apiBaseUrl={apiBaseUrl}
                  copy={dict}
                  page={teamsPage}
                  onPageChange={setTeamsPage}
                  workspaceMainClass={workspaceMainClass}
                />
              ) : workspaceMode === "agent" ? (
                <AgentWorkspace
                  agentSourceType={repair.agentSourceType}
                  code={repair.code}
                  copy={dict}
                  diffApplied={repair.diffApplied}
                  diffDecisionMessage={repair.diffDecisionMessage}
                  entrypointPath={repair.entrypointPath}
                  errorMessage={repair.errorMessage}
                  finalDiff={repair.finalDiff}
                  finalMessage={repair.finalMessage}
                  githubRef={repair.githubRef}
                  githubRepoUrl={repair.githubRepoUrl}
                  inputText={repair.inputText}
                  userPrompt={repair.userPrompt}
                  testCases={repair.testCases}
                  language={repair.language}
                  locale={locale}
                  model={agentModel}
                  modelOptions={availableModels}
                  projectActionLoading={repair.projectActionLoading}
                  projectSubdir={repair.projectSubdir}
                  projectEntrypointOptions={repair.projectEntrypointOptions}
                  projectFilesLoading={repair.projectFilesLoading}
                  languageSupported={repair.languageSupported}
                  runResult={repair.runResult}
                  stages={repair.stages}
                  status={repair.status}
                  statusText={statusText}
                  verificationPassed={repair.verificationPassed}
                  workspaceMainClass={workspaceMainClass}
                  zipFileName={repair.zipFileName}
                  onApplyDiff={repair.handleApplyDiff}
                  onCodeChange={repair.setCode}
                  onEntrypointChange={repair.setEntrypointPath}
                  onGithubRefChange={repair.setGithubRef}
                  onGithubRepoUrlChange={repair.setGithubRepoUrl}
                  onInputTextChange={repair.setInputText}
                  onUserPromptChange={repair.setUserPrompt}
                  onTestCasesChange={repair.setTestCases}
                  onLanguageChange={repair.setLanguage}
                  onModelChange={setAgentModel}
                  onProjectSubdirChange={repair.setProjectSubdir}
                  onReset={handleReset}
                  onSend={repair.handleSend}
                  onSkipDiff={repair.skipApplyingDiff}
                  onSourceTypeChange={repair.setAgentSourceType}
                  onStop={repair.stopStreaming}
                  onZipSelected={repair.handleZipSelected}
                />
              ) : (
                <ChatWorkspace
                  activeModelLabel={activeChatModel?.label ?? chatModel}
                  chatError={chat.chatError}
                  chatInput={chat.chatInput}
                  chatMessages={chat.chatMessages}
                  chatReasoningStreaming={chat.chatReasoningStreaming}
                  chatStreamingText={chat.chatStreamingText}
                  chatThinking={chat.chatThinking}
                  copy={dict}
                  isDesktopLayout={isDesktopLayout}
                  model={chatModel}
                  modelOptions={availableModels}
                  onChatInputChange={chat.setChatInput}
                  onModelChange={setChatModel}
                  onSend={chat.handleChatSend}
                />
              )}
            </main>

            {dict.footer.trim() ? (
              <footer className="mt-2 shrink-0 text-center text-[11px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                {dict.footer}
              </footer>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

export default App;
