import { useCallback, useEffect, useState } from "react";
import type { AppCopy } from "../../i18n";
import type {
  AuthenticatedUser,
  CreditWalletSnapshot,
  ModelCatalogItem,
  ProfilePage,
  ProfileSnapshot,
  UserApiToken,
  UserPreferences,
} from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminMetricCard,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminNumber,
  toStatusTone,
} from "../admin/AdminCommon";

interface ProfileWorkspaceProps {
  apiBaseUrl: string;
  copy: AppCopy;
  currentUser: AuthenticatedUser;
  modelOptions: ModelCatalogItem[];
  page: ProfilePage;
  onPageChange: (page: ProfilePage) => void;
  workspaceMainClass: string;
}

export function ProfileWorkspace({
  apiBaseUrl,
  copy,
  currentUser,
  modelOptions,
  page,
  onPageChange,
  workspaceMainClass,
}: ProfileWorkspaceProps) {
  const [snapshot, setSnapshot] = useState<ProfileSnapshot | null>(null);
  const [wallet, setWallet] = useState<CreditWalletSnapshot | null>(null);
  const [error, setError] = useState("");
  const [savingPrefs, setSavingPrefs] = useState(false);
  const [creatingToken, setCreatingToken] = useState(false);
  const [newTokenName, setNewTokenName] = useState("");
  const [newlyCreatedToken, setNewlyCreatedToken] = useState<UserApiToken | null>(null);
  const [prefsDraft, setPrefsDraft] = useState<UserPreferences | null>(null);

  const fetchOverview = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/profile/overview`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as ProfileSnapshot;
      setSnapshot(data);
      setPrefsDraft(data.preferences);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  const fetchWallet = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/wallet/summary`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setWallet((await response.json()) as CreditWalletSnapshot);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void fetchOverview();
    void fetchWallet();
  }, [fetchOverview, fetchWallet]);

  async function handleSavePreferences() {
    if (!prefsDraft) return;
    setSavingPrefs(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/profile/preferences`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prefsDraft),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      await fetchOverview();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSavingPrefs(false);
    }
  }

  async function handleCreateToken() {
    if (!newTokenName.trim()) return;
    setCreatingToken(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/profile/api-tokens`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token_name: newTokenName.trim(), scope: "repair" }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      const payload = await response.json();
      setNewlyCreatedToken(payload.token as UserApiToken);
      setNewTokenName("");
      await fetchOverview();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setCreatingToken(false);
    }
  }

  async function handleRevokeToken(tokenId: number) {
    const response = await fetch(`${apiBaseUrl}/api/profile/api-tokens/${tokenId}`, {
      method: "DELETE",
      credentials: "include",
    });
    if (!response.ok) return;
    await fetchOverview();
  }

  const preferences = prefsDraft;
  const overview = snapshot?.overview;
  const walletSummary = snapshot?.wallet ?? wallet?.wallet ?? null;
  const tokens = snapshot?.api_tokens ?? [];

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 ${workspaceMainClass}`}
    >
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.profileTitle}
          </div>
          <div className="hidden text-xs text-slate-500 dark:text-white/45 sm:block">
            {copy.profileHint}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {(["overview", "wallet", "preferences", "api_tokens"] as ProfilePage[]).map((tab) => (
            <button
              key={tab}
              onClick={() => onPageChange(tab)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                page === tab
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                  : "bg-black/[0.04] text-slate-600 hover:bg-black/[0.07] dark:bg-white/[0.05] dark:text-white/70 dark:hover:bg-white/[0.09]"
              }`}
            >
              {tab === "overview"
                ? copy.profileOverview
                : tab === "wallet"
                  ? copy.profileWallet
                  : tab === "preferences"
                    ? copy.profilePreferences
                    : copy.profileApiTokens}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <div className="mt-2 rounded-2xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto app-fade-in">
        {page === "overview" ? (
          <div className="space-y-2">
            <AdminSurface>
              <AdminSectionTitle title={copy.profileOverview} />
              <div className="mt-2 grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
                <AdminMetricCard
                  label={copy.profileBalance}
                  value={formatAdminNumber(walletSummary?.balance_credits ?? 0)}
                  tone="emerald"
                />
                <AdminMetricCard
                  label={copy.profileRepairHistories}
                  value={formatAdminNumber(overview?.total_repair_sessions ?? 0)}
                />
                <AdminMetricCard
                  label={copy.profileChatHistories}
                  value={formatAdminNumber(overview?.total_chat_sessions ?? 0)}
                />
                <AdminMetricCard
                  label={copy.profileBenchmarkRuns}
                  value={formatAdminNumber(overview?.total_benchmark_runs ?? 0)}
                />
                <AdminMetricCard
                  label="Tokens"
                  value={formatAdminNumber(overview?.lifetime_tokens ?? 0)}
                  tone="sky"
                />
              </div>
            </AdminSurface>

            <AdminSurface>
              <AdminSectionTitle title={copy.signedInAs} />
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <div className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.03]">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.emailLabel}
                  </div>
                  <div className="mt-1 text-sm font-medium">{currentUser.email}</div>
                </div>
                <div className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.03]">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.nameLabel}
                  </div>
                  <div className="mt-1 text-sm font-medium">{currentUser.display_name}</div>
                </div>
                <div className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.03]">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.adminRole}
                  </div>
                  <div className="mt-1 text-sm font-medium capitalize">{currentUser.role}</div>
                </div>
                <div className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.03]">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.adminLastLogin}
                  </div>
                  <div className="mt-1 text-sm font-medium">
                    {formatAdminDateTime(currentUser.last_login_at)}
                  </div>
                </div>
              </div>
            </AdminSurface>
          </div>
        ) : null}

        {page === "wallet" ? (
          <AdminSurface>
            <AdminSectionTitle
              title={copy.walletTitle}
              hint={copy.walletHint}
              actions={
                <button
                  onClick={fetchWallet}
                  className="rounded-full bg-black/[0.04] px-3 py-1 text-xs dark:bg-white/[0.06]"
                >
                  {copy.adminRefresh}
                </button>
              }
            />
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              <AdminMetricCard
                label={copy.walletBalance}
                value={formatAdminNumber(walletSummary?.balance_credits ?? 0)}
                tone="emerald"
              />
              <AdminMetricCard
                label={copy.profileTotalSpent}
                value={formatAdminNumber(walletSummary?.lifetime_spent ?? 0)}
                tone="amber"
              />
              <AdminMetricCard
                label={copy.profileTotalGranted}
                value={formatAdminNumber(walletSummary?.lifetime_earned ?? 0)}
                tone="sky"
              />
            </div>

            <div className="mt-3 grid gap-2 lg:grid-cols-[1.3fr,1fr]">
              <AdminSurface>
                <AdminSectionTitle title={copy.walletRecent} />
                {wallet && wallet.transactions.length > 0 ? (
                  <div className="mt-2 max-h-[20rem] overflow-y-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                        <tr>
                          <th className="py-1 pr-2">Date</th>
                          <th className="py-1 pr-2">Reason</th>
                          <th className="py-1 pr-2 text-right">Δ</th>
                          <th className="py-1 pr-2 text-right">Balance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {wallet.transactions.map((tx) => (
                          <tr
                            key={tx.id}
                            className="border-t border-black/5 dark:border-white/10"
                          >
                            <td className="py-1 pr-2">{formatAdminDateTime(tx.created_at)}</td>
                            <td className="py-1 pr-2 font-mono">{tx.reason_code}</td>
                            <td
                              className={`py-1 pr-2 text-right font-semibold ${
                                tx.change_credits < 0
                                  ? "text-rose-500 dark:text-rose-300"
                                  : "text-emerald-600 dark:text-emerald-300"
                              }`}
                            >
                              {tx.change_credits > 0 ? "+" : ""}
                              {tx.change_credits}
                            </td>
                            <td className="py-1 pr-2 text-right">{tx.balance_after}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="mt-2">
                    <AdminEmptyState message={copy.walletNoTransactions} />
                  </div>
                )}
              </AdminSurface>

              <AdminSurface>
                <AdminSectionTitle title={copy.walletPricing} />
                <div className="mt-2 space-y-1.5">
                  {wallet?.pricing ? (
                    <div className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 text-xs dark:border-white/10 dark:bg-white/[0.03]">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold capitalize">
                          {wallet.pricing.role_code}
                        </span>
                        <span className="text-slate-500 dark:text-white/45">
                          Free {wallet.pricing.monthly_free_credits}/mo
                        </span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-slate-500 dark:text-white/50">
                        <span>chat · {wallet.pricing.cost_per_chat}</span>
                        <span>repair · {wallet.pricing.cost_per_repair}</span>
                        <span>bench · {wallet.pricing.cost_per_benchmark_run}</span>
                      </div>
                    </div>
                  ) : (
                    <AdminEmptyState message="—" />
                  )}
                </div>
              </AdminSurface>
            </div>
          </AdminSurface>
        ) : null}

        {page === "preferences" && preferences ? (
          <AdminSurface>
            <AdminSectionTitle
              title={copy.profilePreferences}
              actions={
                <button
                  onClick={handleSavePreferences}
                  disabled={savingPrefs}
                  className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                    savingPrefs
                      ? "cursor-wait bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-white/50"
                      : "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                  }`}
                >
                  {savingPrefs ? copy.profileSaving : copy.profileSave}
                </button>
              }
            />
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <label className="flex flex-col gap-1 text-xs">
                <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                  {copy.profileDefaultAgent}
                </span>
                <select
                  value={preferences.default_agent_model ?? ""}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev ? { ...prev, default_agent_model: event.target.value || null } : prev,
                    )
                  }
                  className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                >
                  <option value="">—</option>
                  {modelOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                  {copy.profileDefaultChat}
                </span>
                <select
                  value={preferences.default_chat_model ?? ""}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev ? { ...prev, default_chat_model: event.target.value || null } : prev,
                    )
                  }
                  className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                >
                  <option value="">—</option>
                  {modelOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                  {copy.profileDefaultLanguage}
                </span>
                <input
                  value={preferences.default_language ?? ""}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev ? { ...prev, default_language: event.target.value || null } : prev,
                    )
                  }
                  className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                  {copy.profileTimezone}
                </span>
                <input
                  value={preferences.timezone ?? ""}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev ? { ...prev, timezone: event.target.value || null } : prev,
                    )
                  }
                  className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs sm:col-span-2">
                <span className="uppercase tracking-[0.18em] text-slate-500 dark:text-white/40">
                  {copy.profileBio}
                </span>
                <textarea
                  value={preferences.bio ?? ""}
                  rows={3}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev ? { ...prev, bio: event.target.value } : prev,
                    )
                  }
                  className="rounded-xl border border-black/10 bg-white px-2 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
              </label>
              <label className="flex items-center gap-2 text-xs sm:col-span-2">
                <input
                  type="checkbox"
                  checked={Boolean(preferences.show_site_map_widget)}
                  onChange={(event) =>
                    setPrefsDraft((prev) =>
                      prev
                        ? { ...prev, show_site_map_widget: event.target.checked }
                        : prev,
                    )
                  }
                />
                <span>{copy.profileShowSiteMap}</span>
              </label>
            </div>
          </AdminSurface>
        ) : null}

        {page === "api_tokens" ? (
          <AdminSurface>
            <AdminSectionTitle title={copy.profileApiTokens} />
            <div className="mt-2 flex flex-wrap gap-2">
              <input
                value={newTokenName}
                onChange={(event) => setNewTokenName(event.target.value)}
                placeholder={copy.profileTokenName}
                className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
              />
              <button
                onClick={handleCreateToken}
                disabled={creatingToken || !newTokenName.trim()}
                className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                  creatingToken || !newTokenName.trim()
                    ? "cursor-not-allowed bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-white/50"
                    : "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                }`}
              >
                {copy.profileTokenCreate}
              </button>
            </div>
            {newlyCreatedToken?.revealed_token ? (
              <div className="mt-2 rounded-2xl border border-emerald-400/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-800 dark:text-emerald-100">
                <div className="font-semibold">{copy.profileTokenNew}</div>
                <code className="mt-1 block break-all font-mono">
                  {newlyCreatedToken.revealed_token}
                </code>
              </div>
            ) : null}
            {tokens.length === 0 ? (
              <div className="mt-3">
                <AdminEmptyState message={copy.profileTokensEmpty} />
              </div>
            ) : (
              <div className="mt-3 space-y-1.5">
                {tokens.map((token) => (
                  <div
                    key={token.id}
                    className="flex items-center justify-between rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 text-xs dark:border-white/10 dark:bg-white/[0.03]"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{token.token_name}</span>
                        <AdminBadge
                          label={token.revoked_at ? "revoked" : "active"}
                          tone={token.revoked_at ? "rose" : "emerald"}
                        />
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] text-slate-500 dark:text-white/45">
                        {token.token_prefix}… · scope {token.scope} ·{" "}
                        {formatAdminDateTime(token.created_at)}
                      </div>
                    </div>
                    {!token.revoked_at ? (
                      <button
                        onClick={() => handleRevokeToken(token.id)}
                        className="rounded-full bg-rose-500/10 px-3 py-1 text-[11px] font-semibold text-rose-600 hover:bg-rose-500/15 dark:text-rose-300"
                      >
                        {copy.profileTokenRevoke}
                      </button>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </AdminSurface>
        ) : null}
      </div>
    </section>
  );
}
