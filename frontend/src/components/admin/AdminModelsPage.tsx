import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { AppCopy } from "../../i18n";
import type {
  AdminModelConfigItem,
  AdminModelConfigPayload,
  AdminModelUsageReport,
} from "../../types";
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

interface AdminModelsPageProps {
  copy: AppCopy;
  days: number;
  modelConfigs: AdminModelConfigItem[];
  modelUsage: AdminModelUsageReport | null;
  mutatingModelId: number | "create" | null;
  onDaysChange: (days: number) => void;
  onCreateModelConfig: (payload: AdminModelConfigPayload) => Promise<void>;
  onUpdateModelConfig: (modelConfigId: number, payload: AdminModelConfigPayload) => Promise<void>;
  onDeleteModelConfig: (modelConfigId: number) => Promise<void>;
}

type ModelFormState = AdminModelConfigPayload;

function buildNewModelDraft(sortOrder: number): ModelFormState {
  return {
    provider_code: "openai_compatible",
    provider_name: "OpenAI",
    vendor_name: "OpenAI",
    model_key: "",
    display_name: "",
    api_model_name: "",
    api_base_url: "",
    api_key_env_var: "OPENAI_API_KEY",
    enabled: true,
    is_default_chat: false,
    is_default_repair: false,
    supports_streaming: true,
    supports_json: true,
    thinking_enabled: false,
    sort_order: sortOrder,
    notes: "",
    extra_config: {},
  };
}

function toDraft(item: AdminModelConfigItem): ModelFormState {
  return {
    provider_code: item.provider_code,
    provider_name: item.provider_name,
    vendor_name: item.vendor_name,
    model_key: item.model_key,
    display_name: item.display_name,
    api_model_name: item.api_model_name,
    api_base_url: item.api_base_url ?? "",
    api_key_env_var: item.api_key_env_var ?? "",
    enabled: item.enabled,
    is_default_chat: item.is_default_chat,
    is_default_repair: item.is_default_repair,
    supports_streaming: item.supports_streaming,
    supports_json: item.supports_json,
    thinking_enabled: item.thinking_enabled,
    sort_order: item.sort_order,
    notes: item.notes ?? "",
    extra_config: item.extra_config ?? {},
  };
}

function updateDraftField(
  draft: ModelFormState,
  field: keyof ModelFormState,
  value: string | number | boolean,
): ModelFormState {
  return {
    ...draft,
    [field]: value,
  };
}

interface ModelFormCardProps {
  copy: AppCopy;
  draft: ModelFormState;
  headline: string;
  hint: string;
  isBusy: boolean;
  busyLabel: string;
  onChange: (next: ModelFormState) => void;
  onSubmit: () => void | Promise<void>;
  submitLabel: string;
  onDelete?: () => void | Promise<void>;
  deleteLabel?: string;
  toggleLabel?: string;
  onToggleEnabled?: () => void | Promise<void>;
  footer?: ReactNode;
}

function ModelFormCard({
  copy,
  draft,
  headline,
  hint,
  isBusy,
  busyLabel,
  onChange,
  onSubmit,
  submitLabel,
  onDelete,
  deleteLabel,
  toggleLabel,
  onToggleEnabled,
  footer,
}: ModelFormCardProps) {
  return (
    <AdminSurface className="space-y-2 app-slide-up">
      <AdminSectionTitle title={headline} hint={hint} />

      <div className="grid gap-2 xl:grid-cols-4">
        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminProviderCode}
          </div>
          <select
            value={draft.provider_code}
            onChange={(event) =>
              onChange(updateDraftField(draft, "provider_code", event.target.value as ModelFormState["provider_code"]))
            }
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          >
            <option value="openai_compatible">OpenAI Compatible</option>
            <option value="gemini">Gemini</option>
          </select>
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminProviderName}
          </div>
          <input
            value={draft.provider_name}
            onChange={(event) => onChange(updateDraftField(draft, "provider_name", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminVendorName}
          </div>
          <input
            value={draft.vendor_name}
            onChange={(event) => onChange(updateDraftField(draft, "vendor_name", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminSortOrder}
          </div>
          <input
            type="number"
            value={draft.sort_order}
            onChange={(event) =>
              onChange(updateDraftField(draft, "sort_order", Number.parseInt(event.target.value || "0", 10) || 0))
            }
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminModelKey}
          </div>
          <input
            value={draft.model_key}
            onChange={(event) => onChange(updateDraftField(draft, "model_key", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminDisplayName}
          </div>
          <input
            value={draft.display_name}
            onChange={(event) => onChange(updateDraftField(draft, "display_name", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminApiModelName}
          </div>
          <input
            value={draft.api_model_name}
            onChange={(event) => onChange(updateDraftField(draft, "api_model_name", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminApiKeyEnvVar}
          </div>
          <input
            value={draft.api_key_env_var}
            onChange={(event) => onChange(updateDraftField(draft, "api_key_env_var", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>

        <label className="block xl:col-span-4">
          <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
            {copy.adminApiBaseUrl}
          </div>
          <input
            value={draft.api_base_url}
            onChange={(event) => onChange(updateDraftField(draft, "api_base_url", event.target.value))}
            className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
          />
        </label>
      </div>

      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        {[
          ["enabled", copy.adminModelEnabled],
          ["is_default_chat", copy.adminDefaultChat],
          ["is_default_repair", copy.adminDefaultRepair],
          ["supports_streaming", copy.adminSupportsStreaming],
          ["supports_json", copy.adminSupportsJson],
          ["thinking_enabled", copy.adminThinkingMode],
        ].map(([field, label]) => {
          const checked = Boolean(draft[field as keyof ModelFormState]);
          return (
            <label
              key={field}
              className="flex items-center justify-between rounded-xl border border-black/10 bg-black/[0.025] px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.025]"
            >
              <span>{label}</span>
              <input
                type="checkbox"
                checked={checked}
                onChange={(event) => onChange(updateDraftField(draft, field as keyof ModelFormState, event.target.checked))}
              />
            </label>
          );
        })}
      </div>

      <label className="block">
        <div className="mb-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
          {copy.adminModelNotes}
        </div>
        <textarea
          value={draft.notes}
          onChange={(event) => onChange(updateDraftField(draft, "notes", event.target.value))}
          rows={2}
          className="w-full rounded-xl border border-black/10 bg-white/50 px-2 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.05]"
        />
      </label>

      {footer}

      <div className="flex flex-wrap items-center justify-end gap-2">
        {onToggleEnabled ? (
          <button
            onClick={() => void onToggleEnabled()}
            disabled={isBusy}
            className="rounded-full border border-black/10 px-3 py-1.5 text-xs font-medium transition hover:bg-black/[0.03] disabled:opacity-50 dark:border-white/10 dark:hover:bg-white/[0.04]"
          >
            {toggleLabel}
          </button>
        ) : null}
        {onDelete ? (
          <button
            onClick={() => void onDelete()}
            disabled={isBusy}
            className="rounded-full border border-rose-500/25 px-3 py-1.5 text-xs font-medium text-rose-700 transition hover:bg-rose-500/8 disabled:opacity-50 dark:text-rose-200"
          >
            {isBusy ? busyLabel : deleteLabel}
          </button>
        ) : null}
        <button
          onClick={() => void onSubmit()}
          disabled={isBusy}
          className="rounded-full bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
        >
          {isBusy ? busyLabel : submitLabel}
        </button>
      </div>
    </AdminSurface>
  );
}

export function AdminModelsPage({
  copy,
  days,
  modelConfigs,
  modelUsage,
  mutatingModelId,
  onDaysChange,
  onCreateModelConfig,
  onUpdateModelConfig,
  onDeleteModelConfig,
}: AdminModelsPageProps) {
  const [newModelDraft, setNewModelDraft] = useState<ModelFormState>(() => buildNewModelDraft(100));
  const [drafts, setDrafts] = useState<Record<number, ModelFormState>>({});
  const [expandedModelId, setExpandedModelId] = useState<number | "create" | null>(null);

  useEffect(() => {
    setDrafts(
      Object.fromEntries(modelConfigs.map((item) => [item.id, toDraft(item)])),
    );
  }, [modelConfigs]);

  const items = modelUsage?.items ?? [];
  const totalRequests = items.reduce((sum, item) => sum + item.request_count, 0);
  const totalTokens = items.reduce((sum, item) => sum + item.total_tokens, 0);
  const averageLatency =
    totalRequests > 0
      ? items.reduce((sum, item) => sum + item.avg_latency_ms * item.request_count, 0) / totalRequests
      : 0;
  const dailySeries = modelUsage?.daily_series ?? [];
  const maxSeriesValue = dailySeries.reduce((current, item) => Math.max(current, item.total_tokens), 0);

  const nextSortOrder = useMemo(() => {
    const maxSort = modelConfigs.reduce((current, item) => Math.max(current, item.sort_order), 0);
    return maxSort + 10;
  }, [modelConfigs]);

  return (
    <div className="w-full min-w-0 space-y-2">
      <AdminSurface>
        <AdminSectionTitle
          title={copy.adminModelsTitle}
          actions={
            <>
              {[7, 30, 90].map((value) => (
                <button
                  key={value}
                  onClick={() => onDaysChange(value)}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                    days === value
                      ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                      : "border border-black/10 text-slate-600 dark:border-white/10 dark:text-white/65"
                  }`}
                >
                  {value}d
                </button>
              ))}
            </>
          }
        />
      </AdminSurface>

      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
        <AdminMetricCard label={copy.adminModels} value={formatAdminNumber(items.length)} tone="sky" />
        <AdminMetricCard label={copy.adminRequestCount} value={formatAdminNumber(totalRequests)} tone="emerald" />
        <AdminMetricCard label={copy.adminTokens} value={formatAdminNumber(totalTokens)} tone="amber" />
        <AdminMetricCard label={copy.adminLatency} value={formatAdminDurationMs(averageLatency)} tone="rose" />
      </div>

      <div className="grid gap-2 xl:grid-cols-[0.9fr_1.1fr]">
        <AdminSurface>
          <AdminSectionTitle title={copy.adminDailyModelUsage} hint={`${days}d`} />
          <div className="mt-2 space-y-2">
            {dailySeries.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              dailySeries.map((item, index) => (
                <div key={`${item.day}-${item.model}-${index}`}>
                  <div className="mb-1 flex items-center justify-between gap-2 text-[11px] text-slate-500 dark:text-white/45">
                    <span className="truncate">
                      {item.day} · {item.model}
                    </span>
                    <span>
                      {formatAdminNumber(item.total_tokens)} {copy.adminTokens}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-black/[0.05] dark:bg-white/[0.06]">
                    <div
                      className="h-full rounded-full bg-slate-900 dark:bg-white"
                      style={{
                        width: `${maxSeriesValue > 0 ? Math.max(8, (item.total_tokens / maxSeriesValue) * 100) : 8}%`,
                      }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </AdminSurface>

        <AdminSurface>
          <AdminSectionTitle title={copy.adminModels} hint={`${formatAdminNumber(items.length)} total`} />
          <div className="mt-2 overflow-x-auto">
            {items.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              <table className="w-full min-w-full text-left text-xs">
                <thead className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                  <tr>
                    <th className="px-2 py-2">{copy.model}</th>
                    <th className="px-2 py-2">{copy.adminRequestProvider}</th>
                    <th className="px-2 py-2">{copy.adminRequestCount}</th>
                    <th className="px-2 py-2">{copy.adminInputTokens}</th>
                    <th className="px-2 py-2">{copy.adminOutputTokens}</th>
                    <th className="px-2 py-2">{copy.adminTokens}</th>
                    <th className="px-2 py-2">{copy.adminLatency}</th>
                    <th className="px-2 py-2">{copy.adminLastLogin}</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr
                      key={`${item.provider}-${item.model}`}
                      className="border-t border-black/5 text-slate-700 dark:border-white/10 dark:text-white/75"
                    >
                      <td className="px-2 py-2">
                        <div className="font-medium text-slate-900 dark:text-white">{item.model}</div>
                      </td>
                      <td className="px-2 py-2">
                        <AdminBadge label={item.provider} tone="slate" />
                      </td>
                      <td className="px-2 py-2">{formatAdminNumber(item.request_count)}</td>
                      <td className="px-2 py-2">{formatAdminNumber(item.input_tokens)}</td>
                      <td className="px-2 py-2">{formatAdminNumber(item.output_tokens)}</td>
                      <td className="px-2 py-2">{formatAdminNumber(item.total_tokens)}</td>
                      <td className="px-2 py-2">{formatAdminDurationMs(item.avg_latency_ms)}</td>
                      <td className="px-2 py-2">{formatAdminDateTime(item.last_used_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </AdminSurface>
      </div>

      <div className="flex items-center justify-between">
        <AdminSectionTitle title={copy.adminModelRegistryTitle} hint={copy.adminModelRegistryHint} />
        {expandedModelId !== "create" ? (
          <button
            onClick={() => setExpandedModelId("create")}
            className="rounded-full bg-slate-900 px-4 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
          >
            {copy.adminModelCreate}
          </button>
        ) : null}
      </div>

      {expandedModelId === "create" ? (
        <ModelFormCard
          copy={copy}
          draft={newModelDraft}
          headline={copy.adminModelCreate}
          hint=""
          isBusy={mutatingModelId === "create"}
          busyLabel={copy.adminModelSaving}
          onChange={setNewModelDraft}
          onSubmit={() => {
            void onCreateModelConfig(newModelDraft).then(() => {
              setNewModelDraft(buildNewModelDraft(nextSortOrder + 10));
              setExpandedModelId(null);
            });
          }}
          submitLabel={copy.adminModelSave}
          onDelete={() => setExpandedModelId(null)}
          deleteLabel={copy.cancel}
        />
      ) : null}

      <div className="space-y-2">
        {modelConfigs.length === 0 ? (
          <AdminEmptyState message={copy.modelEmpty} />
        ) : (
          modelConfigs.map((item) => {
            const isExpanded = expandedModelId === item.id;
            const draft = drafts[item.id] ?? toDraft(item);
            const isBusy = mutatingModelId === item.id;

            if (isExpanded) {
              return (
                <ModelFormCard
                  key={item.id}
                  copy={copy}
                  draft={draft}
                  headline={item.display_name}
                  hint={`${item.model_key} · ${item.provider_name}`}
                  isBusy={isBusy}
                  busyLabel={copy.adminModelSaving}
                  onChange={(next) =>
                    setDrafts((current) => ({
                      ...current,
                      [item.id]: next,
                    }))
                  }
                  onSubmit={() => {
                    void onUpdateModelConfig(item.id, draft).then(() => {
                      setExpandedModelId(null);
                    });
                  }}
                  submitLabel={copy.adminModelSave}
                  onDelete={() => {
                    if (!window.confirm(`${copy.adminModelDelete}: ${item.display_name}?`)) {
                      return;
                    }
                    void onDeleteModelConfig(item.id);
                  }}
                  deleteLabel={copy.adminModelDelete}
                  toggleLabel={copy.cancel}
                  onToggleEnabled={() => setExpandedModelId(null)}
                  footer={
                    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-5">
                      <div className="rounded-xl border border-black/5 bg-black/[0.025] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.025]">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                          {copy.adminModelEnabled}
                        </div>
                        <div className="mt-0.5">
                          <AdminBadge
                            label={item.enabled ? copy.yes : copy.no}
                            tone={toStatusTone(item.enabled ? "active" : "suspended")}
                          />
                        </div>
                      </div>
                      <div className="rounded-xl border border-black/5 bg-black/[0.025] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.025]">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                          {copy.adminModelConfigured}
                        </div>
                        <div className="mt-0.5">
                          <AdminBadge
                            label={item.api_key_configured ? copy.yes : copy.no}
                            tone={toStatusTone(item.api_key_configured ? "active" : "failed")}
                          />
                        </div>
                      </div>
                      <div className="rounded-xl border border-black/5 bg-black/[0.025] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.025]">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                          {copy.adminModelUsage30d}
                        </div>
                        <div className="mt-0.5 text-xs font-medium">{formatAdminNumber(item.request_count_30d)}</div>
                      </div>
                      <div className="rounded-xl border border-black/5 bg-black/[0.025] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.025]">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                          {copy.adminModelTokens30d}
                        </div>
                        <div className="mt-0.5 text-xs font-medium">{formatAdminNumber(item.total_tokens_30d)}</div>
                      </div>
                      <div className="rounded-xl border border-black/5 bg-black/[0.025] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.025]">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 dark:text-white/35">
                          {copy.adminLastLogin}
                        </div>
                        <div className="mt-0.5 text-xs font-medium">{formatAdminDateTime(item.last_used_at)}</div>
                      </div>
                      {item.missing_configuration.length > 0 ? (
                        <div className="md:col-span-2 xl:col-span-5 rounded-xl border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-xs text-amber-800 dark:text-amber-100">
                          {copy.adminModelMissingConfig}: {item.missing_configuration.join(", ")}
                        </div>
                      ) : null}
                    </div>
                  }
                />
              );
            }

            return (
              <div
                key={item.id}
                className="flex items-center justify-between gap-2 rounded-xl border border-black/5 bg-white/60 px-3 py-2.5 transition hover:bg-white/50 dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.05]"
              >
                <div className="flex min-w-0 items-center gap-2.5">
                  <div
                    className={`h-2 w-2 shrink-0 rounded-full ${
                      item.enabled && item.api_key_configured ? "bg-emerald-500" : item.enabled ? "bg-amber-500" : "bg-slate-300 dark:bg-slate-700"
                    }`}
                  />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-900 dark:text-white">
                      {item.display_name}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-white/45">
                      <span className="truncate">{item.model_key}</span>
                      <span>·</span>
                      <span>{item.provider_name}</span>
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    onClick={() => setExpandedModelId(item.id)}
                    className="rounded-full bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-600 transition hover:bg-slate-200 dark:bg-white/10 dark:text-white/70 dark:hover:bg-white/15"
                  >
                    {copy.adminEdit}
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
