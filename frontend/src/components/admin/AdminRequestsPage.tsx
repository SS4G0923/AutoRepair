import type { Dispatch, SetStateAction } from "react";
import type { AdminRequestFilters } from "../../app/useAdminConsole";
import type { AppCopy } from "../../i18n";
import type { AdminLlmRequestDetail, AdminLlmRequestList } from "../../types";
import {
  AdminBadge,
  AdminCodeBlock,
  AdminEmptyState,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  formatAdminDurationMs,
  formatAdminNumber,
  toStatusTone,
} from "./AdminCommon";

interface AdminRequestsPageProps {
  copy: AppCopy;
  requestDetail: AdminLlmRequestDetail | null;
  requestDetailLoading: boolean;
  requestFilters: AdminRequestFilters;
  requests: AdminLlmRequestList | null;
  selectedRequestId: number | null;
  setRequestFilters: Dispatch<SetStateAction<AdminRequestFilters>>;
  onSelectRequest: (requestId: number) => void;
}

function RequestMetaRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-black/5 bg-black/[0.02] px-2.5 py-2 dark:border-white/10 dark:bg-white/[0.03]">
      <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500 dark:text-white/35">
        {label}
      </div>
      <div className="mt-0.5 break-words text-xs text-slate-800 dark:text-white/80">{value || "-"}</div>
    </div>
  );
}

export function AdminRequestsPage({
  copy,
  requestDetail,
  requestDetailLoading,
  requestFilters,
  requests,
  selectedRequestId,
  setRequestFilters,
  onSelectRequest,
}: AdminRequestsPageProps) {
  const items = requests?.items ?? [];
  const startIndex = requests ? (requests.page - 1) * requests.page_size + 1 : 0;
  const endIndex = requests ? Math.min(requests.total, startIndex + items.length - 1) : 0;
  const canGoPrevious = Boolean(requests && requests.page > 1);
  const canGoNext = Boolean(requests && requests.page * requests.page_size < requests.total);

  return (
    <div className="space-y-2">
      <AdminSurface>
        <div className="grid gap-2 xl:grid-cols-[1.3fr_1fr_0.7fr_0.7fr]">
          <input
            value={requestFilters.q}
            onChange={(event) =>
              setRequestFilters((current) => ({
                ...current,
                page: 1,
                q: event.target.value,
              }))
            }
            placeholder={copy.adminSearchPlaceholder}
            className="rounded-xl border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
          />
          <input
            value={requestFilters.model}
            onChange={(event) =>
              setRequestFilters((current) => ({
                ...current,
                page: 1,
                model: event.target.value,
              }))
            }
            placeholder={copy.adminFilterModel}
            className="rounded-xl border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
          />
          <select
            value={requestFilters.status}
            onChange={(event) =>
              setRequestFilters((current) => ({
                ...current,
                page: 1,
                status: event.target.value,
              }))
            }
            className="rounded-xl border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
          >
            <option value="">{copy.adminFilterStatus}: {copy.adminFilterAll}</option>
            <option value="completed">{copy.adminFilterCompleted}</option>
            <option value="failed">{copy.adminFilterFailed}</option>
          </select>
          <select
            value={requestFilters.requestMode}
            onChange={(event) =>
              setRequestFilters((current) => ({
                ...current,
                page: 1,
                requestMode: event.target.value,
              }))
            }
            className="rounded-xl border border-black/10 bg-white/70 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
          >
            <option value="">{copy.adminFilterMode}: {copy.adminFilterAll}</option>
            <option value="chat">{copy.adminFilterChat}</option>
            <option value="repair">{copy.adminFilterRepair}</option>
          </select>
        </div>

        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 dark:text-white/60">
          <div>
            {copy.adminShowing} {requests?.total ? startIndex : 0}-{endIndex} {copy.adminOf} {formatAdminNumber(requests?.total ?? 0)}
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() =>
                setRequestFilters((current) => ({
                  ...current,
                  page: Math.max(1, current.page - 1),
                }))
              }
              disabled={!canGoPrevious}
              className="rounded-full border border-black/10 px-3 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminPrevious}
            </button>
            <button
              onClick={() =>
                setRequestFilters((current) => ({
                  ...current,
                  page: current.page + 1,
                }))
              }
              disabled={!canGoNext}
              className="rounded-full border border-black/10 px-3 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10"
            >
              {copy.adminNext}
            </button>
          </div>
        </div>
      </AdminSurface>

      <div className="grid gap-2 xl:grid-cols-[0.78fr_1.22fr]">
        <AdminSurface className="min-h-[18rem]">
          <div className="space-y-2">
            {items.length === 0 ? (
              <AdminEmptyState message={copy.adminNoData} />
            ) : (
              items.map((item) => {
                const active = selectedRequestId === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectRequest(item.id)}
                    className={`w-full rounded-2xl border px-3 py-3 text-left transition ${
                      active
                        ? "border-slate-900 bg-slate-900 text-white shadow-lg dark:border-white dark:bg-white dark:text-slate-950"
                        : "border-black/5 bg-black/[0.02] hover:bg-black/[0.04] dark:border-white/10 dark:bg-white/[0.03] dark:text-white/80 dark:hover:bg-white/[0.05]"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium">#{item.id} · {item.model}</div>
                      </div>
                      <AdminBadge label={item.request_status} tone={toStatusTone(item.request_status)} />
                    </div>
                    <div
                      className={`mt-1 text-[11px] ${
                        active ? "text-white/70 dark:text-slate-950/70" : "text-slate-500 dark:text-white/45"
                      }`}
                    >
                      {item.provider} · {item.request_mode} · {formatAdminNumber(item.total_tokens)} tk · {item.user_display_name || item.user_email || "-"}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </AdminSurface>

        <AdminSurface className="min-h-[18rem]">
          {requestDetailLoading ? (
            <div className="grid min-h-[14rem] place-items-center text-sm text-slate-500 dark:text-white/45">
              {copy.adminRefresh}...
            </div>
          ) : !requestDetail ? (
            <AdminEmptyState message={copy.adminNoSelection} />
          ) : (
            <div className="space-y-2">
              <AdminSectionTitle
                title={copy.adminRequestDetail}
                hint={`#${requestDetail.request.id} · ${requestDetail.request.model}`}
              />

              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                <RequestMetaRow label={copy.adminRequestId} value={String(requestDetail.request.id)} />
                <RequestMetaRow
                  label={copy.adminUserLabel}
                  value={requestDetail.request.user_display_name || requestDetail.request.user_email || "-"}
                />
                <RequestMetaRow
                  label={copy.adminHistoryId}
                  value={requestDetail.request.history_id ? String(requestDetail.request.history_id) : "-"}
                />
                <RequestMetaRow label={copy.adminFilterMode} value={requestDetail.request.request_mode} />
                <RequestMetaRow label={copy.adminFilterStatus} value={requestDetail.request.request_status} />
                <RequestMetaRow label={copy.adminRequestProvider} value={requestDetail.request.provider} />
                <RequestMetaRow label={copy.model} value={requestDetail.request.model} />
                <RequestMetaRow label={copy.adminRequestStage} value={requestDetail.request.stage || "-"} />
                <RequestMetaRow label={copy.adminRequestPurpose} value={requestDetail.request.purpose || "-"} />
                <RequestMetaRow label={copy.adminRequestStarted} value={formatAdminDateTime(requestDetail.request.started_at)} />
                <RequestMetaRow label={copy.adminRequestFinished} value={formatAdminDateTime(requestDetail.request.finished_at)} />
                <RequestMetaRow label={copy.adminLatency} value={formatAdminDurationMs(requestDetail.request.latency_ms)} />
                <RequestMetaRow label={copy.adminRequestStreaming} value={requestDetail.request.is_streaming ? copy.yes : copy.no} />
                <RequestMetaRow label={copy.adminRequestJson} value={requestDetail.request.is_json_response ? copy.yes : copy.no} />
                <RequestMetaRow label={copy.adminRequestSourceType} value={requestDetail.request.source_type || "-"} />
                <RequestMetaRow label={copy.adminTokenSource} value={requestDetail.request.token_source || "-"} />
                <RequestMetaRow label={copy.adminInputTokens} value={formatAdminNumber(requestDetail.request.input_tokens)} />
                <RequestMetaRow label={copy.adminOutputTokens} value={formatAdminNumber(requestDetail.request.output_tokens)} />
                <RequestMetaRow label={copy.adminTokens} value={formatAdminNumber(requestDetail.request.total_tokens)} />
                <RequestMetaRow label={copy.adminCachedTokens} value={formatAdminNumber(requestDetail.request.cached_input_tokens)} />
                <RequestMetaRow label={copy.adminReasoningTokens} value={formatAdminNumber(requestDetail.request.reasoning_tokens)} />
                <RequestMetaRow
                  label={copy.adminRequestChars}
                  value={`${formatAdminNumber(requestDetail.request.prompt_chars)} / ${formatAdminNumber(requestDetail.request.response_chars)}`}
                />
                <RequestMetaRow label={copy.adminRequestError} value={requestDetail.request.error_message || "-"} />
              </div>

              <div className="grid gap-2 xl:grid-cols-2">
                <AdminCodeBlock title={copy.adminSystemPrompt} content={requestDetail.message.system_prompt} />
                <AdminCodeBlock title={copy.adminParsedJson} content={requestDetail.message.parsed_response_json} />
              </div>
              <AdminCodeBlock title={copy.adminPrompt} content={requestDetail.message.prompt_text} />
              <AdminCodeBlock title={copy.adminResponse} content={requestDetail.message.response_text} />

              <div>
                <AdminSectionTitle title={copy.adminToolEvents} />
                <div className="mt-2 space-y-2">
                  {requestDetail.tool_events.length === 0 ? (
                    <AdminEmptyState message={copy.adminToolEventsEmpty} />
                  ) : (
                    requestDetail.tool_events.map((event) => (
                      <div
                        key={event.id}
                        className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-3 dark:border-white/10 dark:bg-white/[0.03]"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div className="font-medium text-slate-900 dark:text-white">
                            {event.tool_name}
                          </div>
                          <div className="flex items-center gap-2">
                            {event.round_index != null ? (
                              <AdminBadge label={`round ${event.round_index}`} tone="slate" />
                            ) : null}
                            <AdminBadge label={event.status} tone={toStatusTone(event.status)} />
                          </div>
                        </div>
                        <div className="mt-2 text-xs text-slate-500 dark:text-white/45">
                          {formatAdminDateTime(event.created_at)}
                        </div>
                        {event.arguments_json ? (
                          <div className="mt-3">
                            <AdminCodeBlock title={copy.toolArguments} content={event.arguments_json} />
                          </div>
                        ) : null}
                        {event.output_preview ? (
                          <div className="mt-3">
                            <AdminCodeBlock title={copy.toolOutput} content={event.output_preview} />
                          </div>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </AdminSurface>
      </div>
    </div>
  );
}
