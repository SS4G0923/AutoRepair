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
  onSelectRequest: (requestId: number | null) => void;
}

function RequestMetaRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-black/5 bg-black/[0.02] px-2 py-1.5 dark:border-white/10 dark:bg-white/[0.03]">
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
            className="rounded-xl border border-black/10 bg-white/50 px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
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
            className="rounded-xl border border-black/10 bg-white/50 px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white dark:placeholder:text-white/28"
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
            className="rounded-xl border border-black/10 bg-white/50 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
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
            className="rounded-xl border border-black/10 bg-white/50 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
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

      <div className="space-y-2">
        {items.length === 0 ? (
          <AdminEmptyState message={copy.adminNoData} />
        ) : (
          items.map((item) => {
            const active = selectedRequestId === item.id;
            return (
              <div
                key={item.id}
                className={`rounded-2xl border px-3 py-3 transition ${
                  active
                    ? "border-slate-900 bg-white/50 dark:border-white/20 dark:bg-white/[0.04]"
                    : "border-black/5 bg-white/60 hover:bg-white/50 dark:border-white/10 dark:bg-white/[0.02] dark:hover:bg-white/[0.04]"
                }`}
              >
                {/* Summary Row */}
                <div
                  className="flex cursor-pointer flex-wrap items-center justify-between gap-3"
                  onClick={() => onSelectRequest(active ? null : item.id)}
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <div
                      className={`h-2 w-2 shrink-0 rounded-full ${
                        item.request_status === "completed"
                          ? "bg-emerald-500"
                          : item.request_status === "failed"
                            ? "bg-rose-500"
                            : "bg-amber-500"
                      }`}
                    />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <div className="truncate text-sm font-semibold text-slate-900 dark:text-white">
                          #{item.id} · {item.model}
                        </div>
                        <AdminBadge label={item.request_status} tone={toStatusTone(item.request_status)} />
                      </div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500 dark:text-white/45">
                        <span>{formatAdminDateTime(item.started_at)}</span>
                        <span>·</span>
                        <span>{item.provider}</span>
                        <span>·</span>
                        <span>{item.request_mode}</span>
                        <span>·</span>
                        <span>{formatAdminNumber(item.total_tokens)} {copy.adminTokens}</span>
                        <span>·</span>
                        <span>{item.user_display_name || item.user_email || "-"}</span>
                      </div>
                    </div>
                  </div>
                  <div
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition ${
                      active
                        ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                        : "bg-slate-100 text-slate-500 dark:bg-white/10 dark:text-white/45"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className={`h-4 w-4 transition-transform duration-200 ${active ? "rotate-180" : ""}`}
                    >
                      <path
                        fillRule="evenodd"
                        d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>

                {/* Expanded Details */}
                {active ? (
                  <div className="mt-3 border-t border-black/5 pt-3 dark:border-white/10">
                    {requestDetailLoading ? (
                      <div className="grid min-h-[10rem] place-items-center text-sm text-slate-500 dark:text-white/45">
                        {copy.adminRefresh}...
                      </div>
                    ) : !requestDetail || requestDetail.request.id !== item.id ? (
                      <AdminEmptyState message={copy.adminNoData} />
                    ) : (
                      <div className="space-y-2">
                        <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-4">
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

                        {requestDetail.tool_events.length > 0 ? (
                          <div>
                            <AdminSectionTitle title={copy.adminToolEvents} />
                            <div className="mt-2 space-y-2">
                              {requestDetail.tool_events.map((event) => (
                                <div
                                  key={event.id}
                                  className="rounded-xl border border-black/5 bg-black/[0.02] px-3 py-2.5 dark:border-white/10 dark:bg-white/[0.03]"
                                >
                                  <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="text-xs font-medium text-slate-900 dark:text-white">
                                      {event.tool_name}
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                      {event.round_index != null ? (
                                        <AdminBadge label={`round ${event.round_index}`} tone="slate" />
                                      ) : null}
                                      <AdminBadge label={event.status} tone={toStatusTone(event.status)} />
                                    </div>
                                  </div>
                                  <div className="mt-1 text-[10px] text-slate-500 dark:text-white/45">
                                    {formatAdminDateTime(event.created_at)}
                                  </div>
                                  {event.arguments_json ? (
                                    <div className="mt-2">
                                      <AdminCodeBlock title={copy.toolArguments} content={event.arguments_json} />
                                    </div>
                                  ) : null}
                                  {event.output_preview ? (
                                    <div className="mt-2">
                                      <AdminCodeBlock title={copy.toolOutput} content={event.output_preview} />
                                    </div>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
