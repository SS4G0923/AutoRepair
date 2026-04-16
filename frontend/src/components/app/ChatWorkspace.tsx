import { AppCopy } from "../../i18n";
import type { ChatMessage, ModelCatalogItem, ModelOptionValue } from "../../types";

interface ChatWorkspaceProps {
  activeModelLabel: string;
  chatError: string;
  chatInput: string;
  chatMessages: ChatMessage[];
  chatStreamingText: string;
  chatThinking: boolean;
  copy: AppCopy;
  isDesktopLayout: boolean;
  model: ModelOptionValue;
  modelOptions: ModelCatalogItem[];
  onChatInputChange: (value: string) => void;
  onModelChange: (value: ModelOptionValue) => void;
  onSend: () => void;
}

export function ChatWorkspace({
  activeModelLabel,
  chatError,
  chatInput,
  chatMessages,
  chatStreamingText,
  chatThinking,
  copy,
  isDesktopLayout,
  model,
  modelOptions,
  onChatInputChange,
  onModelChange,
  onSend,
}: ChatWorkspaceProps) {
  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 ${!isDesktopLayout ? "flex-1" : ""}`}
    >
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">
          {copy.chatTitle}
        </div>
        <select
          value={model}
          onChange={(event) => onModelChange(event.target.value as ModelOptionValue)}
          disabled={modelOptions.length === 0}
          className="rounded-full border border-black/10 bg-white/50 px-3 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
        >
          {modelOptions.length > 0 ? (
            modelOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {copy.model}: {option.label}
              </option>
            ))
          ) : (
            <option value="">{copy.modelEmpty}</option>
          )}
        </select>
      </div>

      <div className="mt-3 min-h-0 flex-1 overflow-hidden rounded-[20px] border border-black/5 bg-black/[0.025] dark:border-white/10 dark:bg-white/[0.025]">
        <div className="flex h-full min-h-[8rem] flex-col gap-2.5 overflow-y-auto p-2.5">
          {chatMessages.length === 0 ? (
            <div className="flex flex-1 items-center justify-center rounded-[24px] border border-dashed border-black/10 px-6 text-center text-sm text-slate-500 dark:border-white/10 dark:text-white/40">
              {copy.chatEmpty}
            </div>
          ) : (
            chatMessages.map((message) => (
              <div
                key={message.id}
                className={`max-w-[85%] rounded-[24px] px-4 py-3 ${
                  message.role === "user"
                    ? "ml-auto bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                    : "bg-white/85 text-slate-700 dark:bg-slate-900/90 dark:text-white"
                }`}
              >
                <div className="whitespace-pre-wrap break-words text-sm leading-7">{message.content}</div>
                <div
                  className={`mt-2 text-[11px] ${
                    message.role === "user"
                      ? "text-white/65 dark:text-slate-950/55"
                      : "text-slate-400 dark:text-white/35"
                  }`}
                >
                  {message.at}
                </div>
              </div>
            ))
          )}

          {chatThinking && !chatStreamingText ? (
            <div className="max-w-[85%] rounded-[24px] bg-white/85 px-4 py-3 text-slate-700 dark:bg-slate-900/90 dark:text-white">
              <div className="text-sm">{copy.chatThinking}</div>
            </div>
          ) : null}

          {chatStreamingText ? (
            <div className="max-w-[85%] rounded-[24px] bg-white/85 px-4 py-3 text-slate-700 dark:bg-slate-900/90 dark:text-white">
              <div className="whitespace-pre-wrap break-words text-sm leading-7">
                {chatStreamingText}
              </div>
            </div>
          ) : null}

          {chatError ? (
            <div className="max-w-[85%] rounded-[24px] border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
              {chatError}
            </div>
          ) : null}
        </div>
      </div>

      <div className="mt-3 shrink-0 rounded-[20px] border border-black/5 bg-white/50 p-2.5 dark:border-white/10 dark:bg-white/[0.03]">
        <textarea
          value={chatInput}
          onChange={(event) => onChatInputChange(event.target.value)}
          placeholder={copy.chatPlaceholder}
          className="min-h-[48px] max-h-[120px] w-full resize-y bg-transparent text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 dark:text-white dark:placeholder:text-white/28"
        />
        <div className="mt-2 flex items-center justify-between gap-3">
          <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/35">
            {copy.model}: {activeModelLabel}
          </div>
          <button
            onClick={onSend}
            disabled={!chatInput.trim() || chatThinking || modelOptions.length === 0}
            className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-45 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
          >
            {copy.chatSend}
          </button>
        </div>
      </div>
    </section>
  );
}
