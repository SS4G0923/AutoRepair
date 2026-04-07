import { useEffect, useRef, useState } from "react";
import { AuthPage } from "./components/AuthPage";
import { CodeEditor } from "./components/CodeEditor";
import { StageCard } from "./components/StageCard";
import { codeTemplates, copy, languageOptions, modelOptions, stageOrder } from "./i18n";
import type {
  AuthenticatedUser,
  AgentHistorySnapshot,
  ChatMessage,
  ChatHistorySnapshot,
  CodeLanguage,
  EventEntry,
  HistoryDetail,
  HistorySummary,
  ModelOptionValue,
  OAuthProvider,
  RunResult,
  SessionStatus,
  StageName,
  StageState,
  ThemeMode,
  ToolEventEntry,
  UiLocale,
  WorkspaceMode,
} from "./types";

const createStageState = (): Record<StageName, StageState> => ({
  run: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  inspect: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  plan: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  code: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
  verify: { status: "idle", explain: "", report: "", diff: "", toolEvents: [] },
});

const SIDEBAR_WIDTH_STORAGE_KEY = "autorepair-sidebar-width";
const DEFAULT_SIDEBAR_WIDTH = 200;
const MIN_SIDEBAR_WIDTH = 160;
const MAX_SIDEBAR_WIDTH = 560;

function formatTimestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function buildSummary(event: string, data: Record<string, unknown>) {
  if (event === "stage") {
    return `${String(data.stage)} · ${String(data.status)}`;
  }
  if (event === "tool_event") {
    return `${String(data.stage)} · ${String(data.tool_name)} · ${String(data.status)}`;
  }
  if (event === "error") {
    return "error";
  }
  if (event === "result") {
    return "result";
  }
  return JSON.stringify(data).slice(0, 140);
}

function parseSseBlock(block: string) {
  const lines = block.split("\n");
  let event = "message";
  const dataParts: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataParts.push(line.slice(5).trimStart());
    }
  }

  return {
    event,
    data: dataParts.length > 0 ? JSON.parse(dataParts.join("\n")) : {},
  };
}

function normalizeStageMap(
  incoming?: Partial<Record<StageName, Partial<StageState>>>,
): Record<StageName, StageState> {
  const base = createStageState();
  if (!incoming) {
    return base;
  }

  for (const stage of Object.keys(base) as StageName[]) {
    const next = incoming[stage];
    if (!next) {
      continue;
    }
    base[stage] = {
      status: next.status ?? base[stage].status,
      explain: next.explain ?? base[stage].explain,
      report: next.report ?? base[stage].report,
      diff: next.diff ?? base[stage].diff,
      toolEvents: Array.isArray(next.toolEvents)
        ? next.toolEvents.map((item, index) => ({
            id:
              typeof item.id === "string" && item.id
                ? item.id
                : `history-tool-${stage}-${index}`,
            tool_name:
              typeof item.tool_name === "string" && item.tool_name ? item.tool_name : "tool",
            status: item.status === "completed" ? "completed" : "started",
            round: typeof item.round === "number" ? item.round : undefined,
            arguments: typeof item.arguments === "string" ? item.arguments : undefined,
            output_preview:
              typeof item.output_preview === "string" ? item.output_preview : undefined,
            output_truncated: Boolean(item.output_truncated ?? false),
            at: typeof item.at === "string" ? item.at : "",
          }))
        : [],
    };
  }

  return base;
}

function applyUnifiedDiffToText(originalText: string, diffText: string) {
  const normalizedOriginal = originalText.replace(/\r\n/g, "\n");
  const normalizedDiff = diffText.replace(/\r\n/g, "\n");
  const sourceLines = normalizedOriginal.split("\n");
  const diffLines = normalizedDiff.split("\n");
  const hadTrailingNewline = normalizedOriginal.endsWith("\n");

  let diffIndex = diffLines.findIndex((line) => line.startsWith("@@"));
  if (diffIndex === -1) {
    throw new Error("Missing hunk header.");
  }

  const result: string[] = [];
  let sourceIndex = 0;

  while (diffIndex < diffLines.length) {
    const header = diffLines[diffIndex];
    if (!header.startsWith("@@")) {
      diffIndex += 1;
      continue;
    }

    const match = header.match(/^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@/);
    if (!match) {
      throw new Error(`Invalid hunk header: ${header}`);
    }

    const targetStart = Math.max(0, Number(match[1]) - 1);
    while (sourceIndex < targetStart && sourceIndex < sourceLines.length) {
      result.push(sourceLines[sourceIndex]);
      sourceIndex += 1;
    }

    diffIndex += 1;
    while (diffIndex < diffLines.length && !diffLines[diffIndex].startsWith("@@")) {
      const line = diffLines[diffIndex];
      if (!line || line === "\\ No newline at end of file") {
        diffIndex += 1;
        continue;
      }
      const prefix = line[0];
      const value = line.slice(1);
      if (prefix === " ") {
        result.push(value);
        sourceIndex += 1;
      } else if (prefix === "-") {
        sourceIndex += 1;
      } else if (prefix === "+") {
        result.push(value);
      }
      diffIndex += 1;
    }
  }

  while (sourceIndex < sourceLines.length) {
    result.push(sourceLines[sourceIndex]);
    sourceIndex += 1;
  }

  let nextText = result.join("\n");
  if (hadTrailingNewline && !nextText.endsWith("\n")) {
    nextText += "\n";
  }
  return nextText;
}

function LanguageIcon() {
  return (
    <span aria-hidden="true" className="flex items-end gap-1 leading-none">
      <span className="text-[15px] font-semibold">文</span>
      <span className="text-slate-400 dark:text-white/35">/</span>
      <span className="text-[14px] font-semibold tracking-[0.08em]">A</span>
    </span>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5">
      <circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2.5v2.2M12 19.3v2.2M21.5 12h-2.2M4.7 12H2.5M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6M18.7 18.7l-1.6-1.6M6.9 6.9L5.3 5.3"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5">
      <path
        d="M19 14.5A7.5 7.5 0 0 1 9.5 5a8.5 8.5 0 1 0 9.5 9.5Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function AgentIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5">
      <path
        d="M6 18h12M8 14h8M9 4h6l3 3v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V7l3-3Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5">
      <path
        d="M5 6.5A2.5 2.5 0 0 1 7.5 4h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H11l-4.5 4v-4H7.5A2.5 2.5 0 0 1 5 12.5Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4">
      <path
        d="M9 4.5h6M5.5 7h13M9.5 10.5v6M14.5 10.5v6M8 19.5h8a1 1 0 0 0 1-1l.7-10.5H6.3L7 18.5a1 1 0 0 0 1 1Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function App() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
  const [locale, setLocale] = useState<UiLocale>("zh");
  const [theme, setTheme] = useState<ThemeMode>("dark");
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
  const [activeChatHistoryId, setActiveChatHistoryId] = useState<number | null>(null);
  const [language, setLanguage] = useState<CodeLanguage>("python");
  const [model, setModel] = useState<ModelOptionValue>("qwen3.5-plus");
  const [code, setCode] = useState(codeTemplates.python);
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [stages, setStages] = useState<Record<StageName, StageState>>(createStageState);
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [finalDiff, setFinalDiff] = useState("");
  const [finalMessage, setFinalMessage] = useState("");
  const [verificationPassed, setVerificationPassed] = useState<boolean | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [diffDecisionMessage, setDiffDecisionMessage] = useState("");
  const [diffApplied, setDiffApplied] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatThinking, setChatThinking] = useState(false);
  const [chatStreamingText, setChatStreamingText] = useState("");
  const [chatError, setChatError] = useState("");

  const streamAbortRef = useRef<AbortController | null>(null);
  const chatAbortRef = useRef<AbortController | null>(null);
  const oauthReturnRef = useRef(false);
  const userMenuRef = useRef<HTMLDivElement | null>(null);
  const historyRequestRef = useRef(0);
  const [sidebarWidthPx, setSidebarWidthPx] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [sidebarResizing, setSidebarResizing] = useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches,
  );
  const sidebarWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH);
  const sidebarResizeStartRef = useRef({ x: 0, w: DEFAULT_SIDEBAR_WIDTH });

  const dict = copy[locale];
  const activeLanguage = languageOptions.find((item) => item.value === language) ?? languageOptions[0];
  const activeModel = modelOptions.find((item) => item.value === model) ?? modelOptions[0];
  const pythonSupported = activeLanguage.supported;

  async function fetchSession() {
    const response = await fetch(`${apiBaseUrl}/api/auth/session`, {
      credentials: "include",
    });
    const data = await response.json();
    setCurrentUser(data.authenticated ? (data.user as AuthenticatedUser) : null);
    setOauthProviders((data.oauth_providers ?? []) as OAuthProvider[]);
    return Boolean(data.authenticated);
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

  function upsertHistoryItem(summary: HistorySummary) {
    setHistoryItems((current) => {
      const filtered = current.filter((item) => item.id !== summary.id);
      return [summary, ...filtered];
    });
  }

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("autorepair-theme") as ThemeMode | null;
    const storedLocale = window.localStorage.getItem("autorepair-locale") as UiLocale | null;
    const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(storedTheme ?? (preferredDark ? "dark" : "light"));
    setLocale(storedLocale ?? "zh");
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem("autorepair-theme", theme);
  }, [theme]);

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
    setActiveChatHistoryId(null);
  }, [currentUser]);

  useEffect(() => {
    return () => {
      chatAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY);
      if (raw) {
        const n = Number(raw);
        if (!Number.isNaN(n)) {
          const clamped = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, n));
          setSidebarWidthPx(clamped);
          sidebarWidthRef.current = clamped;
        }
      }
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    sidebarWidthRef.current = sidebarWidthPx;
  }, [sidebarWidthPx]);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    setIsDesktopLayout(mq.matches);
    const onChange = () => setIsDesktopLayout(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!sidebarResizing) {
      return;
    }
    const onMove = (e: PointerEvent) => {
      const maxW = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.55));
      const next = Math.min(
        maxW,
        Math.max(MIN_SIDEBAR_WIDTH, sidebarResizeStartRef.current.w + (e.clientX - sidebarResizeStartRef.current.x)),
      );
      setSidebarWidthPx(next);
      sidebarWidthRef.current = next;
    };
    const onUp = () => {
      setSidebarResizing(false);
      try {
        window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidthRef.current));
      } catch {
        /* ignore */
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [sidebarResizing]);

  useEffect(() => {
    if (!sidebarResizing) {
      return;
    }
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [sidebarResizing]);

  useEffect(() => {
    setCode(codeTemplates[language]);
  }, [language]);

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

  async function handleSend() {
    if (!pythonSupported) {
      setErrorMessage(dict.unsupported);
      setStatus("error");
      return;
    }

    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    setStatus("streaming");
    setStages(createStageState());
    setEvents([]);
    setRunResult(null);
    setFinalDiff("");
    setFinalMessage("");
    setVerificationPassed(null);
    setErrorMessage("");
    setDiffDecisionMessage("");
    setDiffApplied(false);

    try {
      const response = await fetch(`${apiBaseUrl}/api/repair/stream`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code,
          filename: `snippet.${activeLanguage.extension}`,
          language,
          model,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.text();
        throw new Error(payload || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) {
            continue;
          }
          const parsed = parseSseBlock(block);
          applyEvent(parsed.event, parsed.data as Record<string, unknown>);
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        applyEvent(parsed.event, parsed.data as Record<string, unknown>);
      }

      setStatus((current) => (current === "error" ? current : "done"));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setStatus("error");
    }
  }

  function applyEvent(eventName: string, data: Record<string, unknown>) {
    const entry: EventEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      event: eventName,
      stage: typeof data.stage === "string" ? (data.stage as StageName) : undefined,
      summary: buildSummary(eventName, data),
      at: formatTimestamp(),
    };
    setEvents((current) => [...current, entry]);

    if (eventName === "stage") {
      const stage = data.stage as StageName;
      const nextStatus = data.status as StageState["status"];
      setStages((current) => ({
        ...current,
        [stage]: {
          ...current[stage],
          status: nextStatus,
        },
      }));
      return;
    }

    if (eventName === "run_result") {
      setRunResult({
        stdout: String(data.stdout ?? ""),
        stderr: String(data.stderr ?? ""),
        execution:
          typeof data.execution === "object" && data.execution
            ? {
                returncode: Number((data.execution as Record<string, unknown>).returncode ?? 0),
                duration_sec: Number((data.execution as Record<string, unknown>).duration_sec ?? 0),
                timed_out: Boolean((data.execution as Record<string, unknown>).timed_out ?? false),
              }
            : undefined,
      });
      return;
    }

    if (eventName === "inspect_report") {
      setStages((current) => ({
        ...current,
        inspect: {
          ...current.inspect,
          report: JSON.stringify(data.report ?? {}, null, 2),
        },
      }));
      return;
    }

    if (eventName === "plan_report") {
      setStages((current) => ({
        ...current,
        plan: {
          ...current.plan,
          report: String(data.report ?? ""),
        },
      }));
      return;
    }

    if (eventName === "code_report") {
      const diff = String(data.git_diff ?? "");
      setFinalDiff(diff);
      setStages((current) => ({
        ...current,
        code: {
          ...current.code,
          diff,
          report: diff,
        },
      }));
      return;
    }

    if (eventName === "verify_report") {
      setStages((current) => ({
        ...current,
        verify: {
          ...current.verify,
          report: JSON.stringify(data.report ?? {}, null, 2),
        },
      }));
      return;
    }

    if (eventName === "explain_chunk") {
      const stage = data.stage as StageName;
      const chunk = String(data.chunk ?? "");
      setStages((current) => ({
        ...current,
        [stage]: {
          ...current[stage],
          explain: current[stage].explain + chunk,
          status: "explaining",
        },
      }));
      return;
    }

    if (eventName === "code_diff_chunk") {
      const chunk = String(data.chunk ?? "");
      setStages((current) => ({
        ...current,
        code: {
          ...current.code,
          diff: current.code.diff + chunk,
        },
      }));
      return;
    }

    if (eventName === "tool_event") {
      const stage = data.stage as StageName;
      if (!stage || !["run", "inspect", "plan", "code", "verify"].includes(stage)) {
        return;
      }
      const toolEvent: ToolEventEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        tool_name: String(data.tool_name ?? "tool"),
        status: data.status === "completed" ? "completed" : "started",
        round: typeof data.round === "number" ? data.round : undefined,
        arguments: typeof data.arguments === "string" ? data.arguments : undefined,
        output_preview:
          typeof data.output_preview === "string" ? data.output_preview : undefined,
        output_truncated: Boolean(data.output_truncated ?? false),
        at: formatTimestamp(),
      };
      setStages((current) => ({
        ...current,
        [stage]: {
          ...current[stage],
          toolEvents: [...current[stage].toolEvents, toolEvent],
        },
      }));
      return;
    }

    if (eventName === "error") {
      setErrorMessage(String(data.message ?? ""));
      setVerificationPassed(false);
      if (typeof data.history_id === "number") {
        const historyId = Number(data.history_id);
        setSelectedHistoryId(historyId);
        upsertHistoryItem({
          id: historyId,
          mode: "agent",
          title: `Repair · ${String(data.filename ?? "main.py")}`,
          preview_text: String(data.message ?? (errorMessage || "Agent run failed.")),
          model,
          language,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        void fetchHistoryList();
      }
      setStatus("error");
      return;
    }

    if (eventName === "result") {
      const resultStatus = String(data.status ?? "");
      const passed =
        typeof data.verification_passed === "boolean"
          ? Boolean(data.verification_passed)
          : resultStatus === "verified";
      setVerificationPassed(
        resultStatus === "clean" ? null : resultStatus === "verified" ? true : passed ? true : false,
      );
      setFinalMessage(
        resultStatus === "clean"
          ? dict.cleanMessage
          : resultStatus === "verified"
            ? dict.verificationReady
            : resultStatus === "verify_failed"
              ? dict.verificationFailed
              : typeof data.message === "string" && data.message
                ? data.message
                : dict.repairedMessage,
      );
      if (typeof data.git_diff === "string") {
        setFinalDiff(data.git_diff);
      }
      if (typeof data.history_id === "number") {
        const historyId = Number(data.history_id);
        setSelectedHistoryId(historyId);
        upsertHistoryItem({
          id: historyId,
          mode: "agent",
          title: `Repair · ${String(data.filename ?? "main.py")}`,
          preview_text:
            resultStatus === "clean"
              ? dict.cleanMessage
              : resultStatus === "verified"
                ? dict.verificationReady
                : resultStatus === "verify_failed"
                  ? dict.verificationFailed
                  : dict.repairedMessage,
          model,
          language,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        void fetchHistoryList();
      }
    }
  }

  async function handleLogout() {
    await fetch(`${apiBaseUrl}/api/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
    setUserMenuOpen(false);
    setCurrentUser(null);
    setEvents([]);
    setRunResult(null);
    setFinalDiff("");
    setFinalMessage("");
    setVerificationPassed(null);
    setErrorMessage("");
    setDiffDecisionMessage("");
    setDiffApplied(false);
    setStatus("idle");
    setWorkspaceMode("agent");
    setHistoryItems([]);
    setSelectedHistoryId(null);
    setActiveChatHistoryId(null);
    setChatMessages([]);
    setChatInput("");
    setChatThinking(false);
    setChatStreamingText("");
    setChatError("");
  }

  function handleReset() {
    streamAbortRef.current?.abort();
    setStatus("idle");
    setStages(createStageState());
    setEvents([]);
    setRunResult(null);
    setFinalDiff("");
    setFinalMessage("");
    setVerificationPassed(null);
    setErrorMessage("");
    setDiffDecisionMessage("");
    setDiffApplied(false);
    setSelectedHistoryId(null);
  }

  function startNewAgentSession() {
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setWorkspaceMode("agent");
    setActiveChatHistoryId(null);
    setSelectedHistoryId(null);
    setCode(codeTemplates[language]);
    setChatMessages([]);
    setChatInput("");
    setChatThinking(false);
    setChatStreamingText("");
    setChatError("");
    handleReset();
  }

  function startNewChatSession() {
    streamAbortRef.current?.abort();
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setWorkspaceMode("chat");
    setActiveChatHistoryId(null);
    setSelectedHistoryId(null);
    setStatus("idle");
    setStages(createStageState());
    setEvents([]);
    setRunResult(null);
    setFinalDiff("");
    setFinalMessage("");
    setVerificationPassed(null);
    setErrorMessage("");
    setDiffDecisionMessage("");
    setDiffApplied(false);
    setChatMessages([]);
    setChatInput("");
    setChatThinking(false);
    setChatStreamingText("");
    setChatError("");
  }

  function handleApplyDiff() {
    try {
      const nextCode = applyUnifiedDiffToText(code, finalDiff);
      setCode(nextCode);
      setDiffApplied(true);
      setDiffDecisionMessage(dict.applySuccess);
    } catch (error) {
      setDiffDecisionMessage(
        `${dict.applyFailed}${error instanceof Error ? ` ${error.message}` : ""}`.trim(),
      );
    }
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
      setActiveChatHistoryId(detail.id);
        if (detail.model) {
          setModel(detail.model);
        }
        setChatMessages(
          (snapshot.messages ?? []).map((message, index) => ({
            id: `history-${detail.id}-${index}`,
            role: message.role,
            content: message.content,
            at: message.at,
          })),
        );
        setChatInput("");
        setChatThinking(false);
        setChatStreamingText("");
        setChatError("");
        setVerificationPassed(null);
        setFinalDiff("");
        setFinalMessage("");
        setErrorMessage("");
        setStatus("idle");
        return;
      }

      const snapshot = detail.snapshot as AgentHistorySnapshot;
      setWorkspaceMode("agent");
      setActiveChatHistoryId(null);
      setChatMessages([]);
      setChatInput("");
      setChatThinking(false);
      setChatStreamingText("");
      setChatError("");
      setLanguage(snapshot.language ?? "python");
      setCode(snapshot.code ?? codeTemplates[snapshot.language ?? "python"]);
      if (snapshot.model) {
        setModel(snapshot.model);
      }
      setRunResult(snapshot.run_result ?? null);
      setStages(normalizeStageMap(snapshot.stages));
      setEvents(snapshot.events ?? []);
      setFinalDiff(snapshot.final_diff ?? "");
      setFinalMessage(
        snapshot.final_status === "clean"
          ? dict.cleanMessage
          : snapshot.final_status === "verified"
            ? dict.verificationReady
            : snapshot.final_status === "verify_failed"
              ? dict.verificationFailed
              : snapshot.final_status === "repaired"
                ? dict.repairedMessage
            : "",
      );
      setVerificationPassed(
        snapshot.final_status === "verified"
          ? true
          : snapshot.final_status === "verify_failed"
            ? false
            : null,
      );
      setErrorMessage(snapshot.error_message ?? "");
      setDiffDecisionMessage("");
      setDiffApplied(false);
      setStatus(snapshot.error_message ? "error" : snapshot.final_status ? "done" : "idle");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
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
      if (activeChatHistoryId === historyId) {
        setActiveChatHistoryId(null);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : dict.historyDeleteFailed;
      setErrorMessage(message);
      setChatError(message);
    } finally {
      setDeletingHistoryId(null);
    }
  }

  async function handleChatSend() {
    const nextMessage = chatInput.trim();
    if (!nextMessage || chatThinking) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: nextMessage,
      at: formatTimestamp(),
    };

    setChatMessages((current) => [...current, userMessage]);
    setChatInput("");
    setChatThinking(true);
    setChatStreamingText("");
    setChatError("");

    chatAbortRef.current?.abort();
    const controller = new AbortController();
    chatAbortRef.current = controller;

    const requestMessages = [...chatMessages, userMessage].map((message) => ({
      role: message.role,
      content: message.content,
      at: message.at,
    }));

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat/stream`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: requestMessages,
          model,
          history_id: activeChatHistoryId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.text();
        throw new Error(payload || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalMessage = "";
      let returnedHistoryId: number | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) {
            continue;
          }
          const parsed = parseSseBlock(block);
          if (parsed.event === "chat_chunk") {
            const chunk = String((parsed.data as Record<string, unknown>).chunk ?? "");
            if (chunk) {
              finalMessage += chunk;
              setChatStreamingText((current) => current + chunk);
            }
          } else if (parsed.event === "result") {
            finalMessage = String((parsed.data as Record<string, unknown>).message ?? finalMessage);
            if (typeof (parsed.data as Record<string, unknown>).history_id === "number") {
              returnedHistoryId = Number((parsed.data as Record<string, unknown>).history_id);
            }
            setChatStreamingText(finalMessage);
          } else if (parsed.event === "error") {
            throw new Error(String((parsed.data as Record<string, unknown>).message ?? "Chat request failed."));
          }
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        if (parsed.event === "chat_chunk") {
          const chunk = String((parsed.data as Record<string, unknown>).chunk ?? "");
          if (chunk) {
            finalMessage += chunk;
            setChatStreamingText((current) => current + chunk);
          }
        } else if (parsed.event === "result") {
          finalMessage = String((parsed.data as Record<string, unknown>).message ?? finalMessage);
          if (typeof (parsed.data as Record<string, unknown>).history_id === "number") {
            returnedHistoryId = Number((parsed.data as Record<string, unknown>).history_id);
          }
          setChatStreamingText(finalMessage);
        } else if (parsed.event === "error") {
          throw new Error(String((parsed.data as Record<string, unknown>).message ?? "Chat request failed."));
        }
      }

      const cleanedMessage = finalMessage.trim();
      if (!cleanedMessage) {
        throw new Error("Chat model returned an empty response.");
      }

      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: cleanedMessage,
        at: formatTimestamp(),
      };
      setChatMessages((current) => [...current, assistantMessage]);
      setChatStreamingText("");
      if (returnedHistoryId) {
        setActiveChatHistoryId(returnedHistoryId);
        setSelectedHistoryId(returnedHistoryId);
        upsertHistoryItem({
          id: returnedHistoryId,
          mode: "chat",
          title: nextMessage.slice(0, 80),
          preview_text: cleanedMessage.slice(0, 120),
          model,
          language: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }
      void fetchHistoryList();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      setChatError(error instanceof Error ? error.message : String(error));
      setChatStreamingText("");
    } finally {
      setChatThinking(false);
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null;
      }
    }
  }

  function userInitials(user: AuthenticatedUser) {
    const parts = user.display_name.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) {
      return user.email.slice(0, 2).toUpperCase();
    }
    return parts
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("");
  }

  function handleSidebarResizePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!isDesktopLayout) {
      return;
    }
    event.preventDefault();
    sidebarResizeStartRef.current = { x: event.clientX, w: sidebarWidthPx };
    sidebarWidthRef.current = sidebarWidthPx;
    setSidebarResizing(true);
  }

  function handleSidebarResizeReset() {
    setSidebarWidthPx(DEFAULT_SIDEBAR_WIDTH);
    sidebarWidthRef.current = DEFAULT_SIDEBAR_WIDTH;
    try {
      window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(DEFAULT_SIDEBAR_WIDTH));
    } catch {
      /* ignore */
    }
  }

  const workspaceMainClass = `min-h-0 min-w-0 h-full ${!isDesktopLayout ? "flex-1" : ""}`.trim();

  const statusText =
    status === "streaming"
      ? dict.statusStreaming
      : status === "done"
        ? dict.statusDone
        : status === "error"
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
        <div className="relative z-40 flex shrink-0 items-center justify-between gap-3">
          {currentUser ? (
            <div className="min-w-0">
              <div className="font-display text-xl tracking-tight text-slate-950 dark:text-white sm:text-2xl">
                AutoRepair Studio
              </div>
            </div>
          ) : (
            <div />
          )}
          <div className="flex justify-end gap-3">
            <button
              onClick={() => setLocale((current) => (current === "zh" ? "en" : "zh"))}
              aria-label={dict.locale}
              title={dict.locale}
              className="flex h-11 min-w-11 items-center justify-center rounded-full border border-black/10 bg-white/65 px-3 text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
            >
              <LanguageIcon />
            </button>
            <button
              onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
              aria-label={theme === "dark" ? dict.themeLight : dict.themeDark}
              title={theme === "dark" ? dict.themeLight : dict.themeDark}
              className="flex h-11 min-w-11 items-center justify-center rounded-full border border-black/10 bg-white/65 px-3 text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
            {currentUser ? (
              <div ref={userMenuRef} className="relative z-40 flex items-center">
                <button
                  onClick={() => setUserMenuOpen((current) => !current)}
                  className="flex h-11 items-center gap-2 rounded-full border border-black/10 bg-white/65 px-2.5 pr-3 text-left text-sm text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/80"
                >
                  {currentUser.avatar_url ? (
                    <img
                      src={currentUser.avatar_url}
                      alt={currentUser.display_name}
                      className="h-8 w-8 rounded-full object-cover"
                    />
                  ) : (
                    <div className="grid h-8 w-8 place-items-center rounded-full bg-slate-900 text-xs font-semibold text-white dark:bg-white dark:text-slate-950">
                      {userInitials(currentUser)}
                    </div>
                  )}
                  <div className="min-w-0 max-w-[9rem]">
                    <div className="truncate font-medium">{currentUser.display_name}</div>
                  </div>
                  <div className={`text-xs transition ${userMenuOpen ? "rotate-180" : ""}`}>▾</div>
                </button>
                {userMenuOpen ? (
                  <div className="absolute right-0 top-[calc(100%+0.75rem)] z-[60] min-w-[220px] rounded-[24px] border border-black/10 bg-white/92 p-2 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-ink-900/95 dark:shadow-glow">
                    <div className="rounded-[18px] px-3 py-3 text-sm text-slate-600 dark:text-white/70">
                      <div className="font-medium text-slate-900 dark:text-white">{currentUser.display_name}</div>
                      <div className="mt-1 text-xs text-slate-500 dark:text-white/45">{currentUser.email}</div>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="flex w-full items-center justify-between rounded-[18px] px-3 py-3 text-left text-sm font-medium text-slate-700 transition hover:bg-black/[0.04] dark:text-white/80 dark:hover:bg-white/[0.05]"
                    >
                      <span>{dict.logout}</span>
                      <span className="text-slate-400 dark:text-white/35">↗</span>
                    </button>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        {sessionLoading ? (
          <div className="grid min-h-[82vh] place-items-center">
            <div className="rounded-[32px] border border-black/5 bg-white/72 px-8 py-6 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
              <div className="text-sm uppercase tracking-[0.28em] text-slate-500 dark:text-white/45">{dict.authLoading}</div>
            </div>
          </div>
        ) : null}

        {!sessionLoading && !currentUser ? (
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
              <aside
                className={`flex h-full min-h-0 flex-col gap-2 overflow-hidden pr-1 ${!isDesktopLayout ? "shrink-0" : ""}`}
              >
                <section className="shrink-0 rounded-[22px] border border-black/5 bg-white/72 p-2.5 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
                  <div className="px-2 text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">
                    {dict.sidebarWorkspace}
                  </div>
                  <div className="mt-2 space-y-1.5">
                    <button
                      onClick={startNewAgentSession}
                      className={`flex w-full items-center gap-2 rounded-[16px] px-2.5 py-2.5 text-left transition ${
                        workspaceMode === "agent"
                          ? "bg-slate-900 text-white shadow-lg dark:bg-white dark:text-slate-950"
                          : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="grid h-8 w-8 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
                        <AgentIcon />
                      </div>
                      <div className="min-w-0">
                        <div className="font-medium">{dict.modeAgent}</div>
                        <div className={`text-xs ${workspaceMode === "agent" ? "text-white/70 dark:text-slate-950/70" : "text-slate-500 dark:text-white/40"}`}>
                          {dict.modeAgentHint}
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={startNewChatSession}
                      className={`flex w-full items-center gap-2 rounded-[16px] px-2.5 py-2.5 text-left transition ${
                        workspaceMode === "chat"
                          ? "bg-slate-900 text-white shadow-lg dark:bg-white dark:text-slate-950"
                          : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="grid h-8 w-8 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
                        <ChatIcon />
                      </div>
                      <div className="min-w-0">
                        <div className="font-medium">{dict.modeChat}</div>
                        <div className={`text-xs ${workspaceMode === "chat" ? "text-white/70 dark:text-slate-950/70" : "text-slate-500 dark:text-white/40"}`}>
                          {dict.modeChatHint}
                        </div>
                      </div>
                    </button>
                  </div>
                </section>

                <section className="flex min-h-0 flex-1 flex-col rounded-[22px] border border-black/5 bg-white/72 p-2.5 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
                  <div className="shrink-0 px-2 text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/40">
                    {dict.sidebarHistory}
                  </div>
                  <div className="mt-2 min-h-0 flex-1 space-y-1.5 overflow-y-auto">
                    {historyLoading && historyItems.length === 0 ? (
                      <div className="rounded-[22px] bg-black/[0.03] px-4 py-4 text-sm text-slate-500 dark:bg-white/[0.03] dark:text-white/45">
                        {dict.historyLoading}
                      </div>
                    ) : historyItems.length === 0 ? (
                      <div className="rounded-[22px] bg-black/[0.03] px-4 py-4 text-sm text-slate-500 dark:bg-white/[0.03] dark:text-white/45">
                        {dict.historyEmpty}
                      </div>
                    ) : (
                      <div className="space-y-1.5 pr-1">
                        {historyLoading ? (
                          <div className="px-2 pb-1 text-[11px] uppercase tracking-[0.18em] text-slate-400 dark:text-white/28">
                            {dict.historyLoading}
                          </div>
                        ) : null}
                        {historyItems.map((item) => (
                          <div
                            key={item.id}
                            className={`flex items-start gap-2 rounded-[16px] px-2 py-2 transition ${
                              selectedHistoryId === item.id
                                ? "bg-slate-900 text-white shadow-lg dark:bg-white dark:text-slate-950"
                                : "bg-black/[0.03] text-slate-700 hover:bg-black/[0.05] dark:bg-white/[0.03] dark:text-white/75 dark:hover:bg-white/[0.06]"
                            }`}
                          >
                            <button
                              onClick={() => void handleHistoryOpen(item.id)}
                              className="flex min-w-0 flex-1 items-start gap-2 text-left"
                            >
                              <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-black/10 dark:bg-white/10">
                                {item.mode === "agent" ? <AgentIcon /> : <ChatIcon />}
                              </div>
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-medium">{item.title}</div>
                                <div
                                  className={`mt-1 line-clamp-2 text-xs ${
                                    selectedHistoryId === item.id
                                      ? "text-white/70 dark:text-slate-950/70"
                                      : "text-slate-500 dark:text-white/40"
                                  }`}
                                >
                                  {item.preview_text}
                                </div>
                                <div
                                  className={`mt-2 text-[11px] uppercase tracking-[0.18em] ${
                                    selectedHistoryId === item.id
                                      ? "text-white/60 dark:text-slate-950/60"
                                      : "text-slate-400 dark:text-white/28"
                                  }`}
                                >
                                  {item.model ?? ""} {item.updated_at}
                                </div>
                              </div>
                            </button>
                            <button
                              onClick={() => void handleDeleteHistory(item.id)}
                              disabled={deletingHistoryId === item.id}
                              aria-label={dict.historyDelete}
                              title={dict.historyDelete}
                              className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg transition ${
                                selectedHistoryId === item.id
                                  ? "text-white/75 hover:bg-white/10 dark:text-slate-950/70 dark:hover:bg-slate-950/10"
                                  : "text-slate-400 hover:bg-black/[0.05] hover:text-rose-500 dark:text-white/35 dark:hover:bg-white/[0.06] dark:hover:text-rose-300"
                              } disabled:cursor-not-allowed disabled:opacity-40`}
                            >
                              <TrashIcon />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </section>
              </aside>

              {isDesktopLayout ? (
                <div
                  role="separator"
                  aria-orientation="vertical"
                  aria-valuemin={MIN_SIDEBAR_WIDTH}
                  aria-valuemax={MAX_SIDEBAR_WIDTH}
                  aria-valuenow={Math.round(sidebarWidthPx)}
                  aria-label={locale === "zh" ? "拖动调整侧栏宽度，双击恢复默认" : "Drag to resize sidebar; double-click to reset"}
                  onPointerDown={handleSidebarResizePointerDown}
                  onDoubleClick={handleSidebarResizeReset}
                  className={`relative z-20 w-[6px] shrink-0 cursor-col-resize touch-none select-none ${
                    sidebarResizing ? "bg-black/[0.06] dark:bg-white/[0.08]" : ""
                  }`}
                >
                  <div className="pointer-events-none absolute inset-y-3 left-1/2 w-px -translate-x-1/2 rounded-full bg-black/15 dark:bg-white/20" />
                </div>
              ) : null}

              {workspaceMode === "agent" ? (
                <div className={workspaceMainClass}>
                  <div className="grid h-full min-h-0 items-stretch gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden pr-1">
                      <div className="flex min-h-0 flex-1 flex-col rounded-[24px] border border-black/5 bg-white/72 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
                        <div className="shrink-0 flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                          <div>
                            <div className="text-xs uppercase tracking-[0.32em] text-slate-500 dark:text-white/40">{dict.editorTitle}</div>
                            <div className="mt-2 text-sm text-slate-600 dark:text-white/65">{dict.editorHint}</div>
                          </div>
                          <div className="flex flex-wrap gap-3">
                            <select
                              value={language}
                              onChange={(event) => setLanguage(event.target.value as CodeLanguage)}
                              className="rounded-full border border-black/10 bg-white/70 px-4 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                            >
                              {languageOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                            <select
                              value={model}
                              onChange={(event) => setModel(event.target.value as ModelOptionValue)}
                              className="rounded-full border border-black/10 bg-white/70 px-4 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                            >
                              {modelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {dict.model}: {option.label}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        {!pythonSupported ? (
                          <div className="mt-4 shrink-0 rounded-3xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-200">
                            {dict.unsupported}
                          </div>
                        ) : null}

                        <div className="mt-3 min-h-0 flex-1 flex flex-col">
                          <CodeEditor value={code} onChange={setCode} placeholder={dict.placeholder} />
                        </div>

                        <div className="mt-3 shrink-0 flex flex-wrap items-center gap-2">
                          <button
                            onClick={handleSend}
                            disabled={status === "streaming" || !pythonSupported}
                            className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-45 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                          >
                            {dict.send}
                          </button>
                          <button
                            onClick={() => {
                              streamAbortRef.current?.abort();
                              setStatus("error");
                            }}
                            disabled={status !== "streaming"}
                            className="rounded-full border border-black/10 bg-white/70 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                          >
                            {dict.stop}
                          </button>
                          <button
                            onClick={handleReset}
                            className="rounded-full border border-black/10 bg-white/70 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                          >
                            {dict.reset}
                          </button>
                          <div className="rounded-full border border-black/10 bg-white/70 px-4 py-3 text-sm text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-white/70">
                            {statusText}
                          </div>
                        </div>
                      </div>

                    </section>

                    <section className="h-full min-h-0 min-w-0 space-y-3 overflow-y-auto pr-1">
                      {(runResult || errorMessage || finalMessage || finalDiff) ? (
                        <section className="rounded-[20px] border border-black/5 bg-white/75 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow">
                          <div className="flex items-center justify-between gap-4">
                            <div className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/80">{dict.runOutput}</div>
                            {runResult?.execution ? (
                              <div className="text-xs text-slate-500 dark:text-white/75">
                                rc={runResult.execution.returncode} · {runResult.execution.duration_sec.toFixed(2)}s
                              </div>
                            ) : null}
                          </div>
                          {runResult ? (
                            <div className="mt-4 grid gap-4 md:grid-cols-2">
                              <div>
                                <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/80">{dict.stdout}</div>
                                <pre className="min-h-[120px] whitespace-pre-wrap break-words rounded-3xl bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:bg-slate-950/95 dark:text-white">
                                  {runResult.stdout || "∅"}
                                </pre>
                              </div>
                              <div>
                                <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/80">{dict.stderr}</div>
                                <pre className="min-h-[120px] whitespace-pre-wrap break-words rounded-3xl bg-black/[0.03] p-4 font-mono text-xs leading-6 text-slate-700 [overflow-wrap:anywhere] dark:bg-slate-950/95 dark:text-white">
                                  {runResult.stderr || "∅"}
                                </pre>
                              </div>
                            </div>
                          ) : null}
                          {errorMessage ? (
                            <div className="mt-4 rounded-3xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                              {errorMessage}
                            </div>
                          ) : null}
                          {finalMessage ? (
                            <div className="mt-4 rounded-3xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-200">
                              {finalMessage}
                            </div>
                          ) : null}
                          {finalDiff ? (
                            <div className="mt-4">
                              <div className="mb-2 text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
                                {dict.finalDiff}
                              </div>
                              <pre className="overflow-y-auto whitespace-pre-wrap break-words rounded-3xl bg-slate-950 p-5 font-mono text-xs leading-6 text-slate-100 [overflow-wrap:anywhere] dark:bg-ink-900">
                                {finalDiff}
                              </pre>
                              {verificationPassed ? (
                                <div className="mt-4 flex flex-wrap items-center gap-3">
                                  <div className="text-sm text-slate-600 dark:text-white/70">{dict.applyPrompt}</div>
                                  <button
                                    onClick={handleApplyDiff}
                                    disabled={diffApplied}
                                    className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                                  >
                                    {dict.applyAccept}
                                  </button>
                                  <button
                                    onClick={() => setDiffDecisionMessage(dict.applySkipped)}
                                    className="rounded-full border border-black/10 bg-white/70 px-4 py-2 text-sm text-slate-700 transition hover:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
                                  >
                                    {dict.applyDecline}
                                  </button>
                                </div>
                              ) : null}
                              {diffDecisionMessage ? (
                                <div className="mt-4 rounded-3xl border border-black/10 bg-black/[0.03] px-4 py-3 text-sm text-slate-700 dark:border-white/10 dark:bg-white/[0.03] dark:text-white/75">
                                  {diffDecisionMessage}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </section>
                      ) : null}

                      {stageOrder.map((stage) => (
                        <StageCard
                          key={stage}
                          locale={locale}
                          stage={stage}
                          state={stages[stage]}
                          copy={dict}
                        />
                      ))}
                    </section>
                  </div>
                </div>
              ) : (
                <section
                  className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/72 p-3 shadow-float backdrop-blur-xl dark:border-white/10 dark:bg-white/5 dark:shadow-glow ${!isDesktopLayout ? "flex-1" : ""}`}
                >
                  <div className="shrink-0 flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                    <div>
                      <div className="text-xs uppercase tracking-[0.32em] text-slate-500 dark:text-white/40">{dict.chatTitle}</div>
                      <div className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-white/65">{dict.chatHint}</div>
                    </div>
                    <div className="flex flex-wrap gap-3">
                      <select
                        value={model}
                        onChange={(event) => setModel(event.target.value as ModelOptionValue)}
                        className="rounded-full border border-black/10 bg-white/70 px-4 py-2 text-sm text-slate-900 outline-none dark:border-white/10 dark:bg-white/5 dark:text-white"
                      >
                        {modelOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {dict.model}: {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="mt-3 min-h-0 flex-1 overflow-hidden rounded-[20px] border border-black/5 bg-black/[0.025] dark:border-white/10 dark:bg-white/[0.025]">
                    <div className="flex h-full min-h-[8rem] flex-col gap-2.5 overflow-y-auto p-2.5">
                      {chatMessages.length === 0 ? (
                        <div className="flex flex-1 items-center justify-center rounded-[24px] border border-dashed border-black/10 px-6 text-center text-sm text-slate-500 dark:border-white/10 dark:text-white/40">
                          {dict.chatEmpty}
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
                            <div className={`mt-2 text-[11px] ${message.role === "user" ? "text-white/65 dark:text-slate-950/55" : "text-slate-400 dark:text-white/35"}`}>
                              {message.at}
                            </div>
                          </div>
                        ))
                      )}
                      {chatThinking && !chatStreamingText ? (
                        <div className="max-w-[85%] rounded-[24px] bg-white/85 px-4 py-3 text-slate-700 dark:bg-slate-900/90 dark:text-white">
                          <div className="text-sm">{dict.chatThinking}</div>
                        </div>
                      ) : null}
                      {chatStreamingText ? (
                        <div className="max-w-[85%] rounded-[24px] bg-white/85 px-4 py-3 text-slate-700 dark:bg-slate-900/90 dark:text-white">
                          <div className="whitespace-pre-wrap break-words text-sm leading-7">{chatStreamingText}</div>
                        </div>
                      ) : null}
                      {chatError ? (
                        <div className="max-w-[85%] rounded-[24px] border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                          {chatError}
                        </div>
                      ) : null}
                    </div>
                  </div>

                  <div className="mt-3 shrink-0 rounded-[20px] border border-black/5 bg-white/70 p-2.5 dark:border-white/10 dark:bg-white/[0.03]">
                    <textarea
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      placeholder={dict.chatPlaceholder}
                      className="min-h-[48px] max-h-[120px] w-full resize-y bg-transparent text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 dark:text-white dark:placeholder:text-white/28"
                    />
                    <div className="mt-2 flex items-center justify-between gap-3">
                      <div className="text-xs uppercase tracking-[0.22em] text-slate-500 dark:text-white/35">
                        {dict.model}: {activeModel.label}
                      </div>
                      <button
                        onClick={handleChatSend}
                        disabled={!chatInput.trim() || chatThinking}
                        className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-45 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
                      >
                        {dict.chatSend}
                      </button>
                    </div>
                  </div>
                </section>
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
