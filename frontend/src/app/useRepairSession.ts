import { useEffect, useRef, useState } from "react";
import { AppCopy, codeTemplates, languageOptions } from "../i18n";
import type {
  AgentHistorySnapshot,
  AgentSourceType,
  CodeLanguage,
  EventEntry,
  HistorySummary,
  ModelOptionValue,
  ProjectEntrypointOption,
  RunResult,
  SessionStatus,
  StageName,
  StageState,
  ToolEventEntry,
} from "../types";
import {
  applyUnifiedDiffToText,
  buildSummary,
  createStageState,
  fileToBase64,
  formatTimestamp,
  isStageName,
  normalizeStageMap,
  parseSseBlock,
} from "./utils";

function defaultEntrypointForLanguage(language: CodeLanguage) {
  switch (language) {
    case "javascript":
      return "index.js";
    case "typescript":
      return "index.ts";
    case "java":
      return "Main.java";
    case "go":
      return "main.go";
    case "c":
      return "main.c";
    case "cpp":
      return "main.cpp";
    case "python":
    default:
      return "main.py";
  }
}

const knownDefaultEntrypoints = new Set<string>([
  "main.py",
  "index.js",
  "index.ts",
  "Main.java",
  "main.go",
  "main.c",
  "main.cpp",
]);

interface UseRepairSessionOptions {
  apiBaseUrl: string;
  dict: AppCopy;
  model: ModelOptionValue;
  refreshHistoryList: () => void | Promise<void>;
  selectHistory: (historyId: number) => void;
  upsertHistoryItem: (summary: HistorySummary) => void;
}

export function useRepairSession({
  apiBaseUrl,
  dict,
  model,
  refreshHistoryList,
  selectHistory,
  upsertHistoryItem,
}: UseRepairSessionOptions) {
  const [agentSourceType, setAgentSourceType] = useState<AgentSourceType>("single_file");
  const [language, setLanguage] = useState<CodeLanguage>("python");
  const [code, setCode] = useState(codeTemplates.python);
  const [inputText, setInputText] = useState("");
  const [entrypointPath, setEntrypointPath] = useState(defaultEntrypointForLanguage("python"));
  const [projectSubdir, setProjectSubdir] = useState("");
  const [githubRepoUrl, setGithubRepoUrl] = useState("");
  const [githubRef, setGithubRef] = useState("");
  const [zipFileName, setZipFileName] = useState("");
  const [zipFileBase64, setZipFileBase64] = useState("");
  const [projectEntrypointOptions, setProjectEntrypointOptions] = useState<ProjectEntrypointOption[]>([]);
  const [projectFilesLoading, setProjectFilesLoading] = useState(false);
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

  const streamAbortRef = useRef<AbortController | null>(null);
  const projectFilesRequestRef = useRef(0);
  const activeLanguage = languageOptions.find((item) => item.value === language) ?? languageOptions[0];
  const languageSupported = activeLanguage.supported;

  useEffect(() => {
    if (agentSourceType !== "single_file") {
      return;
    }
    setCode(codeTemplates[language]);
  }, [agentSourceType, language]);

  useEffect(() => {
    setEntrypointPath((current) => {
      const trimmed = current.trim();
      if (!trimmed || knownDefaultEntrypoints.has(trimmed)) {
        return defaultEntrypointForLanguage(language);
      }
      return current;
    });
  }, [language]);

  useEffect(() => {
    return () => {
      streamAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (agentSourceType === "single_file") {
      setProjectEntrypointOptions([]);
      setProjectFilesLoading(false);
      return;
    }

    const payload =
      agentSourceType === "zip"
        ? zipFileBase64
          ? {
              project_zip_base64: zipFileBase64,
              ...(projectSubdir.trim() ? { project_subdir: projectSubdir.trim() } : {}),
            }
          : null
        : githubRepoUrl.trim()
          ? {
              github_repo_url: githubRepoUrl.trim(),
              ...(githubRef.trim() ? { github_ref: githubRef.trim() } : {}),
              ...(projectSubdir.trim() ? { project_subdir: projectSubdir.trim() } : {}),
            }
          : null;

    if (!payload) {
      setProjectEntrypointOptions([]);
      setProjectFilesLoading(false);
      return;
    }

    const requestId = projectFilesRequestRef.current + 1;
    projectFilesRequestRef.current = requestId;
    setProjectFilesLoading(true);

    const timeoutId = window.setTimeout(() => {
      void (async () => {
        try {
          const response = await fetch(`${apiBaseUrl}/api/repair/project-files`, {
            method: "POST",
            credentials: "include",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });
          const data = (await response.json().catch(() => ({}))) as {
            error?: string;
            entrypoint_options?: ProjectEntrypointOption[];
          };
          if (!response.ok) {
            throw new Error(data.error || dict.projectFilesLoadFailed);
          }
          if (projectFilesRequestRef.current !== requestId) {
            return;
          }
          const nextOptions = Array.isArray(data.entrypoint_options) ? data.entrypoint_options : [];
          setProjectEntrypointOptions(nextOptions);
          setEntrypointPath((current) => {
            const preferred =
              nextOptions.find((item) => item.path === current) ??
              nextOptions[0];
            if (preferred) {
              setLanguage(preferred.language);
              return preferred.path;
            }
            return "";
          });
          setErrorMessage(nextOptions.length === 0 ? dict.projectFilesEmpty : "");
          if (nextOptions.length > 0) {
            setStatus((current) => (current === "error" ? "idle" : current));
          }
        } catch (error) {
          if (projectFilesRequestRef.current !== requestId) {
            return;
          }
          setProjectEntrypointOptions([]);
          setEntrypointPath("");
          setErrorMessage(error instanceof Error ? error.message : String(error));
          setStatus("error");
        } finally {
          if (projectFilesRequestRef.current === requestId) {
            setProjectFilesLoading(false);
          }
        }
      })();
    }, 250);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [agentSourceType, apiBaseUrl, dict.projectFilesEmpty, dict.projectFilesLoadFailed, githubRef, githubRepoUrl, projectSubdir, zipFileBase64]);

  function resetRepairState(shouldAbort = true) {
    if (shouldAbort) {
      streamAbortRef.current?.abort();
    }
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
  }

  async function handleZipSelected(file: File | null) {
    if (!file) {
      setZipFileName("");
      setZipFileBase64("");
      setProjectEntrypointOptions([]);
      setEntrypointPath("");
      return;
    }

    try {
      const encoded = await fileToBase64(file);
      setZipFileName(file.name);
      setZipFileBase64(encoded);
      setErrorMessage("");
      setStatus("idle");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setStatus("error");
    }
  }

  function buildRepairPayload() {
    const normalizedEntrypoint = entrypointPath.trim();
    const normalizedSubdir = projectSubdir.trim();
    const selectedProjectEntrypoint =
      normalizedEntrypoint || projectEntrypointOptions[0]?.path || "";
    const selectedProjectLanguage =
      projectEntrypointOptions.find((item) => item.path === selectedProjectEntrypoint)?.language ?? language;

    if (agentSourceType === "single_file") {
      return {
        code,
        input_text: inputText,
        filename: normalizedEntrypoint || defaultEntrypointForLanguage(language),
        language,
        model,
      };
    }

    if (agentSourceType === "zip") {
      if (!zipFileBase64) {
        throw new Error(dict.zipRequired);
      }
      if (!selectedProjectEntrypoint) {
        throw new Error(dict.projectFilesEmpty);
      }
      return {
        filename: selectedProjectEntrypoint,
        input_text: inputText,
        language: selectedProjectLanguage,
        model,
        project_zip_base64: zipFileBase64,
        ...(normalizedSubdir ? { project_subdir: normalizedSubdir } : {}),
      };
    }

    const normalizedRepoUrl = githubRepoUrl.trim();
    const normalizedRef = githubRef.trim();
    if (!normalizedRepoUrl) {
      throw new Error(dict.githubRepoRequired);
    }
    if (!selectedProjectEntrypoint) {
      throw new Error(dict.projectFilesEmpty);
    }
    return {
      filename: selectedProjectEntrypoint,
      input_text: inputText,
      language: selectedProjectLanguage,
      model,
      github_repo_url: normalizedRepoUrl,
      ...(normalizedRef ? { github_ref: normalizedRef } : {}),
      ...(normalizedSubdir ? { project_subdir: normalizedSubdir } : {}),
    };
  }

  function applyEvent(eventName: string, data: Record<string, unknown>) {
    const entry: EventEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      event: eventName,
      stage: isStageName(data.stage) ? data.stage : undefined,
      summary: buildSummary(eventName, data),
      at: formatTimestamp(),
    };
    setEvents((current) => [...current, entry]);

    if (eventName === "stage") {
      const stage = data.stage;
      if (!isStageName(stage)) {
        return;
      }
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
        input_text: typeof data.input_text === "string" ? data.input_text : undefined,
        stdout: String(data.stdout ?? ""),
        stderr: String(data.stderr ?? ""),
        entrypoint: typeof data.entrypoint === "string" ? data.entrypoint : undefined,
        source_type:
          data.source_type === "single_file" || data.source_type === "zip" || data.source_type === "github"
            ? data.source_type
            : undefined,
        file_count: typeof data.file_count === "number" ? Number(data.file_count) : undefined,
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
      const report =
        typeof data.report === "string" && data.report.trim()
          ? data.report
          : diff;
      if (diff) {
        setFinalDiff(diff);
      }
      setStages((current) => ({
        ...current,
        code: {
          ...current.code,
          diff,
          report,
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
      const stage = data.stage;
      if (!isStageName(stage)) {
        return;
      }
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
      const stage = data.stage;
      if (!isStageName(stage)) {
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
      const message = String(data.message ?? "");
      setErrorMessage(message);
      setVerificationPassed(false);
      if (typeof data.history_id === "number") {
        const historyId = Number(data.history_id);
        selectHistory(historyId);
        upsertHistoryItem({
          id: historyId,
          mode: "agent",
          title: `Repair · ${String(data.filename ?? "main.py")}`,
          preview_text: message || "Agent run failed.",
          model,
          language,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        void refreshHistoryList();
      }
      setStatus("error");
      return;
    }

    if (eventName === "result") {
      const resultStatus = String(data.status ?? "");
      const selectionSummary =
        typeof data.selection_summary === "string" && data.selection_summary.trim()
          ? data.selection_summary.trim()
          : "";
      const passed =
        typeof data.verification_passed === "boolean"
          ? Boolean(data.verification_passed)
          : resultStatus === "verified";
      setVerificationPassed(
        resultStatus === "clean" ? null : resultStatus === "verified" ? true : passed ? true : false,
      );
      setFinalMessage(
        [
          resultStatus === "clean"
            ? dict.cleanMessage
            : resultStatus === "verified"
              ? dict.verificationReady
              : resultStatus === "verify_failed"
                ? dict.verificationFailed
                : typeof data.message === "string" && data.message
                  ? data.message
                  : dict.repairedMessage,
          selectionSummary,
        ]
          .filter(Boolean)
          .join("\n"),
      );
      if (typeof data.git_diff === "string") {
        setFinalDiff(data.git_diff);
      }
      if (typeof data.history_id === "number") {
        const historyId = Number(data.history_id);
        selectHistory(historyId);
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
        void refreshHistoryList();
      }
    }
  }

  async function handleSend() {
    if (!languageSupported) {
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
      const payload = buildRepairPayload();
      const response = await fetch(`${apiBaseUrl}/api/repair/stream`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payloadText = await response.text();
        throw new Error(payloadText || `HTTP ${response.status}`);
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
          applyEvent(parsed.event, parsed.data);
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        applyEvent(parsed.event, parsed.data);
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

  function stopStreaming() {
    streamAbortRef.current?.abort();
    setStatus("error");
  }

  function handleApplyDiff() {
    if (agentSourceType !== "single_file") {
      setDiffDecisionMessage(dict.applyProjectOnly);
      return;
    }
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

  function skipApplyingDiff() {
    setDiffDecisionMessage(dict.applySkipped);
  }

  function handleProjectEntrypointChange(value: string) {
    setEntrypointPath(value);
    const matched = projectEntrypointOptions.find((item) => item.path === value);
    if (matched) {
      setLanguage(matched.language);
    }
  }

  function handleProjectSubdirChange(value: string) {
    setProjectSubdir(value);
  }

  function handleGithubRepoUrlChange(value: string) {
    setGithubRepoUrl(value);
  }

  function handleGithubRefChange(value: string) {
    setGithubRef(value);
  }

  function handleSourceTypeChange(value: AgentSourceType) {
    setAgentSourceType(value);
    if (value === "single_file") {
      setProjectEntrypointOptions([]);
      setProjectFilesLoading(false);
      setEntrypointPath(defaultEntrypointForLanguage(language));
    } else {
      setEntrypointPath("");
    }
  }

  function startNewAgentSession() {
    setCode(codeTemplates[language]);
    setInputText("");
    setEntrypointPath(defaultEntrypointForLanguage(language));
    resetRepairState();
  }

  function prepareForChatHistoryView() {
    setVerificationPassed(null);
    setFinalDiff("");
    setFinalMessage("");
    setErrorMessage("");
    setStatus("idle");
    setDiffDecisionMessage("");
    setDiffApplied(false);
  }

  function loadHistorySnapshot(snapshot: AgentHistorySnapshot) {
    setAgentSourceType(snapshot.source_type ?? "single_file");
    setEntrypointPath(snapshot.filename ?? defaultEntrypointForLanguage(snapshot.language ?? "python"));
    setProjectSubdir(snapshot.project_subdir ?? "");
    setGithubRepoUrl(snapshot.github_repo_url ?? "");
    setGithubRef(snapshot.github_ref ?? "");
    setZipFileName("");
    setZipFileBase64("");
    setInputText(snapshot.input_text ?? snapshot.run_result?.input_text ?? "");
    setProjectEntrypointOptions(
      snapshot.source_type && snapshot.source_type !== "single_file" && snapshot.filename
        ? [{ path: snapshot.filename, language: snapshot.language ?? "python" }]
        : [],
    );
    setProjectFilesLoading(false);
    setLanguage(snapshot.language ?? "python");
    setCode(
      snapshot.source_type === "single_file"
        ? snapshot.code ?? codeTemplates[snapshot.language ?? "python"]
        : codeTemplates[snapshot.language ?? "python"],
    );
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
  }

  return {
    agentSourceType,
    language,
    code,
    inputText,
    entrypointPath,
    projectSubdir,
    githubRepoUrl,
    githubRef,
    zipFileName,
    runResult,
    status,
    stages,
    events,
    finalDiff,
    finalMessage,
    verificationPassed,
    errorMessage,
    diffDecisionMessage,
    diffApplied,
    languageSupported,
    projectEntrypointOptions,
    projectFilesLoading,
    setAgentSourceType: handleSourceTypeChange,
    setLanguage,
    setCode,
    setInputText,
    setEntrypointPath: handleProjectEntrypointChange,
    setProjectSubdir: handleProjectSubdirChange,
    setGithubRepoUrl: handleGithubRepoUrlChange,
    setGithubRef: handleGithubRefChange,
    setErrorMessage,
    handleZipSelected,
    handleSend,
    stopStreaming,
    handleApplyDiff,
    skipApplyingDiff,
    resetRepairState,
    startNewAgentSession,
    prepareForChatHistoryView,
    loadHistorySnapshot,
  };
}
