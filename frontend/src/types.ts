export type UiLocale = "en" | "zh";
export type ThemeMode = "light" | "dark";
export type SessionStatus = "idle" | "streaming" | "done" | "error";
export type StageName = "run" | "inspect" | "plan" | "code";
export type AuthMode = "login" | "register";
export type OAuthProvider = "github" | "google";
export type WorkspaceMode = "agent" | "chat";

export type CodeLanguage = "python" | "javascript" | "typescript" | "java" | "go";
export type ModelOptionValue = string;

export interface LanguageOption {
  value: CodeLanguage;
  label: string;
  extension: string;
  supported: boolean;
}

export interface ModelOption {
  value: ModelOptionValue;
  label: string;
}

export interface StageState {
  status: "idle" | "started" | "explaining" | "completed";
  explain: string;
  report: string;
  diff: string;
}

export interface EventEntry {
  id: string;
  event: string;
  stage?: StageName;
  summary: string;
  at: string;
}

export interface RunResult {
  stdout: string;
  stderr: string;
  execution?: {
    returncode: number;
    duration_sec: number;
    timed_out: boolean;
  };
}

export interface AuthenticatedUser {
  id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  auth_source: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  at: string;
}

export interface HistorySummary {
  id: number;
  mode: WorkspaceMode;
  title: string;
  preview_text: string;
  model: string | null;
  language: string | null;
  created_at: string;
  updated_at: string;
}

export interface AgentHistorySnapshot {
  code: string;
  filename: string;
  language: CodeLanguage;
  model: string;
  run_result: RunResult | null;
  stages: Record<StageName, StageState>;
  events: EventEntry[];
  final_diff: string;
  final_status: string;
  error_message: string;
}

export interface ChatHistorySnapshot {
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    at: string;
  }>;
}

export interface HistoryDetail extends HistorySummary {
  snapshot: AgentHistorySnapshot | ChatHistorySnapshot;
}
