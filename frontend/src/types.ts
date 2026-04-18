export type UiLocale = "en" | "zh";
export type ThemeMode = "light" | "dark";
export type SessionStatus = "idle" | "streaming" | "done" | "error";
export type StageName = "run" | "inspect" | "plan" | "code" | "verify";
export type AuthMode = "login" | "register";
export type OAuthProvider = "github" | "google";
export type WorkspaceMode = "agent" | "chat" | "admin" | "billing";
export type AgentSourceType = "single_file" | "zip" | "github";
export type UserRole = "basic" | "advanced" | "admin";
export type AccountStatus = "active" | "suspended";
export type AdminPage = "dashboard" | "users" | "requests" | "models" | "activity" | "payments";
export type PaymentMethodCode = "card" | "paypal" | "wechat" | "alipay";
export type PaymentOrderStatus = "pending" | "paid" | "rejected" | "cancelled" | "failed";

export type CodeLanguage = "python" | "javascript" | "typescript" | "java" | "go" | "c" | "cpp";
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

export interface ModelCatalogItem extends ModelOption {
  provider_code: string;
  provider_name: string;
  vendor_name: string;
  supports_streaming: boolean;
  supports_json: boolean;
  is_default_chat: boolean;
  is_default_repair: boolean;
}

export interface PublicModelCatalog {
  items: ModelCatalogItem[];
  default_chat_model: string | null;
  default_repair_model: string | null;
}

export interface ToolEventEntry {
  id: string;
  tool_name: string;
  status: "started" | "completed";
  round?: number;
  arguments?: string;
  output_preview?: string;
  output_truncated?: boolean;
  at: string;
}

export interface StageState {
  status: "idle" | "started" | "explaining" | "completed";
  reasoning: string;
  explain: string;
  report: string;
  diff: string;
  toolEvents: ToolEventEntry[];
  retryAttempt?: number;
  retryMax?: number;
}

export interface EventEntry {
  id: string;
  event: string;
  stage?: StageName;
  summary: string;
  at: string;
}

export interface RunResult {
  input_text?: string;
  stdout: string;
  stderr: string;
  entrypoint?: string;
  source_type?: AgentSourceType;
  file_count?: number;
  execution?: {
    returncode: number;
    duration_sec: number;
    timed_out: boolean;
  };
}

export interface ProjectEntrypointOption {
  path: string;
  language: CodeLanguage;
}

export interface AuthenticatedUser {
  id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  auth_source: string;
  role: UserRole;
  account_status: AccountStatus;
  created_at: string;
  updated_at: string | null;
  last_login_at: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
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
  input_text?: string;
  model: string;
  source_type?: AgentSourceType;
  github_repo_url?: string;
  github_ref?: string;
  project_subdir?: string;
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
    reasoning?: string;
    at: string;
  }>;
}

export interface HistoryDetail extends HistorySummary {
  snapshot: AgentHistorySnapshot | ChatHistorySnapshot;
}

export interface AdminDashboardSummary {
  total_users: number;
  admin_users: number;
  advanced_users: number;
  new_users_7d: number;
  llm_requests_7d: number;
  chat_requests_7d: number;
  repair_requests_7d: number;
  failed_requests_7d: number;
  total_tokens_7d: number;
  paid_orders_30d: number;
  paid_amount_cents_30d: number;
}

export interface AdminDailyTokenUsage {
  day: string;
  request_count: number;
  total_tokens: number;
}

export interface AdminDailyUserGrowth {
  day: string;
  new_users: number;
  cumulative_users: number;
}

export interface AdminModelUsageItem {
  model: string;
  provider: string;
  request_count: number;
  total_tokens: number;
  input_tokens?: number;
  output_tokens?: number;
  avg_latency_ms: number;
  last_used_at: string | null;
}

export interface AdminLatestRequestItem {
  id: number;
  request_mode: string;
  stage: string | null;
  purpose: string | null;
  provider: string;
  model: string;
  request_status: string;
  started_at: string;
  user_id: number | null;
  user_email: string | null;
  user_display_name: string | null;
  total_tokens: number;
}

export interface AdminDailyPaymentVolume {
  day: string;
  paid_orders: number;
  paid_amount_cents: number;
}

export interface AdminPaymentMethodUsageItem {
  payment_method: PaymentMethodCode;
  paid_orders: number;
  paid_amount_cents: number;
}

export interface AdminDashboardData {
  summary: AdminDashboardSummary;
  daily_token_usage: AdminDailyTokenUsage[];
  daily_user_growth: AdminDailyUserGrowth[];
  daily_payment_volume: AdminDailyPaymentVolume[];
  model_usage: AdminModelUsageItem[];
  payment_method_usage: AdminPaymentMethodUsageItem[];
  latest_requests: AdminLatestRequestItem[];
}

export interface AdminUserItem {
  id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  auth_source: string;
  role: UserRole;
  account_status: AccountStatus;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
  history_count: number;
  llm_request_count: number;
  total_tokens: number;
  payment_order_count: number;
  active_subscription_plan: string | null;
}

export interface AdminLlmRequestItem {
  id: number;
  user_id: number | null;
  user_email: string | null;
  user_display_name: string | null;
  history_id: number | null;
  request_mode: string;
  stage: string | null;
  purpose: string | null;
  provider: string;
  model: string;
  source_type: string | null;
  is_streaming: boolean;
  is_json_response: boolean;
  request_status: string;
  token_source: string | null;
  prompt_chars: number;
  response_chars: number;
  latency_ms: number;
  error_message: string | null;
  started_at: string;
  finished_at: string | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface AdminLlmRequestList {
  items: AdminLlmRequestItem[];
  page: number;
  page_size: number;
  total: number;
}

export interface AdminLlmToolEvent {
  id: number;
  round_index: number | null;
  status: string;
  tool_name: string;
  arguments_json: string;
  output_preview: string;
  output_truncated: boolean;
  created_at: string;
}

export interface AdminLlmRequestDetail {
  request: AdminLlmRequestItem & {
    cached_input_tokens: number;
    reasoning_tokens: number;
  };
  message: {
    system_prompt: string;
    prompt_text: string;
    response_text: string;
    parsed_response_json: string;
  };
  tool_events: AdminLlmToolEvent[];
}

export interface AdminModelUsageReport {
  days: number;
  items: AdminModelUsageItem[];
  daily_series: Array<{
    day: string;
    model: string;
    request_count: number;
    total_tokens: number;
  }>;
}

export interface AdminModelConfigItem {
  id: number;
  provider_code: "openai_compatible" | "gemini";
  provider_name: string;
  vendor_name: string;
  model_key: string;
  display_name: string;
  api_model_name: string;
  api_base_url: string | null;
  api_key_env_var: string | null;
  enabled: boolean;
  is_default_chat: boolean;
  is_default_repair: boolean;
  supports_streaming: boolean;
  supports_json: boolean;
  thinking_enabled: boolean;
  sort_order: number;
  notes: string | null;
  extra_config: Record<string, unknown>;
  api_key_configured: boolean;
  missing_configuration: string[];
  request_count_30d: number;
  total_tokens_30d: number;
  last_used_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AdminModelConfigPayload {
  provider_code: "openai_compatible" | "gemini";
  provider_name: string;
  vendor_name: string;
  model_key: string;
  display_name: string;
  api_model_name: string;
  api_base_url: string;
  api_key_env_var: string;
  enabled: boolean;
  is_default_chat: boolean;
  is_default_repair: boolean;
  supports_streaming: boolean;
  supports_json: boolean;
  thinking_enabled: boolean;
  sort_order: number;
  notes: string;
  extra_config: Record<string, unknown>;
}

export interface AdminLoginEventItem {
  id: number;
  user_id: number | null;
  user_email: string | null;
  user_display_name: string | null;
  email_attempt: string | null;
  login_method: string;
  login_status: string;
  failure_reason: string | null;
  ip_address: string | null;
  user_agent: string | null;
  created_at: string;
}

export interface AdminLoginEventList {
  items: AdminLoginEventItem[];
  page: number;
  page_size: number;
  total: number;
}

export interface BillingPlan {
  id: number;
  plan_code: string;
  plan_name: string;
  role_granted: UserRole;
  billing_cycle: string;
  amount_cents: number;
  currency: string;
  description: string;
  is_active: boolean;
  sort_order: number;
}

export interface BillingPaymentMethod {
  code: PaymentMethodCode;
  provider_code: string;
  provider_name: string;
  display_mode: "card_form" | "paypal_buttons" | "qr_code";
  integration_status: "ready" | "missing_config";
  missing_config: string[];
  script_url: string | null;
  public_config: Record<string, unknown>;
  is_configured: boolean;
}

export interface BillingSubscription {
  id: number;
  user_id: number;
  plan_id: number;
  plan_code: string;
  plan_name: string;
  role_granted: UserRole;
  subscription_status: string;
  started_at: string | null;
  ends_at: string | null;
  revoked_at: string | null;
  activated_by_order_id: number | null;
}

export interface BillingOrderItem {
  id: number;
  order_no: string;
  user_id: number;
  plan_code: string;
  plan_name: string;
  target_role: UserRole;
  amount_cents: number;
  currency: string;
  payment_method: PaymentMethodCode;
  order_status: PaymentOrderStatus;
  provider_status: string | null;
  checkout_action: string;
  checkout_url: string | null;
  provider_reference: string | null;
  session_status: string | null;
  redirect_url: string | null;
  qr_code_text: string | null;
  qr_code_url: string | null;
  provider_code: string;
  provider_name: string;
  display_mode: string;
  integration_status: string | null;
  missing_config: string[];
  script_url: string | null;
  public_config: Record<string, unknown>;
  notify_url: string | null;
  return_url: string | null;
  next_action_path: string | null;
  instructions: string;
  created_at: string;
  updated_at: string;
  paid_at: string | null;
}

export interface BillingOrderSummary {
  total_orders: number;
  paid_orders: number;
  paid_amount_cents: number;
}

export interface BillingSummaryData {
  payment_environment: "prepare" | "live";
  plans: BillingPlan[];
  payment_methods: BillingPaymentMethod[];
  orders: BillingOrderItem[];
  order_summary: BillingOrderSummary;
  current_subscription: BillingSubscription | null;
  user: AuthenticatedUser;
}

export interface BillingOrderSession {
  order: BillingOrderItem;
  session: {
    provider_code: string;
    provider_name: string;
    display_mode: string;
    integration_status: string;
    missing_config: string[];
    script_url: string | null;
    public_config: Record<string, unknown>;
    notify_url: string | null;
    return_url: string | null;
    next_action_path: string | null;
    instructions: string;
    qr_code_url: string | null;
    qr_code_text: string | null;
  };
}

export interface AdminPaymentOrderItem extends BillingOrderItem {
  user_email: string | null;
  user_display_name: string | null;
}

export interface AdminPaymentOrderSummary {
  total_orders: number;
  pending_orders: number;
  paid_orders: number;
  paid_amount_cents: number;
}

export interface AdminPaymentOrderList {
  items: AdminPaymentOrderItem[];
  page: number;
  page_size: number;
  total: number;
  summary: AdminPaymentOrderSummary;
}
