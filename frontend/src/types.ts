export type UiLocale = "en" | "zh";
export type ThemeMode = "light" | "dark";
export type SessionStatus = "idle" | "streaming" | "done" | "error";
export type StageName = "run" | "inspect" | "plan" | "code" | "verify";
export type AuthMode = "login" | "register";
export type OAuthProvider = "github" | "google";
export type WorkspaceMode = "agent" | "chat" | "admin" | "billing" | "benchmark" | "profile" | "teams";
export type AgentSourceType = "single_file" | "zip" | "github";
export type UserRole = "basic" | "advanced" | "admin";
export type AccountStatus = "active" | "suspended";
export type AdminPage =
  | "dashboard"
  | "users"
  | "requests"
  | "models"
  | "activity"
  | "payments"
  | "benchmark";
export type BenchmarkPage = "projects" | "runs" | "leaderboard" | "experiments";
export type ProfilePage = "overview" | "wallet" | "preferences" | "api_tokens";
export type TeamsPage = "organizations" | "projects" | "invites";
export type BenchmarkRunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type BenchmarkRunMode = "inspect_only" | "mock_repair" | "full_repair";
export type BenchmarkStrategy = "full_pipeline" | "naive_chat";
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

export interface RepairTestCase {
  stdin: string;
  expected_stdout: string;
  name?: string;
}

export interface TestCaseResult {
  index: number;
  name: string;
  stdin: string;
  expected_stdout: string;
  expected_provided: boolean;
  stdout: string;
  stderr: string;
  returncode: number;
  timed_out: boolean;
  duration_sec: number;
  runtime_ok: boolean;
  matched_output: boolean | null;
  passed: boolean;
}

export interface TestCasesSummary {
  provided: boolean;
  total: number;
  passed: number;
  failed: number;
  all_passed: boolean;
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
  user_prompt?: string;
  test_cases_summary?: TestCasesSummary;
  test_case_results?: TestCaseResult[];
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
  user_prompt?: string;
  test_cases?: RepairTestCase[];
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

// ---------------------------------------------------------------------------
// Credit Wallet
// ---------------------------------------------------------------------------

export interface CreditPricingRule {
  role_code: UserRole;
  monthly_free_credits: number;
  cost_per_chat: number;
  cost_per_repair: number;
  cost_per_benchmark_run: number;
  updated_at: string | null;
}

export interface CreditTransaction {
  id: number;
  user_id: number;
  change_credits: number;
  balance_after: number;
  reason_code: string;
  reference_type: string | null;
  reference_id: number | null;
  note: string | null;
  actor_user_id: number | null;
  created_at: string;
}

export interface CreditWalletSnapshot {
  wallet: {
    user_id: number;
    balance_credits: number;
    lifetime_earned: number;
    lifetime_spent: number;
    last_grant_at: string | null;
    updated_at: string | null;
  };
  transactions: CreditTransaction[];
  pricing: CreditPricingRule | null;
  role: UserRole | string;
}

export interface AdminWalletBalanceItem {
  user_id: number;
  email: string;
  display_name: string;
  role: UserRole;
  balance_credits: number;
  lifetime_earned: number;
  lifetime_spent: number;
  last_grant_at: string | null;
  updated_at: string | null;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

export interface BenchmarkProject {
  id: number;
  project_code: string;
  display_name: string;
  source_type: "defects4j" | "custom";
  language: string;
  description: string | null;
  tags: string[];
  is_active: boolean;
  sort_order: number;
  bug_count: number;
  last_run_at: string | null;
}

export interface BenchmarkBug {
  id: number;
  project_id: number;
  bug_key: string;
  title: string;
  severity: string;
  defects4j_project: string | null;
  defects4j_bug_id: number | null;
  description: string | null;
  tags: string[];
  is_active: boolean;
}

export interface BenchmarkRunSummary {
  id: number;
  user_id: number;
  organization_id: number | null;
  project_id: number;
  project_code: string | null;
  bug_id: number;
  bug_key: string | null;
  defects4j_project: string | null;
  defects4j_bug_id: number | null;
  model_key: string;
  run_mode: BenchmarkRunMode;
  run_status: BenchmarkRunStatus;
  stage: string | null;
  pass_count: number;
  fail_count: number;
  total_tests: number;
  duration_ms: number;
  credits_spent: number;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
  strategy?: BenchmarkStrategy | string | null;
  experiment_id?: number | null;
  is_plausible?: boolean;
  is_correct?: boolean;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  patch_lines_added?: number;
  patch_lines_removed?: number;
  llm_rounds?: number;
  failed_tests_before?: number;
  failed_tests_after?: number;
}

export interface BenchmarkRunDetail extends BenchmarkRunSummary {
  report: Record<string, unknown> | null;
  patch_diff: string;
}

export interface BenchmarkLeaderboardItem {
  project_id: number;
  project_code: string | null;
  project_display_name: string | null;
  model_key: string;
  sample_count: number;
  success_count: number;
  pass_rate: number;
  avg_duration_ms: number;
  last_run_at: string | null;
}

export interface BenchmarkSummary {
  total_projects: number;
  total_bugs: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
}

export interface BenchmarkExperimentArmConfig {
  strategy: BenchmarkStrategy | string;
  model_key: string;
}

export interface BenchmarkExperimentSummary {
  id: number;
  experiment_code: string;
  title: string | null;
  description: string | null;
  hypothesis: string | null;
  status: string;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  config: {
    arms?: BenchmarkExperimentArmConfig[];
    bug_ids?: number[];
    bug_keys?: string[];
  } | Record<string, unknown>;
  created_by_user_id: number | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface BenchmarkExperimentArmResult {
  strategy: string;
  model_key: string;
  total: number;
  completed: number;
  plausible: number;
  correct: number;
  plausible_rate: number;
  correct_rate: number;
  avg_duration_ms: number;
  avg_tokens: number;
}

export interface BenchmarkExperimentPerBugRow {
  bug_key: string;
  strategy: string;
  model_key: string;
  run_status: string;
  is_plausible: boolean;
  is_correct: boolean;
  duration_ms: number;
  total_tokens: number;
  error_message: string | null;
}

export interface BenchmarkExperimentDetail {
  experiment: BenchmarkExperimentSummary;
  arms: BenchmarkExperimentArmResult[];
  per_bug: BenchmarkExperimentPerBugRow[];
}

export interface AdminBenchmarkRun extends BenchmarkRunSummary {
  user_email: string | null;
  user_display_name: string | null;
}

export interface AdminBenchmarkRefreshResult {
  d4j_home: string;
  projects: Array<{ project_code: string; upserted: number; new: number }>;
  total_imported: number;
  total_new: number;
}

// ---------------------------------------------------------------------------
// Teams & Projects
// ---------------------------------------------------------------------------

export interface TeamOrganization {
  id: number;
  name: string;
  slug: string;
  description: string | null;
  owner_user_id: number;
  plan_code: string;
  member_count: number;
  project_count: number;
  created_at: string;
  updated_at: string;
  member_role: "owner" | "admin" | "member" | null;
}

export interface TeamMember {
  id: number;
  organization_id: number;
  user_id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  role: "owner" | "admin" | "member";
  joined_at: string;
}

export interface TeamInvite {
  id: number;
  organization_id: number;
  email: string;
  invite_token: string;
  invite_status: "pending" | "accepted" | "revoked" | "expired";
  invited_by_user_id: number | null;
  expires_at: string;
  accepted_at: string | null;
  created_at: string;
}

export interface TeamProject {
  id: number;
  organization_id: number;
  owner_user_id: number;
  name: string;
  slug: string;
  language: string | null;
  description: string | null;
  repo_url: string | null;
  default_entrypoint: string | null;
  color_hex: string | null;
  history_count: number;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Profile / Preferences / API Tokens
// ---------------------------------------------------------------------------

export interface UserPreferences {
  default_agent_model: string | null;
  default_chat_model: string | null;
  default_language: string | null;
  locale: UiLocale;
  theme: ThemeMode;
  timezone: string | null;
  bio: string | null;
  show_site_map_widget: boolean;
  updated_at: string | null;
}

export interface ProfileOverview {
  total_histories: number;
  total_repair_sessions: number;
  total_chat_sessions: number;
  total_benchmark_runs: number;
  organization_count: number;
  lifetime_tokens: number;
}

export interface UserApiToken {
  id: number;
  user_id: number;
  token_name: string;
  token_prefix: string;
  scope: string;
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  created_at: string;
  revealed_token?: string | null;
}

export interface ProfileSnapshot {
  user: AuthenticatedUser;
  preferences: UserPreferences;
  overview: ProfileOverview;
  wallet: CreditWalletSnapshot["wallet"];
  organizations: TeamOrganization[];
  api_tokens: UserApiToken[];
}

// ---------------------------------------------------------------------------
// Site map widget
// ---------------------------------------------------------------------------

export interface SiteMapItem {
  code: string;
  label: string;
  path: string;
}

export interface SiteMapGroup {
  code: string;
  title: string;
  items: SiteMapItem[];
}

export interface SiteMapResponse {
  groups: SiteMapGroup[];
}
