import { useEffect, useState } from "react";
import { GitHubLogo, GoogleLogo } from "./AuthIcons";
import { AuthShowcase } from "./AuthShowcase";
import type { AuthMode, AuthenticatedUser, OAuthProvider, UiLocale } from "../types";

interface AuthPageProps {
  apiBaseUrl: string;
  locale: UiLocale;
  oauthProviders: OAuthProvider[];
  onAuthenticated: (user: AuthenticatedUser, providers: OAuthProvider[]) => void;
  initialError: string;
  initialMessage: string;
  onResetNotice: () => void;
  copy: {
    title: string;
    subtitle: string;
    authEyebrow: string;
    authTitle: string;
    authSubtitle: string;
    loginTab: string;
    registerTab: string;
    nameLabel: string;
    emailLabel: string;
    passwordLabel: string;
    authSubmitLogin: string;
    authSubmitRegister: string;
    authSwitchLogin: string;
    authSwitchRegister: string;
    continueGithub: string;
    continueGoogle: string;
    authOr: string;
    oauthUnavailable: string;
  };
}

interface AuthResponse {
  user: AuthenticatedUser;
  oauth_providers: OAuthProvider[];
}

export function AuthPage({
  apiBaseUrl,
  locale,
  oauthProviders,
  onAuthenticated,
  initialError,
  initialMessage,
  onResetNotice,
  copy,
}: AuthPageProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState(initialError);
  const [successMessage, setSuccessMessage] = useState(initialMessage);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    setErrorMessage(initialError);
    setSuccessMessage(initialMessage);
  }, [initialError, initialMessage]);

  function setNotice(error: string, success = "") {
    setErrorMessage(error);
    setSuccessMessage(success);
  }

  async function handleSubmit() {
    setSubmitting(true);
    onResetNotice();
    setNotice("");
    try {
      const endpoint = mode === "login" ? "/api/auth/login" : "/api/auth/register";
      const payload =
        mode === "login"
          ? { email, password }
          : { display_name: displayName, email, password };
      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(String(data.error ?? "Authentication failed."));
      }
      const authResponse = data as AuthResponse;
      onAuthenticated(authResponse.user, authResponse.oauth_providers);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  function handleOAuth(provider: OAuthProvider) {
    onResetNotice();
    if (!oauthProviders.includes(provider)) {
      setNotice(copy.oauthUnavailable);
      return;
    }
    window.location.href = `${apiBaseUrl}/api/auth/oauth/${provider}`;
  }

  return (
    <div className="mx-auto grid h-[calc(100vh-5.5rem)] max-w-[1400px] items-stretch gap-5 overflow-hidden px-4 py-4 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-8">
      <AuthShowcase locale={locale} />

      <section className="flex h-full flex-col overflow-hidden rounded-[36px] border border-black/5 bg-white/50 p-6 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 sm:p-7">
        <div className="grid grid-cols-2 gap-3 rounded-full border border-black/5 bg-black/[0.03] p-1 dark:border-white/10 dark:bg-white/[0.03]">
          <button
            onClick={() => setMode("login")}
            className={`rounded-full px-4 py-3 text-sm font-medium transition ${
              mode === "login"
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                : "text-slate-600 dark:text-white/65"
            }`}
          >
            {copy.loginTab}
          </button>
          <button
            onClick={() => setMode("register")}
            className={`rounded-full px-4 py-3 text-sm font-medium transition ${
              mode === "register"
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                : "text-slate-600 dark:text-white/65"
            }`}
          >
            {copy.registerTab}
          </button>
        </div>

        <div className="mt-5 space-y-3">
          {mode === "register" ? (
            <label className="block">
              <div className="mb-2 text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/45">
                {copy.nameLabel}
              </div>
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
              />
            </label>
          ) : null}

          <label className="block">
            <div className="mb-2 text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/45">
              {copy.emailLabel}
            </div>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
            />
          </label>

          <label className="block">
            <div className="mb-2 text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/45">
              {copy.passwordLabel}
            </div>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
            />
          </label>
        </div>

        {errorMessage ? (
          <div className="mt-5 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
            {errorMessage}
          </div>
        ) : null}
        {successMessage ? (
          <div className="mt-5 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-200">
            {successMessage}
          </div>
        ) : null}

        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="mt-5 w-full rounded-2xl bg-slate-900 px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-white/85"
        >
          {mode === "login" ? copy.authSubmitLogin : copy.authSubmitRegister}
        </button>

        <div className="mt-5 flex items-center gap-4">
          <div className="h-px flex-1 bg-black/10 dark:bg-white/10" />
          <div className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-white/45">{copy.authOr}</div>
          <div className="h-px flex-1 bg-black/10 dark:bg-white/10" />
        </div>

        <div className="mt-5 grid gap-3">
          <button
            onClick={() => handleOAuth("github")}
            className="flex items-center justify-center gap-3 rounded-2xl border border-black/10 bg-white/50 px-4 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
          >
            <GitHubLogo />
            {copy.continueGithub}
          </button>
          <button
            onClick={() => handleOAuth("google")}
            className="flex items-center justify-center gap-3 rounded-2xl border border-black/10 bg-white/50 px-4 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
          >
            <GoogleLogo />
            {copy.continueGoogle}
          </button>
        </div>

        <div className="mt-5 text-sm text-slate-600 dark:text-white/65">
          {mode === "login" ? copy.authSwitchRegister : copy.authSwitchLogin}{" "}
          <button
            onClick={() => setMode(mode === "login" ? "register" : "login")}
            className="font-semibold text-slate-900 underline underline-offset-4 dark:text-white"
          >
            {mode === "login" ? copy.registerTab : copy.loginTab}
          </button>
        </div>
      </section>
    </div>
  );
}
