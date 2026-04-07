import type { RefObject } from "react";
import type { AppCopy } from "../../i18n";
import { getUserInitials } from "../../app/utils";
import type { AuthenticatedUser, ThemeMode } from "../../types";
import { LanguageIcon, MoonIcon, SunIcon } from "./AppIcons";

interface AppHeaderProps {
  currentUser: AuthenticatedUser | null;
  copy: Pick<AppCopy, "locale" | "logout" | "themeDark" | "themeLight" | "title">;
  theme: ThemeMode;
  userMenuOpen: boolean;
  userMenuRef: RefObject<HTMLDivElement>;
  onLogout: () => void;
  onToggleLocale: () => void;
  onToggleTheme: () => void;
  onToggleUserMenu: () => void;
}

export function AppHeader({
  currentUser,
  copy,
  theme,
  userMenuOpen,
  userMenuRef,
  onLogout,
  onToggleLocale,
  onToggleTheme,
  onToggleUserMenu,
}: AppHeaderProps) {
  return (
    <div className="relative z-40 flex shrink-0 items-center justify-between gap-3">
      {currentUser ? (
        <div className="min-w-0">
          <div className="font-display text-xl tracking-tight text-slate-950 dark:text-white sm:text-2xl">
            {copy.title}
          </div>
        </div>
      ) : (
        <div />
      )}

      <div className="flex justify-end gap-3">
        <button
          onClick={onToggleLocale}
          aria-label={copy.locale}
          title={copy.locale}
          className="flex h-11 min-w-11 items-center justify-center rounded-full border border-black/10 bg-white/65 px-3 text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
        >
          <LanguageIcon />
        </button>
        <button
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? copy.themeLight : copy.themeDark}
          title={theme === "dark" ? copy.themeLight : copy.themeDark}
          className="flex h-11 min-w-11 items-center justify-center rounded-full border border-black/10 bg-white/65 px-3 text-slate-700 transition hover:border-slate-400 dark:border-white/10 dark:bg-white/5 dark:text-white/75"
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>

        {currentUser ? (
          <div ref={userMenuRef} className="relative z-40 flex items-center">
            <button
              onClick={onToggleUserMenu}
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
                  {getUserInitials(currentUser)}
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
                  <div className="font-medium text-slate-900 dark:text-white">
                    {currentUser.display_name}
                  </div>
                  <div className="mt-1 text-xs text-slate-500 dark:text-white/45">
                    {currentUser.email}
                  </div>
                </div>
                <button
                  onClick={onLogout}
                  className="flex w-full items-center justify-between rounded-[18px] px-3 py-3 text-left text-sm font-medium text-slate-700 transition hover:bg-black/[0.04] dark:text-white/80 dark:hover:bg-white/[0.05]"
                >
                  <span>{copy.logout}</span>
                  <span className="text-slate-400 dark:text-white/35">↗</span>
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
