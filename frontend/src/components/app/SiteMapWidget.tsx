import { useEffect, useState } from "react";
import type { AppCopy } from "../../i18n";
import type { SiteMapGroup, SiteMapResponse } from "../../types";

interface SiteMapWidgetProps {
  apiBaseUrl: string;
  copy: AppCopy;
}

export function SiteMapWidget({ apiBaseUrl, copy }: SiteMapWidgetProps) {
  const [groups, setGroups] = useState<SiteMapGroup[]>([]);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/site-map`);
        if (!response.ok) return;
        const data = (await response.json()) as SiteMapResponse;
        if (active) setGroups(data.groups ?? []);
      } catch {
        // ignore — widget is decorative
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [apiBaseUrl]);

  if (groups.length === 0) return null;

  return (
    <section className="mt-3 rounded-[22px] border border-black/5 bg-white/40 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.04] app-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.siteMapTitle}
          </div>
          <div className="text-[11px] text-slate-500 dark:text-white/45">{copy.siteMapHint}</div>
        </div>
      </div>
      <div className="mt-2 grid gap-2 sm:grid-cols-3">
        {groups.map((group) => (
          <div
            key={group.code}
            className="rounded-2xl border border-black/5 bg-black/[0.02] p-2.5 text-xs dark:border-white/10 dark:bg-white/[0.03]"
          >
            <div className="mb-1.5 text-[10px] uppercase tracking-[0.22em] text-slate-500 dark:text-white/40">
              {group.title}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {group.items.map((item) => (
                <span
                  key={item.code}
                  className="rounded-full bg-black/[0.05] px-2 py-1 text-[11px] font-medium text-slate-700 dark:bg-white/[0.07] dark:text-white/75"
                  title={item.path}
                >
                  {item.label}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
