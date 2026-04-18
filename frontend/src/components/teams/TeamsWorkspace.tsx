import { useCallback, useEffect, useMemo, useState } from "react";
import type { AppCopy } from "../../i18n";
import type {
  TeamInvite,
  TeamMember,
  TeamOrganization,
  TeamProject,
  TeamsPage,
} from "../../types";
import {
  AdminBadge,
  AdminEmptyState,
  AdminSectionTitle,
  AdminSurface,
  formatAdminDateTime,
  toStatusTone,
} from "../admin/AdminCommon";

interface TeamsWorkspaceProps {
  apiBaseUrl: string;
  copy: AppCopy;
  page: TeamsPage;
  onPageChange: (page: TeamsPage) => void;
  workspaceMainClass: string;
}

export function TeamsWorkspace({
  apiBaseUrl,
  copy,
  page,
  onPageChange,
  workspaceMainClass,
}: TeamsWorkspaceProps) {
  const [orgs, setOrgs] = useState<TeamOrganization[]>([]);
  const [activeOrgId, setActiveOrgId] = useState<number | null>(null);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [invites, setInvites] = useState<TeamInvite[]>([]);
  const [projects, setProjects] = useState<TeamProject[]>([]);
  const [error, setError] = useState("");
  const [newOrgName, setNewOrgName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectLanguage, setNewProjectLanguage] = useState("");
  const [acceptToken, setAcceptToken] = useState("");

  const fetchOrgs = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/teams/organizations`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const items: TeamOrganization[] = data.items ?? [];
      setOrgs(items);
      setActiveOrgId((current) => current ?? items[0]?.id ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [apiBaseUrl]);

  const fetchOrgDetails = useCallback(
    async (orgId: number) => {
      try {
        const [membersResp, invitesResp, projectsResp] = await Promise.all([
          fetch(`${apiBaseUrl}/api/teams/organizations/${orgId}/members`, {
            credentials: "include",
          }),
          fetch(`${apiBaseUrl}/api/teams/organizations/${orgId}/invites`, {
            credentials: "include",
          }),
          fetch(`${apiBaseUrl}/api/teams/organizations/${orgId}/projects`, {
            credentials: "include",
          }),
        ]);
        if (membersResp.ok) {
          const data = await membersResp.json();
          setMembers((data.items ?? []) as TeamMember[]);
        }
        if (invitesResp.ok) {
          const data = await invitesResp.json();
          setInvites((data.items ?? []) as TeamInvite[]);
        }
        if (projectsResp.ok) {
          const data = await projectsResp.json();
          setProjects((data.items ?? []) as TeamProject[]);
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    },
    [apiBaseUrl],
  );

  useEffect(() => {
    void fetchOrgs();
  }, [fetchOrgs]);

  useEffect(() => {
    if (activeOrgId) void fetchOrgDetails(activeOrgId);
  }, [activeOrgId, fetchOrgDetails]);

  async function handleCreateOrg() {
    if (!newOrgName.trim()) return;
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/teams/organizations`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newOrgName.trim() }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      setNewOrgName("");
      await fetchOrgs();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleInvite() {
    if (!activeOrgId || !inviteEmail.trim()) return;
    try {
      const response = await fetch(
        `${apiBaseUrl}/api/teams/organizations/${activeOrgId}/invites`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: inviteEmail.trim() }),
        },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      setInviteEmail("");
      await fetchOrgDetails(activeOrgId);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleAcceptInvite() {
    if (!acceptToken.trim()) return;
    try {
      const response = await fetch(
        `${apiBaseUrl}/api/teams/invites/${acceptToken.trim()}/accept`,
        {
          method: "POST",
          credentials: "include",
        },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      setAcceptToken("");
      await fetchOrgs();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleCreateProject() {
    if (!activeOrgId || !newProjectName.trim()) return;
    try {
      const response = await fetch(
        `${apiBaseUrl}/api/teams/organizations/${activeOrgId}/projects`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: newProjectName.trim(),
            language: newProjectLanguage.trim() || null,
          }),
        },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      setNewProjectName("");
      setNewProjectLanguage("");
      await fetchOrgDetails(activeOrgId);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleDeleteProject(projectId: number) {
    const response = await fetch(`${apiBaseUrl}/api/teams/projects/${projectId}`, {
      method: "DELETE",
      credentials: "include",
    });
    if (!response.ok || !activeOrgId) return;
    await fetchOrgDetails(activeOrgId);
  }

  async function handleRemoveMember(userId: number) {
    if (!activeOrgId) return;
    await fetch(`${apiBaseUrl}/api/teams/organizations/${activeOrgId}/members/${userId}`, {
      method: "DELETE",
      credentials: "include",
    });
    await fetchOrgDetails(activeOrgId);
  }

  const activeOrg = useMemo(
    () => orgs.find((org) => org.id === activeOrgId) ?? null,
    [orgs, activeOrgId],
  );

  return (
    <section
      className={`flex min-h-0 min-w-0 h-full flex-col overflow-hidden rounded-[24px] border border-black/5 bg-white/50 p-3 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 ${workspaceMainClass}`}
    >
      <div className="shrink-0 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            {copy.teamsTitle}
          </div>
          <div className="hidden text-xs text-slate-500 dark:text-white/45 sm:block">
            {copy.teamsHint}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {(["organizations", "projects", "invites"] as TeamsPage[]).map((tab) => (
            <button
              key={tab}
              onClick={() => onPageChange(tab)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                page === tab
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950"
                  : "bg-black/[0.04] text-slate-600 hover:bg-black/[0.07] dark:bg-white/[0.05] dark:text-white/70 dark:hover:bg-white/[0.09]"
              }`}
            >
              {tab === "organizations"
                ? copy.teamsOrganizations
                : tab === "projects"
                  ? copy.teamsProjects
                  : copy.teamsInvites}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <div className="mt-2 rounded-2xl border border-rose-500/20 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:border-rose-400/20 dark:bg-rose-500/10 dark:text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto app-fade-in">
        <div className="grid gap-2 lg:grid-cols-[0.8fr,1.2fr]">
          <AdminSurface>
            <AdminSectionTitle title={copy.teamsOrganizations} />
            <div className="mt-2 flex gap-2">
              <input
                value={newOrgName}
                onChange={(event) => setNewOrgName(event.target.value)}
                placeholder={copy.teamsNameLabel}
                className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
              />
              <button
                onClick={handleCreateOrg}
                className="rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-slate-800 dark:bg-white dark:text-slate-950"
              >
                {copy.teamsCreateOrg}
              </button>
            </div>
            {orgs.length === 0 ? (
              <div className="mt-2">
                <AdminEmptyState message={copy.teamsOrgEmpty} />
              </div>
            ) : (
              <div className="mt-2 space-y-1.5">
                {orgs.map((org) => (
                  <button
                    key={org.id}
                    onClick={() => setActiveOrgId(org.id)}
                    className={`flex w-full items-center justify-between gap-2 rounded-2xl border px-3 py-2 text-left text-xs transition ${
                      activeOrgId === org.id
                        ? "border-slate-900 bg-slate-900 text-white dark:border-white dark:bg-white dark:text-slate-950"
                        : "border-black/5 bg-black/[0.02] hover:bg-black/[0.05] dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="truncate font-semibold">{org.name}</div>
                      <div
                        className={`truncate text-[11px] ${
                          activeOrgId === org.id
                            ? "text-white/70 dark:text-slate-950/70"
                            : "text-slate-500 dark:text-white/45"
                        }`}
                      >
                        {org.member_count} · {org.project_count}
                      </div>
                    </div>
                    <AdminBadge
                      label={org.member_role ?? "—"}
                      tone={toStatusTone(org.member_role ?? "")}
                    />
                  </button>
                ))}
              </div>
            )}
          </AdminSurface>

          {page === "organizations" ? (
            <AdminSurface>
              <AdminSectionTitle
                title={activeOrg?.name ?? copy.teamsOrganizations}
                hint={activeOrg?.description ?? undefined}
              />
              <div className="mt-3 space-y-2">
                <div>
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.teamsMembers}
                  </div>
                  {members.length === 0 ? (
                    <div className="mt-1.5">
                      <AdminEmptyState message="—" />
                    </div>
                  ) : (
                    <div className="mt-1.5 space-y-1">
                      {members.map((member) => (
                        <div
                          key={member.id}
                          className="flex items-center justify-between rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-1.5 text-xs dark:border-white/10 dark:bg-white/[0.03]"
                        >
                          <div>
                            <span className="font-medium">{member.display_name}</span>
                            <span className="ml-2 text-slate-500 dark:text-white/45">
                              {member.email}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <AdminBadge label={member.role} tone={toStatusTone(member.role)} />
                            {member.role !== "owner" ? (
                              <button
                                onClick={() => handleRemoveMember(member.user_id)}
                                className="rounded-full bg-rose-500/10 px-2 py-0.5 text-[11px] text-rose-600 hover:bg-rose-500/15 dark:text-rose-300"
                              >
                                {copy.teamsRemoveMember}
                              </button>
                            ) : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-white/40">
                    {copy.teamsInviteMember}
                  </div>
                  <div className="mt-1.5 flex gap-2">
                    <input
                      value={inviteEmail}
                      onChange={(event) => setInviteEmail(event.target.value)}
                      placeholder={copy.teamsEmailLabel}
                      className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                    />
                    <button
                      onClick={handleInvite}
                      className="rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950"
                    >
                      {copy.teamsInviteMember}
                    </button>
                  </div>
                </div>
              </div>
            </AdminSurface>
          ) : null}

          {page === "projects" ? (
            <AdminSurface>
              <AdminSectionTitle title={copy.teamsProjects} />
              <div className="mt-2 flex flex-wrap gap-2">
                <input
                  value={newProjectName}
                  onChange={(event) => setNewProjectName(event.target.value)}
                  placeholder={copy.teamsNameLabel}
                  className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
                <input
                  value={newProjectLanguage}
                  onChange={(event) => setNewProjectLanguage(event.target.value)}
                  placeholder={copy.teamsLanguageLabel}
                  className="w-28 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
                <button
                  onClick={handleCreateProject}
                  className="rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950"
                >
                  {copy.teamsCreateProject}
                </button>
              </div>
              {projects.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.teamsProjectsEmpty} />
                </div>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {projects.map((project) => (
                    <div
                      key={project.id}
                      className="flex items-center justify-between rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 text-xs dark:border-white/10 dark:bg-white/[0.03]"
                    >
                      <div>
                        <div className="flex items-center gap-2 font-semibold">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: project.color_hex ?? "#64748B" }}
                          />
                          {project.name}
                        </div>
                        <div className="mt-0.5 text-[11px] text-slate-500 dark:text-white/45">
                          {project.language ?? "—"} · {formatAdminDateTime(project.created_at)}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteProject(project.id)}
                        className="rounded-full bg-rose-500/10 px-3 py-1 text-[11px] text-rose-600 hover:bg-rose-500/15 dark:text-rose-300"
                      >
                        {copy.historyDelete}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </AdminSurface>
          ) : null}

          {page === "invites" ? (
            <AdminSurface>
              <AdminSectionTitle title={copy.teamsInvites} />
              <div className="mt-2 flex gap-2">
                <input
                  value={acceptToken}
                  onChange={(event) => setAcceptToken(event.target.value)}
                  placeholder={copy.teamsTokenLabel}
                  className="flex-1 rounded-xl border border-black/10 bg-white px-3 py-1.5 text-sm dark:border-white/10 dark:bg-slate-900"
                />
                <button
                  onClick={handleAcceptInvite}
                  className="rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-white dark:text-slate-950"
                >
                  {copy.teamsAcceptInvite}
                </button>
              </div>
              {invites.length === 0 ? (
                <div className="mt-2">
                  <AdminEmptyState message={copy.teamsInvitesEmpty} />
                </div>
              ) : (
                <div className="mt-2 space-y-1.5">
                  {invites.map((invite) => (
                    <div
                      key={invite.id}
                      className="rounded-2xl border border-black/5 bg-black/[0.02] px-3 py-2 text-xs dark:border-white/10 dark:bg-white/[0.03]"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{invite.email}</span>
                        <AdminBadge
                          label={invite.invite_status}
                          tone={toStatusTone(invite.invite_status)}
                        />
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] text-slate-500 dark:text-white/45">
                        {invite.invite_token} · exp {formatAdminDateTime(invite.expires_at)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </AdminSurface>
          ) : null}
        </div>
      </div>
    </section>
  );
}
