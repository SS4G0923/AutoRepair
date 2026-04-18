from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.store import get_db_connection


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    lowered = name.strip().lower()
    slug = _SLUG_RE.sub("-", lowered).strip("-")
    return slug[:60] or f"org-{secrets.token_hex(4)}"


def _serialize_org(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "slug": row["slug"],
        "description": row.get("description"),
        "owner_user_id": int(row["owner_user_id"]),
        "plan_code": row["plan_code"],
        "member_count": int(row.get("member_count") or 0),
        "project_count": int(row.get("project_count") or 0),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "member_role": row.get("member_role"),
    }


def _serialize_member(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "organization_id": int(row["organization_id"]),
        "user_id": int(row["user_id"]),
        "email": row.get("email"),
        "display_name": row.get("display_name"),
        "avatar_url": row.get("avatar_url"),
        "role": row.get("member_role"),
        "global_role": row.get("user_role"),
        "joined_at": str(row["joined_at"]),
    }


def _serialize_project(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "organization_id": int(row["organization_id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "name": row["name"],
        "slug": row["slug"],
        "language": row.get("language"),
        "description": row.get("description"),
        "repo_url": row.get("repo_url"),
        "default_entrypoint": row.get("default_entrypoint"),
        "color_hex": row.get("color_hex"),
        "history_count": int(row.get("history_count") or 0),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _serialize_invite(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "organization_id": int(row["organization_id"]),
        "email": row["email"],
        "invite_token": row["invite_token"],
        "invite_status": row["invite_status"],
        "invited_by_user_id": int(row["invited_by_user_id"]) if row.get("invited_by_user_id") is not None else None,
        "expires_at": str(row["expires_at"]),
        "accepted_at": str(row["accepted_at"]) if row.get("accepted_at") else None,
        "created_at": str(row["created_at"]),
    }


def _unique_slug(cursor, base: str) -> str:
    candidate = base
    suffix = 1
    while True:
        cursor.execute("SELECT id FROM organizations WHERE slug = %s", (candidate,))
        if cursor.fetchone() is None:
            return candidate
        suffix += 1
        candidate = f"{base}-{suffix}"


def list_organizations_for_user(user_id: int) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT o.id, o.name, o.slug, o.description, o.owner_user_id, o.plan_code,
                       o.created_at, o.updated_at, m.member_role,
                       (SELECT COUNT(*) FROM organization_members mm WHERE mm.organization_id = o.id) AS member_count,
                       (SELECT COUNT(*) FROM projects pp WHERE pp.organization_id = o.id AND pp.deleted_at IS NULL) AS project_count
                FROM organizations o
                INNER JOIN organization_members m ON m.organization_id = o.id AND m.user_id = %s
                WHERE o.deleted_at IS NULL
                ORDER BY o.id DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall() or []
    return [_serialize_org(row) for row in rows]


def create_organization(
    *,
    owner_user_id: int,
    name: str,
    description: str | None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Organization name is required.")
    if len(clean_name) > 120:
        raise ValueError("Organization name must be at most 120 characters.")
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            slug = _unique_slug(cursor, _slugify(clean_name))
            cursor.execute(
                """
                INSERT INTO organizations (name, slug, description, owner_user_id, plan_code)
                VALUES (%s, %s, %s, %s, 'free_team')
                """,
                (clean_name, slug, description, owner_user_id),
            )
            org_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO organization_members (organization_id, user_id, member_role)
                VALUES (%s, %s, 'owner')
                """,
                (org_id, owner_user_id),
            )
        connection.commit()
    return get_organization_for_user(org_id, owner_user_id) or {}


def get_organization_for_user(org_id: int, user_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT o.id, o.name, o.slug, o.description, o.owner_user_id, o.plan_code,
                       o.created_at, o.updated_at, m.member_role,
                       (SELECT COUNT(*) FROM organization_members mm WHERE mm.organization_id = o.id) AS member_count,
                       (SELECT COUNT(*) FROM projects pp WHERE pp.organization_id = o.id AND pp.deleted_at IS NULL) AS project_count
                FROM organizations o
                INNER JOIN organization_members m ON m.organization_id = o.id AND m.user_id = %s
                WHERE o.id = %s AND o.deleted_at IS NULL
                """,
                (user_id, org_id),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return _serialize_org(row)


def list_members(org_id: int) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT m.id, m.organization_id, m.user_id, m.member_role, m.joined_at,
                       u.email, u.display_name, u.avatar_url, u.role AS user_role
                FROM organization_members m
                INNER JOIN users u ON u.id = m.user_id
                WHERE m.organization_id = %s
                ORDER BY (m.member_role = 'owner') DESC, (m.member_role = 'admin') DESC, m.joined_at ASC
                """,
                (org_id,),
            )
            rows = cursor.fetchall() or []
    return [_serialize_member(row) for row in rows]


def invite_member(
    *,
    org_id: int,
    invited_by_user_id: int,
    email: str,
) -> dict[str, Any]:
    clean_email = email.strip().lower()
    if "@" not in clean_email or len(clean_email) < 5:
        raise ValueError("A valid email is required.")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO organization_invites
                    (organization_id, email, invite_token, invited_by_user_id, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (org_id, clean_email, token, invited_by_user_id, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
            invite_id = int(cursor.lastrowid)
        connection.commit()
    return list_invites_for(org_id, invite_id=invite_id)[0]


def list_invites_for(org_id: int, *, invite_id: int | None = None) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            if invite_id is not None:
                cursor.execute(
                    """
                    SELECT id, organization_id, email, invite_token, invite_status,
                           invited_by_user_id, expires_at, accepted_at, created_at
                    FROM organization_invites WHERE id = %s
                    """,
                    (invite_id,),
                )
                rows = cursor.fetchall() or []
            else:
                cursor.execute(
                    """
                    SELECT id, organization_id, email, invite_token, invite_status,
                           invited_by_user_id, expires_at, accepted_at, created_at
                    FROM organization_invites
                    WHERE organization_id = %s
                    ORDER BY id DESC
                    """,
                    (org_id,),
                )
                rows = cursor.fetchall() or []
    return [_serialize_invite(row) for row in rows]


def accept_invite(*, invite_token: str, user_id: int) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, organization_id, invite_status, expires_at
                FROM organization_invites
                WHERE invite_token = %s
                """,
                (invite_token,),
            )
            invite = cursor.fetchone()
            if invite is None:
                raise ValueError("Invite token was not found.")
            if invite["invite_status"] != "pending":
                raise ValueError("Invite has already been handled.")
            cursor.execute("SELECT NOW() AS now_time")
            now_value = (cursor.fetchone() or {}).get("now_time")
            if now_value and str(invite["expires_at"]) < str(now_value):
                raise ValueError("Invite has expired.")

            cursor.execute(
                """
                INSERT IGNORE INTO organization_members (organization_id, user_id, member_role)
                VALUES (%s, %s, 'member')
                """,
                (int(invite["organization_id"]), user_id),
            )
            cursor.execute(
                "UPDATE organization_invites SET invite_status='accepted', accepted_at = NOW() WHERE id = %s",
                (int(invite["id"]),),
            )
        connection.commit()
    return {"organization_id": int(invite["organization_id"])}


def remove_member(*, org_id: int, user_id: int) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT member_role FROM organization_members WHERE organization_id = %s AND user_id = %s",
                (org_id, user_id),
            )
            row = cursor.fetchone()
            if row is None:
                return
            if row["member_role"] == "owner":
                raise ValueError("Organization owner cannot be removed.")
            cursor.execute(
                "DELETE FROM organization_members WHERE organization_id = %s AND user_id = %s",
                (org_id, user_id),
            )
        connection.commit()


def create_project(
    *,
    org_id: int,
    owner_user_id: int,
    name: str,
    language: str | None,
    description: str | None,
    repo_url: str | None,
    color_hex: str | None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Project name is required.")
    base_slug = _slugify(clean_name)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            slug = base_slug
            suffix = 1
            while True:
                cursor.execute(
                    "SELECT id FROM projects WHERE organization_id = %s AND slug = %s",
                    (org_id, slug),
                )
                if cursor.fetchone() is None:
                    break
                suffix += 1
                slug = f"{base_slug}-{suffix}"
            cursor.execute(
                """
                INSERT INTO projects
                    (organization_id, owner_user_id, name, slug, language, description,
                     repo_url, color_hex)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (org_id, owner_user_id, clean_name, slug, language, description, repo_url, color_hex),
            )
            project_id = int(cursor.lastrowid)
        connection.commit()
    return get_project(project_id) or {}


def get_project(project_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.organization_id, p.owner_user_id, p.name, p.slug,
                       p.language, p.description, p.repo_url, p.default_entrypoint,
                       p.color_hex, p.created_at, p.updated_at,
                       (SELECT COUNT(*) FROM conversation_histories ch
                        WHERE ch.project_id = p.id AND ch.deleted_at IS NULL) AS history_count
                FROM projects p
                WHERE p.id = %s AND p.deleted_at IS NULL
                """,
                (project_id,),
            )
            row = cursor.fetchone()
    return _serialize_project(row) if row else None


def list_projects(org_id: int) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.organization_id, p.owner_user_id, p.name, p.slug,
                       p.language, p.description, p.repo_url, p.default_entrypoint,
                       p.color_hex, p.created_at, p.updated_at,
                       (SELECT COUNT(*) FROM conversation_histories ch
                        WHERE ch.project_id = p.id AND ch.deleted_at IS NULL) AS history_count
                FROM projects p
                WHERE p.organization_id = %s AND p.deleted_at IS NULL
                ORDER BY p.updated_at DESC, p.id DESC
                """,
                (org_id,),
            )
            rows = cursor.fetchall() or []
    return [_serialize_project(row) for row in rows]


def delete_project(project_id: int) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE projects SET deleted_at = NOW() WHERE id = %s",
                (project_id,),
            )
        connection.commit()


def user_is_member(*, org_id: int, user_id: int) -> bool:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM organization_members WHERE organization_id = %s AND user_id = %s",
                (org_id, user_id),
            )
            return cursor.fetchone() is not None


def list_all_organizations_for_admin() -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT o.id, o.name, o.slug, o.description, o.owner_user_id, o.plan_code,
                       o.created_at, o.updated_at,
                       (SELECT COUNT(*) FROM organization_members mm WHERE mm.organization_id = o.id) AS member_count,
                       (SELECT COUNT(*) FROM projects pp WHERE pp.organization_id = o.id AND pp.deleted_at IS NULL) AS project_count
                FROM organizations o
                WHERE o.deleted_at IS NULL
                ORDER BY o.id DESC
                """
            )
            rows = cursor.fetchall() or []
    return [_serialize_org(row) for row in rows]
