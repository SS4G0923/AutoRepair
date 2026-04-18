from __future__ import annotations

import hashlib
import secrets
from typing import Any

from backend.auth.store import get_db_connection


def _serialize_preferences(row: dict[str, Any] | None, user_id: int) -> dict[str, Any]:
    if row is None:
        return {
            "user_id": user_id,
            "default_agent_model": None,
            "default_chat_model": None,
            "default_language": None,
            "locale": "zh",
            "theme": "dark",
            "timezone": None,
            "bio": None,
            "show_site_map_widget": True,
            "updated_at": None,
        }
    return {
        "user_id": int(row["user_id"]),
        "default_agent_model": row.get("default_agent_model"),
        "default_chat_model": row.get("default_chat_model"),
        "default_language": row.get("default_language"),
        "locale": row.get("locale") or "zh",
        "theme": row.get("theme") or "dark",
        "timezone": row.get("timezone"),
        "bio": row.get("bio"),
        "show_site_map_widget": bool(row.get("show_site_map_widget", 1)),
        "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
    }


def get_preferences(user_id: int) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, default_agent_model, default_chat_model, default_language,
                       locale, theme, timezone, bio, show_site_map_widget, updated_at
                FROM user_preferences WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
    return _serialize_preferences(row, user_id)


def update_preferences(
    *,
    user_id: int,
    default_agent_model: str | None = None,
    default_chat_model: str | None = None,
    default_language: str | None = None,
    locale: str | None = None,
    theme: str | None = None,
    timezone: str | None = None,
    bio: str | None = None,
    show_site_map_widget: bool | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "default_agent_model": default_agent_model,
        "default_chat_model": default_chat_model,
        "default_language": default_language,
        "locale": locale,
        "theme": theme,
        "timezone": timezone,
        "bio": bio,
    }
    if show_site_map_widget is not None:
        fields["show_site_map_widget"] = 1 if show_site_map_widget else 0

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_preferences
                    (user_id, default_agent_model, default_chat_model, default_language,
                     locale, theme, timezone, bio, show_site_map_widget)
                VALUES (%s, %s, %s, %s, COALESCE(%s,'zh'), COALESCE(%s,'dark'), %s, %s, COALESCE(%s, 1))
                ON DUPLICATE KEY UPDATE
                    default_agent_model = COALESCE(VALUES(default_agent_model), default_agent_model),
                    default_chat_model  = COALESCE(VALUES(default_chat_model), default_chat_model),
                    default_language    = COALESCE(VALUES(default_language), default_language),
                    locale              = VALUES(locale),
                    theme               = VALUES(theme),
                    timezone            = COALESCE(VALUES(timezone), timezone),
                    bio                 = COALESCE(VALUES(bio), bio),
                    show_site_map_widget= VALUES(show_site_map_widget)
                """,
                (
                    user_id,
                    fields["default_agent_model"],
                    fields["default_chat_model"],
                    fields["default_language"],
                    fields["locale"],
                    fields["theme"],
                    fields["timezone"],
                    fields["bio"],
                    fields.get("show_site_map_widget"),
                ),
            )
        connection.commit()
    return get_preferences(user_id)


def _serialize_api_token(row: dict[str, Any], *, reveal_token: str | None = None) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "token_name": row["token_name"],
        "token_prefix": row["token_prefix"],
        "scope": row["scope"],
        "last_used_at": str(row["last_used_at"]) if row.get("last_used_at") else None,
        "expires_at": str(row["expires_at"]) if row.get("expires_at") else None,
        "revoked_at": str(row["revoked_at"]) if row.get("revoked_at") else None,
        "created_at": str(row["created_at"]),
        "revealed_token": reveal_token,
    }


def list_api_tokens(user_id: int) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, token_name, token_prefix, scope,
                       last_used_at, expires_at, revoked_at, created_at
                FROM user_api_tokens
                WHERE user_id = %s
                ORDER BY id DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall() or []
    return [_serialize_api_token(r) for r in rows]


def create_api_token(
    *,
    user_id: int,
    token_name: str,
    scope: str = "repair",
) -> dict[str, Any]:
    clean_name = token_name.strip()[:64]
    if not clean_name:
        raise ValueError("Token name is required.")
    raw_token = f"arp_{secrets.token_urlsafe(32)}"
    prefix = raw_token[:10]
    hashed = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_api_tokens
                    (user_id, token_name, token_prefix, token_hash, scope)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, clean_name, prefix, hashed, scope),
            )
            token_id = int(cursor.lastrowid)
        connection.commit()
    for row in list_api_tokens(user_id):
        if row["id"] == token_id:
            row["revealed_token"] = raw_token
            return row
    raise RuntimeError("Token could not be reloaded after creation.")


def revoke_api_token(*, user_id: int, token_id: int) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE user_api_tokens SET revoked_at = NOW() WHERE id = %s AND user_id = %s",
                (token_id, user_id),
            )
        connection.commit()


def get_profile_overview(user_id: int) -> dict[str, Any]:
    """A lightweight aggregate used by the Personal Center landing card."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM conversation_histories WHERE user_id = %s AND deleted_at IS NULL) AS total_histories,
                    (SELECT COUNT(*) FROM conversation_histories WHERE user_id = %s AND deleted_at IS NULL AND mode = 'agent') AS total_repair_sessions,
                    (SELECT COUNT(*) FROM conversation_histories WHERE user_id = %s AND deleted_at IS NULL AND mode = 'chat') AS total_chat_sessions,
                    (SELECT COUNT(*) FROM benchmark_runs WHERE user_id = %s) AS total_benchmark_runs,
                    (SELECT COUNT(*) FROM organization_members WHERE user_id = %s) AS organization_count,
                    (SELECT COALESCE(SUM(lt.total_tokens), 0)
                     FROM llm_requests lr
                     LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                     WHERE lr.user_id = %s) AS lifetime_tokens
                """,
                (user_id, user_id, user_id, user_id, user_id, user_id),
            )
            overview = cursor.fetchone() or {}
    return {
        "total_histories": int(overview.get("total_histories") or 0),
        "total_repair_sessions": int(overview.get("total_repair_sessions") or 0),
        "total_chat_sessions": int(overview.get("total_chat_sessions") or 0),
        "total_benchmark_runs": int(overview.get("total_benchmark_runs") or 0),
        "organization_count": int(overview.get("organization_count") or 0),
        "lifetime_tokens": int(overview.get("lifetime_tokens") or 0),
    }
