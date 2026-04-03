from __future__ import annotations

import json
from typing import Any

from auth_store import get_db_connection


def _serialize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "mode": row["mode"],
        "title": row["title"],
        "preview_text": row.get("preview_text") or "",
        "model": row.get("model"),
        "language": row.get("language"),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def list_histories_for_user(user_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, mode, title, preview_text, model, language, created_at, updated_at
                FROM conversation_histories
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall() or []
    return [_serialize_history_row(row) for row in rows]


def get_history_for_user(user_id: int, history_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, mode, title, preview_text, model, language, snapshot_json, created_at, updated_at
                FROM conversation_histories
                WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """,
                (history_id, user_id),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    serialized = _serialize_history_row(row)
    serialized["snapshot"] = json.loads(row["snapshot_json"])
    return serialized


def save_history(
    *,
    user_id: int,
    mode: str,
    title: str,
    preview_text: str,
    snapshot: dict[str, Any],
    model: str | None = None,
    language: str | None = None,
    history_id: int | None = None,
) -> dict[str, Any]:
    clean_title = title.strip()[:255] or "Untitled"
    clean_preview = preview_text.strip()
    snapshot_json = json.dumps(snapshot, ensure_ascii=False)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            if history_id is not None:
                cursor.execute(
                    """
                    UPDATE conversation_histories
                    SET title = %s,
                        preview_text = %s,
                        model = %s,
                        language = %s,
                        snapshot_json = %s
                    WHERE id = %s AND user_id = %s AND mode = %s AND deleted_at IS NULL
                    """,
                    (
                        clean_title,
                        clean_preview,
                        model,
                        language,
                        snapshot_json,
                        history_id,
                        user_id,
                        mode,
                    ),
                )
                if cursor.rowcount == 0:
                    raise ValueError("History record was not found for this user.")
                saved_id = history_id
            else:
                cursor.execute(
                    """
                    INSERT INTO conversation_histories (
                        user_id, mode, title, preview_text, model, language, snapshot_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        mode,
                        clean_title,
                        clean_preview,
                        model,
                        language,
                        snapshot_json,
                    ),
                )
                saved_id = int(cursor.lastrowid)
        connection.commit()

    saved = get_history_for_user(user_id, saved_id)
    if saved is None:
        raise RuntimeError("History record was saved but could not be reloaded.")
    return saved


def soft_delete_history_for_user(user_id: int, history_id: int) -> bool:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE conversation_histories
                SET deleted_at = NOW()
                WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """,
                (history_id, user_id),
            )
            deleted = cursor.rowcount > 0
        connection.commit()
    return deleted
