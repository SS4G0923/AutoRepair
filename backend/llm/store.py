from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from backend.auth.store import get_db_connection


ModelPurpose = Literal["chat", "repair"]

PROVIDER_CODE_OPTIONS = {"openai_compatible", "gemini"}


@dataclass(frozen=True)
class RuntimeModelConfig:
    id: int
    model_key: str
    display_name: str
    provider_code: str
    provider_name: str
    vendor_name: str
    api_model_name: str
    api_base_url: str | None
    api_key_env_var: str | None
    enabled: bool
    supports_streaming: bool
    supports_json: bool
    is_default_chat: bool
    is_default_repair: bool
    sort_order: int
    notes: str | None = None
    extra_config: dict[str, Any] | None = None


def _normalize_thinking_enabled(extra_config: dict[str, Any] | None) -> bool:
    if not extra_config:
        return False
    raw_value = extra_config.get("thinking_enabled")
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return False


def _parse_extra_config(raw_value: Any) -> dict[str, Any] | None:
    if raw_value in (None, "", b""):
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _serialize_model_row(row: dict[str, Any]) -> dict[str, Any]:
    api_key_env_var = row.get("api_key_env_var")
    extra_config = _parse_extra_config(row.get("extra_config_json"))
    api_key_configured = True if not api_key_env_var else bool(os.getenv(str(api_key_env_var)))
    missing_configuration: list[str] = []
    if api_key_env_var and not api_key_configured:
        missing_configuration.append(str(api_key_env_var))
    return {
        "id": int(row["id"]),
        "provider_code": row["provider_code"],
        "provider_name": row["provider_name"],
        "vendor_name": row["vendor_name"],
        "model_key": row["model_key"],
        "display_name": row["display_name"],
        "api_model_name": row["api_model_name"],
        "api_base_url": row["api_base_url"],
        "api_key_env_var": api_key_env_var,
        "enabled": bool(row["enabled"]),
        "is_default_chat": bool(row["is_default_chat"]),
        "is_default_repair": bool(row["is_default_repair"]),
        "supports_streaming": bool(row["supports_streaming"]),
        "supports_json": bool(row["supports_json"]),
        "sort_order": int(row["sort_order"] or 0),
        "notes": row.get("notes"),
        "extra_config": extra_config or {},
        "thinking_enabled": _normalize_thinking_enabled(extra_config),
        "api_key_configured": api_key_configured,
        "missing_configuration": missing_configuration,
        "request_count_30d": int(row.get("request_count_30d") or 0),
        "total_tokens_30d": int(row.get("total_tokens_30d") or 0),
        "last_used_at": str(row["last_used_at"]) if row.get("last_used_at") else None,
        "created_at": str(row["created_at"]) if row.get("created_at") else None,
        "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
    }


def _runtime_from_row(row: dict[str, Any]) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        id=int(row["id"]),
        model_key=row["model_key"],
        display_name=row["display_name"],
        provider_code=row["provider_code"],
        provider_name=row["provider_name"],
        vendor_name=row["vendor_name"],
        api_model_name=row["api_model_name"],
        api_base_url=row["api_base_url"],
        api_key_env_var=row["api_key_env_var"],
        enabled=bool(row["enabled"]),
        supports_streaming=bool(row["supports_streaming"]),
        supports_json=bool(row["supports_json"]),
        is_default_chat=bool(row["is_default_chat"]),
        is_default_repair=bool(row["is_default_repair"]),
        sort_order=int(row["sort_order"] or 0),
        notes=row.get("notes"),
        extra_config=_parse_extra_config(row.get("extra_config_json")),
    )


def _fetch_model_row_by_id(cursor, model_config_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT *
        FROM llm_model_configs
        WHERE id = %s AND deleted_at IS NULL
        """,
        (model_config_id,),
    )
    return cursor.fetchone()


def _fetch_model_row_by_key(
    cursor,
    model_key: str,
    *,
    include_disabled: bool,
) -> dict[str, Any] | None:
    sql = """
        SELECT *
        FROM llm_model_configs
        WHERE model_key = %s
          AND deleted_at IS NULL
    """
    params: list[Any] = [model_key]
    if not include_disabled:
        sql += " AND enabled = 1"
    sql += " LIMIT 1"
    cursor.execute(sql, tuple(params))
    return cursor.fetchone()


def _fetch_any_model_row_by_key(cursor, model_key: str) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT *
        FROM llm_model_configs
        WHERE model_key = %s
        LIMIT 1
        """,
        (model_key,),
    )
    return cursor.fetchone()


def _count_enabled_models(cursor) -> int:
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM llm_model_configs
        WHERE deleted_at IS NULL
          AND enabled = 1
        """
    )
    row = cursor.fetchone() or {}
    return int(row.get("total") or 0)


def _ensure_default_flags(cursor) -> None:
    cursor.execute(
        """
        SELECT id
        FROM llm_model_configs
        WHERE deleted_at IS NULL
          AND enabled = 1
        ORDER BY sort_order ASC, id ASC
        LIMIT 1
        """
    )
    first_enabled = cursor.fetchone()
    if first_enabled is None:
        return

    first_enabled_id = int(first_enabled["id"])

    cursor.execute(
        """
        SELECT id
        FROM llm_model_configs
        WHERE deleted_at IS NULL
          AND enabled = 1
          AND is_default_chat = 1
        LIMIT 1
        """
    )
    if cursor.fetchone() is None:
        cursor.execute(
            """
            UPDATE llm_model_configs
            SET is_default_chat = CASE WHEN id = %s THEN 1 ELSE 0 END
            WHERE deleted_at IS NULL
              AND enabled = 1
            """,
            (first_enabled_id,),
        )

    cursor.execute(
        """
        SELECT id
        FROM llm_model_configs
        WHERE deleted_at IS NULL
          AND enabled = 1
          AND is_default_repair = 1
        LIMIT 1
        """
    )
    if cursor.fetchone() is None:
        cursor.execute(
            """
            UPDATE llm_model_configs
            SET is_default_repair = CASE WHEN id = %s THEN 1 ELSE 0 END
            WHERE deleted_at IS NULL
              AND enabled = 1
            """,
            (first_enabled_id,),
        )


def _apply_default_flags(
    cursor,
    *,
    model_config_id: int,
    is_default_chat: bool,
    is_default_repair: bool,
) -> None:
    if is_default_chat:
        cursor.execute(
            """
            UPDATE llm_model_configs
            SET is_default_chat = CASE WHEN id = %s THEN 1 ELSE 0 END
            WHERE deleted_at IS NULL
            """,
            (model_config_id,),
        )
    if is_default_repair:
        cursor.execute(
            """
            UPDATE llm_model_configs
            SET is_default_repair = CASE WHEN id = %s THEN 1 ELSE 0 END
            WHERE deleted_at IS NULL
            """,
            (model_config_id,),
        )


def _get_default_runtime_model(cursor, purpose: ModelPurpose) -> RuntimeModelConfig | None:
    default_column = "is_default_chat" if purpose == "chat" else "is_default_repair"
    cursor.execute(
        f"""
        SELECT *
        FROM llm_model_configs
        WHERE deleted_at IS NULL
          AND enabled = 1
        ORDER BY {default_column} DESC, sort_order ASC, id ASC
        """
    )
    rows = cursor.fetchall() or []
    for row in rows:
        runtime_model = _runtime_from_row(row)
        if runtime_model.api_key_env_var is None or load_model_api_key(runtime_model):
            return runtime_model
    return None


def list_public_model_catalog() -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM llm_model_configs
                WHERE deleted_at IS NULL
                  AND enabled = 1
                ORDER BY
                    is_default_repair DESC,
                    is_default_chat DESC,
                    sort_order ASC,
                    id ASC
                """
            )
            rows = cursor.fetchall() or []

    visible_runtime_models: list[RuntimeModelConfig] = []
    for row in rows:
        runtime_model = _runtime_from_row(row)
        if runtime_model.api_key_env_var is None or load_model_api_key(runtime_model):
            visible_runtime_models.append(runtime_model)

    default_chat = next((item for item in visible_runtime_models if item.is_default_chat), None)
    default_repair = next((item for item in visible_runtime_models if item.is_default_repair), None)
    if default_chat is None:
        default_chat = visible_runtime_models[0] if visible_runtime_models else None
    if default_repair is None:
        default_repair = visible_runtime_models[0] if visible_runtime_models else None

    return {
        "items": [
            {
                "value": item.model_key,
                "label": item.display_name,
                "provider_code": item.provider_code,
                "provider_name": item.provider_name,
                "vendor_name": item.vendor_name,
                "supports_streaming": item.supports_streaming,
                "supports_json": item.supports_json,
                "is_default_chat": item.is_default_chat,
                "is_default_repair": item.is_default_repair,
            }
            for item in visible_runtime_models
        ],
        "default_chat_model": default_chat.model_key if default_chat else None,
        "default_repair_model": default_repair.model_key if default_repair else None,
    }


def list_model_configs_for_admin() -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.*,
                    COALESCE(stats.request_count_30d, 0) AS request_count_30d,
                    COALESCE(stats.total_tokens_30d, 0) AS total_tokens_30d,
                    stats.last_used_at
                FROM llm_model_configs c
                LEFT JOIN (
                    SELECT
                        lr.model,
                        COUNT(*) AS request_count_30d,
                        COALESCE(SUM(lt.total_tokens), 0) AS total_tokens_30d,
                        MAX(lr.started_at) AS last_used_at
                    FROM llm_requests lr
                    LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                    WHERE lr.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    GROUP BY lr.model
                ) stats
                    ON stats.model = c.model_key
                WHERE c.deleted_at IS NULL
                ORDER BY c.sort_order ASC, c.id ASC
                """
            )
            rows = cursor.fetchall() or []
    return [_serialize_model_row(row) for row in rows]


def get_model_config_by_id(model_config_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            row = _fetch_model_row_by_id(cursor, model_config_id)
    return _serialize_model_row(row) if row else None


def get_runtime_model_config(
    model_key: str,
    *,
    include_disabled: bool = False,
) -> RuntimeModelConfig | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            row = _fetch_model_row_by_key(cursor, model_key, include_disabled=include_disabled)
    return _runtime_from_row(row) if row else None


def resolve_model_selection(
    selected_model_key: str | None,
    *,
    purpose: ModelPurpose,
) -> RuntimeModelConfig:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            if selected_model_key:
                row = _fetch_model_row_by_key(cursor, selected_model_key, include_disabled=False)
                if row is None:
                    raise ValueError(f"Model `{selected_model_key}` is not enabled or does not exist.")
                runtime_model = _runtime_from_row(row)
                api_key = load_model_api_key(runtime_model)
                if runtime_model.api_key_env_var and not api_key:
                    raise ValueError(
                        f"Model `{selected_model_key}` is missing environment variable "
                        f"`{runtime_model.api_key_env_var}`."
                    )
                if runtime_model.api_key_env_var is not None and not api_key:
                    raise ValueError(f"Model `{selected_model_key}` does not have a configured API key.")
                return runtime_model

            default_model = _get_default_runtime_model(cursor, purpose)
            if default_model is None:
                raise ValueError(
                    f"No enabled {purpose} model with a configured API key is available."
                )
            return default_model


def create_model_config(
    *,
    provider_code: str,
    provider_name: str,
    vendor_name: str,
    model_key: str,
    display_name: str,
    api_model_name: str,
    api_base_url: str | None,
    api_key_env_var: str | None,
    enabled: bool,
    is_default_chat: bool,
    is_default_repair: bool,
    supports_streaming: bool,
    supports_json: bool,
    sort_order: int,
    notes: str | None,
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if provider_code not in PROVIDER_CODE_OPTIONS:
        raise ValueError("`provider_code` is invalid.")

    effective_default_chat = bool(enabled and is_default_chat)
    effective_default_repair = bool(enabled and is_default_repair)
    extra_config_json = json.dumps(extra_config or {}, ensure_ascii=False)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            existing = _fetch_any_model_row_by_key(cursor, model_key)
            if existing is None:
                cursor.execute(
                    """
                    INSERT INTO llm_model_configs (
                        provider_code,
                        provider_name,
                        vendor_name,
                        model_key,
                        display_name,
                        api_model_name,
                        api_base_url,
                        api_key_env_var,
                        enabled,
                        is_default_chat,
                        is_default_repair,
                        supports_streaming,
                        supports_json,
                        sort_order,
                        notes,
                        extra_config_json,
                        deleted_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
                    """,
                    (
                        provider_code,
                        provider_name,
                        vendor_name,
                        model_key,
                        display_name,
                        api_model_name,
                        api_base_url,
                        api_key_env_var,
                        1 if enabled else 0,
                        1 if effective_default_chat else 0,
                        1 if effective_default_repair else 0,
                        1 if supports_streaming else 0,
                        1 if supports_json else 0,
                        sort_order,
                        notes,
                        extra_config_json,
                    ),
                )
                model_config_id = int(cursor.lastrowid)
            else:
                model_config_id = int(existing["id"])
                cursor.execute(
                    """
                    UPDATE llm_model_configs
                    SET provider_code = %s,
                        provider_name = %s,
                        vendor_name = %s,
                        display_name = %s,
                        api_model_name = %s,
                        api_base_url = %s,
                        api_key_env_var = %s,
                        enabled = %s,
                        is_default_chat = %s,
                        is_default_repair = %s,
                        supports_streaming = %s,
                        supports_json = %s,
                        sort_order = %s,
                        notes = %s,
                        extra_config_json = %s,
                        deleted_at = NULL
                    WHERE id = %s
                    """,
                    (
                        provider_code,
                        provider_name,
                        vendor_name,
                        display_name,
                        api_model_name,
                        api_base_url,
                        api_key_env_var,
                        1 if enabled else 0,
                        1 if effective_default_chat else 0,
                        1 if effective_default_repair else 0,
                        1 if supports_streaming else 0,
                        1 if supports_json else 0,
                        sort_order,
                        notes,
                        extra_config_json,
                        model_config_id,
                    ),
                )

            _apply_default_flags(
                cursor,
                model_config_id=model_config_id,
                is_default_chat=effective_default_chat,
                is_default_repair=effective_default_repair,
            )
            _ensure_default_flags(cursor)
        connection.commit()

    created = get_model_config_by_id(model_config_id)
    if created is None:
        raise RuntimeError("Model config was saved but could not be reloaded.")
    return created


def update_model_config(
    *,
    model_config_id: int,
    provider_code: str,
    provider_name: str,
    vendor_name: str,
    model_key: str,
    display_name: str,
    api_model_name: str,
    api_base_url: str | None,
    api_key_env_var: str | None,
    enabled: bool,
    is_default_chat: bool,
    is_default_repair: bool,
    supports_streaming: bool,
    supports_json: bool,
    sort_order: int,
    notes: str | None,
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if provider_code not in PROVIDER_CODE_OPTIONS:
        raise ValueError("`provider_code` is invalid.")

    effective_default_chat = bool(enabled and is_default_chat)
    effective_default_repair = bool(enabled and is_default_repair)
    extra_config_json = json.dumps(extra_config or {}, ensure_ascii=False)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            current = _fetch_model_row_by_id(cursor, model_config_id)
            if current is None:
                raise ValueError("Model config was not found.")

            duplicate = _fetch_any_model_row_by_key(cursor, model_key)
            if duplicate is not None and int(duplicate["id"]) != model_config_id:
                if duplicate.get("deleted_at") is None:
                    raise ValueError(f"Model key `{model_key}` is already in use.")
                raise ValueError(
                    f"Model key `{model_key}` belongs to a deleted model. Please restore or rename it."
                )

            if bool(current["enabled"]) and not enabled and _count_enabled_models(cursor) <= 1:
                raise ValueError("At least one enabled model must remain configured.")

            cursor.execute(
                """
                UPDATE llm_model_configs
                SET provider_code = %s,
                    provider_name = %s,
                    vendor_name = %s,
                    model_key = %s,
                    display_name = %s,
                    api_model_name = %s,
                    api_base_url = %s,
                    api_key_env_var = %s,
                    enabled = %s,
                    is_default_chat = %s,
                    is_default_repair = %s,
                    supports_streaming = %s,
                    supports_json = %s,
                    sort_order = %s,
                    notes = %s,
                    extra_config_json = %s
                WHERE id = %s
                """,
                (
                    provider_code,
                    provider_name,
                    vendor_name,
                    model_key,
                    display_name,
                    api_model_name,
                    api_base_url,
                    api_key_env_var,
                    1 if enabled else 0,
                    1 if effective_default_chat else 0,
                    1 if effective_default_repair else 0,
                    1 if supports_streaming else 0,
                    1 if supports_json else 0,
                    sort_order,
                    notes,
                    extra_config_json,
                    model_config_id,
                ),
            )

            _apply_default_flags(
                cursor,
                model_config_id=model_config_id,
                is_default_chat=effective_default_chat,
                is_default_repair=effective_default_repair,
            )
            _ensure_default_flags(cursor)
        connection.commit()

    updated = get_model_config_by_id(model_config_id)
    if updated is None:
        raise RuntimeError("Model config was updated but could not be reloaded.")
    return updated


def delete_model_config(model_config_id: int) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            current = _fetch_model_row_by_id(cursor, model_config_id)
            if current is None:
                raise ValueError("Model config was not found.")
            if bool(current["enabled"]) and _count_enabled_models(cursor) <= 1:
                raise ValueError("At least one enabled model must remain configured.")
            cursor.execute(
                """
                UPDATE llm_model_configs
                SET enabled = 0,
                    is_default_chat = 0,
                    is_default_repair = 0,
                    deleted_at = NOW()
                WHERE id = %s
                """,
                (model_config_id,),
            )
            _ensure_default_flags(cursor)
        connection.commit()


def load_model_api_key(runtime_model: RuntimeModelConfig) -> str | None:
    if runtime_model.api_key_env_var:
        return os.getenv(runtime_model.api_key_env_var)
    return ""


def is_model_thinking_enabled(runtime_model: RuntimeModelConfig) -> bool:
    return _normalize_thinking_enabled(runtime_model.extra_config)
