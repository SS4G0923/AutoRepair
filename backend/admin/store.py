from __future__ import annotations

from typing import Any

from backend.auth.store import get_db_connection


def list_users_for_admin(*, limit: int = 200) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    u.id,
                    u.email,
                    u.display_name,
                    u.avatar_url,
                    u.auth_source,
                    u.role,
                    u.account_status,
                    u.created_at,
                    u.updated_at,
                    u.last_login_at,
                    (
                        SELECT COUNT(*)
                        FROM conversation_histories ch
                        WHERE ch.user_id = u.id AND ch.deleted_at IS NULL
                    ) AS history_count,
                    (
                        SELECT COUNT(*)
                        FROM llm_requests lr
                        WHERE lr.user_id = u.id
                    ) AS llm_request_count,
                    (
                        SELECT COALESCE(SUM(lt.total_tokens), 0)
                        FROM llm_requests lr
                        LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                        WHERE lr.user_id = u.id
                    ) AS total_tokens,
                    (
                        SELECT COUNT(*)
                        FROM payment_orders po
                        WHERE po.user_id = u.id
                    ) AS payment_order_count,
                    (
                        SELECT sp.plan_name
                        FROM user_subscriptions us
                        INNER JOIN subscription_plans sp ON sp.id = us.plan_id
                        WHERE us.user_id = u.id AND us.subscription_status = 'active'
                        ORDER BY us.id DESC
                        LIMIT 1
                    ) AS active_subscription_plan
                FROM users u
                ORDER BY u.created_at DESC, u.id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall() or []
    return [
        {
            "id": int(row["id"]),
            "email": row["email"],
            "display_name": row["display_name"],
            "avatar_url": row["avatar_url"],
            "auth_source": row["auth_source"],
            "role": row["role"],
            "account_status": row["account_status"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "last_login_at": str(row["last_login_at"]) if row["last_login_at"] else None,
            "history_count": int(row["history_count"] or 0),
            "llm_request_count": int(row["llm_request_count"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "payment_order_count": int(row["payment_order_count"] or 0),
            "active_subscription_plan": row["active_subscription_plan"],
        }
        for row in rows
    ]


def get_admin_dashboard() -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_users,
                    SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END) AS admin_users,
                    SUM(CASE WHEN role = 'advanced' THEN 1 ELSE 0 END) AS advanced_users,
                    SUM(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 ELSE 0 END) AS new_users_7d
                FROM users
                """
            )
            user_summary = cursor.fetchone() or {}

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS llm_requests_7d,
                    SUM(CASE WHEN request_mode = 'chat' THEN 1 ELSE 0 END) AS chat_requests_7d,
                    SUM(CASE WHEN request_mode = 'repair' THEN 1 ELSE 0 END) AS repair_requests_7d,
                    SUM(CASE WHEN request_status = 'failed' THEN 1 ELSE 0 END) AS failed_requests_7d,
                    COALESCE(SUM(lt.total_tokens), 0) AS total_tokens_7d
                FROM llm_requests lr
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE lr.started_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """
            )
            request_summary = cursor.fetchone() or {}

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS paid_orders_30d,
                    COALESCE(SUM(amount_cents), 0) AS paid_amount_cents_30d
                FROM payment_orders
                WHERE order_status = 'paid' AND paid_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                """
            )
            payment_summary = cursor.fetchone() or {}

            cursor.execute(
                """
                SELECT
                    DATE(lr.started_at) AS day,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(lt.total_tokens), 0) AS total_tokens
                FROM llm_requests lr
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE lr.started_at >= DATE_SUB(CURDATE(), INTERVAL 13 DAY)
                GROUP BY DATE(lr.started_at)
                ORDER BY day ASC
                """
            )
            daily_token_usage = [
                {
                    "day": str(row["day"]),
                    "request_count": int(row["request_count"] or 0),
                    "total_tokens": int(row["total_tokens"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    d.day AS day,
                    d.new_users AS new_users,
                    (
                        SELECT COUNT(*)
                        FROM users u2
                        WHERE DATE(u2.created_at) <= d.day
                    ) AS cumulative_users
                FROM (
                    SELECT
                        DATE(created_at) AS day,
                        COUNT(*) AS new_users
                    FROM users
                    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 13 DAY)
                    GROUP BY DATE(created_at)
                ) d
                ORDER BY d.day ASC
                """
            )
            daily_user_growth = [
                {
                    "day": str(row["day"]),
                    "new_users": int(row["new_users"] or 0),
                    "cumulative_users": int(row["cumulative_users"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    DATE(paid_at) AS day,
                    COUNT(*) AS paid_orders,
                    COALESCE(SUM(amount_cents), 0) AS paid_amount_cents
                FROM payment_orders
                WHERE order_status = 'paid'
                  AND paid_at >= DATE_SUB(CURDATE(), INTERVAL 13 DAY)
                GROUP BY DATE(paid_at)
                ORDER BY day ASC
                """
            )
            daily_payment_volume = [
                {
                    "day": str(row["day"]),
                    "paid_orders": int(row["paid_orders"] or 0),
                    "paid_amount_cents": int(row["paid_amount_cents"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    lr.model,
                    lr.provider,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(lt.total_tokens), 0) AS total_tokens,
                    AVG(lr.latency_ms) AS avg_latency_ms,
                    MAX(lr.started_at) AS last_used_at
                FROM llm_requests lr
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE lr.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY lr.model, lr.provider
                ORDER BY total_tokens DESC, request_count DESC
                LIMIT 8
                """
            )
            model_usage = [
                {
                    "model": row["model"],
                    "provider": row["provider"],
                    "request_count": int(row["request_count"] or 0),
                    "total_tokens": int(row["total_tokens"] or 0),
                    "avg_latency_ms": float(row["avg_latency_ms"] or 0),
                    "last_used_at": str(row["last_used_at"]) if row["last_used_at"] else None,
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    payment_method,
                    COUNT(*) AS paid_orders,
                    COALESCE(SUM(amount_cents), 0) AS paid_amount_cents
                FROM payment_orders
                WHERE order_status = 'paid'
                  AND paid_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY payment_method
                ORDER BY paid_amount_cents DESC, paid_orders DESC
                """
            )
            payment_method_usage = [
                {
                    "payment_method": row["payment_method"],
                    "paid_orders": int(row["paid_orders"] or 0),
                    "paid_amount_cents": int(row["paid_amount_cents"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    lr.id,
                    lr.request_mode,
                    lr.stage,
                    lr.purpose,
                    lr.provider,
                    lr.model,
                    lr.request_status,
                    lr.started_at,
                    u.id AS user_id,
                    u.email AS user_email,
                    u.display_name AS user_display_name,
                    COALESCE(lt.total_tokens, 0) AS total_tokens
                FROM llm_requests lr
                LEFT JOIN users u ON u.id = lr.user_id
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                ORDER BY lr.started_at DESC
                LIMIT 10
                """
            )
            latest_requests = [
                {
                    "id": int(row["id"]),
                    "request_mode": row["request_mode"],
                    "stage": row["stage"],
                    "purpose": row["purpose"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "request_status": row["request_status"],
                    "started_at": str(row["started_at"]),
                    "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
                    "user_email": row["user_email"],
                    "user_display_name": row["user_display_name"],
                    "total_tokens": int(row["total_tokens"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

    return {
        "summary": {
            "total_users": int(user_summary.get("total_users") or 0),
            "admin_users": int(user_summary.get("admin_users") or 0),
            "advanced_users": int(user_summary.get("advanced_users") or 0),
            "new_users_7d": int(user_summary.get("new_users_7d") or 0),
            "llm_requests_7d": int(request_summary.get("llm_requests_7d") or 0),
            "chat_requests_7d": int(request_summary.get("chat_requests_7d") or 0),
            "repair_requests_7d": int(request_summary.get("repair_requests_7d") or 0),
            "failed_requests_7d": int(request_summary.get("failed_requests_7d") or 0),
            "total_tokens_7d": int(request_summary.get("total_tokens_7d") or 0),
            "paid_orders_30d": int(payment_summary.get("paid_orders_30d") or 0),
            "paid_amount_cents_30d": int(payment_summary.get("paid_amount_cents_30d") or 0),
        },
        "daily_token_usage": daily_token_usage,
        "daily_user_growth": daily_user_growth,
        "daily_payment_volume": daily_payment_volume,
        "model_usage": model_usage,
        "payment_method_usage": payment_method_usage,
        "latest_requests": latest_requests,
    }


def list_llm_requests_for_admin(
    *,
    page: int = 1,
    page_size: int = 25,
    query: str = "",
    model: str = "",
    status: str = "",
    request_mode: str = "",
) -> dict[str, Any]:
    safe_page = max(1, page)
    safe_page_size = max(1, min(100, page_size))
    offset = (safe_page - 1) * safe_page_size

    where_clauses: list[str] = ["1 = 1"]
    params: list[Any] = []

    if query.strip():
        like = f"%{query.strip()}%"
        where_clauses.append(
            """
            (
                u.email LIKE %s
                OR u.display_name LIKE %s
                OR lr.model LIKE %s
                OR COALESCE(lr.purpose, '') LIKE %s
            )
            """
        )
        params.extend([like, like, like, like])
    if model.strip():
        where_clauses.append("lr.model = %s")
        params.append(model.strip())
    if status.strip():
        where_clauses.append("lr.request_status = %s")
        params.append(status.strip())
    if request_mode.strip():
        where_clauses.append("lr.request_mode = %s")
        params.append(request_mode.strip())

    where_sql = " AND ".join(where_clauses)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM llm_requests lr
                LEFT JOIN users u ON u.id = lr.user_id
                WHERE {where_sql}
                """,
                params,
            )
            total = int((cursor.fetchone() or {}).get("total") or 0)

            cursor.execute(
                f"""
                SELECT
                    lr.id,
                    lr.user_id,
                    u.email AS user_email,
                    u.display_name AS user_display_name,
                    lr.history_id,
                    lr.request_mode,
                    lr.stage,
                    lr.purpose,
                    lr.provider,
                    lr.model,
                    lr.source_type,
                    lr.is_streaming,
                    lr.is_json_response,
                    lr.request_status,
                    lr.token_source,
                    lr.prompt_chars,
                    lr.response_chars,
                    lr.latency_ms,
                    lr.error_message,
                    lr.started_at,
                    lr.finished_at,
                    COALESCE(lt.input_tokens, 0) AS input_tokens,
                    COALESCE(lt.output_tokens, 0) AS output_tokens,
                    COALESCE(lt.total_tokens, 0) AS total_tokens
                FROM llm_requests lr
                LEFT JOIN users u ON u.id = lr.user_id
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE {where_sql}
                ORDER BY lr.started_at DESC, lr.id DESC
                LIMIT %s OFFSET %s
                """,
                [*params, safe_page_size, offset],
            )
            rows = cursor.fetchall() or []

    return {
        "items": [
            {
                "id": int(row["id"]),
                "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
                "user_email": row["user_email"],
                "user_display_name": row["user_display_name"],
                "history_id": int(row["history_id"]) if row["history_id"] is not None else None,
                "request_mode": row["request_mode"],
                "stage": row["stage"],
                "purpose": row["purpose"],
                "provider": row["provider"],
                "model": row["model"],
                "source_type": row["source_type"],
                "is_streaming": bool(row["is_streaming"]),
                "is_json_response": bool(row["is_json_response"]),
                "request_status": row["request_status"],
                "token_source": row["token_source"],
                "prompt_chars": int(row["prompt_chars"] or 0),
                "response_chars": int(row["response_chars"] or 0),
                "latency_ms": int(row["latency_ms"] or 0),
                "error_message": row["error_message"],
                "started_at": str(row["started_at"]),
                "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
            }
            for row in rows
        ],
        "page": safe_page,
        "page_size": safe_page_size,
        "total": total,
    }


def get_llm_request_detail(request_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    lr.id,
                    lr.user_id,
                    u.email AS user_email,
                    u.display_name AS user_display_name,
                    lr.history_id,
                    lr.request_mode,
                    lr.stage,
                    lr.purpose,
                    lr.provider,
                    lr.model,
                    lr.source_type,
                    lr.is_streaming,
                    lr.is_json_response,
                    lr.request_status,
                    lr.token_source,
                    lr.prompt_chars,
                    lr.response_chars,
                    lr.latency_ms,
                    lr.error_message,
                    lr.started_at,
                    lr.finished_at,
                    COALESCE(lt.input_tokens, 0) AS input_tokens,
                    COALESCE(lt.output_tokens, 0) AS output_tokens,
                    COALESCE(lt.total_tokens, 0) AS total_tokens,
                    COALESCE(lt.cached_input_tokens, 0) AS cached_input_tokens,
                    COALESCE(lt.reasoning_tokens, 0) AS reasoning_tokens,
                    msg.system_prompt,
                    msg.prompt_text,
                    msg.response_text,
                    msg.parsed_response_json
                FROM llm_requests lr
                LEFT JOIN users u ON u.id = lr.user_id
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                LEFT JOIN llm_request_messages msg ON msg.request_id = lr.id
                WHERE lr.id = %s
                """,
                (request_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                """
                SELECT
                    id,
                    round_index,
                    status,
                    tool_name,
                    arguments_json,
                    output_preview,
                    output_truncated,
                    created_at
                FROM llm_request_tool_events
                WHERE request_id = %s
                ORDER BY id ASC
                """,
                (request_id,),
            )
            tool_rows = cursor.fetchall() or []

    return {
        "request": {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
            "user_email": row["user_email"],
            "user_display_name": row["user_display_name"],
            "history_id": int(row["history_id"]) if row["history_id"] is not None else None,
            "request_mode": row["request_mode"],
            "stage": row["stage"],
            "purpose": row["purpose"],
            "provider": row["provider"],
            "model": row["model"],
            "source_type": row["source_type"],
            "is_streaming": bool(row["is_streaming"]),
            "is_json_response": bool(row["is_json_response"]),
            "request_status": row["request_status"],
            "token_source": row["token_source"],
            "prompt_chars": int(row["prompt_chars"] or 0),
            "response_chars": int(row["response_chars"] or 0),
            "latency_ms": int(row["latency_ms"] or 0),
            "error_message": row["error_message"],
            "started_at": str(row["started_at"]),
            "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "cached_input_tokens": int(row["cached_input_tokens"] or 0),
            "reasoning_tokens": int(row["reasoning_tokens"] or 0),
        },
        "message": {
            "system_prompt": row["system_prompt"] or "",
            "prompt_text": row["prompt_text"] or "",
            "response_text": row["response_text"] or "",
            "parsed_response_json": row["parsed_response_json"] or "",
        },
        "tool_events": [
            {
                "id": int(tool_row["id"]),
                "round_index": int(tool_row["round_index"]) if tool_row["round_index"] is not None else None,
                "status": tool_row["status"],
                "tool_name": tool_row["tool_name"],
                "arguments_json": tool_row["arguments_json"] or "",
                "output_preview": tool_row["output_preview"] or "",
                "output_truncated": bool(tool_row["output_truncated"]),
                "created_at": str(tool_row["created_at"]),
            }
            for tool_row in tool_rows
        ],
    }


def get_model_usage_report(*, days: int = 30) -> dict[str, Any]:
    safe_days = max(1, min(120, days))
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    lr.model,
                    lr.provider,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(lt.total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(lt.input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(lt.output_tokens), 0) AS output_tokens,
                    AVG(lr.latency_ms) AS avg_latency_ms,
                    MAX(lr.started_at) AS last_used_at
                FROM llm_requests lr
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE lr.started_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY lr.model, lr.provider
                ORDER BY total_tokens DESC, request_count DESC
                """,
                (safe_days,),
            )
            items = [
                {
                    "model": row["model"],
                    "provider": row["provider"],
                    "request_count": int(row["request_count"] or 0),
                    "total_tokens": int(row["total_tokens"] or 0),
                    "input_tokens": int(row["input_tokens"] or 0),
                    "output_tokens": int(row["output_tokens"] or 0),
                    "avg_latency_ms": float(row["avg_latency_ms"] or 0),
                    "last_used_at": str(row["last_used_at"]) if row["last_used_at"] else None,
                }
                for row in (cursor.fetchall() or [])
            ]

            cursor.execute(
                """
                SELECT
                    DATE(lr.started_at) AS day,
                    lr.model,
                    COUNT(*) AS request_count,
                    COALESCE(SUM(lt.total_tokens), 0) AS total_tokens
                FROM llm_requests lr
                LEFT JOIN llm_token_usage lt ON lt.request_id = lr.id
                WHERE lr.started_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY DATE(lr.started_at), lr.model
                ORDER BY day ASC, total_tokens DESC
                """,
                (safe_days - 1,),
            )
            daily_series = [
                {
                    "day": str(row["day"]),
                    "model": row["model"],
                    "request_count": int(row["request_count"] or 0),
                    "total_tokens": int(row["total_tokens"] or 0),
                }
                for row in (cursor.fetchall() or [])
            ]

    return {
        "days": safe_days,
        "items": items,
        "daily_series": daily_series,
    }


def list_login_events_for_admin(*, page: int = 1, page_size: int = 50) -> dict[str, Any]:
    safe_page = max(1, page)
    safe_page_size = max(1, min(100, page_size))
    offset = (safe_page - 1) * safe_page_size

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM user_login_events")
            total = int((cursor.fetchone() or {}).get("total") or 0)
            cursor.execute(
                """
                SELECT
                    e.id,
                    e.user_id,
                    u.email AS user_email,
                    u.display_name AS user_display_name,
                    e.email_attempt,
                    e.login_method,
                    e.login_status,
                    e.failure_reason,
                    e.ip_address,
                    e.user_agent,
                    e.created_at
                FROM user_login_events e
                LEFT JOIN users u ON u.id = e.user_id
                ORDER BY e.created_at DESC, e.id DESC
                LIMIT %s OFFSET %s
                """,
                (safe_page_size, offset),
            )
            rows = cursor.fetchall() or []

    return {
        "items": [
            {
                "id": int(row["id"]),
                "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
                "user_email": row["user_email"],
                "user_display_name": row["user_display_name"],
                "email_attempt": row["email_attempt"],
                "login_method": row["login_method"],
                "login_status": row["login_status"],
                "failure_reason": row["failure_reason"],
                "ip_address": row["ip_address"],
                "user_agent": row["user_agent"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ],
        "page": safe_page,
        "page_size": safe_page_size,
        "total": total,
    }
