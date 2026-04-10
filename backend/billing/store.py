from __future__ import annotations

import json
import secrets
from datetime import datetime
from typing import Any

from backend.auth.store import get_db_connection, get_user_by_id
from backend.billing.providers import build_checkout_payload, payment_environment, payment_provider_profile

SUPPORTED_PAYMENT_METHODS = ("card", "paypal", "wechat", "alipay")
SUPPORTED_USER_ROLES = ("basic", "advanced", "admin")


def _generate_order_no() -> str:
    return f"AR{datetime.now().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(3).upper()}"


def _generate_transaction_no() -> str:
    return f"TX{datetime.now().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(4).upper()}"


def _serialize_plan(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "plan_code": row["plan_code"],
        "plan_name": row["plan_name"],
        "role_granted": row["role_granted"],
        "billing_cycle": row["billing_cycle"],
        "amount_cents": int(row["amount_cents"]),
        "currency": row["currency"],
        "description": row["description"] or "",
        "is_active": bool(row["is_active"]),
        "sort_order": int(row["sort_order"] or 0),
    }


def _serialize_subscription(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "plan_id": int(row["plan_id"]),
        "plan_code": row["plan_code"],
        "plan_name": row["plan_name"],
        "role_granted": row["role_granted"],
        "subscription_status": row["subscription_status"],
        "started_at": str(row["started_at"]) if row.get("started_at") else None,
        "ends_at": str(row["ends_at"]) if row.get("ends_at") else None,
        "revoked_at": str(row["revoked_at"]) if row.get("revoked_at") else None,
        "activated_by_order_id": (
            int(row["activated_by_order_id"]) if row.get("activated_by_order_id") is not None else None
        ),
    }


def _serialize_order(
    row: dict[str, Any],
    *,
    session_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {}
    if session_row and session_row.get("session_payload_json"):
        try:
            payload = json.loads(session_row["session_payload_json"])
        except json.JSONDecodeError:
            payload = {}

    instructions = ""
    if isinstance(payload, dict):
        instructions = str(payload.get("instructions") or "")
    missing_config = payload.get("missing_config") if isinstance(payload, dict) else []
    if not isinstance(missing_config, list):
        missing_config = []
    public_config = payload.get("public_config") if isinstance(payload, dict) else {}
    if not isinstance(public_config, dict):
        public_config = {}

    return {
        "id": int(row["id"]),
        "order_no": row["order_no"],
        "user_id": int(row["user_id"]),
        "plan_code": row["plan_code"],
        "plan_name": row["plan_name_snapshot"],
        "target_role": row["target_role"],
        "amount_cents": int(row["amount_cents"]),
        "currency": row["currency"],
        "payment_method": row["payment_method"],
        "order_status": row["order_status"],
        "provider_status": row["provider_status"],
        "checkout_action": row["checkout_action"],
        "checkout_url": row["checkout_url"],
        "provider_reference": session_row.get("provider_reference") if session_row else None,
        "session_status": session_row.get("session_status") if session_row else None,
        "redirect_url": session_row.get("redirect_url") if session_row else None,
        "qr_code_text": session_row.get("qr_code_text") if session_row else None,
        "qr_code_url": payload.get("qr_code_url") if isinstance(payload, dict) else None,
        "provider_code": payload.get("provider_code") if isinstance(payload, dict) else row["payment_method"],
        "provider_name": payload.get("provider_name") if isinstance(payload, dict) else row["payment_method"],
        "display_mode": payload.get("display_mode") if isinstance(payload, dict) else row["checkout_action"],
        "integration_status": (
            payload.get("integration_status") if isinstance(payload, dict) else row["provider_status"]
        ),
        "missing_config": [str(item) for item in missing_config if item],
        "script_url": payload.get("script_url") if isinstance(payload, dict) else None,
        "public_config": public_config,
        "notify_url": payload.get("notify_url") if isinstance(payload, dict) else None,
        "return_url": payload.get("return_url") if isinstance(payload, dict) else None,
        "next_action_path": payload.get("next_action_path") if isinstance(payload, dict) else None,
        "instructions": instructions,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "paid_at": str(row["paid_at"]) if row.get("paid_at") else None,
    }


def list_payment_methods() -> list[dict[str, Any]]:
    return [
        {
            "code": method,
            "provider_code": profile["provider_code"],
            "provider_name": profile["provider_name"],
            "display_mode": profile["display_mode"],
            "integration_status": profile["integration_status"],
            "missing_config": profile["missing_config"],
            "is_configured": profile["integration_status"] == "ready",
            "script_url": profile["script_url"],
            "public_config": profile["public_config"],
        }
        for method in SUPPORTED_PAYMENT_METHODS
        for profile in [payment_provider_profile(method)]
    ]


def _build_checkout_payload(
    *,
    payment_method: str,
    order_id: int,
    order_no: str,
    plan_name: str,
    amount_cents: int,
    currency: str,
    user_email: str | None = None,
) -> dict[str, Any]:
    return build_checkout_payload(
        payment_method=payment_method,
        order_id=order_id,
        order_no=order_no,
        plan_name=plan_name,
        amount_cents=amount_cents,
        currency=currency,
        user_email=user_email,
    )


def _get_plan_by_code(cursor, plan_code: str) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            id,
            plan_code,
            plan_name,
            role_granted,
            billing_cycle,
            amount_cents,
            currency,
            description,
            is_active,
            sort_order
        FROM subscription_plans
        WHERE plan_code = %s AND is_active = 1
        """,
        (plan_code,),
    )
    return cursor.fetchone()


def _fetch_latest_provider_session(cursor, order_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            id,
            order_id,
            provider,
            session_status,
            provider_session_id,
            provider_reference,
            redirect_url,
            qr_code_text,
            session_payload_json
        FROM payment_provider_sessions
        WHERE order_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (order_id,),
    )
    return cursor.fetchone()


def _fetch_order(cursor, order_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            id,
            order_no,
            user_id,
            plan_code,
            plan_name_snapshot,
            target_role,
            amount_cents,
            currency,
            payment_method,
            order_status,
            provider_status,
            checkout_action,
            checkout_url,
            created_at,
            updated_at,
            paid_at
        FROM payment_orders
        WHERE id = %s
        """,
        (order_id,),
    )
    return cursor.fetchone()


def _fetch_order_for_update(cursor, order_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            id,
            order_no,
            user_id,
            plan_id,
            plan_code,
            plan_name_snapshot,
            target_role,
            amount_cents,
            currency,
            payment_method,
            order_status,
            provider_status,
            checkout_action,
            checkout_url,
            created_at,
            updated_at,
            paid_at
        FROM payment_orders
        WHERE id = %s
        FOR UPDATE
        """,
        (order_id,),
    )
    return cursor.fetchone()


def _fetch_plan_by_id(cursor, plan_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            id,
            plan_code,
            plan_name,
            role_granted,
            billing_cycle,
            amount_cents,
            currency,
            description,
            is_active,
            sort_order
        FROM subscription_plans
        WHERE id = %s
        """,
        (plan_id,),
    )
    return cursor.fetchone()


def _fetch_active_subscription_for_user(cursor, user_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            us.id,
            us.user_id,
            us.plan_id,
            us.plan_code,
            sp.plan_name,
            us.role_granted,
            us.subscription_status,
            us.starts_at AS started_at,
            us.ends_at,
            us.revoked_at,
            us.activated_by_order_id
        FROM user_subscriptions us
        INNER JOIN subscription_plans sp ON sp.id = us.plan_id
        WHERE us.user_id = %s AND us.subscription_status = 'active'
        ORDER BY us.id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    return cursor.fetchone()


def list_subscription_plans(*, active_only: bool = True) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    id,
                    plan_code,
                    plan_name,
                    role_granted,
                    billing_cycle,
                    amount_cents,
                    currency,
                    description,
                    is_active,
                    sort_order
                FROM subscription_plans
                {"WHERE is_active = 1" if active_only else ""}
                ORDER BY sort_order ASC, amount_cents ASC, id ASC
                """
            )
            rows = cursor.fetchall() or []
    return [_serialize_plan(row) for row in rows]


def list_payment_orders_for_user(user_id: int, *, limit: int = 20) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    order_no,
                    user_id,
                    plan_code,
                    plan_name_snapshot,
                    target_role,
                    amount_cents,
                    currency,
                    payment_method,
                    order_status,
                    provider_status,
                    checkout_action,
                    checkout_url,
                    created_at,
                    updated_at,
                    paid_at
                FROM payment_orders
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall() or []
            session_rows: dict[int, dict[str, Any]] = {}
            if rows:
                cursor.execute(
                    f"""
                    SELECT
                        order_id,
                        session_status,
                        provider_reference,
                        redirect_url,
                        qr_code_text,
                        session_payload_json
                    FROM payment_provider_sessions
                    WHERE order_id IN ({", ".join(["%s"] * len(rows))})
                    ORDER BY id DESC
                    """,
                    [int(row["id"]) for row in rows],
                )
                provider_rows = cursor.fetchall() or []
                for provider_row in provider_rows:
                    order_id = int(provider_row["order_id"])
                    session_rows.setdefault(order_id, provider_row)

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_orders,
                    SUM(CASE WHEN order_status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
                    COALESCE(SUM(CASE WHEN order_status = 'paid' THEN amount_cents ELSE 0 END), 0) AS paid_amount_cents
                FROM payment_orders
                WHERE user_id = %s
                """,
                (user_id,),
            )
            summary_row = cursor.fetchone() or {}
            active_subscription = _fetch_active_subscription_for_user(cursor, user_id)

    return {
        "orders": [_serialize_order(row, session_row=session_rows.get(int(row["id"]))) for row in rows],
        "summary": {
            "total_orders": int(summary_row.get("total_orders") or 0),
            "paid_orders": int(summary_row.get("paid_orders") or 0),
            "paid_amount_cents": int(summary_row.get("paid_amount_cents") or 0),
        },
        "current_subscription": _serialize_subscription(active_subscription),
    }


def get_billing_summary_for_user(user_id: int) -> dict[str, Any]:
    orders_payload = list_payment_orders_for_user(user_id)
    return {
        "payment_environment": payment_environment(),
        "plans": list_subscription_plans(active_only=True),
        "payment_methods": list_payment_methods(),
        "orders": orders_payload["orders"],
        "order_summary": orders_payload["summary"],
        "current_subscription": orders_payload["current_subscription"],
    }


def _record_payment_event(
    cursor,
    *,
    order_id: int,
    event_type: str,
    actor_user_id: int | None,
    actor_role: str,
    payload: dict[str, Any] | None = None,
) -> None:
    cursor.execute(
        """
        INSERT INTO payment_events (
            order_id,
            event_type,
            actor_user_id,
            actor_role,
            event_payload_json
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            order_id,
            event_type,
            actor_user_id,
            actor_role,
            json.dumps(payload or {}, ensure_ascii=False),
        ),
    )


def _apply_user_role_change(
    cursor,
    *,
    user_id: int,
    new_role: str,
    grant_source: str,
    granted_by_user_id: int | None,
    payment_order_id: int | None = None,
    note: str | None = None,
) -> str:
    if new_role not in SUPPORTED_USER_ROLES:
        raise ValueError("Unsupported role.")

    cursor.execute(
        """
        SELECT id, role
        FROM users
        WHERE id = %s
        FOR UPDATE
        """,
        (user_id,),
    )
    user_row = cursor.fetchone()
    if user_row is None:
        raise ValueError("User was not found.")

    previous_role = str(user_row.get("role") or "basic")
    effective_role = new_role

    if previous_role == "admin" and new_role == "advanced" and grant_source.startswith("payment"):
        effective_role = "admin"

    if previous_role == "admin" and effective_role != "admin":
        cursor.execute("SELECT COUNT(*) AS total_admins FROM users WHERE role = 'admin'")
        admin_total = int((cursor.fetchone() or {}).get("total_admins") or 0)
        if admin_total <= 1:
            raise ValueError("At least one admin user must remain.")

    if previous_role != effective_role:
        cursor.execute(
            """
            UPDATE users
            SET role = %s
            WHERE id = %s
            """,
            (effective_role, user_id),
        )

    if effective_role == "basic":
        cursor.execute(
            """
            UPDATE user_subscriptions
            SET subscription_status = 'revoked', revoked_at = NOW()
            WHERE user_id = %s AND subscription_status = 'active'
            """,
            (user_id,),
        )

    cursor.execute(
        """
        INSERT INTO user_role_grants (
            user_id,
            granted_by_user_id,
            previous_role,
            new_role,
            grant_source,
            payment_order_id,
            note
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            granted_by_user_id,
            previous_role,
            effective_role,
            grant_source,
            payment_order_id,
            note,
        ),
    )

    return effective_role


def _activate_subscription(
    cursor,
    *,
    user_id: int,
    plan_row: dict[str, Any],
    order_id: int,
) -> dict[str, Any]:
    cursor.execute(
        """
        UPDATE user_subscriptions
        SET subscription_status = 'superseded', revoked_at = NOW()
        WHERE user_id = %s AND plan_code = %s AND subscription_status = 'active'
        """,
        (user_id, plan_row["plan_code"]),
    )
    cursor.execute(
        """
        INSERT INTO user_subscriptions (
            user_id,
            plan_id,
            plan_code,
            role_granted,
            subscription_status,
            starts_at,
            ends_at,
            activated_by_order_id
        )
        VALUES (%s, %s, %s, %s, 'active', NOW(), NULL, %s)
        """,
        (
            user_id,
            int(plan_row["id"]),
            plan_row["plan_code"],
            plan_row["role_granted"],
            order_id,
        ),
    )
    subscription_id = int(cursor.lastrowid)
    cursor.execute(
        """
        SELECT
            us.id,
            us.user_id,
            us.plan_id,
            us.plan_code,
            sp.plan_name,
            us.role_granted,
            us.subscription_status,
            us.starts_at AS started_at,
            us.ends_at,
            us.revoked_at,
            us.activated_by_order_id
        FROM user_subscriptions us
        INNER JOIN subscription_plans sp ON sp.id = us.plan_id
        WHERE us.id = %s
        """,
        (subscription_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Subscription activation failed.")
    return _serialize_subscription(row) or {}


def create_payment_order_for_user(
    *,
    user_id: int,
    plan_code: str,
    payment_method: str,
) -> dict[str, Any]:
    if payment_method not in SUPPORTED_PAYMENT_METHODS:
        raise ValueError("Unsupported payment method.")

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            plan_row = _get_plan_by_code(cursor, plan_code)
            if plan_row is None:
                raise ValueError("Subscription plan was not found.")

            user = get_user_by_id(user_id)
            if user is None:
                raise ValueError("User was not found.")

            order_no = _generate_order_no()
            profile = payment_provider_profile(payment_method, currency=str(plan_row["currency"]))

            cursor.execute(
                """
                INSERT INTO payment_orders (
                    order_no,
                    user_id,
                    plan_id,
                    plan_code,
                    plan_name_snapshot,
                    target_role,
                    amount_cents,
                    currency,
                    payment_method,
                    order_status,
                    provider_status,
                    checkout_action,
                    checkout_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'created', %s, %s)
                """,
                (
                    order_no,
                    user_id,
                    int(plan_row["id"]),
                    plan_row["plan_code"],
                    plan_row["plan_name"],
                    plan_row["role_granted"],
                    int(plan_row["amount_cents"]),
                    plan_row["currency"],
                    payment_method,
                    profile["display_mode"],
                    None,
                ),
            )
            order_id = int(cursor.lastrowid)
            checkout_payload = _build_checkout_payload(
                payment_method=payment_method,
                order_id=order_id,
                order_no=order_no,
                plan_name=str(plan_row["plan_name"]),
                amount_cents=int(plan_row["amount_cents"]),
                currency=str(plan_row["currency"]),
                user_email=str(user["email"]),
            )

            cursor.execute(
                """
                INSERT INTO payment_provider_sessions (
                    order_id,
                    provider,
                    session_status,
                    provider_session_id,
                    provider_reference,
                    redirect_url,
                    qr_code_text,
                    session_payload_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    order_id,
                    payment_method,
                    "pending_provider_init" if checkout_payload["integration_status"] == "ready" else "missing_config",
                    f"{payment_method}-{order_no.lower()}",
                    None,
                    checkout_payload["checkout_url"],
                    checkout_payload["qr_code_text"],
                    json.dumps(checkout_payload, ensure_ascii=False),
                ),
            )

            _record_payment_event(
                cursor,
                order_id=order_id,
                event_type="order_created",
                actor_user_id=user_id,
                actor_role="user",
                payload={
                    "plan_code": plan_row["plan_code"],
                    "payment_method": payment_method,
                    "payment_environment": payment_environment(),
                },
            )
        connection.commit()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            order_row = _fetch_order(cursor, order_id)
            session_row = _fetch_latest_provider_session(cursor, order_id)
    if order_row is None:
        raise RuntimeError("Payment order could not be reloaded.")
    return _serialize_order(order_row, session_row=session_row)


def get_payment_order_for_user(*, user_id: int, order_id: int) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            order_row = _fetch_order(cursor, order_id)
            if order_row is None:
                raise ValueError("Payment order was not found.")
            if int(order_row["user_id"]) != user_id:
                raise PermissionError("This order does not belong to the current user.")
            session_row = _fetch_latest_provider_session(cursor, order_id)
    return _serialize_order(order_row, session_row=session_row)


def get_payment_order_session_for_user(*, user_id: int, order_id: int) -> dict[str, Any]:
    order = get_payment_order_for_user(user_id=user_id, order_id=order_id)
    return {
        "order": order,
        "session": {
            "provider_code": order.get("provider_code"),
            "provider_name": order.get("provider_name"),
            "display_mode": order.get("display_mode"),
            "integration_status": order.get("integration_status"),
            "missing_config": order.get("missing_config") or [],
            "script_url": order.get("script_url"),
            "public_config": order.get("public_config") or {},
            "notify_url": order.get("notify_url"),
            "return_url": order.get("return_url"),
            "next_action_path": order.get("next_action_path"),
            "instructions": order.get("instructions"),
            "qr_code_url": order.get("qr_code_url"),
            "qr_code_text": order.get("qr_code_text"),
        },
    }


def _mark_order_paid(
    cursor,
    *,
    order_row: dict[str, Any],
    actor_user_id: int | None,
    actor_role: str,
    provider_reference: str,
    note: str | None = None,
) -> dict[str, Any]:
    if order_row["order_status"] == "paid":
        plan_row = _fetch_plan_by_id(cursor, int(order_row["plan_id"]))
        if plan_row is None:
            raise RuntimeError("Plan for order was not found.")
        subscription_row = _fetch_active_subscription_for_user(cursor, int(order_row["user_id"]))
        return {
            "effective_role": get_user_by_id(int(order_row["user_id"]))["role"],
            "subscription": _serialize_subscription(subscription_row),
            "plan_row": plan_row,
        }

    if order_row["order_status"] in {"cancelled", "rejected", "failed"}:
        raise ValueError("This order cannot be completed.")

    plan_row = _fetch_plan_by_id(cursor, int(order_row["plan_id"]))
    if plan_row is None:
        raise RuntimeError("Plan for order was not found.")

    cursor.execute(
        """
        UPDATE payment_orders
        SET order_status = 'paid',
            provider_status = %s,
            paid_at = NOW()
        WHERE id = %s
        """,
        (provider_reference, int(order_row["id"])),
    )
    cursor.execute(
        """
        UPDATE payment_provider_sessions
        SET session_status = 'completed',
            provider_reference = %s
        WHERE order_id = %s
        """,
        (provider_reference, int(order_row["id"])),
    )
    cursor.execute(
        """
        INSERT INTO payment_transactions (
            order_id,
            transaction_no,
            provider,
            payment_method,
            transaction_status,
            amount_cents,
            currency,
            provider_reference,
            raw_payload_json,
            paid_at
        )
        VALUES (%s, %s, %s, %s, 'paid', %s, %s, %s, %s, NOW())
        """,
        (
            int(order_row["id"]),
            _generate_transaction_no(),
            order_row["payment_method"],
            order_row["payment_method"],
            int(order_row["amount_cents"]),
            order_row["currency"],
            provider_reference,
            json.dumps({"note": note or "", "provider_reference": provider_reference}, ensure_ascii=False),
        ),
    )

    subscription = _activate_subscription(
        cursor,
        user_id=int(order_row["user_id"]),
        plan_row=plan_row,
        order_id=int(order_row["id"]),
    )
    effective_role = _apply_user_role_change(
        cursor,
        user_id=int(order_row["user_id"]),
        new_role=str(plan_row["role_granted"]),
        grant_source="payment_order",
        granted_by_user_id=actor_user_id,
        payment_order_id=int(order_row["id"]),
        note=note,
    )
    _record_payment_event(
        cursor,
        order_id=int(order_row["id"]),
        event_type="payment_completed",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={
            "provider_reference": provider_reference,
            "effective_role": effective_role,
            "note": note or "",
        },
    )
    return {
        "effective_role": effective_role,
        "subscription": subscription,
        "plan_row": plan_row,
    }


def complete_payment_order_in_sandbox(*, user_id: int, order_id: int) -> dict[str, Any]:
    raise ValueError("Sandbox payment flow has been disabled. Configure a real payment provider instead.")


def update_user_role_by_admin(
    *,
    target_user_id: int,
    new_role: str,
    admin_user_id: int,
    note: str | None = None,
) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            _apply_user_role_change(
                cursor,
                user_id=target_user_id,
                new_role=new_role,
                grant_source="admin_manual",
                granted_by_user_id=admin_user_id,
                payment_order_id=None,
                note=note,
            )
        connection.commit()

    user = get_user_by_id(target_user_id)
    if user is None:
        raise RuntimeError("Updated user could not be reloaded.")
    return user


def list_payment_orders_for_admin(
    *,
    page: int = 1,
    page_size: int = 25,
    status: str = "",
    payment_method: str = "",
    query: str = "",
) -> dict[str, Any]:
    safe_page = max(1, page)
    safe_page_size = max(1, min(100, page_size))
    offset = (safe_page - 1) * safe_page_size

    where_clauses = ["1 = 1"]
    params: list[Any] = []

    if status.strip():
        where_clauses.append("po.order_status = %s")
        params.append(status.strip())
    if payment_method.strip():
        where_clauses.append("po.payment_method = %s")
        params.append(payment_method.strip())
    if query.strip():
        like = f"%{query.strip()}%"
        where_clauses.append(
            """
            (
                po.order_no LIKE %s
                OR u.email LIKE %s
                OR u.display_name LIKE %s
                OR po.plan_name_snapshot LIKE %s
            )
            """
        )
        params.extend([like, like, like, like])

    where_sql = " AND ".join(where_clauses)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM payment_orders po
                LEFT JOIN users u ON u.id = po.user_id
                WHERE {where_sql}
                """,
                params,
            )
            total = int((cursor.fetchone() or {}).get("total") or 0)

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_orders,
                    SUM(CASE WHEN order_status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,
                    SUM(CASE WHEN order_status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
                    COALESCE(SUM(CASE WHEN order_status = 'paid' THEN amount_cents ELSE 0 END), 0) AS paid_amount_cents
                FROM payment_orders
                """
            )
            summary_row = cursor.fetchone() or {}

            cursor.execute(
                f"""
                SELECT
                    po.id,
                    po.order_no,
                    po.user_id,
                    u.email AS user_email,
                    u.display_name AS user_display_name,
                    po.plan_code,
                    po.plan_name_snapshot,
                    po.target_role,
                    po.amount_cents,
                    po.currency,
                    po.payment_method,
                    po.order_status,
                    po.provider_status,
                    po.checkout_action,
                    po.checkout_url,
                    po.created_at,
                    po.updated_at,
                    po.paid_at
                FROM payment_orders po
                LEFT JOIN users u ON u.id = po.user_id
                WHERE {where_sql}
                ORDER BY po.id DESC
                LIMIT %s OFFSET %s
                """,
                [*params, safe_page_size, offset],
            )
            rows = cursor.fetchall() or []

            order_ids = [int(row["id"]) for row in rows]
            session_rows: dict[int, dict[str, Any]] = {}
            if order_ids:
                placeholders = ", ".join(["%s"] * len(order_ids))
                cursor.execute(
                    f"""
                    SELECT
                        order_id,
                        session_status,
                        provider_reference,
                        redirect_url,
                        qr_code_text,
                        session_payload_json
                    FROM payment_provider_sessions
                    WHERE order_id IN ({placeholders})
                    ORDER BY id DESC
                    """,
                    order_ids,
                )
                for provider_row in cursor.fetchall() or []:
                    order_id = int(provider_row["order_id"])
                    session_rows.setdefault(order_id, provider_row)

    items = []
    for row in rows:
        item = _serialize_order(row, session_row=session_rows.get(int(row["id"])))
        item["user_email"] = row["user_email"]
        item["user_display_name"] = row["user_display_name"]
        items.append(item)

    return {
        "items": items,
        "page": safe_page,
        "page_size": safe_page_size,
        "total": total,
        "summary": {
            "total_orders": int(summary_row.get("total_orders") or 0),
            "pending_orders": int(summary_row.get("pending_orders") or 0),
            "paid_orders": int(summary_row.get("paid_orders") or 0),
            "paid_amount_cents": int(summary_row.get("paid_amount_cents") or 0),
        },
    }


def approve_payment_order(
    *,
    order_id: int,
    admin_user_id: int,
    approve: bool,
    note: str | None = None,
) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            order_row = _fetch_order_for_update(cursor, order_id)
            if order_row is None:
                raise ValueError("Payment order was not found.")

            if approve:
                _mark_order_paid(
                    cursor,
                    order_row=order_row,
                    actor_user_id=admin_user_id,
                    actor_role="admin",
                    provider_reference="admin_approved",
                    note=note or "Approved by admin.",
                )
            else:
                if order_row["order_status"] == "paid":
                    raise ValueError("A paid order cannot be rejected.")
                cursor.execute(
                    """
                    UPDATE payment_orders
                    SET order_status = 'rejected',
                        provider_status = 'admin_rejected'
                    WHERE id = %s
                    """,
                    (order_id,),
                )
                cursor.execute(
                    """
                    UPDATE payment_provider_sessions
                    SET session_status = 'rejected',
                        provider_reference = 'admin_rejected'
                    WHERE order_id = %s
                    """,
                    (order_id,),
                )
                _record_payment_event(
                    cursor,
                    order_id=order_id,
                    event_type="payment_rejected",
                    actor_user_id=admin_user_id,
                    actor_role="admin",
                    payload={"note": note or ""},
                )
        connection.commit()

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            order = _fetch_order(cursor, order_id)
            session_row = _fetch_latest_provider_session(cursor, order_id)
    if order is None:
        raise RuntimeError("Updated payment order could not be reloaded.")
    return _serialize_order(order, session_row=session_row)
