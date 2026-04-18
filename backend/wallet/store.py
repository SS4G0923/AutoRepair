from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.auth.store import get_db_connection


class InsufficientCreditsError(ValueError):
    """Raised when a user tries to spend more credits than they hold."""

    def __init__(self, *, required: int, balance: int) -> None:
        super().__init__(
            f"Insufficient credits: need {required} but wallet has {balance}."
        )
        self.required = required
        self.balance = balance


def _serialize_wallet(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "user_id": int(row["user_id"]),
        "balance_credits": int(row["balance_credits"] or 0),
        "lifetime_earned": int(row["lifetime_earned"] or 0),
        "lifetime_spent": int(row["lifetime_spent"] or 0),
        "last_grant_at": str(row["last_grant_at"]) if row.get("last_grant_at") else None,
        "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
    }


def _serialize_transaction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "change_credits": int(row["change_credits"]),
        "balance_after": int(row["balance_after"]),
        "reason_code": row["reason_code"],
        "reference_type": row.get("reference_type"),
        "reference_id": int(row["reference_id"]) if row.get("reference_id") is not None else None,
        "note": row.get("note"),
        "actor_user_id": int(row["actor_user_id"]) if row.get("actor_user_id") is not None else None,
        "created_at": str(row["created_at"]),
    }


def _serialize_pricing(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "role_code": row["role_code"],
        "monthly_free_credits": int(row["monthly_free_credits"] or 0),
        "cost_per_chat": int(row["cost_per_chat"] or 0),
        "cost_per_repair": int(row["cost_per_repair"] or 0),
        "cost_per_benchmark_run": int(row["cost_per_benchmark_run"] or 0),
        "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
    }


def ensure_wallet_exists(user_id: int) -> dict[str, Any]:
    """Create a wallet row for the user with their welcome credits if missing."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT user_id, balance_credits, lifetime_earned, lifetime_spent, last_grant_at, updated_at "
                "FROM credit_wallets WHERE user_id = %s",
                (user_id,),
            )
            wallet = cursor.fetchone()
            if wallet is not None:
                return _serialize_wallet(wallet)  # type: ignore[return-value]

            cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
            user_row = cursor.fetchone()
            role = (user_row or {}).get("role", "basic") if user_row else "basic"

            cursor.execute(
                "SELECT monthly_free_credits FROM credit_pricing_rules WHERE role_code = %s",
                (role,),
            )
            rule = cursor.fetchone()
            welcome = int((rule or {}).get("monthly_free_credits") or 0)

            cursor.execute(
                """
                INSERT INTO credit_wallets (user_id, balance_credits, lifetime_earned, last_grant_at)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, welcome, welcome, datetime.now(timezone.utc) if welcome > 0 else None),
            )
            if welcome > 0:
                cursor.execute(
                    """
                    INSERT INTO credit_transactions
                        (user_id, change_credits, balance_after, reason_code, note, actor_user_id)
                    VALUES (%s, %s, %s, %s, %s, NULL)
                    """,
                    (user_id, welcome, welcome, "welcome_grant", f"Welcome grant for role '{role}'."),
                )
        connection.commit()
    return get_wallet_snapshot(user_id)


def get_wallet_snapshot(user_id: int) -> dict[str, Any]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT user_id, balance_credits, lifetime_earned, lifetime_spent, last_grant_at, updated_at "
                "FROM credit_wallets WHERE user_id = %s",
                (user_id,),
            )
            wallet = cursor.fetchone()
            cursor.execute(
                """
                SELECT id, user_id, change_credits, balance_after, reason_code,
                       reference_type, reference_id, note, actor_user_id, created_at
                FROM credit_transactions
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT 30
                """,
                (user_id,),
            )
            txs = cursor.fetchall() or []

            cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
            user_row = cursor.fetchone() or {}
            role = user_row.get("role", "basic")
            cursor.execute(
                "SELECT role_code, monthly_free_credits, cost_per_chat, cost_per_repair, "
                "cost_per_benchmark_run, updated_at FROM credit_pricing_rules WHERE role_code = %s",
                (role,),
            )
            pricing_row = cursor.fetchone()

    serialized_wallet = _serialize_wallet(wallet) or {
        "user_id": user_id,
        "balance_credits": 0,
        "lifetime_earned": 0,
        "lifetime_spent": 0,
        "last_grant_at": None,
        "updated_at": None,
    }
    return {
        "wallet": serialized_wallet,
        "transactions": [_serialize_transaction(row) for row in txs],
        "pricing": _serialize_pricing(pricing_row) if pricing_row else None,
        "role": role,
    }


def get_balance(user_id: int) -> int:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT balance_credits FROM credit_wallets WHERE user_id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
    if row is None:
        return 0
    return int(row["balance_credits"] or 0)


def list_pricing_rules() -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT role_code, monthly_free_credits, cost_per_chat, cost_per_repair, "
                "cost_per_benchmark_run, updated_at FROM credit_pricing_rules ORDER BY role_code"
            )
            rows = cursor.fetchall() or []
    return [_serialize_pricing(row) for row in rows]


def update_pricing_rule(
    *,
    role_code: str,
    monthly_free_credits: int,
    cost_per_chat: int,
    cost_per_repair: int,
    cost_per_benchmark_run: int,
) -> dict[str, Any]:
    if role_code not in {"basic", "advanced", "admin"}:
        raise ValueError("role_code must be one of basic/advanced/admin")
    for label, value in (
        ("monthly_free_credits", monthly_free_credits),
        ("cost_per_chat", cost_per_chat),
        ("cost_per_repair", cost_per_repair),
        ("cost_per_benchmark_run", cost_per_benchmark_run),
    ):
        if value < 0 or value > 1_000_000:
            raise ValueError(f"{label} must be between 0 and 1_000_000")

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO credit_pricing_rules
                    (role_code, monthly_free_credits, cost_per_chat, cost_per_repair, cost_per_benchmark_run)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    monthly_free_credits = VALUES(monthly_free_credits),
                    cost_per_chat = VALUES(cost_per_chat),
                    cost_per_repair = VALUES(cost_per_repair),
                    cost_per_benchmark_run = VALUES(cost_per_benchmark_run)
                """,
                (role_code, monthly_free_credits, cost_per_chat, cost_per_repair, cost_per_benchmark_run),
            )
        connection.commit()

    for rule in list_pricing_rules():
        if rule["role_code"] == role_code:
            return rule
    raise RuntimeError("Pricing rule could not be loaded after update.")


def spend_credits(
    *,
    user_id: int,
    amount: int,
    reason_code: str,
    reference_type: str | None = None,
    reference_id: int | None = None,
    note: str | None = None,
    actor_user_id: int | None = None,
) -> dict[str, Any]:
    """Deduct credits atomically. Returns the resulting transaction row."""
    if amount < 0:
        raise ValueError("amount must be >= 0")
    ensure_wallet_exists(user_id)
    if amount == 0:
        return {
            "user_id": user_id,
            "change_credits": 0,
            "balance_after": get_balance(user_id),
            "reason_code": reason_code,
            "skipped": True,
        }

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT balance_credits FROM credit_wallets WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            row = cursor.fetchone()
            current = int((row or {}).get("balance_credits") or 0)
            if current < amount:
                connection.rollback()
                raise InsufficientCreditsError(required=amount, balance=current)
            new_balance = current - amount
            cursor.execute(
                "UPDATE credit_wallets SET balance_credits = %s, lifetime_spent = lifetime_spent + %s "
                "WHERE user_id = %s",
                (new_balance, amount, user_id),
            )
            cursor.execute(
                """
                INSERT INTO credit_transactions
                    (user_id, change_credits, balance_after, reason_code,
                     reference_type, reference_id, note, actor_user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    -amount,
                    new_balance,
                    reason_code,
                    reference_type,
                    reference_id,
                    note,
                    actor_user_id,
                ),
            )
            tx_id = int(cursor.lastrowid)
        connection.commit()
    return {
        "id": tx_id,
        "user_id": user_id,
        "change_credits": -amount,
        "balance_after": new_balance,
        "reason_code": reason_code,
        "reference_type": reference_type,
        "reference_id": reference_id,
        "note": note,
    }


def refund_credits(
    *,
    user_id: int,
    amount: int,
    reason_code: str,
    reference_type: str | None = None,
    reference_id: int | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if amount < 0:
        raise ValueError("amount must be >= 0")
    if amount == 0:
        return {"skipped": True, "balance_after": get_balance(user_id)}
    ensure_wallet_exists(user_id)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT balance_credits FROM credit_wallets WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            row = cursor.fetchone()
            current = int((row or {}).get("balance_credits") or 0)
            new_balance = current + amount
            cursor.execute(
                "UPDATE credit_wallets SET balance_credits = %s, lifetime_spent = GREATEST(0, lifetime_spent - %s) "
                "WHERE user_id = %s",
                (new_balance, amount, user_id),
            )
            cursor.execute(
                """
                INSERT INTO credit_transactions
                    (user_id, change_credits, balance_after, reason_code,
                     reference_type, reference_id, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, amount, new_balance, reason_code, reference_type, reference_id, note),
            )
        connection.commit()
    return {"balance_after": new_balance}


def grant_credits(
    *,
    user_id: int,
    amount: int,
    reason_code: str,
    actor_user_id: int | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if amount <= 0:
        raise ValueError("amount must be > 0")
    ensure_wallet_exists(user_id)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT balance_credits FROM credit_wallets WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            row = cursor.fetchone()
            current = int((row or {}).get("balance_credits") or 0)
            new_balance = current + amount
            cursor.execute(
                "UPDATE credit_wallets SET balance_credits = %s, lifetime_earned = lifetime_earned + %s, "
                "last_grant_at = NOW() WHERE user_id = %s",
                (new_balance, amount, user_id),
            )
            cursor.execute(
                """
                INSERT INTO credit_transactions
                    (user_id, change_credits, balance_after, reason_code, note, actor_user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, amount, new_balance, reason_code, note, actor_user_id),
            )
            tx_id = int(cursor.lastrowid)
        connection.commit()
    return {
        "id": tx_id,
        "user_id": user_id,
        "change_credits": amount,
        "balance_after": new_balance,
        "reason_code": reason_code,
        "note": note,
    }


def list_wallets_for_admin(*, limit: int = 200) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.id AS user_id,
                       u.email,
                       u.display_name,
                       u.role,
                       COALESCE(w.balance_credits, 0) AS balance_credits,
                       COALESCE(w.lifetime_earned, 0) AS lifetime_earned,
                       COALESCE(w.lifetime_spent, 0)  AS lifetime_spent,
                       w.last_grant_at
                FROM users u
                LEFT JOIN credit_wallets w ON w.user_id = u.id
                ORDER BY balance_credits DESC, u.id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall() or []
    return [
        {
            "user_id": int(r["user_id"]),
            "email": r["email"],
            "display_name": r["display_name"],
            "role": r["role"],
            "balance_credits": int(r["balance_credits"] or 0),
            "lifetime_earned": int(r["lifetime_earned"] or 0),
            "lifetime_spent": int(r["lifetime_spent"] or 0),
            "last_grant_at": str(r["last_grant_at"]) if r.get("last_grant_at") else None,
        }
        for r in rows
    ]


def pricing_cost_for_role(role: str, reason_code: str) -> int:
    """Look up cost for a standard action (returns 0 if rule missing)."""
    rules = {r["role_code"]: r for r in list_pricing_rules()}
    rule = rules.get(role) or rules.get("basic") or {}
    key = {
        "chat": "cost_per_chat",
        "repair": "cost_per_repair",
        "benchmark_run": "cost_per_benchmark_run",
    }.get(reason_code)
    if key is None:
        return 0
    return int(rule.get(key, 0) or 0)
