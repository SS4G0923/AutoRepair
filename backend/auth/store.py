from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SQL_DIR = (REPO_ROOT / "SQL").resolve()
IGNORABLE_MIGRATION_ERROR_CODES = {1007, 1050, 1060, 1061}


def _mysql_settings(include_database: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
    }
    if include_database:
        settings["database"] = os.getenv("MYSQL_DATABASE", "AutoRepair")
    return settings


def get_db_connection() -> pymysql.connections.Connection:
    return pymysql.connect(**_mysql_settings(include_database=True))


def _get_server_connection() -> pymysql.connections.Connection:
    return pymysql.connect(**_mysql_settings(include_database=False))


def _iter_sql_files() -> list[Path]:
    sql_files = [path for path in SQL_DIR.glob("*.sql") if path.is_file()]
    return sorted(sql_files, key=lambda path: (0 if path.name == "auth_schema.sql" else 1, path.name))


def _split_sql_statements(sql_text: str) -> list[str]:
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]


def _bootstrap_admin_roles() -> None:
    raw_admin_emails = os.getenv("AUTOREPAIR_ADMIN_EMAILS", "")
    emails = [normalize_email(email) for email in raw_admin_emails.split(",") if email.strip()]
    if not emails:
        return

    placeholders = ", ".join(["%s"] * len(emails))
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE users
                SET role = 'admin'
                WHERE email IN ({placeholders})
                """,
                emails,
            )
        connection.commit()


def init_auth_db() -> None:
    with _get_server_connection() as connection:
        with connection.cursor() as cursor:
            for sql_path in _iter_sql_files():
                for statement in _split_sql_statements(sql_path.read_text(encoding="utf-8")):
                    try:
                        cursor.execute(statement)
                    except pymysql.MySQLError as exc:
                        code = int(exc.args[0]) if exc.args else 0
                        if code in IGNORABLE_MIGRATION_ERROR_CODES:
                            continue
                        raise
        connection.commit()
    _bootstrap_admin_roles()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _serialize_user(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"],
        "avatar_url": row["avatar_url"],
        "auth_source": row["auth_source"],
        "role": row.get("role", "basic"),
        "account_status": row.get("account_status", "active"),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]) if row.get("updated_at") is not None else None,
        "last_login_at": str(row["last_login_at"]) if row.get("last_login_at") is not None else None,
    }


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    email,
                    display_name,
                    avatar_url,
                    auth_source,
                    role,
                    account_status,
                    created_at,
                    updated_at,
                    last_login_at
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
    return _serialize_user(row)


def get_user_with_password(email: str) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    email,
                    display_name,
                    avatar_url,
                    auth_source,
                    role,
                    account_status,
                    created_at,
                    updated_at,
                    last_login_at,
                    password_hash
                FROM users
                WHERE email = %s
                """,
                (normalize_email(email),),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    row["created_at"] = str(row["created_at"])
    return row


def touch_user_last_login(user_id: int) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET last_login_at = NOW()
                WHERE id = %s
                """,
                (user_id,),
            )
        connection.commit()


def record_login_event(
    *,
    user_id: int | None,
    email_attempt: str | None,
    login_method: str,
    login_status: str,
    ip_address: str | None,
    user_agent: str | None,
    failure_reason: str | None = None,
) -> None:
    normalized_email = normalize_email(email_attempt) if email_attempt else None
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_login_events (
                    user_id,
                    email_attempt,
                    login_method,
                    login_status,
                    failure_reason,
                    ip_address,
                    user_agent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    normalized_email,
                    login_method,
                    login_status,
                    failure_reason,
                    ip_address,
                    (user_agent or "")[:255] or None,
                ),
            )
        connection.commit()


def create_local_user(*, email: str, display_name: str, password_hash: str) -> dict[str, Any]:
    normalized_email = normalize_email(email)
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (email, display_name, password_hash, auth_source)
                VALUES (%s, %s, %s, 'local')
                """,
                (normalized_email, display_name.strip(), password_hash),
            )
            user_id = int(cursor.lastrowid)
        connection.commit()
    user = get_user_by_id(user_id)
    if user is None:
        raise RuntimeError("User creation succeeded but user could not be reloaded.")
    return user


def upsert_oauth_user(
    *,
    provider: str,
    provider_user_id: str,
    email: str,
    display_name: str,
    avatar_url: str | None,
) -> dict[str, Any]:
    normalized_email = normalize_email(email)
    clean_display_name = display_name.strip() or normalized_email.split("@", 1)[0]
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id
                FROM oauth_accounts
                WHERE provider = %s AND provider_user_id = %s
                """,
                (provider, provider_user_id),
            )
            existing_link = cursor.fetchone()

            if existing_link is not None:
                user_id = int(existing_link["user_id"])
                cursor.execute(
                    """
                    UPDATE users
                    SET email = %s, display_name = %s, avatar_url = %s
                    WHERE id = %s
                    """,
                    (normalized_email, clean_display_name, avatar_url, user_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT id
                    FROM users
                    WHERE email = %s
                    """,
                    (normalized_email,),
                )
                existing_user = cursor.fetchone()
                if existing_user is None:
                    cursor.execute(
                        """
                        INSERT INTO users (email, display_name, avatar_url, auth_source)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (normalized_email, clean_display_name, avatar_url, provider),
                    )
                    user_id = int(cursor.lastrowid)
                else:
                    user_id = int(existing_user["id"])
                    cursor.execute(
                        """
                        UPDATE users
                        SET display_name = %s, avatar_url = %s
                        WHERE id = %s
                        """,
                        (clean_display_name, avatar_url, user_id),
                    )

                cursor.execute(
                    """
                    INSERT INTO oauth_accounts (user_id, provider, provider_user_id, email)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        email = VALUES(email),
                        user_id = VALUES(user_id)
                    """,
                    (user_id, provider, provider_user_id, normalized_email),
                )

        connection.commit()

    user = get_user_by_id(user_id)
    if user is None:
        raise RuntimeError("OAuth user could not be reloaded after upsert.")
    return user
