from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = (BASE_DIR / "SQL" / "auth_schema.sql").resolve()


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


def init_auth_db() -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = [statement.strip() for statement in schema_sql.split(";") if statement.strip()]
    with _get_server_connection() as connection:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        connection.commit()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _serialize_user(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "avatar_url": row["avatar_url"],
        "auth_source": row["auth_source"],
        "created_at": str(row["created_at"]),
    }


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email, display_name, avatar_url, auth_source, created_at
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
                SELECT id, email, display_name, avatar_url, auth_source, created_at, password_hash
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
