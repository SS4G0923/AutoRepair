from __future__ import annotations

import io
import json
import os
import queue
import secrets
import subprocess
import threading
import zipfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from flask import Flask, Response, jsonify, redirect, request, session, stream_with_context, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth.store import (
    create_local_user,
    get_db_connection,
    get_user_by_id,
    get_user_with_password,
    init_auth_db,
    normalize_email,
    record_login_event,
    touch_user_last_login,
    upsert_oauth_user,
)
from backend.admin.store import (
    get_admin_dashboard,
    get_llm_request_detail,
    get_model_usage_report,
    list_llm_requests_for_admin,
    list_login_events_for_admin,
    list_users_for_admin,
)
from backend.benchmark import store as bench_store
from backend.benchmark.runner import run_benchmark_in_background
from backend.pdf_export.exporter import (
    build_benchmark_report_pdf,
    build_repair_report_pdf,
)
from backend.profile.store import (
    create_api_token,
    get_preferences,
    get_profile_overview,
    list_api_tokens,
    revoke_api_token,
    update_preferences,
)
from backend.teams import store as team_store
from backend.wallet import store as wallet_store
from backend.wallet.store import InsufficientCreditsError
from backend.billing.store import (
    approve_payment_order,
    complete_payment_order_in_sandbox,
    create_payment_order_for_user,
    get_billing_summary_for_user,
    get_payment_order_session_for_user,
    list_payment_orders_for_admin,
    update_user_role_by_admin,
)
from backend.chat.pipeline import ChatRequest, run_chat_pipeline
from backend.history.store import (
    get_history_for_user,
    list_histories_for_user,
    save_history,
    soft_delete_history_for_user,
)
from backend.llm.store import (
    create_model_config,
    delete_model_config,
    list_model_configs_for_admin,
    list_public_model_catalog,
    update_model_config,
)
from backend.repair.languages import get_language_spec, normalize_language
from backend.repair.pipeline import RepairRequest, _apply_unified_diff_to_project, run_repair_pipeline
from backend.repair.workspace import (
    list_project_entrypoint_options,
    normalize_project_path,
    prepare_project_workspace,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from authlib.integrations.flask_client import OAuth
except ImportError:  # pragma: no cover
    OAuth = None


if load_dotenv is not None:
    load_dotenv()


def _parse_allowed_origins() -> set[str]:
    raw = os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    return {origin.strip() for origin in raw.split(",") if origin.strip()}


def _preferred_frontend_origin(allowed_origins: set[str]) -> str:
    configured = os.getenv("FRONTEND_ORIGIN")
    if configured:
        return configured
    return sorted(allowed_origins)[0] if allowed_origins else "http://127.0.0.1:5173"


ALLOWED_ORIGINS = _parse_allowed_origins()
PREFERRED_FRONTEND_ORIGIN = _preferred_frontend_origin(ALLOWED_ORIGINS)
GITHUB_TOKEN_SESSION_KEY = "github_oauth_token"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"

init_auth_db()

oauth = OAuth(app) if OAuth is not None else None
if oauth is not None:
    github_client_id = os.getenv("GITHUB_CLIENT_ID")
    github_client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if github_client_id and github_client_secret:
        oauth.register(
            name="github",
            client_id=github_client_id,
            client_secret=github_client_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email repo"},
        )

    google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if google_client_id and google_client_secret:
        oauth.register(
            name="google",
            client_id=google_client_id,
            client_secret=google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )


def _get_request_origin() -> str | None:
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        return origin
    return None


def _allowed_request_headers() -> str:
    requested = request.headers.get("Access-Control-Request-Headers", "").strip()
    if requested:
        return requested
    return "Content-Type, Authorization"


def _frontend_redirect_url(**params: str) -> str:
    if not params:
        return PREFERRED_FRONTEND_ORIGIN
    return f"{PREFERRED_FRONTEND_ORIGIN}?{urlencode(params)}"


def _current_user() -> dict[str, Any] | None:
    raw_user_id = session.get("user_id")
    if raw_user_id is None:
        return None
    try:
        user = get_user_by_id(int(raw_user_id))
    except (TypeError, ValueError):
        session.clear()
        return None
    if user is None or user.get("account_status") != "active":
        session.clear()
        return None
    return user


def _set_user_session(user: dict[str, Any]) -> None:
    session.clear()
    session["user_id"] = user["id"]


def _store_oauth_token(provider: str, token: dict[str, Any]) -> None:
    access_token = str(token.get("access_token") or "").strip()
    session_key = f"{provider}_oauth_token"
    if not access_token:
        session.pop(session_key, None)
        return
    session[session_key] = {
        "access_token": access_token,
        "token_type": str(token.get("token_type") or "").strip() or None,
        "scope": str(token.get("scope") or "").strip() or None,
    }


def _github_access_token_from_session() -> str | None:
    raw_token = session.get(GITHUB_TOKEN_SESSION_KEY)
    if isinstance(raw_token, dict):
        value = str(raw_token.get("access_token") or "").strip()
        return value or None
    if isinstance(raw_token, str):
        value = raw_token.strip()
        return value or None
    return None


def _request_ip() -> str | None:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for.strip():
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr


def _is_admin(user: dict[str, Any] | None) -> bool:
    return bool(user and user.get("role") == "admin")


def _finalize_login(user: dict[str, Any], *, login_method: str, email_attempt: str | None) -> dict[str, Any]:
    touch_user_last_login(int(user["id"]))
    refreshed_user = get_user_by_id(int(user["id"])) or user
    record_login_event(
        user_id=int(refreshed_user["id"]),
        email_attempt=email_attempt,
        login_method=login_method,
        login_status="success",
        ip_address=_request_ip(),
        user_agent=request.headers.get("User-Agent"),
    )
    _set_user_session(refreshed_user)
    return refreshed_user


def _enabled_oauth_providers() -> list[str]:
    if oauth is None:
        return []
    return [provider for provider in ("github", "google") if oauth.create_client(provider) is not None]


def _validate_registration_payload(payload: dict[str, Any]) -> tuple[str, str, str]:
    display_name = str(payload.get("display_name", "")).strip()
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))
    if len(display_name) < 2:
        raise ValueError("`display_name` must be at least 2 characters.")
    if "@" not in email or len(normalize_email(email)) < 5:
        raise ValueError("`email` must be a valid email address.")
    if len(password) < 8:
        raise ValueError("`password` must be at least 8 characters.")
    return display_name, email, password


def _validate_login_payload(payload: dict[str, Any]) -> tuple[str, str]:
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))
    if "@" not in email or not password:
        raise ValueError("`email` and `password` are required.")
    return email, password


def _validate_role_update_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    role = str(payload.get("role", "")).strip().lower()
    note = str(payload.get("note", "")).strip() or None
    if role not in {"basic", "advanced", "admin"}:
        raise ValueError("`role` must be one of: basic, advanced, admin.")
    return role, note


def _validate_payment_order_payload(payload: dict[str, Any]) -> tuple[str, str]:
    plan_code = str(payload.get("plan_code", "")).strip()
    payment_method = str(payload.get("payment_method", "")).strip().lower()
    if not plan_code:
        raise ValueError("`plan_code` is required.")
    if payment_method not in {"card", "paypal", "wechat", "alipay"}:
        raise ValueError("`payment_method` must be one of: card, paypal, wechat, alipay.")
    return plan_code, payment_method


def _normalize_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _validate_model_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    provider_code = str(payload.get("provider_code", "")).strip().lower()
    provider_name = str(payload.get("provider_name", "")).strip()
    vendor_name = str(payload.get("vendor_name", "")).strip() or provider_name
    model_key = str(payload.get("model_key", "")).strip()
    display_name = str(payload.get("display_name", "")).strip()
    api_model_name = str(payload.get("api_model_name", "")).strip()
    api_base_url = str(payload.get("api_base_url", "")).strip() or None
    api_key_env_var = str(payload.get("api_key_env_var", "")).strip() or None
    notes = str(payload.get("notes", "")).strip() or None
    extra_config = payload.get("extra_config")

    if provider_code not in {"openai_compatible", "gemini"}:
        raise ValueError("`provider_code` must be `openai_compatible` or `gemini`.")
    if not provider_name:
        raise ValueError("`provider_name` is required.")
    if not vendor_name:
        raise ValueError("`vendor_name` is required.")
    if not model_key:
        raise ValueError("`model_key` is required.")
    if not display_name:
        raise ValueError("`display_name` is required.")
    if not api_model_name:
        raise ValueError("`api_model_name` is required.")
    if len(model_key) > 128:
        raise ValueError("`model_key` must be at most 128 characters.")
    if len(display_name) > 120:
        raise ValueError("`display_name` must be at most 120 characters.")
    if len(provider_name) > 64 or len(vendor_name) > 64:
        raise ValueError("`provider_name` and `vendor_name` must be at most 64 characters.")
    if len(api_model_name) > 128:
        raise ValueError("`api_model_name` must be at most 128 characters.")
    if api_base_url is not None and len(api_base_url) > 512:
        raise ValueError("`api_base_url` must be at most 512 characters.")
    if api_key_env_var is not None and len(api_key_env_var) > 64:
        raise ValueError("`api_key_env_var` must be at most 64 characters.")
    if notes is not None and len(notes) > 2000:
        raise ValueError("`notes` must be at most 2000 characters.")

    if extra_config is None:
        normalized_extra_config: dict[str, Any] = {}
    elif isinstance(extra_config, dict):
        normalized_extra_config = extra_config
    else:
        raise ValueError("`extra_config` must be an object when provided.")

    sort_order = payload.get("sort_order", 0)
    if not isinstance(sort_order, int):
        raise ValueError("`sort_order` must be an integer.")

    return {
        "provider_code": provider_code,
        "provider_name": provider_name,
        "vendor_name": vendor_name,
        "model_key": model_key,
        "display_name": display_name,
        "api_model_name": api_model_name,
        "api_base_url": api_base_url,
        "api_key_env_var": api_key_env_var,
        "enabled": _normalize_bool(payload.get("enabled"), default=True),
        "is_default_chat": _normalize_bool(payload.get("is_default_chat"), default=False),
        "is_default_repair": _normalize_bool(payload.get("is_default_repair"), default=False),
        "supports_streaming": _normalize_bool(payload.get("supports_streaming"), default=True),
        "supports_json": _normalize_bool(payload.get("supports_json"), default=True),
        "sort_order": sort_order,
        "notes": notes,
        "extra_config": {
            **normalized_extra_config,
            "thinking_enabled": _normalize_bool(
                payload.get("thinking_enabled", normalized_extra_config.get("thinking_enabled")),
                default=False,
            ),
        },
    }


def _require_login(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if request.method == "OPTIONS":
            return Response(status=204)
        user = _current_user()
        if user is None:
            return jsonify({"error": "Authentication required."}), 401
        return view_func(*args, **kwargs)

    return wrapped


def _require_admin(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if request.method == "OPTIONS":
            return Response(status=204)
        user = _current_user()
        if user is None:
            return jsonify({"error": "Authentication required."}), 401
        if not _is_admin(user):
            return jsonify({"error": "Admin access required."}), 403
        return view_func(*args, **kwargs)

    return wrapped


@app.before_request
def handle_preflight() -> Response | None:
    if request.method == "OPTIONS":
        return Response(status=204)
    return None


@app.after_request
def add_cors_headers(response: Response) -> Response:
    origin = _get_request_origin()
    if origin is not None:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = _allowed_request_headers()
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Max-Age"] = "600"
    return response


def _format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _ui_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _event_summary(event: str, data: dict[str, Any]) -> str:
    if event == "stage":
        return f"{str(data.get('stage'))} · {str(data.get('status'))}"
    if event == "candidate_status":
        return (
            f"{str(data.get('stage'))} · "
            f"{str(data.get('candidate_label') or data.get('candidate_key') or 'candidate')} · "
            f"{str(data.get('status'))}"
        )
    if event == "stage_reasoning_chunk":
        return f"{str(data.get('stage'))} · reasoning"
    if event == "chat_reasoning_chunk":
        return "chat · reasoning"
    if event == "error":
        return "error"
    if event == "result":
        return "result"
    return json.dumps(data, ensure_ascii=False)[:140]


def _truncate_text(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"


def _agent_history_title(repair_request: RepairRequest, filename: str | None = None) -> str:
    return f"Repair · {filename or repair_request.filename or 'project'}"


def _chat_history_title(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return _truncate_text(message["content"], 80)
    return "Chat session"


def _empty_stage_map() -> dict[str, dict[str, Any]]:
    return {
        "run": {"status": "idle", "reasoning": "", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "inspect": {"status": "idle", "reasoning": "", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "plan": {"status": "idle", "reasoning": "", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "code": {"status": "idle", "reasoning": "", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "verify": {"status": "idle", "reasoning": "", "explain": "", "report": "", "diff": "", "toolEvents": []},
    }


def _build_auth_session_response() -> dict[str, Any]:
    user = _current_user()
    return {
        "authenticated": user is not None,
        "user": user,
        "oauth_providers": _enabled_oauth_providers(),
    }


def _get_oauth_client(provider: str):
    if oauth is None:
        return None
    return oauth.create_client(provider)


def _github_email(client, token: dict[str, Any]) -> str | None:
    emails_response = client.get("user/emails", token=token)
    emails_payload = emails_response.json()
    if not isinstance(emails_payload, list):
        return None
    primary_verified = next(
        (
            item.get("email")
            for item in emails_payload
            if isinstance(item, dict) and item.get("primary") and item.get("verified")
        ),
        None,
    )
    if primary_verified:
        return primary_verified
    fallback = next(
        (item.get("email") for item in emails_payload if isinstance(item, dict) and item.get("email")),
        None,
    )
    return fallback


def _oauth_user_payload(provider: str, client) -> tuple[dict[str, Any], str, str, str, str | None]:
    token = client.authorize_access_token()
    if not isinstance(token, dict):
        raise ValueError("OAuth provider did not return a usable token.")
    if provider == "google":
        user_info = token.get("userinfo")
        if not isinstance(user_info, dict):
            user_info = client.get("userinfo", token=token).json()
        provider_user_id = str(user_info.get("sub", "")).strip()
        email = str(user_info.get("email", "")).strip()
        display_name = str(user_info.get("name") or email.split("@", 1)[0]).strip()
        avatar_url = user_info.get("picture")
    elif provider == "github":
        profile = client.get("user", token=token).json()
        provider_user_id = str(profile.get("id", "")).strip()
        email = str(profile.get("email") or _github_email(client, token) or "").strip()
        display_name = str(profile.get("name") or profile.get("login") or email.split("@", 1)[0]).strip()
        avatar_url = profile.get("avatar_url")
    else:
        raise ValueError("Unsupported OAuth provider.")

    if not provider_user_id or not email:
        raise ValueError("OAuth provider did not return a usable email.")
    return token, provider_user_id, email, display_name, avatar_url


@app.get("/healthz")
def healthz() -> Response:
    return jsonify({"status": "ok"})


@app.route("/api/history", methods=["GET", "OPTIONS"])
@_require_login
def history_list() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    return jsonify({"items": list_histories_for_user(int(user["id"]))})


@app.route("/api/history/<int:history_id>", methods=["GET", "OPTIONS"])
@_require_login
def history_detail(history_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    history = get_history_for_user(int(user["id"]), history_id)
    if history is None:
        return jsonify({"error": "History record was not found."}), 404
    return jsonify(history)


@app.route("/api/history/<int:history_id>", methods=["DELETE", "OPTIONS"])
@_require_login
def history_delete(history_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    deleted = soft_delete_history_for_user(int(user["id"]), history_id)
    if not deleted:
        return jsonify({"error": "History record was not found."}), 404
    return jsonify({"ok": True, "history_id": history_id})


@app.route("/api/auth/session", methods=["GET", "OPTIONS"])
def auth_session() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return jsonify(_build_auth_session_response())


@app.route("/api/models", methods=["GET", "OPTIONS"])
def public_models() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return jsonify(list_public_model_catalog())


@app.route("/api/auth/register", methods=["POST", "OPTIONS"])
def register() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        display_name, email, password = _validate_registration_payload(payload)
        if get_user_with_password(email) is not None:
            return jsonify({"error": "An account with this email already exists."}), 409
        user = create_local_user(
            email=email,
            display_name=display_name,
            password_hash=generate_password_hash(password),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    session_user = _finalize_login(user, login_method="local_register", email_attempt=email)
    return jsonify({"user": session_user, "oauth_providers": _enabled_oauth_providers()})


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def login() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        email, password = _validate_login_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user = get_user_with_password(email)
    if user is not None and user.get("account_status") != "active":
        record_login_event(
            user_id=int(user["id"]),
            email_attempt=email,
            login_method="local_login",
            login_status="failed",
            ip_address=_request_ip(),
            user_agent=request.headers.get("User-Agent"),
            failure_reason="account_suspended",
        )
        return jsonify({"error": "Account is suspended."}), 403
    if user is None or not user.get("password_hash") or not check_password_hash(user["password_hash"], password):
        record_login_event(
            user_id=int(user["id"]) if user is not None else None,
            email_attempt=email,
            login_method="local_login",
            login_status="failed",
            ip_address=_request_ip(),
            user_agent=request.headers.get("User-Agent"),
            failure_reason="Invalid email or password.",
        )
        return jsonify({"error": "Invalid email or password."}), 401

    session_user = get_user_by_id(int(user["id"]))
    if session_user is None:
        return jsonify({"error": "User account could not be loaded."}), 500

    session_user = _finalize_login(session_user, login_method="local_login", email_attempt=email)
    return jsonify({"user": session_user, "oauth_providers": _enabled_oauth_providers()})


@app.route("/api/auth/logout", methods=["POST", "OPTIONS"])
def logout() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/auth/oauth/<provider>")
def oauth_start(provider: str) -> Response:
    client = _get_oauth_client(provider)
    if client is None:
        return redirect(_frontend_redirect_url(auth_error="provider_unavailable"))
    redirect_uri = url_for("oauth_callback", provider=provider, _external=True)
    return client.authorize_redirect(redirect_uri)


@app.get("/api/auth/oauth/<provider>/callback")
def oauth_callback(provider: str) -> Response:
    client = _get_oauth_client(provider)
    if client is None:
        return redirect(_frontend_redirect_url(auth_error="provider_unavailable"))

    try:
        token, provider_user_id, email, display_name, avatar_url = _oauth_user_payload(provider, client)
        user = upsert_oauth_user(
            provider=provider,
            provider_user_id=provider_user_id,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
        )
        if user.get("account_status") != "active":
            record_login_event(
                user_id=int(user["id"]),
                email_attempt=email,
                login_method=provider,
                login_status="failed",
                ip_address=_request_ip(),
                user_agent=request.headers.get("User-Agent"),
                failure_reason="account_suspended",
            )
            return redirect(_frontend_redirect_url(auth_error="account_suspended"))
        _finalize_login(user, login_method=provider, email_attempt=email)
        if provider == "github":
            _store_oauth_token(provider, token)
    except Exception:
        record_login_event(
            user_id=None,
            email_attempt=None,
            login_method=provider,
            login_status="failed",
            ip_address=_request_ip(),
            user_agent=request.headers.get("User-Agent"),
            failure_reason="oauth_callback_failed",
        )
        return redirect(_frontend_redirect_url(auth_error="oauth_callback_failed"))

    return redirect(_frontend_redirect_url(auth_success="1"))


@app.route("/api/admin/dashboard", methods=["GET", "OPTIONS"])
@_require_admin
def admin_dashboard() -> Response:
    return jsonify(get_admin_dashboard())


@app.route("/api/admin/users", methods=["GET", "OPTIONS"])
@_require_admin
def admin_users() -> Response:
    limit = request.args.get("limit", default=200, type=int) or 200
    return jsonify({"items": list_users_for_admin(limit=limit)})


@app.route("/api/admin/users/<int:user_id>/role", methods=["POST", "OPTIONS"])
@_require_admin
def admin_user_role_update(user_id: int) -> Response:
    admin_user = _current_user()
    if admin_user is None:
        return jsonify({"error": "Authentication required."}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        role, note = _validate_role_update_payload(payload)
        updated_user = update_user_role_by_admin(
            target_user_id=user_id,
            new_role=role,
            admin_user_id=int(admin_user["id"]),
            note=note,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    response_payload: dict[str, Any] = {"user": updated_user}
    if int(admin_user["id"]) == user_id:
        response_payload["current_user"] = updated_user
    return jsonify(response_payload)


@app.route("/api/admin/llm-requests", methods=["GET", "OPTIONS"])
@_require_admin
def admin_llm_requests() -> Response:
    page = request.args.get("page", default=1, type=int) or 1
    page_size = request.args.get("page_size", default=25, type=int) or 25
    query = request.args.get("q", default="", type=str) or ""
    model = request.args.get("model", default="", type=str) or ""
    status = request.args.get("status", default="", type=str) or ""
    request_mode = request.args.get("request_mode", default="", type=str) or ""
    return jsonify(
        list_llm_requests_for_admin(
            page=page,
            page_size=page_size,
            query=query,
            model=model,
            status=status,
            request_mode=request_mode,
        )
    )


@app.route("/api/admin/llm-requests/<int:request_id>", methods=["GET", "OPTIONS"])
@_require_admin
def admin_llm_request_detail(request_id: int) -> Response:
    detail = get_llm_request_detail(request_id)
    if detail is None:
        return jsonify({"error": "LLM request was not found."}), 404
    return jsonify(detail)


@app.route("/api/admin/model-usage", methods=["GET", "OPTIONS"])
@_require_admin
def admin_model_usage() -> Response:
    days = request.args.get("days", default=30, type=int) or 30
    return jsonify(get_model_usage_report(days=days))


@app.route("/api/admin/model-configs", methods=["GET", "OPTIONS"])
@_require_admin
def admin_model_configs() -> Response:
    return jsonify({"items": list_model_configs_for_admin()})


@app.route("/api/admin/model-configs", methods=["POST", "OPTIONS"])
@_require_admin
def admin_model_configs_create() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        item = create_model_config(**_validate_model_config_payload(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"item": item})


@app.route("/api/admin/model-configs/<int:model_config_id>", methods=["POST", "OPTIONS"])
@_require_admin
def admin_model_configs_update(model_config_id: int) -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        item = update_model_config(
            model_config_id=model_config_id,
            **_validate_model_config_payload(payload),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"item": item})


@app.route("/api/admin/model-configs/<int:model_config_id>", methods=["DELETE", "OPTIONS"])
@_require_admin
def admin_model_configs_delete(model_config_id: int) -> Response:
    try:
        delete_model_config(model_config_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "id": model_config_id})


@app.route("/api/admin/login-events", methods=["GET", "OPTIONS"])
@_require_admin
def admin_login_events() -> Response:
    page = request.args.get("page", default=1, type=int) or 1
    page_size = request.args.get("page_size", default=50, type=int) or 50
    return jsonify(list_login_events_for_admin(page=page, page_size=page_size))


@app.route("/api/admin/payment-orders", methods=["GET", "OPTIONS"])
@_require_admin
def admin_payment_orders() -> Response:
    page = request.args.get("page", default=1, type=int) or 1
    page_size = request.args.get("page_size", default=25, type=int) or 25
    status = request.args.get("status", default="", type=str) or ""
    payment_method = request.args.get("payment_method", default="", type=str) or ""
    query = request.args.get("q", default="", type=str) or ""
    return jsonify(
        list_payment_orders_for_admin(
            page=page,
            page_size=page_size,
            status=status,
            payment_method=payment_method,
            query=query,
        )
    )


@app.route("/api/admin/payment-orders/<int:order_id>/approve", methods=["POST", "OPTIONS"])
@_require_admin
def admin_payment_order_approve(order_id: int) -> Response:
    admin_user = _current_user()
    if admin_user is None:
        return jsonify({"error": "Authentication required."}), 401

    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    note = str(payload.get("note", "")).strip() or None
    try:
        order = approve_payment_order(
            order_id=order_id,
            admin_user_id=int(admin_user["id"]),
            approve=True,
            note=note,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"order": order})


@app.route("/api/admin/payment-orders/<int:order_id>/reject", methods=["POST", "OPTIONS"])
@_require_admin
def admin_payment_order_reject(order_id: int) -> Response:
    admin_user = _current_user()
    if admin_user is None:
        return jsonify({"error": "Authentication required."}), 401

    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    note = str(payload.get("note", "")).strip() or None
    try:
        order = approve_payment_order(
            order_id=order_id,
            admin_user_id=int(admin_user["id"]),
            approve=False,
            note=note,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"order": order})


@app.route("/api/billing/summary", methods=["GET", "OPTIONS"])
@_require_login
def billing_summary() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = get_billing_summary_for_user(int(user["id"]))
    payload["user"] = user
    return jsonify(payload)


@app.route("/api/billing/orders", methods=["POST", "OPTIONS"])
@_require_login
def billing_create_order() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if user.get("role") != "basic":
        return jsonify({"error": "Only basic users can create upgrade orders."}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        plan_code, payment_method = _validate_payment_order_payload(payload)
        order = create_payment_order_for_user(
            user_id=int(user["id"]),
            plan_code=plan_code,
            payment_method=payment_method,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"order": order})


@app.route("/api/billing/orders/<int:order_id>/session", methods=["GET", "OPTIONS"])
@_require_login
def billing_order_session(order_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401

    try:
        payload = get_payment_order_session_for_user(user_id=int(user["id"]), order_id=order_id)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(payload)


@app.route("/api/billing/orders/<int:order_id>/sandbox-complete", methods=["POST", "OPTIONS"])
@_require_login
def billing_complete_sandbox(order_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401

    try:
        payload = complete_payment_order_in_sandbox(user_id=int(user["id"]), order_id=order_id)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    refreshed_user = payload.get("user")
    if isinstance(refreshed_user, dict):
        _set_user_session(refreshed_user)
    return jsonify(payload)


def _payment_provider_placeholder_response(provider: str) -> Response:
    return (
        jsonify(
            {
                "error": (
                    f"{provider} callback endpoint is scaffolded but not fully implemented yet. "
                    "Add provider signature verification, status mapping, and order capture logic."
                )
            }
        ),
        501,
    )


def _validate_project_source_payload(payload: dict[str, Any]) -> dict[str, str | None]:
    project_zip_base64 = payload.get("project_zip_base64")
    github_repo_url = payload.get("github_repo_url")
    github_ref = payload.get("github_ref")
    project_subdir = payload.get("project_subdir")
    preview_path = payload.get("preview_path")

    if project_zip_base64 is not None and (
        not isinstance(project_zip_base64, str) or not project_zip_base64.strip()
    ):
        raise ValueError("`project_zip_base64` must be a non-empty base64 string when provided.")
    if github_repo_url is not None and (
        not isinstance(github_repo_url, str) or not github_repo_url.strip()
    ):
        raise ValueError("`github_repo_url` must be a non-empty string when provided.")
    if github_ref is not None and not isinstance(github_ref, str):
        raise ValueError("`github_ref` must be a string when provided.")
    if project_subdir is not None and (
        not isinstance(project_subdir, str) or not project_subdir.strip()
    ):
        raise ValueError("`project_subdir` must be a non-empty string when provided.")
    if preview_path is not None and (
        not isinstance(preview_path, str) or not preview_path.strip()
    ):
        raise ValueError("`preview_path` must be a non-empty string when provided.")

    source_count = sum(
        1
        for present in (bool(project_zip_base64), bool(github_repo_url))
        if present
    )
    if source_count != 1:
        raise ValueError("Exactly one of `project_zip_base64` or `github_repo_url` must be provided.")

    return {
        "project_zip_base64": project_zip_base64.strip() if isinstance(project_zip_base64, str) else None,
        "github_repo_url": github_repo_url.strip() if isinstance(github_repo_url, str) else None,
        "github_ref": github_ref.strip() if isinstance(github_ref, str) and github_ref.strip() else None,
        "project_subdir": normalize_project_path(project_subdir) if isinstance(project_subdir, str) else None,
        "preview_path": normalize_project_path(preview_path) if isinstance(preview_path, str) else None,
    }


def _validate_project_patch_payload(payload: dict[str, Any]) -> dict[str, str | None]:
    source_payload = _validate_project_source_payload(payload)
    language = normalize_language(payload.get("language", "python"))
    language_spec = get_language_spec(language)

    raw_filename = payload.get("filename")
    if not isinstance(raw_filename, str) or not raw_filename.strip():
        raise ValueError("`filename` must be a non-empty string.")
    filename = normalize_project_path(
        raw_filename,
        required_suffixes=language_spec.source_extensions,
    )

    raw_git_diff = payload.get("git_diff")
    if not isinstance(raw_git_diff, str) or not raw_git_diff.strip():
        raise ValueError("`git_diff` must be a non-empty string.")

    raw_zip_filename = payload.get("zip_filename")
    if raw_zip_filename is not None and not isinstance(raw_zip_filename, str):
        raise ValueError("`zip_filename` must be a string when provided.")

    raw_commit_message = payload.get("commit_message")
    if raw_commit_message is not None and not isinstance(raw_commit_message, str):
        raise ValueError("`commit_message` must be a string when provided.")

    return {
        **source_payload,
        "language": language,
        "filename": filename,
        "git_diff": raw_git_diff,
        "zip_filename": raw_zip_filename.strip() if isinstance(raw_zip_filename, str) and raw_zip_filename.strip() else None,
        "commit_message": (
            raw_commit_message.strip()
            if isinstance(raw_commit_message, str) and raw_commit_message.strip()
            else None
        ),
    }


def _project_file_target(project_root: Path, relative_path: str) -> Path:
    normalized = normalize_project_path(relative_path)
    target = (project_root / normalized).resolve()
    resolved_root = project_root.resolve()
    if resolved_root not in target.parents and target != resolved_root:
        raise ValueError("Path escapes the prepared project root.")
    return target


def _write_patched_project_files(
    project_root: Path,
    *,
    original_files: dict[str, str],
    patched_files: dict[str, str],
) -> None:
    for relative_path in original_files:
        if relative_path in patched_files:
            continue
        target = _project_file_target(project_root, relative_path)
        if target.exists():
            target.unlink()

    for relative_path, content in patched_files.items():
        target = _project_file_target(project_root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _archive_directory(root_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root_dir).as_posix())
    return buffer.getvalue()


def _build_patched_zip_name(raw_filename: str | None) -> str:
    cleaned = Path(raw_filename or "project.zip").name.strip() or "project.zip"
    stem = cleaned[:-4] if cleaned.lower().endswith(".zip") else cleaned
    stem = stem or "project"
    return f"{stem}-patched.zip"


def _default_project_commit_message(filename: str) -> str:
    return f"Apply AutoRepair patch for {Path(filename).name}"


def _github_push_url(repo_url: str, access_token: str) -> str:
    if repo_url.startswith("git@github.com:"):
        repo_path = repo_url.split(":", 1)[1].strip()
    else:
        parsed = urlparse(repo_url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host != "github.com":
            raise ValueError("`github_repo_url` must point to github.com.")
        repo_path = parsed.path.lstrip("/").strip()
    repo_path = repo_path.rstrip("/")
    if repo_path.endswith(".git"):
        repo_path = repo_path[:-4]
    if repo_path.count("/") < 1:
        raise ValueError("`github_repo_url` must include the owner and repository name.")
    return f"https://x-access-token:{quote(access_token, safe='')}@github.com/{repo_path}.git"


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
    timeout_sec: int = 60,
) -> str:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if extra_env:
        env.update(extra_env)
    verb = args[0] if args else "command"
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Git {verb} failed: {message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Git {verb} failed: {exc}") from exc
    return completed.stdout.strip()


@app.route("/api/billing/providers/stripe/webhook", methods=["POST", "OPTIONS"])
def billing_stripe_webhook() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return _payment_provider_placeholder_response("Stripe")


@app.route("/api/billing/providers/paypal/webhook", methods=["POST", "OPTIONS"])
def billing_paypal_webhook() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return _payment_provider_placeholder_response("PayPal")


@app.route("/api/billing/providers/wechat/notify", methods=["POST", "OPTIONS"])
def billing_wechat_notify() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return _payment_provider_placeholder_response("WeChat Pay")


@app.route("/api/billing/providers/alipay/notify", methods=["POST", "OPTIONS"])
def billing_alipay_notify() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return _payment_provider_placeholder_response("Alipay")


@app.route("/api/repair/stream", methods=["POST", "OPTIONS"])
@_require_login
def repair_stream() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        repair_request = RepairRequest.from_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    user_id = int(user["id"])

    repair_cost = wallet_store.pricing_cost_for_role(str(user.get("role") or "basic"), "repair")
    try:
        wallet_store.spend_credits(
            user_id=user_id,
            amount=repair_cost,
            reason_code="repair",
            reference_type="repair_pipeline",
            note=f"repair · {repair_request.language} · {repair_request.model}",
        )
    except InsufficientCreditsError as exc:
        return jsonify(
            {
                "error": str(exc),
                "code": "insufficient_credits",
                "required": exc.required,
                "balance": exc.balance,
            }
        ), 402

    event_queue: queue.Queue[str | object] = queue.Queue()
    sentinel = object()
    captured: dict[str, Any] = {
        "code": repair_request.code or "",
        "filename": repair_request.filename or "",
        "language": repair_request.language,
        "input_text": repair_request.input_text or "",
        "model": repair_request.model,
        "source_type": repair_request.source_type,
        "github_repo_url": repair_request.github_repo_url,
        "github_ref": repair_request.github_ref,
        "project_subdir": repair_request.project_subdir,
        "run_result": None,
        "stages": _empty_stage_map(),
        "events": [],
        "final_diff": "",
        "final_status": "",
        "error_message": "",
    }
    # The `result` event is emitted before background explain threads finish streaming,
    # so their `stage/completed` events arrive *after* the initial save. We remember the
    # saved history id here and re-save once the pipeline worker has joined every
    # background thread, so the persisted snapshot reflects the final stage statuses.
    saved_history_id: int | None = None

    def _final_preview_text() -> str:
        if captured.get("error_message"):
            return _truncate_text(
                str(captured["error_message"]) or "Agent run failed."
            )
        status = str(captured.get("final_status") or "")
        if status == "clean":
            return "No runtime error detected."
        if status == "verified":
            return "Repair diff verified and ready for review."
        if status == "verify_failed":
            return "Repair diff generated, but verification failed."
        if status:
            return "Repair diff is ready."
        return ""

    def emit(event: str, data: dict[str, Any]) -> None:
        nonlocal saved_history_id
        outgoing = dict(data)

        if event == "stage":
            stage = str(outgoing.get("stage", ""))
            if stage in captured["stages"]:
                captured["stages"][stage]["status"] = outgoing.get("status", "idle")
        elif event == "run_result":
            if outgoing.get("entrypoint"):
                captured["filename"] = str(outgoing.get("entrypoint"))
            if isinstance(outgoing.get("entrypoint_code"), str):
                captured["code"] = str(outgoing.get("entrypoint_code"))
            captured["run_result"] = {
                "stdout": outgoing.get("stdout", ""),
                "stderr": outgoing.get("stderr", ""),
                "input_text": outgoing.get("input_text", ""),
                "entrypoint": outgoing.get("entrypoint"),
                "source_type": outgoing.get("source_type"),
                "file_count": outgoing.get("file_count"),
                "execution": outgoing.get("execution"),
            }
        elif event == "inspect_report":
            captured["stages"]["inspect"]["report"] = json.dumps(
                outgoing.get("report", {}),
                ensure_ascii=False,
                indent=2,
            )
        elif event == "plan_report":
            captured["stages"]["plan"]["report"] = str(outgoing.get("report", ""))
        elif event == "code_report":
            diff_text = str(outgoing.get("git_diff", ""))
            report_text = str(outgoing.get("report", diff_text))
            captured["stages"]["code"]["report"] = report_text
            captured["stages"]["code"]["diff"] = diff_text
            if diff_text:
                captured["final_diff"] = diff_text
        elif event == "verify_report":
            captured["stages"]["verify"]["report"] = json.dumps(
                outgoing.get("report", {}),
                ensure_ascii=False,
                indent=2,
            )
        elif event == "explain_chunk":
            stage = str(outgoing.get("stage", ""))
            if stage in captured["stages"]:
                captured["stages"][stage]["explain"] += str(outgoing.get("chunk", ""))
                captured["stages"][stage]["status"] = "explaining"
        elif event == "stage_reasoning_chunk":
            stage = str(outgoing.get("stage", ""))
            if stage in captured["stages"]:
                captured["stages"][stage]["reasoning"] += str(outgoing.get("chunk", ""))
        elif event == "code_diff_chunk":
            captured["stages"]["code"]["diff"] += str(outgoing.get("chunk", ""))
        elif event == "tool_event":
            stage = str(outgoing.get("stage", ""))
            if stage in captured["stages"]:
                captured["stages"][stage]["toolEvents"].append(
                    {
                        "tool_name": str(outgoing.get("tool_name", "tool")),
                        "status": str(outgoing.get("status", "started")),
                        "round": outgoing.get("round"),
                        "arguments": outgoing.get("arguments"),
                        "output_preview": outgoing.get("output_preview"),
                        "output_truncated": bool(outgoing.get("output_truncated", False)),
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    }
                )
        elif event == "error":
            captured["error_message"] = str(outgoing.get("message", ""))
            saved = save_history(
                user_id=user_id,
                mode="agent",
                title=_agent_history_title(repair_request, captured.get("filename")),
                preview_text=_truncate_text(captured["error_message"] or "Agent run failed."),
                model=repair_request.model,
                language=repair_request.language,
                snapshot=captured,
                history_id=saved_history_id,
            )
            saved_history_id = saved["id"]
            outgoing["history_id"] = saved["id"]
        elif event == "result":
            if outgoing.get("filename"):
                captured["filename"] = str(outgoing.get("filename"))
            if outgoing.get("git_diff"):
                captured["final_diff"] = str(outgoing.get("git_diff"))
                captured["stages"]["code"]["diff"] = str(outgoing.get("git_diff"))
            captured["final_status"] = str(outgoing.get("status", ""))
            if captured["final_status"] == "clean":
                preview_text = "No runtime error detected."
            elif captured["final_status"] == "verified":
                preview_text = "Repair diff verified and ready for review."
            elif captured["final_status"] == "verify_failed":
                preview_text = "Repair diff generated, but verification failed."
            else:
                preview_text = "Repair diff is ready."
            saved = save_history(
                user_id=user_id,
                mode="agent",
                title=_agent_history_title(repair_request, captured.get("filename")),
                preview_text=preview_text,
                model=repair_request.model,
                language=repair_request.language,
                snapshot=captured,
                history_id=saved_history_id,
            )
            saved_history_id = saved["id"]
            outgoing["history_id"] = saved["id"]

        captured["events"].append(
            {
                "id": f"history-{len(captured['events']) + 1}",
                "event": event,
                "stage": outgoing.get("stage"),
                "summary": _event_summary(event, outgoing),
                "at": _ui_timestamp(),
            }
        )
        event_queue.put(_format_sse(event, outgoing))

    def worker() -> None:
        nonlocal saved_history_id
        try:
            run_repair_pipeline(repair_request, emit, user_id=user_id)
        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
            # By the time `run_repair_pipeline` returns, every background explain thread
            # has already been joined inside the pipeline. So captured["stages"] now
            # contains the final `completed` status for every stage. Re-persist the
            # snapshot so the UI restores a fully-finished pipeline next time the user
            # re-opens this conversation from history. Best-effort: swallow any error
            # here so we never block tearing down the SSE stream.
            if saved_history_id is not None:
                try:
                    save_history(
                        user_id=user_id,
                        mode="agent",
                        title=_agent_history_title(
                            repair_request, captured.get("filename")
                        ),
                        preview_text=_final_preview_text(),
                        model=repair_request.model,
                        language=repair_request.language,
                        snapshot=captured,
                        history_id=saved_history_id,
                    )
                except Exception:
                    pass
            event_queue.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    @stream_with_context
    def generate():
        yield _format_sse(
            "accepted",
            {
                "filename": repair_request.filename or "",
                "language": repair_request.language,
                "model": repair_request.model,
                "source_type": repair_request.source_type,
            },
        )
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield item

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/repair/project-files", methods=["POST", "OPTIONS"])
@_require_login
def repair_project_files() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        source_payload = _validate_project_source_payload(payload)
        result = list_project_entrypoint_options(
            project_zip_base64=source_payload["project_zip_base64"],
            github_repo_url=source_payload["github_repo_url"],
            github_ref=source_payload["github_ref"],
            project_subdir=source_payload["project_subdir"],
            preview_path=source_payload["preview_path"],
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.route("/api/repair/project-download", methods=["POST", "OPTIONS"])
@_require_login
def repair_project_download() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        request_payload = _validate_project_patch_payload(payload)
        if not request_payload["project_zip_base64"]:
            raise ValueError("`project_zip_base64` is required for ZIP download.")
        with prepare_project_workspace(
            code=None,
            filename=request_payload["filename"],
            language=str(request_payload["language"]),
            project_files=(),
            project_zip_base64=request_payload["project_zip_base64"],
            github_repo_url=None,
            github_ref=None,
            project_subdir=request_payload["project_subdir"],
        ) as workspace:
            patched_files = _apply_unified_diff_to_project(
                workspace.file_map,
                str(request_payload["git_diff"]),
                default_path=workspace.entrypoint,
            )
            _write_patched_project_files(
                workspace.root_dir,
                original_files=workspace.file_map,
                patched_files=patched_files,
            )
            archive_bytes = _archive_directory(workspace.source_root)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    response = Response(archive_bytes, mimetype="application/zip")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{_build_patched_zip_name(request_payload["zip_filename"])}"'
    )
    return response


@app.route("/api/repair/project-push", methods=["POST", "OPTIONS"])
@_require_login
def repair_project_push() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    access_token = _github_access_token_from_session()
    if not access_token:
        return jsonify({"error": "GitHub 推送需要使用 GitHub 登录并重新授权仓库权限。"}), 403

    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401

    try:
        request_payload = _validate_project_patch_payload(payload)
        if not request_payload["github_repo_url"]:
            raise ValueError("`github_repo_url` is required for GitHub push.")
        with prepare_project_workspace(
            code=None,
            filename=request_payload["filename"],
            language=str(request_payload["language"]),
            project_files=(),
            project_zip_base64=None,
            github_repo_url=request_payload["github_repo_url"],
            github_ref=request_payload["github_ref"],
            project_subdir=request_payload["project_subdir"],
        ) as workspace:
            patched_files = _apply_unified_diff_to_project(
                workspace.file_map,
                str(request_payload["git_diff"]),
                default_path=workspace.entrypoint,
            )
            _write_patched_project_files(
                workspace.root_dir,
                original_files=workspace.file_map,
                patched_files=patched_files,
            )
            branch_name = _run_git(["branch", "--show-current"], cwd=workspace.source_root)
            if not branch_name:
                raise RuntimeError("GitHub 推送仅支持分支工作区，不支持 tag 或 detached HEAD。")

            status_output = _run_git(["status", "--porcelain"], cwd=workspace.source_root)
            if not status_output:
                return jsonify(
                    {
                        "ok": True,
                        "branch": branch_name,
                        "message": f"仓库分支 {branch_name} 已经包含这份补丁，没有新的改动需要推送。",
                    }
                )

            author_name = str(user.get("display_name") or "AutoRepair").strip() or "AutoRepair"
            author_email = str(user.get("email") or "autorepair@local.invalid").strip() or "autorepair@local.invalid"
            commit_env = {
                "GIT_AUTHOR_NAME": author_name,
                "GIT_AUTHOR_EMAIL": author_email,
                "GIT_COMMITTER_NAME": author_name,
                "GIT_COMMITTER_EMAIL": author_email,
            }
            commit_message = (
                str(request_payload["commit_message"])
                if request_payload["commit_message"]
                else _default_project_commit_message(str(request_payload["filename"]))
            )

            _run_git(["add", "--all"], cwd=workspace.source_root)
            _run_git(["commit", "-m", commit_message], cwd=workspace.source_root, extra_env=commit_env)
            commit_sha = _run_git(["rev-parse", "HEAD"], cwd=workspace.source_root)
            _run_git(
                [
                    "push",
                    _github_push_url(str(request_payload["github_repo_url"]), access_token),
                    f"HEAD:{branch_name}",
                ],
                cwd=workspace.source_root,
            )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            "branch": branch_name,
            "commit_sha": commit_sha,
            "message": f"已将修复后的提交推送到 GitHub 分支 {branch_name}。",
        }
    )


@app.route("/api/chat/stream", methods=["POST", "OPTIONS"])
@_require_login
def chat_stream() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        chat_request = ChatRequest.from_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    user_id = int(user["id"])

    chat_cost = wallet_store.pricing_cost_for_role(str(user.get("role") or "basic"), "chat")
    try:
        wallet_store.spend_credits(
            user_id=user_id,
            amount=chat_cost,
            reason_code="chat",
            reference_type="chat_pipeline",
            note=f"chat · {chat_request.model}",
        )
    except InsufficientCreditsError as exc:
        return jsonify(
            {
                "error": str(exc),
                "code": "insufficient_credits",
                "required": exc.required,
                "balance": exc.balance,
            }
        ), 402

    event_queue: queue.Queue[str | object] = queue.Queue()
    sentinel = object()
    assistant_reasoning = ""

    def emit(event: str, data: dict[str, Any]) -> None:
        nonlocal assistant_reasoning
        outgoing = dict(data)
        if event == "chat_reasoning_chunk":
            assistant_reasoning += str(outgoing.get("chunk", ""))
        if event == "result":
            assistant_text = str(outgoing.get("message", "")).strip()
            if isinstance(outgoing.get("reasoning"), str) and outgoing.get("reasoning"):
                assistant_reasoning = str(outgoing.get("reasoning"))
            snapshot_messages = [
                {
                    "role": message.role,
                    "content": message.content,
                    "reasoning": getattr(message, "reasoning", None),
                    "at": message.at or _ui_timestamp(),
                }
                for message in chat_request.messages
            ]
            snapshot_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "reasoning": assistant_reasoning.strip() or None,
                    "at": _ui_timestamp(),
                }
            )
            saved = save_history(
                user_id=user_id,
                mode="chat",
                title=_chat_history_title(snapshot_messages),
                preview_text=_truncate_text(assistant_text or snapshot_messages[-2]["content"]),
                model=chat_request.model,
                language=None,
                snapshot={"messages": snapshot_messages},
                history_id=chat_request.history_id,
            )
            outgoing["history_id"] = saved["id"]
        event_queue.put(_format_sse(event, outgoing))

    def worker() -> None:
        try:
            run_chat_pipeline(chat_request, emit, user_id=user_id)
        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
            event_queue.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    @stream_with_context
    def generate():
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield item

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# ---------------------------------------------------------------------------
# Credit Wallet endpoints
# ---------------------------------------------------------------------------


@app.route("/api/wallet/summary", methods=["GET", "OPTIONS"])
@_require_login
def wallet_summary() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    wallet_store.ensure_wallet_exists(int(user["id"]))
    return jsonify(wallet_store.get_wallet_snapshot(int(user["id"])))


@app.route("/api/admin/wallet/balances", methods=["GET", "OPTIONS"])
@_require_admin
def admin_wallet_balances() -> Response:
    limit = request.args.get("limit", default=200, type=int) or 200
    return jsonify({"items": wallet_store.list_wallets_for_admin(limit=limit)})


@app.route("/api/admin/wallet/pricing", methods=["GET", "OPTIONS"])
@_require_admin
def admin_wallet_pricing() -> Response:
    return jsonify({"items": wallet_store.list_pricing_rules()})


@app.route("/api/admin/wallet/pricing", methods=["POST", "OPTIONS"])
@_require_admin
def admin_wallet_pricing_update() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        rule = wallet_store.update_pricing_rule(
            role_code=str(payload.get("role_code", "")).strip().lower(),
            monthly_free_credits=int(payload.get("monthly_free_credits", 0) or 0),
            cost_per_chat=int(payload.get("cost_per_chat", 0) or 0),
            cost_per_repair=int(payload.get("cost_per_repair", 0) or 0),
            cost_per_benchmark_run=int(payload.get("cost_per_benchmark_run", 0) or 0),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"item": rule})


@app.route("/api/admin/wallet/grant", methods=["POST", "OPTIONS"])
@_require_admin
def admin_wallet_grant() -> Response:
    admin_user = _current_user()
    if admin_user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        target_user_id = int(payload.get("user_id") or 0)
        amount = int(payload.get("amount") or 0)
        if target_user_id <= 0 or amount <= 0:
            raise ValueError("user_id and positive amount are required.")
        note = str(payload.get("note", "")).strip() or None
        result = wallet_store.grant_credits(
            user_id=target_user_id,
            amount=amount,
            reason_code="admin_grant",
            actor_user_id=int(admin_user["id"]),
            note=note,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"transaction": result})


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------


@app.route("/api/benchmark/projects", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_projects_list() -> Response:
    return jsonify(
        {
            "items": bench_store.list_benchmark_projects(),
            "summary": bench_store.get_benchmark_summary(),
        }
    )


@app.route("/api/benchmark/projects/<int:project_id>/bugs", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_bugs_list(project_id: int) -> Response:
    return jsonify({"items": bench_store.list_benchmark_bugs(project_id=project_id)})


@app.route("/api/benchmark/runs", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_runs_list() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    limit = request.args.get("limit", default=50, type=int) or 50
    offset = request.args.get("offset", default=0, type=int) or 0
    return jsonify(bench_store.list_runs_for_user(int(user["id"]), limit=limit, offset=offset))


@app.route("/api/benchmark/leaderboard", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_leaderboard() -> Response:
    return jsonify({"items": bench_store.get_leaderboard()})


@app.route("/api/benchmark/runs/<int:run_id>", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_run_detail(run_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    run = bench_store.get_run(run_id, include_heavy=True)
    if run is None or int(run.get("user_id") or 0) != int(user["id"]):
        if run is not None and _is_admin(user):
            return jsonify(run)
        return jsonify({"error": "Benchmark run was not found."}), 404
    return jsonify(run)


@app.route("/api/benchmark/runs", methods=["POST", "OPTIONS"])
@_require_login
def benchmark_run_create() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        bug_id = int(payload.get("bug_id") or 0)
        model_key = str(payload.get("model_key") or "").strip()
        run_mode = str(payload.get("run_mode") or "inspect_only").strip().lower()
        strategy = str(payload.get("strategy") or "").strip().lower() or None
        if bug_id <= 0:
            raise ValueError("`bug_id` is required.")
        if not model_key:
            raise ValueError("`model_key` is required.")
        if run_mode not in {"inspect_only", "mock_repair", "full_repair"}:
            raise ValueError(
                "`run_mode` must be one of: inspect_only, mock_repair, full_repair."
            )
        if run_mode == "full_repair":
            strategy = strategy or "full_pipeline"
            if strategy not in {"full_pipeline", "naive_chat"}:
                raise ValueError("`strategy` must be one of: full_pipeline, naive_chat.")

        bug_ctx = bench_store.get_bug_with_project(bug_id)
        if bug_ctx is None:
            return jsonify({"error": "Benchmark bug was not found."}), 404

        project_source = bug_ctx["project"].get("source_type")
        if project_source != "defects4j":
            return jsonify({"error": f"Only defects4j benchmarks are supported (got {project_source})."}), 400

        role = str(user.get("role") or "basic")
        cost = wallet_store.pricing_cost_for_role(role, "benchmark_run")
        try:
            wallet_store.spend_credits(
                user_id=int(user["id"]),
                amount=cost,
                reason_code="benchmark_run",
                reference_type="benchmark_run",
                note=f"Benchmark {bug_ctx['project']['project_code']}:{bug_ctx['bug']['bug_key']}",
            )
        except InsufficientCreditsError as exc:
            return jsonify(
                {
                    "error": str(exc),
                    "code": "insufficient_credits",
                    "required": exc.required,
                    "balance": exc.balance,
                }
            ), 402

        run_id = bench_store.create_run(
            user_id=int(user["id"]),
            organization_id=None,
            project_id=int(bug_ctx["project"]["id"]),
            bug_id=int(bug_ctx["bug"]["id"]),
            model_key=model_key,
            run_mode=run_mode,
            credits_spent=cost,
            strategy=strategy,
        )

        defects4j_project = str(bug_ctx["bug"].get("defects4j_project") or "").strip()
        defects4j_bug_id = int(bug_ctx["bug"].get("defects4j_bug_id") or 0)
        if not defects4j_project or defects4j_bug_id <= 0:
            bench_store.update_run_progress(
                run_id,
                stage="error",
                run_status="failed",
                error_message="Bug is missing defects4j_project / defects4j_bug_id metadata.",
                finalize=True,
            )
            return jsonify({"error": "Bug is missing defects4j_project / defects4j_bug_id metadata."}), 400

        if run_mode == "full_repair":
            import threading

            from backend.benchmark.repair_runner import run_full_repair_for_bug

            threading.Thread(
                target=run_full_repair_for_bug,
                kwargs={
                    "run_id": run_id,
                    "user_id": int(user["id"]),
                    "organization_id": None,
                    "project_code": str(bug_ctx["project"]["project_code"]),
                    "defects4j_project": defects4j_project,
                    "defects4j_bug_id": defects4j_bug_id,
                    "model_key": model_key,
                    "strategy": strategy or "full_pipeline",
                    "force_checkout": True,
                },
                daemon=True,
                name=f"benchmark-repair-{run_id}",
            ).start()
        else:
            run_benchmark_in_background(
                run_id=run_id,
                user_id=int(user["id"]),
                organization_id=None,
                project_code=str(bug_ctx["project"]["project_code"]),
                defects4j_project=defects4j_project,
                defects4j_bug_id=defects4j_bug_id,
                model_key=model_key,
                run_mode=run_mode,
                force_checkout=True,
            )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/admin/benchmark/runs", methods=["GET", "OPTIONS"])
@_require_admin
def admin_benchmark_runs_list() -> Response:
    limit = request.args.get("limit", default=100, type=int) or 100
    return jsonify({"items": bench_store.list_runs_for_admin(limit=limit)})


@app.route("/api/admin/benchmark/refresh-defects4j", methods=["POST", "OPTIONS"])
@_require_admin
def admin_benchmark_refresh_defects4j() -> Response:
    """Re-import active-bugs.csv for every known Defects4J project."""
    payload = request.get_json(silent=True) or {}
    d4j_home = str(payload.get("d4j_home") or "").strip() or None
    projects = payload.get("projects") or None
    if projects is not None and not isinstance(projects, list):
        return jsonify({"error": "`projects` must be a list of project codes."}), 400
    limit_per_project = payload.get("limit_per_project")
    try:
        from backend.scripts.import_defects4j_bugs import import_all

        summary = import_all(
            d4j_home=d4j_home,
            projects=list(projects) if projects else None,
            limit_per_project=int(limit_per_project) if limit_per_project else None,
        )
    except SystemExit as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Import failed: {exc}"}), 500
    return jsonify({"summary": summary})


# ---------------------------------------------------------------------------
# Benchmark comparison experiments (weak-model-vs-strong-model study)
# ---------------------------------------------------------------------------


def _run_benchmark_experiment_in_background(
    *,
    experiment_id: int,
    user_id: int,
    bug_ids: list[int],
    arms: list[dict[str, str]],
) -> None:
    """Execute every (bug × arm) combo synchronously in a background thread."""
    import logging as _logging

    logger = _logging.getLogger("backend.benchmark.experiment")
    from backend.benchmark.repair_runner import run_full_repair_for_bug

    bench_store.update_experiment_status(experiment_id, status="running", mark_started=True)
    try:
        for bug_id in bug_ids:
            bug_ctx = bench_store.get_bug_with_project(bug_id)
            if bug_ctx is None:
                logger.warning("Experiment %s: bug_id %s not found", experiment_id, bug_id)
                continue
            project_code = str(bug_ctx["project"]["project_code"])
            defects4j_project = str(bug_ctx["bug"].get("defects4j_project") or "").strip()
            defects4j_bug_id = int(bug_ctx["bug"].get("defects4j_bug_id") or 0)
            if not defects4j_project or defects4j_bug_id <= 0:
                logger.warning(
                    "Experiment %s: bug %s missing defects4j metadata", experiment_id, bug_id
                )
                continue
            for arm in arms:
                strategy = arm["strategy"]
                model_key = arm["model_key"]
                run_id = bench_store.create_run(
                    user_id=user_id,
                    organization_id=None,
                    project_id=int(bug_ctx["project"]["id"]),
                    bug_id=int(bug_ctx["bug"]["id"]),
                    model_key=model_key,
                    run_mode="full_repair",
                    credits_spent=0,
                    strategy=strategy,
                    experiment_id=experiment_id,
                )
                try:
                    run_full_repair_for_bug(
                        run_id=run_id,
                        user_id=user_id,
                        organization_id=None,
                        project_code=project_code,
                        defects4j_project=defects4j_project,
                        defects4j_bug_id=defects4j_bug_id,
                        model_key=model_key,
                        strategy=strategy,
                        experiment_id=experiment_id,
                        force_checkout=True,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.exception(
                        "Experiment %s run %s failed: %s", experiment_id, run_id, exc
                    )
        bench_store.recount_experiment(experiment_id)
        bench_store.update_experiment_status(
            experiment_id, status="completed", mark_finished=True
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Experiment %s failed: %s", experiment_id, exc)
        bench_store.update_experiment_status(
            experiment_id, status="failed", mark_finished=True
        )


@app.route("/api/benchmark/experiments", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_experiments_list() -> Response:
    limit = request.args.get("limit", default=50, type=int) or 50
    return jsonify({"items": bench_store.list_experiments(limit=limit)})


@app.route("/api/benchmark/experiments/<int:experiment_id>", methods=["GET", "OPTIONS"])
@_require_login
def benchmark_experiment_detail(experiment_id: int) -> Response:
    data = bench_store.get_experiment_results(experiment_id)
    if not data:
        return jsonify({"error": "Experiment not found."}), 404
    return jsonify(data)


@app.route("/api/benchmark/experiments", methods=["POST", "OPTIONS"])
@_require_login
def benchmark_experiment_create() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    experiment_code = str(payload.get("experiment_code") or "").strip()
    if not experiment_code:
        return jsonify({"error": "`experiment_code` is required."}), 400
    title = str(payload.get("title") or experiment_code).strip() or experiment_code
    description = (str(payload.get("description") or "").strip() or None)
    hypothesis = (str(payload.get("hypothesis") or "").strip() or None)

    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        return jsonify({"error": "`arms` must be a non-empty list."}), 400
    arms: list[dict[str, str]] = []
    for raw in raw_arms:
        if not isinstance(raw, dict):
            return jsonify({"error": "Each arm must be an object with strategy/model_key."}), 400
        strategy = str(raw.get("strategy") or "").strip().lower()
        model_key = str(raw.get("model_key") or "").strip()
        if strategy not in {"full_pipeline", "naive_chat"}:
            return jsonify(
                {"error": f"Unsupported strategy `{strategy}` (expected full_pipeline|naive_chat)."}
            ), 400
        if not model_key:
            return jsonify({"error": "Each arm needs a non-empty `model_key`."}), 400
        arms.append({"strategy": strategy, "model_key": model_key})

    # Resolve bug ids — accept either `bug_ids` or (`project_code`, `limit`)
    raw_bug_ids = payload.get("bug_ids")
    bug_ids: list[int] = []
    if isinstance(raw_bug_ids, list) and raw_bug_ids:
        for v in raw_bug_ids:
            try:
                bug_ids.append(int(v))
            except (TypeError, ValueError):
                continue
    else:
        project_code = str(payload.get("project_code") or "").strip()
        limit_n = int(payload.get("limit") or 5)
        if not project_code:
            return jsonify(
                {"error": "Provide either `bug_ids` or (`project_code`, `limit`)."}
            ), 400
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT b.id FROM benchmark_bugs b
                    INNER JOIN benchmark_projects p ON p.id = b.project_id
                    WHERE p.project_code = %s AND b.is_active = 1
                    ORDER BY b.defects4j_bug_id
                    LIMIT %s
                    """,
                    (project_code, limit_n),
                )
                bug_ids = [int(row["id"]) for row in (cursor.fetchall() or [])]
    if not bug_ids:
        return jsonify({"error": "No bugs were resolved for the experiment."}), 400

    config = {
        "arms": arms,
        "bug_ids": bug_ids,
    }
    experiment = bench_store.get_or_create_experiment(
        experiment_code=experiment_code,
        title=title,
        description=description,
        hypothesis=hypothesis,
        config=config,
        created_by_user_id=int(user["id"]),
    )
    experiment_id = int(experiment["id"])
    bench_store.update_experiment_status(
        experiment_id,
        status="queued",
        config=config,
    )

    import threading

    threading.Thread(
        target=_run_benchmark_experiment_in_background,
        kwargs={
            "experiment_id": experiment_id,
            "user_id": int(user["id"]),
            "bug_ids": bug_ids,
            "arms": arms,
        },
        daemon=True,
        name=f"benchmark-experiment-{experiment_id}",
    ).start()

    return jsonify(
        {
            "experiment_id": experiment_id,
            "experiment_code": experiment_code,
            "bug_count": len(bug_ids),
            "arm_count": len(arms),
        }
    )


# ---------------------------------------------------------------------------
# Teams / Projects endpoints
# ---------------------------------------------------------------------------


def _validate_org_membership(org_id: int, user_id: int) -> bool:
    return team_store.user_is_member(org_id=org_id, user_id=user_id)


@app.route("/api/teams/organizations", methods=["GET", "OPTIONS"])
@_require_login
def teams_list_orgs() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    return jsonify({"items": team_store.list_organizations_for_user(int(user["id"]))})


@app.route("/api/teams/organizations", methods=["POST", "OPTIONS"])
@_require_login
def teams_create_org() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        org = team_store.create_organization(
            owner_user_id=int(user["id"]),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "") or "").strip() or None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"organization": org})


@app.route("/api/teams/organizations/<int:org_id>/members", methods=["GET", "OPTIONS"])
@_require_login
def teams_list_members(org_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(user["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    return jsonify({"items": team_store.list_members(org_id)})


@app.route("/api/teams/organizations/<int:org_id>/invites", methods=["GET", "OPTIONS"])
@_require_login
def teams_list_invites(org_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(user["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    return jsonify({"items": team_store.list_invites_for(org_id)})


@app.route("/api/teams/organizations/<int:org_id>/invites", methods=["POST", "OPTIONS"])
@_require_login
def teams_invite_member(org_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(user["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    payload = request.get_json(silent=True) or {}
    try:
        invite = team_store.invite_member(
            org_id=org_id,
            invited_by_user_id=int(user["id"]),
            email=str(payload.get("email", "")),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"invite": invite})


@app.route("/api/teams/invites/<invite_token>/accept", methods=["POST", "OPTIONS"])
@_require_login
def teams_accept_invite(invite_token: str) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    try:
        result = team_store.accept_invite(invite_token=invite_token, user_id=int(user["id"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/api/teams/organizations/<int:org_id>/members/<int:user_id>", methods=["DELETE", "OPTIONS"])
@_require_login
def teams_remove_member(org_id: int, user_id: int) -> Response:
    actor = _current_user()
    if actor is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(actor["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    try:
        team_store.remove_member(org_id=org_id, user_id=user_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True})


@app.route("/api/teams/organizations/<int:org_id>/projects", methods=["GET", "OPTIONS"])
@_require_login
def teams_list_projects(org_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(user["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    return jsonify({"items": team_store.list_projects(org_id)})


@app.route("/api/teams/organizations/<int:org_id>/projects", methods=["POST", "OPTIONS"])
@_require_login
def teams_create_project(org_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    if not _validate_org_membership(org_id, int(user["id"])):
        return jsonify({"error": "Organization was not found."}), 404
    payload = request.get_json(silent=True) or {}
    try:
        project = team_store.create_project(
            org_id=org_id,
            owner_user_id=int(user["id"]),
            name=str(payload.get("name", "")),
            language=str(payload.get("language", "") or "").strip() or None,
            description=str(payload.get("description", "") or "").strip() or None,
            repo_url=str(payload.get("repo_url", "") or "").strip() or None,
            color_hex=str(payload.get("color_hex", "") or "").strip() or None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"project": project})


@app.route("/api/teams/projects/<int:project_id>", methods=["DELETE", "OPTIONS"])
@_require_login
def teams_delete_project(project_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    project = team_store.get_project(project_id)
    if project is None:
        return jsonify({"error": "Project was not found."}), 404
    if not _validate_org_membership(int(project["organization_id"]), int(user["id"])):
        return jsonify({"error": "Project was not found."}), 404
    team_store.delete_project(project_id)
    return jsonify({"ok": True})


@app.route("/api/admin/teams/organizations", methods=["GET", "OPTIONS"])
@_require_admin
def admin_teams_list_orgs() -> Response:
    return jsonify({"items": team_store.list_all_organizations_for_admin()})


# ---------------------------------------------------------------------------
# Personal Center endpoints
# ---------------------------------------------------------------------------


@app.route("/api/profile/overview", methods=["GET", "OPTIONS"])
@_require_login
def profile_overview() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    wallet_snapshot = wallet_store.get_wallet_snapshot(int(user["id"]))
    return jsonify(
        {
            "user": user,
            "preferences": get_preferences(int(user["id"])),
            "overview": get_profile_overview(int(user["id"])),
            "wallet": wallet_snapshot["wallet"],
            "organizations": team_store.list_organizations_for_user(int(user["id"])),
            "api_tokens": list_api_tokens(int(user["id"])),
        }
    )


@app.route("/api/profile/preferences", methods=["POST", "OPTIONS"])
@_require_login
def profile_preferences_update() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True) or {}
    try:
        prefs = update_preferences(
            user_id=int(user["id"]),
            default_agent_model=payload.get("default_agent_model"),
            default_chat_model=payload.get("default_chat_model"),
            default_language=payload.get("default_language"),
            locale=payload.get("locale"),
            theme=payload.get("theme"),
            timezone=payload.get("timezone"),
            bio=payload.get("bio"),
            show_site_map_widget=payload.get("show_site_map_widget"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"preferences": prefs})


@app.route("/api/profile/api-tokens", methods=["GET", "OPTIONS"])
@_require_login
def profile_list_api_tokens() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    return jsonify({"items": list_api_tokens(int(user["id"]))})


@app.route("/api/profile/api-tokens", methods=["POST", "OPTIONS"])
@_require_login
def profile_create_api_token() -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    payload = request.get_json(silent=True) or {}
    try:
        token = create_api_token(
            user_id=int(user["id"]),
            token_name=str(payload.get("token_name", "")),
            scope=str(payload.get("scope", "repair")),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"token": token})


@app.route("/api/profile/api-tokens/<int:token_id>", methods=["DELETE", "OPTIONS"])
@_require_login
def profile_revoke_api_token(token_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    revoke_api_token(user_id=int(user["id"]), token_id=token_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# PDF Export endpoints
# ---------------------------------------------------------------------------


@app.route("/api/pdf/history/<int:history_id>", methods=["GET", "OPTIONS"])
@_require_login
def pdf_export_history(history_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    try:
        pdf_bytes, filename = build_repair_report_pdf(user_id=int(user["id"]), history_id=history_id)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    response = Response(pdf_bytes, mimetype="application/pdf")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Content-Length"] = str(len(pdf_bytes))
    return response


@app.route("/api/pdf/benchmark/<int:run_id>", methods=["GET", "OPTIONS"])
@_require_login
def pdf_export_benchmark(run_id: int) -> Response:
    user = _current_user()
    if user is None:
        return jsonify({"error": "Authentication required."}), 401
    try:
        pdf_bytes, filename = build_benchmark_report_pdf(user_id=int(user["id"]), run_id=run_id)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    response = Response(pdf_bytes, mimetype="application/pdf")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Content-Length"] = str(len(pdf_bytes))
    return response


# ---------------------------------------------------------------------------
# Site Map (publicly browsable; used by homepage widget)
# ---------------------------------------------------------------------------


@app.route("/api/site-map", methods=["GET", "OPTIONS"])
def site_map() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    return jsonify(
        {
            "groups": [
                {
                    "code": "workspace",
                    "title": "Workspace",
                    "items": [
                        {"code": "agent", "label": "Agent Repair", "path": "/agent"},
                        {"code": "chat", "label": "Chat", "path": "/chat"},
                        {"code": "benchmark", "label": "Benchmark Arena", "path": "/benchmark"},
                        {"code": "teams", "label": "Teams & Projects", "path": "/teams"},
                    ],
                },
                {
                    "code": "account",
                    "title": "Account",
                    "items": [
                        {"code": "profile", "label": "Personal Center", "path": "/profile"},
                        {"code": "wallet", "label": "Credit Wallet", "path": "/profile/wallet"},
                        {"code": "billing", "label": "Upgrade", "path": "/billing"},
                    ],
                },
                {
                    "code": "admin",
                    "title": "Admin",
                    "items": [
                        {"code": "dashboard", "label": "Admin Dashboard", "path": "/admin/dashboard"},
                        {"code": "users", "label": "Users", "path": "/admin/users"},
                        {"code": "models", "label": "Models", "path": "/admin/models"},
                        {"code": "wallet_admin", "label": "Credit Wallets", "path": "/admin/wallet"},
                        {"code": "benchmark_admin", "label": "Benchmark Runs", "path": "/admin/benchmark"},
                        {"code": "teams_admin", "label": "Organizations", "path": "/admin/teams"},
                    ],
                },
            ]
        }
    )


if __name__ == "__main__":
    from backend.app import main

    main()
