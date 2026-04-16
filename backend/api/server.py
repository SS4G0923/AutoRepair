from __future__ import annotations

import json
import os
import queue
import secrets
import threading
from datetime import datetime, timezone
from functools import wraps
from typing import Any
from urllib.parse import urlencode

from flask import Flask, Response, jsonify, redirect, request, session, stream_with_context, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth.store import (
    create_local_user,
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
from backend.repair.pipeline import RepairRequest, run_repair_pipeline
from backend.repair.workspace import list_project_entrypoint_options

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
            client_kwargs={"scope": "read:user user:email"},
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
        "run": {"status": "idle", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "inspect": {"status": "idle", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "plan": {"status": "idle", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "code": {"status": "idle", "explain": "", "report": "", "diff": "", "toolEvents": []},
        "verify": {"status": "idle", "explain": "", "report": "", "diff": "", "toolEvents": []},
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


def _github_email(client) -> str | None:
    emails_response = client.get("user/emails")
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


def _oauth_user_payload(provider: str, client) -> tuple[str, str, str, str | None]:
    token = client.authorize_access_token()
    if provider == "google":
        user_info = token.get("userinfo")
        if not isinstance(user_info, dict):
            user_info = client.get("userinfo").json()
        provider_user_id = str(user_info.get("sub", "")).strip()
        email = str(user_info.get("email", "")).strip()
        display_name = str(user_info.get("name") or email.split("@", 1)[0]).strip()
        avatar_url = user_info.get("picture")
    elif provider == "github":
        profile = client.get("user").json()
        provider_user_id = str(profile.get("id", "")).strip()
        email = str(profile.get("email") or _github_email(client) or "").strip()
        display_name = str(profile.get("name") or profile.get("login") or email.split("@", 1)[0]).strip()
        avatar_url = profile.get("avatar_url")
    else:
        raise ValueError("Unsupported OAuth provider.")

    if not provider_user_id or not email:
        raise ValueError("OAuth provider did not return a usable email.")
    return provider_user_id, email, display_name, avatar_url


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
        provider_user_id, email, display_name, avatar_url = _oauth_user_payload(provider, client)
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

    def emit(event: str, data: dict[str, Any]) -> None:
        outgoing = dict(data)

        if event == "stage":
            stage = str(outgoing.get("stage", ""))
            if stage in captured["stages"]:
                captured["stages"][stage]["status"] = outgoing.get("status", "idle")
        elif event == "run_result":
            if outgoing.get("entrypoint"):
                captured["filename"] = str(outgoing.get("entrypoint"))
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
            )
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
            )
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
        try:
            run_repair_pipeline(repair_request, emit, user_id=user_id)
        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
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

    project_zip_base64 = payload.get("project_zip_base64")
    github_repo_url = payload.get("github_repo_url")
    github_ref = payload.get("github_ref")
    project_subdir = payload.get("project_subdir")

    if project_zip_base64 is not None and (
        not isinstance(project_zip_base64, str) or not project_zip_base64.strip()
    ):
        return jsonify({"error": "`project_zip_base64` must be a non-empty base64 string when provided."}), 400
    if github_repo_url is not None and (
        not isinstance(github_repo_url, str) or not github_repo_url.strip()
    ):
        return jsonify({"error": "`github_repo_url` must be a non-empty string when provided."}), 400
    if github_ref is not None and not isinstance(github_ref, str):
        return jsonify({"error": "`github_ref` must be a string when provided."}), 400
    if project_subdir is not None and (
        not isinstance(project_subdir, str) or not project_subdir.strip()
    ):
        return jsonify({"error": "`project_subdir` must be a non-empty string when provided."}), 400

    source_count = sum(
        1
        for present in (bool(project_zip_base64), bool(github_repo_url))
        if present
    )
    if source_count != 1:
        return jsonify({"error": "Exactly one of `project_zip_base64` or `github_repo_url` must be provided."}), 400

    try:
        result = list_project_entrypoint_options(
            project_zip_base64=project_zip_base64.strip() if isinstance(project_zip_base64, str) else None,
            github_repo_url=github_repo_url.strip() if isinstance(github_repo_url, str) else None,
            github_ref=github_ref.strip() if isinstance(github_ref, str) and github_ref.strip() else None,
            project_subdir=project_subdir.strip() if isinstance(project_subdir, str) else None,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


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

    event_queue: queue.Queue[str | object] = queue.Queue()
    sentinel = object()

    def emit(event: str, data: dict[str, Any]) -> None:
        outgoing = dict(data)
        if event == "result":
            assistant_text = str(outgoing.get("message", "")).strip()
            snapshot_messages = [
                {"role": message.role, "content": message.content, "at": message.at or _ui_timestamp()}
                for message in chat_request.messages
            ]
            snapshot_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
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


if __name__ == "__main__":
    from backend.app import main

    main()
