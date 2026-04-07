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
    upsert_oauth_user,
)
from backend.chat.pipeline import ChatRequest, run_chat_pipeline
from backend.history.store import (
    get_history_for_user,
    list_histories_for_user,
    save_history,
    soft_delete_history_for_user,
)
from backend.repair.pipeline import RepairRequest, run_repair_pipeline

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
        return get_user_by_id(int(raw_user_id))
    except (TypeError, ValueError):
        session.clear()
        return None


def _set_user_session(user: dict[str, Any]) -> None:
    session.clear()
    session["user_id"] = user["id"]


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

    _set_user_session(user)
    return jsonify({"user": user, "oauth_providers": _enabled_oauth_providers()})


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
    if user is None or not user.get("password_hash") or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session_user = get_user_by_id(int(user["id"]))
    if session_user is None:
        return jsonify({"error": "User account could not be loaded."}), 500

    _set_user_session(session_user)
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
        _set_user_session(user)
    except Exception:
        return redirect(_frontend_redirect_url(auth_error="oauth_callback_failed"))

    return redirect(_frontend_redirect_url(auth_success="1"))


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
            captured["stages"]["code"]["report"] = diff_text
            captured["stages"]["code"]["diff"] = diff_text
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
            run_repair_pipeline(repair_request, emit)
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
            run_chat_pipeline(chat_request, emit)
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
