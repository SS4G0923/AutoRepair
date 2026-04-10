from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from backend.auth.store import normalize_email

PAYMENT_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "card": {
        "provider_code": "stripe",
        "provider_name": "Stripe",
        "display_mode": "card_form",
        "required_envs": [
            "AUTOREPAIR_PUBLIC_BASE_URL",
            "STRIPE_SECRET_KEY",
            "STRIPE_PUBLISHABLE_KEY",
            "STRIPE_WEBHOOK_SECRET",
        ],
        "script_template": "https://js.stripe.com/v3/",
        "notify_path": "/api/billing/providers/stripe/webhook",
        "return_path": "/billing/return/stripe",
    },
    "paypal": {
        "provider_code": "paypal",
        "provider_name": "PayPal",
        "display_mode": "paypal_buttons",
        "required_envs": [
            "AUTOREPAIR_PUBLIC_BASE_URL",
            "PAYPAL_CLIENT_ID",
            "PAYPAL_CLIENT_SECRET",
            "PAYPAL_WEBHOOK_ID",
        ],
        "notify_path": "/api/billing/providers/paypal/webhook",
        "return_path": "/billing/return/paypal",
    },
    "wechat": {
        "provider_code": "wechatpay",
        "provider_name": "WeChat Pay",
        "display_mode": "qr_code",
        "required_envs": [
            "AUTOREPAIR_PUBLIC_BASE_URL",
            "WECHATPAY_MCHID",
            "WECHATPAY_APPID",
            "WECHATPAY_SERIAL_NO",
            "WECHATPAY_PRIVATE_KEY_PATH",
            "WECHATPAY_API_V3_KEY",
            "WECHATPAY_NOTIFY_URL",
        ],
        "notify_path": "/api/billing/providers/wechat/notify",
        "return_path": "/billing/return/wechat",
    },
    "alipay": {
        "provider_code": "alipay",
        "provider_name": "Alipay",
        "display_mode": "qr_code",
        "required_envs": [
            "AUTOREPAIR_PUBLIC_BASE_URL",
            "ALIPAY_APP_ID",
            "ALIPAY_PRIVATE_KEY",
            "ALIPAY_PUBLIC_KEY",
            "ALIPAY_NOTIFY_URL",
        ],
        "notify_path": "/api/billing/providers/alipay/notify",
        "return_path": "/billing/return/alipay",
    },
}


def payment_environment() -> str:
    environment = os.getenv("AUTOREPAIR_PAYMENT_ENVIRONMENT", "prepare").strip().lower()
    if environment in {"prepare", "live"}:
        return environment
    return "prepare"


def _public_base_url() -> str:
    return os.getenv("AUTOREPAIR_PUBLIC_BASE_URL", "").rstrip("/")


def payment_provider_profile(payment_method: str, *, currency: str = "CNY") -> dict[str, Any]:
    spec = PAYMENT_PROVIDER_SPECS[payment_method]
    missing_config = [name for name in spec["required_envs"] if not os.getenv(name, "").strip()]
    environment = payment_environment()
    provider_code = spec["provider_code"]
    script_url = None
    public_config: dict[str, Any] = {}

    if payment_method == "card":
        script_url = spec["script_template"]
        public_config = {
            "publishable_key": os.getenv("STRIPE_PUBLISHABLE_KEY", "").strip() or None,
            "currency": currency,
            "elements_mode": "card_number",
        }
    elif payment_method == "paypal":
        client_id = os.getenv("PAYPAL_CLIENT_ID", "").strip()
        if client_id:
            script_url = (
                f"https://www.paypal.com/sdk/js?client-id={quote(client_id)}"
                f"&currency={quote(currency)}&intent=capture&components=buttons"
            )
        public_config = {
            "client_id": client_id or None,
            "currency": currency,
            "intent": "capture",
        }

    return {
        "payment_method": payment_method,
        "provider_code": provider_code,
        "provider_name": spec["provider_name"],
        "display_mode": spec["display_mode"],
        "integration_status": "ready" if not missing_config else "missing_config",
        "missing_config": missing_config,
        "script_url": script_url,
        "public_config": public_config,
        "environment": environment,
        "notify_path": spec["notify_path"],
        "return_path": spec["return_path"],
    }


def build_checkout_payload(
    *,
    payment_method: str,
    order_id: int,
    order_no: str,
    plan_name: str,
    amount_cents: int,
    currency: str,
    user_email: str | None = None,
) -> dict[str, Any]:
    profile = payment_provider_profile(payment_method, currency=currency)
    amount_value = f"{amount_cents / 100:.2f} {currency}"
    public_base_url = _public_base_url()
    return_url = f"{public_base_url}{profile['return_path']}" if public_base_url else None
    notify_url = (
        f"{public_base_url}{profile['notify_path']}" if public_base_url else os.getenv(
            f"{profile['provider_code'].upper()}_NOTIFY_URL", ""
        ).strip() or None
    )

    if payment_method == "card":
        instructions = (
            "Use Stripe.js on the frontend to mount card number / expiry / CVC fields, "
            "then have the backend create a PaymentIntent and return client_secret for confirmation."
        )
        next_action_path = f"/api/billing/orders/{order_id}/card/setup-intent"
    elif payment_method == "paypal":
        instructions = (
            "Load the PayPal JavaScript SDK, render PayPal Buttons, create an order server-side, "
            "and capture the order after onApprove."
        )
        next_action_path = f"/api/billing/orders/{order_id}/paypal/create-order"
    elif payment_method == "wechat":
        instructions = (
            "Call WeChat Pay Native unified order on the backend, store the returned code_url, "
            "then render that code_url as a QR code for the user to scan."
        )
        next_action_path = f"/api/billing/orders/{order_id}/wechat/precreate"
    else:
        instructions = (
            "Call Alipay precreate on the backend, store the returned QR payment string or payment URL, "
            "then render it as a QR code for the user to scan."
        )
        next_action_path = f"/api/billing/orders/{order_id}/alipay/precreate"

    if profile["missing_config"]:
        instructions = (
            f"{instructions} Missing configuration: {', '.join(profile['missing_config'])}."
        )

    qr_code_url = None
    if payment_method in {"wechat", "alipay"}:
        qr_code_url = None

    return {
        "provider_code": profile["provider_code"],
        "provider_name": profile["provider_name"],
        "display_mode": profile["display_mode"],
        "integration_status": profile["integration_status"],
        "missing_config": profile["missing_config"],
        "script_url": profile["script_url"],
        "public_config": profile["public_config"],
        "environment": profile["environment"],
        "notify_url": notify_url,
        "return_url": return_url,
        "order_reference": order_no,
        "amount_label": amount_value,
        "plan_name": plan_name,
        "customer_email": normalize_email(user_email or "") if user_email else None,
        "instructions": instructions,
        "next_action_path": next_action_path,
        "checkout_url": None,
        "qr_code_url": qr_code_url,
        "qr_code_text": None,
    }
