from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .profiles import SetupProfile
from .utils import env as env_utils

_BODY_PREVIEW_CHARS = 300
_PROVIDER_ENV_EXAMPLES = [
    "OPENAI_API_KEY=sk-...",
    "ANTHROPIC_API_KEY=sk-ant-...",
    "OPENROUTER_API_KEY=sk-or-...",
]


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        text = response.text
        if len(text) > _BODY_PREVIEW_CHARS:
            return f"{text[:_BODY_PREVIEW_CHARS]}..."
        return text


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    url = _url(base_url, path)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                data=data,
                files=files,
                json=json_body,
            )
        return {
            "url": url,
            "status_code": response.status_code,
            "ok": response.status_code < 400,
            "body": _response_body(response),
        }
    except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def _headers_for_profile(profile: SetupProfile, env_values: dict[str, str]) -> dict[str, str]:
    if profile.auth_mode != "single_user":
        return {}
    key = (env_values.get("SINGLE_USER_API_KEY") or env_values.get("API_KEY") or "").strip()
    if not key:
        return {}
    return {"X-API-KEY": key}


def _login_multi_user(
    base_url: str,
    env_values: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    username = (env_values.get("ADMIN_USERNAME") or "").strip()
    password = env_values.get("ADMIN_PASSWORD") or ""
    if not username or not password:
        return {
            "status": "missing_credentials",
            "ok": False,
            "detail": "ADMIN_USERNAME and ADMIN_PASSWORD are required for multi-user profile verification.",
        }, {}

    response = _request(
        "POST",
        base_url,
        "/api/v1/auth/login",
        data={"username": username, "password": password},
        timeout=timeout,
    )
    body = response.get("body")
    token = body.get("access_token") if isinstance(body, dict) else None
    action = {
        "status": "ok" if response.get("ok") and token else "error",
        "ok": bool(response.get("ok") and token),
        "url": response.get("url"),
        "status_code": response.get("status_code"),
    }
    if not response.get("ok"):
        action["error"] = response.get("error") or body
    if not token and response.get("ok"):
        action["error"] = "login response did not include an access token"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return action, headers


def _configured_provider_count(body: Any) -> int:
    if not isinstance(body, dict):
        return 0
    total = body.get("total_configured")
    if isinstance(total, int):
        return total
    providers = body.get("providers")
    if isinstance(providers, list):
        return len(providers)
    if isinstance(providers, dict):
        return len(providers)
    return 0


def _provider_check(base_url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    response = _request("GET", base_url, "/api/v1/llm/providers", headers=headers, timeout=timeout)
    result: dict[str, Any] = {
        "url": response.get("url"),
        "status_code": response.get("status_code"),
    }
    if not response.get("ok"):
        result.update({"status": "endpoint_failed", "ok": False})
        if "error" in response:
            result["error"] = response["error"]
        return result

    configured = _configured_provider_count(response.get("body"))
    result["configured"] = configured
    if configured <= 0:
        result.update(
            {
                "status": "provider_missing",
                "ok": True,
                "env_examples": list(_PROVIDER_ENV_EXAMPLES),
            }
        )
        return result
    result.update({"status": "provider_configured", "ok": True})
    return result


def _first_value_check(base_url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    sample = b"# tldw onboarding verification\n\nThis sample verifies ingest and search.\n"
    ingest = _request(
        "POST",
        base_url,
        "/api/v1/media/add",
        headers=headers,
        data={
            "media_type": "document",
            "title": "tldw onboarding verification",
            "keywords": "onboarding,verification",
            "perform_analysis": "false",
            "perform_chunking": "true",
        },
        files={"files": ("tldw-onboarding-verification.md", sample, "text/markdown")},
        timeout=timeout,
    )
    search = _request(
        "POST",
        base_url,
        "/api/v1/media/search",
        headers=headers,
        json_body={"query": "onboarding verification", "fields": ["title", "content"]},
        timeout=timeout,
    )
    return {
        "ingest": "ok" if ingest.get("ok") else "error",
        "search": "ok" if search.get("ok") else "error",
        "ok": bool(ingest.get("ok") and search.get("ok")),
        "details": {"ingest": ingest, "search": search},
    }


def run_profile_checks(
    *,
    profile: SetupProfile,
    base_url: str,
    webui_url: str | None,
    env_path: Path,
    first_value: bool,
    timeout: float = 5.0,
) -> dict[str, Any]:
    env_values = env_utils.load_env(env_path)
    actions: list[dict[str, Any]] = [
        {"server": {"mode": "existing", "profile": profile.name}},
    ]
    notes: list[str] = []

    endpoint_results = {
        "health": _request("GET", base_url, "/health", timeout=timeout),
        "ready": _request("GET", base_url, "/ready", timeout=timeout),
        "docs": _request("GET", base_url, "/docs", timeout=timeout),
        "quickstart": _request("GET", base_url, "/api/v1/config/quickstart", timeout=timeout),
    }
    actions.append({"endpoints": endpoint_results})

    headers = _headers_for_profile(profile, env_values)
    auth_checks: dict[str, Any] = {}
    if profile.auth_mode == "multi_user":
        login_action, login_headers = _login_multi_user(base_url, env_values, timeout)
        auth_checks["login"] = login_action
        headers = login_headers
    auth_checks["me"] = _request("GET", base_url, "/api/v1/auth/me", headers=headers, timeout=timeout)
    actions.append({"auth": auth_checks})

    provider = _provider_check(base_url, headers, timeout)
    actions.append({"chat": provider})
    if provider.get("status") == "provider_missing":
        notes.append("No provider key configured; chat verification skipped.")

    first_value_ok = True
    if first_value:
        first_value_result = _first_value_check(base_url, headers, timeout)
        first_value_ok = bool(first_value_result.get("ok"))
        actions.append({"first_value": first_value_result})

    webui_ok = True
    if profile.includes_webui and webui_url:
        webui_result = _request("GET", webui_url, "/", timeout=timeout)
        webui_ok = bool(webui_result.get("ok"))
        actions.append({"webui": webui_result})

    endpoints_ok = all(result.get("ok") for result in endpoint_results.values())
    auth_ok = all(result.get("ok") for result in auth_checks.values())
    status = "ok" if endpoints_ok and auth_ok and first_value_ok and webui_ok else "error"
    if not endpoints_ok:
        notes.append("One or more API endpoints failed checks.")
    if not auth_ok:
        notes.append("Authentication verification failed.")
    if first_value and not first_value_ok:
        notes.append("First-value ingest/search verification failed.")
    if not webui_ok:
        notes.append("WebUI verification failed.")

    return {"status": status, "actions": actions, "notes": notes}
