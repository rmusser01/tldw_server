from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpx

from .profiles import SetupProfile
from .utils import env as env_utils

_BODY_PREVIEW_CHARS = 300
_PROVIDER_ENV_EXAMPLES = [
    "OPENAI_API_KEY=sk-...",
    "ANTHROPIC_API_KEY=sk-ant-...",
    "OPENROUTER_API_KEY=sk-or-...",
]
_FIRST_VALUE_TITLE = "tldw onboarding verification"
_FIRST_VALUE_UNIQUE_PHRASE = "tldw-onboarding-verification-unique"
_FIRST_VALUE_SAMPLE = (
    f"# {_FIRST_VALUE_TITLE}\n\n" f"This sample verifies ingest and search with {_FIRST_VALUE_UNIQUE_PHRASE}.\n"
)


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _sanitize_url_userinfo(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.netloc or "@" not in parsed.netloc:
        return value
    _, hostinfo = parsed.netloc.rsplit("@", 1)
    return SplitResult(parsed.scheme, hostinfo, parsed.path, parsed.query, parsed.fragment).geturl()


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


def _response_summary(response: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "url": _sanitize_url_userinfo(response.get("url")),
        "status_code": response.get("status_code"),
        "ok": bool(response.get("ok")),
    }
    if "error" in response:
        summary["error"] = "request_failed"
    return summary


def _auth_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    summary = _response_summary(action)
    if "status" in action:
        summary["status"] = action["status"]
    return summary


def _search_body_has_sample(body: Any) -> bool:
    needle = _FIRST_VALUE_UNIQUE_PHRASE.lower()

    def contains_sample(value: Any, depth: int = 0) -> bool:
        if depth > 8:
            return False
        if isinstance(value, str):
            return needle in value.lower()
        if isinstance(value, dict):
            return any(contains_sample(item, depth + 1) for item in value.values())
        if isinstance(value, list):
            return any(contains_sample(item, depth + 1) for item in value)
        return False

    if not isinstance(body, (dict, list)):
        return False
    if isinstance(body, dict):
        results = body.get("results")
        if results == [] or results == {}:
            return False
        if isinstance(results, (list, dict)):
            return contains_sample(results)
    return contains_sample(body)


def _search_result_items(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    for key in ("results", "items"):
        values = body.get(key)
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
    return []


def _media_detail_path(result: dict[str, Any]) -> str | None:
    url = result.get("url")
    if isinstance(url, str) and url.startswith("/api/v1/media/"):
        return url
    media_id = result.get("id")
    if isinstance(media_id, int):
        return f"/api/v1/media/{media_id}"
    if isinstance(media_id, str) and media_id.isdecimal():
        return f"/api/v1/media/{media_id}"
    return None


def _detail_body_has_sample(
    base_url: str,
    headers: dict[str, str],
    timeout: float,
    search_body: Any,
) -> tuple[bool, dict[str, Any] | None]:
    for item in _search_result_items(search_body)[:3]:
        path = _media_detail_path(item)
        if not path:
            continue
        detail = _request("GET", base_url, path, headers=headers, timeout=timeout)
        summary = _response_summary(detail)
        if detail.get("ok") and _search_body_has_sample(detail.get("body")):
            return True, summary
        return False, summary
    return False, None


def _headers_for_profile(profile: SetupProfile, env_values: dict[str, str]) -> dict[str, str]:
    if profile.auth_mode != "single_user":
        return {}
    key = (env_values.get("SINGLE_USER_API_KEY") or env_values.get("API_KEY") or "").strip()
    if not key:
        return {}
    return {"X-API-KEY": key}


def _env_values_for_profile(env_path: Path) -> dict[str, str]:
    env_values = env_utils.load_env(env_path)
    env_values.update(os.environ)
    return env_values


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
    summary = _response_summary(response)
    action = {
        "status": "ok" if response.get("ok") and token else "error",
        "ok": bool(response.get("ok") and token),
        "url": summary.get("url"),
        "status_code": summary.get("status_code"),
    }
    if not response.get("ok"):
        action["error"] = summary.get("error", "login_failed")
    if not token and response.get("ok"):
        action["error"] = "login response did not include an access token"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return action, headers


def _configured_provider_count(body: Any) -> int:
    if not isinstance(body, dict):
        return 0
    providers = body.get("providers")
    provider_entries: list[Any] = []
    if isinstance(providers, list):
        provider_entries = providers
    elif isinstance(providers, dict):
        provider_entries = list(providers.values())

    has_config_status = any(isinstance(provider, dict) and "is_configured" in provider for provider in provider_entries)
    if has_config_status:
        return sum(
            1 for provider in provider_entries if isinstance(provider, dict) and bool(provider.get("is_configured"))
        )

    total = body.get("total_configured")
    if isinstance(total, int):
        return total
    if provider_entries:
        return len(provider_entries)
    return 0


def _provider_check(base_url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    response = _request("GET", base_url, "/api/v1/llm/providers", headers=headers, timeout=timeout)
    result: dict[str, Any] = {
        "url": _sanitize_url_userinfo(response.get("url")),
        "status_code": response.get("status_code"),
    }
    if not response.get("ok"):
        result.update({"status": "endpoint_failed", "ok": False, "error": "request_failed"})
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
    sample = _FIRST_VALUE_SAMPLE.encode("utf-8")
    ingest = _request(
        "POST",
        base_url,
        "/api/v1/media/add",
        headers=headers,
        data={
            "media_type": "document",
            "title": _FIRST_VALUE_TITLE,
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
        json_body={"query": _FIRST_VALUE_UNIQUE_PHRASE, "fields": ["title", "content"]},
        timeout=timeout,
    )
    matched = bool(search.get("ok") and _search_body_has_sample(search.get("body")))
    detail_summary = None
    if not matched and search.get("ok"):
        matched, detail_summary = _detail_body_has_sample(base_url, headers, timeout, search.get("body"))
    search_summary = _response_summary(search)
    search_summary["matched"] = matched
    details = {"ingest": _response_summary(ingest), "search": search_summary}
    if detail_summary is not None:
        details["detail"] = detail_summary
    return {
        "ingest": "ok" if ingest.get("ok") else "error",
        "search": "ok" if matched else "error",
        "ok": bool(ingest.get("ok") and matched),
        "details": details,
    }


def run_profile_checks(
    *,
    profile: SetupProfile,
    base_url: str,
    webui_url: str | None,
    env_path: Path,
    first_value: bool,
    check_provider: bool,
    timeout: float = 5.0,
) -> dict[str, Any]:
    env_values = _env_values_for_profile(env_path)
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
    actions.append({"endpoints": {key: _response_summary(value) for key, value in endpoint_results.items()}})

    headers = _headers_for_profile(profile, env_values)
    auth_checks: dict[str, Any] = {}
    if profile.auth_mode == "multi_user":
        login_action, login_headers = _login_multi_user(base_url, env_values, timeout)
        auth_checks["login"] = login_action
        headers = login_headers
    auth_checks["me"] = _request("GET", base_url, "/api/v1/auth/me", headers=headers, timeout=timeout)
    actions.append({"auth": {key: _auth_action_summary(value) for key, value in auth_checks.items()}})

    provider_ok = True
    if check_provider:
        provider = _provider_check(base_url, headers, timeout)
        provider_ok = bool(provider.get("ok"))
        actions.append({"chat": provider})
        if provider.get("status") == "provider_missing":
            notes.append("No provider key configured; chat verification skipped.")
        elif provider.get("status") == "endpoint_failed":
            notes.append("Provider endpoint failed during verification.")
    else:
        notes.append("Provider verification skipped; pass --check-provider to check chat provider configuration.")

    first_value_ok = True
    if first_value:
        first_value_result = _first_value_check(base_url, headers, timeout)
        first_value_ok = bool(first_value_result.get("ok"))
        actions.append({"first_value": first_value_result})

    webui_ok = True
    if profile.includes_webui and webui_url:
        webui_result = _request("GET", webui_url, "/", timeout=timeout)
        webui_ok = bool(webui_result.get("ok"))
        actions.append({"webui": _response_summary(webui_result)})

    endpoints_ok = all(result.get("ok") for result in endpoint_results.values())
    auth_ok = all(result.get("ok") for result in auth_checks.values())
    status = "ok" if endpoints_ok and auth_ok and provider_ok and first_value_ok and webui_ok else "error"
    if not endpoints_ok:
        notes.append("One or more API endpoints failed checks.")
    if not auth_ok:
        notes.append("Authentication verification failed.")
    if first_value and not first_value_ok:
        notes.append("First-value ingest/search verification failed.")
    if not webui_ok:
        notes.append("WebUI verification failed.")

    return {"status": status, "actions": actions, "notes": notes}
