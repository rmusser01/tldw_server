"""Webhook and notification adapters.

This module includes adapters for webhook and notification operations:
- webhook: Send HTTP webhooks
- notify: Send notifications (Slack/webhook compatible)
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.http_client import create_client as _wf_create_client
from tldw_Server_API.app.core.Security.egress import is_url_allowed, is_url_allowed_for_tenant
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import NotifyConfig, WebhookConfig

_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


def _adapter_error_message(operation: str, exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return f"{operation} timed out"
    return f"{operation} failed"


@registry.register(
    "notify",
    category="integration",
    description="Send notifications",
    parallelizable=True,
    tags=["integration", "notification"],
    config_model=NotifyConfig,
)
async def run_notify_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Send a notification via webhook (Slack/email-compatible JSON).

    Config:
      - url: http(s) webhook URL
      - message: str (templated)
      - subject: str (optional)
      - headers: dict (optional extra headers)

    Output: { dispatched: bool, status_code?, provider?: 'slack'|'webhook' }
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    msg_t = str(config.get("message") or "").strip()
    message = _tmpl(msg_t, context) or msg_t
    subject = str(config.get("subject") or "").strip() or None
    url = str(config.get("url") or "").strip()
    extra_headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}

    if not (url.startswith("http://") or url.startswith("https://")):
        return {"error": "invalid_url"}

    if is_test_mode():
        return {"dispatched": False, "test_mode": True}

    try:
        tenant_id = str(context.get("tenant_id") or "default") if isinstance(context, dict) else "default"
        ok = False
        try:
            ok = is_url_allowed_for_tenant(url, tenant_id)
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
            ok = is_url_allowed(url)

        if not ok:
            return {"dispatched": False, "error": "blocked_egress"}

        headers = {"content-type": "application/json"}
        with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
            headers.update({k: str(v) for k, v in extra_headers.items()})

        body = {"text": message}
        if subject:
            body["subject"] = subject

        timeout = float(os.getenv("WORKFLOWS_NOTIFY_TIMEOUT", "10"))
        with _wf_create_client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=headers)
            ok = 200 <= resp.status_code < 300

        host = urlparse(url).hostname or ""
        prov = "slack" if "slack" in host else "webhook"
        return {"dispatched": ok, "status_code": resp.status_code, "provider": prov}

    except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS as e:
        return {"dispatched": False, "error": _adapter_error_message("Notification dispatch", e)}


@registry.register(
    "webhook",
    category="integration",
    description="Send webhooks",
    parallelizable=True,
    tags=["integration", "webhook"],
    config_model=WebhookConfig,
)
async def run_webhook_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Send an HTTP request (with safe egress) or dispatch a local webhook event.

    Config (HTTP mode when 'url' provided):
      - url: str (templated)
      - method: str = POST (GET|POST|PUT|PATCH|DELETE)
      - headers: dict[str,str] (templated values)
      - body: dict|list|str|number|bool|null - request JSON body (supports simple JSON-path injection)
        Special string values are supported to inject JSON from context:
          - 'JSON:inputs.qa_samples'  => replaces with context['inputs']['qa_samples'] (not a string)
          - 'JSON:prev.response_json.items|pluck:id' => list of id fields from previous step response
      - timeout_seconds: int (default: 10)

    Config (local webhook mode when no 'url' provided):
      - event: str (default 'workflow.event')
      - data: dict (templated minimal)

    Output keys:
      - dispatched: bool
      - status_code: int (HTTP mode)
      - response_json: any (when response is JSON)
      - response_text: str (when response not JSON)
      - error: str (on failure)
    """
    def _render_value(v: Any) -> Any:
        """Render strings via prompt templating; recurse into lists/dicts."""
        from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
        if isinstance(v, str):
            try:
                return _tmpl(v, context)
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                return v
        if isinstance(v, list):
            return [_render_value(x) for x in v]
        if isinstance(v, dict):
            return {k: _render_value(val) for k, val in v.items()}
        return v

    def _resolve_json_ref(expr: str) -> Any:
        """Resolve a limited JSON reference like 'inputs.qa_samples' or 'prev.response_json.items|pluck:id'."""
        path = expr
        pluck_field: str | None = None
        # Support '|pluck:field'
        if "|pluck:" in path:
            path, tail = path.split("|pluck:", 1)
            pluck_field = tail.strip()
        # Walk dotted path from context root
        cur: Any = context
        for part in [p for p in path.strip().split(".") if p]:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                try:
                    cur = getattr(cur, part)
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    cur = None
                    break
        # Optional pluck across list of dicts
        if pluck_field and isinstance(cur, list):
            out = []
            for item in cur:
                try:
                    if isinstance(item, dict) and pluck_field in item:
                        out.append(item[pluck_field])
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    continue
            cur = out
        return cur

    def _inject_json_specials(obj: Any) -> Any:
        """Traverse obj and replace strings starting with 'JSON:' with referenced JSON from context."""
        if isinstance(obj, str):
            if obj.strip().lower().startswith("json:"):
                ref = obj.split(":", 1)[1].strip()
                return _resolve_json_ref(ref)
            return obj
        if isinstance(obj, list):
            return [_inject_json_specials(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _inject_json_specials(v) for k, v in obj.items()}
        return obj

    def _normalize_policy_hosts(entries: Any) -> list[str]:
        out: list[str] = []
        if not entries:
            return out
        if isinstance(entries, str):
            entries = [entries]
        for raw in entries:
            if raw is None:
                continue
            entry = str(raw).strip().lower()
            if not entry:
                continue
            host = ""
            if "://" in entry:
                try:
                    host = urlparse(entry).hostname or ""
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    host = ""
            else:
                # Strip path if present
                if "/" in entry:
                    entry = entry.split("/", 1)[0]
                host = entry
            host = host.strip()
            if host.startswith("*."):
                host = host[2:]
            if host.startswith("."):
                host = host[1:]
            if host.count(":") == 1 and host.rsplit(":", 1)[-1].isdigit():
                host = host.rsplit(":", 1)[0]
            if host:
                out.append(host)
        return out

    def _resolve_signing_secret(ref: str) -> str | None:
        if not ref:
            return None
        try:
            secrets = context.get("secrets") if isinstance(context, dict) else None
            if isinstance(secrets, dict) and ref in secrets:
                return str(secrets.get(ref))
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            val = os.getenv(ref, "")
            return val if val else None
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
            return None

    def _policy_allows(url_val: str) -> tuple[bool, str | None]:
        policy_cfg = config.get("egress_policy") or config.get("egress") or {}
        if not isinstance(policy_cfg, dict) or not policy_cfg:
            return True, None
        allowlist = _normalize_policy_hosts(policy_cfg.get("allowlist") or policy_cfg.get("allow") or [])
        denylist = _normalize_policy_hosts(policy_cfg.get("denylist") or policy_cfg.get("deny") or [])
        block_private = policy_cfg.get("block_private")
        try:
            from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
            result = evaluate_url_policy(
                url_val,
                allowlist=allowlist or None,
                denylist=denylist or None,
                block_private_override=block_private if isinstance(block_private, bool) else None,
            )
            return result.allowed, result.reason
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
            return False, "policy_error"

    def _record_blocked(url_val: str) -> None:
        try:
            host = urlparse(url_val).hostname or ""
            from tldw_Server_API.app.core.Metrics import increment_counter as _inc
            _inc("workflows_webhook_deliveries_total", labels={"status": "blocked", "host": host})
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
            pass

    from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookEvent, webhook_manager

    user_id = resolve_context_user_id(context)
    if not user_id:
        return {"dispatched": False, "error": "missing_user_id"}

    event_name = str(config.get("event") or "workflow.event")
    payload = config.get("data") or {"context": list(context.keys())}
    url = str(config.get("url") or "").strip()

    if is_test_mode():
        # Skip outbound work in tests
        return {"dispatched": False, "test_mode": True}

    if url:
        tenant_id = str(context.get("tenant_id") or "default")

        def _global_allows(url_val: str) -> bool:
            try:
                from tldw_Server_API.app.core.Security.egress import is_webhook_url_allowed_for_tenant
                return is_webhook_url_allowed_for_tenant(url_val, tenant_id)
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                return is_url_allowed(url_val)

        try:
            # Method, headers, timeout
            method = str(config.get("method") or "POST").upper()
            headers_cfg = config.get("headers") or {}
            # Templating for url and headers
            url_t = _render_value(url) or url
            url_t = str(url_t).strip()
            if not url_t:
                return {"dispatched": False, "error": "missing_url"}
            headers_r: dict[str, str] = {}
            if isinstance(headers_cfg, dict):
                for hk, hv in headers_cfg.items():
                    try:
                        headers_r[str(hk)] = str(_render_value(hv))
                    except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                        headers_r[str(hk)] = str(hv)
            # Drop empty headers (avoid sending empty Authorization/X-API-KEY)
            with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
                headers_r = {k: v for k, v in headers_r.items() if isinstance(v, str) and v.strip()}
            # If no explicit auth headers provided, allow secrets from workflow run to supply them
            try:
                secrets = context.get("secrets") if isinstance(context, dict) else None
                if isinstance(secrets, dict):
                    has_auth = any(k.lower() == "authorization" for k in headers_r) or any(k.lower() == "x-api-key" for k in headers_r)
                    if not has_auth:
                        _jwt = secrets.get("jwt") or secrets.get("bearer")
                        _api = secrets.get("api_key") or secrets.get("x_api_key")
                        if _jwt:
                            headers_r["Authorization"] = f"Bearer {_jwt}"
                        elif _api:
                            headers_r["X-API-KEY"] = str(_api)
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                pass
            # Ensure content-type unless provided
            if "content-type" not in {k.lower(): v for k, v in headers_r.items()}:
                headers_r["Content-Type"] = "application/json"
            # Per-step allow/deny policy
            try:
                step_allowed, reason = _policy_allows(url_t)
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                step_allowed, reason = (False, "policy_error")
            if not _global_allows(url_t) or not step_allowed:
                _record_blocked(url_t)
                return {"dispatched": False, "error": "blocked_egress", "reason": reason}
            # Default auth fallbacks for scheduled runs (optional)
            try:
                _had_auth = any(k.lower() == "authorization" for k in headers_r) or any(k.lower() == "x-api-key" for k in headers_r)
                used_fallback = False
                if not _had_auth:
                    _bear = os.getenv("WORKFLOWS_DEFAULT_BEARER_TOKEN", "").strip()
                    _key = os.getenv("WORKFLOWS_DEFAULT_API_KEY", "").strip()
                    if _bear:
                        headers_r["Authorization"] = f"Bearer {_bear}"
                        used_fallback = True
                    elif _key:
                        headers_r["X-API-KEY"] = _key
                        used_fallback = True
                # Optional sanity check for fallback auth (once per run)
                try:
                    if used_fallback and env_flag_enabled("WORKFLOWS_VALIDATE_DEFAULT_AUTH") and not context.get("_wf_default_auth_checked"):
                        base = os.getenv("WORKFLOWS_INTERNAL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
                        _url = f"{base}/api/v1/workflows/auth/check"
                        with _wf_create_client(timeout=5.0, trust_env=False) as _client:
                            _resp = _client.get(_url, headers=headers_r)
                            if _resp.status_code // 100 != 2:
                                return {"dispatched": False, "error": "default_auth_validation_failed", "status_code": _resp.status_code}
                        context["_wf_default_auth_checked"] = True
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    # Non-fatal; allow the request to proceed
                    pass
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                pass
            # Render and prepare body
            body_raw = config.get("body") if ("body" in config) else (config.get("data") if ("data" in config) else None)
            body_r = _render_value(body_raw) if body_raw is not None else None
            body_r = _inject_json_specials(body_r)
            # Inject W3C trace context
            try:
                from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager as _get_tm
                _get_tm().inject_context(headers_r)
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                pass
            secret = os.getenv("WORKFLOWS_WEBHOOK_SECRET", "")
            body_json_str = None
            # Prepare request kwargs
            req_kwargs: dict[str, Any] = {}
            if method == "GET":
                if isinstance(body_r, dict):
                    req_kwargs["params"] = body_r
                elif body_r is not None:
                    # Non-dict body for GET - ignore
                    pass
            else:
                if body_r is not None:
                    # Use JSON body if not already a string
                    req_kwargs["content"] = json.dumps(body_r)
                    body_json_str = req_kwargs["content"]
                else:
                    req_kwargs["content"] = json.dumps(payload)
                    body_json_str = req_kwargs["content"]
            # Optional per-step signing config overrides
            try:
                signing_cfg = config.get("signing")
                if signing_cfg is False or str(signing_cfg).lower() in {"0", "false", "none", "off"}:
                    secret = ""
                elif isinstance(signing_cfg, dict):
                    stype = str(signing_cfg.get("type") or "hmac-sha256").lower()
                    if stype in {"none", "off"}:
                        secret = ""
                    elif stype not in {"hmac-sha256", "hmac_sha256", "hmacsha256"}:
                        return {"dispatched": False, "error": "unsupported_signing_type"}
                    else:
                        sref = str(signing_cfg.get("secret_ref") or "").strip()
                        sdirect = signing_cfg.get("secret")
                        if sref:
                            secret = _resolve_signing_secret(sref) or ""
                        elif sdirect:
                            secret = str(sdirect)
                        if not secret:
                            return {"dispatched": False, "error": "missing_signing_secret"}
                elif signing_cfg:
                    # Truthy non-dict => fall back to env secret if available
                    secret = os.getenv("WORKFLOWS_WEBHOOK_SECRET", "")
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                pass
            if secret:
                sig = hmac.new(secret.encode("utf-8"), (body_json_str or "").encode("utf-8"), hashlib.sha256).hexdigest()
                headers_r["X-Workflows-Signature"] = sig
                headers_r["X-Hub-Signature-256"] = f"sha256={sig}"

            timeout_val = float(config.get("timeout_seconds") or os.getenv("WORKFLOWS_WEBHOOK_TIMEOUT", "10"))
            follow_redirects = bool(config.get("follow_redirects") or config.get("allow_redirects") or False)
            try:
                max_redirects = int(config.get("max_redirects") or os.getenv("HTTP_MAX_REDIRECTS", "5"))
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                max_redirects = int(os.getenv("HTTP_MAX_REDIRECTS", "5"))
            try:
                max_bytes = int(config.get("max_bytes")) if config.get("max_bytes") is not None else None
            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                max_bytes = None
            try:
                client_ctx = _wf_create_client(timeout=timeout_val, trust_env=False)
            except TypeError:
                client_ctx = _wf_create_client(timeout=timeout_val)
            with client_ctx as client:
                # Dispatch with explicit redirect handling
                cur_url = url_t
                resp = None
                redirects = 0
                method_cur = method
                req_kwargs_cur = dict(req_kwargs)
                while True:
                    if not _global_allows(cur_url):
                        _record_blocked(cur_url)
                        return {"dispatched": False, "error": "blocked_egress"}
                    step_allowed, reason = _policy_allows(cur_url)
                    if not step_allowed:
                        _record_blocked(cur_url)
                        return {"dispatched": False, "error": "blocked_egress", "reason": reason}
                    resp = client.request(method_cur, cur_url, headers=headers_r, follow_redirects=False, **req_kwargs_cur)
                    if not follow_redirects or resp.status_code not in (301, 302, 303, 307, 308):
                        break
                    location = resp.headers.get("location")
                    with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
                        resp.close()
                    if not location:
                        return {"dispatched": False, "error": "redirect_missing_location"}
                    redirects += 1
                    if redirects > max_redirects:
                        return {"dispatched": False, "error": "redirects_exceeded"}
                    try:
                        from urllib.parse import urljoin
                        cur_url = urljoin(cur_url, location)
                    except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                        cur_url = location
                    if resp.status_code in (301, 302, 303) and method_cur not in ("GET", "HEAD"):
                        method_cur = "GET"
                        req_kwargs_cur = {k: v for k, v in req_kwargs_cur.items() if k == "params"}
                if resp is None:
                    return {"dispatched": False, "error": "webhook_no_response"}

                # Optional response size guard
                def _read_response_bytes(r) -> bytes:
                    if max_bytes is not None:
                        clen = r.headers.get("content-length")
                        if clen:
                            try:
                                if int(clen) > max_bytes:
                                    with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
                                        r.close()
                                    raise ValueError("response_too_large")
                            except ValueError:
                                raise
                            except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                                pass
                    buf = bytearray()
                    if max_bytes is None:
                        try:
                            data = r.read()
                            return data if data is not None else b""
                        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                            return b""
                        finally:
                            with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
                                r.close()
                    try:
                        for chunk in r.iter_bytes():
                            buf.extend(chunk)
                            if max_bytes is not None and len(buf) > max_bytes:
                                raise ValueError("response_too_large")
                    finally:
                        with contextlib.suppress(_WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS):
                            r.close()
                    return bytes(buf)

                ok = 200 <= resp.status_code < 300
                # Metrics for success/failure
                try:
                    host = urlparse(cur_url).hostname or ""
                    from tldw_Server_API.app.core.Metrics import increment_counter as _inc
                    _inc("workflows_webhook_deliveries_total", labels={"status": ("delivered" if ok else "failed"), "host": host})
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    pass
                # Optional artifact of response metadata
                try:
                    if callable(context.get("add_artifact")):
                        step_run_id = str(context.get("step_run_id") or "")
                        art_dir = resolve_artifacts_dir(step_run_id or f"webhook_{int(time.time()*1000)}")
                        art_dir.mkdir(parents=True, exist_ok=True)
                        fpath = art_dir / "webhook_response.json"
                        data = {"status_code": resp.status_code, "headers": dict(resp.headers)}
                        fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
                        context["add_artifact"](
                            type="webhook_response",
                            uri=f"file://{fpath}",
                            size_bytes=len(fpath.read_bytes() if fpath.exists() else b""),
                            mime_type="application/json",
                            metadata={"url": url},
                        )
                        # Optionally save response body for diagnostics
                        try:
                            if bool(config.get("save_response_json")) or bool(config.get("save_response_body")):
                                body_path = art_dir / "webhook_response_body.json"
                                body_mime = "application/json"
                                try:
                                    body_text = resp.text
                                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                                    body_text = ""
                                # Pretty print JSON when possible
                                try:
                                    parsed = resp.json()
                                    body_text = json.dumps(parsed, indent=2)
                                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                                    # keep as text/plain when not JSON
                                    body_mime = "text/plain"
                                body_path.write_text(body_text, encoding="utf-8")
                                context["add_artifact"](
                                    type="webhook_response_body",
                                    uri=f"file://{body_path}",
                                    size_bytes=len(body_path.read_bytes() if body_path.exists() else b""),
                                    mime_type=body_mime,
                                    metadata={"url": url},
                                )
                        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                            pass
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    pass
                # Build outputs
                out: dict[str, Any] = {"dispatched": ok, "status_code": resp.status_code}
                try:
                    body_bytes = _read_response_bytes(resp)
                except ValueError:
                    out["dispatched"] = False
                    out["error"] = "response_too_large"
                    return out
                try:
                    enc = resp.encoding or "utf-8"
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    enc = "utf-8"
                try:
                    text = body_bytes.decode(enc, errors="replace")
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    text = ""
                try:
                    out["response_json"] = json.loads(text) if text else None
                except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
                    if text:
                        out["response_text"] = text
                return out
        except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS as e:
            return {"dispatched": False, "error": _adapter_error_message("Workflow webhook dispatch", e)}

    # Default: use registered webhooks
    try:
        event = WebhookEvent(event_name)  # type: ignore[arg-type]
    except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS:
        event = WebhookEvent.EVALUATION_PROGRESS
    try:
        await webhook_manager.send_webhook(user_id=user_id, event=event, evaluation_id="workflow", data=payload)
        return {"dispatched": True}
    except _WORKFLOWS_WEBHOOK_NONCRITICAL_EXCEPTIONS as e:
        return {"dispatched": False, "error": _adapter_error_message("Workflow webhook dispatch", e)}
