#!/usr/bin/env python3
"""Emit or run ACP downstream-agent certification smoke manifests.

The manifest reuses existing ACP test suites and runner checks. It does not
invent a second compatibility test framework; it gives contributors one stable
place to find the commands and capability IDs needed for matrix evidence.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import queue
import shlex
import signal
# subprocess is intentionally used to run static manifest argv with shell=False.
import subprocess  # nosec B404
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_STDOUT_EOF = object()
_ERROR_MESSAGE_LIMIT = 240
_STDOUT_LINE_LIMIT = 64 * 1024
_STDOUT_QUEUE_MAXSIZE = 32
_SESSION_ID_PLACEHOLDER = "${session_id}"
_FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
_WORKSPACE_LIVE_E2E_PROMPT = (
    "Create a concise markdown workspace certification artifact with a title, "
    "one evidence bullet, and the literal marker TLDW_WORKSPACE_ACP_CERTIFIED."
)


_MANIFESTS: dict[str, dict[str, Any]] = {
    "stub-smoke": {
        "profile": "stub-smoke",
        "support_state": "supported_with_caveats",
        "verification_level": "stub_smoke_tested",
        "requires_live_agent": False,
        "required_environment": [],
        "notes": [
            "Uses in-repo stub/mocked ACP paths only.",
            "Does not certify a live Codex, Claude Code, OpenCode, or custom agent binary.",
        ],
        "commands": [
            {
                "id": "backend_acp_smoke",
                "description": "Focused backend ACP lifecycle, diagnostics, retention/redaction, and orchestration smoke checks.",
                "cwd": ".",
                "argv": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tldw_Server_API/tests/Agent_Client_Protocol/test_acp_e2e_smoke.py",
                    "tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py",
                    "tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_controls.py",
                    "tldw_Server_API/tests/Agent_Client_Protocol/test_acp_run_handler.py",
                    "tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py",
                    "-q",
                ],
                "capabilities": [
                    "init",
                    "session_new",
                    "prompt",
                    "structured_completion",
                    "diagnostics",
                    "cancel_close",
                    "artifacts",
                    "redacted_support_view",
                    "review_loop",
                ],
                "safe_to_run_by_default": True,
            },
            {
                "id": "go_runner_verify",
                "description": "Build and test the in-repo tldw-agent runner binaries.",
                "cwd": "tools/tldw-agent",
                "argv": ["./scripts/verify-local-build.sh"],
                "capabilities": ["init", "session_new", "prompt", "cancel_close"],
                "safe_to_run_by_default": True,
            },
            {
                "id": "browser_mocked_setup_run_diagnose",
                "description": "Mocked browser ACP setup/run/diagnose flow for Agent Tasks.",
                "cwd": "apps/tldw-frontend",
                "argv": [
                    "./node_modules/.bin/playwright",
                    "test",
                    "e2e/workflows/tier-3-automation/agent-tasks.spec.ts",
                    "--grep",
                    "guide ACP setup",
                    "--reporter=line",
                ],
                "env": {
                    "TLDW_WEB_URL": "http://localhost:18080",
                    "TLDW_WEB_CMD": "bun run dev -- -p 18080",
                },
                "capabilities": [
                    "diagnostics",
                    "artifacts",
                    "review_loop",
                    "workspace_env",
                    "mcp_injection",
                ],
                "safe_to_run_by_default": True,
                "optional": True,
            },
        ],
    },
    "live-e2e": {
        "profile": "live-e2e",
        "support_state": "supported_with_caveats",
        "verification_level": "live_e2e_tested",
        "requires_live_agent": True,
        "required_environment": [
            "TLDW_E2E_SERVER_URL",
            "TLDW_E2E_API_KEY",
            "ACP_AGENT_PROFILE",
        ],
        "notes": [
            "Requires a running backend, API key, configured ACP runner profile, installed downstream agent binary, and provider credentials.",
            "Use this only for named live-agent support claims.",
        ],
        "commands": [
            {
                "id": "live_backend_acp_e2e",
                "description": "Live backend ACP API flow against a configured downstream agent profile.",
                "cwd": ".",
                "argv": [
                    sys.executable,
                    "Helper_Scripts/Testing-related/acp_certification_smoke.py",
                    "--backend-live-e2e",
                ],
                "env": {
                    "TLDW_E2E_SERVER_URL": "${TLDW_E2E_SERVER_URL}",
                    "TLDW_E2E_API_KEY": "${TLDW_E2E_API_KEY}",
                    "ACP_AGENT_PROFILE": "${ACP_AGENT_PROFILE}",
                },
                "capabilities": [
                    "init",
                    "session_new",
                    "prompt",
                    "structured_completion",
                    "artifacts",
                    "diagnostics",
                    "cancel_close",
                    "review_loop",
                    "workspace_env",
                    "mcp_injection",
                    "redacted_support_view",
                ],
                "safe_to_run_by_default": False,
            },
            {
                "id": "live_runner_verify",
                "description": "Verify the local runner build before attributing failures to the downstream agent.",
                "cwd": "tools/tldw-agent",
                "argv": ["./scripts/verify-local-build.sh"],
                "capabilities": ["init", "session_new", "prompt", "cancel_close"],
                "safe_to_run_by_default": False,
            },
        ],
    },
    "workspace-live-e2e": {
        "profile": "workspace-live-e2e",
        "support_state": "supported_with_caveats",
        "verification_level": "live_e2e_tested",
        "requires_live_agent": True,
        "required_environment": [
            "TLDW_E2E_SERVER_URL",
            "TLDW_E2E_API_KEY",
            "ACP_AGENT_PROFILE",
            "ACP_E2E_WORKSPACE_ID",
        ],
        "optional_environment": [
            "ACP_E2E_WORKSPACE_CWD",
            "ACP_E2E_MCP_SERVER_NAME",
            "ACP_E2E_MCP_SERVER_COMMAND",
            "ACP_E2E_MCP_SERVER_ARGS_JSON",
            "ACP_E2E_EXPECT_SANDBOX",
            "ACP_E2E_EXPECT_ARTIFACTS",
            "ACP_E2E_EXPECT_REVIEWER_LOOP",
        ],
        "notes": [
            "Requires a running backend, API key, configured ACP runner profile, installed downstream agent binary, provider credentials, and a workspace id to certify.",
            "Extends live-e2e with Research Workspace session context, non-empty MCP server injection, artifacts, diagnostics, reviewer-loop, and sandbox evidence checks.",
            "Unavailable optional capabilities are reported as skipped unless the matching ACP_E2E_EXPECT_* flag requires them.",
        ],
        "commands": [
            {
                "id": "workspace_live_backend_acp_e2e",
                "description": "Live Research Workspace ACP REST flow against a configured downstream agent profile.",
                "cwd": ".",
                "argv": [
                    sys.executable,
                    "Helper_Scripts/Testing-related/acp_certification_smoke.py",
                    "--backend-workspace-live-e2e",
                ],
                "env": {
                    "TLDW_E2E_SERVER_URL": "${TLDW_E2E_SERVER_URL}",
                    "TLDW_E2E_API_KEY": "${TLDW_E2E_API_KEY}",
                    "ACP_AGENT_PROFILE": "${ACP_AGENT_PROFILE}",
                },
                "capabilities": [
                    "init",
                    "session_new",
                    "prompt",
                    "structured_completion",
                    "artifacts",
                    "diagnostics",
                    "cancel_close",
                    "review_loop",
                    "workspace_env",
                    "mcp_injection",
                    "sandbox",
                    "redacted_support_view",
                ],
                "safe_to_run_by_default": False,
            },
        ],
    },
}


def build_manifest(profile: str) -> dict[str, Any]:
    """Return a copy of the certification manifest for a profile."""
    if profile not in _MANIFESTS:
        raise ValueError(f"Unknown ACP certification profile: {profile}")
    return deepcopy(_MANIFESTS[profile])


def build_agent_profile_manifest(entrypoint: dict[str, Any]) -> dict[str, Any]:
    """Return a profile-specific ACP certification probe manifest."""
    profile = str(entrypoint.get("type") or entrypoint.get("profile_key") or "")
    if not profile:
        raise ValueError("Agent profile manifest requires a profile type")

    blockers = list(entrypoint.get("blockers") or [])
    primary_blocker = entrypoint.get("primary_blocker")
    if entrypoint.get("probe_state") == "custom_template" and not primary_blocker:
        primary_blocker = "custom_template"
    if primary_blocker and primary_blocker not in blockers:
        blockers.insert(0, str(primary_blocker))

    manifest: dict[str, Any] = {
        "profile": profile,
        "name": entrypoint.get("name") or profile,
        "support_state": entrypoint.get("support_state") or "documented_unverified",
        "verification_level": entrypoint.get("verification_level") or "documented_only",
        "requires_live_agent": True,
        "required_environment": [],
        "entrypoint": {
            "entrypoint_strategy": entrypoint.get("entrypoint_strategy"),
            "probe_state": entrypoint.get("probe_state"),
            "acp_command": entrypoint.get("acp_command") or "",
            "acp_args": list(entrypoint.get("acp_args") or []),
            "primary_blocker": primary_blocker,
            "status_message": entrypoint.get("status_message") or "",
            "docs_url": entrypoint.get("docs_url"),
            "adapter_source": entrypoint.get("adapter_source"),
            "adapter_docs_url": entrypoint.get("adapter_docs_url"),
            "adapter_package": entrypoint.get("adapter_package"),
            "adapter_version": entrypoint.get("adapter_version"),
            "adapter_version_policy": entrypoint.get("adapter_version_policy"),
            "adapter_install_source": entrypoint.get("adapter_install_source"),
            "credential_policy": entrypoint.get("credential_policy"),
            "runtime_backend": entrypoint.get("runtime_backend"),
        },
        "blockers": blockers,
        "notes": [
            str(entrypoint.get("status_message") or "Profile-specific ACP certification manifest."),
            "Requires operator-provided runtime state and an installed downstream ACP entrypoint.",
        ],
        "commands": [],
    }

    if entrypoint.get("probe_state") == "ready_to_probe":
        manifest["commands"].append(
            {
                "id": "acp_initialize_probe",
                "description": "Bounded ACP initialize probe for the selected downstream entrypoint.",
                "cwd": ".",
                "argv": [
                    str(entrypoint["acp_command"]),
                    *[str(arg) for arg in entrypoint.get("acp_args", [])],
                ],
                "stdin_jsonl": [
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": 1,
                            "clientInfo": {
                                "name": "tldw-server-certification-smoke",
                                "version": "0",
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(ROOT), "mcpServers": []},
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": _SESSION_ID_PLACEHOLDER,
                            "prompt": [
                                {
                                    "type": "text",
                                    "text": "Reply with a short ACP certification acknowledgement.",
                                }
                            ],
                        },
                    },
                ],
                "timeout_seconds": 120,
                "capabilities": ["init", "session_new", "prompt"],
                "safe_to_run_by_default": False,
            }
        )

    return manifest


def render_manifest_dict(
    manifest: dict[str, Any],
    *,
    output_format: str = "markdown",
) -> str:
    """Render a certification manifest dictionary as JSON or Markdown."""
    if output_format == "json":
        return json.dumps(manifest, indent=2, sort_keys=True)
    if output_format != "markdown":
        raise ValueError(f"Unsupported output format: {output_format}")

    lines = [
        f"# ACP Certification Manifest: {manifest['profile']}",
        "",
        f"- support_state: `{manifest['support_state']}`",
        f"- verification_level: `{manifest['verification_level']}`",
        f"- requires_live_agent: `{str(manifest['requires_live_agent']).lower()}`",
    ]
    if manifest["required_environment"]:
        lines.append(f"- required_environment: `{', '.join(manifest['required_environment'])}`")
    if manifest.get("entrypoint"):
        entrypoint = manifest["entrypoint"]
        lines.extend(
            [
                f"- entrypoint_strategy: `{entrypoint.get('entrypoint_strategy')}`",
                f"- probe_state: `{entrypoint.get('probe_state')}`",
                f"- acp_command: `{entrypoint.get('acp_command')}`",
            ]
        )
        if entrypoint.get("acp_args"):
            lines.append(f"- acp_args: `{shlex.join(str(arg) for arg in entrypoint['acp_args'])}`")
    if manifest.get("blockers"):
        lines.append(f"- blockers: `{', '.join(str(blocker) for blocker in manifest['blockers'])}`")
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"- {note}" for note in manifest["notes"])
    if manifest.get("blockers"):
        lines.append("")
        lines.append("## Blockers")
        lines.extend(f"- {blocker}" for blocker in manifest["blockers"])
    lines.append("")
    lines.append("## Commands")
    if not manifest["commands"]:
        lines.append("")
        lines.append("No runnable commands for this manifest.")
        return "\n".join(lines).rstrip() + "\n"
    for command in manifest["commands"]:
        env_prefix = " ".join(
            f"{key}={_quote_shell_env_value(str(value))}"
            for key, value in sorted(command.get("env", {}).items())
        )
        rendered_command = shlex.join(str(arg) for arg in command["argv"])
        if env_prefix:
            rendered_command = f"{env_prefix} {rendered_command}"
        lines.extend(
            [
                f"### {command['id']}",
                "",
                command["description"],
                "",
                f"- cwd: `{command['cwd']}`",
                f"- safe_to_run_by_default: `{str(command['safe_to_run_by_default']).lower()}`",
                f"- optional: `{str(command.get('optional', False)).lower()}`",
                f"- capabilities: `{', '.join(command['capabilities'])}`",
                "",
                "```bash",
                rendered_command,
                "```",
                "",
            ]
        )
        if command.get("stdin_jsonl"):
            lines.extend(
                [
                    "stdin_jsonl:",
                    "",
                    "```jsonl",
                    *[json.dumps(frame, sort_keys=True) for frame in command["stdin_jsonl"]],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_manifest(profile: str, *, output_format: str = "markdown") -> str:
    """Render a certification manifest as JSON or Markdown."""
    return render_manifest_dict(build_manifest(profile), output_format=output_format)


def _quote_shell_env_value(value: str) -> str:
    """Return a shell-safe env value while preserving ${VAR} placeholders."""
    if value.startswith("${") and value.endswith("}"):
        return value
    return shlex.quote(value)


def _command_env(command: dict[str, Any]) -> dict[str, str]:
    """Merge manifest env overrides into the process environment.

    Values of the form `${NAME}` are resolved from the current process
    environment so live-E2E manifests can declare required variables without
    copying secret values into the manifest itself.
    """
    env = os.environ.copy()
    for key, value in command.get("env", {}).items():
        text = str(value)
        if text.startswith("${") and text.endswith("}"):
            env_name = text[2:-1]
            env[key] = os.environ.get(env_name, "")
        else:
            env[key] = text
    return env


def _is_loopback_host(host: str | None) -> bool:
    """Return whether a parsed URL host is local loopback."""
    if not host:
        return False
    normalized = host.strip().strip("[]").rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _allow_insecure_backend_e2e_http() -> bool:
    """Return whether non-local plaintext HTTP is explicitly allowed."""
    return os.environ.get("ACP_BACKEND_E2E_ALLOW_INSECURE_HTTP", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _normalized_server_url() -> str:
    """Return the configured E2E server URL with a scheme and no trailing slash."""
    raw_url = os.environ.get("TLDW_E2E_SERVER_URL", "").strip()
    if not raw_url:
        raise ValueError("TLDW_E2E_SERVER_URL is required")
    if "://" not in raw_url:
        parsed_without_scheme = urllib.parse.urlparse(f"//{raw_url}")
        if not _is_loopback_host(parsed_without_scheme.hostname):
            raise ValueError(
                "TLDW_E2E_SERVER_URL requires an explicit scheme for non-local hosts"
            )
        raw_url = f"http://{raw_url}"

    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("TLDW_E2E_SERVER_URL must use http:// or https://")
    if not parsed.hostname:
        raise ValueError("TLDW_E2E_SERVER_URL must include a host")
    if (
        parsed.scheme == "http"
        and not _is_loopback_host(parsed.hostname)
        and not _allow_insecure_backend_e2e_http()
    ):
        raise ValueError(
            "Refusing to send X-API-KEY over plaintext HTTP to a non-local host; "
            "use https:// or set ACP_BACKEND_E2E_ALLOW_INSECURE_HTTP=1"
        )
    return raw_url.rstrip("/")


def _backend_e2e_timeout_seconds() -> float:
    """Return the backend live-E2E request timeout."""
    raw_timeout = os.environ.get("ACP_BACKEND_E2E_TIMEOUT_SECONDS", "120").strip()
    return float(raw_timeout or "120")


def _http_json_request(
    method: str,
    path: str,
    body: Any = None,
    timeout_seconds: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send one JSON request to the configured backend and return status/payload."""
    url = f"{_normalized_server_url()}{path}"
    request_body: bytes | None = None
    headers = {
        "Accept": "application/json",
        "X-API-KEY": os.environ.get("TLDW_E2E_API_KEY", ""),
    }
    if body is not None:
        request_body = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=request_body,
        headers=headers,
        method=method,
    )
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _backend_e2e_timeout_seconds()
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            payload_bytes = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        payload_bytes = exc.read()
        status = int(exc.code)
    if not payload_bytes:
        return status, {}
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {"raw_preview": payload_bytes[:240].decode("utf-8", errors="replace")}
    return status, payload if isinstance(payload, dict) else {"data": payload}


def _payload_preview(payload: Any) -> str:
    """Return a bounded one-line JSON preview for failure output."""
    try:
        text = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        text = str(payload)
    text = text.replace("\r", " ").replace("\n", " ")
    return text[:240] + ("..." if len(text) > 240 else "")


def _fail_backend_live_e2e(
    step: str,
    status: int,
    payload: Any,
    *,
    label: str = "live_backend_acp_e2e",
) -> int:
    """Print a bounded backend live-E2E failure and return a failure code."""
    print(
        f"FAIL {label}: {step} returned HTTP {status}: "
        f"{_payload_preview(payload)}",
        file=sys.stderr,
    )
    return 1


def _check_backend_response(
    step: str,
    status: int,
    payload: Any,
    *,
    label: str = "live_backend_acp_e2e",
) -> int | None:
    """Return an exit code when a backend response should fail certification."""
    if status >= 400:
        return _fail_backend_live_e2e(step, status, payload, label=label)
    return None


def _run_backend_live_e2e_from_env() -> int:
    """Run the backend ACP REST lifecycle for the configured live agent."""
    required_env = {
        name: os.environ.get(name, "").strip()
        for name in ("TLDW_E2E_SERVER_URL", "TLDW_E2E_API_KEY", "ACP_AGENT_PROFILE")
    }
    missing = [
        name for name, value in required_env.items()
        if not value
    ]
    if missing:
        print(
            "Refusing to run backend live ACP certification without required environment: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    profile = required_env["ACP_AGENT_PROFILE"]
    workspace_cwd = os.environ.get("ACP_E2E_WORKSPACE_CWD", "").strip() or str(ROOT)
    session_id: str | None = None
    timeout_seconds: float | None = None

    def request(method: str, path: str, body: Any = None) -> tuple[int, dict[str, Any]]:
        return _http_json_request(
            method,
            path,
            body,
            timeout_seconds=timeout_seconds,
        )

    try:
        timeout_seconds = _backend_e2e_timeout_seconds()
        for step, method, path, body in (
            ("health", "GET", "/api/v1/acp/health", None),
            (
                "setup-guide",
                "GET",
                "/api/v1/acp/setup-guide?"
                + urllib.parse.urlencode({"agent_type": profile}),
                None,
            ),
        ):
            status, payload = request(method, path, body)
            failed = _check_backend_response(step, status, payload)
            if failed is not None:
                return failed

        new_body = {
            "cwd": workspace_cwd,
            "agent_type": profile,
            "name": f"ACP live E2E {profile}",
            "mcp_servers": [],
        }
        status, payload = request("POST", "/api/v1/acp/sessions/new", new_body)
        failed = _check_backend_response("sessions/new", status, payload)
        if failed is not None:
            return failed
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return _fail_backend_live_e2e("sessions/new", status, {"detail": "missing session_id"})

        prompt_body = {
            "session_id": session_id,
            "prompt": [
                {
                    "type": "text",
                    "text": "Reply with a short ACP backend live E2E certification acknowledgement.",
                }
            ],
        }
        status, prompt_payload = request("POST", "/api/v1/acp/sessions/prompt", prompt_body)
        failed = _check_backend_response("sessions/prompt", status, prompt_payload)
        if failed is not None:
            return failed

        stop_reason = (
            prompt_payload.get("stop_reason")
            or (prompt_payload.get("raw_result") or {}).get("stopReason")
        )
        redacted_paths = [
            ("detail", f"/api/v1/acp/sessions/{session_id}/detail?redacted=true"),
            ("events", f"/api/v1/acp/sessions/{session_id}/events?redacted=true"),
            ("artifacts", f"/api/v1/acp/sessions/{session_id}/artifacts?redacted=true"),
            ("diagnostics", f"/api/v1/acp/sessions/{session_id}/diagnostics"),
        ]
        evidence: dict[str, Any] = {
            "agent_profile": profile,
            "session_id": session_id,
            "stop_reason": stop_reason,
        }
        for step, path in redacted_paths:
            status, payload = request("GET", path, None)
            failed = _check_backend_response(step, status, payload)
            if failed is not None:
                return failed
            if "total" in payload:
                evidence[f"{step}_total"] = payload.get("total")

        status, payload = request(
            "POST",
            "/api/v1/acp/sessions/cancel",
            {"session_id": session_id},
        )
        failed = _check_backend_response("sessions/cancel", status, payload)
        if failed is not None:
            return failed

        status, payload = request(
            "POST",
            "/api/v1/acp/sessions/close",
            {"session_id": session_id},
        )
        failed = _check_backend_response("sessions/close", status, payload)
        if failed is not None:
            return failed
        session_id = None

        print("PASS live_backend_acp_e2e: " + json.dumps(evidence, sort_keys=True))
        return 0
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"FAIL live_backend_acp_e2e: {exc}", file=sys.stderr)
        return 1
    finally:
        if session_id:
            try:
                request(
                    "POST",
                    "/api/v1/acp/sessions/close",
                    {"session_id": session_id},
                )
            except (OSError, ValueError, urllib.error.URLError) as exc:
                print(
                    f"WARN live_backend_acp_e2e: failed to close session {session_id}: {exc}",
                    file=sys.stderr,
                )


def _env_flag_enabled(name: str) -> bool:
    """Return True when an environment flag requests strict live evidence."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _workspace_live_e2e_workspace_id(profile: str) -> str:
    """Return the workspace id to bind to the live workspace certification run."""
    configured = os.environ.get("ACP_E2E_WORKSPACE_ID", "").strip()
    if configured:
        return configured
    raise ValueError("ACP_E2E_WORKSPACE_ID is required for workspace-live-e2e certification")


def _workspace_live_e2e_mcp_servers() -> list[dict[str, Any]]:
    """Return a non-empty MCP server config for workspace certification."""
    name = os.environ.get("ACP_E2E_MCP_SERVER_NAME", "").strip() or "tldw-workspace-certification"
    command = os.environ.get("ACP_E2E_MCP_SERVER_COMMAND", "").strip() or sys.executable
    raw_args = os.environ.get("ACP_E2E_MCP_SERVER_ARGS_JSON", "").strip()
    if raw_args:
        try:
            args_payload = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError("ACP_E2E_MCP_SERVER_ARGS_JSON must be valid JSON") from exc
        if not isinstance(args_payload, list) or not all(isinstance(arg, str) for arg in args_payload):
            raise ValueError("ACP_E2E_MCP_SERVER_ARGS_JSON must be a JSON array of strings")
        args = args_payload
    else:
        args = [str(Path(__file__).resolve()), "--mcp-certification-server"]
    return [
        {
            "name": name,
            "type": "stdio",
            "command": command,
            "args": args,
            "env": {},
        }
    ]


def _payload_workspace_context(payload: Any) -> dict[str, Any]:
    """Return a payload's workspace_context dict when present."""
    if not isinstance(payload, dict):
        return {}
    context = payload.get("workspace_context")
    return dict(context) if isinstance(context, dict) else {}


def _workspace_context_value(payloads: list[dict[str, Any]], key: str) -> Any:
    """Return the first non-empty workspace_context value for a key."""
    for payload in payloads:
        context = _payload_workspace_context(payload)
        value = context.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _filtered_sessions_include(
    payload: dict[str, Any],
    *,
    session_id: str,
    workspace_id: str,
) -> bool:
    """Return whether a workspace-filtered session list contains the session."""
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return False
    for session in sessions:
        if not isinstance(session, dict):
            continue
        if str(session.get("session_id") or "") != session_id:
            continue
        context = _payload_workspace_context(session)
        listed_workspace_id = session.get("workspace_id") or context.get("workspace_id")
        if str(listed_workspace_id or "") == workspace_id:
            return True
    return False


def _payload_contains_review_evidence(payload: Any) -> bool:
    """Return True when support payloads include structured review-loop evidence."""
    review_evidence_keys = {"review_loop", "reviewer", "review_decision"}

    def is_positive_evidence(value: Any) -> bool:
        if value in (None, False, "", [], {}):
            return False
        if value is True:
            return True
        if isinstance(value, str):
            return bool(value.strip())
        return True

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in review_evidence_keys and is_positive_evidence(value):
                return True
            if isinstance(value, (dict, list)) and _payload_contains_review_evidence(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_payload_contains_review_evidence(item) for item in payload)
    return False


def _required_workspace_live_failures(capabilities: dict[str, str]) -> list[str]:
    """Return capability ids that must pass for workspace live certification."""
    required = (
        "init",
        "session_new",
        "prompt",
        "structured_completion",
        "diagnostics",
        "cancel_close",
        "workspace_env",
        "mcp_injection",
        "redacted_support_view",
    )
    return [capability for capability in required if capabilities.get(capability) != "pass"]


def _optional_expectation_failures(capabilities: dict[str, str]) -> list[str]:
    """Return optional capabilities that were requested but did not pass."""
    expected = {
        "artifacts": "ACP_E2E_EXPECT_ARTIFACTS",
        "sandbox": "ACP_E2E_EXPECT_SANDBOX",
        "review_loop": "ACP_E2E_EXPECT_REVIEWER_LOOP",
    }
    return [
        capability
        for capability, env_name in expected.items()
        if _env_flag_enabled(env_name) and capabilities.get(capability) != "pass"
    ]


def _run_backend_workspace_live_e2e_from_env() -> int:
    """Run the backend ACP REST lifecycle with workspace evidence checks."""
    required_env = {
        name: os.environ.get(name, "").strip()
        for name in (
            "TLDW_E2E_SERVER_URL",
            "TLDW_E2E_API_KEY",
            "ACP_AGENT_PROFILE",
            "ACP_E2E_WORKSPACE_ID",
        )
    }
    missing = [
        name for name, value in required_env.items()
        if not value
    ]
    if missing:
        print(
            "Refusing to run backend workspace live ACP certification without required environment: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    profile = required_env["ACP_AGENT_PROFILE"]
    workspace_id = _workspace_live_e2e_workspace_id(profile)
    workspace_id_source = "env"
    workspace_cwd = os.environ.get("ACP_E2E_WORKSPACE_CWD", "").strip() or str(ROOT)
    mcp_servers = _workspace_live_e2e_mcp_servers()
    session_id: str | None = None
    timeout_seconds: float | None = None

    def request(method: str, path: str, body: Any = None) -> tuple[int, dict[str, Any]]:
        return _http_json_request(
            method,
            path,
            body,
            timeout_seconds=timeout_seconds,
        )

    def check_workspace_response(step: str, status: int, payload: Any) -> int | None:
        return _check_backend_response(
            step,
            status,
            payload,
            label="workspace_live_backend_acp_e2e",
        )

    try:
        timeout_seconds = _backend_e2e_timeout_seconds()
        for step, method, path, body in (
            ("health", "GET", "/api/v1/acp/health", None),
            (
                "setup-guide",
                "GET",
                "/api/v1/acp/setup-guide?"
                + urllib.parse.urlencode({"agent_type": profile}),
                None,
            ),
        ):
            status, payload = request(method, path, body)
            failed = check_workspace_response(step, status, payload)
            if failed is not None:
                return failed

        new_body = {
            "cwd": workspace_cwd,
            "agent_type": profile,
            "name": f"ACP workspace live E2E {profile}",
            "workspace_id": workspace_id,
            "mcp_servers": mcp_servers,
        }
        status, new_payload = request("POST", "/api/v1/acp/sessions/new", new_body)
        failed = check_workspace_response("sessions/new", status, new_payload)
        if failed is not None:
            return failed
        session_id = str(new_payload.get("session_id") or "")
        if not session_id:
            return _fail_backend_live_e2e(
                "sessions/new",
                status,
                {"detail": "missing session_id"},
                label="workspace_live_backend_acp_e2e",
            )

        prompt_body = {
            "session_id": session_id,
            "prompt": [
                {
                    "type": "text",
                    "text": _WORKSPACE_LIVE_E2E_PROMPT,
                }
            ],
        }
        status, prompt_payload = request("POST", "/api/v1/acp/sessions/prompt", prompt_body)
        failed = check_workspace_response("sessions/prompt", status, prompt_payload)
        if failed is not None:
            return failed

        stop_reason = (
            prompt_payload.get("stop_reason")
            or (prompt_payload.get("raw_result") or {}).get("stopReason")
        )
        redacted_paths = [
            ("detail", f"/api/v1/acp/sessions/{session_id}/detail?redacted=true"),
            ("events", f"/api/v1/acp/sessions/{session_id}/events?redacted=true"),
            ("artifacts", f"/api/v1/acp/sessions/{session_id}/artifacts?redacted=true"),
            ("diagnostics", f"/api/v1/acp/sessions/{session_id}/diagnostics"),
        ]
        support_payloads: dict[str, dict[str, Any]] = {}
        evidence: dict[str, Any] = {
            "agent_profile": profile,
            "workspace_id": workspace_id,
            "workspace_id_source": workspace_id_source,
            "session_id": session_id,
            "stop_reason": stop_reason,
        }
        for step, path in redacted_paths:
            status, payload = request("GET", path, None)
            failed = check_workspace_response(step, status, payload)
            if failed is not None:
                return failed
            support_payloads[step] = payload
            if "total" in payload:
                evidence[f"{step}_total"] = payload.get("total")

        filtered_path = (
            "/api/v1/acp/sessions?"
            + urllib.parse.urlencode({"workspace_id": workspace_id, "limit": 20})
        )
        status, session_list_payload = request("GET", filtered_path, None)
        failed = check_workspace_response("sessions?workspace_id", status, session_list_payload)
        if failed is not None:
            return failed

        status, payload = request(
            "POST",
            "/api/v1/acp/sessions/cancel",
            {"session_id": session_id},
        )
        failed = check_workspace_response("sessions/cancel", status, payload)
        if failed is not None:
            return failed

        status, payload = request(
            "POST",
            "/api/v1/acp/sessions/close",
            {"session_id": session_id},
        )
        failed = check_workspace_response("sessions/close", status, payload)
        if failed is not None:
            return failed
        closed_session_id = session_id
        session_id = None

        context_payloads = [
            new_payload,
            support_payloads.get("detail", {}),
            support_payloads.get("diagnostics", {}),
        ]
        workspace_from_new = str(new_payload.get("workspace_id") or "")
        workspace_from_context = str(_workspace_context_value(context_payloads, "workspace_id") or "")
        workspace_env_passed = (
            workspace_from_new == workspace_id
            or workspace_from_context == workspace_id
            or _filtered_sessions_include(
                session_list_payload,
                session_id=closed_session_id,
                workspace_id=workspace_id,
            )
        )
        mcp_count = _workspace_context_value(context_payloads, "mcp_server_count")
        try:
            mcp_count_int = int(mcp_count or 0)
        except (TypeError, ValueError):
            mcp_count_int = 0
        sandbox_session_id = (
            new_payload.get("sandbox_session_id")
            or _workspace_context_value(context_payloads, "sandbox_session_id")
        )
        sandbox_run_id = (
            new_payload.get("sandbox_run_id")
            or _workspace_context_value(context_payloads, "sandbox_run_id")
        )
        artifacts_total = int(support_payloads.get("artifacts", {}).get("total") or 0)
        review_loop_passed = (
            _payload_contains_review_evidence(support_payloads.get("diagnostics"))
            or _payload_contains_review_evidence(support_payloads.get("artifacts"))
            or _payload_contains_review_evidence(prompt_payload)
        )
        capabilities = {
            "init": "pass",
            "session_new": "pass",
            "prompt": "pass",
            "structured_completion": "pass" if stop_reason else "fail",
            "diagnostics": "pass",
            "cancel_close": "pass",
            "workspace_env": "pass" if workspace_env_passed else "fail",
            "mcp_injection": "pass" if mcp_count_int > 0 else "fail",
            "redacted_support_view": "pass",
            "artifacts": "pass" if artifacts_total > 0 else "skip",
            "sandbox": "pass" if sandbox_session_id and sandbox_run_id else "skip",
            "review_loop": "pass" if review_loop_passed else "skip",
        }
        evidence["capabilities"] = capabilities
        evidence["mcp_server_count"] = mcp_count_int
        if sandbox_session_id:
            evidence["sandbox_session_id"] = sandbox_session_id
        if sandbox_run_id:
            evidence["sandbox_run_id"] = sandbox_run_id

        failures = _required_workspace_live_failures(capabilities)
        failures.extend(_optional_expectation_failures(capabilities))
        if failures:
            return _fail_backend_live_e2e(
                "workspace evidence",
                200,
                {"failed_capabilities": failures, "evidence": evidence},
                label="workspace_live_backend_acp_e2e",
            )

        print("PASS workspace_live_backend_acp_e2e: " + json.dumps(evidence, sort_keys=True))
        return 0
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"FAIL workspace_live_backend_acp_e2e: {exc}", file=sys.stderr)
        return 1
    finally:
        if session_id:
            try:
                request(
                    "POST",
                    "/api/v1/acp/sessions/close",
                    {"session_id": session_id},
                )
            except (OSError, ValueError, urllib.error.URLError) as exc:
                print(
                    f"WARN workspace_live_backend_acp_e2e: failed to close session {session_id}: {exc}",
                    file=sys.stderr,
                )


def _run_mcp_certification_server() -> int:
    """Run a minimal stdio MCP server for workspace certification injection."""
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        request_id = request.get("id")
        method = str(request.get("method") or "")
        if request_id is None:
            continue
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "tldw-workspace-certification",
                    "version": "0.1.0",
                },
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "workspace.certification.echo",
                        "description": "Return a static workspace certification marker.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        elif method == "tools/call":
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "TLDW_WORKSPACE_MCP_CERTIFICATION_OK",
                    }
                ]
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unsupported method: {method}"},
            }
            print(json.dumps(response), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
    return 0


def _missing_executable_reason(command: dict[str, Any], cwd: Path) -> str | None:
    """Return a skip/fail reason when a path-like executable is unavailable."""
    argv = command.get("argv", [])
    if not argv:
        return "command argv is empty"

    executable = str(argv[0])
    if "/" not in executable and not executable.startswith("."):
        return None

    executable_path = Path(executable)
    if not executable_path.is_absolute():
        executable_path = cwd / executable_path
    if not executable_path.exists():
        return f"executable not found: {executable_path}"
    if not os.access(executable_path, os.X_OK):
        return f"executable is not runnable: {executable_path}"
    return None


def _handle_missing_prerequisite(command: dict[str, Any], reason: str) -> int | None:
    """Print skip/fail output for missing command prerequisites."""
    command_id = command.get("id", "<unknown>")
    if command.get("optional", False):
        print(f"SKIP {command_id}: {reason}")
        return None
    print(f"FAIL {command_id}: {reason}", file=sys.stderr)
    return 127


def _close_pipe(pipe: Any) -> None:
    """Close a subprocess pipe if it exposes close()."""
    if pipe is None or not hasattr(pipe, "close"):
        return
    try:
        pipe.close()
    except OSError:
        pass


def _wait_for_process(process: subprocess.Popen[str], timeout: float) -> bool:
    """Return True when a process exits before timeout."""
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return True


def _signal_process_group(process: subprocess.Popen[str], sig: int) -> bool:
    """Signal the process group owned by a stdio probe subprocess."""
    if os.name != "posix" or not hasattr(os, "killpg"):
        return False
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


def _stop_stdio_process(process: subprocess.Popen[str], *, force_kill: bool) -> None:
    """Terminate a stdio probe subprocess and any children it spawned."""
    if force_kill:
        if not _signal_process_group(process, _FORCE_KILL_SIGNAL):
            process.kill()
        return

    if _signal_process_group(process, signal.SIGTERM):
        return
    if hasattr(process, "terminate"):
        process.terminate()
        return
    process.kill()


def _cleanup_stdio_process(process: subprocess.Popen[str], *, force_kill: bool) -> None:
    """Close stdio, stop the subprocess, and wait so failure paths do not leak."""
    is_running = True
    if hasattr(process, "poll"):
        try:
            is_running = process.poll() is None
        except OSError:
            is_running = True

    if not force_kill:
        _close_pipe(getattr(process, "stdin", None))
        if _wait_for_process(process, 1):
            _close_pipe(getattr(process, "stdout", None))
            return

    if is_running:
        try:
            _stop_stdio_process(process, force_kill=force_kill)
        except OSError:
            pass

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            _stop_stdio_process(process, force_kill=True)
        except OSError:
            pass
        try:
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except OSError:
        pass
    finally:
        _close_pipe(getattr(process, "stdin", None))
        _close_pipe(getattr(process, "stdout", None))


def _enqueue_stdout_payload(responses: queue.Queue[object], payload: object) -> None:
    """Bound stdout queue memory while preserving the first queued response."""
    try:
        responses.put_nowait(payload)
    except queue.Full:
        return


def _drain_stdout_payloads(responses: queue.Queue[object]) -> None:
    """Discard stale queued responses before advancing to the next request id."""
    while True:
        try:
            responses.get_nowait()
        except queue.Empty:
            return


def _stdout_reader(
    stdout: Any,
    responses: queue.Queue[object],
    stop_event: threading.Event,
    expected_response: dict[str, Any],
    expected_response_condition: threading.Condition,
) -> None:
    """Read complete stdout lines in a daemon thread."""
    while not stop_event.is_set():
        with expected_response_condition:
            while expected_response.get("matched") and not stop_event.is_set():
                expected_response_condition.wait(timeout=0.01)
        try:
            try:
                line = stdout.readline(_STDOUT_LINE_LIMIT + 1)
            except TypeError:
                line = stdout.readline()
        except (OSError, ValueError):
            _enqueue_stdout_payload(responses, _STDOUT_EOF)
            return
        if not line:
            _enqueue_stdout_payload(responses, _STDOUT_EOF)
            return
        if len(line) > _STDOUT_LINE_LIMIT:
            _enqueue_stdout_payload(
                responses,
                {"_reader_error": {"message": "JSON-RPC stdout line exceeds maximum length"}},
            )
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _enqueue_stdout_payload(
                responses,
                {"_reader_error": {"message": "invalid JSON-RPC response"}},
            )
            continue
        if not isinstance(payload, dict):
            _enqueue_stdout_payload(
                responses,
                {"_reader_error": {"message": "JSON-RPC response is not an object"}},
            )
            continue
        with expected_response_condition:
            expected_id = expected_response.get("id")
        if payload.get("id") != expected_id:
            continue
        _enqueue_stdout_payload(responses, payload)
        with expected_response_condition:
            expected_response["matched"] = True
            expected_response_condition.notify_all()


def _finish_stdio_process(
    process: subprocess.Popen[str],
    *,
    force_kill: bool,
    reader_stop: threading.Event | None = None,
    reader_thread: threading.Thread | None = None,
) -> None:
    """Stop the stdout reader and clean up the subprocess."""
    if reader_stop is not None:
        reader_stop.set()
    _cleanup_stdio_process(process, force_kill=force_kill)
    if reader_thread is not None:
        reader_thread.join(timeout=0.1)


def _format_error_message(message: Any) -> str:
    """Return a bounded single-line error message."""
    text = str(message or "JSON-RPC error").replace("\r", " ").replace("\n", " ")
    if len(text) > _ERROR_MESSAGE_LIMIT:
        return text[: _ERROR_MESSAGE_LIMIT - 3] + "..."
    return text


def _sanitize_jsonrpc_error(error: Any) -> dict[str, Any]:
    """Keep stable JSON-RPC error fields without printing arbitrary error data."""
    if not isinstance(error, dict):
        return {"message": _format_error_message(error)}
    sanitized: dict[str, Any] = {"message": _format_error_message(error.get("message"))}
    if "code" in error:
        sanitized["code"] = error["code"]
    return sanitized


def _read_jsonrpc_response(
    responses: queue.Queue[object],
    request_id: Any,
    deadline: float,
) -> dict[str, Any] | None:
    """Read the matching JSON-RPC response for a request id before the deadline."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            payload = responses.get(timeout=remaining)
        except queue.Empty:
            return None
        if payload is _STDOUT_EOF:
            return None
        if not isinstance(payload, dict):
            continue
        if payload.get("_reader_error"):
            return {"id": request_id, "error": payload["_reader_error"]}
        if payload.get("id") != request_id:
            continue
        return payload


def _substitute_runtime_placeholders(value: Any, *, session_id: str | None) -> Any:
    """Return a JSON-RPC value with runtime placeholders resolved."""
    if value == _SESSION_ID_PLACEHOLDER:
        if not session_id:
            raise ValueError(
                f"Runtime placeholder '{_SESSION_ID_PLACEHOLDER}' found but no "
                "sessionId was captured from a previous session/new response."
            )
        return session_id
    if isinstance(value, list):
        return [
            _substitute_runtime_placeholders(item, session_id=session_id)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _substitute_runtime_placeholders(item, session_id=session_id)
            for key, item in value.items()
        }
    return value


def _extract_session_id(response: dict[str, Any]) -> str | None:
    """Return the session id from a session/new response if present."""
    result = response.get("result")
    if not isinstance(result, dict):
        return None
    session_id = result.get("sessionId") or result.get("session_id")
    if session_id is None:
        return None
    return str(session_id)


def _run_stdio_jsonrpc_sequence(command: dict[str, Any], cwd: Path) -> int:
    """Run an ordered stdio JSON-RPC sequence, stopping after the first error."""
    timeout_seconds = float(command.get("timeout_seconds", 10))
    deadline = time.monotonic() + timeout_seconds
    command_id = command.get("id", "<unknown>")
    try:
        process = subprocess.Popen(  # nosec B603
            command["argv"],
            cwd=str(cwd),
            env=_command_env(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=os.name == "posix",
            text=True,
        )
    except FileNotFoundError as exc:
        handled = _handle_missing_prerequisite(command, str(exc))
        return 127 if handled is None else handled

    frames = list(command.get("stdin_jsonl", []))
    responses: queue.Queue[object] = queue.Queue(maxsize=_STDOUT_QUEUE_MAXSIZE)
    if process.stdout is None:
        _finish_stdio_process(process, force_kill=True)
        print(f"FAIL {command_id}: subprocess stdout is unavailable", file=sys.stderr)
        return 1
    reader_stop = threading.Event()
    expected_response: dict[str, Any] = {
        "id": frames[0].get("id") if frames else None,
        "matched": False,
    }
    expected_response_condition = threading.Condition()
    reader_thread = threading.Thread(
        target=_stdout_reader,
        args=(
            process.stdout,
            responses,
            reader_stop,
            expected_response,
            expected_response_condition,
        ),
        daemon=True,
    )
    reader_thread.start()
    session_id: str | None = None

    for frame in frames:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            print(f"FAIL {command_id}: timed out before {frame.get('method')}", file=sys.stderr)
            return 124
        if process.stdin is None:
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            print(f"FAIL {command_id}: subprocess stdin is unavailable", file=sys.stderr)
            return 1
        with expected_response_condition:
            expected_response["id"] = frame.get("id")
            expected_response["matched"] = False
            expected_response_condition.notify_all()
        try:
            frame_to_send = _substitute_runtime_placeholders(frame, session_id=session_id)
        except ValueError as exc:
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            logger.error("FAIL {}: {}", command_id, exc)
            return 1
        try:
            process.stdin.write(json.dumps(frame_to_send) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            print(f"FAIL {command_id}: subprocess closed stdin before {frame.get('method')}", file=sys.stderr)
            return 1

        response = _read_jsonrpc_response(responses, frame.get("id"), deadline)
        if response is None:
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            print(f"FAIL {command_id}: timed out waiting for {frame.get('method')} response", file=sys.stderr)
            return 124
        if response.get("error"):
            _finish_stdio_process(
                process,
                force_kill=True,
                reader_stop=reader_stop,
                reader_thread=reader_thread,
            )
            print(
                f"FAIL {command_id}: {frame.get('method')} returned error: "
                + json.dumps(_sanitize_jsonrpc_error(response["error"]), sort_keys=True),
                file=sys.stderr,
            )
            return 1
        _drain_stdout_payloads(responses)
        if frame.get("method") == "session/new":
            session_id = _extract_session_id(response)

    exit_code = process.poll()
    if exit_code not in (None, 0):
        _finish_stdio_process(
            process,
            force_kill=False,
            reader_stop=reader_stop,
            reader_thread=reader_thread,
        )
        print(f"FAIL {command_id}: subprocess exited with status {exit_code}", file=sys.stderr)
        return int(exit_code)

    _finish_stdio_process(
        process,
        force_kill=False,
        reader_stop=reader_stop,
        reader_thread=reader_thread,
    )
    return 0


def run_manifest_dict(manifest: dict[str, Any]) -> int:
    """Run safe-by-default commands from a certification manifest dictionary."""
    if manifest["requires_live_agent"]:
        blockers = [str(blocker) for blocker in manifest.get("blockers", []) if blocker]
        if blockers:
            print(
                "Refusing to run ACP certification for blocked manifest: "
                + ", ".join(blockers),
                file=sys.stderr,
            )
            return 2
        missing = [
            name for name in manifest["required_environment"]
            if not os.environ.get(name)
        ]
        if missing:
            print(
                "Refusing to run live ACP certification without required environment: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        if not manifest["commands"]:
            print(
                "Refusing to run ACP certification: manifest has no runnable commands.",
                file=sys.stderr,
            )
            return 2

    for command in manifest["commands"]:
        if not command["safe_to_run_by_default"] and not manifest["requires_live_agent"]:
            continue
        cwd = ROOT / command["cwd"]
        if not cwd.exists():
            handled = _handle_missing_prerequisite(command, f"cwd not found: {cwd}")
            if handled is None:
                continue
            return handled
        missing_executable = _missing_executable_reason(command, cwd)
        if missing_executable:
            handled = _handle_missing_prerequisite(command, missing_executable)
            if handled is None:
                continue
            return handled
        print(f"==> {command['id']} ({cwd})")
        if command.get("stdin_jsonl"):
            result_code = _run_stdio_jsonrpc_sequence(command, cwd)
            if result_code != 0:
                return int(result_code)
            print(f"PASS {command['id']}")
            continue
        try:
            result = subprocess.run(  # nosec B603
                command["argv"],
                cwd=str(cwd),
                env=_command_env(command),
                check=False,
            )
        except FileNotFoundError as exc:
            handled = _handle_missing_prerequisite(command, str(exc))
            if handled is None:
                continue
            return handled
        if result.returncode != 0:
            return int(result.returncode)
    return 0


def run_manifest(profile: str) -> int:
    """Run safe-by-default commands for a certification profile."""
    return run_manifest_dict(build_manifest(profile))


def _build_registry_agent_manifest(agent_profile: str) -> dict[str, Any]:
    """Build a manifest for an agent registry entry."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
        classify_agent_entrypoint,
        get_agent_registry,
    )

    registry = get_agent_registry()
    entry = registry.get_entry(agent_profile)
    if entry is None:
        raise ValueError(f"Unknown ACP agent profile: {agent_profile}")
    classification = classify_agent_entrypoint(entry)
    entry_metadata = {
        "support_state": getattr(entry, "support_state", None),
        "verification_level": getattr(entry, "verification_level", None),
        "adapter_source": getattr(entry, "adapter_source", None),
        "adapter_docs_url": getattr(entry, "adapter_docs_url", None),
        "adapter_package": getattr(entry, "adapter_package", None),
        "adapter_version": getattr(entry, "adapter_version", None),
        "adapter_version_policy": getattr(entry, "adapter_version_policy", None),
        "adapter_install_source": getattr(entry, "adapter_install_source", None),
        "credential_policy": getattr(entry, "credential_policy", None),
        "runtime_backend": getattr(entry, "runtime_backend", None),
    }
    return build_agent_profile_manifest(
        classification.as_dict()
        | {key: value for key, value in entry_metadata.items() if value is not None}
        | {"type": entry.type, "name": entry.name}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(_MANIFESTS),
        default="stub-smoke",
        help="Certification profile to emit or run.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Manifest output format.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the manifest commands instead of printing them.",
    )
    parser.add_argument(
        "--agent-profile",
        help="Render or run a registry-backed agent profile manifest.",
    )
    parser.add_argument(
        "--backend-live-e2e",
        action="store_true",
        help="Run the backend REST live-E2E certification flow from environment.",
    )
    parser.add_argument(
        "--backend-workspace-live-e2e",
        action="store_true",
        help="Run the backend REST workspace live-E2E certification flow from environment.",
    )
    parser.add_argument(
        "--mcp-certification-server",
        action="store_true",
        help="Run a minimal stdio MCP server for workspace live certification.",
    )
    args = parser.parse_args(argv)

    if args.mcp_certification_server:
        return _run_mcp_certification_server()

    if args.backend_live_e2e:
        return _run_backend_live_e2e_from_env()

    if args.backend_workspace_live_e2e:
        return _run_backend_workspace_live_e2e_from_env()

    if args.agent_profile:
        manifest = _build_registry_agent_manifest(args.agent_profile)
        if args.run:
            return run_manifest_dict(manifest)
        print(render_manifest_dict(manifest, output_format=args.format))
        return 0

    if args.run:
        return run_manifest(args.profile)

    print(render_manifest(args.profile, output_format=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
