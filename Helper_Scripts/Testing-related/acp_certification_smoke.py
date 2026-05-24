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
                    "python",
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
                    "python",
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
        "support_state": "documented_unverified",
        "verification_level": "documented_only",
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


def _fail_backend_live_e2e(step: str, status: int, payload: Any) -> int:
    """Print a bounded backend live-E2E failure and return a failure code."""
    print(
        f"FAIL live_backend_acp_e2e: {step} returned HTTP {status}: "
        f"{_payload_preview(payload)}",
        file=sys.stderr,
    )
    return 1


def _check_backend_response(step: str, status: int, payload: Any) -> int | None:
    """Return an exit code when a backend response should fail certification."""
    if status >= 400:
        return _fail_backend_live_e2e(step, status, payload)
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
            if force_kill:
                process.kill()
            elif hasattr(process, "terminate"):
                process.terminate()
            else:
                process.kill()
        except OSError:
            pass

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
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
    return build_agent_profile_manifest(
        classification.as_dict() | {"type": entry.type, "name": entry.name}
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
    args = parser.parse_args(argv)

    if args.backend_live_e2e:
        return _run_backend_live_e2e_from_env()

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
