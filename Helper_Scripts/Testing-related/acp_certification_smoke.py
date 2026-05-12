#!/usr/bin/env python3
"""Emit or run ACP downstream-agent certification smoke manifests.

The manifest reuses existing ACP test suites and runner checks. It does not
invent a second compatibility test framework; it gives contributors one stable
place to find the commands and capability IDs needed for matrix evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
# subprocess is intentionally used to run static manifest argv with shell=False.
import subprocess  # nosec B404
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_STDOUT_EOF = object()
_ERROR_MESSAGE_LIMIT = 240


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
                "description": "Live backend/browser ACP flow against a configured downstream agent profile.",
                "cwd": "apps/tldw-frontend",
                "argv": [
                    "bunx",
                    "playwright",
                    "test",
                    "e2e/workflows/tier-3-automation/acp-playground.spec.ts",
                    "e2e/workflows/tier-3-automation/agent-registry.spec.ts",
                    "e2e/workflows/tier-3-automation/agent-tasks.spec.ts",
                    "--reporter=line",
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
    if primary_blocker and primary_blocker not in blockers:
        blockers.insert(0, str(primary_blocker))

    manifest: dict[str, Any] = {
        "profile": profile,
        "name": entrypoint.get("name") or profile,
        "support_state": "documented_unverified",
        "verification_level": "documented_only",
        "requires_live_agent": True,
        "required_environment": [
            "TLDW_E2E_SERVER_URL",
            "TLDW_E2E_API_KEY",
            "ACP_AGENT_PROFILE",
        ],
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
                            "protocolVersion": "1",
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
                        "params": {"cwd": ".", "mcpServers": []},
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "prompt": "Reply with a short ACP certification acknowledgement."
                        },
                    },
                ],
                "timeout_seconds": 10,
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


def _cleanup_stdio_process(process: subprocess.Popen[str], *, force_kill: bool) -> None:
    """Close stdio, stop the subprocess, and wait so failure paths do not leak."""
    is_running = True
    if hasattr(process, "poll"):
        try:
            is_running = process.poll() is None
        except OSError:
            is_running = True

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


def _stdout_reader(
    stdout: Any,
    responses: queue.Queue[object],
    stop_event: threading.Event,
) -> None:
    """Read complete stdout lines in a daemon thread."""
    while not stop_event.is_set():
        try:
            line = stdout.readline()
        except (OSError, ValueError):
            responses.put(_STDOUT_EOF)
            return
        if not line:
            responses.put(_STDOUT_EOF)
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            responses.put({"_reader_error": {"message": "invalid JSON-RPC response"}})
            continue
        if not isinstance(payload, dict):
            responses.put({"_reader_error": {"message": "JSON-RPC response is not an object"}})
            continue
        responses.put(payload)


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

    responses: queue.Queue[object] = queue.Queue()
    if process.stdout is None:
        _finish_stdio_process(process, force_kill=True)
        print(f"FAIL {command_id}: subprocess stdout is unavailable", file=sys.stderr)
        return 1
    reader_stop = threading.Event()
    reader_thread = threading.Thread(
        target=_stdout_reader,
        args=(process.stdout, responses, reader_stop),
        daemon=True,
    )
    reader_thread.start()

    for frame in command.get("stdin_jsonl", []):
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
        try:
            process.stdin.write(json.dumps(frame) + "\n")
            process.stdin.flush()
        except BrokenPipeError:
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
    args = parser.parse_args(argv)

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
