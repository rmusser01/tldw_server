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
import shlex
# subprocess is intentionally used to run static manifest argv with shell=False.
import subprocess  # nosec B404
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


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


def render_manifest(profile: str, *, output_format: str = "markdown") -> str:
    """Render a certification manifest as JSON or Markdown."""
    manifest = build_manifest(profile)
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
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"- {note}" for note in manifest["notes"])
    lines.append("")
    lines.append("## Commands")
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
    return "\n".join(lines).rstrip() + "\n"


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


def run_manifest(profile: str) -> int:
    """Run safe-by-default commands for a certification profile."""
    manifest = build_manifest(profile)
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
    args = parser.parse_args(argv)

    if args.run:
        return run_manifest(args.profile)

    print(render_manifest(args.profile, output_format=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
