#!/usr/bin/env python3
"""Run a one-off UAT pass against the MCP Unified standalone user guide.

The harness intentionally lives outside the packaged ``mcp_unified`` module.
It validates the package boundary from an isolated workspace by installing the
package-local project and then executing the documented CLI flows through argv
subprocess calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

_MAX_CAPTURE_CHARS = 6000
_DEFAULT_TIMEOUT_SECONDS = 120.0
_FIXTURE_HOST = "127.0.0.1"
_SERVER_STOP_TIMEOUT_SECONDS = 5.0

_STDIO_FIXTURE_SOURCE = '''\
from __future__ import annotations

import asyncio
import sys

from mcp_unified.gateway.stdio import GatewayStdioServer
from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime


async def main() -> None:
    server = GatewayStdioServer(SmokeFixtureGatewayRuntime(include_denied_tool=True))
    for line in sys.stdin:
        response = await server.handle_line(line)
        if response is not None:
            sys.stdout.write(response)
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
'''

_ASGI_FIXTURE_SOURCE = '''\
from __future__ import annotations

from mcp_unified.gateway.fastapi import create_gateway_app
from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime


app = create_gateway_app(SmokeFixtureGatewayRuntime(include_denied_tool=True))
'''


@dataclass(frozen=True)
class UatStep:
    """One executable or synthetic user-guide UAT step."""

    step_id: str
    description: str
    command: list[str] | None = None
    cwd: Path | None = None
    required: bool = True
    expected_exit_codes: tuple[int, ...] = (0,)
    write_json_path: Path | None = None
    write_json_payload: dict[str, Any] | None = None
    write_text_path: Path | None = None
    write_text_payload: str | None = None
    skip_reason: str | None = None
    background_process_key: str | None = None
    stop_process_key: str | None = None
    health_check_host: str | None = None
    health_check_port: int | None = None
    health_check_path: str | None = None


@dataclass
class UatStepResult:
    """Redacted result for one UAT step."""

    step_id: str
    description: str
    status: str
    required: bool
    duration_ms: float
    command: list[str] | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str | None = None


@dataclass
class UatRunContext:
    """Paths and settings shared across one UAT run."""

    repo_root: Path
    workspace: Path
    bootstrap_python: str
    gateway_url: str | None
    admin_key: str | None
    timeout_seconds: float
    package_install_args: list[str]
    secrets: list[str] = field(default_factory=list)
    processes: dict[str, subprocess.Popen[str]] = field(default_factory=dict)


def default_package_project(repo_root: Path) -> Path:
    """Return the standalone MCP package project path."""

    return repo_root / "apps" / "mcp-unified"


def package_install_spec(
    *,
    repo_root: Path,
    wheel_path: Path | None,
    editable: bool,
) -> list[str]:
    """Return pip install arguments for the standalone package under test.

    Wheel installs take precedence over editable source installs.
    """

    if wheel_path is not None:
        return [str(wheel_path.resolve())]
    project = default_package_project(repo_root)
    if editable:
        return ["-e", f"{project}[gateway]"]
    return [f"{project}[gateway]"]


def build_uat_plan(
    *,
    repo_root: Path,
    workspace: Path,
    python_executable: str,
    gateway_executable: str,
    smoke_executable: str,
    gateway_url: str | None,
    package_install_args: list[str],
    fixture_port: int | None = None,
) -> list[UatStep]:
    """Build the ordered standalone user-guide UAT command plan."""

    venv_dir = workspace / ".venv"
    venv_python = _venv_python(venv_dir)
    gateway_config = workspace / "gateway.json"
    reporting_config = workspace / "gateway-reporting.json"
    policy_args = workspace / "policy-args.json"
    external_server = workspace / "search-server.json"
    credential_grant = workspace / "researcher-search-grant.json"
    snapshot = workspace / "gateway-snapshot.json"
    smoke_report = workspace / "mcp-smoke-inprocess.json"
    stdio_fixture = workspace / "smoke_stdio_fixture.py"
    stdio_smoke_report = workspace / "mcp-smoke-stdio.json"
    asgi_fixture = workspace / "smoke_asgi_fixture.py"
    http_smoke_report = workspace / "mcp-smoke-http.json"
    websocket_smoke_report = workspace / "mcp-smoke-websocket.json"
    fixture_port = fixture_port or _fixture_port_for_workspace(workspace)

    steps = [
        UatStep(
            "create_venv",
            "Create an isolated virtual environment.",
            command=[
                python_executable,
                "-m",
                "venv",
                str(venv_dir),
            ],
            cwd=workspace,
        ),
        UatStep(
            "install_package_boundary",
            "Install the package-local MCP Unified boundary as documented.",
            command=[
                str(venv_python),
                "-m",
                "pip",
                "install",
                *package_install_args,
            ],
            cwd=workspace,
        ),
        UatStep(
            "gateway_package_info",
            "Confirm the gateway CLI reports package status.",
            command=[gateway_executable, "package-info"],
            cwd=workspace,
        ),
        UatStep(
            "smoke_cli_help",
            "Confirm the documented smoke CLI is installed.",
            command=[smoke_executable, "--help"],
            cwd=workspace,
        ),
        UatStep(
            "write_gateway_config",
            "Write the guide's minimal SQLite gateway config.",
            write_json_path=gateway_config,
            write_json_payload={
                "store": {
                    "kind": "sqlite",
                    "sqlite_path": "./mcp-gateway.db",
                },
                "default_preset_id": "project-researcher",
            },
        ),
        UatStep(
            "validate_gateway_config",
            "Validate the minimal gateway config.",
            command=[gateway_executable, "validate-config", str(gateway_config)],
            cwd=workspace,
        ),
        UatStep(
            "list_presets",
            "List bundled profile presets.",
            command=[gateway_executable, "list-presets"],
            cwd=workspace,
        ),
        UatStep(
            "show_project_researcher_preset",
            "Inspect the project-researcher preset.",
            command=[gateway_executable, "show-preset", "project-researcher"],
            cwd=workspace,
        ),
        UatStep(
            "duplicate_project_researcher_preset",
            "Duplicate project-researcher into persistent storage.",
            command=[
                gateway_executable,
                "duplicate-preset",
                "project-researcher",
                "--profile-id",
                "researcher",
                "--name",
                "Project Researcher",
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "set_default_profile",
            "Set the duplicated profile as the default.",
            command=[
                gateway_executable,
                "set-default-profile",
                "researcher",
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "get_default_profile",
            "Read back the active default profile.",
            command=[gateway_executable, "get-default-profile", "--config", str(gateway_config)],
            cwd=workspace,
        ),
        UatStep(
            "write_policy_args",
            "Write safe policy-explain arguments.",
            write_json_path=policy_args,
            write_json_payload={"path": "docs/example.md"},
        ),
        UatStep(
            "explain_policy",
            "Explain one local profile/tool policy decision.",
            command=[
                gateway_executable,
                "explain-policy",
                "--profile",
                "researcher",
                "--tool",
                "fs.read",
                "--args-json-file",
                str(policy_args),
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "preview_profile_tools",
            "Preview the profile's filesystem tool surface.",
            command=[
                gateway_executable,
                "preview-profile-tools",
                "--profile",
                "researcher",
                "--category",
                "filesystem",
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "write_external_server",
            "Write the guide's external server registry example.",
            write_json_path=external_server,
            write_json_payload={
                "id": "search",
                "name": "Search MCP Server",
                "transport": "stdio",
                "command": ["python", "-m", "search_mcp_server"],
                "env_allowlist": ["PATH", "SEARCH_API_ENDPOINT"],
                "credential_slots": ["search_api_key"],
                "enabled": True,
                "auto_start": False,
            },
        ),
        UatStep(
            "create_external_server",
            "Create an external server registry entry.",
            command=[
                gateway_executable,
                "create-external-server",
                "--server-file",
                str(external_server),
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "list_external_servers",
            "List registered external servers.",
            command=[gateway_executable, "list-external-servers", "--config", str(gateway_config)],
            cwd=workspace,
        ),
        UatStep(
            "write_credential_grant",
            "Write the guide's metadata-only credential grant.",
            write_json_path=credential_grant,
            write_json_payload={
                "id": "researcher-search-api-key",
                "profile_id": "researcher",
                "external_server_id": "search",
                "broker_id": "env",
                "credential_slot": "search_api_key",
                "scopes": ["search:read"],
                "enabled": True,
            },
        ),
        UatStep(
            "create_credential_grant",
            "Create the metadata-only credential grant.",
            command=[
                gateway_executable,
                "create-credential-grant",
                "--grant-file",
                str(credential_grant),
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "list_credential_grants",
            "List credential grants for the researcher profile.",
            command=[
                gateway_executable,
                "list-credential-grants",
                "--profile-id",
                "researcher",
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "export_config_snapshot",
            "Export a gateway configuration snapshot.",
            command=[
                gateway_executable,
                "export-config",
                "--output",
                str(snapshot),
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "import_config_snapshot_dry_run",
            "Validate snapshot import without writing.",
            command=[
                gateway_executable,
                "import-config",
                "--snapshot-file",
                str(snapshot),
                "--dry-run",
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "import_config_snapshot_apply",
            "Apply the exported snapshot to the configured store.",
            command=[
                gateway_executable,
                "import-config",
                "--snapshot-file",
                str(snapshot),
                "--config",
                str(gateway_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "write_reporting_config",
            "Write a config with metadata-only tool-use reporting enabled.",
            write_json_path=reporting_config,
            write_json_payload={
                "store": {
                    "kind": "sqlite",
                    "sqlite_path": "./mcp-reporting-gateway.db",
                },
                "default_preset_id": "project-researcher",
                "tool_use_reporting": {
                    "enabled": True,
                    "store": {
                        "kind": "sqlite",
                        "sqlite_path": "./mcp-tool-events.db",
                    },
                    "retention_max_age_days": 30,
                    "retention_max_events": 100000,
                },
            },
        ),
        UatStep(
            "tool_events_report",
            "Build an empty aggregate tool-use report from the configured store.",
            command=[
                gateway_executable,
                "tool-events",
                "report",
                "--group-by",
                "profile",
                "--config",
                str(reporting_config),
            ],
            cwd=workspace,
        ),
        UatStep(
            "smoke_inprocess",
            "Run the documented in-process MCP smoke scenario.",
            command=[
                smoke_executable,
                "inprocess",
                "--json-report",
                str(smoke_report),
            ],
            cwd=workspace,
        ),
        UatStep(
            "write_stdio_fixture",
            "Write a temporary stdio MCP fixture server.",
            write_text_path=stdio_fixture,
            write_text_payload=_STDIO_FIXTURE_SOURCE,
        ),
        UatStep(
            "smoke_stdio_subprocess",
            "Run the smoke client against a stdio subprocess fixture.",
            command=[
                smoke_executable,
                "stdio",
                "--command",
                str(venv_python),
                "--arg",
                str(stdio_fixture),
                "--cwd",
                str(workspace),
                "--json-report",
                str(stdio_smoke_report),
            ],
            cwd=workspace,
        ),
        UatStep(
            "write_asgi_fixture",
            "Write a temporary FastAPI MCP fixture gateway.",
            write_text_path=asgi_fixture,
            write_text_payload=_ASGI_FIXTURE_SOURCE,
        ),
        UatStep(
            "start_fixture_gateway",
            "Start the local fixture gateway for live HTTP/WebSocket smoke tests.",
            command=[
                str(venv_python),
                "-m",
                "uvicorn",
                "smoke_asgi_fixture:app",
                "--host",
                _FIXTURE_HOST,
                "--port",
                str(fixture_port),
                "--log-level",
                "warning",
            ],
            cwd=workspace,
            background_process_key="fixture_gateway",
            health_check_host=_FIXTURE_HOST,
            health_check_port=fixture_port,
            health_check_path="/mcp/status",
        ),
        UatStep(
            "smoke_http",
            "Run the smoke client against the live HTTP fixture gateway.",
            command=[
                smoke_executable,
                "http",
                "--url",
                f"http://{_FIXTURE_HOST}:{fixture_port}/mcp/request",
                "--json-report",
                str(http_smoke_report),
            ],
            cwd=workspace,
        ),
        UatStep(
            "smoke_websocket",
            "Run the smoke client against the live WebSocket fixture gateway.",
            command=[
                smoke_executable,
                "websocket",
                "--url",
                f"ws://{_FIXTURE_HOST}:{fixture_port}/mcp/ws",
                "--json-report",
                str(websocket_smoke_report),
            ],
            cwd=workspace,
        ),
        UatStep(
            "stop_fixture_gateway",
            "Stop the local fixture gateway.",
            required=False,
            stop_process_key="fixture_gateway",
        ),
    ]

    if gateway_url:
        steps.append(
            UatStep(
                "remote_runtime_list",
                "List remote runtime state from a configured gateway URL.",
                command=[gateway_executable, "runtime-list", "--gateway-url", gateway_url],
                cwd=workspace,
                required=False,
            )
        )
    else:
        steps.append(
            UatStep(
                "remote_runtime_skipped",
                "Remote runtime commands require a live gateway URL.",
                required=False,
                skip_reason="Set --gateway-url or MCP_UNIFIED_GATEWAY_URL to run remote runtime UAT.",
            )
        )

    return steps


def redact_text(
    text: str,
    *,
    secrets: list[str],
    sensitive_paths: list[Path],
) -> str:
    """Redact secrets, bearer values, and local paths from captured text."""

    redacted = text
    for path in sorted({str(path) for path in sensitive_paths if path}, key=len, reverse=True):
        if path:
            redacted = redacted.replace(path, "<redacted-path>")
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted-secret>")
    redacted = re.sub(
        r"Bearer\s+[^\s\"']+",
        "Bearer <redacted-secret>",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(X-MCP-Gateway-Admin-Key\s*[:=]\s*)[^\s\"']+",
        r"\1<redacted-secret>",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def run_uat(context: UatRunContext) -> dict[str, Any]:
    """Run the standalone user-guide UAT plan and return a JSON-safe report."""

    started = perf_counter()
    started_at = datetime.now(UTC).isoformat()
    context.workspace.mkdir(parents=True, exist_ok=True)
    venv_bin = _venv_bin_dir(context.workspace / ".venv")
    gateway_executable = str(venv_bin / _script_name("mcp-unified-gateway"))
    smoke_executable = str(venv_bin / _script_name("mcp-unified-smoke"))
    plan = build_uat_plan(
        repo_root=context.repo_root,
        workspace=context.workspace,
        python_executable=context.bootstrap_python,
        gateway_executable=gateway_executable,
        smoke_executable=smoke_executable,
        gateway_url=context.gateway_url,
        package_install_args=context.package_install_args,
    )
    try:
        results = [_run_step(step, context) for step in plan]
    finally:
        _stop_remaining_processes(context)
    ok = all(result.status in {"passed", "skipped"} for result in results if result.required)
    return {
        "ok": ok,
        "started_at": started_at,
        "duration_ms": _elapsed_ms(started),
        "workspace": "<redacted-path>",
        "repo_root": "<redacted-path>",
        "guide": "apps/mcp-unified/USER_GUIDE.md",
        "steps": [_step_result_payload(result, context) for result in results],
        "summary": {
            "passed": sum(1 for result in results if result.status == "passed"),
            "failed": sum(1 for result in results if result.status == "failed"),
            "skipped": sum(1 for result in results if result.status == "skipped"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the user-guide UAT harness from the command line."""

    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    gateway_url = args.gateway_url or os.getenv("MCP_UNIFIED_GATEWAY_URL")
    admin_key = os.getenv("MCP_UNIFIED_GATEWAY_ADMIN_KEY")
    secrets = [admin_key] if admin_key else []
    package_install_args = package_install_spec(
        repo_root=repo_root,
        wheel_path=args.wheel,
        editable=args.editable,
    )

    if args.workspace:
        workspace = args.workspace.resolve()
        report = run_uat(
            UatRunContext(
                repo_root=repo_root,
                workspace=workspace,
                bootstrap_python=sys.executable,
                gateway_url=gateway_url,
                admin_key=admin_key,
                timeout_seconds=args.timeout_seconds,
                package_install_args=package_install_args,
                secrets=secrets,
            )
        )
    else:
        with tempfile.TemporaryDirectory(prefix="mcp-standalone-uat-") as temp_dir:
            report = run_uat(
                UatRunContext(
                    repo_root=repo_root,
                    workspace=Path(temp_dir),
                    bootstrap_python=sys.executable,
                    gateway_url=gateway_url,
                    admin_key=admin_key,
                    timeout_seconds=args.timeout_seconds,
                    package_install_args=package_install_args,
                    secrets=secrets,
                )
            )

    _write_report(report, args.json_report)
    return 0 if report["ok"] else 1


def _run_step(step: UatStep, context: UatRunContext) -> UatStepResult:
    started = perf_counter()
    if step.skip_reason:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="skipped",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason=step.skip_reason,
        )
    if step.stop_process_key is not None:
        return _stop_background_process(step, context, started)
    if step.background_process_key is not None:
        return _start_background_process(step, context, started)
    if step.write_json_path is not None:
        if step.write_json_payload is None:
            return UatStepResult(
                step_id=step.step_id,
                description=step.description,
                status="failed",
                required=step.required,
                duration_ms=_elapsed_ms(started),
                reason="write step is missing a JSON payload",
            )
        step.write_json_path.parent.mkdir(parents=True, exist_ok=True)
        step.write_json_path.write_text(
            json.dumps(step.write_json_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="passed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason=f"wrote {step.write_json_path.name}",
        )
    if step.write_text_path is not None:
        if step.write_text_payload is None:
            return UatStepResult(
                step_id=step.step_id,
                description=step.description,
                status="failed",
                required=step.required,
                duration_ms=_elapsed_ms(started),
                reason="write step is missing a text payload",
            )
        step.write_text_path.parent.mkdir(parents=True, exist_ok=True)
        step.write_text_path.write_text(step.write_text_payload, encoding="utf-8")
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="passed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason=f"wrote {step.write_text_path.name}",
        )
    if step.command is None:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="failed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason="step has no command, write payload, or skip reason",
        )

    env = os.environ.copy()
    if context.admin_key:
        env["MCP_UNIFIED_GATEWAY_ADMIN_KEY"] = context.admin_key
    try:
        # Commands are fixed argv lists, never shell strings.
        completed = subprocess.run(  # nosec B603
            step.command,
            cwd=step.cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=context.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="failed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            command=step.command,
            reason=f"{exc.__class__.__name__}: {exc}",
        )

    status = "passed" if completed.returncode in step.expected_exit_codes else "failed"
    return UatStepResult(
        step_id=step.step_id,
        description=step.description,
        status=status,
        required=step.required,
        duration_ms=_elapsed_ms(started),
        command=step.command,
        exit_code=completed.returncode,
        stdout=_truncate(completed.stdout),
        stderr=_truncate(completed.stderr),
        reason=None if status == "passed" else "unexpected exit code",
    )


def _start_background_process(
    step: UatStep,
    context: UatRunContext,
    started: float,
) -> UatStepResult:
    if step.command is None:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="failed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason="background process step is missing a command",
        )
    if step.health_check_port is None or step.health_check_path is None:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="failed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            command=step.command,
            reason="background process step is missing a health check",
        )

    env = os.environ.copy()
    if context.admin_key:
        env["MCP_UNIFIED_GATEWAY_ADMIN_KEY"] = context.admin_key
    try:
        # Commands are fixed argv lists, never shell strings.
        process = subprocess.Popen(  # nosec B603
            step.command,
            cwd=step.cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="failed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            command=step.command,
            reason=f"{exc.__class__.__name__}: {exc}",
        )

    context.processes[step.background_process_key or step.step_id] = process
    health_reason = _wait_for_http_health(
        process,
        host=step.health_check_host or _FIXTURE_HOST,
        port=step.health_check_port,
        path=step.health_check_path,
        timeout_seconds=context.timeout_seconds,
    )
    if health_reason is None:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="passed",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            command=step.command,
            reason="health check passed",
        )

    context.processes.pop(step.background_process_key or step.step_id, None)
    stdout, stderr = _terminate_process(process)
    return UatStepResult(
        step_id=step.step_id,
        description=step.description,
        status="failed",
        required=step.required,
        duration_ms=_elapsed_ms(started),
        command=step.command,
        exit_code=process.returncode,
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        reason=health_reason,
    )


def _stop_background_process(
    step: UatStep,
    context: UatRunContext,
    started: float,
) -> UatStepResult:
    process = context.processes.pop(step.stop_process_key or "", None)
    if process is None:
        return UatStepResult(
            step_id=step.step_id,
            description=step.description,
            status="skipped",
            required=step.required,
            duration_ms=_elapsed_ms(started),
            reason="process not running",
        )
    stdout, stderr = _terminate_process(process)
    return UatStepResult(
        step_id=step.step_id,
        description=step.description,
        status="passed",
        required=step.required,
        duration_ms=_elapsed_ms(started),
        exit_code=process.returncode,
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        reason="process stopped",
    )


def _wait_for_http_health(
    process: subprocess.Popen[str],
    *,
    host: str,
    port: int,
    path: str,
    timeout_seconds: float,
) -> str | None:
    deadline = perf_counter() + timeout_seconds
    last_error = "health check did not complete"
    while perf_counter() < deadline:
        if process.poll() is not None:
            return f"process exited before health check passed: {process.returncode}"
        connection: HTTPConnection | None = None
        try:
            connection = HTTPConnection(host, port, timeout=0.5)
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return None
            last_error = f"health check returned HTTP {response.status}"
        except OSError as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            if connection is not None:
                connection.close()
        sleep(0.1)
    return last_error


def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=_SERVER_STOP_TIMEOUT_SECONDS)


def _stop_remaining_processes(context: UatRunContext) -> None:
    for process in list(context.processes.values()):
        try:
            _terminate_process(process)
        except Exception as exc:  # noqa: BLE001 - cleanup must not abort on unexpected process errors.
            sys.stderr.write(
                "warning: failed to stop background process "
                f"{process.pid}: {exc.__class__.__name__}: {exc}\n"
            )
    context.processes.clear()


def _step_result_payload(result: UatStepResult, context: UatRunContext) -> dict[str, Any]:
    return {
        "id": result.step_id,
        "description": result.description,
        "status": result.status,
        "required": result.required,
        "duration_ms": result.duration_ms,
        "command": _redact_command(result.command, context) if result.command else None,
        "exit_code": result.exit_code,
        "stdout": redact_text(
            result.stdout,
            secrets=context.secrets,
            sensitive_paths=[context.workspace, context.repo_root],
        ),
        "stderr": redact_text(
            result.stderr,
            secrets=context.secrets,
            sensitive_paths=[context.workspace, context.repo_root],
        ),
        "reason": result.reason,
    }


def _redact_command(command: list[str], context: UatRunContext) -> list[str]:
    return [
        redact_text(
            token,
            secrets=context.secrets,
            sensitive_paths=[context.workspace, context.repo_root],
        )
        for token in command
    ]


def _write_report(report: dict[str, Any], json_report: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if json_report is None:
        sys.stdout.write(rendered)
        return
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(rendered, encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-off UAT against the MCP Unified standalone user guide.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the apps/mcp-unified standalone project.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional workspace directory to keep after the run.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Path to write the redacted JSON report. Defaults to stdout.",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Built wheel to install for installed-artifact UAT.",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help=(
            "Install the app package project in editable mode for local guide iteration. "
            "Ignored when --wheel is supplied."
        ),
    )
    parser.add_argument(
        "--gateway-url",
        help="Optional live mounted gateway URL for remote runtime UAT.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Timeout per subprocess command.",
    )
    return parser.parse_args(argv)


def _script_name(name: str) -> str:
    if os.name == "nt":
        return f"{name}.exe"
    return name


def _venv_bin_dir(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def _venv_python(venv_dir: Path) -> Path:
    return _venv_bin_dir(venv_dir) / _script_name("python")


def _fixture_port_for_workspace(workspace: Path) -> int:
    # Keep the plan side-effect-free; the actual bind happens in the Uvicorn step.
    return 18000 + (sum(str(workspace).encode("utf-8")) % 1000)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _truncate(value: str | None) -> str:
    if value is None:
        return ""
    if len(value) <= _MAX_CAPTURE_CHARS:
        return value
    return value[:_MAX_CAPTURE_CHARS] + "\n<truncated>"


if __name__ == "__main__":
    raise SystemExit(main())
