"""Command-line entrypoint for MCP Unified smoke scenarios."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from loguru import logger

from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
from mcp_unified.smoke.reporting import SmokeReport, report_to_json
from mcp_unified.smoke.scenarios import ScenarioMode, run_baseline_scenario
from mcp_unified.smoke.transports import (
    InProcessGatewayTransport,
    LiveHttpTransport,
    LiveWebSocketTransport,
    McpSmokeTransport,
    McpSmokeTransportError,
    StdioSubprocessTransport,
)

_EXIT_SUCCESS = 0
_EXIT_SCENARIO_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TRANSPORT_FAILED = 3
_EXIT_STRICT_CAPABILITY_UNAVAILABLE = 4
_SCENARIO_CHOICES = ("baseline",)
_MODE_CHOICES = ("best-effort", "strict")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MCP smoke CLI and return a process exit code."""

    parser = _build_parser()
    try:
        args = parser.parse_args(_normalize_stdio_arg_tokens(argv))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else _EXIT_USAGE

    try:
        report = asyncio.run(_run(args))
    except (McpSmokeTransportError, OSError) as exc:
        logger.error("transport error: {}", exc)
        return _EXIT_TRANSPORT_FAILED
    except ValueError as exc:
        logger.error("error: {}", exc)
        return _EXIT_USAGE

    _write_report(report, args.json_report, stdout=sys.stdout)
    if report.ok:
        return _EXIT_SUCCESS
    if _has_strict_capability_failure(report):
        return _EXIT_STRICT_CAPABILITY_UNAVAILABLE
    return _EXIT_SCENARIO_FAILED


def _build_parser() -> argparse.ArgumentParser:
    common = _build_common_parser(with_defaults=True)
    subcommand_common = _build_common_parser(with_defaults=False)
    parser = argparse.ArgumentParser(
        prog="mcp-unified-smoke",
        description="Run MCP Unified JSON-RPC smoke scenarios.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="transport", required=True)

    inprocess = subparsers.add_parser(
        "inprocess",
        parents=[subcommand_common],
        help="Run against the deterministic in-process fixture runtime.",
    )
    inprocess.add_argument(
        "--disable-resources",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    http = subparsers.add_parser(
        "http",
        parents=[subcommand_common],
        help="Run against a live MCP HTTP endpoint.",
    )
    http.add_argument("--url", required=True, help="MCP HTTP JSON-RPC endpoint URL.")

    websocket = subparsers.add_parser(
        "websocket",
        parents=[subcommand_common],
        help="Run against a live MCP WebSocket endpoint.",
    )
    websocket.add_argument("--url", required=True, help="MCP WebSocket endpoint URL.")

    stdio = subparsers.add_parser(
        "stdio",
        parents=[subcommand_common],
        help="Run against an MCP server subprocess over stdio.",
    )
    stdio.add_argument("--command", required=True, help="Executable path or command.")
    stdio.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Argument to pass to the subprocess; repeat for multiple args.",
    )
    stdio.add_argument("--cwd", help="Subprocess working directory.")
    stdio.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable name to inherit; repeat for multiple names.",
    )
    return parser


def _normalize_stdio_arg_tokens(argv: Sequence[str] | None) -> list[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--arg" and index + 1 < len(tokens):  # nosec B105
            value = tokens[index + 1]
            if value.startswith("-"):
                normalized.append(f"--arg={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _build_common_parser(*, with_defaults: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    default = None if with_defaults else argparse.SUPPRESS
    parser.add_argument(
        "--scenario",
        choices=_SCENARIO_CHOICES,
        default="baseline" if with_defaults else argparse.SUPPRESS,
        help="Smoke scenario to run.",
    )
    parser.add_argument(
        "--mode",
        choices=_MODE_CHOICES,
        default="best-effort" if with_defaults else argparse.SUPPRESS,
        help="Scenario mode. strict fails missing advertised capabilities.",
    )
    parser.add_argument(
        "--profile-id",
        default=default,
        help="Gateway profile id to select when supported.",
    )
    parser.add_argument(
        "--api-key-env",
        default=default,
        help="Environment variable holding an API key.",
    )
    parser.add_argument(
        "--bearer-token-env",
        default=default,
        help="Environment variable holding a bearer token.",
    )
    parser.add_argument(
        "--json-report",
        metavar="PATH",
        default=default,
        help="Write the redacted JSON report to PATH, or '-' for stdout.",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        default=False if with_defaults else argparse.SUPPRESS,
        help="Record debug-trace intent in report metadata.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0 if with_defaults else argparse.SUPPRESS,
        help="Transport startup/request timeout in seconds.",
    )
    parser.add_argument(
        "--safe-tool-name",
        default="echo.search" if with_defaults else argparse.SUPPRESS,
        help="Safe tool name expected to be callable by the baseline scenario.",
    )
    parser.add_argument(
        "--safe-tool-arguments-json",
        default='{"query":"smoke"}' if with_defaults else argparse.SUPPRESS,
        help="JSON object passed to the safe tool call.",
    )
    parser.add_argument(
        "--safe-resource-uri",
        default=default,
        help="Safe resource URI to read when resources are available.",
    )
    parser.add_argument(
        "--safe-prompt-name",
        default=default,
        help="Safe prompt name to get when prompts are available.",
    )
    parser.add_argument(
        "--safe-prompt-arguments-json",
        default='{"topic":"smoke"}' if with_defaults else argparse.SUPPRESS,
        help="JSON object passed to the safe prompt get call.",
    )
    return parser


async def _run(args: argparse.Namespace) -> SmokeReport:
    if args.scenario != "baseline":
        raise ValueError(f"unsupported smoke scenario: {args.scenario}")

    transport = _build_transport(args)
    report = await run_baseline_scenario(
        transport,
        mode=_scenario_mode(args.mode),
        safe_tool_name=args.safe_tool_name,
        safe_tool_arguments=_json_object(args.safe_tool_arguments_json, "safe tool arguments"),
        safe_resource_uri=args.safe_resource_uri,
        safe_prompt_name=args.safe_prompt_name,
        safe_prompt_arguments=_json_object(
            args.safe_prompt_arguments_json,
            "safe prompt arguments",
        ),
    )
    report.metadata["scenario"] = args.scenario
    report.metadata["mode"] = _scenario_mode(args.mode)
    if args.profile_id:
        report.metadata["profile_id"] = args.profile_id
    if args.debug_trace:
        report.metadata["debug_trace"] = True
    return report


def _build_transport(args: argparse.Namespace) -> McpSmokeTransport:
    api_key = _env_value(args.api_key_env, "api key") if args.api_key_env else None
    bearer_token = (
        _env_value(args.bearer_token_env, "bearer token")
        if args.bearer_token_env
        else None
    )
    timeout = _positive_timeout(args.timeout)
    transport_name = cast(str, args.transport)

    if transport_name == "inprocess":
        runtime = _build_inprocess_runtime(disable_resources=args.disable_resources)
        return InProcessGatewayTransport(runtime)
    if transport_name == "http":
        return LiveHttpTransport(
            args.url,
            bearer_token=bearer_token,
            api_key=api_key,
            profile_id=args.profile_id,
            timeout=timeout,
        )
    if transport_name == "websocket":
        return LiveWebSocketTransport(
            args.url,
            bearer_token=bearer_token,
            api_key=api_key,
            profile_id=args.profile_id,
            timeout=timeout,
        )
    if transport_name == "stdio":
        return StdioSubprocessTransport(
            args.command,
            args=tuple(args.arg),
            cwd=args.cwd,
            env_allowlist=tuple(args.env),
            startup_timeout=timeout,
            request_timeout=timeout,
        )
    raise ValueError(f"unsupported transport: {transport_name}")


def _build_inprocess_runtime(*, disable_resources: bool) -> Any:
    runtime = SmokeFixtureGatewayRuntime(include_denied_tool=True)
    if not disable_resources:
        return runtime
    return _NoResourceFixtureRuntime(runtime)


class _NoResourceFixtureRuntime:
    """Fixture runtime wrapper that intentionally omits resource methods."""

    def __init__(self, runtime: SmokeFixtureGatewayRuntime) -> None:
        self._runtime = runtime
        self.name = runtime.name
        self.version = runtime.version
        self.denied_tool_name = runtime.denied_tool_name

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        return await self._runtime.list_tools(context)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return await self._runtime.call_tool(name, arguments, context)

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        return await self._runtime.list_prompts(context)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return await self._runtime.get_prompt(name, arguments, context)

    async def list_modules(self, context: Any) -> list[dict[str, Any]]:
        return await self._runtime.list_modules(context)

    async def get_modules_health(self, context: Any) -> dict[str, Any]:
        return await self._runtime.get_modules_health(context)


def _scenario_mode(value: str) -> ScenarioMode:
    normalized = value.replace("-", "_")
    return cast(ScenarioMode, normalized)


def _json_object(raw: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _env_value(name: str, label: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise ValueError(f"{label} env var is not set: {name}")
    return value


def _positive_timeout(value: float) -> float:
    if value <= 0:
        raise ValueError("timeout must be greater than zero")
    return value


def _write_report(report: SmokeReport, destination: str | None, *, stdout: TextIO) -> None:
    payload = report_to_json(report)
    if destination == "-":
        json.dump(payload, stdout, indent=2, sort_keys=True)
        stdout.write("\n")
        return
    if destination:
        path = Path(destination)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    status = "PASS" if payload["ok"] else "FAIL"
    stdout.write(f"{status}: {report.transport}\n")
    for step in payload["steps"]:
        marker = "ok" if step["ok"] else "failed"
        reason = step.get("reason_code")
        suffix = f" ({reason})" if reason else ""
        stdout.write(f"- {step['name']}: {marker}{suffix}\n")


def _has_strict_capability_failure(report: SmokeReport) -> bool:
    if report.metadata.get("mode") != "strict":
        return False
    return any(
        step.reason_code == "required_capability_unavailable"
        for step in report.steps
        if not step.ok
    )


if __name__ == "__main__":  # pragma: no cover - exercised through console script.
    raise SystemExit(main())


__all__ = ["main"]
