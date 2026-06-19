"""Smoke-test helpers for MCP Unified JSON-RPC clients."""

from mcp_unified.smoke.client import McpSmokeClient
from mcp_unified.smoke.exceptions import McpSmokeClientError, McpSmokeTransportError
from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
from mcp_unified.smoke.reporting import (
    SmokeReport,
    SmokeStepReport,
    SmokeTraceSummary,
    redact_detail,
    report_to_json,
    summarize_result,
)
from mcp_unified.smoke.scenarios import run_baseline_scenario
from mcp_unified.smoke.transports import (
    InProcessFastApiTransport,
    InProcessGatewayTransport,
    LiveHttpTransport,
    LiveWebSocketTransport,
    StdioSubprocessTransport,
)
from mcp_unified.smoke.types import McpSmokeTransport

__all__ = [
    "InProcessFastApiTransport",
    "InProcessGatewayTransport",
    "LiveHttpTransport",
    "LiveWebSocketTransport",
    "McpSmokeClient",
    "McpSmokeClientError",
    "McpSmokeTransport",
    "McpSmokeTransportError",
    "StdioSubprocessTransport",
    "SmokeFixtureGatewayRuntime",
    "SmokeReport",
    "SmokeStepReport",
    "SmokeTraceSummary",
    "redact_detail",
    "report_to_json",
    "run_baseline_scenario",
    "summarize_result",
]
