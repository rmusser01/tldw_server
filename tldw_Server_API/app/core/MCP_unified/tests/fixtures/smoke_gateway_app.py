"""Standalone FastAPI gateway app used by live smoke validation."""

from mcp_unified.gateway.fastapi import create_gateway_app
from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime

app = create_gateway_app(SmokeFixtureGatewayRuntime(), prefix="/mcp")
