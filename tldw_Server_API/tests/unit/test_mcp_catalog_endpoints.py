"""Tests for the MCP catalog API endpoints (list and connection test)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import (
    MCPConnectionTestRequest,
    list_mcp_catalog,
    check_mcp_connection,
)
import tldw_Server_API.app.core.MCP_unified.catalog_loader as _catalog_mod
from tldw_Server_API.app.core.MCP_unified.catalog_loader import load_mcp_catalog

pytestmark = pytest.mark.unit

# Path to the real catalog YAML shipped with the project.
_CATALOG_YAML = (
    Path(__file__).resolve().parents[2]
    / "Config_Files"
    / "mcp_server_catalog.yaml"
)


@pytest.fixture(autouse=True)
def _load_real_catalog():
    """Load the real catalog YAML before each test and clear after."""
    _catalog_mod._CATALOG_CACHE = []
    load_mcp_catalog(_CATALOG_YAML)
    yield
    _catalog_mod._CATALOG_CACHE = []


# -- list_mcp_catalog --------------------------------------------------------


@pytest.mark.asyncio
async def test_list_catalog():
    """Calling with no filter should return all catalog entries."""
    result = await list_mcp_catalog()

    assert isinstance(result, list)
    assert len(result) >= 1
    keys = {e.key for e in result}
    # The real YAML should contain at least these well-known entries.
    assert "github" in keys
    assert "arxiv" in keys


@pytest.mark.asyncio
async def test_list_catalog_with_archetype_filter():
    """Filtering by archetype_key should narrow the returned entries."""
    all_entries = await list_mcp_catalog()

    filtered = await list_mcp_catalog(archetype_key="research_assistant")

    assert isinstance(filtered, list)
    assert len(filtered) >= 1
    # Filtered set must be a subset of the full catalog.
    assert len(filtered) <= len(all_entries)
    # Every returned entry must list the requested archetype.
    for entry in filtered:
        assert "research_assistant" in entry.suggested_for


@pytest.mark.asyncio
async def test_list_catalog_unknown_archetype_returns_empty():
    """An archetype key that no entry maps to should yield an empty list."""
    result = await list_mcp_catalog(archetype_key="nonexistent_archetype_xyz")

    assert result == []


# -- test_mcp_connection -----------------------------------------------------


@pytest.mark.asyncio
async def test_connection_unreachable(monkeypatch: pytest.MonkeyPatch):
    """Catalog connection failures should return a safe unreachable response."""
    req = MCPConnectionTestRequest(url="https://api.github.com")

    async def _failing_probe(_url: str, _headers: dict[str, str]) -> None:
        raise OSError("network down")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._is_private_ip",
        lambda _host: False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._probe_mcp_connection",
        _failing_probe,
    )

    resp = await check_mcp_connection(req)

    assert resp.reachable is False
    assert resp.error is not None
    assert isinstance(resp.error, str)
    assert resp.tools_discovered == []


@pytest.mark.asyncio
async def test_connection_uses_api_key_header(monkeypatch: pytest.MonkeyPatch):
    captured_headers: dict[str, str] = {}

    async def _fake_probe(url: str, headers: dict[str, str]):
        assert url == "https://api.github.com"
        captured_headers.update(headers)
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._is_private_ip",
        lambda _host: False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._probe_mcp_connection",
        _fake_probe,
    )

    req = MCPConnectionTestRequest(
        url="https://api.github.com",
        auth_type="api_key",
        secret="secret-token",
    )

    resp = await check_mcp_connection(req)

    assert resp.reachable is True
    assert captured_headers == {"X-API-Key": "secret-token"}


@pytest.mark.asyncio
async def test_connection_rejects_unsupported_auth_type():
    req = MCPConnectionTestRequest(
        url="https://api.github.com",
        auth_type="digest",
        secret="secret-token",
    )

    with pytest.raises(HTTPException) as excinfo:
        await check_mcp_connection(req)

    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_connection_uses_central_url_safety_guard(monkeypatch: pytest.MonkeyPatch):
    captured_urls: list[str] = []

    def _fake_assert_url_safe(url: str) -> None:
        captured_urls.append(url)
        raise HTTPException(status_code=400, detail="blocked")

    async def _unexpected_probe(_url: str, _headers: dict[str, str]) -> None:  # pragma: no cover - defensive
        raise AssertionError("_probe_mcp_connection should not run when the safety guard blocks")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint.assert_url_safe",
        _fake_assert_url_safe,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._is_private_ip",
        lambda _host: False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._probe_mcp_connection",
        _unexpected_probe,
    )

    req = MCPConnectionTestRequest(url="https://api.github.com")

    with pytest.raises(HTTPException) as excinfo:
        await check_mcp_connection(req)

    assert captured_urls == ["https://api.github.com"]
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_connection_rejects_uncurated_url(monkeypatch: pytest.MonkeyPatch):
    async def _unexpected_probe(_url: str, _headers: dict[str, str]) -> None:  # pragma: no cover - defensive
        raise AssertionError("_probe_mcp_connection should not run for uncataloged URLs")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint._probe_mcp_connection",
        _unexpected_probe,
    )

    resp = await check_mcp_connection(MCPConnectionTestRequest(url="https://8.8.8.8/mcp"))

    assert resp.reachable is False
    assert resp.error == "URL must match a curated MCP catalog entry"
