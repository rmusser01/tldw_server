from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rpg_module import RPGModule
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext
from tldw_Server_API.app.core.RPG.service import RPGService


class _RPGRegistryStub:
    def __init__(self, module: RPGModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str) -> RPGModule | None:
        return self.module if tool_name in self._tool_names else None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        return "rpg" if tool_name in self._tool_names else None

    async def get_all_modules(self) -> dict[str, RPGModule]:
        return {"rpg": self.module}

    @property
    def _tool_names(self) -> set[str]:
        return {
            "rpg.adapters.list",
            "rpg.sessions.get",
            "rpg.rules.lookup",
            "rpg.context.build",
            "rpg.events.record",
            "rpg.proposals.apply",
            "rpg.proposals.reject",
        }


class _RPGPolicy:
    def __init__(self, tool_permissions: set[str], *, module_read: bool = True) -> None:
        self.tool_permissions = tool_permissions
        self.module_read = module_read

    async def check_permission(self, user_id, resource, action, resource_id=None):  # noqa: ANN001
        del user_id, action
        if resource.value == "module":
            return self.module_read and resource_id == "rpg"
        if resource.value == "tool":
            tool_id = str(resource_id or "")
            return (
                "*" in self.tool_permissions
                or tool_id in self.tool_permissions
                or any(
                    pattern.endswith(".*") and tool_id.startswith(pattern[:-1])
                    for pattern in self.tool_permissions
                )
            )
        return True


def _chacha_path(tmp_path: Path) -> str:
    return str(tmp_path / "rpg-mcp-chacha.sqlite")


def _seed_session(chacha_path: str) -> int:
    repo = RPGRepository.initialized(CharactersRAGDB(chacha_path, "rpg-mcp-seed"))
    service = RPGService(repo=repo, owner_user_id=42)
    campaign = service.create_campaign(
        "MCP Campaign",
        None,
        "fate",
        idempotency_key="mcp-campaign",
    )
    session = service.create_session(
        campaign.id,
        "MCP Session",
        adapter_key="fate",
        idempotency_key="mcp-session",
    )
    return session.id


def _context(
    tmp_path: Path,
    *,
    user_id: str | None = "42",
    allowed_tools: list[str] | None = None,
) -> RequestContext:
    metadata: dict[str, Any] = {}
    if allowed_tools is not None:
        metadata["allowed_tools"] = allowed_tools
    return RequestContext(
        request_id="rpg-mcp",
        user_id=user_id,
        client_id="unit",
        metadata=metadata,
        db_paths={"chacha": _chacha_path(tmp_path)},
    )


def _protocol(tool_permissions: set[str]) -> MCPProtocol:
    module = RPGModule(ModuleConfig(name="rpg"))
    proto = MCPProtocol()
    proto.module_registry = _RPGRegistryStub(module)
    proto.rbac_policy = _RPGPolicy(tool_permissions)
    return proto


@pytest.mark.asyncio
async def test_rpg_module_lists_read_and_write_tools() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    tools = await module.get_tools()
    tool_by_name = {tool["name"]: tool for tool in tools}

    assert tool_by_name["rpg.adapters.list"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert tool_by_name["rpg.events.record"]["metadata"]["category"] == "management"  # nosec B101
    assert tool_by_name["rpg.events.record"]["metadata"]["is_write"] is True  # nosec B101
    assert module.is_write_tool_def(tool_by_name["rpg.events.record"]) is True  # nosec B101
    assert module.is_write_tool_def(tool_by_name["rpg.context.build"]) is False  # nosec B101
    assert "rpg.proposals.apply" in tool_by_name  # nosec B101


@pytest.mark.asyncio
async def test_rpg_module_lists_adapters_without_database_context() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool("rpg.adapters.list", {}, context=None)

    assert [item["adapter_key"] for item in result["adapters"]] == ["dnd5e_srd", "fate", "pf2e"]  # nosec B101


@pytest.mark.asyncio
async def test_rpg_database_tools_fail_closed_without_user_context(tmp_path: Path) -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="authenticated user context"):
        await module.execute_tool("rpg.sessions.get", {"session_id": 1}, context=None)

    context = _context(tmp_path, user_id=None)
    with pytest.raises(ValueError, match="authenticated user context"):
        await module.execute_tool("rpg.sessions.get", {"session_id": 1}, context=context)

    missing_db_context = RequestContext(request_id="missing-db", user_id="42", client_id="unit")
    with pytest.raises(ValueError, match="ChaChaNotes DB path"):
        await module.execute_tool("rpg.sessions.get", {"session_id": 1}, context=missing_db_context)


@pytest.mark.asyncio
async def test_rpg_write_validation_runs_before_context_binding() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="idempotencyKey is required"):
        await module.execute_tool(
            "rpg.events.record",
            {
                "session_id": 1,
                "expected_last_event_sequence": 0,
                "events": [{"event_type": "note.added", "event_payload": {"text": "Missing key"}}],
            },
            context=None,
        )

    with pytest.raises(ValueError, match="expected_last_event_sequence must be non-negative"):
        await module.execute_tool(
            "rpg.events.record",
            {
                "session_id": 1,
                "expected_last_event_sequence": -1,
                "events": [{"event_type": "note.added", "event_payload": {"text": "Bad sequence"}}],
                "idempotencyKey": "bad-sequence",
            },
            context=None,
        )

    with pytest.raises(ValueError, match="idempotencyKey must be <= 256 characters"):
        await module.execute_tool(
            "rpg.events.record",
            {
                "session_id": 1,
                "expected_last_event_sequence": 0,
                "events": [{"event_type": "note.added", "event_payload": {"text": "Long key"}}],
                "idempotencyKey": "x" * 257,
            },
            context=None,
        )


@pytest.mark.asyncio
async def test_rpg_module_rejects_invalid_arguments_before_db_lookup(tmp_path: Path) -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="session_id must be a positive integer"):
        await module.execute_tool("rpg.sessions.get", {"session_id": 0}, context=_context(tmp_path))

    with pytest.raises(ValueError, match="max_chars must be between 1000 and 24000"):
        await module.execute_tool(
            "rpg.context.build",
            {"session_id": 1, "max_chars": 999},
            context=None,
        )

    with pytest.raises(ValueError, match="query must be <= 500 characters"):
        await module.execute_tool(
            "rpg.rules.lookup",
            {"session_id": 1, "query": "x" * 501},
            context=None,
        )

    with pytest.raises(ValueError, match="proposal_id must be an integer"):
        module.validate_tool_arguments(
            "rpg.proposals.apply",
            {
                "session_id": 1,
                "proposal_id": True,
                "expected_last_event_sequence": 0,
                "idempotencyKey": "bad-bool",
            },
        )


@pytest.mark.asyncio
async def test_rpg_module_gets_session_snapshot_from_chacha_context(tmp_path: Path) -> None:
    chacha_path = _chacha_path(tmp_path)
    session_id = _seed_session(chacha_path)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.sessions.get",
        {"session_id": session_id},
        context=_context(tmp_path),
    )

    assert result["session"]["id"] == session_id  # nosec B101
    assert result["snapshot"]["last_event_sequence"] == 0  # nosec B101


@pytest.mark.asyncio
async def test_protocol_denies_rpg_tool_without_execute_permission(tmp_path: Path) -> None:
    session_id = _seed_session(_chacha_path(tmp_path))
    proto = _protocol(set())

    response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rpg.sessions.get", "arguments": {"session_id": session_id}},
            id="deny-rpg",
        ),
        _context(tmp_path),
    )
    listed = await proto.process_request(MCPRequest(method="tools/list", params={}, id="list-rpg"), _context(tmp_path))
    tools = {tool["name"]: tool for tool in listed.result["tools"]}

    assert response.error is not None  # nosec B101
    assert response.error.code == -32001  # nosec B101
    assert tools["rpg.sessions.get"]["canExecute"] is False  # nosec B101


@pytest.mark.asyncio
async def test_protocol_allows_read_permission_and_denies_write_permission(tmp_path: Path) -> None:
    session_id = _seed_session(_chacha_path(tmp_path))
    proto = _protocol({"rpg.sessions.get", "rpg.adapters.list"})

    read_response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rpg.sessions.get", "arguments": {"session_id": session_id}},
            id="read-rpg",
        ),
        _context(tmp_path),
    )
    write_response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={
                "name": "rpg.events.record",
                "arguments": {
                    "session_id": session_id,
                    "expected_last_event_sequence": 0,
                    "events": [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Denied"}}],
                    "idempotencyKey": "mcp-denied-write",
                },
            },
            id="write-denied-rpg",
        ),
        _context(tmp_path),
    )

    assert read_response.error is None  # nosec B101
    assert read_response.result["tool"] == "rpg.sessions.get"  # nosec B101
    assert write_response.error is not None  # nosec B101
    assert write_response.error.code == -32001  # nosec B101


@pytest.mark.asyncio
async def test_protocol_allows_exact_write_permission_for_record_events(tmp_path: Path) -> None:
    session_id = _seed_session(_chacha_path(tmp_path))
    proto = _protocol({"rpg.events.record"})

    response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={
                "name": "rpg.events.record",
                "arguments": {
                    "session_id": session_id,
                    "expected_last_event_sequence": 0,
                    "events": [
                        {
                            "event_type": "note.added",
                            "event_payload": {"note_id": "n1", "text": "Recorded through MCP"},
                        }
                    ],
                    "idempotencyKey": "mcp-record-exact",
                },
            },
            id="write-exact-rpg",
        ),
        _context(tmp_path),
    )

    assert response.error is None  # nosec B101
    payload = response.result["content"][0]["json"]
    assert payload["committed_events"] == []  # nosec B101
    assert payload["proposal"]["status"] == "pending"  # nosec B101
    assert payload["proposal"]["proposed_events"][0]["source_type"] == "mcp"  # nosec B101


@pytest.mark.asyncio
async def test_protocol_wildcard_tool_permission_marks_rpg_tools_executable(tmp_path: Path) -> None:
    _seed_session(_chacha_path(tmp_path))
    proto = _protocol({"rpg.*"})

    response = await proto.process_request(
        MCPRequest(method="tools/list", params={}, id="list-wildcard-rpg"),
        _context(tmp_path),
    )
    tools = {tool["name"]: tool for tool in response.result["tools"]}

    assert tools["rpg.adapters.list"]["canExecute"] is True  # nosec B101
    assert tools["rpg.events.record"]["canExecute"] is True  # nosec B101
    assert tools["rpg.proposals.apply"]["canExecute"] is True  # nosec B101


@pytest.mark.asyncio
async def test_protocol_allowed_tools_metadata_denies_unlisted_rpg_tool(tmp_path: Path) -> None:
    session_id = _seed_session(_chacha_path(tmp_path))
    proto = _protocol({"rpg.*"})

    denied = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rpg.sessions.get", "arguments": {"session_id": session_id}},
            id="allowed-tools-deny-rpg",
        ),
        _context(tmp_path, allowed_tools=["notes.search"]),
    )
    allowed = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rpg.sessions.get", "arguments": {"session_id": session_id}},
            id="allowed-tools-allow-rpg",
        ),
        _context(tmp_path, allowed_tools=["rpg.sessions.get"]),
    )

    assert denied.error is not None  # nosec B101
    assert "not allowed by execution context" in denied.error.message  # nosec B101
    assert allowed.error is None  # nosec B101


async def _capture_default_registrations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[dict[str, Any]]:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    registrations: list[dict[str, Any]] = []

    async def _capture_registration(module_id: str, module_type: type[Any], config: Any) -> None:
        registrations.append(
            {"module_id": module_id, "module_type": module_type, "config": config}
        )

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(tmp_path / "missing-modules.yaml"))
    monkeypatch.delenv("MCP_MODULES", raising=False)
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_BROWSER_CDP_MODULE", "0")
    monkeypatch.delenv("MCP_BROWSER_CDP_URL", raising=False)

    await server._register_default_modules()
    return registrations


@pytest.mark.asyncio
async def test_server_registers_rpg_module_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MCP_ENABLE_RPG_MODULE", "1")

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    registration = next(item for item in registrations if item["module_id"] == "rpg")
    assert registration["module_type"].__name__ == "RPGModule"  # nosec B101
    assert registration["config"].name == "RPG"  # nosec B101
    assert registration["config"].department == "management"  # nosec B101


@pytest.mark.asyncio
async def test_server_does_not_register_rpg_module_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MCP_ENABLE_RPG_MODULE", raising=False)

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    assert "rpg" not in {item["module_id"] for item in registrations}  # nosec B101
