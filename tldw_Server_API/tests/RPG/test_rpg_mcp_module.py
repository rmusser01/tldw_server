from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import rpg_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rpg_module import RPGModule
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RPG.context import SessionContext
from tldw_Server_API.app.core.RPG.rules.answering import RulesAnswerOptions
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupResult
from tldw_Server_API.app.core.RPG.service import RPGService

_MCP_RPG_TEST_PERMISSIONS = [
    "rpg.campaigns.read",
    "rpg.campaigns.manage",
    "rpg.sessions.read",
    "rpg.sessions.manage",
    "rpg.proposals.review",
    "rpg.rules.read",
    "media.read",
    "chat.completions",
]


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
            "rpg.campaigns.rules_packs.get",
            "rpg.campaigns.rules_packs.replace",
            "rpg.sessions.rules_packs.get",
            "rpg.sessions.rules_packs.replace",
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


def _seed_campaign_and_session(chacha_path: str) -> tuple[int, int]:
    repo = RPGRepository.initialized(CharactersRAGDB(chacha_path, "rpg-mcp-campaign-seed"))
    service = RPGService(repo=repo, owner_user_id=42)
    campaign = service.create_campaign(
        "MCP Campaign",
        None,
        "fate",
        idempotency_key="mcp-campaign-pair",
    )
    session = service.create_session(
        campaign.id,
        "MCP Session",
        adapter_key="fate",
        idempotency_key="mcp-session-pair",
    )
    return campaign.id, session.id


def _seed_session_with_rules_pack(chacha_path: str, media_id: int) -> int:
    repo = RPGRepository.initialized(CharactersRAGDB(chacha_path, "rpg-mcp-rules-seed"))
    service = RPGService(repo=repo, owner_user_id=42)
    campaign = service.create_campaign(
        "MCP Rules Campaign",
        None,
        "fate",
        idempotency_key="mcp-rules-campaign",
    )
    session = service.create_session(
        campaign.id,
        "MCP Rules Session",
        adapter_key="fate",
        idempotency_key="mcp-rules-session",
    )
    repo.replace_session_rules_pack_refs(
        owner_user_id=42,
        session_id=session.id,
        expected_version=session.version,
        rules_pack_refs=[
            {
                "ref_id": f"media_item:{media_id}",
                "source_type": "media_item",
                "source_id": media_id,
                "display_name": "MCP Rules",
                "enabled": True,
                "metadata": {},
            }
        ],
        idempotency_key="mcp-rules-pack-refs",
        request_payload_hash="mcp-rules-pack-refs-hash",
        source_type="mcp",
    )
    return session.id


def _seed_media(media_path: str, *, title: str, content: str) -> int:
    db = MediaDatabase(db_path=media_path, client_id="42")
    try:
        media_uuid = str(uuid.uuid4())
        last_modified = db._get_current_utc_timestamp_str()
        cursor = db.execute_query(
            "INSERT INTO Media (title, type, content, author, content_hash, uuid, last_modified, client_id, owner_user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (title, "document", content, "Test Author", f"hash:{media_uuid}", media_uuid, last_modified, "42", 42),
            commit=True,
        )
        media_id = getattr(cursor, "lastrowid", None)
        if media_id:
            return int(media_id)
        row = db.execute_query("SELECT id FROM Media WHERE uuid = ?", (media_uuid,)).fetchone()
        return int(row["id"] if isinstance(row, dict) else row[0])
    finally:
        db.close_connection()


def _context(
    tmp_path: Path,
    *,
    user_id: str | None = "42",
    allowed_tools: list[str] | None = None,
    media_path: str | None = None,
    permissions: list[str] | None = None,
    answer_generation_controls: bool = True,
) -> RequestContext:
    metadata: dict[str, Any] = {
        "permissions": list(_MCP_RPG_TEST_PERMISSIONS if permissions is None else permissions),
    }
    if answer_generation_controls:
        metadata["mcp_rpg_answer_generation_controls"] = "enforced"
    if allowed_tools is not None:
        metadata["allowed_tools"] = allowed_tools
    return RequestContext(
        request_id="rpg-mcp",
        user_id=user_id,
        client_id="unit",
        metadata=metadata,
        db_paths={"chacha": _chacha_path(tmp_path), **({"media": media_path} if media_path else {})},
    )


class _FakeRagRetriever:
    calls: list[dict[str, Any]] = []
    instances: list[_FakeRagRetriever] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.close_called = False
        self.__class__.instances.append(self)

    def close(self):
        self.close_called = True

    async def retrieve_from_plan(self, plan, **kwargs):
        self.__class__.calls.append({"plan": plan, **kwargs})
        media_id = int(kwargs["allowed_media_ids"][0])
        return [
            Document(
                id=str(media_id),
                content="MCP retrieved rules evidence",
                metadata={"media_id": media_id, "title": "MCP Rules"},
                source=DataSource.MEDIA_DB,
                score=0.9,
            )
        ]


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
async def test_rpg_mcp_tool_list_includes_rules_pack_ref_tools() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    tools = await module.get_tools()
    tool_names = {tool["name"] for tool in tools}

    assert {  # nosec B101
        "rpg.campaigns.rules_packs.get",
        "rpg.campaigns.rules_packs.replace",
        "rpg.sessions.rules_packs.get",
        "rpg.sessions.rules_packs.replace",
    } <= tool_names


@pytest.mark.asyncio
async def test_rpg_mcp_rules_pack_ref_tools_have_read_write_metadata() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    tools = await module.get_tools()
    tool_by_name = {tool["name"]: tool for tool in tools}

    campaign_get = tool_by_name["rpg.campaigns.rules_packs.get"]
    session_replace = tool_by_name["rpg.sessions.rules_packs.replace"]

    assert campaign_get["metadata"]["readOnlyHint"] is True  # nosec B101
    assert campaign_get["metadata"]["required_permissions"] == ["rpg.campaigns.read", "media.read"]  # nosec B101
    assert session_replace["metadata"]["readOnlyHint"] is False  # nosec B101
    assert session_replace["metadata"]["required_permissions"] == ["rpg.sessions.manage", "media.read"]  # nosec B101
    assert module.is_write_tool_def(session_replace) is True  # nosec B101


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

    missing_db_context = RequestContext(
        request_id="missing-db",
        user_id="42",
        client_id="unit",
        metadata={"permissions": list(_MCP_RPG_TEST_PERMISSIONS)},
    )
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
async def test_rpg_mcp_rules_pack_ref_replace_validates_expected_version() -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="expected_version must be a positive integer"):
        await module.execute_tool(
            "rpg.sessions.rules_packs.replace",
            {
                "session_id": 1,
                "expected_version": 0,
                "refs": [],
                "idempotencyKey": "mcp-rules-pack-version",
            },
            context=None,
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
async def test_rpg_module_session_lookup_does_not_open_media_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chacha_path = _chacha_path(tmp_path)
    session_id = _seed_session(chacha_path)

    def fail_media_database(*args, **kwargs):
        raise AssertionError("media db should not be opened for session metadata")

    monkeypatch.setattr(rpg_module, "MediaDatabase", fail_media_database)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.sessions.get",
        {"session_id": session_id},
        context=_context(tmp_path, media_path=str(tmp_path / "unused-media.sqlite")),
    )

    assert result["session"]["id"] == session_id  # nosec B101


@pytest.mark.asyncio
async def test_rpg_module_rules_lookup_uses_attached_media_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_path = str(tmp_path / "rpg-mcp-media.sqlite")
    media_id = _seed_media(media_path, title="MCP Rules", content="MCP rules source")
    session_id = _seed_session_with_rules_pack(_chacha_path(tmp_path), media_id)
    _FakeRagRetriever.calls = []
    _FakeRagRetriever.instances = []
    monkeypatch.setattr(rpg_module, "MultiDatabaseRetriever", _FakeRagRetriever, raising=False)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.rules.lookup",
        {"session_id": session_id, "query": "stress"},
        context=_context(tmp_path, media_path=media_path),
    )

    assert result["results"][0]["origin"] == "user_provided"  # nosec B101
    assert result["results"][0]["text"] == "MCP retrieved rules evidence"  # nosec B101
    assert _FakeRagRetriever.calls[0]["allowed_media_ids"] == [media_id]  # nosec B101
    assert _FakeRagRetriever.instances[0].close_called is True  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_tool_requires_declared_domain_permissions(tmp_path: Path) -> None:
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(PermissionError, match="media.read"):
        await module.execute_tool(
            "rpg.rules.lookup",
            {"session_id": 1, "query": "stress"},
            context=_context(tmp_path, permissions=["rpg.rules.read"]),
        )

    with pytest.raises(PermissionError, match="rpg.sessions.manage"):
        await module.execute_tool(
            "rpg.sessions.rules_packs.replace",
            {
                "session_id": 1,
                "expected_version": 1,
                "refs": [],
                "idempotencyKey": "mcp-rules-pack-missing-manage",
            },
            context=_context(tmp_path, permissions=["media.read"]),
        )


@pytest.mark.asyncio
async def test_protocol_denies_rpg_tool_when_domain_permission_missing(tmp_path: Path) -> None:
    _seed_session(_chacha_path(tmp_path))
    proto = _protocol({"rpg.rules.lookup"})

    response = await proto.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rpg.rules.lookup", "arguments": {"session_id": 1, "query": "stress"}},
            id="deny-domain-rpg",
        ),
        _context(tmp_path, permissions=["rpg.rules.read"]),
    )

    assert response.error is not None  # nosec B101
    assert response.error.code == -32001  # nosec B101
    assert response.error.message == "tool_execution_error"  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_answer_mode_requires_chat_permissions_and_generation_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_lookup_rules(self, *, session_id, query, mode="lookup", answer_options=None):
        nonlocal called
        called = True
        return RuleLookupResult(
            query=query,
            mode=mode,
            results=[],
            answer=None,
            answer_status="no_evidence",
            answer_citation_ids=[],
            diagnostics={},
        )

    monkeypatch.setattr(RPGService, "lookup_rules", fake_lookup_rules)
    module = RPGModule(ModuleConfig(name="rpg"))
    args = {"session_id": 1, "query": "stress", "mode": "answer"}

    with pytest.raises(PermissionError, match="chat.completions"):
        await module.execute_tool(
            "rpg.rules.lookup",
            args,
            context=_context(tmp_path, permissions=["rpg.rules.read", "media.read"]),
        )

    with pytest.raises(PermissionError, match="answer generation controls"):
        await module.execute_tool(
            "rpg.rules.lookup",
            args,
            context=_context(
                tmp_path,
                permissions=["rpg.rules.read", "media.read", "chat.completions"],
                answer_generation_controls=False,
            ),
        )

    assert called is False  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_session_rules_pack_replace_succeeds_and_replays(tmp_path: Path) -> None:
    media_path = str(tmp_path / "rpg-mcp-replace-media.sqlite")
    media_id = _seed_media(media_path, title="Replacement Rules", content="replacement rules")
    _, session_id = _seed_campaign_and_session(_chacha_path(tmp_path))
    module = RPGModule(ModuleConfig(name="rpg"))
    args = {
        "session_id": session_id,
        "expected_version": 1,
        "refs": [{"source_type": "media_item", "source_id": media_id}],
        "idempotency_key": "mcp-rules-pack-replace-success",
    }
    context = _context(tmp_path, media_path=media_path)

    result = await module.execute_tool("rpg.sessions.rules_packs.replace", args, context=context)
    replay = await module.execute_tool("rpg.sessions.rules_packs.replace", args, context=context)

    assert result["version"] == 2  # nosec B101
    assert result["refs"][0]["source_id"] == media_id  # nosec B101
    assert replay["replayed"] is True  # nosec B101

    with pytest.raises(Exception, match="idempotency_key_conflict"):
        await module.execute_tool(
            "rpg.sessions.rules_packs.replace",
            {**args, "refs": []},
            context=context,
        )


@pytest.mark.asyncio
async def test_rpg_mcp_rules_pack_replace_does_not_construct_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = str(tmp_path / "rpg-mcp-replace-media-no-retriever.sqlite")
    media_id = _seed_media(media_path, title="Replacement Rules", content="replacement rules")
    _, session_id = _seed_campaign_and_session(_chacha_path(tmp_path))

    def fail_retriever(*args, **kwargs):
        raise AssertionError("rules-pack replacement should not construct a retriever")

    monkeypatch.setattr(rpg_module, "MultiDatabaseRetriever", fail_retriever)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.sessions.rules_packs.replace",
        {
            "session_id": session_id,
            "expected_version": 1,
            "refs": [{"source_type": "media_item", "source_id": media_id}],
            "idempotency_key": "mcp-rules-pack-replace-no-retriever",
        },
        context=_context(tmp_path, media_path=media_path),
    )

    assert result["refs"][0]["source_id"] == media_id  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_service_construction_closes_opened_resources_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class FakeChaChaDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def close_connection(self):
            closed.append("chacha")

    class FakeMediaDB:
        backend = object()

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def close_connection(self):
            closed.append("media")

    def fail_retriever(*args, **kwargs):
        raise RuntimeError("retriever unavailable")

    monkeypatch.setattr(rpg_module, "CharactersRAGDB", FakeChaChaDB)
    monkeypatch.setattr(rpg_module, "MediaDatabase", FakeMediaDB)
    monkeypatch.setattr(rpg_module.CollectionsDatabase, "from_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(rpg_module, "MultiDatabaseRetriever", fail_retriever)
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(RuntimeError, match="retriever unavailable"):
        await module.execute_tool(
            "rpg.rules.lookup",
            {"session_id": 1, "query": "stress"},
            context=_context(tmp_path, media_path=str(tmp_path / "media.sqlite")),
        )

    assert closed == ["media", "chacha"]  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_rules_lookup_accepts_answer_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_lookup_rules(self, *, session_id, query, mode="lookup", answer_options=None):
        captured.update(
            {
                "session_id": session_id,
                "query": query,
                "mode": mode,
                "answer_options": answer_options,
            }
        )
        return RuleLookupResult(
            query=query,
            mode=mode,
            results=[],
            answer=None,
            answer_status="no_evidence",
            answer_citation_ids=[],
            diagnostics={
                "bundled_policy": "no_match",
                "result_mode": "citation_index",
                "linked_rules_pack_count": 0,
                "enabled_rules_pack_count": 0,
                "ready_media_item_count": 0,
                "retrieval_result_count": 0,
                "bundled_citation_count": 0,
                "skipped_refs": [],
                "broad_fallback_used": False,
            },
        )

    monkeypatch.setattr(RPGService, "lookup_rules", fake_lookup_rules)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.rules.lookup",
        {
            "session_id": 1,
            "query": "stress",
            "mode": "answer",
            "provider": "openai",
            "model": "gpt-test",
            "temperature": 0.4,
            "max_tokens": 321,
        },
        context=_context(tmp_path),
    )

    assert result["mode"] == "answer"  # nosec B101
    assert captured["mode"] == "answer"  # nosec B101
    options = captured["answer_options"]
    assert isinstance(options, RulesAnswerOptions)  # nosec B101
    assert options.provider == "openai"  # nosec B101
    assert options.model == "gpt-test"  # nosec B101
    assert options.temperature == 0.4  # nosec B101
    assert options.max_tokens == 321  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_context_build_awaits_async_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_build_context(self, *, session_id, query=None, max_chars=24000):
        captured.update({"session_id": session_id, "query": query, "max_chars": max_chars})
        return SessionContext(text="async context", diagnostics={"rules_lookup": {}})

    monkeypatch.setattr(RPGService, "build_context", fake_build_context)
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool(
        "rpg.context.build",
        {"session_id": 7, "query": "stress", "max_chars": 1000},
        context=_context(tmp_path),
    )

    assert result["text"] == "async context"  # nosec B101
    assert captured == {"session_id": 7, "query": "stress", "max_chars": 1000}  # nosec B101


@pytest.mark.asyncio
async def test_rpg_mcp_read_tools_require_media_read_for_attached_refs(tmp_path: Path) -> None:
    session_id = _seed_session_with_rules_pack(_chacha_path(tmp_path), media_id=123)
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="Media DB path not available"):
        await module.execute_tool(
            "rpg.rules.lookup",
            {"session_id": session_id, "query": "stress"},
            context=_context(tmp_path),
        )


@pytest.mark.asyncio
async def test_rpg_mcp_write_tools_require_media_read_for_source_validation(tmp_path: Path) -> None:
    _, session_id = _seed_campaign_and_session(_chacha_path(tmp_path))
    module = RPGModule(ModuleConfig(name="rpg"))

    with pytest.raises(ValueError, match="Media DB path not available"):
        await module.execute_tool(
            "rpg.sessions.rules_packs.replace",
            {
                "session_id": session_id,
                "expected_version": 1,
                "refs": [{"source_type": "media_item", "source_id": 123}],
                "idempotencyKey": "mcp-rules-pack-no-media",
            },
            context=_context(tmp_path),
        )


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
