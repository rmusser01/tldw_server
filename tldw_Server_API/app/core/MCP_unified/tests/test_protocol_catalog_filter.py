import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest


class _NoopRateLimiter:
    async def check_rate_limit(self, _key: str, *, category: str = "default") -> None:
        del category
        return None


class _NoopMetrics:
    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class _NoopTelemetry:
    def trace_context(
        self,
        _operation_name: str,
        _attributes: dict[str, Any] | None = None,
    ) -> Any:
        class _Span:
            def set_attribute(self, _key: str, _value: Any) -> None:
                return None

        class _Context:
            def __enter__(self) -> _Span:
                return _Span()

            def __exit__(self, *_exc_info: Any) -> None:
                return None

        return _Context()


def _protocol_dependencies(*, tool_catalog_provider: Any) -> SimpleNamespace:
    return SimpleNamespace(
        module_registry=object(),
        rbac_policy=object(),
        rate_limiter=_NoopRateLimiter(),
        metrics_collector=_NoopMetrics(),
        telemetry_provider=_NoopTelemetry(),
        redis_client_factory=lambda **_kwargs: None,
        tool_catalog_provider=tool_catalog_provider,
        effective_policy_resolver=object(),
        approval_evaluator=object(),
        path_scope_enforcer=object(),
        external_access_evaluator=object(),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_catalog_resolution_uses_injected_provider() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ToolCatalogProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def resolve_tool_names(
            self,
            *,
            catalog_name: str | None,
            catalog_id: Any,
            metadata: dict[str, Any],
            strict: bool,
        ) -> set[str] | None:
            self.calls.append(
                {
                    "catalog_name": catalog_name,
                    "catalog_id": catalog_id,
                    "metadata": metadata,
                    "strict": strict,
                }
            )
            return {"media.search"}

    provider = _ToolCatalogProvider()
    proto = MCPProtocol(dependencies=_protocol_dependencies(tool_catalog_provider=provider))
    ctx = RequestContext(
        request_id="catalog-provider",
        user_id="1",
        client_id="unit",
        metadata={"team_id": 7, "org_id": 5},
    )

    resolved = await proto._resolve_catalog_tool_names(  # noqa: SLF001
        {"catalog": "A", "catalog_id": "123", "catalog_strict": "yes"},
        ctx,
    )

    assert resolved == {"media.search"}
    assert provider.calls == [
        {
            "catalog_name": "A",
            "catalog_id": "123",
            "metadata": {"team_id": 7, "org_id": 5},
            "strict": True,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_catalog_provider_failure_fails_closed_by_default() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _FailingToolCatalogProvider:
        async def resolve_tool_names(
            self,
            *,
            catalog_name: str | None,
            catalog_id: Any,
            metadata: dict[str, Any],
            strict: bool,
        ) -> set[str] | None:
            del catalog_name, catalog_id, metadata, strict
            raise RuntimeError("catalog lookup failed at /private/authnz.db")

    proto = MCPProtocol(
        dependencies=_protocol_dependencies(tool_catalog_provider=_FailingToolCatalogProvider())
    )
    ctx = RequestContext(request_id="catalog-provider-failure", user_id="1", client_id="unit")

    non_strict = await proto._resolve_catalog_tool_names(  # noqa: SLF001
        {"catalog": "A"},
        ctx,
    )
    fail_open = await proto._resolve_catalog_tool_names(  # noqa: SLF001
        {"catalog": "A", "catalog_fail_open": True},
        ctx,
    )
    strict = await proto._resolve_catalog_tool_names(  # noqa: SLF001
        {"catalog": "A", "catalog_strict": True},
        ctx,
    )
    strict_fail_open = await proto._resolve_catalog_tool_names(  # noqa: SLF001
        {"catalog": "A", "catalog_strict": True, "catalog_fail_open": True},
        ctx,
    )

    assert non_strict == set()
    assert fail_open is None
    assert strict == set()
    assert strict_fail_open == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tldw_tool_catalog_provider_handles_tuple_rows(monkeypatch):
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    class _PoolStub:
        async def fetchone(self, query: str, *args):
            assert "tool_catalogs" in query
            assert args == ("A", 7)
            return (123,)

        async def fetchall(self, query: str, *args):
            assert "tool_catalog_entries" in query
            assert args == (123,)
            return [("media.search",), (), {"tool_name": "notes.search"}]

    async def _get_db_pool_stub():
        return _PoolStub()

    import tldw_Server_API.app.core.AuthNZ.database as db_mod

    monkeypatch.setattr(db_mod, "get_db_pool", _get_db_pool_stub)

    provider = tldw_runtime.TldwToolCatalogProvider()

    resolved = await provider.resolve_tool_names(
        catalog_name="A",
        catalog_id=None,
        metadata={"team_id": 7},
        strict=True,
    )

    assert resolved == {"media.search", "notes.search"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tldw_tool_catalog_provider_sanitizes_failure_logs(monkeypatch):
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    async def _get_db_pool_stub():
        raise RuntimeError("catalog lookup failed at /private/authnz.db")

    records: list[tuple[str, tuple[Any, ...]]] = []

    def _debug(message: str, *args: Any, **_kwargs: Any) -> None:
        records.append((message, args))

    import tldw_Server_API.app.core.AuthNZ.database as db_mod

    monkeypatch.setattr(db_mod, "get_db_pool", _get_db_pool_stub)
    monkeypatch.setattr(tldw_runtime, "logger", SimpleNamespace(debug=_debug))

    provider = tldw_runtime.TldwToolCatalogProvider()

    resolved = await provider.resolve_tool_names(
        catalog_name="A",
        catalog_id=None,
        metadata={},
        strict=True,
    )

    logged = "\n".join([message + " " + " ".join(map(str, args)) for message, args in records])
    assert resolved == set()
    assert "RuntimeError" in logged
    assert "/private/" not in logged
    assert "catalog lookup failed" not in logged


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_list_catalog_filter(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    # Stub DB pool with catalog id resolution and entries
    class _PoolStub:
        async def fetchone(self, query: str, *args):
            # Return an id for catalog name resolution
            return {"id": 123}

        async def fetchall(self, query: str, *args):
            # Return only 'media.search' in catalog
            return [{"tool_name": "media.search"}]

    async def _get_db_pool_stub():
        return _PoolStub()

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "false")
    monkeypatch.setenv("MCP_MODULES", "")

    # Patch DB pool getter
    import tldw_Server_API.app.core.AuthNZ.database as db_mod
    monkeypatch.setattr(db_mod, "get_db_pool", _get_db_pool_stub)

    # Stub module registry with a single module exposing two tools
    class _ModuleStub:
        name = "Media"

        async def get_tools(self):
            return [
                {"name": "media.search", "inputSchema": {"type": "object"}},
                {"name": "ingest_media", "inputSchema": {"type": "object"}},
            ]

    class _RegistryStub:
        async def get_all_modules(self):
            return {"media": _ModuleStub()}

    # Build protocol and monkeypatch registry and RBAC checks
    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()
    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(request_id="test", user_id="1", client_id="unit", session_id=None, metadata={})

    # Run tools/list with catalog
    result = await proto._handle_tools_list({"catalog": "A"}, ctx)
    assert isinstance(result, dict)
    tools = result.get("tools", [])
    names = {t.get("name") for t in tools}
    assert "media.search" in names
    assert "ingest_media" not in names  # filtered out by catalog
    assert result.get("_meta", {}).get("catalog", {}).get("status") == "resolved"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_list_unresolved_catalog_returns_empty_with_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolved catalog filters should return no tools and explicit metadata."""
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ToolCatalogProvider:
        """Catalog provider that simulates an unresolved catalog lookup."""

        async def resolve_tool_names(
            self,
            *,
            catalog_name: str | None,
            catalog_id: Any,
            metadata: dict[str, Any],
            strict: bool,
        ) -> set[str] | None:
            del catalog_name, catalog_id, metadata, strict
            return None

    class _ModuleStub:
        """Module stub with two discoverable media tools."""

        name = "Media"

        async def get_tools(self) -> list[dict[str, Any]]:
            """Return tools that should be hidden when the catalog is unresolved."""
            return [
                {"name": "media.search", "inputSchema": {"type": "object"}},
                {"name": "ingest_media", "inputSchema": {"type": "object"}},
            ]

    class _RegistryStub:
        """Registry stub exposing the media module."""

        async def get_all_modules(self) -> dict[str, Any]:
            """Return registered modules for tools/list."""
            return {"media": _ModuleStub()}

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "false")
    monkeypatch.setenv("MCP_MODULES", "")

    proto = MCPProtocol(dependencies=_protocol_dependencies(tool_catalog_provider=_ToolCatalogProvider()))
    proto.module_registry = _RegistryStub()

    async def _allow_mod(*_args: Any, **_kwargs: Any) -> bool:
        """Allow all module discovery in this test."""
        return True

    async def _allow_tool(*_args: Any, **_kwargs: Any) -> bool:
        """Allow all tool discovery in this test."""
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(request_id="catalog-miss", user_id="1", client_id="unit", metadata={})
    result = await proto._handle_tools_list({"catalog": "typo-catalog"}, ctx)

    assert result.get("tools") == []
    assert result.get("_meta", {}).get("catalog") == {
        "status": "unresolved",
        "filtered": True,
        "toolCount": 0,
        "hint": "Check catalog name/id or remove the catalog filter.",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_catalog_strict_overrides_fail_open_for_unresolved_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict catalog lookup should still fail closed when fail-open is also requested."""
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ToolCatalogProvider:
        """Catalog provider that exposes whether strict was forwarded."""

        def __init__(self) -> None:
            """Track strict values passed by the protocol."""
            self.strict_values: list[bool] = []

        async def resolve_tool_names(
            self,
            *,
            catalog_name: str | None,
            catalog_id: Any,
            metadata: dict[str, Any],
            strict: bool,
        ) -> set[str] | None:
            """Return strict-mode empty sets and fail-open None otherwise."""
            del catalog_name, catalog_id, metadata
            self.strict_values.append(strict)
            return set() if strict else None

    class _ModuleStub:
        """Module stub with tools that strict catalog lookup should hide."""

        name = "Media"

        async def get_tools(self) -> list[dict[str, Any]]:
            """Return tools that should be hidden when strict lookup cannot resolve."""
            return [
                {"name": "media.search", "inputSchema": {"type": "object"}},
                {"name": "ingest_media", "inputSchema": {"type": "object"}},
            ]

    class _RegistryStub:
        """Registry stub exposing the media module."""

        async def get_all_modules(self) -> dict[str, Any]:
            """Return registered modules for tools/list."""
            return {"media": _ModuleStub()}

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "false")
    monkeypatch.setenv("MCP_MODULES", "")

    provider = _ToolCatalogProvider()
    proto = MCPProtocol(dependencies=_protocol_dependencies(tool_catalog_provider=provider))
    proto.module_registry = _RegistryStub()

    async def _allow_mod(*_args: Any, **_kwargs: Any) -> bool:
        """Allow all module discovery in this test."""
        return True

    async def _allow_tool(*_args: Any, **_kwargs: Any) -> bool:
        """Allow all tool discovery in this test."""
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(request_id="catalog-fail-open", user_id="1", client_id="unit", metadata={})
    result = await proto._handle_tools_list(
        {"catalog": "typo-catalog", "catalog_strict": True, "catalog_fail_open": True},
        ctx,
    )

    assert provider.strict_values == [True]
    assert result.get("tools") == []
    assert result.get("_meta", {}).get("catalog") == {
        "status": "unresolved",
        "filtered": True,
        "toolCount": 0,
        "hint": "Check catalog name/id or remove the catalog filter.",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_list_module_filter(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _MediaModuleStub:
        name = "Media"

        async def get_tools(self):
            return [{"name": "media.search", "inputSchema": {"type": "object"}}]

    class _NotesModuleStub:
        name = "Notes"

        async def get_tools(self):
            return [{"name": "notes.search", "inputSchema": {"type": "object"}}]

    class _RegistryStub:
        async def get_all_modules(self):
            return {"media": _MediaModuleStub(), "notes": _NotesModuleStub()}

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(request_id="mod-filter", user_id="1", client_id="unit", session_id=None, metadata={})
    result = await proto._handle_tools_list({"module": "notes"}, ctx)
    tools = result.get("tools", [])
    names = {t.get("name") for t in tools}
    assert names == {"notes.search"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_tools_list_scope_filters_tools(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _ModuleStub:
        name = "Media"

        async def get_tools(self):
            return [
                {"name": "media.search", "inputSchema": {"type": "object"}},
                {"name": "media.get", "inputSchema": {"type": "object"}},
            ]

    class _RegistryStub:
        async def get_all_modules(self):
            return {"media": _ModuleStub()}

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_mod(ctx, mid):
        return True

    async def _allow_tool(ctx, name, **_kwargs):
        return True

    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    ctx = RequestContext(
        request_id="scoped",
        user_id="1",
        client_id="unit",
        session_id=None,
        metadata={"permissions": ["mcp:tool:media.search"]},
    )

    result = await proto._handle_tools_list({}, ctx)
    tools = result.get("tools", [])
    names = {t.get("name") for t in tools}
    assert names == {"media.search"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_catalog_resolution_precedence(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    # Stub DB pool to simulate team/org/global resolution
    class _PoolStub:
        async def fetchone(self, query: str, *args):
            # Team first
            if "team_id = ?" in query and args == ("A", 7):
                return {"id": 100}
            # Org next
            if "org_id = ?" in query and "team_id IS NULL" in query and args == ("A", 5):
                return {"id": 200}
            # Global last
            if "org_id IS NULL" in query and "team_id IS NULL" in query:
                return {"id": 300}
            return None

        async def fetchall(self, query: str, *args):
            # Return a distinct tool per resolved catalog id
            cat_id = args[0]
            if cat_id == 100:
                return [{"tool_name": "team.only"}]
            if cat_id == 200:
                return [{"tool_name": "org.only"}]
            if cat_id == 300:
                return [{"tool_name": "global.only"}]
            return []

    async def _get_db_pool_stub():
        return _PoolStub()

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "false")
    monkeypatch.setenv("MCP_MODULES", "")

    import tldw_Server_API.app.core.AuthNZ.database as db_mod
    monkeypatch.setattr(db_mod, "get_db_pool", _get_db_pool_stub)

    # Registry stub provides two tools; catalog should filter to the resolved one
    class _ModuleStub:
        name = "Media"
        async def get_tools(self):
            return [
                {"name": "team.only", "inputSchema": {"type": "object"}},
                {"name": "org.only", "inputSchema": {"type": "object"}},
                {"name": "global.only", "inputSchema": {"type": "object"}},
            ]

    class _RegistryStub:
        async def get_all_modules(self):
            return {"media": _ModuleStub()}

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()
    async def _allow_mod(*_args, **_kwargs):
        return True
    async def _allow_tool(*_args, **_kwargs):
        return True
    proto._has_module_permission = _allow_mod  # type: ignore
    proto._has_tool_permission = _allow_tool  # type: ignore

    # team_id present: prefer team scoped catalog
    ctx_team = RequestContext(request_id="r", user_id="1", client_id="c", metadata={"team_id": 7, "org_id": 5})
    res_team = await proto._handle_tools_list({"catalog": "A"}, ctx_team)
    names_team = {t.get("name") for t in res_team.get("tools", [])}
    assert names_team == {"team.only"}

    # no team: fall back to org scoped
    ctx_org = RequestContext(request_id="r2", user_id="1", client_id="c", metadata={"org_id": 5})
    res_org = await proto._handle_tools_list({"catalog": "A"}, ctx_org)
    names_org = {t.get("name") for t in res_org.get("tools", [])}
    assert names_org == {"org.only"}

    # neither: fall back to global
    ctx_global = RequestContext(request_id="r3", user_id="1", client_id="c", metadata={})
    res_global = await proto._handle_tools_list({"catalog": "A"}, ctx_global)
    names_global = {t.get("name") for t in res_global.get("tools", [])}
    assert names_global == {"global.only"}

    # unresolved catalog (fetchone returns None for all): fail closed with explicit metadata
    class _PoolNone:
        async def fetchone(self, *a, **k):
            return None
        async def fetchall(self, *a, **k):
            return []

    monkeypatch.setattr(db_mod, "get_db_pool", lambda: _PoolNone())
    res_unres = await proto._handle_tools_list({"catalog": "missing"}, ctx_global)
    names_unres = {t.get("name") for t in res_unres.get("tools", [])}
    assert names_unres == set()
    assert res_unres.get("_meta", {}).get("catalog", {}).get("status") == "unresolved"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_resources_list_uses_catalog_filter(monkeypatch):
    os.environ["TEST_MODE"] = "true"

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

    class _PoolStub:
        async def fetchone(self, query: str, *args):
            return {"id": 10}

        async def fetchall(self, query: str, *args):
            return [{"tool_name": "allowed.tool"}]

    async def _get_db_pool_stub():
        return _PoolStub()

    import tldw_Server_API.app.core.AuthNZ.database as db_mod
    monkeypatch.setattr(db_mod, "get_db_pool", _get_db_pool_stub)

    class _AllowedModule:
        name = "Allowed"

        async def get_tools(self):
            return [{"name": "allowed.tool"}]

        async def get_resources(self):
            return [{"uri": "allowed://one"}]

    class _BlockedModule:
        name = "Blocked"

        async def get_tools(self):
            return [{"name": "other.tool"}]

        async def get_resources(self):
            return [{"uri": "blocked://one"}]

    class _RegistryStub:
        async def get_all_modules(self):
            return {"allowed": _AllowedModule(), "blocked": _BlockedModule()}

    proto = MCPProtocol()
    proto.module_registry = _RegistryStub()

    async def _allow_module(*_args, **_kwargs):
        return True

    async def _allow_resource(*_args, **_kwargs):
        return True

    proto._has_module_permission = _allow_module  # type: ignore
    proto._has_resource_permission = _allow_resource  # type: ignore

    ctx = RequestContext(request_id="r", user_id="u", client_id="c", metadata={})
    result = await proto._handle_resources_list({"catalog": "A"}, ctx)
    uris = {res.get("uri") for res in result.get("resources", [])}
    assert "allowed://one" in uris
    assert "blocked://one" not in uris


@pytest.mark.unit
@pytest.mark.asyncio
async def test_catalog_membership_does_not_grant_rag_execute_permission() -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
    from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext

    class _ToolCatalogProvider:
        async def resolve_tool_names(
            self,
            *,
            catalog_name: str | None,
            catalog_id: Any,
            metadata: dict[str, Any],
            strict: bool,
        ) -> set[str]:
            del catalog_name, catalog_id, metadata, strict
            return {"rag.search", "rag.answer", "knowledge.search"}

    class _RagModuleStub(BaseModule):
        async def on_initialize(self) -> None:
            return None

        async def on_shutdown(self) -> None:
            return None

        async def check_health(self) -> dict[str, bool]:
            return {"ok": True}

        async def get_tools(self) -> list[dict[str, Any]]:
            return [
                create_tool_definition(
                    name="rag.search",
                    description="RAG search",
                    parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
                    metadata={"category": "search", "readOnlyHint": True},
                ),
                create_tool_definition(
                    name="rag.answer",
                    description="RAG answer",
                    parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
                    metadata={"category": "rag_generation", "readOnlyHint": True},
                ),
            ]

        def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
            del tool_name
            if "query" not in arguments:
                raise ValueError("query required")

        async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            context: RequestContext | None = None,
        ) -> dict[str, str]:
            del arguments, context
            return {"tool": tool_name}

    registry = ModuleRegistry()
    await registry.register_module("rag", _RagModuleStub, ModuleConfig(name="rag"))
    proto = MCPProtocol(dependencies=_protocol_dependencies(tool_catalog_provider=_ToolCatalogProvider()))
    proto.module_registry = registry

    async def _allow_module(_context: RequestContext, _module_id: str) -> bool:
        return True

    async def _deny_answer(_context: RequestContext, tool_name: str, **_kwargs: Any) -> bool:
        return tool_name != "rag.answer"

    proto._has_module_permission = _allow_module  # type: ignore[method-assign]
    proto._has_tool_permission = _deny_answer  # type: ignore[method-assign]

    ctx = RequestContext(request_id="rag-catalog", user_id="1", client_id="unit")
    listed = await proto._handle_tools_list({"catalog": "library-rag", "catalog_strict": True}, ctx)
    by_name = {tool["name"]: tool for tool in listed["tools"]}

    assert set(by_name) == {"rag.search", "rag.answer"}  # nosec B101
    assert by_name["rag.search"]["canExecute"] is True  # nosec B101
    assert by_name["rag.answer"]["canExecute"] is False  # nosec B101

    denied = await proto.process_request(
        MCPRequest(method="tools/call", params={"name": "rag.answer", "arguments": {"query": "q"}}, id=1),
        ctx,
    )

    assert denied.error is not None  # nosec B101
