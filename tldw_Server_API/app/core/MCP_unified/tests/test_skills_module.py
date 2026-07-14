"""Contract tests for the read-only MCP Skills catalog and renderer."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import skills_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.skills_module import (
    DEFAULT_LIST_PAGE_SIZE,
    HARD_MAX_RENDERED_SKILL_CHARS,
    SkillsModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol_types import RequestContext
from tldw_Server_API.app.core.Skills.exceptions import SkillNotFoundError
from tldw_Server_API.app.core.Skills.skill_executor import SkillExecutionResult, SkillExecutor
from tldw_Server_API.app.core.Skills.skills_service import (
    SKILL_NAME_PATTERN,
    SkillMetadata,
    SkillsService,
)

pytestmark = pytest.mark.unit


@dataclass
class UserCatalog:
    user_id: int
    base_path: Path
    db_path: Path
    db: CharactersRAGDB
    service: SkillsService
    context: RequestContext


@pytest.fixture
def user_catalogs(tmp_path: Path) -> dict[int, UserCatalog]:
    """Create isolated real Skills catalogs for two authenticated users."""
    catalogs: dict[int, UserCatalog] = {}
    for user_id in (1, 2):
        base_path = tmp_path / f"user-{user_id}"
        base_path.mkdir()
        db_path = base_path / "ChaChaNotes.db"
        db = CharactersRAGDB(db_path=db_path, client_id=f"seed-skills-{user_id}")
        service = SkillsService(user_id=user_id, base_path=base_path, db=db)
        catalogs[user_id] = UserCatalog(
            user_id=user_id,
            base_path=base_path,
            db_path=db_path,
            db=db,
            service=service,
            context=RequestContext(
                request_id=f"skills-{user_id}",
                user_id=str(user_id),
                db_paths={"chacha": str(db_path)},
            ),
        )
    yield catalogs
    for catalog in catalogs.values():
        catalog.db.close_all_connections()


async def _module(settings: dict[str, Any] | None = None) -> SkillsModule:
    module = SkillsModule(ModuleConfig(name="skills", settings=settings or {}))
    await module.on_initialize()
    return module


def _metadata(name: str = "review-paper", *, context: str = "inline") -> SkillMetadata:
    return SkillMetadata(
        id="internal-id",
        name=name,
        description="Review a paper",
        argument_hint="[issue]",
        user_invocable=True,
        disable_model_invocation=False,
        allowed_tools=["rag.search"],
        model=None,
        context=context,
        directory_path="/private/catalog/path",
        content_hash="private-hash",
        version=1,
    )


def _skill_data(
    name: str = "review-paper",
    *,
    content: str = "Review $ARGUMENTS",
    context: str = "inline",
    supporting_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Review a paper",
        "argument_hint": "[issue]",
        "user_invocable": True,
        "disable_model_invocation": False,
        "allowed_tools": ["rag.search"],
        "model": None,
        "context": context,
        "content": content,
        "raw_content": content,
        "supporting_files": supporting_files or {},
        "directory_path": "/private/catalog/path",
        "version": 1,
    }


async def _seed_review_skill(
    service: SkillsService,
    *,
    name: str = "review-paper",
    context: str = "inline",
    content: str = "Review $ARGUMENTS",
    supporting_files: dict[str, str] | None = None,
) -> None:
    await service.create_skill(
        name,
        (
            "---\n"
            "description: Review a paper\n"
            'argument-hint: "[issue]"\n'
            "allowed-tools:\n"
            "  - rag.search\n"
            f"context: {context}\n"
            "---\n"
            f"{content}"
        ),
        supporting_files=supporting_files,
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


@pytest.mark.asyncio
async def test_tool_catalog_is_exact_and_read_only() -> None:
    module = await _module({"list_page_size": 17})

    tools = await module.get_tools()

    assert [tool["name"] for tool in tools] == ["skills.list", "skills.get", "skills.render"]
    assert all(tool["metadata"]["readOnlyHint"] is True for tool in tools)
    assert [tool["metadata"]["category"] for tool in tools] == ["search", "retrieval", "retrieval"]
    list_tool = tools[0]
    assert list_tool["inputSchema"]["properties"]["limit"]["default"] == 17
    assert list_tool["inputSchema"]["properties"]["q"]["maxLength"] == 200
    assert tools[2]["inputSchema"]["properties"]["arguments"]["maxLength"] == 10_000
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in tools)
    for tool, field in ((tools[1], "name"), (tools[2], "skill_name")):
        name_schema = tool["inputSchema"]["properties"][field]
        assert name_schema["maxLength"] == 64
        assert name_schema["pattern"] == SKILL_NAME_PATTERN.pattern


@pytest.mark.parametrize(
    ("settings", "expected_page_size", "expected_render_limit"),
    [
        ({}, DEFAULT_LIST_PAGE_SIZE, HARD_MAX_RENDERED_SKILL_CHARS),
        (
            {"list_page_size": True, "max_rendered_skill_chars": False},
            DEFAULT_LIST_PAGE_SIZE,
            HARD_MAX_RENDERED_SKILL_CHARS,
        ),
        (
            {"list_page_size": "25", "max_rendered_skill_chars": "900"},
            DEFAULT_LIST_PAGE_SIZE,
            HARD_MAX_RENDERED_SKILL_CHARS,
        ),
        ({"list_page_size": 0, "max_rendered_skill_chars": 0}, 1, 1),
        ({"list_page_size": 25, "max_rendered_skill_chars": 900}, 25, 900),
        (
            {"list_page_size": 101, "max_rendered_skill_chars": 100_001},
            100,
            HARD_MAX_RENDERED_SKILL_CHARS,
        ),
    ],
)
@pytest.mark.asyncio
async def test_settings_use_defaults_for_invalid_types_and_clamp_integers(
    settings: dict[str, Any],
    expected_page_size: int,
    expected_render_limit: int,
) -> None:
    module = await _module(settings)

    assert module._list_page_size == expected_page_size
    assert module._max_rendered_skill_chars == expected_render_limit
    assert isinstance(module._executor, SkillExecutor)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("skills.list", {"unexpected": 1}, "unexpected arguments"),
        ("skills.get", {"name": "valid", "unexpected": 1}, "unexpected arguments"),
        ("skills.render", {"skill_name": "valid", "extra": 1}, "unexpected arguments"),
        ("skills.list", {"limit": True}, "limit must be an integer"),
        ("skills.list", {"offset": False}, "offset must be an integer"),
        ("skills.list", {"offset": -1}, "offset must be >= 0"),
        ("skills.list", {"limit": 0}, "limit must be 1..100"),
        ("skills.list", {"limit": 101}, "limit must be 1..100"),
        ("skills.list", {"q": "q" * 201}, "q must be at most 200 characters"),
        ("skills.list", {"q": None}, "q must be a string"),
        ("skills.list", {"q": 1}, "q must be a string"),
        ("skills.get", {}, "name must be a valid skill name"),
        ("skills.get", {"name": "Invalid_Name"}, "name must be a valid skill name"),
        (
            "skills.get",
            {"name": f" {'a' * 63} "},
            "name must be a valid skill name",
        ),
        ("skills.render", {}, "skill_name must be a valid skill name"),
        ("skills.render", {"skill_name": True}, "skill_name must be a valid skill name"),
        (
            "skills.render",
            {"skill_name": f" {'a' * 63} "},
            "skill_name must be a valid skill name",
        ),
        (
            "skills.render",
            {"skill_name": "valid", "arguments": "x" * 10_001},
            "arguments must be at most 10000 characters",
        ),
        ("skills.render", {"skill_name": "valid", "arguments": 1}, "arguments must be a string"),
    ],
)
@pytest.mark.asyncio
async def test_exact_argument_validation(
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    module = await _module()

    with pytest.raises(ValueError, match=message):
        module.validate_tool_arguments(tool_name, arguments)


@pytest.mark.asyncio
async def test_execute_rejects_non_dictionary_arguments() -> None:
    module = await _module()

    with pytest.raises(TypeError, match="arguments must be an object"):
        await module.execute_tool("skills.list", [])  # type: ignore[arg-type]


@pytest.mark.parametrize("user_id", [None, "", "abc", "0", "-1", False, True])
@pytest.mark.asyncio
async def test_invalid_authenticated_user_ids_fail_closed(tmp_path: Path, user_id: Any) -> None:
    module = await _module()
    context = SimpleNamespace(
        user_id=user_id,
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )

    with pytest.raises(PermissionError, match="^skills_user_context_required$"):
        await module.execute_tool("skills.list", {}, context=context)


@pytest.mark.parametrize(
    "context",
    [
        None,
        SimpleNamespace(user_id="1", db_paths={}),
        SimpleNamespace(user_id="1", db_paths={"chacha": ""}),
        SimpleNamespace(user_id="1", db_paths={"chacha": False}),
    ],
)
@pytest.mark.asyncio
async def test_missing_database_context_fails_closed(context: Any) -> None:
    module = await _module()

    with pytest.raises(PermissionError, match="^skills_user_context_required$"):
        await module.execute_tool("skills.list", {}, context=context)


@pytest.mark.asyncio
async def test_list_uses_one_filtered_page_call_and_exact_pagination(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_one = user_catalogs[1]
    await user_one.service.create_skill("alpha", "---\ndescription: Alpha\n---\nAlpha")
    await user_one.service.create_skill("beta", "---\ndescription: Beta\n---\nBeta")
    await user_one.service.create_skill("blocked", "---\ndescription: Blocked\n---\nBlocked")
    await user_one.service.create_skill("hidden", "---\nuser-invocable: false\n---\nHidden")
    await user_one.service.create_skill(
        "manual-only",
        "---\ndisable-model-invocation: true\n---\nManual",
    )
    await user_one.service.create_skill("deleted", "Deleted")
    await user_one.service.delete_skill("deleted")
    await user_catalogs[2].service.create_skill("other-user", "Other user")

    original_allowed = SkillsService._is_skill_allowed

    def allow_except_blocked(self: SkillsService, name: str, *, purpose: str) -> bool:
        return name != "blocked" and original_allowed(self, name, purpose=purpose)

    original_page = SkillsService.list_model_visible_skills_page
    page_calls: list[tuple[str | None, int, int]] = []

    async def counted_page(
        self: SkillsService,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SkillMetadata], int]:
        page_calls.append((q, limit, offset))
        return await original_page(self, q=q, limit=limit, offset=offset)

    monkeypatch.setattr(SkillsService, "_is_skill_allowed", allow_except_blocked)
    monkeypatch.setattr(SkillsService, "list_model_visible_skills_page", counted_page)
    module = await _module()

    first = await module.execute_tool(
        "skills.list",
        {"limit": 1, "offset": 0},
        context=user_one.context,
    )
    final = await module.execute_tool(
        "skills.list",
        {"limit": 1, "offset": 1},
        context=user_one.context,
    )

    assert first["count"] == 1
    assert first["total"] == 2
    assert first["limit"] == 1
    assert first["offset"] == 0
    assert first["next_offset"] == 1
    assert [item["name"] for item in first["skills"]] == ["alpha"]
    assert final["count"] == 1
    assert final["total"] == 2
    assert final["next_offset"] is None
    assert [item["name"] for item in final["skills"]] == ["beta"]
    assert page_calls == [(None, 1, 0), (None, 1, 1)]


@pytest.mark.asyncio
async def test_list_preserves_query_and_normalizes_only_whitespace_at_delegation(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated_queries: list[str | None] = []

    async def capture_page(
        _self: SkillsService,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SkillMetadata], int]:
        delegated_queries.append(q)
        assert limit == 50
        assert offset == 0
        return [], 0

    monkeypatch.setattr(SkillsService, "list_model_visible_skills_page", capture_page)
    module = await _module()

    await module.execute_tool(
        "skills.list",
        {"q": "flags --all /* literal */"},
        context=user_catalogs[1].context,
    )
    await module.execute_tool("skills.list", {"q": " \n\t "}, context=user_catalogs[1].context)

    assert delegated_queries == ["flags --all /* literal */", None]


@pytest.mark.asyncio
async def test_get_matches_list_metadata_and_excludes_private_fields(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    original_get = SkillsService.get_model_visible_skill_metadata
    calls = 0

    async def counted_get(self: SkillsService, name: str) -> SkillMetadata:
        nonlocal calls
        calls += 1
        return await original_get(self, name)

    monkeypatch.setattr(SkillsService, "get_model_visible_skill_metadata", counted_get)
    module = await _module()

    listed = await module.execute_tool("skills.list", {}, context=user_catalogs[1].context)
    fetched = await module.execute_tool(
        "skills.get",
        {"name": "review-paper"},
        context=user_catalogs[1].context,
    )

    assert fetched == listed["skills"][0]
    assert set(fetched) == {
        "name",
        "description",
        "argument_hint",
        "user_invocable",
        "disable_model_invocation",
        "declared_tools",
        "model",
        "context",
        "runtime",
        "version",
    }
    assert calls == 1
    assert not (
        _all_keys(fetched)
        & {
            "id",
            "created_at",
            "last_modified",
            "content",
            "raw_content",
            "supporting_files",
            "directory_path",
            "path",
            "content_hash",
            "hash",
        }
    )


@pytest.mark.asyncio
async def test_hidden_model_disabled_deleted_blocked_and_other_user_skills_are_not_found(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_one = user_catalogs[1]
    await user_one.service.create_skill("hidden", "---\nuser-invocable: false\n---\nHidden")
    await user_one.service.create_skill(
        "manual-only",
        "---\ndisable-model-invocation: true\n---\nManual",
    )
    await user_one.service.create_skill("deleted", "Deleted")
    await user_one.service.delete_skill("deleted")
    await user_one.service.create_skill("blocked", "Blocked")
    await user_catalogs[2].service.create_skill("other-user", "Other")
    original_allowed = SkillsService._is_skill_allowed

    def allow_except_blocked(self: SkillsService, name: str, *, purpose: str) -> bool:
        return name != "blocked" and original_allowed(self, name, purpose=purpose)

    monkeypatch.setattr(SkillsService, "_is_skill_allowed", allow_except_blocked)
    module = await _module()

    listed = await module.execute_tool("skills.list", {}, context=user_one.context)
    assert listed["skills"] == []

    for name in ("hidden", "manual-only", "deleted", "blocked", "other-user", "unknown"):
        with pytest.raises(ValueError, match="^skill_not_found$"):
            await module.execute_tool("skills.get", {"name": name}, context=user_one.context)
        with pytest.raises(ValueError, match="^skill_not_found$"):
            await module.execute_tool(
                "skills.render",
                {"skill_name": name},
                context=user_one.context,
            )


@pytest.mark.asyncio
async def test_render_is_forced_dry_and_preserves_arguments(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    module = await _module()
    execute = AsyncMock(
        return_value=SkillExecutionResult(
            skill_name="review-paper",
            rendered_prompt="Review --formal /* literal */\nnext",
            allowed_tools=["rag.search"],
            model_override=None,
            execution_mode="inline",
            dry_run=True,
        )
    )
    monkeypatch.setattr(module._executor, "execute", execute)

    result = await module.execute_tool(
        "skills.render",
        {"skill_name": "review-paper", "arguments": "--formal /* literal */\nnext"},
        context=user_catalogs[1].context,
    )

    assert result == {
        "skill_name": "review-paper",
        "rendered_prompt": "Review --formal /* literal */\nnext",
        "declared_tools": ["rag.search"],
        "model_override": None,
        "execution_mode": "inline",
        "supporting_files_omitted": False,
        "dry_run": True,
        "version": 1,
    }
    execute.assert_awaited_once()
    call = execute.await_args
    assert call.args[1] == "--formal /* literal */\nnext"
    assert call.kwargs == {"context": None, "dry_run": True}


@pytest.mark.asyncio
async def test_render_calls_visibility_gate_and_verified_load_once(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    original_metadata = SkillsService.get_model_visible_skill_metadata
    original_skill = SkillsService.get_skill
    metadata_calls = 0
    skill_calls = 0

    async def counted_metadata(self: SkillsService, name: str) -> SkillMetadata:
        nonlocal metadata_calls
        metadata_calls += 1
        return await original_metadata(self, name)

    async def counted_skill(
        self: SkillsService,
        name: str,
        *,
        enforce_integrity: bool = True,
    ) -> dict[str, Any]:
        nonlocal skill_calls
        skill_calls += 1
        return await original_skill(self, name, enforce_integrity=enforce_integrity)

    monkeypatch.setattr(SkillsService, "get_model_visible_skill_metadata", counted_metadata)
    monkeypatch.setattr(SkillsService, "get_skill", counted_skill)
    module = await _module()

    await module.execute_tool(
        "skills.render",
        {"skill_name": "review-paper", "arguments": "issue 42"},
        context=user_catalogs[1].context,
    )

    assert metadata_calls == 1
    assert skill_calls == 1


@pytest.mark.asyncio
async def test_render_applies_schema_defaults_to_null_visibility_flags(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    original_get_skill = SkillsService.get_skill

    async def legacy_get_skill(
        self: SkillsService,
        name: str,
        *,
        enforce_integrity: bool = True,
    ) -> dict[str, Any]:
        skill = await original_get_skill(self, name, enforce_integrity=enforce_integrity)
        skill["user_invocable"] = None
        skill["disable_model_invocation"] = None
        return skill

    monkeypatch.setattr(SkillsService, "get_skill", legacy_get_skill)
    module = await _module()

    result = await module.execute_tool(
        "skills.render",
        {"skill_name": "review-paper", "arguments": "issue 42"},
        context=user_catalogs[1].context,
    )

    assert result["rendered_prompt"] == "Review issue 42"


@pytest.mark.asyncio
async def test_supporting_files_are_disclosed_only_as_boolean(
    user_catalogs: dict[int, UserCatalog],
) -> None:
    await _seed_review_skill(
        user_catalogs[1].service,
        supporting_files={"private-reference.md": "private supporting content"},
    )
    module = await _module()

    result = await module.execute_tool(
        "skills.render",
        {"skill_name": "review-paper", "arguments": "issue 42"},
        context=user_catalogs[1].context,
    )

    assert result["supporting_files_omitted"] is True
    assert isinstance(result["supporting_files_omitted"], bool)
    assert "private-reference.md" not in str(result)
    assert "private supporting content" not in str(result)


@pytest.mark.asyncio
async def test_inline_and_fork_render_never_enter_execution_paths(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service, name="inline-skill")
    await _seed_review_skill(user_catalogs[1].service, name="fork-skill", context="fork")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("dry render must not execute inline or fork paths")

    monkeypatch.setattr(SkillExecutor, "_execute_inline", forbidden)
    monkeypatch.setattr(SkillExecutor, "_execute_forked", forbidden)
    module = await _module()

    inline = await module.execute_tool(
        "skills.render",
        {"skill_name": "inline-skill"},
        context=user_catalogs[1].context,
    )
    fork = await module.execute_tool(
        "skills.render",
        {"skill_name": "fork-skill"},
        context=user_catalogs[1].context,
    )

    assert inline["execution_mode"] == "inline"
    assert fork["execution_mode"] == "fork"
    assert inline["dry_run"] is True
    assert fork["dry_run"] is True


@pytest.mark.asyncio
async def test_oversized_render_is_rejected_atomically(user_catalogs: dict[int, UserCatalog]) -> None:
    await _seed_review_skill(user_catalogs[1].service, content="12345678901")
    module = await _module({"max_rendered_skill_chars": 10})

    with pytest.raises(ValueError) as exc_info:
        await module.execute_tool(
            "skills.render",
            {"skill_name": "review-paper"},
            context=user_catalogs[1].context,
        )

    assert str(exc_info.value) == "rendered_skill_too_large: limit=10"
    assert "12345678901" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("user_invocable", False), ("disable_model_invocation", True)],
)
@pytest.mark.asyncio
async def test_render_rechecks_visibility_after_verified_load(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    original_get_skill = SkillsService.get_skill

    async def raced_get_skill(
        self: SkillsService,
        name: str,
        *,
        enforce_integrity: bool = True,
    ) -> dict[str, Any]:
        skill = await original_get_skill(self, name, enforce_integrity=enforce_integrity)
        skill[field] = value
        return skill

    monkeypatch.setattr(SkillsService, "get_skill", raced_get_skill)
    module = await _module()
    monkeypatch.setattr(module._executor, "execute", AsyncMock())

    with pytest.raises(ValueError, match="^skill_not_found$"):
        await module.execute_tool(
            "skills.render",
            {"skill_name": "review-paper"},
            context=user_catalogs[1].context,
        )
    module._executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_integrity_race_is_bounded(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)

    async def blocked_load(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ContextIntegrityBlocked("skill:user:1/review-paper", "quarantined")

    monkeypatch.setattr(SkillsService, "get_skill", blocked_load)
    module = await _module()

    with pytest.raises(PermissionError, match="^context_integrity_blocked$"):
        await module.execute_tool(
            "skills.render",
            {"skill_name": "review-paper"},
            context=user_catalogs[1].context,
        )


@pytest.mark.asyncio
async def test_discovery_integrity_exception_is_hidden_as_not_found(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_metadata(*_args: Any, **_kwargs: Any) -> SkillMetadata:
        raise ContextIntegrityBlocked("skill:user:1/review-paper", "quarantined")

    monkeypatch.setattr(SkillsService, "get_model_visible_skill_metadata", blocked_metadata)
    module = await _module()

    with pytest.raises(ValueError, match="^skill_not_found$"):
        await module.execute_tool(
            "skills.get",
            {"name": "review-paper"},
            context=user_catalogs[1].context,
        )
    with pytest.raises(ValueError, match="^skill_not_found$"):
        await module.execute_tool(
            "skills.render",
            {"skill_name": "review-paper"},
            context=user_catalogs[1].context,
        )


class TrackingDB:
    instances: list[TrackingDB] = []

    def __init__(self, db_path: str | Path, client_id: str) -> None:
        self.db_path = db_path
        self.client_id = client_id
        self.closed = False
        self.closed_after_worker = False
        type(self).instances.append(self)

    def close_all_connections(self) -> None:
        self.closed = True


class ScenarioService:
    scenario = "list"

    def __init__(self, user_id: int, base_path: Path, db: TrackingDB) -> None:
        self.user_id = user_id
        self.base_path = base_path
        self.db = db

    async def list_model_visible_skills_page(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SkillMetadata], int]:
        del q, limit, offset
        if self.scenario == "storage":
            raise OSError("SENTINEL_CONTENT /sentinel/private/path")
        return [_metadata()], 1

    async def get_model_visible_skill_metadata(self, name: str) -> SkillMetadata:
        if self.scenario == "not_found":
            raise SkillNotFoundError(name)
        return _metadata(name)

    async def get_skill(self, name: str, *, enforce_integrity: bool = True) -> dict[str, Any]:
        assert enforce_integrity is True
        if self.scenario == "integrity":
            raise ContextIntegrityBlocked(f"skill:user:1/{name}", "quarantined")
        if self.scenario == "oversized":
            return _skill_data(name, content="x" * 11)
        return _skill_data(name)


@pytest.mark.parametrize(
    ("scenario", "tool_name", "arguments", "settings", "error_type", "message"),
    [
        ("list", "skills.list", {}, {}, None, None),
        ("get", "skills.get", {"name": "review-paper"}, {}, None, None),
        ("render", "skills.render", {"skill_name": "review-paper"}, {}, None, None),
        (
            "not_found",
            "skills.get",
            {"name": "missing"},
            {},
            ValueError,
            "skill_not_found",
        ),
        (
            "integrity",
            "skills.render",
            {"skill_name": "review-paper"},
            {},
            PermissionError,
            "context_integrity_blocked",
        ),
        (
            "oversized",
            "skills.render",
            {"skill_name": "review-paper"},
            {"max_rendered_skill_chars": 10},
            ValueError,
            "rendered_skill_too_large: limit=10",
        ),
        ("storage", "skills.list", {}, {}, RuntimeError, "skills_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_database_closes_on_every_opened_result_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario: str,
    tool_name: str,
    arguments: dict[str, Any],
    settings: dict[str, Any],
    error_type: type[Exception] | None,
    message: str | None,
) -> None:
    TrackingDB.instances = []
    ScenarioService.scenario = scenario
    monkeypatch.setattr(skills_module, "CharactersRAGDB", TrackingDB)
    monkeypatch.setattr(skills_module, "SkillsService", ScenarioService)
    module = await _module(settings)
    context = RequestContext(
        request_id="tracking",
        user_id="1",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )

    if error_type is None:
        await module.execute_tool(tool_name, arguments, context=context)
    else:
        with pytest.raises(error_type, match=f"^{message}$"):
            await module.execute_tool(tool_name, arguments, context=context)

    assert len(TrackingDB.instances) == 1
    assert TrackingDB.instances[0].closed is True


@pytest.mark.asyncio
async def test_service_construction_failure_closes_partial_database_in_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    TrackingDB.instances = []
    constructor_thread: list[int] = []
    close_thread: list[int] = []

    class FailingService:
        def __init__(self, user_id: int, base_path: Path, db: TrackingDB) -> None:
            del user_id, base_path, db
            constructor_thread.append(threading.get_ident())
            raise RuntimeError("private constructor failure")

    original_close = TrackingDB.close_all_connections

    def record_close(self: TrackingDB) -> None:
        close_thread.append(threading.get_ident())
        original_close(self)

    monkeypatch.setattr(TrackingDB, "close_all_connections", record_close)
    monkeypatch.setattr(skills_module, "CharactersRAGDB", TrackingDB)
    monkeypatch.setattr(skills_module, "SkillsService", FailingService)
    module = await _module()
    event_loop_thread = threading.get_ident()
    context = RequestContext(
        request_id="constructor-failure",
        user_id="1",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )

    with pytest.raises(RuntimeError, match="^skills_unavailable$"):
        await module.execute_tool("skills.list", {}, context=context)

    assert TrackingDB.instances[0].closed is True
    assert constructor_thread == close_thread
    assert constructor_thread != [event_loop_thread]


@pytest.mark.asyncio
async def test_storage_failure_log_excludes_exception_message_content_and_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    TrackingDB.instances = []
    ScenarioService.scenario = "storage"
    monkeypatch.setattr(skills_module, "CharactersRAGDB", TrackingDB)
    monkeypatch.setattr(skills_module, "SkillsService", ScenarioService)
    module = await _module()
    context = RequestContext(
        request_id="storage-log",
        user_id="7",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )
    messages: list[str] = []
    sink_id = logger.add(
        messages.append,
        format="{message}",
        filter=lambda record: record["message"].startswith("skills operation="),
    )

    try:
        with pytest.raises(RuntimeError, match="^skills_unavailable$"):
            await module.execute_tool("skills.list", {}, context=context)
    finally:
        logger.remove(sink_id)

    log_text = "".join(messages)
    assert "skills operation=skills.list user_id=7 exception=OSError" in log_text
    assert "SENTINEL_CONTENT" not in log_text
    assert "/sentinel/private/path" not in log_text
    assert str(tmp_path) not in log_text


@pytest.mark.parametrize("blocked_phase", ["construction", "service", "closure"])
@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_lifecycle_and_database_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    blocked_phase: str,
) -> None:
    TrackingDB.instances = []
    phase_started = threading.Event()
    release_phase = threading.Event()
    phase_finished = threading.Event()
    operation_calls = 0

    def block_phase() -> None:
        phase_started.set()
        release_phase.wait(timeout=5)
        phase_finished.set()

    class BlockingDB(TrackingDB):
        def __init__(self, db_path: str | Path, client_id: str) -> None:
            super().__init__(db_path, client_id)
            if blocked_phase == "construction":
                block_phase()

        def close_all_connections(self) -> None:
            if blocked_phase == "closure":
                block_phase()
            super().close_all_connections()

    class BlockingService(ScenarioService):
        async def list_model_visible_skills_page(
            self,
            *,
            q: str | None = None,
            limit: int = 50,
            offset: int = 0,
        ) -> tuple[list[SkillMetadata], int]:
            nonlocal operation_calls
            del q, limit, offset
            operation_calls += 1
            if blocked_phase == "service":
                await asyncio.to_thread(block_phase)
            return [], 0

    monkeypatch.setattr(skills_module, "CharactersRAGDB", BlockingDB)
    monkeypatch.setattr(skills_module, "SkillsService", BlockingService)
    module = await _module()
    context = RequestContext(
        request_id="cancel",
        user_id="1",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )
    task = asyncio.create_task(module.execute_tool("skills.list", {}, context=context))
    assert await asyncio.to_thread(phase_started.wait, 2)

    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.sleep(0.02)
    completed_before_release = task.done()
    closed_before_release = TrackingDB.instances[0].closed

    release_phase.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed_before_release is False
    assert closed_before_release is False
    assert phase_finished.is_set()
    assert TrackingDB.instances[0].closed is True
    if blocked_phase == "construction":
        assert operation_calls == 0


@pytest.mark.asyncio
async def test_successful_operation_close_failure_is_logged_and_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CloseFailingDB(TrackingDB):
        def close_all_connections(self) -> None:
            super().close_all_connections()
            raise OSError("SENTINEL_CLOSE /sentinel/close/path")

    TrackingDB.instances = []
    ScenarioService.scenario = "list"
    monkeypatch.setattr(skills_module, "CharactersRAGDB", CloseFailingDB)
    monkeypatch.setattr(skills_module, "SkillsService", ScenarioService)
    module = await _module()
    context = RequestContext(
        request_id="close-failure",
        user_id="7",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )
    messages: list[str] = []
    sink_id = logger.add(
        messages.append,
        format="{message}",
        filter=lambda record: record["message"].startswith("skills operation="),
    )

    try:
        with pytest.raises(RuntimeError, match="^skills_unavailable$"):
            await module.execute_tool("skills.list", {}, context=context)
    finally:
        logger.remove(sink_id)

    log_text = "".join(messages)
    assert "skills operation=skills.list user_id=7 exception=OSError" in log_text
    assert "SENTINEL_CLOSE" not in log_text
    assert "/sentinel/close/path" not in log_text
    assert str(tmp_path) not in log_text


@pytest.mark.asyncio
async def test_bounded_operation_error_takes_precedence_over_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CloseFailingDB(TrackingDB):
        def close_all_connections(self) -> None:
            super().close_all_connections()
            raise OSError("private close failure")

    TrackingDB.instances = []
    ScenarioService.scenario = "not_found"
    monkeypatch.setattr(skills_module, "CharactersRAGDB", CloseFailingDB)
    monkeypatch.setattr(skills_module, "SkillsService", ScenarioService)
    module = await _module()
    context = RequestContext(
        request_id="bounded-precedence",
        user_id="1",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )

    with pytest.raises(ValueError, match="^skill_not_found$"):
        await module.execute_tool("skills.get", {"name": "missing"}, context=context)


@pytest.mark.asyncio
async def test_cancellation_preserves_cancelled_error_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingCloseFailingDB(TrackingDB):
        def close_all_connections(self) -> None:
            close_started.set()
            release_close.wait(timeout=5)
            super().close_all_connections()
            raise OSError("SENTINEL_CLOSE /sentinel/close/path")

    TrackingDB.instances = []
    ScenarioService.scenario = "list"
    monkeypatch.setattr(skills_module, "CharactersRAGDB", BlockingCloseFailingDB)
    monkeypatch.setattr(skills_module, "SkillsService", ScenarioService)
    module = await _module()
    context = RequestContext(
        request_id="cancel-close-failure",
        user_id="9",
        db_paths={"chacha": str(tmp_path / "ChaChaNotes.db")},
    )
    messages: list[str] = []
    sink_id = logger.add(
        messages.append,
        format="{message}",
        filter=lambda record: record["message"].startswith("skills operation="),
    )
    task = asyncio.create_task(module.execute_tool("skills.list", {}, context=context))
    assert await asyncio.to_thread(close_started.wait, 2)

    try:
        task.cancel()
        await asyncio.sleep(0.02)
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_close.set()
        logger.remove(sink_id)

    log_text = "".join(messages)
    assert "skills operation=skills.list user_id=9 exception=OSError" in log_text
    assert "SENTINEL_CLOSE" not in log_text
    assert "/sentinel/close/path" not in log_text
