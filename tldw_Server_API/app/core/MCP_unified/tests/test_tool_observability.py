"""Tests for shared MCP tool observability metadata helpers."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import (
    BaseModule,
    ModuleConfig,
    create_tool_definition,
)
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_execution_eval_metadata,
    build_tool_eval_metadata,
)


def test_build_tool_eval_metadata_is_stable_and_non_empty() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id="mcp.git.status.v1",
        tool_prompt_version="2026.06.04",
        task_families=["code_review"],
        expected_result_kind="structured_git_state",
        success_signals=["avoided_mutation"],
    )

    assert metadata == {
        "eval": {
            "tool_prompt_id": "mcp.git.status.v1",
            "tool_prompt_version": "2026.06.04",
            "task_families": ["code_review"],
            "expected_result_kind": "structured_git_state",
            "success_signals": ["avoided_mutation"],
            "prompt_variant": "builtin",
        }
    }


def test_build_tool_eval_metadata_strips_values_and_removes_blank_list_items() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id=" mcp.git.diff.v1 ",
        tool_prompt_version=" 2026.06.04 ",
        task_families=[" code_review ", "", "  ", " repository_research"],
        expected_result_kind=" bounded_diff ",
        success_signals=[" avoided_mutation ", "", " selected_correct_scope "],
        prompt_variant="   ",
    )

    assert metadata == {
        "eval": {
            "tool_prompt_id": "mcp.git.diff.v1",
            "tool_prompt_version": "2026.06.04",
            "task_families": ["code_review", "repository_research"],
            "expected_result_kind": "bounded_diff",
            "success_signals": ["avoided_mutation", "selected_correct_scope"],
            "prompt_variant": "builtin",
        }
    }


def test_build_tool_eval_metadata_ignores_non_list_iterable_values() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id="mcp.git.diff.v1",
        tool_prompt_version="2026.06.04",
        task_families="search",
        expected_result_kind="bounded_diff",
        success_signals=None,  # type: ignore[arg-type]
    )

    assert metadata["eval"]["task_families"] == []
    assert metadata["eval"]["success_signals"] == []


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("tool_prompt_id", {"tool_prompt_id": " "}),
        ("tool_prompt_version", {"tool_prompt_version": ""}),
        ("expected_result_kind", {"expected_result_kind": "\t"}),
    ],
)
def test_build_tool_eval_metadata_rejects_blank_required_strings(field: str, kwargs: dict[str, str]) -> None:
    values = {
        "tool_prompt_id": "mcp.git.status.v1",
        "tool_prompt_version": "2026.06.04",
        "task_families": ["code_review"],
        "expected_result_kind": "structured_git_state",
        "success_signals": ["avoided_mutation"],
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=field):
        build_tool_eval_metadata(**values)


def test_build_execution_eval_metadata_returns_only_safe_scalar_fields() -> None:
    metadata = build_execution_eval_metadata(
        tool_name=" git.status ",
        tool_prompt_id=" mcp.git.status.v1 ",
        tool_prompt_version=" 2026.06.04 ",
        action_family=" status ",
        result_kind=" structured_git_state ",
        profile_id=" code-reviewer ",
        path_filter_used=True,
        truncated=False,
        reason_code=" ok ",
        duration_ms=12.75,
    )

    assert metadata == {
        "tool_name": "git.status",
        "tool_prompt_id": "mcp.git.status.v1",
        "tool_prompt_version": "2026.06.04",
        "action_family": "status",
        "result_kind": "structured_git_state",
        "profile_id": "code-reviewer",
        "path_filter_used": True,
        "truncated": False,
        "reason_code": "ok",
        "duration_ms": 12.75,
    }
    assert all(isinstance(value, str | bool | int | float) for value in metadata.values())


def test_build_execution_eval_metadata_omits_blank_optional_fields() -> None:
    metadata = build_execution_eval_metadata(
        tool_name="git.diff",
        tool_prompt_id="mcp.git.diff.v1",
        tool_prompt_version="2026.06.04",
        action_family="diff",
        result_kind="bounded_diff",
        profile_id=" ",
        path_filter_used=None,
        truncated=None,
        reason_code="",
        duration_ms=None,
    )

    assert metadata == {
        "tool_name": "git.diff",
        "tool_prompt_id": "mcp.git.diff.v1",
        "tool_prompt_version": "2026.06.04",
        "action_family": "diff",
        "result_kind": "bounded_diff",
    }


def test_build_execution_eval_metadata_does_not_accept_arbitrary_label_dicts() -> None:
    signature = inspect.signature(build_execution_eval_metadata)

    assert "labels" not in signature.parameters
    assert "metadata" not in signature.parameters
    assert "raw_payload" not in signature.parameters
    assert "absolute_path" not in signature.parameters
    assert "raw_diff" not in signature.parameters
    assert "file_contents" not in signature.parameters
    assert "author_email" not in signature.parameters

    with pytest.raises(TypeError):
        build_execution_eval_metadata(
            tool_name="git.diff",
            tool_prompt_id="mcp.git.diff.v1",
            tool_prompt_version="2026.06.04",
            action_family="diff",
            result_kind="bounded_diff",
            labels={
                "raw_payload": "diff --git a/secret.txt b/secret.txt",
                "absolute_path": "/Users/example/work/secret.txt",
                "author_email": "author@example.com",
            },
        )


def test_create_tool_definition_adds_default_eval_metadata() -> None:
    tool = create_tool_definition(
        name="docs.search",
        description="Search indexed docs.",
        parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        metadata={"category": "search", "readOnlyHint": True},
    )

    metadata = tool["metadata"]
    eval_metadata = metadata["eval"]

    assert metadata["category"] == "search"
    assert metadata["readOnlyHint"] is True
    assert eval_metadata["tool_prompt_id"] == "mcp.docs.search.v1"
    assert eval_metadata["tool_prompt_version"]
    assert eval_metadata["task_families"] == ["search"]
    assert eval_metadata["expected_result_kind"] == "search_result"
    assert "avoided_mutation" in eval_metadata["success_signals"]
    assert eval_metadata["prompt_variant"] == "builtin"


def test_create_tool_definition_preserves_explicit_eval_metadata() -> None:
    explicit_eval = build_tool_eval_metadata(
        tool_prompt_id="mcp.custom.docs_search.v2",
        tool_prompt_version="2026.06.04-custom",
        task_families=["custom_docs"],
        expected_result_kind="custom_doc_matches",
        success_signals=["matched_custom_prompt"],
        prompt_variant="operator_patch",
    )

    tool = create_tool_definition(
        name="docs.search",
        description="Search indexed docs.",
        parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        metadata={"category": "search", **explicit_eval},
    )

    assert tool["metadata"]["eval"] == explicit_eval["eval"]


def test_create_tool_definition_sanitizes_explicit_eval_metadata() -> None:
    tool = create_tool_definition(
        name="docs.search",
        description="Search indexed docs.",
        parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        metadata={
            "category": "search",
            "eval": {
                "tool_prompt_id": "mcp.docs.search.custom.v1",
                "tool_prompt_version": "2026.06.04-custom",
                "task_families": ["search", {"raw": "do-not-leak"}],
                "expected_result_kind": "search_result",
                "success_signals": ["completed_without_error", {"raw": "secret"}],
                "prompt_variant": "operator_patch",
                "raw_payload": "diff --git a/secret.txt b/secret.txt",
            },
        },
    )

    assert tool["metadata"]["eval"] == {
        "tool_prompt_id": "mcp.docs.search.custom.v1",
        "tool_prompt_version": "2026.06.04-custom",
        "task_families": ["search"],
        "expected_result_kind": "search_result",
        "success_signals": ["completed_without_error"],
        "prompt_variant": "operator_patch",
    }


def test_create_tool_definition_merges_partial_explicit_eval_metadata() -> None:
    tool = create_tool_definition(
        name="docs.search",
        description="Search indexed docs.",
        parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        metadata={
            "category": "search",
            "eval": {
                "tool_prompt_id": "mcp.docs.search.operator.v1",
                "task_families": ["operator_docs"],
                "success_signals": ["matched_operator_prompt"],
            },
        },
    )

    assert tool["metadata"]["eval"] == {
        "tool_prompt_id": "mcp.docs.search.operator.v1",
        "tool_prompt_version": "2026.06.04",
        "task_families": ["operator_docs"],
        "expected_result_kind": "search_result",
        "success_signals": ["matched_operator_prompt"],
        "prompt_variant": "builtin",
    }


def test_create_tool_definition_rejects_non_string_required_eval_fields() -> None:
    tool = create_tool_definition(
        name="docs.search",
        description="Search indexed docs.",
        parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        metadata={
            "category": "search",
            "eval": {
                "tool_prompt_id": {"raw": "do-not-leak"},
                "tool_prompt_version": "2026.06.04-custom",
                "expected_result_kind": "custom_search_result",
            },
        },
    )

    eval_metadata = tool["metadata"]["eval"]
    assert eval_metadata["tool_prompt_id"] == "mcp.docs.search.v1"
    assert "do-not-leak" not in str(eval_metadata)
    assert eval_metadata["tool_prompt_version"] == "2026.06.04-custom"
    assert eval_metadata["expected_result_kind"] == "custom_search_result"


class _ManualObservableModule(BaseModule):
    """Test module that intentionally bypasses the shared definition helper."""

    def __init__(
        self,
        config: ModuleConfig,
        *,
        result_override: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._result_override = result_override

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "manual.inspect",
                "description": "Inspect a manual result.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                "metadata": {"category": "inspection", "readOnlyHint": True},
            }
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if self._result_override is not None:
            return dict(self._result_override)
        return {"target": arguments["target"], "ok": True}


class _ManualRegistryStub:
    def __init__(self, module: _ManualObservableModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str) -> _ManualObservableModule | None:
        return self.module if tool_name == "manual.inspect" else None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        return "manual" if tool_name == "manual.inspect" else None

    async def get_all_modules(self) -> dict[str, _ManualObservableModule]:
        return {"manual": self.module}


def _manual_observable_protocol(
    *,
    result_override: dict[str, Any] | None = None,
) -> MCPProtocol:
    protocol = MCPProtocol()
    module = _ManualObservableModule(ModuleConfig(name="manual"), result_override=result_override)
    protocol.module_registry = _ManualRegistryStub(module)  # type: ignore[assignment]

    async def _allow_module(_context: RequestContext, _module_id: str | None) -> bool:
        return True

    async def _allow_tool(
        _context: RequestContext,
        _tool_name: str,
        **_kwargs: Any,
    ) -> bool:
        return True

    protocol._has_module_permission = _allow_module  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow_tool  # type: ignore[method-assign]
    return protocol


@pytest.mark.asyncio
async def test_tools_list_adds_eval_metadata_to_manual_tool_definitions() -> None:
    protocol = _manual_observable_protocol()
    context = RequestContext(request_id="manual-list", metadata={})

    payload = await protocol._handle_tools_list({}, context)
    tool = payload["tools"][0]
    eval_metadata = tool["metadata"]["eval"]

    assert eval_metadata["tool_prompt_id"] == "mcp.manual.inspect.v1"
    assert eval_metadata["task_families"] == ["inspection"]
    assert eval_metadata["expected_result_kind"] == "inspection_result"
    assert "avoided_mutation" in eval_metadata["success_signals"]


@pytest.mark.asyncio
async def test_tools_call_adds_execution_eval_metadata_to_structured_results() -> None:
    protocol = _manual_observable_protocol()
    context = RequestContext(
        request_id="manual-call",
        user_id="user-1",
        client_id="client-1",
        metadata={"profile_id": "researcher"},
    )

    payload = await protocol._handle_tools_call(
        {"name": "manual.inspect", "arguments": {"target": "readme"}},
        context,
    )

    result_json = payload["content"][0]["json"]
    eval_metadata = result_json["eval"]

    assert result_json["target"] == "readme"
    assert eval_metadata["tool_name"] == "manual.inspect"
    assert eval_metadata["tool_prompt_id"] == "mcp.manual.inspect.v1"
    assert eval_metadata["tool_prompt_version"]
    assert eval_metadata["action_family"] == "inspection"
    assert eval_metadata["result_kind"] == "inspection_result"
    assert eval_metadata["profile_id"] == "researcher"
    assert isinstance(eval_metadata["duration_ms"], float)
    assert payload["eval"]["tool_name"] == "manual.inspect"
    assert payload["eval"]["profile_id"] == "researcher"


@pytest.mark.asyncio
async def test_tools_call_omits_unsafe_profile_id_from_eval_metadata() -> None:
    protocol = _manual_observable_protocol()
    context = RequestContext(
        request_id="manual-call",
        user_id="user-1",
        client_id="client-1",
        metadata={"profile_id": "researcher@example.com"},
    )

    payload = await protocol._handle_tools_call(
        {"name": "manual.inspect", "arguments": {"target": "readme"}},
        context,
    )

    result_eval = payload["content"][0]["json"]["eval"]
    assert "profile_id" not in result_eval
    assert "profile_id" not in payload["eval"]


@pytest.mark.asyncio
async def test_tools_call_does_not_promote_tool_returned_eval_to_top_level() -> None:
    protocol = _manual_observable_protocol(
        result_override={
            "target": "readme",
            "eval": {
                "tool_name": "malicious.override",
                "raw_payload": "diff --git a/secret.txt b/secret.txt",
                "profile_id": "operator@example.com",
            },
        }
    )
    context = RequestContext(
        request_id="manual-call",
        user_id="user-1",
        client_id="client-1",
        metadata={"profile_id": "researcher"},
    )

    payload = await protocol._handle_tools_call(
        {"name": "manual.inspect", "arguments": {"target": "readme"}},
        context,
    )

    assert payload["content"][0]["json"]["eval"]["tool_name"] == "malicious.override"
    assert payload["eval"]["tool_name"] == "manual.inspect"
    assert payload["eval"]["profile_id"] == "researcher"
    assert "raw_payload" not in payload["eval"]
