from __future__ import annotations

from typing import Any

import pytest

from mcp_unified.interfaces.path_scope import PathScopeCandidate, normalize_path_scope_candidate
from tldw_Server_API.app.core.MCP_unified.modules.base import (
    BaseModule,
    ModuleConfig,
    create_tool_definition,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext


_PATCH_TEXT = """--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-old
+new
"""


class _AllowAllRbac:
    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _RegistryStub:
    def __init__(self, module: BaseModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str) -> BaseModule | None:
        return self.module if tool_name == "fs.patch" else None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        return "filesystem" if tool_name == "fs.patch" else None


class _RecordingPathEnforcer:
    def __init__(self) -> None:
        self.received_candidates: list[PathScopeCandidate] | None = None

    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: RequestContext,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        path_scope_candidates: list[PathScopeCandidate] | None = None,
    ) -> dict[str, Any]:
        del effective_policy, context, tool_name, tool_args, tool_def
        self.received_candidates = path_scope_candidates
        return {
            "enabled": True,
            "within_scope": True,
            "reason": None,
            "force_approval": False,
            "normalized_paths": [candidate.path for candidate in path_scope_candidates or []],
            "scope_payload": {"path_decisions": []},
        }


class _OldShapePathEnforcer:
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: RequestContext,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del effective_policy, context, tool_name, tool_args, tool_def
        return {
            "enabled": True,
            "within_scope": True,
            "reason": None,
            "force_approval": False,
            "normalized_paths": [],
            "scope_payload": None,
        }


class _PatchModule(BaseModule):
    def __init__(
        self,
        *,
        candidates: list[PathScopeCandidate] | None = None,
        include_candidate_hook: bool = True,
    ) -> None:
        super().__init__(ModuleConfig(name="filesystem"))
        self._candidates = candidates
        self._include_candidate_hook = include_candidate_hook

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ready": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="fs.patch",
                description="Patch files.",
                parameters={
                    "properties": {"diff": {"type": "string"}},
                    "required": ["diff"],
                },
                metadata={
                    "category": "management",
                    "write_capable": True,
                    "path_scope_candidate_source": "module",
                },
            )
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "fs.patch" and not str(arguments.get("diff") or ""):
            raise ValueError("diff is required")

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        del _depth
        return input_data

    async def extract_path_scope_candidates(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> list[PathScopeCandidate]:
        del tool_name, arguments, context
        if not self._include_candidate_hook:
            return await super().extract_path_scope_candidates("fs.patch", {}, None)
        return list(self._candidates or [])

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        del tool_name, arguments, context
        return {"ok": True}


def _build_protocol(
    module: BaseModule,
    path_scope_enforcer: Any,
    *,
    effective_policy_enabled: bool = True,
    path_scope_mode: str = "workspace_root",
) -> MCPProtocol:
    protocol = MCPProtocol()
    protocol.module_registry = _RegistryStub(module)  # type: ignore[assignment]
    protocol.rbac_policy = _AllowAllRbac()
    protocol.dependencies.path_scope_enforcer = path_scope_enforcer

    async def _effective_policy(_context: RequestContext) -> dict[str, Any]:
        return {
            "enabled": effective_policy_enabled,
            "allowed_tools": ["fs.patch"],
            "denied_tools": [],
            "capabilities": [],
            "policy_document": {"path_scope_mode": path_scope_mode},
            "sources": [],
        }

    async def _approval_allow(**_kwargs: Any) -> dict[str, Any]:
        return {"status": "allow", "reason": "test"}

    async def _governance_noop(**_kwargs: Any) -> None:
        return None

    protocol._resolve_effective_tool_policy = _effective_policy  # type: ignore[method-assign]
    protocol._evaluate_runtime_approval = _approval_allow  # type: ignore[method-assign]
    protocol._run_governance_preflight = _governance_noop  # type: ignore[method-assign]
    return protocol


def _context() -> RequestContext:
    return RequestContext(
        request_id="candidate-test",
        user_id="1",
        client_id="pytest",
        metadata={"mcp_policy_context_enabled": True},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_passes_module_path_candidates_to_enforcer() -> None:
    candidates = [PathScopeCandidate(path="src/app.py", action="edit", source="module")]
    module = _PatchModule(candidates=candidates)
    enforcer = _RecordingPathEnforcer()
    protocol = _build_protocol(module, enforcer)

    prepared = await protocol.prepare_tool_call(
        params={"name": "fs.patch", "arguments": {"diff": _PATCH_TEXT}},
        context=_context(),
    )

    assert prepared.tool_name == "fs.patch"
    assert enforcer.received_candidates == candidates


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_fails_closed_when_module_candidates_unavailable() -> None:
    module = _PatchModule(include_candidate_hook=False)
    protocol = _build_protocol(module, _RecordingPathEnforcer())

    with pytest.raises(PermissionError, match="path_scope_candidates_unavailable"):
        await protocol.prepare_tool_call(
            params={"name": "fs.patch", "arguments": {"diff": _PATCH_TEXT}},
            context=_context(),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_fails_closed_when_enforcer_cannot_accept_required_candidates() -> None:
    module = _PatchModule(candidates=[PathScopeCandidate(path="src/app.py", action="edit", source="module")])
    protocol = _build_protocol(module, _OldShapePathEnforcer())

    with pytest.raises(PermissionError, match="path_scope_candidates_unsupported"):
        await protocol.prepare_tool_call(
            params={"name": "fs.patch", "arguments": {"diff": _PATCH_TEXT}},
            context=_context(),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_protocol_uses_real_filesystem_patch_candidates() -> None:
    module = FilesystemModule(ModuleConfig(name="filesystem"))
    enforcer = _RecordingPathEnforcer()
    protocol = _build_protocol(module, enforcer)

    await protocol.prepare_tool_call(
        params={"name": "fs.patch", "arguments": {"diff": _PATCH_TEXT}},
        context=_context(),
    )

    assert enforcer.received_candidates == [  # nosec B101
        PathScopeCandidate(
            path="src/app.py",
            action="edit",
            source="filesystem_diff",
            requires_existing_file=True,
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effective_policy_enabled", "path_scope_mode"),
    [
        (False, "workspace_root"),
        (True, "none"),
    ],
)
async def test_protocol_skips_candidate_extraction_when_path_scope_inactive(
    effective_policy_enabled: bool,
    path_scope_mode: str,
) -> None:
    module = _PatchModule(include_candidate_hook=False)
    protocol = _build_protocol(
        module,
        _RecordingPathEnforcer(),
        effective_policy_enabled=effective_policy_enabled,
        path_scope_mode=path_scope_mode,
    )

    prepared = await protocol.prepare_tool_call(
        params={"name": "fs.patch", "arguments": {"diff": _PATCH_TEXT}},
        context=_context(),
    )

    assert prepared.tool_name == "fs.patch"


@pytest.mark.unit
def test_path_scope_candidate_string_false_flags_remain_false() -> None:
    candidate = normalize_path_scope_candidate(
        {
            "path": "src/app.py",
            "action": "edit",
            "source": "test",
            "requires_existing_file": "false",
            "creates_file": "0",
        }
    )

    assert candidate.requires_existing_file is False
    assert candidate.creates_file is False


@pytest.mark.unit
def test_path_scope_candidate_rejects_non_boolean_flags() -> None:
    with pytest.raises(ValueError, match="requires_existing_file must be a boolean"):
        normalize_path_scope_candidate(
            {
                "path": "src/app.py",
                "action": "edit",
                "requires_existing_file": "sometimes",
            }
        )
