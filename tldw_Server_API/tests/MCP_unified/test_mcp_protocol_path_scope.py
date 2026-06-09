from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeModule:
    name = "files"

    def __init__(self, tool_def: dict) -> None:
        self._tool_def = dict(tool_def)
        self.execute_calls: list[dict] = []

    async def get_tools(self) -> list[dict]:
        return [dict(self._tool_def)]

    async def get_tool_def(self, tool_name: str) -> dict | None:
        if tool_name == self._tool_def.get("name"):
            return dict(self._tool_def)
        return None

    def is_write_tool_def(self, tool_def: dict) -> bool:
        return False

    def sanitize_input(self, input_data):  # noqa: ANN001
        return input_data

    def validate_tool_arguments(self, tool_name: str, arguments: dict) -> None:  # noqa: ARG002
        return None

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):  # noqa: ANN001, ARG002
        self.execute_calls.append({"tool_name": tool_name, "arguments": dict(arguments or {})})
        return {"ok": True}

    async def execute_with_circuit_breaker(
        self,
        func,  # noqa: ANN001
        tool_name: str,
        arguments: dict,
        context=None,  # noqa: ANN001
    ):
        return await func(tool_name, arguments, context=context)


class _FakeRegistry:
    def __init__(self, module: _FakeModule) -> None:
        self.module = module

    async def find_module_for_tool(self, tool_name: str):  # noqa: ANN001
        if tool_name == self.module._tool_def.get("name"):
            return self.module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        if tool_name == self.module._tool_def.get("name"):
            return self.module.name
        return None


class _FakePathEnforcementService:
    def __init__(self, result: dict) -> None:
        self.result = dict(result)
        self.calls: list[dict] = []

    async def evaluate_tool_call(self, **kwargs) -> dict:  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return dict(self.result)


class _FakeApprovalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def evaluate_tool_call(self, **kwargs) -> dict:  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return {
            "status": "approval_required",
            "approval": {
                "approval_policy_id": 1,
                "tool_name": kwargs["tool_name"],
                "reason": kwargs["approval_reason"],
                "scope_context": dict(kwargs["scope_payload"] or {}),
                "duration_options": ["once", "session"],
                "arguments_summary": {"path": "../README.md"},
            },
        }


class _NoopApprovalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def evaluate_tool_call(self, **kwargs) -> dict:  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return {"status": "approval_required", "approval": {"tool_name": kwargs["tool_name"]}}


class _FakeWorkspaceRootResolver:
    def __init__(self, result: dict) -> None:
        self.result = dict(result)
        self.calls: list[dict] = []

    async def resolve_for_context(self, **kwargs) -> dict:  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return dict(self.result)


def _filesystem_tool_def(name: str, action: str) -> dict:
    return {
        "name": name,
        "description": name,
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "path_scope_action": action,
        },
    }


def _effective_path_policy(path_grants: list[dict]) -> dict:
    return {
        "enabled": True,
        "allowed_tools": ["fs.read", "fs.edit", "fs.write", "fs.patch"],
        "policy_document": {
            "path_scope_mode": "workspace_root",
            "path_grants": path_grants,
        },
    }


@pytest.mark.asyncio
async def test_handle_tools_call_raises_approval_for_path_scope_violation(monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_path_service = _FakePathEnforcementService(
        {
            "enabled": True,
            "within_scope": False,
            "reason": "path_outside_current_folder_scope",
            "force_approval": True,
            "normalized_paths": ["/tmp/project/README.md"],
            "scope_payload": {
                "path_scope_mode": "cwd_descendants",
                "workspace_root": "/tmp/project",
                "scope_root": "/tmp/project/src",
                "normalized_paths": ["/tmp/project/README.md"],
                "reason": "path_outside_current_folder_scope",
            },
        }
    )
    fake_approval_service = _FakeApprovalService()

    async def _fake_get_path_service():
        return fake_path_service

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "cwd_descendants",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-path-scope",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"persona_id": "researcher", "cwd": "src"},
    )

    with pytest.raises(ApprovalRequiredError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "../README.md"}},
            context,
        )

    approval = exc.value.approval or {}
    assert approval["reason"] == "path_outside_current_folder_scope"
    assert approval["scope_context"]["path_scope_mode"] == "cwd_descendants"
    assert approval["scope_context"]["normalized_paths"] == ["/tmp/project/README.md"]
    assert fake_path_service.calls[0]["tool_name"] == "files.read"
    assert fake_approval_service.calls[0]["within_effective_policy"] is False
    assert fake_approval_service.calls[0]["force_approval"] is True
    assert fake_approval_service.calls[0]["approval_reason"] == "path_outside_current_folder_scope"


@pytest.mark.asyncio
async def test_handle_tools_call_raises_approval_for_path_allowlist_violation(monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_path_service = _FakePathEnforcementService(
        {
            "enabled": True,
            "within_scope": False,
            "reason": "path_outside_allowlist_scope",
            "force_approval": True,
            "normalized_paths": ["/tmp/project/src2/README.md"],
            "scope_payload": {
                "path_scope_mode": "workspace_root",
                "workspace_root": "/tmp/project",
                "scope_root": "/tmp/project",
                "normalized_paths": ["/tmp/project/src2/README.md"],
                "path_allowlist_prefixes": ["src"],
                "reason": "path_outside_allowlist_scope",
            },
        }
    )
    fake_approval_service = _FakeApprovalService()

    async def _fake_get_path_service():
        return fake_path_service

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "path_allowlist_prefixes": ["src"],
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-path-allowlist",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"persona_id": "researcher"},
    )

    with pytest.raises(ApprovalRequiredError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "src2/README.md"}},
            context,
        )

    approval = exc.value.approval or {}
    assert approval["reason"] == "path_outside_allowlist_scope"
    assert approval["scope_context"]["path_allowlist_prefixes"] == ["src"]
    assert fake_approval_service.calls[0]["approval_reason"] == "path_outside_allowlist_scope"


@pytest.mark.asyncio
async def test_handle_tools_call_requires_approval_for_fs_write_text_out_of_scope_before_execution(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod

    tool_def = {
        "name": "fs.write_text",
        "description": "Write a text file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "management",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "capabilities": ["filesystem.write"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_path_service = _FakePathEnforcementService(
        {
            "enabled": True,
            "within_scope": False,
            "reason": "path_outside_current_folder_scope",
            "force_approval": True,
            "normalized_paths": ["/tmp/project/README.md"],
            "scope_payload": {
                "path_scope_mode": "cwd_descendants",
                "workspace_root": "/tmp/project",
                "scope_root": "/tmp/project/src",
                "normalized_paths": ["/tmp/project/README.md"],
                "reason": "path_outside_current_folder_scope",
            },
        }
    )
    fake_approval_service = _FakeApprovalService()

    async def _fake_get_path_service():
        return fake_path_service

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["fs.write_text"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "cwd_descendants",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-fs-write-path-scope",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"persona_id": "researcher", "cwd": "src"},
    )

    with pytest.raises(ApprovalRequiredError):
        await protocol._handle_tools_call(
            {"name": "fs.write_text", "arguments": {"path": "../README.md", "content": "blocked"}},
            context,
        )

    assert fake_path_service.calls[0]["tool_name"] == "fs.write_text"
    assert fake_module.execute_calls == []


@pytest.mark.asyncio
async def test_handle_tools_call_requires_approval_for_run_chain_preflight_path_scope(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
        RunCommandModule,
    )
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod

    fs_write_tool = {
        "name": "fs.write_text",
        "description": "Write text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "management",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "capabilities": ["filesystem.write"],
        },
    }
    fs_read_tool = {
        "name": "fs.read_text",
        "description": "Read text",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "capabilities": ["filesystem.read"],
        },
    }

    class _FilesystemModule:
        name = "filesystem"

        def __init__(self) -> None:
            self.execute_calls: list[dict] = []

        async def get_tools(self) -> list[dict]:
            return [dict(fs_write_tool), dict(fs_read_tool)]

        async def get_tool_def(self, tool_name: str) -> dict | None:
            if tool_name == "fs.write_text":
                return dict(fs_write_tool)
            if tool_name == "fs.read_text":
                return dict(fs_read_tool)
            return None

        def is_write_tool_def(self, tool_def: dict) -> bool:
            return tool_def.get("name") == "fs.write_text"

        def sanitize_input(self, input_data):  # noqa: ANN001
            return input_data

        def validate_tool_arguments(self, tool_name: str, arguments: dict) -> None:  # noqa: ARG002
            return None

        async def execute_tool(self, tool_name: str, arguments: dict, context=None):  # noqa: ANN001, ARG002
            self.execute_calls.append({"tool_name": tool_name, "arguments": dict(arguments or {})})
            return {"ok": True}

        async def execute_with_circuit_breaker(
            self,
            func,  # noqa: ANN001
            tool_name: str,
            arguments: dict,
            context=None,  # noqa: ANN001
        ):
            return await func(tool_name, arguments, context=context)

    class _MultiRegistry:
        def __init__(self, modules: dict[str, object], tool_to_module: dict[str, str]) -> None:
            self._modules = dict(modules)
            self._tool_to_module = dict(tool_to_module)

        async def find_module_for_tool(self, tool_name: str):  # noqa: ANN001
            module_id = self._tool_to_module.get(tool_name)
            if module_id is None:
                return None
            return self._modules.get(module_id)

        def get_module_id_for_tool(self, tool_name: str) -> str | None:
            return self._tool_to_module.get(tool_name)

        async def get_all_modules(self) -> dict[str, object]:
            return dict(self._modules)

    class _ConditionalPathService:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def evaluate_tool_call(self, **kwargs) -> dict:  # noqa: ANN003
            self.calls.append(dict(kwargs))
            if kwargs.get("tool_name") == "fs.write_text":
                return {
                    "enabled": True,
                    "within_scope": False,
                    "reason": "path_outside_current_folder_scope",
                    "force_approval": True,
                    "normalized_paths": ["/tmp/project/secret.txt"],
                    "scope_payload": {
                        "path_scope_mode": "cwd_descendants",
                        "workspace_root": "/tmp/project",
                        "scope_root": "/tmp/project/src",
                        "normalized_paths": ["/tmp/project/secret.txt"],
                        "reason": "path_outside_current_folder_scope",
                    },
                }
            return {
                "enabled": True,
                "within_scope": True,
                "reason": None,
                "force_approval": False,
                "normalized_paths": [],
                "scope_payload": None,
            }

    class _ConditionalApprovalService:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def evaluate_tool_call(self, **kwargs) -> dict:  # noqa: ANN003
            self.calls.append(dict(kwargs))
            if kwargs.get("tool_name") == "fs.write_text":
                return {
                    "status": "approval_required",
                    "approval": {
                        "approval_policy_id": 1,
                        "tool_name": "fs.write_text",
                        "reason": kwargs.get("approval_reason"),
                        "scope_context": dict(kwargs.get("scope_payload") or {}),
                    },
                }
            return {"status": "allow"}

    path_service = _ConditionalPathService()
    approval_service = _ConditionalApprovalService()

    async def _fake_get_path_service():
        return path_service

    async def _fake_get_approval_service():
        return approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    run_module = RunCommandModule(ModuleConfig(name="run", settings={"protocol": protocol}))
    filesystem_module = _FilesystemModule()
    protocol.module_registry = _MultiRegistry(
        modules={"run_command": run_module, "filesystem": filesystem_module},
        tool_to_module={
            "run": "run_command",
            "fs.write_text": "filesystem",
            "fs.read_text": "filesystem",
        },
    )

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["run", "fs.write_text", "fs.read_text"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "cwd_descendants",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-run-preflight-path-scope",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"persona_id": "researcher", "cwd": "src"},
    )

    with pytest.raises(ApprovalRequiredError):
        await protocol._handle_tools_call(
            {"name": "run", "arguments": {"command": "write ../secret.txt hi && cat ../secret.txt"}},
            context,
        )

    assert filesystem_module.execute_calls == []
    assert [call.get("tool_name") for call in path_service.calls] == ["run", "fs.write_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocked_reason", "scope_payload"),
    [
        (
            "required_slot_not_granted",
            {
                "server_id": "docs",
                "requested_slots": ["token_readonly"],
                "missing_bound_slots": ["token_readonly"],
                "blocked_reason": "required_slot_not_granted",
            },
        ),
        (
            "required_slot_secret_missing",
            {
                "server_id": "docs",
                "requested_slots": ["token_readonly"],
                "missing_secret_slots": ["token_readonly"],
                "blocked_reason": "required_slot_secret_missing",
            },
        ),
    ],
)
async def test_handle_tools_call_hard_denies_external_slot_blockers_without_approval(
    monkeypatch,
    blocked_reason: str,
    scope_payload: dict,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod

    tool_def = {
        "name": "ext.docs.search",
        "description": "Search docs",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_network": True,
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_approval_service = _NoopApprovalService()

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["ext.docs.search"],
            "approval_policy_id": 1,
            "approval_mode": "ask_outside_profile",
            "sources": [{"assignment_id": 11, "profile_id": 7}],
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    async def _path_scope(*_args, **_kwargs):
        return {"enabled": False, "within_scope": True, "reason": None, "scope_payload": None}

    async def _external_access(*_args, **_kwargs):
        return {
            "enabled": True,
            "within_scope": False,
            "reason": blocked_reason,
            "scope_payload": dict(scope_payload),
        }

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    protocol._evaluate_path_scope = _path_scope  # type: ignore[method-assign]
    protocol._evaluate_external_access = _external_access  # type: ignore[method-assign]

    context = RequestContext(
        request_id=f"req-{blocked_reason}",
        user_id="7",
        client_id="test-client",
        session_id="sess-ext-deny",
        metadata={"persona_id": "researcher"},
    )

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "ext.docs.search", "arguments": {"query": "approval needed"}},
            context,
        )

    assert fake_approval_service.calls == []


def test_external_slot_scope_key_includes_requested_slots() -> None:
    from tldw_Server_API.app.services.mcp_hub_approval_service import _scope_key_for_tool_call

    scope_key_a = _scope_key_for_tool_call(
        "ext.docs.search",
        {"query": "same"},
        scope_payload={
            "server_id": "docs",
            "requested_slots": ["token_readonly"],
            "blocked_reason": "external_confirmation_required",
        },
    )
    scope_key_b = _scope_key_for_tool_call(
        "ext.docs.search",
        {"query": "same"},
        scope_payload={
            "server_id": "docs",
            "requested_slots": ["token_readonly", "token_write"],
            "blocked_reason": "external_confirmation_required",
        },
    )
    scope_key_c = _scope_key_for_tool_call(
        "ext.docs.write",
        {"query": "same"},
        scope_payload={
            "server_id": "docs",
            "requested_slots": ["token_readonly"],
            "blocked_reason": "external_confirmation_required",
        },
    )

    assert scope_key_a != scope_key_b
    assert scope_key_a != scope_key_c


def test_path_scope_key_includes_active_workspace_id() -> None:
    from tldw_Server_API.app.services.mcp_hub_approval_service import _scope_key_for_tool_call

    scope_key_a = _scope_key_for_tool_call(
        "files.read",
        {"path": "src/notes.txt"},
        scope_payload={
            "path_scope_mode": "workspace_root",
            "workspace_id": "workspace-alpha",
            "normalized_paths": ["/tmp/project/src/notes.txt"],
            "blocked_reason": "path_outside_allowlist_scope",
        },
    )
    scope_key_b = _scope_key_for_tool_call(
        "files.read",
        {"path": "src/notes.txt"},
        scope_payload={
            "path_scope_mode": "workspace_root",
            "workspace_id": "workspace-beta",
            "normalized_paths": ["/tmp/project/src/notes.txt"],
            "blocked_reason": "path_outside_allowlist_scope",
        },
    )

    assert scope_key_a != scope_key_b


@pytest.mark.asyncio
async def test_path_grants_keep_read_edit_and_write_distinct(tmp_path) -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    workspace_root = str(tmp_path)
    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=_FakeWorkspaceRootResolver(
                {
                    "workspace_root": workspace_root,
                    "workspace_id": "workspace-actions",
                    "source": "test_workspace",
                    "reason": None,
                }
            ),
        )
    )
    context = SimpleNamespace(
        user_id="7",
        session_id="sess-actions",
        metadata={"workspace_id": "workspace-actions"},
    )

    async def _evaluate(path_grants: list[dict], tool_name: str, action: str) -> dict:
        return await path_service.evaluate_tool_call(
            effective_policy=_effective_path_policy(path_grants),
            context=context,
            tool_name=tool_name,
            tool_args={"path": "docs/a.txt"},
            tool_def=_filesystem_tool_def(tool_name, action),
        )

    def _assert_not_granted(result: dict, requested_action: str) -> None:
        assert result["within_scope"] is False  # nosec B101
        assert result["reason"] == "path_action_not_granted"  # nosec B101
        assert workspace_root not in repr(result)  # nosec B101
        decision = result["scope_payload"]["path_decisions"][0]
        assert decision["requested_action"] == requested_action  # nosec B101
        assert decision["reason_code"] == "path_action_not_granted"  # nosec B101
        assert decision["redacted"] is True  # nosec B101

    read_policy_grants = [{"prefix": "docs", "actions": ["read"]}]
    read_allowed = await _evaluate(read_policy_grants, "fs.read", "read")
    read_edit_denied = await _evaluate(read_policy_grants, "fs.edit", "edit")
    read_write_denied = await _evaluate(read_policy_grants, "fs.write", "write")

    assert read_allowed["within_scope"] is True  # nosec B101
    assert read_allowed["scope_payload"]["path_decisions"][0]["requested_action"] == "read"  # nosec B101
    _assert_not_granted(read_edit_denied, "edit")
    _assert_not_granted(read_write_denied, "write")

    edit_policy_grants = [{"prefix": "docs", "actions": ["edit"]}]
    edit_allowed = await _evaluate(edit_policy_grants, "fs.edit", "edit")
    edit_read_denied = await _evaluate(edit_policy_grants, "fs.read", "read")
    edit_write_denied = await _evaluate(edit_policy_grants, "fs.write", "write")

    assert edit_allowed["within_scope"] is True  # nosec B101
    assert edit_allowed["scope_payload"]["path_decisions"][0]["requested_action"] == "edit"  # nosec B101
    _assert_not_granted(edit_read_denied, "read")
    _assert_not_granted(edit_write_denied, "write")

    write_policy_grants = [{"prefix": "docs", "actions": ["write"]}]
    write_allowed = await _evaluate(write_policy_grants, "fs.write", "write")
    write_read_denied = await _evaluate(write_policy_grants, "fs.read", "read")
    write_edit_denied = await _evaluate(write_policy_grants, "fs.edit", "edit")

    assert write_allowed["within_scope"] is True  # nosec B101
    assert write_allowed["scope_payload"]["path_decisions"][0]["requested_action"] == "write"  # nosec B101
    _assert_not_granted(write_read_denied, "read")
    _assert_not_granted(write_edit_denied, "edit")


@pytest.mark.asyncio
async def test_path_grants_deny_override_wins_for_nested_prefix(tmp_path) -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=_FakeWorkspaceRootResolver(
                {
                    "workspace_root": str(tmp_path),
                    "workspace_id": "workspace-deny-override",
                    "source": "test_workspace",
                    "reason": None,
                }
            ),
        )
    )

    result = await path_service.evaluate_tool_call(
        effective_policy=_effective_path_policy(
            [
                {"prefix": "docs", "actions": ["read", "edit", "write"]},
                {"prefix": "docs/private", "actions": ["edit", "write"], "effect": "deny"},
            ]
        ),
        context=SimpleNamespace(
            user_id="7",
            session_id="sess-deny-override",
            metadata={"workspace_id": "workspace-deny-override"},
        ),
        tool_name="fs.edit",
        tool_args={"path": "docs/private/a.txt"},
        tool_def=_filesystem_tool_def("fs.edit", "edit"),
    )

    assert result["within_scope"] is False  # nosec B101
    assert result["reason"] == "path_action_denied"  # nosec B101
    decision = result["scope_payload"]["path_decisions"][0]
    assert decision["reason_code"] == "path_action_denied"  # nosec B101
    assert decision["matched_grant_prefix"] == "docs/private"  # nosec B101
    assert decision["matched_grant_effect"] == "deny"  # nosec B101


@pytest.mark.asyncio
async def test_path_grants_patch_bundle_fails_closed_when_create_needs_write(tmp_path) -> None:
    from mcp_unified.interfaces.path_scope import PathScopeCandidate

    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=_FakeWorkspaceRootResolver(
                {
                    "workspace_root": str(tmp_path),
                    "workspace_id": "workspace-patch-bundle",
                    "source": "test_workspace",
                    "reason": None,
                }
            ),
        )
    )

    result = await path_service.evaluate_tool_call(
        effective_policy=_effective_path_policy(
            [
                {"prefix": "docs", "actions": ["edit"]},
            ]
        ),
        context=SimpleNamespace(
            user_id="7",
            session_id="sess-patch-bundle",
            metadata={"workspace_id": "workspace-patch-bundle"},
        ),
        tool_name="fs.patch",
        tool_args={"diff": "not-inspected-here"},
        tool_def=_filesystem_tool_def("fs.patch", "edit"),
        path_scope_candidates=[
            PathScopeCandidate(path="docs/allowed.txt", action="edit", source="filesystem_diff"),
            PathScopeCandidate(
                path="docs/new.txt",
                action="write",
                source="filesystem_diff",
                creates_file=True,
            ),
        ],
    )

    assert result["within_scope"] is False  # nosec B101
    assert result["reason"] == "path_action_not_granted"  # nosec B101
    decisions = result["scope_payload"]["path_decisions"]
    write_decision = next(decision for decision in decisions if decision["requested_action"] == "write")
    assert write_decision["normalized_path"] == "docs/new.txt"  # nosec B101
    assert write_decision["reason_code"] == "path_action_not_granted"  # nosec B101
    assert write_decision["redacted"] is True  # nosec B101


@pytest.mark.asyncio
async def test_handle_tools_call_allows_direct_workspace_scoped_reader(monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": "/tmp/mcp-hub-direct/project",
            "workspace_id": "workspace-direct",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=fake_resolver,
        )
    )

    async def _fake_get_path_service():
        return path_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-direct-workspace-root",
        user_id="7",
        client_id="test-client",
        session_id=None,
        metadata={"workspace_id": "workspace-direct", "cwd": "src"},
    )

    result = await protocol._handle_tools_call(
        {"name": "files.read", "arguments": {"path": "notes.txt"}},
        context,
    )

    assert result["tool"] == "files.read"


@pytest.mark.asyncio
async def test_handle_tools_call_hard_denies_workspace_not_allowed_for_assignment_without_approval(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import GovernanceDeniedError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_approval_service = _NoopApprovalService()

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "approval_mode": "ask_outside_profile",
            "sources": [{"assignment_id": 11, "profile_id": 7}],
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    async def _path_scope(*_args, **_kwargs):
        return {
            "enabled": True,
            "within_scope": False,
            "reason": "workspace_not_allowed_for_assignment",
            "force_approval": False,
            "scope_payload": {
                "workspace_id": "workspace-beta",
                "allowed_workspace_ids": ["workspace-alpha"],
                "reason": "workspace_not_allowed_for_assignment",
            },
        }

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    protocol._evaluate_path_scope = _path_scope  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-workspace-set-deny",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-beta"},
    )

    with pytest.raises(GovernanceDeniedError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "notes.txt"}},
            context,
        )

    assert fake_approval_service.calls == []
    assert exc.value.governance["reason_code"] == "workspace_not_allowed_for_assignment"
    assert exc.value.governance["path_scope"]["workspace_id"] == "workspace-beta"
    assert exc.value.governance["path_scope"]["allowed_workspace_ids"] == ["workspace-alpha"]


@pytest.mark.asyncio
async def test_handle_tools_call_hard_denies_shared_registry_workspace_without_approval(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import GovernanceDeniedError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_approval_service = _NoopApprovalService()

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "approval_mode": "ask_outside_profile",
            "sources": [{"assignment_id": 18, "profile_id": None}],
            "selected_workspace_trust_source": "shared_registry",
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    async def _path_scope(*_args, **_kwargs):
        return {
            "enabled": True,
            "within_scope": False,
            "reason": "workspace_not_allowed_for_assignment",
            "force_approval": False,
            "scope_payload": {
                "workspace_id": "shared-secret",
                "allowed_workspace_ids": ["shared-docs"],
                "selected_workspace_trust_source": "shared_registry",
                "reason": "workspace_not_allowed_for_assignment",
            },
        }

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    protocol._evaluate_path_scope = _path_scope  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-shared-workspace-deny",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"workspace_id": "shared-secret"},
    )

    with pytest.raises(GovernanceDeniedError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "notes.txt"}},
            context,
        )

    assert fake_approval_service.calls == []
    assert exc.value.governance["reason_code"] == "workspace_not_allowed_for_assignment"
    assert exc.value.governance["path_scope"]["selected_workspace_trust_source"] == "shared_registry"


@pytest.mark.asyncio
async def test_handle_tools_call_requires_approval_for_trusted_workspace_not_allowed_by_assignment(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_approval_service = _FakeApprovalService()

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "approval_mode": "ask_outside_profile",
            "selected_assignment_id": 11,
            "selected_workspace_trust_source": "shared_registry",
            "selected_workspace_source_mode": "named",
            "sources": [{"assignment_id": 11, "profile_id": 7}],
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    async def _path_scope(*_args, **_kwargs):
        return {
            "enabled": True,
            "within_scope": False,
            "reason": "workspace_not_allowed_but_trusted",
            "force_approval": True,
            "scope_payload": {
                "workspace_id": "workspace-beta",
                "allowed_workspace_ids": ["workspace-alpha"],
                "selected_workspace_trust_source": "shared_registry",
                "selected_assignment_id": 11,
                "workspace_source_mode": "named",
                "reason": "workspace_not_allowed_but_trusted",
            },
        }

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    protocol._evaluate_path_scope = _path_scope  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-workspace-set-approval",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-beta"},
    )

    with pytest.raises(ApprovalRequiredError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "notes.txt"}},
            context,
        )

    assert exc.value.approval["reason"] == "workspace_not_allowed_but_trusted"
    assert exc.value.approval["scope_context"]["workspace_id"] == "workspace-beta"
    assert exc.value.approval["scope_context"]["selected_assignment_id"] == 11


@pytest.mark.asyncio
async def test_handle_tools_call_hard_denies_unresolvable_workspace_for_trust_source_without_approval(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import GovernanceDeniedError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_approval_service = _NoopApprovalService()

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "approval_mode": "ask_outside_profile",
            "selected_assignment_id": 12,
            "selected_workspace_trust_source": "shared_registry",
            "selected_workspace_source_mode": "named",
            "sources": [{"assignment_id": 12, "profile_id": None}],
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    async def _path_scope(*_args, **_kwargs):
        return {
            "enabled": True,
            "within_scope": False,
            "reason": "workspace_unresolvable_for_trust_source",
            "force_approval": False,
            "scope_payload": {
                "workspace_id": "workspace-missing",
                "selected_workspace_trust_source": "shared_registry",
                "selected_assignment_id": 12,
                "workspace_source_mode": "named",
                "reason": "workspace_unresolvable_for_trust_source",
            },
        }

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    protocol._evaluate_path_scope = _path_scope  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-workspace-unresolvable-deny",
        user_id="7",
        client_id="test-client",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-missing"},
    )

    with pytest.raises(GovernanceDeniedError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "notes.txt"}},
            context,
        )

    assert fake_approval_service.calls == []
    assert exc.value.governance["reason_code"] == "workspace_unresolvable_for_trust_source"
    assert exc.value.governance["path_scope"]["workspace_id"] == "workspace-missing"


@pytest.mark.asyncio
async def test_handle_tools_call_requires_approval_when_direct_workspace_root_missing(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": None,
            "workspace_id": "workspace-direct",
            "source": "sandbox_workspace_lookup",
            "reason": "workspace_root_unavailable",
        }
    )
    fake_approval_service = _FakeApprovalService()
    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=fake_resolver,
        )
    )

    async def _fake_get_path_service():
        return path_service

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-direct-missing-root",
        user_id="7",
        client_id="test-client",
        session_id=None,
        metadata={"workspace_id": "workspace-direct", "cwd": "src"},
    )

    with pytest.raises(ApprovalRequiredError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "notes.txt"}},
            context,
        )

    approval = exc.value.approval or {}
    assert approval["reason"] == "workspace_root_unavailable"
    assert approval["scope_context"]["path_scope_mode"] == "workspace_root"


@pytest.mark.asyncio
async def test_handle_tools_call_direct_cwd_descendants_stays_narrower_than_workspace_root(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import ApprovalRequiredError
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
    from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
    from tldw_Server_API.app.services import mcp_hub_approval_service as approval_service_mod
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )
    from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService

    tool_def = {
        "name": "files.read",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        },
    }
    fake_module = _FakeModule(tool_def)
    fake_resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": "/tmp/mcp-hub-direct/project",
            "workspace_id": "workspace-direct",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    fake_approval_service = _FakeApprovalService()
    path_service = McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            sandbox_service=object(),
            workspace_root_resolver=fake_resolver,
        )
    )

    async def _fake_get_path_service():
        return path_service

    async def _fake_get_approval_service():
        return fake_approval_service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_get_path_service)
    monkeypatch.setattr(approval_service_mod, "get_mcp_hub_approval_service", _fake_get_approval_service)

    protocol = MCPProtocol()
    protocol.module_registry = _FakeRegistry(fake_module)

    async def _resolve_effective_policy(_context):
        return {
            "enabled": True,
            "allowed_tools": ["files.read"],
            "approval_policy_id": 1,
            "policy_document": {
                "path_scope_mode": "cwd_descendants",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-direct-cwd-descendants",
        user_id="7",
        client_id="test-client",
        session_id=None,
        metadata={"workspace_id": "workspace-direct", "cwd": "src"},
    )

    with pytest.raises(ApprovalRequiredError) as exc:
        await protocol._handle_tools_call(
            {"name": "files.read", "arguments": {"path": "../README.md"}},
            context,
        )

    approval = exc.value.approval or {}
    assert approval["reason"] == "path_outside_current_folder_scope"
    assert approval["scope_context"]["path_scope_mode"] == "cwd_descendants"
    expected_scope_root = Path("/").joinpath("tmp", "mcp-hub-direct", "project", "src").resolve(strict=False)
    assert approval["scope_context"]["scope_root"] == str(expected_scope_root)
