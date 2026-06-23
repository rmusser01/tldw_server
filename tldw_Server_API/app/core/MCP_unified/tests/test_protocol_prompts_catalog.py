"""Protocol coverage for MCP prompt catalog routing."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import Action, Resource
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    LIBRARY_PROMPT_PREFIX,
    PromptCatalogCursor,
    PromptCatalogError,
    encode_prompt_cursor,
)
from tldw_Server_API.app.core.MCP_unified.protocol import (
    InvalidParamsException,
    MCPRequest,
    MCPProtocol,
    RequestContext,
)

pytestmark = pytest.mark.unit


class PromptOnlyRegistry:
    def __init__(self, modules: dict[str, BaseModule] | None = None) -> None:
        self.modules = modules or {}
        self.find_calls: list[str] = []

    async def get_all_modules(self) -> dict[str, BaseModule]:
        return self.modules

    async def get_module(self, module_id: str) -> BaseModule | None:
        return self.modules.get(module_id)

    async def find_module_for_prompt(self, name: str) -> None:
        self.find_calls.append(name)
        return None

    def get_module_id_for_prompt(self, name: str) -> None:
        return None


class PromptPermissionPolicy:
    def __init__(self, *, prompt_read: bool, module_read: bool) -> None:
        self.prompt_read = prompt_read
        self.module_read = module_read

    def check_permission(
        self,
        user_id: str | None,
        resource: Resource,
        action: Action,
        resource_id: str | None = None,
    ) -> bool:
        if action is not Action.READ:
            return False
        if resource is Resource.PROMPT:
            return self.prompt_read
        if resource is Resource.MODULE:
            return self.module_read
        return False


class ContextPromptModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    async def get_prompts_for_context(
        self,
        context: RequestContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "prompts": [
                {
                    "name": "library:550e8400-e29b-41d4-a716-446655440000",
                    "description": "Prompt library entry",
                    "arguments": [],
                }
            ],
            "nextCursor": "next-page-token",
            "_meta": {
                "tldw": {
                    "warnings": [
                        {"code": "config_skipped", "message": "Config prompt skipped"},
                        "ignored",
                    ]
                }
            },
        }

    async def get_prompt_for_context(
        self,
        name: str,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]:
        return {
            "description": f"Resolved {name}",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"argument-count={len(arguments)}",
                    },
                }
            ],
        }


class FailingPromptModule(ContextPromptModule):
    async def get_prompts_for_context(
        self,
        context: RequestContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        raise PromptCatalogError(
            "prompt_db_unavailable",
            "Prompt library is unavailable.",
            internal=True,
        )


class ScopedWarningPromptModule(ContextPromptModule):
    async def get_prompts_for_context(
        self,
        context: RequestContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_uuid = "550e8400-e29b-41d4-a716-446655440000"
        denied_uuid = "550e8400-e29b-41d4-a716-446655440001"
        return {
            "prompts": [
                {
                    "name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                    "description": "Allowed prompt library entry",
                    "arguments": [],
                },
                {
                    "name": f"{LIBRARY_PROMPT_PREFIX}{denied_uuid}",
                    "description": "Denied prompt library entry",
                    "arguments": [],
                },
            ],
            "_meta": {
                "tldw": {
                    "warnings": [
                        {
                            "source": "library",
                            "code": "prompt_unavailable",
                            "prompt_uuid": allowed_uuid,
                        },
                        {
                            "source": "library",
                            "code": "prompt_unavailable",
                            "prompt_uuid": denied_uuid,
                        },
                    ]
                }
            },
        }


class ScopedCursorPromptModule(ContextPromptModule):
    async def get_prompts_for_context(
        self,
        context: RequestContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_uuid = "550e8400-e29b-41d4-a716-446655440000"
        denied_uuid = "550e8400-e29b-41d4-a716-446655440001"
        denied_name = "Denied Private Prompt"
        return {
            "prompts": [
                {
                    "name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                    "description": "Allowed prompt library entry",
                    "arguments": [],
                }
            ],
            "nextCursor": encode_prompt_cursor(
                PromptCatalogCursor(
                    library_after_name=denied_name,
                    library_after_uuid=denied_uuid,
                )
            ),
        }


class IdentifierWarningPromptModule(ContextPromptModule):
    async def get_prompts_for_context(
        self,
        context: RequestContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_uuid = "550e8400-e29b-41d4-a716-446655440000"
        return {
            "prompts": [
                {
                    "name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                    "description": "Allowed prompt library entry",
                    "arguments": [],
                }
            ],
            "_meta": {
                "tldw": {
                    "warnings": [
                        {
                            "source": "library",
                            "code": "prompt_unavailable",
                            "_prompt_name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                            "prompt_name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                            "prompt_uuid": allowed_uuid,
                            "prompt_id": 42,
                            "id": "library-row-42",
                        },
                        {
                            "source": "config",
                            "code": "config_prompt_unavailable",
                            "_prompt_name": f"{LIBRARY_PROMPT_PREFIX}{allowed_uuid}",
                            "prompt_uuid": allowed_uuid,
                            "prompt_id": 43,
                            "id": "mismatched.config.id",
                        },
                    ]
                }
            },
        }


class ErrorPromptModule(ContextPromptModule):
    def __init__(self, config: ModuleConfig, error: PromptCatalogError) -> None:
        super().__init__(config)
        self.error = error

    async def get_prompt_for_context(
        self,
        name: str,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]:
        raise self.error


def _handler_with_registry(registry: PromptOnlyRegistry) -> MCPProtocol:
    handler = MCPProtocol()
    handler.module_registry = registry
    return handler


def _prompt_module() -> ContextPromptModule:
    return ContextPromptModule(
        ModuleConfig(
            name="prompts",
            version="1.0.0",
            description="Prompt catalog test module",
        )
    )


def _context() -> RequestContext:
    return RequestContext(request_id="prompt-catalog", user_id="u1", client_id="unit")


async def _async_true(*args: Any, **kwargs: Any) -> bool:
    return True


def _decoded_cursor_text(cursor: str) -> str:
    padding = "=" * (-len(cursor) % 4)
    payload = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
    return json.dumps(json.loads(payload.decode("utf-8")), sort_keys=True)


@pytest.mark.asyncio
async def test_initialize_declares_mcp_prompt_capability() -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": _prompt_module()}))

    result = await handler._handle_initialize({"clientInfo": {"name": "unit"}}, _context())

    assert result["capabilities"]["prompts"] == {  # nosec B101
        "available": True,
        "listChanged": False,
    }


@pytest.mark.asyncio
async def test_prompts_list_uses_context_hook_and_preserves_cursor_and_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": _prompt_module()}))

    async def _allow_namespaced(context: RequestContext, prompt_name: str) -> bool:
        return True

    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _allow_namespaced)

    result = await handler._handle_prompts_list({}, _context())

    assert result["prompts"] == [  # nosec B101
        {
            "name": "library:550e8400-e29b-41d4-a716-446655440000",
            "description": "Prompt library entry",
            "arguments": [],
            "module": "prompts",
        }
    ]
    assert result["nextCursor"] == "next-page-token"  # nosec B101
    assert result["_meta"]["tldw"]["warnings"] == [  # nosec B101
        {"code": "config_skipped", "message": "Config prompt skipped"}
    ]


@pytest.mark.asyncio
async def test_prompts_list_filters_and_sanitizes_scoped_warning_metadata() -> None:
    module = ScopedWarningPromptModule(
        ModuleConfig(
            name="prompts",
            version="1.0.0",
            description="Scoped warning prompt catalog test module",
        )
    )
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)
    allowed_prompt_name = f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000"
    denied_prompt_name = f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440001"
    context = RequestContext(
        request_id="prompt-catalog-scoped-warnings",
        user_id="u1",
        client_id="unit",
        metadata={"permissions": [f"mcp:prompt:{allowed_prompt_name}"]},
    )

    result = await handler._handle_prompts_list({}, context)

    assert [prompt["name"] for prompt in result["prompts"]] == [allowed_prompt_name]  # nosec B101
    assert denied_prompt_name not in str(result)  # nosec B101
    assert "550e8400-e29b-41d4-a716-446655440001" not in str(result)  # nosec B101
    assert result["_meta"]["tldw"]["warnings"] == [  # nosec B101
        {"source": "library", "code": "prompt_unavailable"}
    ]


@pytest.mark.asyncio
async def test_prompts_list_does_not_return_denied_identifier_cursor_for_scoped_callers() -> None:
    module = ScopedCursorPromptModule(
        ModuleConfig(
            name="prompts",
            version="1.0.0",
            description="Scoped cursor prompt catalog test module",
        )
    )
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)
    allowed_prompt_name = f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000"
    denied_uuid = "550e8400-e29b-41d4-a716-446655440001"
    denied_name = "Denied Private Prompt"
    context = RequestContext(
        request_id="prompt-catalog-scoped-cursor",
        user_id="u1",
        client_id="unit",
        metadata={"permissions": [f"mcp:prompt:{allowed_prompt_name}"]},
    )

    result = await handler._handle_prompts_list({}, context)

    assert [prompt["name"] for prompt in result["prompts"]] == [allowed_prompt_name]  # nosec B101
    assert denied_name not in str(result["prompts"])  # nosec B101
    assert denied_uuid not in str(result["prompts"])  # nosec B101
    if "nextCursor" in result:
        decoded_cursor = _decoded_cursor_text(result["nextCursor"])
        assert denied_name not in decoded_cursor  # nosec B101
        assert denied_uuid not in decoded_cursor  # nosec B101


@pytest.mark.asyncio
async def test_visible_prompt_warning_strips_identifier_fields_after_explicit_name_resolution() -> None:
    module = IdentifierWarningPromptModule(
        ModuleConfig(
            name="prompts",
            version="1.0.0",
            description="Identifier warning prompt catalog test module",
        )
    )
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)
    allowed_prompt_name = f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000"
    context = RequestContext(
        request_id="prompt-catalog-warning-identifiers",
        user_id="u1",
        client_id="unit",
        metadata={"permissions": [f"mcp:prompt:{allowed_prompt_name}"]},
    )

    result = await handler._handle_prompts_list({}, context)

    assert result["_meta"]["tldw"]["warnings"] == [  # nosec B101
        {"source": "library", "code": "prompt_unavailable"},
        {"source": "config", "code": "config_prompt_unavailable"},
    ]


@pytest.mark.asyncio
async def test_prompts_list_maps_internal_catalog_error_to_runtime_error() -> None:
    handler = _handler_with_registry(
        PromptOnlyRegistry(
            {
                "prompts": FailingPromptModule(
                    ModuleConfig(
                        name="prompts",
                        version="1.0.0",
                        description="Failing prompt catalog test module",
                    )
                )
            }
        )
    )

    with pytest.raises(RuntimeError, match="Failed to list prompts") as excinfo:
        await handler._handle_prompts_list({}, _context())

    assert excinfo.value.__cause__ is None  # nosec B101
    assert excinfo.value.__suppress_context__ is True  # nosec B101


@pytest.mark.asyncio
async def test_prompts_get_dispatches_namespaced_prompt_before_global_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = PromptOnlyRegistry({"prompts": _prompt_module()})
    handler = _handler_with_registry(registry)

    async def _allow_namespaced(context: RequestContext, prompt_name: str) -> bool:
        return True

    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _allow_namespaced)

    result = await handler._handle_prompts_get(
        {
            "name": "library:550e8400-e29b-41d4-a716-446655440000",
            "arguments": {"topic": "catalog"},
        },
        _context(),
    )

    assert result["description"] == "Resolved library:550e8400-e29b-41d4-a716-446655440000"  # nosec B101
    assert result["messages"][0]["content"]["text"] == "argument-count=1"  # nosec B101
    assert registry.find_calls == []  # nosec B101


@pytest.mark.asyncio
async def test_prompts_get_rejects_non_object_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": _prompt_module()}))

    async def _allow_namespaced(context: RequestContext, prompt_name: str) -> bool:
        return True

    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _allow_namespaced)

    with pytest.raises(InvalidParamsException, match="Prompt arguments must be an object"):
        await handler._handle_prompts_get(
            {
                "name": "library:550e8400-e29b-41d4-a716-446655440000",
                "arguments": [],
            },
            _context(),
        )


@pytest.mark.asyncio
async def test_protocol_maps_catalog_invalid_params_without_body_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _handler_with_registry(
        PromptOnlyRegistry(
            {
                "prompts": ErrorPromptModule(
                    ModuleConfig(
                        name="prompts",
                        version="1.0.0",
                        description="Error prompt catalog test module",
                    ),
                    PromptCatalogError(
                        "missing_required_variable",
                        "Missing required variable: topic",
                    ),
                )
            }
        )
    )
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _async_true)

    with pytest.raises(InvalidParamsException) as excinfo:
        await handler._handle_prompts_get(
            {
                "name": f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000",
                "arguments": {},
            },
            _context(),
        )

    assert "Missing required variable: topic" in str(excinfo.value)  # nosec B101


@pytest.mark.asyncio
async def test_protocol_maps_internal_catalog_error_to_internal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _handler_with_registry(
        PromptOnlyRegistry(
            {
                "prompts": ErrorPromptModule(
                    ModuleConfig(
                        name="prompts",
                        version="1.0.0",
                        description="Internal error prompt catalog test module",
                    ),
                    PromptCatalogError(
                        "prompt_db_unavailable",
                        "Prompt body SENSITIVE",
                        internal=True,
                    ),
                )
            }
        )
    )
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _async_true)

    with pytest.raises(RuntimeError) as excinfo:
        await handler._handle_prompts_get(
            {
                "name": f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000",
                "arguments": {},
            },
            _context(),
        )

    assert "Failed to get prompt" in str(excinfo.value)  # nosec B101
    assert "SENSITIVE" not in str(excinfo.value)  # nosec B101
    assert excinfo.value.__cause__ is None  # nosec B101
    assert excinfo.value.__suppress_context__ is True  # nosec B101


@pytest.mark.asyncio
async def test_process_request_internal_catalog_error_does_not_leak_to_response_or_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _handler_with_registry(
        PromptOnlyRegistry(
            {
                "prompts": ErrorPromptModule(
                    ModuleConfig(
                        name="prompts",
                        version="1.0.0",
                        description="Internal error prompt catalog test module",
                    ),
                    PromptCatalogError(
                        "prompt_db_unavailable",
                        "Prompt body SENSITIVE",
                        internal=True,
                    ),
                )
            }
        )
    )
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", _async_true)
    captured_logs: list[str] = []
    sink_id = logger.add(lambda message: captured_logs.append(str(message)), level="DEBUG")

    try:
        response = await handler.process_request(
            MCPRequest(
                method="prompts/get",
                params={
                    "name": f"{LIBRARY_PROMPT_PREFIX}550e8400-e29b-41d4-a716-446655440000",
                    "arguments": {},
                },
                id="prompt-internal-error",
            ),
            _context(),
        )
    finally:
        logger.remove(sink_id)

    assert response is not None  # nosec B101
    assert response.error is not None  # nosec B101
    assert response.error.code == -32603  # nosec B101
    assert response.error.message == "Internal error"  # nosec B101
    assert "SENSITIVE" not in str(response)  # nosec B101
    assert "SENSITIVE" not in "".join(captured_logs)  # nosec B101


@pytest.mark.asyncio
async def test_prompts_list_allows_prompts_read_without_module_read_for_namespaced_prompts() -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": _prompt_module()}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)

    response = await handler.process_request(
        MCPRequest(method="prompts/list", params={}, id="prompt-list-no-module-read"),
        _context(),
    )

    assert response is not None  # nosec B101
    assert response.error is None  # nosec B101
    assert response.result is not None  # nosec B101
    assert [prompt["name"] for prompt in response.result["prompts"]] == [  # nosec B101
        "library:550e8400-e29b-41d4-a716-446655440000"
    ]


@pytest.mark.asyncio
async def test_namespaced_prompt_denies_when_only_module_read_is_granted() -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": _prompt_module()}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=False, module_read=True)

    result = await handler._handle_prompts_list({}, _context())

    assert result["prompts"] == []  # nosec B101
