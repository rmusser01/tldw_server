from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args, get_origin, get_type_hints
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.MCP_unified import protocol as protocol_module
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
from tldw_Server_API.app.core.MCP_unified.protocol_types import (
    AuthenticatedExecutionScope,
    InvalidParamsException,
    RequestContext,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.canonical import (
    CanonicalJsonTooLarge,
    canonical_json_bytes,
    decode_canonical_json_object,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.models import (
    CanonicalJsonSnapshot,
    PreparedExecutionPolicy,
)


class _AllowAllRBAC:
    async def check_permission(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True


class _IntegrityWriteModule(BaseModule):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.tool_name = f"{config.name}.write"
        self.source_tool_def: dict[str, Any] = {
            "name": self.tool_name,
            "description": "Prepared execution integrity test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "idempotencyKey": {"type": "string"},
                },
                "required": ["value"],
                "additionalProperties": True,
            },
            "metadata": {
                "category": "management",
                "rate_limit_fail_closed": True,
            },
        }

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ready": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self.source_tool_def]

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del arguments, tool_def
        return tool_name == self.tool_name

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == self.tool_name and not arguments.get("value"):
            raise ValueError("value is required")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, context
        return {"value": arguments["value"]}


def _runtime_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "validate_input_schema": True,
        "disable_write_tools": False,
        "idempotency_ttl_seconds": 300,
        "idempotency_cache_size": 512,
        "idempotency_wait_seconds": 5,
        "idempotency_finalize_seconds": 5,
        "idempotency_result_max_bytes": 256_000,
        "module_timeout": 30,
        "tool_category_map": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def _prepare_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context: RequestContext | None = None,
    config: SimpleNamespace | None = None,
    module_timeout: Any = 30,
    metadata: dict[str, Any] | None = None,
    scope_payload: dict[str, Any] | None = None,
    idempotency_key: str = "idem-key",
) -> tuple[MCPProtocol, _IntegrityWriteModule, Any, dict[str, Any]]:
    module_id = f"prepared_integrity_{uuid4().hex}"
    registry = get_module_registry()
    await registry.register_module(
        module_id,
        _IntegrityWriteModule,
        ModuleConfig(name=module_id, timeout_seconds=module_timeout),
    )
    module = await registry.get_module(module_id)
    assert isinstance(module, _IntegrityWriteModule)
    if metadata is not None:
        module.source_tool_def["metadata"] = metadata

    mutable_scope = scope_payload if scope_payload is not None else {"path_scope_mode": "workspace"}

    async def _evaluate_path_scope(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"within_scope": True, "scope_payload": mutable_scope}

    runtime_config = config or _runtime_config()
    monkeypatch.setattr(protocol_module, "get_config", lambda: runtime_config)

    protocol = MCPProtocol()
    protocol.rbac_policy = _AllowAllRBAC()
    monkeypatch.setattr(protocol, "_evaluate_path_scope", _evaluate_path_scope)
    prepared = await protocol.prepare_tool_call(
        params={
            "name": module.tool_name,
            "arguments": {"value": "alpha"},
        },
        context=context or RequestContext(request_id=f"request-{module_id}", user_id="user-1"),
        idempotency_key=idempotency_key,
    )
    return protocol, module, prepared, mutable_scope


def test_canonical_json_is_sorted_compact_unicode_preserving_and_stable() -> None:
    value = {"z": [3, {"b": True, "a": None}], "accent": "café"}

    first = canonical_json_bytes(value, max_bytes=1_000)
    second = canonical_json_bytes(value, max_bytes=1_000)

    assert first == b'{"accent":"caf\xc3\xa9","z":[3,{"a":null,"b":true}]}'
    assert second == first


@dataclass
class _UnsupportedDataclass:
    value: str


class _UnsupportedObject:
    pass


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string-key"},
        ("tuple",),
        {"set"},
        b"bytes",
        Path("/tmp/not-json"),
        _UnsupportedDataclass("value"),
        _UnsupportedObject(),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_canonical_json_rejects_non_json_values(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes(value, max_bytes=1_000)


@pytest.mark.parametrize("container_kind", ["list", "dict"])
def test_canonical_json_rejects_cycles_deterministically(container_kind: str) -> None:
    if container_kind == "list":
        value: Any = []
        value.append(value)
    else:
        value = {}
        value["self"] = value

    with pytest.raises(TypeError, match="Cyclic JSON structure"):
        canonical_json_bytes(value, max_bytes=1_000)


def test_canonical_json_allows_exact_byte_limit_and_rejects_one_byte_less() -> None:
    value = {"message": "é"}
    encoded = canonical_json_bytes(value, max_bytes=10_000)

    assert canonical_json_bytes(value, max_bytes=len(encoded)) == encoded
    with pytest.raises(CanonicalJsonTooLarge) as exc_info:
        canonical_json_bytes(value, max_bytes=len(encoded) - 1)

    assert exc_info.value.max_bytes == len(encoded) - 1
    assert exc_info.value.actual_bytes == len(encoded)


def test_canonical_object_decoder_returns_fresh_objects_and_validates_shape() -> None:
    encoded = canonical_json_bytes({"nested": {"value": 1}}, max_bytes=1_000)

    first = decode_canonical_json_object(encoded, max_bytes=1_000)
    second = decode_canonical_json_object(encoded, max_bytes=1_000)
    assert first == second
    assert first is not second
    assert first["nested"] is not second["nested"]

    with pytest.raises(TypeError, match="top-level JSON object"):
        decode_canonical_json_object(b"[]", max_bytes=1_000)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "at least one"),
        ({"active_org_id": 0}, "positive non-boolean integer"),
        ({"active_org_id": -1}, "positive non-boolean integer"),
        ({"active_org_id": True}, "positive non-boolean integer"),
        ({"active_team_id": 1.5}, "positive non-boolean integer"),
        ({"active_team_id": "7"}, "positive non-boolean integer"),
    ],
)
def test_authenticated_execution_scope_rejects_empty_or_invalid_ids(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AuthenticatedExecutionScope(**kwargs)


def test_authenticated_execution_scope_accepts_positive_non_bool_ids() -> None:
    scope = AuthenticatedExecutionScope(active_org_id=11, active_team_id=22)

    assert scope.active_org_id == 11
    assert scope.active_team_id == 22


def test_public_prepared_execution_annotations_resolve_at_runtime() -> None:
    prepared_hints = get_type_hints(protocol_module.PreparedToolCall)
    scope_hints = get_type_hints(protocol_module.AuthenticatedExecutionScope)
    canonical_object_hints = get_type_hints(
        protocol_module.AuthenticatedExecutionScope.canonical_object,
    )

    assert prepared_hints["policy"] is PreparedExecutionPolicy
    assert prepared_hints["tool_definition_snapshot"] is CanonicalJsonSnapshot
    assert prepared_hints["scope_reporting_snapshot"] is CanonicalJsonSnapshot
    assert scope_hints == {"active_org_id": int | None, "active_team_id": int | None}
    canonical_return = canonical_object_hints["return"]
    assert get_origin(canonical_return) is dict
    assert get_args(canonical_return)[0] is str


def test_fingerprinting_rejects_forged_all_none_authenticated_scope() -> None:
    scope = AuthenticatedExecutionScope(active_org_id=11)
    object.__setattr__(scope, "active_org_id", None)
    context = RequestContext(
        request_id="forged-empty-scope",
        user_id="user-1",
        server_auth_scope=scope,
    )
    protocol = MCPProtocol()

    with pytest.raises(ValueError, match="at least one"):
        protocol._tool_execution_security.fingerprint_idempotency_scope(context)


@pytest.mark.asyncio
async def test_preparation_detaches_tool_and_scope_snapshots_and_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(
        idempotency_ttl_seconds=45,
        idempotency_cache_size=73,
        idempotency_wait_seconds=6,
        idempotency_finalize_seconds=7,
        idempotency_result_max_bytes=300_000,
    )
    source_scope = {"path_scope_mode": "workspace", "allowed": ["alpha"]}
    protocol, module, prepared, _ = await _prepare_call(
        monkeypatch,
        config=config,
        module_timeout=20,
        scope_payload=source_scope,
    )
    del protocol

    module.source_tool_def["metadata"]["category"] = "network"
    module.source_tool_def["inputSchema"]["properties"].pop("idempotencyKey")
    source_scope["allowed"].append("mutated")
    config.idempotency_ttl_seconds = 999
    config.idempotency_cache_size = 999

    first_tool_def = prepared.tool_def
    first_scope = prepared.scope_payload
    assert first_tool_def is not None
    assert first_scope is not None
    first_tool_def["metadata"]["category"] = "changed-by-observer"
    first_scope["allowed"].append("changed-by-observer")

    assert prepared.tool_def["metadata"]["category"] == "management"
    assert prepared.scope_payload == {"allowed": ["alpha"], "path_scope_mode": "workspace"}
    assert prepared.policy.rate_limit_category == "management"
    assert prepared.policy.rate_limit_fail_closed is True
    assert prepared.policy.idempotency.inject_argument is True
    assert prepared.policy.idempotency.ttl_seconds == 45
    assert prepared.policy.idempotency.contention_wait_seconds == 6
    assert prepared.policy.idempotency.finalize_seconds == 7
    assert prepared.policy.idempotency.lock_ttl_seconds == 47
    assert prepared.policy.idempotency.max_entries == 73
    assert prepared.policy.idempotency.max_result_bytes == 300_000


@pytest.mark.parametrize(
    ("metadata_value", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
@pytest.mark.asyncio
async def test_rate_limit_fail_closed_requires_literal_json_true(
    monkeypatch: pytest.MonkeyPatch,
    metadata_value: Any,
    expected: bool,
) -> None:
    metadata = {"category": "read", "rate_limit_fail_closed": metadata_value}
    _, _, prepared, _ = await _prepare_call(monkeypatch, metadata=metadata)

    assert prepared.policy.rate_limit_fail_closed is expected


@pytest.mark.parametrize(
    ("config_overrides", "module_timeout"),
    [
        ({"idempotency_ttl_seconds": True}, 30),
        ({"idempotency_ttl_seconds": 0}, 30),
        ({"idempotency_ttl_seconds": 604_801}, 30),
        ({"idempotency_cache_size": True}, 30),
        ({"idempotency_cache_size": 0}, 30),
        ({"idempotency_cache_size": 100_001}, 30),
        ({}, True),
        ({"idempotency_ttl_seconds": 1}, 302_398),
    ],
)
@pytest.mark.asyncio
async def test_preparation_rejects_invalid_runtime_policy_values(
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: dict[str, Any],
    module_timeout: Any,
) -> None:
    with pytest.raises(InvalidParamsException, match="Invalid idempotency execution policy"):
        await _prepare_call(
            monkeypatch,
            config=_runtime_config(**config_overrides),
            module_timeout=module_timeout,
        )


@pytest.mark.asyncio
async def test_prepared_integrity_rejects_manually_signed_noncanonical_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _, prepared, _ = await _prepare_call(monkeypatch)
    observer_tool_def = prepared.tool_def
    assert observer_tool_def is not None
    noncanonical = json.dumps(observer_tool_def, ensure_ascii=False, indent=2).encode("utf-8")
    snapshot = CanonicalJsonSnapshot(
        encoded=noncanonical,
        sha256=hashlib.sha256(noncanonical).hexdigest(),
    )
    forged = replace(prepared, tool_definition_snapshot=snapshot)
    forged = replace(
        forged,
        integrity_tag=protocol._build_prepared_tool_call_integrity_tag(
            tool_name=forged.tool_name,
            module_id=forged.module_id,
            policy=forged.policy,
            idempotency_cache_key=forged.idempotency_cache_key,
            normalized_idempotency_key_digest=forged.normalized_idempotency_key_digest,
            arguments_hash=forged.arguments_hash,
            context_fingerprint=forged.context_fingerprint,
            idempotency_scope_fingerprint=forged.idempotency_scope_fingerprint,
            tool_definition_sha256=forged.tool_definition_snapshot.sha256,
            scope_reporting_sha256=forged.scope_reporting_snapshot.sha256,
        ),
    )

    with pytest.raises(InvalidParamsException, match="non-canonical tool definition snapshot"):
        protocol._verify_prepared_tool_call_integrity(forged)


@pytest.mark.parametrize(
    "tamper_target",
    [
        "policy",
        "oversized_policy",
        "tool_snapshot",
        "scope_snapshot",
        "raw_key",
        "cache_key",
        "arguments",
        "context",
        "explicit_scope",
    ],
)
@pytest.mark.asyncio
async def test_prepared_integrity_rejects_authoritative_state_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    context = RequestContext(
        request_id=f"tamper-{tamper_target}",
        user_id="user-1",
        metadata={"workspace": "alpha"},
        server_auth_scope=AuthenticatedExecutionScope(active_org_id=101),
    )
    protocol, _, prepared, _ = await _prepare_call(monkeypatch, context=context)
    candidate = prepared

    if tamper_target == "policy":
        candidate = replace(
            prepared,
            policy=replace(prepared.policy, rate_limit_category="network"),
        )
    elif tamper_target == "oversized_policy":
        candidate = replace(
            prepared,
            policy=replace(prepared.policy, rate_limit_category="x" * 64_000),
        )
    elif tamper_target == "tool_snapshot":
        candidate = replace(
            prepared,
            tool_definition_snapshot=replace(
                prepared.tool_definition_snapshot,
                encoded=b"null",
            ),
        )
    elif tamper_target == "scope_snapshot":
        candidate = replace(
            prepared,
            scope_reporting_snapshot=replace(
                prepared.scope_reporting_snapshot,
                encoded=b"null",
            ),
        )
    elif tamper_target == "raw_key":
        candidate = replace(prepared, normalized_idempotency_key="different-key")
    elif tamper_target == "cache_key":
        assert prepared.idempotency_cache_key is not None
        candidate = replace(prepared, idempotency_cache_key=f"{prepared.idempotency_cache_key}-tampered")
    elif tamper_target == "arguments":
        prepared.tool_args["value"] = "tampered"
    elif tamper_target == "context":
        prepared.context.metadata["workspace"] = "tampered"
    elif tamper_target == "explicit_scope":
        prepared.context.server_auth_scope = AuthenticatedExecutionScope(active_org_id=202)

    with pytest.raises(InvalidParamsException, match="Prepared tool call integrity check failed"):
        protocol._verify_prepared_tool_call_integrity(candidate)


def test_metadata_org_and_team_ids_do_not_define_authenticated_scope_domain() -> None:
    protocol = MCPProtocol()
    first = RequestContext(
        request_id="scope-metadata",
        user_id="user-1",
        metadata={"org_id": 11, "team_id": 22},
    )
    second = RequestContext(
        request_id="scope-metadata",
        user_id="user-1",
        metadata={"org_id": 33, "team_id": 44},
    )

    assert protocol._tool_execution_security.fingerprint_idempotency_scope(first) == ""
    assert protocol._tool_execution_security.fingerprint_idempotency_scope(second) == ""
    assert protocol._make_idempotency_cache_key(first, "module-a", "tool.write", "key-1") == (
        protocol._make_idempotency_cache_key(second, "module-a", "tool.write", "key-1")
    )


def test_personal_idempotency_key_preserves_exact_legacy_shape() -> None:
    protocol = MCPProtocol()
    context = RequestContext(request_id="personal", user_id="user-1")

    assert protocol._make_idempotency_cache_key(context, "module-a", "tool.write", "key-1") == (
        "user:user-1|module:module-a|tool:tool.write|key:key-1"
    )


@pytest.mark.asyncio
async def test_preparation_preserves_patched_protocol_cache_key_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MCPProtocol._make_idempotency_cache_key
    calls: list[str] = []

    def _patched_facade(
        protocol: MCPProtocol,
        context: RequestContext,
        module_name: str,
        tool_name: str,
        idempotency_key: str,
    ) -> str:
        calls.append(idempotency_key)
        return original(protocol, context, module_name, tool_name, idempotency_key)

    monkeypatch.setattr(MCPProtocol, "_make_idempotency_cache_key", _patched_facade)

    protocol, _, prepared, _ = await _prepare_call(monkeypatch)

    assert calls == ["idem-key"]
    assert prepared.idempotency_cache_key == protocol._tool_execution_security.make_idempotency_cache_key(
        prepared.context,
        prepared.module_id or prepared.module.name,
        prepared.tool_name,
        "idem-key",
    )


def test_explicit_scopes_append_distinct_fixed_format_digests_without_raw_ids() -> None:
    protocol = MCPProtocol()
    org_context = RequestContext(
        request_id="scoped",
        user_id="user-1",
        server_auth_scope=AuthenticatedExecutionScope(active_org_id=101_001),
    )
    team_context = RequestContext(
        request_id="scoped",
        user_id="user-1",
        server_auth_scope=AuthenticatedExecutionScope(active_team_id=202_002),
    )

    org_key = protocol._make_idempotency_cache_key(org_context, "module-a", "tool.write", "key-1")
    team_key = protocol._make_idempotency_cache_key(team_context, "module-a", "tool.write", "key-1")
    org_digest = org_key.rsplit("|scope:sha256:", 1)[1]
    team_digest = team_key.rsplit("|scope:sha256:", 1)[1]

    assert org_key != team_key
    assert re.fullmatch(r"[0-9a-f]{64}", org_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", team_digest)
    assert "101001" not in org_key
    assert "202002" not in team_key
