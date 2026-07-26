from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, get_args, get_origin, get_type_hints
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.MCP_unified import protocol as protocol_module
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import (
    ModuleStatus,
    get_module_registry,
)
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
        self.tool_name = str(config.settings.get("tool_name") or f"{config.name}.write")
        self.breaker_entry_count = 0
        self.execute_count = 0
        self.last_arguments: dict[str, Any] | None = None
        self.tool_def_calls = 0
        self.block_tool_def_on_call: int | None = None
        self.tool_def_entered = asyncio.Event()
        self.tool_def_release = asyncio.Event()
        self.has_tool_calls = 0
        self.block_has_tool_on_call: int | None = None
        self.has_tool_entered = asyncio.Event()
        self.has_tool_release = asyncio.Event()
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

    async def get_tool_def(self, tool_name: str) -> dict[str, Any] | None:
        self.tool_def_calls += 1
        if self.tool_def_calls == self.block_tool_def_on_call:
            self.tool_def_entered.set()
            await self.tool_def_release.wait()
        return await super().get_tool_def(tool_name)

    async def has_tool(self, tool_name: str) -> bool:
        self.has_tool_calls += 1
        if self.has_tool_calls == self.block_has_tool_on_call:
            self.has_tool_entered.set()
            await self.has_tool_release.wait()
        return await super().has_tool(tool_name)

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
        self.execute_count += 1
        self.last_arguments = dict(arguments)
        return {"value": arguments["value"]}

    async def execute_with_circuit_breaker(
        self,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.breaker_entry_count += 1
        return await super().execute_with_circuit_breaker(operation, *args, **kwargs)


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
    idempotency_key: str | None = "idem-key",
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


class _RateLimiterProbe:
    def __init__(self) -> None:
        self.calls = 0

    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls += 1


class _IdempotencyProbe:
    def __init__(self) -> None:
        self.bound_keys: list[str] = []
        self.run_keys: list[str] = []

    async def bind_arguments(
        self,
        key: str,
        arguments_hash: str,
        *,
        ttl: int,
        max_size: int,
    ) -> bool:
        del arguments_hash, ttl, max_size
        self.bound_keys.append(key)
        return True

    async def run(
        self,
        key: str,
        execute: Any,
        *,
        ttl: int,
        max_size: int,
        lock_ttl: int,
    ) -> tuple[Any, bool]:
        del ttl, max_size, lock_ttl
        self.run_keys.append(key)
        return await execute(), False


class _BlockingRateLimiter(_RateLimiterProbe):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls += 1
        self.entered.set()
        await self.release.wait()


class _BlockingIdempotency(_IdempotencyProbe):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def run(
        self,
        key: str,
        execute: Any,
        *,
        ttl: int,
        max_size: int,
        lock_ttl: int,
    ) -> tuple[Any, bool]:
        del ttl, max_size, lock_ttl
        self.run_keys.append(key)
        self.entered.set()
        await self.release.wait()
        return await execute(), False


class _BlockingSecondLookupRegistry:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls = 0
        self.block_when: Callable[[], bool] = lambda: False
        self.blocked = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def find_module_for_tool(self, tool_name: str) -> BaseModule | None:
        self.calls += 1
        if not self.blocked and self.block_when():
            self.blocked = True
            self.entered.set()
            await self.release.wait()
        return await self.delegate.find_module_for_tool(tool_name)

    async def get_module(self, module_id: str) -> BaseModule | None:
        return await self.delegate.get_module(module_id)

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        return self.delegate.get_module_id_for_tool(tool_name)


def _assert_stale_prepared_call(payload: dict[str, Any]) -> None:
    execution_eval = payload["eval"]
    assert payload == {
        "content": [
            {
                "type": "json",
                "json": {
                    "status": "failed",
                    "reason_code": "stale_prepared_call",
                    "message": "The prepared tool call is no longer valid.",
                },
            }
        ],
        "isError": True,
        "module": payload["module"],
        "tool": payload["tool"],
        "eval": execution_eval,
    }


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


@pytest.mark.asyncio
async def test_async_prepared_verifier_always_checks_integrity_without_live_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _, prepared, _ = await _prepare_call(monkeypatch)
    forged = replace(
        prepared,
        policy=replace(prepared.policy, rate_limit_category="read"),
    )

    with pytest.raises(InvalidParamsException, match="Prepared tool call integrity check failed"):
        await protocol._tool_execution_security.verify_prepared_tool_call(
            forged,
            require_live_binding=False,
        )


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


def test_scoped_idempotency_key_cannot_collide_with_crafted_unscoped_raw_key() -> None:
    protocol = MCPProtocol()
    scoped_context = RequestContext(
        request_id="scoped-collision",
        user_id="user-1",
        server_auth_scope=AuthenticatedExecutionScope(active_org_id=101_001),
    )
    unscoped_context = RequestContext(request_id="unscoped-collision", user_id="user-1")
    scope_digest = protocol._tool_execution_security.fingerprint_idempotency_scope(
        scoped_context,
    )
    scoped_key = protocol._make_idempotency_cache_key(
        scoped_context,
        "module-a",
        "tool.write",
        "key-1",
    )
    crafted_raw_key = f"key-1|scope:sha256:{scope_digest}"
    crafted_unscoped_key = protocol._make_idempotency_cache_key(
        unscoped_context,
        "module-a",
        "tool.write",
        crafted_raw_key,
    )

    assert scoped_key == (
        f"user:user-1|module:module-a|tool:tool.write|scope:sha256:{scope_digest}|key:key-1"
    )
    assert crafted_unscoped_key == (
        f"user:user-1|module:module-a|tool:tool.write|key:{crafted_raw_key}"
    )
    assert crafted_unscoped_key != scoped_key


def test_explicit_scopes_use_distinct_fixed_format_digests_without_raw_ids() -> None:
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
    org_digest = protocol._tool_execution_security.fingerprint_idempotency_scope(org_context)
    team_digest = protocol._tool_execution_security.fingerprint_idempotency_scope(team_context)

    assert org_key != team_key
    assert re.fullmatch(r"[0-9a-f]{64}", org_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", team_digest)
    assert org_key == (
        f"user:user-1|module:module-a|tool:tool.write|scope:sha256:{org_digest}|key:key-1"
    )
    assert team_key == (
        f"user:user-1|module:module-a|tool:tool.write|scope:sha256:{team_digest}|key:key-1"
    )
    assert "101001" not in org_key
    assert "202002" not in team_key


@pytest.mark.parametrize(
    "drift",
    [
        "unregistered",
        "disabled",
        "replacement_same_id",
        "remapped",
        "module_id_drift",
        "renamed",
        "definition_changed",
        "resolution_failure",
    ],
)
@pytest.mark.asyncio
async def test_first_live_check_blocks_stale_registry_bindings_before_admission(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    protocol, module, prepared, _ = await _prepare_call(monkeypatch)
    registry = get_module_registry()

    if drift == "unregistered":
        assert prepared.module_id is not None
        await registry.unregister_module(prepared.module_id)
    elif drift == "disabled":
        module.config.enabled = False
    elif drift == "replacement_same_id":
        assert prepared.module_id is not None
        await registry.unregister_module(prepared.module_id)
        await registry.register_module(
            prepared.module_id,
            _IntegrityWriteModule,
            ModuleConfig(name=prepared.module_id),
        )
    elif drift == "remapped":
        remapped_id = f"prepared_remap_{uuid4().hex}"
        await registry.register_module(
            remapped_id,
            _IntegrityWriteModule,
            ModuleConfig(
                name=remapped_id,
                settings={"tool_name": prepared.tool_name},
            ),
        )
    elif drift == "module_id_drift":
        class _ModuleIdDriftRegistry:
            async def find_module_for_tool(self, tool_name: str) -> BaseModule | None:
                return module if tool_name == prepared.tool_name else None

            def get_module_id_for_tool(self, tool_name: str) -> str | None:
                return "different-module-id" if tool_name == prepared.tool_name else None

        protocol.module_registry = _ModuleIdDriftRegistry()
    elif drift == "renamed":
        module.source_tool_def["name"] = f"{prepared.tool_name}.renamed"
        module.invalidate_capability_caches()
    elif drift == "definition_changed":
        module.source_tool_def["description"] = "mutated after preparation"
        module.invalidate_capability_caches()
    else:
        class _RegistryResolutionError(Exception):
            pass

        class _FailingRegistry:
            async def find_module_for_tool(self, _tool_name: str) -> BaseModule | None:
                raise _RegistryResolutionError("private registry failure detail")

            def get_module_id_for_tool(self, _tool_name: str) -> str | None:
                raise AssertionError("module id lookup must not follow failed resolution")

        protocol.module_registry = _FailingRegistry()

    rate_limiter = _RateLimiterProbe()
    idempotency = _IdempotencyProbe()
    protocol.rate_limiter = rate_limiter
    protocol._idempotency = idempotency

    payload = await protocol.execute_prepared_tool_call(prepared)

    _assert_stale_prepared_call(payload)
    assert "private registry failure detail" not in json.dumps(payload)
    assert rate_limiter.calls == 0
    assert idempotency.bound_keys == []
    assert idempotency.run_keys == []
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0


@pytest.mark.parametrize(
    "status",
    [
        ModuleStatus.PENDING,
        ModuleStatus.INITIALIZING,
        ModuleStatus.INACTIVE,
        ModuleStatus.ERROR,
    ],
)
@pytest.mark.asyncio
async def test_live_check_requires_actual_registry_registration_to_be_operational(
    monkeypatch: pytest.MonkeyPatch,
    status: ModuleStatus,
) -> None:
    protocol, module, prepared, _ = await _prepare_call(monkeypatch)
    assert prepared.module_id is not None
    registry = get_module_registry()
    registry._modules[prepared.module_id].status = status

    assert await registry.find_module_for_tool(prepared.tool_name) is None

    rate_limiter = _RateLimiterProbe()
    idempotency = _IdempotencyProbe()
    protocol.rate_limiter = rate_limiter
    protocol._idempotency = idempotency

    payload = await protocol.execute_prepared_tool_call(prepared)

    _assert_stale_prepared_call(payload)
    assert rate_limiter.calls == 0
    assert idempotency.bound_keys == []
    assert idempotency.run_keys == []
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0


@pytest.mark.parametrize("lookup_path", ["cached", "fallback"])
@pytest.mark.parametrize("drift", ["status", "instance"])
@pytest.mark.asyncio
async def test_actual_registry_revalidates_operational_binding_after_has_tool(
    monkeypatch: pytest.MonkeyPatch,
    lookup_path: str,
    drift: str,
) -> None:
    _, module, prepared, _ = await _prepare_call(monkeypatch)
    assert prepared.module_id is not None
    registry = get_module_registry()
    if lookup_path == "fallback":
        registry._tool_registry.pop(prepared.tool_name, None)
    module.block_has_tool_on_call = module.has_tool_calls + 1

    lookup = asyncio.create_task(registry.find_module_for_tool(prepared.tool_name))
    await asyncio.wait_for(module.has_tool_entered.wait(), timeout=2)
    registration = registry._modules[prepared.module_id]
    if drift == "status":
        registration.status = ModuleStatus.INACTIVE
    else:
        registration.module_instance = _IntegrityWriteModule(
            ModuleConfig(
                name=prepared.module_id,
                settings={"tool_name": prepared.tool_name},
            ),
        )
    module.has_tool_release.set()

    assert await asyncio.wait_for(lookup, timeout=2) is None
    assert registry.get_module_id_for_tool(prepared.tool_name) != prepared.module_id


@pytest.mark.parametrize("tamper_target", ["arguments", "context"])
@pytest.mark.asyncio
async def test_second_live_check_rechecks_integrity_after_awaited_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    context = RequestContext(
        request_id=f"second-check-race-{tamper_target}",
        user_id="user-1",
        metadata={"workspace": "alpha"},
    )
    protocol, module, prepared, _ = await _prepare_call(
        monkeypatch,
        context=context,
        idempotency_key=None,
    )
    registry = _BlockingSecondLookupRegistry(get_module_registry())
    rate_limiter = _RateLimiterProbe()
    registry.block_when = lambda: rate_limiter.calls == 1
    protocol.module_registry = registry
    protocol.rate_limiter = rate_limiter

    execution = asyncio.create_task(protocol.execute_prepared_tool_call(prepared))
    await asyncio.wait_for(registry.entered.wait(), timeout=2)
    if tamper_target == "arguments":
        prepared.tool_args["value"] = "tampered"
    else:
        prepared.context.metadata["workspace"] = "tampered"
    registry.release.set()

    with pytest.raises(
        InvalidParamsException,
        match="Prepared tool call integrity check failed",
    ):
        await asyncio.wait_for(execution, timeout=2)

    assert registry.calls == 4
    assert rate_limiter.calls == 1
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0


@pytest.mark.parametrize("drift", ["status", "mapping", "enabled"])
@pytest.mark.asyncio
async def test_second_live_check_rechecks_binding_after_definition_await(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    protocol, module, prepared, _ = await _prepare_call(
        monkeypatch,
        idempotency_key=None,
    )
    assert prepared.module_id is not None
    registry = get_module_registry()
    rate_limiter = _RateLimiterProbe()
    protocol.rate_limiter = rate_limiter
    module.block_tool_def_on_call = module.tool_def_calls + 2

    execution = asyncio.create_task(protocol.execute_prepared_tool_call(prepared))
    await asyncio.wait_for(module.tool_def_entered.wait(), timeout=2)
    if drift == "status":
        registry._modules[prepared.module_id].status = ModuleStatus.INACTIVE
    elif drift == "mapping":
        remapped_id = f"prepared_race_remap_{uuid4().hex}"
        await registry.register_module(
            remapped_id,
            _IntegrityWriteModule,
            ModuleConfig(
                name=remapped_id,
                settings={"tool_name": prepared.tool_name},
            ),
        )
    else:
        module.config.enabled = False
    module.tool_def_release.set()
    payload = await asyncio.wait_for(execution, timeout=2)

    _assert_stale_prepared_call(payload)
    assert rate_limiter.calls == 1
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0


@pytest.mark.asyncio
async def test_second_live_check_blocks_definition_mutation_during_rate_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared, _ = await _prepare_call(
        monkeypatch,
        idempotency_key=None,
    )
    rate_limiter = _BlockingRateLimiter()
    protocol.rate_limiter = rate_limiter

    execution = asyncio.create_task(protocol.execute_prepared_tool_call(prepared))
    await asyncio.wait_for(rate_limiter.entered.wait(), timeout=2)
    module.source_tool_def["description"] = "changed during rate admission"
    module.invalidate_capability_caches()
    rate_limiter.release.set()
    payload = await asyncio.wait_for(execution, timeout=2)

    _assert_stale_prepared_call(payload)
    assert rate_limiter.calls == 1
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0


@pytest.mark.asyncio
async def test_second_live_check_blocks_definition_mutation_during_idempotency_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, module, prepared, _ = await _prepare_call(monkeypatch)
    rate_limiter = _RateLimiterProbe()
    idempotency = _BlockingIdempotency()
    protocol.rate_limiter = rate_limiter
    protocol._idempotency = idempotency

    execution = asyncio.create_task(protocol.execute_prepared_tool_call(prepared))
    await asyncio.wait_for(idempotency.entered.wait(), timeout=2)
    module.source_tool_def["description"] = "changed during idempotency contention"
    module.invalidate_capability_caches()
    idempotency.release.set()
    payload = await asyncio.wait_for(execution, timeout=2)

    _assert_stale_prepared_call(payload)
    assert rate_limiter.calls == 1
    assert len(idempotency.bound_keys) == 1
    assert idempotency.run_keys == idempotency.bound_keys
    assert module.breaker_entry_count == 0
    assert module.execute_count == 0
