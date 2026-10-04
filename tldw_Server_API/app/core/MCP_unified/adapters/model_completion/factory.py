"""Lazy host composition of a certified, explicitly governed completion port."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from tldw_Server_API.app.core.AuthNZ.byok_helpers import load_server_config_snapshot
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import _trusted_endpoint_from_snapshot
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import ProviderUsageReservationsRepo
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Billing.enforcement import get_billing_enforcer
from tldw_Server_API.app.core.config import get_config_section
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.accounting import (
    CompletionAccountingPolicy,
    ModelCompletionAccounting,
)
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.adapter import ModelCompletionAdapter
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import (
    OpenAICompletionTransport,
    OpenAITransportPolicy,
)
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ManagedModelCompletionPort,
    ModelCompletionCapabilities,
    ModelCompletionFailure,
    ModelCompletionPortSettings,
    ModelCompletionRequest,
    ModelCompletionResult,
    ModelFailureDomain,
    ModelInvocationIdentity,
)
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, ResourceGovernor

_UNCERTIFIED = ModelCompletionCapabilities(False, False, False, False, False)
_POLICY_ID = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_MAX_GOVERNOR_INTEGER = 2**53 - 1
_ACCOUNTING_FIELDS = frozenset(
    {
        "input_cost_units_per_token",
        "output_cost_units_per_token",
        "monthly_token_limit",
        "monthly_cost_limit",
        "governor_policy_id",
        "input_token_overhead",
        "governor_tokens_per_cost_unit",
    }
)


class _UnavailableCompletionPort:
    """Terminal, content-free result of unsupported or invalid host composition."""

    __slots__ = ("_code",)

    def __init__(self, code: str) -> None:
        self._code = code

    @property
    def capabilities(self) -> ModelCompletionCapabilities:
        return _UNCERTIFIED

    def is_healthy(self) -> bool:
        return False

    async def complete(
        self, request: ModelCompletionRequest, identity: ModelInvocationIdentity
    ) -> ModelCompletionResult:
        raise_detached_error(ModelCompletionFailure(self._code, ModelFailureDomain.SHARED_INFRASTRUCTURE))

    async def shutdown(self) -> None:
        """No work was admitted by this unavailable port."""

    async def wait_for_shutdown_completion(self) -> None:
        """There is no owned work to drain."""


class _LazyAuthNZPool:
    """Resolve the canonical pool at admission, preserving each transaction's backend."""

    def __init__(self) -> None:
        self._active: ContextVar[DatabasePool | None] = ContextVar("mcp_completion_pool", default=None)

    @property
    def pool(self) -> Any:
        active = self._active.get()
        if active is None:
            raise RuntimeError("Completion accounting requires an active transaction")
        return active.pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        active = await get_db_pool()
        token = self._active.set(active)
        try:
            async with active.transaction() as connection:
                yield connection
        finally:
            self._active.reset(token)


class _FrozenCompletionGovernor(MemoryResourceGovernor):
    """Use one detached explicit policy, never ingress safety-net substitution."""

    def __init__(self, policy_id: str, policy: dict[str, Any], clock: Callable[[], float]) -> None:
        self._completion_policy_id = policy_id
        self._completion_policy = copy.deepcopy(policy)
        super().__init__(time_source=clock, default_handle_ttl=policy["jobs"]["ttl_sec"])

    def _get_policy(self, policy_id: str) -> dict[str, Any]:
        if policy_id != self._completion_policy_id:
            raise ValueError("Invalid completion governor policy")
        return copy.deepcopy(self._completion_policy)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate completion policy field")
        result[key] = value
    return result


def _load_operator_policy() -> dict[str, Any]:
    section = get_config_section("MCP-Model-Completion")
    text = section.get("policy_json")
    if type(text) is not str or not 1 <= len(text.encode("utf-8")) <= 65536:
        raise ValueError("Explicit completion policy is required")
    policy = json.loads(text, object_pairs_hook=_unique_json_object)
    if type(policy) is not dict:
        raise ValueError("Completion policy must be an object")
    return policy


def _positive_governor_integer(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_GOVERNOR_INTEGER:
        raise ValueError("Invalid completion governor integer")
    return value


def _governor_snapshot(raw: Any, accounting: CompletionAccountingPolicy, settings: ModelCompletionPortSettings) -> dict:
    if type(raw) is not dict or set(raw) != {"requests", "tokens", "jobs", "scopes", "fail_mode"}:
        raise ValueError("Incomplete completion governor policy")
    policy = copy.deepcopy(raw)
    if (
        policy["fail_mode"] != "fail_closed"
        or type(policy["scopes"]) is not list
        or not policy["scopes"]
        or any(type(scope) is not str or scope not in {"user", "global"} for scope in policy["scopes"])
        or "user" not in policy["scopes"]
        or len(set(policy["scopes"])) != len(policy["scopes"])
        or type(accounting.governor_policy_id) is not str
        or _POLICY_ID.fullmatch(accounting.governor_policy_id) is None
        or accounting.governor_policy_id == "default"
    ):
        raise ValueError("Invalid completion governor scope")
    capacities = {}
    for category, rate in (("requests", "rpm"), ("tokens", "per_min")):
        block = policy[category]
        if type(block) is not dict or set(block) != {rate, "burst"}:
            raise ValueError("Incomplete completion governor bucket")
        capacities[category] = _positive_governor_integer(block[rate]) * _positive_governor_integer(block["burst"])
        _positive_governor_integer(capacities[category])
    jobs = policy["jobs"]
    if type(jobs) is not dict or set(jobs) != {"max_concurrent", "ttl_sec"}:
        raise ValueError("Incomplete completion governor concurrency")
    _positive_governor_integer(jobs["max_concurrent"])
    ttl = _positive_governor_integer(jobs["ttl_sec"])
    if not settings.run_timeout_seconds + settings.cancellation_cleanup_seconds <= ttl <= 3600:
        raise ValueError("Invalid completion governor lease lifetime")
    # The shared governor clamps oversized token reservations. A certified
    # snapshot must instead be able to hold every hard-bounded request in full.
    max_tokens = 400000 + accounting.input_token_overhead + 8192
    if capacities["tokens"] < max_tokens:
        raise ValueError("Completion governor cannot cover bounded exposure")
    return policy


def build_tldw_model_completion_port(
    settings: ModelCompletionPortSettings,
    *,
    operator_policy_snapshot: Mapping[str, Any] | None = None,
    server_config_snapshot: Mapping[str, Any] | None = None,
    reservations: Any = None,
    usage: Any = None,
    billing: Any = None,
    governor: ResourceGovernor | None = None,
    clock: Callable[[], float] = time.monotonic,
    transport_factory: Callable[[OpenAITransportPolicy], OpenAICompletionTransport] | None = None,
) -> ManagedModelCompletionPort:
    """Build on demand without DB/network I/O; invalid configuration stays unavailable.

    Trusted host injection seams are not request parameters. The production
    backend is a dedicated in-process governor with a frozen, non-fallback
    policy; durable scope quotas remain canonical across processes. AuthNZ DB
    initialization is deferred until accounted admission on the native loop.
    """
    code = "model_factory_policy_invalid"
    try:
        if type(settings) is not ModelCompletionPortSettings:
            raise ValueError("Invalid completion settings")
        if type(settings.provider) is not str or settings.provider != "openai":
            return _UnavailableCompletionPort("model_provider_unsupported")
        captured = ModelCompletionPortSettings(
            settings.provider, settings.model, settings.run_timeout_seconds, settings.cancellation_cleanup_seconds
        )
        if not (1 <= captured.run_timeout_seconds <= 120 and 1 <= captured.cancellation_cleanup_seconds <= 15):
            raise ValueError("Invalid completion time limits")
        policy = (
            copy.deepcopy(dict(operator_policy_snapshot))
            if operator_policy_snapshot is not None
            else _load_operator_policy()
        )
        if set(policy) != {"accounting", "governor", "output_token_field"}:
            raise ValueError("Incomplete completion policy")
        values = policy["accounting"]
        if type(values) is not dict or set(values) != _ACCOUNTING_FIELDS:
            raise ValueError("Incomplete completion accounting")
        accounting_policy = CompletionAccountingPolicy(provider=captured.provider, model=captured.model, **values)
        governor_policy = _governor_snapshot(policy["governor"], accounting_policy, captured)
        if governor is None and os.getenv("RG_BACKEND", "memory") != "memory":
            raise ValueError("Uncertified completion governor backend")
        config = (
            copy.deepcopy(dict(server_config_snapshot))
            if server_config_snapshot is not None
            else copy.deepcopy(load_server_config_snapshot())
        )
        endpoint = _trusted_endpoint_from_snapshot("openai", config, authoritative=True)
        transport_policy = OpenAITransportPolicy(
            captured.provider, captured.model, endpoint, captured.run_timeout_seconds, policy["output_token_field"]
        )
        if not callable(clock) or not math.isfinite(clock()):
            raise ValueError("Invalid completion clock")
        if transport_factory is not None and not callable(transport_factory):
            raise ValueError("Invalid completion transport factory")
        transport = (OpenAICompletionTransport if transport_factory is None else transport_factory)(transport_policy)
        pool = _LazyAuthNZPool()
        accounting = ModelCompletionAccounting(
            policy=accounting_policy,
            reservations=reservations if reservations is not None else ProviderUsageReservationsRepo(pool),
            usage=usage if usage is not None else AuthnzUsageRepo(pool),
            billing=billing if billing is not None else get_billing_enforcer(),
            governor=(
                governor
                if governor is not None
                else _FrozenCompletionGovernor(accounting_policy.governor_policy_id, governor_policy, clock)
            ),
        )
        return ModelCompletionAdapter(captured, config, accounting, transport)
    except Exception:  # noqa: BLE001 - fail closed without logging or retaining private operator state
        return _UnavailableCompletionPort(code)
