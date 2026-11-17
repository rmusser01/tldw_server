"""Lazy, fail-closed host composition for bounded model completion."""

import asyncio
import copy
import importlib
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionPortSettings,
    ModelCompletionRequest,
    ModelFailureDomain,
    ModelInvocationIdentity,
)


def api():
    return importlib.import_module("tldw_Server_API.app.core.MCP_unified.adapters.model_completion.factory")


@pytest.fixture(autouse=True)
def isolated_operator_environment(monkeypatch):
    monkeypatch.setenv("RG_BACKEND", "memory")
    monkeypatch.delenv("HTTP_CERT_PINS", raising=False)


def settings(**changes):
    return replace(ModelCompletionPortSettings("openai", "fixed", 7, 2), **changes)


def operator_policy():
    return {
        "accounting": {
            "input_cost_units_per_token": 2,
            "output_cost_units_per_token": 3,
            "monthly_token_limit": 1000000,
            "monthly_cost_limit": 10000000,
            "governor_policy_id": "mcp.completion",
            "input_token_overhead": 32,
            "governor_tokens_per_cost_unit": 1000,
        },
        "governor": {
            "requests": {"rpm": 60, "burst": 1},
            "tokens": {"per_min": 500000, "burst": 1},
            "jobs": {"max_concurrent": 2, "ttl_sec": 135},
            "scopes": ["user"],
            "fail_mode": "fail_closed",
        },
        "output_token_field": "max_completion_tokens",
    }


def config():
    return {"openai_api": {"api_key": "private-key", "api_base_url": "https://fixed.example/v1"}}


def build(**kwargs):
    kwargs.setdefault("operator_policy_snapshot", operator_policy())
    kwargs.setdefault("server_config_snapshot", config())
    return api().build_tldw_model_completion_port(settings(), **kwargs)


def test_factory_builds_ready_native_port_without_provider_or_database_io(monkeypatch):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.AuthNZ import database

    module = api()

    def forbidden(*args, **kwargs):
        raise AssertionError("construction must not perform I/O")

    monkeypatch.setattr(database, "get_db_pool", forbidden)
    monkeypatch.setattr(module, "get_db_pool", forbidden)
    monkeypatch.setattr(http_client, "create_async_client", forbidden)
    monkeypatch.setenv("TLDW_HTTP_CERT_PINS", "")
    port = build()
    assert port.is_healthy() is True
    assert all(getattr(port.capabilities, name) is True for name in port.capabilities.__slots__)


def test_factory_io_guard_does_not_leak_across_first_import():
    probe = """
from pytest import MonkeyPatch
from tldw_Server_API.app.core.AuthNZ import database
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_factory import (
    api, test_factory_builds_ready_native_port_without_provider_or_database_io,
)
original = database.get_db_pool
with MonkeyPatch.context() as patch:
    patch.setenv('RG_BACKEND', 'memory')
    patch.delenv('HTTP_CERT_PINS', raising=False)
    test_factory_builds_ready_native_port_without_provider_or_database_io(patch)
assert database.get_db_pool is original
assert api().get_db_pool is original, 'factory retained the test-only database stub'
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("provider", ["OpenAI", "openai ", "custom-openai-api", "anthropic", "auto"])
async def test_unsupported_provider_returns_terminal_unavailable_port(provider):
    port = api().build_tldw_model_completion_port(settings(provider=provider))
    assert port.is_healthy() is False
    assert port.capabilities.native_async_cancellation is False
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(
            ModelCompletionRequest("sys", "user", 1, 10, 10, 100),
            ModelInvocationIdentity(7, None, None, "3f213c23-8e27-4fcb-86f8-c14c6a11e0d3"),
        )
    assert caught.value.code == "model_provider_unsupported"
    assert caught.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
    assert caught.value.__context__ is None
    await port.shutdown()
    await port.wait_for_shutdown_completion()
    assert port.is_healthy() is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_timeout_seconds", 121),
        ("cancellation_cleanup_seconds", 16),
        ("model", " fixed"),
        ("model", "private\nmodel"),
    ],
)
def test_invalid_settings_fail_closed(field, value):
    port = api().build_tldw_model_completion_port(
        settings(**{field: value}), operator_policy_snapshot=operator_policy(), server_config_snapshot=config()
    )
    assert port.is_healthy() is False


@pytest.mark.parametrize("missing", ["accounting", "governor", "output_token_field"])
def test_missing_policy_blocks_construction(missing):
    policy = operator_policy()
    del policy[missing]
    assert build(operator_policy_snapshot=policy).is_healthy() is False


@pytest.mark.parametrize("field", list(operator_policy()["accounting"]))
def test_every_accounting_value_is_explicit(field):
    policy = operator_policy()
    del policy["accounting"][field]
    assert build(operator_policy_snapshot=policy).is_healthy() is False


@pytest.mark.parametrize(
    "field",
    [
        "input_cost_units_per_token",
        "output_cost_units_per_token",
        "monthly_token_limit",
        "monthly_cost_limit",
        "input_token_overhead",
        "governor_tokens_per_cost_unit",
    ],
)
@pytest.mark.parametrize("value", [True, "2", 1.5, -1, 2**63])
def test_accounting_values_reject_coercion_and_overflow(field, value):
    policy = operator_policy()
    policy["accounting"][field] = value
    assert build(operator_policy_snapshot=policy).is_healthy() is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(governor={}),
        lambda p: p["governor"].update(fail_mode="fallback_memory"),
        lambda p: p["governor"].update(scopes=[]),
        lambda p: p["governor"].update(scopes=["global"]),
        lambda p: p["governor"].update(scopes=["user", "ip"]),
        lambda p: p["governor"].update(requests={}),
        lambda p: p["governor"]["requests"].update(rpm=True),
        lambda p: p["governor"]["tokens"].update(per_min=0),
        lambda p: p["governor"]["tokens"].update(per_min=1),
        lambda p: p["governor"]["jobs"].update(max_concurrent=0),
        lambda p: p["governor"]["jobs"].update(ttl_sec=8),
        lambda p: p["governor"].update(unknown="private-value"),
        lambda p: p["accounting"].update(governor_policy_id="default"),
        lambda p: p.update(output_token_field="tools"),
    ],
)
def test_incomplete_or_fallback_governor_policy_is_unavailable(mutation):
    policy = operator_policy()
    mutation(policy)
    assert build(operator_policy_snapshot=policy).is_healthy() is False


def test_captures_independent_operator_and_server_snapshots():
    policy, server = operator_policy(), config()
    captured = []

    def transport_factory(value):
        from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import OpenAICompletionTransport

        captured.append(value)
        return OpenAICompletionTransport(value)

    port = build(operator_policy_snapshot=policy, server_config_snapshot=server, transport_factory=transport_factory)
    policy["accounting"]["input_cost_units_per_token"] = 999
    policy["governor"]["jobs"]["max_concurrent"] = 999
    server["openai_api"]["api_base_url"] = "https://other.example/v1"
    assert port.is_healthy() is True
    assert captured[0].endpoint.base_url == "https://fixed.example/v1/"
    assert port._accounting.policy.input_cost_units_per_token == 2
    assert port._config["openai_api"]["api_base_url"] == "https://fixed.example/v1"
    assert port._accounting._governor._get_policy("mcp.completion")["jobs"]["max_concurrent"] == 2


def test_static_governor_never_uses_unknown_or_default_policy():
    governor = build()._accounting._governor
    with pytest.raises(ValueError, match="completion governor policy"):
        governor._get_policy("default")
    snapshot = governor._get_policy("mcp.completion")
    snapshot["requests"]["rpm"] = 100000
    assert governor._get_policy("mcp.completion")["requests"]["rpm"] == 60


def test_unsupported_governor_cost_rate_is_rejected_not_silently_ignored():
    policy = operator_policy()
    policy["governor"]["cost_units"] = {"per_min": 410, "burst": 1}
    assert build(operator_policy_snapshot=policy).is_healthy() is False


def test_default_configuration_is_read_once_on_factory_call(monkeypatch):
    module = api()
    calls = []
    policy = operator_policy()
    section = {"policy_json": json.dumps(policy)}
    monkeypatch.setattr(module, "get_config_section", lambda name: calls.append(name) or section)
    monkeypatch.setattr(module, "load_server_config_snapshot", lambda: calls.append("server") or config())
    port = module.build_tldw_model_completion_port(settings())
    assert calls == ["MCP-Model-Completion", "server"]
    assert port.is_healthy() is True
    section["policy_json"] = "broken"
    assert port.is_healthy() is True
    assert calls == ["MCP-Model-Completion", "server"]


@pytest.mark.parametrize("text", ["", "{}", "[]", "null", '{"accounting":{},"accounting":{}}', "x" * 65537])
def test_bad_operator_json_is_unavailable_without_leak(monkeypatch, text):
    monkeypatch.setattr(api(), "get_config_section", lambda name: {"policy_json": text})
    port = api().build_tldw_model_completion_port(settings(), server_config_snapshot=config())
    assert port.is_healthy() is False
    assert text not in repr(port) or not text


def test_configuration_failure_is_detached_and_does_not_log_private_values(monkeypatch, caplog):
    def fail(*args):
        raise RuntimeError("private-key https://fixed.example/?secret=prompt")

    monkeypatch.setattr(api(), "get_config_section", fail)
    port = api().build_tldw_model_completion_port(settings())
    assert port.is_healthy() is False
    assert "private-key" not in repr(port) + caplog.text


@pytest.mark.parametrize("base", [None, "https://fixed.example/v1?private-key", "https://a:b@fixed.example/v1"])
def test_missing_or_unsafe_configured_endpoint_is_unavailable(base):
    server = config()
    server["openai_api"]["api_base_url"] = base
    assert build(server_config_snapshot=server).is_healthy() is False


def test_uncertified_transport_stays_unavailable():
    from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import OpenAICompletionTransport

    def transport_factory(policy):
        transport = OpenAICompletionTransport(policy)
        transport._capabilities = replace(transport.capabilities, native_async_cancellation=False)
        return transport

    assert build(transport_factory=transport_factory).is_healthy() is False


def test_false_valued_transport_factory_does_not_fall_back():
    from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import OpenAICompletionTransport

    class TransportFactory:
        def __bool__(self):
            return False

        def __call__(self, policy):
            transport = OpenAICompletionTransport(policy)
            transport._capabilities = replace(transport.capabilities, native_async_cancellation=False)
            return transport

    assert build(transport_factory=TransportFactory()).is_healthy() is False


def test_injected_services_are_used_without_production_lookup():
    services = [SimpleNamespace() for _ in range(4)]
    port = build(reservations=services[0], usage=services[1], billing=services[2], governor=services[3])
    assert port._accounting._reservations is services[0]
    assert port._accounting._usage is services[1]
    assert port._accounting._billing is services[2]
    assert port._accounting._governor is services[3]


def test_nullable_explicit_monthly_limits_and_zero_prices_are_valid():
    policy = copy.deepcopy(operator_policy())
    policy["accounting"].update(
        monthly_token_limit=None, monthly_cost_limit=None, input_cost_units_per_token=0, output_cost_units_per_token=0
    )
    assert build(operator_policy_snapshot=policy).is_healthy() is True


@pytest.mark.parametrize("backend", ["redis", "invalid", "Memory", ""])
def test_uncertified_production_governor_backend_never_falls_back(monkeypatch, backend):
    monkeypatch.setenv("RG_BACKEND", backend)
    assert build().is_healthy() is False


async def test_lazy_pool_keeps_concurrent_transaction_backend_and_cleanup_isolated(monkeypatch):
    module = api()
    pool = module._LazyAuthNZPool()
    first_entered, second_entered, first_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    exits = []

    class Pool:
        def __init__(self, marker):
            self.pool = marker

        @asynccontextmanager
        async def transaction(self):
            try:
                yield self.pool
            finally:
                exits.append(self.pool)

    pools = [Pool("postgres"), Pool(None)]

    async def get_pool():
        return pools.pop(0)

    monkeypatch.setattr(module, "get_db_pool", get_pool)

    async def first():
        async with pool.transaction() as conn:
            assert conn == "postgres"
            first_entered.set()
            await second_entered.wait()
            assert pool.pool == "postgres"
        with pytest.raises(RuntimeError, match="active transaction"):
            _ = pool.pool
        first_done.set()

    async def second():
        await first_entered.wait()
        async with pool.transaction() as conn:
            assert conn is None
            second_entered.set()
            await first_done.wait()
            assert pool.pool is None

    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert exits == ["postgres", None]
    with pytest.raises(RuntimeError, match="active transaction"):
        _ = pool.pool


async def test_lazy_pool_cancellation_preserves_transaction_cleanup_and_context(monkeypatch):
    module = api()
    proxy = module._LazyAuthNZPool()
    entered = asyncio.Event()
    exited = []

    class Pool:
        pool = "postgres"

        @asynccontextmanager
        async def transaction(self):
            try:
                yield "connection"
            finally:
                exited.append(True)

    async def get_pool():
        return Pool()

    monkeypatch.setattr(module, "get_db_pool", get_pool)

    async def work():
        try:
            async with proxy.transaction():
                entered.set()
                await asyncio.Event().wait()
        finally:
            with pytest.raises(RuntimeError, match="active transaction"):
                _ = proxy.pool

    task = asyncio.create_task(work())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel("caller")
    with pytest.raises(asyncio.CancelledError, match="caller"):
        await task
    assert exited == [True]
