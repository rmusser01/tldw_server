"""Adversarial host composition on migrated AuthNZ storage and native HTTPX."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from loguru import logger

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.AuthNZ import byok_runtime, user_provider_secrets
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import fetch_all
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import (
    BillingScope,
    ProviderUsageReservationsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.Billing import enforcement
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion import factory, normalization
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import OpenAICompletionTransport
from tldw_Server_API.app.core.MCP_unified.auth_scope import project_authenticated_execution_scope
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
    ModelInvocationIdentity,
)
from tldw_Server_API.app.core.MCP_unified.protocol_types import RequestContext
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_factory import operator_policy, settings
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_transport import Body
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_transport import (
    asyncio_diagnostics as asyncio_diagnostics,
)
from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_runtime_sqlite import _upsert_shared_key, _upsert_user_key
from tldw_Server_API.tests.AuthNZ_SQLite.test_provider_scope_resolution_sqlite import (
    _execute,
    _UnavailablePool,
    seed_scope_state,
)
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_pool as reservation_pool,
)
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_schema as reservation_schema,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
BASE = "https://fixed.example/v1"
URL = BASE + "/chat/completions"
KEYS = {source: f"sk-integration-{source}-sentinel" for source in ("user", "team", "org", "server")}
SYSTEM = "private-system-sentinel"
PROMPT = "private-prompt-sentinel"
CONTENT = " completion-sentinel\r\nline "
RAW = "raw-provider-body-sentinel"
OVERRIDE = "https://attacker.example/v1?private-query-sentinel"
PRIVATE = " ".join((*KEYS.values(), SYSTEM, PROMPT, CONTENT, RAW, OVERRIDE))


def request(**changes):
    """Keep provider usage below the genuine worst-case reservation."""
    return replace(ModelCompletionRequest(SYSTEM, PROMPT, 32, 100, 400, 2048), **changes)


def envelope(content=CONTENT):
    return {
        "id": "provider-private-id",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "private": PRIVATE},
        "raw": PRIVATE,
    }


def identity_from_context(context):
    """Model the host handoff without admitting any caller metadata to the port."""
    scope = context.server_auth_scope
    return ModelInvocationIdentity(
        int(context.user_id),
        scope.active_team_id if scope else None,
        scope.active_org_id if scope else None,
        str(uuid4()),
    )


async def usage_rows(pool, execution_id):
    async with pool.acquire() as conn:
        return await fetch_all(
            conn,
            pool.pool is not None,
            "SELECT * FROM llm_usage_log WHERE request_id = ? AND operation = 'mcp_model_completion'",
            (execution_id,),
            (),
        )


def assert_private_absent(value):
    text = repr(value)
    for sentinel in (
        *KEYS.values(),
        "sk-unrelated-sentinel",
        "caller-key",
        "provider-private-id",
        SYSTEM,
        PROMPT,
        "completion-sentinel",
        RAW,
        OVERRIDE,
        "private-query-sentinel",
    ):
        assert sentinel not in text


def assert_failure(error, code, domain):
    assert type(error) is ModelCompletionFailure
    assert (error.code, error.domain, str(error), error.args) == (code, domain, code, (code,))
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__ is True
    assert_private_absent((repr(error), vars(error)))


class FakeProvider:
    """Only the external provider is fake; observe committed admission at HTTP entry."""

    def __init__(self, pool):
        self.pool = pool
        self.execution_id = None
        self.requests = []
        self.dispatched_rows = []
        self.clients = []
        self.client_class = httpx.AsyncClient
        self.bodies = []
        self.payload = envelope()
        self.status = 200
        self.headers = {"content-type": "application/json"}
        self.error = None
        self.before_response = None
        self.body_factory = None

    async def handle(self, incoming):
        self.requests.append(incoming)
        row = await ProviderUsageReservationsRepo(self.pool).get(self.execution_id)
        self.dispatched_rows.append(row)
        assert row["state"] == "dispatched"
        if self.before_response is not None:
            await self.before_response()
        if self.error is not None:
            raise self.error
        body = self.body_factory() if self.body_factory else Body([json.dumps(self.payload).encode()])
        self.bodies.append(body)
        return httpx.Response(self.status, headers=self.headers, stream=body, request=incoming)

    def transport(self, policy):
        def client_factory(**kwargs):
            client = self.client_class(transport=httpx.MockTransport(self.handle), **kwargs)
            self.clients.append(client)
            return client

        return OpenAICompletionTransport(policy, client_factory=client_factory)


@pytest_asyncio.fixture
async def composed(reservation_pool, monkeypatch, caplog, asyncio_diagnostics):
    """Default lazy repos, real encrypted credentials, billing and frozen governor."""
    state = await seed_scope_state(reservation_pool)
    pool = reservation_pool
    logs, validations, ports = [], [], []
    caplog.set_level(logging.DEBUG, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="httpcore")
    sink = logger.add(
        lambda message: logs.append((str(message), repr(message.record["extra"]), repr(message.record["exception"])))
    )
    monkeypatch.setenv("RG_BACKEND", "memory")
    monkeypatch.setenv("BYOK_LAST_USED_THROTTLE_SECONDS", "0")
    monkeypatch.delenv("HTTP_CERT_PINS", raising=False)
    monkeypatch.setattr(
        user_provider_secrets,
        "get_settings",
        lambda: SimpleNamespace(
            BYOK_ENCRYPTION_KEY=base64.b64encode(b"k" * 32).decode(), BYOK_SECONDARY_ENCRYPTION_KEY=None
        ),
    )
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda provider: True)

    async def db_pool():
        return pool

    async def local_operator_quota():
        return False

    async def validate(url, **kwargs):
        validations.append((url, kwargs))

    monkeypatch.setattr(byok_runtime, "get_db_pool", db_pool)
    monkeypatch.setattr(factory, "get_db_pool", db_pool)
    monkeypatch.setattr(enforcement, "billing_checks_active", local_operator_quota)
    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    provider = FakeProvider(pool)

    def build(*, policy=None, port_settings=None):
        port = factory.build_tldw_model_completion_port(
            settings() if port_settings is None else port_settings,
            server_config_snapshot={"openai_api": {"api_key": KEYS["server"], "api_base_url": BASE}},
            operator_policy_snapshot=policy if policy is not None else operator_policy(),
            transport_factory=provider.transport,
        )
        ports.append(port)
        return port

    def context(metadata=None):
        scope = project_authenticated_execution_scope(
            authenticated_user_id=str(state["user"]),
            principal_user_id=state["user"],
            principal_active_org_id=state["org"],
            principal_active_team_id=state["team"],
        )
        return RequestContext("mcp-request", user_id=str(state["user"]), server_auth_scope=scope, metadata=metadata)

    env = SimpleNamespace(
        state=state,
        pool=pool,
        provider=provider,
        build=build,
        context=context,
        logs=logs,
        validations=validations,
        caplog=caplog,
        diagnostics=asyncio_diagnostics,
    )
    try:
        yield env
    finally:
        try:
            for port in ports:
                await port.shutdown()
                await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
            assert asyncio_diagnostics[0] == []
            assert_private_absent((logs, caplog.text, asyncio_diagnostics))
        finally:
            logger.remove(sink)


async def seed_keys(env, sources=("user", "team", "org"), *, credential_fields=None):
    user_repo = AuthnzUserProviderSecretsRepo(env.pool)
    shared_repo = AuthnzOrgProviderSecretsRepo(env.pool)
    for source in sources:
        if source == "user":
            await _upsert_user_key(user_repo, env.state["user"], "openai", KEYS[source], credential_fields)
        else:
            await _upsert_shared_key(shared_repo, source, env.state[source], "openai", KEYS[source], credential_fields)


async def assert_safe_surfaces(env, invocation, *, result=None):
    row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
    assert_private_absent(
        (
            row,
            env.provider.dispatched_rows,
            await usage_rows(env.pool, invocation.execution_id),
            env.logs,
            env.caplog.text,
            env.diagnostics,
        )
    )
    if result is not None:
        assert type(result).__slots__ == ("content",)
        assert set(asdict(result)) == {"content"}
        assert not hasattr(result, "raw_metadata")
    assert all(client.is_closed for client in env.provider.clients)
    assert all(body.closed for body in env.provider.bodies)
    return row


@pytest.mark.parametrize("selected", ["user", "team", "org", "server"])
async def test_authenticated_completion_uses_absence_only_precedence_and_frozen_endpoint(composed, selected):
    env = composed
    sources = ("user", "team", "org")
    await seed_keys(
        env,
        sources[sources.index(selected) :] if selected != "server" else (),
        credential_fields={"base_url": OVERRIDE},
    )
    # Unrelated keys and caller claims must never supply an active scope or endpoint.
    await _upsert_shared_key(
        AuthnzOrgProviderSecretsRepo(env.pool), "team", env.state["other_team"], "openai", "sk-unrelated-sentinel"
    )
    context = env.context(
        {
            "user_id": 1,
            "active_team_id": env.state["other_team"],
            "active_org_id": env.state["other_org"],
            "execution_id": "caller-execution",
            "server_auth_scope": {"active_team_id": env.state["other_team"]},
            "base_url": OVERRIDE,
            "api_key": "caller-key",
            "trusted_base_url_override": True,
        }
    )
    invocation = identity_from_context(context)
    env.provider.execution_id = invocation.execution_id
    port = env.build()
    assert port.is_healthy()
    result = await port.complete(request(), invocation)
    assert result.content == CONTENT.replace("\r\n", "\n")
    assert (invocation.user_id, invocation.active_team_id, invocation.active_organization_id) == (
        env.state["user"],
        env.state["team"],
        env.state["org"],
    )
    assert len(env.provider.requests) == 1
    incoming = env.provider.requests[0]
    assert str(incoming.url) == URL
    assert incoming.headers["authorization"] == f"Bearer {KEYS[selected]}"
    assert json.loads(incoming.content) == {
        "model": "fixed",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
        "stream": False,
        "n": 1,
        "tools": None,
        "max_completion_tokens": 32,
    }
    row = await assert_safe_surfaces(env, invocation, result=result)
    assert (row["state"], row["billing_scope_type"], row["billing_scope_id"], row["actual_cost_units"]) == (
        "reconciled",
        "org",
        env.state["org"],
        43,
    )
    assert row["reserved_input_tokens"] == len(SYSTEM.encode()) + len(PROMPT.encode()) + 32
    usage = await usage_rows(env.pool, invocation.execution_id)
    assert len(usage) == 1
    assert (
        usage[0]["user_id"],
        usage[0]["billing_org_id"],
        usage[0]["prompt_tokens"],
        usage[0]["completion_tokens"],
        usage[0]["estimated"],
    ) == (env.state["user"], env.state["org"], 17, 3, 0)
    if selected == "user":
        stored = await AuthnzUserProviderSecretsRepo(env.pool).fetch_secret_for_user(env.state["user"], "openai")
    elif selected in {"team", "org"}:
        stored = await AuthnzOrgProviderSecretsRepo(env.pool).fetch_secret(selected, env.state[selected], "openai")
    else:
        stored = None
    if stored is not None:
        assert stored["last_used_at"] is not None
        assert KEYS[selected] not in stored["encrypted_blob"]


class Gate:
    """Deterministic cancellation/read boundary with no wall-clock scheduling."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self):
        self.entered.set()
        await self.release.wait()


def gate_method(monkeypatch, owner, name, gate, *, after=False):
    """Wrap an actual operation, preserving its database and lifecycle behavior."""
    original = getattr(owner, name)

    async def wrapped(*args, **kwargs):
        if not after:
            await gate.wait()
        result = await original(*args, **kwargs)
        if after:
            await gate.wait()
        return result

    monkeypatch.setattr(owner, name, wrapped)


async def drain_call(port, call, gate):
    gate.release.set()
    if not call.done():
        call.cancel()
    await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), 5)
    await port.shutdown()
    await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)


def fail_client_close(provider, *, gate=None, after_close=True):
    """Inject only the native client teardown fault, keeping all provider I/O real."""
    original = httpx.AsyncClient.aclose
    tasks = []

    class FailingCloseClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            tasks.append(asyncio.current_task())
            if gate is not None:
                await gate.wait()
            if after_close:
                await super().aclose()
            raise RuntimeError(PRIVATE) from ValueError(PRIVATE)

    provider.client_class = FailingCloseClient
    return original, tasks


@pytest.mark.parametrize("after_close", [True, False])
async def test_paid_receipt_survives_client_cleanup_error_and_latches_unhealthy(composed, after_close):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    original_close, close_tasks = fail_client_close(env.provider, after_close=after_close)
    port = env.build()
    assert port.is_healthy()
    try:
        result = await port.complete(request(), invocation)
        assert result.content == CONTENT.replace("\r\n", "\n")
        assert len(env.provider.requests) == 1
        assert all(task.done() and not task._log_traceback for task in close_tasks)
        assert not port.is_healthy()
        assert port.capabilities.native_async_cancellation is False
        assert not port._owned
        assert not port._retained
        assert not port._accounting._governor._handles
        row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
        assert (row["state"], row["actual_input_tokens"], row["actual_output_tokens"], row["actual_cost_units"]) == (
            "reconciled",
            17,
            3,
            43,
        )
        usage = await usage_rows(env.pool, invocation.execution_id)
        assert len(usage) == 1
        assert (usage[0]["prompt_tokens"], usage[0]["completion_tokens"], usage[0]["estimated"]) == (17, 3, 0)
        stored = await AuthnzUserProviderSecretsRepo(env.pool).fetch_secret_for_user(env.state["user"], "openai")
        assert stored["last_used_at"] is not None
        assert await ProviderUsageReservationsRepo(env.pool).outstanding(BillingScope("org", env.state["org"])) == {
            "tokens": 0,
            "cost_units": 0,
        }
        for subsequent in (invocation, identity_from_context(env.context())):
            with pytest.raises(ModelCompletionFailure) as denied:
                await port.complete(request(), subsequent)
            assert_failure(denied.value, "model_adapter_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        assert len(env.provider.requests) == len(env.provider.clients) == 1
    finally:
        # An unhealthy port cannot certify a client whose native close failed.
        for client in env.provider.clients:
            await original_close(client)
    await assert_safe_surfaces(env, invocation, result=result)


@pytest.mark.parametrize(
    "outcome,code,domain",
    [
        ("connection", "model_transport_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
        ("rejected", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("invalid", "invalid_model_output", ModelFailureDomain.REQUEST),
    ],
)
async def test_client_cleanup_error_before_receipt_preserves_original_failure(composed, outcome, code, domain):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    _, close_tasks = fail_client_close(env.provider)
    if outcome == "connection":
        env.provider.error = httpx.ConnectError(PRIVATE)
    elif outcome == "rejected":
        env.provider.status = 429
    else:
        env.provider.payload = envelope(" \r\n ")
    port = env.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(request(), invocation)
    assert_failure(caught.value, code, domain)
    assert not port.is_healthy()
    assert port.capabilities.native_async_cancellation is False
    assert len(env.provider.requests) == 1
    assert all(task.done() and not task._log_traceback for task in close_tasks)
    row = await assert_safe_surfaces(env, invocation)
    assert row["state"] == "ambiguous"
    assert await usage_rows(env.pool, invocation.execution_id) == []
    stored = await AuthnzUserProviderSecretsRepo(env.pool).fetch_secret_for_user(env.state["user"], "openai")
    assert stored["last_used_at"] is None


@pytest.mark.parametrize("outcome", ["cancel", "shutdown"])
async def test_client_cleanup_error_cannot_publish_receipt_after_cancellation_or_shutdown(composed, outcome):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    gate = Gate()
    _, close_tasks = fail_client_close(env.provider, gate=gate)
    port = env.build()
    call = asyncio.create_task(port.complete(request(), invocation))
    drain = None
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        if outcome == "cancel":
            call.cancel("original-client-close-cancel")
            await asyncio.sleep(0)
            call.cancel("repeated-client-close-cancel")
        else:
            await port.shutdown()
        drain = asyncio.create_task(port.wait_for_shutdown_completion())
        await asyncio.sleep(0)
        assert not call.done()
        assert not drain.done()
        assert all(not task.done() for task in close_tasks)
        gate.release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(call, 5)
        if outcome == "cancel":
            assert caught.value.args == ("original-client-close-cancel",)
        assert PRIVATE not in str(caught.value)
        await asyncio.wait_for(drain, 5)
        assert not port.is_healthy()
        assert port.capabilities.native_async_cancellation is False
        assert all(task.done() and not task._log_traceback for task in close_tasks)
        assert len(env.provider.requests) == 1
        row = await assert_safe_surfaces(env, invocation)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
        assert not port._owned
        assert not port._retained
        assert not port._accounting._governor._handles
    finally:
        await drain_call(port, call, gate)
        if drain is not None:
            await asyncio.wait_for(drain, 5)


async def test_client_cleanup_error_after_allowance_remains_owned_and_cannot_recover_health(composed):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    gate = Gate()
    _, close_tasks = fail_client_close(env.provider, gate=gate)
    port = env.build()
    call = asyncio.create_task(port.complete(request(), invocation))
    drain = None
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        call.cancel("client-close-allowance-cancel")
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(call, 5)
        assert caught.value.args == ("client-close-allowance-cancel",)
        assert not port.is_healthy()
        assert port._retained
        assert all(not task.done() for task in close_tasks)
        drain = asyncio.create_task(port.wait_for_shutdown_completion())
        await asyncio.sleep(0)
        assert not drain.done()
        gate.release.set()
        await asyncio.wait_for(drain, 5)
        assert port.capabilities.native_async_cancellation is False
        assert not port.is_healthy()
        assert not port._owned
        assert not port._retained
        assert all(task.done() and not task._log_traceback for task in close_tasks)
        assert len(env.provider.requests) == 1
        row = await assert_safe_surfaces(env, invocation)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
        assert not port._accounting._governor._handles
        with pytest.raises(ModelCompletionFailure) as denied:
            await port.complete(request(), identity_from_context(env.context()))
        assert_failure(denied.value, "model_adapter_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    finally:
        await drain_call(port, call, gate)
        if drain is not None:
            await asyncio.wait_for(drain, 5)


@pytest.mark.parametrize("cleanup", ["past_run_deadline", "past_cleanup_allowance"])
async def test_timely_transport_receipt_survives_client_close_past_adapter_deadlines(composed, monkeypatch, cleanup):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    gate = Gate()
    _, close_tasks = fail_client_close(env.provider, gate=gate)
    loop = asyncio.get_running_loop()
    receipts = []
    original_normalize = normalization.normalize_model_completion_response

    def observe_receipt(envelope, bounded):
        receipt = original_normalize(envelope, bounded)
        receipts.append((loop.time(), receipt))
        return receipt

    monkeypatch.setattr(normalization, "normalize_model_completion_response", observe_receipt)
    port = env.build(port_settings=settings(run_timeout_seconds=1, cancellation_cleanup_seconds=1))
    call = asyncio.create_task(port.complete(request(), invocation))
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        owned_invocation = next(iter(port._invocations))
        assert len(receipts) == 1
        assert receipts[0][0] < owned_invocation.run_deadline
        if cleanup == "past_run_deadline":
            await asyncio.sleep(max(0, owned_invocation.run_deadline - loop.time()) + 0.05)
            gate.release.set()
        result = await asyncio.wait_for(call, 3)
        assert result.content == receipts[0][1].content == CONTENT.replace("\r\n", "\n")
        assert not port.is_healthy()
        if cleanup == "past_cleanup_allowance":
            assert port._retained
            assert all(not task.done() for task in close_tasks)
            drain = asyncio.create_task(port.wait_for_shutdown_completion())
            await asyncio.sleep(0)
            assert not drain.done()
            gate.release.set()
            await asyncio.wait_for(drain, 5)
        else:
            await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert port.capabilities.native_async_cancellation is False
        assert not port.is_healthy()
        assert all(task.done() and not task._log_traceback for task in close_tasks)
        assert not port._owned
        assert not port._retained
        assert not port._accounting._governor._handles
        assert len(env.provider.requests) == 1
        row = await assert_safe_surfaces(env, invocation, result=result)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
    finally:
        await drain_call(port, call, gate)


async def test_transport_receipt_after_original_deadline_is_not_cached_on_cleanup_error(composed, monkeypatch):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    _, close_tasks = fail_client_close(env.provider)
    original_normalize = normalization.normalize_model_completion_response
    observed = []

    def late_receipt(envelope, bounded):
        result = original_normalize(envelope, bounded)
        time.sleep(1.05)
        observed.append(next(iter(port._invocations)))
        return result

    monkeypatch.setattr(normalization, "normalize_model_completion_response", late_receipt)
    port = env.build(port_settings=settings(run_timeout_seconds=1, cancellation_cleanup_seconds=1))
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(request(), invocation)
    assert_failure(caught.value, "model_completion_timeout", ModelFailureDomain.REQUEST)
    assert len(observed) == 1
    assert observed[0].result is None
    assert not port.is_healthy()
    assert all(task.done() and not task._log_traceback for task in close_tasks)
    assert len(env.provider.requests) == 1
    row = await assert_safe_surfaces(env, invocation)
    assert row["state"] == "ambiguous"
    assert await usage_rows(env.pool, invocation.execution_id) == []


@pytest.mark.parametrize("cancel_kind", ["task_cancel", "task_cancel_return", "raised_cancel"])
async def test_native_client_close_self_cancellation_keeps_receipt_but_not_health(composed, cancel_kind):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    original_close = httpx.AsyncClient.aclose
    close_tasks = []

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            close_tasks.append(asyncio.current_task())
            if cancel_kind != "raised_cancel":
                asyncio.current_task().cancel(PRIVATE)
                if cancel_kind == "task_cancel":
                    await asyncio.sleep(0)
            else:
                raise asyncio.CancelledError(PRIVATE)

    env.provider.client_class = SelfCancellingClient
    port = env.build()
    try:
        result = await port.complete(request(), invocation)
        assert result.content == CONTENT.replace("\r\n", "\n")
        assert not asyncio.current_task().cancelling()
        await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert not env.provider.clients[0].is_closed
        assert not port.is_healthy()
        assert port.capabilities.native_async_cancellation is False
        assert all(task.done() and task.cancelled() for task in close_tasks)
        assert len(env.provider.requests) == 1
        row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
        with pytest.raises(ModelCompletionFailure) as denied:
            await port.complete(request(), identity_from_context(env.context()))
        assert_failure(denied.value, "model_adapter_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    finally:
        for client in env.provider.clients:
            await original_close(client)
    await assert_safe_surfaces(env, invocation, result=result)


@pytest.mark.parametrize("cancel_kind", ["task_cancel", "task_cancel_return", "raised_cancel"])
@pytest.mark.parametrize(
    "outcome,code,domain",
    [
        ("connection", "model_transport_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
        ("rejected", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("invalid", "invalid_model_output", ModelFailureDomain.REQUEST),
    ],
)
async def test_native_cleanup_cancellation_preserves_known_failure(composed, cancel_kind, outcome, code, domain):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    close_tasks = []

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self):
            close_tasks.append(asyncio.current_task())
            if cancel_kind == "raised_cancel":
                raise asyncio.CancelledError(PRIVATE)
            asyncio.current_task().cancel(PRIVATE)
            if cancel_kind == "task_cancel":
                await asyncio.sleep(0)

    env.provider.client_class = SelfCancellingClient
    if outcome == "connection":
        env.provider.error = httpx.ConnectError(PRIVATE)
    elif outcome == "rejected":
        env.provider.status = 429
    else:
        env.provider.payload = envelope(" \r\n ")
    port = env.build()
    try:
        with pytest.raises(ModelCompletionFailure) as caught:
            await port.complete(request(), invocation)
        assert_failure(caught.value, code, domain)
        assert not asyncio.current_task().cancelling()
        await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert not port.is_healthy()
        assert port.capabilities.native_async_cancellation is False
        assert len(env.provider.requests) == len(close_tasks) == 1
        assert all(task.done() and not task._log_traceback for task in close_tasks)
        row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
    finally:
        for client in env.provider.clients:
            await httpx.AsyncClient.aclose(client)
    await assert_safe_surfaces(env, invocation)


@pytest.mark.parametrize("outcome", ["connection", "rejected", "invalid"])
@pytest.mark.parametrize("cancellation", ["caller", "shutdown"])
async def test_cancellation_wins_known_failure_during_native_close(composed, outcome, cancellation):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    entered, release = asyncio.Event(), asyncio.Event()

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self):
            entered.set()
            await release.wait()
            asyncio.current_task().cancel(PRIVATE)

    env.provider.client_class = SelfCancellingClient
    if outcome == "connection":
        env.provider.error = httpx.ConnectError(PRIVATE)
    elif outcome == "rejected":
        env.provider.status = 429
    else:
        env.provider.payload = envelope(" \r\n ")
    port = env.build()
    call = asyncio.create_task(port.complete(request(), invocation))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancellation == "caller":
            call.cancel("original-caller-cancel")
        else:
            await port.shutdown()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await call
        assert caught.value.args == (("original-caller-cancel",) if cancellation == "caller" else ())
        await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert not port.is_healthy()
        assert len(env.provider.requests) == 1
        row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        for client in env.provider.clients:
            await httpx.AsyncClient.aclose(client)
    await assert_safe_surfaces(env, invocation)


async def test_caller_cancel_during_healthy_client_close_does_not_latch_unhealthy(composed):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    gate = Gate()

    class GatedCloseClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            await gate.wait()
            await super().aclose()

    env.provider.client_class = GatedCloseClient
    port = env.build()
    call = asyncio.create_task(port.complete(request(), invocation))
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        call.cancel("healthy-client-close-cancel")
        await asyncio.sleep(0)
        assert not call.done()
        gate.release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(call, 5)
        assert caught.value.args == ("healthy-client-close-cancel",)
        await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert port.is_healthy()
        assert port.capabilities.native_async_cancellation is True
        assert not port._owned
        assert not port._retained
        assert len(env.provider.requests) == 1
        row = await assert_safe_surfaces(env, invocation)
        assert row["state"] == "ambiguous"
        assert await usage_rows(env.pool, invocation.execution_id) == []
    finally:
        await drain_call(port, call, gate)


@pytest.mark.parametrize(
    "loss",
    [
        "user_inactive",
        "user_key_revoked",
        "team_member_revoked",
        "org_member_revoked",
        "team_relationship_changed",
        "team_key_revoked",
        "org_key_revoked",
        "invalid_user_blob",
    ],
)
async def test_scope_loss_immediately_before_resolution_never_falls_back(composed, monkeypatch, loss):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    gate = Gate()
    gate_method(monkeypatch, AuthnzUserProviderSecretsRepo, "resolve_authorized_secret", gate)
    port = env.build()
    call = asyncio.create_task(port.complete(request(), invocation))
    mutations = {
        "user_key_revoked": (
            "UPDATE user_provider_secrets SET revoked_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (env.state["user"],),
        ),
        "team_member_revoked": (
            "UPDATE team_members SET status = 'inactive' WHERE user_id = ? AND team_id = ?",
            (env.state["user"], env.state["team"]),
        ),
        "org_member_revoked": (
            "UPDATE org_members SET status = 'inactive' WHERE user_id = ? AND org_id = ?",
            (env.state["user"], env.state["org"]),
        ),
        "team_relationship_changed": (
            "UPDATE teams SET org_id = ? WHERE id = ?",
            (env.state["other_org"], env.state["team"]),
        ),
        "team_key_revoked": (
            "UPDATE org_provider_secrets SET revoked_at = CURRENT_TIMESTAMP "
            "WHERE scope_type = 'team' AND scope_id = ?",
            (env.state["team"],),
        ),
        "org_key_revoked": (
            "UPDATE org_provider_secrets SET revoked_at = CURRENT_TIMESTAMP "
            "WHERE scope_type = 'org' AND scope_id = ?",
            (env.state["org"],),
        ),
        "invalid_user_blob": (
            "UPDATE user_provider_secrets SET encrypted_blob = ? WHERE user_id = ?",
            (PRIVATE, env.state["user"]),
        ),
    }
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        if loss == "user_inactive":
            await UsersDB(env.pool).update_user(env.state["user"], is_active=False)
        else:
            sql, params = mutations[loss]
            await _execute(env.pool, sql, *params)
        gate.release.set()
        with pytest.raises(ModelCompletionFailure) as caught:
            await asyncio.wait_for(call, 5)
        assert_failure(
            caught.value,
            "invalid_provider_credentials" if loss == "invalid_user_blob" else "credential_scope_revoked",
            ModelFailureDomain.CREDENTIAL_SCOPE,
        )
        assert env.provider.requests == []
        assert env.validations == []
        assert await assert_safe_surfaces(env, invocation) is None
    finally:
        await drain_call(port, call, gate)


async def test_gzip_expansion_is_bounded_before_json_parsing(composed, monkeypatch):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    compressed = gzip.compress(json.dumps(envelope(CONTENT + "x" * 8192)).encode())
    assert len(compressed) < 1024
    env.provider.headers.update({"content-encoding": "gzip", "content-length": str(len(compressed))})
    env.provider.body_factory = lambda: Body([compressed, b"must-not-be-read"])
    parses = []

    def observed_loads(body):
        parses.append(body)
        return json.loads(body)

    monkeypatch.setattr(hc, "json", SimpleNamespace(loads=observed_loads))
    port = env.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(request(max_provider_response_bytes=1024), invocation)
    assert_failure(caught.value, "model_response_invalid", ModelFailureDomain.REQUEST)
    assert parses == []
    assert len(env.provider.requests) == 1
    assert env.provider.bodies[0].reads == 1
    row = await assert_safe_surfaces(env, invocation)
    assert row["state"] == "ambiguous"
    assert await usage_rows(env.pool, invocation.execution_id) == []


@pytest.mark.parametrize("invalid", ["prompt_unicode", "request_limit", "identity"])
async def test_invalid_request_or_identity_is_neutral_without_provider_io(composed, invalid):
    env = composed
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    bounded = request()
    if invalid == "prompt_unicode":
        bounded = request(user_prompt=PROMPT + "\ud800")
    elif invalid == "request_limit":
        object.__setattr__(bounded, "max_output_tokens", True)
    else:
        object.__setattr__(invocation, "user_id", True)
    port = env.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(bounded, invocation)
    assert_failure(
        caught.value,
        "model_identity_invalid" if invalid == "identity" else "model_request_invalid",
        ModelFailureDomain.REQUEST,
    )
    assert env.provider.requests == []
    assert env.validations == []
    assert await assert_safe_surfaces(env, invocation) is None


@pytest.mark.parametrize(
    "outcome,code,domain",
    [
        ("empty", "invalid_model_output", ModelFailureDomain.REQUEST),
        ("tools", "invalid_model_output", ModelFailureDomain.REQUEST),
        ("characters", "invalid_model_output", ModelFailureDomain.REQUEST),
        ("bytes", "invalid_model_output", ModelFailureDomain.REQUEST),
        ("unicode", "invalid_model_output", ModelFailureDomain.REQUEST),
        ("malformed_json", "model_response_invalid", ModelFailureDomain.REQUEST),
        ("401", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("429", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("500", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("connection", "model_transport_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
    ],
)
async def test_only_explicit_shared_transport_failure_is_breaker_eligible(composed, outcome, code, domain):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    bounded = request()
    if outcome == "empty":
        env.provider.payload = envelope(" \r\n ")
    elif outcome == "tools":
        env.provider.payload["choices"][0]["message"]["tool_calls"] = [{"private": PRIVATE}]
    elif outcome == "characters":
        bounded = request(max_output_chars=3)
    elif outcome == "bytes":
        env.provider.payload = envelope("\u00fc" * 10)
        bounded = request(max_output_bytes=10)
    elif outcome == "unicode":
        env.provider.payload = envelope(CONTENT + "\ud800")
    elif outcome == "malformed_json":
        env.provider.body_factory = lambda: Body([PRIVATE.encode()])
    elif outcome == "connection":
        env.provider.error = httpx.ConnectError(PRIVATE)
    else:
        env.provider.status = int(outcome)
    port = env.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(bounded, invocation)
    assert_failure(caught.value, code, domain)
    assert len(env.provider.requests) == 1
    row = await assert_safe_surfaces(env, invocation)
    assert row["state"] == "ambiguous"
    assert await usage_rows(env.pool, invocation.execution_id) == []


async def test_paid_receipt_survives_atomic_usage_write_failure_without_retry(composed, monkeypatch):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    original = AuthnzUsageRepo.insert_mcp_completion_usage
    writes = []

    async def fail_after_write(*args, **kwargs):
        await original(*args, **kwargs)
        writes.append(invocation.execution_id)
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(AuthnzUsageRepo, "insert_mcp_completion_usage", fail_after_write)
    port = env.build()
    result = await port.complete(request(), invocation)
    assert result.content == CONTENT.replace("\r\n", "\n")
    assert writes == [invocation.execution_id]
    assert len(env.provider.requests) == 1
    row = await assert_safe_surfaces(env, invocation, result=result)
    assert row["state"] == "ambiguous"
    assert row["actual_input_tokens"] is None
    assert await usage_rows(env.pool, invocation.execution_id) == []
    assert await ProviderUsageReservationsRepo(env.pool).outstanding(BillingScope("org", env.state["org"])) == {
        "tokens": row["reserved_input_tokens"] + row["reserved_output_tokens"],
        "cost_units": row["reserved_cost_units"],
    }


@pytest.mark.parametrize(
    "boundary,state,attempts,usage_count",
    [
        ("credential_before_read", None, 0, 0),
        ("credential_after_read", None, 0, 0),
        ("admission_before_write", None, 0, 0),
        ("admission_after_commit", "released", 0, 0),
        ("governor", "released", 0, 0),
        ("dispatch_before_commit", "reserved", 0, 0),
        ("dispatch_after_commit", "ambiguous", 0, 0),
        ("provider_before_response", "ambiguous", 1, 0),
        ("provider_stream", "ambiguous", 1, 0),
        ("credential_usage", "ambiguous", 1, 0),
        ("reconciliation_inside_write", "ambiguous", 1, 0),
        ("reconciliation_after_commit", "reconciled", 1, 1),
    ],
)
async def test_caller_cancellation_preserves_boundary_exposure_and_drains_owned_work(
    composed, monkeypatch, boundary, state, attempts, usage_count
):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build()
    gate = Gate()
    if boundary.startswith("credential_") and boundary != "credential_usage":
        gate_method(
            monkeypatch,
            AuthnzUserProviderSecretsRepo,
            "resolve_authorized_secret",
            gate,
            after=boundary == "credential_after_read",
        )
    elif boundary.startswith("admission_"):
        gate_method(
            monkeypatch, ProviderUsageReservationsRepo, "reserve", gate, after=boundary == "admission_after_commit"
        )
    elif boundary == "governor":
        gate_method(monkeypatch, type(port._accounting._governor), "reserve", gate)
    elif boundary.startswith("dispatch_"):
        gate_method(
            monkeypatch,
            ProviderUsageReservationsRepo,
            "mark_dispatched",
            gate,
            after=boundary == "dispatch_after_commit",
        )
    elif boundary == "provider_before_response":
        env.provider.before_response = gate.wait
    elif boundary == "provider_stream":
        env.provider.body_factory = lambda: Body([b'{"choices":'], entered=gate.entered)
    elif boundary == "credential_usage":
        gate_method(monkeypatch, ProviderCredentialRuntime, "mark_used", gate)
    elif boundary == "reconciliation_inside_write":
        gate_method(monkeypatch, AuthnzUsageRepo, "insert_mcp_completion_usage", gate, after=True)
    else:
        gate_method(monkeypatch, ProviderUsageReservationsRepo, "reconcile", gate, after=True)
    call = asyncio.create_task(port.complete(request(), invocation))
    try:
        await asyncio.wait_for(gate.entered.wait(), 3)
        call.cancel("public-cancel")
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(call, 5)
        assert caught.value.args == ("public-cancel",)
        await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
        assert len(env.provider.requests) == attempts
        row = await assert_safe_surfaces(env, invocation)
        assert (row["state"] if row else None) == state
        assert len(await usage_rows(env.pool, invocation.execution_id)) == usage_count
        if state in {"reserved", "ambiguous"}:
            assert await ProviderUsageReservationsRepo(env.pool).outstanding(BillingScope("org", env.state["org"])) == {
                "tokens": row["reserved_input_tokens"] + row["reserved_output_tokens"],
                "cost_units": row["reserved_cost_units"],
            }
        else:
            assert await ProviderUsageReservationsRepo(env.pool).outstanding(BillingScope("org", env.state["org"])) == {
                "tokens": 0,
                "cost_units": 0,
            }
        # A completed lifecycle must not leave an invocation, lease, or unusable pool.
        assert not port._owned
        assert not port._retained
        assert not port._accounting._governor._handles
        assert (
            await AuthnzOrgProviderSecretsRepo(env.pool).fetch_secret("team", env.state["team"], "openai")
        ) is not None
    finally:
        await drain_call(port, call, gate)


async def test_reconstruction_retains_ambiguous_exposure_and_forbids_replay(composed, monkeypatch):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    env.provider.status = 429
    port = env.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await port.complete(request(), invocation)
    assert_failure(caught.value, "model_response_rejected", ModelFailureDomain.REQUEST)
    before = await assert_safe_surfaces(env, invocation)
    assert before["state"] == "ambiguous"
    await port.shutdown()
    await asyncio.wait_for(port.wait_for_shutdown_completion(), 5)
    await env.pool.close()

    # Recreate process-local state, retaining only the durable database path.
    reopened = DatabasePool()
    reopened.db_path = env.pool.db_path
    reopened._sqlite_uri = False
    reopened._initialized = True

    async def fresh_pool():
        return reopened

    monkeypatch.setattr(factory, "get_db_pool", fresh_pool)
    monkeypatch.setattr(byok_runtime, "get_db_pool", fresh_pool)
    env.provider.pool = reopened
    restarted_ports = []
    try:
        repo = ProviderUsageReservationsRepo(reopened)
        assert await repo.get(invocation.execution_id) == before
        assert await repo.outstanding(BillingScope("org", env.state["org"])) == {
            "tokens": before["reserved_input_tokens"] + before["reserved_output_tokens"],
            "cost_units": before["reserved_cost_units"],
        }
        restarted = env.build()
        restarted_ports.append(restarted)
        assert restarted._accounting._governor is not port._accounting._governor
        with pytest.raises(ModelCompletionFailure) as replay:
            await restarted.complete(request(), invocation)
        assert_failure(replay.value, "accounting_unavailable", ModelFailureDomain.REQUEST)
        policy = operator_policy()
        policy["accounting"]["monthly_token_limit"] = before["reserved_input_tokens"] + before["reserved_output_tokens"]
        quota_port = env.build(policy=policy)
        restarted_ports.append(quota_port)
        subsequent = identity_from_context(env.context())
        env.provider.execution_id = subsequent.execution_id
        with pytest.raises(ModelCompletionFailure) as denied:
            await quota_port.complete(request(), subsequent)
        assert_failure(denied.value, "model_quota_exceeded", ModelFailureDomain.REQUEST)
        assert await repo.get(subsequent.execution_id) is None
        assert await repo.get(invocation.execution_id) == before
        assert await usage_rows(reopened, invocation.execution_id) == []
        assert len(env.provider.requests) == 1
        assert_private_absent((env.logs, env.caplog.text))
    finally:
        try:
            for restarted_port in restarted_ports:
                await restarted_port.shutdown()
                await asyncio.wait_for(restarted_port.wait_for_shutdown_completion(), 5)
        finally:
            await reopened.close()


async def test_metadata_cannot_populate_omitted_authenticated_scopes(composed):
    env = composed
    await seed_keys(env, ("team", "org"))
    scope = project_authenticated_execution_scope(
        authenticated_user_id=env.state["user"],
        principal_user_id=env.state["user"],
    )
    context = RequestContext(
        "mcp-request",
        user_id=str(env.state["user"]),
        server_auth_scope=scope,
        metadata={
            "active_team_id": env.state["team"],
            "active_org_id": env.state["org"],
            "user_id": 1,
            "base_url": OVERRIDE,
            "trusted_base_url_override": True,
        },
    )
    invocation = identity_from_context(context)
    assert (invocation.active_team_id, invocation.active_organization_id) == (None, None)
    env.provider.execution_id = invocation.execution_id
    result = await env.build().complete(request(), invocation)
    assert result.content == CONTENT.replace("\r\n", "\n")
    assert len(env.provider.requests) == 1
    assert env.provider.requests[0].headers["authorization"] == f"Bearer {KEYS['server']}"
    assert str(env.provider.requests[0].url) == URL
    row = await assert_safe_surfaces(env, invocation, result=result)
    assert (
        row["state"],
        row["billing_scope_type"],
        row["billing_scope_id"],
        row["active_team_id"],
        row["active_organization_id"],
    ) == ("reconciled", "user", env.state["user"], None, None)
    usage = await usage_rows(env.pool, invocation.execution_id)
    assert len(usage) == 1
    assert usage[0]["billing_org_id"] is None


async def test_authoritative_store_unavailable_never_falls_back(composed, monkeypatch):
    env = composed
    await seed_keys(env)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id

    async def unavailable_pool():
        return _UnavailablePool(env.pool)

    # Real repository/driver failure, with no replacement credential resolver.
    monkeypatch.setattr(byok_runtime, "get_db_pool", unavailable_pool)
    with pytest.raises(ModelCompletionFailure) as caught:
        await env.build().complete(request(), invocation)
    assert_failure(caught.value, "credential_store_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert env.provider.requests == []
    assert env.validations == []
    assert await assert_safe_surfaces(env, invocation) is None
