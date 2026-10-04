"""Managed completion ownership, deadline, authority, and privacy tests."""

import asyncio
import gc
import importlib
import logging
import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.accounting import CompletionAccountingPolicy
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.normalization import NormalizedModelCompletion
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import OpenAITransportPolicy
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionCapabilities,
    ModelCompletionFailure,
    ModelCompletionPortSettings,
    ModelCompletionRequest,
    ModelFailureDomain,
    ModelInvocationIdentity,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

MODULE = "tldw_Server_API.app.core.MCP_unified.adapters.model_completion.adapter"
CAPS = ModelCompletionCapabilities(True, True, True, True, True)
PRIVATE = "private-key-prompt-endpoint"


async def turns(count=5):
    for _ in range(count):
        await asyncio.sleep(0)


@pytest_asyncio.fixture
async def asyncio_diagnostics():
    """Capture loop diagnostics while retaining default asyncio reporting."""
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    contexts, messages = [], []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(self.format(record))

    def report(current_loop, context):
        contexts.append(context)
        current_loop.default_exception_handler(context)

    handler = Capture(level=logging.ERROR)
    logger = logging.getLogger("asyncio")
    logger.addHandler(handler)
    loop.set_exception_handler(report)
    try:
        yield contexts, messages
    finally:
        loop.set_exception_handler(previous)
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def harness():
    events = []
    gates = SimpleNamespace(
        resolve=None, reserve=None, dispatch=None, mark=None, reconcile=None, ambiguous=None, close=None
    )
    failures = SimpleNamespace(resolve=None, dispatch=None, mark=None, reconcile=None, ambiguous=None, close=None)

    class Runtime:
        mark_result = True
        closed = False

        @property
        def has_pending_shutdown_work(self):
            return False

        async def wait_for_shutdown_completion(self):
            return None

        async def resolve(self, provider, *, model):
            events.append("resolve")
            assert (provider, model) == ("openai", "fixed")
            if gates.resolve:
                await gates.resolve.wait()
            if failures.resolve:
                raise failures.resolve
            return "issued-credentials"

        async def mark_used(self, credentials):
            events.append("mark_used")
            assert credentials == "issued-credentials"
            if gates.mark:
                await gates.mark.wait()
            if failures.mark:
                raise failures.mark
            return self.mark_result

        async def close(self):
            events.append("close")
            if gates.close:
                await gates.close.wait()
            self.closed = True
            if failures.close:
                raise failures.close

    class Accounting:
        policy = CompletionAccountingPolicy("openai", "fixed", 2, 3, 1000, 2000, "mcp.default")
        state = None
        received = None
        reconcile_calls = 0
        reconcile_result = True
        ambiguous_result = True

        async def reserve(self, request, identity):
            events.append("reserve")
            self.received = (request, identity)
            if gates.reserve:
                await gates.reserve.wait()
            self.state = "reserved"
            return object()

        async def mark_dispatched(self, handle):
            events.append("dispatch")
            if gates.dispatch:
                await gates.dispatch.wait()
            if failures.dispatch:
                raise failures.dispatch
            self.state = "dispatched"

        async def release_before_dispatch(self, handle):
            events.append("release")
            self.state = "released"

        async def retain_ambiguous(self, handle):
            events.append("ambiguous")
            if gates.ambiguous:
                await gates.ambiguous.wait()
            if failures.ambiguous:
                raise failures.ambiguous
            if self.state != "reconciled":
                self.state = "ambiguous"
            return self.ambiguous_result

        async def reconcile(self, handle, **counts):
            events.append("reconcile")
            self.reconcile_calls += 1
            if gates.reconcile:
                try:
                    await gates.reconcile.wait()
                except asyncio.CancelledError:
                    await gates.reconcile.wait()
            if failures.reconcile:
                raise failures.reconcile
            if self.state == "ambiguous":
                return False
            if self.reconcile_result:
                self.state = "reconciled"
            return self.reconcile_result

    class Transport:
        capabilities = CAPS
        policy = OpenAITransportPolicy(
            "openai",
            "fixed",
            TrustedProviderEndpoint(
                "https://api.openai.com/v1/", ConfiguredEndpointScope.from_url("https://api.openai.com/v1/")
            ),
            1,
            "max_tokens",
        )
        started = asyncio.Event()
        cancelled = asyncio.Event()
        finish = asyncio.Event()
        mode = "success"
        error = None
        calls = 0
        received = None
        child_name = None

        async def complete(self, request, credentials, *, on_receipt=None):
            events.append("transport")
            self.calls += 1
            self.received = request
            self.child_name = asyncio.current_task().get_name()
            self.started.set()
            if self.mode != "success":
                try:
                    await self.finish.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    if self.mode == "cooperative":
                        raise
                    await self.finish.wait()
            if self.error:
                raise self.error
            result = NormalizedModelCompletion("valid content", 3, 4)
            if on_receipt is not None:
                on_receipt(result.content)
            return result

    runtimes, identities = [], []

    def factory(identity):
        identities.append(identity)
        runtime = Runtime()
        runtimes.append(runtime)
        return runtime

    accounting, transport = Accounting(), Transport()
    settings = ModelCompletionPortSettings("openai", "fixed", 1, 1)
    config = {"openai": {"api_key": "captured"}}

    def build(**kwargs):
        assert importlib.util.find_spec(MODULE) is not None, "managed adapter is missing"
        adapter_class = importlib.import_module(MODULE).ModelCompletionAdapter
        return adapter_class(
            settings=kwargs.pop("settings", settings),
            server_config_snapshot=config,
            accounting=accounting,
            transport=transport,
            trusted_credential_runtime_factory=kwargs.pop("factory", factory),
            **kwargs,
        )

    return SimpleNamespace(
        build=build,
        events=events,
        gates=gates,
        failures=failures,
        accounting=accounting,
        transport=transport,
        settings=settings,
        config=config,
        runtimes=runtimes,
        identities=identities,
        factory=factory,
        runtime_type=Runtime,
        request=ModelCompletionRequest("sys", "user", 10, 100, 200, 2000),
        identity=ModelInvocationIdentity(7, 11, 13, str(uuid4())),
    )


async def invoke(h, adapter):
    return await adapter.complete(h.request, h.identity)


async def cancel_started(h, adapter):
    task = asyncio.create_task(invoke(h, adapter))
    await h.transport.started.wait()
    task.cancel("original cancellation")
    return task


@pytest.mark.asyncio
async def test_success_orders_authority_dispatch_usage_settlement_and_close(harness):
    h = harness
    adapter = h.build()
    result = await invoke(h, adapter)
    assert result.content == "valid content"
    assert h.events == ["resolve", "reserve", "dispatch", "transport", "mark_used", "reconcile", "close"]
    assert h.accounting.received[0] is h.transport.received
    assert h.accounting.received[0] is not h.request
    assert h.identities[0] is h.accounting.received[1]
    assert h.identities[0] is not h.identity
    assert h.transport.calls == 1
    assert h.transport.child_name.startswith("mcp-model-completion-")
    assert h.runtimes[0].closed
    assert adapter.is_healthy()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,domain",
    [
        ("invalid_provider_credentials", ModelFailureDomain.CREDENTIAL_SCOPE),
        ("credential_scope_revoked", ModelFailureDomain.CREDENTIAL_SCOPE),
        ("credential_store_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
    ],
)
async def test_authoritative_resolution_failure_is_detached(harness, code, domain):
    h = harness
    h.failures.resolve = ByokResolutionError(code, "openai")
    h.failures.resolve.__context__ = RuntimeError(PRIVATE)
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.domain is domain
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert h.events == ["resolve", "close"]
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(PRIVATE), RuntimeError(PRIVATE)])
async def test_unexpected_scope_exception_is_not_inferred_shared(harness, error):
    h = harness
    h.failures.resolve = error
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.domain is ModelFailureDomain.CREDENTIAL_SCOPE
    assert caught.value.__context__ is None
    assert PRIVATE not in str(caught.value)


@pytest.mark.asyncio
async def test_cancel_before_dispatch_releases_unused_reservation(harness):
    h = harness
    h.gates.reserve = asyncio.Event()
    original = h.accounting.reserve

    async def reserve(*args):
        try:
            return await original(*args)
        except asyncio.CancelledError:
            h.gates.reserve.set()
            return await original(*args)

    h.accounting.reserve = reserve
    adapter = h.build()
    task = asyncio.create_task(invoke(h, adapter))
    while "reserve" not in h.events:
        await turns(1)
    task.cancel("pre-dispatch")
    with pytest.raises(asyncio.CancelledError, match="pre-dispatch"):
        await task
    assert h.accounting.state == "released"
    assert h.transport.calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_timeout_cancels_child_and_keeps_conservative_exposure(harness):
    h = harness
    h.transport.mode = "cooperative"
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.code == "model_completion_timeout"
    assert caught.value.domain is ModelFailureDomain.REQUEST
    assert h.transport.cancelled.is_set()
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_native_caller_cancel_wins_terminating_child(harness):
    h = harness
    h.transport.mode = "cooperative"
    task = await cancel_started(h, h.build())
    with pytest.raises(asyncio.CancelledError, match="original cancellation"):
        await task
    assert h.transport.cancelled.is_set()
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("late_error", [None, RuntimeError(PRIVATE)])
async def test_swallowed_cancel_retains_discards_and_requires_explicit_health(harness, late_error):
    h = harness
    h.transport.mode = "swallow"
    h.transport.error = late_error
    adapter = h.build()
    task = await cancel_started(h, adapter)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter.is_healthy()
    assert h.accounting.state == "ambiguous"
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, adapter)
    h.transport.finish.set()
    await turns(30)
    assert h.accounting.reconcile_calls == 0
    # Termination callbacks must not reopen admission implicitly.
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, adapter)
    assert adapter.is_healthy()
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_repeated_cancel_preserves_first_cancel_and_cleanup_deadline(harness):
    h = harness
    h.transport.mode = "swallow"
    adapter = h.build()
    task = await cancel_started(h, adapter)
    await h.transport.cancelled.wait()
    for _ in range(4):
        task.cancel("replacement")
        await turns()
    with pytest.raises(asyncio.CancelledError, match="original cancellation"):
        await asyncio.wait_for(task, 1.5)
    assert not adapter.is_healthy()
    h.transport.finish.set()
    await adapter.shutdown()
    await adapter.wait_for_shutdown_completion()
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_shutdown_owns_retained_work_and_never_recovers_closing_port(harness):
    h = harness
    h.transport.mode = "swallow"
    adapter = h.build()
    task = await cancel_started(h, adapter)
    with pytest.raises(asyncio.CancelledError):
        await task
    await adapter.shutdown()
    drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
    await turns()
    assert not drain.done()
    h.transport.finish.set()
    await drain
    assert h.runtimes[0].closed
    assert not adapter.is_healthy()
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, adapter)
    assert h.transport.calls == 1


@pytest.mark.asyncio
async def test_shutdown_during_admission_prevents_dispatch_and_waits_runtime_close(harness):
    h = harness
    h.gates.resolve = asyncio.Event()
    h.gates.close = asyncio.Event()
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    while not h.runtimes:
        await turns(1)
    await adapter.shutdown()
    drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
    await turns()
    assert not drain.done()
    h.gates.resolve.set()
    h.gates.close.set()
    await drain
    with pytest.raises(asyncio.CancelledError):
        await call
    assert h.transport.calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, RuntimeError(PRIVATE)])
async def test_mark_used_failure_keeps_valid_content_without_downward_reconcile(harness, failure):
    h = harness
    if failure is False:
        h.runtime_type.mark_result = False
    else:
        h.failures.mark = failure
    result = await invoke(h, h.build())
    assert result.content == "valid content"
    assert h.accounting.state == "ambiguous"
    assert h.accounting.reconcile_calls == 0


@pytest.mark.asyncio
async def test_dispatch_persistence_failure_never_releases_after_attempt(harness):
    h = harness
    h.failures.dispatch = ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, h.build())
    assert "release" not in h.events
    assert h.accounting.state == "ambiguous"
    assert h.transport.calls == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "OpenAI"),
        ("model", "other"),
        ("run_timeout_seconds", 121),
        ("cancellation_cleanup_seconds", 16),
        ("run_timeout_seconds", True),
        ("cancellation_cleanup_seconds", 0),
    ],
)
def test_constructor_revalidates_settings_and_matching_policies(harness, field, value):
    h = harness
    settings = replace(h.settings)
    object.__setattr__(settings, field, value)
    with pytest.raises(ModelCompletionFailure):
        h.build(settings=settings)
    assert h.events == []


def test_constructor_rejects_http_deadline_mismatch(harness):
    h = harness
    h.transport.policy = replace(h.transport.policy, timeout_seconds=2)
    with pytest.raises(ModelCompletionFailure):
        h.build()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,field,value",
    [
        ("identity", "user_id", True),
        ("identity", "execution_id", "private-id"),
        ("request", "max_output_tokens", 8193),
        ("request", "user_prompt", "\ud800"),
    ],
)
async def test_request_identity_revalidated_before_any_admission(harness, target, field, value):
    h = harness
    object.__setattr__(getattr(h, target), field, value)
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.domain is ModelFailureDomain.REQUEST
    assert h.events == []


@pytest.mark.asyncio
async def test_original_request_identity_mutation_cannot_change_admitted_snapshot(harness):
    h = harness
    h.gates.resolve = asyncio.Event()
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    while "resolve" not in h.events:
        await turns(1)
    object.__setattr__(h.request, "user_prompt", "changed")
    object.__setattr__(h.identity, "user_id", 99)
    h.gates.resolve.set()
    await call
    assert h.accounting.received[0].user_prompt == "user"
    assert h.accounting.received[1].user_id == 7


@pytest.mark.asyncio
async def test_health_revalidates_all_capabilities_without_dispatch(harness):
    h = harness
    adapter = h.build()
    h.transport.capabilities = replace(CAPS, native_async_cancellation=False)
    assert not adapter.is_healthy()
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, adapter)
    h.transport.capabilities = CAPS
    assert adapter.is_healthy()
    assert h.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_cancel_wins_late_completion_inside_cleanup_allowance(harness, outcome):
    h = harness
    h.transport.mode = "swallow"
    if outcome == "error":
        h.transport.error = RuntimeError(PRIVATE)
    adapter = h.build()
    call = await cancel_started(h, adapter)
    await h.transport.cancelled.wait()
    h.transport.finish.set()
    with pytest.raises(asyncio.CancelledError, match="original cancellation"):
        await call
    assert h.accounting.state == "ambiguous"
    assert h.accounting.reconcile_calls == 0
    assert h.runtimes[0].closed
    assert adapter.is_healthy()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["transport", "close"])
@pytest.mark.parametrize("retained", [False, True])
async def test_late_private_errors_never_reach_loop_or_default_logger(harness, asyncio_diagnostics, boundary, retained):
    h = harness
    h.transport.mode = "swallow"
    if boundary == "transport":
        h.transport.error = RuntimeError(PRIVATE)
    else:
        h.gates.close = asyncio.Event()
        h.failures.close = RuntimeError(PRIVATE)
    adapter = h.build()
    call = await cancel_started(h, adapter)
    await h.transport.cancelled.wait()
    if not retained:
        h.transport.finish.set()
        if h.gates.close:
            h.gates.close.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    if retained:
        h.transport.finish.set()
        if h.gates.close:
            h.gates.close.set()
    await adapter.shutdown()
    await adapter.wait_for_shutdown_completion()
    gc.collect()
    await turns(20)
    contexts, messages = asyncio_diagnostics
    assert contexts == []
    assert messages == []
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("late_error", [None, RuntimeError(PRIVATE)])
async def test_timeout_retains_late_success_and_error_without_publishing(harness, late_error):
    h = harness
    h.transport.mode = "swallow"
    h.transport.error = late_error
    adapter = h.build()
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, adapter)
    assert caught.value.code == "model_completion_cleanup_incomplete"
    assert caught.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
    assert not adapter.is_healthy()
    h.transport.finish.set()
    await adapter.wait_for_shutdown_completion()
    assert h.accounting.state == "ambiguous"
    assert h.accounting.reconcile_calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_noncaller_timeout_cleanup_miss_is_shared_and_records_abandonment(harness):
    from loguru import logger

    h = harness
    h.transport.mode = "swallow"
    adapter = h.build()
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        with pytest.raises(ModelCompletionFailure) as caught:
            await invoke(h, adapter)
        assert caught.value.code == "model_completion_cleanup_incomplete"
        assert caught.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
        assert not adapter.is_healthy()
        assert sum("MCP completion cleanup allowance exceeded" in message for message in messages) == 1
        assert all(PRIVATE not in message for message in messages)
    finally:
        h.transport.finish.set()
        await adapter.wait_for_shutdown_completion()
        logger.remove(sink)


@pytest.mark.asyncio
async def test_abandonment_latch_precedes_blocked_ambiguous_persistence(harness):
    h = harness
    h.transport.mode = "swallow"
    h.gates.ambiguous = asyncio.Event()
    adapter = h.build()
    call = await cancel_started(h, adapter)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, 1.5)
    assert not adapter.is_healthy()
    h.transport.finish.set()
    await turns(30)
    assert h.accounting.reconcile_calls == 0
    assert not h.runtimes[0].closed
    h.gates.ambiguous.set()
    await adapter.wait_for_shutdown_completion()
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_inflight_reconciliation_is_fenced_by_repository_state(harness):
    h = harness
    h.gates.reconcile = asyncio.Event()
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    while "reconcile" not in h.events:
        await turns(1)
    call.cancel("during-reconcile")
    with pytest.raises(asyncio.CancelledError):
        await call
    await turns()
    assert h.accounting.state == "ambiguous"
    h.gates.reconcile.set()
    await adapter.wait_for_shutdown_completion()
    assert h.accounting.state == "ambiguous"
    assert h.accounting.reconcile_calls == 1
    assert h.events.index("mark_used") < h.events.index("reconcile")


@pytest.mark.asyncio
async def test_noncooperative_credential_close_remains_owned_after_public_deadline(harness):
    h = harness
    h.gates.close = asyncio.Event()
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    while "close" not in h.events:
        await turns(1)
    call.cancel("during-close")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, 1.5)
    assert not adapter.is_healthy()
    await adapter.shutdown()
    drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
    await turns()
    assert not drain.done()
    h.gates.close.set()
    await drain
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_noncooperative_admission_is_owned_and_cannot_dispatch_late(harness):
    h = harness
    h.gates.resolve = asyncio.Event()

    def factory(identity):
        runtime = h.factory(identity)
        original = runtime.resolve

        async def resolve(*args, **kwargs):
            try:
                return await original(*args, **kwargs)
            except asyncio.CancelledError:
                await h.gates.resolve.wait()
                return "issued-credentials"

        runtime.resolve = resolve
        return runtime

    adapter = h.build(factory=factory)
    call = asyncio.create_task(invoke(h, adapter))
    while not h.runtimes:
        await turns(1)
    call.cancel("during-admission")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, 1.5)
    assert not adapter.is_healthy()
    await adapter.shutdown()
    drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
    await turns()
    assert not drain.done()
    h.gates.resolve.set()
    await drain
    assert h.transport.calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_shutdown_drain_does_not_wait_for_callers_unrelated_followup(harness):
    h = harness
    adapter = h.build()
    returned, followup = asyncio.Event(), asyncio.Event()

    async def caller():
        await invoke(h, adapter)
        returned.set()
        await followup.wait()

    task = asyncio.create_task(caller())
    await returned.wait()
    await adapter.shutdown()
    await asyncio.wait_for(adapter.wait_for_shutdown_completion(), 0.5)
    assert not task.done()
    followup.set()
    await task


@pytest.mark.asyncio
async def test_shutdown_between_owned_completion_and_publication_blocks_content(harness, monkeypatch):
    h = harness
    adapter = h.build()
    original_wait = asyncio.wait
    caller = asyncio.current_task()

    async def wait(tasks, **kwargs):
        result = await original_wait(tasks, **kwargs)
        if asyncio.current_task() is caller:
            await adapter.shutdown()
        return result

    monkeypatch.setattr(asyncio, "wait", wait)
    with pytest.raises(asyncio.CancelledError):
        await invoke(h, adapter)


@pytest.mark.asyncio
async def test_default_runtime_uses_only_authoritative_scope_and_captured_config(harness, monkeypatch):
    h = harness
    captured = []
    module = importlib.import_module(MODULE)

    def runtime(**kwargs):
        captured.append(kwargs)
        return h.factory(h.identity)

    monkeypatch.setattr(module, "ProviderCredentialRuntime", runtime)
    adapter = h.build(factory=None)
    h.config["openai"]["api_key"] = "mutated-after-construction"
    await invoke(h, adapter)
    await invoke(h, adapter)
    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["user_id"] == 7
        assert kwargs["team_ids"] == kwargs["org_ids"] == []
        assert kwargs["trusted_base_url_override"] is False
        scope = kwargs["authoritative_scope"]
        assert (scope.user_id, scope.team_id, scope.organization_id) == (7, 11, 13)
        assert kwargs["server_config_snapshot"] == {"openai": {"api_key": "captured"}}
    assert h.runtimes[0] is not h.runtimes[1]
    assert all(runtime.closed for runtime in h.runtimes)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", list(ModelFailureDomain))
async def test_trusted_failure_domain_is_preserved_without_raw_exception_chain(harness, domain):
    h = harness
    error = ModelCompletionFailure("trusted_failure", domain)
    error.__cause__ = RuntimeError(PRIVATE)
    h.transport.error = error
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value is not error
    assert caught.value.code == "trusted_failure"
    assert caught.value.domain is domain
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert h.accounting.state == "ambiguous"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(PRIVATE), RuntimeError(PRIVATE)])
async def test_untyped_transport_failure_is_breaker_neutral(harness, error):
    h = harness
    h.transport.error = error
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.domain is ModelFailureDomain.REQUEST
    assert PRIVATE not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, RuntimeError(PRIVATE)])
async def test_reconciliation_failure_preserves_valid_content_and_exposure(harness, failure):
    h = harness
    if failure is False:
        h.accounting.reconcile_result = False
    else:
        h.failures.reconcile = failure
    result = await invoke(h, h.build())
    assert result.content == "valid content"
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_unavailable_ambiguous_write_leaves_dispatched_exposure_and_contains_late_work(harness):
    h = harness
    h.transport.mode = "swallow"
    h.failures.ambiguous = RuntimeError(PRIVATE)
    adapter = h.build()
    call = await cancel_started(h, adapter)
    with pytest.raises(asyncio.CancelledError):
        await call
    assert h.accounting.state == "dispatched"
    assert not adapter.is_healthy()
    h.transport.finish.set()
    await adapter.wait_for_shutdown_completion()
    assert h.accounting.reconcile_calls == 0
    assert h.accounting.state == "dispatched"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_cancel_during_shutdown_drain_cannot_orphan_cleanup(harness):
    h = harness
    h.gates.close = asyncio.Event()
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    while "close" not in h.events:
        await turns(1)
    await adapter.shutdown()
    drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
    await turns()
    drain.cancel("drain-cancel")
    await turns()
    drain.cancel("second")
    assert not drain.done()
    h.gates.close.set()
    with pytest.raises(asyncio.CancelledError, match="drain-cancel"):
        await drain
    with pytest.raises(asyncio.CancelledError):
        await call
    assert h.runtimes[0].closed


@pytest.mark.parametrize("policy", ["accounting", "transport"])
def test_policy_model_mismatch_fails_closed(harness, policy):
    h = harness
    dependency = getattr(h, policy)
    dependency.policy = replace(dependency.policy, model="different")
    with pytest.raises(ModelCompletionFailure):
        h.build()


@pytest.mark.parametrize(
    "field",
    [
        "native_async_cancellation",
        "response_limit_before_decode",
        "native_max_output_tokens",
        "tool_suppression",
        "automatic_retries_disabled",
    ],
)
@pytest.mark.asyncio
async def test_missing_capability_at_construction_blocks_admission(harness, field):
    h = harness
    h.transport.capabilities = replace(CAPS, **{field: False})
    adapter = h.build()
    assert not adapter.is_healthy()
    with pytest.raises(ModelCompletionFailure):
        await invoke(h, adapter)
    assert h.events == []


@pytest.mark.asyncio
async def test_settings_are_copied_before_later_lowlevel_mutation(harness):
    h = harness
    adapter = h.build()
    object.__setattr__(h.settings, "provider", "other")
    object.__setattr__(h.settings, "run_timeout_seconds", 0)
    result = await invoke(h, adapter)
    assert result.content == "valid content"
    assert h.transport.calls == 1


@pytest.mark.asyncio
async def test_late_credential_usage_after_abandonment_cannot_enter_reconcile(harness):
    h = harness
    finish, cancelled = asyncio.Event(), asyncio.Event()

    def factory(identity):
        runtime = h.factory(identity)

        async def mark_used(credentials):
            h.events.append("mark_used")
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
            return True

        runtime.mark_used = mark_used
        return runtime

    adapter = h.build(factory=factory)
    call = asyncio.create_task(invoke(h, adapter))
    while "mark_used" not in h.events:
        await turns(1)
    call.cancel("during-usage-mark")
    await cancelled.wait()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert not adapter.is_healthy()
    assert h.accounting.state == "ambiguous"
    finish.set()
    await adapter.wait_for_shutdown_completion()
    assert h.accounting.reconcile_calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_provider_self_cancellation_remains_native_and_closes_runtime(harness):
    h = harness
    h.transport.error = asyncio.CancelledError("provider-self-cancel")
    with pytest.raises(asyncio.CancelledError, match="provider-self-cancel"):
        await invoke(h, h.build())
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_success_leaves_no_live_adapter_owned_tasks(harness):
    h = harness
    adapter = h.build()
    await invoke(h, adapter)
    await adapter.wait_for_shutdown_completion()
    assert not any(task.get_name().startswith("mcp-model-") for task in asyncio.all_tasks())


@pytest.fixture
def native_runtime(harness, monkeypatch):
    """Use the actual default authoritative runtime with only its resolver gated."""
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionStatus, ResolvedByokCredentials

    monkeypatch.setattr(runtime_module, "RESOLUTION_TASK_CANCEL_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(runtime_module, "USAGE_TASK_DRAIN_TIMEOUT_SECONDS", 0.01)
    started, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    state = SimpleNamespace(phase="resolution", runtimes=[], tasks=[], error=None, usage_cancel_kind=None)
    original = runtime_module.ProviderCredentialRuntime

    async def resist():
        state.tasks.append(asyncio.current_task())
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()
        if state.error:
            raise state.error

    async def touch():
        await resist()
        if state.usage_cancel_kind == "task_cancel":
            asyncio.current_task().cancel(PRIVATE)
        elif state.usage_cancel_kind == "raised_cancel":
            raise asyncio.CancelledError(PRIVATE)

    async def resolver(provider, **kwargs):
        scope = kwargs["authoritative_scope"]
        assert (scope.user_id, scope.team_id, scope.organization_id) == (7, 11, 13)
        if state.phase == "resolution":
            await resist()
        return ResolvedByokCredentials(
            provider=provider,
            api_key="issued-key",
            app_config={},
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
            _touch_cb=touch if state.phase == "usage" else None,
        )

    def runtime(**kwargs):
        created = original(**kwargs)
        state.runtimes.append(created)
        return created

    monkeypatch.setattr(runtime_module, "resolve_byok_credentials", resolver)
    monkeypatch.setattr(importlib.import_module(MODULE), "ProviderCredentialRuntime", runtime)
    state.started, state.cancelled, state.finish = started, cancelled, finish
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["paid", "cancel", "shutdown"])
@pytest.mark.parametrize("cancel_kind", ["task_cancel", "raised_cancel"])
async def test_native_usage_self_cancel_after_paid_receipt_is_not_public_cancellation(
    harness,
    native_runtime,
    cancel_kind,
    outcome,
    asyncio_diagnostics,
):
    from loguru import logger

    h, native = harness, native_runtime
    native.phase = "usage"
    native.usage_cancel_kind = cancel_kind

    async def transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    call = asyncio.create_task(invoke(h, adapter))
    await native.started.wait()
    try:
        if outcome == "cancel":
            call.cancel("first-native-usage-human-cancel")
            await asyncio.sleep(0)
            call.cancel("second-native-usage-human-cancel")
        elif outcome == "shutdown":
            await adapter.shutdown()
        native.finish.set()
        if outcome == "paid":
            assert call.cancelling() == 0
            assert (await call).content == "valid content"
            assert call.cancelling() == 0
            assert not call.cancelled()
        else:
            with pytest.raises(asyncio.CancelledError) as cancellation:
                await call
            if outcome == "cancel":
                assert cancellation.value.args == ("first-native-usage-human-cancel",)
            assert PRIVATE not in str(cancellation.value)
        await adapter.wait_for_shutdown_completion()
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert all(task.done() for task in native.tasks)
        assert h.transport.calls == 1
        assert h.accounting.reconcile_calls == 0
        assert h.accounting.state == "ambiguous"
        assert sum(message == "MCP completion post-call bookkeeping cancelled" for message in messages) == (
            1 if outcome == "paid" else 0
        )
        assert all(PRIVATE not in message and len(message) < 100 for message in messages)
        assert asyncio_diagnostics == ([], [])
    finally:
        native.finish.set()
        await asyncio.gather(call, return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["paid", "cancel", "shutdown"])
@pytest.mark.parametrize("cancel_kind", ["task_cancel", "raised_cancel"])
@pytest.mark.parametrize("boundary", ["reconcile", "close", "fence"])
async def test_postpaid_owned_cancellation_preserves_only_non_cancelled_public_receipt(
    harness,
    monkeypatch,
    boundary,
    cancel_kind,
    outcome,
    asyncio_diagnostics,
):
    from loguru import logger

    h = harness
    started, finish = asyncio.Event(), asyncio.Event()

    async def operation(*args, **kwargs):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            await finish.wait()
        if boundary == "close":
            h.runtimes[0].closed = True
        elif boundary == "fence":
            h.accounting.state = "ambiguous"
        else:
            h.accounting.reconcile_calls += 1
        if cancel_kind == "task_cancel":
            asyncio.current_task().cancel(PRIVATE)
        else:
            raise asyncio.CancelledError(PRIVATE)
        return False if boundary == "reconcile" else None if boundary == "close" else True

    if boundary == "close":
        monkeypatch.setattr(h.runtime_type, "close", operation)
    elif boundary == "fence":
        h.runtime_type.mark_result = False
        h.accounting.retain_ambiguous = operation
    else:
        h.accounting.reconcile = operation
    adapter = h.build()
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    call = asyncio.create_task(invoke(h, adapter))
    await started.wait()
    try:
        if outcome == "cancel":
            call.cancel("first-postpaid-human-cancel")
            await asyncio.sleep(0)
            call.cancel("second-postpaid-human-cancel")
        elif outcome == "shutdown":
            await adapter.shutdown()
        finish.set()
        if outcome == "paid":
            assert call.cancelling() == 0
            assert (await call).content == "valid content"
            assert call.cancelling() == 0
            assert not call.cancelled()
        else:
            with pytest.raises(asyncio.CancelledError) as cancellation:
                await call
            if outcome == "cancel":
                assert cancellation.value.args == ("first-postpaid-human-cancel",)
            assert PRIVATE not in str(cancellation.value)
        await adapter.wait_for_shutdown_completion()
        assert h.runtimes[0].closed
        assert h.transport.calls == 1
        assert h.accounting.state == ("reconciled" if boundary == "close" else "ambiguous")
        assert sum(message == "MCP completion post-call bookkeeping cancelled" for message in messages) == (
            1 if outcome == "paid" else 0
        )
        assert all(PRIVATE not in message and len(message) < 100 for message in messages)
        assert asyncio_diagnostics == ([], [])
    finally:
        finish.set()
        await asyncio.gather(call, return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout", "shutdown"])
async def test_native_resolution_cleanup_stays_owned_until_terminal(
    harness, native_runtime, termination, asyncio_diagnostics
):
    h, native = harness, native_runtime
    native.error = RuntimeError(PRIVATE)
    adapter = h.build(factory=None)
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    await native.started.wait()
    try:
        if termination == "cancel":
            call.cancel("original-native-cancel")
        elif termination == "shutdown":
            await adapter.shutdown()
        if termination == "timeout":
            with pytest.raises(ModelCompletionFailure) as caught:
                await call
            assert caught.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
        else:
            with pytest.raises(asyncio.CancelledError):
                await call
        assert not adapter.is_healthy()
        assert not native.tasks[0].done()
        assert h.transport.calls == 0
        await adapter.shutdown()
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        native.finish.set()
        await drain
        assert all(task.done() for task in native.tasks)
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert asyncio_diagnostics == ([], [])
    finally:
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_native_usage_cleanup_keeps_paid_receipt_or_first_native_cancel(
    harness, native_runtime, cancel_caller, asyncio_diagnostics
):
    h, native = harness, native_runtime
    native.phase = "usage"
    native.error = RuntimeError(PRIVATE)

    # The transport receives a real issued handle, not the harness's text sentinel.
    async def transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    await native.started.wait()
    try:
        if cancel_caller:
            call.cancel("first-native-cancel")
            await native.cancelled.wait()
            call.cancel("second-native-cancel")
            with pytest.raises(asyncio.CancelledError, match="first-native-cancel"):
                await call
        else:
            result = await call
            assert result.content == "valid content"
        assert not adapter.is_healthy()
        assert h.accounting.state == "ambiguous"
        assert h.accounting.reconcile_calls == 0
        assert native.runtimes[0].has_pending_shutdown_work is True
        await adapter.shutdown()
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        native.finish.set()
        await drain
        assert all(task.done() for task in native.tasks)
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert asyncio_diagnostics == ([], [])
    finally:
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["mark", "reconcile", "ambiguous", "close"])
async def test_timely_paid_receipt_survives_ancillary_cleanup_miss(harness, boundary):
    h = harness
    finish = asyncio.Event()
    if boundary == "mark":

        async def mark_used(self, credentials):
            h.events.append("mark_used")
            try:
                await finish.wait()
            except asyncio.CancelledError:
                await finish.wait()
            return True

        h.runtime_type.mark_used = mark_used
    else:
        setattr(h.gates, boundary, finish)
        if boundary == "ambiguous":
            h.runtime_type.mark_result = False
    adapter = h.build()
    try:
        result = await invoke(h, adapter)
        assert result.content == "valid content"
        assert not adapter.is_healthy()
        with pytest.raises(ModelCompletionFailure):
            await invoke(h, adapter)
    finally:
        finish.set()
        await adapter.wait_for_shutdown_completion()
    if boundary in ("mark", "ambiguous"):
        assert h.accounting.reconcile_calls == 0
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("has_pending_shutdown_work", "not-a-bool"),
        ("wait_for_shutdown_completion", None),
        ("wait_for_shutdown_completion", lambda: None),
    ],
)
async def test_malformed_runtime_lifecycle_rejected_before_paid_call(harness, field, value):
    h = harness
    setattr(h.runtime_type, field, value)
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert h.transport.calls == 0
    assert "reserve" not in h.events
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,message",
    [
        ("mark", "MCP completion credential usage degraded"),
        ("reconcile", "MCP completion post-call accounting degraded"),
        ("ambiguous", "MCP completion conservative accounting degraded"),
    ],
)
async def test_false_ancillary_outcome_emits_static_warning(harness, boundary, message):
    from loguru import logger

    h = harness
    if boundary == "mark":
        h.runtime_type.mark_result = False
    elif boundary == "reconcile":
        h.accounting.reconcile_result = False
    else:
        h.runtime_type.mark_result = False
        h.accounting.ambiguous_result = False
    messages = []
    sink = logger.add(lambda event: messages.append(event.record["message"]), level="WARNING")
    try:
        result = await invoke(h, h.build())
        assert result.content == "valid content"
        assert message in messages
        assert all(PRIVATE not in item and len(item) < 100 for item in messages)
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
async def test_provider_receipt_after_original_deadline_never_caches_content(harness):
    h = harness

    async def late_transport(request, credentials, *, on_receipt=None):
        # Starve the timer callback: receipt time must still be checked locally.
        time.sleep(1.05)
        if on_receipt is not None:
            on_receipt("late content")
        return NormalizedModelCompletion("late content", 3, 4)

    h.transport.complete = late_transport
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert caught.value.code == "model_completion_timeout"
    assert h.accounting.reconcile_calls == 0
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
async def test_transport_type_error_never_retries_without_receipt_callback(harness):
    h = harness

    async def broken_transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        raise TypeError(PRIVATE)

    h.transport.complete = broken_transport
    with pytest.raises(ModelCompletionFailure) as caught:
        await invoke(h, h.build())
    assert (caught.value.code, caught.value.domain) == ("model_completion_failed", ModelFailureDomain.REQUEST)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert h.transport.calls == 1
    assert h.accounting.reconcile_calls == 0
    assert h.accounting.state == "ambiguous"
    assert h.runtimes[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_native_cancel_at_receipt_wins_cached_result_or_late_error(harness, outcome, asyncio_diagnostics):
    h = harness
    call = None

    async def transport(request, credentials, *, on_receipt=None):
        call.cancel("receipt-cancel")
        if outcome == "error":
            raise RuntimeError(PRIVATE)
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build()
    call = asyncio.create_task(invoke(h, adapter))
    with pytest.raises(asyncio.CancelledError, match="receipt-cancel"):
        await call
    await adapter.wait_for_shutdown_completion()
    assert h.runtimes[0].closed
    assert asyncio_diagnostics == ([], [])


@pytest.mark.asyncio
async def test_shutdown_suppresses_cached_paid_result_while_native_usage_pending(harness, native_runtime):
    h, native = harness, native_runtime
    native.phase = "usage"

    async def transport(request, credentials, *, on_receipt=None):
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    await native.started.wait()
    try:
        await adapter.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await call
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        native.finish.set()
        await drain
        assert native.runtimes[0].has_pending_shutdown_work is False
    finally:
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)


@pytest.fixture
def postclose_lifecycle_fault(native_runtime, monkeypatch):
    """Corrupt only the public post-close API of an actual native runtime."""
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime

    pending = ProviderCredentialRuntime.has_pending_shutdown_work
    original_drain = ProviderCredentialRuntime.wait_for_shutdown_completion
    original_close = ProviderCredentialRuntime.close
    state = SimpleNamespace(
        enabled=True,
        closed=False,
        kind="getter_raise",
        reads=0,
        drains=0,
        drain_started=asyncio.Event(),
        terminal_cancel=asyncio.Event(),
        supervisor=None,
    )

    async def close(runtime):
        await original_close(runtime)
        state.closed = True

    def has_pending(runtime):
        state.reads += 1
        if state.closed and state.enabled:
            if state.kind == "getter_raise":
                raise RuntimeError(PRIVATE)
            if state.kind == "getter_type":
                return "not-a-bool"
        native_pending = pending.fget(runtime)
        if (
            state.closed
            and state.enabled
            and state.kind in {"getter_terminal_task_cancel", "getter_terminal_task_cancel_soon"}
            and native_pending is False
            and not state.terminal_cancel.is_set()
        ):
            state.supervisor = asyncio.current_task()
            state.terminal_cancel.set()
            if state.kind == "getter_terminal_task_cancel":
                state.supervisor.cancel(PRIVATE)
            else:
                asyncio.get_running_loop().call_soon(state.supervisor.cancel, PRIVATE)
        return native_pending

    async def drain(runtime):
        state.drains += 1
        if state.closed and state.enabled:
            if state.kind == "drain_raise":
                raise RuntimeError(PRIVATE)
            if state.kind == "drain_cancel":
                raise asyncio.CancelledError(PRIVATE)
            if state.kind in {"drain_task_cancel", "drain_task_cancel_soon"}:
                state.supervisor = asyncio.current_task()
                state.drain_started.set()
                if state.kind == "drain_task_cancel":
                    state.supervisor.cancel(PRIVATE)
                else:
                    asyncio.get_running_loop().call_soon(state.supervisor.cancel, PRIVATE)
                return None
            if state.kind == "drain_early":
                return None
        state.drain_started.set()
        await original_drain(runtime)
        if (
            state.closed
            and state.enabled
            and state.kind in {"drain_terminal_task_cancel", "drain_terminal_raise_cancel"}
        ):
            state.supervisor = asyncio.current_task()
            state.terminal_cancel.set()
            if state.kind == "drain_terminal_task_cancel":
                state.supervisor.cancel(PRIVATE)
            else:
                raise asyncio.CancelledError(PRIVATE)

    def drain_method(runtime):
        if state.closed and state.enabled and state.kind == "drain_shape":
            return lambda: None
        return drain.__get__(runtime, ProviderCredentialRuntime)

    monkeypatch.setattr(ProviderCredentialRuntime, "close", close)
    monkeypatch.setattr(ProviderCredentialRuntime, "has_pending_shutdown_work", property(has_pending))
    monkeypatch.setattr(ProviderCredentialRuntime, "wait_for_shutdown_completion", property(drain_method))
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["paid", "cancel", "shutdown"])
@pytest.mark.parametrize(
    "fault_kind",
    [
        "drain_terminal_task_cancel",
        "drain_terminal_raise_cancel",
        "getter_terminal_task_cancel",
        "getter_terminal_task_cancel_soon",
    ],
)
async def test_terminal_lifecycle_self_cancel_is_quarantined_from_public_caller(
    harness,
    native_runtime,
    postclose_lifecycle_fault,
    fault_kind,
    outcome,
    asyncio_diagnostics,
):
    from loguru import logger

    h, native, fault = harness, native_runtime, postclose_lifecycle_fault
    native.phase = "usage"
    fault.kind = fault_kind

    async def transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    call = asyncio.create_task(invoke(h, adapter))
    await native.started.wait()
    try:
        if fault_kind.startswith("drain_"):
            await fault.drain_started.wait()
        native.finish.set()
        await fault.terminal_cancel.wait()
        if outcome == "cancel":
            call.cancel("first-terminal-human-cancel")
            await asyncio.sleep(0)
            call.cancel("second-terminal-human-cancel")
        elif outcome == "shutdown":
            await adapter.shutdown()
        if outcome == "paid":
            assert call.cancelling() == 0
            assert (await call).content == "valid content"
            assert call.cancelling() == 0
            assert not call.cancelled()
        else:
            with pytest.raises(asyncio.CancelledError) as cancellation:
                await call
            if outcome == "cancel":
                assert cancellation.value.args == ("first-terminal-human-cancel",)
            assert PRIVATE not in str(cancellation.value)
        await adapter.wait_for_shutdown_completion()
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert h.transport.calls == 1
        assert h.accounting.reconcile_calls <= 1
        if fault_kind.startswith("drain_"):
            assert fault.drains == 1
        else:
            assert fault.drains == 0
        assert all(PRIVATE not in message and len(message) < 100 for message in messages)
        assert sum(message == "MCP completion credential lifecycle supervision degraded" for message in messages) <= 1
        if outcome != "shutdown":
            with pytest.raises(ModelCompletionFailure):
                await invoke(h, adapter)
            assert adapter.is_healthy()
        else:
            assert not adapter.is_healthy()
        assert asyncio_diagnostics == ([], [])
    finally:
        fault.enabled = False
        native.finish.set()
        await asyncio.gather(call, return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["paid", "cancel"])
@pytest.mark.parametrize("fault_kind", ["drain_task_cancel", "drain_task_cancel_soon"])
async def test_task_self_cancel_cannot_skip_or_restart_lifecycle_backoff(
    harness,
    native_runtime,
    postclose_lifecycle_fault,
    outcome,
    fault_kind,
    asyncio_diagnostics,
):
    from loguru import logger

    h, native, fault = harness, native_runtime, postclose_lifecycle_fault
    native.phase = "usage" if outcome == "paid" else "resolution"
    fault.kind = fault_kind

    async def transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    cancellations = []
    await native.started.wait()
    try:
        if outcome == "cancel":
            call.cancel("first-backoff-caller-cancel")
            await native.cancelled.wait()
            call.cancel("second-backoff-caller-cancel")
        await fault.drain_started.wait()
        loop = asyncio.get_running_loop()
        cancellations = [loop.call_later(delay, fault.supervisor.cancel, PRIVATE) for delay in (0.03, 0.06, 0.09)]
        await asyncio.sleep(0.15)
        assert 2 <= fault.drains <= 4
        assert fault.supervisor in adapter._owned
        assert fault.supervisor in adapter._retained
        assert native.runtimes[0].has_pending_shutdown_work is True
        assert not adapter.is_healthy()
        if outcome == "paid":
            assert (await call).content == "valid content"
            assert h.accounting.state == "ambiguous"
            assert h.accounting.reconcile_calls == 0
        else:
            with pytest.raises(asyncio.CancelledError, match="first-backoff-caller-cancel"):
                await call
            assert h.transport.calls == 0
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        assert sum(message == "MCP completion credential lifecycle supervision degraded" for message in messages) == 1
        assert all(PRIVATE not in message and len(message) < 100 for message in messages)
        native.finish.set()
        await drain
        with pytest.raises(ModelCompletionFailure):
            await invoke(h, adapter)
        assert adapter.is_healthy()
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert asyncio_diagnostics == ([], [])
    finally:
        for cancellation in cancellations:
            cancellation.cancel()
        fault.enabled = False
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_kind", ["getter_raise", "getter_type", "drain_raise", "drain_cancel", "drain_early", "drain_shape"]
)
async def test_postclose_lifecycle_fault_preserves_receipt_and_requires_positive_terminal_proof(
    harness,
    native_runtime,
    postclose_lifecycle_fault,
    fault_kind,
    asyncio_diagnostics,
):
    from loguru import logger

    h, native, fault = harness, native_runtime, postclose_lifecycle_fault
    native.phase = "usage"
    fault.kind = fault_kind

    async def transport(request, credentials, *, on_receipt=None):
        h.transport.calls += 1
        if on_receipt is not None:
            on_receipt("valid content")
        return NormalizedModelCompletion("valid content", 3, 4)

    h.transport.complete = transport
    adapter = h.build(factory=None)
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    await native.started.wait()
    try:
        result = await call
        assert result.content == "valid content"
        assert not adapter.is_healthy()
        assert h.accounting.state == "ambiguous"
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        await asyncio.sleep(0.15)
        assert not drain.done()
        assert not adapter.is_healthy()
        assert sum(message == "MCP completion credential lifecycle supervision degraded" for message in messages) == 1
        assert all(PRIVATE not in message and len(message) < 100 for message in messages)
        assert fault.reads + fault.drains < 100
        native.finish.set()
        await asyncio.gather(*native.tasks, return_exceptions=True)
        if fault_kind in {"getter_raise", "getter_type", "drain_shape"}:
            # Terminal native work does not repair malformed public metadata.
            await asyncio.sleep(0.15)
            assert not drain.done()
            assert not adapter.is_healthy()
            fault.enabled = False
        else:
            # Exact False plus the valid async shape is sufficient terminal
            # proof, even if an earlier drain attempt raised or returned early.
            done, _ = await asyncio.wait({drain}, timeout=0.35)
            assert drain in done
        await drain
        with pytest.raises(ModelCompletionFailure):
            await invoke(h, adapter)
        assert adapter.is_healthy()
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert h.accounting.reconcile_calls == 0
        assert asyncio_diagnostics == ([], [])
    finally:
        fault.enabled = False
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.shutdown()
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault_kind", ["getter_raise", "drain_raise", "drain_cancel", "drain_early"])
async def test_postclose_lifecycle_fault_cannot_orphan_cancelled_no_success_invocation(
    harness,
    native_runtime,
    postclose_lifecycle_fault,
    fault_kind,
    asyncio_diagnostics,
):
    h, native, fault = harness, native_runtime, postclose_lifecycle_fault
    fault.kind = fault_kind
    adapter = h.build(factory=None)
    call = asyncio.create_task(invoke(h, adapter))
    drain = None
    await native.started.wait()
    try:
        call.cancel("first-no-receipt-cancel")
        await native.cancelled.wait()
        call.cancel("second-no-receipt-cancel")
        with pytest.raises(asyncio.CancelledError, match="first-no-receipt-cancel"):
            await call
        assert not adapter.is_healthy()
        assert h.transport.calls == 0
        await adapter.shutdown()
        drain = asyncio.create_task(adapter.wait_for_shutdown_completion())
        await turns()
        assert not drain.done()
        fault.enabled = False
        native.finish.set()
        await drain
        assert native.runtimes[0].has_pending_shutdown_work is False
        assert not adapter.is_healthy()
        assert asyncio_diagnostics == ([], [])
    finally:
        fault.enabled = False
        native.finish.set()
        await asyncio.gather(call, *(() if drain is None else (drain,)), return_exceptions=True)
        await adapter.wait_for_shutdown_completion()
        await asyncio.gather(*native.tasks, return_exceptions=True)
