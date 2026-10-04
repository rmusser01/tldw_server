"""Bounded model completion with conservative accounting and owned teardown."""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    AuthoritativeProviderScope,
    ByokResolutionError,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.accounting import (
    CompletionAccountingPolicy,
    ModelCompletionAccounting,
)
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport import (
    OpenAICompletionTransport,
    OpenAITransportPolicy,
    snapshot_model_completion_request,
)
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionCapabilities,
    ModelCompletionFailure,
    ModelCompletionPortSettings,
    ModelCompletionRequest,
    ModelCompletionResult,
    ModelFailureDomain,
    ModelInvocationIdentity,
)

_T = TypeVar("_T")


def _failure(code: str, domain: ModelFailureDomain) -> ModelCompletionFailure:
    return ModelCompletionFailure(code, domain)


def _trusted_failure(error: Exception) -> ModelCompletionFailure:
    """Copy only explicit trusted provenance; arbitrary exceptions are neutral."""
    if type(error) is ModelCompletionFailure:
        return _failure(error.code, error.domain)
    return _failure("model_completion_failed", ModelFailureDomain.REQUEST)


@dataclass(eq=False, slots=True)
class _Invocation:
    """Invocation-local ownership, independent of a caller's later work."""

    task: asyncio.Task | None = None
    child: asyncio.Task | None = None
    reservation: Any = None
    dispatch_attempted: bool = False
    cancelled: bool = False
    abandoned: bool = False
    settled: bool = False
    fence: asyncio.Task | None = None
    work: set[asyncio.Task] = field(default_factory=set)
    run_deadline: float = 0.0
    result: ModelCompletionResult | None = None


class ModelCompletionAdapter:
    """Compose one certified transport with authoritative per-call credentials.

    The runtime factory is a trusted internal injection seam, not a request
    parameter. Public deadlines contain even non-cooperative admission,
    accounting, and credential teardown without relinquishing ownership.
    """

    def __init__(
        self,
        settings: ModelCompletionPortSettings,
        server_config_snapshot: Mapping[str, Any],
        accounting: ModelCompletionAccounting,
        transport: OpenAICompletionTransport,
        trusted_credential_runtime_factory: Callable[[ModelInvocationIdentity], Any] | None = None,
    ) -> None:
        invalid = False
        try:
            if type(settings) is not ModelCompletionPortSettings:
                raise ValueError("Invalid settings type")
            captured = ModelCompletionPortSettings(
                settings.provider,
                settings.model,
                settings.run_timeout_seconds,
                settings.cancellation_cleanup_seconds,
            )
            if captured.provider != "openai" or not (
                1 <= captured.run_timeout_seconds <= 120 and 1 <= captured.cancellation_cleanup_seconds <= 15
            ):
                raise ValueError("Invalid settings bounds")
            accounting_policy = accounting.policy
            transport_policy = transport.policy
            if (
                type(accounting_policy) is not CompletionAccountingPolicy
                or type(transport_policy) is not OpenAITransportPolicy
                or (accounting_policy.provider, accounting_policy.model) != (captured.provider, captured.model)
                or (transport_policy.provider, transport_policy.model) != (captured.provider, captured.model)
                or transport_policy.timeout_seconds != captured.run_timeout_seconds
            ):
                raise ValueError("Mismatched frozen policies")
            # Reconstruct to revalidate frozen objects modified through low-level APIs.
            CompletionAccountingPolicy(
                **{f.name: getattr(accounting_policy, f.name) for f in fields(accounting_policy)}
            )
            OpenAITransportPolicy(**{f.name: getattr(transport_policy, f.name) for f in fields(transport_policy)})
            config = copy.deepcopy(dict(server_config_snapshot))
            if trusted_credential_runtime_factory is not None and not callable(trusted_credential_runtime_factory):
                raise ValueError("Invalid trusted runtime factory")
        except Exception:  # noqa: BLE001 - detach malformed operator state
            invalid = True
        if invalid:
            raise_detached_error(_failure("model_adapter_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE))
        self._settings = captured
        self._config = config
        self._accounting = accounting
        self._transport = transport
        self._runtime_factory = trusted_credential_runtime_factory
        self._closing = False
        self._unhealthy = not self._certified()
        self._invocations: set[_Invocation] = set()
        self._owned: set[asyncio.Task] = set()
        self._retained: set[asyncio.Task] = set()

    @property
    def capabilities(self) -> ModelCompletionCapabilities:
        """Expose the transport's current certification without probing a provider."""
        return self._transport.capabilities

    def _certified(self) -> bool:
        try:
            caps = self.capabilities
            return type(caps) is ModelCompletionCapabilities and all(
                getattr(caps, item.name) is True for item in fields(caps)
            )
        except Exception:  # noqa: BLE001 - unavailable certification fails closed
            return False

    def is_healthy(self) -> bool:
        """Explicitly re-establish readiness only after retained work terminates."""
        healthy = not self._closing and not self._retained and self._certified()
        self._unhealthy = not healthy
        return healthy

    def _own(self, task: asyncio.Task, invocation: _Invocation) -> asyncio.Task:
        self._owned.add(task)
        invocation.work.add(task)
        if invocation.abandoned:
            self._retained.add(task)

        def completed(done: asyncio.Task) -> None:
            # Always retrieve terminal exceptions, including private late failures.
            if not done.cancelled():
                done.exception()
            self._owned.discard(done)
            self._retained.discard(done)
            invocation.work.discard(done)
            if done is invocation.task:
                self._invocations.discard(invocation)
            # Removing ownership never clears the unhealthy latch.

        task.add_done_callback(completed)
        return task

    async def _await_owned(
        self,
        task: asyncio.Task[_T],
        *,
        cancel_on_cancellation: bool,
    ) -> _T:
        cancellation = None
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                    if cancel_on_cancellation:
                        task.cancel(error.args[0] if error.args else None)
        if cancellation is not None:
            if not task.cancelled():
                task.exception()
            raise cancellation
        return task.result()

    def _operation(self, operation: Awaitable[_T], invocation: _Invocation, name: str) -> asyncio.Task[_T]:
        return self._own(asyncio.create_task(operation, name=name), invocation)

    def _guard(self, invocation: _Invocation) -> None:
        if invocation.cancelled or invocation.abandoned or self._closing:
            raise asyncio.CancelledError

    def _cancel(self, invocation: _Invocation, message: Any = None) -> None:
        if not invocation.cancelled:
            invocation.cancelled = True
            if invocation.task is not None:
                invocation.task.cancel(message)

    def _fence(self, invocation: _Invocation) -> asyncio.Task:
        if invocation.fence is None:
            invocation.fence = self._operation(
                self._retain_ambiguous(invocation),
                invocation,
                "mcp-model-accounting-fence",
            )
        return invocation.fence

    async def _retain_ambiguous(self, invocation: _Invocation) -> None:
        retained = False
        try:
            retained = await self._accounting.retain_ambiguous(invocation.reservation) is True
        except Exception:  # noqa: BLE001 - preserve chargeable exposure and contain private storage errors
            retained = False
        if not retained:
            logger.warning("MCP completion conservative accounting degraded")

    def _abandon(self, invocation: _Invocation) -> None:
        # Install the local publication/settlement fence before any persistence await.
        invocation.abandoned = True
        self._unhealthy = True
        self._retained.update(task for task in invocation.work if not task.done())
        logger.warning("MCP completion cleanup allowance exceeded")
        if invocation.reservation is not None and invocation.dispatch_attempted:
            self._fence(invocation)

    def _runtime(self, identity: ModelInvocationIdentity) -> Any:
        if self._runtime_factory is not None:
            return self._runtime_factory(identity)
        return ProviderCredentialRuntime(
            user_id=identity.user_id,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            authoritative_scope=AuthoritativeProviderScope(
                identity.user_id,
                identity.active_team_id,
                identity.active_organization_id,
            ),
            server_config_snapshot=self._config,
        )

    @staticmethod
    def _runtime_lifecycle(runtime: Any) -> tuple[bool, Callable[[], Awaitable[None]]]:
        pending = runtime.has_pending_shutdown_work
        drain = runtime.wait_for_shutdown_completion
        if type(pending) is not bool or not inspect.iscoroutinefunction(drain):
            raise ValueError("Invalid credential lifecycle contract")
        return pending, drain

    def _validate_runtime_lifecycle(self, runtime: Any) -> None:
        """Require truthful native lifecycle operations before a paid attempt."""
        valid = False
        try:
            self._runtime_lifecycle(runtime)
            valid = True
        except Exception:  # noqa: BLE001 - trusted composition defects must still be detached
            valid = False
        if not valid:
            raise_detached_error(
                _failure("model_credential_lifecycle_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            )

    async def _supervise_runtime_shutdown(self, runtime: Any) -> None:
        """Retain ownership until fresh public lifecycle proof certifies teardown."""
        supervisor = asyncio.current_task()
        loop = asyncio.get_running_loop()
        warned = False
        while True:
            try:
                pending, drain = self._runtime_lifecycle(runtime)
                if pending:
                    self._retained.add(supervisor)
                    self._unhealthy = True
                    await drain()
                    pending, _ = self._runtime_lifecycle(runtime)
                if pending is False:
                    # Deliver ancillary self-cancellation before certifying a
                    # normal supervisor return, without touching the caller.
                    await asyncio.sleep(0)
                    return
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - ancillary API faults cannot release ownership
                pass
            self._retained.add(supervisor)
            self._unhealthy = True
            if not warned:
                logger.warning("MCP completion credential lifecycle supervision degraded")
                warned = True
            # Retry only lifecycle observation/drain, with one attempt at a time.
            deadline = loop.time() + 0.1
            while (remaining := deadline - loop.time()) > 0:
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:
                    # Cancellation cannot skip or restart the backoff interval.
                    continue

    async def _provider(
        self,
        invocation: _Invocation,
        request: ModelCompletionRequest,
        runtime: Any,
        credentials: Any,
    ) -> ModelCompletionResult:
        self._guard(invocation)
        invocation.dispatch_attempted = True
        await self._accounting.mark_dispatched(invocation.reservation)
        self._guard(invocation)
        normalized = await self._transport.complete(request, credentials)
        self._guard(invocation)
        if asyncio.get_running_loop().time() > invocation.run_deadline:
            raise_detached_error(_failure("model_completion_timeout", ModelFailureDomain.REQUEST))
        result = ModelCompletionResult(normalized.content)
        # Only this timely certified receipt can survive ancillary teardown failure.
        invocation.result = result
        marked = False
        try:
            marked = await runtime.mark_used(credentials) is True
        except Exception:  # noqa: BLE001 - valid paid output survives credential-use persistence failure
            marked = False
        self._guard(invocation)
        if not marked:
            logger.warning("MCP completion credential usage degraded")
        if marked:
            try:
                invocation.settled = (
                    await self._accounting.reconcile(
                        invocation.reservation,
                        input_tokens=normalized.input_tokens,
                        output_tokens=normalized.output_tokens,
                    )
                    is True
                )
            except Exception:  # noqa: BLE001 - valid paid output survives accounting persistence failure
                invocation.settled = False
            if not invocation.settled:
                logger.warning("MCP completion post-call accounting degraded")
            self._guard(invocation)
        return result

    async def _run(
        self,
        invocation: _Invocation,
        request: ModelCompletionRequest,
        identity: ModelInvocationIdentity,
    ) -> ModelCompletionResult:
        runtime = None
        lifecycle_validated = False
        try:
            failure = None
            try:
                runtime = self._runtime(identity)
                self._validate_runtime_lifecycle(runtime)
                lifecycle_validated = True
                credentials = await runtime.resolve(self._settings.provider, model=self._settings.model)
            except ByokResolutionError as error:
                failure = _failure(
                    error.code,
                    (
                        ModelFailureDomain.SHARED_INFRASTRUCTURE
                        if error.code == "credential_store_unavailable"
                        else ModelFailureDomain.CREDENTIAL_SCOPE
                    ),
                )
            except Exception as error:  # noqa: BLE001 - arbitrary scope defects are not shared by inference
                failure = (
                    _trusted_failure(error)
                    if type(error) is ModelCompletionFailure
                    else _failure("model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
                )
            if failure is not None:
                raise_detached_error(failure)
            self._guard(invocation)
            invocation.reservation = await self._accounting.reserve(request, identity)
            self._guard(invocation)
            invocation.child = self._operation(
                self._provider(invocation, request, runtime, credentials),
                invocation,
                "mcp-model-completion-" + identity.execution_id,
            )
            return await self._await_owned(invocation.child, cancel_on_cancellation=True)
        finally:
            try:
                if invocation.reservation is not None and not invocation.settled:
                    cleanup = (
                        self._fence(invocation)
                        if invocation.dispatch_attempted
                        else self._operation(
                            self._accounting.release_before_dispatch(invocation.reservation),
                            invocation,
                            "mcp-model-accounting-release",
                        )
                    )
                    await self._await_owned(cleanup, cancel_on_cancellation=False)
            finally:
                if runtime is not None:
                    close = self._operation(runtime.close(), invocation, "mcp-model-credential-close")
                    try:
                        await self._await_owned(close, cancel_on_cancellation=False)
                    except Exception:  # noqa: BLE001 - runtime cleanup contains private failures
                        logger.warning("MCP completion credential cleanup degraded")
                    finally:
                        if lifecycle_validated:
                            drain = self._operation(
                                self._supervise_runtime_shutdown(runtime), invocation, "mcp-model-credential-drain"
                            )
                            await self._await_owned(drain, cancel_on_cancellation=False)

    async def complete(
        self,
        request: ModelCompletionRequest,
        identity: ModelInvocationIdentity,
    ) -> ModelCompletionResult:
        """Snapshot before admission and contain a single invocation's lifetime."""
        snapshot = snapshot_model_completion_request(request)
        invalid = False
        try:
            if type(identity) is not ModelInvocationIdentity:
                raise ValueError("Invalid identity type")
            identity_snapshot = ModelInvocationIdentity(
                identity.user_id,
                identity.active_team_id,
                identity.active_organization_id,
                identity.execution_id,
            )
        except Exception:  # noqa: BLE001 - untrusted malformed identity is request-local
            invalid = True
        if invalid:
            raise_detached_error(_failure("model_identity_invalid", ModelFailureDomain.REQUEST))
        if self._closing or self._unhealthy or self._retained or not self._certified():
            raise_detached_error(_failure("model_adapter_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE))
        invocation = _Invocation(run_deadline=asyncio.get_running_loop().time() + self._settings.run_timeout_seconds)
        self._invocations.add(invocation)
        invocation.task = self._operation(
            self._run(invocation, snapshot, identity_snapshot),
            invocation,
            "mcp-model-invocation",
        )
        cancellation = None
        timed_out = False
        try:
            done, _ = await asyncio.wait(
                {invocation.task}, timeout=max(0.0, invocation.run_deadline - asyncio.get_running_loop().time())
            )
            timed_out = not done
        except asyncio.CancelledError as error:
            cancellation = error
        if timed_out or cancellation is not None:
            self._cancel(invocation, cancellation.args[0] if cancellation and cancellation.args else None)
            deadline = asyncio.get_running_loop().time() + self._settings.cancellation_cleanup_seconds
            while not invocation.task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait({invocation.task}, timeout=remaining)
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
            if not invocation.task.done():
                self._abandon(invocation)
            elif not invocation.task.cancelled():
                invocation.task.exception()
            if cancellation is not None:
                raise cancellation
            if self._closing:
                raise asyncio.CancelledError
            if invocation.result is not None:
                return invocation.result
            if invocation.abandoned:
                raise_detached_error(
                    _failure("model_completion_cleanup_incomplete", ModelFailureDomain.SHARED_INFRASTRUCTURE)
                )
            raise_detached_error(_failure("model_completion_timeout", ModelFailureDomain.REQUEST))
        self._guard(invocation)
        failure = None
        try:
            result = invocation.task.result()
        except asyncio.CancelledError:
            # Only completed owned work reaches here; caller cancellation and
            # shutdown have already won before this cached-receipt fallback.
            if invocation.result is None:
                raise
            logger.warning("MCP completion post-call bookkeeping cancelled")
            return invocation.result
        except Exception as error:  # noqa: BLE001 - expose only explicit trusted failure provenance
            failure = _trusted_failure(error)
        if failure is not None:
            raise_detached_error(failure)
        return result

    async def shutdown(self) -> None:
        """Stop admission and forward cancellation once to active invocations."""
        self._closing = True
        for invocation in tuple(self._invocations):
            self._cancel(invocation)

    async def wait_for_shutdown_completion(self) -> None:
        """Drain all owned admission, provider, accounting, and runtime work."""
        cancellation = None
        while self._owned:
            try:
                await asyncio.wait(set(self._owned))
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation
