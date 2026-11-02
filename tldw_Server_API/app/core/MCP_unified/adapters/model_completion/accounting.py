"""Conservative admission and settlement for one bounded model completion."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, cast

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, await_cancellation_safe_cleanup
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import _TransactionBoundPool
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import (
    MAX_INT,
    BillingScope,
    ProviderUsageReservation,
    ProviderUsageReservationError,
    ProviderUsageReservationsRepo,
    ReservationActuals,
    ReservationQuotaSnapshot,
)
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import COST_UNITS_PER_USD, AuthnzUsageRepo
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
    ModelInvocationIdentity,
)
from tldw_Server_API.app.core.Resource_Governance.governor import ResourceGovernor, RGRequest


def _count(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= MAX_INT:
        raise ValueError("Invalid bounded accounting value")
    return value


def _governor_cost(tokens: int, policy: CompletionAccountingPolicy) -> int:
    return _count((tokens + policy.governor_tokens_per_cost_unit - 1) // policy.governor_tokens_per_cost_unit + 1)


@dataclass(frozen=True, slots=True)
class CompletionAccountingPolicy:
    """Frozen nanodollar pricing and separate Resource Governor unit weights."""

    provider: str
    model: str
    input_cost_units_per_token: int
    output_cost_units_per_token: int
    monthly_token_limit: int | None
    monthly_cost_limit: int | None
    governor_policy_id: str
    input_token_overhead: int = 32
    governor_tokens_per_cost_unit: int = 1000

    def __post_init__(self) -> None:
        for text in (self.provider, self.model, self.governor_policy_id):
            if type(text) is not str or not text.strip():
                raise ValueError("Invalid accounting policy")
        for value in (self.input_cost_units_per_token, self.output_cost_units_per_token, self.input_token_overhead):
            _count(value)
        _count(self.governor_tokens_per_cost_unit, positive=True)
        for limit in (self.monthly_token_limit, self.monthly_cost_limit):
            if limit is not None:
                _count(limit)


@dataclass(slots=True)
class _DispatchBoundary:
    attempted: bool = False
    release_started: bool = False


@dataclass(frozen=True, slots=True, repr=False)
class AccountingReservation:
    """An execution-owned admission handle; it carries no prompt or output."""

    reservation: ProviderUsageReservation
    governor_handle: str
    _owner: object = field(repr=False)
    _dispatch: _DispatchBoundary = field(default_factory=_DispatchBoundary, repr=False, compare=False)


class ModelCompletionAccounting:
    """Compose durable quota authority with rate/concurrency admission."""

    def __init__(
        self,
        *,
        policy: CompletionAccountingPolicy,
        reservations: ProviderUsageReservationsRepo,
        usage: AuthnzUsageRepo,
        billing: BillingEnforcer,
        governor: ResourceGovernor,
        quota_pool_factory: Callable[[], Awaitable[DatabasePool]] | None = None,
    ) -> None:
        if type(policy) is not CompletionAccountingPolicy or any(
            service is None for service in (reservations, usage, billing, governor)
        ):
            raise ValueError("Accounting requires frozen policy and all services")
        self._policy = policy
        self._reservations = reservations
        self._usage = usage
        self._billing = billing
        self._governor = governor
        if quota_pool_factory is not None and not callable(quota_pool_factory):
            raise ValueError("Quota pool factory must be callable")
        self._quota_pool_factory = quota_pool_factory
        self._owner = object()

    @property
    def policy(self) -> CompletionAccountingPolicy:
        """Expose the frozen policy for adapter composition validation."""
        return self._policy

    def _handle(self, handle: AccountingReservation) -> ProviderUsageReservation:
        if type(handle) is not AccountingReservation or handle._owner is not self._owner:
            raise ModelCompletionFailure("invalid_accounting_handle", ModelFailureDomain.REQUEST)
        return handle.reservation

    def _exposure(
        self, request: ModelCompletionRequest, identity: ModelInvocationIdentity, credential_source: str | None
    ) -> ProviderUsageReservation:
        if type(request) is not ModelCompletionRequest or type(identity) is not ModelInvocationIdentity:
            raise ValueError("Invalid bounded request")
        scope = (
            BillingScope("org", identity.active_organization_id)
            if identity.active_organization_id is not None
            else (
                BillingScope("team", identity.active_team_id)
                if identity.active_team_id is not None
                else BillingScope("user", identity.user_id)
            )
        )
        input_tokens = _count(
            len(request.system_prompt.encode("utf-8"))
            + len(request.user_prompt.encode("utf-8"))
            + self._policy.input_token_overhead
        )
        output_tokens = _count(request.max_output_tokens, positive=True)
        _count(input_tokens + output_tokens)
        cost = _count(
            input_tokens * self._policy.input_cost_units_per_token
            + output_tokens * self._policy.output_cost_units_per_token
        )
        return ProviderUsageReservation(
            identity.execution_id,
            identity.user_id,
            identity.active_team_id,
            identity.active_organization_id,
            scope,
            self._policy.provider,
            self._policy.model,
            input_tokens,
            output_tokens,
            cost,
            credential_source,
        )

    async def _user_quota_refused(
        self, reservation: ProviderUsageReservation, conn: Any, quota_pool: DatabasePool | None
    ) -> bool:
        """Use native profile precedence and counters, never an ambient caller identity.

        MCP's required accounting remains fail-closed, so use the uncached
        resolver rather than the general chat gate's fail-open/logging wrapper.
        """
        if reservation.credential_source == "user":
            return False
        from tldw_Server_API.app.core.Usage import quota_resolver

        if not quota_resolver.usage_quotas_enabled():
            return False
        postgres = self._usage._is_postgres_backend()
        if quota_pool is not None and (quota_pool.pool is not None) != postgres:
            raise ValueError("Native quotas require the admission backend")
        pool = cast(DatabasePool, _TransactionBoundPool(conn, postgres=postgres))
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for key, since, tokens in (
            ("limits.llm_tokens_per_month", now.replace(day=1), True),
            ("limits.llm_cost_per_day_usd", now, False),
            ("limits.llm_cost_per_month_usd", now.replace(day=1), False),
        ):
            limit = await quota_resolver.uncached_user_quota(reservation.user_id, key, db_pool=pool)
            if limit is None:
                continue
            bound = Decimal(str(limit))
            used_tokens, used_costs = await self._usage.read_user_llm_exposure(conn, reservation.user_id, since)
            exposure = (
                Decimal(_count(used_tokens + reservation.reserved_input_tokens + reservation.reserved_output_tokens))
                if tokens
                else Decimal(_count(used_costs + reservation.reserved_cost_units)) / COST_UNITS_PER_USD
            )
            if not bound.is_finite() or bound < 0:
                raise ValueError("Invalid native quota accounting")
            if bound == 0 or exposure > bound:
                return True
        return False

    async def reserve(
        self,
        request: ModelCompletionRequest,
        identity: ModelInvocationIdentity,
        *,
        credential_source: str | None = None,
    ) -> AccountingReservation:
        """Reserve durably before acquiring the governor; unwind only pre-dispatch work."""
        reservation = None
        invalid = False
        try:
            reservation = self._exposure(request, identity, credential_source)
        except (ValueError, UnicodeError, ProviderUsageReservationError):
            invalid = True
        if invalid or reservation is None:
            raise_detached_error(ModelCompletionFailure("invalid_model_request", ModelFailureDomain.REQUEST))

        async def snapshot_reader(conn: Any) -> ReservationQuotaSnapshot:
            if await self._user_quota_refused(reservation, conn, quota_pool):
                raise ProviderUsageReservationError("quota_exceeded")
            hosted_limit = None
            if reservation.credential_source != "user":
                hosted_limit = await self._billing.get_mcp_token_limit(
                    reservation.billing_scope, operator_limit=None, conn=conn
                )
            month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            tokens, costs = await self._usage.read_provider_scope_usage(conn, reservation.billing_scope, month_start)
            hosted_tokens = 0
            if hosted_limit is not None:
                hosted_tokens, _ = await self._usage.read_provider_scope_usage(
                    conn, reservation.billing_scope, month_start, operator_paid_only=True
                )
            return ReservationQuotaSnapshot(
                tokens,
                self._policy.monthly_token_limit,
                costs,
                self._policy.monthly_cost_limit,
                hosted_used_tokens=hosted_tokens,
                hosted_token_limit=hosted_limit,
            )

        quota_failed = False
        quota_pool: DatabasePool | None = None
        try:
            if self._quota_pool_factory is not None:
                quota_pool = await self._quota_pool_factory()
        except Exception:  # noqa: BLE001 - keep required quota-service failures content-free
            quota_failed = True
        if quota_failed:
            raise_detached_error(
                ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            )
        durable_owned = False
        governor_handle = None
        failure = None

        def on_reserved() -> None:
            nonlocal durable_owned
            durable_owned = True

        try:
            await self._reservations.reserve(reservation, snapshot_reader=snapshot_reader, on_reserved=on_reserved)
            total_tokens = _count(reservation.reserved_input_tokens + reservation.reserved_output_tokens)
            decision, governor_handle = await self._governor.reserve(
                RGRequest(
                    entity=f"user:{identity.user_id}",
                    categories={
                        "requests": {"units": 1},
                        "tokens": {"units": total_tokens},
                        "cost_units": {"units": _governor_cost(total_tokens, self._policy)},
                        "jobs": {"units": 1},
                    },
                    tags={"policy_id": self._policy.governor_policy_id, "service": "mcp_model_completion"},
                ),
                op_id=identity.execution_id,
            )
            if decision.allowed is not True:
                failure = ModelCompletionFailure("model_quota_exceeded", ModelFailureDomain.REQUEST)
            elif type(governor_handle) is not str or not governor_handle:
                failure = ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            else:
                return AccountingReservation(reservation, governor_handle, self._owner)
        except asyncio.CancelledError:
            if durable_owned:
                await await_cancellation_safe_cleanup(
                    self._release_pre_dispatch(reservation.execution_id, governor_handle)
                )
            raise
        except ProviderUsageReservationError as exc:
            domain = (
                ModelFailureDomain.REQUEST
                if exc.code in {"quota_exceeded", "duplicate_execution"}
                else ModelFailureDomain.SHARED_INFRASTRUCTURE
            )
            failure = ModelCompletionFailure(
                "model_quota_exceeded" if exc.code == "quota_exceeded" else "accounting_unavailable", domain
            )
        except Exception:  # noqa: BLE001 - detach arbitrary required-service failures at the public boundary
            failure = ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        if durable_owned:
            await await_cancellation_safe_cleanup(self._release_pre_dispatch(reservation.execution_id, governor_handle))
        raise_detached_error(
            failure or ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        )

    async def _release_pre_dispatch(
        self,
        execution_id: str,
        governor_handle: str | None,
        dispatch: _DispatchBoundary | None = None,
    ) -> bool:
        released = False
        can_release_governor = True
        operation_logger = logger.bind(operation="mcp_model_completion.pre_dispatch_cleanup")
        try:
            await self._reservations.release_before_dispatch(execution_id)
            released = True
        except ProviderUsageReservationError as exc:
            can_release_governor = exc.code != "invalid_transition"
            operation_logger.warning("MCP completion pre-dispatch accounting cleanup degraded")
        except Exception:  # noqa: BLE001 - preserve conservative exposure when pre-dispatch cleanup fails
            operation_logger.warning("MCP completion pre-dispatch accounting cleanup degraded")
        finally:
            # Storage failure does not retain a never-dispatched concurrency lease.
            # An explicit repository dispatch fence still forbids a refund.
            if can_release_governor and governor_handle is not None and not (dispatch and dispatch.attempted):
                try:
                    await self._governor.release(governor_handle)
                except Exception:  # noqa: BLE001 - cleanup remains conservative and content-free
                    released = False
                    operation_logger.warning("MCP completion pre-dispatch governor cleanup degraded")
        return released

    async def mark_dispatched(self, handle: AccountingReservation) -> None:
        """Persist the dispatch boundary before any provider request attempt."""
        reservation = self._handle(handle)
        if handle._dispatch.release_started:
            raise ModelCompletionFailure("invalid_accounting_state", ModelFailureDomain.REQUEST)
        handle._dispatch.attempted = True
        failed = False
        try:
            await self._reservations.mark_dispatched(reservation.execution_id)
        except Exception:  # noqa: BLE001 - sanitize storage/driver failures before dispatch
            failed = True
        if failed:
            raise_detached_error(
                ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            )

    async def release_before_dispatch(self, handle: AccountingReservation) -> None:
        """Release an unused reservation; repository state rejects post-dispatch release."""
        reservation = self._handle(handle)
        if handle._dispatch.attempted:
            raise ModelCompletionFailure("invalid_accounting_state", ModelFailureDomain.REQUEST)
        handle._dispatch.release_started = True
        released = await await_cancellation_safe_cleanup(
            self._release_pre_dispatch(reservation.execution_id, handle.governor_handle, handle._dispatch)
        )
        if not released:
            raise_detached_error(
                ModelCompletionFailure("accounting_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            )

    async def retain_ambiguous(self, handle: AccountingReservation) -> bool:
        """Keep worst-case durable exposure and finish the governor conservatively."""
        reservation = self._handle(handle)
        persisted = True
        try:
            await self._reservations.retain_ambiguous(reservation.execution_id)
        except Exception:  # noqa: BLE001 - retain chargeable state on arbitrary persistence failure
            persisted = False
            logger.warning("MCP completion ambiguous accounting persistence degraded")
        try:
            await self._governor.commit(handle.governor_handle)
        except Exception:  # noqa: BLE001 - post-call governor failure must not replace valid output
            persisted = False
            logger.warning("MCP completion conservative governor settlement degraded")
        return persisted

    async def reconcile(
        self,
        handle: AccountingReservation,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> bool:
        """Return degraded-accounting status without replacing a valid model result."""
        reservation = self._handle(handle)
        trusted_counts = (
            type(input_tokens) is int
            and 0 <= input_tokens <= reservation.reserved_input_tokens
            and type(output_tokens) is int
            and 0 <= output_tokens <= reservation.reserved_output_tokens
        )
        if not trusted_counts:
            input_tokens, output_tokens = reservation.reserved_input_tokens, reservation.reserved_output_tokens
        input_tokens = cast(int, input_tokens)
        output_tokens = cast(int, output_tokens)
        input_cost = _count(input_tokens * self._policy.input_cost_units_per_token)
        actuals = ReservationActuals(
            input_tokens, output_tokens, _count(input_cost + output_tokens * self._policy.output_cost_units_per_token)
        )

        async def writer(conn: Any, row: ProviderUsageReservation, amounts: ReservationActuals) -> None:
            await self._usage.insert_mcp_completion_usage(
                conn,
                row,
                amounts,
                estimated=not trusted_counts,
                input_cost_units=input_cost,
            )

        try:
            total_tokens = _count(input_tokens + output_tokens)
            # Finish the governor before making the reservation non-chargeable.
            # Failure at either boundary therefore leaves conservative exposure.
            await self._governor.commit(
                handle.governor_handle,
                actuals={
                    "requests": 1,
                    "tokens": total_tokens,
                    "cost_units": _governor_cost(total_tokens, self._policy),
                    "jobs": 1,
                },
            )
            await self._reservations.reconcile(reservation.execution_id, actuals, usage_writer=writer)
        except asyncio.CancelledError:
            await await_cancellation_safe_cleanup(self.retain_ambiguous(handle))
            raise
        except Exception:  # noqa: BLE001 - preserve valid output and conservative state on settlement failure
            logger.warning("MCP completion post-call accounting degraded")
            await self.retain_ambiguous(handle)
            return False
        return True
