"""Durable, conservative admission and settlement for bounded MCP completions.

Every admission reads its canonical snapshot under the same scope lock used by
settlement. Unresolved reservations carry forward without age filtering. Legacy
Chat does not take these locks and is not atomically coordinated by this API.
Cost units are caller-frozen integer nanodollars; no prices or content are stored.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from uuid import RFC_4122, UUID

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import execute, fetch_all

MAX_INT = 2**63 - 1
_ERROR_CODES = frozenset(
    {
        "invalid_reservation",
        "integer_overflow",
        "duplicate_execution",
        "quota_exceeded",
        "snapshot_unavailable",
        "reservation_not_found",
        "invalid_transition",
        "conflicting_replay",
        "actuals_exceed_reservation",
        "usage_write_failed",
        "reservation_store_unavailable",
    }
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


class ProviderUsageReservationError(Exception):
    """Stable, content-free error with no retained backend exception graph."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_CODES else "reservation_store_unavailable"
        super().__init__(self.code)


def _repository_errors(method: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """Detach exception context after transaction cleanup, preserving cancellation."""

    @wraps(method)
    async def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        code = "reservation_store_unavailable"
        try:
            return await method(*args, **kwargs)
        except ProviderUsageReservationError as exc:
            code = exc.code
        except Exception:  # noqa: BLE001 - sanitize backend failures; not cancellation
            code = "reservation_store_unavailable"
        raise ProviderUsageReservationError(code)

    return guarded


def _integer(value: Any, *, positive: bool = False, code: str = "invalid_reservation") -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ProviderUsageReservationError(code)
    if value > MAX_INT:
        raise ProviderUsageReservationError("integer_overflow")
    return value


def _sum(*values: int) -> int:
    total = 0
    for value in values:
        total += _integer(value)
        if total > MAX_INT:
            raise ProviderUsageReservationError("integer_overflow")
    return total


def _execution_id(value: Any) -> str:
    valid = False
    if type(value) is str:
        try:
            parsed = UUID(value)
            valid = str(parsed) == value and parsed.version == 4 and parsed.variant == RFC_4122
        except ValueError:
            pass
    if not valid:
        raise ProviderUsageReservationError("invalid_reservation")
    return value


@dataclass(frozen=True)
class BillingScope:
    """Exact authenticated billing identity, never inferred from memberships."""

    kind: str
    value: int

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in {"user", "team", "org"}:
            raise ProviderUsageReservationError("invalid_reservation")
        _integer(self.value, positive=True)


@dataclass(frozen=True)
class ProviderUsageReservation:
    """Immutable, content-free worst-case provider exposure."""

    execution_id: str
    user_id: int
    active_team_id: int | None
    active_organization_id: int | None
    billing_scope: BillingScope
    provider: str
    model: str
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_cost_units: int

    def __post_init__(self) -> None:
        _execution_id(self.execution_id)
        _integer(self.user_id, positive=True)
        for value in (self.active_team_id, self.active_organization_id):
            if value is not None:
                _integer(value, positive=True)
        if type(self.billing_scope) is not BillingScope:
            raise ProviderUsageReservationError("invalid_reservation")
        expected = (
            BillingScope("org", self.active_organization_id)
            if self.active_organization_id is not None
            else (
                BillingScope("team", self.active_team_id)
                if self.active_team_id is not None
                else BillingScope("user", self.user_id)
            )
        )
        if self.billing_scope != expected:
            raise ProviderUsageReservationError("invalid_reservation")
        for value in (self.provider, self.model):
            if type(value) is not str or not value.strip():
                raise ProviderUsageReservationError("invalid_reservation")
        _sum(self.reserved_input_tokens, self.reserved_output_tokens)
        _integer(self.reserved_cost_units)


@dataclass(frozen=True)
class ReservationQuotaSnapshot:
    """Uncached canonical actual usage plus explicit limits; None is unlimited."""

    used_tokens: int
    token_limit: int | None
    used_cost_units: int
    cost_limit: int | None

    def __post_init__(self) -> None:
        for value in (self.used_tokens, self.used_cost_units):
            _integer(value, code="snapshot_unavailable")
        for value in (self.token_limit, self.cost_limit):
            if value is not None:
                _integer(value, code="snapshot_unavailable")


@dataclass(frozen=True)
class ReservationActuals:
    """Bounded actual input/output tokens and exact integer cost units."""

    input_tokens: int
    output_tokens: int
    cost_units: int

    def __post_init__(self) -> None:
        _sum(self.input_tokens, self.output_tokens)
        _integer(self.cost_units)


SnapshotReader = Callable[[Any], Awaitable[ReservationQuotaSnapshot]]
UsageWriter = Callable[[Any, ProviderUsageReservation, ReservationActuals], Awaitable[None]]


@dataclass
class ProviderUsageReservationsRepo:
    """State machine whose callbacks must use only the supplied transaction."""

    db_pool: DatabasePool

    def _pg(self) -> bool:
        return self.db_pool.pool is not None

    async def _execute(self, conn: Any, sql: str, *args: Any) -> None:
        await execute(conn, self._pg(), sql, args)

    async def _rows(self, conn: Any, sql: str, *args: Any) -> list[dict[str, Any]]:
        return await fetch_all(conn, self._pg(), sql, args, ())

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Any]:
        # DatabasePool translates body exceptions. Retain only our fixed code,
        # and publish a fresh exception outside all handlers after rollback.
        code = None
        failed = False
        try:
            async with self.db_pool.transaction() as conn:
                try:
                    if self._pg():
                        # Lock waiters must see the preceding holder's commit, not
                        # a snapshot captured before the scope row lock was acquired.
                        await self._execute(conn, "SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    yield conn
                except ProviderUsageReservationError as exc:
                    code = exc.code
                    raise
        except Exception:  # noqa: BLE001 - rollback completed before sanitized publication
            failed = True
        if failed:
            raise ProviderUsageReservationError(code or "reservation_store_unavailable")

    async def _lock(self, conn: Any, scope: BillingScope) -> None:
        await self._execute(
            conn,
            "INSERT INTO provider_usage_scope_locks (billing_scope_type, billing_scope_id) "
            "VALUES (?, ?) ON CONFLICT (billing_scope_type, billing_scope_id) DO NOTHING",
            scope.kind,
            scope.value,
        )
        if self._pg():
            await self._rows(
                conn,
                "SELECT billing_scope_id FROM provider_usage_scope_locks "
                "WHERE billing_scope_type = ? AND billing_scope_id = ? FOR UPDATE",
                scope.kind,
                scope.value,
            )

    def _now(self) -> datetime | str:
        now = datetime.now(timezone.utc)
        return now if self._pg() else now.isoformat()

    async def _get(self, conn: Any, execution_id: str) -> dict[str, Any] | None:
        result = await self._rows(
            conn, "SELECT * FROM provider_usage_reservations " "WHERE execution_id = ?", execution_id
        )
        return result[0] if result else None

    async def _locked_row(self, conn: Any, execution_id: str) -> dict[str, Any]:
        row = await self._get(conn, execution_id)
        if row is None:
            raise ProviderUsageReservationError("reservation_not_found")
        await self._lock(conn, BillingScope(row["billing_scope_type"], row["billing_scope_id"]))
        # The first read establishes immutable scope only; re-read state after locking.
        row = await self._get(conn, execution_id)
        if row is None:
            raise ProviderUsageReservationError("reservation_not_found")
        return row

    @staticmethod
    def _reservation(row: dict[str, Any]) -> ProviderUsageReservation:
        return ProviderUsageReservation(
            execution_id=row["execution_id"],
            user_id=row["user_id"],
            active_team_id=row["active_team_id"],
            active_organization_id=row["active_organization_id"],
            billing_scope=BillingScope(row["billing_scope_type"], row["billing_scope_id"]),
            provider=row["provider"],
            model=row["model"],
            reserved_input_tokens=row["reserved_input_tokens"],
            reserved_output_tokens=row["reserved_output_tokens"],
            reserved_cost_units=row["reserved_cost_units"],
        )

    async def _outstanding(self, conn: Any, scope: BillingScope) -> dict[str, int]:
        result = await self._rows(
            conn,
            "SELECT reserved_input_tokens, reserved_output_tokens, reserved_cost_units "
            "FROM provider_usage_reservations WHERE billing_scope_type = ? AND billing_scope_id = ? "
            "AND state IN ('reserved', 'dispatched', 'ambiguous')",
            scope.kind,
            scope.value,
        )
        tokens = cost = 0
        for row in result:
            tokens = _sum(tokens, row["reserved_input_tokens"], row["reserved_output_tokens"])
            cost = _sum(cost, row["reserved_cost_units"])
        return {"tokens": tokens, "cost_units": cost}

    @_repository_errors
    async def reserve(
        self,
        reservation: ProviderUsageReservation,
        *,
        snapshot_reader: SnapshotReader,
    ) -> dict[str, Any]:
        """Commit exposure only after locked, fail-closed quota reads."""
        if type(reservation) is not ProviderUsageReservation or not callable(snapshot_reader):
            raise ProviderUsageReservationError("invalid_reservation")
        async with self._transaction() as conn:
            await self._lock(conn, reservation.billing_scope)
            if await self._get(conn, reservation.execution_id) is not None:
                raise ProviderUsageReservationError("duplicate_execution")
            snapshot = None
            try:
                snapshot = await snapshot_reader(conn)
            except Exception:  # noqa: BLE001 - required snapshot callback fails closed
                snapshot = None
            if type(snapshot) is not ReservationQuotaSnapshot:
                raise ProviderUsageReservationError("snapshot_unavailable")
            outstanding = await self._outstanding(conn, reservation.billing_scope)
            tokens = _sum(
                snapshot.used_tokens,
                outstanding["tokens"],
                reservation.reserved_input_tokens,
                reservation.reserved_output_tokens,
            )
            cost = _sum(snapshot.used_cost_units, outstanding["cost_units"], reservation.reserved_cost_units)
            if (snapshot.token_limit is not None and tokens > snapshot.token_limit) or (
                snapshot.cost_limit is not None and cost > snapshot.cost_limit
            ):
                raise ProviderUsageReservationError("quota_exceeded")
            now = self._now()
            inserted = await self._rows(
                conn,
                """INSERT INTO provider_usage_reservations
                (execution_id, user_id, active_team_id, active_organization_id,
                 billing_scope_type, billing_scope_id, provider, model,
                 reserved_input_tokens, reserved_output_tokens, reserved_cost_units,
                 state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
                ON CONFLICT (execution_id) DO NOTHING RETURNING execution_id""",
                reservation.execution_id,
                reservation.user_id,
                reservation.active_team_id,
                reservation.active_organization_id,
                reservation.billing_scope.kind,
                reservation.billing_scope.value,
                reservation.provider,
                reservation.model,
                reservation.reserved_input_tokens,
                reservation.reserved_output_tokens,
                reservation.reserved_cost_units,
                now,
                now,
            )
            if not inserted:
                raise ProviderUsageReservationError("duplicate_execution")
            row = await self._get(conn, reservation.execution_id)
        return row

    async def _transition(self, execution_id: str, source: str, target: str) -> dict[str, Any]:
        _execution_id(execution_id)
        async with self._transaction() as conn:
            row = await self._locked_row(conn, execution_id)
            if row["state"] != target:
                if row["state"] != source:
                    raise ProviderUsageReservationError("invalid_transition")
                now = self._now()
                await self._execute(
                    conn,
                    "UPDATE provider_usage_reservations SET state = ?, updated_at = ?, "
                    "dispatched_at = ?, resolved_at = ? WHERE execution_id = ?",
                    target,
                    now,
                    now if target == "dispatched" else row["dispatched_at"],
                    now if target == "released" else None,
                    execution_id,
                )
                row = await self._get(conn, execution_id)
        return row

    @_repository_errors
    async def mark_dispatched(self, execution_id: str) -> dict[str, Any]:
        """Record dispatch; replay acknowledges state but never authorizes another call."""
        return await self._transition(execution_id, "reserved", "dispatched")

    @_repository_errors
    async def release_before_dispatch(self, execution_id: str) -> dict[str, Any]:
        """Release only exposure known never to have been dispatched."""
        return await self._transition(execution_id, "reserved", "released")

    @_repository_errors
    async def retain_ambiguous(self, execution_id: str) -> dict[str, Any]:
        """Permanently retain uncertain exposure; late settlement is forbidden."""
        return await self._transition(execution_id, "dispatched", "ambiguous")

    @_repository_errors
    async def reconcile(
        self,
        execution_id: str,
        actuals: ReservationActuals,
        *,
        usage_writer: UsageWriter,
    ) -> dict[str, Any]:
        """Insert canonical usage and settle atomically under the admission scope lock."""
        _execution_id(execution_id)
        if type(actuals) is not ReservationActuals or not callable(usage_writer):
            raise ProviderUsageReservationError("invalid_reservation")
        async with self._transaction() as conn:
            row = await self._locked_row(conn, execution_id)
            actual_values = (actuals.input_tokens, actuals.output_tokens, actuals.cost_units)
            stored_values = (row["actual_input_tokens"], row["actual_output_tokens"], row["actual_cost_units"])
            if row["state"] == "reconciled":
                if actual_values != stored_values:
                    raise ProviderUsageReservationError("conflicting_replay")
            else:
                if row["state"] != "dispatched":
                    raise ProviderUsageReservationError("invalid_transition")
                ceilings = (row["reserved_input_tokens"], row["reserved_output_tokens"], row["reserved_cost_units"])
                if any(actual > ceiling for actual, ceiling in zip(actual_values, ceilings)):
                    raise ProviderUsageReservationError("actuals_exceed_reservation")
                wrote = False
                try:
                    await usage_writer(conn, self._reservation(row), actuals)
                    wrote = True
                except Exception:  # noqa: BLE001 - callback errors must not expose provider data
                    wrote = False
                if not wrote:
                    raise ProviderUsageReservationError("usage_write_failed")
                now = self._now()
                await self._execute(
                    conn,
                    "UPDATE provider_usage_reservations SET state = 'reconciled', "
                    "actual_input_tokens = ?, actual_output_tokens = ?, actual_cost_units = ?, "
                    "updated_at = ?, resolved_at = ? WHERE execution_id = ?",
                    *actual_values,
                    now,
                    now,
                    execution_id,
                )
                row = await self._get(conn, execution_id)
        return row

    @_repository_errors
    async def get(self, execution_id: str) -> dict[str, Any] | None:
        """Read a durable row; return None only on authoritative absence."""
        _execution_id(execution_id)
        async with self._transaction() as conn:
            row = await self._get(conn, execution_id)
        return row

    @_repository_errors
    async def outstanding(self, billing_scope: BillingScope) -> dict[str, int]:
        """Return checked totals for all unresolved rows, regardless of age."""
        if type(billing_scope) is not BillingScope:
            raise ProviderUsageReservationError("invalid_reservation")
        async with self._transaction() as conn:
            await self._lock(conn, billing_scope)
            result = await self._outstanding(conn, billing_scope)
        return result
