"""Canonical admin-webhook registration lifecycle and availability rules."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Generic, NoReturn, Protocol, TypeAlias, TypeVar

from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    UNSET as REPOSITORY_UNSET,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AdminWebhookUnitOfWork,
    CommittedDeliveryOutcome,
    IdempotencyLookup,
    IdempotencyLookupKind,
    MigrationState,
    RegistrationInsert,
    RegistrationPatch,
    RegistrationTarget,
    StoredWebhookRegistration,
    WebhookRepositoryError,
    WebhookRepositoryErrorCode,
)

from .audit import MutationAction, MutationAudit, MutationAuditSink, MutationOutcome
from .catalog import (
    EVENT_API_VERSION,
    EVENT_CATALOG,
    WebhookCatalogItem,
    normalize_subscriptions,
)
from .config import AdminWebhookMode, AdminWebhookSettings
from .crypto import (
    ProtectedValue,
    WebhookKeyError,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
    load_webhook_key_ring,
)
from .domain import (
    AdminWebhookActivationCheck,
    AdminWebhookActivationPhase,
    AdminWebhookActivationReasonCode,
    DeliveryBacklogCounts,
    DeliveryCapabilityStatus,
    DeliveryComponentStatus,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryRuntimeComponent,
    DeliveryRuntimeReasonCode,
    DeliveryState,
    IdempotencyScope,
    ValidatedWebhookTarget,
    WebhookError,
    WebhookErrorCode,
    WebhookLimits,
    WebhookMigrationSummary,
    WebhookRegistration,
    WebhookStatus,
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
    parse_registration_etag,
    validate_idempotency_key,
    validate_webhook_target,
)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SIGNING_SECRET_PATTERN = re.compile(r"^whsec_[0-9a-f]{64}$")
_ACTIVE_ROTATION_PHASES = frozenset({"rewriting", "verifying", "awaiting_primary_cutover"})
_FAILED_AUDIT_CODES = frozenset(
    {
        WebhookErrorCode.KEY_UNAVAILABLE,
        WebhookErrorCode.KEY_CONFIGURATION_MISMATCH,
        WebhookErrorCode.KEY_ROTATION_IN_PROGRESS,
        WebhookErrorCode.DATABASE_BUSY,
    }
)
_ADMISSION_DENIAL_CODES = frozenset(
    {
        WebhookErrorCode.REGISTRATION_LIMIT,
        WebhookErrorCode.ACTIVE_LIMIT,
        WebhookErrorCode.DELIVERY_UNAVAILABLE,
        WebhookErrorCode.SECRET_ROTATION_REQUIRED,
    }
)


def evaluate_activation_readiness(
    status: WebhookStatus,
    *,
    phase: AdminWebhookActivationPhase,
    max_backlog_age_seconds: int,
) -> AdminWebhookActivationCheck:
    """Evaluate one read-only activation phase from sanitized current state."""

    if not isinstance(phase, AdminWebhookActivationPhase):
        raise TypeError("activation phase is invalid")
    if (
        isinstance(max_backlog_age_seconds, bool)
        or not isinstance(max_backlog_age_seconds, int)
        or not 1 <= max_backlog_age_seconds <= 86_400
    ):
        raise ValueError("activation backlog age is invalid")
    delivery = status.delivery
    schema_ready = bool(
        status.schema_ready
        and delivery.schema_ready
        and delivery.delivery_schema_ready
        and delivery.canonical_schema_version == 1
    )
    migration_complete = bool(status.migration.phase == "complete" and delivery.migration_complete)
    key_available = bool(status.key_state == WebhookKeyLoadCode.AVAILABLE.value and delivery.key_ready)
    key_ready = bool(key_available and delivery.key_primary_match)
    jobs_ready = bool(delivery.jobs_database_ready and delivery.queue_ready and delivery.job_type_ready)
    limits_ready = not (status.limits.registrations_over_limit or status.limits.active_registrations_over_limit)
    worker_ready = delivery.worker.ready
    reconciler_ready = delivery.reconciler.ready
    retention_ready = delivery.retention.ready
    runtime_ready = worker_ready and reconciler_ready and retention_ready
    oldest_age = delivery.oldest_nonterminal_age_seconds
    backlog_age_ready = oldest_age is None or oldest_age <= max_backlog_age_seconds
    required_mode = (
        AdminWebhookMode.MIGRATE.value if phase is AdminWebhookActivationPhase.PREDEPLOY else AdminWebhookMode.ON.value
    )
    reasons: list[AdminWebhookActivationReasonCode] = []
    if status.mode != required_mode:
        reasons.append(AdminWebhookActivationReasonCode.PHASE_MISMATCH)
    if not schema_ready:
        reasons.append(AdminWebhookActivationReasonCode.SCHEMA_UNREADY)
    if not migration_complete:
        reasons.append(AdminWebhookActivationReasonCode.MIGRATION_PENDING)
    if not key_available:
        reasons.append(AdminWebhookActivationReasonCode.KEY_UNAVAILABLE)
    elif not key_ready:
        reasons.append(AdminWebhookActivationReasonCode.KEY_CONFIGURATION_MISMATCH)
    if not jobs_ready:
        reasons.append(AdminWebhookActivationReasonCode.JOBS_UNAVAILABLE)
    if status.limits.registrations_over_limit:
        reasons.append(AdminWebhookActivationReasonCode.REGISTRATION_LIMIT_EXCEEDED)
    if status.limits.active_registrations_over_limit:
        reasons.append(AdminWebhookActivationReasonCode.ACTIVE_LIMIT_EXCEEDED)
    if phase is AdminWebhookActivationPhase.LIVE:
        if not worker_ready:
            reasons.append(AdminWebhookActivationReasonCode.WORKER_UNAVAILABLE)
        if not reconciler_ready:
            reasons.append(AdminWebhookActivationReasonCode.RECONCILER_UNAVAILABLE)
        if not retention_ready:
            reasons.append(AdminWebhookActivationReasonCode.RETENTION_UNAVAILABLE)
        if not backlog_age_ready:
            reasons.append(AdminWebhookActivationReasonCode.BACKLOG_AGE_EXCEEDED)
    return AdminWebhookActivationCheck(
        phase=phase,
        ready=not reasons,
        mode=status.mode,
        schema_ready=schema_ready,
        migration_complete=migration_complete,
        key_ready=key_ready,
        jobs_ready=jobs_ready,
        limits_ready=limits_ready,
        worker_ready=worker_ready,
        reconciler_ready=reconciler_ready,
        retention_ready=retention_ready,
        runtime_ready=runtime_ready,
        backlog_age_ready=backlog_age_ready,
        oldest_nonterminal_age_seconds=oldest_age,
        max_backlog_age_seconds=max_backlog_age_seconds,
        reason_codes=tuple(reasons),
    )


_CANCELLATION_TOKEN_DOMAIN = b"tldw-admin-webhook-cancel-v1\x00"


def _random_cancellation_seed() -> bytes:
    """Return a fresh 256-bit seed for deterministic cancellation tokens."""

    return secrets.token_bytes(32)


@dataclass(frozen=True)
class _CancellationTokenSource:
    """Issue transaction-local cancellation tokens from one random seed."""

    seed: bytes

    def __post_init__(self) -> None:
        """Reject seeds that are not exactly 256 bits."""

        if not isinstance(self.seed, bytes) or len(self.seed) != 32:
            raise ValueError("cancellation seed must be 256 bits")

    def attempt_factory(self) -> Callable[[], str]:
        """Return a factory that issues unique ordered HMAC tokens."""

        ordinal = 0

        def issue() -> str:
            """Issue the next lowercase SHA-256 cancellation token."""

            nonlocal ordinal
            token = hmac.new(
                self.seed,
                _CANCELLATION_TOKEN_DOMAIN + ordinal.to_bytes(8, "big"),
                hashlib.sha256,
            ).hexdigest()
            ordinal += 1
            return token

        return issue


class DeliveryCapability(Protocol):
    """Current sanitized delivery readiness supplied by the data plane."""

    async def status(self, now: datetime) -> DeliveryCapabilityStatus:
        """Return whether a registration may currently be activated."""


class _ControlMetrics(Protocol):
    """Observe committed control-plane mutations without affecting them."""

    def registration_counts(self, *, total: int, active: int) -> None:
        """Record the current committed registration gauges."""

        ...

    def admission_denied(self, reason: WebhookErrorCode) -> None:
        """Record a bounded registration-admission denial reason."""

        ...

    def delivery_committed(
        self,
        *,
        state: DeliveryState,
        kind: DeliveryKind,
        reason_code: DeliveryReasonCode | None,
        status_code: int | None,
    ) -> None:
        """Record a committed delivery transition using bounded labels."""

        ...


class UnavailableDeliveryCapability:
    """Fixed fail-closed capability for off, migrate, and isolated tests."""

    async def status(self, now: datetime) -> DeliveryCapabilityStatus:
        _utc(now)
        components = {
            DeliveryRuntimeComponent.WORKER: DeliveryRuntimeReasonCode.WORKER_UNAVAILABLE,
            DeliveryRuntimeComponent.RECONCILER: (DeliveryRuntimeReasonCode.RECONCILER_UNAVAILABLE),
            DeliveryRuntimeComponent.RETENTION: (DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE),
        }
        statuses = {
            component: DeliveryComponentStatus(
                component=component,
                ready=False,
                reason_code=reason,
                heartbeat_age_seconds=None,
            )
            for component, reason in components.items()
        }
        return DeliveryCapabilityStatus(
            canonical_schema_version=0,
            schema_ready=False,
            delivery_schema_ready=False,
            migration_complete=False,
            key_ready=False,
            key_primary_match=False,
            jobs_database_ready=False,
            queue_ready=False,
            job_type_ready=False,
            jobs_backend="unavailable",
            worker=statuses[DeliveryRuntimeComponent.WORKER],
            reconciler=statuses[DeliveryRuntimeComponent.RECONCILER],
            retention=statuses[DeliveryRuntimeComponent.RETENTION],
            backlog=DeliveryBacklogCounts(),
            oldest_nonterminal_age_seconds=None,
            acquisition_ready=False,
            acquisition_reason_code=DeliveryRuntimeReasonCode.SCHEMA_UNREADY,
            delivery_capability_ready=False,
        )


def _database_unavailable_delivery_status(
    *,
    migration: MigrationState,
    key_ring_result: WebhookKeyRingLoadResult,
) -> DeliveryCapabilityStatus:
    """Build a sanitized fail-closed status for an unavailable database."""

    reason = DeliveryRuntimeReasonCode.DATABASE_UNAVAILABLE
    components = {
        component: DeliveryComponentStatus(
            component=component,
            ready=False,
            reason_code=reason,
            heartbeat_age_seconds=None,
        )
        for component in DeliveryRuntimeComponent
    }
    ring = key_ring_result.ring
    key_ready = ring is not None
    return DeliveryCapabilityStatus(
        canonical_schema_version=migration.schema_version,
        schema_ready=migration.schema_version == 1,
        delivery_schema_ready=False,
        migration_complete=(migration.phase == "complete" and migration.completed_at is not None),
        key_ready=key_ready,
        key_primary_match=(key_ready and migration.active_primary_key_id == ring.primary_id),
        jobs_database_ready=False,
        queue_ready=False,
        job_type_ready=False,
        jobs_backend="unavailable",
        worker=components[DeliveryRuntimeComponent.WORKER],
        reconciler=components[DeliveryRuntimeComponent.RECONCILER],
        retention=components[DeliveryRuntimeComponent.RETENTION],
        backlog=DeliveryBacklogCounts(),
        oldest_nonterminal_age_seconds=None,
        acquisition_ready=False,
        acquisition_reason_code=reason,
        delivery_capability_ready=False,
    )


class _Omitted:
    __slots__ = ()


OMITTED = _Omitted()


@dataclass(frozen=True)
class RegistrationChanges:
    """Caller-visible PATCH fields; signing secrets are intentionally absent."""

    description: str | _Omitted = OMITTED
    url: str | _Omitted = OMITTED
    event_types: tuple[str, ...] | _Omitted = OMITTED
    active: bool | _Omitted = OMITTED
    timeout_seconds: int | _Omitted = OMITTED


def _validate_command_identity(
    *,
    actor_id: object,
    request_id: object,
    now: object,
    webhook_id: object | None = None,
) -> None:
    if isinstance(actor_id, bool) or not isinstance(actor_id, int) or actor_id < 1:
        raise ValueError("actor_id must be a positive integer")
    if webhook_id is not None and (isinstance(webhook_id, bool) or not isinstance(webhook_id, int) or webhook_id < 1):
        raise ValueError("webhook_id must be a positive integer")
    if not isinstance(request_id, str) or _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("request_id is invalid")
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")


@dataclass(frozen=True)
class CreateRegistrationCommand:
    actor_id: int
    idempotency_key: str
    url: str
    event_types: tuple[str, ...]
    description: str
    timeout_seconds: int
    request_id: str
    now: datetime

    def __post_init__(self) -> None:
        _validate_command_identity(
            actor_id=self.actor_id,
            request_id=self.request_id,
            now=self.now,
        )


@dataclass(frozen=True)
class PatchRegistrationCommand:
    actor_id: int
    webhook_id: int
    if_match: str | None
    changes: RegistrationChanges
    request_id: str
    now: datetime

    def __post_init__(self) -> None:
        _validate_command_identity(
            actor_id=self.actor_id,
            webhook_id=self.webhook_id,
            request_id=self.request_id,
            now=self.now,
        )
        if not isinstance(self.changes, RegistrationChanges):
            raise TypeError("changes must be RegistrationChanges")


@dataclass(frozen=True)
class DeleteRegistrationCommand:
    actor_id: int
    webhook_id: int
    if_match: str | None
    request_id: str
    now: datetime

    def __post_init__(self) -> None:
        _validate_command_identity(
            actor_id=self.actor_id,
            webhook_id=self.webhook_id,
            request_id=self.request_id,
            now=self.now,
        )


@dataclass(frozen=True)
class RotateSecretCommand:
    actor_id: int
    webhook_id: int
    if_match: str | None
    idempotency_key: str
    request_id: str
    now: datetime

    def __post_init__(self) -> None:
        _validate_command_identity(
            actor_id=self.actor_id,
            webhook_id=self.webhook_id,
            request_id=self.request_id,
            now=self.now,
        )


@dataclass(frozen=True)
class SecretMutationResult:
    registration: WebhookRegistration
    secret: str
    replayed: bool


@dataclass(frozen=True)
class MutationResult:
    registration: WebhookRegistration
    changed: bool


@dataclass(frozen=True)
class WebhookCatalog:
    api_version: str
    events: tuple[WebhookCatalogItem, ...]
    registration_limit: int
    active_limit: int


@dataclass(frozen=True)
class WebhookRegistrationPage:
    """One bounded offset page plus its server-side total."""

    items: tuple[WebhookRegistration, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True)
class _NormalizedCreate:
    description: str
    target: ValidatedWebhookTarget
    event_types: tuple[str, ...]
    timeout_seconds: int

    def request_body(self) -> Mapping[str, object]:
        return {
            "description": self.description,
            "event_types": list(self.event_types),
            "timeout_seconds": self.timeout_seconds,
            "url": self.target.url,
        }


@dataclass(frozen=True)
class _NormalizedChanges:
    description: str | _Omitted = OMITTED
    target: ValidatedWebhookTarget | _Omitted = OMITTED
    event_types: tuple[str, ...] | _Omitted = OMITTED
    active: bool | _Omitted = OMITTED
    timeout_seconds: int | _Omitted = OMITTED


@dataclass
class _MutationContext:
    actor_id: int
    action: MutationAction
    request_id: str
    webhook_id: int | None = None
    target_hostname: str | None = None
    event_types: tuple[str, ...] = ()
    emitted: bool = False
    accepted_emitted: bool = False

    def record(
        self,
        *,
        outcome: MutationOutcome,
        reason_code: WebhookErrorCode | None = None,
    ) -> MutationAudit:
        return MutationAudit(
            actor_id=self.actor_id,
            action=self.action,
            webhook_id=self.webhook_id,
            target_hostname=self.target_hostname,
            event_types=self.event_types,
            outcome=outcome,
            request_id=self.request_id,
            reason_code=reason_code,
        )


class _AuditSinkUnavailable(MandatoryAuditWriteError):
    pass


T = TypeVar("T")


@dataclass(frozen=True)
class _TransactionOutcome(Generic[T]):
    value: T
    audit_outcome: MutationOutcome
    delivery_outcomes: tuple[CommittedDeliveryOutcome, ...] = ()


_TransactionOperation: TypeAlias = Callable[
    [AdminWebhookUnitOfWork],
    Awaitable[_TransactionOutcome[T]],
]


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _validate_description(value: object) -> str:
    if not isinstance(value, str) or len(value) > 500:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    return value


def _validate_timeout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 30:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    return value


def _map_exception(exc: BaseException) -> WebhookError | None:
    if isinstance(exc, WebhookError):
        return exc
    if isinstance(exc, WebhookRepositoryError):
        mapping = {
            WebhookRepositoryErrorCode.DATABASE_BUSY: WebhookErrorCode.DATABASE_BUSY,
            WebhookRepositoryErrorCode.NOT_FOUND: WebhookErrorCode.NOT_FOUND,
            WebhookRepositoryErrorCode.STALE_REVISION: WebhookErrorCode.PRECONDITION_FAILED,
            WebhookRepositoryErrorCode.REGISTRATION_LIMIT: WebhookErrorCode.REGISTRATION_LIMIT,
            WebhookRepositoryErrorCode.ACTIVE_LIMIT: WebhookErrorCode.ACTIVE_LIMIT,
        }
        code = mapping.get(exc.code)
        return WebhookError(code) if code is not None else None
    if isinstance(exc, WebhookKeyError):
        return WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
    return None


def _error_audit_outcome(error: WebhookError | None) -> MutationOutcome:
    if error is None or error.code in _FAILED_AUDIT_CODES:
        return "failed"
    return "denied"


def _log_metrics_failure(metric: str, exc: BaseException) -> None:
    """Log only the metric name and exception type for fail-open observers."""

    logger.warning(
        "Admin webhook metrics update failed metric={} error_type={}",
        metric,
        type(exc).__name__,
    )


def _migration_registration_ids(value: object) -> set[int]:
    found: set[int] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, bool):
            continue
        if isinstance(current, int):
            if current > 0:
                found.add(current)
            continue
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return found


class AdminWebhookControlPlane:
    """Own canonical registration rules without SQL, HTTP, or route concerns."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        settings: AdminWebhookSettings,
        key_ring_result: WebhookKeyRingLoadResult,
        delivery_capability: DeliveryCapability,
        cancellation_seed_factory: Callable[[], bytes] = _random_cancellation_seed,
        metrics: _ControlMetrics | None = None,
    ) -> None:
        if not callable(cancellation_seed_factory):
            raise TypeError("cancellation seed factory is required")
        self._repository = repository
        self._settings = settings
        self._key_ring_result = key_ring_result
        self._delivery_capability = delivery_capability
        self._cancellation_seed_factory = cancellation_seed_factory
        self._metrics = metrics

    def _new_cancellation_token_source(self) -> _CancellationTokenSource:
        return _CancellationTokenSource(self._cancellation_seed_factory())

    async def _emit(
        self,
        context: _MutationContext,
        sink: MutationAuditSink,
        *,
        outcome: MutationOutcome,
        reason_code: WebhookErrorCode | None = None,
    ) -> None:
        try:
            await sink(context.record(outcome=outcome, reason_code=reason_code))
        except Exception as exc:
            raise _AuditSinkUnavailable from exc
        context.emitted = True
        if outcome == "accepted":
            context.accepted_emitted = True

    async def _raise_after_audit(
        self,
        context: _MutationContext,
        sink: MutationAuditSink,
        exc: BaseException,
    ) -> NoReturn:
        if isinstance(exc, _AuditSinkUnavailable):
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
        mapped = _map_exception(exc)
        try:
            await self._emit(
                context,
                sink,
                outcome=_error_audit_outcome(mapped),
                reason_code=mapped.code if mapped is not None else None,
            )
        except _AuditSinkUnavailable:
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
        if mapped is not None:
            raise mapped from None
        raise exc

    async def _attempt_correlated_failed_audit(
        self,
        context: _MutationContext,
        sink: MutationAuditSink,
        exc: BaseException,
    ) -> None:
        mapped = _map_exception(exc)
        try:
            await sink(
                context.record(
                    outcome="failed",
                    reason_code=mapped.code if mapped is not None else None,
                )
            )
        except Exception:  # noqa: BLE001 - a failed follow-up audit cannot mask commit failure
            return

    async def _run_transactional_mutation(
        self,
        context: _MutationContext,
        sink: MutationAuditSink,
        operation: _TransactionOperation[T],
    ) -> T:
        try:
            async with self._repository.transaction() as tx:
                try:
                    outcome = await operation(tx)
                except Exception as exc:  # noqa: BLE001 - boundary must audit unexpected failures
                    await self._raise_after_audit(context, sink, exc)
                await self._emit(
                    context,
                    sink,
                    outcome=outcome.audit_outcome,
                )
            value = outcome.value
            if self._metrics is not None and outcome.audit_outcome == "accepted":
                try:
                    counts = await self._repository.registration_counts()
                    self._metrics.registration_counts(
                        total=counts.total,
                        active=counts.active,
                    )
                except Exception as exc:  # noqa: BLE001 - metrics are fail-open
                    _log_metrics_failure("registration_counts", exc)
                for delivery_outcome in outcome.delivery_outcomes:
                    try:
                        self._metrics.delivery_committed(
                            state=delivery_outcome.state,
                            kind=delivery_outcome.kind,
                            reason_code=delivery_outcome.reason_code,
                            status_code=delivery_outcome.status_code,
                        )
                    except Exception as exc:  # noqa: BLE001 - metrics are fail-open
                        _log_metrics_failure("delivery_committed", exc)
                        continue
            return value
        except _AuditSinkUnavailable:
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
        except Exception as exc:
            if context.accepted_emitted:
                await self._attempt_correlated_failed_audit(context, sink, exc)
            if context.emitted:
                mapped = _map_exception(exc)
                if mapped is not None:
                    if self._metrics is not None and mapped.code in _ADMISSION_DENIAL_CODES:
                        try:
                            self._metrics.admission_denied(mapped.code)
                        except Exception as metric_exc:  # noqa: BLE001 - metrics are fail-open
                            _log_metrics_failure("admission_denied", metric_exc)
                    raise mapped from None
                raise
            await self._raise_after_audit(context, sink, exc)

    async def _prepare_or_audit(
        self,
        context: _MutationContext,
        sink: MutationAuditSink,
        prepare: Callable[[], Awaitable[T]],
    ) -> T:
        try:
            return await prepare()
        except Exception as exc:  # noqa: BLE001 - boundary must audit unexpected failures
            await self._raise_after_audit(context, sink, exc)

    async def _require_surface_available(self) -> MigrationState:
        if self._settings.mode is AdminWebhookMode.OFF:
            raise WebhookError(WebhookErrorCode.DISABLED)
        if self._settings.mode is AdminWebhookMode.MIGRATE:
            raise WebhookError(WebhookErrorCode.MIGRATION_PENDING)
        state = await self._repository.get_migration_state()
        self._require_migration_ready(state)
        return state

    @staticmethod
    def _require_migration_ready(state: MigrationState) -> None:
        if state.phase != "complete" or state.completed_at is None:
            raise WebhookError(WebhookErrorCode.MIGRATION_PENDING)

    def _require_protected_write_key(self, state: MigrationState) -> WebhookKeyRing:
        if state.rotation_phase in _ACTIVE_ROTATION_PHASES:
            raise WebhookError(WebhookErrorCode.KEY_ROTATION_IN_PROGRESS)
        ring = self._key_ring_result.ring
        if ring is None:
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
        if state.active_primary_key_id != ring.primary_id:
            raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
        return ring

    def _normalize_create(self, command: CreateRegistrationCommand) -> _NormalizedCreate:
        validate_idempotency_key(command.idempotency_key)
        description = _validate_description(command.description)
        timeout_seconds = _validate_timeout(command.timeout_seconds)
        if not isinstance(command.event_types, tuple):
            raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)
        event_types = normalize_subscriptions(command.event_types)
        target = validate_webhook_target(
            command.url,
            allow_http_dev=self._settings.allow_http_dev,
            allow_e2e_loopback=self._settings.allow_e2e_loopback,
        )
        return _NormalizedCreate(
            description=description,
            target=target,
            event_types=event_types,
            timeout_seconds=timeout_seconds,
        )

    def _normalize_changes(self, changes: RegistrationChanges) -> _NormalizedChanges:
        if all(
            value is OMITTED
            for value in (
                changes.description,
                changes.url,
                changes.event_types,
                changes.active,
                changes.timeout_seconds,
            )
        ):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
        description: str | _Omitted = OMITTED
        target: ValidatedWebhookTarget | _Omitted = OMITTED
        event_types: tuple[str, ...] | _Omitted = OMITTED
        active: bool | _Omitted = OMITTED
        timeout_seconds: int | _Omitted = OMITTED
        if changes.description is not OMITTED:
            description = _validate_description(changes.description)
        if changes.url is not OMITTED:
            if not isinstance(changes.url, str):
                raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
            target = validate_webhook_target(
                changes.url,
                allow_http_dev=self._settings.allow_http_dev,
                allow_e2e_loopback=self._settings.allow_e2e_loopback,
            )
        if changes.event_types is not OMITTED:
            if not isinstance(changes.event_types, tuple):
                raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)
            event_types = normalize_subscriptions(changes.event_types)
        if changes.active is not OMITTED:
            if not isinstance(changes.active, bool):
                raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
            active = changes.active
        if changes.timeout_seconds is not OMITTED:
            timeout_seconds = _validate_timeout(changes.timeout_seconds)
        return _NormalizedChanges(
            description=description,
            target=target,
            event_types=event_types,
            active=active,
            timeout_seconds=timeout_seconds,
        )

    async def create(
        self,
        command: CreateRegistrationCommand,
        *,
        audit_sink: MutationAuditSink,
    ) -> SecretMutationResult:
        context = _MutationContext(
            actor_id=command.actor_id,
            action="admin_webhook.create",
            request_id=command.request_id,
        )

        async def prepare() -> tuple[_NormalizedCreate, IdempotencyScope, str, str]:
            await self._require_surface_available()
            normalized = self._normalize_create(command)
            context.target_hostname = normalized.target.hostname
            context.event_types = normalized.event_types
            scope = build_idempotency_scope(
                actor_id=command.actor_id,
                operation="create",
                route="/admin/webhooks",
            )
            lookup_digest = idempotency_lookup_digest(command.idempotency_key, scope)
            request_fingerprint = canonical_request_hash(
                command.idempotency_key,
                scope=scope,
                body=normalized.request_body(),
                conditional_version=None,
            )
            return normalized, scope, lookup_digest, request_fingerprint

        normalized, scope_object, lookup_digest, request_fingerprint = await self._prepare_or_audit(
            context, audit_sink, prepare
        )

        async def operation(tx: AdminWebhookUnitOfWork) -> _TransactionOutcome[SecretMutationResult]:
            claim = await tx.claim_idempotency(
                lookup_digest=lookup_digest,
                scope=scope_object,
                request_fingerprint=request_fingerprint,
                now=_utc(command.now),
                expires_at=_utc(command.now) + timedelta(seconds=self._settings.idempotency_ttl_seconds),
            )
            if claim.kind is IdempotencyLookupKind.CONFLICT:
                raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
            if claim.kind is IdempotencyLookupKind.IN_PROGRESS:
                raise WebhookError(WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS)
            migration = await tx.lock_migration_state()
            self._require_migration_ready(migration)
            if claim.kind is IdempotencyLookupKind.REPLAY:
                replay = await self._resolve_secret_replay(
                    tx=tx,
                    claim=claim,
                    lookup_digest=lookup_digest,
                    migration=migration,
                )
                context.webhook_id = replay.registration.id
                context.target_hostname = replay.registration.target_hostname
                context.event_types = replay.registration.event_types
                return _TransactionOutcome(replay, "no_op")

            ring = self._require_protected_write_key(migration)
            await tx.enforce_registration_limit(limit=self._settings.registration_limit)
            webhook_id = await tx.allocate_registration_id()
            context.webhook_id = webhook_id
            secret_value = "whsec_" + secrets.token_hex(32)
            initial_version = 1
            target = ring.encrypt_text(
                purpose="registration.target",
                identity={
                    "registration_id": webhook_id,
                    "target_version": initial_version,
                },
                plaintext=normalized.target.url,
            )
            protected_secret = ring.encrypt_text(
                purpose="registration.secret",
                identity={
                    "registration_id": webhook_id,
                    "secret_version": initial_version,
                },
                plaintext=secret_value,
            )
            registration = await tx.insert_registration(
                RegistrationInsert(
                    id=webhook_id,
                    description=normalized.description,
                    target=RegistrationTarget(
                        protected=target,
                        hostname=normalized.target.hostname,
                        display=normalized.target.target_display,
                    ),
                    event_types=normalized.event_types,
                    active=False,
                    timeout_seconds=normalized.timeout_seconds,
                    secret=protected_secret,
                    secret_rotation_required=False,
                    actor_user_id=command.actor_id,
                    now=_utc(command.now),
                )
            )
            replay_secret = self._encrypt_replay_secret(
                ring=ring,
                lookup_digest=lookup_digest,
                registration=registration,
                secret=secret_value,
            )
            await tx.complete_idempotency(
                lookup_digest=lookup_digest,
                request_fingerprint=request_fingerprint,
                resource_id=registration.id,
                resource_version=registration.revision,
                secret_version=registration.secret_version,
                replay_secret=replay_secret,
                response_status=201,
                response_metadata={"result_kind": "created"},
                at=_utc(command.now),
            )
            await tx.mark_first_canonical_activity(
                "registration_mutation",
                _utc(command.now),
            )
            result = SecretMutationResult(
                registration=registration,
                secret=secret_value,
                replayed=False,
            )
            del secret_value
            return _TransactionOutcome(result, "accepted")

        return await self._run_transactional_mutation(context, audit_sink, operation)

    def _encrypt_replay_secret(
        self,
        *,
        ring: WebhookKeyRing,
        lookup_digest: str,
        registration: WebhookRegistration,
        secret: str,
    ) -> ProtectedValue:
        return ring.encrypt_text(
            purpose="idempotency.secret_replay",
            identity={
                "lookup_digest": lookup_digest,
                "registration_id": registration.id,
                "secret_version": registration.secret_version,
            },
            plaintext=secret,
        )

    async def _resolve_secret_replay(
        self,
        *,
        tx: AdminWebhookUnitOfWork,
        claim: IdempotencyLookup,
        lookup_digest: str,
        migration: MigrationState,
    ) -> SecretMutationResult:
        if (
            claim.resource_superseded
            or claim.resource_id is None
            or claim.secret_version is None
            or claim.replay_secret is None
        ):
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED)
        stored = await tx.get_protected_registration(
            claim.resource_id,
            include_deleted=True,
            lock=False,
        )
        if (
            stored is None
            or stored.registration.deleted_at is not None
            or stored.registration.secret_version != claim.secret_version
        ):
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED)
        ring = self._require_protected_write_key(migration)
        secret = ring.decrypt_text(
            purpose="idempotency.secret_replay",
            identity={
                "lookup_digest": lookup_digest,
                "registration_id": stored.registration.id,
                "secret_version": claim.secret_version,
            },
            protected=claim.replay_secret,
        )
        if _SIGNING_SECRET_PATTERN.fullmatch(secret) is None:
            raise RuntimeError("stored webhook replay secret is invalid")
        return SecretMutationResult(
            registration=stored.registration,
            secret=secret,
            replayed=True,
        )

    async def patch(
        self,
        command: PatchRegistrationCommand,
        *,
        audit_sink: MutationAuditSink,
    ) -> MutationResult:
        context = _MutationContext(
            actor_id=command.actor_id,
            action="admin_webhook.patch",
            request_id=command.request_id,
            webhook_id=command.webhook_id,
        )

        async def prepare() -> tuple[
            int,
            _NormalizedChanges,
            _CancellationTokenSource,
        ]:
            await self._require_surface_available()
            expected_revision = parse_registration_etag(
                command.if_match,
                expected_webhook_id=command.webhook_id,
            )
            normalized = self._normalize_changes(command.changes)
            if isinstance(normalized.target, ValidatedWebhookTarget):
                context.target_hostname = normalized.target.hostname
            if isinstance(normalized.event_types, tuple):
                context.event_types = normalized.event_types
            return (
                expected_revision,
                normalized,
                self._new_cancellation_token_source(),
            )

        expected_revision, normalized, cancellation_tokens = await self._prepare_or_audit(
            context,
            audit_sink,
            prepare,
        )
        delivery_status: DeliveryCapabilityStatus | None = None
        if normalized.active is True:
            try:
                delivery_status = await self._delivery_capability.status(_utc(command.now))
            except Exception:  # noqa: BLE001 - activation fails closed
                delivery_status = None

        async def operation(tx: AdminWebhookUnitOfWork) -> _TransactionOutcome[MutationResult]:
            disposition_token_factory = cancellation_tokens.attempt_factory()
            migration = await tx.lock_migration_state()
            self._require_migration_ready(migration)
            current = await tx.get_protected_registration(
                command.webhook_id,
                include_deleted=False,
                lock=True,
            )
            if current is None:
                raise WebhookError(WebhookErrorCode.NOT_FOUND)
            self._require_revision(current, expected_revision)
            context.target_hostname = current.registration.target_hostname
            context.event_types = current.registration.event_types

            repository_target: RegistrationTarget | object = REPOSITORY_UNSET
            if isinstance(normalized.target, ValidatedWebhookTarget):
                ring = self._require_protected_write_key(migration)
                current_url = ring.decrypt_text(
                    purpose="registration.target",
                    identity={
                        "registration_id": current.registration.id,
                        "target_version": current.registration.target_version,
                    },
                    protected=current.target,
                )
                if normalized.target.url != current_url:
                    next_target_version = current.registration.target_version + 1
                    repository_target = RegistrationTarget(
                        protected=ring.encrypt_text(
                            purpose="registration.target",
                            identity={
                                "registration_id": current.registration.id,
                                "target_version": next_target_version,
                            },
                            plaintext=normalized.target.url,
                        ),
                        hostname=normalized.target.hostname,
                        display=normalized.target.target_display,
                    )
                    context.target_hostname = normalized.target.hostname

            if normalized.active is True and not current.registration.active:
                ring = self._require_protected_write_key(migration)
                if current.registration.secret_rotation_required:
                    raise WebhookError(WebhookErrorCode.SECRET_ROTATION_REQUIRED)
                self._require_registration_decryptable(current, ring)
                if delivery_status is None or not delivery_status.delivery_capability_ready:
                    raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
                await tx.enforce_active_registration_limit(limit=self._settings.active_limit)

            repository_patch = RegistrationPatch(
                description=(normalized.description if isinstance(normalized.description, str) else REPOSITORY_UNSET),
                target=(repository_target if isinstance(repository_target, RegistrationTarget) else REPOSITORY_UNSET),
                event_types=(normalized.event_types if isinstance(normalized.event_types, tuple) else REPOSITORY_UNSET),
                active=(normalized.active if isinstance(normalized.active, bool) else REPOSITORY_UNSET),
                timeout_seconds=(
                    normalized.timeout_seconds if isinstance(normalized.timeout_seconds, int) else REPOSITORY_UNSET
                ),
            )
            patched = await tx.patch_registration(
                command.webhook_id,
                expected_revision=expected_revision,
                patch=repository_patch,
                actor_user_id=command.actor_id,
                at=_utc(command.now),
            )
            context.target_hostname = patched.registration.target_hostname
            context.event_types = patched.registration.event_types
            delivery_config_changed = (
                patched.registration.target_version != current.registration.target_version
                or patched.registration.event_types != current.registration.event_types
                or patched.registration.timeout_seconds != current.registration.timeout_seconds
            )
            cancellation_reason: DeliveryReasonCode | None = None
            delivery_outcomes: tuple[CommittedDeliveryOutcome, ...] = ()
            if current.registration.active and not patched.registration.active:
                cancellation_reason = DeliveryReasonCode.CANCELED_DISABLED
            elif delivery_config_changed:
                cancellation_reason = DeliveryReasonCode.SUPERSEDED_CONFIG
            if cancellation_reason is not None:
                terminal_batch = await tx.cancel_registration_work_with_outcomes(
                    command.webhook_id,
                    (
                        patched.registration.delivery_config_version,
                        patched.registration.secret_version,
                    ),
                    cancellation_reason,
                    disposition_token_factory,
                    _utc(command.now),
                )
                delivery_outcomes = terminal_batch.outcomes
            if patched.changed:
                await tx.mark_first_canonical_activity(
                    "registration_mutation",
                    _utc(command.now),
                )
            return _TransactionOutcome(
                MutationResult(
                    registration=patched.registration,
                    changed=patched.changed,
                ),
                "accepted" if patched.changed else "no_op",
                delivery_outcomes,
            )

        return await self._run_transactional_mutation(context, audit_sink, operation)

    @staticmethod
    def _require_revision(
        current: StoredWebhookRegistration,
        expected_revision: int,
    ) -> None:
        if current.registration.revision != expected_revision:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)

    @staticmethod
    def _require_registration_decryptable(
        current: StoredWebhookRegistration,
        ring: WebhookKeyRing,
    ) -> None:
        if not ring.can_decrypt(
            purpose="registration.target",
            identity={
                "registration_id": current.registration.id,
                "target_version": current.registration.target_version,
            },
            protected=current.target,
        ) or not ring.can_decrypt(
            purpose="registration.secret",
            identity={
                "registration_id": current.registration.id,
                "secret_version": current.registration.secret_version,
            },
            protected=current.secret,
        ):
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)

    async def delete(
        self,
        command: DeleteRegistrationCommand,
        *,
        audit_sink: MutationAuditSink,
    ) -> MutationResult:
        context = _MutationContext(
            actor_id=command.actor_id,
            action="admin_webhook.delete",
            request_id=command.request_id,
            webhook_id=command.webhook_id,
        )

        async def prepare() -> tuple[int, _CancellationTokenSource]:
            await self._require_surface_available()
            return (
                parse_registration_etag(
                    command.if_match,
                    expected_webhook_id=command.webhook_id,
                ),
                self._new_cancellation_token_source(),
            )

        expected_revision, cancellation_tokens = await self._prepare_or_audit(
            context,
            audit_sink,
            prepare,
        )

        async def operation(tx: AdminWebhookUnitOfWork) -> _TransactionOutcome[MutationResult]:
            disposition_token_factory = cancellation_tokens.attempt_factory()
            migration = await tx.lock_migration_state()
            self._require_migration_ready(migration)
            current = await tx.get_protected_registration(
                command.webhook_id,
                include_deleted=False,
                lock=True,
            )
            if current is None:
                raise WebhookError(WebhookErrorCode.NOT_FOUND)
            self._require_revision(current, expected_revision)
            context.target_hostname = current.registration.target_hostname
            context.event_types = current.registration.event_types
            deleted = await tx.soft_delete_registration(
                command.webhook_id,
                expected_revision=expected_revision,
                actor_user_id=command.actor_id,
                at=_utc(command.now),
            )
            terminal_batch = await tx.cancel_registration_work_with_outcomes(
                command.webhook_id,
                (
                    deleted.delivery_config_version,
                    deleted.secret_version,
                ),
                DeliveryReasonCode.CANCELED_DELETED,
                disposition_token_factory,
                _utc(command.now),
            )
            await tx.mark_first_canonical_activity(
                "registration_mutation",
                _utc(command.now),
            )
            return _TransactionOutcome(
                MutationResult(registration=deleted, changed=True),
                "accepted",
                terminal_batch.outcomes,
            )

        return await self._run_transactional_mutation(context, audit_sink, operation)

    async def rotate_secret(
        self,
        command: RotateSecretCommand,
        *,
        audit_sink: MutationAuditSink,
    ) -> SecretMutationResult:
        context = _MutationContext(
            actor_id=command.actor_id,
            action="admin_webhook.rotate_secret",
            request_id=command.request_id,
            webhook_id=command.webhook_id,
        )

        async def prepare() -> tuple[
            int,
            IdempotencyScope,
            str,
            str,
            _CancellationTokenSource,
        ]:
            await self._require_surface_available()
            validate_idempotency_key(command.idempotency_key)
            expected_revision = parse_registration_etag(
                command.if_match,
                expected_webhook_id=command.webhook_id,
            )
            scope = build_idempotency_scope(
                actor_id=command.actor_id,
                operation="rotate_secret",
                route=f"/admin/webhooks/{command.webhook_id}/rotate-secret",
                webhook_id=command.webhook_id,
            )
            lookup_digest = idempotency_lookup_digest(command.idempotency_key, scope)
            request_fingerprint = canonical_request_hash(
                command.idempotency_key,
                scope=scope,
                body={},
                conditional_version=expected_revision,
            )
            return (
                expected_revision,
                scope,
                lookup_digest,
                request_fingerprint,
                self._new_cancellation_token_source(),
            )

        (
            expected_revision,
            scope_object,
            lookup_digest,
            request_fingerprint,
            cancellation_tokens,
        ) = await self._prepare_or_audit(context, audit_sink, prepare)

        async def operation(tx: AdminWebhookUnitOfWork) -> _TransactionOutcome[SecretMutationResult]:
            disposition_token_factory = cancellation_tokens.attempt_factory()
            claim = await tx.claim_idempotency(
                lookup_digest=lookup_digest,
                scope=scope_object,
                request_fingerprint=request_fingerprint,
                now=_utc(command.now),
                expires_at=_utc(command.now) + timedelta(seconds=self._settings.idempotency_ttl_seconds),
            )
            if claim.kind is IdempotencyLookupKind.CONFLICT:
                raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
            if claim.kind is IdempotencyLookupKind.IN_PROGRESS:
                raise WebhookError(WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS)
            migration = await tx.lock_migration_state()
            self._require_migration_ready(migration)
            if claim.kind is IdempotencyLookupKind.REPLAY:
                replay = await self._resolve_secret_replay(
                    tx=tx,
                    claim=claim,
                    lookup_digest=lookup_digest,
                    migration=migration,
                )
                context.target_hostname = replay.registration.target_hostname
                context.event_types = replay.registration.event_types
                return _TransactionOutcome(replay, "no_op")

            current = await tx.get_protected_registration(
                command.webhook_id,
                include_deleted=False,
                lock=True,
            )
            if current is None:
                raise WebhookError(WebhookErrorCode.NOT_FOUND)
            self._require_revision(current, expected_revision)
            context.target_hostname = current.registration.target_hostname
            context.event_types = current.registration.event_types
            if current.registration.active:
                raise WebhookError(WebhookErrorCode.REGISTRATION_ACTIVE)
            ring = self._require_protected_write_key(migration)
            next_secret_version = current.registration.secret_version + 1
            secret_value = "whsec_" + secrets.token_hex(32)
            protected_secret = ring.encrypt_text(
                purpose="registration.secret",
                identity={
                    "registration_id": current.registration.id,
                    "secret_version": next_secret_version,
                },
                plaintext=secret_value,
            )
            patched = await tx.patch_registration(
                command.webhook_id,
                expected_revision=expected_revision,
                patch=RegistrationPatch(
                    secret=protected_secret,
                    secret_rotation_required=False,
                ),
                actor_user_id=command.actor_id,
                at=_utc(command.now),
            )
            registration = patched.registration
            terminal_batch = await tx.cancel_registration_work_with_outcomes(
                command.webhook_id,
                (
                    registration.delivery_config_version,
                    registration.secret_version,
                ),
                DeliveryReasonCode.CANCELED_SECRET_ROTATION,
                disposition_token_factory,
                _utc(command.now),
            )
            replay_secret = self._encrypt_replay_secret(
                ring=ring,
                lookup_digest=lookup_digest,
                registration=registration,
                secret=secret_value,
            )
            await tx.complete_idempotency(
                lookup_digest=lookup_digest,
                request_fingerprint=request_fingerprint,
                resource_id=registration.id,
                resource_version=registration.revision,
                secret_version=registration.secret_version,
                replay_secret=replay_secret,
                response_status=200,
                response_metadata={"result_kind": "rotated"},
                at=_utc(command.now),
            )
            await tx.mark_first_canonical_activity(
                "registration_mutation",
                _utc(command.now),
            )
            result = SecretMutationResult(
                registration=registration,
                secret=secret_value,
                replayed=False,
            )
            del secret_value
            return _TransactionOutcome(
                result,
                "accepted",
                terminal_batch.outcomes,
            )

        return await self._run_transactional_mutation(context, audit_sink, operation)

    async def catalog(self) -> WebhookCatalog:
        try:
            await self._require_surface_available()
        except Exception as exc:
            mapped = _map_exception(exc)
            if mapped is not None:
                raise mapped from None
            raise
        return WebhookCatalog(
            api_version=EVENT_API_VERSION,
            events=EVENT_CATALOG,
            registration_limit=self._settings.registration_limit,
            active_limit=self._settings.active_limit,
        )

    async def list_page(self, *, limit: int, offset: int) -> WebhookRegistrationPage:
        """Return one public offset page without exposing repository primitives."""
        if not 1 <= limit <= 100 or not 0 <= offset <= 1_000:
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
        try:
            await self._require_surface_available()
            items = await self._repository.list_registrations(
                limit=limit,
                offset=offset,
            )
            total = await self._repository.count_registrations()
        except Exception as exc:
            mapped = _map_exception(exc)
            if mapped is not None:
                raise mapped from None
            raise
        return WebhookRegistrationPage(
            items=tuple(items),
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get(self, webhook_id: int) -> WebhookRegistration:
        if isinstance(webhook_id, bool) or not isinstance(webhook_id, int) or webhook_id < 1:
            raise WebhookError(WebhookErrorCode.NOT_FOUND)
        try:
            await self._require_surface_available()
            registration = await self._repository.get_registration(webhook_id)
        except Exception as exc:
            mapped = _map_exception(exc)
            if mapped is not None:
                raise mapped from None
            raise
        if registration is None:
            raise WebhookError(WebhookErrorCode.NOT_FOUND)
        return registration

    async def status(self, *, now: datetime | None = None) -> WebhookStatus:
        observed_at = _utc(now or datetime.now(timezone.utc))
        try:
            migration = await self._repository.get_migration_state()
            registration_state = await self._repository.registration_limit_state(
                limit=self._settings.registration_limit
            )
            active_state = await self._repository.active_registration_limit_state(limit=self._settings.active_limit)
            secret_rotation_required = await self._repository.count_secret_rotation_required()
        except Exception as exc:
            mapped = _map_exception(exc)
            if mapped is not None:
                raise mapped from None
            raise
        try:
            delivery = await self._delivery_capability.status(observed_at)
        except Exception:  # noqa: BLE001 - status degrades on capability failure
            delivery = _database_unavailable_delivery_status(
                migration=migration,
                key_ring_result=self._key_ring_result,
            )
        mode_reason = {
            AdminWebhookMode.OFF: DeliveryRuntimeReasonCode.MODE_OFF,
            AdminWebhookMode.MIGRATE: DeliveryRuntimeReasonCode.MODE_MIGRATE,
        }.get(self._settings.mode)
        if mode_reason is not None:
            delivery = replace(
                delivery,
                acquisition_ready=False,
                acquisition_reason_code=mode_reason,
                delivery_capability_ready=False,
            )
        key_state = self._status_key_state(migration)
        imported_ids = _migration_registration_ids(migration.source_mapping)
        rollback_permitted = bool(
            migration.phase == "complete"
            and migration.rollback_retirement_phase == "retained"
            and migration.rollback_expires_at is not None
            and migration.rollback_expires_at > observed_at
            and migration.first_canonical_activity_at is None
            and migration.first_canonical_activity_kind is None
        )
        return WebhookStatus(
            mode=self._settings.mode.value,
            route_selection="canonical",
            schema_ready=migration.schema_version >= 1,
            key_state=key_state,
            delivery_capability_ready=delivery.delivery_capability_ready,
            delivery=delivery,
            limits=WebhookLimits(
                registrations=self._settings.registration_limit,
                active_registrations=self._settings.active_limit,
                current_registrations=registration_state.current,
                current_active_registrations=active_state.current,
                registrations_over_limit=registration_state.over_limit,
                active_registrations_over_limit=active_state.over_limit,
            ),
            migration=WebhookMigrationSummary(
                phase=migration.phase,
                imported_count=len(imported_ids),
                unresolved_count=0,
                rejected_count=len(migration.source_rejections),
                secret_rotation_required_count=secret_rotation_required,
                legacy_file_restore_permitted=rollback_permitted,
                rollback_expires_at=migration.rollback_expires_at,
            ),
        )

    def _status_key_state(self, migration: MigrationState) -> str:
        if migration.rotation_phase in _ACTIVE_ROTATION_PHASES:
            return WebhookErrorCode.KEY_ROTATION_IN_PROGRESS.value
        ring = self._key_ring_result.ring
        if ring is None:
            return f"{self._key_ring_result.code.value}"
        if migration.phase == "complete" and migration.active_primary_key_id != ring.primary_id:
            return WebhookErrorCode.KEY_CONFIGURATION_MISMATCH.value
        return f"{WebhookKeyLoadCode.AVAILABLE.value}"


async def get_admin_webhook_control_plane() -> AdminWebhookControlPlane:
    """Build a stateless service around the application-scoped AuthNZ pool."""
    pool = await get_db_pool()
    repository = AdminWebhookRepository(pool)
    settings = AdminWebhookSettings.from_environment(os.environ)
    key_ring_result = load_webhook_key_ring()
    from .observability import AdminWebhookMetrics

    metrics = AdminWebhookMetrics()
    try:
        counts = await repository.registration_counts()
        metrics.registration_counts(total=counts.total, active=counts.active)
    except Exception as exc:  # noqa: BLE001 - metrics initialization is fail-open
        _log_metrics_failure("registration_counts_initialization", exc)
    from .observability import (
        AdminWebhookDeliveryCapability,
        UnavailableJobsCapabilityProbe,
    )

    jobs_probe = UnavailableJobsCapabilityProbe()
    if settings.mode is AdminWebhookMode.ON:
        try:
            from tldw_Server_API.app.core.Jobs.manager import JobManager

            from .observability import JobManagerJobsCapabilityProbe

            manager = await asyncio.to_thread(JobManager)
            jobs_probe = JobManagerJobsCapabilityProbe(manager)
        except Exception:  # noqa: BLE001 - status remains available and fail-closed
            pass
    delivery_capability: DeliveryCapability = AdminWebhookDeliveryCapability(
        repository=repository,
        key_ring_result=key_ring_result,
        jobs_probe=jobs_probe,
        heartbeat_freshness_seconds=(settings.delivery_heartbeat_freshness_seconds),
        metrics=metrics,
    )
    return AdminWebhookControlPlane(
        repository=repository,
        settings=settings,
        key_ring_result=key_ring_result,
        delivery_capability=delivery_capability,
        metrics=metrics,
    )
