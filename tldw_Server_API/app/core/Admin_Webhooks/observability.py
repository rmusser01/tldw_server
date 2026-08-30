"""Closed health and metrics adapters for canonical webhook delivery."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricsRegistry,
    MetricType,
    get_metrics_registry,
)

from .catalog import EVENT_CATALOG
from .crypto import WebhookKeyRingLoadResult
from .domain import (
    DeliveryCapabilityStatus,
    DeliveryHealthSnapshot,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryRuntimeReasonCode,
    DeliveryState,
    WebhookErrorCode,
)

_CANONICAL_SCHEMA_VERSION = 1
_ALLOWED_LABEL_KEYS = frozenset(
    {
        "state",
        "kind",
        "event_type",
        "reason",
        "status_class",
        "component",
        "backend",
    }
)


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    metric_type: MetricType
    description: str
    labels: tuple[str, ...] = ()


_METRIC_SPECS = (
    _MetricSpec("admin_webhooks_registrations", MetricType.GAUGE, "Registrations by state", ("state",)),
    _MetricSpec(
        "admin_webhooks_admission_denials_total", MetricType.COUNTER, "Registration admission denials", ("reason",)
    ),
    _MetricSpec("admin_webhooks_events_total", MetricType.COUNTER, "Committed canonical events", ("event_type",)),
    _MetricSpec("admin_webhooks_fanout_total", MetricType.COUNTER, "Committed delivery fanout", ("event_type",)),
    _MetricSpec("admin_webhooks_enqueue_claims_total", MetricType.COUNTER, "Committed enqueue claims", ("backend",)),
    _MetricSpec(
        "admin_webhooks_enqueue_recoveries_total",
        MetricType.COUNTER,
        "Committed enqueue recoveries",
        ("reason", "backend"),
    ),
    _MetricSpec("admin_webhooks_enqueue_failures_total", MetricType.COUNTER, "Enqueue failures", ("reason", "backend")),
    _MetricSpec(
        "admin_webhooks_deliveries_total",
        MetricType.COUNTER,
        "Committed delivery outcomes",
        ("state", "kind", "reason", "status_class"),
    ),
    _MetricSpec(
        "admin_webhooks_attempts_total",
        MetricType.COUNTER,
        "Committed delivery attempts",
        ("kind", "reason", "status_class"),
    ),
    _MetricSpec(
        "admin_webhooks_attempt_latency_seconds",
        MetricType.HISTOGRAM,
        "Committed attempt latency",
        ("kind", "status_class"),
    ),
    _MetricSpec("admin_webhooks_retries_total", MetricType.COUNTER, "Committed delivery retries", ("reason",)),
    _MetricSpec("admin_webhooks_expiries_total", MetricType.COUNTER, "Committed delivery expiries"),
    _MetricSpec("admin_webhooks_backlog", MetricType.GAUGE, "Current nonterminal delivery backlog", ("state",)),
    _MetricSpec("admin_webhooks_oldest_nonterminal_age_seconds", MetricType.GAUGE, "Oldest nonterminal delivery age"),
    _MetricSpec("admin_webhooks_heartbeat_age_seconds", MetricType.GAUGE, "Runtime heartbeat age", ("component",)),
    _MetricSpec("admin_webhooks_heartbeat_ready", MetricType.GAUGE, "Runtime heartbeat readiness", ("component",)),
    _MetricSpec(
        "admin_webhooks_retention_deletions_total", MetricType.COUNTER, "Committed retention deletions", ("kind",)
    ),
    _MetricSpec("admin_webhooks_key_errors_total", MetricType.COUNTER, "Closed key readiness errors", ("reason",)),
    _MetricSpec(
        "admin_webhooks_migration_errors_total", MetricType.COUNTER, "Closed migration readiness errors", ("reason",)
    ),
    _MetricSpec("admin_webhooks_ssrf_denials_total", MetricType.COUNTER, "Closed SSRF denials", ("reason",)),
)
_METRIC_SPEC_BY_NAME = {spec.name: spec for spec in _METRIC_SPECS}


def _status_class(status_code: int | None) -> str:
    if status_code is None:
        return "none"
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise TypeError("status code is invalid")
    if 200 <= status_code <= 299:
        return "2xx"
    if 300 <= status_code <= 399:
        return "3xx"
    if 400 <= status_code <= 499:
        return "4xx"
    if 500 <= status_code <= 599:
        return "5xx"
    return "other"


class AdminWebhookMetrics:
    """Fail-open typed access to the fixed admin-webhook metric registry."""

    def __init__(self, *, registry: MetricsRegistry | object | None = None) -> None:
        self._registry = registry or get_metrics_registry()
        for spec in _METRIC_SPECS:
            try:
                self._registry.register_metric(
                    MetricDefinition(
                        name=spec.name,
                        type=spec.metric_type,
                        description=spec.description,
                        labels=list(spec.labels),
                    )
                )
            except Exception:  # noqa: BLE001 - registration is fail-open
                continue

    def _emit_counter(
        self,
        name: str,
        *,
        value: float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        spec = _METRIC_SPEC_BY_NAME.get(name)
        supplied = labels or {}
        if (
            spec is None
            or spec.metric_type is not MetricType.COUNTER
            or tuple(supplied) != spec.labels
            or not set(supplied) <= _ALLOWED_LABEL_KEYS
        ):
            raise ValueError("admin webhook metric schema is invalid")
        try:
            self._registry.increment(name, value, labels=supplied)
        except Exception:  # noqa: BLE001 - metrics are fail-open
            return

    def _emit_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        spec = _METRIC_SPEC_BY_NAME.get(name)
        supplied = labels or {}
        if (
            spec is None
            or spec.metric_type is not MetricType.GAUGE
            or tuple(supplied) != spec.labels
            or not set(supplied) <= _ALLOWED_LABEL_KEYS
        ):
            raise ValueError("admin webhook metric schema is invalid")
        try:
            self._registry.set_gauge(name, value, labels=supplied)
        except Exception:  # noqa: BLE001 - metrics are fail-open
            return

    def _emit_histogram(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str],
    ) -> None:
        spec = _METRIC_SPEC_BY_NAME.get(name)
        if (
            spec is None
            or spec.metric_type is not MetricType.HISTOGRAM
            or tuple(labels) != spec.labels
            or not set(labels) <= _ALLOWED_LABEL_KEYS
        ):
            raise ValueError("admin webhook metric schema is invalid")
        try:
            self._registry.observe(name, value, labels=labels)
        except Exception:  # noqa: BLE001 - metrics are fail-open
            return

    def events_committed(self, *, event_type: str, fanout_count: int) -> None:
        """Count one newly committed event and its bounded fanout."""
        if event_type not in EVENT_CATALOG:
            raise ValueError("event type metric value is invalid")
        if isinstance(fanout_count, bool) or fanout_count < 0:
            raise ValueError("fanout metric value is invalid")
        labels = {"event_type": event_type}
        self._emit_counter("admin_webhooks_events_total", labels=labels)
        self._emit_counter(
            "admin_webhooks_fanout_total",
            value=fanout_count,
            labels=labels,
        )

    def registration_counts(self, *, total: int, active: int) -> None:
        """Set current active and inactive registration gauges."""
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (total, active))
            or active > total
        ):
            raise ValueError("registration metric counts are invalid")
        self._emit_gauge(
            "admin_webhooks_registrations",
            active,
            labels={"state": "active"},
        )
        self._emit_gauge(
            "admin_webhooks_registrations",
            total - active,
            labels={"state": "inactive"},
        )

    def admission_denied(self, reason: WebhookErrorCode) -> None:
        """Count a closed registration or activation admission denial."""
        if reason not in {
            WebhookErrorCode.REGISTRATION_LIMIT,
            WebhookErrorCode.ACTIVE_LIMIT,
            WebhookErrorCode.DELIVERY_UNAVAILABLE,
            WebhookErrorCode.SECRET_ROTATION_REQUIRED,
        }:
            raise TypeError("admission denial metric reason is invalid")
        self._emit_counter(
            "admin_webhooks_admission_denials_total",
            labels={"reason": reason.value},
        )

    def enqueue_failure(self, failure: Enum, *, backend: str) -> None:
        """Count one closed reconciler enqueue failure."""
        reason = getattr(failure, "value", None)
        if reason not in {
            "admission_rejected",
            "backend_unavailable",
            "identity_conflict",
        }:
            raise TypeError("enqueue failure metric value is invalid")
        if backend not in {"sqlite", "postgres", "unavailable"}:
            raise ValueError("enqueue backend metric value is invalid")
        self._emit_counter(
            "admin_webhooks_enqueue_failures_total",
            labels={"reason": reason, "backend": backend},
        )

    def enqueue_success(self, success: Enum, *, backend: str) -> None:
        """Count one post-commit claim, queue attach, or disposition repair."""
        kind = getattr(success, "value", None)
        if kind not in {"claimed", "queued", "disposition_recovered"}:
            raise TypeError("enqueue success metric value is invalid")
        if backend not in {"sqlite", "postgres", "unavailable"}:
            raise ValueError("enqueue backend metric value is invalid")
        if kind == "claimed":
            self._emit_counter(
                "admin_webhooks_enqueue_claims_total",
                labels={"backend": backend},
            )
            return
        self._emit_counter(
            "admin_webhooks_enqueue_recoveries_total",
            labels={"reason": kind, "backend": backend},
        )

    def attempt_committed(
        self,
        *,
        state: DeliveryState,
        kind: DeliveryKind,
        reason_code: DeliveryReasonCode | None,
        status_code: int | None,
        latency_ms: int | None,
    ) -> None:
        """Observe one durable attempt outcome using only closed executor facts."""
        self.delivery_committed(
            state=state,
            kind=kind,
            reason_code=reason_code,
            status_code=status_code,
        )
        reason = reason_code.value if reason_code is not None else "none"
        status_class = _status_class(status_code)
        self._emit_counter(
            "admin_webhooks_attempts_total",
            labels={
                "kind": kind.value,
                "reason": reason,
                "status_class": status_class,
            },
        )
        if latency_ms is not None:
            if isinstance(latency_ms, bool) or latency_ms < 0:
                raise ValueError("attempt latency metric value is invalid")
            self._emit_histogram(
                "admin_webhooks_attempt_latency_seconds",
                latency_ms / 1_000,
                labels={"kind": kind.value, "status_class": status_class},
            )
        if state is DeliveryState.RETRY_WAIT:
            self._emit_counter(
                "admin_webhooks_retries_total",
                labels={"reason": reason},
            )
        if reason_code in {
            DeliveryReasonCode.TARGET_INVALID,
            DeliveryReasonCode.TARGET_REJECTED,
            DeliveryReasonCode.HTTP_HOP_DNS_ADDRESS_DENIED,
            DeliveryReasonCode.HTTP_HOP_PEER_VERIFICATION_FAILED,
        }:
            self._emit_counter(
                "admin_webhooks_ssrf_denials_total",
                labels={"reason": reason},
            )

    def expiries_committed(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("expiry metric count is invalid")
        if count:
            self._emit_counter("admin_webhooks_expiries_total", value=count)

    def retention_committed(self, result: object) -> None:
        """Count committed retention deletions by one fixed row category."""
        for field, kind in (
            ("deliveries", "delivery"),
            ("events", "event"),
            ("expired_idempotency", "idempotency"),
            ("heartbeats", "heartbeat"),
            ("registrations", "registration"),
        ):
            count = getattr(result, field, None)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TypeError("retention metric result is invalid")
            if count:
                self._emit_counter(
                    "admin_webhooks_retention_deletions_total",
                    value=count,
                    labels={"kind": kind},
                )

    def health_snapshot(self, status: DeliveryCapabilityStatus) -> None:
        """Set current gauges from exactly one sanitized capability snapshot."""
        if not isinstance(status, DeliveryCapabilityStatus):
            raise TypeError("delivery capability metric status is invalid")
        for state, count in status.backlog.__dict__.items():
            self._emit_gauge(
                "admin_webhooks_backlog",
                count,
                labels={"state": state},
            )
        self._emit_gauge(
            "admin_webhooks_oldest_nonterminal_age_seconds",
            status.oldest_nonterminal_age_seconds or 0,
        )
        for component in (status.worker, status.reconciler, status.retention):
            labels = {"component": component.component.value}
            self._emit_gauge(
                "admin_webhooks_heartbeat_age_seconds",
                component.heartbeat_age_seconds or 0,
                labels=labels,
            )
            self._emit_gauge(
                "admin_webhooks_heartbeat_ready",
                int(component.ready),
                labels=labels,
            )
        key_reason = None
        if not status.key_ready:
            key_reason = DeliveryRuntimeReasonCode.KEY_UNAVAILABLE
        elif not status.key_primary_match:
            key_reason = DeliveryRuntimeReasonCode.KEY_CONFIGURATION_MISMATCH
        if key_reason is not None:
            self._emit_counter(
                "admin_webhooks_key_errors_total",
                labels={"reason": key_reason.value},
            )
        migration_reason = None
        if not status.schema_ready or not status.delivery_schema_ready:
            migration_reason = DeliveryRuntimeReasonCode.SCHEMA_UNREADY
        elif not status.migration_complete:
            migration_reason = DeliveryRuntimeReasonCode.MIGRATION_PENDING
        if migration_reason is not None:
            self._emit_counter(
                "admin_webhooks_migration_errors_total",
                labels={"reason": migration_reason.value},
            )

    def delivery_committed(
        self,
        *,
        state: DeliveryState,
        kind: DeliveryKind,
        reason_code: DeliveryReasonCode | None,
        status_code: int | None,
    ) -> None:
        """Count one delivery outcome only after its durable commit."""
        if not isinstance(state, DeliveryState) or not isinstance(kind, DeliveryKind):
            raise TypeError("delivery metric values are invalid")
        if reason_code is not None and not isinstance(reason_code, DeliveryReasonCode):
            raise TypeError("delivery metric reason is invalid")
        self._emit_counter(
            "admin_webhooks_deliveries_total",
            labels={
                "state": state.value,
                "kind": kind.value,
                "reason": reason_code.value if reason_code is not None else "none",
                "status_class": _status_class(status_code),
            },
        )


@dataclass(frozen=True)
class JobsCapabilityStatus:
    """Closed Jobs readiness without row or connection detail."""

    database_ready: bool
    queue_ready: bool
    job_type_ready: bool
    backend: str

    def __post_init__(self) -> None:
        if self.backend not in {"sqlite", "postgres", "unavailable"}:
            raise ValueError("Jobs backend is invalid")
        if any(
            not isinstance(value, bool)
            for value in (
                self.database_ready,
                self.queue_ready,
                self.job_type_ready,
            )
        ):
            raise TypeError("Jobs readiness is invalid")


class _HealthRepository(Protocol):
    async def get_delivery_health_snapshot(self, **kwargs: object) -> DeliveryHealthSnapshot: ...


class _JobsProbe(Protocol):
    async def status(self) -> JobsCapabilityStatus: ...


class JobManagerJobsCapabilityProbe:
    """Bounded read-only preflight over the public Jobs manager surface."""

    def __init__(self, manager: object) -> None:
        self._manager = manager

    async def status(self) -> JobsCapabilityStatus:
        raw_backend = str(getattr(self._manager, "backend", "unavailable"))
        backend = raw_backend if raw_backend in {"sqlite", "postgres"} else "unavailable"
        queues = getattr(self._manager, "DOMAIN_ALLOWED_QUEUES", {})
        queue_ready = isinstance(queues, dict) and ADMIN_WEBHOOK_DELIVERY_QUEUE in queues.get(
            ADMIN_WEBHOOK_DELIVERY_DOMAIN, ()
        )
        allowed_job_types = {
            item.strip()
            for variable in (
                "JOBS_ALLOWED_JOB_TYPES",
                "JOBS_ALLOWED_JOB_TYPES_ADMIN_WEBHOOKS",
            )
            for item in os.getenv(variable, "").split(",")
            if item.strip()
        }
        job_type_ready = (
            ADMIN_WEBHOOK_DELIVERY_JOB_TYPE == "admin_webhook_delivery"
            and callable(getattr(self._manager, "admit_job", None))
            and (not allowed_job_types or ADMIN_WEBHOOK_DELIVERY_JOB_TYPE in allowed_job_types)
        )
        get_job = getattr(self._manager, "get_job", None)
        database_ready = False
        if callable(get_job) and backend != "unavailable":
            try:
                await asyncio.to_thread(get_job, 0)
                database_ready = True
            except Exception:  # noqa: BLE001 - readiness is fail-closed
                database_ready = False
        return JobsCapabilityStatus(
            database_ready=database_ready,
            queue_ready=queue_ready,
            job_type_ready=job_type_ready,
            backend=backend,
        )


class AdminWebhookDeliveryCapability:
    """Compose one current AuthNZ snapshot with one bounded Jobs probe."""

    def __init__(
        self,
        *,
        repository: _HealthRepository,
        key_ring_result: WebhookKeyRingLoadResult,
        jobs_probe: _JobsProbe,
        heartbeat_freshness_seconds: int,
        metrics: AdminWebhookMetrics | None = None,
    ) -> None:
        if not 1 <= heartbeat_freshness_seconds <= 60:
            raise ValueError("heartbeat freshness is invalid")
        self._repository = repository
        self._key_ring_result = key_ring_result
        self._jobs_probe = jobs_probe
        self._heartbeat_freshness_seconds = heartbeat_freshness_seconds
        self._metrics = metrics

    async def status(self, now: datetime) -> DeliveryCapabilityStatus:
        """Return one sanitized capability projection at ``now``."""
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("capability time must be timezone-aware")
        observed_at = now.astimezone(timezone.utc)
        ring = self._key_ring_result.ring
        health = await self._repository.get_delivery_health_snapshot(
            now=observed_at,
            heartbeat_freshness_seconds=self._heartbeat_freshness_seconds,
            key_available=ring is not None,
            expected_primary_key_id=(ring.primary_id if ring is not None else None),
        )
        jobs = await self._jobs_probe.status()
        checks = (
            (health.canonical_schema_version == _CANONICAL_SCHEMA_VERSION, DeliveryRuntimeReasonCode.SCHEMA_UNREADY),
            (health.delivery_schema_ready, DeliveryRuntimeReasonCode.SCHEMA_UNREADY),
            (health.migration_complete, DeliveryRuntimeReasonCode.MIGRATION_PENDING),
            (health.key_ready, DeliveryRuntimeReasonCode.KEY_UNAVAILABLE),
            (health.key_primary_match, DeliveryRuntimeReasonCode.KEY_CONFIGURATION_MISMATCH),
            (jobs.database_ready, DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE),
            (jobs.queue_ready, DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE),
            (jobs.job_type_ready, DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE),
            (
                health.reconciler.ready,
                health.reconciler.reason_code or DeliveryRuntimeReasonCode.RECONCILER_UNAVAILABLE,
            ),
        )
        acquisition_reason = next((reason for ready, reason in checks if not ready), None)
        acquisition_ready = acquisition_reason is None
        oldest_age = None
        if health.oldest_nonterminal_created_at is not None:
            oldest_age = max(
                0,
                int((observed_at - health.oldest_nonterminal_created_at).total_seconds()),
            )
        status = DeliveryCapabilityStatus(
            canonical_schema_version=health.canonical_schema_version,
            schema_ready=health.canonical_schema_version == _CANONICAL_SCHEMA_VERSION,
            delivery_schema_ready=health.delivery_schema_ready,
            migration_complete=health.migration_complete,
            key_ready=health.key_ready,
            key_primary_match=health.key_primary_match,
            jobs_database_ready=jobs.database_ready,
            queue_ready=jobs.queue_ready,
            job_type_ready=jobs.job_type_ready,
            jobs_backend=jobs.backend,
            worker=health.worker,
            reconciler=health.reconciler,
            retention=health.retention,
            backlog=health.backlog,
            oldest_nonterminal_age_seconds=oldest_age,
            acquisition_ready=acquisition_ready,
            acquisition_reason_code=acquisition_reason,
            delivery_capability_ready=acquisition_ready and health.worker.ready,
        )
        if self._metrics is not None:
            try:
                self._metrics.health_snapshot(status)
            except Exception:  # noqa: BLE001 - metrics are fail-open
                pass
        return status


__all__ = [
    "AdminWebhookDeliveryCapability",
    "AdminWebhookMetrics",
    "JobManagerJobsCapabilityProbe",
    "JobsCapabilityStatus",
]
