"""Shallow completion-only data contracts, not yet routed by production Jobs.

Caller inputs and prepared observer arguments retain their original identity.
Only internal result disposition and preparation order are validated here; no
policy evaluation, driver imports, environment reads or observer dispatch occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class CompleteJobCommand:
    """Retain unconverted facade inputs and its two explicitly captured flags."""

    job_id: int
    result: dict[str, Any] | None
    worker_id: str | None
    lease_id: str | None
    completion_token: str | None
    expected_uuid: str | None
    enforce: bool
    outbox_enabled: bool


class CompletionOutcome(str, Enum):
    """Distinguish a new transition from replay and rejection."""

    APPLIED = "applied"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    NO_TRANSITION = "no_transition"


class CompletionNoTransitionReason(str, Enum):
    """Describe only rejection facts already available to completion."""

    MISSING_ROW = "missing_row"
    IDENTITY_MISMATCH = "identity_mismatch"
    TERMINAL_STATE = "terminal_state"
    GUARD_REJECTED = "guard_rejected"


class CompletionOrigin(str, Enum):
    """Identify the counter bucket from which completion transitioned."""

    PROCESSING = "processing"
    QUEUED = "queued"


@dataclass(frozen=True)
class JsonTruncatedEffect:
    """Prepared optional truncation metric with an attempt-local binding."""

    binding_id: int
    job: dict[str, Any]
    kind: Literal["result"] = "result"


@dataclass(frozen=True)
class DurationEffect:
    """Prepared duration metric, preserving raw backend-specific timestamps."""

    binding_id: int
    job: dict[str, Any]
    started_at: Any
    completed_at: Any


@dataclass(frozen=True)
class CompletedEffect:
    """Prepared completed counter metric, not the final lifecycle observer."""

    binding_id: int
    job: dict[str, Any]


@dataclass(frozen=True)
class GaugesEffect:
    """Prepared gauge labels for the observer's keyword arguments."""

    binding_id: int
    domain: Any
    queue: Any
    job_type: Any


@dataclass(frozen=True)
class TraceEffect:
    """Prepared completion trace using the original tracing projection."""

    binding_id: int
    job: dict[str, Any]
    operation: Literal["job.complete"] = "job.complete"


@dataclass(frozen=True)
class SlaWarningEffect:
    """Prepared warning for an optional SLA persistence failure."""

    binding_id: int
    job_id: int
    error_type: str


@dataclass(frozen=True)
class SlaEventEffect:
    """Prepared SLA lifecycle observer independent of outbox persistence."""

    binding_id: int
    job: dict[str, Any]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class SlaMetricEffect:
    """Prepared SLA duration-breach metric after its lifecycle observer."""

    binding_id: int
    job: dict[str, Any]
    kind: Literal["duration"] = "duration"


@dataclass(frozen=True)
class CompletionEventEffect:
    """Prepared final completion lifecycle observer with no attributes."""

    binding_id: int
    job: dict[str, Any]
    attrs: None = None


CompletionEffect = (
    JsonTruncatedEffect
    | DurationEffect
    | CompletedEffect
    | GaugesEffect
    | TraceEffect
    | SlaWarningEffect
    | SlaEventEffect
    | SlaMetricEffect
    | CompletionEventEffect
)


@dataclass(frozen=True)
class CompletionDurableEvent:
    """An inserted outbox fact, containing neither a callback nor a binding ID."""

    event_type: Literal["job.sla_breached", "job.completed"]
    job: dict[str, Any]
    attrs: dict[str, Any] | None


def _validate_effects(effects: tuple[CompletionEffect, ...]) -> None:
    """Accept optional metric prefixes followed by SLA and final completion."""
    metric_types = (DurationEffect, CompletedEffect, GaugesEffect, TraceEffect)
    allowed_types = (
        JsonTruncatedEffect,
        *metric_types,
        SlaWarningEffect,
        SlaEventEffect,
        SlaMetricEffect,
        CompletionEventEffect,
    )
    types = tuple(type(effect) for effect in effects)
    if any(effect_type not in allowed_types for effect_type in types):
        raise ValueError("Unknown completion effect")
    bindings = [effect.binding_id for effect in effects]
    if len(set(bindings)) != len(bindings):
        raise ValueError("Completion binding IDs must be unique within an attempt")

    position = int(bool(types) and types[0] is JsonTruncatedEffect)
    # Binding/argument preparation may stop anywhere in this suppression group.
    for metric_type in metric_types:
        if position == len(types) or types[position] is not metric_type:
            break
        position += 1
    if types[position:] not in (
        (CompletionEventEffect,),
        (SlaWarningEffect, CompletionEventEffect),
        (SlaEventEffect, SlaMetricEffect, CompletionEventEffect),
    ):
        raise ValueError("Invalid completion effect preparation order")


def _validate_durable_events(events: tuple[CompletionDurableEvent, ...]) -> None:
    """Require inserted SLA facts to precede the single completion fact."""
    if any(not isinstance(event, CompletionDurableEvent) for event in events):
        raise ValueError("Unknown completion durable event")
    if tuple(event.event_type for event in events) not in (
        (),
        ("job.completed",),
        ("job.sla_breached", "job.completed"),
    ):
        raise ValueError("Invalid completion durable event order")


@dataclass(frozen=True)
class CompleteJobResult:
    """Validated internal disposition retaining shallow prepared data references."""

    outcome: CompletionOutcome
    reason: CompletionNoTransitionReason | None = None
    origin: CompletionOrigin | None = None
    locked_row: dict[str, Any] | None = None
    effects: tuple[CompletionEffect, ...] = ()
    durable_events: tuple[CompletionDurableEvent, ...] = ()

    def __post_init__(self) -> None:
        """Validate internal result consistency without inspecting job payloads."""
        if not isinstance(self.outcome, CompletionOutcome):
            raise ValueError("Unknown completion outcome")
        if self.outcome is CompletionOutcome.APPLIED:
            if self.reason is not None or not isinstance(self.origin, CompletionOrigin) or self.locked_row is None:
                raise ValueError("Applied completion requires row/origin and no reason")
            _validate_effects(self.effects)
            _validate_durable_events(self.durable_events)
            return
        if self.origin is not None or self.locked_row is not None or self.effects or self.durable_events:
            raise ValueError("Non-applied completion cannot carry transition payload")
        if self.outcome is CompletionOutcome.NO_TRANSITION:
            if not isinstance(self.reason, CompletionNoTransitionReason):
                raise ValueError("No-transition completion requires a reason")
        elif self.reason is not None:
            raise ValueError("Completion replay cannot carry a reason")

    @classmethod
    def applied(
        cls,
        *,
        origin: CompletionOrigin,
        locked_row: dict[str, Any],
        effects: tuple[CompletionEffect, ...],
        durable_events: tuple[CompletionDurableEvent, ...],
    ) -> CompleteJobResult:
        """Describe a new transition with its already prepared facts."""
        return cls(
            CompletionOutcome.APPLIED,
            origin=origin,
            locked_row=locked_row,
            effects=effects,
            durable_events=durable_events,
        )

    @classmethod
    def replay(cls) -> CompleteJobResult:
        """Describe an idempotent public success without transition facts."""
        return cls(CompletionOutcome.IDEMPOTENT_REPLAY)

    @classmethod
    def no_transition(cls, reason: CompletionNoTransitionReason) -> CompleteJobResult:
        """Describe a public false outcome with its existing rejection reason."""
        return cls(CompletionOutcome.NO_TRANSITION, reason=reason)

    @property
    def public_success(self) -> bool:
        """Map applied/replay dispositions onto the unchanged public boolean."""
        return self.outcome is not CompletionOutcome.NO_TRANSITION


class CompletionPolicies(Protocol):
    """Lazy executor-free preparation; implementations never dispatch observers.

    Callable binding, parsing and clock sampling belong together in later facade
    preparation, not in these contracts or a separate eager duration callback.
    """

    def transform_result(self, result: Any, domain: str) -> Any:
        """Transform a result only at its existing backend execution point."""
        ...

    def queued_domain_allowed(self, domain: Any) -> bool:
        """Read the queued-domain policy lazily."""
        ...

    def counters_enabled(self) -> bool:
        """Read counter enablement at the existing reconciliation point."""
        ...

    def prepare_metric_effects(self, backend: str, job: dict[str, Any], result: Any) -> tuple[CompletionEffect, ...]:
        """Prepare optional truncation and a possibly partial metric prefix."""
        ...

    def duration_sla_breach(self, job: dict[str, Any]) -> tuple[float, float] | None:
        """Evaluate SLA using its later independent policy/clock sample."""
        ...

    def prepare_lifecycle_event(
        self, event_type: str, job: dict[str, Any], attrs: dict[str, Any] | None
    ) -> SlaEventEffect | CompletionEventEffect:
        """Prepare the lifecycle observer's original job and attributes."""
        ...

    def prepare_sla_metric(self, job: dict[str, Any]) -> SlaMetricEffect:
        """Prepare a duration SLA metric without invoking it."""
        ...

    def prepare_sla_warning(self, job_id: int, error_type: str) -> SlaWarningEffect:
        """Prepare the optional persistence warning without logging it."""
        ...
