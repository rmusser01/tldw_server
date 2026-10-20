"""Direct tests for the inert, shallow completion-only persistence boundary."""

from __future__ import annotations

import importlib
import multiprocessing
import sys
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from typing import Any, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.jobs_completion_contracts import (
    CompletedEffect,
    CompleteJobCommand,
    CompleteJobResult,
    CompletionDurableEvent,
    CompletionEffect,
    CompletionEventEffect,
    CompletionNoTransitionReason,
    CompletionOrigin,
    CompletionOutcome,
    CompletionPolicies,
    DurationEffect,
    GaugesEffect,
    JsonTruncatedEffect,
    SlaEventEffect,
    SlaMetricEffect,
    SlaWarningEffect,
    TraceEffect,
)

pytestmark = pytest.mark.unit


class Untouchable:
    """Expose accidental input conversion, copying or serialization immediately."""

    def __int__(self) -> int:
        """Reject eager ID coercion."""
        raise AssertionError("unexpected int conversion")

    def __str__(self) -> str:
        """Reject eager UUID/token coercion."""
        raise AssertionError("unexpected str conversion")

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """Reject copying nested caller data."""
        raise AssertionError("unexpected deepcopy")

    def __iter__(self) -> Any:
        """Reject attempts to serialize by iteration."""
        raise AssertionError("unexpected iteration")


class UnserializableResult(dict[str, Any]):
    """Make even a swallowed JSON serialization attempt fail this test."""

    def items(self) -> Any:
        """Reject eager JSON traversal of a caller's result mapping."""
        raise AssertionError("unexpected JSON conversion")


def test_command_is_shallow_and_inert() -> None:
    """Eager coercion/copying/freezing would destroy the facade's input timing."""
    raw = Untouchable()
    original = UnserializableResult(raw=raw, nested=[])
    command = CompleteJobCommand(raw, original, raw, raw, raw, raw, raw, raw)
    assert command.result is original
    for name in (
        "job_id",
        "worker_id",
        "lease_id",
        "completion_token",
        "expected_uuid",
        "enforce",
        "outbox_enabled",
    ):
        assert getattr(command, name) is raw
    with pytest.raises(FrozenInstanceError):
        command.result = {}
    original["nested"].append(raw)
    assert command.result["nested"][0] is raw


@pytest.mark.parametrize("value", [None, "", "  NONcanonical-UUID  ", Untouchable()])
def test_command_preserves_raw_optional_identity(value: Any) -> None:
    """Even empty and legacy identities must reach backend conversion unchanged."""
    command = CompleteJobCommand(1, None, value, value, value, value, False, False)
    assert command.result is None
    assert command.expected_uuid is value
    assert command.completion_token is value
    assert command.worker_id is value
    assert command.lease_id is value


def prepared_effects(row: dict[str, Any]) -> tuple[CompletionEffect, ...]:
    """Return a hand-ordered full preparation with distinct local bindings."""
    return (
        JsonTruncatedEffect(10, row),
        DurationEffect(11, row, row.get("started_at"), row.get("completed_at")),
        CompletedEffect(12, row),
        GaugesEffect(13, row.get("domain"), row.get("queue"), row.get("job_type")),
        TraceEffect(14, row),
        SlaEventEffect(15, row, row),
        SlaMetricEffect(16, row),
        CompletionEventEffect(17, row),
    )


@pytest.mark.parametrize(
    "stamp",
    [None, "not-a-timestamp", Untouchable(), datetime(2030, 1, 2), datetime(2030, 1, 2, tzinfo=timezone.utc)],
)
def test_effects_and_durable_facts_are_shallow_and_frozen(stamp: Any) -> None:
    """Prepared timestamps, labels, jobs and attrs must never be normalized."""
    row = {"started_at": stamp, "completed_at": stamp, "domain": stamp, "queue": stamp, "job_type": stamp}
    effects = prepared_effects(row)
    warning = SlaWarningEffect(18, stamp, stamp)
    event = CompletionDurableEvent("job.completed", row, row)
    for effect in (*effects, warning, event):
        field = fields(effect)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(effect, field, None)
        if hasattr(effect, "job"):
            assert effect.job is row
    assert effects[1].started_at is stamp
    assert effects[1].completed_at is stamp
    assert effects[3].domain is stamp
    assert effects[3].queue is stamp
    assert effects[3].job_type is stamp
    assert effects[5].attrs is row
    assert warning.job_id is stamp
    assert warning.error_type is stamp
    assert event.attrs is row
    row["later"] = stamp
    assert event.job["later"] is stamp
    assert set(get_args(CompletionEffect)) == {type(e) for e in (*effects, warning)}


@pytest.mark.parametrize("origin", list(CompletionOrigin))
def test_applied_keeps_envelope_identity(origin: CompletionOrigin) -> None:
    """Applied results expose the original locked projection and prepared facts."""
    row: dict[str, Any] = {}
    effects = prepared_effects(row)
    durable = (CompletionDurableEvent("job.sla_breached", row, row), CompletionDurableEvent("job.completed", row, None))
    result = CompleteJobResult.applied(origin=origin, locked_row=row, effects=effects, durable_events=durable)
    assert result.public_success is True
    assert result.outcome is CompletionOutcome.APPLIED
    assert result.reason is None
    assert result.origin is origin
    assert result.locked_row is row
    assert result.effects is effects
    assert result.durable_events is durable
    with pytest.raises(FrozenInstanceError):
        result.reason = CompletionNoTransitionReason.MISSING_ROW


def test_replay_is_payload_free_public_success() -> None:
    """Replay is successful without pretending it performed a transition."""
    result = CompleteJobResult.replay()
    assert result == CompleteJobResult(CompletionOutcome.IDEMPOTENT_REPLAY)
    assert result.public_success is True


@pytest.mark.parametrize("reason", list(CompletionNoTransitionReason))
def test_no_transition_is_payload_free_public_false(reason: CompletionNoTransitionReason) -> None:
    """Each existing rejection reason maps to the same public false disposition."""
    result = CompleteJobResult.no_transition(reason)
    assert result == CompleteJobResult(CompletionOutcome.NO_TRANSITION, reason=reason)
    assert result.public_success is False


@pytest.mark.parametrize("outcome", [CompletionOutcome.IDEMPOTENT_REPLAY, CompletionOutcome.NO_TRANSITION])
@pytest.mark.parametrize(
    "payload",
    [
        {"origin": CompletionOrigin.PROCESSING},
        {"locked_row": {}},
        {"effects": (CompletionEventEffect(1, {}),)},
        {"durable_events": (CompletionDurableEvent("job.completed", {}, None),)},
    ],
)
def test_non_applied_rejects_transition_payload(outcome: CompletionOutcome, payload: dict[str, Any]) -> None:
    """Replay/rejection cannot dispatch effects or claim durable writes."""
    reason = CompletionNoTransitionReason.MISSING_ROW if outcome is CompletionOutcome.NO_TRANSITION else None
    with pytest.raises(ValueError):
        CompleteJobResult(outcome, reason=reason, **payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"origin": None},
        {"locked_row": None},
        {"origin": "unknown"},
        {"reason": CompletionNoTransitionReason.GUARD_REJECTED},
        {"outcome": "unknown"},
    ],
)
def test_applied_rejects_inconsistent_disposition(changes: dict[str, Any]) -> None:
    """Only an applied, reason-free result can own the locked-row projection."""
    result = CompleteJobResult.applied(
        origin=CompletionOrigin.QUEUED, locked_row={}, effects=(CompletionEventEffect(1, {}),), durable_events=()
    )
    with pytest.raises(ValueError):
        replace(result, **changes)


@pytest.mark.parametrize(
    "outcome,reason",
    [
        (CompletionOutcome.IDEMPOTENT_REPLAY, CompletionNoTransitionReason.MISSING_ROW),
        (CompletionOutcome.NO_TRANSITION, None),
        (CompletionOutcome.NO_TRANSITION, "unknown"),
    ],
)
def test_reason_belongs_only_to_no_transition(outcome: CompletionOutcome, reason: Any) -> None:
    """A malformed internal disposition cannot silently map to a boolean."""
    with pytest.raises(ValueError):
        CompleteJobResult(outcome, reason=reason)


@pytest.mark.parametrize("prefix_length", range(5))
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("sla", ["none", "success", "warning"])
def test_all_optional_metric_prefixes_are_legal(prefix_length: int, truncated: bool, sla: str) -> None:
    """Suppressed optional preparation must not turn a valid completion into failure."""
    full = prepared_effects({})
    effects = full[:1] if truncated else ()
    effects += full[1 : 1 + prefix_length]
    effects += {"none": (), "success": full[5:7], "warning": (SlaWarningEffect(19, 1, "RuntimeError"),)}[sla]
    effects += full[-1:]
    result = CompleteJobResult.applied(
        origin=CompletionOrigin.PROCESSING, locked_row={}, effects=effects, durable_events=()
    )
    assert result.effects is effects


@pytest.mark.parametrize(
    "indices",
    [
        (),
        (0,),
        (7, 7),
        (7, 0),
        (2, 7),
        (1, 3, 7),
        (1, 2, 4, 7),
        (0, 0, 7),
        (1, 0, 7),
        (6, 5, 7),
        (5, 7),
        (6, 7),
        (8, 5, 6, 7),
        (5, 6, 8, 7),
        (5, 6, 5, 6, 7),
        (8, 8, 7),
        (5, 6, 1, 7),
    ],
)
def test_invalid_effect_order_is_rejected(indices: tuple[int, ...]) -> None:
    """Reject missing/duplicate completion, metric gaps and conflicting SLA forms."""
    full = (*prepared_effects({}), SlaWarningEffect(18, 1, "RuntimeError"))
    # Distinct bindings isolate grammar errors from duplicate-binding validation.
    effects = tuple(replace(full[index], binding_id=100 + n) for n, index in enumerate(indices))
    with pytest.raises(ValueError):
        CompleteJobResult.applied(origin=CompletionOrigin.PROCESSING, locked_row={}, effects=effects, durable_events=())


@pytest.mark.parametrize("event_types", [(), ("job.completed",), ("job.sla_breached", "job.completed")])
def test_durable_events_are_independent_of_observer_preparation(event_types: tuple[str, ...]) -> None:
    """Outbox facts can be absent or present without manufacturing SLA observers."""
    durable = tuple(CompletionDurableEvent(name, {}, None) for name in event_types)
    result = CompleteJobResult.applied(
        origin=CompletionOrigin.QUEUED, locked_row={}, effects=(CompletionEventEffect(1, {}),), durable_events=durable
    )
    assert result.durable_events is durable


@pytest.mark.parametrize(
    "event_types",
    [
        ("job.sla_breached",),
        ("job.completed", "job.sla_breached"),
        ("job.completed", "job.completed"),
        ("job.sla_breached", "job.sla_breached", "job.completed"),
        ("unknown",),
    ],
)
def test_invalid_durable_event_order_is_rejected(event_types: tuple[str, ...]) -> None:
    """Inserted facts cannot reverse or duplicate lifecycle order."""
    with pytest.raises(ValueError):
        CompleteJobResult.applied(
            origin=CompletionOrigin.QUEUED,
            locked_row={},
            effects=(CompletionEventEffect(1, {}),),
            durable_events=tuple(CompletionDurableEvent(name, {}, None) for name in event_types),
        )


@given(st.lists(st.integers(), min_size=2, max_size=2, unique=True))
def test_binding_ids_are_unique_only_within_attempt(bindings: list[int]) -> None:
    """Opaque IDs may be any integer and can be reused in another attempt."""
    effects = (DurationEffect(bindings[0], {}, None, None), CompletionEventEffect(bindings[1], {}))
    for _ in range(2):
        assert CompleteJobResult.applied(
            origin=CompletionOrigin.PROCESSING, locked_row={}, effects=effects, durable_events=()
        ).public_success
    with pytest.raises(ValueError):
        CompleteJobResult.applied(
            origin=CompletionOrigin.PROCESSING,
            locked_row={},
            effects=(effects[0], replace(effects[1], binding_id=bindings[0])),
            durable_events=(),
        )


@given(st.permutations(range(4)))
def test_metric_permutations_require_a_prepared_prefix(order: list[int]) -> None:
    """Only original preparation order is valid, not arbitrary metric permutations."""
    full = prepared_effects({})
    effects = tuple(full[index + 1] for index in order) + full[-1:]
    if tuple(order) == (0, 1, 2, 3):
        assert CompleteJobResult.applied(
            origin=CompletionOrigin.QUEUED, locked_row={}, effects=effects, durable_events=()
        ).public_success
    else:
        with pytest.raises(ValueError):
            CompleteJobResult.applied(origin=CompletionOrigin.QUEUED, locked_row={}, effects=effects, durable_events=())


@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_arbitrary_payloads_keep_identity(row: dict[str, Any]) -> None:
    """Envelope validation must not turn into job-payload validation or copying."""
    effect = CompletionEventEffect(1, row)
    result = CompleteJobResult.applied(
        origin=CompletionOrigin.QUEUED, locked_row=row, effects=(effect,), durable_events=()
    )
    assert result.locked_row is row
    assert result.effects[0].job is row


class RecordingPolicies(CompletionPolicies):
    """Inert executor-free policy stub: every invocation is explicit and recorded."""

    def __init__(self, effects: tuple[CompletionEffect, ...]) -> None:
        """Keep a prepared prefix without executing policy work."""
        self.effects = effects
        self.calls: list[str] = []

    def transform_result(self, result: Any, domain: str) -> Any:
        """Record result transformation only when requested."""
        self.calls.append("transform_result")
        return result

    def queued_domain_allowed(self, domain: Any) -> bool:
        """Record allowlist evaluation only when requested."""
        self.calls.append("queued_domain_allowed")
        return True

    def counters_enabled(self) -> bool:
        """Record counter settings only when requested."""
        self.calls.append("counters_enabled")
        return False

    def prepare_metric_effects(self, backend: str, job: dict[str, Any], result: Any) -> tuple[CompletionEffect, ...]:
        """Return an already prepared partial prefix, with no dispatch."""
        self.calls.append("prepare_metric_effects")
        return self.effects

    def duration_sla_breach(self, job: dict[str, Any]) -> tuple[float, float] | None:
        """Record the later independent SLA evaluation."""
        self.calls.append("duration_sla_breach")
        return None

    def prepare_lifecycle_event(
        self, event_type: str, job: dict[str, Any], attrs: dict[str, Any] | None
    ) -> SlaEventEffect | CompletionEventEffect:
        """Prepare data without invoking a lifecycle observer."""
        self.calls.append("prepare_lifecycle_event")
        return CompletionEventEffect(50, job)

    def prepare_sla_metric(self, job: dict[str, Any]) -> SlaMetricEffect:
        """Prepare data without invoking a metric observer."""
        self.calls.append("prepare_sla_metric")
        return SlaMetricEffect(51, job)

    def prepare_sla_warning(self, job_id: int, error_type: str) -> SlaWarningEffect:
        """Prepare data without logging a warning."""
        self.calls.append("prepare_sla_warning")
        return SlaWarningEffect(52, job_id, error_type)


def test_policy_protocol_stays_lazy_and_permits_partial_preparation() -> None:
    """Constructing envelopes never evaluates policies or samples a clock."""
    row: dict[str, Any] = {}
    prefix = (DurationEffect(1, row, None, Untouchable()), CompletedEffect(2, row))
    policy = RecordingPolicies(prefix)
    command = CompleteJobCommand(1, row, None, None, None, None, False, False)
    CompleteJobResult.replay()
    CompleteJobResult.no_transition(CompletionNoTransitionReason.MISSING_ROW)
    CompleteJobResult.applied(
        origin=CompletionOrigin.QUEUED,
        locked_row=row,
        effects=prefix + (CompletionEventEffect(3, row),),
        durable_events=(),
    )
    assert policy.calls == []
    prepared = policy.prepare_metric_effects("postgres", row, command.result)
    result = CompleteJobResult.applied(
        origin=CompletionOrigin.QUEUED,
        locked_row=row,
        effects=prepared + (CompletionEventEffect(3, row),),
        durable_events=(),
    )
    assert result.effects[:2] == prefix
    assert policy.calls == ["prepare_metric_effects"]


def cold_import_check() -> None:
    """Check imports in a fresh spawned interpreter, outside all pytest fixtures."""
    forbidden = ("psycopg", "psycopg2", "tldw_Server_API.app.core.Jobs")
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
    module_name = "tldw_Server_API.app.core.DB_Management.jobs_completion_contracts"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)


def test_cold_import_loads_no_driver_or_jobs_modules() -> None:
    """A DB-layer data boundary must not pull in drivers or Jobs orchestration."""
    process = multiprocessing.get_context("spawn").Process(target=cold_import_check)
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("cold import did not finish")
    assert process.exitcode == 0
