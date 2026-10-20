"""Closed workflow facts must not admit ambiguous or mutable authority."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowAdmissionCommand,
    WorkflowAdmissionResult,
    WorkflowControls,
    WorkflowEffect,
    WorkflowEffectResult,
    WorkflowHorizonResult,
    WorkflowLease,
    WorkflowLookupResult,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowProgress,
    WorkflowProgressKind,
    WorkflowProgressResult,
    WorkflowReceipt,
    WorkflowRoot,
    WorkflowScope,
    bounded_workflow_json,
    decode_workflow_receipt,
    encode_workflow_receipt,
    workflow_effect_fingerprint,
)
from tldw_Server_API.tests.Jobs.workflow_test_support import ROOT_REFERENCE, ROOT_UUID

pytestmark = pytest.mark.unit
NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


def controls():
    return WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT)


def root():
    return WorkflowRoot(
        ROOT_UUID, "owner-a", WorkflowScope("embeddings", "workflow", "root"), ROOT_REFERENCE, "key", controls()
    )


def receipt():
    return WorkflowReceipt("a" * 64, "b" * 64, ROOT_UUID, WorkflowAction.COMPLETE, "completed", NOW)


def test_controls_reject_retry_consuming_expiry():
    with pytest.raises(ValueError):
        WorkflowControls(3, None, ExpiredLeasePolicy.CONSUME_RETRY, 5)


@pytest.mark.parametrize("limit", [0, 6, True])
def test_controls_bound_no_progress(limit):
    with pytest.raises(ValueError):
        WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT, limit)


def test_owner_limit_is_utf8_bytes():
    value = controls()
    WorkflowAdmissionCommand("\u00e9" * 256, ROOT_REFERENCE, "key", value)
    with pytest.raises(ValueError):
        WorkflowAdmissionCommand("\u00e9" * 257, ROOT_REFERENCE, "key", value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_retries", True),
        ("max_retries", -1),
        ("max_retries", 1.0),
        ("max_retries", float("inf")),
        ("quarantine_threshold", 0),
        ("quarantine_threshold", True),
        ("no_progress_limit", 1.5),
        ("expired_lease_policy", "requeue_no_attempt"),
    ],
)
def test_controls_are_exact(field, value):
    with pytest.raises(ValueError):
        replace(controls(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_user_id", " "),
        ("idempotency_key", "\u00e9" * 101),
        ("reference_id", "00000000-0000-1000-8000-000000000001"),
        ("reference_id", ROOT_REFERENCE.upper().replace("0001", "000A")),
        ("controls", {}),
    ],
)
def test_admission_rejects_nonexact_facts(field, value):
    command = WorkflowAdmissionCommand("owner", ROOT_REFERENCE, "\u00e9" * 100, controls())
    with pytest.raises(ValueError):
        replace(command, **{field: value})


@pytest.mark.parametrize(
    "code", ["", "a" * 65, "a/b", "A", "\u00e9", "a\n", "embeddings\x00/invalid", "a" * 64 + "\x00suffix"]
)
def test_scope_rejects_nonfixed_codes(code):
    with pytest.raises(ValueError):
        WorkflowScope(code, "workflow", "root")


@pytest.mark.parametrize("field", ["job_uuid", "reference_id"])
def test_root_uuid_rejects_embedded_nul(field):
    value = root()
    with pytest.raises(ValueError):
        replace(value, **{field: getattr(value, field) + "\x00suffix"})


def test_policy_rejects_reserved_scope_and_missing_infrastructure_reason():
    with pytest.raises(ValueError):
        WorkflowScope("admin_webhooks", "delivery", "other")
    with pytest.raises(ValueError):
        WorkflowPolicy(root().scope, "reference_id", frozenset({"other"}))


def test_dto_copy_cannot_mutate_identity():
    reasons = {"transient", "no_progress_exhausted"}
    policy = WorkflowPolicy(root().scope, "reference_id", reasons)
    reasons.clear()
    assert policy.reason_codes == frozenset({"transient", "no_progress_exhausted"})
    with pytest.raises(FrozenInstanceError):
        root().owner_user_id = "other"


def test_policy_requires_reason_membership():
    policy = WorkflowPolicy(root().scope, "reference_id", frozenset({"no_progress_exhausted"}))
    with pytest.raises(ValueError):
        policy.validate_effect(WorkflowEffect("a" * 64, WorkflowAction.FAIL, "b" * 64, "other"))


@pytest.mark.parametrize(
    "action,reason,schedule",
    [
        (WorkflowAction.CONTINUE, None, None),
        (WorkflowAction.RETRY, "transient", None),
        (WorkflowAction.RETRY, None, NOW),
        (WorkflowAction.FAIL, None, None),
        (WorkflowAction.COMPLETE, "irrelevant", None),
        (WorkflowAction.CANCEL, None, NOW),
        (WorkflowAction.FAIL, "permanent", NOW),
        (WorkflowAction.CONTINUE, "irrelevant", NOW),
    ],
)
def test_effect_rejects_irrelevant_or_missing_facts(action, reason, schedule):
    with pytest.raises(ValueError):
        WorkflowEffect("a" * 64, action, "b" * 64, reason, schedule)


@pytest.mark.parametrize(
    "key", ["A" * 64, "a" * 63, "g" * 64, True, "a" * 64 + "\x00suffix", "a" * 32 + "\x00" + "g" * 31]
)
def test_keys_and_fingerprints_are_exact(key):
    with pytest.raises(ValueError):
        WorkflowProgress(key, WorkflowProgressKind.CURSOR, "b" * 64)
    with pytest.raises(ValueError):
        WorkflowEffect("a" * 64, WorkflowAction.COMPLETE, key)


def test_aware_dates_normalize_and_naive_dates_fail():
    effect = WorkflowEffect(
        "a" * 64, WorkflowAction.CONTINUE, "b" * 64, not_before_at=NOW.astimezone(timezone(timedelta(hours=3)))
    )
    assert effect.not_before_at == NOW
    assert effect.not_before_at.tzinfo is timezone.utc
    with pytest.raises(ValueError):
        replace(effect, not_before_at=NOW.replace(tzinfo=None))


def test_effect_fingerprint_ignores_delivery_lease_and_hint():
    effect = WorkflowEffect("a" * 64, WorkflowAction.RETRY, "b" * 64, "transient", NOW)
    digest = workflow_effect_fingerprint(root(), effect)
    WorkflowLease("worker-a", "lease-a")
    WorkflowLease("worker-b", "lease-b")
    assert workflow_effect_fingerprint(replace(root(), job_id_hint=42), effect) == digest
    changed = replace(receipt(), applied_at=NOW + timedelta(days=1), released=True)
    assert changed != receipt()
    assert workflow_effect_fingerprint(root(), effect) == digest
    for candidate in [
        replace(root(), owner_user_id="other"),
        replace(root(), job_uuid=ROOT_REFERENCE),
        replace(root(), scope=WorkflowScope("other", "workflow", "root")),
        replace(root(), reference_id=ROOT_UUID),
        replace(root(), idempotency_key="other"),
        replace(root(), controls=replace(controls(), max_retries=4)),
    ]:
        assert workflow_effect_fingerprint(candidate, effect) != digest
    for candidate in [
        replace(effect, guard_fingerprint="c" * 64),
        replace(effect, reason_code="other"),
        replace(effect, not_before_at=NOW + timedelta(microseconds=1)),
        WorkflowEffect("a" * 64, WorkflowAction.FAIL, "b" * 64, "transient"),
    ]:
        assert workflow_effect_fingerprint(root(), candidate) != digest


def test_effect_facts_accept_4096_reject_4097_bytes():
    assert bounded_workflow_json(b"{}" + b" " * 4094) == {}
    with pytest.raises(ValueError):
        bounded_workflow_json(b"{}" + b" " * 4095)
    raw = json.dumps({"text": "\u00e9" * 2048}, ensure_ascii=False).encode("utf-8")
    with pytest.raises(ValueError):
        bounded_workflow_json(raw)


@pytest.mark.parametrize("raw", [b"\xff", b"{bad", b"NaN", b'{"x":1,"x":2}', b"[" * 2000 + b"]" * 2000])
def test_bounded_parser_rejects_bad_json_without_echoing_data(raw):
    with pytest.raises(ValueError) as error:
        bounded_workflow_json(raw)
    assert "bad" not in str(error.value)


def test_receipt_roundtrip_is_exact_and_rejects_unknown_nested_facts():
    raw = encode_workflow_receipt(receipt())
    assert decode_workflow_receipt(raw) == receipt()
    facts = json.loads(raw)
    for changed in [
        dict(facts, unknown=True),
        dict(facts, action={"value": "complete"}),
        dict(facts, released=1),
        dict(facts, resulting_status="queued"),
    ]:
        with pytest.raises(ValueError):
            decode_workflow_receipt(json.dumps(changed).encode())
    assert decode_workflow_receipt(raw + b" " * (4096 - len(raw))) == receipt()
    with pytest.raises(ValueError):
        decode_workflow_receipt(raw + b" " * (4097 - len(raw)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WorkflowAdmissionResult(WorkflowOutcome.APPLIED, None),
        lambda: WorkflowAdmissionResult(WorkflowOutcome.REJECTED, root()),
        lambda: WorkflowLookupResult(WorkflowOutcome.MISSING, root(), "queued"),
        lambda: WorkflowLookupResult(WorkflowOutcome.ACTIVE, root(), None),
        lambda: WorkflowEffectResult(WorkflowOutcome.APPLIED, None),
        lambda: WorkflowEffectResult(WorkflowOutcome.STALE_LEASE, receipt()),
        lambda: WorkflowHorizonResult(WorkflowOutcome.APPLIED, None, 300),
        lambda: WorkflowHorizonResult(WorkflowOutcome.INSUFFICIENT_HORIZON, NOW, 300),
        lambda: WorkflowProgressResult(WorkflowOutcome.CONFLICT, True),
        lambda: WorkflowProgressResult("applied", True),
    ],
)
def test_results_reject_contradictory_fields(factory):
    with pytest.raises(ValueError):
        factory()


def test_valid_result_variants():
    assert WorkflowAdmissionResult(WorkflowOutcome.APPLIED, root()).root == root()
    assert WorkflowLookupResult(WorkflowOutcome.ACTIVE, root(), "processing").status == "processing"
    assert WorkflowEffectResult(WorkflowOutcome.ALREADY_APPLIED, receipt()).receipt == receipt()
    assert WorkflowHorizonResult(WorkflowOutcome.APPLIED, NOW, 300).guaranteed_seconds == 300
    assert WorkflowProgressResult(WorkflowOutcome.APPLIED, True).reset is True


@given(st.binary(max_size=32), st.integers(min_value=4097, max_value=4200))
def test_oversize_is_rejected_before_any_decoding(prefix, size):
    raw = prefix + b" " * (size - len(prefix))
    with pytest.raises(ValueError, match="4096"):
        bounded_workflow_json(raw)
