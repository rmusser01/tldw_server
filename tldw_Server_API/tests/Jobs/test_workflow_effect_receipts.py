"""Logical workflow effects retain immutable proof beyond Jobs history."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowFenceError,
    WorkflowOutcome,
    WorkflowReceipt,
    encode_workflow_receipt,
    workflow_effect_fingerprint,
)


def test_earlier_effect_survives_later_transition_and_archive(workflow_case):
    workflow_case.admit()
    first = workflow_case.effect("a" * 64, WorkflowAction.CONTINUE)
    workflow_case.apply(first, workflow_case.acquire())
    last = workflow_case.effect("b" * 64, WorkflowAction.COMPLETE)
    workflow_case.apply(last, workflow_case.acquire())
    workflow_case.archive_root(compressed=True)
    assert workflow_case.capability.lookup(workflow_case.root).outcome is WorkflowOutcome.ARCHIVED
    with workflow_case.connection() as conn:
        archived = conn.execute("SELECT payload,payload_compressed FROM jobs_archive").fetchone()
    assert archived[0] is None
    assert archived[1] is not None
    replay = workflow_case.capability.lookup_effect(workflow_case.root, first)
    assert replay.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert replay.receipt.resulting_status == "queued"
    assert workflow_case.status() == "completed"


def test_retry_lost_ack_does_not_spend_twice(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("c" * 64, WorkflowAction.RETRY, "provider_transient")
    lease = workflow_case.acquire()
    workflow_case.apply(effect, lease)
    workflow_case.apply(effect, lease)
    assert workflow_case.retry_count() == 1
    assert workflow_case.receipt_count() == 1


@pytest.mark.parametrize("archived", [False, True])
def test_pruned_history_does_not_erase_unreleased_proof(workflow_case, archived):
    workflow_case.admit()
    effect = workflow_case.effect("d" * 64, WorkflowAction.COMPLETE)
    workflow_case.apply(effect, workflow_case.acquire())
    if archived:
        workflow_case.archive_root(compressed=True)
    workflow_case.prune_root_history()
    assert workflow_case.job_count() == 0
    assert workflow_case.capability.lookup_effect(workflow_case.root, effect).outcome is WorkflowOutcome.ALREADY_APPLIED
    assert workflow_case.admit().outcome in {WorkflowOutcome.MISSING, WorkflowOutcome.UNAVAILABLE}
    assert workflow_case.capability.release_effect(workflow_case.root, effect).receipt.released


def test_effect_insert_failure_rolls_back_status_counter_and_event(workflow_case, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "1")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    workflow_case.admit()
    workflow_case.install_effect_guard()
    lease = workflow_case.acquire()
    before = workflow_case.snapshot()
    effect = workflow_case.effect("a" * 64, WorkflowAction.RETRY, "transient")
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    store = workflow_store(workflow_case.manager.backend)
    monkeypatch.setattr(
        store, "insert_effect_receipt", Mock(side_effect=RuntimeError("receipt insert failed")), raising=False
    )
    with pytest.raises(RuntimeError, match="receipt insert failed"):
        workflow_case.apply(effect, lease)
    assert workflow_case.snapshot() == before


def test_new_lease_delivers_same_intent_after_locked_guard_revalidation(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    old = workflow_case.acquire()
    current = workflow_case.acquire()
    with pytest.raises(WorkflowFenceError) as stale:
        workflow_case.apply(effect, old)
    assert stale.value.outcome is WorkflowOutcome.STALE_LEASE
    assert workflow_case.receipt_count() == 0
    result = workflow_case.apply(effect, current)
    assert result.outcome is WorkflowOutcome.APPLIED
    assert result.receipt.effect_key == effect.effect_key
    assert result.receipt.facts_fingerprint == workflow_effect_fingerprint(workflow_case.root, effect)


def test_stale_domain_guard_aborts_without_receipt(workflow_case):
    workflow_case.admit()
    workflow_case.install_effect_guard()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    lease = workflow_case.acquire()
    with workflow_case.connection() as conn:
        workflow_case.execute(conn, "UPDATE workflow_effect_domain SET guard=?", ("f" * 64,))
        conn.commit()
    before = workflow_case.snapshot()
    with pytest.raises(WorkflowFenceError):
        workflow_case.apply(effect, lease)
    assert workflow_case.snapshot() == before


def test_verified_guard_must_match_intent(workflow_case):
    workflow_case.admit()
    lease = workflow_case.acquire()
    with pytest.raises(WorkflowFenceError) as rejected:
        workflow_case.apply(workflow_case.effect("a" * 64, WorkflowAction.COMPLETE), lease, "f" * 64)
    assert rejected.value.outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.receipt_count() == 0
    assert workflow_case.status() == "processing"


@pytest.mark.parametrize("released", [False, True])
@pytest.mark.parametrize("fact", ["action", "guard", "schedule", "reason"])
def test_same_key_changed_facts_conflict_even_after_release(workflow_case, released, fact):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.RETRY, "transient")
    workflow_case.apply(effect, workflow_case.acquire())
    if released:
        workflow_case.capability.release_effect(workflow_case.root, effect)
    changed = {
        "action": replace(effect, action=WorkflowAction.CONTINUE, reason_code=None),
        "guard": replace(effect, guard_fingerprint="f" * 64),
        "schedule": replace(effect, not_before_at=datetime(2025, 1, 1, tzinfo=timezone.utc)),
        "reason": replace(effect, reason_code="permanent"),
    }[fact]
    assert workflow_case.capability.lookup_effect(workflow_case.root, changed).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.capability.release_effect(workflow_case.root, changed).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.retry_count() == 1


def test_release_lost_ack_replays_without_reapplying(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.CONTINUE)
    lease = workflow_case.acquire()
    workflow_case.apply(effect, lease)
    first = workflow_case.capability.release_effect(workflow_case.root, effect)
    second = workflow_case.capability.release_effect(workflow_case.root, effect)
    assert first.receipt == second.receipt
    assert first.receipt.released
    assert second.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert workflow_case.apply(effect, lease).receipt == first.receipt
    assert workflow_case.receipt_count() == 1


@pytest.mark.parametrize("replacement", [False, True])
def test_terminal_or_reused_uuid_cannot_receive_unapplied_effect(workflow_case, replacement):
    workflow_case.admit()
    lease = workflow_case.acquire()
    workflow_case.apply(workflow_case.effect("a" * 64, WorkflowAction.COMPLETE), lease)
    if replacement:
        workflow_case.replace_numeric_id_with_other_uuid()
    before = workflow_case.snapshot()
    with pytest.raises(WorkflowFenceError) as rejected:
        workflow_case.apply(workflow_case.effect("b" * 64, WorkflowAction.COMPLETE), lease)
    assert rejected.value.outcome is (WorkflowOutcome.MISSING if replacement else WorkflowOutcome.TERMINAL)
    assert workflow_case.snapshot() == before


def test_receipt_capacity_exhaustion_precedes_transition(workflow_case):
    workflow_case.admit()
    lease = workflow_case.acquire()
    retained = workflow_case.effect("a" * 64, WorkflowAction.CONTINUE)
    retained_proof = WorkflowReceipt(
        retained.effect_key,
        workflow_effect_fingerprint(workflow_case.root, retained),
        workflow_case.root.job_uuid,
        retained.action,
        "queued",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    with workflow_case.connection() as conn:
        workflow_case.insert(
            conn,
            "job_workflow_effect_receipts",
            workflow_case.receipt_values(
                job_uuid=workflow_case.root.job_uuid,
                action="continue",
                resulting_status="queued",
                facts_fingerprint=retained_proof.facts_fingerprint,
                facts=encode_workflow_receipt(retained_proof).decode(),
                slot=32768,
            ),
        )
        conn.commit()
    effect = workflow_case.effect("b" * 64, WorkflowAction.COMPLETE)
    with workflow_case.capability.transaction(workflow_case.root.owner_user_id) as tx:
        fence = tx.lock_live_root(workflow_case.root, lease)
        result = tx.apply_effect(fence, effect, effect.guard_fingerprint)
    assert result.outcome is WorkflowOutcome.EXHAUSTED
    assert workflow_case.status() == "processing"
    assert workflow_case.receipt_count() == 1
    assert workflow_case.capability.lookup_effect(workflow_case.root, retained).receipt == retained_proof
    assert workflow_case.capability.release_effect(workflow_case.root, retained).receipt.released
    assert workflow_case.status() == "processing"


@pytest.mark.parametrize(
    "action,reason,status",
    [
        (WorkflowAction.CONTINUE, None, "queued"),
        (WorkflowAction.RETRY, "transient", "queued"),
        (WorkflowAction.COMPLETE, None, "completed"),
        (WorkflowAction.FAIL, "permanent", "failed"),
        (WorkflowAction.CANCEL, None, "cancelled"),
    ],
)
def test_first_effect_lifecycle_counters_events_and_consumed_fence(workflow_case, monkeypatch, action, reason, status):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "1")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, action, reason)
    lease = workflow_case.acquire()
    result = workflow_case.apply(effect, lease)
    row = workflow_case.job_row()
    assert result.outcome is WorkflowOutcome.APPLIED
    assert row["status"] == status
    assert (row["worker_id"], row["lease_id"], row["leased_until"]) == (None, None, None)
    assert row["retry_count"] == int(action is WorkflowAction.RETRY)
    assert workflow_case.counters() == (0, int(status == "queued"))
    assert workflow_case.snapshot()[2] == 2
    if status == "queued":
        assert datetime.fromisoformat(str(row["available_at"])).replace(tzinfo=timezone.utc) == effect.not_before_at
    else:
        assert row["completion_token"] == effect.effect_key


@pytest.mark.parametrize("max_retries,threshold,status,retries", [(0, None, "failed", 0), (3, 1, "quarantined", 1)])
def test_retry_respects_budget_and_quarantine(workflow_case, max_retries, threshold, status, retries):
    workflow_case.admit(max_retries=max_retries, quarantine_threshold=threshold)
    effect = workflow_case.effect("a" * 64, WorkflowAction.RETRY, "transient")
    result = workflow_case.apply(effect, workflow_case.acquire())
    assert result.receipt.resulting_status == status
    assert workflow_case.retry_count() == retries


@pytest.mark.parametrize("field", ["owner_user_id", "reference_id", "idempotency_key", "controls", "scope"])
def test_retained_proof_requires_exact_root(workflow_case, field):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    workflow_case.apply(effect, workflow_case.acquire())
    root = workflow_case.root
    changes = {
        "owner_user_id": "owner-b",
        "reference_id": str(uuid4()),
        "idempotency_key": "other",
        "controls": replace(root.controls, max_retries=2),
        "scope": replace(root.scope, queue="other"),
    }
    bad = replace(root, **{field: changes[field]})
    assert workflow_case.capability.lookup_effect(bad, effect).outcome in {
        WorkflowOutcome.MISSING,
        WorkflowOutcome.CONFLICT,
    }
    assert workflow_case.capability.release_effect(bad, effect).receipt is None
    assert workflow_case.capability.lookup_effect(replace(root, job_id_hint=99999), effect).receipt is not None


def test_no_lease_lookup_and_release_cannot_apply_missing_effect(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    assert workflow_case.capability.lookup_effect(workflow_case.root, effect).outcome is WorkflowOutcome.MISSING
    assert workflow_case.capability.release_effect(workflow_case.root, effect).outcome is WorkflowOutcome.MISSING
    assert workflow_case.status() == "queued"


def test_observers_only_after_commit_once_and_never_resolve_numeric_id(workflow_case, monkeypatch):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    lease = workflow_case.acquire()
    observed = []

    def observe(job, **kwargs):
        assert workflow_case.status() == "completed"
        assert job["uuid"] == workflow_case.root.job_uuid
        observed.append(dict(job))

    monkeypatch.setattr("tldw_Server_API.app.core.Jobs.manager.increment_completed", observe)
    monkeypatch.setattr(workflow_case.manager, "get_job", Mock(side_effect=AssertionError("numeric lookup")))
    workflow_case.apply(effect, lease)
    workflow_case.apply(effect, lease)
    workflow_case.capability.lookup_effect(workflow_case.root, effect)
    workflow_case.capability.release_effect(workflow_case.root, effect)
    assert len(observed) == 1


@pytest.mark.parametrize("fake", [None, {}, []])
def test_unminted_effect_fence_is_rejected(workflow_case, fake):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    with workflow_case.capability.transaction(workflow_case.root.owner_user_id) as tx:
        with pytest.raises(WorkflowFenceError) as rejected:
            tx.apply_effect(fake, effect, effect.guard_fingerprint)
        assert rejected.value.outcome is WorkflowOutcome.STALE_LEASE
    assert workflow_case.status() == "queued"


def test_receipt_action_must_agree_with_exact_intent(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    fingerprint = workflow_effect_fingerprint(workflow_case.root, effect)
    inconsistent = WorkflowReceipt(
        effect.effect_key,
        fingerprint,
        workflow_case.root.job_uuid,
        WorkflowAction.FAIL,
        "failed",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    with workflow_case.connection() as conn:
        workflow_case.insert(
            conn,
            "job_workflow_effect_receipts",
            workflow_case.receipt_values(
                job_uuid=workflow_case.root.job_uuid,
                action="fail",
                resulting_status="failed",
                facts_fingerprint=fingerprint,
                facts=encode_workflow_receipt(inconsistent).decode(),
            ),
        )
        conn.commit()
    assert workflow_case.capability.lookup_effect(workflow_case.root, effect).outcome is WorkflowOutcome.CONFLICT


def test_update_boundary_expiry_rolls_back_domain_and_jobs(workflow_case, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    workflow_case.admit()
    workflow_case.install_effect_guard()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    lease = workflow_case.acquire()
    before = workflow_case.snapshot()
    store = workflow_store(workflow_case.manager.backend)
    original = store.transition_effect

    def expire_then_transition(connection, *args, **kwargs):
        workflow_case.expire(connection)
        return original(connection, *args, **kwargs)

    monkeypatch.setattr(store, "transition_effect", expire_then_transition)
    with pytest.raises(WorkflowFenceError) as rejected:
        workflow_case.apply(effect, lease)
    assert rejected.value.outcome is WorkflowOutcome.STALE_LEASE
    assert workflow_case.snapshot() == before


def test_consumed_fence_replays_but_cannot_apply_another_intent(workflow_case):
    workflow_case.admit()
    workflow_case.install_effect_guard()
    effect = workflow_case.effect("a" * 64, WorkflowAction.CONTINUE)
    lease = workflow_case.acquire()
    with workflow_case.capability.transaction(workflow_case.root.owner_user_id) as tx:
        fence = tx.lock_live_root(workflow_case.root, lease)
        result = tx.apply_effect(fence, effect, workflow_case.guard_cas(tx, effect))
        assert tx.apply_effect(fence, effect, "not-an-authorization").receipt == result.receipt
        with pytest.raises(WorkflowFenceError):
            tx.apply_effect(fence, workflow_case.effect("b" * 64, WorkflowAction.COMPLETE), "e" * 64)
        assert tx.connection is not None
    assert workflow_case.status() == "queued"


def test_consuming_one_root_does_not_weaken_other_fences(workflow_case):
    workflow_case.admit()
    workflow_case.install_effect_guard()
    root = workflow_case.root
    lease = workflow_case.acquire()
    other = workflow_case.admit_new_reference().root
    workflow_case.root = other
    other_lease = workflow_case.acquire()
    workflow_case.root = root
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    before = workflow_case.snapshot()
    with pytest.raises(WorkflowFenceError) as rejected:
        with workflow_case.capability.transaction(root.owner_user_id) as tx:
            first_fence = tx.lock_live_root(root, lease)
            tx.lock_live_root(other, other_lease)
            tx.apply_effect(first_fence, effect, workflow_case.guard_cas(tx, effect))
            workflow_case.root = other
            workflow_case.expire(tx.connection)
            workflow_case.root = root
    workflow_case.root = root
    assert rejected.value.outcome is WorkflowOutcome.STALE_LEASE
    assert workflow_case.snapshot() == before


def test_observer_failure_does_not_undo_committed_proof(workflow_case, monkeypatch):
    workflow_case.admit()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager.increment_completed", Mock(side_effect=RuntimeError("observer failed"))
    )
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    result = workflow_case.apply(effect, workflow_case.acquire())
    assert result.outcome is WorkflowOutcome.APPLIED
    assert workflow_case.capability.lookup_effect(workflow_case.root, effect).receipt == result.receipt


def test_proof_replay_after_id_reuse_never_observes_replacement(workflow_case, monkeypatch):
    workflow_case.admit()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    lease = workflow_case.acquire()
    original = workflow_case.apply(effect, lease)
    workflow_case.replace_numeric_id_with_other_uuid()
    observer = Mock(side_effect=AssertionError("replayed observer"))
    monkeypatch.setattr(workflow_case.manager, "_emit_workflow_effect_side_effects", observer)
    replay = workflow_case.apply(effect, lease)
    assert replay.receipt == original.receipt
    assert workflow_case.capability.release_effect(workflow_case.root, effect).receipt.released
    with workflow_case.connection() as conn:
        replacement = conn.execute("SELECT uuid,status FROM jobs").fetchone()
    assert replacement[0] != workflow_case.root.job_uuid
    assert replacement[1] == "queued"
    observer.assert_not_called()


def test_retry_budget_is_charged_only_for_permitted_retries(workflow_case, monkeypatch):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    workflow_case.admit(max_retries=2)
    for index, expected in enumerate(("queued", "queued", "failed")):
        effect = workflow_case.effect(f"{index:064x}", WorkflowAction.RETRY, "transient")
        lease = workflow_case.acquire()
        applied = workflow_case.apply(effect, lease)
        assert applied.receipt.resulting_status == expected
        assert workflow_case.apply(effect, lease).receipt == applied.receipt
    assert workflow_case.retry_count() == 2
    assert workflow_case.job_row()["failure_streak_count"] == 2
    with workflow_case.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_events WHERE event_type='job.retry_scheduled'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM job_events WHERE event_type='job.failed'").fetchone()[0] == 1


def test_continue_is_retry_and_streak_neutral(workflow_case):
    workflow_case.admit(quarantine_threshold=2)
    workflow_case.apply(workflow_case.effect("a" * 64, WorkflowAction.RETRY, "transient"), workflow_case.acquire())
    workflow_case.apply(workflow_case.effect("b" * 64, WorkflowAction.CONTINUE), workflow_case.acquire())
    row = workflow_case.job_row()
    assert (row["retry_count"], row["failure_streak_count"], row["failure_streak_code"]) == (1, 1, "transient")
    result = workflow_case.apply(
        workflow_case.effect("c" * 64, WorkflowAction.RETRY, "transient"), workflow_case.acquire()
    )
    assert result.receipt.resulting_status == "quarantined"
    assert workflow_case.retry_count() == 2


def test_commit_failure_does_not_publish_receipt_or_observers(workflow_case, monkeypatch):
    workflow_case.admit()
    workflow_case.install_effect_guard()
    effect = workflow_case.effect("a" * 64, WorkflowAction.COMPLETE)
    lease = workflow_case.acquire()
    before = workflow_case.snapshot()
    observer = Mock()
    monkeypatch.setattr(workflow_case.manager, "_emit_workflow_effect_side_effects", observer)
    original_connect = workflow_case.manager._connect

    class CommitFailure:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            raise RuntimeError("controlled commit failure")

    with monkeypatch.context() as context:
        context.setattr(workflow_case.manager, "_connect", lambda: CommitFailure(original_connect()))
        with pytest.raises(RuntimeError, match="controlled commit failure"):
            workflow_case.apply(effect, lease)
    assert workflow_case.snapshot() == before
    observer.assert_not_called()


@pytest.mark.parametrize("offset,due", [(-60, 1), (60, 0)])
def test_original_schedule_uses_jobs_database_time_comparison(workflow_case, offset, due):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    workflow_case.admit()
    with workflow_case.connection() as conn:
        now = workflow_store(workflow_case.manager.backend).database_now(conn)
    effect = replace(
        workflow_case.effect("a" * 64, WorkflowAction.CONTINUE), not_before_at=now + timedelta(seconds=offset)
    )
    workflow_case.apply(effect, workflow_case.acquire())
    with workflow_case.connection() as conn:
        count = workflow_case.execute(
            conn,
            "SELECT COUNT(*) FROM jobs WHERE uuid=? AND available_at<=CURRENT_TIMESTAMP",
            (workflow_case.root.job_uuid,),
        ).fetchone()[0]
    assert count == due
    stored = datetime.fromisoformat(str(workflow_case.job_row()["available_at"]))
    assert stored.replace(tzinfo=timezone.utc) == effect.not_before_at
