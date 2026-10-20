"""Independent command model checked against fresh SQLite and runtime-role PG stores."""

import hashlib
import json
from dataclasses import dataclass, field, replace

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowEffect,
    WorkflowFenceError,
    WorkflowLease,
    WorkflowOutcome,
    WorkflowProgressKind,
    decode_workflow_receipt,
)

pytestmark = [pytest.mark.integration, pytest.mark.property]
STALE = WorkflowLease("stale-worker", "stale-lease")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def episode_key(version, root_uuid, epoch):
    return digest(json.dumps([version, root_uuid, epoch], separators=(",", ":")))


def fingerprint(root, effect):
    """Independent wire oracle: deliberately does not call production hashing."""
    facts = {
        "version": "jobs_workflow_effect_v1",
        "owner_user_id": root.owner_user_id,
        "job_uuid": root.job_uuid,
        "scope": ["embeddings", "workflow", "embedding_workflow"],
        "reference_id": root.reference_id,
        "idempotency_key": root.idempotency_key,
        "controls": [root.controls.max_retries, root.controls.quarantine_threshold, "requeue_no_attempt", 5],
        "effect_key": effect.effect_key,
        "action": effect.action.value,
        "guard_fingerprint": effect.guard_fingerprint,
        "reason_code": effect.reason_code,
        "not_before_at": effect.not_before_at.isoformat() if effect.not_before_at else None,
    }
    return digest(json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


@dataclass
class Model:
    root_uuid: str
    retry_count: int = 0
    no_progress_count: int = 0
    status: str = "queued"
    epoch: int = 0
    closed_epoch: int = 0
    progress_flag: int = 0
    expired: bool = False
    archived: bool = False
    retry_streak: int = 0
    lease_digest: str | None = None
    effects: dict = field(default_factory=dict)
    receipts: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    effect_fingerprints: dict = field(default_factory=dict)
    released: dict = field(default_factory=dict)

    @property
    def terminal(self):
        return self.status in {"completed", "failed", "cancelled", "quarantined"}

    @property
    def terminal_status(self):
        return self.status if self.terminal else None


class WorkflowMachine(RuleBasedStateMachine):
    """Every expected transition is derived from commands, never observed counters."""

    def __init__(self, factory, quarantine=None):
        super().__init__()
        self.case = factory()
        self.case.command = replace(
            self.case.command, controls=replace(self.case.command.controls, quarantine_threshold=quarantine)
        )
        assert self.case.admit().outcome is WorkflowOutcome.APPLIED
        self.root = self.case.root
        self.controls = self.case.command.controls
        assert self.root.controls == self.controls
        assert (self.root.owner_user_id, self.root.reference_id, self.root.idempotency_key) == (
            self.case.command.owner_user_id,
            self.case.command.reference_id,
            self.case.command.idempotency_key,
        )
        self.model = Model(self.root.job_uuid)
        self.lease = None
        self.serial = 0
        self.check_store()

    def remember(self, effect):
        model = self.model
        model.effects[effect.effect_key] = effect
        model.receipts[effect.effect_key] = (effect.action, model.status)
        model.effect_fingerprints[effect.effect_key] = fingerprint(self.root, effect)
        model.released[effect.effect_key] = False

    def close_episode(self, charge):
        model = self.model
        model.closed_epoch = model.epoch
        if charge and not model.progress_flag:
            model.no_progress_count += 1

    def exhaust(self):
        key = episode_key("jobs_workflow_no_progress_v1", self.root.job_uuid, self.model.epoch)
        self.model.status = "failed"
        self.remember(WorkflowEffect(key, WorkflowAction.FAIL, key, "no_progress_exhausted"))

    def recover_expiry(self):
        model = self.model
        self.close_episode(True)
        if model.no_progress_count == 5:
            self.exhaust()
        else:
            # The maintenance schedule is deliberately database-private. Capture
            # its acknowledgment once, not counters or expected transitions.
            key = episode_key("jobs_workflow_expiry_v1", self.root.job_uuid, model.epoch)
            retained = self.case.effect_receipt(self.root, key)
            receipt = decode_workflow_receipt(retained["facts"].encode())
            assert (receipt.job_uuid, receipt.effect_key, receipt.action, receipt.resulting_status) == (
                self.root.job_uuid,
                key,
                WorkflowAction.CONTINUE,
                "queued",
            )
            assert retained["facts_fingerprint"] == receipt.facts_fingerprint
            model.effect_fingerprints[key] = receipt.facts_fingerprint
            model.effects[key] = None
            model.receipts[key] = (WorkflowAction.CONTINUE, "queued")
            model.released[key] = False
            model.status = "queued"
        model.expired = False
        self.lease = None

    @rule()
    def admit(self):
        result = self.case.admit()
        assert result.outcome is WorkflowOutcome.ALREADY_APPLIED
        assert result.root == self.root

    @rule(changed=st.booleans())
    def replay(self, changed=False):
        effects = [effect for effect in self.model.effects.values() if effect is not None]
        if not effects:
            assert self.case.admit(max_retries=4).outcome is WorkflowOutcome.CONFLICT
            return
        effect = effects[-1]
        if changed:
            effect = replace(effect, guard_fingerprint="d" * 64)
        result = self.case.apply(effect, self.lease or STALE)
        assert result.outcome is (WorkflowOutcome.CONFLICT if changed else WorkflowOutcome.ALREADY_APPLIED)

    @rule()
    def acquire(self):
        model = self.model
        reclaim = model.status == "processing" and model.expired
        lease = self.case.acquire(replace_current=False)
        if reclaim:
            self.recover_expiry()
        if model.status == "queued":
            assert lease is not None
            self.lease = lease
            model.lease_digest = digest(
                json.dumps(["jobs_workflow_lease_v1", lease.worker_id, lease.lease_id], separators=(",", ":"))
            )
            model.epoch += 1
            model.progress_flag = 0
            model.status = "processing"
        else:
            assert lease is None

    def live_error(self, stale=False):
        if self.model.terminal:
            return WorkflowOutcome.TERMINAL
        if stale or self.model.status != "processing" or self.model.expired:
            return WorkflowOutcome.STALE_LEASE
        return None

    @rule(stale=st.booleans())
    def renew(self, stale=False):
        expected = self.live_error(stale)
        try:
            with self.case.capability.transaction(self.root.owner_user_id) as tx:
                fence = tx.lock_live_root(self.root, STALE if stale else self.lease or STALE)
                result = tx.ensure_horizon(fence, 120)
            assert expected is None
            assert result.outcome is WorkflowOutcome.APPLIED
        except WorkflowFenceError as error:
            assert error.outcome is expected

    @rule(sweep=st.booleans())
    def expire(self, sweep=False):
        model = self.model
        if model.status == "processing" and not model.expired:
            self.case.expire_current_lease()
            model.expired = True
        if sweep:
            self.case.recover("sweep")
            if model.status == "processing" and model.expired:
                self.recover_expiry()

    def transition(self, action, stale=False):
        model = self.model
        self.serial += 1
        effect = self.case.effect(
            digest(f"action/{self.serial}"), action, "provider_transient" if action is WorkflowAction.RETRY else None
        )
        expected = self.live_error(stale)
        try:
            result = self.case.apply(effect, STALE if stale else self.lease or STALE)
            assert expected is None
        except WorkflowFenceError as error:
            assert error.outcome is expected
            return
        if action in {WorkflowAction.CONTINUE, WorkflowAction.RETRY}:
            self.close_episode(action is WorkflowAction.CONTINUE)
        if action is WorkflowAction.CONTINUE and model.no_progress_count == 5:
            assert result.outcome is WorkflowOutcome.EXHAUSTED
            assert result.receipt is None
            assert self.case.capability.lookup_effect(self.root, effect).outcome is WorkflowOutcome.MISSING
            self.exhaust()
        else:
            assert result.outcome is WorkflowOutcome.APPLIED
            if action is WorkflowAction.RETRY:
                model.retry_streak += 1
                if model.retry_count >= self.controls.max_retries:
                    model.status = "failed"
                elif self.controls.quarantine_threshold and model.retry_streak >= self.controls.quarantine_threshold:
                    model.status = "quarantined"
                else:
                    model.status = "queued"
                if model.status != "failed":
                    model.retry_count += 1
            else:
                model.status = "queued" if action is WorkflowAction.CONTINUE else "completed"
            self.remember(effect)
        self.lease = None

    @rule(stale=st.booleans())
    def continue_(self, stale=False):
        self.transition(WorkflowAction.CONTINUE, stale)

    @rule(stale=st.booleans())
    def retry(self, stale=False):
        self.transition(WorkflowAction.RETRY, stale)

    @rule()
    def cancel(self):
        if self.model.archived:
            return
        changed = self.case.manager.cancel_job(self.root.job_id_hint, expected_uuid=self.root.job_uuid)
        assert changed is (not self.model.terminal)
        if changed:
            self.model.status = "cancelled"
            self.model.expired = False
            self.lease = None

    @rule(stale=st.booleans())
    def complete(self, stale=False):
        self.transition(WorkflowAction.COMPLETE, stale)

    @rule(index=st.integers(0, 3), changed=st.booleans(), kind=st.sampled_from(list(WorkflowProgressKind)))
    def progress(self, index=0, changed=False, kind=WorkflowProgressKind.CURSOR):
        model = self.model
        key, facts = digest(f"progress/{index}"), "d" * 64 if changed else "b" * 64
        expected = self.live_error()
        self.case.current_lease = self.lease or STALE
        try:
            result = self.case.progress(key, kind, facts)
            assert expected is None
        except WorkflowFenceError as error:
            assert error.outcome is expected
            return
        if key in model.progress:
            assert result.outcome is (
                WorkflowOutcome.ALREADY_APPLIED if model.progress[key] == (kind, facts) else WorkflowOutcome.CONFLICT
            )
            assert not result.reset
        else:
            assert result.outcome is WorkflowOutcome.APPLIED
            assert result.reset
            model.progress[key] = (kind, facts)
            model.effect_fingerprints[key] = facts
            model.released[key] = False
            model.no_progress_count = 0
            model.progress_flag = 1

    @rule()
    def release(self):
        effects = [effect for effect in self.model.effects.values() if effect is not None]
        if effects:
            effect = effects[-1]
            result = self.case.capability.release_effect(self.root, effect)
            assert result.outcome is (
                WorkflowOutcome.ALREADY_APPLIED if self.model.released[effect.effect_key] else WorkflowOutcome.APPLIED
            )
            self.model.released[effect.effect_key] = True

    @rule(compressed=st.booleans())
    def archive(self, compressed=False):
        if self.model.terminal and not self.model.archived:
            self.case.archive_root(compressed=compressed)
            self.model.archived = True

    @rule()
    def restart(self):
        self.case.restart()

    @invariant()
    def check_store(self):
        observed, model = self.case.observe(), self.model
        assert observed.root_uuid == model.root_uuid
        assert observed.retry_count == model.retry_count
        assert observed.no_progress_count == model.no_progress_count
        assert observed.effect_fingerprints == model.effect_fingerprints
        assert observed.released == model.released
        assert not model.terminal or observed.status == model.terminal_status
        assert observed.status == model.status
        assert (observed.current_epoch, observed.last_closed_epoch, observed.progress_flag) == (
            model.epoch,
            model.closed_epoch,
            model.progress_flag,
        )
        support = self.case.support_row()
        assert (
            support["max_retries"],
            support["quarantine_threshold"],
            support["no_progress_limit"],
            support["expired_lease_policy"],
            support["lease_digest"],
        ) == (
            self.controls.max_retries,
            self.controls.quarantine_threshold,
            5,
            "requeue_no_attempt",
            model.lease_digest,
        )
        if model.status == "processing":
            row = self.case.job_row()
            assert (row["worker_id"], row["lease_id"]) == (self.lease.worker_id, self.lease.lease_id)
        for key, expected in model.receipts.items():
            row = self.case.effect_receipt(self.root, key)
            receipt = decode_workflow_receipt(row["facts"].encode())
            assert (receipt.action, receipt.resulting_status) == expected
            assert (row["action"], row["resulting_status"]) == (expected[0].value, expected[1])


@pytest.fixture(params=[False, True], ids=["two-step", "single-update"])
def acquisition_mode(request, monkeypatch):
    for backend in ("SQLITE", "PG"):
        monkeypatch.setenv(f"JOBS_{backend}_SINGLE_UPDATE_ACQUIRE", str(request.param))


@pytest.mark.parametrize("quarantine", [None, 2])
def test_real_store_state_machine(workflow_example_factory, acquisition_mode, quarantine):
    run_state_machine_as_test(
        lambda: WorkflowMachine(workflow_example_factory, quarantine),
        settings=settings(settings.get_profile("ci"), max_examples=8, stateful_step_count=35),
    )


TRACES = [
    [
        "acquire",
        "continue_",
        "acquire",
        "continue_",
        "acquire",
        "continue_",
        "acquire",
        "continue_",
        "acquire",
        "continue_",
        "replay",
        "release",
        "restart",
        "archive",
        "replay",
    ],
    ["acquire", "retry", "replay", "acquire", "retry", "acquire", "retry", "acquire", "retry", "replay", "archive"],
    [
        "acquire",
        "continue_",
        "acquire",
        "progress",
        "continue_",
        "acquire",
        "continue_",
        "acquire",
        "progress",
        "expire",
        "acquire",
        "renew",
        "complete",
        "archive",
        "restart",
        "replay",
    ],
    [
        "acquire",
        "expire",
        "acquire",
        "expire",
        "acquire",
        "expire",
        "acquire",
        "expire",
        "acquire",
        "expire",
        "acquire",
        "restart",
        "admit",
    ],
    ["acquire", "cancel", "retry", "complete", "restart", "archive", "admit"],
    [
        "acquire",
        ("retry", {"stale": True}),
        ("renew", {"stale": True}),
        "progress",
        ("progress", {"changed": True}),
        "continue_",
        ("complete", {"stale": True}),
        "acquire",
        "renew",
        "retry",
        "replay",
        ("replay", {"changed": True}),
        "acquire",
        "complete",
    ],
]


@pytest.mark.parametrize(
    "trace", TRACES, ids=["exhaustion", "retry-budget", "progress-replay", "expiry", "cancel", "invalid-stale"]
)
def test_identical_backend_traces(workflow_example_factory, acquisition_mode, trace):
    machine = WorkflowMachine(workflow_example_factory)
    try:
        for command in trace:
            name, arguments = command if isinstance(command, tuple) else (command, {})
            getattr(machine, name)(**arguments)
            machine.check_store()
        machine.replay(changed=True)
        machine.renew(stale=True)
        machine.check_store()
    finally:
        machine.teardown()


@pytest.mark.parametrize("quarantine", [None, 2])
def test_retry_and_no_progress_budgets_are_independent(workflow_example_factory, acquisition_mode, quarantine):
    machine = WorkflowMachine(workflow_example_factory, quarantine)
    try:
        for command in ["acquire", "continue_", "acquire", "retry", "replay", "acquire", "retry"]:
            getattr(machine, command)()
            machine.check_store()
        assert machine.model.no_progress_count == 1
        assert machine.model.retry_count == 2
        assert machine.model.status == ("quarantined" if quarantine else "queued")
    finally:
        machine.teardown()


@pytest.mark.parametrize("kind", list(WorkflowProgressKind))
def test_each_progress_kind_resets_once_across_epochs(workflow_example_factory, acquisition_mode, kind):
    machine = WorkflowMachine(workflow_example_factory)
    try:
        machine.acquire()
        machine.continue_()
        machine.acquire()
        machine.progress(kind=kind)
        machine.check_store()
        assert machine.model.no_progress_count == 0
        machine.continue_()
        machine.acquire()
        machine.continue_()
        machine.acquire()
        machine.progress(kind=kind)
        machine.check_store()
        assert machine.model.no_progress_count == 1
        machine.complete()
        machine.check_store()
    finally:
        machine.teardown()
