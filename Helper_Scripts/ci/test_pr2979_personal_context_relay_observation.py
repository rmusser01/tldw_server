"""Synthetic delegation/privacy controls; no application or database imports."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pr2979_personal_context_relay_observation as probe
import pytest

TARGET = "test_abandoned_preparation_expires_without_covering_source"
PRIVATE = "private-profile-ciphertext-exception-sentinel"


@pytest.fixture()
def world(monkeypatch, tmp_path):
    calls = []
    result = SimpleNamespace(
        staged_rows=0,
        inspected_rows=1,
        source_exhausted=False,
        visible_lookahead=False,
        continuation="personal_context_relay_pending",
    )

    class Budget:
        def __init__(self):
            self.remaining_rows = 1
            self.deadline_ns = time.monotonic_ns() + 100_000_000

        def deadline_open(self):
            return time.monotonic_ns() < self.deadline_ns

        def can_inspect(self):
            return self.remaining_rows > 0 and self.deadline_open()

        def consume(self):
            return self.consume_returned()

        def consume_returned(self):
            self.remaining_rows -= 1
            return self.deadline_open()

    class Store:
        suppress = False
        granted = True

        @contextmanager
        def profile_lease(self, *args, **kwargs):
            calls.append("enter")
            try:
                yield object() if self.granted else None
            except RuntimeError:
                if not self.suppress:
                    raise
            finally:
                calls.append("exit")

        def unfinished_stage_identities(self, *args, **kwargs):
            calls.append("unfinished")
            return ()

        def earliest_nonterminal_batch(self, *args, **kwargs):
            calls.append("earliest")
            kwargs["budget"].consume_returned()
            return None

        def renew_lease(self, *args, **kwargs):
            return True

    class Relay:
        def __init__(self):
            self.publications = Store()
            self.error = None

        def relay_profile(self, **kwargs):
            budget = Budget()
            with self.publications.profile_lease(PRIVATE, blocking=False) as lease:
                if lease is not None:
                    return self._relay_owned(budget=budget)
            return result

        def _relay_owned(self, *, budget):
            if self.error is not None:
                raise self.error
            if budget.can_inspect():
                self.publications.unfinished_stage_identities(PRIVATE, budget=budget)
                self.publications.earliest_nonterminal_batch(PRIVATE, budget=budget)
            return result

    for name, values in (
        (
            "tldw_Server_API.app.core.Sync.v2.personal_context_relay",
            {"PersonalContextRelay": Relay, "PersonalContextRecoveryBudget": Budget},
        ),
        (
            "tldw_Server_API.app.core.Personalization.personal_context_publication",
            {"PersonalContextPublicationRelayStore": Store},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("UAT558_PROBE_OUTPUT", str(tmp_path / "observation"))
    item = SimpleNamespace(name=TARGET + "[True]", module=ModuleType("synthetic_test"))
    originals = (Relay.relay_profile, Relay._relay_owned, Store.profile_lease, Budget.deadline_open)
    return SimpleNamespace(
        Relay=Relay,
        Store=Store,
        Budget=Budget,
        result=result,
        calls=calls,
        item=item,
        originals=originals,
        tmp=tmp_path,
    )


def finish(hook, world):
    with pytest.raises(StopIteration):
        next(hook)
    assert (
        world.Relay.relay_profile,
        world.Relay._relay_owned,
        world.Store.profile_lease,
        world.Budget.deadline_open,
    ) == world.originals


@pytest.mark.parametrize("legacy", [False, True])
def test_original_result_and_content_free_case_record(world, legacy):
    world.item.name = TARGET + f"[{legacy}]"
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    assert (
        world.Relay().relay_profile(
            user_id=PRIVATE, profile_id=PRIVATE, dataset_id=PRIVATE, after_server_cursor=None, row_budget=1
        )
        is world.result
    )
    finish(hook, world)
    output = (world.tmp / f"observation-legacy_{str(legacy).lower()}.json").read_text()
    assert PRIVATE not in output
    doc = json.loads(output)
    assert doc["case"] == f"legacy_{str(legacy).lower()}" and doc["complete"] is True
    assert doc["records"][0]["result"]["inspected_rows"] == 1
    assert world.calls == ["enter", "unfinished", "earliest", "exit"]
    assert {e["phase"] for e in doc["records"][0]["events"]} >= {
        "lease.enter",
        "lease.exit",
        "budget.deadline_open",
        "source.earliest_nonterminal_batch",
    }


@pytest.mark.parametrize("error", [None, RuntimeError(PRIVATE), SystemExit(11)])
def test_output_unavailable_preserves_original_outcome(world, monkeypatch, capsys, error):
    def unavailable(*args, **kwargs):
        raise OSError(PRIVATE)

    monkeypatch.setattr(Path, "write_text", unavailable)
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    relay = world.Relay()
    relay.error = error
    if error is None:
        assert relay.relay_profile() is world.result
    else:
        with pytest.raises(type(error)) as raised:
            relay.relay_profile()
        assert raised.value is error
    finish(hook, world)
    assert capsys.readouterr().out.strip() == "UAT558_DIAGNOSTIC_UNAVAILABLE"
    assert world.calls.count("enter") == world.calls.count("exit") == 1


def test_original_context_manager_suppression_and_release(world):
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    relay = world.Relay()
    relay.error = RuntimeError(PRIVATE)
    relay.publications.suppress = True
    assert relay.relay_profile() is world.result
    finish(hook, world)
    assert world.calls == ["enter", "exit"]


def test_unrelated_case_has_no_patch_or_record(world):
    world.item.name = "unrelated"
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    finish(hook, world)
    assert not list(world.tmp.iterdir())


def test_missing_optional_method_delegates_without_observation(world, monkeypatch, capsys):
    monkeypatch.delattr(world.Store, "renew_lease")
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    assert world.Relay().relay_profile() is world.result
    finish(hook, world)
    assert capsys.readouterr().out.strip() == "UAT558_DIAGNOSTIC_UNAVAILABLE"


def test_clock_memory_fault_does_not_replace_result(world, monkeypatch):
    def unavailable():
        raise MemoryError

    monkeypatch.setattr(probe.time, "perf_counter_ns", unavailable)
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    assert world.Relay().relay_profile() is world.result
    finish(hook, world)
    doc = json.loads((world.tmp / "observation-legacy_true.json").read_text())
    assert doc["complete"] is False


def test_original_unavailable_lease_retains_outcome(world):
    hook = probe.pytest_runtest_call(world.item)
    next(hook)
    relay = world.Relay()
    relay.publications.granted = False
    assert relay.relay_profile() is world.result
    finish(hook, world)
    doc = json.loads((world.tmp / "observation-legacy_true.json").read_text())
    assert next(e for e in doc["records"][0]["events"] if e["phase"] == "lease.enter")["granted"] is False
    assert world.calls == ["enter", "exit"]
