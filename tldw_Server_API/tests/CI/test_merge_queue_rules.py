"""Table tests for the merge queue's pure rules (spec 4.1 and section 6)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from Helper_Scripts.ci import merge_queue as mq

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
BACKEND, E2E, CONTAINER = "backend-required", "e2e-required", "container-build-check"
LICENSE = "frontend-license-policy/trusted/dev"


def _run(context=BACKEND, conclusion="success", minutes_ago=1, status="completed", url="u", started_ago=None):
    done = NOW - timedelta(minutes=minutes_ago) if status == "completed" else None
    started = NOW - timedelta(minutes=started_ago) if started_ago is not None else None
    return mq.CheckRun(context, status, conclusion if status == "completed" else None, done, url, started_at=started)


def _green(minutes_ago=1, without=()):
    """Every required context passed, except the ones named in `without`."""
    return tuple(_run(name, minutes_ago=minutes_ago) for name in mq.ALL_CONTEXTS if name not in without)


def _pr(**kw):
    base = {
        "number": 1, "node_id": "PR_1", "head_sha": "a" * 40, "head_ref": "feat/x", "same_repo": True,
        "armed_at": NOW - timedelta(hours=1), "is_draft": False, "merge_state": "CLEAN",
        "head_committed_at": NOW - timedelta(hours=2), "base_sha": "d" * 40, "checks": (),
    }
    base.update(kw)
    return mq.PrState(**base)


def test_the_seven_required_contexts_and_how_each_is_started():
    """Spec 4.1 table: six check runs dispatched on the PR branch, one status dispatched on dev."""
    assert [(c.name, c.workflow, c.kind, c.base_sha) for c in mq.CONTEXTS] == [
        ("backend-required", "backend-required.yml", "check", True),
        ("security-required", "security-required.yml", "check", True),
        ("coverage-required", "coverage-required.yml", "check", True),
        ("frontend-required", "frontend-required.yml", "check", True),
        ("e2e-required", "e2e-required.yml", "check", True),
        ("container-build-check", "container-build-check.yml", "check", False),
        ("frontend-license-policy/trusted/dev", "frontend-license-gate.yml", "status", True),
    ]
    assert mq.REPO.count("/") == 1 and mq.BASE == "dev"


def test_line_is_armed_non_draft_same_repo_oldest_first():
    a = _pr(number=1, armed_at=NOW - timedelta(minutes=5))
    b = _pr(number=2, armed_at=NOW - timedelta(minutes=50))
    unarmed = _pr(number=3, armed_at=None)
    draft = _pr(number=4, is_draft=True)
    fork = _pr(number=5, same_repo=False)
    bot = _pr(number=6, human_author=False)
    assert [p.number for p in mq.line_of([a, b, unarmed, draft, fork, bot])] == [2, 1]


def test_rearmed_pr_rejoins_at_the_back():
    first = _pr(number=1, armed_at=NOW - timedelta(minutes=30))
    rearmed = _pr(number=2, armed_at=NOW - timedelta(minutes=1))  # was first before eviction
    assert [p.number for p in mq.line_of([rearmed, first])] == [1, 2]


@pytest.mark.parametrize(
    ("runs", "state"),
    [
        ((), "missing"),
        ((_run(conclusion="cancelled"),), "missing"),
        ((_run(status="in_progress"),), "running"),
        ((_run(status="queued"), _run(conclusion="failure", minutes_ago=9)), "running"),
        ((_run(),), "passed"),
        ((_run(conclusion="neutral"),), "passed"),
        ((_run(conclusion="skipped"),), "passed"),
        ((_run(conclusion="failure"),), "failed"),
        ((_run(conclusion="timed_out"),), "failed"),
        ((_run(conclusion="failure", minutes_ago=9), _run(conclusion="cancelled", minutes_ago=2)), "failed"),
        ((_run(conclusion="failure", minutes_ago=9), _run(conclusion="failure", minutes_ago=2)), "failed twice"),
        ((_run(conclusion="failure", minutes_ago=9), _run(minutes_ago=2)), "passed"),
        ((_run(minutes_ago=9), _run(conclusion="failure", minutes_ago=2)), "failed"),
        ((_run(status="pending", started_ago=5),), "running"),
        ((_run(status="pending", started_ago=31),), "missing"),
        ((_run(conclusion="failure", minutes_ago=40), _run(status="pending", started_ago=31)), "failed"),
    ],
    ids=[
        "no-run", "only-cancelled", "live", "live-after-a-failure", "success", "neutral", "skipped",
        "one-failure", "timed-out", "failure-then-cancelled", "two-failures", "retry-that-passed",
        "failure-after-a-pass", "pending-status", "stale-pending-status", "stale-pending-after-a-failure",
    ],
)
def test_context_state_table(runs, state):
    assert mq.context_state(list(runs), NOW) == state


@pytest.mark.parametrize(
    ("pr", "kind", "contexts"),
    [
        (_pr(merge_state="UNKNOWN"), "wait", ()),
        (_pr(merge_state="BEHIND"), "rebase", mq.ALL_CONTEXTS),
        (_pr(merge_state="DIRTY"), "evict", ()),
        # Spec 4.1, one row each.
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)) + (_run(E2E, status="in_progress"),)),
         "wait", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)) + (
            _run(E2E, "failure", 30), _run(E2E, "failure", 2))), "evict", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)) + (_run(E2E, "failure"),)), "retry", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E, LICENSE))), "dispatch", (E2E, LICENSE)),
        (_pr(merge_state="BLOCKED", checks=()), "dispatch", mq.ALL_CONTEXTS),
        (_pr(merge_state="BLOCKED", checks=tuple(_run(n, "cancelled") for n in mq.ALL_CONTEXTS)),
         "dispatch", mq.ALL_CONTEXTS),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)), head_committed_at=NOW - timedelta(minutes=2)),
         "wait", (E2E,)),
        # Section 6, the green rows.
        (_pr(merge_state="CLEAN", checks=_green(5)), "wait", ()),
        (_pr(merge_state="UNSTABLE", checks=_green(5)), "wait", ()),
        (_pr(merge_state="CLEAN", checks=_green(16)), "evict", ()),
        (_pr(merge_state="CLEAN", checks=_green(16, without=(E2E,)) + (_run(E2E, minutes_ago=14),)), "wait", ()),
        (_pr(merge_state="BLOCKED", checks=_green(), unresolved_threads=1), "evict", ()),
        (_pr(merge_state="BLOCKED", checks=_green(), unresolved_threads=0), "wait", ()),
        (_pr(merge_state="CLEAN", checks=_green(3) + (_run(E2E, "failure", 20),)), "wait", ()),
        (_pr(merge_state="DRAFT", checks=_green()), "wait", ()),
        # Combinations: the order of the 4.1 rows.
        (_pr(merge_state="BLOCKED", checks=(_run(BACKEND, "failure", 30), _run(BACKEND, "failure", 2),
                                            _run(E2E, "failure"))), "evict", (BACKEND,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E, CONTAINER)) + (_run(E2E, "failure"),)),
         "retry", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E, CONTAINER)) + (_run(E2E, status="queued"),)),
         "wait", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)) + (_run(E2E, "failure"),), unresolved_threads=2),
         "retry", (E2E,)),
    ],
    ids=[
        "unknown-waits", "behind-rebases-and-starts-all-seven", "dirty-evicts",
        "running-waits", "failed-twice-evicts", "failed-once-retries-only-that-context",
        "missing-dispatches-only-those-contexts", "no-run-at-all-dispatches-all-seven",
        "only-cancelled-dispatches", "missing-on-a-young-head-waits",
        "green-clean-waits", "green-unstable-waits", "stuck-green-evicts",
        "green-is-timed-from-the-last-context-to-finish", "green-blocked-unresolved-evicts",
        "green-blocked-nothing-unresolved-waits", "retry-that-passed-waits", "green-other-merge-state-waits",
        "failed-twice-beats-failed-once", "failed-once-beats-missing", "running-beats-missing",
        "a-red-context-is-retried-before-threads-matter",
    ],
)
def test_decide_front_table(pr, kind, contexts):
    action = mq.decide_front(pr, NOW)
    assert (action.kind, action.contexts) == (kind, contexts)


def test_a_failure_is_acted_on_while_other_contexts_are_still_running():
    """Only a FAILED gate wakes the queue (spec 4.6). If it waited for the slower gates, and they
    then passed, nothing would wake it again and the failed context would never be retried."""
    still_running = tuple(_run(n, status="in_progress") for n in mq.ALL_CONTEXTS if n != BACKEND)
    once = mq.decide_front(_pr(merge_state="BLOCKED", checks=still_running + (_run(BACKEND, "failure"),)), NOW)
    assert (once.kind, once.contexts) == ("retry", (BACKEND,))
    twice = mq.decide_front(_pr(merge_state="BLOCKED", checks=still_running + (
        _run(BACKEND, "failure", 30), _run(BACKEND, "failure", 2))), NOW)
    assert (twice.kind, twice.contexts, twice.slug) == ("evict", (BACKEND,), "failed-twice")


def test_a_context_being_retried_is_running_not_failed():
    """The retry is in flight: no second retry, no eviction, until it finishes."""
    checks = _green(without=(BACKEND,)) + (_run(BACKEND, "failure", 9), _run(BACKEND, status="queued"))
    action = mq.decide_front(_pr(merge_state="BLOCKED", checks=checks), NOW)
    assert (action.kind, action.contexts) == ("wait", (BACKEND,))


def test_failed_twice_links_both_runs_of_each_such_context():
    checks = _green(without=(BACKEND, LICENSE)) + (
        _run(BACKEND, "failure", 40, url="b0"), _run(BACKEND, "failure", 30, url="b1"),
        _run(BACKEND, "failure", 2, url="b2"),
        _run(LICENSE, "failure", 20, url="l1"), _run(LICENSE, "failure", 1, url="l2"),
    )
    action = mq.decide_front(_pr(merge_state="BLOCKED", checks=checks), NOW)
    assert action.links == ("b1", "b2", "l1", "l2")
    assert action.reason == f"{BACKEND}, {LICENSE} failed twice"


def test_retry_links_the_failed_run_of_each_failed_context():
    checks = _green(without=(BACKEND, E2E)) + (_run(BACKEND, "failure", url="b"), _run(E2E, "failure", url="e"))
    action = mq.decide_front(_pr(merge_state="BLOCKED", checks=checks), NOW)
    assert (action.kind, action.contexts, action.links) == ("retry", (BACKEND, E2E), ("b", "e"))


def test_a_run_of_an_unknown_context_is_ignored():
    checks = _green() + (mq.CheckRun("ci / lint", "completed", "failure", NOW, "x"),)
    assert mq.decide_front(_pr(merge_state="CLEAN", checks=checks), NOW).kind == "wait"
