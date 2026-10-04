"""The merge queue's read layer, action layer and run loop, against a fake gh (spec sections 4, 6, 7)."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pytest

from Helper_Scripts.ci import merge_queue as mq

pytestmark = pytest.mark.unit

SCRIPT = Path(mq.__file__)
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
OLD = "a" * 40
NEW = "b" * 40
DEV = "d" * 40
BACKEND, E2E, CONTAINER = "backend-required", "e2e-required", "container-build-check"
LICENSE = "frontend-license-policy/trusted/dev"
LICENSE_GATE = "frontend-license-gate.yml"
CHECK_NAMES = tuple(c.name for c in mq.CONTEXTS if c.kind == "check")


@pytest.fixture(autouse=True)
def _no_real_run_id(monkeypatch):
    """Isolate GITHUB_RUN_ID: these tests must not depend on (or be confused by) this
    process's own CI run id. Tests that exercise self-exclusion set it explicitly."""
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)


def _node(number, *, head=OLD, armed="2026-10-04T10:00:00Z", state="BEHIND", repo=None, draft=False,
          committed="2026-10-04T09:00:00Z", ref=None, threads=(), author="User", base=DEV):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "headRefOid": head,
        "headRefName": ref or f"feat/{number}", "mergeStateStatus": state,
        "author": {"__typename": author} if author else None,
        "headRepository": {"nameWithOwner": repo or mq.REPO},
        "baseRef": {"target": {"oid": base}} if base else None,
        "autoMergeRequest": {"enabledAt": armed} if armed else None,
        "commits": {"nodes": [{"commit": {"committedDate": committed}}]},
        "reviewThreads": {"nodes": [{"isResolved": resolved} for resolved in threads]},
    }


def _check(name=BACKEND, conclusion="success", completed="2026-10-04T11:58:00Z", url=None, suite=None):
    check = {"name": name, "status": "completed", "conclusion": conclusion, "completed_at": completed,
             "html_url": url or f"https://check/{name}"}
    if suite is not None:
        check["check_suite"] = {"id": suite}
    return check


def _checks(overrides=None):
    """The six check-run contexts, all passed; `overrides` maps a context to another conclusion,
    or to None for no run at all."""
    conclusions = {name: "success" for name in CHECK_NAMES}
    conclusions.update(overrides or {})
    return [_check(name, conclusion) for name, conclusion in conclusions.items() if conclusion is not None]


def _status(state="success", created="2026-10-04T11:58:00Z", url="https://license/1", context=LICENSE):
    return {"context": context, "state": state, "created_at": created, "target_url": url}


def _green(overrides=None, statuses=None):
    """FakeGh keyword arguments for a head whose seven contexts all passed, with exceptions."""
    return {"checks": {OLD: _checks(overrides)}, "statuses": {OLD: [_status()] if statuses is None else statuses}}


def _run_of(workflow, rid, status="in_progress", conclusion=None, event="workflow_dispatch", **extra):
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": status,
            "conclusion": conclusion, "html_url": f"https://run/{rid}", **extra}


def _dispatch_call(context, ref="feat/1", number=1, base=DEV):
    """The exact dispatch the queue sends for one context (spec 4.1 table)."""
    ctx = next(c for c in mq.CONTEXTS if c.name == context)
    if ctx.kind == "status":
        return ("dispatch", ctx.workflow, {"ref": "dev", "inputs[pr]": str(number)})
    if not ctx.base_sha:
        return ("dispatch", ctx.workflow, {"ref": ref})
    return ("dispatch", ctx.workflow, {"ref": ref, "inputs[base_sha]": base})


# The license gate goes first, then the six gates in table order.
ALL_SEVEN = [LICENSE, *CHECK_NAMES]


# gh's real error format for API failures: `gh: <message> (HTTP NNN)`.
DISPATCH_ERRORS = {
    422: "gh: Workflow does not have 'workflow_dispatch' trigger (HTTP 422)",
    404: "gh: No ref found for: feat/1 (HTTP 404)",
    403: "gh: Resource not accessible by integration (HTTP 403)",
    502: "gh: Server Error (HTTP 502)",
}


class FakeGh:
    """Records every mutating call; serves scripted reads.

    Like the real API, the rebase mutation returns the PRE-rebase head and the branch moves a
    moment later: the first `rebase_lag` single-PR rereads after a successful rebase still show
    the old head, later ones show NEW (and `rebased_base` as dev's tip, if given).
    `rebase_lag=None`: it never moves. `events` holds the recorded calls interleaved with
    ("read_pr", head) for every single-PR read. A `reread` value may be a list: successive
    single-PR reads walk it, the last entry sticks.

    Lists page like the real API: the PR line `line_page_size` at a time through a cursor, REST
    lists `mq.PER_PAGE` at a time. Check runs are served per `check_name`, as the real endpoint
    filters them; `check_names_read` records the names asked for. Statuses come back as a bare
    list, newest first.

    `late_runs[sha]` joins the head's runs from its second unfiltered runs read on, and
    `late_statuses[sha]` its statuses from the second statuses read on: a run another queue run
    started after this one decided. A dispatch of a `dispatch_refused` workflow fails with
    `DISPATCH_ERRORS[dispatch_status]`.
    """

    def __init__(self, nodes, *, checks=None, statuses=None, runs=None, comments=None, rebase_error=False,
                 reread=None, dispatch_refused=(), rebase_lag=1, disarm_error=False, line_page_size=None,
                 late_runs=None, late_statuses=None, dispatch_status=422, rebased_base=None):
        self.nodes = {n["number"]: n for n in nodes}
        self.checks = checks or {}
        self.statuses = statuses or {}
        self.runs = runs or {}
        self.comments = comments or {}
        self.rebase_error = rebase_error
        self.reread = reread or {}
        self.dispatch_refused = set(dispatch_refused)
        self.rebase_lag = rebase_lag
        self.disarm_error = disarm_error
        self.line_page_size = line_page_size
        self.late_runs = late_runs or {}
        self.late_statuses = late_statuses or {}
        self.dispatch_status = dispatch_status
        self.rebased_base = rebased_base
        self.runs_reads = {}
        self.statuses_reads = {}
        self.reread_counts = {}
        self.rereads_since_rebase = None
        self.check_names_read = []
        self.calls = []
        self.events = []
        self.reads = 0
        self.line_cursors = []

    def _record(self, call):
        self.calls.append(call)
        self.events.append(call)

    def dispatched(self):
        return [c for c in self.calls if c[0] == "dispatch"]

    def graphql(self, query, **v):
        if "updatePullRequestBranch" in query:
            self._record(("rebase", v["id"], v["oid"]))
            if self.rebase_error:
                raise mq.GhError("rebase refused")
            self.rereads_since_rebase = 0
            return {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": v["oid"]}}}}
        if "disablePullRequestAutoMerge" in query:
            self._record(("disarm", v["id"]))
            if self.disarm_error:
                raise mq.GhError("Pull request is not in the correct state to disable auto-merge")
            return {"data": {}}
        self.reads += 1
        if "comments(last" in query:
            bodies = self.comments.get(v["number"], [])
            return {"data": {"repository": {"pullRequest": {"comments": {"nodes": [{"body": b} for b in bodies]}}}}}
        if "pullRequest(number" in query:
            node = self.reread.get(v["number"], self.nodes[v["number"]])
            if isinstance(node, list):
                seen = self.reread_counts.get(v["number"], 0)
                self.reread_counts[v["number"]] = seen + 1
                node = node[min(seen, len(node) - 1)]
            if self.rereads_since_rebase is not None:
                self.rereads_since_rebase += 1
                if self.rebase_lag is not None and self.rereads_since_rebase > self.rebase_lag:
                    node = dict(node, headRefOid=NEW, mergeStateStatus="BLOCKED")
                    if self.rebased_base:
                        node["baseRef"] = {"target": {"oid": self.rebased_base}}
            self.events.append(("read_pr", node["headRefOid"]))
            return {"data": {"repository": {"pullRequest": node}}}
        if "pullRequests(" in query:
            nodes = list(self.nodes.values())
            self.line_cursors.append(v.get("after"))
            size = self.line_page_size or len(nodes) or 1
            start = int(v.get("after") or 0)
            more = start + size < len(nodes)
            return {"data": {"repository": {"pullRequests": {
                "pageInfo": {"hasNextPage": more, "endCursor": str(start + size) if more else None},
                "nodes": nodes[start:start + size],
            }}}}
        raise AssertionError(f"unexpected query {query[:60]!r}")

    @staticmethod
    def _page(items, path):
        page = int(re.search(r"[?&]page=(\d+)", path).group(1))
        assert f"per_page={mq.PER_PAGE}" in path
        return items[(page - 1) * mq.PER_PAGE:page * mq.PER_PAGE]

    def rest(self, method, path, fields=None):
        if method == "GET" and "/check-runs" in path:
            self.reads += 1
            assert "filter=all" in path, "a re-run's earlier failure must stay visible"
            name = unquote(re.search(r"[?&]check_name=([^&]+)", path).group(1))
            self.check_names_read.append(name)
            sha = path.split("/commits/")[1].split("/")[0]
            checks = [c for c in self.checks.get(sha, []) if c["name"] == name]
            return {"total_count": len(checks), "check_runs": self._page(checks, path)}
        if method == "GET" and "/statuses" in path:
            self.reads += 1
            sha = path.split("/commits/")[1].split("/")[0]
            if "page=1" in re.findall(r"[?&](page=\d+)", path):
                self.statuses_reads[sha] = self.statuses_reads.get(sha, 0) + 1
            statuses = list(self.statuses.get(sha, []))
            if self.statuses_reads.get(sha, 0) >= 2:
                statuses += self.late_statuses.get(sha, [])
            newest_first = sorted(statuses, key=lambda s: s["created_at"], reverse=True)
            return self._page(newest_first, path)
        if method == "GET" and "/actions/runs?" in path:
            self.reads += 1
            sha = path.split("head_sha=")[1].split("&")[0]
            runs = self.runs.get(sha, [])
            if "status=action_required" in path:
                runs = [r for r in runs if r.get("conclusion") == "action_required"]
            elif "page=1" in re.findall(r"[?&](page=\d+)", path):
                self.runs_reads[sha] = self.runs_reads.get(sha, 0) + 1
            if self.runs_reads.get(sha, 0) >= 2 and "status=" not in path:
                runs = runs + self.late_runs.get(sha, [])
            return {"total_count": len(runs), "workflow_runs": self._page(runs, path)}
        if method == "POST" and path.endswith("/dispatches"):
            workflow = path.split("/workflows/")[1].split("/")[0]
            self._record(("dispatch", workflow, dict(fields or {})))
            if workflow in self.dispatch_refused:
                raise mq.GhError(DISPATCH_ERRORS[self.dispatch_status])
            return None
        if method == "POST" and path.endswith("/cancel"):
            self._record(("cancel", path.split("/runs/")[1].split("/")[0]))
            return None
        if method == "DELETE" and "/actions/runs/" in path:
            self._record(("delete", path.rsplit("/", 1)[1]))
            return None
        if method == "POST" and path.endswith("/comments"):
            self._record(("comment", int(path.split("/issues/")[1].split("/")[0]), fields["body"]))
            return None
        raise AssertionError(f"unexpected rest call {method} {path}")


def _run(gh, mode="on"):
    return mq.run(gh, mode, now=lambda: NOW, sleep=lambda s: None, log=lambda m: None)


# --- modes -----------------------------------------------------------------------------------


def test_mode_values():
    for mode in ("", "off", "yes", "true", None):
        gh = FakeGh([_node(1)])
        assert _run(gh, mode) == [] and gh.calls == [] and gh.reads == 0
    for mode in ("On", " on ", "DRY"):
        gh = FakeGh([_node(1)])
        assert _run(gh, mode)[0][1].kind == "rebase"


@pytest.mark.parametrize("front", [
    {"state": "DIRTY"},
    {"state": "BLOCKED"},
    {"state": "BLOCKED", "repo": "someone/fork"},
], ids=["evict", "dispatch", "fork-note"])
def test_dry_mode_makes_no_mutating_calls(front):
    gh = FakeGh([_node(1, **front), _node(2, armed="2026-10-04T11:00:00Z")],
                runs={OLD: [_run_of("ci.yml", 21, "completed", "action_required",
                                    triggering_actor={"login": "github-actions[bot]"})]})
    decisions = _run(gh, "dry")
    assert decisions and gh.calls == []


def test_dry_mode_decides_for_the_next_front_after_an_eviction():
    gh = FakeGh([_node(1, state="DIRTY"), _node(2, armed="2026-10-04T11:00:00Z")])
    assert [(n, a.kind) for n, a in _run(gh, "dry")] == [(1, "evict"), (2, "rebase")]
    assert gh.calls == []


# --- reading the seven contexts --------------------------------------------------------------


def test_check_runs_are_read_by_name_for_exactly_the_six_check_contexts():
    """A head here carries 250+ check runs. Only the six required names are asked for, so the
    other checks on the head can never crowd a required one off the pages the queue reads."""
    noise = [_check(f"ci / shard {i}", "failure") for i in range(300)]
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: noise + _checks()}, statuses={OLD: [_status()]})
    assert _run(gh, "dry")[0][1].reason == "green; auto-merge should fire"
    assert gh.check_names_read == list(CHECK_NAMES)


def test_workflow_runs_are_read_before_check_runs():
    """A run seen as completed then already has its check runs, so 'completed without reporting
    its check' cannot be an artefact of the order the two lists were read in."""
    order = []

    class Recording(FakeGh):
        def rest(self, method, path, fields=None):
            order.append("runs" if "/actions/runs?" in path else "checks" if "/check-runs" in path else "statuses")
            return super().rest(method, path, fields)

    mq.read_contexts(Recording([]), OLD)
    assert order == ["runs"] + ["checks"] * 6 + ["statuses"]


def test_dev_tip_is_read_together_with_the_pr_state():
    assert "baseRef { target { oid } }" in mq.PR_FIELDS
    assert mq.PR_FIELDS in mq.LINE_QUERY and mq.PR_FIELDS in mq.PR_QUERY
    gh = FakeGh([_node(1, base="c" * 40)])
    assert [p.base_sha for p in mq.read_prs(gh)] == ["c" * 40]
    assert mq.read_pr(gh, 1).base_sha == "c" * 40


@pytest.mark.parametrize(("statuses", "kind", "contexts"), [
    ([], "dispatch", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:50:00Z")], "wait", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:50:00Z"), _status("success", "2026-10-04T11:52:00Z")], "wait", ()),
    ([_status("pending", "2026-10-04T11:50:00Z"), _status("failure", "2026-10-04T11:52:00Z")], "retry", (LICENSE,)),
    ([_status("error", "2026-10-04T11:52:00Z")], "retry", (LICENSE,)),
    ([_status("failure", "2026-10-04T11:40:00Z"), _status("pending", "2026-10-04T11:50:00Z")], "wait", (LICENSE,)),
    ([_status("failure", "2026-10-04T11:30:00Z"), _status("pending", "2026-10-04T11:40:00Z"),
      _status("success", "2026-10-04T11:50:00Z")], "wait", ()),
    ([_status("pending", "2026-10-04T11:20:00Z"), _status("failure", "2026-10-04T11:30:00Z"),
      _status("pending", "2026-10-04T11:40:00Z"), _status("error", "2026-10-04T11:50:00Z")], "evict", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:44:00Z")], "dispatch", (LICENSE,)),
    ([_status("success", context="frontend-license-policy/trusted/main"),
      _status("failure", context="qodo")], "dispatch", (LICENSE,)),
], ids=[
    "no-status-is-missing", "pending-is-running", "pending-then-success-passed", "failure-failed-once",
    "error-is-a-failure", "retry-in-flight-is-running", "retry-that-passed", "two-failures-failed-twice",
    "a-pending-older-than-15-minutes-is-a-lost-run", "other-contexts-are-ignored",
])
def test_the_license_context_is_read_from_the_commit_status_history(statuses, kind, contexts):
    """The license context is a commit status, not a check run. The status list keeps every
    status posted for the head: each success/failure/error is one finished run, and a pending
    counts only while it is the newest entry."""
    gh = FakeGh([_node(1, state="BLOCKED")], **_green(statuses=statuses))
    action = _run(gh, "dry")[0][1]
    assert (action.kind, action.contexts) == (kind, contexts)


def test_license_failed_twice_evicts_with_both_status_links():
    statuses = [_status("failure", "2026-10-04T11:30:00Z", "https://license/1"),
                _status("failure", "2026-10-04T11:50:00Z", "https://license/2")]
    gh = FakeGh([_node(1, state="BLOCKED")], **_green(statuses=statuses))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.links) == ("evict", ("https://license/1", "https://license/2"))
    assert ("disarm", "PR_1") in gh.calls and gh.dispatched() == []
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-failed-twice:{OLD} -->" in comment[2] and f"{LICENSE} failed twice" in comment[2]


def test_the_license_status_on_the_second_page_of_statuses_is_seen():
    """Statuses come newest first; a head with 100 newer statuses of other contexts must still
    show the license failure behind them."""
    noise = [_status("success", "2026-10-04T11:59:00Z", context=f"bot/{i}") for i in range(mq.PER_PAGE)]
    gh = FakeGh([_node(1, state="BLOCKED")],
                **_green(statuses=noise + [_status("failure", "2026-10-04T11:00:00Z")]))
    action = _run(gh, "dry")[0][1]
    assert (action.kind, action.contexts) == ("retry", (LICENSE,))


def test_more_statuses_than_the_page_cap_fails_instead_of_deciding_on_part_of_them():
    noise = [_status("success", context=f"bot/{i}") for i in range(mq.MAX_PAGES * mq.PER_PAGE)]
    with pytest.raises(mq.GhError, match="partial view"):
        mq.read_statuses(FakeGh([], statuses={OLD: noise}), OLD)


# --- dispatch contract (spec 4.1 table) ------------------------------------------------------


def test_dispatch_requests_per_context():
    pr = mq.read_pr(FakeGh([_node(7, ref="feat/queue-me")]), 7)
    assert {c.name: mq.dispatch_request(c, pr) for c in mq.CONTEXTS} == {
        "backend-required": ("backend-required.yml", "feat/queue-me", {"base_sha": DEV}),
        "security-required": ("security-required.yml", "feat/queue-me", {"base_sha": DEV}),
        "coverage-required": ("coverage-required.yml", "feat/queue-me", {"base_sha": DEV}),
        "frontend-required": ("frontend-required.yml", "feat/queue-me", {"base_sha": DEV}),
        "e2e-required": ("e2e-required.yml", "feat/queue-me", {"base_sha": DEV}),
        "container-build-check": ("container-build-check.yml", "feat/queue-me", {}),
        "frontend-license-policy/trusted/dev": ("frontend-license-gate.yml", "dev", {"pr": "7"}),
    }


def test_no_run_at_all_dispatches_all_seven_with_their_inputs():
    gh = FakeGh([_node(1, state="BLOCKED")])
    assert _run(gh)[0][1].kind == "dispatch"
    assert gh.calls == [_dispatch_call(name) for name in ALL_SEVEN]


def test_missing_contexts_only_are_dispatched():
    gh = FakeGh([_node(1, state="BLOCKED")], **_green({E2E: None, CONTAINER: "cancelled"}, statuses=[]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("dispatch", (E2E, CONTAINER, LICENSE))
    assert gh.calls == [_dispatch_call(LICENSE), _dispatch_call(E2E), _dispatch_call(CONTAINER)]


def test_first_failure_retries_only_the_failed_contexts():
    gh = FakeGh([_node(1, state="BLOCKED")], **_green({BACKEND: "failure"}))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))
    assert gh.dispatched() == [_dispatch_call(BACKEND)]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:retry:{OLD} -->" in comment[2]
    assert f"{BACKEND} failed once" in comment[2] and comment[2].endswith(f"\n- https://check/{BACKEND}")
    assert [c[0] for c in gh.calls] == ["dispatch", "comment"]


def test_a_failed_license_status_retries_only_the_license_gate_on_dev():
    """The license gate posts its status without a target_url: the comment then lists no link."""
    gh = FakeGh([_node(5, state="BLOCKED")], **_green(statuses=[_status("failure", url=None)]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.links) == ("retry", ())
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "5"})]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[2] == (f"<!-- merge-queue:retry:{OLD} -->\nMerge queue: {LICENSE} failed once on "
                          f"`{OLD[:10]}`; retrying with a fresh run.")


def test_a_failure_is_retried_while_other_contexts_are_still_running():
    """Only a failed gate wakes the queue: waiting for the others would lose the retry."""
    runs = {OLD: [_run_of("e2e-required.yml", 70)]}
    gh = FakeGh([_node(1, state="BLOCKED")], runs=runs, **_green({BACKEND: "failure", E2E: None}))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))
    assert gh.dispatched() == [_dispatch_call(BACKEND)]


def test_an_unknown_dev_tip_dispatches_nothing_and_fails_the_run():
    """Without base_sha a gate compares against HEAD^ and can pass with nothing tested."""
    gh = FakeGh([_node(1, state="BLOCKED", base=None)])
    with pytest.raises(mq.GhError, match="refusing to dispatch backend-required.yml without base_sha"):
        _run(gh)
    assert gh.calls == []


# --- rebase ----------------------------------------------------------------------------------


def test_on_mode_rebases_front_only_and_starts_exactly_the_seven_contexts():
    runs = {OLD: [
        _run_of("backend-required.yml", 11, event="pull_request"),
        _run_of("ci.yml", 12, event="pull_request"),
        _run_of("pre-commit.yml", 13, "completed", "success", event="pull_request"),
        _run_of("merge-queue.yml", 14, "completed", "success", event="pull_request"),
    ]}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-04T11:00:00Z")], runs=runs)
    _run(gh)
    assert ("rebase", "PR_1", OLD) in gh.calls
    assert gh.dispatched() == [_dispatch_call(name) for name in ALL_SEVEN], "ci.yml and the rest are not re-run"
    assert [c for c in gh.calls if c[0] == "cancel"] == [("cancel", "11"), ("cancel", "12")]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[1] == 1 and f"<!-- merge-queue:rebased:{NEW} -->" in comment[2]
    assert "started the 7 required checks" in comment[2]
    assert all(c[1] != "PR_2" for c in gh.calls if c[0] in ("rebase", "disarm"))
    assert [c[0] for c in gh.calls].count("rebase") == 1


def test_rebase_dispatches_with_the_dev_tip_read_alongside_the_new_head():
    tip = "e" * 40
    gh = FakeGh([_node(1)], rebased_base=tip)
    _run(gh)
    assert gh.dispatched() == [_dispatch_call(name, base=tip) for name in ALL_SEVEN]


def test_rebase_failure_with_moved_head_never_evicts():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, head=NEW, state="BEHIND")})
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase"]


def test_rebase_refused_on_conflict_evicts():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="DIRTY")})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert any(c[0] == "comment" and "conflicts with dev" in c[2] for c in gh.calls)


def test_first_rebase_failure_warns_without_evicting():
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="BEHIND")})
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase", "comment"]
    body = gh.calls[1][2]
    assert f"<!-- merge-queue:rebase-failed:{OLD} -->" in body
    assert "rebasing onto dev failed (rebase refused); will retry once" in body


def test_repeated_rebase_failure_evicts():
    """A PR that stays BEHIND (not DIRTY) while every rebase fails would otherwise stall the line."""
    gh = FakeGh([_node(1)], rebase_error=True, reread={1: _node(1, state="BEHIND")},
                comments={1: [f"<!-- merge-queue:rebase-failed:{OLD} -->\nfirst failure"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-rebase:{OLD} -->" in comment[2]
    assert "rebase onto dev keeps failing: rebase refused" in comment[2]


def test_refused_rebase_rereads_once_more_before_counting_it_as_a_failure():
    """A racing run's rebase was accepted, but the ref moves about 1 s later. Our pinned-head
    mutation is refused inside that window; the first reread still shows the old head. One more
    look after REBASE_POLL_S sees the moved head: no comment, no eviction."""
    sleeps = []
    gh = FakeGh([_node(1)], rebase_error=True,
                reread={1: [_node(1, state="BEHIND"), _node(1, head=NEW, state="BLOCKED")]})
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert [c[0] for c in gh.calls] == ["rebase"]
    assert sleeps == [mq.REBASE_POLL_S]


def test_rebase_dispatches_only_after_the_new_head_appears():
    """updatePullRequestBranch returns the PRE-rebase head and the branch moves about 1 s later.
    The queue polls until a reread shows the new head, and only then dispatches and comments
    with the NEW sha."""
    sleeps = []
    gh = FakeGh([_node(1)], rebase_lag=2)
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    first_dispatch = next(e for e in gh.events if e[0] == "dispatch")
    assert gh.events.index(("read_pr", NEW)) < gh.events.index(first_dispatch)
    assert sleeps == [mq.REBASE_POLL_S] * 3
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:rebased:{NEW} -->" in comment[2] and f"`{NEW[:10]}`" in comment[2]
    assert OLD not in comment[2]


def test_rebase_whose_head_never_moves_dispatches_nothing():
    sleeps = []
    gh = FakeGh([_node(1)], rebase_lag=None)
    mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert [c[0] for c in gh.calls] == ["rebase"]
    assert sleeps == [mq.REBASE_POLL_S] * mq.REBASE_POLLS


def test_rebase_dispatches_before_cancelling_and_spares_the_queue(monkeypatch):
    """A queue-tick runs inside its own required-workflow run (id 11), and a merge-queue.yml run
    (id 99) shares the old head too: cancelling either would kill the run doing the cancelling.
    Every other live run on the old head is cancelled, after all seven dispatches went out."""
    monkeypatch.setenv("GITHUB_RUN_ID", "11")
    runs = {OLD: [
        _run_of("merge-queue.yml", 99, event="pull_request"),
        _run_of("backend-required.yml", 11, event="pull_request"),
        _run_of("ci.yml", 12, event="pull_request"),
        _run_of("e2e-required.yml", 15, "queued", event="pull_request"),
    ]}
    gh = FakeGh([_node(1)], runs=runs)
    _run(gh)
    kinds = [c[0] for c in gh.calls]
    assert kinds[:8] == ["rebase"] + ["dispatch"] * 7
    assert [c for c in gh.calls if c[0] == "cancel"] == [("cancel", "12"), ("cancel", "15")]
    assert kinds.index("cancel") > max(i for i, k in enumerate(kinds) if k == "dispatch")


def test_rebase_deletes_the_new_heads_empty_approval_runs():
    runs = {NEW: [
        _run_of("ci.yml", 31, "completed", "action_required", triggering_actor={"login": "github-actions[bot]"}),
        _run_of("ci.yml", 32, "completed", "action_required", triggering_actor={"login": "someone"}),
        _run_of("backend-required.yml", 33),
    ]}
    gh = FakeGh([_node(1)], runs=runs)
    _run(gh)
    assert [c for c in gh.calls if c[0] == "delete"] == [("delete", "31")]
    kinds = [c[0] for c in gh.calls]
    assert kinds.index("delete") > max(i for i, k in enumerate(kinds) if k == "dispatch")


# --- eviction and comments -------------------------------------------------------------------


def test_evicted_front_hands_over_in_the_same_run():
    gh = FakeGh([_node(1, state="DIRTY"), _node(2, armed="2026-10-04T11:00:00Z")])
    _run(gh)
    assert gh.calls[0] == ("disarm", "PR_1")
    assert ("rebase", "PR_2", OLD) in gh.calls


def test_comments_are_deduplicated_by_marker():
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-conflict:{OLD} -->\nold"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert not any(c[0] == "comment" for c in gh.calls)


def test_a_second_retry_on_the_same_head_dispatches_without_a_second_comment():
    gh = FakeGh([_node(1, state="BLOCKED")], comments={1: [f"<!-- merge-queue:retry:{OLD} -->\nold"]},
                **_green({E2E: "failure"}))
    _run(gh)
    assert gh.calls == [_dispatch_call(E2E)]


def test_eviction_for_a_new_reason_on_the_same_head_still_comments():
    """A re-armed PR evicted again on the same head, for a different reason, must be told why."""
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-stuck:{OLD} -->\nold"]})
    _run(gh)
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-conflict:{OLD} -->" in comment[2]


def test_disarm_failure_still_comments():
    """A merge fires push:dev and pull_request:closed; the losing run's disarm hits an
    already-disarmed PR. That must not crash the run or skip the comment."""
    gh = FakeGh([_node(1, state="DIRTY")], disarm_error=True)
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert any(c[0] == "comment" and "conflicts with dev" in c[2] for c in gh.calls)


def test_blocked_green_evicts_only_with_unresolved_threads():
    """All green with BLOCKED is often mergeStateStatus lagging; only real unresolved
    conversations evict at once."""
    gh = FakeGh([_node(1, state="BLOCKED", threads=(True, False))], **_green())
    decision = _run(gh)[0][1]
    assert decision.kind == "evict" and "unresolved conversations" in decision.reason
    gh = FakeGh([_node(1, state="BLOCKED", threads=(True,))], **_green())
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def test_stuck_green_is_timed_from_the_last_of_the_seven_contexts():
    late_license = [_status(created="2026-10-04T11:50:00Z")]
    checks = {OLD: [_check(name, completed="2026-10-04T11:00:00Z") for name in CHECK_NAMES]}
    waiting = FakeGh([_node(1, state="CLEAN")], checks=checks, statuses={OLD: late_license})
    assert _run(waiting)[0][1].kind == "wait"
    stuck = FakeGh([_node(1, state="CLEAN")], checks=checks, statuses={OLD: [_status(created="2026-10-04T11:44:00Z")]})
    decision = _run(stuck)[0][1]
    assert (decision.kind, decision.slug) == ("evict", "stuck")
    assert ("disarm", "PR_1") in stuck.calls


def test_armed_fork_gets_one_comment_and_is_never_queued():
    gh = FakeGh([_node(1, repo="someone/fork", state="BEHIND")])
    decisions = _run(gh)
    assert decisions == []
    assert [c[0] for c in gh.calls] == ["comment"]
    assert f"<!-- merge-queue:fork:{OLD} -->" in gh.calls[0][2] and "fork" in gh.calls[0][2]
    again = FakeGh([_node(1, repo="someone/fork")], comments={1: [f"<!-- merge-queue:fork:{OLD} -->\nnote"]})
    _run(again)
    assert again.calls == []


def test_bot_authored_prs_are_never_queued():
    """A queue dispatch runs as github-actions[bot], so it would skip the token cap GitHub puts
    on Dependabot runs and the approval gate on agent pushes. Bot PRs are told once and left
    for a maintainer; the next user PR is the front."""
    gh = FakeGh([_node(1, author="Bot"), _node(2, armed="2026-10-04T11:00:00Z", state="DIRTY")])
    decisions = _run(gh)
    assert [n for n, _ in decisions] == [2]
    bot_comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:bot:{OLD} -->" in bot_comment[2] and "bot or app" in bot_comment[2]
    assert not any(c[0] in ("rebase", "dispatch", "disarm") and "PR_1" in c for c in gh.calls)
    assert _run(FakeGh([_node(1, author=None)]), "dry") == [], "a deleted (ghost) author is not a user"


# --- UNKNOWN, loop bound ---------------------------------------------------------------------


def test_unknown_state_is_reread_before_deciding():
    gh = FakeGh([_node(1, state="UNKNOWN")], reread={1: _node(1, state="DIRTY")})
    assert _run(gh)[0][1].kind == "evict"


def test_still_unknown_after_rereads_waits():
    """After each merge the next front is routinely UNKNOWN for a while: 12 rereads, 10 s apart."""
    sleeps = []
    gh = FakeGh([_node(1, state="UNKNOWN")], reread={1: _node(1, state="UNKNOWN")})
    decisions = mq.run(gh, "on", now=lambda: NOW, sleep=sleeps.append, log=lambda m: None)
    assert decisions[0][1].kind == "wait"
    assert sleeps == [10] * 12
    assert gh.calls == []


def test_max_fronts_per_run_is_bounded():
    nodes = [_node(i, state="DIRTY") for i in range(1, 13)]
    gh = FakeGh(nodes)
    decisions = _run(gh, "dry")
    assert len(decisions) == mq.MAX_FRONTS_PER_RUN == 10
    on = FakeGh(nodes)
    _run(on)
    assert [c[0] for c in on.calls].count("disarm") == 10


# --- workflow-run stand-ins ------------------------------------------------------------------


def test_live_required_run_without_check_waits():
    """A required check is a needs-gated job: it has NO check run while the jobs before it are
    still running. A live run of its workflow on the head must still read as 'in flight', not as
    'no run at all' (which would dispatch a duplicate on every wake)."""
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_run_of("e2e-required.yml", 77)]}, **_green({E2E: None}))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("wait", (E2E,))
    assert gh.calls == []


def test_live_retry_run_after_one_failure_waits():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_run_of("e2e-required.yml", 78)]},
                **_green({E2E: "failure"}))
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def test_a_live_run_of_another_workflow_is_not_a_required_context():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_run_of("ci.yml", 79), _run_of(LICENSE_GATE, 80)]},
                **_green({E2E: None}))
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts) == ("dispatch", (E2E,))


def test_own_run_is_not_counted_as_live(monkeypatch):
    """A queue-tick runs inside the required workflow run whose gate just failed."""
    monkeypatch.setenv("GITHUB_RUN_ID", "55")
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_run_of("backend-required.yml", 55)]},
                **_green({BACKEND: "failure"}))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))
    assert gh.dispatched() == [_dispatch_call(BACKEND)]


def test_queue_tick_failure_is_not_a_gate_failure():
    """The run failed (its queue-tick did), but its check suite reported the gate green."""
    runs = {OLD: [_run_of("backend-required.yml", 9, "completed", "failure", check_suite_id=900,
                          updated_at="2026-10-04T11:59:00Z")]}
    checks = _checks({BACKEND: None}) + [_check(BACKEND, suite=900)]
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: checks}, statuses={OLD: [_status()]}, runs=runs)
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def _broken_run(rid, conclusion, updated, workflow="security-required.yml"):
    """A required-workflow run that completed red without ever reporting its check."""
    return _run_of(workflow, rid, "completed", conclusion, check_suite_id=500 + rid, updated_at=updated)


def test_run_that_failed_without_reporting_its_check_counts_as_a_failure_of_that_context():
    """A startup failure (e.g. a broken workflow file on the branch) reports no check run. It
    must count as a failure of its own context -- one retries, two evict -- not as 'no run',
    which would dispatch forever."""
    green = _green({"security-required": None})
    once = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_broken_run(31, "startup_failure", "2026-10-04T11:00:00Z")]},
                  **green)
    decision = _run(once)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", ("security-required",))
    assert once.dispatched() == [_dispatch_call("security-required")]
    twice = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [
        _broken_run(31, "startup_failure", "2026-10-04T11:00:00Z"),
        _broken_run(32, "timed_out", "2026-10-04T11:30:00Z"),
    ]}, **green)
    decision = _run(twice)[0][1]
    assert decision.kind == "evict" and decision.links == ("https://run/31", "https://run/32")
    assert ("disarm", "PR_1") in twice.calls


def _queue_run(rid, conclusion="success", actor="github-actions[bot]", event="workflow_dispatch", suite=None):
    return _run_of("frontend-required.yml", rid, "completed", conclusion, event=event,
                   check_suite_id=suite or 700 + rid, updated_at=f"2026-10-04T11:{rid}:00Z",
                   triggering_actor={"login": actor})


def test_a_queue_dispatched_run_that_finished_green_without_its_check_is_a_failure_not_a_reason_to_redispatch():
    """frontend-required.yml on dev names its gate job `frontend-required-diagnostic` on a
    dispatch. A branch whose dispatched run never reports the required context would otherwise
    read as 'no run' after every green run, and be dispatched again at every wake, forever."""
    green = _green({"frontend-required": None})
    once = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_queue_run(31)]}, **green)
    decision = _run(once, "dry")[0][1]
    assert (decision.kind, decision.contexts) == ("retry", ("frontend-required",))
    twice = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_queue_run(31), _queue_run(32)]}, **green)
    decision = _run(twice, "dry")[0][1]
    assert (decision.kind, decision.links) == ("evict", ("https://run/31", "https://run/32"))


@pytest.mark.parametrize("run", [
    _queue_run(31, actor="someone"),
    _queue_run(31, event="pull_request"),
    _queue_run(31, conclusion="cancelled"),
    _queue_run(31, conclusion="skipped"),
], ids=["a-human-diagnostic-dispatch", "a-pull-request-run", "cancelled", "skipped"])
def test_other_runs_without_the_check_are_not_failures(run):
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [run]}, **_green({"frontend-required": None}))
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts) == ("dispatch", ("frontend-required",))


def test_a_queue_dispatched_run_that_reported_its_check_is_judged_by_the_check():
    checks = _checks({"frontend-required": None}) + [_check("frontend-required", suite=731)]
    gh = FakeGh([_node(1, state="CLEAN")], checks={OLD: checks}, statuses={OLD: [_status()]},
                runs={OLD: [_queue_run(31)]})
    assert _run(gh, "dry")[0][1].reason == "green; auto-merge should fire"


def test_a_broken_run_of_one_workflow_does_not_fail_another_context():
    """Suite ids are matched per context: a suite that reported e2e says nothing about backend."""
    runs = {OLD: [_run_of("backend-required.yml", 41, "completed", "failure", check_suite_id=900,
                          updated_at="2026-10-04T11:00:00Z")]}
    checks = _checks({BACKEND: None, E2E: None}) + [_check(E2E, suite=900)]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, statuses={OLD: [_status()]}, runs=runs)
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))


def test_cleanup_deletes_only_bot_approval_runs():
    runs = {OLD: [
        {"id": 21, "conclusion": "action_required", "triggering_actor": {"login": "github-actions[bot]"}},
        {"id": 22, "conclusion": "action_required", "triggering_actor": {"login": "someone"}},
    ]}
    gh = FakeGh([_node(1, state="CLEAN")], runs=runs, **_green())
    _run(gh)
    assert gh.calls == [("delete", "21")]


# --- racing queue runs -----------------------------------------------------------------------


def test_a_run_that_appeared_since_the_decision_stops_only_that_contexts_dispatch():
    """merge-queue.yml and a queue-tick can decide the same dispatch for one head. The one that
    dispatches second re-reads the live runs first and stands down for the contexts already
    started."""
    gh = FakeGh([_node(1, state="BLOCKED")], late_runs={OLD: [_run_of("backend-required.yml", 88)]},
                **_green({BACKEND: "failure", E2E: "failure"}))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND, E2E))
    assert gh.dispatched() == [_dispatch_call(E2E)]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"{E2E} failed once" in comment[2] and BACKEND not in comment[2].split("\n- ")[0]


@pytest.mark.parametrize("overrides", [{BACKEND: None}, {BACKEND: "failure"}], ids=["dispatch", "retry"])
def test_nothing_is_dispatched_or_said_when_every_targeted_context_already_started(overrides):
    gh = FakeGh([_node(1, state="BLOCKED")], late_runs={OLD: [_run_of("backend-required.yml", 88)]},
                **_green(overrides))
    assert _run(gh)[0][1].kind in ("dispatch", "retry")
    assert gh.calls == []


def test_a_license_status_that_turned_pending_since_the_decision_is_not_dispatched_again():
    gh = FakeGh([_node(1, state="BLOCKED")], late_statuses={OLD: [_status("pending", "2026-10-04T11:59:30Z")]},
                **_green(statuses=[_status("failure", "2026-10-04T11:50:00Z")]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (LICENSE,))
    assert gh.calls == []


# --- refused dispatches ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [422, 404])
@pytest.mark.parametrize(("overrides", "decided"), [({E2E: "failure"}, "retry"), ({E2E: None}, "dispatch")])
def test_a_dispatch_refused_by_the_pr_branch_evicts_and_the_line_moves_on(overrides, decided, status):
    """A branch that refuses the dispatch (422: its workflow is broken, lacks the trigger or the
    input; 404: its ref is gone) must not stall the line: evict it with the error, then decide
    for the next PR in the same run."""
    gh = FakeGh([_node(1, state="BLOCKED"), _node(2, armed="2026-10-04T11:00:00Z", state="DIRTY")],
                dispatch_refused={"e2e-required.yml"}, dispatch_status=status, **_green(overrides))
    decisions = _run(gh)
    assert [(n, a.kind) for n, a in decisions] == [(1, decided), (2, "evict")]
    assert ("disarm", "PR_1") in gh.calls and ("disarm", "PR_2") in gh.calls
    comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:evict-dispatch:{OLD} -->" in comment[2]
    assert f"CI dispatch of e2e-required.yml failed: {DISPATCH_ERRORS[status]}" in comment[2]
    assert not any(c[0] == "comment" and "merge-queue:retry:" in c[2] for c in gh.calls)


@pytest.mark.parametrize("status", [422, 404])
def test_a_dispatch_refused_after_a_rebase_evicts_on_the_new_head(status):
    runs = {OLD: [_run_of("ci.yml", 12, event="pull_request")]}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-04T11:00:00Z", state="DIRTY")], runs=runs,
                dispatch_refused={"security-required.yml"}, dispatch_status=status)
    decisions = _run(gh)
    assert [(n, a.kind) for n, a in decisions] == [(1, "rebase"), (2, "evict")]
    comment = next(c for c in gh.calls if c[0] == "comment" and c[1] == 1)
    assert f"<!-- merge-queue:evict-dispatch:{NEW} -->" in comment[2]
    assert f"CI dispatch of security-required.yml failed: {DISPATCH_ERRORS[status]}" in comment[2]
    assert [c[1] for c in gh.dispatched()] == [LICENSE_GATE, "backend-required.yml", "security-required.yml"]
    assert not any(c[0] == "cancel" for c in gh.calls), "an evicted PR's old runs are left alone"
    assert not any(c[0] == "comment" and "merge-queue:rebased:" in c[2] for c in gh.calls)


@pytest.mark.parametrize("status", [422, 404, 502, 403])
@pytest.mark.parametrize("state", ["BLOCKED", "BEHIND"], ids=["dispatch", "after-rebase"])
def test_a_refused_license_dispatch_fails_the_run_and_disarms_nobody(state, status):
    """The license gate is dispatched on dev, so no PR can be blamed for its refusal: a 422 or
    404 there means dev's workflow is broken or not landed yet. Evicting would disarm every
    front in turn. The run fails instead, before any of the six gates is started."""
    nodes = [_node(1, state=state)] + [_node(i, armed=f"2026-10-04T1{i}:00:00Z", state="BLOCKED") for i in (2, 3)]
    gh = FakeGh(nodes, dispatch_refused={LICENSE_GATE}, dispatch_status=status)
    with pytest.raises(mq.GhError, match=rf"\(HTTP {status}\)"):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment", "cancel") for c in gh.calls)
    assert [c[1] for c in gh.dispatched()] == [LICENSE_GATE]


@pytest.mark.parametrize("status", [502, 403])
@pytest.mark.parametrize(("state", "overrides"), [("BLOCKED", {E2E: None}), ("BLOCKED", {E2E: "failure"}),
                                                  ("BEHIND", None)], ids=["dispatch", "retry", "after-rebase"])
def test_a_github_side_dispatch_error_fails_the_run_and_disarms_nobody(state, overrides, status):
    """A 5xx, rate limit or 403 is GitHub's error, not the branch's. Evicting on it, with the
    same-run hand-over, would disarm every front one outage touches. The run must fail instead,
    before any disarm, so the next event retries."""
    nodes = [_node(1, state=state)] + [_node(i, armed=f"2026-10-04T1{i}:00:00Z", state="BLOCKED") for i in range(2, 6)]
    gh = FakeGh(nodes, dispatch_refused={"e2e-required.yml"}, dispatch_status=status, **_green(overrides))
    with pytest.raises(mq.GhError, match=rf"\(HTTP {status}\)"):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)
    assert {c[2]["ref"] for c in gh.dispatched()} <= {"feat/1", "dev"}, "nothing behind the front was touched"
    assert {c[2].get("inputs[pr]", "1") for c in gh.dispatched()} == {"1"}


# --- pagination ------------------------------------------------------------------------------


def test_line_is_read_across_pages_and_the_oldest_armed_pr_leads():
    """With more open PRs than one page, the oldest armed PR may sit on a later page."""
    gh = FakeGh([
        _node(1, armed="2026-10-04T11:00:00Z", state="DIRTY"),
        _node(2, armed="2026-10-04T11:30:00Z", state="DIRTY"),
        _node(3, armed="2026-10-04T09:00:00Z", state="DIRTY"),
    ], line_page_size=2)
    decisions = _run(gh, "dry")
    assert [n for n, _ in decisions] == [3, 1, 2]
    assert gh.line_cursors == [None, "2"]


def test_a_line_longer_than_the_page_cap_fails_instead_of_deciding_on_part_of_it():
    gh = FakeGh([_node(i, state="DIRTY") for i in range(1, mq.MAX_PAGES + 2)], line_page_size=1)
    with pytest.raises(mq.GhError, match="partial line"):
        _run(gh, "dry")


def test_a_failure_on_the_second_page_of_a_contexts_check_runs_is_seen():
    """100 cancelled runs fill page 1; the failure on page 2 must still count."""
    cancelled = [_check(E2E, "cancelled", url=f"https://run/c{i}") for i in range(mq.PER_PAGE)]
    checks = _checks({E2E: None}) + cancelled + [_check(E2E, "failure", url="https://run/f")]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, statuses={OLD: [_status()]})
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts, decision.links) == ("retry", (E2E,), ("https://run/f",))


def test_a_live_required_run_on_the_second_page_of_workflow_runs_is_seen():
    """A live required run behind 100 other runs on the head still means 'in flight'."""
    others = [_run_of("ci.yml", 1000 + i, "completed", "success", event="pull_request") for i in range(mq.PER_PAGE)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: others + [_run_of("e2e-required.yml", 77)]},
                **_green({E2E: None}))
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []


def test_rest_pages_stop_at_total_count_without_a_false_cap_error():
    """Exactly MAX_PAGES full pages is a complete list when total_count says so; one more item is not."""
    full = [_check(BACKEND, "cancelled", url=f"https://run/c{i}") for i in range(mq.MAX_PAGES * mq.PER_PAGE)]
    assert len(mq.read_checks(FakeGh([], checks={OLD: full}), OLD)) == mq.MAX_PAGES * mq.PER_PAGE
    with pytest.raises(mq.GhError, match="partial view"):
        mq.read_checks(FakeGh([], checks={OLD: full + [_check()]}), OLD)


# --- the forbidden operations, and the gh wrapper --------------------------------------------


def test_the_queue_can_never_merge_arm_or_push():
    """Spec section 3: the queue may only DISABLE auto-merge. The eviction comment's
    re-arm hint (`gh pr merge <n> --auto --merge`) is text for a human, not a call."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("enablePullRequestAutoMerge", "mergePullRequest", '/merge"', "/merge'", "git push", '"push"',
                      "'push'", "createCommitOnBranch", "updateRef", "/git/refs"):
        assert forbidden not in source, forbidden
    assert not re.search(r"\[\s*[\"']pr[\"']", source), "the queue never runs `gh pr ...` subcommands"
    assert re.findall(r"mutation\(.*?\{\s*(\w+)", source) == ["updatePullRequestBranch", "disablePullRequestAutoMerge"]
    assert source.count("subprocess.run(") == 1 and '["gh", *args]' in source, "gh is the only program ever run"


def test_gh_only_ever_calls_the_api_subcommand():
    seen = []
    gh = mq.Gh(runner=lambda args: (seen.append(args), "{}")[1])
    gh.graphql("query { viewer { login } }", n=1)
    gh.rest("POST", "repos/x/y/issues/1/comments", {"body": "b"})
    assert seen and all(args[0] == "api" for args in seen)


def test_gh_argument_typing():
    seen = []
    gh = mq.Gh(runner=lambda args: (seen.append(args), "{}")[1])
    gh.graphql("query { x }", number=5, id="X")
    mq.dispatch(gh, "w.yml", "feat/1", {"base_sha": DEV})
    mq.dispatch(gh, "w.yml", "feat/1")
    graphql_args, dispatch_args, bare_args = seen
    assert graphql_args[graphql_args.index("-F") + 1] == "number=5"
    assert "id=X" in graphql_args
    assert dispatch_args[-4:] == ["-f", "ref=feat/1", "-f", f"inputs[base_sha]={DEV}"]
    assert bare_args[-2:] == ["-f", "ref=feat/1"] and not any("inputs" in a for a in bare_args)


def test_a_failed_gh_call_raises_with_the_http_status_the_refusal_rule_reads(monkeypatch):
    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="gh: Not Found (HTTP 404)\n")

    monkeypatch.setattr(mq.subprocess, "run", failing)
    with pytest.raises(mq.GhError, match=r"\(HTTP 404\)"):
        mq.Gh().rest("POST", "repos/x/y/actions/workflows/w.yml/dispatches", {"ref": "gone"})


def test_main_drives_the_real_gh_argv_end_to_end(monkeypatch, tmp_path):
    """main() through the real Gh and _run_gh down to the subprocess argv, with canned JSON for
    a BEHIND front PR: line query, context reads, pinned-head rebase, poll reread, the seven
    dispatches with their inputs, comment."""
    owner, name = mq.REPO.split("/")
    node = _node(7, ref="feat/queue-me")
    seen = []

    def fake_subprocess_run(argv, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        seen.append(argv)
        args = argv[1:]
        if args[:2] == ["api", "graphql"]:
            query = args[3][len("query="):]
            out = {
                mq.LINE_QUERY: {"data": {"repository": {"pullRequests": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [node]}}}},
                mq.REBASE_MUTATION: {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": OLD}}}},
                mq.PR_QUERY: {"data": {"repository": {"pullRequest": dict(node, headRefOid=NEW,
                                                                          mergeStateStatus="BLOCKED")}}},
                mq.COMMENTS_QUERY: {"data": {"repository": {"pullRequest": {"comments": {"nodes": []}}}}},
            }[query]
        elif args[2] == "GET":
            out = [] if "/statuses" in args[3] else {"total_count": 0, "check_runs": [], "workflow_runs": []}
        elif args[3].endswith("/dispatches"):
            out = None  # 204 No Content
        else:
            out = {"id": 1}
        return subprocess.CompletedProcess(argv, 0, stdout="" if out is None else json.dumps(out), stderr="")

    monkeypatch.setattr(mq.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(mq, "REBASE_POLL_S", 0)
    monkeypatch.setenv("MERGE_QUEUE", "on")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert mq.main() == 0

    repo = f"repos/{owner}/{name}"
    page = "per_page=100&page=1"
    get = ["gh", "api", "-X", "GET"]
    who = ["-f", f"owner={owner}", "-f", f"name={name}"]

    def post_dispatch(workflow, *fields):
        return ["gh", "api", "-X", "POST", f"{repo}/actions/workflows/{workflow}/dispatches", *fields]

    assert seen[:-1] == [
        ["gh", "api", "graphql", "-f", f"query={mq.LINE_QUERY}", *who],
        [*get, f"{repo}/actions/runs?head_sha={OLD}&{page}"],
        *[[*get, f"{repo}/commits/{OLD}/check-runs?check_name={check}&filter=all&{page}"] for check in CHECK_NAMES],
        [*get, f"{repo}/commits/{OLD}/statuses?{page}"],
        [*get, f"{repo}/actions/runs?head_sha={OLD}&status=action_required&{page}"],
        ["gh", "api", "graphql", "-f", f"query={mq.REBASE_MUTATION}", "-f", "id=PR_7", "-f", f"oid={OLD}"],
        ["gh", "api", "graphql", "-f", f"query={mq.PR_QUERY}", *who, "-F", "number=7"],
        [*get, f"{repo}/actions/runs?head_sha={OLD}&{page}"],
        post_dispatch("frontend-license-gate.yml", "-f", "ref=dev", "-f", "inputs[pr]=7"),
        post_dispatch("backend-required.yml", "-f", "ref=feat/queue-me", "-f", f"inputs[base_sha]={DEV}"),
        post_dispatch("security-required.yml", "-f", "ref=feat/queue-me", "-f", f"inputs[base_sha]={DEV}"),
        post_dispatch("coverage-required.yml", "-f", "ref=feat/queue-me", "-f", f"inputs[base_sha]={DEV}"),
        post_dispatch("frontend-required.yml", "-f", "ref=feat/queue-me", "-f", f"inputs[base_sha]={DEV}"),
        post_dispatch("e2e-required.yml", "-f", "ref=feat/queue-me", "-f", f"inputs[base_sha]={DEV}"),
        post_dispatch("container-build-check.yml", "-f", "ref=feat/queue-me"),
        [*get, f"{repo}/actions/runs?head_sha={NEW}&status=action_required&{page}"],
        ["gh", "api", "graphql", "-f", f"query={mq.COMMENTS_QUERY}", *who, "-F", "number=7"],
    ]
    assert seen[-1][:6] == ["gh", "api", "-X", "POST", f"{repo}/issues/7/comments", "-f"]
    assert seen[-1][6].startswith(f"body=<!-- merge-queue:rebased:{NEW} -->\nMerge queue: this PR is next.")
    assert "| #7 | rebase | behind dev |" in summary.read_text(encoding="utf-8")
