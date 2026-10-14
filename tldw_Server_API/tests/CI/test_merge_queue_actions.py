"""The merge queue's read layer, action layer and run loop, against a fake gh (spec sections 4, 6, 7)."""

from __future__ import annotations

import ast
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
BACKEND, E2E, CONTAINER = "backend-required", "e2e-required", "container-build-check"
SECURITY = "security-required"
LICENSE = "frontend-license-policy/trusted/dev"
LICENSE_GATE = "frontend-license-gate.yml"
CHECK_NAMES = tuple(c.name for c in mq.CONTEXTS if c.kind == "check")


@pytest.fixture(autouse=True)
def _no_real_run_id(monkeypatch):
    """Isolate GITHUB_RUN_ID and GITHUB_WORKFLOW_REF from the CI run executing these tests; a
    rebase never cancels the run executing the queue."""
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_WORKFLOW_REF", raising=False)


def _node(number, *, head=OLD, armed="2026-10-04T10:00:00Z", state="BEHIND", repo=None, draft=False,
          committed="2026-10-04T09:00:00Z", ref=None, author="User"):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "headRefOid": head,
        "headRefName": ref or f"feat/{number}", "mergeStateStatus": state,
        "author": {"__typename": author} if author else None,
        "headRepository": {"nameWithOwner": repo or mq.REPO},
        "autoMergeRequest": {"enabledAt": armed} if armed else None,
        "commits": {"nodes": [{"commit": {"committedDate": committed}}]},
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


def _run_of(workflow, rid, status="in_progress", conclusion=None, event="pull_request", **extra):
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": status,
            "conclusion": conclusion, "html_url": f"https://run/{rid}", **extra}


GATE_FILES = tuple(c.workflow for c in mq.CONTEXTS if c.kind == "check")


def _held(rid, workflow="backend-required.yml", actor="github-actions[bot]", event="pull_request", repo=None):
    """A run GitHub holds for approval because the queue's token caused its event (spec S2)."""
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": "completed",
            "conclusion": "action_required", "triggering_actor": {"login": actor},
            "head_repository": {"full_name": repo or mq.REPO}, "html_url": f"https://run/{rid}"}


def _six_held(first=31):
    """The six gate workflows' held runs on a queue-rebased head."""
    return [_held(first + i, workflow) for i, workflow in enumerate(GATE_FILES)]


def _pr_state(**kw):
    """A front PrState for calling one action function directly."""
    base = {"number": 1, "node_id": "PR_1", "head_sha": OLD, "head_ref": "feat/1", "same_repo": True,
            "armed_at": NOW, "is_draft": False, "merge_state": "BLOCKED", "head_committed_at": NOW}
    base.update(kw)
    return mq.PrState(**base)




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
    the old head, later ones show NEW.
    `rebase_lag=None`: it never moves. `events` holds the recorded calls interleaved with
    ("read_pr", head) for every single-PR read. A `reread` value may be a list: successive
    single-PR reads walk it, the last entry sticks.

    Lists page like the real API: the PR line `line_page_size` at a time through a cursor, REST
    lists `mq.PER_PAGE` at a time. Check runs are served per `check_name`, as the real endpoint
    filters them; `check_names_read` records the names asked for. Statuses come back as a bare
    list, newest first.

    `late_nodes`, if given, replaces the line from its second read on: what another queue run
    or the author did between this run's decision and its action.

    `late_runs[sha]` joins the head's runs from its second unfiltered runs read on, and
    `late_statuses[sha]` its statuses from the second statuses read on: a run another queue run
    started after this one decided. A dispatch of a `dispatch_refused` workflow fails with
    `DISPATCH_ERRORS[dispatch_status]`.
    """

    def __init__(self, nodes, *, checks=None, statuses=None, runs=None, comments=None, rebase_error=False,
                 reread=None, dispatch_refused=(), rebase_lag=1, disarm_error=False, line_page_size=None,
                 late_runs=None, late_statuses=None, dispatch_status=422,
                 late_nodes=None, rebase_error_text="gh: rebase refused (HTTP 422)", approve_error=None, rerun_error=None,
                 run_status=None, behind_by=None):
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
        self.late_nodes = late_nodes
        self.rebase_error_text = rebase_error_text
        self.approve_error = approve_error
        self.rerun_error = rerun_error
        self.run_status = run_status or {}
        self.run_reads = {}
        self.behind_by = behind_by or {}
        self.line_reads = 0
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

    def _set_run(self, rid, **fields):
        for runs in self.runs.values():
            for run in runs:
                if run.get("id") == rid:
                    run.update(fields)

    def dispatched(self):
        return [c for c in self.calls if c[0] == "dispatch"]

    def graphql(self, query, **v):
        if "updatePullRequestBranch" in query:
            self._record(("rebase", v["id"], v["oid"]))
            if self.rebase_error:
                raise mq.GhError(self.rebase_error_text)
            self.rereads_since_rebase = 0
            return {"data": {"updatePullRequestBranch": {"pullRequest": {"headRefOid": v["oid"]}}}}
        if "disablePullRequestAutoMerge" in query:
            self._record(("disarm", v["id"]))
            if self.disarm_error:
                raise mq.GhError("Pull request is not in the correct state to disable auto-merge")
            return {"data": {}}
        self.reads += 1
        if "comments(last" in query:
            nodes = []
            for index, entry in enumerate(self.comments.get(v["number"], []), start=1):
                body, created = entry if isinstance(entry, tuple) else (entry, None)
                nodes.append({"databaseId": index, "body": body, "createdAt": created})
            return {"data": {"repository": {"pullRequest": {"comments": {"nodes": nodes}}}}}
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
            self.events.append(("read_pr", node["headRefOid"]))
            return {"data": {"repository": {"pullRequest": node}}}
        if "pullRequests(" in query:
            if v.get("after") is None:
                self.line_reads += 1
            nodes = list(self.nodes.values())
            if self.late_nodes is not None and self.line_reads >= 2:
                nodes = list(self.late_nodes)
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
        if method == "POST" and path.endswith("/approve"):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("approve", rid))
            if self.approve_error:
                raise mq.GhError(self.approve_error)
            self._set_run(int(rid), status="queued", conclusion=None)  # GitHub starts it
            return None
        if method == "POST" and path.endswith(("/rerun", "/rerun-failed-jobs")):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("rerun", rid, "failed" if path.endswith("-failed-jobs") else "all"))
            if self.rerun_error:
                raise mq.GhError(self.rerun_error)
            self._set_run(int(rid), status="queued", conclusion=None)
            return None
        if method == "GET" and re.search(r"/actions/runs/\d+$", path):
            rid = int(path.rsplit("/", 1)[1])
            seen = self.run_reads.get(rid, 0)
            self.run_reads[rid] = seen + 1
            if rid in self.run_status:
                statuses = self.run_status[rid]
                entry = statuses[min(seen, len(statuses) - 1)]
                return {"id": rid, **entry} if isinstance(entry, dict) else {"id": rid, "status": entry}
            run = next((r for runs in self.runs.values() for r in runs if r.get("id") == rid), {"status": "completed"})
            return dict(run)
        if method == "GET" and "/compare/" in path:
            self.reads += 1
            behind = self.behind_by.get(path.split("...")[1].split("?")[0], 0)
            if isinstance(behind, Exception):
                raise behind
            return {"behind_by": behind}
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


@pytest.mark.parametrize(("statuses", "kind", "contexts"), [
    ([], "start", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:50:00Z")], "wait", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:50:00Z"), _status("success", "2026-10-04T11:52:00Z")], "wait", ()),
    ([_status("pending", "2026-10-04T11:50:00Z"), _status("failure", "2026-10-04T11:52:00Z")], "retry", (LICENSE,)),
    ([_status("error", "2026-10-04T11:52:00Z")], "retry", (LICENSE,)),
    ([_status("failure", "2026-10-04T11:40:00Z"), _status("pending", "2026-10-04T11:50:00Z")], "wait", (LICENSE,)),
    ([_status("failure", "2026-10-04T11:30:00Z"), _status("pending", "2026-10-04T11:40:00Z"),
      _status("success", "2026-10-04T11:50:00Z")], "wait", ()),
    ([_status("pending", "2026-10-04T11:20:00Z"), _status("failure", "2026-10-04T11:30:00Z"),
      _status("pending", "2026-10-04T11:40:00Z"), _status("error", "2026-10-04T11:50:00Z")], "evict", (LICENSE,)),
    ([_status("pending", "2026-10-04T11:44:00Z")], "start", (LICENSE,)),
    ([_status("success", context="frontend-license-policy/trusted/main"),
      _status("failure", context="qodo")], "start", (LICENSE,)),
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


def test_a_failed_license_status_retries_only_the_license_gate_on_dev():
    """The license gate posts its status without a target_url: the comment then lists no link."""
    gh = FakeGh([_node(5, state="BLOCKED")], **_green(statuses=[_status("failure", url=None)]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.links) == ("retry", ())
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "5"})]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[2] == (f"<!-- merge-queue:retry-{LICENSE}:{OLD} -->\nMerge queue: {LICENSE} failed once on "
                          f"`{OLD[:10]}`; dispatched the license gate on `dev` again.")


# --- rebase ----------------------------------------------------------------------------------


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
    assert "rebasing onto dev failed (gh: rebase refused (HTTP 422)); will retry" in body


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


# --- eviction and comments -------------------------------------------------------------------


def test_evicted_front_hands_over_in_the_same_run():
    gh = FakeGh([_node(1, state="DIRTY"), _node(2, armed="2026-10-04T11:00:00Z")])
    _run(gh)
    assert [c[0] for c in gh.calls[:2]] == ["comment", "disarm"]
    assert ("rebase", "PR_2", OLD) in gh.calls


def test_comments_are_deduplicated_by_marker():
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-conflict:{OLD} -->\nold"]})
    _run(gh)
    assert ("disarm", "PR_1") in gh.calls
    assert not any(c[0] == "comment" for c in gh.calls)


def test_eviction_for_a_new_reason_on_the_same_head_still_comments():
    """A re-armed PR evicted again on the same head, for a different reason, must be told why."""
    gh = FakeGh([_node(1, state="DIRTY")], comments={1: [f"<!-- merge-queue:evict-stuck:{OLD} -->\nold"]})
    _run(gh)
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"<!-- merge-queue:evict-conflict:{OLD} -->" in comment[2]


def test_blocked_green_waits_for_auto_merge_and_never_asks_about_review_threads():
    """dev does not require conversations to be resolved, so BLOCKED with everything green is
    the merge state lagging the checks. The queue must not evict a mergeable PR over a thread,
    and does not even read them."""
    gh = FakeGh([_node(1, state="BLOCKED")], **_green())
    assert _run(gh)[0][1].kind == "wait"
    assert gh.calls == []
    assert "reviewThreads" not in mq.PR_FIELDS


# --- acting on a stale snapshot --------------------------------------------------------------


def test_a_pr_disarmed_since_the_decision_gets_no_checks():
    """Another queue run evicted it, or its author disarmed it, while this run was reading the
    seven contexts. Starting checks now would spend runners on a PR that has left the line."""
    gh = FakeGh([_node(1, state="BLOCKED")], late_nodes=[_node(1, state="BLOCKED", armed=None)])
    decisions = _run(gh)
    assert [d[1].kind for d in decisions] == ["start"]
    assert gh.calls == []


def test_a_pr_pushed_to_since_the_decision_is_left_for_the_next_wake():
    gh = FakeGh([_node(1)], late_nodes=[_node(1, head=NEW)])
    _run(gh)
    assert gh.calls == []


def test_a_pr_that_is_no_longer_first_in_line_is_not_acted_on():
    earlier = _node(2, armed="2026-10-04T08:00:00Z", state="BLOCKED")
    gh = FakeGh([_node(1)], late_nodes=[_node(1), earlier])
    _run(gh)
    assert gh.calls == []


def test_a_wait_decision_does_not_reread_the_line():
    gh = FakeGh([_node(1, state="BLOCKED")], **_green())
    _run(gh)
    assert gh.line_reads == 1


# --- a rebase the token is not allowed to make -----------------------------------------------


WORKFLOW_REFUSAL = (
    "gh: refusing to allow a GitHub App to create or update workflow `.github/workflows/ci.yml` "
    "without `workflows` permission"
)


def test_a_rebase_across_workflow_changes_evicts_at_once_with_the_hand_rebase_steps():
    """GITHUB_TOKEN can never update a branch with dev's workflow-file changes. Retrying cannot
    work and waiting would block everyone behind, so the PR leaves the line straight away, told
    exactly what to do, and the next PR is taken in the same run."""
    gh = FakeGh([_node(1), _node(2, armed="2026-10-04T11:00:00Z")], rebase_error=True,
                rebase_error_text=WORKFLOW_REFUSAL)
    _run(gh)
    assert gh.calls[0] == ("rebase", "PR_1", OLD)
    assert gh.calls[1][0] == "comment" and gh.calls[2] == ("disarm", "PR_1")
    comments = [c for c in gh.calls if c[0] == "comment" and c[1] == 1]
    assert len(comments) == 1
    body = comments[0][2]
    assert f"<!-- merge-queue:evict-workflows:{OLD} -->" in body
    assert "workflow files" in body and "git rebase origin/dev" in body and "force-push" in body
    assert "will retry once" not in body
    assert ("rebase", "PR_2", OLD) in gh.calls


def test_any_other_rebase_refusal_still_gets_one_retry():
    gh = FakeGh([_node(1)], rebase_error=True)
    _run(gh)
    assert not any(c[0] == "disarm" for c in gh.calls)
    assert any(c[0] == "comment" and "will retry" in c[2] for c in gh.calls)


# --- hand-started frontend diagnostics -------------------------------------------------------


def test_two_failed_hand_diagnostics_do_not_evict_a_green_pr():
    """Both diagnostics finished after the real check went green, so if they counted they would
    be the two latest runs of the context: failed twice."""
    runs = {OLD: [
        _run_of("frontend-required.yml", rid, "completed", "failure", event="workflow_dispatch", actor={"login": "octocat"},
                updated_at=f"2026-10-04T11:59:{rid}Z") for rid in (10, 20)
    ]}
    gh = FakeGh([_node(1, state="BLOCKED")], runs=runs, **_green())
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
    assert (decision.kind, decision.contexts) == ("start", (E2E,))


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
    which would be started forever."""
    green = _green({"security-required": None})
    once = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_broken_run(31, "startup_failure", "2026-10-04T11:00:00Z")]},
                  **green)
    decision = _run(once)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", ("security-required",))
    assert ("rerun", "31", "all") in once.calls and not once.dispatched()
    twice = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [
        _broken_run(31, "startup_failure", "2026-10-04T11:00:00Z"),
        _broken_run(32, "timed_out", "2026-10-04T11:30:00Z"),
    ]}, **green)
    decision = _run(twice)[0][1]
    assert decision.kind == "evict" and decision.links == ("https://run/31", "https://run/32")
    assert ("disarm", "PR_1") in twice.calls


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


@pytest.mark.parametrize("overrides", [{BACKEND: None}, {BACKEND: "failure"}], ids=["start", "retry"])
def test_nothing_is_dispatched_or_said_when_every_targeted_context_already_started(overrides):
    gh = FakeGh([_node(1, state="BLOCKED")], late_runs={OLD: [_run_of("backend-required.yml", 88)]},
                **_green(overrides))
    assert _run(gh)[0][1].kind in ("start", "retry")
    assert gh.calls == []


def test_a_license_status_that_turned_pending_since_the_decision_is_not_dispatched_again():
    gh = FakeGh([_node(1, state="BLOCKED")], late_statuses={OLD: [_status("pending", "2026-10-04T11:59:30Z")]},
                **_green(statuses=[_status("failure", "2026-10-04T11:50:00Z")]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (LICENSE,))
    assert gh.calls == []


# --- refused dispatches ----------------------------------------------------------------------


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
    assert [c[1] for c in gh.dispatched()] == ([LICENSE_GATE, "security-required.yml"] if state == "BEHIND" else [LICENSE_GATE])


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
    # gh is the only program the script can run. Checked on the syntax tree, so reformatting or
    # renaming does not trip it, while a second process call or a different program does.
    tree = ast.parse(source)
    process_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id in ("subprocess", "os")
        and node.func.attr in ("run", "Popen", "call", "check_call", "check_output", "system", "popen",
                               "execv", "execvp", "spawnv")
    ]
    assert len(process_calls) == 1, "exactly one process is ever started"
    argv = process_calls[0].args[0]
    assert isinstance(argv, ast.List) and isinstance(argv.elts[0], ast.Constant) and argv.elts[0].value == "gh"
    assert not any(keyword.arg == "shell" for keyword in process_calls[0].keywords), "never through a shell"
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not imported & {"pty", "shlex", "ctypes", "multiprocessing"}, imported


def test_the_guard_reads_real_commands_not_only_the_source():
    """The same prohibition seen from outside: whatever the queue does in a full pass, every
    command it hands to the runner is `gh api ...`, and none of them merges, arms or pushes."""
    seen: list[list[str]] = []

    def runner(args: list[str]) -> str:
        seen.append(args)
        joined = " ".join(args)
        if "pullRequests(" in joined:
            return json.dumps({"data": {"repository": {"pullRequests": {
                "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [_node(1, state="DIRTY")],
            }}}})
        if "comments(last" in joined:
            return json.dumps({"data": {"repository": {"pullRequest": {"comments": {"nodes": []}}}}})
        if "pullRequest(number" in joined:
            return json.dumps({"data": {"repository": {"pullRequest": _node(1, state="DIRTY")}}})
        if "/check-runs" in joined:
            return json.dumps({"total_count": 0, "check_runs": []})
        if "/statuses" in joined:
            return "[]"
        if "/actions/runs?" in joined:
            return json.dumps({"total_count": 0, "workflow_runs": []})
        return "{}"

    mq.run(mq.Gh(runner=runner), "on", now=lambda: NOW, sleep=lambda s: None, log=lambda m: None)
    assert seen and all(args[0] == "api" for args in seen)
    text = " ".join(" ".join(args) for args in seen)
    assert "disablePullRequestAutoMerge" in text, "the eviction really went through this runner"
    for forbidden in ("enablePullRequestAutoMerge", "mergePullRequest", "/merge", "git/refs"):
        assert forbidden not in text, forbidden


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
    mq.wake(gh, 42)
    mq.dispatch_license(gh, _pr_state(number=7), lambda m: None)
    gh.rest("POST", f"repos/{mq.REPO}/actions/runs/31/approve")
    gh.rest("POST", f"repos/{mq.REPO}/actions/runs/31/rerun-failed-jobs")
    graphql_args, kick_args, license_args, approve_args, rerun_args = seen
    assert graphql_args[graphql_args.index("-F") + 1] == "number=5" and "id=X" in graphql_args
    assert kick_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/workflows/security-required.yml/dispatches",
                         "-f", "ref=dev", "-f", "inputs[queue_kick]=true", "-f", "inputs[wait_run]=42"]
    assert license_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/workflows/frontend-license-gate.yml/dispatches",
                            "-f", "ref=dev", "-f", "inputs[pr]=7"]
    assert approve_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/runs/31/approve"]
    assert rerun_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/runs/31/rerun-failed-jobs"]

def test_a_failed_gh_call_raises_with_the_http_status_the_refusal_rule_reads(monkeypatch):
    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="gh: Not Found (HTTP 404)\n")

    monkeypatch.setattr(mq.subprocess, "run", failing)
    with pytest.raises(mq.GhError, match=r"\(HTTP 404\)"):
        mq.Gh().rest("POST", "repos/x/y/actions/workflows/w.yml/dispatches", {"ref": "gone"})


def test_main_drives_the_real_gh_argv_end_to_end(monkeypatch, tmp_path):
    """main() through the real Gh and _run_gh down to the subprocess argv, with canned JSON for
    a BEHIND front PR: line query, context reads, pinned-head rebase, poll reread, the license
    dispatch, the six approvals, the non-gate cleanup, comment."""
    owner, name = mq.REPO.split("/")
    node = _node(7, ref="feat/queue-me")
    seen = []
    held = [_held(31 + i, workflow) for i, workflow in enumerate(GATE_FILES)] + [_held(40, "ci.yml")]

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
        elif args[2] == "GET" and f"head_sha={NEW}&status=action_required" in args[3]:
            out = {"total_count": len(held), "workflow_runs": list(held)}
        elif args[2] == "POST" and args[3].endswith("/approve"):
            rid = int(args[3].split("/runs/")[1].split("/")[0])
            held[:] = [r for r in held if r["id"] != rid]  # GitHub starts it; it is no longer held
            out = None
        elif args[2] == "GET":
            out = [] if "/statuses" in args[3] else {"total_count": 0, "check_runs": [], "workflow_runs": []}
        elif args[3].endswith("/dispatches"):
            out = None  # 204 No Content
        else:
            out = {"id": 1}
        return subprocess.CompletedProcess(argv, 0, stdout="" if out is None else json.dumps(out), stderr="")

    monkeypatch.setattr(mq.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(mq, "REBASE_POLL_S", 0)
    monkeypatch.setattr(mq, "HELD_RUN_POLL_S", 0)
    monkeypatch.setenv("MERGE_QUEUE", "on")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert mq.main() == 0

    repo = f"repos/{owner}/{name}"
    mutating = [a for a in seen if a[3:4] in (["POST"], ["DELETE"]) or a[4:5] == [f"query={mq.REBASE_MUTATION}"]]
    assert mutating[0] == ["gh", "api", "graphql", "-f", f"query={mq.REBASE_MUTATION}", "-f", "id=PR_7", "-f", f"oid={OLD}"]
    assert mutating[1] == ["gh", "api", "-X", "POST", f"{repo}/actions/workflows/frontend-license-gate.yml/dispatches",
                           "-f", "ref=dev", "-f", "inputs[pr]=7"]
    assert [a[4] for a in mutating[2:8]] == [f"{repo}/actions/runs/{31 + i}/approve" for i in range(6)]
    assert mutating[8] == ["gh", "api", "-X", "DELETE", f"{repo}/actions/runs/40"]
    assert mutating[9][:6] == ["gh", "api", "-X", "POST", f"{repo}/issues/7/comments", "-f"]
    assert mutating[9][6].startswith(f"body=<!-- merge-queue:rebased:{NEW} -->\nMerge queue: this PR is next.")
    assert len(mutating) == 10
    assert "| #7 | rebase | behind dev |" in summary.read_text(encoding="utf-8")


# ======== revised plan: new tests ========

@pytest.mark.parametrize("actor", ["github-actions[bot]", "someone"])
@pytest.mark.parametrize(("status", "conclusion"), [("in_progress", None), ("completed", "success"),
                                                     ("completed", "failure"), ("completed", "startup_failure")])
def test_a_dispatched_gate_run_is_ignored_whoever_started_it(actor, status, conclusion):
    """Spec section 7, stand-ins: a dispatched gate run never counts (S5), so it can neither hold the
    line as in flight nor fail a context."""
    run = _run_of("e2e-required.yml", 77, status, conclusion, event="workflow_dispatch",
                  actor={"login": actor}, triggering_actor={"login": actor}, check_suite_id=770)
    assert mq.required_run_stand_ins([run], ()) == ()


def test_a_dispatched_runs_check_does_not_count():
    """A dispatched run reports its check on the head commit, but the check is not in the PR's rollup (S5):
    a green one must not read as passed, a red one must not read as failed. A pull_request run's check counts."""
    dispatched = _run_of("e2e-required.yml", 77, "completed", "success", event="workflow_dispatch", check_suite_id=770)
    pr_run = _run_of("backend-required.yml", 78, "completed", "success", check_suite_id=780)
    from_dispatch = mq.CheckRun(E2E, "completed", "success", NOW, "u", 770)
    from_pr = mq.CheckRun(BACKEND, "completed", "failure", NOW, "u", 780)
    unlisted = mq.CheckRun(SECURITY, "completed", "success", NOW, "u", 999)
    assert mq.counted((from_dispatch, from_pr, unlisted), [dispatched, pr_run]) == (from_pr, unlisted)


@pytest.mark.parametrize(("conclusion", "stands_in"), [
    ("success", True), ("neutral", True), ("skipped", True), ("failure", True), ("startup_failure", True),
    ("timed_out", True), ("cancelled", False), ("action_required", False),
])
def test_a_completed_gate_run_without_its_check_is_one_failure_unless_cancelled_or_held(conclusion, stands_in):
    """A run that finished without reporting its check (a renamed gate job, a startup
    failure) must count as a failure, so it is retried once and then evicted, not re-run on every wake.
    A cancelled run is restarted instead, and a held run is approved."""
    run = _run_of("e2e-required.yml", 70, "completed", conclusion, check_suite_id=700, updated_at="2026-10-04T11:00:00Z")
    stand_ins = mq.required_run_stand_ins([run], ())
    assert [(c.context, c.conclusion) for c in stand_ins] == ([(E2E, "failure")] if stands_in else [])
    reported = (mq.CheckRun(E2E, "completed", "success", NOW, "u", 700),)
    assert mq.required_run_stand_ins([run], reported) == (), "a run whose suite reported the check speaks through it"


@pytest.mark.parametrize(("run", "approvable"), [
    (_held(1, "backend-required.yml"), True),
    (_held(1, "container-build-check.yml"), True),
    (_held(1, "ci.yml"), False),                                   # D1: never a non-gate workflow
    (_held(1, "pre-commit.yml"), False),
    (_held(1, actor="some-agent[bot]"), False),                    # held for a person to approve
    (_held(1, repo="someone/fork"), False),                        # GitHub's outside-contributor gate
    (_held(1, event="workflow_dispatch"), False),
    (dict(_held(1), conclusion="success"), False),                 # not held at all
])
def test_only_the_queues_own_held_gate_runs_are_approvable(run, approvable):
    assert mq._queue_held(run) is approvable


def test_approve_held_runs_approves_only_the_queues_gate_runs():
    runs = [_held(31, "backend-required.yml"), _held(32, "ci.yml"), _held(33, "e2e-required.yml"),
            _held(34, actor="person")]
    gh = FakeGh([_node(1)], runs={OLD: runs})
    assert mq.approve_held_runs(gh, OLD, lambda m: None) == {"backend-required.yml", "e2e-required.yml"}
    assert [c for c in gh.calls if c[0] == "approve"] == [("approve", "31"), ("approve", "33")]


@pytest.mark.parametrize("error", [
    "gh api -X POST failed: gh: Server Error (HTTP 502)",
    "gh api graphql failed: gh: Something went wrong while executing your query. (HTTP 200)",
    'gh api -X POST failed: Post "https://api.github.com/repos/x/y/actions/runs/1/approve": EOF',
    "gh api -X POST failed: gh: API rate limit exceeded (HTTP 403)",
])
def test_a_transient_approve_error_raises_and_disarms_nobody(error):
    gh = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=error)
    with pytest.raises(mq.GhError):
        mq._approve_strictly(gh, _pr_state(), _held(31), lambda m: None, lambda: NOW)
    assert [c[0] for c in gh.calls] == ["approve"]


def test_a_refused_approval_of_a_run_no_longer_held_stands_down():
    gh = FakeGh([_node(1)], approve_error="gh: Not Found (HTTP 404)", run_status={31: [{"status": "queued"}]})
    assert mq._approve_strictly(gh, _pr_state(), _held(31), lambda m: None, lambda: NOW) is False
    assert [c[0] for c in gh.calls] == ["approve"]


def test_a_refused_approval_strikes_then_evicts_after_the_gap():
    refused = "gh: Resource not accessible by integration (HTTP 403)"
    first = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=refused)
    assert mq._approve_strictly(first, _pr_state(), _held(31), lambda m: None, lambda: NOW) is False
    assert first.calls[-1][0] == "comment" and f"merge-queue:approve-error:{OLD}" in first.calls[-1][2]
    warned = (f"<!-- merge-queue:approve-error:{OLD} -->\nx", "2026-10-04T11:49:00Z")  # 11 minutes old
    later = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=refused, comments={1: [warned]})
    assert mq._approve_strictly(later, _pr_state(), _held(31), lambda m: None, lambda: NOW) is True
    assert [c[0] for c in later.calls] == ["approve", "comment", "disarm"]
    assert f"merge-queue:evict-approve:{OLD}" in later.calls[1][2]


def test_a_strike_inside_the_gap_neither_warns_again_nor_evicts():
    warned = (f"<!-- merge-queue:no-run:{OLD} -->\nx", "2026-10-04T11:55:00Z")  # 5 minutes old
    gh = FakeGh([_node(1)], comments={1: [warned]})
    evicted = mq._strike(gh, _pr_state(), "no-run", "w", mq.Action("evict", "e", slug="no-run"), False,
                         lambda m: None, lambda: NOW)
    assert evicted is False and gh.calls == []


def test_evict_comments_first_then_disarms():
    gh = FakeGh([_node(1)])
    assert mq._evict(gh, _pr_state(), mq.Action("evict", "why", slug="conflict"), lambda m: None) is True
    assert [c[0] for c in gh.calls] == ["comment", "disarm"]


def test_a_failed_disarm_of_a_pr_still_armed_raises_but_an_already_disarmed_pr_is_fine():
    still_armed = FakeGh([_node(1)], disarm_error=True)
    with pytest.raises(mq.GhError):
        mq._evict(still_armed, _pr_state(), mq.Action("evict", "why", slug="x"), lambda m: None)
    already = FakeGh([_node(1)], disarm_error=True, reread={1: _node(1, armed=None)})
    assert mq._evict(already, _pr_state(), mq.Action("evict", "why", slug="x"), lambda m: None) is True


def test_wake_dispatches_the_kick_on_dev_and_nothing_else():
    gh = FakeGh([_node(1)])
    mq.wake(gh)
    mq.wake(gh, 99)
    assert gh.dispatched() == [
        ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true"}),
        ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true", "inputs[wait_run]": "99"}),
    ]


def test_wait_for_run_returns_once_the_run_completes_and_gives_up_after_the_bound():
    done = FakeGh([_node(1)], run_status={5: ["in_progress", "in_progress", "completed"]})
    mq.wait_for_run(done, 5, lambda s: None, lambda m: None)
    assert done.run_reads[5] == 3
    lines = []
    stuck = FakeGh([_node(1)], run_status={5: ["in_progress"]})
    mq.wait_for_run(stuck, 5, lambda s: None, lines.append)
    assert stuck.run_reads[5] == mq.WAIT_RUN_TRIES and "deciding anyway" in lines[-1]


@pytest.mark.parametrize(("error", "outcome"), [
    ("gh: The workflow run cannot be rerun (HTTP 422)", "refused"),
    ("gh: Conflict (HTTP 409)", "refused"),
    ("gh: Unable to re-run this workflow run because it was created over a month ago (HTTP 403)", "refused"),
])
def test_rerun_refusals(error, outcome):
    run = _run_of("e2e-required.yml", 70, "completed", "failure")
    gh = FakeGh([_node(1)], runs={OLD: [run]}, rerun_error=error)
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == outcome


def test_a_rerun_answered_with_an_error_while_a_racing_run_re_ran_it_stands_down():
    run = _run_of("e2e-required.yml", 70, "completed", "failure")
    gh = FakeGh([_node(1)], runs={OLD: [run]}, rerun_error="gh: Conflict (HTTP 409)",
                run_status={70: ["in_progress"]})
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == "started"


def test_a_held_rerun_is_approved():
    run = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    gh = FakeGh([_node(1)], runs={OLD: [run, _held(71, "e2e-required.yml")]},
                run_status={70: ["completed", "in_progress"]})
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == "started"
    assert ("rerun", "70", "failed") in gh.calls and ("approve", "71") in gh.calls


def test_comment_once_removes_its_own_copy_when_a_racing_run_posted_first():
    posted = []

    class Racing(FakeGh):
        def rest(self, method, path, fields=None):
            if method == "POST" and path.endswith("/comments"):
                posted.append(fields["body"])
                return {"id": 50}
            if method == "DELETE" and "/issues/comments/" in path:
                self._record(("delete-comment", path.rsplit("/", 1)[1]))
                return None
            return super().rest(method, path, fields)

        def graphql(self, query, **v):
            if "comments(last" in query and posted:  # the second read sees the racer's copy, id 7
                nodes = [{"databaseId": 7, "body": posted[0], "createdAt": None},
                         {"databaseId": 50, "body": posted[0], "createdAt": None}]
                return {"data": {"repository": {"pullRequest": {"comments": {"nodes": nodes}}}}}
            return super().graphql(query, **v)

    gh = Racing([_node(1)])
    assert mq.comment_once(gh, 1, "rebased", NEW, "body", lambda m: None) is False
    assert ("delete-comment", "50") in gh.calls


def test_rebase_dispatches_the_license_then_approves_exactly_the_six_gates_then_cleans_up_then_cancels():
    held, ci = _six_held(), _held(40, "ci.yml")
    old_live, queue_run = _run_of("backend-required.yml", 11), _run_of("merge-queue.yml", 12)
    gh = FakeGh([_node(1)], runs={OLD: [old_live, queue_run], NEW: held + [ci]})
    assert _run(gh)[0][1].kind == "rebase"
    calls = gh.calls
    assert calls[0] == ("rebase", "PR_1", OLD)
    license_at = calls.index(("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"}))
    approvals = [i for i, c in enumerate(calls) if c[0] == "approve"]
    assert sorted(calls[i][1] for i in approvals) == sorted(str(r["id"]) for r in held)
    assert license_at < min(approvals), "spec 4.4: the license is dispatched before any gate is approved"
    assert ("approve", "40") not in calls and ("delete", "40") in calls, "D1: ci.yml's held run is deleted"
    assert not any(c[0] == "delete" and c[1] in {str(r["id"]) for r in held} for c in calls)
    assert calls.index(("cancel", "11")) > max(approvals)
    assert ("cancel", "12") not in calls, "a merge-queue.yml run on the old head is never cancelled"
    assert calls[-1][0] == "comment" and f"merge-queue:rebased:{NEW}" in calls[-1][2]
    assert len(gh.dispatched()) == 1, "nothing else is dispatched: all six gates were approved"


def test_a_rebase_whose_gates_are_not_all_held_in_time_wakes_a_kick():
    gh = FakeGh([_node(1)], runs={NEW: _six_held()[:4]})
    _run(gh)
    assert ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true"}) in gh.calls
    assert "approved 4 of its 6 required gates" in gh.calls[-1][2]


def test_rebase_approves_only_after_the_new_head_appears():
    gh = FakeGh([_node(1)], runs={NEW: _six_held()}, rebase_lag=2)
    _run(gh)
    first_approve = next(i for i, e in enumerate(gh.events) if e[0] == "approve")
    assert ("read_pr", NEW) in gh.events[:first_approve]


def test_a_rebase_whose_head_never_moves_strikes_and_wakes():
    gh = FakeGh([_node(1)], rebase_lag=None)
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase", "comment", "dispatch"]
    assert f"merge-queue:rebase-unmoved:{OLD}" in gh.calls[1][2]
    assert gh.calls[2][1] == "security-required.yml"
    warned = (f"<!-- merge-queue:rebase-unmoved:{OLD} -->\nx", "2026-10-04T11:40:00Z")
    later = FakeGh([_node(1)], rebase_lag=None, comments={1: [warned]})
    _run(later)
    assert [c[0] for c in later.calls][-2:] == ["comment", "disarm"]


def test_a_transient_rebase_error_with_the_head_moved_stands_down_and_with_it_unmoved_fails_the_run():
    error = "gh api graphql failed: gh: Something went wrong while executing your query. (HTTP 200)"
    moved = FakeGh([_node(1)], rebase_error=True, rebase_error_text=error, reread={1: _node(1, head=NEW)})
    _run(moved)
    assert not any(c[0] in ("disarm", "comment") for c in moved.calls)
    unmoved = FakeGh([_node(1)], rebase_error=True, rebase_error_text=error)
    with pytest.raises(mq.GhError):
        _run(unmoved)
    assert not any(c[0] in ("disarm", "comment") for c in unmoved.calls)


def test_repeated_rebase_failure_evicts_only_after_the_strike_gap():
    refused = "gh: Unprocessable (HTTP 422)"
    first = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused)
    _run(first)
    assert [c[0] for c in first.calls] == ["rebase", "comment"]
    soon = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused,
                  comments={1: [(f"<!-- merge-queue:rebase-failed:{OLD} -->\nx", "2026-10-04T11:58:00Z")]})
    _run(soon)
    assert [c[0] for c in soon.calls] == ["rebase"]
    late = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused,
                  comments={1: [(f"<!-- merge-queue:rebase-failed:{OLD} -->\nx", "2026-10-04T11:45:00Z")]})
    _run(late)
    assert [c[0] for c in late.calls] == ["rebase", "comment", "disarm"]


def test_cleanup_deletes_only_the_queues_non_gate_held_runs():
    runs = [_held(31, "backend-required.yml"), _held(40, "ci.yml"), _held(41, "pre-commit.yml", actor="person")]
    gh = FakeGh([_node(1)], runs={OLD: runs})
    mq.cleanup_approval_runs(gh, _pr_state(), lambda m: None)
    assert gh.calls == [("delete", "40")]


def test_a_failure_after_the_rebase_took_effect_wakes_a_kick_and_fails_the_run():
    """The new head's gate runs are held, and nothing else would wake the queue
    to approve them if this run dies after the branch moved."""
    gh = FakeGh([_node(1)], dispatch_refused={LICENSE_GATE}, dispatch_status=502)
    with pytest.raises(mq.GhError):
        _run(gh)
    assert [c[1] for c in gh.dispatched()] == [LICENSE_GATE, "security-required.yml"]
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)


def test_decide_approves_held_gate_runs_first_so_they_read_as_running():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(35, "e2e-required.yml")]}, **_green({E2E: None}))
    decision = _run(gh)[0][1]
    assert ("approve", "35") in gh.calls
    assert (decision.kind, decision.contexts) == ("wait", (E2E,))


def test_decide_never_approves_a_non_gate_held_run():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(40, "ci.yml")]}, **_green())
    _run(gh)
    assert ("approve", "40") not in gh.calls


def test_start_restarts_a_cancelled_run_of_a_missing_gate_once_then_evicts():
    """A cancelled run says nothing about the PR, so it is restarted in full, once per
    gate per head; a second cancellation on the same head evicts instead of restarting forever."""
    def cancelled():
        return _run_of("e2e-required.yml", 70, "completed", "cancelled", created_at="2026-10-04T11:00:00Z")
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled()]}, run_status={70: ["in_progress"]},
                **_green({E2E: None}))
    assert _run(gh)[0][1].kind == "start"
    assert ("rerun", "70", "all") in gh.calls
    assert gh.calls[-1][0] == "comment" and f"merge-queue:restart-{E2E}:{OLD}" in gh.calls[-1][2]
    assert not gh.dispatched(), "a gate is never dispatched"
    again = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled()]},
                   comments={1: [f"<!-- merge-queue:restart-{E2E}:{OLD} -->\nx"]}, **_green({E2E: None}))
    _run(again)
    assert [c[0] for c in again.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in again.calls[0][2]


def test_start_restarts_a_run_whose_gate_job_was_cancelled_whatever_the_run_concluded():
    """A run can conclude `failure` (another job failed) while its gate job was cancelled; the
    context is missing, and the run is restarted once like a cancelled run."""
    run = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "cancelled", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [run]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].kind == "start"
    assert ("rerun", "70", "all") in gh.calls


def test_start_approves_a_held_gate_run_the_decide_step_could_not_approve():
    pr = _pr_state()
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(35, "e2e-required.yml")]})
    assert mq._start(gh, pr, (E2E,), lambda m: None, lambda s: None, lambda: NOW) is False
    assert gh.calls == [("approve", "35")]


def test_start_dispatches_a_missing_license_on_dev_and_nothing_for_the_gates_that_have_runs():
    gh = FakeGh([_node(1, state="BLOCKED")], **_green(statuses=[]))
    assert _run(gh)[0][1].contexts == (LICENSE,)
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]


def test_start_with_no_run_at_all_warns_then_evicts_after_the_gap():
    first = FakeGh([_node(1, state="BLOCKED")], **_green({E2E: None}))
    _run(first)
    assert [c[0] for c in first.calls] == ["comment"] and f"merge-queue:no-run:{OLD}" in first.calls[0][2]
    warned = (f"<!-- merge-queue:no-run:{OLD} -->\nx", "2026-10-04T11:45:00Z")
    later = FakeGh([_node(1, state="BLOCKED")], comments={1: [warned]}, **_green({E2E: None}))
    _run(later)
    assert [c[0] for c in later.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-no-run:{OLD}" in later.calls[0][2]


def test_first_failure_re_runs_the_failed_jobs_of_that_gates_run_and_comments_once():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].kind == "retry"
    assert ("rerun", "70", "failed") in gh.calls
    assert gh.calls[-1][0] == "comment" and f"merge-queue:retry-{E2E}:{OLD}" in gh.calls[-1][2]
    assert not gh.dispatched()


def test_a_broken_runs_second_retry_on_the_same_head_evicts():
    broken = _run_of("security-required.yml", 70, "completed", "startup_failure", check_suite_id=700)
    marker = f"<!-- merge-queue:retry-{SECURITY}:{OLD} -->\nx"
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [broken]}, comments={1: [marker]},
                **_green({SECURITY: None}))
    _run(gh)
    assert [c[0] for c in gh.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-failed-twice:{OLD}" in gh.calls[0][2]


def test_a_green_gate_run_without_its_check_is_retried_once_then_evicted():
    """A PR whose own workflow copy renamed its gate job finishes green with no check.
    That is one failure: re-run once in full, then evicted, never re-run on every wake."""
    def silent():
        return _run_of("e2e-required.yml", 70, "completed", "success", check_suite_id=700)
    first = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [silent()]}, run_status={70: ["in_progress"]},
                   **_green({E2E: None}))
    assert _run(first)[0][1].kind == "retry"
    assert ("rerun", "70", "all") in first.calls
    marker = f"<!-- merge-queue:retry-{E2E}:{OLD} -->\nx"
    second = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [silent()]}, comments={1: [marker]},
                    **_green({E2E: None}))
    _run(second)
    assert [c[0] for c in second.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-failed-twice:{OLD}" in second.calls[0][2]


def test_a_retry_attempt_that_ends_cancelled_is_restarted_once_then_evicted():
    """A retry that keeps ending cancelled (a gate hitting its timeout) must not be
    re-run on every wake."""
    def state(comments):
        run = _run_of("e2e-required.yml", 70, "completed", "cancelled", check_suite_id=700)
        checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
        return FakeGh([_node(1, state="BLOCKED")], runs={OLD: [run]}, checks={OLD: checks},
                      statuses={OLD: [_status()]}, comments={1: comments}, run_status={70: ["in_progress"]})
    retried = f"<!-- merge-queue:retry-{E2E}:{OLD} -->\nx"
    first = state([retried])
    _run(first)
    assert ("rerun", "70", "all") in first.calls
    assert f"merge-queue:restart-{E2E}:{OLD}" in first.calls[-1][2]
    second = state([retried, f"<!-- merge-queue:restart-{E2E}:{OLD} -->\ny"])
    second.run_status = {}
    _run(second)
    assert [c[0] for c in second.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in second.calls[0][2]


def test_a_failed_license_is_dispatched_again_before_any_gate_is_re_run():
    """With LICENSE_FIRST_CI_ENABLED a gate waits for the license status, so the
    license goes first on a retry as it does on a start."""
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status("failure")]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].contexts == (E2E, LICENSE)
    assert [c[0] for c in gh.calls if c[0] in ("dispatch", "rerun")] == ["dispatch", "rerun"]


def test_a_failed_gate_whose_run_is_still_live_is_left_for_its_kick():
    """The failed run's own queue-tick is still running when a racing queue run decides."""
    live = _run_of("e2e-required.yml", 70, "in_progress", None, check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [live]}, checks={OLD: checks}, statuses={OLD: [_status()]})
    _run(gh)
    assert gh.calls == []


def test_a_refused_rerun_evicts():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, rerun_error="gh: The workflow run cannot be rerun (HTTP 422)")
    _run(gh)
    assert [c[0] for c in gh.calls][-2:] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in gh.calls[-2][2]


def test_a_transient_rerun_error_fails_the_run_and_disarms_nobody():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, rerun_error="gh: Server Error (HTTP 502)")
    with pytest.raises(mq.GhError):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)


HOURS_OLD = "2026-10-04T04:00:00Z"


def _old_green():
    return {"checks": {OLD: [_check(n, "success", completed=HOURS_OLD) for n in CHECK_NAMES]},
            "statuses": {OLD: [_status(created=HOURS_OLD)]}}


def test_an_old_green_head_that_dev_has_moved_past_is_rebased_not_evicted_as_stuck():
    """Spec section 6, the stuck row (TASK-34948 in tldw_chatbook): right after a merge the next PR's
    merge state can still read up to date while its green runs are hours old."""
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: 2}, **_old_green())
    decision = _run(gh)[0][1]
    assert decision.kind == "rebase" and gh.calls[0] == ("rebase", "PR_1", OLD)
    assert not any(c[0] == "disarm" for c in gh.calls)


def test_an_old_green_head_up_to_date_with_dev_is_still_evicted_as_stuck():
    gh = FakeGh([_node(1, state="CLEAN")], **_old_green())
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.slug) == ("evict", "stuck")


def test_a_fresh_green_head_that_dev_has_moved_past_is_rebased_at_once():
    """The stale merge state also hides a head whose green runs are minutes old;
    auto-merge cannot fire on a head that is behind, so waiting would hold the line."""
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: 3}, **_green())
    decision = _run(gh)[0][1]
    assert decision.kind == "rebase" and gh.calls[0] == ("rebase", "PR_1", OLD)


def test_a_failed_behind_check_never_disarms():
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: mq.GhError("gh: Server Error (HTTP 502)")}, **_old_green())
    with pytest.raises(mq.GhError):
        _run(gh)
    assert gh.calls == []


def test_a_kick_waits_for_its_run_before_reading_the_line():
    gh = FakeGh([_node(1, state="DIRTY")], run_status={99: ["in_progress", "completed"]})
    mq.run(gh, "on", now=lambda: NOW, sleep=lambda s: None, log=lambda m: None, wait_run=99)
    assert gh.run_reads[99] == 2 and gh.line_reads >= 1


def test_a_young_head_is_waited_out_once_then_started():
    young = _node(1, state="BLOCKED", committed="2026-10-04T11:59:00Z")
    gh = FakeGh([young], **_green({E2E: None}))
    slept = []
    decisions = mq.run(gh, "on", now=lambda: NOW, sleep=slept.append, log=lambda m: None)
    assert decisions[0][1].kind == "start" and slept and max(slept) <= mq.YOUNG_HEAD.total_seconds() + 1


def test_main_passes_only_a_numeric_wait_run(monkeypatch):
    seen = {}

    def fake_run(gh, mode, wait_run=None):
        seen["wait"] = wait_run
        return []

    monkeypatch.setattr(mq, "run", fake_run)
    monkeypatch.setenv("MERGE_QUEUE", "on")
    for value, expected in (("123", 123), ("", None), ("12; rm -rf /", None)):
        seen.clear()
        monkeypatch.setenv("WAIT_RUN", value)
        mq.main()
        assert seen.get("wait") == expected, value


def test_missing_contexts_are_started_without_any_gate_dispatch():
    cancelled = _run_of("container-build-check.yml", 71, "completed", "cancelled")
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled]}, run_status={71: ["in_progress"]},
                **_green({CONTAINER: "cancelled"}, statuses=[]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("start", (CONTAINER, LICENSE))
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]
    assert ("rerun", "71", "all") in gh.calls


def test_a_failure_is_retried_while_other_contexts_are_still_running():
    """Only a failed gate wakes the queue: waiting for the others would lose the retry."""
    failed = _run_of("backend-required.yml", 60, "completed", "failure", check_suite_id=600)
    checks = _checks({BACKEND: None, E2E: None}) + [_check(BACKEND, "failure", suite=600)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed, _run_of("e2e-required.yml", 70)]},
                checks={OLD: checks}, statuses={OLD: [_status()]}, run_status={60: ["in_progress"]})
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))
    assert ("rerun", "60", "failed") in gh.calls and not gh.dispatched()


def test_on_mode_rebases_the_front_only():
    runs = {OLD: [
        _run_of("backend-required.yml", 11),
        _run_of("ci.yml", 12),
        _run_of("pre-commit.yml", 13, "completed", "success"),
        _run_of("merge-queue.yml", 14, "completed", "success"),
    ], NEW: _six_held()}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-04T11:00:00Z")], runs=runs)
    _run(gh)
    assert [c for c in gh.calls if c[0] == "rebase"] == [("rebase", "PR_1", OLD)]
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]
    assert [c for c in gh.calls if c[0] == "cancel"] == [("cancel", "11"), ("cancel", "12")]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[1] == 1 and f"<!-- merge-queue:rebased:{NEW} -->" in comment[2]
    assert "approved its 6 required gates" in comment[2]
    assert all(c[1] != "PR_2" for c in gh.calls if c[0] in ("rebase", "disarm"))


def test_a_run_that_appeared_since_the_decision_stops_only_that_contexts_retry():
    """merge-queue.yml and a kick can decide the same retry for one head. The second re-reads the live
    runs first and leaves the contexts already started."""
    checks = _checks({BACKEND: None, E2E: None}) + [_check(BACKEND, "failure", suite=600),
                                                    _check(E2E, "failure", suite=700)]
    runs = [_run_of("backend-required.yml", 60, "completed", "failure", check_suite_id=600),
            _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, statuses={OLD: [_status()]}, runs={OLD: runs},
                late_runs={OLD: [_run_of("backend-required.yml", 88)]}, run_status={70: ["in_progress"]})
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND, E2E))
    assert [c for c in gh.calls if c[0] == "rerun"] == [("rerun", "70", "failed")]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"merge-queue:retry-{E2E}:{OLD}" in comment[2]


@pytest.mark.parametrize("path", ["approve", "rerun"])
@pytest.mark.parametrize("error", ["gh: Server Error (HTTP 502)", "gh: API rate limit exceeded (HTTP 403)"])
def test_a_github_side_error_on_approve_or_rerun_fails_the_run_and_disarms_nobody(path, error):
    """A 5xx or rate limit is GitHub's error, not the PR's: the run fails before any disarm, so the next
    event retries, and nothing behind the front is touched."""
    nodes = [_node(1, state="BLOCKED")] + [_node(i, armed=f"2026-10-04T1{i}:00:00Z", state="BLOCKED") for i in range(2, 6)]
    if path == "approve":
        gh = FakeGh(nodes, runs={OLD: [_held(35, "e2e-required.yml")]}, approve_error=error, **_green({E2E: None}))
    else:
        checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
        gh = FakeGh(nodes, runs={OLD: [_run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)]},
                    checks={OLD: checks}, statuses={OLD: [_status()]}, rerun_error=error)
    with pytest.raises(mq.GhError):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)
    assert not any(c[0] == "rebase" for c in gh.calls), "nothing behind the front was touched"


@pytest.mark.parametrize(("run", "kind"), [
    (_run_of("frontend-required.yml", 31, "completed", "success", event="workflow_dispatch", check_suite_id=731,
             actor={"login": "someone"}), "start"),
    (_run_of("frontend-required.yml", 31, "completed", "cancelled", check_suite_id=731), "start"),
    (_run_of("frontend-required.yml", 31, "completed", "success", check_suite_id=731), "retry"),
    (_run_of("frontend-required.yml", 31, "completed", "skipped", check_suite_id=731), "retry"),
], ids=["a-dispatch-is-ignored", "cancelled-is-no-run", "green-without-its-check-fails", "skipped-without-its-check-fails"])
def test_what_a_gate_run_without_its_check_means(run, kind):
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [dict(run)]}, **_green({"frontend-required": None}))
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts) == (kind, ("frontend-required",))


def test_the_queue_never_dispatches_a_check_context():
    """Spec section 7: the only dispatches are the license gate and the kick, both on dev. Checked on
    the syntax tree (only `dispatch_license` and `wake` call `dispatch`) and on behaviour (every
    dispatch any scenario in this file records is one of the two)."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    callers = {
        fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dispatch"
    }
    assert callers == {"dispatch_license", "wake"}
    assert SCRIPT.read_text(encoding="utf-8").count("/dispatches") == 1, "one place builds a dispatch"


def _assert_only_allowed_dispatches(gh):
    for _, workflow, fields in gh.dispatched():
        assert fields["ref"] == "dev", fields
        license_call = workflow == LICENSE_GATE and set(fields) == {"ref", "inputs[pr]"}
        kick = (workflow == "security-required.yml" and fields.get("inputs[queue_kick]") == "true"
                and set(fields) <= {"ref", "inputs[queue_kick]", "inputs[wait_run]"})
        assert license_call or kick, (workflow, fields)


@pytest.mark.parametrize("scenario", [
    {"nodes": [_node(1)], "runs": {NEW: []}},                                             # rebase, nothing held
    {"nodes": [_node(1, state="BLOCKED")], "checks": {OLD: []}, "statuses": {OLD: []}},  # nothing at all
    {"nodes": [_node(1, state="BLOCKED")], **_green({E2E: "failure"})},                    # failed once
    {"nodes": [_node(1, state="BLOCKED")], **_green(statuses=[_status("failure")])},       # license failed once
], ids=["rebase", "nothing", "gate-failed", "license-failed"])
def test_no_scenario_dispatches_anything_but_the_license_and_the_kick(scenario):
    scenario = dict(scenario)
    nodes = scenario.pop("nodes")
    gh = FakeGh(nodes, **scenario)
    try:
        _run(gh)
    except mq.GhError:
        pass
    _assert_only_allowed_dispatches(gh)
