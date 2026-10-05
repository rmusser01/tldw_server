#!/usr/bin/env python3
"""One-at-a-time merge queue for `dev`.

Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md

Runs inside GitHub Actions with the built-in GITHUB_TOKEN (merge-queue.yml and the queue-tick
jobs of the required workflows). Each run reads the line of armed PRs, decides one action for
the PR at the front and, in `on` mode, performs it. PRs behind the front are never touched.

Never enables auto-merge, never merges, never pushes: a merge made with GITHUB_TOKEN pushes
to dev without triggering any workflow, which would silently stop this queue and dev's
post-merge checks (spec section 3).

Mode comes from the MERGE_QUEUE repository variable: unset/off = do nothing,
dry = decide and log only, on = act.

Run as `python3 -m Helper_Scripts.ci.merge_queue`.
"""

from __future__ import annotations

import json
import os
# subprocess only ever runs the `gh` CLI with a fixed argv list, never a shell.
import subprocess  # nosec B404
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol
from urllib.parse import quote

REPO = os.environ.get("GITHUB_REPOSITORY", "rmusser01/tldw_server")
BASE = "dev"
QUEUE_WORKFLOW = "merge-queue.yml"
YOUNG_HEAD = timedelta(minutes=3)
STUCK_GREEN = timedelta(minutes=15)
# A commit status has no run behind it that the queue can watch: a license run cancelled after
# it posted `pending` publishes nothing more, and would read as running forever and stall the
# line. A `pending` older than this is ignored. The license job's own timeout is 5 minutes; this
# must stay longer.
STALE_PENDING = timedelta(minutes=15)
MAX_FRONTS_PER_RUN = 10
UNKNOWN_REREADS = 12
UNKNOWN_SLEEP_S = 10
# updatePullRequestBranch returns the PRE-rebase head; the branch moves about 1 s later.
REBASE_POLLS = 10
REBASE_POLL_S = 3
# Every list read follows pages to the end. A list longer than this fails the run rather than
# deciding on a partial view (an armed PR or a live run on a later page would be invisible).
PER_PAGE = 100
MAX_PAGES = 10
PASSING = frozenset({"success", "neutral", "skipped"})
LIVE_RUN_STATUSES = frozenset({"queued", "in_progress", "waiting", "requested", "pending"})
BROKEN_RUN_CONCLUSIONS = frozenset({"failure", "startup_failure", "timed_out"})
# Who a run started with GITHUB_TOKEN belongs to: the queue's own dispatches and rebases.
QUEUE_ACTOR = "github-actions[bot]"
# gh reports API errors as `gh: <message> (HTTP NNN)`. These two mean the PR's branch itself
# refuses the dispatch; every other error is GitHub's and fails the run so the next event retries.
BRANCH_REFUSALS = ("(HTTP 422)", "(HTTP 404)")
# GitHub refuses to let an Actions token create or update a file under .github/workflows
# ("refusing to allow a GitHub App to create or update workflow ... without `workflows`
# permission"), and GITHUB_TOKEN can never hold that permission. A rebase that would bring
# dev's workflow changes into the PR branch therefore fails every time it is tried.
WORKFLOW_REFUSAL = ("workflow", "permission")

MISSING, RUNNING, PASSED, FAILED, FAILED_TWICE = "missing", "running", "passed", "failed", "failed twice"


@dataclass(frozen=True)
class Context:
    """One required status on `dev` and how the queue starts it (spec 4.1).

    Attributes:
        name: The context, as branch protection names it.
        workflow: The workflow file that reports it.
        kind: `check` for a check run on the PR head, produced by the workflow dispatched on
            the PR's branch; `status` for a commit status on the PR head, produced by the
            workflow dispatched on `dev` with the `pr` input.
        base_sha: Whether a `check` dispatch takes dev's tip as its `base_sha` input.
        queue_only_dispatch: Whether a dispatch of the workflow reports this context only when
            the queue made it. `frontend-required.yml` names its gate
            `frontend-required-diagnostic` on a dispatch by anyone else (spec 4.8), so such a
            run is not a run of this context at all.
    """

    name: str
    workflow: str
    kind: str = "check"
    base_sha: bool = True
    queue_only_dispatch: bool = False


CONTEXTS = (
    Context("backend-required", "backend-required.yml"),
    Context("security-required", "security-required.yml"),
    Context("coverage-required", "coverage-required.yml"),
    Context("frontend-required", "frontend-required.yml", queue_only_dispatch=True),
    Context("e2e-required", "e2e-required.yml"),
    Context("container-build-check", "container-build-check.yml", base_sha=False),
    Context("frontend-license-policy/trusted/dev", "frontend-license-gate.yml", kind="status"),
)
ALL_CONTEXTS = tuple(c.name for c in CONTEXTS)


@dataclass(frozen=True)
class CheckRun:
    """One run of a required context on a commit: a check run, a commit status, or a stand-in.

    Attributes:
        context: The required context this run reports.
        status: The run's status (`queued`, `in_progress`, `completed`, ...).
        conclusion: The conclusion once completed (`success`, `failure`, ...), else None.
        completed_at: When it completed, else None.
        url: The run's web URL, quoted in comments.
        suite_id: The check suite it belongs to, which links it to its workflow run.
        started_at: When a `pending` commit status was posted (see STALE_PENDING), else None.
    """

    context: str
    status: str
    conclusion: str | None
    completed_at: datetime | None
    url: str
    suite_id: int | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class PrState:
    """What the queue needs to know about one open PR into dev.

    Attributes:
        number: The PR number.
        node_id: The PR's GraphQL node id, used by mutations.
        head_sha: The current head commit.
        head_ref: The head branch name, the ref the check workflows are dispatched on.
        same_repo: Whether the head branch lives in this repository (not a fork).
        armed_at: When auto-merge was enabled, or None if it is not armed.
        is_draft: Whether the PR is a draft.
        merge_state: GitHub's `mergeStateStatus` (`BEHIND`, `CLEAN`, `DIRTY`, ...).
        head_committed_at: The head commit's date.
        base_sha: dev's tip, read together with the PR; sent as the `base_sha` dispatch input.
        checks: The required contexts' runs on the head, stand-ins included.
        human_author: Whether a user, not a bot or app, opened the PR.
    """

    number: int
    node_id: str
    head_sha: str
    head_ref: str
    same_repo: bool
    armed_at: datetime | None
    is_draft: bool
    merge_state: str
    head_committed_at: datetime
    base_sha: str = ""
    checks: tuple[CheckRun, ...] = ()
    human_author: bool = True


@dataclass(frozen=True)
class Action:
    """The single decision for the front PR: wait, rebase, dispatch, retry or evict.

    `slug` names an eviction's cause in its comment marker (`evict-<slug>`), so a re-armed PR
    evicted again on the same head for a different reason is still told why.

    Attributes:
        kind: `wait`, `rebase`, `dispatch`, `retry` or `evict`.
        reason: A human-readable cause, logged and quoted in comments.
        links: Run URLs quoted in the comment (the failed runs).
        slug: An eviction's cause, used in its comment marker.
        contexts: The required contexts the action targets: the ones a rebase, dispatch or
            retry starts, and the ones a wait or eviction is about.
    """

    kind: str
    reason: str
    links: tuple[str, ...] = ()
    slug: str = ""
    contexts: tuple[str, ...] = ()


def line_of(prs: list[PrState]) -> list[PrState]:
    """Return the queue: armed, non-draft, same-repo, user-authored PRs, oldest arming first.

    Args:
        prs: Every open PR into dev.

    Returns:
        The PRs in queue order.
    """
    eligible = [p for p in prs if p.armed_at is not None and not p.is_draft and p.same_repo and p.human_author]
    return sorted(eligible, key=lambda p: (p.armed_at, p.number))


def _finished(runs: list[CheckRun], now: datetime) -> list[CheckRun]:
    """Completed runs that count (a cancelled run is no run), oldest first."""
    done = (c for c in runs if c.status == "completed" and c.conclusion != "cancelled")
    return sorted(done, key=lambda c: c.completed_at or now)


def context_state(runs: list[CheckRun], now: datetime) -> str:
    """Reduce one context's runs on the head to its state (spec 4.1).

    A live run means `running`, whatever came before it. Otherwise the latest finished run
    decides between `passed` and failed, and the number of failed runs on this head between
    `failed` (once) and `failed twice`.

    Args:
        runs: The context's runs on the head.
        now: The current time (UTC).

    Returns:
        `missing`, `running`, `passed`, `failed` or `failed twice`.
    """
    if any(c.status != "completed" and not (c.started_at and now - c.started_at > STALE_PENDING) for c in runs):
        return RUNNING
    finished = _finished(runs, now)
    if not finished:
        return MISSING
    if finished[-1].conclusion in PASSING:
        return PASSED
    return FAILED_TWICE if sum(c.conclusion not in PASSING for c in finished) >= 2 else FAILED


def decide_front(pr: PrState, now: datetime) -> Action:
    """Decide the one action for the PR at the front of the line (spec 4.1 and section 6).

    A failed context is acted on while other contexts are still running. Spec 4.1 lists
    "any context running: wait" first, but only a FAILED gate wakes the queue (spec 4.6): a gate
    that fails early while a slower gate later passes would never be retried or evicted.

    Args:
        pr: The front PR, with the required contexts' runs on the current head.
        now: The current time (UTC).

    Returns:
        The action to take, naming the contexts it targets.
    """
    state = pr.merge_state
    if state == "UNKNOWN":
        return Action("wait", "merge state unknown")
    if state == "BEHIND":
        return Action("rebase", "behind dev", contexts=ALL_CONTEXTS)
    if state == "DIRTY":
        return Action("evict", "conflicts with dev", slug="conflict")
    runs = {name: [c for c in pr.checks if c.context == name] for name in ALL_CONTEXTS}
    states = {name: context_state(runs[name], now) for name in ALL_CONTEXTS}

    def named(wanted: str) -> tuple[str, ...]:
        """The contexts in the wanted state, in table order."""
        return tuple(name for name in ALL_CONTEXTS if states[name] == wanted)

    def failed_urls(names: tuple[str, ...], last: int) -> tuple[str, ...]:
        """The URLs of each named context's last `last` failed runs."""
        urls: list[str] = []
        for name in names:
            failed = [c for c in _finished(runs[name], now) if c.conclusion not in PASSING][-last:]
            urls += [c.url for c in failed if c.url]  # a commit status may carry no link
        return tuple(urls)

    twice, once = named(FAILED_TWICE), named(FAILED)
    if twice:
        return Action("evict", f"{', '.join(twice)} failed twice", failed_urls(twice, 2), "failed-twice", twice)
    if once:
        return Action("retry", f"{', '.join(once)} failed once; retrying", failed_urls(once, 1), contexts=once)
    running, missing = named(RUNNING), named(MISSING)
    if running:
        return Action("wait", f"running: {', '.join(running)}", contexts=running)
    if missing:
        if now - pr.head_committed_at <= YOUNG_HEAD:
            return Action("wait", "head is under 3 minutes old; its own runs may not be visible yet", contexts=missing)
        return Action("dispatch", f"no run on the up-to-date head: {', '.join(missing)}", contexts=missing)
    # BLOCKED with everything green is mergeStateStatus lagging the checks. Review threads are
    # not consulted: dev's rules do not require conversations to be resolved, so an unresolved
    # thread never blocks the merge and must never evict a green PR.
    if state in ("CLEAN", "UNSTABLE", "HAS_HOOKS", "BLOCKED"):
        last_green = max(_finished(runs[name], now)[-1].completed_at or now for name in ALL_CONTEXTS)
        if now - last_green > STUCK_GREEN:
            return Action(
                "evict", "green for over 15 minutes but auto-merge did not fire; re-arm to retry", slug="stuck",
            )
        return Action("wait", "green; auto-merge should fire")
    return Action("wait", f"merge state {state}")


class GhError(RuntimeError):
    """A gh CLI call failed, or a list read was longer than the queue will page through."""


class GhApi(Protocol):
    """What the queue needs from GitHub; `Gh` in production, a fake in tests."""

    def graphql(self, query: str, **variables: object) -> dict:
        """Run a GraphQL query or mutation.

        Args:
            query: The GraphQL document.
            **variables: Its variables; ints are sent typed, everything else as strings.

        Returns:
            The decoded response (`{"data": ...}`).

        Raises:
            GhError: The call failed.
        """
        ...

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object:
        """Call a REST endpoint.

        Args:
            method: The HTTP method.
            path: The path relative to the API root, query string included.
            fields: String fields sent as the request body.

        Returns:
            The decoded JSON response, or None for an empty body.

        Raises:
            GhError: The call failed.
        """
        ...


def _run_gh(args: list[str]) -> str:
    """Run `gh` with the given arguments and return its stdout; raise GhError on a non-zero exit."""
    # `gh` is resolved from PATH (preinstalled on hosted runners); argv is a list built from
    # constants and API values, with no shell.
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)  # nosec B603 B607
    if proc.returncode != 0:
        raise GhError(f"gh {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


class Gh:
    """Thin wrapper over the gh CLI (preinstalled on hosted runners; auth via GH_TOKEN).

    Only ever runs `gh api`; see `GhApi` for the method contracts.
    """

    def __init__(self, runner: Callable[[list[str]], str] | None = None) -> None:
        """Create the wrapper.

        Args:
            runner: Runs `gh` with the given arguments and returns its stdout; defaults to a
                subprocess call that raises GhError on a non-zero exit.
        """
        self._run = runner or _run_gh

    def graphql(self, query: str, **variables: object) -> dict:
        """See `GhApi.graphql`."""
        args = ["api", "graphql", "-f", f"query={query}"]
        for key, value in variables.items():
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        return json.loads(self._run(args))

    def rest(self, method: str, path: str, fields: dict[str, str] | None = None) -> object:
        """See `GhApi.rest`."""
        args = ["api", "-X", method, path]
        for key, value in (fields or {}).items():
            args += ["-f", f"{key}={value}"]
        out = self._run(args)
        return json.loads(out) if out.strip() else None


# baseRef is the live branch, so its target is dev's tip at the moment the PR state was read.
PR_FIELDS = """
  number id isDraft headRefOid headRefName mergeStateStatus
  author { __typename }
  headRepository { nameWithOwner }
  baseRef { target { oid } }
  autoMergeRequest { enabledAt }
  commits(last: 1) { nodes { commit { committedDate } } }
"""
LINE_QUERY = (
    "query($owner: String!, $name: String!, $after: String) { repository(owner: $owner, name: $name) {"
    f' pullRequests(states: OPEN, baseRefName: "{BASE}", first: {PER_PAGE}, after: $after) {{'
    f" pageInfo {{ hasNextPage endCursor }} nodes {{ {PR_FIELDS} }} }} }} }}"
)
PR_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {"
    f" pullRequest(number: $number) {{ {PR_FIELDS} }} }} }}"
)
COMMENTS_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {"
    " pullRequest(number: $number) { comments(last: 100) { nodes { body } } } } }"
)
REBASE_MUTATION = (
    "mutation($id: ID!, $oid: GitObjectID!) { updatePullRequestBranch(input: "
    "{pullRequestId: $id, expectedHeadOid: $oid, updateMethod: REBASE}) { pullRequest { headRefOid } } }"
)
DISARM_MUTATION = (
    "mutation($id: ID!) { disablePullRequestAutoMerge(input: {pullRequestId: $id}) { clientMutationId } }"
)


def _ts(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp (`...Z`) into an aware datetime; None stays None."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _owner_name() -> tuple[str, str]:
    """Split REPO (`owner/name`) into the two GraphQL variables."""
    owner, name = REPO.split("/")
    return owner, name


def _parse_pr(node: dict) -> PrState:
    """Build a PrState (without its checks) from one GraphQL pull-request node of PR_FIELDS."""
    commits = node["commits"]["nodes"]
    auto = node.get("autoMergeRequest")
    head_repo = (node.get("headRepository") or {}).get("nameWithOwner")
    return PrState(
        number=node["number"],
        node_id=node["id"],
        head_sha=node["headRefOid"],
        head_ref=node["headRefName"],
        same_repo=head_repo == REPO,
        armed_at=_ts(auto["enabledAt"]) if auto else None,
        is_draft=node["isDraft"],
        merge_state=node["mergeStateStatus"],
        head_committed_at=_ts(commits[0]["commit"]["committedDate"]) if commits else datetime.now(timezone.utc),
        base_sha=((node.get("baseRef") or {}).get("target") or {}).get("oid") or "",
        human_author=(node.get("author") or {}).get("__typename") == "User",
    )


def read_prs(gh: GhApi) -> list[PrState]:
    """Read every open PR into dev, following the GraphQL cursor to the last page.

    Args:
        gh: The GitHub client.

    Returns:
        Every open PR into dev, in API order.

    Raises:
        GhError: A call failed, or there are more than MAX_PAGES pages of open PRs.
    """
    owner, name = _owner_name()
    prs: list[PrState] = []
    cursor = None
    for _ in range(MAX_PAGES):
        variables = {"owner": owner, "name": name}
        if cursor:
            variables["after"] = cursor
        page = gh.graphql(LINE_QUERY, **variables)["data"]["repository"]["pullRequests"]
        prs += [_parse_pr(n) for n in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]
    raise GhError(f"more than {MAX_PAGES * PER_PAGE} open PRs into {BASE}; refusing to decide on a partial line")


def read_pr(gh: GhApi, number: int) -> PrState:
    """Read one PR's current state, and dev's tip with it.

    Args:
        gh: The GitHub client.
        number: The PR number.

    Returns:
        The PR's state, without checks.

    Raises:
        GhError: The call failed.
    """
    owner, name = _owner_name()
    data = gh.graphql(PR_QUERY, owner=owner, name=name, number=number)
    return _parse_pr(data["data"]["repository"]["pullRequest"])


def _rest_pages(gh: GhApi, path: str, key: str | None = None) -> list[dict]:
    """Every item of a paged REST list.

    Pages are followed until `total_count` (when the response has one) is reached or a short
    page comes back. `key` names the list inside an object response; None means the response
    is the list itself.
    """
    items: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        data = gh.rest("GET", f"{path}{'&' if '?' in path else '?'}per_page={PER_PAGE}&page={page}")
        batch = (data or []) if key is None else (data or {}).get(key, [])
        items += batch
        total = None if key is None else (data or {}).get("total_count")
        if len(batch) < PER_PAGE or (total is not None and len(items) >= total):
            return items
    raise GhError(f"more than {MAX_PAGES * PER_PAGE} items for {path}; refusing to decide on a partial view")


def read_checks(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read every run of the check-run contexts on a commit.

    A head here carries over 250 check runs, more with re-runs, so each context is read by name
    (`check_name`) rather than by paging through all of them: no number of other checks on the
    head can push a required one past the page cap.

    Args:
        gh: The GitHub client.
        sha: The commit.

    Returns:
        The check-run contexts' runs, all pages.

    Raises:
        GhError: A call failed, or one context has more than MAX_PAGES pages of runs.
    """
    runs: list[CheckRun] = []
    for ctx in CONTEXTS:
        if ctx.kind != "check":
            continue
        path = f"repos/{REPO}/commits/{sha}/check-runs?check_name={quote(ctx.name, safe='')}&filter=all"
        runs += [
            CheckRun(ctx.name, c["status"], c.get("conclusion"), _ts(c.get("completed_at")), c["html_url"],
                     (c.get("check_suite") or {}).get("id"))
            for c in _rest_pages(gh, path, "check_runs")
        ]
    return tuple(runs)


def read_statuses(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read the commit-status contexts on a commit, as runs.

    A commit's status list keeps every status ever posted for a context, newest first. Each
    `success`, `failure` or `error` entry is one finished run, so two `failure`/`error` entries
    on the same head are "failed twice". A `pending` entry is a live run only while it is the
    newest entry; an older one was superseded by the result posted after it.

    Args:
        gh: The GitHub client.
        sha: The commit.

    Returns:
        The status contexts' runs.

    Raises:
        GhError: A call failed, or the commit has more than MAX_PAGES pages of statuses.
    """
    names = [c.name for c in CONTEXTS if c.kind == "status"]
    runs: list[CheckRun] = []
    entries = sorted(_rest_pages(gh, f"repos/{REPO}/commits/{sha}/statuses"), key=lambda s: s.get("created_at") or "")
    for name in names:
        history = [s for s in entries if s.get("context") == name]
        for index, status in enumerate(history):
            posted, url = _ts(status.get("created_at")), status.get("target_url") or ""
            if status.get("state") != "pending":
                conclusion = "success" if status.get("state") == "success" else "failure"
                runs.append(CheckRun(name, "completed", conclusion, posted, url))
            elif index == len(history) - 1:
                runs.append(CheckRun(name, "pending", None, None, url, started_at=posted))
    return tuple(runs)


def runs_on(gh: GhApi, sha: str, status: str | None = None) -> list[dict]:
    """Read every workflow run on a commit.

    Args:
        gh: The GitHub client.
        sha: The head commit.
        status: Only runs with this status (e.g. `action_required`), if given.

    Returns:
        The workflow runs, all pages, as the API returns them.

    Raises:
        GhError: A call failed, or there are more than MAX_PAGES pages.
    """
    path = f"repos/{REPO}/actions/runs?head_sha={sha}"
    if status:
        path += f"&status={status}"
    return _rest_pages(gh, path, "workflow_runs")


def _workflow_name(run: dict) -> str:
    """The workflow file's basename, stripped of any `@ref` suffix the API may add."""
    return str(run.get("path", "")).split("/")[-1].split("@")[0]


def _own_run_id() -> int:
    """The id of the workflow run executing this script, or 0 outside Actions."""
    return int(os.environ.get("GITHUB_RUN_ID", "0") or 0)


def _host_workflow() -> str:
    """The workflow file whose run is executing this script, or "" outside Actions.

    `GITHUB_WORKFLOW_REF` is `owner/repo/.github/workflows/<file>@<ref>`.
    """
    ref = os.environ.get("GITHUB_WORKFLOW_REF", "")
    return ref.split("@", 1)[0].rsplit("/", 1)[-1]


def required_run_stand_ins(runs: list[dict], checks: tuple[CheckRun, ...]) -> tuple[CheckRun, ...]:
    """Required-workflow runs on a head that have not reported their check, as stand-ins.

    - A LIVE run stands in as an in-flight check: each required check is a needs-gated job with
      no check run until the jobs before it finish, so an in-flight front PR would otherwise
      look like it has no run and be dispatched again on every wake.
    - A COMPLETED run with no check run for its context in its check suite stands in as one
      failure when it failed (`BROKEN_RUN_CONCLUSIONS`: a startup failure, e.g. a broken
      workflow file on the branch), or when the queue itself dispatched it and it finished
      green without ever reporting the context (the branch's copy names its gate job
      differently). Either would otherwise read as "no run" and be dispatched forever. A failed
      run whose suite DID report the check is not a gate failure (its queue-tick may have
      failed); that check run already speaks for it.

    A hand-started `frontend-required.yml` dispatch is skipped: it publishes a diagnostic name,
    never the required one (spec 4.8).

    A workflow-run conclusion is only ever read as a failure here, never as a merge signal.
    The queue's own run (GITHUB_RUN_ID) is never counted: a queue-tick runs inside the required
    workflow run it reports on. The license gate runs on `dev`, not on this head, so it has no
    stand-in; its commit status is all the queue sees of it.

    Args:
        runs: The workflow runs on the head, from `runs_on`.
        checks: The check-run contexts' runs on that head, from `read_checks`.

    Returns:
        One stand-in per live run and per completed run that counts as a failure.
    """
    by_workflow = {c.workflow: c for c in CONTEXTS if c.kind == "check"}
    reported = {(c.context, c.suite_id) for c in checks if c.suite_id is not None}
    stand_ins = []
    for run in runs:
        ctx = by_workflow.get(_workflow_name(run))
        if ctx is None or run.get("id") == _own_run_id():
            continue
        # A diagnostic dispatch can never report the required name, so neither its being live
        # nor its failing says anything about the context. `actor` is who started the run and,
        # unlike `triggering_actor`, does not change when someone re-runs it.
        if (ctx.queue_only_dispatch and run.get("event") == "workflow_dispatch"
                and (run.get("actor") or {}).get("login") != QUEUE_ACTOR):
            continue
        context = ctx.name
        url, conclusion = run.get("html_url", ""), run.get("conclusion")
        if run.get("status") in LIVE_RUN_STATUSES:
            stand_ins.append(CheckRun(context, run["status"], None, None, url))
            continue
        silent_green = (conclusion == "success" and run.get("event") == "workflow_dispatch"
                        and (run.get("triggering_actor") or {}).get("login") == QUEUE_ACTOR)
        if (run.get("status") == "completed" and (conclusion in BROKEN_RUN_CONCLUSIONS or silent_green)
                and (context, run.get("check_suite_id")) not in reported):
            stand_ins.append(CheckRun(context, "completed", "failure", _ts(run.get("updated_at")), url))
    return tuple(stand_ins)


def read_contexts(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read all seven required contexts' runs on a commit (spec 4.1).

    Workflow runs are read before check runs: a run seen as completed already has its check
    runs, so "completed without reporting its check" is never an artefact of the read order.

    Args:
        gh: The GitHub client.
        sha: The head commit.

    Returns:
        Check runs, workflow-run stand-ins and commit statuses, each tagged with its context.

    Raises:
        GhError: A call failed.
    """
    runs = runs_on(gh, sha)
    checks = read_checks(gh, sha)
    return checks + required_run_stand_ins(runs, checks) + read_statuses(gh, sha)


def _best_effort(log: Callable[[str], None], what: str, fn: Callable[[], object]) -> None:
    """Run `fn`; log a GhError as `what` instead of raising, for calls the queue can do without."""
    try:
        fn()
    except GhError as exc:
        log(f"  best-effort {what} failed: {exc}")


def comment_once(gh: GhApi, number: int, kind: str, sha: str, body: str) -> bool:
    """Post a comment unless one of this kind already exists for this head.

    Args:
        gh: The GitHub client.
        number: The PR number.
        kind: The comment's kind, part of its hidden marker.
        sha: The head the comment is about, part of its hidden marker.
        body: The visible text.

    Returns:
        True if it posted, False if the marker was already there.

    Raises:
        GhError: A call failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    bodies = [n.get("body") or "" for n in data["data"]["repository"]["pullRequest"]["comments"]["nodes"]]
    if any(marker in b for b in bodies):
        return False
    gh.rest("POST", f"repos/{REPO}/issues/{number}/comments", {"body": f"{marker}\n{body}"})
    return True


def dispatch(gh: GhApi, workflow: str, ref: str, inputs: dict[str, str] | None = None) -> None:
    """Start a workflow_dispatch run.

    Args:
        gh: The GitHub client.
        workflow: The workflow file's basename.
        ref: The branch to run it on (its copy of the workflow is the one that runs).
        inputs: The dispatch inputs, if any.

    Raises:
        GhError: GitHub refused the dispatch (e.g. HTTP 422, no workflow_dispatch trigger).
    """
    fields = {"ref": ref}
    for key, value in (inputs or {}).items():
        fields[f"inputs[{key}]"] = value
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{workflow}/dispatches", fields)


def dispatch_request(ctx: Context, pr: PrState) -> tuple[str, str, dict[str, str]]:
    """Say how one required context is started for a PR's head (spec 4.1 table).

    A check context runs the PR branch's copy of its workflow, with dev's tip as `base_sha`
    where the workflow takes one. The license status runs dev's copy of the license gate with
    the PR number, so the policy code is always dev's.

    Args:
        ctx: The context to start.
        pr: The PR; `head_ref`, `base_sha` and `number` are used.

    Returns:
        The workflow file, the ref to dispatch it on, and its inputs.

    Raises:
        GhError: dev's tip is unknown. Without `base_sha` a gate would compare against `HEAD^`
            and could pass with nothing tested (spec 4.3).
    """
    if ctx.kind == "status":
        return ctx.workflow, BASE, {"pr": str(pr.number)}
    if not ctx.base_sha:
        return ctx.workflow, pr.head_ref, {}
    if not pr.base_sha:
        raise GhError(f"{BASE}'s tip is unknown for #{pr.number}; refusing to dispatch {ctx.workflow} without base_sha")
    return ctx.workflow, pr.head_ref, {"base_sha": pr.base_sha}


def _evict(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> bool:
    """Disarm the PR and tell it why, once per cause and head; always returns True (it left the line)."""
    # Best-effort: a merge fires both push:dev and pull_request:closed, so two racing runs can
    # evict the same PR and the second disarm hits an already-disarmed PR.
    _best_effort(log, f"disarm #{pr.number}", lambda: gh.graphql(DISARM_MUTATION, id=pr.node_id))
    links = "".join(f"\n- {u}" for u in action.links)
    comment_once(
        gh, pr.number, f"evict-{action.slug}", pr.head_sha,
        f"Merge queue: removed from the line ({action.reason}). Auto-merge is now off. Fix the cause, then "
        f"re-arm with `gh pr merge {pr.number} --auto --merge` to rejoin at the back.{links}",
    )
    log(f"  evicted #{pr.number}: {action.reason}")
    return True


def _dispatch_contexts(
    gh: GhApi, pr: PrState, names: tuple[str, ...], log: Callable[[str], None],
    finish: Callable[[], None] = lambda: None,
) -> bool:
    """Dispatch the named contexts on the PR's head; a refusal by the PR's branch evicts the PR.

    Only HTTP 422 (the branch's workflow is broken, lacks the trigger or rejects the input) and
    HTTP 404 (the ref is gone) on a dispatch to the PR's OWN branch are the PR's fault. Anything
    else re-raises, so the run fails and the next event retries: a 5xx, rate limit, network
    error or 403, and any error at all from the license gate, which is dispatched on `dev`.
    Evicting on those would disarm every front that an outage, or a broken `dev`, touches.

    The license gate goes first: it is the one dispatch the PR cannot be blamed for, so if it
    fails the run stops before any runner is spent on the six gates.

    The workflow this script is running in goes last. A queue-tick runs inside a required
    workflow's own run, and dispatching that same workflow on the same branch cancels the run
    executing this script (one concurrency group per workflow, event and ref, with
    cancel-in-progress). Everything the caller still has to do once the runs are started is
    passed as `finish`: it runs after the last dispatch, or just before the host workflow's
    dispatch when there is one, so a cancelled tick loses nothing.

    Args:
        gh: The GitHub client.
        pr: The PR whose head the contexts are started on.
        names: The contexts to start.
        log: Receives one line per dispatch.
        finish: The caller's remaining work (cancel superseded runs, comment). Not run if the
            PR is evicted before the point where it would run.

    Returns:
        True if the PR was evicted (the line moves on), False if the runs were started.
    """
    host = _host_workflow()
    ordered = sorted(
        (c for c in CONTEXTS if c.name in names), key=lambda c: (c.workflow == host, c.kind != "status"),
    )
    requests = [(ctx, dispatch_request(ctx, pr)) for ctx in ordered]  # all resolved before any is sent
    finished = False
    for ctx, (workflow, ref, inputs) in requests:
        if workflow == host and not finished:
            finish()
            finished = True
        try:
            dispatch(gh, workflow, ref, inputs)
        except GhError as exc:
            if ctx.kind == "status" or not any(code in str(exc) for code in BRANCH_REFUSALS):
                raise
            reason = f"CI dispatch of {workflow} failed: {str(exc)[:200]}"
            return _evict(gh, pr, Action("evict", reason, slug="dispatch"), log)
        log(f"  dispatched {workflow} on {ref} for #{pr.number}")
    if not finished:
        finish()
    return False


def _rebase(gh: GhApi, pr: PrState, log: Callable[[str], None], sleep: Callable[[float], None]) -> bool:
    """Rebase the front PR onto dev with its head pinned, then start the seven contexts (spec section 7).

    Returns:
        True if the PR was evicted (conflicts, a rebase the token may not do, a second failed
        rebase, or a refused dispatch), False otherwise.
    """
    try:
        gh.graphql(REBASE_MUTATION, id=pr.node_id, oid=pr.head_sha)
    except GhError as exc:
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha == pr.head_sha and fresh.merge_state != "DIRTY":
            # A racing run's rebase may have been accepted with its ref update still landing
            # (about 1 s): look once more before counting this as our own failure.
            sleep(REBASE_POLL_S)
            fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            log(f"  rebase skipped: head moved to {fresh.head_sha[:10]}")
            return False
        if fresh.merge_state == "DIRTY":
            return _evict(gh, fresh, Action("evict", "conflicts with dev (rebase refused)", slug="conflict"), log)
        if all(word in str(exc).lower() for word in WORKFLOW_REFUSAL):
            # Retrying cannot help, and waiting would hold up everyone behind this PR.
            reason = (
                "dev changed workflow files since this branch was cut, and the queue's token is not allowed to "
                "rebase across them; rebase onto dev by hand (`git fetch origin dev && git rebase origin/dev`, "
                "then force-push with lease)"
            )
            return _evict(gh, fresh, Action("evict", reason, slug="workflows"), log)
        error = str(exc)[:200]
        posted = comment_once(
            gh, pr.number, "rebase-failed", pr.head_sha,
            f"Merge queue: rebasing onto dev failed ({error}); will retry once, then remove from the line.",
        )
        if posted:
            log(f"  rebase failed, will retry once: {exc}")
            return False
        return _evict(gh, fresh, Action("evict", f"rebase onto dev keeps failing: {error}", slug="rebase"), log)
    # The mutation returns the PRE-rebase headRefOid and the branch moves about 1 s later, so
    # wait for the new head to appear before dispatching CI on it.
    rebased = None
    for _ in range(REBASE_POLLS):
        sleep(REBASE_POLL_S)
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            rebased = fresh
            break
    if rebased is None:
        log(f"  rebase of #{pr.number} accepted but the head never moved; a later wake recovers it")
        return False
    old_runs = runs_on(gh, pr.head_sha)

    def finish() -> None:
        """Cancel the old head's runs, drop empty approval runs and announce the rebase."""
        # The run executing this script is never cancelled: a queue-tick runs inside a required
        # workflow's run on the old head, and an auto_merge_enabled-triggered merge-queue.yml
        # run shares that head too.
        for run in old_runs:
            if run.get("status") not in LIVE_RUN_STATUSES:
                continue
            if run.get("id") == _own_run_id() or _workflow_name(run) == QUEUE_WORKFLOW:
                continue
            _best_effort(log, f"cancel run {run['id']}",
                         lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
        cleanup_approval_runs(gh, rebased, log)
        comment_once(
            gh, pr.number, "rebased", rebased.head_sha,
            f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{rebased.head_sha[:10]}`) and started "
            f"the {len(CONTEXTS)} required checks.",
        )

    # A rebase made with GITHUB_TOKEN starts no pull_request run, so the queue starts all seven
    # contexts itself (spec 4.2), and nothing else (spec 4.5). The old head's runs are cancelled
    # only once the new ones are started (see _dispatch_contexts for the one exception).
    if _dispatch_contexts(gh, rebased, ALL_CONTEXTS, log, finish):
        return True
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {rebased.head_sha[:10]}")
    return False


def _appeared(gh: GhApi, pr: PrState, names: tuple[str, ...], now: datetime) -> tuple[str, ...]:
    """The named contexts that have a live run on the head right now."""
    kinds = {c.kind for c in CONTEXTS if c.name in names}
    fresh: tuple[CheckRun, ...] = ()
    if "check" in kinds:
        fresh += required_run_stand_ins(runs_on(gh, pr.head_sha), ())
    if "status" in kinds:
        fresh += read_statuses(gh, pr.head_sha)
    return tuple(n for n in names if context_state([c for c in fresh if c.context == n], now) == RUNNING)


def apply(
    gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None],
    now: datetime,
) -> bool:
    """Perform one decided action (spec section 7).

    Args:
        gh: The GitHub client.
        pr: The front PR, as decided on.
        action: The decided action.
        log: Receives one line per notable step.
        sleep: Waits between rereads (injected by tests).
        now: The current time (UTC).

    Returns:
        True if the PR left the line (evicted, by decision or because an action failed), so
        the caller moves on to the next front; False otherwise.

    Raises:
        GhError: A read or a non-best-effort call failed.
    """
    if action.kind == "rebase":
        return _rebase(gh, pr, log, sleep)
    if action.kind in ("dispatch", "retry"):
        # Two queue runs (merge-queue.yml and a queue-tick) can decide the same dispatch for the
        # same head. Once the first one's run is listed, the other sees it here and stands down
        # for that context. This narrows the race but cannot close it: the window left is the
        # time between this read and the POST, plus GitHub's delay between a dispatch returning
        # and its run showing up (for the license gate: until its run posts `pending`), which
        # no read can see. A duplicate is one extra run.
        appeared = _appeared(gh, pr, action.contexts, now)
        if appeared:
            log(f"  #{pr.number}: a run appeared since the decision for {', '.join(appeared)}; not dispatching those")
        targets = tuple(n for n in action.contexts if n not in appeared)
        if not targets:
            return False

        def finish() -> None:
            """Announce a retry; a plain dispatch says nothing."""
            if action.kind == "retry":
                links = "".join(f"\n- {u}" for u in action.links)
                comment_once(
                    gh, pr.number, "retry", pr.head_sha,
                    f"Merge queue: {', '.join(targets)} failed once on `{pr.head_sha[:10]}`; retrying with a "
                    f"fresh run.{links}",
                )

        return _dispatch_contexts(gh, pr, targets, log, finish)
    if action.kind == "evict":
        return _evict(gh, pr, action, log)
    return False


def cleanup_approval_runs(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Delete the empty approval-pending runs the queue's own token rebase created.

    Best-effort end to end: a failed listing must not abort the pass.

    Args:
        gh: The GitHub client.
        pr: The front PR; runs on its head are cleaned.
        log: Receives a line per failed best-effort call.
    """
    try:
        runs = runs_on(gh, pr.head_sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list approval-pending runs failed: {exc}")
        return
    for run in runs:
        if (run.get("triggering_actor") or {}).get("login") != QUEUE_ACTOR:
            continue
        _best_effort(log, f"delete approval-pending run {run['id']}",
                     lambda rid=run["id"]: gh.rest("DELETE", f"repos/{REPO}/actions/runs/{rid}"))


UNQUEUED_NOTES = {
    "fork": "Merge queue: fork PRs are not queued, because GitHub cannot dispatch workflows on a fork's branch. "
            "A maintainer merges this one by hand.",
    "bot": "Merge queue: PRs opened by a bot or app are not queued, because a queue dispatch would run this "
           "branch's workflows without the token limits and approval gates GitHub applies to bot-authored "
           "runs. A maintainer merges this one by hand.",
}


def comment_unqueued(gh: GhApi, prs: list[PrState], mode: str, log: Callable[[str], None]) -> None:
    """Tell armed fork and bot-authored PRs, once per head, that the queue skips them.

    Best-effort: a failed comment never aborts the run.

    Args:
        gh: The GitHub client.
        prs: Every open PR into dev.
        mode: `dry` only logs; `on` also comments.
        log: Receives one line per skipped PR.
    """
    for pr in prs:
        if pr.armed_at is None or (pr.same_repo and pr.human_author):
            continue
        kind = "fork" if not pr.same_repo else "bot"
        log(f"#{pr.number}: {kind} PR armed; not queued")
        if mode == "on":
            _best_effort(
                log, f"comment on {kind} PR #{pr.number}",
                lambda p=pr, k=kind: comment_once(gh, p.number, k, p.head_sha, UNQUEUED_NOTES[k]),
            )


def settle_unknown(gh: GhApi, pr: PrState, sleep: Callable[[float], None]) -> PrState:
    """Re-read a PR whose merge state is UNKNOWN until it settles, up to UNKNOWN_REREADS times.

    Args:
        gh: The GitHub client.
        pr: The front PR.
        sleep: Waits between rereads (injected by tests).

    Returns:
        The latest state read; still UNKNOWN if it never settled.

    Raises:
        GhError: A reread failed.
    """
    for _ in range(UNKNOWN_REREADS):
        if pr.merge_state != "UNKNOWN":
            return pr
        sleep(UNKNOWN_SLEEP_S)
        pr = read_pr(gh, pr.number)
    return pr


def _still_front(gh: GhApi, pr: PrState, left: set[int]) -> bool:
    """Whether the PR is still armed, on the same head, and first in line.

    A decision is taken on a snapshot, and reading seven contexts takes a dozen calls. In that
    time another queue run may have evicted the PR, or its author may have disarmed it or
    pushed. Acting on the stale snapshot would start checks on a PR that has left the line.

    Args:
        gh: The GitHub client.
        pr: The PR as it was decided on.
        left: PRs this run has already removed from the line. Their disarm is best-effort and
            may not show in a read made a moment later, so they never count as being ahead.

    Returns:
        True if a fresh read of the line, without `left`, still starts with this PR on this head.
    """
    line = [p for p in line_of(read_prs(gh)) if p.number not in left]
    return bool(line) and line[0].number == pr.number and line[0].head_sha == pr.head_sha


def run(
    gh: GhApi,
    mode: str | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> list[tuple[int, Action]]:
    """One queue pass: decide for the front PR and, in `on` mode, act; repeat after evictions.

    Args:
        gh: The GitHub client.
        mode: The MERGE_QUEUE value; only `dry` and `on` (any case or padding) do anything.
        now: The clock (injected by tests).
        sleep: Waits between rereads (injected by tests).
        log: Receives the run's log lines.

    Returns:
        The (PR number, action) decisions taken or, in dry mode, proposed.

    Raises:
        GhError: A read or a non-best-effort call failed; the next event retries.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    prs = read_prs(gh)
    comment_unqueued(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    left: set[int] = set()
    for pr in line_of(prs)[:MAX_FRONTS_PER_RUN]:
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        pr = replace(pr, checks=read_contexts(gh, pr.head_sha))
        action = decide_front(pr, now())
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        left_line = action.kind == "evict"
        if mode == "on":
            if action.kind != "wait" and not _still_front(gh, pr, left):
                log(f"#{pr.number}: no longer the armed front PR on this head; nothing done, the next wake decides")
                break
            cleanup_approval_runs(gh, pr, log)
            left_line = apply(gh, pr, action, log, sleep, now())
        if not left_line:
            break
        left.add(pr.number)
    return decisions


def main() -> int:
    """Run one queue pass with the real gh CLI and append the decisions to the job summary.

    Returns:
        The process exit code (0; a failed call raises instead).
    """
    decisions = run(Gh(), os.environ.get("MERGE_QUEUE"))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Merge queue\n\n| PR | Action | Reason |\n|---|---|---|\n")
            for number, action in decisions:
                fh.write(f"| #{number} | {action.kind} | {action.reason} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
