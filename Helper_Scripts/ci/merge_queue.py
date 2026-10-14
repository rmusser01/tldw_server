#!/usr/bin/env python3
"""One-at-a-time merge queue for `dev`.

Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md, amended by
Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md

Runs inside GitHub Actions with the built-in GITHUB_TOKEN (merge-queue.yml, and the
queue-kick job of security-required.yml that a failed gate's queue-tick dispatches). Each run reads the line of armed PRs, decides one action for
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
# A queue rebase (GITHUB_TOKEN) makes GitHub create the PR's pull_request runs held for approval
# (spec S2). Approving a gate's held run is what makes its check count (S4); a dispatched run's
# check never reaches the PR's rollup (S5). So the queue approves and never dispatches a gate.
QUEUE_ACTOR = "github-actions[bot]"
# The queue's own comments, as GraphQL names their author: a Bot whose login has no `[bot]`
# (checked on tldw_chatbook's queue comments, 2026-10-14). Retry, restart and strike markers count
# only in these: in anyone else's comment they would let any commenter turn a first failure into
# an eviction.
QUEUE_COMMENTER = "github-actions"
HELD_RUN_POLLS = 10
HELD_RUN_POLL_S = 3
# The workflow the queue dispatches to wake itself (spec 4.6, D3): GitHub starts a dispatch only
# for a workflow file that exists on main, and merge-queue.yml does not yet.
KICK_WORKFLOW = "security-required.yml"
# How a re-run GitHub answered with an error is read (`_rerun_error`): `final` (a 403 saying the
# run is over a month old: no re-run will ever be accepted), `refused` (409/422: refused now, a
# strike, since a racing re-run that is not visible yet can cause one), `transient` (`_transient`,
# re-raised so the next event retries; never disarms), or `unknown` (re-raised with a warning per
# gate, and counted as final once that warning is STRIKE_GAP old).
RERUN_REFUSALS = ("HTTP 409", "HTTP 422")
RERUN_REFUSAL_403_TEXT = "month ago"
# How long a kick waits for the run that sent it to complete before deciding. The kick's job
# timeout is 15 minutes (as merge-queue.yml's); the wait plus one front's worst case (an UNKNOWN
# settle, a rebase's polls, re-runs of up to six gates) must fit.
# ponytail: a run still live after this is a GitHub fault; the kick decides anyway and the next event recovers.
WAIT_RUN_TRIES = 30
WAIT_RUN_DELAY_S = 6.0
# A queue run takes on no further front, and starts no young-head wait that would end, past this;
# it wakes a fresh run instead.
RUN_BUDGET = timedelta(minutes=4)
# A lasting failure to get a head going (a rebase that did not take effect, no run to start, an
# unclassified re-run error, a refused approval) is warned about once per head and evicts only
# once the warning is at least this old (spec section 7, errors).
STRIKE_GAP = timedelta(minutes=10)
# GitHub refuses to let an Actions token create or update a file under .github/workflows
# ("refusing to allow a GitHub App to create or update workflow ... without `workflows`
# permission"), and GITHUB_TOKEN can never hold that permission.
WORKFLOW_REFUSAL = ("workflow", "permission")

MISSING, RUNNING, PASSED, FAILED, FAILED_TWICE = "missing", "running", "passed", "failed", "failed twice"


@dataclass(frozen=True)
class Context:
    """One required status on `dev` and the workflow that reports it (spec 4.1).

    Attributes:
        name: The context, as branch protection names it.
        workflow: The workflow file that reports it.
        kind: `check` for a gate's check run, which counts only from a `pull_request` run on the
            PR head; `status` for the license commit status, posted by the license gate the
            queue dispatches on `dev`.
    """

    name: str
    workflow: str
    kind: str = "check"


CONTEXTS = (
    Context("backend-required", "backend-required.yml"),
    Context("security-required", "security-required.yml"),
    Context("coverage-required", "coverage-required.yml"),
    Context("frontend-required", "frontend-required.yml"),
    Context("e2e-required", "e2e-required.yml"),
    Context("container-build-check", "container-build-check.yml"),
    Context("frontend-license-policy/trusted/dev", "frontend-license-gate.yml", kind="status"),
)
ALL_CONTEXTS = tuple(c.name for c in CONTEXTS)
BY_NAME = {c.name: c for c in CONTEXTS}
GATE_WORKFLOWS = frozenset(c.workflow for c in CONTEXTS if c.kind == "check")
LICENSE = next(c for c in CONTEXTS if c.kind == "status")


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
        head_ref: The head branch name.
        same_repo: Whether the head branch lives in this repository (not a fork).
        armed_at: When auto-merge was enabled, or None if it is not armed.
        is_draft: Whether the PR is a draft.
        merge_state: GitHub's `mergeStateStatus` (`BEHIND`, `CLEAN`, `DIRTY`, ...).
        head_committed_at: The head commit's date.
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
    checks: tuple[CheckRun, ...] = ()
    human_author: bool = True


@dataclass(frozen=True)
class Action:
    """The single decision for the front PR: wait, rebase, start, retry or evict.

    `slug` names an eviction's cause in its comment marker (`evict-<slug>`), so a re-armed PR
    evicted again on the same head for a different reason is still told why.

    Attributes:
        kind: `wait`, `rebase`, `start`, `retry` or `evict`.
        reason: A human-readable cause, logged and quoted in comments.
        links: Run URLs quoted in the comment (the failed runs).
        slug: An eviction's cause, used in its comment marker; `young` on the young-head wait.
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
        if LICENSE.name in missing:
            # The license gate runs on dev and has no queue-tick: with its status missing while the
            # gates run, nothing would start it once they finish, and the line would wait for an
            # unrelated event. On a young head its run may not be visible yet: wait that out.
            if abs(now - pr.head_committed_at) <= YOUNG_HEAD:
                return Action("wait", f"running: {', '.join(running)}; the license status may not be visible yet",
                              slug="young", contexts=running + (LICENSE.name,))
            return Action("start", f"no license status while {', '.join(running)} run", contexts=(LICENSE.name,))
        return Action("wait", f"running: {', '.join(running)}", contexts=running)
    if missing:
        # The commit date comes from the author's clock: one a little ahead counts as young too, one
        # further ahead does not, or the line would wait for that clock.
        if abs(now - pr.head_committed_at) <= YOUNG_HEAD:
            return Action("wait", "head is under 3 minutes old; its own runs may not be visible yet",
                          slug="young", contexts=missing)
        return Action("start", f"no counted run on the up-to-date head: {', '.join(missing)}", contexts=missing)
    # BLOCKED with everything green is mergeStateStatus lagging the checks. Review threads are
    # not consulted: dev's rules do not require conversations to be resolved, so an unresolved
    # thread never blocks the merge and must never evict a green PR.
    if state in ("CLEAN", "UNSTABLE", "HAS_HOOKS", "BLOCKED"):
        last_green = max(_finished(runs[name], now)[-1].completed_at or now for name in ALL_CONTEXTS)
        if now - last_green > STUCK_GREEN:
            return Action(
                "evict", "green for over 15 minutes but auto-merge did not fire; re-arm to retry", slug="stuck",
            )
        return Action("wait", "green; auto-merge should fire", slug="green")
    return Action("wait", f"merge state {state}")


class GhError(RuntimeError):
    """A gh CLI call failed, or a list read was longer than the queue will page through."""


class CommentRefused(GhError):
    """GitHub will never take the queue's comment on this PR (its conversation is locked)."""


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


PR_FIELDS = """
  number id isDraft headRefOid headRefName mergeStateStatus
  author { __typename }
  headRepository { nameWithOwner }
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
    " pullRequest(number: $number) { comments(last: 100) { nodes { databaseId body createdAt"
    " author { login __typename } } } } } }"
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
    """Read one PR's current state.

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


def counted(checks: tuple[CheckRun, ...], runs: list[dict]) -> tuple[CheckRun, ...]:
    """Drop gate check runs that came from a run other than `pull_request` (a `workflow_dispatch`).

    Such a check is attached to the head commit but absent from the PR's status rollup, so it
    never satisfies the context (spec S5). A check whose run is not listed is kept.

    Args:
        checks: The gate contexts' check runs on the head, from `read_checks`.
        runs: Every workflow run on the head, from `runs_on`.

    Returns:
        The checks that count.
    """
    not_counted = {r.get("check_suite_id") for r in runs
                   if _workflow_name(r) in GATE_WORKFLOWS and r.get("event") != "pull_request"}
    return tuple(c for c in checks if c.suite_id is None or c.suite_id not in not_counted)


# A completed run with one of these conclusions is no result: a cancelled run is restarted, a
# held one approved. Any other completed run that reported no check is a failure of its gate.
NO_RESULT_CONCLUSIONS = frozenset({"cancelled", "action_required"})


def required_run_stand_ins(runs: list[dict], checks: tuple[CheckRun, ...]) -> tuple[CheckRun, ...]:
    """Gate `pull_request` runs on a head that have not reported their check, as stand-ins.

    - A LIVE run stands in as an in-flight check: each gate is a needs-gated job with no check run
      until the jobs before it finish, so an in-flight front PR would otherwise look like it has
      no run.
    - A COMPLETED run with no check run for its context in its check suite stands in as one
      failure, whatever it concluded, unless it was cancelled or is held (`NO_RESULT_CONCLUSIONS`):
      a startup failure, a timeout before the gate job, or a gate job the branch renamed or removed
      would otherwise read as "no run" and be started again on every wake. As a failure it is
      retried once and then evicted. A run whose suite DID report the check is judged by that check
      (a failed run whose queue-tick failed is not a gate failure).

    Only `pull_request` runs stand in: a dispatched gate run can never satisfy a context (spec
    S5), whoever started it, so waiting on one would only stall the line. A workflow-run
    conclusion is only ever read as a failure here, never as a merge signal. The license gate runs
    on `dev`, not on this head, so it has no stand-in.

    Args:
        runs: The workflow runs on the head, from `runs_on`.
        checks: The gate contexts' counted check runs on that head.

    Returns:
        One stand-in per live run and per completed run that reported no check.
    """
    by_workflow = {c.workflow: c for c in CONTEXTS if c.kind == "check"}
    reported = {(c.context, c.suite_id) for c in checks if c.suite_id is not None}
    stand_ins = []
    for run in runs:
        ctx = by_workflow.get(_workflow_name(run))
        if ctx is None or run.get("event") != "pull_request":
            continue
        url = run.get("html_url", "")
        if run.get("status") in LIVE_RUN_STATUSES:
            stand_ins.append(CheckRun(ctx.name, run["status"], None, None, url))
        elif (run.get("status") == "completed" and run.get("conclusion") not in NO_RESULT_CONCLUSIONS
              and (ctx.name, run.get("check_suite_id")) not in reported):
            stand_ins.append(CheckRun(ctx.name, "completed", "failure", _ts(run.get("updated_at")), url))
    return tuple(stand_ins)


def read_contexts(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read all seven required contexts' runs on a commit (spec 4.1).

    Workflow runs are read before check runs: a run seen as completed already has its check
    runs, so "completed without reporting its check" is never an artefact of the read order.

    Args:
        gh: The GitHub client.
        sha: The head commit.

    Returns:
        Counted check runs, workflow-run stand-ins and commit statuses, each tagged with its context.

    Raises:
        GhError: A call failed.
    """
    runs = runs_on(gh, sha)
    checks = counted(read_checks(gh, sha), runs)
    return checks + required_run_stand_ins(runs, checks) + read_statuses(gh, sha)


def _best_effort(log: Callable[[str], None], what: str, fn: Callable[[], object]) -> None:
    """Run `fn`; log a GhError as `what` instead of raising, for calls the queue can do without."""
    try:
        fn()
    except GhError as exc:
        log(f"  best-effort {what} failed: {exc}")


def _transient(exc: GhError) -> bool:
    """Whether an error is GitHub's or the network's, saying nothing about the PR.

    Matches what gh prints: a REST error `gh: <message> (HTTP 502)`, a non-JSON one `gh: HTTP 502`,
    a GraphQL timeout `gh: Something went wrong while executing your query...`, rate limits and
    abuse detection, and a network failure, where gh prints no `gh: ` server message at all
    (e.g. `Post "https://api.github.com/graphql": EOF`).

    Args:
        exc: The error a call raised.

    Returns:
        True for an outage or throttling error; False for GitHub's answer about this PR.
    """
    message = str(exc).split(" failed: ", 1)[-1]
    lowered = message.lower()
    # ponytail: "no `gh: ` line" = network. Should gh ever drop that prefix, every error reads as
    # transient: queue runs fail visibly and nothing is disarmed; match the new shape then.
    server_answered = any(line.startswith("gh: ") for line in message.splitlines())
    return ("HTTP 5" in message or "HTTP 429" in message or "rate limit" in lowered or "abuse detection" in lowered
            or "something went wrong while executing your query" in lowered or not server_answered)


def _queue_comments(gh: GhApi, number: int) -> list[dict]:
    """The PR's last 100 comments that the queue itself posted (see QUEUE_COMMENTER).

    Raises:
        GhError: The read failed.
    """
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    nodes = data["data"]["repository"]["pullRequest"]["comments"]["nodes"]
    return [n for n in nodes
            if (n.get("author") or {}).get("login") == QUEUE_COMMENTER and (n.get("author") or {}).get("__typename") == "Bot"]


def _marker(gh: GhApi, number: int, kind: str, sha: str) -> dict | None:
    """The queue's comment of this kind for this head, if any.

    Raises:
        GhError: The read failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    return next((n for n in _queue_comments(gh, number) if marker in (n.get("body") or "")), None)


def has_comment(gh: GhApi, number: int, kind: str, sha: str) -> bool:
    """Whether the queue already posted a comment of this kind for this head.

    Raises:
        GhError: The read failed.
    """
    return _marker(gh, number, kind, sha) is not None


def marker_age(gh: GhApi, number: int, kind: str, sha: str, now: datetime) -> timedelta | None:
    """How long ago the queue posted a comment of this kind for this head.

    Returns:
        Its age, `timedelta.max` if it has no timestamp, or None if there is no such comment.

    Raises:
        GhError: The read failed.
    """
    node = _marker(gh, number, kind, sha)
    if node is None:
        return None
    posted = _ts(node.get("createdAt"))
    return timedelta.max if posted is None else now - posted


def _attempt_note(run: dict) -> str:
    """The hidden note a retry or restart comment carries: the run attempt the queue re-ran."""
    return f"<!-- merge-queue-attempt:{int(run.get('run_attempt') or 1)} -->"


def marker_attempt(gh: GhApi, number: int, kind: str, sha: str) -> int | None:
    """The run attempt the queue recorded in its comment of this kind for this head.

    Returns:
        The attempt (0 when the comment carries none), or None if there is no such comment.

    Raises:
        GhError: The read failed.
    """
    node = _marker(gh, number, kind, sha)
    if node is None:
        return None
    body = node.get("body") or ""
    start = body.find("<!-- merge-queue-attempt:")
    if start < 0:
        return 0
    digits = body[start + len("<!-- merge-queue-attempt:"):].split(" ", 1)[0]
    return int(digits) if digits.isascii() and digits.isdigit() else 0


def comment_once(gh: GhApi, number: int, kind: str, sha: str, body: str,
                 log: Callable[[str], None] = print) -> bool:
    """Post a comment unless one of this kind already exists for this head.

    Returns:
        True if it posted, False if the marker was already there (or a racing queue run posted
        the same comment first, in which case this run's copy is removed).

    Raises:
        CommentRefused: The PR's conversation is locked.
        GhError: A call failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    if any(marker in (n.get("body") or "") for n in _queue_comments(gh, number)):
        return False
    try:
        posted = gh.rest("POST", f"repos/{REPO}/issues/{number}/comments", {"body": f"{marker}\n{body}"})
    except GhError as exc:
        if "is locked" in str(exc).lower() and not _transient(exc):
            raise CommentRefused(str(exc)) from exc
        raise
    # Two racing queue runs can both pass the check above and both post. Each then looks again and
    # the later one removes its own copy, so one comment remains.
    own_id = posted.get("id") if isinstance(posted, dict) else None
    if own_id is None:
        return True
    try:
        ids = [n["databaseId"] for n in _queue_comments(gh, number)
               if marker in (n.get("body") or "") and n.get("databaseId") is not None]
        if ids and min(ids) < own_id:
            gh.rest("DELETE", f"repos/{REPO}/issues/comments/{own_id}")
            return False
    except GhError as exc:
        # The comment is posted; tidying up must never stop what the caller does next. At worst two copies stay.
        log(f"  best-effort comment de-duplication on #{number} failed: {exc}")
    return True


def _comment_if_allowed(gh: GhApi, number: int, kind: str, sha: str, log: Callable[[str], None], body: str) -> None:
    try:
        comment_once(gh, number, kind, sha, body, log)
    except CommentRefused as exc:
        log(f"  cannot comment on #{number} ({exc}); skipped")


def _disarm(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    try:
        gh.graphql(DISARM_MUTATION, id=pr.node_id)
    except GhError as exc:
        # Two racing runs can evict the same PR (a merge fires both push:dev and
        # pull_request:closed), and the second disarm hits an already-disarmed PR. Any other
        # failure must not leave the PR armed under a "removed" comment: fail the run, and the
        # next one evicts it again.
        if read_pr(gh, pr.number).armed_at is not None:
            raise
        log(f"  #{pr.number} was already disarmed ({exc})")


def _evict(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> bool:
    """Tell the PR why, then disarm it; always returns True (it left the line)."""
    # Comment first: if the job dies between the two calls the PR stays armed and the next run
    # evicts it again (the comment is deduplicated), instead of being disarmed with no reason given.
    links = "".join(f"\n- {u}" for u in action.links)
    comment_once(
        gh, pr.number, f"evict-{action.slug}", pr.head_sha,
        f"Merge queue: removed from the line ({action.reason}). Auto-merge is now off. Fix the cause, then "
        f"re-arm with `gh pr merge {pr.number} --auto --merge` to rejoin at the back.{links}",
        log,
    )
    _disarm(gh, pr, log)
    log(f"  evicted #{pr.number}: {action.reason}")
    return True


def _strike(gh: GhApi, pr: PrState, kind: str, warning: str, eviction: Action, wake_first: bool,
            log: Callable[[str], None], now: Callable[[], datetime]) -> bool:
    """Count one failure to get this head going (see STRIKE_GAP).

    Args:
        gh: The GitHub client.
        pr: The front PR.
        kind: The warning comment's kind (`rebase-failed`, `rebase-unmoved`, `no-run`, `approve-error`).
        warning: The warning's text.
        eviction: The eviction for a strike at least STRIKE_GAP after the warning.
        wake_first: Wake a queue kick after the warning, to look again soon.
        log: Receives one line per notable step.
        now: The clock.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A call failed.
    """
    age = marker_age(gh, pr.number, kind, pr.head_sha, now())
    if age is None:
        comment_once(gh, pr.number, kind, pr.head_sha, warning, log)
        if wake_first:
            _wake_best_effort(gh, log)
        return False
    if age < STRIKE_GAP:
        log(f"  #{pr.number}: {kind} again within {STRIKE_GAP}; a later event decides")
        return False
    return _evict(gh, pr, eviction, log)


def dispatch(gh: GhApi, workflow: str, ref: str, inputs: dict[str, str] | None = None) -> None:
    """Start a workflow_dispatch run. Only `dispatch_license` and `wake` call this (guard-tested).

    Raises:
        GhError: GitHub refused the dispatch.
    """
    fields = {"ref": ref}
    for key, value in (inputs or {}).items():
        fields[f"inputs[{key}]"] = value
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{workflow}/dispatches", fields)


def dispatch_license(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Start dev's copy of the license gate for the PR (spec 4.4): the one context the queue dispatches.

    Any error re-raises: the dispatch runs on dev, so a refusal is never the PR's fault.
    """
    dispatch(gh, LICENSE.workflow, BASE, {"pr": str(pr.number)})
    log(f"  dispatched {LICENSE.workflow} on {BASE} for #{pr.number}")


def wake(gh: GhApi, wait_run: int | None = None) -> None:
    """Start a queue kick: `security-required.yml` on dev with `queue_kick=true` (spec 4.6, D3).

    Its gate jobs skip and only its `queue-kick` job runs, which waits for `wait_run` (if given)
    and then runs the queue. Up to six gate runs can each send one. GitHub keeps one kick running
    and one pending, and a newer pending kick replaces an older one, losing its `wait_run`. That
    costs nothing: a gate run completes seconds after its tick sends the kick, so it is done by
    the time any later kick runs, and every kick decides from fresh state.

    Raises:
        GhError: The dispatch failed.
    """
    inputs = {"queue_kick": "true"}
    if wait_run is not None:
        inputs["wait_run"] = str(wait_run)
    dispatch(gh, KICK_WORKFLOW, BASE, inputs)


def _wake_best_effort(gh: GhApi, log: Callable[[str], None]) -> None:
    _best_effort(log, "wake a queue kick", lambda: wake(gh))


def wait_for_run(gh: GhApi, run_id: int, sleep: Callable[[float], None], log: Callable[[str], None]) -> None:
    """Wait, bounded, for a workflow run to complete; a run still live after the bound is left."""
    error: GhError | None = None
    for _ in range(WAIT_RUN_TRIES):
        try:
            if gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}").get("status") not in LIVE_RUN_STATUSES:
                return
            error = None
        except GhError as exc:  # a failed read is retried within the bound, never fatal
            error = exc
        sleep(WAIT_RUN_DELAY_S)
    seen = f"last read failed: {error}" if error else "still live"
    log(f"run {run_id} not seen complete after {WAIT_RUN_TRIES * WAIT_RUN_DELAY_S:.0f}s ({seen}); deciding anyway")


def _is_held(run: dict) -> bool:
    """Whether GitHub holds this run for approval (it lists as completed, `action_required`)."""
    return "action_required" in (run.get("status"), run.get("conclusion"))


def _queue_held(run: dict) -> bool:
    """Whether this is a held run the queue may approve (spec section 7, "What may be approved").

    A gate workflow's pull_request run that the queue's own token caused, on this repository's
    branch. Never a non-gate workflow (D1: `ci.yml` alone is 200+ jobs), never a run from another
    repository (GitHub's approval gate for outside contributors), never one another actor caused,
    such as an agent's push waiting for a person.
    """
    return (_is_held(run) and run.get("event") == "pull_request"
            and _workflow_name(run) in GATE_WORKFLOWS
            and (run.get("triggering_actor") or {}).get("login") == QUEUE_ACTOR
            and (run.get("head_repository") or {}).get("full_name") == REPO)


def approve_held_runs(gh: GhApi, sha: str, log: Callable[[str], None]) -> set[str]:
    """Approve the gate runs GitHub holds on this head because the queue's token caused them.

    Best-effort per run: one refused approval must not stop the others.

    Returns:
        The workflow names whose held runs were approved.
    """
    try:
        held = runs_on(gh, sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list held runs failed: {exc}")
        return set()
    approved: set[str] = set()
    for run in held:
        if not _queue_held(run):
            continue
        try:
            gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
        except GhError as exc:
            log(f"  best-effort approve run {run['id']} failed: {exc}")
            continue
        approved.add(_workflow_name(run))
        log(f"  approved held run {run['id']} ({_workflow_name(run)})")
    return approved


def _approve_strictly(gh: GhApi, pr: PrState, run: dict, log: Callable[[str], None],
                      now: Callable[[], datetime]) -> bool:
    """Approve one held gate run a start or retry depends on.

    Not best-effort: a transient error re-raises (the run fails; nobody is disarmed). Any other
    refusal is re-read first: a run no longer held was approved by a racing queue run. One still
    held is an `approve-error` strike.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A transient error, or a failed re-read.
    """
    try:
        gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
    except GhError as exc:
        if _transient(exc):
            raise
        if not _is_held(gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}")):
            log(f"  run {run['id']} is no longer held; a racing queue run approved it")
            return False
        error = str(exc)[:200]
        return _strike(
            gh, pr, "approve-error",
            f"Merge queue: GitHub refused to approve one of this PR's gate runs ({error}). It will try again, and "
            "remove it from the line if GitHub still refuses 10 minutes from now.",
            Action("evict", f"GitHub keeps refusing to approve its gate run: {error}", slug="approve"), False, log, now,
        )
    log(f"  approved held run {run['id']} ({_workflow_name(run)})")
    return False


def _going_again(gh: GhApi, run_id: int) -> bool:
    """Whether a run is live or held again, i.e. a racing queue run has re-run it since we looked."""
    fresh = gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}")
    return fresh.get("status") in LIVE_RUN_STATUSES or _is_held(fresh)


def _rerun_error(exc: GhError) -> str:
    """Classify a re-run error as `final`, `refused`, `transient` or `unknown` (see RERUN_REFUSALS)."""
    text = str(exc)
    if "HTTP 403" in text and RERUN_REFUSAL_403_TEXT in text.lower():
        return "final"
    if any(code in text for code in RERUN_REFUSALS):
        return "refused"
    if _transient(exc):
        return "transient"
    return "unknown"


def _rerun(gh: GhApi, pr: PrState, run: dict, failed_only: bool, log: Callable[[str], None],
           sleep: Callable[[float], None], now: Callable[[], datetime], context: str = "") -> str:
    """Re-run a gate run in place, then approve the re-run if GitHub holds it.

    A re-run adds an attempt to the same check suite, so it replaces the old result for branch
    protection while `read_checks` (filter=all) still lists the old one.

    Args:
        context: The gate, which names its `rerun-error-<context>` warning.

    Returns:
        `started` (by this run, or by a racing queue run), `pending` (re-run, not live after the
        approval polls: the caller wakes a kick once it has recorded the attempt), `refused`
        (409/422: the caller strikes) or `final` (GitHub will never re-run it: the caller evicts).

    Raises:
        GhError: A transient error, or an unknown one before its warning is STRIKE_GAP old.
    """
    path = f"repos/{REPO}/actions/runs/{run['id']}/{'rerun-failed-jobs' if failed_only else 'rerun'}"
    try:
        gh.rest("POST", path)
    except GhError as exc:
        kind = _rerun_error(exc)
        if kind == "transient":
            raise
        # Whatever code GitHub refused with, a racing queue run may have re-run it first.
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was re-run by a racing queue run; standing down")
            return "started"
        if kind == "unknown":
            warning = f"rerun-error-{context}" if context else "rerun-error"
            age = marker_age(gh, pr.number, warning, pr.head_sha, now())
            if age is None:
                comment_once(
                    gh, pr.number, warning, pr.head_sha,
                    f"Merge queue: re-running a required gate failed ({str(exc)[:200]}); will try again, and "
                    "remove it from the line if it still fails 10 minutes from now.",
                    log,
                )
                raise
            if age < STRIKE_GAP:
                raise
            kind = "final"
        log(f"  #{pr.number}: re-run of run {run['id']} refused: {exc}")
        return kind
    # The re-run's actor is the queue's token, so GitHub may hold it for approval again.
    for _ in range(HELD_RUN_POLLS):
        sleep(HELD_RUN_POLL_S)
        approve_held_runs(gh, pr.head_sha, log)
        if gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}").get("status") in LIVE_RUN_STATUSES:
            log(f"  #{pr.number}: re-ran run {run['id']}")
            return "started"
    log(f"  #{pr.number}: re-ran run {run['id']}; not live yet")
    return "pending"


def _wake_if_moved(gh: GhApi, pr: PrState, sleep: Callable[[float], None]) -> bool:
    """Wake a kick if the PR's head moved; return whether it did.

    Raises:
        GhError: The re-read or the wake failed. The caller then treats the head as unmoved, so a
            wake that never went out is not mistaken for a stand-down.
    """
    sleep(REBASE_POLL_S)  # an accepted rebase moves the branch about 1 s after the mutation returns
    if read_pr(gh, pr.number).head_sha == pr.head_sha:
        return False
    wake(gh)
    return True


def _approve_gates(gh: GhApi, sha: str, log: Callable[[str], None], sleep: Callable[[float], None]) -> set[str]:
    """Approve the six held gate runs on a fresh head, waiting (bounded) for them to appear."""
    approved: set[str] = set()
    for attempt in range(HELD_RUN_POLLS):
        if attempt:
            sleep(HELD_RUN_POLL_S)
        approved |= approve_held_runs(gh, sha, log)
        if approved >= GATE_WORKFLOWS:
            break
    return approved


def _rebase(gh: GhApi, pr: PrState, log: Callable[[str], None], sleep: Callable[[float], None],
            now: Callable[[], datetime]) -> bool:
    """Rebase the front PR onto dev with its head pinned, then get its seven contexts going (spec section 6).

    Returns:
        True if the PR was evicted, False otherwise.

    Raises:
        GhError: A transient error with the head unmoved, or the license dispatch failed.
    """
    try:
        gh.graphql(REBASE_MUTATION, id=pr.node_id, oid=pr.head_sha)
    except GhError as exc:
        if _transient(exc):
            # Each merge wakes two queue runs, and the one that loses the rebase race is answered
            # with a generic error; a lost response looks the same. With the head moved, nothing
            # failed: stand down. With it unmoved it is GitHub's error: fail the run so the next
            # event retries; it never counts toward an eviction.
            try:
                moved = _wake_if_moved(gh, pr, sleep)
            except GhError as read_exc:
                log(f"  re-read or wake after a failed rebase failed: {read_exc}")
                moved = False
            if moved:
                log(f"  rebase of #{pr.number} was answered with an error, but its head moved; standing down")
                return False
            raise
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha == pr.head_sha and fresh.merge_state != "DIRTY":
            # A racing run's rebase may have been accepted with its ref update still landing.
            sleep(REBASE_POLL_S)
            fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            log(f"  rebase skipped: head moved to {fresh.head_sha[:10]}")
            return False
        if fresh.merge_state == "DIRTY":
            return _evict(gh, fresh, Action("evict", "conflicts with dev (rebase refused)", slug="conflict"), log)
        if all(word in str(exc).lower() for word in WORKFLOW_REFUSAL):
            reason = (
                "dev changed workflow files since this branch was cut, and the queue's token is not allowed to "
                "rebase across them; rebase onto dev by hand (`git fetch origin dev && git rebase origin/dev`, "
                "then force-push with lease)"
            )
            return _evict(gh, fresh, Action("evict", reason, slug="workflows"), log)
        error = str(exc)[:200]
        log(f"  rebase failed: {exc}")
        return _strike(
            gh, fresh, "rebase-failed",
            f"Merge queue: rebasing onto dev failed ({error}); will retry, and remove it from the line if "
            "rebasing still fails 10 minutes from now.",
            Action("evict", f"rebase onto dev keeps failing: {error}", slug="rebase"), False, log, now,
        )
    # The mutation returns the PRE-rebase headRefOid and the branch moves about 1 s later.
    rebased = None
    for _ in range(REBASE_POLLS):
        sleep(REBASE_POLL_S)
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            rebased = fresh
            break
    if rebased is None:
        log(f"  rebase of #{pr.number} accepted but the head never moved")
        # If the branch moves later its held runs need a queue run to approve them: the warning
        # wakes a kick. A later unmoved rebase evicts, or every event would rebase again.
        return _strike(
            gh, pr, "rebase-unmoved",
            "Merge queue: the rebase onto dev was accepted, but the branch has not moved yet; a queue run will "
            "look again shortly, and remove it from the line if the branch still does not move 10 minutes from now.",
            Action("evict", "rebase onto dev was accepted but the branch never moved, twice at least 10 minutes apart",
                   slug="rebase-unmoved"), True, log, now,
        )
    try:
        old_runs = runs_on(gh, pr.head_sha)
        # Spec 4.4: the license status first, so each gate's await_license finds it (when
        # LICENSE_FIRST_CI_ENABLED is true). Then the gates: approving their held runs is what makes
        # their checks count (S4). Then the other held runs go (D1), then the old head's live runs.
        dispatch_license(gh, rebased, log)
        approved = _approve_gates(gh, rebased.head_sha, log, sleep)
    except GhError:
        # The branch has moved and its gate runs are held: nothing else would wake the queue to
        # approve them. The kick finds an up-to-date head and approves, so this cannot chain.
        _wake_best_effort(gh, log)
        raise
    cleanup_approval_runs(gh, rebased, log)
    # GitHub may list a non-gate held run a moment after the gates' (D1): look once more.
    sleep(HELD_RUN_POLL_S)
    cleanup_approval_runs(gh, rebased, log)
    for run in old_runs:
        if run.get("status") not in LIVE_RUN_STATUSES:
            continue
        if run.get("id") == _own_run_id() or _workflow_name(run) == QUEUE_WORKFLOW:
            continue
        _best_effort(log, f"cancel run {run['id']}",
                     lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
    total = len(GATE_WORKFLOWS)
    if len(approved) < total:
        # The rest are still held, and a held run never runs its own queue-tick: wake a kick.
        _wake_best_effort(gh, log)
        started = f"approved {len(approved)} of its {total} required gates; a woken queue run approves the rest"
    else:
        started = f"approved its {total} required gates"
    _comment_if_allowed(
        gh, pr.number, "rebased", rebased.head_sha, log,
        f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{rebased.head_sha[:10]}`); dispatched the "
        f"license gate and {started}.",
    )
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {rebased.head_sha[:10]}")
    return False


def cleanup_approval_runs(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Delete the non-gate runs GitHub holds on the front head because the queue's token caused them (D1).

    A queue rebase makes GitHub hold a run for every pull_request workflow; only the six gates are
    approved. Deleting the others removes the "Approve and run" button that would start `ci.yml`'s
    200+ jobs on a click. A gate's held run is never deleted here. Best-effort end to end.
    """
    try:
        runs = runs_on(gh, pr.head_sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list approval-pending runs failed: {exc}")
        return
    for run in runs:
        if (run.get("triggering_actor") or {}).get("login") != QUEUE_ACTOR or _workflow_name(run) in GATE_WORKFLOWS:
            continue
        _best_effort(log, f"delete approval-pending run {run['id']}",
                     lambda rid=run["id"]: gh.rest("DELETE", f"repos/{REPO}/actions/runs/{rid}"))


def _gate_runs(gh: GhApi, sha: str) -> list[dict]:
    """The gate workflows' pull_request runs on this head, newest first."""
    runs = [r for r in runs_on(gh, sha) if _workflow_name(r) in GATE_WORKFLOWS and r.get("event") == "pull_request"]
    return sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True)


def _after_rerun(gh: GhApi, pr: PrState, ctx: Context, run: dict, attempt: int,
                 went_wrong: Callable[[dict, str], bool], eviction: Action, log: Callable[[str], None],
                 now: Callable[[], datetime]) -> bool:
    """Decide on a gate the queue already re-ran once on this head (its retry or restart marker exists).

    The decision being acted on may be older than that re-run: a racing queue run made it, or
    GitHub has not shown it yet. So the run is read fresh, and the PR is evicted only when the
    queue's re-run has run (a higher `run_attempt` than the marker recorded) and still went wrong.
    A run that is live or held, or whose gate is fine since the decision, is left alone. A re-run
    not visible yet is a `rerun-unseen-<context>` strike: one warning and a kick, an eviction only
    if it still has not run 10 minutes later.

    Args:
        gh: The GitHub client.
        pr: The front PR.
        ctx: The gate.
        run: The gate's run, as listed.
        attempt: The attempt the marker recorded (0 if it recorded none).
        went_wrong: Given the fresh run and the gate's fresh state, whether the re-run failed again.
        eviction: The eviction when it did.
        log: Receives one line per notable step.
        now: The clock.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A read failed.
    """
    fresh = gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}")
    if fresh.get("status") in LIVE_RUN_STATUSES or _is_held(fresh):
        log(f"  #{pr.number}: {ctx.name} run {run['id']} is going again; standing down")
        return False
    if int(fresh.get("run_attempt") or 1) <= attempt:
        return _strike(
            gh, pr, f"rerun-unseen-{ctx.name}",
            f"Merge queue: the queue's re-run of {ctx.name} on `{pr.head_sha[:10]}` has not started yet. A queue run "
            "will look again, and remove it from the line if it still has not run 10 minutes from now.",
            Action("evict", f"the queue's re-run of {ctx.name} never started", eviction.links, "rerun"), True, log, now,
        )
    state = context_state([c for c in read_contexts(gh, pr.head_sha) if c.context == ctx.name], now())
    if not went_wrong(fresh, state):
        log(f"  #{pr.number}: {ctx.name} is {state} since the decision; standing down")
        return False
    return _evict(gh, pr, eviction, log)


def _refused_rerun(gh: GhApi, pr: PrState, ctx: Context, links: tuple[str, ...], log: Callable[[str], None],
                   now: Callable[[], datetime]) -> bool:
    """A re-run GitHub refused with 409/422: a strike, since a racing re-run can cause one (spec section 7)."""
    return _strike(
        gh, pr, f"rerun-refused-{ctx.name}",
        f"Merge queue: GitHub refused to re-run {ctx.name} on `{pr.head_sha[:10]}`. It will try again, and remove it "
        "from the line if GitHub still refuses 10 minutes from now.",
        Action("evict", f"GitHub keeps refusing to re-run {ctx.name}", links, "rerun"), False, log, now,
    )


def _restart_cancelled(gh: GhApi, pr: PrState, ctx: Context, run: dict, links: tuple[str, ...],
                       log: Callable[[str], None], sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Re-run a cancelled gate run in full, once per gate per head (the `restart-<context>` marker).

    A cancelled run says nothing about the PR (a timeout, a person, a superseded run), so it is
    restarted rather than counted as a failure. A second cancellation on the same head evicts:
    a gate that keeps being cancelled, such as one that hits its job timeout, would otherwise be
    restarted on every wake and hold the line forever.

    Returns:
        True if the PR was evicted.
    """
    kind = f"restart-{ctx.name}"
    attempt = marker_attempt(gh, pr.number, kind, pr.head_sha)
    if attempt is not None:
        return _after_rerun(
            gh, pr, ctx, run, attempt,
            lambda fresh, state: fresh.get("conclusion") == "cancelled" or state == MISSING,
            Action("evict", f"{ctx.name} was cancelled again after the queue restarted it", links, "rerun"), log, now,
        )
    note = _attempt_note(run)  # the attempt being re-run, read before the re-run changes it
    outcome = _rerun(gh, pr, run, False, log, sleep, now, ctx.name)
    if outcome == "final":
        return _evict(gh, pr, Action("evict", f"GitHub will not restart the cancelled {ctx.name} run; push a "
                                     "commit, or close and reopen the PR", links, "no-run"), log)
    if outcome == "refused":
        return _refused_rerun(gh, pr, ctx, links, log, now)
    comment_once(gh, pr.number, kind, pr.head_sha,
                 f"Merge queue: {ctx.name}'s run on `{pr.head_sha[:10]}` was cancelled; restarting it once."
                 f"\n{note}", log)
    if outcome == "pending":
        _wake_best_effort(gh, log)
    return False


def _start(gh: GhApi, pr: PrState, names: tuple[str, ...], log: Callable[[str], None],
           sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Get the named missing contexts going on an up-to-date head (spec 4.1, start).

    The license is dispatched on dev. A gate's held run is approved; with none, its newest completed
    pull_request run (cancelled, or whose gate job was cancelled) is restarted once
    (`_restart_cancelled`). A run that finished without reporting its check is a failure stand-in,
    so its context is failed, not missing, and goes through `_retry_gate`. A dispatch would not
    count (S5), so a gate with no run at all gets a `no-run` strike. It looks again for up to
    HELD_RUN_POLLS x HELD_RUN_POLL_S first: a head's age is its commit date, not its push, so its
    runs may not be listed yet.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A call failed for a reason other than a refusal.
    """
    if any(BY_NAME[n].kind == "status" for n in names):
        dispatch_license(gh, pr, log)
    pending = {BY_NAME[n].workflow: n for n in names if BY_NAME[n].kind == "check"}
    for attempt in range(HELD_RUN_POLLS):
        if not pending:
            return False
        if attempt:
            sleep(HELD_RUN_POLL_S)
        for run in runs_on(gh, pr.head_sha, status="action_required"):
            if _workflow_name(run) in pending and _queue_held(run):
                if _approve_strictly(gh, pr, run, log, now):
                    return True
                pending.pop(_workflow_name(run), None)
        runs = _gate_runs(gh, pr.head_sha)
        for workflow, name in list(pending.items()):
            mine = [r for r in runs if _workflow_name(r) == workflow]
            if any(r.get("status") in LIVE_RUN_STATUSES for r in mine):
                log(f"  #{pr.number}: a {name} run appeared on {pr.head_sha[:10]}; standing down for it")
                del pending[workflow]
                continue
            # Restart only a run that was cancelled, or whose gate job was: any other completed run
            # without the check is a failure stand-in, and a run that passed since the decision must
            # not be restarted. (A held run is neither: it has no conclusion and no checks yet.)
            cancelled_suites = {c.suite_id for c in pr.checks
                                if c.context == name and c.conclusion == "cancelled" and c.suite_id is not None}
            redo = next((r for r in mine if r.get("status") == "completed"
                         and (r.get("conclusion") == "cancelled" or r.get("check_suite_id") in cancelled_suites)), None)
            if redo is None:
                continue
            if _restart_cancelled(gh, pr, BY_NAME[name], redo, (), log, sleep, now):
                return True
            del pending[workflow]
    if not pending:
        return False
    which = ", ".join(sorted(pending.values()))
    return _strike(
        gh, pr, "no-run",
        f"Merge queue: found no run of {which} on this head that the queue can start. It will look again, and "
        "remove it from the line if there is still none 10 minutes from now. A push, or closing and reopening the "
        "PR, starts its CI.",
        Action("evict", f"no run of {which} on this head that the queue can start; push a commit, or close and "
               "reopen the PR", slug="no-run"), False, log, now,
    )


def _retry_gate(gh: GhApi, pr: PrState, ctx: Context, links: tuple[str, ...], log: Callable[[str], None],
                sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Re-run the run whose gate failed once, in its own check suite.

    The failed run's own queue-tick sends a kick that waits for the run to complete, because
    GitHub only re-runs a completed run; a run still live is left for that kick. A run held again
    (its re-run's approval failed) is approved. One retry per gate per head: a run that failed
    without reporting its check keeps its run id when re-run, so its failures never add up to two;
    the `retry-<context>` marker caps it instead. A retry attempt that ended cancelled is restarted
    once (`_restart_cancelled`), not counted.

    Returns:
        True if the PR was evicted.
    """
    failed = sorted((c for c in pr.checks if c.context == ctx.name and c.status == "completed"
                     and c.conclusion not in PASSING and c.conclusion != "cancelled"),
                    key=lambda c: c.completed_at or datetime.min.replace(tzinfo=timezone.utc))
    suite = failed[-1].suite_id if failed else None
    runs = [r for r in _gate_runs(gh, pr.head_sha) if _workflow_name(r) == ctx.workflow]
    run = next((r for r in runs if suite is not None and r.get("check_suite_id") == suite), None)
    if run is None:  # a stand-in failure: a completed run that reported no check
        reported = {c.suite_id for c in pr.checks if c.context == ctx.name and c.suite_id is not None}
        run = next((r for r in runs if r.get("status") == "completed"
                    and r.get("conclusion") not in NO_RESULT_CONCLUSIONS
                    and r.get("check_suite_id") not in reported), None)
    if run is None:
        return _evict(gh, pr, Action("evict", f"{ctx.name} failed and its run cannot be found to re-run",
                                     links, "rerun"), log)
    if run.get("status") in LIVE_RUN_STATUSES:
        log(f"  #{pr.number}: {ctx.name} run {run['id']} is still live; its kick re-runs it")
        return False
    if _queue_held(run):
        return _approve_strictly(gh, pr, run, log, now)
    if _is_held(run):
        # Held for someone else's approval (an agent's push waiting for a person): never ours to
        # approve or re-run (spec section 7).
        return _strike(
            gh, pr, f"held-{ctx.name}",
            f"Merge queue: {ctx.name}'s run on `{pr.head_sha[:10]}` is waiting for someone else's approval, which the "
            "queue never gives. Approve it, or the PR leaves the line 10 minutes from now.",
            Action("evict", f"{ctx.name}'s run is waiting for someone else's approval", links, "held"), False, log, now,
        )
    if run.get("conclusion") == "cancelled":
        return _restart_cancelled(gh, pr, ctx, run, links, log, sleep, now)
    kind = f"retry-{ctx.name}"
    attempt = marker_attempt(gh, pr.number, kind, pr.head_sha)
    if attempt is not None:
        why = f"{ctx.name} failed again after its retry"
        if suite is None:  # the run reported no check at all
            why += " (its run ended without reporting the check; was the gate job renamed or removed?)"
        return _after_rerun(gh, pr, ctx, run, attempt, lambda fresh, state: state in (FAILED, FAILED_TWICE),
                            Action("evict", why, links, "failed-twice"), log, now)
    note = _attempt_note(run)  # the attempt being re-run, read before the re-run changes it
    outcome = _rerun(gh, pr, run, suite is not None, log, sleep, now, ctx.name)
    if outcome == "final":
        return _evict(gh, pr, Action("evict", f"{ctx.name} failed once and GitHub will not re-run it",
                                     links, "rerun"), log)
    if outcome == "refused":
        return _refused_rerun(gh, pr, ctx, links, log, now)
    listed = "".join(f"\n- {u}" for u in links)
    comment_once(gh, pr.number, kind, pr.head_sha,
                 f"Merge queue: {ctx.name} failed once on `{pr.head_sha[:10]}`; re-running it.{listed}"
                 f"\n{note}", log)
    if outcome == "pending":
        _wake_best_effort(gh, log)
    return False


def _retry(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None],
           now: Callable[[], datetime]) -> bool:
    """Retry each context that failed once: the license by a new dispatch on dev, a gate by a re-run.

    Returns:
        True if the PR was evicted.
    """
    listed = "".join(f"\n- {u}" for u in action.links)
    # The license first, as on a start (spec 4.4): with LICENSE_FIRST_CI_ENABLED a gate's
    # await_license waits for it. (Re-enabling license-first also needs a gate the license refused
    # to be re-run in full: a failed-jobs re-run keeps the old await_license verdict.)
    for name in sorted(action.contexts, key=lambda n: BY_NAME[n].kind != "status"):
        ctx = BY_NAME[name]
        if ctx.kind == "status":
            dispatch_license(gh, pr, log)
            comment_once(gh, pr.number, f"retry-{name}", pr.head_sha,
                         f"Merge queue: {name} failed once on `{pr.head_sha[:10]}`; dispatched the license gate "
                         f"on `{BASE}` again.{listed}", log)
            continue
        if _retry_gate(gh, pr, ctx, action.links, log, sleep, now):
            return True
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
    now: Callable[[], datetime],
) -> bool:
    """Perform one decided action (spec section 7).

    Returns:
        True if the PR left the line (evicted), so the caller moves on to the next front.

    Raises:
        GhError: A read or a non-best-effort call failed.
    """
    try:
        if action.kind == "rebase":
            return _rebase(gh, pr, log, sleep, now)
        if action.kind in ("start", "retry"):
            # Two queue runs can decide the same action for the same head. Once the first one's run
            # is live (or the license status pending), the other sees it here and stands down for
            # that context.
            appeared = _appeared(gh, pr, action.contexts, now())
            if appeared:
                log(f"  #{pr.number}: a run appeared since the decision for {', '.join(appeared)}; leaving those")
            targets = tuple(n for n in action.contexts if n not in appeared)
            if not targets:
                return False
            if action.kind == "start":
                return _start(gh, pr, targets, log, sleep, now)
            return _retry(gh, pr, replace(action, contexts=targets), log, sleep, now)
        if action.kind == "evict":
            return _evict(gh, pr, action, log)
        return False
    except CommentRefused as exc:
        # The queue keeps its retry and strike counts in comments; on a locked PR it can neither
        # count nor explain, so it takes the PR out of the line rather than stall the line on it.
        log(f"  cannot comment on #{pr.number} ({exc}); removing it from the line without a comment")
        _disarm(gh, pr, log)
        return True


UNQUEUED_NOTES = {
    "fork": "Merge queue: fork PRs are not queued, because approving a fork's held runs would bypass GitHub's "
            "approval gate for outside contributors' workflows. A maintainer merges this one by hand.",
    "bot": "Merge queue: PRs opened by a bot or app are not queued, because a queue-approved run would run this "
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


def _decide(gh: GhApi, pr: PrState, mode: str, now: Callable[[], datetime],
            log: Callable[[str], None]) -> tuple[PrState, Action]:
    """Read the front PR's contexts and decide (spec section 6).

    Raises:
        GhError: A read failed.
    """
    if mode == "on" and pr.merge_state not in ("BEHIND", "DIRTY"):
        # Held gate runs left by an earlier rebase or re-run: approving them first lets the
        # decision below see them as the live runs they become. A BEHIND front is about to be
        # rebased and a DIRTY one evicted. Only for a PR still armed on this head (spec section 7):
        # its author may have disarmed or pushed since the line was read.
        fresh = read_pr(gh, pr.number)
        if fresh.armed_at is not None and fresh.head_sha == pr.head_sha:
            approve_held_runs(gh, pr.head_sha, log)
    pr = replace(pr, checks=read_contexts(gh, pr.head_sha))
    action = decide_front(pr, now())
    if action.slug in ("stuck", "green"):
        # Right after a merge the next PR's merge state can still read up to date (tldw_chatbook
        # #3070, TASK-34948). Auto-merge cannot fire on a head that is behind, so a green head is
        # checked against dev before the queue waits on it or evicts it as stuck. One GET.
        compare = gh.rest("GET", f"repos/{REPO}/compare/{BASE}...{pr.head_sha}?per_page=1") or {}
        if compare.get("behind_by", 0) > 0:
            action = Action("rebase", "behind dev (its merge state had not caught up yet)", contexts=ALL_CONTEXTS)
    return pr, action


def run(
    gh: GhApi,
    mode: str | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    wait_run: int | None = None,
) -> list[tuple[int, Action]]:
    """One queue pass: decide for the front PR and, in `on` mode, act; repeat after evictions.

    Args:
        gh: The GitHub client.
        mode: The MERGE_QUEUE value; only `dry` and `on` (any case or padding) do anything.
        now: The clock (injected by tests).
        sleep: Waits between rereads (injected by tests).
        log: Receives the run's log lines.
        wait_run: A run to wait for (bounded) before deciding: set by a kick that a failed gate's
            queue-tick sent, so the failed run has completed and can be re-run.

    Returns:
        The (PR number, action) decisions taken or, in dry mode, proposed.

    Raises:
        GhError: A read or a non-best-effort call failed; the next event retries.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    started = now()
    if wait_run:
        wait_for_run(gh, wait_run, sleep, log)
    prs = read_prs(gh)
    comment_unqueued(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    left: set[int] = set()
    for index, pr in enumerate(line_of(prs)[:MAX_FRONTS_PER_RUN]):
        if index and now() - started > RUN_BUDGET:
            log(f"run budget spent before #{pr.number}; a fresh queue run continues")
            if mode == "on":
                _wake_best_effort(gh, log)
            break
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        pr, action = _decide(gh, pr, mode, now, log)
        if mode == "on" and action.slug == "young":
            # A kick woken right after a rebase whose held runs had not appeared can land before
            # GitHub lists the head's runs, and nothing may wake the queue again for this head.
            # Wait out the window once and decide again (capped at one window).
            remaining = (pr.head_committed_at + YOUNG_HEAD - now()).total_seconds()
            pause = min(max(0.0, remaining), YOUNG_HEAD.total_seconds()) + 1
            if now() - started + timedelta(seconds=pause) > RUN_BUDGET:
                log(f"#{pr.number}: no budget left to wait out its young head; a fresh queue run will")
                _wake_best_effort(gh, log)
            else:
                sleep(pause)
                fresh = settle_unknown(gh, read_pr(gh, pr.number), sleep)
                if fresh.armed_at != pr.armed_at:
                    break  # disarmed or re-armed meanwhile; that event wakes a fresh run
                same_head = fresh.head_sha == pr.head_sha
                pr, action = _decide(gh, fresh, mode, now, log)
                if action.slug == "young" and same_head:
                    # Waited a full window and it still looks young: its commit clock runs ahead.
                    action = Action("start", "no run after waiting out the young-head window", contexts=action.contexts)
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        left_line = action.kind == "evict"
        if mode == "on":
            if action.kind != "wait" and not _still_front(gh, pr, left):
                log(f"#{pr.number}: no longer the armed front PR on this head; nothing done, the next wake decides")
                break
            cleanup_approval_runs(gh, pr, log)
            left_line = apply(gh, pr, action, log, sleep, now)
        if not left_line:
            break
        left.add(pr.number)
    return decisions


def main() -> int:
    """Run one queue pass with the real gh CLI and append the decisions to the job summary.

    `WAIT_RUN` (a kick's `wait_run` input) is used only when it is all ASCII digits.

    Returns:
        The process exit code (0; a failed call raises instead).
    """
    wait = os.environ.get("WAIT_RUN", "").strip()
    decisions = run(Gh(), os.environ.get("MERGE_QUEUE"), wait_run=int(wait) if wait.isascii() and wait.isdigit() else None)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Merge queue\n\n| PR | Action | Reason |\n|---|---|---|\n")
            for number, action in decisions:
                fh.write(f"| #{number} | {action.kind} | {action.reason} |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
