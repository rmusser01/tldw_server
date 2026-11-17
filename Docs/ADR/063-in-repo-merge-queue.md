# ADR-063: PRs merge into dev through an in-repo queue

**Status:** Proposed
**Date:** 2026-10-04
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (direction given 2026-10-04; becomes Accepted when `MERGE_QUEUE` is set to `on`)
**Related task:** TASK-13462
**Related spec/plan:** `Docs/superpowers/specs/2026-10-04-merge-queue-design.md`; `Docs/Development/CI_REQUIRED_GATES.md`

## Decision

Pull requests into `dev` merge one at a time through a queue that runs in GitHub Actions on `dev`. A PR joins the line when auto-merge is armed on it. Only the PR at the front is rebased onto `dev` and has the seven required statuses re-run; every other armed PR is left alone until it reaches the front. The queue is `Helper_Scripts/ci/merge_queue.py`, woken by `.github/workflows/merge-queue.yml` and by a failure-only `queue-tick` job in each required workflow. It is controlled by the repository variable `MERGE_QUEUE` (unset or `off`, `dry`, `on`) and ships unset.

The queue uses only the built-in `GITHUB_TOKEN`. It never enables auto-merge, never merges and never pushes.

## Context

`dev` requires seven statuses on a head that is up to date with `dev`, and no ruleset grants a bypass. Each merge therefore puts every other ready PR behind. Authors and agents then each rebase and restart the required gates, and only one of them can win the next merge.

- On PR #3155 (2026-10-04) the required gates took 65 to 120 minutes per cycle and `dev` moved twice during them, forcing three full cycles for one PR while four other ready PRs were behind in the same way.
- `CI_REQUIRED_GATES.md` already records that landing several PRs is serial (four PRs, about four hours, 2026-09-22).
- GitHub's native merge queue is not available: the repository is owned by a user account, not an organization.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| GitHub's native merge queue | Organization-owned repositories only. |
| Drop strict (up-to-date) enforcement | A PR could merge without the gates ever having run on the combination that lands. |
| A bot using a personal access token or GitHub App | Owner rule: no long-lived credential in workflows. A token-less design exists. |
| A convention ("only the oldest ready PR rebases") | Needs every session to poll and agree; nothing enforces it and it stalls when a session stops. |
| Re-run every PR workflow on the rebased head, as the tldw_chatbook queue does | `ci.yml` is over 200 jobs per PR and its dispatch form starts the macOS and Windows matrices; runner capacity is the constraint the queue exists to relieve. |

## Consequences

- A queued PR is tested once per position at the front instead of once per merge that happens anywhere.
- After a rebase the queue approves the six held `pull_request` gate runs GitHub creates for the new head and dispatches only the license gate on `dev`; a dispatched gate's check would not count (amendment below).
- Non-required workflows are not re-run on the rebased head. They ran on the author's last pushed head, and `dev`'s push workflows run after the merge.
- The queue cannot rebase a PR across a change to workflow files on `dev`: an Actions token may not write under `.github/workflows`. Such a PR is removed from the line with hand-rebase steps. This is the price of using no personal access token or App key.
- Review threads do not affect the queue, because `dev`'s rules do not require conversations to be resolved.
- `frontend-required.yml` keeps its guard: no dispatch publishes the protected check name.
- A required gate whose change-detection job failed now reports red on every event instead of being skipped. This is the one behaviour change that applies with the queue off.
- Auto-merge must be enabled in repository settings before the queue can be switched on. Agents follow mode-dependent merge rules in `AGENTS.md` and `CLAUDE.md`.
- Fork PRs and PRs opened by bots or apps are never queued.

## Follow-up

- Run the `dry` comparison and the one-PR trial in the spec's rollout section; the trial settles the platform questions that cannot be checked without a real run on this repository.
- Set this ADR to Accepted when `MERGE_QUEUE=on` is in effect for all traffic.

## Amendment (2026-10-14): approve held runs, never dispatch a gate

The queue as first merged (#3188) could not land a PR: a `workflow_dispatch` run's check never enters the PR's status
rollup (probe #3220). Probe #3221 showed that a `GITHUB_TOKEN` rebase makes GitHub hold a `pull_request` run for every
workflow, and that approving a gate's held run makes its check count. The queue now approves the six gate runs,
re-runs a failed gate in place, deletes the other held runs, and wakes itself through a `security-required.yml` kick
on `dev`. Spec: `Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md` (TASK-13639). Owner decisions
D1-D3 (2026-10-14): delete the non-gate held runs; remove the `frontend-required` bot exemption; host the kick in
`security-required.yml`.
