---
id: TASK-13452
title: Build the in-repo merge queue for dev
status: Done
labels:
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PRs into dev race each other: dev requires seven statuses on an up-to-date head, so every merge puts every other ready PR behind and all of them restart the required gates (65-120 minutes per cycle on PR #3155, which was pushed behind twice on 2026-10-04). Build a one-at-a-time queue so only the PR at the front is rebased and tested. Port of the tldw_chatbook queue (tldw_chatbook PR #2996), adapted to this repository's seven required contexts. Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md. ADR-063.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Armed PRs into dev are processed one at a time in arming order; PRs behind the front are never rebased, dispatched or commented on
- [x] #2 After a queue rebase all seven required contexts are produced on the new head, with change detection compared against dev's tip
- [x] #3 A failed context is retried once and the PR is evicted with the reason on the second failure
- [x] #4 The queue never arms auto-merge, merges or pushes, enforced by a guard test
- [x] #5 With MERGE_QUEUE unset the queue does nothing; dry logs decisions without side effects
- [x] #6 A required gate whose change detection failed reports red instead of being skipped
- [x] #7 Existing CI contract tests stay green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Built 2026-10-04/05 as a port of the tldw_chatbook queue (tldw_chatbook PR #2996). Ships off: MERGE_QUEUE is unset, so the queue does nothing until the owner enables auto-merge and sets the variable.

What was built:
- Helper_Scripts/ci/merge_queue.py: pure line_of/decide_front over seven required contexts, read layer, action layer. Never arms auto-merge, merges or pushes (guard test).
- .github/workflows/merge-queue.yml: wakes on arm, disarm, close and pushes to dev.
- Six required workflows: optional base_sha dispatch input on four (frontend-required already had it), comparison base resolved from it, a failure-only queue-tick job, and the gates now fail instead of being skipped when change detection did not succeed.
- frontend-required.yml: the manual-dispatch diagnostic-name guard is kept and exempts only dispatches by github-actions[bot].
- security-required.yml: dependency review also runs on a queue dispatch.
- frontend-license-gate.yml: a dev-only dispatch job that resolves the PR through the API and runs the audit job's steps verbatim.
- Docs: ADR-063, CI_REQUIRED_GATES.md Merge Queue section, merge rules in AGENTS.md and CLAUDE.md.

Decisions made during the build (spec sections 4.1, 4.6, 4.8, 4.9):
- A failed context is acted on while others still run, because only a failed gate wakes the queue.
- The script dispatches its host workflow last: a tick that re-dispatches its own workflow is cancelled by the shared concurrency group.
- Non-required workflows are not re-run on the rebased head; ci.yml is never dispatched.
- Owner decision 2026-10-05: the license-first contract keeps two narrow, test-pinned allowances for the queue. merge-queue.yml is listed by name as a queue-control workflow, and the queue-tick job may hold the queue's permissions in exactly one pinned shape. Every rule still applies unchanged to all other workflows and jobs.

Verification: tldw_Server_API/tests/CI plus tests/Infrastructure/test_workflow_concurrency_policy.py, with xdist: dev 75ab2240 473 passed / 4 skipped; branch 753 passed / 5 skipped / 0 failed. Mutation checks on the queue script, the gate workflows, the license dispatch job and the contract allowances were all detected. pre-commit over the diff, actionlint 1.7.12 with shellcheck over all workflows, and Bandit on the script are clean.

Not verifiable without a real queue run on this repository (spec section 8, step 4): the rebase mutation with branch updates switched off; that a GITHUB_TOKEN dispatch reports actor github-actions[bot]; the license dispatch on dev; dependency review with explicit refs on a dispatch; queue-tick's job-level write permissions; the result a timed-out gate job reports.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
In-repo merge queue for dev built and shipped switched off. Owner steps to turn it on: enable auto-merge in repository settings, set MERGE_QUEUE=dry for a day, then on with one low-risk PR.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
