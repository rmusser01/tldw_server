---
id: TASK-13639
title: 'Merge queue: approve held gate runs instead of dispatching them'
status: In Progress
labels:
- ci
- merge-queue
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The queue as merged in #3188 cannot land a PR: it starts the six gates with workflow_dispatch, and a dispatched run's check never enters the PR's status rollup, so the PR stays BLOCKED with every gate green (probe #3220). Probe #3221 showed what works with the built-in token alone: a token rebase creates held pull_request runs, and approving a gate's held run makes its check count. Spec: Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md (owner-approved 2026-10-14, D1-D3 as recommended).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a rebase the queue dispatches the license gate on dev, approves exactly the six held gate runs, deletes the other held runs and cancels the old head's live runs
- [ ] #2 A missing gate is started by approving its held run or restarting its cancelled run (once per head), never by a dispatch; with neither, a no-run strike evicts after 10 minutes
- [ ] #3 A gate that failed once is re-run in place (failed jobs) and its held re-run approved; a second failure evicts; a run that never reported its check counts as a failure
- [ ] #4 No code path dispatches a check context; a guard test fails if one does
- [ ] #5 A failed gate's queue-tick only dispatches security-required.yml on dev with queue_kick=true and its run id; the kick waits for that run, then runs the queue, and never cancels another kick
- [ ] #6 frontend-required.yml publishes frontend-required-diagnostic on every dispatch
- [ ] #7 Transient GitHub errors fail the queue run and disarm nobody; other refusals strike before evicting
- [ ] #8 A green front PR is rebased, not waited on or evicted as stuck, when dev has commits its head lacks
- [ ] #9 Workflow contract tests, CI_REQUIRED_GATES.md, AGENTS.md, CLAUDE.md and ADR-063 match the new behaviour
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-14-merge-queue-approve-held-runs.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
