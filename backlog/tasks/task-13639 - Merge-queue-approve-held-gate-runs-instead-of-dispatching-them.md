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
- [x] #1 After a rebase the queue dispatches the license gate on dev, approves exactly the six held gate runs, deletes the other held runs and cancels the old head's live runs
- [x] #2 A missing gate is started by approving its held run or restarting its cancelled run (once per head), never by a dispatch; with neither, a no-run strike evicts after 10 minutes
- [x] #3 A gate that failed once is re-run in place (failed jobs) and its held re-run approved; a second failure evicts; a run that never reported its check counts as a failure
- [x] #4 No code path dispatches a check context; a guard test fails if one does
- [x] #5 A failed gate's queue-tick only dispatches security-required.yml on dev with queue_kick=true and its run id; the kick waits for that run, then runs the queue, and never cancels another kick
- [x] #6 frontend-required.yml publishes frontend-required-diagnostic on every dispatch
- [x] #7 Transient GitHub errors fail the queue run and disarm nobody; other refusals strike before evicting
- [x] #8 A green front PR is rebased, not waited on or evicted as stuck, when dev has commits its head lacks
- [x] #9 Workflow contract tests, CI_REQUIRED_GATES.md, AGENTS.md, CLAUDE.md and ADR-063 match the new behaviour
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-14-merge-queue-approve-held-runs.md.

Approach: the action layer is a port of tldw_chatbook scripts/merge_queue.py at 2716b844e1 (approve held runs, strikes, comment de-dup, transient classification, kick with wait_run, young-head wait, run budget), adapted from one required check to six gate contexts plus the license status. decide_front stays pure; _decide approves held gate runs before reading, then checks a green head against dev (compare) before waiting on it or evicting it as stuck. Caps so nothing is re-run on every wake: a completed gate run that reported no check counts as one failure (retried once, then evicted); a cancelled run, or one whose gate job was cancelled, is restarted once per gate per head (restart-<context> marker). The license goes first on start and retry. A failure after a rebase took effect wakes a kick. Workflows: each queue-tick only dispatches security-required.yml on dev with queue_kick=true (string input) and its run id; security-required.yml's queue-kick job (timeout 15) waits for that run, then runs the queue from dev; its gate jobs skip on a kick on refs/heads/dev; cancel-in-progress is false for kicks. frontend-required publishes the diagnostic name on every dispatch (D2).

Reviews: two independent reviews of the plan (the second applied it literally to a copy of the repo: 390 passed); a final whole-branch review of the code. Findings fixed before code was written: M1-M5, m1-m5, m7, n1, n4, I2, I4, I5 (see the plan's Review Focus and Accepted residuals).

Verification on the branch: queue and workflow CI tests 391 passed, 1 skipped (baseline on dev 336 passed, 1 skipped). Mutation checks, each failing its named test: a dispatch added to _start; the gate filter dropped from _queue_held; cleanup deleting gate runs; the compare guard removed; the restart cap removed; no-check runs not counted as failures; kick cancel-in-progress true; the frontend bot exemption restored. actionlint 1.7.12 clean. ruff: no new findings beyond import-order warnings already on dev.

Not verified (needs the owner's one-PR trial, spec section 7): the string queue_kick input accepted live; rerun-failed-jobs re-running the dependent queue-tick; what a timed-out gate reports; whether the dev ruleset's require_extra_approval_for_unattributed_changes asks for an approval on the queue's github-actions[bot]-committed rebase; job permissions inside approved runs.

Files: Helper_Scripts/ci/merge_queue.py; .github/workflows/{security,backend,coverage,e2e,frontend}-required.yml, container-build-check.yml; tldw_Server_API/tests/CI/test_merge_queue_{rules,actions,gate_workflows,frontend_required}.py, test_license_first_workflow_contracts.py, test_admin_ui_vitest_ratchet_workflow.py; Docs/Development/CI_REQUIRED_GATES.md, AGENTS.md, CLAUDE.md, Docs/ADR/063-in-repo-merge-queue.md (+ Docs/Published copy), both specs, the plan.
Final whole-branch code review (2026-10-14): 0 blocker, 1 major, 6 minor, 5 nits; all fixed. Major: a retry or restart marker alone was read as a second failure, so a stale decision (a racing run's re-run had already passed) or GitHub's lag in showing an accepted re-run could evict a healthy PR. Now the marker records the attempt re-run; eviction needs a fresh read showing a higher attempt and the gate still failed (or cancelled again); an unseen re-run is a strike. Minors: a missing license is dispatched while gates run; 409/422 re-run refusals strike (only the over-a-month 403 is final); a run held for someone else is never re-run (strike); the non-gate held-run cleanup runs twice; markers count only in the queue's own comments (GraphQL author github-actions, a Bot; checked on tldw_chatbook); every new guard mutation-checked (14 mutants, all killed, plus the earlier 7). Nits: rerun-error warning per gate; WAIT_RUN ASCII digits only; stale workflow comments; approvals skipped for a PR disarmed or pushed since the line was read. CI test dir after: 824 passed, 5 skipped.
Code review of #3237 (requesting-code-review, 2026-10-14; owner waived Qodo): 0 critical, 1 important, 5 minor; all fixed. I1: a held gate run the queue did not approve read as missing behind running gates and waited for an unrelated event; now any missing context on a head past the young window is started even while others run (decision order amended in the spec). M1: _start again never restarts a held run (a held re-run keeps its suite), and a run held for someone else is a held-<context> strike there too. M2: a retry attempt whose gate job was cancelled is restarted, whatever the run concluded. M3: no-run and approve-error strikes are per gate. M4: the spec's license claim corrected; the plan's known gap now covers a lost license with no decision during the run. M5: has_comment removed (no callers), _rerun's context is required, spec status, merge-queue.yml header and the gates doc updated. From the #3113 review: the attempt note is read fresh; the rerun-unseen strike is keyed by attempt. CI dir: 829 passed, 5 skipped; 6 new mutants plus the earlier 14 all killed; actionlint and pre-commit clean.
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
