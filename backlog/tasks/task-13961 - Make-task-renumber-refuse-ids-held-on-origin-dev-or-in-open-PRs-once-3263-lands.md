---
id: TASK-13961
title: 'Make task renumber refuse ids held on origin/dev or in open PRs once #3263
  lands'
status: Done
labels:
- backlog
dependencies:
- TASK-13960
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13960 shipped backlog-py task renumber with a local id check only, so that the renumbering did not wait for #3263 (TASK-13854), which adds the origin/dev and open-PR check to task create. Once #3263 is on dev, renumber_task should run the same check (first plan review, M3). Helper_Scripts/renumber_duplicate_task_ids.py must then pass check_remote=False, because backlog-py's live check sees the renumbering PR's own block (second plan review, I1). The appendix of plan_dupids_v3.md holds the code and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 task renumber refuses a top-level id that origin/dev or an open PR holds, with the same message as task create
- [x] #2 renumber_task(..., check_remote=False) skips that check, and the generator's apply passes it
- [x] #3 Tests cover the refusal, the skip, and apply taking the ids its own open PR shows
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-07-renumber-remote-id-check.md. #3263 (TASK-13854) is on dev; renumber_task calls its _reject_remote_id (no second copy). ADR-004 Change summary: waived by the owner on 2026-11-07 ('Yes, waive for all').
RED/GREEN: Task 1 2 failed (the refusal: assert 'None' == 'Task id already exists on open pull requests: TASK-15000 ...'; the skip: TypeError, unexpected keyword argument 'check_remote') -> 2 passed, backlog-py 179 -> 181 passed; mutations (default False; the guard dropped) fail one test each. Task 2 1 failed (TaskMutationError from apply) -> 8 passed. Offline probe (remote_ids._run and the generator's run/git raise): 189 passed. tests/Docs + lint + CI 1586 passed, 6 skipped (plus 2 failed + 1 error, env-only: no coverage/setuptools in the shared venv); shard coverage, test-quality ratchet, check_backlog_task_ids unchanged; Bandit: no issues; ruff clean.
Docs: AGENTS.md's Renumber bullet and the backlog-py README's Task ids section say renumber refuses a remote-held id; neither is mirrored to Docs/Published. No CLI flag for the skip (Decision 2).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
task renumber refuses a top-level id that origin/dev or an open PR holds, with task create's message: renumber_task calls #3263's _reject_remote_id after its local check, before it reads the file. Child ids skip it, as in create, and an unreachable source falls back to the local check. renumber_task(..., check_remote=False) skips it, and Helper_Scripts/renumber_duplicate_task_ids.py's apply passes that, since a re-run's live check would list the renumbering PR's own block. Tests cover the refusal (through the CLI, compared with create's message), the skip, and apply taking an id its own open PR shows. AGENTS.md and the backlog-py README updated.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
