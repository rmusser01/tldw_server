---
id: TASK-199
title: Implement VN Play branch navigation API
status: Done
assignee: []
created_date: '2026-05-09 22:22'
updated_date: '2026-05-10 00:52'
labels:
  - vn-play
  - api
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1463'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/superpowers/specs/2026-05-09-vn-play-branch-navigation-api-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vn-play-branch-navigation-api-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1463 from the reviewed design spec and saved implementation plan. Scope covers backend branch navigation read model, session action idempotency and shared turn/restore mutation gate, branch-aware event filtering, guarded branch/checkpoint restore, API schemas/endpoints, docs, tests, Bandit, and PR-ready verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure branch navigation read model derives active path, parent branch ids, direct/subtree event ranges, and stable warning payloads.
- [x] #2 Repository persists session restore actions, enforces restore idempotency, and shares a session mutation gate with turn requests.
- [x] #3 Service exposes branch navigation, branch-aware event filtering, branch restore, and checkpoint restore idempotency while preserving existing Freeform and Story turn behavior.
- [x] #4 API exposes branch-navigation and branch restore endpoints, extends events filtering compatibly, maps stable errors, and documents the contract.
- [x] #5 Focused VN Play tests pass, Bandit is run for touched backend scope, and final diff hygiene is clean.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/superpowers/plans/2026-05-09-vn-play-branch-navigation-api-implementation-plan.md as the implementation plan of record.

Execution mode: subagent-driven development, one implementation subagent per plan task, with spec compliance review followed by code quality review before moving to the next task.

Initial sequence:
1. Run preflight and focused VN Play baseline.
2. Dispatch Task 1 implementer for the pure branch navigation read model only.
3. Review Task 1 for spec compliance and code quality.
4. Integrate and commit Task 1 before dispatching Task 2.
5. Continue task-by-task through repository session actions, service integration, restore semantics, API endpoints, docs, and final verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preflight completed on rebased branch codex/vn-play-branch-navigation-api at origin/dev 8e52700d0 plus local planning commits. Focused VN Play baseline run: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 71 passed, 5 warnings in 19.26s.

Task 1 complete after subagent implementation plus review loop. Commits: a8eca8712 Add VN Play branch navigation read model; f2bb5b56d Fix VN Play branch replay warnings; 83de8625f Handle VN Play branch navigation edge cases. Spec review passed after replay cap and ambiguous attribution fixes. Code-quality review passed after limit=0 and restore default-target fixes. Controller verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q => 16 passed, 5 warnings; git diff --check => exit 0.

Task 2 started in worktree .worktrees/vn-play-branch-navigation-api at afedc842c. Scope limited to VNPlay_DB.py and test_vn_play_db.py. TDD plan: add failing repository tests for session_actions schema/idempotency/update decoding and shared mutation gate behavior; run focused red test; implement schema migration/table/indexes, session action CRUD/update decoding, session active marker decode/update, and cross-lock acquisition/clear helpers; run required focused VN Play tests, Bandit on VNPlay_DB.py, and git diff --check before committing.

Task 2 verification complete. Red evidence: focused DB pytest failed before production edits with missing session_actions schema and missing session-action/gate repository methods (3 failed, 10 passed). Green evidence: test_vn_play_db.py => 13 passed, 5 warnings; test_vn_play_db.py + test_vn_play_branch_navigation.py => 24 passed, 5 warnings; git diff --check => exit 0; Bandit on VNPlay_DB.py => exit 0 with results: [] and errors: [].

Task 2 committed as 2dc0736b2 Add VN Play session action locking. Post-commit status: branch codex/vn-play-branch-navigation-api is ahead 9 and behind origin/dev by 2, with no unstaged tracked file changes reported by git status --short --branch.

Task 2 review follow-up complete. Red evidence: new regression tests failed before fixes with 3 failures covering invalid session-action locks, mismatched active action lookup, and owner-scoped/immutable update behavior. Green evidence: test_vn_play_db.py => 16 passed, 5 warnings; test_vn_play_db.py + test_vn_play_branch_navigation.py => 27 passed, 5 warnings; git diff --check => exit 0; Bandit on VNPlay_DB.py => exit 0 with results: [] and errors: [] at /tmp/bandit_vn_play_branch_navigation_task2_fix.json.

Task 2 review follow-up committed as daeb82fd1 Harden VN Play session action locking. Post-commit status: branch codex/vn-play-branch-navigation-api is ahead 10 and behind origin/dev by 2, with no unstaged tracked file changes reported by git status --short --branch.

Task 2 complete after subagent implementation plus review loop. Commits: 2dc0736b2 Add VN Play session action locking; daeb82fd1 Harden VN Play session action locking. Spec review passed. Code-quality review passed after hardening action ownership/session/status checks, scoped update returns, and immutable action identity fields. Controller verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py -q => 27 passed, 5 warnings; git diff --check => exit 0. Bandit reported by implementer: VNPlay_DB.py results/errors empty at /tmp/bandit_vn_play_branch_navigation_task2_fix.json.

Rebased codex/vn-play-branch-navigation-api onto origin/dev 90b3b767a before Task 3. New current head before Task 3 dispatch: b6984ca30. Post-rebase focused VN Play baseline: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 88 passed, 5 warnings in 22.12s.

Task 3 service slice started in .worktrees/vn-play-branch-navigation-api at 9bad34d8d. Scope constrained to service.py, branch_navigation.py if needed, and VN Play turn/navigation tests. Plan: add failing service tests for branch navigation, metadata-aware branch event filtering, branch ownership errors, pagination, and branch_node_id tagging; implement get_branch_navigation/list_events_with_metadata/list_events wrapper and Story active-branch tagging; then run focused pytest, diff check, Bandit, and commit if clean.

Task 3 service slice complete and committed as 19eb33a72. Red evidence: new focused service tests failed before implementation with missing get_branch_navigation/list_events_with_metadata APIs and branch-tagging expectations. Green evidence: test_vn_play_turns.py => 33 passed, 5 warnings; test_vn_play_branch_navigation.py + test_vn_play_turns.py => 44 passed, 5 warnings; full tldw_Server_API/tests/VN_Play => 96 passed, 5 warnings. Hygiene: git diff --check exit 0; Bandit service.py + branch_navigation.py exit 0 with /tmp/bandit_vn_play_branch_navigation_task3.json results/errors empty. Scope changed only service.py and test_vn_play_turns.py in the Task 3 worktree commit.

Task 3 complete after subagent implementation plus review loop. Commit: 19eb33a72 Expose VN Play branch navigation service. Spec review passed. Code-quality review passed with no critical/important/minor issues; only future performance consideration for large histories. Controller verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q => 44 passed, 5 warnings; git diff --check => exit 0. Bandit reported by implementer: service.py/branch_navigation.py results/errors empty at /tmp/bandit_vn_play_branch_navigation_task3.json.

Task 4 guarded restore slice started in .worktrees/vn-play-branch-navigation-api at d4531d84c. Scope constrained to service/repository core and VN Play tests per user handoff. TDD plan: add failing service tests for branch_latest restore replay/idempotency, choice_point restore semantics, stale/active-lock/idempotency conflicts, Freeform rejection, and checkpoint restore idempotency/versioning; verify focused red failures; implement service restore_branch plus checkpoint_restore idempotency using vn_play_session_actions and a focused atomic repository completion helper if needed; run focused VN Play pytest, Bandit on touched production scope, diff check, then commit.

Task 4 guarded restore slice verification complete. Red evidence: initial new restore-focused test run failed with 11 failures for missing restore_branch/checkpoint idempotency behavior; later self-review regression test reproduced a target-unavailable lock leak before fix. Green evidence after fixes: restore-focused subset => 11 passed; target cleanup regression => 1 passed; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q => 61 passed, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 108 passed, 5 warnings; git diff --check => exit 0; Bandit on VN Play core + VNPlay_DB.py => exit 0, /tmp/bandit_vn_play_branch_navigation_task4.json results/errors empty.

Task 4 committed as 914f0b714 Add guarded VN Play branch restore. Post-commit status in .worktrees/vn-play-branch-navigation-api: branch codex/vn-play-branch-navigation-api ahead 15 of origin/dev with no unstaged tracked changes reported by git status --short --branch.

Task 4 quality review fix loop started. Validated findings: create_session_action duplicate-at-insert race should replay/conflict deterministically, and failed/abandoned restore action status plus session lock clearing should be atomic in one repository transaction. Worker asked to add regression coverage, rerun focused/full VN Play tests, diff check, and Bandit.

Task 4 quality review fix started at 914f0b714 in .worktrees/vn-play-branch-navigation-api. Scope: VNPlay_DB.py, service.py, and VN Play tests only. Plan: add failing DB regression tests for duplicate-at-insert session action idempotency and terminal-action lock clearing; add/keep service coverage for target failure lock cleanup through the service helper; implement race-safe create_session_action insert handling and a transactional terminal action+guarded lock helper; rerun focused VN Play tests, diff check, Bandit, and commit.

Quality review fixes complete: added race-safe create_session_action duplicate-at-insert recovery for matching idempotency payload hashes and deterministic idempotency_key_conflict for hash mismatches; added transactional mark_session_action_terminal helper that marks failed/abandoned and clears active_session_action_id only when it still points at that action; updated VNPlayService failure/abandon paths to use the atomic helper. TDD red evidence: targeted new tests failed before implementation with raw sqlite UNIQUE constraint, missing mark_session_action_terminal, and service still using separate lock clear. Green/verification: targeted new tests passed; focused DB+turn suite 63 passed; full VN_Play suite 110 passed; git diff --check clean; Bandit touched production scope wrote /tmp/bandit_vn_play_branch_navigation_task4_fix.json with 0 errors and 0 results.

Task 4 review fix loop complete. Fix commit 2957038cb Fix VN Play restore action races. Code-quality re-review found no Critical/Important/Minor issues and approved continuing to Task 5. Controller verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q => 63 passed, 5 warnings. git diff --check => exit 0. Bandit artifact /tmp/bandit_vn_play_branch_navigation_task4_fix.json has results/errors empty.

Rebased codex/vn-play-branch-navigation-api onto latest origin/dev before Task 5. Post-rebase head b104bbb06; Task 4 commits rewritten as 480595fad Add guarded VN Play branch restore and 689d972cf Fix VN Play restore action races. Post-rebase focused VN Play baseline: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 110 passed, 5 warnings in 29.49s.

Task 5 API slice started in .worktrees/vn-play-branch-navigation-api at 847bbbee8 after rebase. Scope constrained to vn_play_schemas.py, vn_play.py, and test_vn_play_api.py. Plan: write failing API tests for branch-navigation, branch-aware events including warning header, branch restore, stale/branch error mappings, and checkpoint restore idempotency wiring; add schemas/endpoints/query parameters/error mapping; run focused API tests, full VN Play tests if practical, diff check, Bandit on touched API files, and commit.

Task 5 API slice complete after worker handoff required controller cleanup. API schema/endpoint/test patch preserved; malformed Backlog note collapse rejected and restored in worktree. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q => 30 passed, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 117 passed, 5 warnings in 32.60s; git diff --check => exit 0; Bandit API scope wrote /tmp/bandit_vn_play_branch_navigation_task5.json with results/errors empty.

Task 5 review fix complete. Fixed legacy events compatibility so omitted unfiltered limit remains unbounded while branch-filtered omitted limit defaults to 100. Re-review found no Critical Important or Minor issues and approved commit. Verification after fix: API tests 32 passed, VN_Play suite 119 passed, diff check clean, Bandit task5 JSON empty.

Task 5 API implementation is complete. AC4 remains unchecked until Task 6 documents the API contract.

Task 6 docs and final verification complete. Updated VN Play API docs for branch navigation branch-aware events warning headers branch restore restore idempotency and stable error mapping. Final verification: VN_Play suite 119 passed. Bandit final JSON empty. Diff check clean.

Post-rebase final verification on branch codex/vn-play-branch-navigation-api: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 121 passed, 5 warnings in 33.76s. Bandit final scope wrote /tmp/bandit_vn_play_branch_navigation.json with results/errors empty. git diff --check => exit 0. Final review fixes included duplicate branch path hardening and checkpoint restore replay response preservation.

Opened PR #1483 against dev: https://github.com/rmusser01/tldw_server/pull/1483. Initial GitHub check state: pending CI; mergeStateStatus UNSTABLE because checks are still running.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented backend-owned VN Play branch navigation API for issue #1463. Added a pure branch-navigation read model, branch-aware event filtering, stable warning/error payloads, session action idempotency, shared turn/restore mutation locking, guarded branch restore, checkpoint restore replay semantics, API schemas/endpoints, and API documentation.

The design keeps navigation derived from persisted VN events instead of introducing a parallel branch store, and uses repository-backed session actions for idempotency/concurrency so custom frontends get the same standalone API behavior.

Verification: pytest tldw_Server_API/tests/VN_Play -q => 121 passed, 5 warnings; Bandit touched backend scope => 0 results/errors; git diff --check => clean. Known skips or blockers: none.
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
