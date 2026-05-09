---
id: TASK-199
title: Implement VN Play branch navigation API
status: In Progress
assignee: []
created_date: '2026-05-09 22:22'
updated_date: '2026-05-09 23:07'
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
- [ ] #4 API exposes branch-navigation and branch restore endpoints, extends events filtering compatibly, maps stable errors, and documents the contract.
- [ ] #5 Focused VN Play tests pass, Bandit is run for touched backend scope, and final diff hygiene is clean.
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
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
