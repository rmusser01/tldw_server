---
id: TASK-2387
title: >-
  Implement Workspace activity and contained-resource index contract for issue
  1994
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-18 20:12'
labels:
  - workspaces
  - acp
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1994'
  - 'https://github.com/rmusser01/tldw_server/issues/1984'
documentation:
  - Docs/Design/Workspace_Container_Contract_2026_06.md
  - Docs/superpowers/plans/2026-06-18-workspace-activity-index-contract-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1994 as the Workspace Phase 2 activity/index slice: durable workspace activity events, a contained-resource index endpoint, minimal frontend TypeScript contract normalizers, docs, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace activity events are stored durably, listed newest-first, and redact secrets/path-like metadata.
- [x] #2 Workspace index endpoint returns identity, grouped resource previews, membership totals, runtime summary, warnings, recent activity, and partial-error field.
- [x] #3 Membership and runtime binding write paths append best-effort activity events without breaking primary writes.
- [x] #4 Frontend TypeScript contract normalizers provide stable display-safe defaults and preserve server owner hrefs.
- [x] #5 Design docs, Backlog task, backend/frontend tests, diff check, Bandit, commit, push, and PR are completed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-18-workspace-activity-index-contract-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan in isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-activity-index-contract`.

Implemented ChaChaNotes schema v50 activity storage; activity DB APIs; WorkspaceActivityIndexService; `GET /api/v1/workspaces/{workspace_id}/index`; best-effort membership/runtime binding activity hooks; frontend `src/services/workspace-index` contract normalizers; and docs updates.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py -q` passed 138 tests after rebase onto `origin/dev` (`a798b78f60`).
- `./node_modules/.bin/vitest run src/services/workspace-index/__tests__/normalizers.test.ts --maxWorkers=1` passed 3 tests.
- `git diff --check` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Workspaces/activity_index.py tldw_Server_API/app/core/Workspaces/membership_service.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_workspace_activity_index.json` completed with zero results.

PR opened: https://github.com/rmusser01/tldw_server/pull/2396. Earlier Git DNS fetch blocker is resolved; branch was rebased onto `origin/dev` before final verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Workspace #1994 activity/index contract in PR https://github.com/rmusser01/tldw_server/pull/2396. The backend now stores secret-safe workspace activity events, exposes a contained-resource index endpoint with grouped previews/runtime warnings/recent activity, and records best-effort membership/runtime binding events. The frontend has minimal TypeScript contract normalizers for future UI consumption, and the design doc records the endpoint as an inspection/navigation contract rather than a duplicate workspace dashboard. Verification passed: 138 focused backend tests, 3 frontend normalizer tests, `git diff --check`, and Bandit with zero results.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review follow-up for PR #2396: rebased branch against latest origin/dev (already up to date), addressed Gemini/Qodo comments by moving runtime-binding activity event construction into core, isolating the workspace index builder in run_in_threadpool, narrowing runtime-binding upsert activity to normalized user-field changes, replacing activity listing dynamic SQL with static parameterized query variants, returning deleted-workspace index payloads with workspace_deleted warnings, and replacing list+scan activity insert readback with direct primary-key lookup plus return_row=False support for best-effort write hooks.

Review verification:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed 107 tests.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py -q` passed 139 tests.
- `./node_modules/.bin/vitest run src/services/workspace-index/__tests__/normalizers.test.ts --maxWorkers=1` passed 3 tests.
- `git diff --check` passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Workspaces/activity_index.py tldw_Server_API/app/core/Workspaces/membership_service.py tldw_Server_API/app/core/Workspaces/runtime_bindings.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_workspace_activity_index_review.json` completed with zero results/errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
