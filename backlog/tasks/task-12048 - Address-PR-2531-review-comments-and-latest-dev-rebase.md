---
id: TASK-12048
title: Address PR 2531 review comments and latest dev rebase
status: Done
created_date: 2026-06-26 17:50
labels:
- webui
- pr-review
- raw-error
- tests
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2531
- TASK-12030
- TASK-12047
modified_files:
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlaygroundRecovery.test.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx
- apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx
- apps/packages/ui/src/components/Option/AgentRegistry/index.tsx
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- apps/packages/ui/src/components/Option/DataTables/SourceSelector.tsx
- apps/packages/ui/src/components/Option/DataTables/__tests__/SourceSelector.recovery.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
- apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx
- apps/packages/ui/src/components/Option/Models/__tests__/AvailableModelsList.test.tsx
- apps/packages/ui/src/components/Option/Models/index.tsx
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
- apps/packages/ui/src/components/ui/state/__tests__/capability-state.test.ts
- apps/packages/ui/src/components/ui/state/capability-state.ts
- apps/packages/ui/src/utils/__tests__/server-error-message.test.ts
- apps/packages/ui/src/utils/server-error-message.ts
- tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py
- tldw_Server_API/tests/UserProfile/test_user_profile_read.py
updated_date: 2026-06-26 18:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2531 onto the latest dev branch and address actionable review feedback from automated PR reviewers. Current visible findings cover pytest marker/docstring policy on newly added backend tests and sanitization of recovery diagnostics rendered through buildCapabilityState.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is based on the latest fetched origin/dev and pushed back to PR #2531.
- [x] #2 New or modified backend tests added by this PR carry accepted pytest markers and docstrings where required by the review feedback.
- [x] #3 Recovery diagnostics rendered from buildCapabilityState redact raw backend messages and sensitive diagnostic values before display.
- [x] #4 Focused regression tests and lint/security checks are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Fetch latest dev and rebase the PR branch if needed.', 'Inspect PR comments/checks and validate each actionable finding against the codebase.', 'Add tests for recovery diagnostic redaction and apply a shared sanitization fix at the capability-state boundary.', 'Add pytest markers/docstrings to the reviewed backend tests.', 'Run focused WebUI/backend tests, lint, diff checks, Bandit, then push with force-with-lease if the rebase rewrites the branch.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased codex/webui-auth-persistence onto fetched origin/dev (6fe09bb9). Addressed PR review comments by adding pytest markers/docstrings, sanitizing buildCapabilityState diagnostics, routing Models/DataTables/Skills/Agent Registry/Agent Tasks error details through the shared sanitizeServerErrorMessage utility, and updating recovery tests to assert redacted diagnostics. Verification so far: focused frontend batch passed (24 files, 212 tests); backend focused tests passed (22 tests); eslint on changed TS/TSX exited 0 with existing warnings only; git diff --check passed; Bandit ran on touched Python tests and reported only LOW B101 assert findings in test files; design-system state verifier remains blocked by missing local typescript package.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2531 onto fetched origin/dev (6fe09bb9) and force-pushed codex/webui-auth-persistence. Addressed visible review feedback by adding pytest markers/docstrings to reviewed backend tests, sanitizing buildCapabilityState diagnostics, reusing sanitizeServerErrorMessage across Models, Data Tables, Skills Manager, Agent Registry, and Agent Tasks, and updating tests to assert redacted diagnostics. Verification recorded: focused frontend batch passed (24 files, 212 tests), backend focused tests passed (22 tests), eslint on changed TS/TSX completed with warnings only, git diff --check passed, Bandit ran on touched Python tests with only LOW B101 pytest-assert findings, and design-state verification remains blocked by the local missing typescript package.
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
