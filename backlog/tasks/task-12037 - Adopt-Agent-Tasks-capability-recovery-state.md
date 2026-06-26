---
id: TASK-12037
title: Adopt Agent Tasks capability recovery state
status: Done
created_date: 2026-06-26 01:21
labels:
- webui
- agents
- ux
- accessibility
priority: medium
references:
- TASK-418.11
- TASK-214
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage8-agent-tasks-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
updated_date: 2026-06-26 01:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the standalone Agent Tasks route. Replace top-level unsupported, setup, workspace setup, and project-load alert states with shared user-language recovery/state primitives and diagnostics where appropriate, while preserving existing task/project workflows when orchestration is available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Tasks renders shared RecoveryCallout/StatePanel capability states for unsupported orchestration routes, ACP setup gaps, workspace setup gaps, and project-load failures instead of local alert-only states.
- [x] #2 Recovery states include user-language title/message, retry or navigation action as appropriate, and non-secret diagnostics for method/path/status/raw message/server URL when request failure data exists.
- [x] #3 Existing project/task lists, workspace filtering, route hydration, task diagnostics, and successful canonical connection behavior remain available when requests succeed.
- [x] #4 Focused tests cover successful rendering plus unsupported orchestration, ACP setup, workspace setup, and project-load failure paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 8 plan document for Agent Tasks capability recovery.
2. Add failing focused tests that expect shared recovery/state primitives for unsupported, setup, workspace, and project-load states.
3. Track request failure diagnostics for top-level project loading and map page-level states through shared capability primitives.
4. Run focused Agent Tasks tests, touched-file ESLint, whitespace checks, and record Bandit applicability.
5. Update Backlog and commit the Stage 8 slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 8 Agent Tasks capability recovery. Top-level unsupported orchestration now renders a shared RecoveryCallout with method/path/server diagnostics; ACP setup and workspace setup render shared StatePanel states while preserving existing navigation actions; project-list load failures now track structured, redacted request metadata and render a retryable RecoveryCallout with method/path/status/server/raw-message diagnostics. Successful project/task loading, workspace filtering, route hydration, and task diagnostics remain covered by the focused connection test.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Agent Tasks now uses shared RecoveryCallout/StatePanel capability states for unsupported orchestration, ACP setup gaps, workspace setup gaps, and project-list load failures. Added focused coverage for shared primitives and diagnostics while retaining existing successful project/task workflow tests. Verification: focused Agent Tasks Vitest passed; touched-file ESLint passed with only the known repo-level Next pages-directory notice; git diff --check passed. Bandit not applicable because changes are TS/TSX/docs/task metadata only.
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
