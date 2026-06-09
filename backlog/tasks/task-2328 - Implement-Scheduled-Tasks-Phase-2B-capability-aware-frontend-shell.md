---
id: TASK-2328
title: Implement Scheduled Tasks Phase 2B capability-aware frontend shell
status: Done
labels:
- scheduled-tasks
- webui
- ux
- phase-2b
- frontend
- implementation
priority: high
references:
- TASK-2326
- TASK-2325
- TASK-2324
- TASK-2327
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Add a pure ScheduledTasks capability helper module with gate enforcement, explicit creation-adapter guard, source intent/result/notification copy, and redaction coverage.
- [ ] #2 Add limited_availability to ScheduledTasks template state, labels, filtering, and effective-template lookup without changing Reminder behavior.
- [ ] #3 Render capability-aware Limited availability UI in the Create panel without Watch/Ingest create actions or scheduled success copy.
- [ ] #4 Wire default empty capability shell through /scheduled-tasks so Watch/Ingest remain non-creating by default and source-vendor IA copy is avoided.
- [ ] #5 Run focused ScheduledTasks unit/component tests plus whitespace verification; record Bandit skip rationale if no Python files are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 2B.2 capability-aware Scheduled Tasks frontend shell. Added a pure capability model with required Watch/Ingest gates, explicit creationAdapterSupported guard, source-intent/result/notification copy helpers, and private-source redaction. Added limited_availability state support, effective-template filtering/lookup, Create panel capability rendering, extension-width coverage, and page-level default empty capability wiring. Watch/Ingest remain non-creating by default and the empty state no longer frames GitHub or YouTube as primary IA. Verification: capability/template/Create panel/page focused suite passed with 65 tests using --testTimeout=20000 after the exact plan command exposed an existing 5s page-test timing variance; route-state regression passed with 8 tests; git diff --check exited 0; product touched-file placeholder scan had no matches. Bandit skipped because no Python files were touched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation follows the Phase 2B.2 plan scope and avoids Watchlists/backend/Home/RAG/API changes.
- [ ] #2 Tests are written before production changes and focused tests pass.
- [ ] #3 No new placeholder markers or unscoped TODOs are introduced.
- [ ] #4 Backlog task records files changed, verification, known skips, and final summary.
- [ ] #5 Changes are committed in focused increments.
<!-- DOD:END -->
