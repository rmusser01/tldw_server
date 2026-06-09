---
id: TASK-45.44.21
title: Migrate ScheduledTasks product states to design-system primitives
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-09 20:22
labels:
- design-system
- webui
- product-state
- scheduled-tasks
dependencies: []
references:
- apps/packages/ui/src/components/Option/ScheduledTasks
- apps/packages/ui/src/components/ui/primitives/Alert.tsx
- apps/packages/ui/src/components/ui/primitives/Badge.tsx
- apps/packages/ui/src/components/ui/feedback/EmptyState.tsx
- apps/packages/ui/src/components/ui/feedback/LoadingState.tsx
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2336
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
- Docs/Design/tldw_web_design_system_baseline_reporting.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by resolving the current ScheduledTasks product-state guard drift on latest dev. Scope is limited to apps/packages/ui/src/components/Option/ScheduledTasks: replace product-state AntD Alert/Tag/Empty/Spin usage with shared design-system primitives and route ScheduledTasks canonical labels through the design-system state registry while preserving existing scheduled task behavior, copy, and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ScheduledTasks product-state Alerts render through the shared design-system Alert primitive while preserving copy and actions.
- [x] #2 ScheduledTasks status Tags render through the shared design-system Badge primitive with canonical state mapping where applicable.
- [x] #3 ScheduledTasks empty and loading states render through shared EmptyState/LoadingState primitives.
- [x] #4 ScheduledTasks hardcoded canonical labels reported by the guard are routed through the design-system state registry.
- [x] #5 Focused ScheduledTasks tests cover migrated product-state markers or registry labels.
- [x] #6 Direct product-state guard scan over ScheduledTasks reports zero findings; full verifier status is recorded with unrelated drift noted if any.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation replaced ScheduledTasks AntD product-state Alert/Empty/Spin/Tag usage with DS Alert, EmptyState, LoadingState, and Badge primitives, including the read-only Watchlists panel. ScheduledTasks blocked/unavailable labels now come from the design-system registry. Verification: focused ScheduledTasks vitest suite passed (99 tests), direct ScheduledTasks product-state guard scan passed with zero findings, git diff --check passed, full product-state verifier failed only on unrelated Skills/KnowledgeQA/Onboarding/ACP blockers, and TypeScript failed only on unrelated Notes/background/voice-cloning diagnostics.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ScheduledTasks product-state surfaces now render through shared design-system primitives, and the ScheduledTasks source tree has zero direct product-state guard findings. Focused ScheduledTasks tests pass. The full product-state verifier still reports unrelated blockers in Skills, KnowledgeQA, Onboarding, and ACP readiness. TypeScript still reports inherited non-ScheduledTasks debt in Notes tests, background.ts, and voice-cloning.ts. Bandit was not applicable because this task touched TypeScript/TSX and markdown only.
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
