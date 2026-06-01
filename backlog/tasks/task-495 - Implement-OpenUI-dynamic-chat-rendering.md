---
id: TASK-495
title: Implement OpenUI dynamic chat rendering
status: In Progress
references:
- TASK-491
- TASK-493
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
- Docs/superpowers/plans/2026-06-01-openui-dynamic-chat-rendering-implementation-plan.md
- https://github.com/pewdiepie-archdaemon/odysseus/pull/151
documentation:
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
- Docs/superpowers/plans/2026-06-01-openui-dynamic-chat-rendering-implementation-plan.md
modified_files:
- Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md
- apps/tldw-frontend/package.json
- apps/packages/ui/package.json
- apps/bun.lock
- apps/packages/ui/src/types/dynamic-ui.ts
- apps/packages/ui/src/utils/dynamic-ui.ts
- apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts
- backlog/tasks/task-495 - Implement-OpenUI-dynamic-chat-rendering.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan for OpenUI dynamic chat rendering. Scope includes runtime feasibility, shared Dynamic UI metadata/utilities, persistence, renderer registry, OpenUI adapter, /chat request mode, action bridge, shared-surface fallbacks, verification, and Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 0 runtime feasibility completed and recorded before OpenUI dependency use
- [ ] #2 Dynamic UI envelope/types/validation/persistence implemented with test-first coverage
- [ ] #3 /chat supports temporary OpenUI request mode and active rendering only on opted-in web chat surface
- [ ] #4 OpenUI actions round-trip as visible user messages with host-attached provenance metadata
- [ ] #5 Extension sidepanel and workspace render source fallback unless explicitly enabled later
- [ ] #6 Focused unit/build/browser/security verification completed or documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tasks 0-1 complete. Task 0 committed OpenUI runtime feasibility review and dependency gate in ec3e71c6b7, with follow-up Backlog status commit ef2035486a. Task 1 added Dynamic UI shared types, validation helpers, source preflight, action normalization, sensitive-value blocking, and focused tests across commits 2e08f1e38c, 8ec5a81a2b, and 69d1d36890. Focused utility suite passes with 17 tests. Task 1 passed spec-compliance and code-quality review.
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
