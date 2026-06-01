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
Task 0 in progress: OpenUI runtime feasibility audit completed with conditional PASS, package manifests updated for WebUI/shared UI, extension dependency intentionally deferred due React 18.2 peer mismatch, and lockfile updated via `bun install --ignore-scripts` because normal workspace postinstall hangs in existing `wxt-prepare` behavior. Baseline Task 1 test command failed with no test files found as expected. Awaiting Task 0 review/commit.
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
