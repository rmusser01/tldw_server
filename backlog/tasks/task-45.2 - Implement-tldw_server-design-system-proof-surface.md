---
id: TASK-45.2
title: Implement tldw_server design-system proof surface
status: In Progress
assignee: []
created_date: '2026-05-04 17:49'
labels:
  - frontend
  - design-system
  - implementation
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - >-
    Docs/superpowers/plans/2026-05-04-tldw-web-design-system-proof-surface-implementation-plan.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first governed WebUI/browser-extension design-system migration slice from the approved contract and implementation plan. The work should add state token aliases, a typed canonical state registry, shared state primitives, and migrate only setup, backend recovery, configuration/readiness gates, health diagnostics, and /admin/server to shared product-state language while preserving extension compatibility and AntD as a mechanics substrate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 State tokens and Tailwind mappings are added as aliases to the existing semantic palette for WebUI and extension builds.
- [ ] #2 A typed design-system state registry and shared state primitives are added and exported from the shared UI package.
- [ ] #3 Backend recovery, configuration/readiness gates, setup, health diagnostics, and /admin/server use canonical state labels, actions, and diagnostics without migrating unrelated admin routes.
- [ ] #4 Focused Vitest coverage verifies token aliases, state registry, shared primitives, recovery, readiness, setup, health, admin states, and proof-surface drift guards.
- [ ] #5 WebUI compile/token sync and extension compile/build token sync pass or blockers are documented.
- [ ] #6 Visual smoke checks for setup, health, and admin proof routes are run or blockers are documented.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
