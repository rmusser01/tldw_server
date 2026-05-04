---
id: TASK-45.2
title: Implement tldw_server design-system proof surface
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-04 17:49'
updated_date: '2026-05-04 20:28'
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
- [x] #1 State tokens and Tailwind mappings are added as aliases to the existing semantic palette for WebUI and extension builds.
- [x] #2 A typed design-system state registry and shared state primitives are added and exported from the shared UI package.
- [x] #3 Backend recovery, configuration/readiness gates, setup, health diagnostics, and /admin/server use canonical state labels, actions, and diagnostics without migrating unrelated admin routes.
- [x] #4 Focused Vitest coverage verifies token aliases, state registry, shared primitives, recovery, readiness, setup, health, admin states, and proof-surface drift guards.
- [x] #5 WebUI compile/token sync and extension compile/build token sync pass or blockers are documented.
- [x] #6 Visual smoke checks for setup, health, and admin proof routes are run or blockers are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Final executed plan: 1) Preserve the approved contract and implementation plan while isolating work on branch codex/tldw-web-design-system-proof-surface. 2) Add state token aliases and Tailwind mappings as aliases to existing semantic tokens, then add a typed design-system state registry and shared state primitives in packages/ui. 3) Migrate only the agreed proof surface: backend recovery, route/error-boundary recovery, ConfigurationErrorScreen, ServerReadinessGate, setup/onboarding, /settings/health, and /admin/server state surfaces. 4) Tighten review findings with actionable admin guard recovery actions, runtime onboarding state tests, and a proof-surface static guard. 5) Run focused Vitest coverage, existing admin regression coverage, WebUI compile/token sync, extension build/token sync, and a browser smoke of /settings/health; document the one pre-existing extension compile blocker and Bandit skip.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worker 3 started Task 5 in .worktrees/tldw-web-design-system-proof-surface on branch codex/tldw-web-design-system-proof-surface. Shared state primitives and design-system registry are already present; this slice will only touch the owned setup/onboarding/health/admin files.

Final verification recorded on 2026-05-04: focused proof-surface Vitest suite passed from apps/tldw-frontend with 12 files / 38 tests; admin media-budget regression passed with 1 file / 3 tests; git diff --check passed. WebUI compile passed with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 and token sync OK. Extension Chrome dev build passed with token sync OK; extension bun run compile is blocked before touched code by existing wxt.config.ts missing a declaration for ./scripts/post-build-tasks.mjs. Browser smoke opened http://localhost:18081/settings/health; canonical Degraded and Unavailable states rendered. Console errors were expected missing local backend/API key requests. Bandit skipped because this task touched frontend TypeScript/TSX/docs/task files only and no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first governed tldw_server WebUI/browser-extension design-system proof surface.

What changed:
- Added and verified canonical product-state infrastructure: state token aliases, registry coverage, shared state primitives, and proof-surface drift guards.
- Migrated the agreed v1 proof surface only: backend recovery, route/error-boundary recovery, configuration/readiness gates, setup/onboarding states, /settings/health, and /admin/server state affordances.
- Tightened review issues before finalization: admin guard primary actions now open the existing tldw_server documentation instead of rendering no-op buttons; onboarding progress labels now show Retrying only while busy, Sign in required for auth failures, and Ready for successful connection; onboarding tests now render runtime states instead of checking source strings.

Why:
- The contract calls for canonical state language without introducing a new palette or broadly migrating admin/product routes. This implementation aliases existing semantic tokens, keeps AntD as the mechanical substrate where already used, and guards the proof surface against drift while leaving broader inventory/migration for later.

Verification:
- Focused Vitest proof-surface suite: 12 files / 38 tests passed.
- Admin media-budget regression: 1 file / 3 tests passed.
- git diff --check passed.
- WebUI compile/token sync passed with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000.
- Extension Chrome dev build/token sync passed.
- Browser smoke for /settings/health rendered canonical Degraded and Unavailable states.

Known blocker/skips:
- Extension bun run compile is blocked by existing wxt.config.ts missing a declaration for ./scripts/post-build-tasks.mjs before touched code is checked.
- Bandit skipped because no Python code was touched.
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
