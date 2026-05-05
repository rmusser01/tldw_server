---
id: TASK-45.2
title: Implement tldw_server design-system proof surface
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-04 17:49'
updated_date: '2026-05-05 03:14'
labels:
  - frontend
  - design-system
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1272'
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

PR #1272 review-fix pass: 1) remove substrate leaks and invalid HTML wrappers in state primitives; 2) forward secondary action loading state and add coverage; 3) make design-system static guards cwd-independent; 4) harden state-key type guard and tests; 5) make setup CTA focus the server URL input and cover it; 6) rewrite onboarding runtime-state tests to avoid mocking React primitives; 7) clarify the two backlog review nits included in the PR; 8) run focused Vitest, diff checks, and update/push the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worker 3 started Task 5 in .worktrees/tldw-web-design-system-proof-surface on branch codex/tldw-web-design-system-proof-surface. Shared state primitives and design-system registry are already present; this slice will only touch the owned setup/onboarding/health/admin files.

Final verification recorded on 2026-05-04: focused proof-surface Vitest suite passed from apps/tldw-frontend with 12 files / 38 tests; admin media-budget regression passed with 1 file / 3 tests; git diff --check passed. WebUI compile passed with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 and token sync OK. Extension Chrome dev build passed with token sync OK; extension bun run compile is blocked before touched code by existing wxt.config.ts missing a declaration for ./scripts/post-build-tasks.mjs. Browser smoke opened http://localhost:18081/settings/health; canonical Degraded and Unavailable states rendered. Console errors were expected missing local backend/API key requests. Bandit skipped because this task touched frontend TypeScript/TSX/docs/task files only and no Python code.

Draft PR opened on 2026-05-04: https://github.com/rmusser01/tldw_server/pull/1272

PR #1272 review pass complete on rebased branch codex/tldw-web-design-system-proof-surface. Addressed Gemini/CodeRabbit comments: StatePanel now avoids rounded-pill and paragraph-wrapped ReactNode messages, DiagnosticRow no longer wraps arbitrary ReactNode values in span, ServerAdminPage no longer relies on ant-alert, secondary actions forward loading/disabled state, state-key guard uses own-property checks, static proof guards resolve from import.meta.url, setup CTA focuses the server URL input, onboarding runtime-state tests no longer mock React primitives, and the TASK-14 wording nit is clarified. The old TASK-16 status nit became obsolete after rebasing onto origin/dev because that task is already Done with DoD checked.

2026-05-05 follow-up PR review refresh: Qodo added two actionable findings on PR #1272 after the prior review pass. Planned narrow fixes: replace tldw-frontend deep UI state imports with the exported @tldw/ui package root, and add noopener to ServerAdminPage external documentation links. CI is still queued/pending with no failures to debug yet.

Validation update: package-root imports failed local Vitest/Vite resolution because current tldw-frontend aliases map @tldw/ui to apps/packages/ui/src rather than apps/packages/ui/index.ts. Using the reviewer-supported alternative instead: keep the established deep imports and add an explicit @tldw/ui ./components/ui/state subpath export in apps/packages/ui/package.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and reviewed the first governed tldw_server WebUI/browser-extension design-system proof surface.

What changed:
- Added canonical product-state infrastructure: state token aliases, registry coverage, shared state primitives, and proof-surface drift guards.
- Migrated only the agreed v1 proof surface: backend recovery, route/error-boundary recovery, configuration/readiness gates, setup/onboarding states, /settings/health, and /admin/server state affordances.
- Addressed PR #1272 review comments: removed AntD/internal styling leakage from StatePanel usage, fixed invalid generic ReactNode wrappers, forwarded secondary action loading state, hardened state-key checks, made static guards cwd-independent, focused setup on the server URL input, rewrote onboarding state tests without mocking React primitives, and clarified the TASK-14 wording nit.
- Rebased the branch onto origin/dev; the old TASK-16 status nit is obsolete because dev already has TASK-16 as Done with Definition of Done checked.

Why:
- The contract calls for canonical state language without introducing a new palette or broadly migrating admin/product routes. This implementation aliases existing semantic tokens, keeps AntD only as the mechanical substrate where already used, and guards the proof surface against drift while leaving broader inventory/migration for later.

Verification:
- Focused Vitest proof-surface suite after rebase: 12 files / 39 tests passed.
- Review-specific Vitest slice: 6 files / 21 tests passed.
- git diff --check passed.
- Review-pattern rg scan found no remaining targeted patterns.
- WebUI compile/token sync passed with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000.
- Extension Chrome dev build/token sync passed; warnings were existing duplicate import/font/chunk-size warnings.

Known skips:
- Bandit skipped because this review pass touched frontend TypeScript/TSX and Backlog/docs metadata only; no Python code was touched.

Follow-up PR #1272 review pass addressed Qodo's two latest actionable findings: added an explicit @tldw/ui ./components/ui/state subpath export instead of changing imports because current frontend aliases resolve @tldw/ui to apps/packages/ui/src, and updated ServerAdminPage external documentation anchors to rel="noopener noreferrer".

Follow-up verification: focused Vitest for ConfigurationErrorScreen, ServerReadinessGate, and ServerAdminPage design-system tests passed with 3 files / 9 tests; git diff --check passed; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed with token sync OK; apps/extension bun run build:chrome:dev passed with token sync OK and only existing duplicate-import/font/chunk-size warnings. Bandit skipped because no Python code was touched.
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
