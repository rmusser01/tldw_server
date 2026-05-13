---
id: TASK-304.3
title: Implement Research Studio canonical route aliases
status: Done
assignee:
  - Codex
created_date: '2026-05-12 16:27'
updated_date: '2026-05-12 17:45'
labels:
  - implementation
  - research-studio
  - webui
  - extension
  - routing
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The WebUI has a canonical /research-studio page that renders the existing Research Studio surface
- [x] #2 /workspace-playground and /workspace-studio are compatibility aliases to /research-studio
- [x] #3 Route aliases preserve search params and hash state such as tab shared prefill and source-transfer data
- [x] #4 Shared UI and extension route registries expose the canonical route and retain legacy aliases
- [x] #5 Focused route and registry tests cover canonical route and aliases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Next route, route constants, RouteRedirect tests, shared route registry, and extension route registry.
2. Add failing route/registry tests for /research-studio and legacy alias query/hash preservation.
3. Implement canonical Next page, legacy redirect pages, shared route constant updates, and React Router alias preservation helper if needed.
4. Run focused route/registry tests and diff hygiene.
5. Update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the canonical /research-studio route by reusing the existing WorkspacePlayground surface and converting /workspace-playground plus /workspace-studio into compatibility redirects.

Added a shared React Router alias helper that preserves location.search and location.hash, then registered /research-studio as canonical in the shared UI route registry and extension route registry while retaining both legacy aliases.

TDD notes: initial route tests failed because the canonical page, legacy redirect page, route constants, registry aliases, and alias helper were missing. An earlier render-based registry test exposed an unrelated lazy OCR dependency import, so the focused tests were narrowed to static route and helper contracts.

Browser/CDP verification: with the local Next dev server on 127.0.0.1:3002, a temporary Playwright smoke passed for /research-studio, /workspace-studio, and /workspace-playground. The smoke validated that the pages compile and do not show the missing-route or module-not-found failures. Auth state can still send the canonical route to /login in this local environment, so the smoke did not assert authenticated Studio content.

Verification run:
- bun run test:run __tests__/navigation/route-redirect.test.ts __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/research-studio-route-files.test.ts __tests__/extension/route-registry.research-studio-alias.test.ts __tests__/extension/route-registry.workspace-playground.test.ts -> 5 files passed, 17 tests passed.
- bunx vitest run src/routes/__tests__/research-studio-route-alias.test.ts -> 1 file passed, 3 tests passed.
- TLDW_WEB_AUTOSTART=false bunx playwright test e2e/research-studio-route-smoke.codex-temp.spec.ts --reporter=line --workers=1 -> 3 passed.
- git diff --check -> clean.

Bandit was not run because this slice touched only frontend TypeScript, route registry tests, and Backlog metadata; no Python/backend code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added /research-studio as the canonical Research Studio route and retained /workspace-playground plus /workspace-studio as compatibility aliases that preserve search and hash state. Updated WebUI and extension route registries and added focused tests for the canonical route, redirect pages, and alias helper behavior.
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
