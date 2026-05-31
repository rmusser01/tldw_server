---
id: TASK-401
title: Polish chat cockpit status and recovery states
status: Done
assignee: []
created_date: '2026-05-16 02:30'
updated_date: '2026-05-16 02:42'
labels:
  - chat
  - cockpit
  - webui
  - ux
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unrelated degraded server health appears as a warning without blocking chat.
- [x] #2 Streaming, loading, missing-model, degraded, error, and unavailable status strip states have clear priority and copy.
- [x] #3 Disabled or blocked cockpit controls expose a concrete recovery reason.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-chat-cockpit-error-degraded-recovery-polish.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Impeccable PRODUCT/DESIGN preflight unavailable in this worktree; proceeding from existing cockpit roadmap/spec and design-system conventions.

Verification: Vitest focused status/cockpit suite passed 24 tests via bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx.

Verification: real-server Playwright passed 9/9 via e2e/workflows/chat-cockpit.real-server.spec.ts against http://127.0.0.1:8000 and http://localhost:8080 with no mocked routes.

Verification: git diff --check passed. Targeted ESLint completed with 0 errors and existing warnings in legacy touched files. Bandit not applicable because this slice changes frontend TypeScript and docs only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Polished main /chat cockpit status and recovery states. Degraded unrelated health now remains warning-only with chat-available copy, streaming stays primary over degraded warnings, missing model and context-loading states have explicit recovery/status copy, and blocked server readiness now surfaces as chat-critical unavailable state in both the footer and runtime rail. Added focused Vitest coverage and extended the real-server Playwright cockpit proof.
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
