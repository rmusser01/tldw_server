---
id: TASK-2318
title: Implement Skills readiness and capability gates on dev
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recreate the Skills readiness/capability fix from a clean dev branch. Scope is limited to the shared readiness gate, Skills workspace capability state, focused tests, and the readiness recovery E2E assertion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server readiness health checks use the persisted configured server URL when present instead of a stale module-level origin.
- [x] #2 Readiness timeout renders one blocking recovery surface without preserving hidden route content.
- [x] #3 Readiness checks restart when the configured server URL changes after hydration.
- [x] #4 Skills page remains usable in connected degraded state when Skills capability is available.
- [x] #5 Skills unreachable state uses Skills-specific recovery copy.
- [x] #6 Focused unit and E2E coverage verifies the changed behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Closed polluted stacked PR #2314 and recreated the change from a clean `origin/dev` worktree.
- Scope is limited to readiness gate behavior, Skills capability/unreachable state, focused tests, and this Backlog task.
- PR review follow-up normalized configured readiness URLs to origin-only health endpoints, added non-sensitive warnings for invalid/unsupported configured URLs, removed the redundant config timestamp effect dependency, and used timer return types compatible with the runtime timer APIs.
- Verification:
  - `git diff --check`
  - `apps/tldw-frontend`: `node_modules/.bin/vitest run components/networking/__tests__/ServerReadinessGate.test.tsx` (19 passed)
  - `apps/packages/ui`: `node_modules/.bin/vitest run src/components/Option/Skills/__tests__/SkillsWorkspace.test.tsx` (6 passed)
  - `apps/tldw-frontend`: `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' node_modules/.bin/playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts --reporter=line` (2 passed)
  - `apps/tldw-frontend`: `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' node_modules/.bin/playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --reporter=line` (3 skipped by backend availability guard)
- Bandit skipped because touched implementation files are frontend TypeScript/Playwright only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented clean Skills readiness/capability fixes on top of `dev`: origin-normalized configured server URL readiness probing, one blocking timeout recovery surface, URL-change retry coverage, degraded Skills access, Skills-specific unreachable copy, and review-requested readiness URL/error handling hardening.
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
