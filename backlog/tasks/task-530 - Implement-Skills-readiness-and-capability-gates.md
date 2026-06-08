---
id: TASK-530
title: Implement Skills readiness and capability gates
status: Done
labels:
- ux
- skills
- frontend
priority: High
ordinal: 531
modified_files:
- apps/tldw-frontend/components/networking/ServerReadinessGate.tsx
- apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx
- apps/tldw-frontend/e2e/ux-audit/knowledge-readiness-recovery.spec.ts
- apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/SkillsWorkspace.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first /skills UX remediation slice: make route readiness checks respect configured server URLs, expose one blocking recovery state, and keep Skills route capability messaging distinct after global readiness passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Readiness health checks use the persisted configured server URL when available instead of a stale module-load page origin.
- [x] #2 Readiness checks restart when the configured backend URL or config timestamp changes.
- [x] #3 Timeout recovery exposes one blocking recovery surface and does not mount route content behind it.
- [x] #4 SkillsWorkspace shows the manager for `connected_degraded` when Skills support is available, unsupported-capability recovery for `hasSkills: false`, and Skills-specific unreachable guidance for unreachable servers.
- [x] #5 Focused Vitest and Playwright verification results are recorded, including skipped backend-gated checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the reviewed readiness plan: add failing ServerReadinessGate tests for configured server origin and single blocking state; move readiness health URL resolution into component scope using persisted connection state; restart readiness checks when configured backend changes; stop mounting route children behind timeout recovery; harden SkillsWorkspace connection/capability branches; suppress unrelated splash overlays during readiness recovery; verify with focused Vitest and Playwright where available.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the `/skills` readiness/capability gate slice in branch `codex/skills-readiness-gates`. `ServerReadinessGate` now resolves the health endpoint at render time from the persisted connection `serverUrl`, falls back to the public API origin while config is unhydrated, restarts checks when the configured URL or timestamp changes, and no longer mounts route children behind the blocking timeout recovery panel. `SkillsWorkspace` now provides route-specific unreachable copy while still allowing `connected_degraded` plus `hasSkills` to show the manager. App shell overlay changes were not needed because `ServerReadinessGate` wraps `FirstRunGate` and blocking recovery now withholds children.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first /skills remediation slice. Added tests for saved backend URL readiness probes, fallback-to-config hydration, configured URL changes, single recovery surface/no duplicate main landmarks, degraded-but-usable Skills manager rendering, unsupported Skills capability recovery, and unreachable Skills-specific copy. Updated the knowledge readiness Playwright assertion to match the single blocking recovery state. Verification: ServerReadinessGate Vitest passed (10 tests); SkillsWorkspace Vitest passed (6 tests); knowledge readiness Playwright passed (2 tests); tier-5 Skills Playwright smoke ran and skipped 3 tests because its backend-availability guard skipped the suite; scoped git diff --check passed. Bandit skipped because this slice only touched frontend TypeScript/Playwright files.
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
