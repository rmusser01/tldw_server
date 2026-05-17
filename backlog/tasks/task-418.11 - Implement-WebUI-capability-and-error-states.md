---
id: TASK-418.11
title: Implement WebUI capability and error states
status: In Progress
labels:
- ux
- webui
- extension
- implementation
- states
priority: high
parent_task_id: TASK-418
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first WP2 capability/error-state remediation slice for the WebUI/extension. Scope: shared user-language capability states and first route adopters /sources, /scheduled-tasks, and /integrations. Preserve existing tables/forms/dense controls, keep raw endpoint/status details behind diagnostics, and do not change backend APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared state primitive expectations are locked with focused tests or documented as already covered.
- [ ] #2 A pure capability-state mapping helper is added only if two or more first-adopter routes would otherwise duplicate mapping logic.
- [ ] #3 /sources top-level unavailable/error/empty states use shared user-language state UI, with raw endpoint details only in diagnostics.
- [ ] #4 /scheduled-tasks top-level unavailable/error/degraded states use shared user-language state UI, with raw endpoint details only in diagnostics.
- [ ] #5 /integrations top-level unsupported/error states use shared user-language state UI, with provider-card details left scoped to cards unless they leak raw route state.
- [ ] #6 Focused Vitest route/component tests pass for changed state primitives and first adopters.
- [ ] #7 Browser QA or Playwright evidence is recorded for /sources, /scheduled-tasks, and /integrations, with any environment gaps documented.
- [ ] #8 Later route-family adopters are listed in the task notes instead of silently skipped.
- [ ] #9 No backend API changes or broad visual redesign are included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline: created clean worktree from `origin/dev` at `a1d24c7f4` after PR #1830 merge. Main checkout remains dirty and unrelated. The committed WP2 plan/spec are present; the referenced audit markdown is missing from the clean `origin/dev` worktree but exists as an untracked file in the main checkout, so this implementation branch treats the committed spec and child plan as authoritative and does not copy unrelated untracked audit artifacts into this PR.

Shared state foundation: `state-primitives.test.tsx` passed at baseline (7 tests), so no primitive production change was needed. Added a focused locking test proving raw endpoint details can live in diagnostics while primary copy stays user-language. Added pure `capability-state.ts` after a red test failed because the helper did not exist. Helper maps common capability failures to existing design-system state keys and builds diagnostics for method/endpoint/status/server/raw message. Verification: `bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts src/components/ui/state/__tests__/state-primitives.test.tsx` passed 12 tests.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
