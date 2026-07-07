---
id: TASK-12168
title: Implement character expression editor onboarding
status: In Progress
labels:
- frontend
- characters
- emotes
- implementation
priority: Medium
references:
- TASK-12167
- TASK-12164
- Docs/superpowers/specs/2026-07-07-character-expression-editor-onboarding-design.md
documentation:
- Docs/superpowers/specs/2026-07-07-character-expression-editor-onboarding-design.md
- Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
- backlog/tasks/task-12168 - Implement-character-expression-editor-onboarding.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved shared character expression image editor, chat discovery nudge, and browser-extension handoff from the 2026-07-07 design spec. Work should stay in shared WebUI/extension UI paths and reuse existing character metadata and avatar image handling patterns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md. Plan-document review loop completed. Reviewer iterations found and plan fixed: nudge dismissal server/user/character scoping, circular dependency between row helpers and character utils, and invalid raw extensions behavior for untouched empty starter rows. Final review status: Approved. Advisory starter-state load test was added to the plan.

Task 1 metadata helper contract completed in commit e5ec2d45cb. Added regression coverage for arbitrary safe expression state image keys, canonical mood image precedence over legacy aliases, canonical write cleanup, empty-map removal, and legacy alias resolution fallback. Verification: initial focused Vitest run failed on nested legacy alias cleanup and legacy resolver fallback; final `bunx vitest run src/utils/__tests__/character-mood.test.ts` from apps/packages/ui passed 11 tests. `git diff --check` passed. Bandit was attempted against touched TypeScript files with the project venv and produced no findings, with expected TypeScript parse errors because Bandit is Python-only.

Task 1 review fix: `upsertCharacterMoodImage` and `removeCharacterMoodImage` now normalize image-map keys with `normalizeCharacterEmoteState`, matching the arbitrary safe state contract. Added regression coverage for upserting/removing `smirk`. Red check failed because `smirk` upsert resolved to an empty string; final `bunx vitest run src/utils/__tests__/character-mood.test.ts` from apps/packages/ui passed 12 tests and `git diff --check` passed. Bandit was attempted against touched TypeScript files with the project venv and produced no findings, with expected TypeScript parse errors because Bandit is Python-only.
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
