---
id: TASK-238
title: Add Persona Garden setup smoke coverage
status: Done
assignee:
  - Codex
created_date: '2026-05-10 17:04'
updated_date: '2026-05-10 19:02'
labels:
  - persona
  - buddy
  - stage-1
  - testing
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1533'
documentation:
  - Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next Stage 1 Persona/Buddy reliability slice from epic #1510. Focus on smoke-hardening the existing Persona Garden assistant setup path so route-level setup entry, resume, recovery, and handoff states are covered without changing the setup product flow. Keep this limited to Persona Garden/Buddy; do not add VN/CYOA behavior, native/background voice behavior, or new persona capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing Persona Garden setup entry/resume/finish paths have focused smoke coverage.
- [x] #2 Smoke coverage verifies setup handoff state and at least one recovery path using existing contracts.
- [x] #3 Implementation keeps normal healthy setup behavior unchanged and avoids new backend contracts.
- [x] #4 Focused frontend verification is run and unrelated baseline failures, if any, are recorded separately.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing Persona Garden setup tests and route/component boundaries.
2. Add the smallest failing smoke test for the uncovered setup path.
3. Patch test utilities or code only if the smoke test exposes a real gap.
4. Run focused Vitest/Playwright-appropriate checks plus diff hygiene and update tracker notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a route-level Persona Garden setup smoke test that starts at persona choice, retries a failed starter-template creation, completes safety/test steps, and verifies dry-run handoff state. The smoke exposed a real retry gap in SetupStarterCommandsStep: a failed template creation left the template selected, so the next click only unchecked it. Patched the component to treat selected-template clicks as retries while an error is visible, preserving normal unchecked behavior when no error is present.

Verification passed: bun run test src/components/PersonaGarden/__tests__/SetupStarterCommandsStep.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx --maxWorkers=1 (84 tests); git diff --check. Bandit skipped because touched files are TypeScript/Backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Persona Garden setup smoke coverage for the full persona-choice to dry-run handoff path, including starter-command failure recovery. Fixed the retry behavior so a selected starter template can be retried while an error is visible instead of only being unchecked. Verification: focused Vitest files passed with 84 tests; git diff --check passed; Bandit skipped for TypeScript-only changes.
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
