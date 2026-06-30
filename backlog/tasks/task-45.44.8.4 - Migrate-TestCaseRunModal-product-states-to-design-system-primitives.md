---
id: TASK-45.44.8.4
title: Migrate TestCaseRunModal product states to design-system primitives
status: Done
labels:
- design-system
- webui
- product-state
- prompt-studio
priority: medium
parent_task_id: TASK-45.44.8
references:
- apps/packages/ui/src/components/Option/Prompt/Studio/TestCases/TestCaseRunModal.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.8 - Migrate-design-system-product-state-Prompt-and-Prompt-Studio.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate TestCaseRunModal run guidance and result status labels from AntD product-state primitives to the shared design-system Alert and Badge primitives, keeping the product-state guard baseline free of TestCaseRunModal entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TestCaseRunModal no longer imports or renders AntD Alert/Tag for product-state messaging.
- [x] #2 Focused Prompt Studio test-case modal coverage asserts the migrated design-system Alert and Badge markers render.
- [x] #3 The product-state baseline contains no TestCaseRunModal Alert/Tag exceptions, and `verify:design-system-state` passes.
- [x] #4 Verification, known skips, and Bandit disposition are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused design-system coverage for the run guidance alert and run result status badges.
- Replaced TestCaseRunModal AntD Alert/Tag product-state rendering with design-system Alert/Badge primitives while preserving existing Modal, Select, Spin, Table, Button, labels, and result summary behavior.
- Confirmed the rebased product-state baseline contains no TestCaseRunModal product-state entries.
- TDD red check: `bunx vitest run src/components/Option/Prompt/Studio/TestCases/__tests__/TestCaseRunModal.design-system.test.tsx --reporter=dot` failed because the run guidance and result labels were not rendered inside design-system markers.
- Verification: `bunx vitest run src/components/Option/Prompt/Studio/TestCases/__tests__/TestCaseRunModal.design-system.test.tsx --reporter=dot` passed 2 tests; Prompt Studio DS combined suite passed 3 files / 7 tests; `bun run verify:design-system-state` passed with no TestCaseRunModal baseline entries; `git diff --check` passed; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- PR review follow-up rebased onto latest `origin/dev`, removed the no-longer-needed design-system Alert import alias, and routed the run result status/summary labels through the existing `t()` translator while preserving default visible copy.
- Bandit not applicable: touched code is frontend TypeScript/TSX, JSON baseline, and Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated TestCaseRunModal run guidance and result status labels from AntD Alert/Tag to the design-system Alert/Badge primitives. Added focused render coverage, kept the rebased product-state baseline free of TestCaseRunModal entries, addressed PR feedback for import clarity and localized result labels, and verified focused tests, Prompt Studio DS tests, the product-state verifier, diff check, and package TypeScript.
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
