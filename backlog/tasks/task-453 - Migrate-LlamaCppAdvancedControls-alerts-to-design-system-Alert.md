---
id: TASK-453
title: Migrate LlamaCppAdvancedControls alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 04:55
labels:
- design-system
- product-state
- ui
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1886
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the llama.cpp advanced controls grammar-support and extra-body-conflict notices from AntD Alert to the canonical design-system Alert while preserving message content and removing the matching baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LlamaCppAdvancedControls renders grammar support and extra body conflict notices through the canonical design-system Alert primitive.
- [x] #2 Existing notice title/message content is preserved for grammar unsupported and reserved extra body key conflict states.
- [x] #3 The two LlamaCppAdvancedControls Alert baseline exceptions are removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented test-first: the LlamaCppAdvancedControls regression renders mocked llama.cpp provider metadata and failed on zero canonical Alert markers before replacing the AntD Alerts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated LlamaCppAdvancedControls grammar-support and raw extra-body conflict notices from AntD Alert to the canonical design-system Alert. Added focused coverage for both notices and removed the two matching baseline entries, reducing product-state baseline exceptions from 330 to 328.

PR review follow-up:
- Removed the no-longer-needed DesignSystemAlert alias after AntD Alert was removed from the file and rendered the canonical primitive as Alert directly.

Verification:
- RED: bunx vitest run src/components/Common/Settings/__tests__/LlamaCppAdvancedControls.test.tsx --reporter=dot failed on zero data-ds-component="Alert" markers.
- GREEN: bunx vitest run src/components/Common/Settings/__tests__/LlamaCppAdvancedControls.test.tsx --reporter=dot passed.
- REVIEW FIX: bunx vitest run src/components/Common/Settings/__tests__/LlamaCppAdvancedControls.test.tsx --reporter=dot passed after the alias cleanup.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 328.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for LlamaCppAdvancedControls/task-453/baseline matched 0 lines.
- Bandit skipped because this slice changes TypeScript UI/test, JSON baseline, and task metadata only; no Python code touched.
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
