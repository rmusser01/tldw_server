---
id: TASK-12169
title: Fix Stage 6 Knowledge QA smoke harness failure
status: Done
labels:
- webui
- e2e
- smoke
priority: medium
modified_files:
- apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the Stage 6 interaction smoke failure where the Knowledge QA no-results test times out behind the Next dev overlay and never reaches the deterministic RAG assertion. Scope is limited to the smoke harness unless investigation proves a product bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the Stage 6 Knowledge QA smoke fixture to provide runtime-config single-user auth, return a valid `Helpful AI Assistant` character fixture, split the RAG stream and non-stream route mocks, and use keyboard activation for the fallback Ask action. Verified with focused Playwright, full Stage 6 interaction stage 2 Playwright file, and ESLint on the edited spec. Bandit not applicable because only a TypeScript E2E spec changed.
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
