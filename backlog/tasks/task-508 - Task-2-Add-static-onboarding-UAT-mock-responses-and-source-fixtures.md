---
id: TASK-508
title: 'Task 2: Add static onboarding UAT mock responses and source fixtures'
status: Done
labels:
- onboarding-uat
- fixtures
- test
priority: medium
modified_files:
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/hosted-success.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/chat-fail-once.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/model-unavailable.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/default.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/source-summary.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/embeddings/default.json
- apps/tldw-frontend/e2e/fixtures/media/onboarding-uat-note.md
- apps/tldw-frontend/public/e2e/onboarding-uat-research-note.html
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add static mock_openai_server configuration/response fixtures and synthetic first-source media fixtures for the onboarding UAT harness, with focused fixture validation tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented static onboarding UAT mock_openai_server configs/responses and synthetic first-source Markdown/HTML fixtures. Added focused Vitest fixture validation covering config presence/shape, model lists, scenario_failures, secret-marker checks, stable chat/source-summary text, embedding shape, and source fixture content. Verification: `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts` from `apps/tldw-frontend` passed with 1 file / 4 tests. `git diff --check` passed. Bandit skipped because the touched implementation scope is frontend/static JSON, Markdown, HTML, and TypeScript test fixtures with no Python source changes.
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
