---
id: TASK-13198
title: Make Document Insights guidance customizable through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-06 02:44'
updated_date: '2026-09-06 03:00'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2914'
documentation:
  - Docs/Design/document-insights-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement approved bounded Document Insights slice using existing owner-scoped Service Prompts storage and shared Settings. Preserve locked structured-output contract, provider settings and content limits; include effective prompt fingerprint in the response cache key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared Settings exposes Document Insights guidance and supports existing save/reset workflow.
- [x] #2 Each insights request uses one authenticated-owner prompt snapshot with prompt-aware caching.
- [x] #3 Default messages, locked JSON contract, response normalization, provider configuration and content limits remain compatible.
- [x] #4 Tests cover customization, defaults/reset, owner isolation, cache changes and malformed output; lint, Bandit and review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Record approved design and baseline. Stage 2: Add failing public API/model-facing regressions. Stage 3: Implement registry, owner resolution, prompt-aware cache and shared metadata. Stage 4: Verify compatibility, security and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Approved bounded design implemented on isolated codex/document-insights-service-prompt from dev 5ed30d7683. Baseline: 10 passed. RED: 4 failed and 2 passed, demonstrating ignored saved guidance and stale cache reuse. GREEN/final: 111 focused backend tests passed, including 11 HTTP/storage/cache/model-boundary tests; shared Settings 77 passed, WebUI Settings 77 passed, prompt-service/transport 124 passed. Ruff check and format check pass, compilation passes, Bandit on both touched runtime files reports zero findings. OpenAPI export and fingerprint check pass unchanged; TypeScript schema generation passes. Export required existing repository package source paths on PYTHONPATH (no dependency change). Independent review found no actionable findings and independently passed all 11 feature tests. Full repository suite, full frontend typecheck and browser smoke were not run. Temporary dependency symlinks removed after verification; no new dependency. All four implementation stages complete; own temporary plan removed per repository convention. Implementation ready for integration choice; no PR created.

Published PR #2914 against dev at requester option 2. Implementation commit 78239bb760; verification recorded above applies to unchanged implementation. Worktree preserved for review follow-up. No merge or recurring monitor initiated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made Document Insights analysis and presentation guidance editable through the existing Service Prompts registry, owner storage and shared WebUI/extension Settings. Kept the structured-output instructions, request category/content carriers, provider controls and normalization unchanged, with exact default-message regression coverage. Resolve and close one owner snapshot on the same worker, then fingerprint the assembled prompt in the existing cache key so prompt edits, resets and in-flight changes cannot mix guidance. Reusing the existing cache and editor avoids a separate configuration system. Verified 111 backend and 278 frontend tests, clean Bandit/lint/format/compilation, unchanged API fingerprint and independent review.
<!-- SECTION:FINAL_SUMMARY:END -->

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
