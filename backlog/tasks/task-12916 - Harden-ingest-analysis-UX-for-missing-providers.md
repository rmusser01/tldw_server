---
id: TASK-12916
title: Harden ingest analysis UX for missing providers
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-08 03:14
labels: []
dependencies: []
priority: high
modified_files:
- apps/packages/ui/src/services/tldw/quick-ingest-batch.ts
- apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
- apps/packages/ui/src/components/Common/QuickIngestModal.tsx
- apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
- apps/packages/ui/src/components/Common/hooks/useIngestResults.tsx
- apps/packages/ui/src/components/Common/QuickIngest/types.ts
- apps/packages/ui/src/components/Common/QuickIngest/ResultsListItem.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/ResultsListItem.status.test.tsx
- apps/packages/ui/src/components/Media/AnalysisModal.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx
- tldw_Server_API/app/services/web_scraping_service.py
- tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address UX/HCI findings from the analysis-provider error investigation: prevent Quick Ingest from silently requesting analysis without a configured provider, avoid treating analyzer error strings as generated analysis content, and surface analysis skipped/failed states as recoverable warnings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick Ingest warns users when analysis is enabled without an analysis provider before work starts.
- [x] #2 Web ingest does not store analyzer Error: strings as analysis content.
- [x] #3 Ingest result normalization separates analysis warnings/errors from successful analysis text.
- [x] #4 Existing media Analysis modal keeps actionable error handling for missing provider failures.
- [x] #5 Focused frontend/backend tests cover the new behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the UX/HCI recommendations by adding an early Quick Ingest provider preflight for analysis, aligning the check with the backend api_name contract, and surfacing the warning in both the classic modal and wizard before work starts. Web ingest now records missing-provider and analyzer error-string outcomes as structured analysis_status/analysis_error warnings instead of returning Error: text as analysis content. Quick Ingest result normalization now suppresses analysis error markers from the analysis text slot and shows recoverable warnings in result rows. The media Analysis modal now maps the missing-provider failure to actionable retry copy.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quick Ingest now blocks analysis runs without an `api_name`, backend web-ingest analysis failures are represented as skipped/failed metadata instead of generated content, and result rows/modals show actionable recovery copy. Verification: Vitest focused suite passed 37/37 tests; pytest focused backend/provider suite passed 10/10 tests; `git diff --check` passed; Bandit reported 0 findings for `web_scraping_service.py`.
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
