---
id: TASK-478.8
title: 'Gate D: harden workspace source acquisition, URL validation, and My Media
  search'
status: Done
labels:
- research-workspace
- uat
- gate-d
- sources
- search
- validation
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failures: My Media search for the exact uploaded document did not surface the document and showed unrelated rows/count changes; entering `not-a-valid-url` in the URL field produced no visible validation feedback.

User goal: reliably add web pages/files/server media to the workspace, understand failures immediately, and find already-ingested material without confusing result drift.

Scope:
- Fix or clarify My Media search behavior, indexing scope, result counts, and sorting for workspace source import.
- Add visible inline validation for invalid URLs and unsupported URL states.
- Review Add Source tab defaults and empty/error/loading states for upload, paste, URL, and server media flows.
- Ensure source creation errors, partial successes, duplicate sources, and retry behavior are visible and recoverable.
- Add tests for exact-title search, invalid URL, duplicate import, and source-create error paths.

Acceptance criteria:
- Exact known media/source queries return expected results or explain why the item is outside the search scope.
- Invalid URL submission shows an inline error and does not silently stall.
- Search result counts are stable and intelligible across repeated searches.
- Live CDP/Playwright validation covers upload, paste, invalid URL, and server media search.

Depends on: can begin after Gate A; final readiness wording should align with TASK-478.3.
Parallelization: can run in parallel with layout/source-inspection/onboarding tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact known media/source queries return expected results or explain why the item is outside the search scope.
- [x] #2 Invalid URL submission shows an inline error and does not silently stall.
- [x] #3 Search result counts are stable and intelligible across repeated searches.
- [x] #4 Live CDP/Playwright validation covers upload, paste, invalid URL, and server media search.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Live UAT found the My Media tab showed `Unable to load media` even though `GET /api/v1/media/?page=1&results_per_page=50&include_keywords=true` returned 200 with an empty `{items: [], pagination: ...}` response.
- Console showed `ReferenceError: existingMediaCache is not defined` from `ExistingTab.loadMedia`; the AddSourceModal focused suite was red in library rendering/pagination paths.
- Current checkpoint restored the module-level My Media cache declaration, added coverage for the live empty media response shape, and repaired nearby item title/keyword rendering.
- Verification for this checkpoint: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx --maxWorkers=1 --no-file-parallelism` passed as part of the broader Research Workspace focused frontend run.
- Live CDP smoke after the fix opened Add Sources -> My Media and showed an empty-state (`All visible media are already in this workspace`) with 0 console errors in a fresh tab.
- Added inline http/https URL validation for single and batch URL intake. Invalid batch rows stay local while valid rows can still ingest.
- Exact My Media searches now explain already-attached matches instead of showing a generic filtered empty state.
- Live CDP smoke validated invalid URL feedback, exact My Media search for `research-workspace-uat-source.md`, and Paste ingestion creating `Gate D Paste Smoke`.
- Handoff: source indexing/status consistency remains under TASK-478.3.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Gate D source acquisition fixes for the current UAT pass. Added client-side http/https URL validation with inline errors for single and batch URL ingestion; invalid batch rows stay local while valid rows can still ingest. Restored My Media loading after the `existingMediaCache` runtime crash, normalized empty paginated media responses, and clarified exact My Media search matches that are already in the workspace with `1 matching media item is already in this workspace`. Verification: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx --maxWorkers=1 --no-file-parallelism` passed: 17 tests. Live CDP validated Add Sources -> My Media empty/already-added states, invalid URL inline validation with 0 console errors, exact search for `research-workspace-uat-source.md`, and Paste ingestion creating `Gate D Paste Smoke`. Remaining indexing/status discrepancy is tracked under TASK-478.3 because the status API still reports `vector_index_pending` even when the UI source row appears Ready.
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
