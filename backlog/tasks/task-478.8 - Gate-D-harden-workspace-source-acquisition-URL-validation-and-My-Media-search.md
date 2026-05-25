---
id: TASK-478.8
title: 'Gate D: harden workspace source acquisition, URL validation, and My Media
  search'
status: To Do
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
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
