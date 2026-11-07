---
id: TASK-15428
title: Retain legacy chat history through interrupted browser migrations
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Browser-local migration should make progress offline and recover from a partial write without losing either copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed copy leaves source data intact and does not mark migration complete.
- [x] #2 Offline migration can run; retries preserve existing destination content and ownership.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: reproduce partial writes and offline scheduling, make migration transactional and retryable, run focused tests and review. ADR not required; this repairs the existing local migration contract.
Reviewed destination ownership, transaction boundaries and browser storage errors. Existing scoped histories reject unseen messages and files. The legacy source remains on-device after successful copying as well as failures. The completion flag prevents ordinary repeat migration; copies already destroyed by past versions cannot be reconstructed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
38 tests passed across migration, hook lifecycle, prompt ownership, Firefox prompt ordering and legacy database coverage. WebUI typecheck passed. Touched-file ESLint reported zero errors. Headless Chromium with synthetic legacy input and real IndexedDB verified write rollback, concurrent retries, source retention and edited destination preservation. Native extension, full application build and real profiles were not exercised. Bandit applies to Python changes; this change is TypeScript only. Ready for requester-written change summary and merge review.
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
