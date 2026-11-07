---
id: TASK-15428
title: Retain legacy chat history through interrupted browser migrations
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/3431
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
CI follow-up: reproduced two migration.retry.test.ts failures with the committed shared-UI Vitest config. The browser fixture lacks chrome.storage.sync, required by real Plasmo Storage for model nicknames; previous frontend-config checks used its shim. Complete the fixture and ensure failure assertions exercise intended branches; run both configurations, focused lint and typecheck, and independent review before pushing.
CI fixture fix and independent review: added the absent chrome.storage.sync.get API and retained it during the injected browser-read failure. Strengthened history failure checks to assert actual migration errors and session failures to identify the correct phase; kept a runAllMigrations rejection assertion after review. Both committed shared-UI and frontend Vitest configurations pass all 38 focused tests. Shared-UI v8 coverage mode passes all 6 migration tests. Touched-file ESLint, git diff --check and canonical task-format validation pass. No production files changed; Bandit is not applicable to this TypeScript test-only follow-up. Full frontend typecheck is being verified after restoring the local @tldw/api-types workspace dependency link.
Final CI-fix verification: full frontend tsc --noEmit --incremental false passes after restoring the local workspace dependency link. Independent review accepted the fixture fix and stronger failure checks with the public-entry rejection assertion retained. The requester waived the human-written Change summary requirement for this PR; recorded in PR 3431 by the coordinating agent. Ready to publish the reviewed follow-up and await current-head required CI; no merge or auto-merge authorized.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
38 focused tests pass under each committed shared-UI and frontend Vitest configuration; shared-UI coverage mode passes all 6 migration tests. Full WebUI typecheck passes. Touched-file ESLint reports zero errors. CI failures came from the test fixture omitting chrome.storage.sync; the fixture now supports real Plasmo Storage and all failure scenarios verify their intended errors. Prior headless Chromium smoke with synthetic legacy input and real IndexedDB verified write rollback, concurrent retries, source retention and edited destination preservation. Native extension, full application build and real profiles were not exercised. Bandit applies to Python changes; this work is TypeScript only. Independently reviewed; awaiting current-head required CI before dev integration.
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
