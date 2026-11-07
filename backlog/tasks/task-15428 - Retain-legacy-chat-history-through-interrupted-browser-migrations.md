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
UX CI follow-up authorized: run 42333366426 failed NE-03 before outline assertions because the first-visit prompt can disappear when note selection changes the query string after its two-second timer. Exact base 2f402bd7e34015ca6823793fd05cee68a16092b2 and head 72a22653053458b288301033bb11714775e6020c have identical prompt and E2E source; an isolated four-case base/head timer reproduction confirmed the inherited race. Plan: move the unchanged prompt assertion and dismissal before note selection in NE-03 and NE-06, validate both real extension cases on synthetic local state plus focused lint/typecheck, and obtain independent review before commit or push. No product changes or weaker assertions; ADR not required for test sequencing repair.
UX harness repair independently reviewed with no actionable findings: moved only the two notes-open-button clicks below the existing prompt assertion/dismissal blocks in NE-03 and NE-06. The current-head Chrome production extension built successfully. Actual extension NE-06 passed in 27.978s and NE-03 passed in 32.955s, each with retries=0, fresh browser profiles, unchanged fixtures/assertions, and synthetic local backend data. Used the committed Playwright config with --project=ux-regression --workers=1 --retries=0 and TLDW_WEB_AUTOSTART=false; direct selection was --grep="NE-(03|06): extension options", followed only by --grep="NE-03: extension options" after the first runner shut down its backend during NE-03 saved-content verification (ECONNREFUSED). Standard live-tier setup could not start unused Next WebUI because Turbopack rejects the existing external node_modules symlink. The final NE-03 lifecycle reused exported live-tier profile/command/process helpers, reserved fresh ports, verified backend health before/after plus isolated database paths and loopback-only requests, and stopped its services. Focused ESLint and full bun run typecheck --incremental false (the repository 8192MiB script) pass. Default-memory typecheck setup attempts were superseded by that successful standard-script run. No product source, assertions, timeouts, dependencies or build configuration changed. Full UX suite not rerun; Bandit not applicable to this TypeScript-only follow-up.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recoverable local migration preserves legacy history, destination ownership and edited content through failures and retries. All 38 focused tests pass under both committed Vitest configurations; shared-UI coverage passes all 6 migration tests. Full frontend typecheck and focused ESLint pass. Browser smoke verified migration rollback/retry behavior; actual Chrome extension NE-06 and NE-03 also pass with retries disabled after the inherited tutorial-dismissal race was repaired by ordering two existing test actions. The local full UX runner was blocked by Turbopack external dependency symlinks, so the two extension cases used the unchanged Playwright config and isolated synthetic backend directly. Full UX suite and real user profiles were not exercised. Independently reviewed; awaiting current-head required CI before dev integration.
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
