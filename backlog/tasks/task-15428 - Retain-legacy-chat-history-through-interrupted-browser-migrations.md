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
November 7 requester explicitly authorized fixing remaining inherited/advisory CI failures. Investigate same-key presentation draft coverage race, reproduce delayed validation before storage persistence, and wait for the behavior the existing test asserts without changing product behavior, assertions, retries or timeouts. Validate committed shared-UI configuration, lint/typecheck and independent review before push. ADR not required for test synchronization with existing async persistence.
Presentation advisory repair: keep the displayed-editor wait and wrap the same SECOND_EDIT sessionStorage assertion in the existing waitFor, because SHA256 validation completes before onAcceptedChange persists the draft. A temporary controlled 50 ms digest delay reproduced the exact CI storage assertion failure before the change and passed afterward (2 selected cases; the initial existing load test warms the lazy Monaco import). All 111 tests in the unchanged workspace suite now pass, including same-key cleanup/recovery checks. Full WebUI typecheck with the standard 8192 MiB setting and scoped ESLint pass. Independent review confirmed unchanged assertions, timeouts, product code and post-reload recovery checks. Temporary reproduction copy and task-created shared-UI dependency symlink removed. ADR required: no; synchronizes a test with its existing asynchronous persistence contract. Bandit not applicable to this TypeScript test/task-only correction. Fresh hosted CI required after push; waiver retained.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recoverable history migration and the reviewed onboarding harness corrections remain intact. The authorized advisory follow-up also waits for the actual persisted same-key presentation draft before the test reloads, preserving all original assertions and timeouts. Controlled reproduction changes from the exact CI storage failure to success; all 111 workspace tests, full WebUI TypeScript, scoped ESLint and independent review pass. No product behavior changes in this follow-up. Temporary local reproduction artifacts within the worktree were removed. New-head hosted CI remains required; requester waiver retained, no merge or beta changes.
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
