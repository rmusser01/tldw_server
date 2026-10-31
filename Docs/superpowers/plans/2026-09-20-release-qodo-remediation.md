# Release Qodo remediation — 2026-09-20

Tracking: TASK-13263.1, under TASK-13263. Original PR: #2761 (published 0.1.42); candidate: #2972; PyPI recovery: #2973.

Published 0.1.42 source and grants remain immutable. Server repairs land in the 0.1.43 candidate; cross-repository client repairs use an isolated Chatbook checkout. A resolved review finding means a verified fix, an already-landed fix, or an evidence-backed rejection, never merely an unread thread.

## Original-review stages (historical completion)

These stages describe the original review batch. The additional18-finding agentic batch has verified fixes/dispositions in the linked agentic ledger. All18 threads are answered/resolved; fresh complete inventories show no unresolved threads on original2761, candidate2972, recovery2973 or companion2763. Remote CI remains tracked under the parent release task.

## Stage 1: Inventory
**Goal**: Capture every Qodo finding.
**Success Criteria**: All 23 original threads represented; new and recovery PR reviews checked.
**Tests**: GitHub review/thread inventory.
**Status**: Complete

## Stage 2: Repairs and contract verification
**Goal**: Reproduce and repair valid findings; verify disputed claims against implementation and approved specifications.
**Success Criteria**: Each ledger row has evidence.
**Tests**: Focused regressions for affected behavior; scoped Bandit.
**Status**: Complete

## Stage 3: Integration and release records
**Goal**: Review combined repairs and regenerate the candidate protected-source manifest.
**Success Criteria**: Tests, type checking and source verification pass; release plan and PRs reference final commits.
**Tests**: Release contracts, targeted suites, TypeScript, package build, Bandit.
**Status**: Complete

## Stage 4: Review closeout
**Goal**: Reply to Qodo threads and check new PR review feedback.
**Success Criteria**: Every thread has an accurate disposition and link; remaining external release gates are explicit.
**Tests**: Fresh GitHub review and CI inventory.
**Status**: In Progress

## Findings

| # | Finding | Disposition and evidence |
|---|---|---|
| 1 | [Credentials can hijack upstream sessions](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308619) | Fixed in 7da1f8e67e: shared brokered-credential validation rejects case-insensitive transport-owned headers before dispatch in HTTP and SSE. Reserved headers and normal credentials are covered by 74 passing transport/WebSocket tests. |
| 2 | [Revoked recipients retain clone access](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308620) | Verified intentional contract. Owner-bound receipts and operation status survive revocation; they do not authorize new work. Added revoked/clone-disabled tests proving replay + polling succeed while a fresh key is denied and no second Job exists. Existing worker revocation tests pass. |
| 3 | [Users can read another dataset's links](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308621) | Not reproducible: the authority resolver permits exactly one canonical default personal dataset and rejects all other explicit IDs before lookup. Expanded route tests cover graph/list/detail rejection; real Sync authority tests pass. Adding dataset columns would contradict the current owner-bound product projection. |
| 4 | [Deleted links disappear from sync](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308623) | Fixed: include_deleted now includes deleted endpoints in link-store pagination. Real SQLite regression covers two tombstones across pages and default exclusion. |
| 5 | [Failed runs remain stuck as running](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308625) | Fixed: terminal run persistence failures propagate before notifications and are retryable in the actual worker. A failed durable write cannot be acknowledged as successful. |
| 6 | [One scheduled slot can run twice](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308626) | Fixed after independent overlap probes: atomic non-stealable execution claims fence slots, incoming Jobs leases are verified, and busy attempts never acknowledge/release an active lease. Explicit executor Future completion guards claim release even under repeated cancellation. Uncertain crashed/interrupted claims require verified-stopped operator reconciliation; see Automation_Admin_Operations. |
| 7 | [Malformed values can authorize purges](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308630) | Fixed: strict request booleans and purge generation reject malformed primitives before invoking services, while preserving JSON timestamp/array contracts. |
| 8 | [Health checks reject connected clients](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308636) | Rejected security-weakening recommendation. The cited unauthenticated client is a mocked unit test; production TLDWAPIClient supports API keys/bearer tokens. Public /health is minimal liveness; diagnostics require system.logs. Added connected-client guidance to Long_Term_Admin_Guide. All 37 permission tests pass. |
| 9 | [Readiness becomes unknown to clients](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308641) | Fixed: /api/v1/health/ready retains ready, engine, db and timezone-aware time alongside sanitized operator fields. Both ready/not-ready regressions pass without weakening authorization. |
| 10 | [Workspace cloning rejects client requests](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308644) | Fixed in [Chatbook PR2763](https://github.com/rmusser01/tldw_chatbook/pull/2763), commit3cbf5effab: canonical request/header, complete operation receipts, scoped polling and stable authenticated-account retry identity survive response loss and panel remount. Explicit new-copy action retires only selected intent; quota never evicts uncertain keys. |
| 11 | [Shared sources cannot be parsed](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308645) | Fixed in Chatbook PR2763: source-page metadata and canonical source_id/origin_url survive parsing; legacy aliases remain; list convenience traverses pages and rejects stalled pagination. |
| 12 | [Connected clients cannot delete links](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308646) | Fixed in Chatbook PR2763: selected dataset/version/idempotency/reason pass through Notes scope, service and HTTP client. Stale409/missing428 stay visible; no fetch-latest bypass. |
| 13 | [One websocket test lacks its marker](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308600) | Fixed in 7da1f8e67e: integration category marker retained alongside asyncio; category collection selects the websocket case. |
| 14 | [A shutdown test depends on internals](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308605) | Fixed in 7da1f8e67e: public protocol write and server.shutdown prove execution-before-module-teardown ordering; all 137 extraction contract tests pass. |
| 15 | [Draft recovery text cannot localize](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308610) | Fixed: presentation recovery, conflict, status and action text uses playground locale keys; pending messages update with locale changes. Included in 222 passing focused frontend tests. |
| 16 | [Cancelled refresh continues the request](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308612) | Fixed: propagate cancellation during refresh fetch or body read rather than consuming original 401. Both abort variants pass. |
| 17 | [Shared speech bypasses its wrapper](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308614) | Fixed: speech uses wxt/browser with an extension-runtime guard, preserving WebUI speechSynthesis fallback. Wrapper-only and fallback tests pass. |
| 18 | [Login logic remains in the page](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308617) | Already fixed in candidate: pages/login.tsx re-exports @web/routes/login; both login suites pass. Next-specific imports remain in the web package. |
| 19 | [Macro retries can duplicate replies](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308627) | Fixed: message, final-run marker and metadata share one outer transaction on the same verified database instance; injected marker failures roll back visible output. |
| 20 | [Successful batch ingests look incomplete](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308632) | Fixed: scan all batch results and identifier aliases for first valid persisted ID. Failed/skipped first item and invalid ID regressions pass. |
| 21 | [One owner's jobs reorder another's](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308633) | Fixed: owner-scoped SQLite chatbook acquisition applies the same owner predicate to the scheduled ordering heuristic. |
| 22 | [Malformed upstream replies appear valid](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308634) | Fixed in 7da1f8e67e: shared validator requires jsonrpc=2.0 on HTTP JSON, matching HTTP SSE events and legacy SSE replies. Wrong/missing versions fail; unrelated malformed notifications remain ignored. |
| 23 | [Webhook error bypasses core module](https://github.com/rmusser01/tldw_server/pull/2761#discussion_r4001308597) | Fixed in 7da1f8e67e: central WebhookKeyError retains compatibility exports and consumer behavior; 206 webhook tests pass. |

## Confirmed contract decisions

- Finding 2: The approved `2026-08-25-shared-workspace-clone-jobs-design.md` explicitly requires recipient-owned operation receipts/status to survive share revocation. Replays do not admit new work; workers recheck live share/clone authority before copy and publication. Verify endpoint and worker regressions before disposition.
- Finding 3: `resolve_notes_link_dataset_authority` selects exactly one active default personal Chatbook dataset and rejects any different explicit ID. Product link rows live in the owner-bound database. Verify real authority and route tests before disposition.
- Finding 8: The cited credential-free client is a unit test that mocks `_request`; the real client supports both X-API-KEY and bearer credentials. Operator diagnostics deliberately require SYSTEM_LOGS; `/health` is the public liveness route. Preserve this security boundary and document it for clients.
- Finding 18: Current candidate `pages/login.tsx` already delegates to `@web/routes/login`. Verify current login regressions.

## Release gates retained

PR2973 waiver was explicitly approved by the requester after the earlier automatic rejection. It merged at cd2dbc792b and publication retry35540174556 is running. PR2972 legal dates remain proposed, and its final CI/review remain open. These gates do not prevent Qodo repairs.

## Verification evidence

- 82 backend tests passed across clone endpoints/worker, Notes authority/links, and readiness; added disposition canaries: 5 passed. Parent final link-store/readiness run: 25 passed.
- Health control-plane/sanitizer tests: 13 passed after fixing test log capture to attach after application logging startup. The initial failure was a removed test log sink, not leaked diagnostic data.
- Health authorization matrix: 37 passed, including anonymous denial, unprivileged denial, and SYSTEM_LOGS/admin access.
- Frontend: 222 tests passed; full TypeScript check passed. Presentation test hook naming was corrected to satisfy React Hooks lint; its 111 tests passed again.
- Parent touched production Python Bandit: no findings. Test-only scan excludes B101 assertions; three unchanged B106 synthetic token/password fixtures remain (access/hash), with no new secret finding.
- Cross-repository findings 10–12 tracked in Chatbook TASK-32881 at isolated `/private/tmp/tldw-chatbook-release-compat`; 143 focused tests pass; two mounted panel tests independently rerun. Companion PR2763 is reviewable; no client release is claimed.

## Integration verification

- Transport/webhook group: 417 tests passed, Ruff clean, no new Bandit findings (three unchanged B110s). Detailed report: [transport](../reviews/2026-09-20-release-qodo-transport.md).
- Frontend: 222 tests passed; 111 presentation tests rerun after correcting test hook lint naming. Full TypeScript passed again; all touched-file ESLint has zero errors and existing warnings. Detailed report: [frontend](../reviews/2026-09-20-release-qodo-frontend.md).
- Scheduled/macro/Jobs/Personal Context: 115 unaffected targeted tests plus the final 69 scheduled consumer/database tests passed. All seven touched production modules have zero Bandit findings. Independent review reproduced and closed both lease-replacement and repeated-cancellation overlap cases. Detailed report: [Jobs](../reviews/2026-09-20-release-qodo-jobs.md).
- Release verification found an omitted existing Whisper path-resolution test in all five explicit CI media shards. Added that path and updated the shard contract; all 133 release/docs/workflow tests pass afterward.
- Refreshed protected source: `89cf5448b1888ceefb1ea68bdc6fe18700e81808`, 7,321 files; manifest SHA-256 `ce4bd9dc1854b5550fa4de12b3eca5b8998d3b0bd33e4b459e11b5d528efabd7`. Explicit checkout verification: 12 passed. Published 0.1.42 grant/tag unchanged.
- Strict MkDocs build passes after curated documentation refresh. Candidate wheel/sdist, Twine and backend-only package checks passed after repairs.
- Recovery PR2973 exact-head CI is now entirely green/CLEAN. Explicit requester waiver is pending; no merge or PyPI dispatch has occurred.

## Fresh candidate review

Qodo review on candidate head8edcb4fae9 raised two Notes focus areas. The existing selected-owner helper emits the same required-owner predicate, so no broader shared/legacy access is introduced. PostgreSQL search now follows that helper consistently. The duplicate-ID concern is a real error-classification bug: only the notes identifier collision should map to that conflict; graph projection uniqueness failures must remain database errors. Fixed in2241ea23e8: 47 focused tests pass, source Ruff/Bandit pass; the additional live PostgreSQL test skips through the official unavailable fixture. See [Notes report](../reviews/2026-09-20-release-qodo-notes.md).

Chatbook PR2763 closes the three cross-repository findings with 143 focused tests, zero Bandit findings across ten production files, and zero new Ruff diagnostics relative to its baseline. Full ToolsSettings tests still encounter an existing pre-panel configuration-bootstrap failure; the scoped mounted panel tests pass. App restart recovery requires a caller-persisted request/receipt lifecycle; panel memory is app-lifetime only. Detailed [client report](../reviews/2026-09-20-release-qodo-chatbook.md) and [independent review](../reviews/2026-09-20-release-qodo-chatbook-independent.md).

The license gate passed, but its completed workflow event cancelled parallel PR jobs through their shared concurrency groups while LICENSE_FIRST_CI_ENABLED was unset. Replacement workflow-run jobs were skipped. Final-head PR jobs must be rerun after all license-gate metadata events finish; cancelled jobs are not passing tests.

All 23 original Qodo threads now have published dispositions and are resolved. Candidate review findings are implemented; fresh final review and CI are being checked. Additional CodeQL alert2693 is repaired in7bd99f427d: buffered character streams use a constant error with 129 owning-module tests passing and independent security review complete. See [stream report](../reviews/2026-09-20-release-buffered-stream-error.md).

Latest Qodo review at10b11f6acb raised keyword and lifecycle PostgreSQL boolean literals plus relationship endpoint ownership. All three are already enforced by the shared PostgreSQL SQL transformer and dual endpoint EXISTS owner predicates. Added executable driver-boundary and malformed-link canaries:46 passed; live PostgreSQL11 skipped through official unavailable fixtures. See [second Notes report](../reviews/2026-09-20-release-qodo-notes-second.md). CodeQL test-fixture dispositions2688–2692 are published at https://github.com/rmusser01/tldw_server/pull/2972#issuecomment-5752715588 and dismissed as used in tests.

## Current agentic-review batch — in progress

The deeper review returned 18 additional findings after the original threads were closed. Every item is tracked in [the agentic review ledger](../reviews/2026-09-20-release-qodo-agentic.md), including source-authority reconciliation, account isolation, retry persistence, OSCE controls, coverage reporting and host-test dispositions. Finish fixes, independent boundary review, manifest refresh and final CI before treating this batch as closed. Earlier approvals/pending-source statements in this document are historical checkpoints; PR2973 approval is resolved and its publication retry is running.
