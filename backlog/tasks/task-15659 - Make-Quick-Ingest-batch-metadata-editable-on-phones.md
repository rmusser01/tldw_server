---
id: TASK-15659
title: Make Quick Ingest batch metadata editable on phones
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15658
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the responsive source editor from the full-modal review (#3633) on the WebUI from #3638.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Title, speaker, track, tags and inclusion for each source are labeled and reachable at 320 and 390 CSS pixels without horizontal scrolling.
- [x] #2 The desktop editor keeps its columns and each source retains its values and selection across layout changes and submission.
- [x] #3 Publish a PR against dev with desktop and phone screenshots and regression coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new architecture decision. The accepted review calls for a responsive presentation of the same source overrides. Draft persistence and API contracts stay the same.
Approved design in #3633: retain desktop comparison columns and use labeled per-source editors with visible inclusion on phones. Verify via Media → Add media and the submitted collection payload.
Implemented one keyed editor per source. The phone disclosure and inclusion label share a 44px header; each expanded field wraps within the modal. The desktop columns reuse those inputs. Red-to-green component regression and 497 passing scoped tests. Chromium at 320, 390 and 1280px verified geometry, inclusion, keyboard focus across reflow and the collection item payload. The two screenshot widths are 390 and 1280 pixels.
Verified the full Quick Ingest Chromium suite (16 passed), scoped Vitest suite (497 passed), frontend TypeScript, scoped ESLint, translation parity and diff whitespace. Bandit inspected the touched directories; there are no Python code changes. The WebUI source guide and published copy document phone editing. GitHub confirms both screenshot blobs and dev as the base of https://github.com/rmusser01/tldw_server/pull/3639.
Review found that source-only edits do not trigger a collection and native uploads drop metadata. Verify actual collection-item and file-job request payloads for those two supported inputs.
Follow-up saved per-source-only metadata with the default collection name and added durable item bindings for uploaded video files. Mixed URL/file requests retain URL-list binding semantics. A repeated file request reuses the owner-scoped job and original cancellation batch. Regression tests first showed zero planned items and HTTP 422 for a bound native upload. Final checks passed 501 UI tests, 35 API/collection tests (including an actual file job, repeat submit, worker and collection DB read), frontend TypeScript, scoped ESLint and diff whitespace. Bandit found no application issues; the existing B106 in the test fixture and three baseline Ruff import findings are unchanged. User guides updated. Existing ADR assessment still applies.
Resolve the cross-review failure path: a partial item plan must stop all ingest dispatches including shared-only edits, and its already-created rows must reflect the failure.
Also verify the owner and collection binding on URL retry replay, preserve an actual cancellation target when a grouped request reuses jobs, and retain unbound clients’ existing idempotency keys.
Partial shared-only planning reproduced two unintended uploads. After the correction 80 service tests passed, including both a rejected second row and an accepted second row whose response was lost. All confirmed pending rows become submit_failed and no source is dispatched. Frontend TypeScript, scoped ESLint and whitespace checks passed.
Bound URL retries now validate the authenticated owner, stored URL and collection item before acknowledgment. Intact replays reuse the original cancellation batch (including same-owner legacy raw keys); known split groups return 409 before new admissions. During a competing admission, already-accepted jobs have a truthful batch and rejected sources are identified individually. Unbound client keys retain their existing storage format. The original implementation reproduced a fresh batch ID on replay. Final verification passed 68 SQLite endpoint, worker and collection tests plus the two focused identity/race regressions after adding an unbound-key assertion. Bandit reports zero application findings; the test-only B106 and the three Ruff import findings match baseline. API guides and published copy agree. Existing Jobs/collection architecture and ADR assessment still apply.
The cross-review identified queue switching when a heavy worker appears. Verify both bound URL and native upload replay use the same job across availability changes, including persisted receipts from an older queue.
Bound URL and keyed upload retries now recover owner-scoped active or archived receipts across queues. They admit new work to the configured primary queue regardless of dynamic heavy-worker availability. Native replays still pass through durable upload admission and ambiguous identities leave the existing collection status intact. Real availability-toggle and lost-receipt-race controls initially produced duplicate native/URL jobs, and client/server conflict tests initially overwrote collection state. Verified 71 endpoint/worker/collection tests, 81 frontend transport tests, 57 SQLite Jobs idempotency tests, and a real PostgreSQL active/archive/ambiguous lookup test. ESLint and whitespace passed. Scoped Bandit found only existing Jobs B608 (8153967cc7) and the test-fixture B106. The three baseline Ruff import findings remain.
The requester included TASK-15659.1 in this PR. Repeating a committed setup request now reuses the authenticated collection and original source rows. Desktop and mobile browser tests preserved edited titles, and a real worker completed both original sources after a lost acknowledgement. ADR-103 documents the durable plan and source identity.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phone-width source cards expose title, speaker, track and tags and keep inclusion visible while collapsed. The desktop grid reuses the same inputs. PR #3639 includes desktop and mobile screenshots. TASK-15659.1 adds same-collection recovery after an interrupted setup, including committed-response-loss verification.
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
