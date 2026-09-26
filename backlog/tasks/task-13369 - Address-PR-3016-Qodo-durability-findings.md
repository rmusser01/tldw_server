---
id: TASK-13369
title: Address PR 3016 Qodo durability findings
status: In Progress
assignee: []
created_date: '2026-09-26 00:52'
updated_date: '2026-09-26 03:24'
labels:
  - vn-assets
  - review
  - durability
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3016'
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 3016 onto dev; address all eight Qodo review threads, verify worker and storage durability, then merge only after review and CI gates pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enqueue failure retries the original batch and deterministic parent Job.
- [x] #2 Storage handoff failures replay without terminalizing the variant.
- [x] #3 Concurrent deliveries use a fenced claim and cannot publish duplicate assets.
- [x] #4 Cancellation clears outstanding reservation capacity and preserves counters.
- [ ] #5 All remaining review comments are addressed with tests or reasoned thread replies.
- [ ] #6 PR checks and human summary gate pass before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_vn_pr_3016_review.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resumed after disk capacity was restored. Confirmed PR 3016 remains open and latest dev has four newer commits. Original eight Qodo threads remain outstanding. Independent review exposed missing-byte/quota accounting recovery and stale claim/publication windows; assigned non-overlapping Storage/AuthNZ and VN worker/DB implementers. Added and observed four new fencing regressions fail before implementation. Frontend verification underway; latest dev fetched; requester-provided Change summary preserved.

Frontend verification: 37 VN asset Vitest tests passed; TypeScript tsc --noEmit passed; scoped ESLint passed; 4 Chromium VN asset smoke tests passed in 58.7s (desktop/mobile and reload recovery). Temporary worktree UI dependency symlink removed after verification. Backend worker/storage fixes and independent re-review still underway.

Storage verification: 177 tests passed in sandbox; 15 PG cases initially skipped due Docker socket sandbox access. Required escalated official pg_temp_db run then passed all 15 PostgreSQL durability cases (100.53s), no PG skips. Independent final review found completed V1 cleanup fails because recipe item_id foreign key prevents item deletion after file cleanup; accepted as blocking and queued for final fix wave.

Final independent review (Rawls) verified three blocking regressions: completed V1 cleanup foreign-key failure; worker replay accepts invalid/truncated bytes and bypasses validation for attached planned items; definitive missing-slot 404 leaves browser pending receipt blocking future generation. Assigned one fresh implementer (Halley) all findings and full-suite failures for a final fix wave. Storage code finalized and PG15 cases passed. Worker code frozen while original full-suite report completes; not merge-ready.

Local implementation and independent re-review complete: all original eight Qodo findings implemented, plus three final-review regressions fixed. Final VN suite 361 passed; Storage/AuthNZ 177 passed; official required PostgreSQL 15 passed; adjacent tests 21 passed; frontend 44 passed; TypeScript/ESLint and four Chromium smoke tests passed. Production Bandit zero findings. Ruff only two pre-existing BLE001 warnings verified against baseline. Rawls independently passed 31 focused backend regressions and found final fix delta clean. Pending: commit/rebase, reply to external threads, fresh Qodo/CI gates, merge.

Rebased cleanly onto dev 3f909e133b and pushed head 87b808188f9599140e0f840da2f6b2e48da774e6. Source blobs unchanged by rebase; post-rebase 45 focused regressions passed. All eight original Qodo threads received individual evidence-backed replies and were resolved; Qodo edited summary 5836873877 marks all eight resolved and references current head. Requested fresh full Qodo review via comment 5842468613. Human requester Change summary preserved verbatim in PR body. Fresh GitHub CI remains queued and merge is blocked; AC6 and task completion remain pending. This tracking update is local pending final integration recording, not a new code change.

Fresh exact-head Qodo review 5324270842 completed at 02:48Z with eight NEW findings numbered 5-12: endpoint recovery ownership, AuthNZ SQL boundary, cleanup observability, PG fixture isolation, test tier/type annotations, checksum verification and missing-file retry behavior. Original eight resolved remain untouched. Reviewing new findings against current behavior and approved preservation/retry contracts before scoped fixes. CI is still queued; no merge attempt.

Fresh-review progress: Task8 implemented with DB_Management-owned VN lock/lookup SQL preserving transaction ownership; required shared isolated_test_environment now owns PostgreSQL durability cases. Agent verified all15SQLite+15PostgreSQL without skips,28 repository/boundary unit cases; productionBandit/Ruff clean. Task9 checksum and definitive-vs-transient failure handling implemented: completed approvals stay immutable, explicit regeneration produces a new draft, unfinished invalid variants fail once and release capacity. Main verified92focusedworker/helper tests,9integrity tests,4 checksum/transient regressions after3 valid checksum negative-control failures,1WorkerSDK nonretryable test; fullStorage142passed. Task7 core receipt workflow extraction and fullVN/independentreviews pending. No merge or new push yet.

Fresh eight findings implemented and independent re-review clean. Core receipt recovery now owns persistence; DB_Management owns new VN SQL on caller transaction; shared isolated_test_environment owns required PostgreSQL cases. Verification: full VN395 passed, final Storage/worker227 passed on previously failing seed, required SQLite15+PG15 passed without skips, AuthNZ28 unit passed, independent67 narrow cases passed. Explicit stat avoids Python3.14 predicate error suppression; off-loop assertions retained. Bandit8 productionfiles zero findings/errors; compileall/diffcheck pass; Ruff only2 verifiedbaselineBLE001. Missing/corrupt completed assets are nonretryable and preserve approvals; explicit regeneration creates new draft, no silent repair. Repository-wide attempt stopped at unrelated MCP flashcards KeyError rows, independently reproduced; source/test/exporter identicaldev, not whole-repo green. Dev advanced59bd584503 unrelatedMCP; final commit/rebase/push, new eight evidence replies, fresh Qodo and requiredCI remain pending. AC5/6 remain unchecked until actual external gates.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
