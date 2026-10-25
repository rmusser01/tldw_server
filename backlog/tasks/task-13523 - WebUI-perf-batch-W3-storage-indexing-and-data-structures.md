---
id: TASK-13523
title: 'WebUI perf batch W3: storage indexing and data structures'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-3-storage-indexing-implementation-plan.md. Scope: Dexie index scans + bulkPut, Set/Map dedupes, keyset pagination, metadata counters, LRU caches, mechanical sweep. WP-23..35. Runs after W2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (WP-23, chat search index scan + Set dedupe) implemented on codex/webui-perf-w3-storage-indexing: added searchTokens side table (schema v17, additive, with streaming backfill upgrade), token-prefiltered content search in fullTextSearchChatHistories (no messages-table materialization; candidate-only verify via history_id index; streaming fallback for punctuation-only queries), Set-based dedupe replacing the O(n^2) findIndex merge, and token indexing wired into every message write path (addMessage, updateMessage, importChatHistoryV2, server mirror reconcile, fork commit, history-selection admits) with searchTokens added to the affected Dexie transaction table lists. Verified: 28 db suites (484 tests) + 5 Sidebar/harness suites (100 tests) green; FE typecheck exit 0; no new deps (memory-dexie harness extended with startsWithIgnoreCase/anyOfIgnoreCase). Stages 2-4 remain.
Fix round 1 after review: v17 upgrade handler now imports the tokenizer statically (no non-Dexie promise awaited inside the version-change transaction); helpers.ts restoreChat wired as the seventh token-indexed message write path (searchTokens added to its tx table list, bulk index after the adds) and pinned via the branch-projection undo test. Re-verified: 18/18 Stage 1 tests, full src/db 28 files/484 tests, FE typecheck exit 0, whitespace clean.
Stage 2 (WP-24) server-chat mirror reconcile: overlap check at HEAD confirmed all plan anchors live (recoverCorrelatedUsers current.some/filter, recoverAnchoredUsers scans, per-message get/put/add + final re-read in reconcileServerChatMirror); residual found beyond plan anchors: greeting incoming.some scan inside reconcileServerChatMessages local loop, hoisted. Characterization golden captured BEFORE refactor against the memory-dexie harness (interleaved fixture: correlation claim, anchored legacy user, version-guarded local edit, captured live row, synthetic reply id, absent remote, remote-wins edit; canonical JSON with sorted keys pins rows in Dexie index-scan PK order, localIds, steady-state second call, and the Stage-1 token side-table incl. stale extras) and held byte-identical across the refactor; new spy test pins one bulkGet + one messages.bulkPut + one searchTokens.bulkPut and zero per-message put/add. Refactor: Set/Map indexes in both recover* helpers, persisted/captured lookup Maps, one bulkGet existence read (cross-history guard preserved), single bulkPut, bulk token index, final all-messages re-read dropped in favor of constructed PK-sorted rows; owner contract (ownerKey/server_scope_key checks) untouched and pinned by existing suites. Test seams taught the new API: hand-rolled fakes in server-chat-mirror/server-chat-chronology tests gained bulkGet/bulkPut, useChatActions.character.integration bindMirrorStorage binds them and the mirror-rollback boundary test now hooks the first bulkPut instead of add; one mock assertion corrected to Dexie PK-scan order. ADR: none needed - mechanical O(n) refactor inside the existing WP-24 plan decisions, no new durable architecture rule. Verification: src/db 488/488 green; useChatActions.character.integration failures identical to HEAD (4 pre-existing); useMessage.history-selection (12), saved-normal (7), temporary-fork (5), history-selection (3), normalChatMode (90) failure sets byte-identical to HEAD via stash-diff - zero new failures; tldw-frontend 'bun run typecheck --incremental false' exit 0.
Stage 3 (WP-25..29, query-shape fixes across Dexie helpers) implemented. Overlap check at HEAD 4f0cbb30ba: (1) prompts deletedAt - index existed since schema v9, queries were JS scans -> GAP (query rewrite); key discovery: Dexie where().equals(null) throws and IndexedDB never indexes null, so only the trashed side (numeric deletedAt) is indexable - getDeletedPrompts/getSharedLocalPrompts(true)/emptyTrash/autoCleanupTrash now use where('deletedAt') ranges, non-deleted family keeps its output-proportional JS filter, countStoredPromptsForMigration -> count(); (2) metadata - GAP: schema v18 adds [history_id+createdAt] compound index (empty upgrade, additive; satisfies lazy-metadata ruling - no counters, no backfill, no staleness by construction): getHistoriesWithMetadata does per-history index-only counts + two-seek newest lookup (exact tie semantics = old PK-ordered reduce) in one read tx - no messages-table materialization on sidebar open; (3) pagination - PARTIAL: keyset variant existed dead at HEAD; Sidebar.tsx switched to it; rewrote it from the stateful fastForward predicate to an exact [createdAt+id] compound range keyset (v18) + search continuation by id + exact one-row-probe hasMore; offset variant retained - NOT caller-free (LocalChatList.tsx scoped on-device path + its tests still use it; scoped keyset is a different contract, out of scope); (4) nickname cache - GAP: module-level cache in db/dexie/nickname.ts, invalidated by saveModelNickname/importDataV2 and PageAssistDatabase.clear/clearDB, failures not cached, per-context limitation documented in comment; (5) drafts quota - GAP: getDraftAssetsTotalBytes and storeDraftAsset cap check now stream via each() accumulation (browser-wide cap semantics preserved, owner filter preserved). Characterization first: prompts trash + metadata outcome suites captured green at HEAD pre-rewrite; mechanism tests (no table materialization / no full-table JS scans) red at HEAD, green after; keyset suite pins no-dup/no-skip across inserts incl. exact next-30. Harness: memory-dexie schema mirrors v18; found+avoided harness deviation (filter predicates run in ascending order during materialization vs Dexie cursor-walk order - the dead fastForward shape cannot be expressed; pure range keyset sidesteps it). Test seams extended: chat.on-device fake gained compound where/between/above + first/last; three prompt fakes gained deletedAt ranges + count. ADR: none needed - query-shape work inside the plan's existing decisions; v18 follows the established additive-index pattern. Verification: src/db 34 files 525/525 green; 17 caller suites (617 tests) green; five known-red hooks suites failure sets identical to HEAD (normalChatMode.history-selection 90 stash-diffed byte-identical, useMessage 12, saved-normal 7, temporary-fork 5, character.integration 4); tldw-frontend typecheck exit 0; whitespace clean; lint N/A for packages/ui (no lint script; outside tldw-frontend eslint base). Stages 4-5 remain.
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
