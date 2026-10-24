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
