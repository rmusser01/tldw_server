---
id: TASK-13522
title: 'WebUI perf batch W2: persistence write-path'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-2-persistence-write-path-implementation-plan.md. Scope: Request-history sidecar, workspace persist chain debounce+diff, ModelDb namespacing, legacy chrome.storage quarantine, session-state serialization, settings registry. WP-15..22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (WP-15 request-history sidecar) started: in-memory array + idle/trailing-debounce(500ms) flush + visibilitychange/beforeunload flush + redact/truncate at insert per Docs/Plans/2026-10-06-webui-perf-batch-2-persistence-write-path-implementation-plan.md Stage 1. ADR assessment: no new ADR needed - mechanical perf fix inside one lib module following the approved batch plan (no new durable architecture rule); governing plan referenced above.
Stage 1 (WP-15 request-history sidecar) COMPLETE (partial prior attempt audited, corrected, and finished): lib/history.ts buffers entries in a module array; redact + 2KB responseBody truncation applied exactly once at insert (historyInternals seam for the spy); flush = 500ms trailing debounce + requestIdleCallback (sync fallback under jsdom) + immediate flush on visibilitychange->hidden and beforeunload; MAX=200 newest-first unchanged; legacy un-truncated entries hydrate readable (migration-on-read kept, never re-truncated); new public flushRequestHistory() lands pending writes for readers/tests. W0 bench kept green and retargeted (100KB blob requestBody->responseBody, drives now-async flush, resets module buffer between runs, exactly-1-setItem assertion): post-fix shape 1 setItem/run, 115,651 bytes attempted=landed, 0 quota errors, 1.0x amplification (pre-fix: 50 calls, ~130.8MB attempted, 2 quota errors). Existing api-client.fetch.test.ts (8 red under async flush) fixed via helper flush only - assertions untouched. Verified: lib/__tests__ 84/84 green, scoped eslint clean, tsc --noEmit clean, Bandit n/a for TS-only scope (0 findings). Baseline-doc delta append belongs to the batch final task. Stages 2-6 remain.
Stage 2 sub-step 3 (artifact offload dedup, WP-16) committed as 55b070bb90, completing over a partial left by a prior session: offload stamps (WeakMap per adapter, record key -> content hash + updatedAt) skip re-puts of unchanged payloads and reuse the stored pointer timestamp for snapshot byte-stability; remaining puts parallelized via Promise.all; stamps invalidated on record deletion paths and removeItem. New tests workspace.persist-artifact-dedup.test.ts (3/3). Store __tests__ 368 pass / 1 documented pre-existing split-storage failure; tsc --noEmit 400 pre-existing errors, none new; W0 bench green untouched, numbers unchanged at this sub-step (jsdom lacks IndexedDB).
Stage 2 whole-stage review round 1 fixed in 426fc547af: coalescer pending slots keyed per storage key (cross-key writes no longer drop the buffered envelope); lifecycle flushes (pagehide/beforeunload/hidden) drain in durability mode that skips all IndexedDB awaits so localStorage lands synchronously, next pass re-offloads; literal 300ms trailing timer added (WORKSPACE_PERSIST_TRAILING_FLUSH_MS, renamed from COALESCE_WINDOW) alongside idle/microtask delivery; getItem comment corrected + prefill now flushes via exported flushWorkspacePersistence() before its verify-read (adapter stays storage-wins for reload/rehydrate); dedup invalidation (removed-then-re-added re-puts) and failed-put-retry tests added. Store dir 374/375 (1 documented pre-existing), bench untouched+green (numbers unchanged), tsc no-new.
Stage 3 (WP-17 + WP-61 alarm part) audit + completion: most of Stage 3 had already landed via TASK-13511 (ff8803237e: model: prefix + __tldwModelDbIndexV1 index + one-time idempotent legacy migration + batched createMany + warmModels Set + non-forced 6h alarm w/ isCatalogFresh gate). Remaining gaps closed this stage: ModelDb.replaceAll (one batched set incl. index + one batched remove, write-before-remove crash-safe, same-id rewrites preserved) and bulkAddModelsFB rewired to it (was 2 sets + 1 remove per pass). Tests-first: 6 new vitest cases in src/db/__tests__/models.storage.test.ts (100-model O(1) sync = 1 index read + 1 set, isLookupExist semantics, migration idempotency incl. crash-after-write tolerance, bulkAddModelsFB 1 set + 1 remove, replaceAll same-id), new TldwModels.warm-cache-ttl.test.ts (warm cache 0 fetches + TTL-bound refetch, fake timers), background-init cold-start warmModels(false) + alarm pin, and bun source guard apps/extension/tests/unit/modeldb-get-null-guard.test.ts (.get(null confined to migrateLegacyRecords). Verified: scoped vitest 21 pass (2 background-init OpenAPI-drift failures pre-existing at HEAD, shown via stash), bun guard 1 pass, packages/ui tsc 400 pre-existing/0 touched, extension compile green.
Stage 4 (WP-18 legacy chrome.storage blob DB quarantine) audit + completion: overlap check found the repoint largely pre-landed - messageHandlers already imported only a type from the legacy module (runtime graph fully Dexie; it no longer calls getPromptById/getSessionFiles at all), production callers of deleteHistoriesByDateRange (Sidebar, LocalChatList) and getPromptById already used db/dexie/helpers, and content search already had the Dexie fullTextSearchChatHistories. Remaining gaps closed: legacy deleteHistoriesByDateRange batched to ONE chrome.storage.remove([...ids]) + ONE chatHistories index rewrite (was per-history deleteMessage+removeChatHistory = O(N^2) index rewrites), legacy searchChatHistories reduced to titles-only (never deserializes per-history message blobs; zero external callers so no content-search rewiring was needed), module-header deprecation note with the Firefox-private-mode fallback method map, and the last legacy reference in messageHandlers repointed (type-only UploadedFile now from @/db/dexie/types). Tests-first: new legacy-blob-db.quarantine.test.ts (10 cases: exactly-one batched remove + one index rewrite, zero-write empty ranges, titles-only via getChatHistory spy + storage-key guard, legacy getPromptById/session-files oracle), helpers.repoint-characterization.test.ts (6 cases: Dexie getPromptById mirrors legacy for existing/missing/empty-id/soft-deleted, session-files shape, DatabaseClosedError->getPromptByIdFB chrome.storage fallback), messageHandlers.legacy-import-guard.test.ts (source guard: no legacy @/db import, Dexie helpers import pinned). Red confirmed on the 5 gap cases pre-implementation. Verified: scoped vitest db+handlers 321 pass + 3 pre-existing messageHandlers.edit failures (reproduced at HEAD with changes stashed), ChatSidebar suites 37/37, packages/ui tsc --noEmit 400 pre-existing errors with 0 in touched files, eslint N/A (no eslint config covers apps/packages/ui), Bandit N/A (TS-only scope). ADR assessment: no new ADR needed - quarantine step inside the approved batch plan (plan Stage 4), no new durable architecture rule.
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
