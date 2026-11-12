# WebUI Perf Batch W2 — Persistence Write-Path Implementation Plan

**Backlog task:** TASK-13522 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e`
**Findings:** WP-15..WP-22

**Goal:** Kill the whole-world read-modify-write persistence patterns that tax every keystroke, every HTTP response, every storage sync, and every ingest poll tick.

**Architecture:** Four independent fixes sharing one principle: write only what changed, when it's safe to write (debounced), in one batched storage operation. In-repo precedent for each exists (the coalescing session writer, Dexie bulkPut, zustand `partialize`).

Paths relative to `apps/packages/ui/src` unless noted.

---

## Stage 1: Request-history sidecar (WP-15)

**Goal:** Stop re-parsing/redacting/rewriting up to 200 full response bodies on every HTTP call.
**Files:** Modify `apps/tldw-frontend/lib/history.ts` (anchor: `grep -n "localStorage.setItem(KEY" lib/history.ts`), `apps/tldw-frontend/lib/api.ts` (call sites `recordSuccess`/`recordFailure`).
**Approach:**
- Keep entries in a module-level array; flush to localStorage via `requestIdleCallback` + trailing debounce (500ms), or `navigator.sendBeacon`-style on `visibilitychange`.
- Redact once at insert (`redactHistoryItem(item)` on the new entry only); never re-redact the array.
- Truncate stored `responseBody` to 2 KB; keep status/duration/url full.
**Success Criteria:** W0 Stage-2 bench: bytes-written per request drops from O(history) to O(entry); no synchronous `setItem` in the request path.
**Tests:** Unit: 50 sequential `addRequestHistory` calls with 100 KB bodies produce ≤ 1 `setItem` and stored bodies are truncated; redaction applied per entry exactly once (spy).
**Status:** Not Started

## Stage 2: Workspace store persist chain (WP-16)

**Goal:** A trivial `set()` no longer re-serializes every workspace, string-compares snapshots, and sequentially offloads artifacts.
**Files:** Modify `store/workspace.ts` (persist chain: `writeSplitWorkspacePersistence` at ~1523, per-workspace loop ~1588-1634, artifact offload ~1009-1031, `partialize` ~3908; anchors: `grep -n "writeSplitWorkspacePersistence\\|readWorkspaceSnapshotFromStorage" store/workspace.ts`).
**Approach (staged, behavior-preserving):**
1. Debounce the persist write 300ms trailing (cancel on unload → immediate flush). Reuse the coalescing-writer pattern from `entries/background-session-store.ts:352-382`.
2. Per-workspace diff: maintain `revision` counters in each workspace snapshot; skip re-serialize/compare for workspaces whose revision didn't change. Replace full-string compare with revision + `updatedAt` etag.
3. Artifact offload: skip artifacts already offloaded (key by artifact id + content hash); parallelize remaining puts via `Promise.all` over `indexedDbAdapter.putArtifactPayloadRecord`.
**Success Criteria:** Typing in a note with 10 workspaces × 50 artifacts: 1 workspace re-serialized, 0 unchanged-workspace reads, ≤ 12.5 persist passes/sec.
**Tests:** Unit: mutate workspace A; assert only A's snapshot key written (localStorage spy); artifact with unchanged hash not re-put; debounce collapses 10 rapid mutations into 1 write; unload flush preserves data.
**Status:** Not Started

## Stage 3: `ModelDb` storage hygiene (WP-17, WP-61 partial)

**Goal:** Model persistence stops scanning the entire `chrome.storage.local` area per operation.
**Files:** Modify `db/models.ts` (anchors: `grep -n "get(null\\|isLookupExist\\|bulkAddModelsFB" db/models.ts`), `entries/background.ts` (`warmModels` ~391-413), `entries/shared/background-init.ts` (hourly alarm, `MODEL_WARM_INTERVAL_MINUTES`).
**Approach:**
- Namespace keys `model:<id>` + one `model:index` lookup set; `getAll` becomes `get` over namespaced keys via the index.
- `createManyModels`: build the lookup `Set` once before the loop (as `warmModels` already does); batch writes into a single `chrome.storage.local.set({...})`; `bulkAddModelsFB`: batched `set` then one `remove([...])`.
- Hourly warm alarm: drop `force=true` so the 15-min TTL cache makes idle wakes no-ops (details also in W5 Stage 3).
**Success Criteria:** Creating 100 models performs 1 index read + 2 storage calls total (from ~200); hourly warm is a no-op when cache is fresh.
**Tests:** Unit (fake chrome.storage): createManyModels issues O(1) storage calls, all models persisted; isLookupExist true for existing, false for new; warmModels with warm cache issues 0 fetches.
**Status:** Not Started

## Stage 4: Legacy `chrome.storage` blob DB quarantine (WP-18)

**Goal:** Hot paths stop reading/rewriting whole collections; the legacy class remains only as the Firefox-private-mode fallback.
**Files:** Modify `db/index.ts` (anchors: `grep -n "searchChatHistories\\|deleteHistoriesByDateRange\\|getPromptById" db/index.ts`), `hooks/handlers/messageHandlers.ts:12` (legacy import).
**Approach:**
- Repoint `messageHandlers` imports to the Dexie implementations (`db/dexie/helpers.ts` `getPromptById` = `db.prompts.get(id)`; `getSessionFiles` equivalent).
- `deleteHistoriesByDateRange`: batch `chrome.storage.remove([...ids])` instead of per-history rewrite (O(N²) → O(N)).
- `searchChatHistories`: superseded by W3 Stage 1 Dexie search; reduce legacy implementation to titles-only (messages search routes to Dexie).
- Leave remaining legacy methods untouched (fallback-only surface); mark module header with a deprecation note.
**Success Criteria:** No production import path outside the Firefox-private-mode bootstrap reaches the legacy class; per-message mutations never rewrite a whole history array.
**Tests:** Unit: `deleteHistoriesByDateRange` removes exactly the selected ids with one batched remove call; `getPromptById` from Dexie path returns identical shape to legacy for a fixture prompt.
**Status:** Not Started

## Stage 5: Background session-state + settings registry (WP-19, WP-20, WP-22)

**Goal:** Ingest status ticks stop re-serializing the world; settings reads stop writing.
**Files:** Modify `entries/background.ts` (`persistSessionState` call on `emitIngestStatus` ~1043-1046; anchors: `grep -n "persistSessionState\\|serializeQuickIngestBatches" entries/background.ts`), `entries/background-session-store.ts`, `services/settings/registry.ts` (anchor: `grep -n "writeLocalStorageValue(setting, normalized)" services/settings/registry.ts`), `services/tldw/chat-request-debug.ts`.
**Approach:**
- Move serialization inside the coalescing writer so it runs only when a write drains; strip `result.data` bodies from `collectedResults` before persisting (keep id/status/error — all resume needs); move to per-funnel incremental keys `tldw:ingest:<funnelId>` with one small index key.
- Registry: cache values in a module `Map`; mirror to localStorage only when normalized value differs from cached; invalidate on `storage.onChanged`.
- `chat-request-debug.ts`: gate the `JSON.parse(JSON.stringify(body))` snapshot behind the debug-enabled flag.
**Success Criteria:** 100-item ingest run: per-tick persist bytes O(1) in batch size; `getSetting` 100× issues 0 writes after first mirror.
**Tests:** Unit: emitIngestStatus 20× with writer coalescing → ≤ 1 serialize per drain window; persisted record contains no `result.data`; registry read-then-read issues no second write.
**Status:** Not Started

## Stage 6: Deep-clone cleanup (WP-21)

**Files / fixes:** `store/workflow-editor.ts:327-337` undo history: `structuredClone` (available in all targets per repo browserslist) instead of JSON round-trip; `store/data-tables.tsx:384,577` same.
**Success Criteria:** No `JSON.parse(JSON.stringify(` in store mutation paths (`grep -rn "JSON.parse(JSON.stringify" store/` returns only test fixtures).
**Tests:** Existing workflow-editor undo/redo + data-tables suites green.
**Status:** Not Started

## Verification & DoD

- [ ] Each stage: failing test first, then implementation, scoped `vitest run` green, commit referencing TASK-13522.
- [ ] W0 Stage-2 bench re-run for WP-15/WP-16; delta recorded in the baseline doc.
- [ ] Full `apps/packages/ui` `vitest run` + `apps/tldw-frontend` `test:run` green; extension unit tests green (`vitest` in `apps/extension`).
- [ ] `bun run lint` clean on touched paths.
