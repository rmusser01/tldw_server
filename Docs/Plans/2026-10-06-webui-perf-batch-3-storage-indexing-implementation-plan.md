# WebUI Perf Batch W3 — Storage Indexing & Data Structures Implementation Plan

**Backlog task:** TASK-13523 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e`
**Findings:** WP-23..WP-35
**Runs after:** Batch W2 (shared `src/db/**` files — sequential, never parallel)

**Goal:** IndexedDB queries use indexes and bulk operations instead of JS full-table scans; O(n²) dedupes/matches become `Map`/`Set`/`bulkPut`; pagination becomes keyset.

**Architecture:** Dexie schema version bump adding indexes (client-side only — no backend migration); query rewrites against `where(...)`; a repo-wide mechanical sweep for `.find`/`.filter`/`.includes` inside loops.

Paths relative to `apps/packages/ui/src` unless noted.

---

## Stage 1: Chat search — index scan + Set dedupe (WP-23)

**Goal:** `searchChatHistories` stops materializing every message row and doing O(n²) dedupe.
**Files:** Modify `db/dexie/chat.ts` (search ~141-171; anchor: `grep -n "self.findIndex" db/dexie/chat.ts`), Dexie schema in `db/dexie/db.ts` (or wherever `stores()` versions live).
**Approach:**
- Add `messages: 'id, history_id, &[history_id+server_message_id]'`-style index if absent (check current schema version first; bump with an upgrade handler — empty upgrade for pure index additions).
- Title search: existing indexed `title` query (keep). Content search: `db.messages.where('history_id').anyOf(candidateHistoryIds)` when titles prefilter; otherwise maintain a small side-table `searchTokens(word, history_id)` populated on message write (tokenize on write, query by word prefix — O(matches) instead of O(table)).
- Replace `self.findIndex` dedupe with `new Set(ids)`.
**Success Criteria:** Search over 50k messages touches ≤ candidate rows; zero full `.toArray()` of the messages table in the search path.
**Tests:** Unit: seed 100 histories × 100 messages; search matches only relevant histories; result ids unique; bench: `db.messages.toArray` not called (spy on Dexie table).
**Status:** Not Started

## Stage 2: Server-chat mirror reconcile — Maps + bulkPut (WP-24)

**Goal:** Reconciliation becomes O(n) with batched reads/writes.
**Files:** Modify `db/dexie/server-chat-mirror.ts` (anchors: `grep -n "acknowledgedRows.find\\|current.some\\|current.filter" db/dexie/server-chat-mirror.ts`).
**Approach:**
- `recoverCorrelatedUsers` / `recoverAnchoredUsers`: build `Map<serverId, row>` / `Map<canonicalId, msg>` once; replace `some`+`filter` scans with map lookups.
- Main loop: replace per-message `await db.messages.get(id)` with one `bulkGet(ids)`; collect `next` rows; single `bulkPut`; drop the final all-messages re-read by returning constructed rows.
**Success Criteria:** 500-message reconcile: ≤ 3 IDB round-trips (1 bulkGet + 1 bulkPut + acknowledge write), linear comparisons.
**Tests:** Unit (fake-indexeddb): reconcile fixture with interleaved server/local messages → identical outcome to current implementation (characterization test first, then refactor); spy asserts bulkPut used.
**Status:** Not Started

## Stage 3: Query-shape fixes across Dexie helpers (WP-25..WP-29)

**Files / fixes (anchors in parens):**
- `db/dexie/chat.ts:469-483,542,551-553`: add `deletedAt` index; prompts queries become `where('deletedAt').equals(null).reverse().sortBy('createdAt')`-family (`grep -n "prompts.filter" db/dexie/chat.ts`).
- `db/dexie/chat.ts:211-285`: `getHistoryMetadata` → `where('history_id').equals(id).count()` + `.last()`; `getHistoriesWithMetadata` → maintained `messageCount`/`lastMessageAt` counters on `chatHistories` (updated in message add/update/delete paths), falling back to count queries until backfilled.
- `db/dexie/chat.ts:411-427`: switch `Sidebar.tsx:181` consumer to the existing keyset `getChatHistoriesPaginatedOptimized` (429-466); delete the offset variant once no callers remain (`grep -rn "getChatHistoriesPaginated(" --include="*.ts*"`).
- `db/dexie/chat.ts:124-135`: cache `getAllModelNicknames()` map in memory, invalidated via a Dexie liveQuery or a bump-counter written alongside nickname mutations (`db/dexie/nickname.ts`).
- `db/dexie/drafts.ts:104-116`: quota sum via `db.draftAssets.each()` accumulation (no blob materialization) or a maintained `meta.totalBytes` record (`grep -n "toArray()" db/dexie/drafts.ts`).
**Success Criteria:** Sidebar open on 200 histories: no full messages-table load; prompt list query uses the index (Dexie `explain`-style assertion or spy).
**Tests:** Per fix: characterization unit test on fake-indexeddb; keyset pagination returns stable pages across inserts (no duplicates/skips).
**Status:** Not Started

## Stage 4: Store/API cache hygiene (WP-30, WP-31)

**Files / fixes:**
- `store/folder.tsx:376-391`: diff-sync by `last_modified` (already fetched) — upsert changed rows, delete vanished ids; fallback path (429-434) loads only when the diff is unresolvable (`grep -n "folders.clear()" store/folder.tsx`).
- `services/tldw/TldwApiClient.ts:1676-1685`: LRU-cap `characterCache`/`chatMessagesCache` (simple `Map` + insert-order eviction, cap 100); evict expired entries on a 60s timer rather than read-path; cap per-chat entries at 5 query-shapes (`grep -n "characterCache = new Map" services/tldw/TldwApiClient.ts`).
**Success Criteria:** Folder refresh with no changes issues 0 bulk writes; 30-min session memory for caches bounded (~100 entries).
**Tests:** Unit: LRU evicts oldest beyond cap; expired entries not returned; folder diff applies only changed rows (storage spy).
**Status:** Not Started

## Stage 5: Map/Set mechanical sweep (WP-32, WP-35)

**Files / fixes:**
- `components/Option/KanbanPlayground/BoardView.tsx:335,367`: `new Map(l.cards.map(c => [c.id, c]))` before id→card maps (`grep -n ".find((c) => c.id === id)" components/Option/KanbanPlayground/BoardView.tsx`).
- `components/Review/hooks/useMediaSearch.ts:396-399,449-452,497-500`: `availableMediaTypes` as `Set` (`grep -n "availableMediaTypes.includes" components/Review/hooks/useMediaSearch.ts`).
- `db/dexie/chat.ts:842-863` `importSessionFilesV2`: pre-index `mergedFiles` by id (O(n²) → O(n)); `db/dexie/nickname.ts:41-50` sequential puts → `bulkPut`.
- `apps/tldw-frontend/pages/research.tsx` `textToStringList`: `new Set` dedupe (if not already landed in W1 Stage 5).
**Success Criteria:** `grep -rn "\.find(.*=>.*\.id ===" components/ hooks/ db/ | grep -v test` shows no in-loop occurrences; import paths linear.
**Tests:** Existing suites for each touched module green; import fixture test asserting merged file count correct.
**Status:** Not Started

## Deferred (documented in index, no work)

- `services/settings/local-bucket.ts` cleanup parallelization; `companion-home.ts` rebuild dedupe; flashcards panel virtualization.

## Verification & DoD

- [ ] Each stage: characterization test → refactor → scoped `vitest run` green; commit referencing TASK-13523.
- [ ] Dexie schema bump additive-only; fresh-install + existing-DB upgrade paths both covered by unit tests.
- [ ] Full `apps/packages/ui` `vitest run` green; `bun run lint` clean on touched paths.
- [ ] Sidebar open / chat search bench numbers recorded in the baseline doc.
