# WebUI Perf Batch W4 — HTTP Architecture Implementation Plan

**Backlog task:** TASK-13524 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e`
**Findings:** WP-36..WP-55

**Goal:** HTTP latency stops being additive in N: batch what the server already supports, parallelize independent round-trips, and replace interval polling with the SSE streams that already exist.

**Architecture:** Client-only changes land here. Stages needing new server endpoints are marked **[BE]** — they are requests against the backend program's Batch 3 (N+1 batching, TASK-13515), not edits to `tldw_Server_API/**`. The repo's SSE infrastructure (`lib/sse.ts`, `subscribeNotificationsStream`, `subscribeResearchRunEvents`) is the target architecture for all polling stages.

Paths relative to `apps/packages/ui/src` unless noted; `FE` = `apps/tldw-frontend`.

---

## Stage 1: Extension quick-ingest batching (WP-36, WP-37, WP-38)

**Goal:** An N-item ingest batch stops costing ~3N+1 sequential round-trips.
**Files:** Modify `entries/background.ts` (URL loop ~3339-3445, file loop ~3447-3544, poll `pollIngestJobsForSession` ~2271-2276, conference item loop ~3062-3104, result PATCH loops ~2056-2072 and ~3552-3569, cancel loop ~2421-2431; anchors: `grep -n "pollIngestJobsForSession\\|patchConferenceCollectionItem" entries/background.ts`), `services/tldw/ingest-jobs-orchestrator.ts` (~217-218).
**Approach:**
1. Group entries with identical `buildFields()` output into single `/api/v1/media/ingest/jobs` submissions with `urls: [u1..uk]` (endpoint already accepts arrays); multi-file uploads append to one FormData via the existing `handleUpload` multi-file support (~1551-1555).
2. Heterogeneous groups: bounded concurrency pool (4) over `lease.upload` via `Promise.all` on chunks; keep per-entry progress by mapping returned job ids back to entry ids.
3. Poll rounds: `await Promise.all(Array.from(unresolved).map(fetchJob))` instead of sequential awaits.
4. Result PATCHes and cancel DELETEs: `Promise.all` (independent per item); prefer batch-cancel whenever any job carries a `batchId`.
5. Conference planning item POSTs: `Promise.all` with the same bounded pool until **[BE]** `POST .../collections/{id}/items/bulk` exists.
**Success Criteria:** 10-item quick-ingest: ≤ 4 submission requests (from 10), poll rounds fully parallel, terminal PATCHes concurrent; wall time for the W0 10-item baseline drops ≥ 60%.
**Tests:** Unit (mock `handleTldwRequest`): N same-config entries produce 1 job submission with k urls; poll round issues J concurrent gets (all inflight simultaneously — assert via deferred promises); PATCH fan-out concurrent. Extension e2e `queued-requests` spec stays green.
**Status:** Not Started

## Stage 2: Polling → SSE/visibility gating (WP-39, WP-40, WP-41, WP-42)

**Files / fixes:**
- `components/Common/PersonaBuddy/IndependentBuddyHost.tsx:225` (mounted globally from `FE/_app.tsx:516`): fetch once on mount; refresh on `visibilitychange`→visible and on focus; back off to 60s when the buddy panel is closed; no interval while `document.hidden` (`grep -n "setInterval(() => void refresh(), 5000)" components/Common/PersonaBuddy/IndependentBuddyHost.tsx`).
- `FE/pages/research.tsx:1028`: `refetchInterval: (q) => hasActiveRun(query) ? 5000 : false`, `refetchIntervalInBackground: false`; invalidate `['research-runs']` when the existing SSE `terminal` event arrives (the stream already carries the run payload).
- `FE/components/notifications/NotificationLifecycleProvider.tsx:391`: poll only while `streamOpenRef.current === false` (degraded fallback with backoff); clear interval in `onOpen`; move `hasMissingConfiguredAuth()` from render into a memo keyed on auth-scope changes (`grep -n "pollNotificationState" FE` path above).
- `components/Option/ResearchWorkspace/index.tsx` (two 5s pollers ~2101, ~3253): gate on `processingMediaIds.length > 0` and `!document.hidden`; single poller instead of two (`grep -n "WORKSPACE_SOURCE_STATUS_POLL_INTERVAL_MS" components/Option/ResearchWorkspace/index.tsx`).
**Success Criteria:** Idle visible tab on `/` + `/chat`: 0 background request baseline (from ~2-4/5s); hidden tab: 0 periodic requests.
**Tests:** Unit per component: fake timers — hidden document produces no polls; buddy panel closed produces no 5s polls; SSE terminal event triggers exactly one list invalidation; notification poll starts on stream error and stops on reopen.
**Status:** Not Started

## Stage 3: Media search correctness + cost (WP-43)

**Files:** Modify `components/Review/hooks/useMediaSearch.ts`.
**Fixes:** `Promise.all` the media + notes searches (independent); delete the duplicate mount-time `refetch()` (type-discovery effect ~892-967 calls it again at ~964 — keep exactly one initial run); derive media-type facets from the first search response instead of fetching pages 1-3 of `/api/v1/media/` (if incomplete facets matter, **[BE]** request a facets endpoint); `availableMediaTypes` as `Set` (landed in W3 Stage 5 if sequenced earlier).
**Success Criteria:** Page mount performs exactly 1 media + 1 notes request (from 2 searches + 3 discovery pages).
**Tests:** Unit: mount effect fires `refetch` once; facets derived from search response; parallelism asserted via deferred-promise spy.
**Status:** Not Started

## Stage 4: N+1 fan-outs (WP-44, WP-45, WP-46, WP-47, WP-48, WP-51)

**Files / fixes:**
- `components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx:346-395`: fetch task details lazily on row expand (first page only) instead of eager 2+N+M layers; merge tasks+details layers where possible.
- `components/Option/ResearchWorkspace/ChatPane/index.tsx:2240-2252` + `StudioPane/hooks/useArtifactGeneration.tsx:967-981`: reuse already-ingested source text from the workspace store by `(mediaId, version)`; fetch only misses; **[BE]** batch `POST /media/details/batch {ids}` as the eventual replacement.
- `components/Flashcards/hooks/useFlashcardQueries.ts:1205-1215`: **[BE]** `GET /flashcards/due-counts` aggregate; until then cache per-deck counts with react-query `staleTime: 60s` so N+1 happens once per minute, not per render.
- `components/Option/WorldBooks/Manager.tsx:271-288`: **[BE]** per-book stats endpoint; until then gate the fetch-all behind the stats panel being opened (not on page mount).
- `components/Option/KnowledgeQA/KnowledgeQAProvider.tsx:2130-2156`: accept the unsynced tail as one payload where ordering allows (**[BE]** if a server endpoint change is needed); minimum: parallelize the per-message rag-context writes that don't depend on parent chaining.
- `services/tldw/TldwApiClient.ts:4918-4943`: route character-list filtering through the existing `searchCharacters` (4948) instead of `listAllCharacters` fetch-all-then-filter (`grep -n "listAllCharacters" services/tldw/TldwApiClient.ts`).
**Success Criteria:** Opening ACP modal issues 1 + N requests (no eager detail layer); "ask with sources" with 5 selected sources issues 0 content fetches when text is already in the workspace store.
**Tests:** Unit per path with request spies; e2e workspace-parity suites stay green.
**Status:** Not Started

## Stage 5: Auth/config read caching (WP-52, WP-63)

**Files / fixes:**
- `FE/lib/authStorage.ts` + `FE/lib/api.ts`: cache the parsed effective config in a module variable; invalidate via the existing `storage` event / `auth-events.ts` listeners; compute auth headers once per config generation (`grep -n "getEffectiveStoredTldwConfig" FE/lib/authStorage.ts`).
- `services/tldw/single-user-credential.ts:394-448`: batch the 4-5 sequential reads into one `chrome.storage.local.get([k1,k2,k3])`; `waitForNewerCurrentAccessToken` (25ms poll, ~233-275): subscribe to `storage.onChanged` for the rotation key and resolve on event; decode each JWT once per call.
**Success Criteria:** A request burst of 50 calls performs ≤ 1 storage read + 0 polls for token rotation (event-driven).
**Tests:** Unit: header builder reads storage once across sequential calls until a storage event fires; rotation wait resolves immediately on the onChanged event (no polling loop iterations).
**Status:** Not Started

## Stage 6: Model-catalog fetch discipline (WP-49, WP-54)

**Files / fixes:**
- `components/Option/Playground/Playground.tsx:465-502`: drop `forceRefresh: true` on mount; migrate the hand-rolled `chatModelsCache` to react-query `useQuery(['chat-models'], …, { staleTime: 15*60_000 })` using the app-wide `QueryClient` (`grep -n "forceRefresh: true" components/Option/Playground/Playground.tsx`).
- `FE/hooks/useVlmBackends.ts:32-42`: react-query dependent query (`['rag/capabilities']` → enabled backends query) instead of uncached mount fetches.
**Success Criteria:** Navigating `/chat` → away → back re-issues 0 model-catalog requests within staleTime (from 1 full catalog + provider status per mount).
**Tests:** Unit: two mounts within staleTime produce one fetch (react-query test wrapper); explicit user refresh still forces.
**Status:** Not Started

## Deferred / parked

- WP-53 prompt bulk-actions server batch — parallel on the client already; **[BE]** follow-up.
- WP-55 `services/agent/agent-loop.ts` sequential tool execution — parked pending its own design discussion (behavior risk).

## Verification & DoD

- [ ] Each stage: failing test first, then implementation, scoped `vitest run` green; commit referencing TASK-13524.
- [ ] W0 page-load request-count baseline re-measured for `/` and `/chat`; delta recorded in the baseline doc.
- [ ] Extension e2e (`test:e2e:queued-requests`, `test:e2e:workspace-parity`) green; full `vitest run` in both `apps/packages/ui` and `apps/tldw-frontend`.
- [ ] **[BE]** items filed as notes on TASK-13515 (backend Batch 3) — no `tldw_Server_API/**` edits from this batch.
- [ ] `bun run lint` clean on touched paths.
