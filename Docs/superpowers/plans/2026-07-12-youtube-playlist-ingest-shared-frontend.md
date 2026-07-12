# YouTube Playlist Ingest Shared Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared WebUI/browser-extension Quick Ingest flow inspect every YouTube playlist before queueing and show every selected video through review, processing, recovery, and results.

**Architecture:** Keep one controller, contract client, queue model, and run-status adapter in `@tldw/ui`; WebUI and extension differ only in transport preference and runtime lifetime. Reuse the backend's version-2 preflight/run resources, TanStack Virtual, Dexie, Zustand, and the current Quick Ingest wizard. Delete the direct playlist-to-queue bypass instead of adding guards to each caller.

**Tech Stack:** React 18, TypeScript, Ant Design/design-system primitives, Zustand, Dexie, `@tanstack/react-virtual`, Vitest/Testing Library, Playwright, WXT extension runtime.

**Backlog:** `TASK-12111` (depends on backend contract task `TASK-12110`)

**Spec:** `Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md`

---

## File map

**Create**

- `apps/packages/ui/src/services/tldw/playlist-ingest.ts` — normalized version-2 HTTP models, cursor paging, run transport, and typed public errors.
- `apps/packages/ui/src/components/Common/QuickIngest/usePlaylistInspection.ts` — the single mandatory Add/Enter/extension-seed inspection controller.
- `apps/packages/ui/src/db/dexie/quick-ingest.ts` — compact session persistence, migration, retention, quota reporting, and single-writer lease helpers.
- `apps/packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts` — contract normalization, paging, run, SSE/poll, and error tests.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.playlist-ingest.test.tsx` — fail-closed shared entry tests.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.virtualization.test.tsx` — scale, selection, focus, and ARIA tests.
- `apps/packages/ui/src/store/__tests__/quick-ingest-indexeddb.test.ts` — migration, quota, cleanup, and multi-tab tests.
- `apps/tldw-frontend/e2e/workflows/quick-ingest-playlist.spec.ts` — WebUI journey against deterministic routes/fixtures.

**Modify**

- `apps/packages/ui/src/services/tldw/domains/media.ts` — expose version-2 playlist/run methods through the existing media-domain mixin on `TldwApiClient`.
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`, `server-capabilities.ts`, and their tests — allow and require the version-2 route/version signal.
- `apps/packages/ui/src/services/tldw/playlist-preflight.ts` — compatibility exports only; occurrence IDs come from the server.
- `apps/packages/ui/src/components/Common/QuickIngest/types.ts` — materialization, occurrence, lifecycle/outcome, run, and client-derived file reattachment types.
- `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx` — remove the playlist bypass; delegate all candidate handling to the controller.
- `apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx` — asynchronous summary, complete pagination, virtual rows, selection tools, and typed recovery.
- `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`, `ReviewStep.tsx`, and `ItemMetadataTable.tsx` — preserve identities, virtualize/filter Review, and submit Review overrides.
- `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx` — create/reconcile runs and bridge wizard state to shared services.
- `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`, `WizardResultsStep.tsx`, `FloatingProgressWidget.tsx`, and `QueuedFileRow.tsx` — truthful lifecycle/outcome UI, run cancellation, retry, and client-derived reattachment.
- `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts` and `quick-ingest-session-reattach.ts` — delegate version-2 runs to the shared run client; keep legacy handling only for non-playlist/old-server compatibility.
- `apps/packages/ui/src/store/quick-ingest-session.ts` — async IndexedDB-backed persistence with visible failures.
- `apps/packages/ui/src/db/dexie/schema.ts` and `types.ts` — version-15 `quickIngestSessions` table.
- `apps/packages/ui/src/entries/background.ts`, `entries/shared/quick-ingest-session-runtime.ts`, and `entries/shared/background-init.ts` — extension polling/reattachment over the same run contract.
- `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx` and tests — active-tab handoff only; no separate ingest logic.
- Existing Quick Ingest service/component/store/integration tests and English locale strings.

## Stage 1: Contract client and mandatory inspection

**Goal:** Make playlist classification and version-2 inspection one shared, fail-closed path.

**Success Criteria:** Add, Enter, WebUI paste, and extension seed cannot create an opaque playlist queue row; typed inspection state survives component rerenders.

**Tests:** Media-domain tests, capability tests, controller tests, AddContent integration tests.

**Status:** Not Started

### Task 1: Add version-2 client models and capability gating

- [ ] **Step 1: Write failing client/capability tests**

Extend `tldw-api-client.media-ingest.test.ts` and `server-capabilities.test.ts` to cover create/get/page/materialize/run/items/cancel/retry routes and `mediaPlaylistIngestContractVersion >= 2`.

```ts
it("requires the complete playlist ingest contract for version 2", async () => {
  mockDocsInfo({ capabilities: { mediaPlaylistIngestContractVersion: 2 } })
  mockOpenApiPaths([
    "/api/v1/media/playlist-preflights",
    "/api/v1/media/playlist-preflights/{preflight_id}",
    "/api/v1/media/playlist-preflights/{preflight_id}/items",
    "/api/v1/media/playlist-preflights/{preflight_id}/materializations",
    "/api/v1/media/ingest/runs",
    "/api/v1/media/ingest/runs/{run_id}",
    "/api/v1/media/ingest/runs/{run_id}/items",
    "/api/v1/media/ingest/runs/{run_id}/events/stream",
    "/api/v1/media/ingest/runs/{run_id}/cancel",
    "/api/v1/media/ingest/runs/{run_id}/retry",
  ])
  expect((await getServerCapabilities()).hasMediaPlaylistIngestV2).toBe(true)
})
```

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts ../packages/ui/src/services/__tests__/server-capabilities.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because the version-2 client and capability are missing.

- [ ] **Step 3: Implement the contract client**

Add exact server wire types and normalized camelCase types in `playlist-ingest.ts`. Use `bgRequest`/`bgUpload` through `domains/media.ts`; extend `openapi-guard.ts` for only the required paths. Keep cursor strings opaque. Map typed server codes to stable UI errors without exposing raw extractor output.

- [ ] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/services/tldw apps/packages/ui/src/services/__tests__
git commit -m "feat: add playlist ingest v2 client (TASK-12111)"
```

### Task 2: Route every playlist candidate through one inspection controller

- [ ] **Step 1: Write failing behavior tests**

Test mixed ordinary/playlist URLs, Add click, Enter, unavailable capability, inspection failure, multiple candidates with bounded concurrency, extension `playlist_preflight` seed, and a session duplicate index spanning queued direct URLs plus multiple playlists. Assert queue mutation never receives a candidate URL directly and Configure/Quick Process stays disabled while any candidate is unresolved.

```tsx
it("blocks Add until every playlist candidate is materialized", async () => {
  renderWizard({ urlInput: `${ordinaryUrl}\n${playlistUrl}` })
  await user.click(screen.getByRole("button", { name: /add urls/i }))
  expect(screen.getByText(/inspection required/i)).toBeInTheDocument()
  expect(queueItems()).toEqual([expect.objectContaining({ url: ordinaryUrl })])
  expect(queueItems()).not.toEqual(expect.arrayContaining([expect.objectContaining({ url: playlistUrl })]))
  expect(screen.getByRole("button", { name: /configure/i })).toBeDisabled()
})
```

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.playlist-ingest.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because `handleAddUrls` still queues the playlist directly.

- [ ] **Step 3: Implement `usePlaylistInspection` and delete the bypass**

The hook owns candidate records keyed by the original line, a maximum concurrent inspection count, resource polling, cancel/retry, seed handling, and a session duplicate index built from existing queue rows plus all loaded candidates. `handleAddUrls` delegates parsed lines once: ordinary lines become staged rows; playlist lines become inspection records. Enter calls the same handler. Candidate detection remains the small trusted-host helper already tested in `AddContentStep.tsx`.

- [ ] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/usePlaylistInspection.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__
git commit -m "fix: require playlist inspection before queueing (TASK-12111)"
```

## Stage 2: Complete virtual preview and stable queue identity

**Goal:** Show the complete ordered snapshot without unbounded DOM/network work, then queue server-authoritative occurrences.

**Success Criteria:** Every page loads exactly once, large lists stay virtualized/accessibly navigable, and Add materializes selected occurrence IDs before queue mutation.

**Tests:** Paging normalizer tests, virtual list tests, selection reconciliation, queue serialization tests.

**Status:** Not Started

### Task 3: Paginate and virtualize the preflight panel

- [ ] **Step 1: Write failing paging and virtualization tests**

Use a 500-item fixture. Assert cursors are followed until null, item order/occurrence IDs are stable, unavailable rows remain visible/disabled, Select all/none/new works, mounted rows remain bounded, keyboard focus survives scroll, and `aria-setsize`/`aria-posinset` are correct. With two playlists and a queued direct URL containing repeated videos, assert the first occurrence remains selected, later occurrences use session-level duplicate evidence, and explicit refresh reconciles selection by normalized source ID plus occurrence index among repeats.

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.virtualization.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because the panel eagerly renders the compatibility response.

- [ ] **Step 3: Implement bounded page loading and TanStack Virtual rows**

Poll summary until `ready`/blocked terminal state, then request pages sequentially by opaque cursor with an `AbortController`. Merge only by server `occurrenceId`; reject duplicates or count mismatch as `preflight_incomplete`. Reconcile the complete snapshot against the session duplicate index using normalized source ID plus repeat index, retaining unambiguous selections and surfacing reorder/add/remove ambiguity after explicit refresh. Use `useVirtualizer` with stable occurrence keys and overscan. Render ordinal/title first and channel/duration/availability/duplicate second; keep URL in details and do not load thumbnails by default.

- [ ] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS with a bounded mounted-row assertion.

```bash
git add apps/packages/ui/src/services/tldw/playlist-ingest.ts apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/packages/ui/src/services/tldw/__tests__
git commit -m "feat: show complete virtualized playlist previews (TASK-12111)"
```

### Task 4: Materialize occurrences and carry Review-time overrides

- [ ] **Step 1: Write failing queue/review tests**

Assert Add sends selected occurrence IDs, creates no rows on materialization failure, stores materialization ID/token plus occurrence ID, renders title/ordinal as primary text, and lets Review edit per-occurrence duplicate policy and only title/author/keywords-add metadata. Add 500-item tests proving both the queue and Review render bounded row counts with filters, and an expired-materialization test proving Start Processing requires reinspection rather than cached URL fallback.

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --maxWorkers=1 --no-file-parallelism --testNamePattern="playlist|materialization|review"`

Expected: FAIL because queue rows contain client-generated IDs and preflight URLs.

- [ ] **Step 3: Extend queue types and materialization handling**

Add a discriminated source reference to `WizardQueueItem`:

```ts
type WizardSourceRef =
  | { kind: "materialized_playlist_item"; materializationId: string; token: string; occurrenceId: string }
  | { kind: "direct_url"; occurrenceId: string; url: string }
  | { kind: "file_stub"; occurrenceId: string }
```

Playlist row `id` equals the server occurrence ID. Preserve compact title/playlist/ordinal/channel/duration display data, but never treat cached playlist URLs as authoritative after materialization expiry. Keep duplicate policy/metadata patch in Review state, not materialization state. Virtualize the Add-step queue and `ItemMetadataTable` with stable occurrence keys. Add queue filters for playlist/type/duplicate state and Review filters for selected/duplicates/policy; filters change visibility only, never selection.

- [ ] **Step 4: Build the exact Start Processing payload**

Serialize selected input records plus `review_overrides[occurrenceId]`. Include an explicit duplicate policy for every current duplicate and include a patch only for explicitly edited allowlisted fields. On backend `review_required`, merge refreshed duplicate evidence and return to Review without marking rows as submitted.

- [ ] **Step 5: Run tests and commit**

Run the command from Step 2 without the name filter.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/types.ts apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx apps/packages/ui/src/components/Common/QuickIngest/ItemMetadataTable.tsx apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__
git commit -m "feat: preserve playlist occurrence identity in review (TASK-12111)"
```

## Stage 3: Shared run submission and status transport

**Goal:** Replace WebUI/extension submission differences with one occurrence-aware run client.

**Success Criteria:** Both clients create the same run, submit bounded chunks, merge status by occurrence, and reattach without per-item polling fan-out.

**Tests:** Run client tests, ambiguous retry, dynamic events, extension runtime tests.

**Status:** Not Started

### Task 5: Implement the shared run client and bounded submission

- [ ] **Step 1: Write failing run-client tests**

Test run creation, Review-required response, processing-only chunk selection, structured partial acceptance, global stop with `Retry-After`, URL/file aligned arrays, server-returned authoritative URLs overriding cached queue display URLs, same-attempt ambiguous retry, and polling/SSE snapshots merged by occurrence ID.

```ts
expect(submitCalls[0].fields).toMatchObject({
  run_id: "run-1",
  occurrence_ids: ["occ-1", "occ-2"],
  attempts: [0, 0],
  urls: [video1, video2],
})
```

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism --testNamePattern="run|occurrence|chunk|ambiguous"`

Expected: FAIL because Quick Ingest submits/waits through legacy item logic.

- [ ] **Step 3: Implement one run client**

In `playlist-ingest.ts`, add `createRun`, `submitPendingChunks`, `getRun`, `listRunItems`, `streamRunEvents`, `cancelRun`, and `retryRunItems`. Use a conservative exported chunk-size constant. Build URL submission fields only from the authoritative processing occurrences returned by run creation/items, never from cached queue URLs. A run item is merged only by `occurrenceId`; state and terminal outcome are separate. Treat `resync_required` as a full summary/items reload.

- [ ] **Step 4: Delegate legacy services without duplicating logic**

In `quick-ingest-batch.ts`, detect version-2 run payloads and call the shared run client; retain the current legacy branch for non-playlist old-server sessions only. In `quick-ingest-session-reattach.ts`, prefer `runId` snapshots over job-ID fan-out. WebUI prefers SSE with polling fallback; extension supplies a polling preference.

- [ ] **Step 5: Run tests and commit**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts ../packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: PASS.

```bash
git add apps/packages/ui/src/services/tldw/playlist-ingest.ts apps/packages/ui/src/services/tldw/quick-ingest-batch.ts apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts apps/packages/ui/src/services/__tests__ apps/packages/ui/src/services/tldw/__tests__
git commit -m "feat: submit quick ingest through shared runs (TASK-12111)"
```

### Task 6: Make the extension runtime a thin transport adapter

- [ ] **Step 1: Write failing runtime parity tests**

Assert the background runtime stores only `runId`/compact mappings, polls the run endpoint after worker recreation, emits occurrence-aware events, delegates cancellation to run cancel, and never independently expands/classifies a playlist.

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/entries/shared/__tests__/quick-ingest-session-runtime.test.ts ../packages/ui/src/entries/__tests__/background.web-clipper.test.ts ../packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL on run reattachment/parity assertions.

- [ ] **Step 3: Update the runtime and active-tab handoff**

`ControlRow` continues to pass only typed open detail. The wizard controller owns inspection. `background.ts` calls the same run client with `transportPreference: "poll"`; remove its duplicate submit/poll loop for version-2 runs. Runtime context tracks `runId` and cancellation, not an in-memory job list as the recovery source of truth.

- [ ] **Step 4: Run extension tests and commit**

Run the command from Step 2, then run: `cd apps/tldw-frontend && bun run test:extension -- --run ../packages/ui/src/entries/shared/__tests__/quick-ingest-session-runtime.test.ts`

Expected: PASS.

```bash
git add apps/packages/ui/src/entries apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__
git commit -m "feat: align extension playlist run transport (TASK-12111)"
```

## Stage 4: Truthful lifecycle UI and durable local recovery

**Goal:** Render server evidence directly and survive reload/runtime loss without silently losing a large session.

**Success Criteria:** Queue/progress/results use occurrence identity and state/outcome axes; IndexedDB migration and failures are visible; file reattachment remains client-derived.

**Tests:** Component lifecycle tests, Dexie migration/quota/cleanup tests, multi-tab tests.

**Status:** Not Started

### Task 7: Update processing, cancellation, retry, and result groups

- [ ] **Step 1: Write failing lifecycle UI tests**

Cover `awaiting_upload`, client-derived file reattach, submit pending/queued/running, cancellation requested, status unavailable, all eight terminal result groups, real row/run cancellation, and retry reconciliation. Assert no fabricated analyzing/storing stage when the backend supplies only generic progress. At 500 items, assert bounded mounted rows and useful Active/Needs attention/Terminal plus outcome filters.

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because existing progress uses fabricated stage tokens and fewer outcomes.

- [ ] **Step 3: Replace UI status inference with run snapshots**

Extend `ItemProgress`/`WizardResultItem` with server lifecycle and terminal outcome. Derive `file_reattach_required` only when server state is `awaiting_upload` and no local `File` exists. Keep `status_unavailable` recoverable with Check again/Reconnect. Virtualize processing/results above the same scale threshold used by preview and preserve title/ordinal primary labels.

- [ ] **Step 4: Wire real cancellation and retry**

Before run creation, cancellation is local removal. After run creation, row cancellation always calls `POST /runs/{runId}/cancel` with that occurrence ID: the server terminalizes unsent items or cancels accepted jobs. Retain `cancellation_requested` until terminal. Whole-run cancellation calls the same route without occurrence IDs. Retry sends eligible occurrence IDs and waits for the server's reconciled attempt/action response.

- [ ] **Step 5: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
git commit -m "feat: show per-occurrence ingest lifecycle (TASK-12111)"
```

### Task 8: Move compact sessions to IndexedDB with visible recovery failures

- [ ] **Step 1: Write failing persistence tests**

Test Dexie v15 migration, one-time sessionStorage import, interruption idempotency, compact 500-item storage, exclusion of `File`/thumbnail bytes, quota/write failure surfaced in store state, active-run retention, terminal cleanup, and two-tab single-writer lease behavior.

- [ ] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/store/__tests__/quick-ingest-session.test.ts ../packages/ui/src/store/__tests__/quick-ingest-indexeddb.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because persistence is synchronous sessionStorage and ignores writes.

- [ ] **Step 3: Add the Dexie record and adapter**

Add `QuickIngestSessionDbRecord` and `quickIngestSessions: "id, lifecycle, updatedAt, expiresAt"` to schema version 15. `db/dexie/quick-ingest.ts` implements async Zustand `StateStorage`, migration marker, retention cleanup, and an atomic lease field. Persist compact display/mapping data only. Keep origins isolated naturally; do not attempt WebUI-extension IndexedDB sharing.

- [ ] **Step 4: Surface persistence and coordination state**

Add `persistenceStatus: "ready" | "migrating" | "unavailable" | "quota_error"` and `isSubmissionOwner` to the session store. Block Start Processing in non-owner tabs, allow takeover after lease expiry, and show a recovery warning when persistence cannot guarantee resume.

- [ ] **Step 5: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/db/dexie apps/packages/ui/src/store/quick-ingest-session.ts apps/packages/ui/src/store/__tests__
git commit -m "feat: persist quick ingest runs in indexeddb (TASK-12111)"
```

## Stage 5: Cross-client, accessibility, and release gates

**Goal:** Prove one behavior across WebUI and extension at realistic playlist scale.

**Success Criteria:** Focused suites, type/lint checks, deterministic WebUI browser journey, extension parity tests, and accessibility checks pass.

**Tests:** Vitest focused gate, Playwright journey, axe/keyboard/virtual-list tests, no fan-out structural assertions.

**Status:** Not Started

### Task 9: Add final browser journeys and verification

- [ ] **Step 1: Add deterministic browser fixtures and failing E2E**

Mock the version-2 API with the existing 34-item conference fixture. Test paste → inspect → complete preview → select → materialize → Review → run → progress → reload/reattach → results. Add an extension-level integration test for active-tab seed and background recreation. Do not hit live YouTube in required CI.

- [ ] **Step 2: Run the browser test and verify RED**

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/quick-ingest-playlist.spec.ts --project=chromium --reporter=line`

Expected: FAIL until route mocks and final UI wiring are complete.

- [ ] **Step 3: Complete route mocks, copy, and accessibility details**

Add English Quick Ingest strings for typed errors/actions. Verify live-region summary deduplication, virtual row position metadata, keyboard selection/focus recovery, no-referrer thumbnail opt-in, bounded mounted rows for 500 items, bounded chunk requests, and zero per-item status polling.

- [ ] **Step 4: Run the focused frontend gate**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts ../packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts ../packages/ui/src/store/__tests__/quick-ingest-session.test.ts ../packages/ui/src/store/__tests__/quick-ingest-indexeddb.test.ts ../packages/ui/src/components/Common/QuickIngest/__tests__ ../packages/ui/src/entries/shared/__tests__/quick-ingest-session-runtime.test.ts --maxWorkers=1 --no-file-parallelism`

Run: `cd apps/tldw-frontend && bunx tsc --noEmit`

Run: `cd apps/tldw-frontend && bunx eslint ../packages/ui/src/services/tldw/playlist-ingest.ts ../packages/ui/src/components/Common/QuickIngest ../packages/ui/src/store/quick-ingest-session.ts ../packages/ui/src/db/dexie/quick-ingest.ts`

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/quick-ingest-playlist.spec.ts --project=chromium --reporter=line`

Expected: all commands exit 0.

- [ ] **Step 5: Verify diff, update task, and commit**

Run: `git diff --check`

Record test counts, browser result, accessibility assertions, and any explicit skips in `TASK-12111`. Bandit is not applicable to this TypeScript-only plan; record the skip.

```bash
git add apps/packages/ui apps/tldw-frontend/e2e backlog/tasks
git commit -m "test: verify shared playlist ingest experience (TASK-12111)"
```
