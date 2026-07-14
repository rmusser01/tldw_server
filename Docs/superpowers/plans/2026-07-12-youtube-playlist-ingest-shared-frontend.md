# YouTube Playlist Ingest Shared Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared WebUI/browser-extension Quick Ingest flow inspect every YouTube playlist before queueing and show every selected video through review, processing, recovery, and results.

**Architecture:** Keep one controller, contract client, queue model, and run-status adapter in `@tldw/ui`; WebUI and extension differ only in transport preference and runtime lifetime. Reuse the backend's version-2 preflight/run resources, TanStack Virtual, Dexie, Zustand, and the current Quick Ingest wizard. Delete the direct playlist-to-queue bypass instead of adding guards to each caller.

**Tech Stack:** React 18, TypeScript, Ant Design/design-system primitives, Zustand, Dexie, `@tanstack/react-virtual`, Vitest/Testing Library, Playwright, WXT extension runtime.

**Backlog:** `TASK-12113` (depends on backend contract task `TASK-12112`)

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

**Status:** Complete

### Task 1: Add version-2 client models and capability gating

- [x] **Step 1: Write failing client/capability tests**

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

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts ../packages/ui/src/services/__tests__/server-capabilities.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because the version-2 client and capability are missing.

- [x] **Step 3: Implement the contract client**

Add exact server wire types and normalized camelCase types in `playlist-ingest.ts`. Use `bgRequest`/`bgUpload` through `domains/media.ts`; extend `openapi-guard.ts` for only the required paths. Keep cursor strings opaque. Map typed server codes to stable UI errors without exposing raw extractor output.

- [x] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/services/tldw apps/packages/ui/src/services/__tests__
git commit -m "feat: add playlist ingest v2 client (TASK-12113)"
```

Verification before commit: initial RED was 11 failed / 49 passed; review-remediation RED was 8 failed / 68 passed. Final focused Vitest passed 76/76, ESLint exited 0 with no errors, Prettier and `git diff --check` passed. Full TypeScript checking remains blocked by the unrelated repository baseline after the required three-attempt audit. Specification and code-quality re-reviews both approved the final Task 1 diff.

### Task 2: Route every playlist candidate through one inspection controller

- [x] **Step 1: Write failing behavior tests**

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

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.playlist-ingest.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because `handleAddUrls` still queues the playlist directly.

- [x] **Step 3: Implement `usePlaylistInspection` and delete the bypass**

The hook owns candidate records keyed by the original line, a maximum concurrent inspection count, resource polling, cancel/retry, seed handling, and a session duplicate index built from existing queue rows plus all loaded candidates. `handleAddUrls` delegates parsed lines once: ordinary lines become staged rows; playlist lines become inspection records. Enter calls the same handler. Candidate detection remains the small trusted-host helper already tested in `AddContentStep.tsx`.

- [x] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/usePlaylistInspection.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__
git commit -m "fix: require playlist inspection before queueing (TASK-12113)"
```

Verification before commit: behavior RED was 13 failed / 7 passed; Strict Mode seed review RED was 1 failed / 25 passed; quality-remediation RED was 7 failed / 24 passed. Final focused Vitest passed 31/31. Targeted ESLint exited 0 with no errors, the two new files passed the frontend Prettier configuration, and `git diff --check` passed. Full TypeScript was not rerun because the repository-baseline audit reached its three-attempt cap in Task 1. Specification and code-quality re-reviews both approved the final Task 2 diff. Bandit is not applicable to this TypeScript-only task.

## Stage 2: Complete virtual preview and stable queue identity

**Goal:** Show the complete ordered snapshot without unbounded DOM/network work, then queue server-authoritative occurrences.

**Success Criteria:** Every page loads exactly once, large lists stay virtualized/accessibly navigable, and Add materializes selected occurrence IDs before queue mutation.

**Tests:** Paging normalizer tests, virtual list tests, selection reconciliation, queue serialization tests.

**Status:** Complete

### Task 3: Paginate and virtualize the preflight panel

- [x] **Step 1: Write failing paging and virtualization tests**

Use a 500-item fixture. Assert cursors are followed until null, item order/occurrence IDs are stable, unavailable rows remain visible/disabled, Select all/none/new works, mounted rows remain bounded, keyboard focus survives scroll, and `aria-setsize`/`aria-posinset` are correct. With two playlists and a queued direct URL containing repeated videos, assert the first occurrence remains selected, later occurrences use session-level duplicate evidence, and explicit refresh reconciles selection by normalized source ID plus occurrence index among repeats.

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.virtualization.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because the panel eagerly renders the compatibility response.

- [x] **Step 3: Implement bounded page loading and TanStack Virtual rows**

Poll summary until `ready`/blocked terminal state, then request pages sequentially by opaque cursor with an `AbortController`. Merge only by server `occurrenceId`; reject duplicates or count mismatch as `preflight_incomplete`. Reconcile the complete snapshot against the session duplicate index using normalized source ID plus repeat index, retaining unambiguous selections and surfacing reorder/add/remove ambiguity after explicit refresh. Use `useVirtualizer` with stable occurrence keys and overscan. Render ordinal/title first and channel/duration/availability/duplicate second; keep URL in details and do not load thumbnails by default.

- [x] **Step 4: Run tests and commit**

Run the command from Step 2.

Expected: PASS with a bounded mounted-row assertion.

```bash
git add apps/packages/ui/src/services/tldw/playlist-ingest.ts apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/packages/ui/src/services/tldw/__tests__
git commit -m "feat: show complete virtualized playlist previews (TASK-12113)"
```

Verification before commit: the initial Task 3 paging/panel RED was 13/13 failures, hardening RED was 10 failed / 13 passed, specification-remediation RED was 3 failed / 30 passed plus 1 failed / 9 passed, and quality-remediation RED was 6 failed / 45 passed. Final focused Vitest passed 66/66 across the paging service, controller integration, virtualized panel, and legacy panel suites. Repository-pinned ESLint exited 0 (apart from the existing Next.js pages-directory informational message), scoped Prettier and `git diff --check` passed, and the final specification and code-quality re-reviews approved the diff. Full TypeScript was not rerun after the Task 1 three-attempt repository-baseline cap. Bandit is not applicable to this TypeScript-only task. Task 4 remains intentionally unimplemented: playlist rows are not materialized or added to the queue yet.

### Task 4: Materialize occurrences and carry Review-time overrides

- [x] **Step 1: Write failing queue/review tests**

Assert Add sends selected occurrence IDs, creates no rows on materialization failure, stores materialization ID/token plus occurrence ID, renders title/ordinal as primary text, and lets Review edit per-occurrence duplicate policy and only title/author/keywords-add metadata. Add 500-item tests proving both the queue and Review render bounded row counts with filters, and an expired-materialization test proving Start Processing requires reinspection rather than cached URL fallback.

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --maxWorkers=1 --no-file-parallelism --testNamePattern="playlist|materialization|review"`

Expected: FAIL because queue rows contain client-generated IDs and preflight URLs.

- [x] **Step 3: Extend queue types and materialization handling**

Add a discriminated source reference to `WizardQueueItem`:

```ts
type WizardSourceRef =
  | { kind: "materialized_playlist_item"; materializationId: string; token: string; occurrenceId: string }
  | { kind: "direct_url"; occurrenceId: string; url: string }
  | { kind: "file_stub"; occurrenceId: string }
```

Playlist row `id` equals the server occurrence ID. Preserve compact title/playlist/ordinal/channel/duration display data, but never treat cached playlist URLs as authoritative after materialization expiry. Keep duplicate policy/metadata patch in Review state, not materialization state. Virtualize the Add-step queue and `ItemMetadataTable` with stable occurrence keys. Add queue filters for playlist/type/duplicate state and Review filters for selected/duplicates/policy; filters change visibility only, never selection.

- [x] **Step 4: Build the exact Start Processing payload**

Serialize selected input records plus `review_overrides[occurrenceId]`. Include an explicit duplicate policy for every current duplicate and include a patch only for explicitly edited allowlisted fields. On backend `review_required`, merge refreshed duplicate evidence and return to Review without marking rows as submitted.

- [x] **Step 5: Run tests and commit**

Run the command from Step 2 without the name filter.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/types.ts apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx apps/packages/ui/src/components/Common/QuickIngest/ItemMetadataTable.tsx apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__
git commit -m "feat: preserve playlist occurrence identity in review (TASK-12113)"
```

Verification before commit: Task 4 was implemented through focused RED/GREEN cycles covering authoritative occurrence materialization, atomic queue mutation, virtualized queue/Review navigation, allowlisted per-occurrence overrides, exact run-request serialization, fail-closed persistence recovery, the 500-input limit, stale duplicate evidence, cached-authority rejection, and evidence-none Review recovery. Final frontend Vitest passed 200/200 across the five Task 4 suites. Backend playlist service and endpoint tests passed 149/149. Repository-pinned production ESLint exited 0 with zero rule findings (apart from the existing Next.js pages-directory informational message), Bandit reported zero findings across the touched 1,334-line backend service, and `git diff --check` passed. Final specification and code-quality re-reviews approved the diff. Full TypeScript was not rerun after the Task 1 three-attempt repository-baseline cap. A blanket Prettier write was not used because the shared package has conflicting frontend/extension configurations; changed hunks were reviewed directly and the whitespace gate passed.

## Stage 3: Shared run submission and status transport

**Goal:** Replace WebUI/extension submission differences with one occurrence-aware run client.

**Success Criteria:** Both clients create the same run, submit bounded chunks, merge status by occurrence, and reattach without per-item polling fan-out.

**Tests:** Run client tests, ambiguous retry, dynamic events, extension runtime tests.

**Status:** In Progress

### Task 5: Implement the shared run client and bounded submission

- [x] **Step 1: Write failing run-client tests**

Test run creation, Review-required response, processing-only chunk selection, structured partial acceptance, global stop with `Retry-After`, URL/file aligned arrays, server-returned authoritative URLs overriding cached queue display URLs, same-attempt ambiguous retry, and polling/SSE snapshots merged by occurrence ID.

```ts
expect(submitCalls[0].fields).toMatchObject({
  run_id: "run-1",
  occurrence_ids: ["occ-1", "occ-2"],
  attempts: [1, 1],
  urls: [video1, video2],
})
```

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism --testNamePattern="run|occurrence|chunk|ambiguous"`

Expected: FAIL because Quick Ingest submits/waits through legacy item logic.

- [x] **Step 3: Implement one run client**

In `playlist-ingest.ts`, add `createRun`, `submitPendingChunks`, `getRun`, `listRunItems`, `streamRunEvents`, `cancelRun`, and `retryRunItems`. Use a conservative exported chunk-size constant. Build URL submission fields only from the authoritative processing occurrences returned by run creation/items, never from cached queue URLs. A run item is merged only by `occurrenceId`; state and terminal outcome are separate. Treat `resync_required` as a full summary/items reload.

- [x] **Step 4: Delegate legacy services without duplicating logic**

In `quick-ingest-batch.ts`, detect version-2 run payloads and call the shared run client; retain the current legacy branch for non-playlist old-server sessions only. In `quick-ingest-session-reattach.ts`, prefer `runId` snapshots over job-ID fan-out. WebUI prefers SSE with polling fallback; extension supplies a polling preference.

- [x] **Step 5: Run tests and commit**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/tldw/__tests__/playlist-ingest.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts ../packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: PASS.

```bash
git add apps/packages/ui/src/services/tldw/playlist-ingest.ts apps/packages/ui/src/services/tldw/quick-ingest-batch.ts apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts apps/packages/ui/src/services/__tests__ apps/packages/ui/src/services/tldw/__tests__
git commit -m "feat: submit quick ingest through shared runs (TASK-12113)"
```

Task 5 verification before review/commit: the new delegate RED timed out in legacy per-job polling, while run-ID reattachment returned `interrupted`; the shared-client cases initially lacked the high-level run operations. Review remediation moved normalized processing options, conference collection creation, and per-occurrence conference metadata into the authoritative run-create body without cached materialized URLs; preserved monitoring after a later chunk is rate-limited while cancelling only unsent occurrences; made run cancellation fail explicitly except for genuine old-server fallback; and made SSE reload version-advanced items and ignore its unchanged initial snapshot. A final cleanup-failure RED proved that an occurrence-cancel failure could otherwise reattach forever; the service now returns an explicit cleanup-failed state and the Modal preserves tracking/accepted work in an interrupted recovery state without starting that loop. The Modal gates run-ID reattachment on explicit submit acknowledgement rather than timing, and the regression keeps submission pending after publishing tracking to prove the race is closed; the ordinary accepted-run control proves acknowledgement enables reattachment. The final expanded Vitest gate passed 224/224 across nine suites, including the three required run/delegate suites, both WizardModal suites, bounded persisted run identity/runtime, background upload Retry-After propagation, terminal no-job outcomes, partial chunk acceptance, cleanup failure, and refreshed run cancellation. Repository-pinned focused ESLint exited 0 with zero errors, and `git diff --check` passed. Final code re-review approved the remediated diff. Full TypeScript was not rerun after the Task 1 three-attempt repository-baseline cap. Bandit is not applicable to this TypeScript-only task. A blanket Prettier write was not used because the shared package still reports conflicting/baseline formatting across every touched file. Step 5 remains open only for the root-agent commit.

Task 5 formal blocker remediation completed test-first, with Step 5 and final approval still open pending both requested re-reviews. The durable tracking contract now persists `creating_run` before create, publishes `run_created` before upload, and merges accepted batch/job mappings after each chunk; reload reconstructs materialized authority and reattaches post-create/partial runs without cached-URL fallback. Missing file/unusable processing occurrences stop submission and are cancelled while accepted work remains attached. Run paging fails closed above 500 occurrences or a 4096-character cursor, retained SSE events cannot regress authoritative lifecycle/progress, and reattachment falls back to legacy jobs only for explicit 404/405/501 compatibility while 429/503 remain retryable and 401/403 surface authorization recovery. Cleanup failure interrupts even when the first chunk accepted nothing. Direct cancellation is observed while create is pending and between chunks, preventing later uploads and cancelling the server run. Formal RED gate: 17 failed / 118 passed across five files. GREEN gate: 135/135. Expanded Task 5 gate: 241/241 across nine files. Repository-pinned scoped ESLint `--quiet` and `git diff --check` exit 0 (only the existing Next pages-directory informational output). Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only changes. No files were staged or committed.

Task 5 second formal-review remediation completed test-first, with Step 5 and final approval still open pending both final re-reviews. A restored `creating_run` marker now fails closed as interrupted and cannot reconstruct or restart a request, including after materialization expiry. A dedicated, sanitized `submissionOccurrenceIds` field preserves at most 500 occurrence identities without overloading collection planning metadata. Reload of `run_created` or `submitting` tracking cancels only server-authoritative unsent states (`staged`, `awaiting_upload`, `submit_pending`) and repolls; accepted/running work is preserved, and cleanup 503 remains retryable without false terminalization. SSE streams without a trustworthy event boundary reload authoritative summary/items for occurrence events, preventing same- or higher-state retained replay from regressing metadata. Second formal RED gate: 8 failed / 132 passed across five files; the additional persisted-bound control brought the final focused gate to 141/141. Expanded Task 5 gate: 247/247 across nine files. Repository-pinned scoped ESLint `--quiet` and `git diff --check` exit 0 (only the existing Next pages-directory informational output). Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only changes. No files were staged or committed.

Task 5 bounded-transport remediation completed test-first, with both final re-reviews still pending. The 500-item unknown-cursor RED proved that replaying retained occurrence events caused 1002 authoritative REST requests instead of the expected two. Reattachment now returns the complete authoritative poll snapshot whenever no trustworthy event high-water mark exists and does not open SSE in that state; the existing stale-terminal control confirms retained events cannot weaken authoritative state. The focused five-file gate passes 142/142 and the expanded nine-file gate passes 248/248. Repository-pinned scoped ESLint `--quiet` exits 0 with only the existing Next pages-directory informational output. Full TypeScript remains skipped under the Task 1 three-attempt baseline cap; Bandit is not applicable to TypeScript-only changes. No files were staged or committed.

Task 5 final formal specification/code-quality re-reviews: approved with no actionable findings. The reviewers independently confirmed fail-closed restored-create recovery, bounded dedicated submission occurrence tracking, authoritative unsent-only cleanup with retryable cancellation failure, bounded unknown-cursor polling, and cursor-backed SSE correctness. One reviewer reran the requested focused and expanded gates at 142/142 and 248/248; the other independently expanded the nine-file verification to 254/254. Both reported scoped ESLint `--quiet` and `git diff --check` clean. The root agent independently reran the expanded nine-file gate at 248/248, linted every changed TypeScript file with the repository-pinned ESLint configuration, and confirmed `git diff --check` clean before commit.

### Task 6: Make the extension runtime a thin transport adapter

- [x] **Step 1: Write failing runtime parity tests**

Assert the background runtime stores only `runId`/compact mappings, polls the run endpoint after worker recreation, emits occurrence-aware events, delegates cancellation to run cancel, and never independently expands/classifies a playlist.

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/entries/shared/__tests__/quick-ingest-session-runtime.test.ts ../packages/ui/src/entries/__tests__/background.web-clipper.test.ts ../packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL on run reattachment/parity assertions.

- [x] **Step 3: Update the runtime and active-tab handoff**

`ControlRow` continues to pass only typed open detail. The wizard controller owns inspection. `background.ts` calls the same run client with `transportPreference: "poll"`; remove its duplicate submit/poll loop for version-2 runs. Runtime context tracks `runId` and cancellation, not an in-memory job list as the recovery source of truth.

- [x] **Step 4: Run extension tests and commit**

Run the command from Step 2, then run: `cd apps/tldw-frontend && bun run test:extension -- --run ../packages/ui/src/entries/shared/__tests__/quick-ingest-session-runtime.test.ts`

Expected: PASS.

```bash
git add apps/packages/ui/src/entries apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__
git commit -m "feat: align extension playlist run transport (TASK-12113)"
```

Task 6 implementation and verification completed test-first; Step 4 remains open only for the root-owned review/commit. The initial exact RED gate collected 23 tests with 4 intended runtime parity failures. Follow-up RED cycles covered versioned and bounded compact persistence, stale cleanup, the v2 background delegate, continued terminal polling, immediate cancellation visibility, direct background poll/cancel transport, and serialized storage cleanup. The final exact three-file gate passes 32/32, the extension-specific runtime gate passes 12/12, and the three shared run-service regression suites pass 90/90. The extension runtime persists only versioned `sessionId`/`runId`/occurrence mappings, rejects malformed or oversized records (255-character identities, 500 occurrences, 500 job mappings), surfaces storage access/write failures, serializes storage mutations, and uses compare-by-run cleanup so stale terminal work cannot delete or resurrect a replacement run. Version-2 background work returns before the legacy classifier and delegates create/submit, direct run polling, and run cancellation to the shared clients; legacy payload behavior is unchanged. `ControlRow` remains a typed active-tab handoff only. Repository-pinned scoped ESLint `--quiet` and `git diff --check` both exit 0. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only task. No files were staged or committed.

Task 6 formal-review remediation completed test-first for all nine requested blockers; Step 4 and formal re-approval remain open. The combined RED gate was 12 failed / 119 passed across the run client, extension runtime, MV3 transport, and Modal session suites. Direct transport now survives summary/paged-item resync; compact records preserve bounded submission state; interrupted runs remain recoverable; cancel/create and restore/cancel races are coordinated; MV3 pending v2 work stays in the durable worker across recreation; persisted UI sessions query worker replay instead of restarting; runtime progress and partial terminal failures carry occurrence-aware results; and accepted cancellation retains/polls the authoritative run until terminal. Self-review also corrected cancelled-item normalization when the item lacked its own error string. Final remediation gate: 133/133. Fresh exact Task 6 gate: 38/38; extension runtime: 18/18; shared Task 5 service regressions: 92/92. Repository-pinned scoped ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the Task 1 three-attempt cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 6 third formal-review remediation completed test-first for all six blocker groups; Step 4 and formal re-approval remain open. The combined RED gate was 20 failed / 118 passed. The runtime now treats `cleanup_required` as recoverable, retains and reconciles a known post-create run after storage failure, retries failed reconciliation polling, and surfaces/retries startup storage reads. Caller-generated extension session IDs plus a compact pre-create fingerprint marker make repeat delivery idempotent, reject conflicting reuse, and prevent direct submission after ambiguous delivery. Bounded 24-hour terminal tombstones survive worker recreation and require an explicit session/run/generation replay acknowledgement before compare-safe deletion. The Modal keeps authoritative cancellation nonterminal until the runtime reports completed, failed, or cancelled, preserves local no-run cancellation, and queries interrupted extension sessions on reopen. The final five-file remediation gate passes 139/139; the fresh exact Task 6 gate passes 47/47; the standalone extension runtime passes 26/26; and shared Task 5 service regressions pass 95/95. Final self-review confirmed that persisted fingerprints contain only a bounded hash and occurrence identities, terminal events whitelist/bound result fields and enforce TTL/ack identity, storage failures retain cleanup authority, and legacy direct-session paths remain isolated. Repository-pinned scoped ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 6 fourth formal-review remediation completed test-first for all eleven requested blocker groups; Step 4 and formal re-approval remain open. The combined RED gate was 24 failed / 118 passed across runtime, background, transport, and Modal suites. Automatic Modal replay acknowledgement was removed so terminal tombstones remain until explicit acknowledgement, TTL expiry, or deterministic capacity eviction. Runtime start acknowledgement now waits for the durable pre-create marker; caller-owned opaque attempt tokens replace payload fingerprints; generation-aware CAS protects marker-to-run/terminal transitions, polling, and explicit replay acknowledgement; failed active-record persistence retries during later reconciliation. Structured Review-required recovery now crosses the background/runtime/Modal boundary. Storage is bounded to 64 unique sessions, 512 KiB per complete terminal record, and 2 MiB aggregate terminal data while never evicting active cleanup authority. Extension cancellation fails closed when the runtime is unavailable or times out. Ambiguous start plus replay timeout retains one stable interrupted extension identity, and Modal replay is open-gated, bounded to three attempts, actionable on exhaustion, repeatable on each reopen/recovery request, and cannot restart a tracked extension run. Self-review added two further RED/GREEN controls: durable replay-ack CAS rejection retains the in-memory tombstone, and the 512 KiB limit measures the whole persisted record rather than only its event. Final combined gate: 144/144; exact Task 6 gate: 60/60; standalone extension runtime: 33/33; shared Task 5 service regressions: 97/97; context regression control: 54/54. Repository-pinned scoped ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 6 fifth formal-review remediation completed test-first for all five requested blocker groups; Step 4 and formal re-approval remain open. The combined behavioral RED gate was 8 failed / 95 passed across the runtime, background adapter, and Modal session suites. The real background adapter now preserves structured `reviewRequired` results. Review recovery is a bounded 24-hour, generation-CAS replay tombstone that survives lost responses and worker recreation, participates in the 64-session/2 MiB deterministic replay cap, and cannot evict active recovery authority. The Modal clears its started flag, active runtime identity, in-memory persisted tracking, and durable extension tracking before applying corrected Review state, so a manual corrected submission starts exactly once. Valid 500-item terminal snapshots that exceed the rich 512 KiB representation now deterministically fall back to an essential form retaining every occurrence ID, terminal status, and outcome instead of entering an interrupt/poll loop. Restore isolates a failed run poll, schedules that session's retry, and continues restoring later records. The first production run passed 102/103; the remaining failure was a test-only Review mock that lacked the real manual Start action, and correcting the harness produced the final 103/103 combined gate without another production change. Fresh verification passed the exact Task 6 gate 67/67, standalone runtime 37/37, shared Task 5 services 97/97, Modal 47/47, context 54/54, and expanded combined gate 153/153. Repository-pinned full touched-scope ESLint `--quiet` and `git diff --check` pass. Self-review confirmed Review TTL/whole-record bounds/generation CAS, deterministic replay eviction before active authority, no replay acknowledgement for Review tombstones, complete essential terminal identity/status/outcome retention, per-session restore retry isolation, exactly-once corrected restart, and unchanged legacy delegation behavior. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 6 sixth formal-review remediation completed test-first for the four requested blocker groups; Step 4 and formal re-approval remain open. The accepted combined RED gate was 6 failed / 103 passed across runtime, background, and Modal session behavior, followed by focused atomic-handoff and cleanup-timer RED controls. Restore now isolates every record, including expired terminal/Review cleanup, and retains only the failed record for a bounded retry that is cancelled when a later restore succeeds. Pending polls re-check generation/run authority, terminal CAS is limited to active records, rejected terminal writers stop without emission or rescheduling, and terminal tombstones cannot replace one another. Oversized essential terminal outcomes normalize through the same Modal consumer as rich results. Review handoff now computes one pure Review snapshot, synchronously writes and exact-reads a Zustand-compatible persistence envelope before mutating store/reducer/ref state, guards stale Provider writes by snapshot revision, and leaves extension replay authority untouched when persistence fails or throws. Final self-review confirmed thrown handoff guards are released, manual persistence rehydrates through the normal store, stale cleanup timers are cleared, and delayed/rejected terminal writers cannot duplicate events. Fresh verification passes exact Task 6 72/72, standalone runtime 41/41, shared Task 5 services 97/97, Modal/context/store 123/123, and the expanded five-file gate 180/180. Repository-pinned full touched-scope ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 6 final formal specification and code-quality re-reviews approved the sixth-pass diff with no actionable findings. The reviewers independently confirmed per-record restore isolation, canonical compact outcomes, exactly-once terminal authority, active-only CAS, atomic Review persistence and guard cleanup, MV3 recovery, bounded storage, privacy, cancellation, replay, and legacy-path invariants. The root agent independently reran the exact Task 6 gate at 72/72, shared Task 5 service regressions at 97/97, Modal/context/store controls at 123/123, repository-pinned ESLint across every changed TypeScript file, and `git diff --check`; all passed before commit.

## Stage 4: Truthful lifecycle UI and durable local recovery

**Goal:** Render server evidence directly and survive reload/runtime loss without silently losing a large session.

**Success Criteria:** Queue/progress/results use occurrence identity and state/outcome axes; IndexedDB migration and failures are visible; file reattachment remains client-derived.

**Tests:** Component lifecycle tests, Dexie migration/quota/cleanup tests, multi-tab tests.

**Status:** In Progress

### Task 7: Update processing, cancellation, retry, and result groups

- [x] **Step 1: Write failing lifecycle UI tests**

Cover `awaiting_upload`, client-derived file reattach, submit pending/queued/running, cancellation requested, status unavailable, all eight terminal result groups, real row/run cancellation, and retry reconciliation. Assert no fabricated analyzing/storing stage when the backend supplies only generic progress. At 500 items, assert bounded mounted rows and useful Active/Needs attention/Terminal plus outcome filters.

- [x] **Step 2: Run and verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because existing progress uses fabricated stage tokens and fewer outcomes.

- [x] **Step 3: Replace UI status inference with run snapshots**

Extend `ItemProgress`/`WizardResultItem` with server lifecycle and terminal outcome. Derive `file_reattach_required` only when server state is `awaiting_upload` and no local `File` exists. Keep `status_unavailable` recoverable with Check again/Reconnect. Virtualize processing/results above the same scale threshold used by preview and preserve title/ordinal primary labels.

- [x] **Step 4: Wire real cancellation and retry**

Before run creation, cancellation is local removal. After run creation, row cancellation always calls `POST /runs/{runId}/cancel` with that occurrence ID: the server terminalizes unsent items or cancels accepted jobs. Retain `cancellation_requested` until terminal. Whole-run cancellation calls the same route without occurrence IDs. Retry sends eligible occurrence IDs and waits for the server's reconciled attempt/action response.

- [x] **Step 5: Run tests and commit**

Run the command from Step 2.

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Common/QuickIngest apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
git commit -m "feat: show per-occurrence ingest lifecycle (TASK-12113)"
```

Task 7 implementation and verification completed test-first; Step 5 remains open for the root-owned formal review and commit. The accepted exact RED gate was 9 failed / 73 passed, with two additional service REDs proving occurrence-scoped cancellation and canonical reattach metadata. Processing now renders authoritative lifecycle/message/percentage evidence without the former simulated analyzing/storing timer; file reattachment is derived only from `awaiting_upload` plus a missing local `File`; status-unavailable rows expose Check again/Reconnect; and Processing/Floating Progress share Active/Needs attention/Terminal semantics. Results preserve all eight canonical terminal outcomes, canonical retry eligibility, and bounded outcome filtering. Row/run cancellation use the shared run-cancel client, retry uses the shared run-retry client followed by authoritative reconciliation, and an exact-signature guard prevents duplicate immediate retry reattachment while allowing changed tracking signatures to poll. Small lists render normally and lists at the existing 100-row threshold use stable-key TanStack virtualization with list position semantics; 500-row gates remain bounded. Final exact Task 7 gate: 83/83; service/playlist gates: 98/98; Modal/context/store controls: 123/123; extension runtime/background controls: 61/61. Repository-pinned touched-scope ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only task. No files were staged or committed.

Task 7 formal-review remediation completed test-first for all eleven requested finding groups; Step 5 and formal specification/code-quality approval remain open. The accepted combined RED was 21 failed / 294 passed across eight files. Canonical lifecycle/progress/retry/run identity now crosses the runtime boundary; occurrence cancel/retry use the shared authority path; unchanged retry snapshots continue polling; reattach transport failures remain recoverable; pre-run, cancellation, results, file reselection, local removal, late-callback, virtual keyboard/focus, and translated floating-summary behavior are all covered. First combined GREEN attempt: 311/315; final combined GREEN: 315/315. Fresh final gates: exact Task 7 89/89; playlist/batch/reattach services 100/100; Modal/context/store 126/126; extension runtime/background/handoff 76/76. Repository-pinned touched-scope ESLint `--quiet` and `git diff --check` pass. Full TypeScript was not rerun under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 7 second formal-review remediation completed test-first for all eleven follow-up finding groups; Step 5 and root-owned review/commit remain open. The accepted combined RED was 22 failed / 275 passed across eight files. The first GREEN attempt was 286/297; after correcting the four expectation/fixture mismatches, the second attempt was 297/297. A final retry-generation CAS regression brought the fresh combined gate to 298/298 with zero skips. Fresh supporting gates are exact Task 7 95/95, playlist/batch/reattach services 105/105, Modal/context/store 197/197, and extension runtime/background/handoff 127/127, all with zero skips. Canonical run IDs and retryability now survive runtime replay, retry accepted by the backend re-arms the same retained session/run through a generation CAS, and persistence/CAS failure retains the terminal tombstone. Retryable run-status transport failures remain recoverable; pre-run occurrence cancellation is live-filtered before creation; same-run file reselection submits exactly one occurrence and reconciles authoritatively; durable Retry all excludes canonical non-retryable failures; late whole-run cancellation callbacks preserve terminal authority; direct and extension results retain ordinal/title identity; ETA copy uses translation keys; occurrence-scoped cancellation cannot fall back to whole-batch cancellation; and Check again reconciles both direct and extension sessions. Repository-pinned touched-scope ESLint `--quiet` and `git diff --check` pass. Full TypeScript was not rerun under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 7 third formal-review remediation completed test-first for all eight requested blocker groups; Step 5 and root-owned formal approval/commit remain open. The accepted four-file RED gate was 15 failed / 225 passed with zero skips. Direct-session cancellation is now reserved before backend create, distinguishes whole-stop from occurrence cancellation, cancels newly created occurrence rows before upload, rechecks row cancellation before every chunk, and always releases the in-memory registry. Extension retry uses a bounded persisted `retrying` generation reservation before backend mutation, permits one concurrent winner, rolls rejection back through generation CAS, reconciles CAS loss, and keeps accepted work polling if active persistence throws. File reads fall back when `File.arrayBuffer` is unavailable and retained bytes can be retried explicitly. The Modal records row/run cancellation before authority exists, applies retry generation before reconciliation, ignores stale whole-run cancellation settlement from an older generation, and routes direct Check again through one refresh nonce. The first combined GREEN exposed one whole-stop unsent-reporting compatibility regression; after separating whole-stop from row cancellation, the fresh combined gate passed 240/240. Fresh supporting gates pass exact Task 7 96/96, playlist/batch/reattach services 107/107, Modal/context/store 203/203, and extension runtime/background/handoff 84/84, all with zero skips. Repository-pinned full dirty-TypeScript ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 7 fourth formal-review remediation completed test-first for all nine requested authority and recovery groups; Step 5 and root-owned formal approval/commit remain open. The accepted six-file RED gate was 14 failed / 278 passed with zero skips and no unhandled errors. Pending row cancellation is forwarded only after acknowledged or indeterminate extension identity; direct submission responses are guarded by session/run/generation or pre-authority cancellation keys; row and whole cancellation settlements are generation-scoped; accepted degraded retries return and adopt their reserved generation while live monitoring continues with a visible warning. Existing-run file reselection preserves the authoritative occurrence attempt. Retry reservations receive a fresh bounded recovery TTL, count as non-evictable records in the 2 MiB recovery budget, and fail closed at capacity. Timeout/lost-response retries remain indeterminate under the reserved generation and reconcile without a second backend mutation, while explicit HTTP rejection alone rolls back. Accepted direct retries return a fresh opaque generation so later cancellation keys cannot match old callbacks. The final combined gate passes 292/292; fresh supporting gates pass exact Task 7 96/96, playlist/batch/reattach services 108/108, Modal/context/store 211/211, extension runtime/background/handoff 88/88, and standalone runtime 51/51, all zero-skip with no unhandled errors. Repository-pinned full dirty-TypeScript ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to this TypeScript-only remediation. No files were staged or committed.

Task 7 fifth formal-review remediation completed test-first for the final four authority findings; Step 5 and root-owned formal approval/commit remain open. The accepted four-file RED gate was 11 failed / 200 passed with no unhandled errors. Authoritative upload attempts now flow from extension polling through the runtime event and Modal store and survive validation, file selection, upload failure, explicit replacement retry, and queued success; both replacement submissions retain attempt 4. Extension cancellation carries an optional expected generation, and runtime/background cancellation rejects stale generation before abort or cancellation mutation; direct retry authority is also generation-fenced. HTTP 500/502/503/504 and status-less timeout retry responses remain indeterminate and retain/reconcile the reserved generation, while explicit 409 rejection remains determinate. A live retry owner prevents a staggered second caller from invoking generic restore, promoting its retrying reservation, polling early, or mutating the backend twice. Final combined gate: 211/211. Supporting gates: exact Task 7 96/96, playlist/batch/reattach services 109/109, Modal/context/store 211/211, extension runtime/background/handoff 95/95, and standalone runtime 53/53. Repository-pinned full dirty-TypeScript ESLint `--quiet` and `git diff --check` pass. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only work. No files were staged or committed.

Task 7 sixth formal-review remediation completed test-first for authoritative retry resubmission and reservation recovery; Step 5 and both root-owned formal approvals/commit remain open. The initial focused RED gate was 14 failed / 144 passed across the batch service, extension background, and extension runtime; source self-review then added three more failing controls to a 163-test gate. Retry success now consumes only the backend's authoritative staged non-file occurrences and attempts, creates jobs, and monitors them while file stubs remain awaiting reselection; cached/display URLs cannot enter resubmission. Status-less and HTTP 500/502/503/504 failures retain the reserved generation, deterministic rejection alone rolls back, replacement upload cannot clear retry authority, and stale-generation cancellation is rejected. Direct reservations are bounded to 64 live non-evictable entries with 24-hour expiry and fail closed before backend mutation. Direct and durable extension reservations reconcile a fresh manifest before any second retry, retain authority through unavailable or partially advanced mixed manifests, submit newly staged occurrences idempotently, and release only after every selected non-file occurrence is active or resolved. Final focused retry gate: 163/163. Supporting gates: exact Task 7 96/96; playlist/batch/reattach services 123/123; Modal/context/store 211/211; extension runtime/background/handoff 99/99; standalone runtime 55/55. Repository-pinned full dirty-TypeScript ESLint `--quiet` across 23 files and `git diff --check` pass. Expected test-harness output was limited to Node localStorage warnings, the deliberate outside-provider exception, mocked unconfigured-server reconciliation, and simulated extension listener delivery retries. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only work. The formatter check was not rerun because its earlier `bunx prettier --check` invocation was blocked before analysis by sandbox tempdir access; no formatter mutation occurred. No files were staged or committed.

Task 7 seventh formal-review remediation completed test-first for retry-owner, mixed-manifest, retry-cadence, lost-delivery, pending-cancellation, and generation-lifecycle follow-ups; Step 5 and root-owned approval/commit remain open. The accepted four-file RED was 9 failed / 237 passed, and source review added focused RED controls for direct empty-response classification (1 failed / 81 skipped), extension empty-response classification (1 failed / 34 skipped), and Modal unmount cleanup (1 failed / 73 skipped). One shared authoritative manifest classifier now gives URL status-unavailable evidence precedence over file awaiting-upload, distinguishes not-advanced terminal retryable URLs from active/resolved rows, submits only fresh staged non-files, and handles successful empty retry responses consistently in direct and extension paths. Direct g2 owners coalesce concurrent calls, reservations survive partial advancement and explicit retry, expiry pruning runs before generation fencing, and active/terminal/cancelled authority retires without weakening stale-generation rejection. Extension lost-before-delivery preserves `notAdvanced`, re-owners exactly one later POST, and pending g2 scoped cancellation reaches the authoritative run while stale g1 is rejected and the durable reservation remains fenced. Modal recovery retries the reservation on its existing cadence with one in-flight owner, a three-POST bound, actionable terminal fallback, and unmount cleanup that prevents late retries; ordinary reattach remains the rendering authority. Final focused gate: 248/248. Supporting gates: exact Task 7 96/96; playlist/batch/reattach services 130/130; Modal/context/store 212/212; extension runtime/background/handoff 103/103; standalone runtime 57/57; focused retry 174/174. Repository-pinned frontend ESLint `--quiet` across all 23 dirty TypeScript files and `git diff --check` pass. Expected output was limited to Node localStorage, deliberate provider-boundary, mocked unconfigured-server, and simulated extension-delivery warnings. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only work. No formatter, staging, or commit action was taken.

Task 7 eighth formal-review remediation completed test-first for ambiguous file-only retries, mixed-manifest advancement, whole-run pending-generation cancellation, Modal reattach ownership, and direct-generation retirement; Step 5 and root-owned approval/commit remain open. The accepted four-file RED gate was 6 failed / 248 passed. The first GREEN attempt was 3 failed / 251 passed; after making the manual retry path the atomic single reattach owner, the focused gate passed 254/254. Source self-review added an interrupted-status non-retirement RED (1 failed / 78 skipped), after which the Modal gate passed 79/79 and the final focused four-file gate passed 257/257. File-only and partially advanced mixed retries now remain explicitly not advanced until file reselection submits exactly one later retry request without creating jobs. Pending whole-run cancellation omits occurrence IDs, preserves the current-generation reservation until genuine terminal reconciliation, and rejects stale-generation cancellation. Modal retry publication and reattachment have one owner even under StrictMode, ignore delayed work after unmount, and retire direct generation authority only through a CAS-safe helper after genuine completed, cancelled, or partial-failure reattachment—not after interrupted or unavailable status. Fresh broad gates pass exact Task 7 96/96; playlist/batch/reattach services 133/133; Modal/context/store 217/217; extension runtime/background/handoff 104/104; standalone runtime 58/58; focused retry batch/background/runtime 178/178; and the expanded batch/runtime/background/Modal gate 257/257. A preliminary repository-pinned ESLint invocation exited 2 before code analysis because the repo-root call omitted an explicit config; the corrected repository-pinned invocation with `apps/tldw-frontend/eslint.config.mjs` passed `--quiet` across all 23 dirty TypeScript files, with only the existing Next pages-directory informational output. `git diff --check` passes. Expected test-harness output was limited to Node localStorage, deliberate or mocked unconfigured-server reconciliation, and simulated extension-delivery warnings. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only work. No formatter, staging, or commit action was taken.

Task 7 ninth formal-review remediation completed test-first for selected-occurrence resolution while an unselected sibling remains active; Step 5 and root-owned approval/commit remain open. The accepted focused RED was 1 failed / 3 passed / 84 skipped: an empty idempotent retry returned a fresh g2 after the selected occurrence reached terminal completed attempt 2, but selected-only reconciliation retired whole-run authority despite the authoritative summary and sibling both remaining running, so stale g1 cancellation reached the backend. The minimal fix separates retry-reservation release from whole-session generation retirement: selected resolution always releases its reservation, while g2 retires only when the complete authoritative run summary is `completed`, `cancelled`, or `partial_failure` and the generation CAS still matches. Running sibling authority therefore keeps stale g1 fenced with no cancellation request while current g2 cancellation succeeds; interrupted/unavailable reconciliation and Modal terminal CAS retirement remain unchanged. Focused lifecycle controls pass 4/4, the complete batch service passes 88/88, and the focused batch/runtime/background/Modal gate passes 260/260. Fresh broad gates pass exact Task 7 96/96; playlist/batch/reattach services 136/136; Modal/context/store 217/217; extension runtime/background/handoff 104/104; standalone runtime 58/58; focused retry batch/background/runtime 181/181; and the expanded four-file gate 260/260. Repository-pinned frontend ESLint with explicit `apps/tldw-frontend/eslint.config.mjs` passes `--quiet` across all 23 dirty TypeScript files with only the existing Next pages-directory informational output; `git diff --check` passes. Expected test-harness output remains limited to Node localStorage, the deliberate provider-context exception, mocked unconfigured-server reconciliation, and simulated extension-delivery warnings. Full TypeScript remains skipped under the documented Task 1 three-attempt repository-baseline cap; Bandit is not applicable to TypeScript-only work. No Prettier, unpinned ESLint, staging, or commit action was taken.

Task 7 final closeout approved. Both formal reviewers approved the exact ninth-pass diff with no actionable findings. Root verification is green: exact UI 96/96, playlist/batch/reattach services 136/136, Modal/context/store 217/217, and extension runtime/background/handoff 104/104. Repository-pinned ESLint with the explicit frontend config passed all 23 dirty TypeScript files, including the untracked `apps/packages/ui/src/components/Common/QuickIngest/file-bytes.ts`, and `git diff --check` passed. Task 7 Step 5 is complete; TASK-12113 remains In Progress for Tasks 8 and 9.

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
git commit -m "feat: persist quick ingest runs in indexeddb (TASK-12113)"
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

Record test counts, browser result, accessibility assertions, and any explicit skips in `TASK-12113`. Bandit is not applicable to this TypeScript-only plan; record the skip.

```bash
git add apps/packages/ui apps/tldw-frontend/e2e backlog/tasks
git commit -m "test: verify shared playlist ingest experience (TASK-12113)"
```
