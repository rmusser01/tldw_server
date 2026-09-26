# Authorized Workspace Opening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Open an existing owned workspace completely and without writes, draft loss, or cross-account state reuse.
**Architecture:** Put a complete read-only loader before the existing editable route. Commit its result through one scoped store activation action, keeping browser drafts separate from canonical server data and legacy local snapshots. Preserve existing workspace APIs and shared-recipient routing.
**Tech Stack:** TypeScript, React, Zustand persistence, existing tldw client, Vitest, Playwright over CDP, FastAPI/Jobs acceptance fixture.
**Spec:** [Authorized server-owned workspace opening](../specs/2026-09-13-owned-workspace-opening-design.md).
**Tracking:** TASK-12020.50; Stages 1-2 implemented and reviewed, route integration and live acceptance pending.

## Global Constraints

- Use existing workspace APIs; no new bootstrap endpoint in this slice.
- No route aliases, redirects, or additional trust banners.
- Opening and revalidation are read-only; only explicit user edits cause writes.
- Preserve legacy local workspaces without inferring their server/account owner.
- Use CDP and a real isolated backend/WebUI for acceptance.
- Preserve existing TASK-12020.49 and TASK-12020.50 changes.
- Do not claim full clone certification until the outstanding acceptance matrix passes.

## File Responsibilities

Paths below are relative to this worktree. Do not restructure unrelated large modules.

| File | Responsibility |
| --- | --- |
| `apps/packages/ui/src/store/workspace-api.ts` | Replace incomplete hydration contract with complete validated reads; reuse source/artifact mappers |
| `apps/packages/ui/src/services/tldw/domains/workspace-api.ts` | Existing typed reads; backward-compatible cancellation/request-context plumbing only if required |
| `apps/packages/ui/src/store/owned-workspace-state.ts` (new) | Owned activation/scope types and pure scoped draft/activation transitions |
| `apps/packages/ui/src/store/workspace.ts` | Active origin, readiness, atomic action, persisted-state isolation |
| `apps/packages/ui/src/store/workspace-slices/workspace-list-slice.ts` | Preserve outgoing state and avoid putting owned content into legacy lists |
| `apps/packages/ui/src/hooks/useOwnedWorkspaceOpening.ts` (new) | Principal verification, hydration readiness, timeout/generation and lifecycle subscriptions |
| `apps/packages/ui/src/components/Option/ResearchWorkspace/ResearchWorkspaceRouteGate.tsx` | Gate owned target rendering; accessible loading/error/retry states |
| `apps/packages/ui/src/components/Option/ResearchWorkspace/research-workspace-route-state.ts` | Owned selector validation without changing deep-research return context |
| `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx` | Consume activation readiness, canonical notes, and mutation identity guards |
| `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts` | Keep legacy reconciliation out of server-owned opening |
| `apps/packages/ui/src/routes/route-paths.ts` and `components/Option/Workspaces/WorkspacesManagerPage.tsx` under the UI source root | Canonical owned Open link producer and manager use |
| `apps/packages/ui/src/assets/locale/en/common.json` | Targeted opening/recovery copy, no banner |

Before editing, inspect nested AGENTS.md instructions and current task status. Record
new test/verification results through Backlog CLI. Keep existing unrelated changes
out of incremental commits; do not stage the entire worktree.

## Stage 1: Complete Read-Only Loader

**Goal:** Replace the partial hydration contract with a complete, validated bundle.
**Success Criteria:** Exact target and all required resources, no writes, failures propagate.
**Status:** Complete

**Interfaces:** Define these exports in `store/workspace-api.ts`, importing the four
existing API response types from `services/tldw/domains/workspace-api.ts`:

```ts
export type OwnedWorkspaceBundle = {
  workspace: WorkspaceApiResponse
  sources: WorkspaceSourceApiResponse[]
  artifacts: WorkspaceArtifactApiResponse[]
  notes: WorkspaceNoteApiResponse[]
}
export type OwnedWorkspaceReader = {
  getWorkspace(id: string): Promise<WorkspaceApiResponse>
  getWorkspaceSources(id: string): Promise<WorkspaceSourceApiResponse[]>
  getWorkspaceArtifacts(id: string): Promise<WorkspaceArtifactApiResponse[]>
  getWorkspaceNotes(id: string): Promise<WorkspaceNoteApiResponse[]>
}
export async function loadOwnedWorkspace(
  id: string,
  reader: OwnedWorkspaceReader,
  signal: AbortSignal
): Promise<OwnedWorkspaceBundle>
```

The injected reader must be bound to the verified request context. The signal
guards dispatch and completion; transport cancellation may additionally be wired
through optional domain read options. Never mutate shared client auth state to pin
a request. Reject aborted responses even where the transport cannot cancel.

- [x] Replace the existing test expecting absent subresources to become empty
  arrays in `store/__tests__/workspace-api-first.test.ts`. Add a typed complete
  fixture there with non-default policy/audio/banner/defaults and explicit selected
  and unselected sources. Reuse existing artifact mapping examples including review
  and lineage fields.
- [x] Add parameterized failures for each of the four reader methods, mismatched
  ID, archived/deleted target, malformed collections, mismatched associations, and
  cancellation before/after dispatch. Empty successful arrays remain valid. This is
  the required rejection assertion pattern (with the typed fixture reader):

  ```ts
  reader.getWorkspaceNotes.mockRejectedValueOnce(new Error("notes unavailable"))
  await expect(loadOwnedWorkspace(id, reader, controller.signal))
    .rejects.toThrow("notes unavailable")
  ```

- [x] Run RED from `apps/tldw-frontend`:
  `bunx vitest run ../packages/ui/src/store/__tests__/workspace-api-first.test.ts --maxWorkers=1`.
- [x] Implement metadata-first validation, then sources/artifacts/notes reads with
  `Promise.all`. Return a bundle only after all validate. Preserve API fields; map
  into local types at activation using existing mappers. Remove the unused old
  incomplete helper after migrating its test references, not the independent
  optimistic-update tests.
- [x] Run GREEN with the same command and domain workspace tests. Confirm all read
  spies target exactly `id`; no mutation helper appears in the loader dependency.

Stage 1 execution notes: loader cases live in the new
`store/__tests__/owned-workspace-load.test.ts`; source/artifact mapping and
optimistic-update coverage remains in `workspace-api-first.test.ts`. The partial
helper and its permissive absent-collection expectation were removed. No runtime
caller used that helper. Mappers are exported for the upcoming activation action.
An independent review found missing effective-assistant status relationships;
RED/GREEN tests now enforce the backend's available/unavailable/none rules for raw
and normalized representations. No API, route, or active-store behavior changed.

Verification: 121 tests passed in five loader/mapping/domain files; scoped TypeScript
check passed. ESLint found no errors, three existing `any` warnings in the unchanged
optimistic-update helper, and the existing shared-UI pages-path configuration warning.
Node v26 native Web Storage conflicts with JSDOM quota-test hooks; the focused run
uses `NODE_OPTIONS=--no-experimental-webstorage`. Broader storage-suite limitations
are recorded in the live acceptance matrix, not hidden by a green loader result.

## Stage 2: Scoped Atomic State And Drafts

**Goal:** Commit authorized data once without losing or mis-scoping current work.
**Success Criteria:** No partial activation, late hydration overwrite, global owned snapshots, or account collisions.
**Status:** Complete

Both checkpoints are implemented: the scoped draft foundation and its Zustand
action/persistence integration. The store now distinguishes legacy-local from
server-owned state and commits authorized bundles atomically. Route callers,
principal verification and canonical mutation UI remain Stage 3, not enabled by
the store checkpoint alone.

- [x] Repair the baseline split-storage spy without changing production storage:
  JSDOM instance assignments create string storage entries rather than replace the
  method, so observe `Storage.prototype.setItem` and restore mocks after each test.
- [x] Implement scoped draft keys, validated wire-format drafts, independent
  persistence, and detached in-memory recovery for failed writes.
- [x] Implement pure activation preparation: stale-attempt rejection, explicit
  pending-edit/note conflicts, clean-note refresh, and folder reference pruning.
- [x] Wire these helpers into one Zustand activation commit, capture the latest
  outgoing draft there, and fence every legacy save/switch/persistence boundary.

**Interfaces:** In `store/owned-workspace-state.ts`, consume `OwnedWorkspaceBundle`
from Stage 1 and define:

```ts
export type OwnedWorkspaceScope = {
  serverBase: string // normalized origin plus deployment subpath
  principalId: string
  organizationId: string | null
}
export type OwnedWorkspaceAttempt = {
  scope: OwnedWorkspaceScope
  generation: number
  workspaceId: string
}
export type OwnedWorkspaceActivationResult = "activated" | "stale" | "draft-conflict"
export function ownedWorkspaceDraftKey(
  scope: OwnedWorkspaceScope, workspaceId: string
): string {
  const base = new URL(scope.serverBase)
  // The implementation rejects credentials, query/hash and non-HTTP(S) bases.
  const normalizedBase = `${base.origin}${base.pathname.replace(/\/+$/, "")}`
  return "tldw:research-workspace:owned-drafts:v1:" + encodeURIComponent(
    JSON.stringify([normalizedBase, scope.principalId, scope.organizationId, workspaceId])
  )
}
```

Foundation interfaces in `store/owned-workspace-state.ts`:
- `createOwnedWorkspaceDraftStore(getStorage)` exposes `save`, `load`, and `remove`
  with explicit saved/unavailable/invalid/missing outcomes. `load` reports whether
  a ready draft is durable. Failed writes remain recoverable in memory; corrupt
  records remain untouched. Successful saves do not retain a stale read cache.
- `OwnedWorkspaceDraft` is a versioned, validated wire format for note/composer,
  folder/layout state and quarantined JSON `pendingChanges`. It does not contain
  credentials, readiness, or an authorization grant. Folder timestamps are strings;
  the store integration must serialize/revive them explicitly.
- `prepareOwnedWorkspaceActivation(attempt, expected, bundle, recoveredDraft?)`
  returns a discriminated preparation result. It consumes a validated loader
  bundle, never fetches or writes, and returns detached content plus draft state.
  The actual action must check the attempt again and commit atomically, since
  calling this helper alone does not activate anything.
- Pending canonical edits are retained for explicit reconciliation, not silently
  applied to loaded sources/settings. A dirty existing note requires its own
  matching version; workspace metadata version does not substitute for it.

Foundation verification: 39 new state/draft tests pass; the expanded 18-file
workspace/loader/domain run passes all 278 tests with Node native Web Storage
disabled. Scoped TypeScript passes. ESLint has zero errors, one pre-existing `any`
warning in the split-storage test and the existing shared-UI pages-path warning.
No live backend/browser acceptance is implied by these tests.

Store action signatures:
`activateOwnedWorkspace(attempt: OwnedWorkspaceAttempt, bundle: OwnedWorkspaceBundle): OwnedWorkspaceActivationResult`
and `invalidateOwnedWorkspace(): void`.
Use a discriminated active-origin field (legacy-local versus server-owned) rather
than treating a UUID as proof of origin. Generation/readiness are not persisted as
authorization. A draft envelope retains browser editor state and applicable base
versions; it never grants read access or substitutes for required GETs.

- [x] Create `store/__tests__/owned-workspace-state.test.ts`, add store integration
  coverage in `owned-workspace-store.test.ts`, and retain existing workspace and
  split-storage regressions. Begin with a failing
  key-isolation test:

  ```ts
  expect(ownedWorkspaceDraftKey({serverBase: "https://one.test/tldw",
    principalId: "2", organizationId: null}, "same-id"))
    .not.toBe(ownedWorkspaceDraftKey({serverBase: "https://two.test/tldw",
      principalId: "2", organizationId: null}, "same-id"))
  ```

- [x] Add store cases: note edit while loading retained at commit; late persistence
  hydration cannot overwrite activation; same-target dirty conflict stays
  recoverable; failed target leaves current draft intact; account/org changes
  reject stale activation; owned-to-local and local-to-owned switches; browser
  folder references pruned; effective assistant default retained; storage failure
  retains in-memory recovery and reports reload risk.
- [x] Run the new store integration tests RED from the frontend; initial actions
  were absent. Add targeted RED/GREEN reproductions for retry, collection
  preservation, stale undo, account clearing, hydration callback invalidation,
  outgoing draft-loss risk, invalid editor state and ingestion status projection.
- [x] Implement the pure transition plus one Zustand `set(state => ...)` commit.
  Capture the outgoing draft from that latest `state`, not a pre-fetch closure.
  Recover only same-scope local editor fields, keep loaded collections canonical,
  and set readiness/clean baseline in the same transition. If a dirty field cannot
  be reconciled, return `draft-conflict` without discarding it or writing remotely.
- [x] Exclude owned state from the legacy persistence partializer and saved lists.
  Use a versioned namespace `tldw:research-workspace:owned-drafts:v1` with the scoped
  key above. Preserve unknown legacy records unchanged. Fence existing save/switch
  paths with active origin; restore legacy local snapshots only in local mode.
- [x] Run GREEN and all existing workspace storage/migration/API tests. Inspect
  persisted payloads for tokens, owned content in UUID-only maps, and schema churn.

Final store verification: **312/312 tests in 19 files**, including 34 actual store
integration cases; scoped TypeScript passes. Scoped ESLint has zero errors and
16 pre-existing unused-import warnings in the legacy store/list slice, plus the
existing pages-path configuration warning. No new Python changes; Bandit is not
applicable to this checkpoint. Independent review's hydration-resurrection and
outstanding-draft-risk findings were reproduced, fixed, and cleared on re-review.

Additional StudioPane integration check: **25 passed, 2 failed**, identically on
the current worktree and exact tracked HEAD loaded through the baseline Vite
plugin. Both failures expect a one-argument KittenTTS voice-catalog call while
the implementation supplies a second `undefined` argument. Reports:
`/tmp/tldw-owned-studio-current.json` and `/tmp/tldw-owned-studio-baseline.json`.
These existing audio test assertions are not changed or described as passing.

Runtime contract for Stage 3: `beginOwnedWorkspace` requires hydrated storage and
verified scope from its caller; `activateOwnedWorkspace` consumes the validated
loader bundle, while `invalidateOwnedWorkspace` clears the active owned view.
Editor actions persist scoped drafts and `ownedWorkspaceDraftStatus` retains
reload-risk across targets and rehydration. Legacy export/duplicate/archive/delete,
transfer, undo and chat-cache actions fail explicitly in owned mode; the route
must use canonical operations rather than invoking those legacy actions. Wire
the composer to `setOwnedWorkspaceComposer`, render explicit recovery conflicts,
and inspect the complete bundle for server settings beyond the legacy UI shape.

## Stage 3: Gate, Lifecycle, Notes And Editing

**Goal:** Wire owned opening without changing shared-view or legacy-local behavior.
**Success Criteria:** Read-only mount; live status works; explicit edits save; scope changes remove stale content immediately.
**Status:** In Progress

### Automatic-Write And Composer Checkpoint (2026-09-13)

The main page now excludes legacy upsert/source reconciliation, migration,
initialization and Knowledge QA prefill for owned origins. Status readiness comes
from validated activation. Owned status failures do not fall back to generic media
readiness. A view lifetime distinguishes account/server/org changes, including
same-ID and A -> B -> A transitions; pending legacy reconciliation, migration
callbacks, prefill, note selection and status responses cannot mutate the next view.

Main-page global note search uses only the validated canonical note associations,
including full content, keywords and versions, without account-wide/tag search or
generic note-detail requests. Legacy chat caches are excluded from owned search.
QuickNotes notebook/modal reads and canonical note writes are still pending.

ChatPane excludes legacy caching, binds the composer to scoped owned drafts, and
clears global/child history across owned scope lifetimes. Review found optimistic
clearing could lose an unsent draft during source preparation. Owned drafts now
remain until successful submission, preserving newer typing. A subsequent review
found retained text allowed repeated sends during capability preparation; an
immediate in-flight guard and busy Send control now prevent duplicate dispatch.
Both issues have RED/GREEN regressions and passed independent re-review. This does
not yet scope-pin the shared streaming transport or restore owned conversation
history; those remain required before route wiring.

Audio catalog discovery no longer normalizes canonical owned settings on mount;
explicit controls and legacy normalization are preserved. Saved-view review found
its UUID-only controller lifetime could retain or accept another account's views.
The controller now takes a stable operation scope, invalidates on scope/readiness
changes, hides stale views synchronously and rejects retained callbacks/retries.
Commit-time ref invalidation runs before child layout effects and does not cancel
committed requests during abandoned/suspended renders. Independent review cleared
this repair and ran 137 saved-view tests successfully. Request transport pinning
is still required before editable-route integration.

The editable route remains deliberately unwired. Next: canonical QuickNotes
association reads and explicit versioned note editing; then bind all owned
mutation/read dispatches to the verified request context, including streaming chat
and post-submit history callbacks. Finish the remaining Header/source/artifact/
sharing/transfer action audit before mounting the owned editor through the route.
This checkpoint is not a live backend/WebUI/CDP acceptance result and does not
complete Stage 3 or TASK-12020.50.

Checkpoint verification: **747/747 tests across 31 files**, covering the new mount,
composer, audio and saved-view cases plus owned loader/store/lifecycle, legacy
workspace behavior and clone UI/recovery regressions. Scoped TypeScript passed.
ESLint reported zero errors and 18 pre-existing warnings in the large page/chat
files and test harness, plus the existing pages-path notice. The existing route
suite emits an i18next initialization warning. `git diff --check` passed.
The unchanged broader StudioPane suite still has its two previously baseline-
reproduced KittenTTS mock-arity failures; its 25 other tests pass. No Python was
changed in this checkpoint, so Bandit is not applicable. Independent review cleared
the three substantive review findings after regression fixes; all remaining work
listed above is still an integration/acceptance gate, not silently deferred success.

### Opening Foundation Checkpoint (2026-09-13)

Implemented the opening-specific read context, lifecycle hook, and strict selector
parser. The context captures server/auth/org configuration once, preserves server
deployment subpaths, uses the existing request core without automatic credential
refresh, and verifies the backend principal before and after complete bundle reads.
Header-auth requests omit browser cookies; hosted/quickstart reads retain their
canonical session transport. Redirects are rejected. Raw metadata is validated
before assistant-default normalization.

The hook waits for hydration, enforces one 30-second opening deadline, cancels stale
generations, invalidates visible owned data on session/config/page boundaries, and
preserves scoped drafts without treating them as authentication evidence. The new
parser rejects empty, duplicate, malformed and mixed owned/shared selectors while
leaving deep-research return context and the existing shared parser unchanged.

**Not wired to the editable page yet.** `ResearchWorkspaceRouteGate` still uses its
existing parser. Hook StrictMode tests prove opening/store behavior, not a zero-write
editable page. Do not mark Stage 3 or any live acceptance row complete from this
checkpoint. The selector admits UUIDs and existing URL-safe opaque IDs (ASCII
letters/digits/underscore/hyphen, 1-128 characters); it does not normalize identity.

The initial read-only integration audit identified these boundaries. The
automatic-write/composer checkpoint above addresses the mount fences and main
search; the remaining mutation/QuickNotes/transport work is still required:

- `ResearchWorkspace/index.tsx`: legacy reconciliation (~1417), migration (~1754),
  fallback initialization (~2755), and Knowledge QA prefill (~2844) must not run as
  owned-opening side effects. Owned source-status/saved-view readiness must come
  from verified activation, not reconciliation.
- `ChatPane/index.tsx`: legacy session caching (~2131/2150) must be excluded; composer
  state must use scoped drafts, with request/completion scope guards.
- `StudioPane/hooks/useAudioTtsSettings.tsx`: automatic voice normalization (~284)
  must not turn server settings into an unsaved canonical change on mount.
- `index.tsx` note search/detail reads (~2278/2326) and `QuickNotesSection.tsx`
  modal/notebook reads (~444/488) must use canonical workspace associations rather
  than tag matches or account-wide notes. Note save completion (~655) needs scope,
  note-version and submitted-draft revision checks.
- Header, sources, saved views, artifacts, sharing and transfer handlers need the
  same current-scope operation boundary. UUID-only completion checks and remounts
  are insufficient. Unsupported owned actions must not call fenced legacy actions.

Checkpoint verification: 424 tests passed across 19 focused/regression files;
changed-scope TypeScript passed. Lint uses the frontend's installed ESLint binary
with its config from the shared UI directory (not a cached `bunx eslint` version).
No Python was edited for this checkpoint, so Bandit is not applicable. The existing
route test emits an i18next initialization warning; live acceptance remains pending.

Foundation review found two issues and both have regression fixes: principal reads
now use `/api/v1/users/me/profile?sections=identity` rather than the disableable
legacy `/auth/me` endpoint; the 30-second deadline includes waiting for hydration,
does not restart when hydration completes, and Retry reattempts hydration. A failed
required read also aborts sibling requests. The post-review focused run passes
94 tests across the reader, lifecycle, and selector suites. Final broader rerun:
426/426 tests across 19 files; TypeScript, scoped ESLint and `git diff --check`
passed. Follow-up independent review reported no actionable findings in this
foundation checkpoint and independently reran the 94 focused tests.

**Interfaces:** `useOwnedWorkspaceOpening(workspaceId: string)` returns a union of
`{status: "loading"}`, `{status: "ready"}`, or
`{status: "error", reason: "denied" | "unavailable" | "connection" | "timeout" | "invalid-response" | "draft-conflict", retry: () => void}`.
Its ready state is valid only for the current route and verified scope. It consumes
Stage 1's loader and Stage 2's attempt/actions. Do not export a generic auth manager.

- [ ] Extend `ResearchWorkspaceRouteGate.test.tsx`, `research-workspace-route-state.test.ts`,
  `workspace-server-reconcile.test.ts`, and `WorkspacesManagerPage.test.tsx` in their
  existing component `__tests__` directories. Add
  `hooks/__tests__/useOwnedWorkspaceOpening.test.tsx` and
  `components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.owned-notes.test.tsx`.
- [ ] Assert empty/duplicate/invalid `workspace`, combined `shared`/`workspace`,
  denied/deleted/missing targets, and failed required reads never mount an empty
  workspace. No selector stays local; `source_workspace_id` alone retains its
  current return-context meaning. Open from manager and completed clone supplies
  the canonical selector without a redirect.
- [ ] Add StrictMode and ordinary-mount mutation assertions, using the existing
  client mocks:

  ```ts
  expect(client.upsertWorkspace).not.toHaveBeenCalled()
  expect(client.addWorkspaceSource).not.toHaveBeenCalled()
  expect(client.updateWorkspaceSourceSelection).not.toHaveBeenCalled()
  ```

  Also assert source status GET occurs after readiness and an explicit name or
  selection edit still reaches the proper mutation API. A 409 preserves the dirty
  edit and presents conflict recovery rather than overwriting the server.
- [ ] Add fake-timer 30-second timeout/retry tests; account/server/org changes both
  during loading and after ready; token refresh without identity change; same UUID
  in two scopes; unmount cleanup; stale mutation completion cannot affect the new
  scope. Verify scoped draft recovery never bypasses principal verification.
- [ ] Add a canonical note associated with the workspace but not its tag, and an
  unrelated same-tag note. The notebook must include only the associated note;
  searching and selecting it keeps that boundary. Rejected note GET is an error,
  not empty content, and must not clear the unsaved note draft.
- [ ] Run these exact files RED with `bunx vitest run ... --maxWorkers=1` from the
  frontend, using their `../packages/ui/src/` paths.
- [ ] Implement the hook using existing auth/config event and request-context
  patterns from `useSharedWorkspaceClones.ts`, without importing its clone manager.
  Await hydration, verify principal, bind reads to that scope, load with the
  deadline, and activate only if target/scope/generation still match. The route
  gate must reject previous ready state synchronously when its target/scope changes,
  rather than waiting for an effect to hide old content.
- [ ] Separate opening readiness from legacy reconciliation in `index.tsx`. Route
  all owned mutation entry points through current-scope checks and pinned request
  identity; discard old completions. Load/search saved notes by workspace association.
  Keep drafts independent. Preserve status/capability degradation and do not start
  sandbox/MCP/agent resources as an opening side effect.
- [ ] Add localized copy: "Couldn't open this workspace", "You no longer have
  access to this workspace", "Workspace loading timed out", "Retry", and "Back
  to Workspaces". State-specific detail must not expose another account's content.
  Use existing DS error/loading surfaces and keyboard-focus conventions.
- [ ] Run GREEN plus shared-recipient route tests, clone UI/hooks, connection,
  workspace persistence, and route-path tests. Check lint and changed-scope types.

## Stage 4: Live Acceptance And Review

**Goal:** Close the actual target-opening blocker with production runtime evidence.
**Success Criteria:** Exact copied content/notes, zero opening mutations, real grounded answer/citations, honest remaining gaps.
**Status:** Not Started

- [ ] Reuse isolated fixture `/tmp/tldw-clone-uat-jIXQx6` only after inspecting its
  scripts/configuration. Confirm task-owned ports are free, current dependencies
  resolve to one Vitest instance, and no runtime points at user databases. Keep
  credentials out of tracked files and command output.
- [ ] Start the real backend/Jobs and WebUI with isolated data; attach Playwright
  to Chrome through CDP. Verify configured provider/model availability, without
  requiring a specific provider or using a mocked completion.
- [ ] Open the completed clone in a clean profile, reload, and compare the route,
  visible workspace ID/name, source selection, saved notes, settings and artifacts
  to authenticated API reads. Record that opening emitted no workspace/source
  mutations. Wait for explicit text/citation readiness before asking a question
  about the fixture sentinel and following the cited source.
- [ ] Exercise an outgoing unsaved note, a failed target load, retry, account/server
  change, and same-target reopen. Record distinct scope and failure behavior, not
  merely successful HTTP status codes. Re-run existing clone recovery/fault cases
  under the separate acceptance plan; this loader alone does not certify them.
- [ ] Update the live matrix and Backlog with exact revision plus uncommitted-diff
  fingerprint, runtime configuration (redacted), screenshots/network evidence, and
  PostgreSQL/provider limitations. Do not replace older failed evidence with a
  success claim; append the verified repair result.
- [ ] Run focused regressions, broader clone integration coverage, ESLint, and
  `git diff --check`. If Python changes become necessary, use the project venv,
  test SQLite/PostgreSQL behavior, and run Bandit on touched Python paths. For this
  frontend-only slice, record Bandit as not applicable, not as an executed pass.
- [ ] Review the diff against all five findings and the spec. Stop task-owned
  processes and remove only temporary dependency links created for this run.
  Commit only separately approved, verified scope; no PR/merge is implied here.

## Sequencing And Review Gates

Stages 1 and 2 have separate test boundaries. Agree the bundle/scope interfaces
first; after that, loader tests and pure draft-transition tests can be developed
independently with non-overlapping file ownership. Stage 3 depends on both and owns
the route/store wiring. Stage 4 depends on verified Stage 3. Do not dispatch parallel
edits to `workspace.ts` or `index.tsx`.

The loader and scoped store activation are implemented and reviewed, not full
opening completion. The storage-harness failure is repaired, and review findings
have regression coverage. Next is Stage 3 route/account-lifecycle integration,
canonical note loading, and explicit mutation handling. Stage 4 supplies live
backend/WebUI/CDP acceptance; no acceptance row passes from unit tests alone.

## Canonical Notes Request Checkpoint

At this earlier checkpoint, the notes transport was available but QuickNotes was
not yet connected. The editor checkpoint below supersedes that integration status;
Stage 3 remains in progress and the owned route remains disabled.

- `createOwnedWorkspaceNotesContext` reuses the opening connection capture, checks
  it against the activated server/principal/organization, and exposes canonical
  workspace-associated list, create, and versioned update operations. The public
  opening interface still exposes reads only.
- Each notes request sends the existing `X-TLDW-Expected-User-ID` assertion. The
  four backend notes routes enforce it before acquiring the notes database. The
  header is optional for existing clients; a mismatch returns 412 with no-store.
- The transport captures credentials and submitted fields, does not refresh or
  automatically retry writes, rejects invalid or foreign receipts, requires a
  safe positive update version, and discards aborted completions. Explicit empty
  keyword updates are preserved. A save cannot silently adopt subsequent edits
  made to its input object while request setup is awaiting configuration.
- TDD evidence: 33 new transport cases failed before implementation; two additional
  mutation-snapshot cases reproduced the asynchronous input race before its fix.
  The backend guard had five expected pre-fix failures.
- Verification: 782 tests passed across the 31-file frontend regression set;
  105 tests passed across workspace API, subresource, and route-binding suites.
  Scoped TypeScript passed. ESLint has zero errors and three existing optimistic
  update helper warnings; the known shared-UI pages-path notice remains. Bandit
  on the touched production endpoint and `git diff --check` passed. Independent
  review found no actionable defect in this checkpoint.

Next editor work must use this context with an activation-lifetime AbortSignal,
load notes and keyword suggestions by canonical association, preserve newer local
edits when a save returns, update canonical note versions without changing another
editor session, and provide explicit conflict recovery. Ambiguous create failures
must not trigger automatic retries because the notes POST has no idempotency key.
Do not expose owned Clear/Undo through the fenced legacy undo manager.

The hosted browser-fetch tests prove request construction, not deployed proxy
forwarding. Live acceptance must verify that `/api/proxy/*` forwards the expected
user header and that an account switch returns 412 without a notes write. This
checkpoint is not evidence of live editor functionality or full clone acceptance.

### Prior Contract Blocker

At the preceding checkpoint, contract refresh was blocked by fingerprint drift. The canonical
export succeeds with `PYTHONPATH=apps/mcp-unified/src:packages/tldw_profile_core/src`
and produces 2094 paths / 3203 schemas, while the checked-in fingerprint describes
2094 / 3202. Read-only diagnosis found that the notes guard adds only four optional
header parameters and no schemas; the prior sharing dependency edit has no schema
effect. The additional `PromptPersistenceAuthorization` schema is already committed
and unrelated to this slice. Removing those known differences still does not
reproduce the checked-in hash, so environment causation is unproven and further
baseline comparison is required. Do not refresh the checked-in fingerprint until
that discrepancy is classified. Temporary exported evidence is in
`/tmp/tldw-owned-notes-openapi.json` and its companion fingerprint/log files.

### Contract Reconciliation

Resolved by a same-runtime historical comparison, without reverting working files.
The Python sources at `dd44c15cbd` (the last fingerprint-writing commit) were loaded
using a read-only import overlay for changed files. Their canonical export itself
produces 2094 paths / 3203 schemas and hash `e7c4c68fc3ce...`, not the fingerprint
checked in at that commit. Comparing that full historical schema to this worktree
shows differences only at `/api/v1/workspaces/{workspace_id}/notes` and its
`/{note_id}` path: the four optional expected-user headers. All component schemas
match. This establishes inherited fingerprint drift and isolates this slice's
actual contract change; it does not establish why the old export was different.

Regenerated the canonical fingerprint and ignored frontend API types using the
existing exporter and installed `openapi-typescript` CLI. A separate fresh export
passes `--check`; current hash is `e2e558143fea...`, 2094 paths / 3203 schemas.
The exporter requires the two monorepo Python source roots in PYTHONPATH with this
existing virtual environment; no dependencies or unrelated API behavior changed.
Evidence: `/tmp/tldw-owned-openapi-fingerprint-revision.{json,log}` and
`/tmp/tldw-owned-notes-contract-{refresh,check}.log`. Contract refresh is no longer
blocking the editor integration.

## Canonical QuickNotes Editor Checkpoint

The owned editor now uses the canonical notes context. Its list, search, selection,
and keyword suggestions use workspace associations, not matching tags or global
notes. Mounting performs no note writes. Existing legacy behavior is retained,
with stale asynchronous completions fenced across workspace/account changes.

- Save receipts update the canonical note/version separately from the editor.
  Newer typing is retained as dirty, while a changed editor session cannot be
  overwritten by an old save. Concurrent refresh cannot resurrect a removed note.
- A version conflict offers explicit recovery containing the latest server text
  and the current unsaved draft for review, preserving both titles/content and
  keywords instead of merely advancing the version and overwriting remote edits.
- Before creating a note, the scoped draft durably records an uncertain-create
  marker. Unavailable recovery storage blocks dispatch. An ambiguous result keeps
  that marker across remount/reload and blocks repeated creates; explicit refresh
  lets the user inspect saved notes. Definite rejection and accepted receipts clear
  the marker. This is client recovery protection, not server POST idempotency.
- Clear/Undo is bounded by the existing undo window, captured activation, editor
  session and unchanged cleared draft. It does not use the legacy global undo path.
- Notes reads bypass the browser cache; malformed keyword payloads fail validation
  rather than silently becoming empty editable fields.

Independent review cleared the store/transport/editor changes and confirmed the
historical OpenAPI comparison. The final regression run passed 859 tests in 35
files; the separate direct-HTTP checkpoint passed 13 requests against the isolated
real backend. Browser/proxy behavior is not inferred from either result. New
localized copy is under `playground:studio.ownedNotes`, with resource and extension
mirror tests. Final review caught nine missing extension mirror entries; their
focused parity test failed before the entries were added and passes afterward.
Re-review found no remaining issue in this scope. The broader existing
`playground-locale-mirror.test.ts` still fails on `composer_rolePlaySaveError`, an
omission independently confirmed in HEAD and unrelated to this checkpoint.
Scoped loader/service/editor TypeScript checks passed; scoped ESLint has
zero errors, five existing QuickNotes warnings and the existing pages-path notice.
Formatting of the new hook/test and `git diff --check` passed. No Python source
changed in this editor checkpoint, so it does not require a new Bandit run; the
preceding notes endpoint checkpoint records its separate security verification.
Final regression log: `/tmp/tldw-owned-quicknotes-combined-final.log`.

Remaining Stage 3 work: pin and fence Header/settings, sources/ingestion,
artifacts, sharing/transfer and chat streaming/history mutations; then connect the
owned route and its Open link producers. Stage 4 still requires real
backend/WebUI/CDP acceptance, including hosted proxy assertion forwarding,
account switches, grounded chat/citations and the recovery/fault matrix.

## Canonical Metadata Request Checkpoint

Stage 3 remains in progress. `createOwnedWorkspaceMetadataContext` now exposes
explicit authenticated GET and versioned PATCH for the activated workspace,
reusing the frozen connection and existing patch serializer. It has no create
operation, does not refresh credentials or retry writes, and excludes archive and
project-profile changes from this metadata-only interface.

- The GET/PATCH backend routes enforce the optional expected-user assertion as
  their first dependency, before workspace DB access or assistant-reference
  validation. Existing unasserted clients remain compatible; PUT is unchanged.
- Responses pass the same complete workspace validator as opening before
  assistant-default normalization. Updates require a safe positive version and a
  nonempty patch, and reject foreign, unavailable or non-advancing receipts.
- Submitted fields, including nested assistant defaults and explicit null/empty
  values, are captured before asynchronous request setup. Account/org/server
  mismatches and aborted lifetimes prevent dispatch or discard late results.
- Test-first evidence: 25 missing-context failures preceded implementation; three
  additional lifecycle-dispatch failures preceded the metadata-only restriction.
  The focused context/loader/opening run passes 182 tests. The broader 35-file run
  passes 889 tests; existing workspace client/domain suites pass another 37.
- Backend verification passes 159 tests including auth hardening; a separate
  parent rerun of the four workspace/guard files passes 112 tests. Bandit reports
  no findings in the endpoint or tests (assertion-only B101 excluded for tests).
  Existing warnings and pytest temporary-directory cleanup warnings remain in
  the verification logs. Independent backend and frontend reviews are clear.
- Canonical OpenAPI export/types were regenerated and a separate fingerprint
  check passed. Structured comparison with the preceding notes export confirms
  only two optional expected-user header additions; paths/schemas remain 2094/3203.
- Real isolated backend verification passes 14 HTTP requests, including successful
  versioned edits, 409 stale version, GET/PATCH account mismatch 412/no-store,
  unasserted compatibility, missing-target PATCH 409 followed by GET 404, and
  temporary-workspace cleanup. Missing-target PATCH 409 is the existing DB
  contract, not a newly introduced status or a successful creation.

Header rename still uses its legacy local-state handler and is not wired to this
context yet. The next slice must retain the rename draft and its starting version,
commit scoped metadata receipts without overwriting newer edits, present explicit
conflict recovery, and reject callbacks from an old account/workspace/editor.
Do not substitute a fresh GET version and silently overwrite a conflict. Complete
the other mutation boundaries before enabling the owned route.

Verification logs: `/tmp/tldw-owned-metadata-{final,regressions,domain-regressions}.log`;
HTTP evidence: `/tmp/tldw-clone-uat-jIXQx6/owned-metadata-http-evidence.json`.
Scoped TypeScript passes. New service/validator scope has no ESLint errors (three
existing optimistic-helper warnings); the serializer's containing domain file
still has its existing `no-control-regex` error at line 966, reproduced from HEAD.
This is transport/backend verification, not live WebUI/CDP acceptance.

## Owned Header Rename Integration Checkpoint

Stage 3 remains in progress. The next bounded implementation connects only Header
rename to the canonical metadata context; the owned route stays disabled until
the remaining mutation boundaries are certified.

- A scoped rename draft carries the user's text and original version separately
  from canonical metadata. The optional draft field is compatible with existing
  stored drafts; reopening never silently substitutes the latest server version.
- Receipt application checks activation identity and the expected metadata
  object, not the whole bundle, so a concurrent notes refresh remains safe. It
  refreshes clean metadata projections and their baseline while preserving
  unrelated local edits. Newer typing and cancelled/reopened editor sessions are
  not discarded by a late response.
- Conflict recovery requires an explicit authorized GET and user choice before
  accepting the server name or resubmitting against its inspected version. No
  write retry or create-capable PUT fallback is introduced.
- Recovery review identified an unchanged-version case: a failed PATCH may never
  have applied. An explicit GET review may therefore accept the same version;
  the separate PATCH receipt action still requires a strictly newer version.
  This avoids trapping an unchanged draft after a successful recovery read.
- Initial store TDD: all 12 new receipt/draft cases failed on missing actions;
  implementation passes those cases plus the preceding 55 store tests. Five
  additional persistence cases cover reload and invalid base versions. The
  initial store/state/legacy persistence check passes 133 tests; scoped TypeScript and
  ESLint pass (existing pages-directory configuration notice only).

Header rename now uses the canonical context for owned workspaces and preserves
the existing local handler for legacy workspaces. Pending, account/storage error,
and explicit conflict choices appear beside the existing editor, not in a new
workspace banner. Cancellation before context setup completes suppresses dispatch;
an already-dispatched receipt may refresh canonical metadata without overwriting
a cancelled/reopened editor. English UI copy and extension mirrors are tested.

Final verification: **1047 tests pass across 40 files**, including 22 rename-hook,
9 owned-Header and 69 unchanged legacy-Header tests. Store/state tests pass 115
cases; the two same-version review cases were added after their missing-action RED.
An additional RED/GREEN assertion prevents unchanged review results from falsely
claiming a server conflict. Scoped Header/hook/store TypeScript checks pass.
Scoped ESLint has zero errors and one pre-existing Header unused-value warning
(`formattedStorageUsage`, also present in HEAD), plus the existing pages-path
notice. New/updated focused files pass formatting; `git diff --check` is clean.
Independent store and hook/UI reviews, including the recovery fix, are clear.
No Python changed in this checkpoint, so Bandit is not applicable to this TS-only
delta; the earlier backend checkpoint retains its separate Bandit evidence.

Combined log: `/tmp/tldw-owned-rename-combined-final.log`. Tests retain existing
React/AntD/i18n and intentional storage-failure diagnostics; no test was skipped
to obtain this result. No commit, push or route enablement occurred. This is not
real WebUI/CDP acceptance and does not complete Stage 3 or TASK-12020.50.

Next: bind the remaining Header settings/default-assistant mutations to the
canonical metadata context, then the source/artifact/sharing/chat boundaries.
Do not enable the owned route before those boundaries and live acceptance pass.

## Owned Default-Assistant Checkpoint

Stage 3 continues with the existing Header default-assistant modal. Scope is
versioned assignment/clearing of an existing Persona default, not banner/image
storage or a new settings framework. Legacy workspace behavior stays separate.

- The assistant draft persists only selected Persona, memory mode and base
  version. Read-write consent is ephemeral and must be confirmed again after
  selection, account or conflict-review changes. Drafts never become canonical
  settings until an authorized receipt is accepted.
- Metadata receipts reuse rename's scoped metadata projection rules while each
  editor keeps an independent session/draft. A notes refresh remains compatible;
  a concurrent canonical metadata replacement invalidates an older receipt.
- The existing Persona client actually reads `/persona/catalog`. That route can
  bootstrap a default Persona when empty. A backward-compatible
  `ensure_default=false` mode makes the owned picker read-only; optional
  expected-user assertion runs before DB and feature dependencies. Legacy
  callers retain the default behavior. `/persona/profiles` is unchanged.
- Owned catalog requests use the captured metadata connection, validate choices,
  reject malformed/duplicate rows, and do not turn failed requests into empty
  successful lists. Catalog failure is distinct from metadata failure so an
  unavailable/deleted default can still be cleared when workspace metadata is
  authorized and available.
- Conflict handling retains the original draft/version, requires explicit review
  and acknowledgment, and never silently refreshes the version to retry a write.
  Explicit unchanged-version review remains valid after an unapplied failed save.

The owned Header modal now uses this context for opening, versioned assignment,
clearing and explicit conflict inspection. Legacy handlers stay separate. A
cancelled/reopened modal is not replaced by an already-dispatched receipt, and
restored read-write selections always require fresh confirmation. The existing
modal contains the error/review copy; no workspace-wide banner was added.

Final checkpoint verification: **1117 tests pass in 42 frontend files**, including 33
assistant-hook, 10 owned-assistant Header, 69 legacy Header and the existing rename
regressions. Store/service/state foundation review is clear. Scoped UI TypeScript,
focused formatting and `git diff --check` pass. ESLint reports zero errors, the
existing Header `formattedStorageUsage` warning, and the existing pages-path
configuration notice. Combined log: `/tmp/tldw-owned-assistant-combined-final.log`.
Independent modal review is clear. Scoped store TypeScript also passes.

Backend tests pass **67 cases**, including optional expected-user dependency
ordering, empty read-only catalog, legacy bootstrap behavior and workspace
subresources. Bandit reports no findings in the changed Persona endpoint or tests
(only test assertion B101 excluded). The regenerated OpenAPI contract matches its
fingerprint: 2094 paths, 3203 schemas; the catalog adds two optional parameters.
Actual isolated FastAPI HTTP verification passes 16 requests covering API-key
auth, no implicit creation, account mismatch, explicit read-write confirmation,
stale-version conflicts, set/clear and cleanup. Lifespan orchestration was disabled;
see the acceptance report for precise scope and local evidence locations.

No route enablement, commit, push or full browser acceptance occurred. Stage 3 and
TASK-12020.50 remain in progress. Next: canonical Header banner/settings behavior,
then remaining source/artifact/sharing/chat mutations before the real WebUI/CDP
walkthrough. The API does not currently persist banner images; resolve that
storage contract explicitly rather than treating a browser-only image as durable.

## Owned Banner Text Checkpoint

User approved a bounded text-first slice: use the existing canonical metadata
GET/PATCH for title/subtitle save and confirmed reset. Preserve `banner_color`
and all other fields. Disable image upload for owned workspaces with an explicit
storage limitation; leave legacy workspace image behavior unchanged. Durable
image storage, cleanup and clone semantics remain a separate design decision.

- [x] Add optional scoped persisted text/base-version draft, independent editor
  session, strict no-image schema, and guarded metadata receipt/review actions.
- [x] Integrate the existing Header modal without mount writes or legacy owned
  updates. Keep drafts on failure, require explicit conflict review, and prevent
  late saves/reset confirmations/image processing from affecting another editor.
- [x] Verify the existing backend via real isolated HTTP save/reset and reload,
  account/version rejection, unrelated-field preservation and cleanup.
- [x] Complete component/regression verification, independent review and record
  the precise remaining live WebUI/CDP acceptance boundary.

Initial TDD: all 14 new store/persistence cases fail before implementation; all
144 cases in those two files pass afterward. Scoped store TypeScript passes.
Real FastAPI probe passed 13 requests with API-key auth and isolated SQLite;
lifespan startup was off, so it does not certify full application startup or
multi-user/hosted operation. Backend stopped automatically after verification.
Logs: `/tmp/tldw-owned-banner-store-{red-2,green}.log` and
`/tmp/tldw-owned-banner-http.log`. No new backend API or schema required.

Final regression: **1165 tests pass in 44 files**, including 25 banner-hook and
9 owned-banner Header tests. After refining the legacy callback identity object
to remove a new lint warning, a fresh focused run passes 103 banner/legacy Header
tests. Scoped UI/store TypeScript, focused formatting and `git diff --check`
pass. ESLint has zero errors and only the pre-existing Header unused-value warning
plus the existing pages-path notice. Independent foundation and modal reviews,
including the callback refinement rereview, have no findings.

Additional RED/GREEN coverage catches missing Header integration and save-time
trimming/length limits; restored drafts must obey the same bounds as typed text.
Reset keeps its exact modal callback and has no hidden retry intent after a
conflict. Legacy image processing checks scope, editor and upload identity before
publishing success, error or completion. Owned image actions are disabled with
accessible explanatory text. No image storage was added.

Logs: `/tmp/tldw-owned-banner-combined-final.log`,
`/tmp/tldw-owned-banner-parent-postreview.log`,
`/tmp/tldw-owned-banner-parent-ui-tsc-final.log`, and
`/tmp/tldw-owned-banner-parent-eslint-final.log`. This slice changes no repository
Python; Bandit is not applicable to the TypeScript-only delta. Prior endpoint
Bandit evidence remains separate.

Stage 3 remains in progress and the owned route stays disabled. Remaining Header
lifecycle actions and source/artifact/sharing/chat mutation boundaries precede
route enablement and real WebUI/CDP acceptance. No commit or push occurred.

## Owned Archive/Restore Checkpoint

User approved account-pinned versioned archive/restore, retention of local drafts,
explicit conflict and uncertain-response review, and restoration through the
existing Workspaces manager. Legacy local archive/Undo remains unchanged. Delete,
duplicate, route enablement and unrelated manager redesign are excluded.

- [x] Add a lifecycle-only captured request context and archived-record validator;
  preserve the stricter active-workspace metadata validator.
- [x] Preserve current drafts before archive dispatch and latest drafts before
  clearing the active view. Refuse to leave when persistence is unavailable.
- [x] Pin directory/context reads to the verified account and reject mismatched
  expected-user requests before database acquisition.
- [x] Connect the owned Header confirmation and explicit status recovery without
  local undo, automatic retries or stale response/navigation effects.
- [x] Guard manager archive/restore against account changes, stale callbacks and
  uncertain writes; require a refreshed server version before another attempt.
- [x] Verify real isolated HTTP archive/restore, regressions, scope typechecks,
  lint, Bandit and independent review; record remaining acceptance limits.

Foundation RED: 26 failures before service/store implementation; backend directory
guards fail twice before dependency binding. GREEN: 293 loader/service/store tests
and 30 backend tests pass. These results are not full WebUI/CDP acceptance.

Final verification: **1328 frontend tests in 51 files** pass (1244 workspace
regressions plus 84 directory tests). Scoped TypeScript includes service/store,
Header, actual workspace shortcut tests, manager and root-panel tests. Endpoint
Bandit reports no findings; OpenAPI fingerprint and ignored generated types were
refreshed and the fingerprint checked. ESLint has zero new warnings/errors; the
existing Header unused-value warning and pages-path notice remain. The complete
design-system guard still fails on pre-existing findings. Comparison of touched
workspace screens against HEAD finds two existing canonical-label findings and
no introduced findings; the new recovery message uses `DsAlert` without an
exception.

Independent review findings were fixed with RED/GREEN tests and rereview:

- Check target-draft durability, not unrelated pending drafts from another account.
- Validate and narrow directory context to the fields the manager consumes;
  reject foreign operation identity and discard unchecked context fields.
- Preserve same-account create/rename/root drafts and selected workspace during
  revalidation. Hide/disable the retained view until verified; discard it on an
  actual account boundary.
- Suspend retained root-panel polling and ignore retired async completions without
  resetting editor input or replaying writes.
- Prevent workspace shortcuts from bypassing archive completion recovery when
  draft storage is unavailable, while retaining Tab/Enter recovery interaction.

The isolated real API-key/SQLite backend passed **19 HTTP checks**, preserving
notes and metadata through archive/restore and rejecting stale/wrong-account
requests. Lifespan was off; full startup, JWT/hosted cookies, PostgreSQL and CDP
acceptance remain outside this checkpoint. Backend stopped after the probe.

Evidence: `/tmp/tldw-owned-archive-regression-final.log`,
`/tmp/tldw-owned-archive-directory-postreview.log`,
`/tmp/tldw-owned-archive-tsc-postreview.log`,
`/tmp/tldw-owned-archive-eslint-final.log`,
`/tmp/tldw-owned-archive-design-system-delta.log`, and
`/tmp/tldw-owned-archive-http.log`. Final scoped reviews have no findings.
TASK-12020.50 and Stage 3 remain In Progress; owned route enablement,
delete/duplicate and other unfinished mutation boundaries still precede full
WebUI/CDP acceptance. No commit, push or PR action occurred.

## Deletion And Unmerged-Stream Review (2026-09-20)

The approved deletion corrections are tracked in
`2026-09-20-owned-workspace-deletion.md`. Local transaction/API foundation is a
prerequisite, not a claim that child-writer fencing, sharing cleanup, or the owned
deletion UI is complete. The owned route stays disabled.

A fresh independent review of the earlier .49/.50 patch found six open defects,
recorded in `../../Development/workspace-unmerged-review-2026-09-20.md`. They cover
prehydration persistence, principal-bound cookie reads, concurrent drafts,
cross-tab clone scope, target-vs-global durability and the legacy readiness probe.
These findings supersede earlier clean checkpoint reviews for merge readiness.
The 1328-test frontend regression rerun is green but does not cover these newly
reproduced cases. Correct the findings before treating the stream as merge-ready
or resuming route enablement.
