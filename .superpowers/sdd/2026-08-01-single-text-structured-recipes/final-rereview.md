# Track B final whole-branch re-review

## Verdict

**NEEDS FIXES.** The final fix range
`2ee920220b..e751f76395` closes the ordinary null-owner, different-owner,
unknown/provisional, durable-restart, authority-failure, menu-visibility, and
v1 compatibility cases from the prior review. It does not close two
same-local-ID races at the authority/Dexie boundary, and one remaining hook
still reports a false pull result as success.

There are two Important findings and one Minor finding. No Critical finding was
found. This is the one permitted final re-review; no further fix wave is
assumed.

## Findings

### Important 1 — pull releases exact-ID authority before publishing the final local status

`apps/packages/ui/src/services/prompt-sync.ts:525-563` first copies server
content while retaining `syncStatus: "error"`, which is correct. It then calls
`reconcileRecipePersistenceExact` at line 546. The registry implementation
atomically checks the exact ID and immediately clears the matching scoped
marker (`apps/packages/ui/src/services/recipe-persistence-registry.ts:66-78`).
Only afterward does `prompt-sync.ts:556-560` asynchronously update the Dexie
row to `syncStatus: "synced"`.

That leaves an unguarded interval between the authority clear and the final
local write. A second create/update can reserve the same ID in that interval.
If that request settles ambiguously and writes the durable error before the
pull's final Dexie continuation, the pull overwrites the new durable lock with
`synced` while the new scoped registry marker remains. A reload or MV3 worker
restart then drops the only remaining guard and makes another duplicate write
possible. This is the same restart/data-loss consequence as prior Important 1,
now reachable through a narrower concurrency window.

Reviewer probe: I temporarily paused the second `db.prompts.update`, installed
a fresh same-ID dispatch reservation after `reconcileExact` returned, and
required pull to retain `error`. The isolated run failed exactly at that
assertion: **1 failed / 56 skipped**, returning `success:true` and
`syncStatus:"synced"`; the registry still read `scoped`. The probe was removed
and the worktree restored before the maintained gates.

Required correction: exact reconciliation needs a token-bound exclusive phase
held across the final local status commit, rather than clearing the marker and
exposing a clear registry first. New dispatch reservation must remain blocked
until the local commit completes. A failed local commit must retain or restore
the matching scoped evidence and durable error. Cover direct and background
authority, successful commit, local write failure, release failure, and a new
reservation attempted between authority acquisition and local commit.

### Important 2 — unlink uses a stale durable-status check before acquiring its lease

`unlinkPrompt` reads the row at `apps/packages/ui/src/services/prompt-sync.ts:1141`,
checks that snapshot for `syncStatus === "error"` at lines 1168-1169, and then
awaits lease acquisition at lines 1170-1180. Its eventual Dexie update at lines
1183-1194 is an unconditional object update; it never rechecks the current row
inside the write transaction.

A concurrent ownerless/untrusted pull can therefore write copied server
content plus the required durable `error` after unlink's initial read but
before lease acquisition. That pull creates no scoped registry marker because
it has no trustworthy actual owner, so `beginExclusive` still succeeds.
Unlink then replaces the newly written error with `local` and removes
`serverId`/project linkage. The blocked pull returns an error, but the durable
restart lock and the reconciliation pointer are already gone. After restart,
Save-as-new can duplicate the possibly existing remote mutation. The lease
correctly blocks authority-reserved writes while held, but it does not make the
earlier Dexie status read part of the protected commit.

Reviewer probe: I temporarily delayed `beginRecipePersistenceUnlink` after
unlink's initial row read, completed a null-owner pull that wrote `error`, then
released lease acquisition. The isolated run failed exactly at the safety
assertion: **1 failed / 56 skipped**; unlink returned `success:true` /
`syncStatus:"local"` instead of preserving `error`, `serverId:101`, and the
project linkage. The probe was removed afterward.

Required correction: after acquiring the exact-ID lease, perform the v2
durable-status/linkage validation and unlink mutation in one Dexie
readwrite/callback transaction. If the current row is now `error` or otherwise
no longer matches the validated linkage, return a blocked result using the
fresh row and leave all linkage/status fields intact. Add the ownerless-pull
interleaving plus authority-loss and write-failure controls; retain the current
v1 path.

### Minor 1 — Prompt Studio import still treats a false pull result as success

The fix correctly makes `pullFromStudioMutation` and `unlinkPromptMutation`
throw on `SyncResult.success === false`, and keep-server leaves its recovery
modal open. However, `importFromStudioMutation` at
`apps/packages/ui/src/components/Option/Prompt/hooks/usePromptSync.tsx:319-333`
still returns `pullFromStudio(serverId)` directly and always runs its success
notification. `pullFromStudio` can select an already-linked local recipe by
`serverId` and return a blocked false result, so this path can show “Prompt
imported” while the row remains error-locked.

Reviewer probe: a temporary hook test returned the same blocked false pull
result used by the maintained pull/unlink tests. The isolated run failed **1
test / 8 skipped** because `notification.error` had zero calls; the import
success path ran instead. The probe was removed.

Required correction: apply the same false-result check to import before its
success callback and invalidate the prompt query on settlement.

## Prior-finding disposition and contract audit

- Pull now copies v2 content without initially overwriting the durable error,
  rejects null actual owner, rejects a different/residual owner, and refuses
  unknown, provisional, or exclusive state. A matching owner can reconcile a
  restart-only durable lock. The remaining defect is the post-check/pre-commit
  race in Important 1.
- Unlink now hides for known durable-error recipes in `PromptActionsMenu`,
  rejects durable error and all current registry uncertainty, uses an exact ID
  plus UUIDv4 token, and releases only the matching token. The remaining defect
  is the stale Dexie check in Important 2. Pull remains visible as the intended
  recovery action.
- Direct WebUI and extension background use the same pure registry semantics.
  Local IDs are non-empty and bounded to 512 characters, owners must match the
  opaque lowercase SHA-256 form, operation tokens must be canonical UUIDv4,
  extra message fields are rejected, and boolean protocol responses disclose
  neither competing owners nor transport material.
- The new messages carry only exact local ID, opaque owner when required, and
  unlink operation token. No API key, bearer token, cookie, authorization
  revision, headers, request snapshot, or server body field can supply or
  override reconciliation authority. The maintained malformed/credential-
  bearing protocol tests passed.
- Schema-v1 pull and unlink retain their existing non-owner-aware behavior.
  The fix range contains no backend schema/rendering/persistence change, no
  capability-support change, no Prompt Assist/composer placement or layout
  change, and no dependency or lockfile change.

## Fresh verification

All positive runs below used clean production/test sources at `e751f76395`.
Temporary review probes were removed before these commands.

- Exact focused affected gate, shuffled seed 12984: **9 files / 1,192 tests
  passed**.
- Maintained owner/sync gate from the recovery plan, shuffled seed 12984:
  **34 files / 1,690 tests passed**.
- WebUI `/chat` recipe Playwright spec: **6/6 passed** in 1.3 minutes.
- Packaged-extension recipe Playwright spec: production Chrome MV3 build
  completed and **5/5 passed** in 1.1 minutes.
- Extension `bun run compile`: exit 0 (`tsc --noEmit -p tsconfig.compile.json`).
- Backend capability controls: **2/2 passed**; the fix range has no Python
  product change.
- ESLint over the six changed production TS/TSX files: exit 0, **0 errors / 109
  existing-style warnings** concentrated in the large background file and
  untouched hook lines. The new authority/lease paths emitted no error.
- `git diff --check 2ee920220b..e751f76395` and worktree `git diff --check`:
  exit 0. No package manifest or lockfile changed.

The green suites substantiate the intended ordinary-path behavior, v1
compatibility, protocol validation, direct/background parity, browser journeys,
and compilation. They do not invalidate the three controlled failing probes;
the two Important races are absent from the maintained test matrices.

## Status

`TASK-12984.2` remains **In Progress**. This review changes no production or
test source and does not modify the deferred Improve-button placement.
