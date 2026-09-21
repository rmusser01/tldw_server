# Track B scoped final re-review 2

## Verdict

**NEEDS FIXES.** The scoped fix range `4d5647f291..2e8f287135` closes the
stale-unlink race and the false Prompt Studio import result. It also closes the
ordinary post-reconciliation reservation/clear race, but its new release-
failure compensation can still exit without restoring either guaranteed
durable or authority-side evidence. There is one Important finding and no
Critical finding.

## Finding

### Important — a failed durable compensation write skips authority cleanup and can expose a synced recipe

`apps/packages/ui/src/services/prompt-sync.ts:581-606` correctly treats a
missing committed-release acknowledgement as ambiguous, but the catch block
awaits the durable `syncStatus: "error"` update at lines 592-595 before it
attempts the token-matched abort and scoped re-mark at lines 596-602. If that
Dexie update rejects, execution leaves the catch immediately. Neither
authority operation runs.

This is materially unsafe for the exact ambiguity the branch is compensating:
the background authority may already have executed the committed finish (or
may execute it after the page-side timeout), clearing the lease and scoped
marker, while the failed Dexie write leaves the row `synced`. The outer pull
catch then reads that synced row and does not add `recipeWriteBlocked`
(`prompt-sync.ts:1079-1099`). At that point there is no guaranteed guard in the
current process, and after a worker/page restart another write can be sent for
an operation whose remote outcome was not safely reconciled.

The maintained release-failure test at
`apps/packages/ui/src/services/__tests__/prompt-sync.uncertainty.test.ts:657-697`
only rejects the first `finishRecipePersistenceReconciliation` call; its
subsequent Dexie compensation succeeds, so it does not exercise this branch.

Required correction: make the abort/re-mark compensation execute even when
the durable error rewrite rejects (and treat an update count of zero
explicitly). Preserve the ordering that prevents a late committed finish from
erasing the re-mark, and return a blocked/unsafe result rather than allowing
the outer catch to describe a still-synced row as unblocked. Add a control with
an ambiguous committed finish plus a rejecting durable compensation write;
assert that authority compensation is still attempted and the result remains
write-blocked.

## Prior-finding disposition

- **Important 1 — NOT ADDRESSED.** The UUIDv4 reconciliation lease is exact
  ID/owner/token-bound, remains held across the final Dexie synced commit,
  blocks same-ID reservations, and prevents an ordinary matching-owner clear.
  A successful local commit clears only through an exact committed finish; a
  failed final local commit aborts while retaining the scoped marker. However,
  the required release-failure fail-closed behavior is incomplete because the
  new double-failure path above skips both authority compensations.
- **Important 2 — ADDRESSED.** V2 unlink snapshots linkage, acquires the exact
  token-bound lease, then re-reads durable status and validates all linkage
  fields in the same Dexie readwrite transaction as the unlink mutation. A
  fresh ownerless-pull error or changed linkage returns a blocked result using
  the fresh row; a write failure rolls back; the lease is released in
  `finally`. The v1 direct-update branch is unchanged.
- **Minor 1 — ADDRESSED.** Prompt Studio import now throws when
  `pullFromStudio` returns `success:false`, cannot reach the success
  notification, reports the returned error, and invalidates
  `fetchAllPrompts` from `onSettled`.

## Contract and scope audit

- Direct WebUI and extension background paths share the same typed registry
  implementation and boolean-only facade. Reconciliation and unlink leases
  cannot release one another; exact ID, opaque SHA-256 owner, canonical UUIDv4
  operation token, and exact message key sets are validated.
- Unknown/provisional state and another owner prevent reconciliation;
  unknown/provisional/scoped/exclusive state prevent unlink. Mismatched
  operation or owner finishes return false without exposing the stored owner
  or token.
- The new protocol accepts no credentials, headers, authorization revision,
  request snapshot, or server-body authority fields, and returns no such
  material.
- No backend, schema, capability, composer/Improve-button placement, layout,
  package manifest, lockfile, or dependency file is changed in the reviewed
  range. The only production changes are the owner authority lifecycle,
  prompt sync/unlink service, and existing Prompt Studio hook.

## Fresh evidence

- Exact focused gate, shuffled seed 12984: **5 files / 123 tests passed**.
- Extension `bun run compile`: exit 0 (`tsc --noEmit -p
  tsconfig.compile.json`).
- `git diff --check 4d5647f291..2e8f287135`: exit 0.
- Changed-file and manifest/lockfile inspection confirmed the scoped surface
  described above. The previously evidenced browser matrices were not
  repeated because this fix range does not change browser layout or the
  exercised user journeys, and code inspection exposed no browser-specific
  reason to rerun them.

## Status

`TASK-12984.2` remains **In Progress**. This review changes no production or
test source.
