# Track B final focused re-review 3

## Verdict

**APPROVED.** The sole Important finding from `final-rereview-2.md` is
**ADDRESSED**. No Critical or Important regression was found in the scoped
micro-fix `f6b9f696d8..e70d532286`.

## Findings

No Critical or Important findings.

## Prior-finding disposition

### Important — ADDRESSED

`apps/packages/ui/src/services/prompt-sync.ts:581-616` now keeps the first
actionable error as the sanitized constant `Recipe uncertainty authority is
unavailable`. The durable `syncStatus: "error"` rewrite is explicitly best
effort: both a rejected update and a zero update count are caught locally and
cannot exit the committed-release catch.

The service then independently attempts, in order:

1. `finishRecipePersistenceReconciliation(local.id, ownerId,
   reconciliationOperationId, false)`, with its own non-throwing catch; and
2. `markRecipePersistenceScoped(local.id, ownerId)`, also with its own
   non-throwing catch.

Both operations reuse the exact local ID, actual opaque owner, and original
UUIDv4 reconciliation token. A failure in the durable rewrite cannot skip the
abort, and an abort failure cannot skip the scoped re-mark. Cleanup errors
cannot replace the sanitized reconciliation-authority error.

The abort-before-re-mark ordering covers both authority outcomes. If the
committed finish already ran, the abort no longer matches and the subsequent
exact owner/ID mark restores evidence. If it did not run, the matching abort
removes the lease while retaining the marker; the re-mark is idempotent, and a
late committed finish can no longer match the removed token-bound lease. A
fresh deterministic registry probe exercised both commit-first and abort-first
orders and retained `scoped` in each.

`updateExistingFromServer` returns `{safe:false}` regardless of whether the
physical durable rewrite succeeded. Both existing-row pull branches map that
result to `syncStatus:"error"`, `recipeWriteBlocked:true`, and the sanitized
error, so the caller remains fail-closed even when the deliberately rejected
Dexie update leaves the physical row `synced`.

## Regression-test review

The new test in
`apps/packages/ui/src/services/__tests__/prompt-sync.uncertainty.test.ts:699-738`
correctly makes the first two prompt updates succeed, rejects the third
(durable compensation) update, and rejects the committed finish. It asserts:

- false/error/write-blocked API output with the sanitized authority error;
- the intentionally still-synced physical row;
- scoped authority state; and
- abort reuse of the exact ID, owner, and token captured from the committed
  finish call.

The test's final scoped-state assertion can also be satisfied by the abort
retaining the original marker, so it does not independently prove that the
explicit re-mark function was invoked after an acknowledgement-lost-but-
committed outcome. This is a non-blocking coverage refinement, not a product
finding: the production call is explicit and independently non-throwing, and
the registry probe verified that re-marked state survives both possible
finish/abort orders. The existing adjacent tests continue to cover ordinary
successful release and release failure with a successful durable rewrite.

## Scope audit

- The micro-fix changes only `prompt-sync.ts`, its uncertainty test, and the
  existing Backlog record.
- Unlink, Prompt Studio import, UI/layout/Improve-button placement, capability,
  protocol/background authority, backend, schema, package manifest, lockfile,
  and dependency surfaces are unchanged.
- No identity, credential, header, authorization revision, request snapshot,
  or server body material was added to logs, errors, inputs, or outputs.

## Fresh verification

- Exact focused gate, shuffled seed 12984: **5 files / 124 tests passed**.
- Extension `bun run compile`: exit 0 (`tsc --noEmit -p
  tsconfig.compile.json`).
- Deterministic registry ordering probe: `commit-first scoped` and
  `abort-first scoped`.
- `git diff --check f6b9f696d8..e70d532286`: exit 0.
- Changed-file inspection found no manifest or lockfile change. The broad
  browser matrices were not repeated because the micro-fix has no browser UI,
  layout, or protocol change.

## Status

`TASK-12984.2` remains **In Progress**. This review changes no production or
test source.
