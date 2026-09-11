# Task 5 independent re-review — round 4

## Verdict

**APPROVED. R3-1 and R3-2 are addressed. No open Critical, Important, or Minor findings in the requested round-4 scope.**

Reviewed exact range `c7d14dbd896a77006b02f951449d2c417a56293a..48669b556737e6a53361aec7346df8e298815dff` in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/single-text-prompt-recipes`. Inspected the full five-file diff, updated implementer report and ledger ruling, and the current sync-result producers/consumers against the Task 5 plan/spec contract. No speculative scope expansion or production/test edits were made.

Backlog association `TASK-12984.2.1` was confirmed In Progress through the CLI before the artifact edit. No review-stage Backlog status change was appropriate; only this review artifact is committed. Verification-before-completion was used for the fresh evidence below.

## Resolved findings

### R3-1 — all v2 pending writes preserve a concurrent durable error

`markLocalPending` at `apps/packages/ui/src/services/prompt-sync.ts:124` owns the only actual pending writes. Its synchronous Dexie callback checks the current row and returns false before changing any field when a v2 row is error-locked. All three callers now return `durableRecipeBlock` when the transition is suppressed:

| Caller | Location | Confirmed behavior |
| --- | --- | --- |
| No-project auto-sync | `prompt-sync.ts:780` | Preserves the earlier atomic fix; a blocked row remains error, with validation/not-dispatched and explicit block metadata. |
| Known-project transient fallback | `prompt-sync.ts:800` | A failed second local read cannot cause another writer's durable error to become pending. Direct/background regression cases retain error and refuse a second mutation after fresh authority initialization. |
| Local linking after a server GET | `prompt-sync.ts:995` | Uses strictly parsed local v2 identity; a concurrent error prevents both the pending status and new reference fields from being written. The entire locked row remains equal to its prior value. |

A fresh search of actual `syncStatus: "pending"` writes finds no fourth site in `prompt-sync.ts`; remaining occurrences construct results only. The existing central ambiguous-result path continues to attempt durable error and cannot enter auto-sync's proven-pre-dispatch fallback.

Dexie semantics remain correct: `Table.update` delegates to exact-key `Collection.modify` in a locked readwrite transaction; callback false suppresses the mutation, but the result counts matched keys. Production uses the callback's `durableError` flag, not the count. Both changed test fixtures now correctly return 1 for an existing matched row when the callback returns false, resolving the prior fixture-fidelity limitation. No live IndexedDB-engine test is claimed.

### R3-2 — authority blocks are explicit and cannot become builder rollback

`readRecipeWriteBlock` at `prompt-sync.ts:96` returns `recipeWriteBlocked: true` when the expected owner's authority is scoped, unknown, or unavailable. It reads current local status after the authority await for display only; failure of that local read does not remove the block. It neither marks a new uncertainty entry nor invents a dispatched mutation or actual owner.

- Manual persistence (`prompt-sync.ts:550`) and auto-sync (`prompt-sync.ts:766`) return the helper's complete blocked result directly.
- The late not-dispatched transport branch (`prompt-sync.ts:581`) uses the same helper and passes the exact original transport ownership object through. A reservation rejection remains not-dispatched rather than being relabeled as an ambiguous dispatch.
- The shared builder (`PromptRecipeBuilder.tsx:233`) checks `recipeWriteBlocked` independently of display/durable status, after exact-ID and transport classification. It throws the existing retention sentinel before either Save deletion or Update snapshot restoration can run.
- Eight delayed-background Save/Update cases verify both scoped and unavailable authority, with durable storage both succeeding and failing. Even when the row still says local/synced because durable marking failed, the exact edited recovery row remains present, the real authority remains scoped, the unverified-outcome notice appears, and there is only one mutation.
- Two delayed-background manual cases verify a fresh error status plus `recipeWriteBlocked`, validation, and `not_dispatched`/null actual owner.
- Four direct/background late-reservation cases verify zero mutation fetches, the explicit block, and unchanged scoped/unknown authority.

## SyncResult and caller audit

- `durableRecipeBlock` creates an explicit shared-state block with accurate local ID and no-dispatch evidence. `readRecipeWriteBlock` preserves that flag while using current display status and, where supplied, exact transport metadata. The new field is result-only; its production references are the result type/helper and builder guard, not storage or HTTP serialization.
- `pushToStudio`, auto-sync, keep-local conflict resolution, and keep-both persistence return complete blocked results without reconstructing/dropping the flag. The three suppressed pending transitions return the same durable-block result. Existing parser/missing-owner validation and successful reconciliation retain their separate established contracts.
- Existing dispatched ambiguity remains protected by its original dispatch classification plus error result. Typed known-no-mutation rejections and valid reconciliation retain their existing cleanup/rollback behavior; the new flag does not pretend those outcomes dispatched differently.
- The recipe builder is the consumer that can delete a newly saved row or restore an update snapshot. It consumes the explicit block before those actions.
- Manual library mutation, conflict, batch, and bulk consumers treat blocked results as unsuccessful and do not perform local rollback, clear uncertainty, or write pending status. A later explicit retry still goes through central preflight/atomic transition protection.
- `usePromptSync.syncPromptAfterLocalSave` projects a reduced result and does not forward the new flag. Its current library-editor consumer nevertheless receives unsuccessful validation rather than transient/pending, and has no destructive rollback branch. This projection does not erase the reviewed safety invariant; no claim is made that every UI receives the new metadata.
- `PromptSearch` remains intentionally ownerless/fail-closed for v2 and warning-only, as ruled for Task 5. Its broader caller usability audit remains Task 6. Pull/reconciliation and explicit unlink behavior were not expanded in this fix.

No remaining unsafe block-to-rollback conversion or error-to-pending transition was found in the reviewed paths.

## Preserved behavior

- The four isolated direct/background Save/Update no-project cases still retain the edited Markdown definition as pending, show the local-pending notice, keep uncertainty clear, and make zero fetches.
- Valid v1/legacy calls remain owner-optional; their project discovery and ordinary pending update behavior use the unchanged non-recipe branch. The focused set includes v1 create/update, legacy partial update, and the idempotency-key call shape.
- Strict outer/inner local identity validation, exact-ID uncertainty classification, real dispatch reservation, matching cleanup, existing safe rollback, and prior restart protections remain passing.
- No capability flag, composer rendering/layout, locale, dependency/lockfile, backend, durable owner storage, or global chat/session scope change appears in this round.

## Fresh verification

Commands below were run by this reviewer, not inferred from the implementer report.

From `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx src/services/__tests__/prompt-sync.uncertainty.test.ts -t 'delayed .*authority|known-project transient fallback|atomically rejects .* uncertainty introduced|linking cannot' --reporter=dot
```

**17 passed / 82 name-filter skips across 2 files**, exit 0. This is 13 new cases plus 4 strengthened late-reservation cases. Log: `/tmp/task5-rereview4-seventeen.log`. No test was permanently skipped or disabled.

The exact full 15-suite set was run with `--reporter=dot --sequence.shuffle --sequence.seed=12984`:

```text
src/services/__tests__/prompt-sync.structured-prompts.test.ts
src/services/__tests__/prompt-sync.auto-sync.test.ts
src/services/__tests__/prompt-sync.uncertainty.test.ts
src/db/dexie/__tests__/prompt-rollback.test.ts
src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx
src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx
src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx
src/services/__tests__/prompt-studio.recipe-policy.test.ts
src/components/Option/Prompt/__tests__/prompt-sync.owner-callers.test.tsx
src/components/Option/PromptStudio/__tests__/PromptStudioPlaygroundPage.recipe-policy.test.tsx
src/services/__tests__/request-core.persistence-scope.test.ts
src/services/__tests__/recipe-persistence-authority.test.ts
src/entries/__tests__/background.recipe-persistence-owner.test.ts
src/services/__tests__/recipe-persistence-registry.test.ts
src/services/__tests__/api-send.test.ts
```

**15 files / 1,295 tests passed**, no skips, exit 0. Log: `/tmp/task5-rereview4-focused.log`.

Additional checks:

- `bun run compile` from `apps/extension` (`tsc --noEmit -p tsconfig.compile.json`): exit 0; `/tmp/task5-rereview4-extension.log`.
- Repository frontend ESLint binary/config on all four changed TS/TSX files: exit 0, no findings; `/tmp/task5-rereview4-eslint.log`. Existing Next pages-directory configuration notice only.
- `git diff --check c7d14dbd..48669b5`: exit 0.
- `git status --short` remained empty before creating this artifact; this round required no temporary probes or production/test edits.
- Full frontend typecheck was not independently rerun. The implementer reports the unchanged 86-diagnostic baseline; that is not presented as reviewer-run evidence. Previously recorded optional drawer failures remain outside the requested focused run.
- Network tests end at mocked fetch and use controlled storage boundaries; no live remote server or IndexedDB-engine claim is made. Bandit is non-applicable to this TypeScript-only fix and Markdown artifact; no Python security scan is claimed.

## Handoff

R3-1 and R3-2 are closed, the earlier addressed findings remain closed, and this review adds no new finding. Task 6/deferred capability and caller audit work remains outside this approval. Only this review artifact is committed; the verified commit and worktree status are reported separately to the controller.
