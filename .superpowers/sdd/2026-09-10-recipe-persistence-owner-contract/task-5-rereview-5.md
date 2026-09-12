# Task 5 independent re-review — round 5

## Verdict

**APPROVED for Task 6 Important 1. No open Critical, Important, or Minor finding in this final Task 5 protocol-fix scope.**

Reviewed exact range `8c4256ac54288d77d2fa8fba76cf5cda3ddc06aa..18c6def94b9f8a14a4464ab7a389c6dba97ef5f9` in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/single-text-prompt-recipes`. Inspected the full nine-file diff and surrounding dispatch, response, authority, and cleanup paths against the Task 5 contract, Task 6 Important 1, the updated implementer report, and both round-5 controller rulings. Task 6 Important 2 (real-surface bridge coverage) is explicitly not closed by this approval.

Backlog `TASK-12984.2.1` was confirmed In Progress through the CLI before the temporary test/artifact edits. No review-stage status change is appropriate. Verification-before-completion was used for fresh evidence. No production code was edited; one temporary reviewer test was fully removed, and only this review artifact is committed.

## Resolved Important finding — unacknowledged background delivery

The original failure no longer depends on either page recovery channel succeeding. The authoritative background installs a provisional, all-owner quarantine for the exact local ID before mutation fetch. Receipt acknowledgement removes only that provisional state, with separate scoped/unknown evidence preserved.

### Installation and acknowledgement

- `entries/background.ts:1716` validates the dispatch marker, generates a fresh `crypto.randomUUID()` operation token, and calls `reserve(id, ownerId, operationId)` before assigning the response receipt. `tldw/request-core.ts:514` awaits that authority callback before setting dispatched metadata and calling fetch at line 524. Failure to generate/reserve stops before fetch.
- `recipe-persistence-registry.ts:27` performs check, scoped installation, and provisional installation synchronously in the one authority. There is no await between these state changes. `read` at line 15 prioritizes exact-ID provisional/unknown state over owner-scoped state, including reads by another owner or null owner. A competing reservation cannot replace that provisional tuple.
- `recipe-persistence-registry.ts:35` looks up the exact ID and requires both the stored owner and operation to match. The only successful ACK deletion is from the provisional map. Wrong ID, owner, operation, or an old operation cannot clear it; ACK does not clear unknown/scoped entries. Duplicate ACK is harmless and returns false.
- `entries/background.ts:3515` validates the message before reaching the ACK handler at line 3519. `recipe-persistence-uncertainty.ts:20` bounds exact IDs to 512 characters, requires opaque owner hashes and canonical UUID-v4 tokens, and rejects extra message/receipt fields. The facade uses the existing background authority and requires the exact successful response shape; it does not fall back to direct mutation on a lost channel.
- `api-send.ts:159` acknowledges only a received require-policy response whose receipt matches the requested exact ID, expected owner, and returned dispatched actual owner. ACK failure is caught inside this received-response branch, preserving the original response and known dispatch. A request-response timeout/loss never enters that branch and cannot manufacture a receipt ACK.

### Observed state and recovery behavior

| Event | Independently confirmed result |
| --- | --- |
| POST response lost; follow-up unknown message fails; durable error write fails; owner A changes to B without worker restart | Exact ID remains `unknown_owner`; second push returns `recipeWriteBlocked` with `not_dispatched`; total mutation count stays one. |
| ACK request is not delivered | Original returned `dispatched` metadata is retained; provisional state blocks all owners; second-owner mutation count stays one. |
| ACK arrives but its reply is lost | Receipt is proven; only provisional state is removed; original `dispatched` metadata is retained rather than relabeled unknown. |
| Ordinary received ambiguity is acknowledged, with durable storage failing | A remains scoped; B sees clear. Direct/background parity cases preserve the intended owner separation. |
| Wrong ID/owner/operation, stale operation, malformed UUID, or credential-bearing extra ACK fields | No matching provisional state is released and no other uncertainty evidence is changed. |
| Successful reconciliation or typed known-no-mutation cleanup while the ACK request was lost | Matching scoped state can clear, but provisional state remains all-owner blocking. |
| Exact `clearScoped` followed by confirmed Forget | Scoped cleanup never removes provisional/unknown. Forget removes only the exact ID's unknown/provisional state, does not clear another ID or any remaining scoped evidence, and makes no network request. |

The permanent live-background regression at `PromptRecipeBuilder.dispatch.test.tsx:423` executes real `pushToStudio → Prompt Studio → apiSend → request-core → background registry`. It asserts the POST method, one actual mutation, both channel-failure flags, one failed durable error-write attempt, and a row that is not error. It then restores messaging and changes the authoritative owner without reinitializing the background before checking the blocked second push. This is the requested current-process failure, not a restart approximation.

The request/reply-loss cases start at line 335; successful/typed cleanup and exact Forget cases start at line 368; direct/background acknowledged-ambiguity parity starts at line 470. Background tests also observe provisional state from inside both create/update fetches before ACK. Registry cases at lines 5–43 cover tuple mismatches, stale ACK, unknown/scoped retention, and another-ID Forget. The maintained successful-pull-before-Forget owner-gate case passes, consistent with the controller's scoped-cleanup clarification.

## Receipt integrity, leakage, and response-overlay audit

- The receipt contains only exact local ID, opaque owner hash, and random operation token. No credentials, connection material, or authorization revision were added. Production reference search confines the new receipt/ACK protocol to the registry, background response/message handler, uncertainty facade, and `apiSend`; it is not persisted in Dexie or added to outgoing HTTP headers/body.
- `tldw/request-core.ts:220` overlays its own dispatch metadata after the server-response wrapper. Server JSON is parsed as nested `data` at line 683, not spread into transport metadata. `entries/background.ts:1760` subsequently overlays its own receipt. `apiSend` reads that root receipt, not server data. The server cannot use a JSON `recipeDelivery` or `recipePersistence` property to substitute the authoritative fields.
- An independent temporary real-background test returned server data containing both a forged otherwise-valid receipt and false transport metadata. It checked the ordinary outgoing body byte-for-byte, absence of receipt/owner fields in outgoing headers, provisional state inside fetch, the trusted root dispatched owner, and a different generated operation token. The forged ACK failed and kept B quarantined; the real ACK succeeded, leaving A scoped and B clear. **1 passed /19 name-filter skips**, exit 0, `/tmp/task5-rereview5-overlay-probe.log`. The test was then removed exactly and the permanent protocol suites rerun successfully.
- Native UUID-v4 generation avoids deterministic counter/time reuse and makes accidental token collision negligible; uniqueness is not claimed as a mathematical guarantee. Exact ID and owner matching also prevent a same-token receipt from releasing a different tuple. The stale-operation regression verifies that an old receipt cannot release a newer reservation. Tokens are local delivery proof, not a new authentication credential or server idempotency mechanism.

No spoofing, sensitive-material leak, server-body contamination, unsafe ACK-to-unknown conversion, or unrelated cleanup was found in the reviewed protocol.

## Fresh verification

All GREEN results below were run by this reviewer at the reviewed HEAD, not inferred from the implementer report. Vitest commands ran from `apps/packages/ui` with `--reporter=dot --sequence.shuffle --sequence.seed=12984` unless noted.

1. Four protocol suites: `PromptRecipeBuilder.dispatch.test.tsx`, `background.recipe-persistence-owner.test.ts`, `recipe-persistence-registry.test.ts`, and `api-send.test.ts`: **105/105 passed, 4 files**, no skips, exit 0. `/tmp/task5-rereview5-protocol.log`. Repeated after removal of the temporary probe: **105/105 passed**, `/tmp/task5-rereview5-protocol-restored.log`.
2. Exact Task 6 plan owner gate: **442/442 passed, 16 files**, no skips, exit 0; `/tmp/task5-rereview5-owner-gate.log`. Paths:

   ```text
   src/services/__tests__/recipe-persistence-owner-contract.test.ts
   src/services/__tests__/recipe-request-snapshot.test.ts
   src/services/__tests__/recipe-persistence-registry.test.ts
   src/services/__tests__/recipe-persistence-authority.test.ts
   src/services/__tests__/recipe-persistence-owner.contract.test.ts
   src/services/__tests__/request-core.persistence-scope.test.ts
   src/services/__tests__/api-send.test.ts
   src/services/__tests__/prompt-sync.uncertainty.test.ts
   src/db/dexie/__tests__/firefox-prompt-write-order.test.ts
   src/db/dexie/__tests__/prompt-rollback.test.ts
   src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx
   src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx
   src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx
   src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx
   src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx
   src/entries/__tests__/background.recipe-persistence-owner.test.ts
   ```

3. Exact 15-suite Task 5 focused set listed in `task-5-rereview-4.md`: **1,313/1,313 passed**, no skips, exit 0; `/tmp/task5-rereview5-focused.log`. Prior no-project, durable-block, rollback, strict identity, v1/legacy, policy, and caller regressions remain passing.
4. Independent read-only historical RED/GREEN: extracted the registry source with `git show`, transpiled it in memory, and ran the same Node assertion sequence at base and HEAD: reserve `(one, A, operation)`, require B's read to be unknown, clear scoped A without releasing provisional, reject B's ACK, accept A's exact ACK, then require B clear. **Base exits 1** (`actual: clear`, `expected: unknown_owner`); **HEAD exits 0**. Logs: `/tmp/task5-rereview5-registry-base-red.log`, `/tmp/task5-rereview5-registry-head-green.log`. No production files were mutated for this check.
5. Inspected implementer RED logs for the full live-background failure, wrong-requested-owner ACK, and deliberate ACK-catch removal: `/tmp/task5-fix5-red.log`, `/tmp/task5-fix5-receipt-owner-red.log`, `/tmp/task5-fix5-ack-loss-red.log`. Their corresponding permanent GREEN cases are included in the reviewer runs above. Those historical integration RED runs are not claimed as reviewer-executed runs.
6. `bun run compile` from `apps/extension`: exit 0 (`tsc --noEmit -p tsconfig.compile.json`); `/tmp/task5-rereview5-extension.log`.
7. Frontend ESLint binary/config on all eight changed TS/TSX paths: exit 0, **0 errors /111 warnings**; `/tmp/task5-rereview5-eslint.log`. Independent lint of base-version background and api-send sources via stdin produces the same **107 +4 warnings**, respectively; `/tmp/task5-rereview5-eslint-background-base.log`, `/tmp/task5-rereview5-eslint-api-base.log`. The other six touched paths add no warnings. Existing Next pages-directory notice only.
8. `git diff --check 8c4256ac..18c6def` and working-tree `git diff --check`: exit 0. Tracked worktree was clean after removing the temporary test, before creating this artifact.

## Scope and handoff

No capability, global chat/session scope, composer layout, dependency/lockfile, backend, durable owner metadata, or server idempotency change appears in this fix. Direct transport still uses ordinary owner-scoped reservation; provisional delivery state addresses the worker-to-page boundary only.

Tests use controlled runtime/storage/fetch boundaries. No packaged-browser, live IndexedDB engine, or live remote-server validation is claimed. Full frontend typecheck/whole-file background formatting were not independently rerun; the implementer records their unchanged baseline failures. Bandit is non-applicable to this TypeScript-only fix and Markdown artifact; no Python scan is claimed.

Task 6 Important 1 is closed. **Task 6 Important 2 remains open and assigned to the subsequent coverage fix.** No other issue is reopened or speculatively added. Only this review artifact is committed; commit identity and clean status are reported separately to the controller.
