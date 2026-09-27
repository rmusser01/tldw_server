# Task 6 independent review — complete recipe persistence owner recovery

## Verdict

**NEEDS FIXES. Two Important findings remain; the recovery range is not approved.**

Reviewed exact range `36610500c9bea57c51dedf99c5b9372e10c32e35..f3034b59ea9f4a06a262ed9f31fb8dbe6d501150`, with separate scrutiny of Task 6 range `9a85f40ae4..f3034b59ea`. I read the repository instructions, recovery plan, specification section 10.6, progress ledger, Task 1–6 reports, and every prior review/rereview artifact. I traced the WebUI system/composer and extension sidepanel/pop-out production adapters through owner resolution, capability refresh, shared builder, synchronization, request-core/background reservation, immutable fetch, result classification, reopen, reconciliation, deletion, and Forget.

The production owner derivation, opaque credential handling, immutable request snapshot, same-subject refresh rule, v1 compatibility, exact-ID cleanup, action-specific capability gating, unchanged global scope, and unchanged composer-shell layout are otherwise consistent with the contract. The findings below are not pre-existing repository baselines.

## Findings

### Important 1 — An extension response loss can evade the required current-process unknown-owner quarantine and dispatch the exact ID twice

Locations: `apps/packages/ui/src/services/api-send.ts:165`, `apps/packages/ui/src/services/prompt-sync.ts:639`, `apps/packages/ui/src/services/recipe-persistence-registry.ts:8`, `apps/packages/ui/src/entries/background.ts:1714`.

When the background completes a version-2 POST but the response channel fails, `apiSend` correctly returns an unknown dispatch. `persistServerPrompt` then attempts `markRecipePersistenceUnknown(localId)`, but deliberately swallows a failure of that background message, and `failure()` separately swallows a failure of the durable `syncStatus: "error"` write. The background still contains only the reservation made before fetch: a scoped marker for owner A. `RecipePersistenceRegistry.read(id, ownerB)` intentionally ignores owner A's scoped marker, so after the channel/storage recover and the authoritative owner changes to B, the same still-running background process reports clear and accepts another exact-ID mutation.

This is not the specification's acknowledged restart recovery boundary: no application/background restart occurred. Section 10.6 requires an unknown response to create an exact-ID, all-owner quarantine for the current process, with a restart becoming a recovery boundary only if durable marking also failed. Here the current process remains alive but loses the all-owner lock.

A temporary reviewer-only regression used the real extension background listener and real `pushToStudio -> apiSend -> request-core -> registry` path. The first POST completed in the background; its response was made to throw, the follow-up `mark-unknown` message was made unavailable, and the exact durable error update was made to fail. After restoring both boundaries and changing from owner A to owner B, a second `pushToStudio` issued a second POST. The required assertion failed with `expected 2 to be 1`. The probe was fully reverted and `git status --short` returned clean before this artifact.

Fix the protocol so background authority cannot retain only cross-owner-invisible scoped evidence after an unacknowledged response. For example, keep an exact-ID provisional all-owner quarantine until the page positively acknowledges receipt, or establish equivalent pre-dispatch durable evidence and clear it only after a conclusive response. Preserve the contract that an ordinary owner-scoped ambiguity is not exposed to another owner. Add a permanent direct/background regression covering response loss plus unknown-marker failure plus durable-marker failure, recovery without process restart, owner change, and a one-mutation assertion.

### Important 2 — The new “four-surface” matrix labels lower-level calls as surfaces without executing the real adapters

Locations: `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts:308`, `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts:449`, `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts:799`.

The rows named `WebUI system`, `WebUI composer`, `extension sidepanel`, and `extension pop-out` do not render or call those surface adapters:

- The owner-to-dispatch table constructs an owner with the pure snapshot resolver and calls `tldwRequest` with a new test-local registry. The extension-labelled rows never use `apiSend`, runtime messaging, the background listener, or a sidepanel/pop-out component.
- The reopen table renders only `useRecipePersistenceOwner`; for extension-labelled rows it mocks `sendMessage` to return a precomputed view rather than installing the real background.
- `expectLocalApply(surface)` is a reducer/compiler helper. The four names affect only generated text and target choice; it never renders the system modal, composer action, sidepanel, or pop-out.
- The direct “full-stack” table calls `resolveRecipePersistenceOwnerView` and `pushToStudio` directly. The two extension authority tests use the real background, but `sidepanel` and `popout` are local variable/comment labels around repeated `pushToStudio` calls, not real surface shells.

The separate real adapter suites do verify open/reopen and capability behavior, but both replace `autoSyncPrompt` (`PromptSelect.system-prompt-modal.test.tsx:92` and `PromptAssistComposerAction.test.tsx:80`). The builder dispatch suite starts at a directly supplied `persistenceScope`. Consequently there is still no test that proves the owner emitted by either actual adapter is the immutable expected owner that reaches request dispatch and returns through reconciliation. A wiring regression that drops or substitutes the owner between an actual surface and the builder would pass the claimed matrix.

Task 6 explicitly requires a cross-layer matrix for the four real surfaces and instructs the reviewer to trace one owner from each real surface to dispatch and back. Add bridging tests that render the real `PromptSelect` and composer adapter while retaining real builder/sync/request-core layers. Exercise the composer adapter under both extension sidepanel and pop-out authority/shell contexts, or accurately prove and document the single shared adapter boundary while testing both real extension entry paths. Mock only platform storage/runtime/network boundaries, and assert the actual adapter-resolved owner, exact local ID, mutation count, reopen lock, reconciliation, and local Apply behavior. Do not satisfy this by renaming the current cosmetic rows.

## Contract audit beyond the findings

- Owner resolution is centralized and the public view contains only `ownerId` and `authorizationRevision`. Manual API keys, runtime keys, bearer/refresh tokens, origins, headers, and organization/principal source material are absent from the view and synchronized rows. No new credential logging or persistence was found.
- Request-core resolves one immutable request snapshot, checks expected versus actual owner before reservation/fetch, preserves the same base and owner across a refresh retry, and overlays dispatch metadata outside server response data. Direct and background paths both reserve immediately before fetch.
- All registry `reserve`, `markScoped`, `markUnknown`, `read`, `clearScoped`, and `forgetUnknown` production sites were inspected. Successful create/update/pull and permanent deletion clear only a trustworthy returned/current owner and exact local ID. Forget removes only exact unknown quarantine. The exception is finding 1's inability to install that quarantine after a failed extension response channel.
- Every v2 pending write in `prompt-sync.ts` uses the atomic lock-aware `markLocalPending`. `recipeWriteBlocked` reaches the destructive recipe-builder rollback guard. Library, batch, bulk, conflict, and reduced `usePromptEditor` consumers treat blocked results as failures and perform no destructive rollback or authority clear; dropping the result-only flag in the reduced projection does not make its validation/error outcome retryable.
- Version-1 create/update/pull behavior remains owner-optional. Strict version-2 identity parsing and exact local-ID metadata remain enforced before project discovery or mutation.
- The owner/revision capability query key and forced reopen revalidation remain current, action-specific authorization remains separate, and an unavailable capability object keeps persistence disabled while local editing/preview/Apply remains available. The deferred unavailable-representation ruling is technically safe; the problem is only the new suite's false claim that its reducer helper proves all four surfaces.
- `PromptSearch` still creates/edits legacy-shaped local records. If a version-2 row reaches its ownerless auto-sync, central identity/owner validation stops before defaults/project discovery/network and the caller reports a warning. The Task 6 ruling not to expand that secondary editor is justified for this recovery scope.
- No recovery-range diff exists in `server-capabilities.ts`, `chat-surface-scope.ts`, the four shell/layout forwarding files, or Task 9 E2E paths. The capability support flag remains false. No dependency/lockfile or Python production file changed.
- Task 6's `PromptBody.search-pagination` mock update accurately follows the approved owner argument. Its `structured-prompt-utils` fixture path is tied to the documented package working directory, and the exact package compatibility gate passed; I do not treat either test-only compatibility edit as a finding.

## Fresh reviewer verification

All commands below were run in the recovery worktree at `f3034b59ea` before creating this artifact.

- Exact Task 6 shuffled owner gate (`--sequence.shuffle --sequence.seed=12984`): **16 suites / 424 tests passed**, no skips, exit 0.
- Exact focused Task 5 concurrency filter over builder dispatch and uncertainty suites: **17 passed / 82 name-filter skips**, exit 0.
- Exact compatibility/Track A/Track B gate: **53 suites / 1,555 tests passed**, no skips, exit 0.
- Extension `bun run compile`: exit 0.
- Extension `locales:sync -- --check`: exit 0 with no diff.
- Backend capability/authorization subset: **7 passed**, exit 0, including `single_text_recipe_v2.supported === false`, limits, admin write rules, ordinary-role catalog scope, and authentication enforcement. Existing configuration/test-cleanup warnings are baseline.
- ESLint over owner-contract production files and Task 6 changed tests: exit 0, **0 errors / 30 existing `no-explicit-any` warnings** in `PromptBody.search-pagination.test.tsx`, `api-send.ts`, and `request-core.ts`. The new contract suite emitted no warning.
- `git diff --check 36610500c9..f3034b59ea`: exit 0.
- The cached repository frontend typecheck result is byte-identical to the Task 5 baseline (`cmp -s` exit 0): 86 unrelated diagnostics / 117 lines. Owner/touched-path scan is empty. The package diagnostic log likewise contains no new contract-suite diagnostic.
- Reviewer probe: **1 failed as intended** (`expected 2 to be 1`) with 27 name-filter skips, proving finding 1. The temporary test was reverted before review-artifact creation.

Passing broad gates do not overrule the reproduced duplicate mutation or the missing real-adapter proof. Bandit is not applicable to this TypeScript/Markdown review range, and no live remote server, packaged-browser extension, or physical IndexedDB claim is made.

## Handoff

Return finding 1 to the Task 5 owner-contract implementation with strict RED/GREEN. Return finding 2 to Task 6 coverage. Re-run the exact 16-suite shuffled gate, focused concurrency cases, compatibility gate, extension compile, static checks, and the new real-surface/double-failure regressions before another independent review. Keep `TASK-12984.2.1` In Progress; this review does not authorize completion.
