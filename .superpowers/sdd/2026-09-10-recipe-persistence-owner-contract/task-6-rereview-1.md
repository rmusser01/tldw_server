# Task 6 independent re-review — fix round 1

## Verdict

**APPROVED. Task 6 Important 2 is closed. No open Critical, Important, or Minor finding remains in `09b3221604..a794a77e9a`.**

I retained the complete recovery-range context and re-reviewed this test-only fix against section 10.6, the recovery plan, Task 6 initial review, Task 5 round-5 approval, updated report/ledger, all changed files, and the surrounding real adapter and route sources. `TASK-12984.2.1` remains In Progress; this review does not mark it Done.

## Important 2 closure

The new `recipe-persistence-owner.surface-bridge.test.tsx` renders the production `PromptSelect` system modal and production `PromptAssistComposerAction`. Both lazy-load the production `PromptRecipeBuilder`. The suite does **not** mock `useRecipePersistenceOwner`, `resolveRecipePersistenceOwnerView`, capability query functions, the builder/compiler, `autoSyncPrompt`, `pushToStudio`, Prompt Studio clients, `apiSend`, request-core, the recipe registry/facade, or background recipe handlers.

The remaining substitutions are appropriate UI/platform boundaries: Ant Design and translation rendering, online/private-mode state, browser runtime/events/storage, safe/local storage, Dexie tables/helpers, deterministic test IDs/default project, and `fetch`. The Dexie substitutes preserve exact-ID update/add/read behavior needed by the sync contract. The network substitute distinguishes capability GET, structured-v2 create POST, and exact prompt pull; it observes the actual production registry reservation at fetch time.

The four bridge scenarios now establish the required behavior through real handoffs:

1. `PromptSelect` resolves the production owner and authorization revision, passes its owner to the real builder, creates the deterministic exact local ID, reserves that ID/owner immediately before the one direct POST, reconciles it to synced/clear, and applies the compiled value only to the real system-prompt callback.
2. The direct composer passes its production-hook owner into real sync/transport, retains one owner-A scoped ambiguity, reopens under owner B with Update locked, keeps local Apply working, prevents owner B's pull from clearing A, and clears only after owner A's exact pull reconciliation.
3. The composer under an extension runtime resolves through the installed production background listener, carries exact ID/owner plus the Task 5 receipt into the background registry, observes provisional all-owner state during the one POST, acknowledges and reconciles successfully, and keeps Apply local.
4. A lost extension response retains exact-ID all-owner quarantine across an owner change and actual composer reopen, permits local Apply, issues no replay, and exposes the real confirmed Forget action. Forget clears exact provisional/unknown state while owner A's scoped evidence and the one-mutation bound remain.

The capability query-key assertions join the production hook result to the exact owner/revision, while the actual `RecipePersistenceRegistry.reserve` assertions join the builder/sync handoff to dispatch. A substituted or missing handoff cannot pass merely through a cosmetic label.

## Independent mutation verification

I temporarily changed only the two production builder props from `persistenceScope={recipeOwner?.ownerId ?? null}` to `persistenceScope={null}`, ran the four-case bridge suite, and observed a genuine **4/4 RED**: every case failed because `Save as new recipe` remained disabled at the real adapter/builder boundary. I restored both lines exactly with `apply_patch`; a source diff over both production files was empty. The bridge plus shell suite then passed **11/11**. No temporary production or test mutation remains in the worktree.

This RED would catch either actual adapter dropping its owner. Supplying a different syntactically valid owner would also fail before mutation because request-core compares the adapter value with the immutable actual snapshot; the successful mutation/reconciliation assertions would fail.

## Surface and route truth audit

The revised lower-level contract suite removed `WebUI system`, `WebUI composer`, `extension sidepanel`, and `extension pop-out` labels. Its tables are now accurately named owner/auth/base/background protocol cases, and its reducer helper is described as compiled local Apply rather than surface execution.

The extension sidepanel source chain is real:

`entries/sidepanel/App.tsx -> shared SidepanelApp -> SidepanelRouteShell -> sidepanel /chat -> SidepanelChat -> SidepanelForm -> PromptAssistComposerAction`.

The formally named `option-quick-chat-popout.tsx` is also represented truthfully: it renders `QuickChatInput` and has no `PromptAssistComposerAction` or recipe persistence adapter.

There is additionally a sidepanel **Open full chat in WebUI** path to `/options.html#/chat`. It resolves through `option-chat-route-registry.tsx -> OptionChat -> Playground -> PlaygroundForm -> ComposerToolbar -> PromptAssistComposerAction`. This is not another recipe adapter: it is the same WebUI/shared composer hosted in an extension options tab. The bridge exercises that component once with direct authority and twice with a real extension background runtime, while the shell test establishes its Playground placement. Therefore the full `/chat` tab is not behaviorally omitted, and the report's narrower statement about the separate Quick Chat pop-out remains accurate. A future recipe integration into `QuickChatInput` would require a new adapter and bridge before that formal pop-out could be claimed.

## Task 5 safety and preserved scope

Task 5 Important 1 remains closed. The fresh protocol gate covers pre-fetch provisional quarantine, exact receipt acknowledgement, response/ACK loss, cross-owner blocking, exact reconciliation, and Forget boundaries. The Task 6 bridge independently exercises the installed background success and response-loss paths without bypassing the protocol.

No production, backend, capability flag, dependency/lockfile, locale, global chat/session scope, composer layout/placement, or Task 9 E2E file changed in this round. `single_text_recipe_v2.supported` remains false. The existing PromptSearch and unavailable-capability rulings remain justified and were not relabeled as surface proof.

## Fresh reviewer evidence

All positive results were run after restoring the mutation-test lines:

- Reversible owner-handoff mutation: **4/4 failed as required**; restored bridge/shell run: **2 files / 11 tests passed**.
- Updated exact shuffled owner gate, seed 12984: **18 files / 453 tests passed**, no skips, exit 0.
- Task 5 protocol gate: **4 files / 105 tests passed**, no skips, exit 0.
- Sidepanel full-screen route, formal Quick Chat pop-out, deferred option route, and shell-wiring check: **4 files / 17 tests passed**, exit 0.
- Extension `bun run compile`: exit 0.
- Repository frontend ESLint/config over all three changed test files: exit 0; only the existing pages-directory configuration notice was printed.
- `git diff --check 09b3221604..a794a77e9a`: exit 0.
- Static scans show that the lower-level contract has no old surface/pop-out labels and that every TypeScript source change in this round is a test file.
- `git status --short --untracked-files=all` was empty after reversible mutation restoration and before creating this artifact.

The test environment uses controlled runtime/storage/network boundaries rather than a packaged extension, physical IndexedDB, or live server. That limitation does not weaken the reviewed adapter handoff because the production hook, builder, synchronization, authority, request, and background protocol remain intact.

## Handoff

Task 6 fix round 1 honestly closes Important 2. The controller may proceed with final Task 6/whole-recovery completion checks; only the controller should finalize `TASK-12984.2.1` after accepting this approval.
