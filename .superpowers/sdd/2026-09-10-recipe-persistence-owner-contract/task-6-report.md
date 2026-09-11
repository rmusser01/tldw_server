# Task 6 implementer report — fix round 1/5

## Outcome

Task 6 review finding Important 2 is addressed on the Task 5-approved base `09b3221604`. The permanent contract now distinguishes lower-layer owner/auth/base protocol cases from actual recipe-capable UI adapters. It renders the real WebUI system modal (`PromptSelect`) and the real shared composer action (`PromptAssistComposerAction`) while retaining the production owner hook, recipe builder/compiler, sync layer, Prompt Studio client, `apiSend`, request core, registry, and extension background listener.

Only test, report, ledger, and Backlog metadata changed. There is no production, backend, capability, dependency, locale, global chat-scope, layout/placement, or Task 9 E2E diff. `single_text_recipe_v2.supported` remains false.

## Files

- Created `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx`.
- Updated `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts` to remove cosmetic WebUI/extension surface labels from lower-level protocol cases.
- Updated `apps/packages/ui/src/components/Chat/composer/__tests__/PromptAssistComposerAction.shell-wiring.test.ts` to prove the real extension sidepanel entry-to-composer chain and document the separate Quick Chat pop-out boundary.
- Updated `backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md` only through Backlog CLI. It remains **In Progress** for independent review.
- Updated this report and the ignored SDD progress ledger.

## Actual adapter bridge

The four permanent bridge cases cover:

1. **System modal, direct authority:** `PromptSelect` resolves an opaque owner in its real hook/query, passes the exact generated local recipe ID and owner through the real builder and sync stack, reserves immediately before fetch, sends one structured-v2 mutation, reconciles the exact marker, and applies compiled text through the real system-prompt callback without a second mutation.
2. **Composer, direct ambiguous outcome:** `PromptAssistComposerAction` passes its actual owner and local ID into immutable dispatch, creates one scoped ambiguity, remains Update-locked after unmount/reopen under a different owner, keeps local Apply usable, refuses to clear owner A with owner B's successful pull, and clears only when owner A reconciles.
3. **Shared extension composer, actual background authority:** the same real composer adapter reaches the installed production background listener, including Task 5's operation receipt. The exact ID and opaque owner reach the background registry, its provisional all-owner state is present at fetch, the success path acknowledges/reconciles it, one mutation occurs, and Apply remains local.
4. **Extension response loss:** a completed background mutation whose page response is lost remains quarantined across an owner change and an actual composer reopen. Update stays disabled, Apply remains local, no replay occurs, and the real Forget UI clears only exact all-owner quarantine while owner A's scoped evidence remains.

The bridge mocks only UI providers and platform/network boundaries: translated/Ant Design rendering, browser runtime/events/storage, safe/local storage, Dexie tables/helpers, online/private-mode state, deterministic Prompt Studio defaults/IDs, and `fetch`. It does not mock owner resolution, capability state, builder, compiler, sync, Prompt Studio, `apiSend`, request core, registry, or the background recipe handlers.

## Sidepanel and pop-out scope

There is one extension recipe-capable composer adapter. The shell-wiring regression proves:

`entries/sidepanel/App.tsx -> shared SidepanelApp -> SidepanelRouteShell -> /chat SidepanelChat -> SidepanelForm -> PromptAssistComposerAction`.

The product's `option-quick-chat-popout.tsx` renders `QuickChatInput` and does not mount `PromptAssistComposerAction`. It therefore cannot truthfully be called a recipe-persistence surface. The prior lower-level tests that used “sidepanel” and “pop-out” as labels around direct helper calls were renamed to owner/auth/base or background-caller cases. This fix claims real adapter coverage for the system modal, WebUI/shared composer, and extension sidepanel only; it records the actual non-recipe Quick Chat pop-out boundary instead of relabeling it.

## Strict TDD provenance

After writing the bridge, I temporarily broke both real adapter handoffs by replacing their production `persistenceScope={recipeOwner?.ownerId ?? null}` value with `null`. The bridge suite then failed **4/4** at the real UI boundary because “Save as new recipe” remained disabled. I immediately restored both production lines; the same suite passed **4/4**. The mutation was never committed and the final diff contains no production component change.

This is a genuine adapter-wiring RED: the tests would fail if either actual surface stopped handing its resolved owner to the shared builder. It is not an already-green lower-layer proof.

## Deferred audits

- **PromptSearch:** the earlier ruling remains unchanged. Its unowned schema-v2 auto-sync fails centrally before defaults, project discovery, or network mutation while preserving the local record and warning. No new real-surface test proved a separate safety or availability defect, so this task did not expand that secondary editor.
- **Capability failure representation:** the explicit unavailable capability object continues to disable Save/Update while edit/preview/Apply stays local. The new real adapters exercise the same production capability gate. No test justified normalizing the value to `undefined`.

## Verification

Fresh final code runs:

- Updated exact shuffled owner gate, seed 12984: **18 suites / 453 tests passed**, no skips.
- Exact compatibility and Track A/Track B gate: **53 suites / 1,562 tests passed**, no skips.
- Task 5 protocol gate: **4 suites / 105 tests passed**.
- Task 5 focused regression gate: **15 suites / 1,313 tests passed**.
- Directly changed bridge/contract/shell suites: **3 suites / 38 tests passed**.
- Extension `compile`: exit 0.
- Extension `locales:sync` and `locales:sync -- --check`: exit 0. The full sync exposed unrelated generated English locale drift; it was removed with no locale diff retained.
- Backend capability/authorization subset: **7 passed**, including the assertion that recipe support remains false.
- ESLint over every owner-contract production path plus Task 6 tests: exit 0, **0 errors / 30 existing `no-explicit-any` warnings** in maintained files. The new bridge and modified contract/shell suites emit no warning.
- Explicit repository-compatible Prettier check over all Task 6 fix files: exit 0.
- `git diff --check`: exit 0.

### TypeScript baseline

The package TypeScript command completed with the established 8 GB heap at exit 2 and **189 existing diagnostics**. The touched-path scan contains only the two pre-existing `recipe-request-snapshot.test.ts` fixture diagnostics and none from the new bridge, modified contract, or shell-wiring suites. Log: `/tmp/task-12984-2-1-tsc-fix1-final.log`.

The repository frontend `bun run typecheck` completed at exit 2 with **86 existing diagnostics / 117 lines**, byte-for-byte identical to `/tmp/task5-fix4-typecheck.log`; its touched-path scan is empty. Log: `/tmp/task6-fix1-frontend-typecheck.log`. Neither whole-repository baseline is represented as passing.

## Security and scope audit

- Added-line scans found no logging, unsafe HTML, `Record<string, any>`, TODO/FIXME, browser credential persistence, global-scope duplication, or layout/style change.
- Credential-looking strings in the bridge are fixed fake test values and remain behind the mocked platform-storage boundary. Production owner views and dispatched metadata remain opaque.
- All implementation changes are TypeScript tests. Bandit is non-applicable; no Python success claim is made.
- The final diff contains no capability flag, backend, production adapter, layout/DOM/class/order, dependency, locale, or Task 9 change.

## Review handoff and limitations

`TASK-12984.2.1` intentionally remains In Progress pending independent review.

These tests use real application layers with controlled browser/runtime/storage/network boundaries, not a packaged extension process, live server, or physical IndexedDB. The extension sidepanel shell proof is a source-wiring regression paired with a direct real-composer/background bridge. The separate Quick Chat pop-out has no recipe adapter to exercise. Task 9's browser journeys remain unchanged and unrun.
