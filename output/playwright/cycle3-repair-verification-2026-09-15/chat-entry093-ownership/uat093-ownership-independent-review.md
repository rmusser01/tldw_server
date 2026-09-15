# TASK13260.33 ownership correction — independent review

## Final disposition — 23:08:52.251 UTC refreeze

**CLEAR for the reviewed implementation scope. No remaining actionable finding. Native acceptance remains parent-owned and pending.**

The sole follow-up adds a pending accepted-destination reference to the Drawer. Scene initialization preserves the current edited draft only when its props identify that still-current accepted destination. Other destination changes and close→reopen use normal initialization; the reference is cleared when the apply attempt finishes. Existing explicit-action, lifetime, restore-revision and destination predicates remain intact.

Fresh independent evidence:

- Unchanged original own-detachment failure/retry probe: **1 passed/21 filtered**, `/private/tmp/uat093-draft-independent-original-green.log`. The actual edited notes survive the failed scene save and are submitted intact on retry.
- Actual Drawer and Form suites: **50 passed in2 files** (24 Drawer,26 Form), `/private/tmp/uat093-draft-independent-focused-green.log`. Includes normal external replacement/reopen scene loads and late scene/owner guards. These overlap the prior194-suite run; no summed full-suite claim.
- All12 current source/test hashes match `/private/tmp/uat093-ownership-final-manifest.json` refrozen23:08:52.251UTC. Current Drawer SHA-256: `d647b11b1dc957fd5100eeb7040c74d833e30e08cb5870de544edb8e15a27b9f`.
- Minimal four-hunk production diff checked against `/private/tmp/uat093-pre-draft-fix-drawer.reconstructed.tsx`. This is explicitly a reconstruction, whose SHA-256 `024398fcf20eafb57d5b7e9f3e4639eeeca414a43d873b5f297e35b4daa57e9c` matches the recorded prior freeze; no contemporaneous source-copy claim.
- Fresh repository-root ESLint with explicit frontend config:12files,1unchanged baseline error,132warnings,0added. `/private/tmp/uat093-draft-independent-eslint-comparison.json` and full JSON retained.
- Parent completed the combined compiler; reviewer read `/private/tmp/uat-cycle3-reviewed-combined-typecheck-comparison.json`:90baseline/90current signatures,0added/0removed at23:13:28.866UTC. This is a comparison of existing diagnostics, not a clean typecheck or an independent rerun.

Commands: from `apps/tldw-frontend`, run `./node_modules/.bin/vitest run --config /private/tmp/uat093-review-draft-after-detach.config.ts --maxWorkers=1 --no-file-parallelism -t 'review: accepted detachment'`. From `apps/packages/ui`, run `./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx --maxWorkers=1 --no-file-parallelism`.

The original three findings were already resolved and verified in the preceding review below; this re-review was limited to the sole draft-loss correction. No production/tests/runtime/browser/global-document/Git edits by reviewer. The actual combined native Drawer→Form acceptance is not inferred from separate fixture suites.

## Final re-review of 23:01:15 UTC freeze

**One remaining P2: own-detachment scene reload discards the edited draft before a failed save can be retried.**

Actual Drawer interaction: edit Scene notes; Apply synchronously detaches the old conversation; parent rerenders Drawer with the accepted null history/server destination; scene save is held and then returns false. The existing scene-load effect keyed on history/server resets `sceneDraft` and loads defaults for the new destination. The failure notice says the draft remains, but the actual Scene notes textarea is empty. A subsequent Apply would use that replacement default scene.

Location: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`, effect beginning near line187 and new `applyWithScene` destination transition near line345. Preserve the staged scene through the drawer's own accepted destination transition, while retaining actual foreign-target/unmount/close→reopen invalidation. Do not suppress unrelated destination reloads.

Evidence: `/private/tmp/uat093-review-draft-after-detach.config.ts`, `/private/tmp/uat093-review-draft-after-detach-fixture.tsx`, `/private/tmp/uat093-review-draft-after-detach.log`. **1 failure**: expected textarea `Keep my edited scene after detachment`, received empty. Actual Drawer and option/actor stores run; actor storage is a controlled boundary. No native failure is claimed.

The original findings below are otherwise resolved in the frozen correction:

- Original unchanged delayed-template replacement/unmount and neutral none→none probes: **3 passed**, `/private/tmp/uat093-ownership-independent-original-green.log`.
- The original scene probe expected identity application only after scene saving. The approved contract now applies identity/settings before saving scene against the accepted destination, so that historical ordering assertion is deliberately superseded, not rewritten as a passing original probe. The permanent actual Drawer destination/false-result/late-presentation/reopen/partial-failure controls pass; the additional parent-prop-rerender failure above exposes the remaining gap.
- Fresh full focused/adjacent verification: **194 passed in11 suites**, `/private/tmp/uat093-ownership-independent-broader.log`. This includes actual WebUI storage with six loaders, real picker transitions, deferred metadata/profile/messages, mirror controls and changed action/scene suites. Counts overlap with focused probes.
- Fresh compatible replay of the retained actual-WebUI immediate, held-profile and cross-tab bodies: **3 passed/31 filtered**, `/private/tmp/uat093-ownership-independent-original-web-green.log`. Historical production transforms are excluded as documented by the implementer's compatibility config.
- All12 source/test hashes and byte sizes in `/private/tmp/uat093-ownership-final-manifest.json` matched the 23:01:15.839 UTC freeze.
- After the remaining finding was delivered, the implementer resumed the Drawer test file; the final recheck found only that test changed from the reviewed manifest. All7 production hashes still matched. The next correction needs a new freeze/review checkpoint.
- Fresh ESLint from repository root, explicit frontend config, all12 paths: baseline1error/133warnings, current1error/132warnings,0added. `/private/tmp/uat093-ownership-independent-eslint-comparison.json`; full baseline/current JSON retained. Existing error is the unchanged Form fixture require-import rule; no clean-lint claim.
- No compiler, runtime, browser, production/test or global-document edits. Parent owns combined compiler and native verification. Native acceptance remains pending.

## Initial review findings — retained history, disposition above

1. **P2: delayed saved-template completion publishes after its owner changes.** The new `await applyAssistantSelection` in `usePromptTemplates.applyStartupTemplateBundle` was followed by unconditional global model/prompt/pinned context writes. Actual Form→template controls held selection persistence and then replaced the active chat or unmounted the Form. Both still called global `updateChatModelSettings` after release. `/private/tmp/uat093-review-late-template.config.ts` and `/private/tmp/uat093-review-form-fixture.tsx`; initial final log `/private/tmp/uat093-review-late-template-final.log`:2 failures. Implementer has added a scoped action result/guard and permanent regressions; re-review pending.

2. **P2: clearing an absent identity detaches a neutral saved conversation.** The new `applyAssistantSelection` matched only character/persona identities, so null→null was treated as replacement. Actual Form with metadata-ready neutral saved Chat plus visible Clear identity action called `setServerChatId(null)`. `/private/tmp/uat093-review-neutral-same-identity.log`:1 failure. The case is appended to the same read-only fixture/config. Tracked identity→null must still detach; neutral identity→null should retain the same target.

3. **P2: outer scene persistence can invoke an obsolete identity action.** Actual `RolePlaySetupDrawer.applyWithScene` awaits scene storage before calling Form's identity action. Replacement of its history/server props or unmount during that await still invokes the old `onApply` after release. The new Form guard captures ownership only when this delayed callback starts, so it does not guard this earlier stage. `/private/tmp/uat093-review-late-scene.config.ts`, `/private/tmp/uat093-review-drawer-fixture.tsx`, `/private/tmp/uat093-review-late-scene.log`:2 failures. This drawer behavior preexists the current diff but is exposed by the required actual-action ownership contract. The saved-template preview's `role-play-setup` branch also awaits old-target scene storage before identity application; this is a source-traced related branch, not an independently captured native failure.

## Review limits

No production/tests/global documentation/runtime/browser/Git changes by reviewer. Temporary probes copy the existing test fixtures and execute actual Form/template or Drawer handlers; external storage boundaries are controlled, and the first two use the existing mocked Drawer shell. They do not claim native mouse, resize or route acceptance. One initial temporary probe used an eager query before a lazy component mounted; corrected to await the visible button before the retained2-failure result. Vitest source-map lines refer to the injected fixture rather than current source line positions.

Canonical saved tracked identity precedence and removal of destructive rendered-mismatch clearing appear consistent with the approved design. The loader's delayed profile selection guards and separate local-history cancellation remain in place. Final exact source hashes, actual WebUI storage/coordinator suites and broader controls will be checked after the implementer freezes the complete correction.
