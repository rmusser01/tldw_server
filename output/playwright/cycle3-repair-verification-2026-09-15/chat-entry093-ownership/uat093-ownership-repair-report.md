# TASK13260.33 / UAT093 — explicit saved Chat identity ownership

## Final status

Frozen **2026-09-15T23:08:52.251Z**, independent final review CLEAR; parent final combined compiler matches the existing90-signature baseline; remaining native acceptance is parent-owned. No runtime/browser operations or commits by this agent. Exact twelve-file manifest: `/private/tmp/uat093-ownership-final-manifest.json`; seven production files are also listed in `/private/tmp/uat093-ownership-production-freeze.json`.

**Prior full validation:194 tests /11 suites pass; final narrow correction:50 tests /2 suites plus the unchanged reviewer probe1/1 pass.** Original real-WebUI-storage behavior replay: **3/3 pass**. Scoped ESLint: **no added diagnostics**; existing one error and 132 warnings remain, and one existing warning was removed. This is not a claim of clean lint or clean TypeScript. Parent owns final combined compiler against the merged 90-diagnostic baseline. Bandit does not apply to these TypeScript-only changes.

## Cause and bounded correction

Actual WebUI storage gives every hook separate React state. The previous shared Zustand test fixture hid a render lag: saved Robot metadata could be ready while another consumer still rendered Cedar preference. Playground treated that mismatch as a deliberate picker change and destroyed the saved selection/messages. The actual sidebar also publishes metadata directly, bypassing the unsuccessful premetadata setter/wait workaround. Native proof is retained in `/private/tmp/uat093-native-sidebar-robot-trace.json`; reviewer causal probes and reassessment remain unchanged.

- Remove the destructive render-mismatch effect and the added premetadata selection write/commit-chain wait. Keep the preexisting selection queue, captured profile operation/authority guards, consumed route target, owned mirror and local-load cancellation.
- Valid saved tracked metadata owns effective identity. Raw/global/cross-tab draft changes do not replace it; fresh conversations still use the draft. Actual AssistantSelect already detaches a different saved identity synchronously and retains the same identity. Real legacy confirmation cancellation retains the saved target.
- Inline role-play and saved/startup templates perform one explicit identity operation. Different or cleared tracked identity detaches before persistence; confirmed plain saved none→none and equal tracked identities remain attached. Unknown metadata is not treated as confirmed plain identity.
- The Form captures mounted state, local explicit-action revision, restore revision and post-detach target/history. Existing selected-assistant commit guards receive this predicate. Accepted/cancelled Boolean results stop stale template/inline trailing settings. This reuses existing cancellation boundaries; it does not certify legacy local ownership or introduce network/auth prerequisites.
- Actual RolePlaySetupDrawer now runs the explicit identity/settings callback first and captures its destination before awaiting it. Only an accepted current result may save a scene, and that scene saves against the accepted destination. It checks destination/history/restore revision, local open/lifetime generation and an optional captured Form action guard before presentation/closing. It does not infer intent from global storage revision, so harmless preference hydration is not an explicit replacement.
- Saved preview uses the same accepted identity action before scene save, keeps its preview on failure, and catches errors with the existing context-backed notification hook. The drawer retains its draft on scene failure. Copy truthfully says identity/behavior were applied while the scene failed. Old scene-first ordering and rollback expectations are deliberately superseded; no atomic all-or-nothing claim or rollback into another target.

The final review found one additional scene-lifecycle interaction: own detachment rerenders the Drawer against its accepted destination, which previously reinitialized and erased the edited scene before a failed save. A local pending destination/current-action ref now suppresses only that owned scene reinitialization. It is cleared after application. External replacement and close→reopen still load their new scene normally. Permanent three-case RED is `/private/tmp/uat093-draft-scope-red.log`; final50/2 GREEN is `/private/tmp/uat093-draft-correction-green.log`; unchanged reviewer1/1 GREEN is `/private/tmp/uat093-draft-original-green.log` using `/private/tmp/uat093-review-draft-after-detach.config.ts`. Parent had independently verified the prior combined compiler still matched90 baseline signatures; parent final combined check also matches90baseline/90current signatures with0added/0removed at23:13:28.866UTC.

## Exact scope

Production:

1. `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
2. `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
3. `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
4. `apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts`
5. `apps/packages/ui/src/hooks/chat/effective-assistant-state.ts`
6. `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
7. `apps/packages/ui/src/hooks/useSelectedAssistant.ts`

Tests: coordinator integration, Form role-play integration, RolePlaySetupDrawer, usePromptTemplates role-play apply and effective-assistant-state. Paths/hashes are in the manifest. Backlog13260.33 is the only associated task record. Root-owned session/title, Settings and layout changes are excluded.

## RED → GREEN evidence

- `/private/tmp/uat093-ownership-red.log`: five actual WebUI-storage failures (immediate, held profile/messages, real sidebar and cross-tab identity) plus two resolver failures; two real picker same/different controls already passed.
- `/private/tmp/uat093-inline-real-red.log`: two actual Form ordering failures against retained exact HEAD Form, after unrelated merged optional PromptAssist/HomeMilestone fixture setup was corrected. Earlier fixture errors are not product RED evidence.
- `/private/tmp/uat093-template-boundary-final-red.log`: two real Form→template-hook failures against retained original Form+template code and valid role-play fixtures. Character/persona single-write and new destination are covered in final tests.
- `/private/tmp/uat093-continuation-red.log`: four real Form/template late-settings failures (replacement and unmount).
- `/private/tmp/uat093-neutral-scene-red.log`: confirmed plain none→none detachment and old-target preview scene save; metadata-unknown negative control passes.
- `/private/tmp/uat093-drawer-red.log`: five actual Drawer scene ownership/accepted-result/lifetime/partial-failure failures.
- `/private/tmp/uat093-drawer-generation-red.log`: close→reopen and same-null-destination newer owner-action failures.
- `/private/tmp/uat093-ownership-final.log`: prior full **194/11 pass**, including canonical/profile/messages, same-turn preference writes, principal invalidation, mirror ownership, real picker same/different and cancelled legacy confirmation, Form same identity/clear/target/history/restore/unmount, actual Drawer accepted destination/false result/held scene/reopen/owner change/failure/retry.
- `/private/tmp/uat093-ownership-eslint{,-baseline,-comparison}.json`: twelve-file current/HEAD comparison, normalized line references in warning prose; no added diagnostics.

The Form fixture renders the actual Form/template hook but mocks the drawer presentation; the Drawer suite renders the actual Drawer with actual option/actor stores and controlled callback/persistence boundaries. The Form's actual captured action guard is separately exercised at the prop seam. These do not replace a native combined Drawer→Form acceptance test.

## Original probes and compatibility

All original reviewer configs/probe bodies are retained unchanged. `/private/tmp/uat093-ownership-web-compat.config.ts` replays original immediate, held-profile and cross-tab behaviors using the matching pre-correction coordinator fixture `/private/tmp/uat093-before-ownership-coordinator.tsx`. It retains the original actual-shim alias, isolation and valid StorageEvent adapters, and excludes historical hypothetical production transforms. All production loaded is current. Result `/private/tmp/uat093-ownership-original-web-green.log`: **3/3 pass**.

The original raw-setter test that treated a global preference update as explicit user intent is not a valid assertion under the approved ownership contract. Permanent current tests deliberately assert raw preference retains the saved target, while real picker handlers explicitly detach. Original late-scene tests requiring onApply to wait until after scene persistence also describe the superseded ordering: current settings acceptance occurs before the held scene, while new controls assert no late actor presentation/close. These original artifacts remain historical RED, not silently rewritten GREEN.

## Reproduce final tests

Working directory: `apps/packages/ui`.

```sh
npx vitest run src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx src/components/Option/Playground/__tests__/usePromptTemplates.role-play-apply.test.ts src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/hooks/__tests__/useSelectedAssistant.test.tsx src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --maxWorkers=1
npx vitest run --config /private/tmp/uat093-ownership-web-compat.config.ts --maxWorkers=1 -t 'preserves saved entry.*(immediate|held-profile|cross-tab)'
```

Lint reproducer from repository root: `node /private/tmp/uat093-ownership-lint.cjs`. Original baseline Form/template reproduction configs and retained source copies are `/private/tmp/uat093-action-baseline.config.ts`, `/private/tmp/uat093-template-baseline.config.ts`, `/private/tmp/uat093-action-baseline-form.tsx`, and `/private/tmp/uat093-action-baseline-prompts.ts`.

Native acceptance remains pending for canonical URL entry, sidebar, Note backlink and normal reload in both modes. Earlier reviewed timing-only green results did not establish native success and are superseded by this causal repair.

## Final narrow correction provenance

The exact pre-draft-fix Drawer source was not captured contemporaneously. Removing only the four pending-destination additions from the final frozen source reconstructs the candidate at `/private/tmp/uat093-pre-draft-fix-drawer.reconstructed.tsx`, SHA256 `024398fcf20eafb57d5b7e9f3e4639eeeca414a43d873b5f297e35b4daa57e9c`, 28,319 bytes. Parent verified the full SHA against the contemporaneous 23:02:23 aggregate manifest: exact match. It is a reconstructed candidate, not the older HEAD scene-first source.

Final narrow validation commands (same UI working directory):

```sh
npx vitest run src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx --maxWorkers=1
npx vitest run --config /private/tmp/uat093-review-draft-after-detach.config.ts --maxWorkers=1 -t 'review: accepted detachment'
```

Parent reported native multi canonical Robot acceptance at23:04:57 on the preceding candidate: original Robot5, two original messages/BEEP BOOP, title, loaded and idle. This evidence precedes the final scene-only correction; no native scene Drawer acceptance is claimed.

## Independent final re-review

Reviewer report `/private/tmp/uat093-ownership-independent-review.md` is CLEAR. The unchanged draft-after-detach probe passes1/1, actual Drawer/Form suites pass50/2, all12 hashes match before/after, and fresh lint reports the same1baseline error/132warnings with0added. Evidence: `uat093-draft-independent-original-green.log`, `uat093-draft-independent-focused-green.log`, `uat093-draft-independent-hash-check.json`, and `uat093-draft-independent-eslint{,-comparison}.json`. Reviewer made no repository edits.

Parent final compiler evidence is `/private/tmp/uat-cycle3-reviewed-combined-typecheck.log`, `uat-cycle3-reviewed-combined-typecheck-comparison.json`, and `uat-cycle3-reviewed-runtime-source-manifest.json`. The20-file combined manifest includes adjacent frozen scopes and this final Drawer correction; no claim of clean TypeScript.
