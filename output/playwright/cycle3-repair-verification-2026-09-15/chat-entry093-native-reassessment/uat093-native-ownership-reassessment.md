# UAT093: native failure ownership reassessment

Status: **confirmed causal finding; implementation not changed**. Read-only production investigation on reviewed `c3368a70f4`; only temporary probe/config/report files written. No browser, runtime, backend, repository source/test or commit changes.

## Confirmed cause

`apps/packages/ui/src/components/Option/Playground/Playground.tsx:832–852` treats a difference between rendered shared assistant selection and current conversation metadata as a deliberate current-tab picker action. It then destroys the active target, messages, metadata and persisted-session reference. These values have different owners and update mechanisms: canonical chat metadata lives in Zustand, while the actual WebUI selected-assistant storage hook holds separate React state per mounted consumer.

The original permanent coordinator test replaced all those storage-hook states with one `testAssistant` field in the same Zustand store. That masks the actual boundary. Next aliases `@plasmohq/storage` and `/hook` to `apps/tldw-frontend/extension/shims/plasmo-storage{,-hook}` (`next.config.mjs:190–191,211–215`). The installed Plasmo package is not the WebUI runtime. The WebUI hook applies its own React value before awaiting persistence, then broadcasts through the real storage watch registry. Waiting for storage/selection commit-chain settlement does not establish that every consumer's React render has committed before Zustand metadata readiness.

The fresh probe uses both actual WebUI shim modules, real `useSelectedAssistant`, real server loader/effective resolver/Playground effects, six actual loader instances and existing controlled HTTP/session/view boundaries. Starting with persisted Cedar4 and successful Robot5 metadata/messages/profile responses, it records:

1. Canonical Robot metadata becomes ready.
2. The destructive effect sees `serverChatId=robot`, `serverChatCharacterId=5`, but `selectedTrackedCharacterId=4` and rendered Cedar.
3. It clears target/messages. The persisted selection ends as minimal character5 named `Assistant`; the loader cannot enrich/publish after losing its target.

This reproduces the native final shape without greeting/default-character bootstrapping. The extra native profile reads can therefore be downstream symptoms; `useCharacterGreeting` only fetches its profile when no server target is active. Successful endpoint responses do not establish successful state publication.

## Fresh causal comparisons

All commands use the existing UI Vitest runner from `apps/packages/ui`, `--maxWorkers=1`. Original configs remain retained; wrappers adapt only the stated test/runtime boundary or source comparison.

- `/private/tmp/uat093-web-storage-isolated.config.ts`, `-t 'UAT093 actual WebUI storage boundary'`: **2 failed saved-entry controls, 1 passed newer-setter negative control**. Log `/private/tmp/uat093-web-storage-isolated.log`, including exact `MISMATCH_AT_CLEAR` values. Wrapper resets new cases independently; the first exploratory config/log is retained but its later cases lacked reset and are not evidence.
- `/private/tmp/uat093-web-storage-no-mismatch.config.ts`, `-t 'independent hook state: (immediate|held-profile)'`: **2 passed** after a temporary transform removes only the destructive effect.
- `/private/tmp/uat093-web-storage-simple-loader.config.ts`, same filter: **2 passed** after additionally removing only the new pre-metadata minimal selection write and commit-chain wait (`useServerChatLoader.ts:814–829`). This does not remove the pre-existing queue, operation ownership, scope/controller guards or deferred profile lifetime.
- `/private/tmp/uat093-web-storage-cross-tab-event.config.ts`, `-t 'independent hook state: cross-tab'`: **1 failed**. After those source-removal comparisons, a real storage event changing the global preference to character7 leaves Robot's target/messages intact but resolves displayed identity7. Existing MemoryStorage is not a jsdom `Storage`, so this event uses standard nullable `storageArea`; the original incompatible event-construction log is retained and is not a product failure.
- `/private/tmp/uat093-web-storage-canonical-owner.config.ts`, `-t 'independent hook state: (immediate|held-profile|cross-tab)'`: **3 passed** with a further comparison making existing valid tracked metadata precede unrelated global tracked preference. This verifies identity/target/message ownership only. After cross-tab preference changes, the current data model falls back to name `Assistant` because its enriched profile was stored globally; it does not yet prove stable per-conversation presentation.
- Existing real AssistantSelect click-handler test independently **1 passed**: `src/components/Common/__tests__/AssistantSelect.behavior.test.tsx -t 'clears the active server chat when selecting a different tracked character'`. Log `/private/tmp/uat093-actual-picker-handler.log`. It runs actual UI selection logic with mocked option actions, so it is not a complete mounted-loader/handler integration claim.

Logs for each comparison use its config basename with `.log`.

## Minimal complete ownership direction

1. Canonical metadata owns the identity of an active saved conversation. Shared preference hydration, migration and cross-tab storage updates are not current-tab replacement intent. `useMessageOption.tsx:308–319` supplies the global draft to `effective-assistant-state.ts:105–118`, which currently grants every differing tracked draft precedence; deletion of the clear effect alone is therefore incomplete.
2. Explicit identity replacement actions own detachment before asynchronous selection persistence. `AssistantSelect.tsx:552–563` already clears a differing tracked server target before its setter; `CharacterSelect.tsx:818–825` confirms and clears before persistence. Preserve those actual behaviors and test them through the real loader.
3. Finish the bounded action inventory before deleting the fallback: `PlaygroundForm.tsx:2189–2204` currently detaches after awaiting selection; `hooks/usePromptTemplates.ts:335–353` writes selection directly with no detach. The latter also uses paired assistant/legacy setters despite both routing to the same shared assistant hook. These need an explicit action contract, not another render-derived fallback. Sibling round2_media is investigating these paths.
4. The new minimal pre-metadata storage write and global wait are unnecessary to prevent detachment once the erroneous effect is gone; the comparisons demonstrate their removal. Keep independently justified deferred-profile lifetime, exact owner/target checks and operation-revision guard until actual picker/metadata/profile races prove they can be simplified safely. Do not remove the older selection serialization merely because this new waiter is unnecessary.

Alternatives considered: a single mounted load owner would reduce duplicate requests but does not distinguish user intent from hydration; adding render-flush barriers or converting all preference storage to another shared store only synchronizes the wrong ownership rule. Neither is required to fix the demonstrated cause. Prefer explicit identity actions plus canonical saved-conversation identity, with bounded presentation ownership if stable profile labels are required.

## Prior review correction and limits

The previous independent clear at `/private/tmp/uat093-followup-independent-review.md` is superseded by this evidence. Its installed-Plasmo timing statement was correct for the package but incorrectly generalized to the Next WebUI runtime. Green controlled-storage tests did not justify the statement that prior rendered character could no longer trigger detachment. A correction notice is added to that report; the original results remain historical evidence.

This reassessment does not certify full production fixes, actual picker integration, all offline/account transitions or native acceptance. Existing HTTP/session/view mocks and test MemoryStorage remain. Parent owns native Robot/Cedar/Aster acceptance and compiler; no broad unchanged suite was rerun.

## Native corroboration and second metadata publisher

Parent subsequently supplied `/private/tmp/uat093-native-sidebar-robot-trace.json`, which I inspected. An observer attached to the already exposed option-store subscription records real keyboard selection of Robot from the Server sidebar at22:29:27.543, followed by metadata readiness at22:29:27.547 and target clearing at22:29:27.574. The retained stack identifies `setServerChatId` called from `Playground.useEffect`; final state has `serverChatId=null`, `isLoading=true`, no messages. The observer did not mutate state.

`apps/packages/ui/src/hooks/chat/useSelectServerChat.ts:127–151` publishes the selected server summary identity and metadata readiness synchronously and never updates shared assistant preference. Thus this real sidebar path bypasses the loader's new pre-metadata selection/wait entirely. Both publishers reach the same invalid render-derived ownership rule. This native path provides an independent reason to correct ownership rather than adding another loader timing barrier. Sidebar mouse geometry was separately reported by parent and is outside this reassessment.
