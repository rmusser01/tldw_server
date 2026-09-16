# UAT013 / TASK13260.3 — Home source grounding

## Result and cause

Frozen for independent review. A valid Home `rag_media` handoff supplied the filename/question and media ID, but did not enable `fileRetrievalEnabled`. Actual `useChatActions` routing requires that flag plus media IDs, so it selected ordinary Chat and sent old Cedar history without Rowan evidence. This matches native captures139–146 on3c30685611; those captures did not prove a restore race.

The permanent actual-boundary tests additionally proved a restore race: an awaited persisted session can replay old mode/source IDs over the new handoff, including when the handoff finishes before restoreSession is invoked. That race has separate RED evidence.

## Change and exact ownership

- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`: enable the existing retrieval flag when accepting a valid numeric source ID in a RAG media handoff. Existing owner checks and ordinary full-content handoff behavior remain.
- `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx`: compare meaningful mode/source selection before replay. The mount baseline is limited to the existing initial restore lifecycle and matching persisted scope/history/server target. Later restores use their invocation-time selection, preserving requested replay and changes made during awaits. Existing restoreRevision/authority and transcript restoration remain intact.
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx`: extend the existing fixture with actual Form submission, useChatActions, RAG pipeline, session restore and API-domain request serialization. Network/model/DB operations and unrelated UI remain fixtures; this is not native acceptance or a full auth UI test.
- Official Backlog task13260.3 notes updated using CLI. No global tracker/plan edits.

The session store currently auto-hydrates synchronously from localStorage with a synchronous migration. A permanent real storage control calls rehydrate and checks completion/state before any await, then exercises the handoff. No guarantee is made for a hypothetical future asynchronous storage adapter.

## Behavior verified

Restored Cedar history remains while Rowan42 is the exact retrieval source and its returned text reaches the model prompt/source list. Five old-normal/old-RAG ordering combinations cover restore first, an in-flight DB restore, and handoff before restore invocation. Empty/error retrieval never falls through to ordinary generation. Ordinary full-content handoff clears source IDs even from an active RAG selection. Foreign-owner and delayed A→B→A storage payloads are rejected. A captured invalidated request cannot publish its late source answer. Later same-hook restores and different persisted conversation/account targets retain requested restoration semantics.

## Reproduction and verification

All Vitest commands use the installed `./node_modules/.bin/vitest`, from `apps/packages/ui`.

- Initial test-first RED: `cycle5-repair-chat-013-red.log`:4 expected failures/1 ordinary control pass,9 unselected, before any production edit. Intermediate fixture setup errors were resolved before accepting this RED.
- Final frozen-source replay: `cycle5-repair-chat-013-final-baseline-red.log`:7 expected failures/1 pass,16 unselected. Config `cycle5-repair-chat-013-replay.config.mts` substitutes only the two production modules from3c30685611 in memory, with current permanent test bodies; no checkout/source swapping.
  `./node_modules/.bin/vitest run --config /private/tmp/cycle5-repair-chat-013-replay.config.mts src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'retrieves Rowan before|after (empty|error) selected-source|full-content ordinary'`
- Separate restore replay: `cycle5-repair-chat-013-final-restore-red.log`:3 expected failures/2 passes,17 unselected. Current Form activation + old session hook; later test additions do not change these five bodies.
  `UAT013_REPLAY=restore ./node_modules/.bin/vitest run --config /private/tmp/cycle5-repair-chat-013-replay.config.mts src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'retrieves Rowan before'`
- Broader GREEN: `cycle5-repair-chat-013-final-green.log`:154/8 suites passed before the two final hydration/account controls.
  `./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts src/services/__tests__/media-chat-handoff.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx`
- Final affected GREEN: `cycle5-repair-chat-013-final-form-green.log`:34/2 suites pass (Form24 + session10). These overlap the broader run; final unique coverage is156 tests across8 suites, not188.
  `./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx`
- Root-scoped ESLint, explicit frontend config, all3 code/test files:0 errors,90 baseline warnings,0 added/removed. Evidence `cycle5-repair-chat-013-eslint-{baseline,final,comparison}.json`, comparison script `cycle5-repair-chat-013-lint-compare.mjs`. The root pages-directory advisory is unchanged.
- Scoped git diff --check passes. Bandit is not applicable: no Python changes. Whole TypeScript comparison is parent-owned, not rerun here. Vitest retains the installed Node localStorage experimental advisory.

## Limits

No browser, runtime, API, inference, staging or commit operations. Native recheck and independent review remain pending. This repair does not delete prior saved history, change general RAG default routing, or change session schema/retrieval preference persistence. It protects the accepted newer source using existing selection values and restore lifecycle, without adding global intent metadata. Invalid/legacy handoff validation remains the existing contract.

Exact4-path freeze (2 production,1 test,1 task) is in `cycle5-repair-chat-013-manifest.json`; production subset is `cycle5-repair-chat-013-production-freeze.json`.
