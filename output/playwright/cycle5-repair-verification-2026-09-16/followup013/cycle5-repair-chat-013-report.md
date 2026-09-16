# UAT013 / TASK13260.3 — Home source grounding

## Result

Refrozen for independent re-review after correcting the same-value intent finding. Final165 tests/9 suites pass; the original unchanged reviewer probe passes. No native acceptance is claimed.

Native source handoff set mode/source IDs but left `fileRetrievalEnabled` false. The actual send router therefore selected ordinary Chat, using old Cedar history and only Rowan's filename/question. Enabling retrieval corrects that dispatch gate. Actual-boundary tests then proved that asynchronous session restore can overwrite accepted source selection, including before restore invocation. The first value-comparison repair missed an accepted same-value handoff. That approach has been replaced by explicit accepted-handoff intent.

## Final change and exact ownership

1. `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`: accepted valid RAG media handoffs enable the existing retrieval flag. Both validated RAG and ordinary media handoffs mark source-selection intent synchronously, even when the new values are identical. Existing owner rejection remains before this mutation.
2. `apps/packages/ui/src/store/playground-session.tsx`: one ephemeral sourceSelectionRevision and increment action in the existing store. It is excluded from persisted data/partialize; no migration or new storage framework. Session clear retains its monotonic value while the existing restoreRevision still cancels obsolete restore work.
3. `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx`: use accepted-intent revision instead of value equality when replaying persisted mode/source IDs. The mount baseline applies only to initial hydration of the matching scope/history/server target. Later requested restores capture current revision at invocation. Old transcript/metadata restore continues; existing restoreRevision/authority checks remain.
4. `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx`: actual Form submission, useChatActions, RAG pipeline, session restore, API-domain request serialization and returned source metadata. Network/model/DB operations and unrelated UI remain controlled fixtures.
5. `apps/packages/ui/src/store/__tests__/playground-session-store.test.ts`: accepted operation counting, clear lifecycle and exclusion from persisted JSON.
6. Official Backlog task13260.3 updated through CLI. No global tracker/plan edits.

## Reassessment and compatibility

The three failure angles and alternatives are documented in `/private/tmp/cycle5-repair-chat-013-intent-reassessment.md`. Existing selectedAssistantOperationRevision, session restoreRevision and Form assistantActionRevision patterns were inspected. Value tuple expansion cannot express same-value intent; cancelPendingRestore would incorrectly cancel transcript restoration. A single source-specific revision reuses the existing store and lifecycle.

The session store's current localStorage adapter and migration hydrate synchronously. A permanent real-storage test calls rehydrate, asserts completion before any await, mounts the owner and verifies Rowan42 survives old Cedar7. This does not assert support for a hypothetical future asynchronous storage adapter.

Verified boundaries include old normal/RAG history; restore before, during and after handoff; same-value RAG and ordinary full-content handoffs; Rowan→Cedar→Rowan during a pending restore; later intentional restore; different persisted conversation/account targets; foreign/stale A→B→A payload rejection without advancing intent; invalidated request output; empty/error retrieval without ordinary-generation fallback. Existing original Cedar history remains. Ordinary full-content handoff remains ordinary even when retrieval was previously enabled.

## RED evidence

All Vitest commands use installed `./node_modules/.bin/vitest` from `apps/packages/ui`.

- Original test-first dispatch RED: `cycle5-repair-chat-013-red.log`,4 expected failures/1 ordinary control pass before production edits. Fixture setup errors were resolved before accepting this RED.
- Frozen3c30685611 source replay: `cycle5-repair-chat-013-final-baseline-red.log`,7 expected failures/1 pass,16 unselected. In-memory source substitution only; no checkout/source swapping. Config `cycle5-repair-chat-013-replay.config.mts`.
- Separate restore-only RED: `cycle5-repair-chat-013-final-restore-red.log`,3 expected failures/2 passes,17 unselected. It uses the current retrieval activation with the old session hook.
- Unchanged independent same-value probe: `cycle5-repair-chat-013-same-intent-original-red.log`,1 failure/24 unselected. Probe/config are `/private/tmp/cycle5-uat013-independent-private-test.txt` and `-private.config.mts`.
- Permanent same-value RAG/ordinary before/during + ABA RED: `cycle5-repair-chat-013-intent-red.log`,5 failures/24 unselected, prior candidate unchanged.
- Store contract RED before adding the ephemeral action: `cycle5-repair-chat-013-intent-store-red.log`,2 failures/2 passes.

Prior candidate report/manifest and source copies are retained as `cycle5-repair-chat-013-before-intent-*`; they are historical failed-candidate evidence, not the final design.

## Final verification

- `cycle5-repair-chat-013-intent-final-green.log`:165 passed/9 suites, exit0, no skipped tests.
  `./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/store/__tests__/playground-session-store.test.ts src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts src/services/__tests__/media-chat-handoff.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx`
- Original unchanged reviewer probe GREEN: `cycle5-repair-chat-013-same-intent-original-green.log`,1 pass/29 intentionally unselected.
  `./node_modules/.bin/vitest run --config /private/tmp/cycle5-uat013-independent-private.config.mts src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'independent:' --maxWorkers=1`
- Earlier43/3 focused green (`-intent-green.log`) overlaps the final165; do not add those counts. Earlier154/8 and34/2 logs document the failed value-comparison candidate and are superseded by the final run.
- Repository-root ESLint with explicit `apps/tldw-frontend/eslint.config.mjs`, all5 source/test files:0 errors,91 unchanged baseline warnings,0 added/removed. Evidence `cycle5-repair-chat-013-eslint-{baseline,final,comparison}.json`; comparator `cycle5-repair-chat-013-lint-compare.mjs`, baseline3c30685611. Root pages-directory advisory is unchanged.
- Scoped `git diff --check` passes. Bandit is not applicable: TS/TSX only. Whole TypeScript comparison is parent-owned. Installed Node localStorage experimental warnings remain in Vitest output.

## Limits and freeze

No browser, runtime, API, inference, staging or commit operations. Independent re-review and native recheck remain pending. This is not a full auth UI test. The repair does not erase saved history, change RAG defaults/request routes, change retrieval preference persistence, or add persisted handoff metadata. Existing legacy/invalid payload validation remains the contract.

Exact6-path freeze (3 production,2 tests,1 task): `/private/tmp/cycle5-repair-chat-013-manifest.json`. Production subset: `cycle5-repair-chat-013-production-freeze.json`.
