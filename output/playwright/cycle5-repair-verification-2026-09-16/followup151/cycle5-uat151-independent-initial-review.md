# Independent review — UAT151 initial frozen candidate

Candidate freeze:2026-09-16T18:05:49.805Z; baseline12b683dc49e853ee1731134c11ba36c0ecbe87c3. **Not yet clear:** one independently reproduced P2 behavior gap plus the controller's separately confirmed compiler regression require correction/refreeze. No repository/source/test/task changes made by this reviewer.

## P2 — Created-deck proof permanently overrides later authoritative removal

`GeneratePanel.tsx:83–88` and `ImageOcclusionTransferPanel.tsx:142–147` merge the local `createdDeck` into every subsequent query result whenever that ID is absent. The proof has no expiry or reconciliation boundary. After creating a deck, even a later successful current-owner list that omits/deletes it cannot clear its selector entry. Numeric save validation checks this augmented list, so it continues treating the absent deck as a valid target.

Independent actual Manager + QueryClient + real AntD selectors/service hooks reproduction:

1. Empty current-owner catalogue.
2. Generate one draft, create deck12, save to it; simulated server now lists the acknowledged deck.
3. A later successful refetch returns an empty list under the same verified owner.
4. Selector still reads `Generated Flashcards`; expected removal assertion fails.

This is an automated current-owner membership regression, not an additional native cross-account disclosure claim. The same merge exists in both panels; the private reproduction mounts the Generate path. Preserve immediate save before the initial refetch, but retire/reconcile the create acknowledgment when an appropriate subsequent authoritative list arrives. Keep this bounded; no global cache redesign.

Private probe artifacts (preserve unchanged for rereview):

- `/private/tmp/cycle5-uat151-created-deck-removal-probe.txt`
- `/private/tmp/cycle5-uat151-created-deck-removal.config.mts`
- `/private/tmp/cycle5-uat151-created-deck-removal.log` —1 expected failure /8 unselected controls,3.94s. Positive create/save/selection preceding the removed-label assertion passes.

Command from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/cycle5-uat151-created-deck-removal.config.mts src/components/Flashcards/__tests__/FlashcardsManager.deck-authority.integration.test.tsx -t 'independent fresh catalogue' --maxWorkers=1
```

The Vite transform adds only a private test to the existing actual Manager fixture; it does not replace production code or write repository tests.

## Controller-confirmed compiler blocker

`useDecksQuery`'s inferred masked-result union causes12 new diagnostics in existing ManageTab/SchedulerTab consumers (90baseline→102). Evidence: `/private/tmp/cycle5-154-combined-typecheck-comparison.json`. Correct the query owner's return typing rather than casting consumers. This was reported by the controller and inspected here; no duplicate full compiler run was performed.

## Remaining source review

No other actionable finding identified in the seven production paths at this checkpoint:

- Distinct `flashcards:decks:scoped` keys isolate the panel catalogue from the legacy `flashcards:decks` cache updater. Captured scope and QueryClient cancellation are combined; null/aborted scope disables reads and masks data.
- Actual private-state clearing relies on existing Manager `generationKey` remount and the handoff hook's synchronous invalidation signal/revision. Account/server invalidation clears accepted intent/scope and remounts ImportExport; pending list/write callbacks check the captured signal/object and mounted owner. This is appropriate for the actual Manager path; the two panels need not create a second auth resolver. Legacy omitted-scope use stays compatible and is not independently private-authority certified.
- Occlusion checks authority before/after canvas work, each upload, new deck acknowledgment, bulk completion and retained Undo's version-read/delete chain. Upload checks cancellation on both sides of file.arrayBuffer. Captured request scope is forwarded through existing transport; no unsafe replay/compensating writes are added.
- Added route policy is bounded to deck GET, asset/bulk POST and canonical-card UUID GET/DELETE, preserving unrelated negative controls, payload arrays, expected-version query and visibility options.
- Same-owner benign-event/draft retention, colliding IDs, unresolved authority, server switch, actual QueryClient history remount, delayed ABA list and legacy cache update are covered in the permanent actual Manager suite. Some lower-level continuation tests use controlled hooks; they are not native permission proof.

## Independent checks / integrity

- Independently ran real service/transport scope controls: **45 passed /2 files,exit0,936ms**, `/private/tmp/cycle5-uat151-independent-transport.log`.
- Command: local Vitest run `src/services/__tests__/flashcards.private-scope.test.ts src/services/__tests__/flashcard-assets.test.ts --maxWorkers=1`.
- All13 code/test bytes match the initial freeze: `/private/tmp/cycle5-uat151-independent-initial-hashes.json` (7production,6tests). Task hash excluded from code verdict.
- Inspected author198/14 evidence and lint comparison:0errors,18warnings,0added/8removed. Final independent full affected-suite run is deferred until the confirmed behavior/type corrections are refrozen.

Native Alice→Bob original history acceptance remains pending. Other Flashcards tabs/initial summary still use legacy catalogue queries by approved scope; this repair does not certify them. No browser, runtime, inference, source, task, staging or commit actions occurred in this review.

## Proof-expiry rereview — 18:11:16.206Z candidate

The independently reproduced created-deck finding is resolved in the revised panels: each local create acknowledgment is bound to the latest successful catalogue dataUpdatedAt at acknowledgment time, read from a current ref. A later successful list supersedes it; loading/error does not create a new successful data revision. Both permanent removal regressions were retained with valid RED2 evidence. The original private actual Manager probe/config ran unchanged and passes1/1 (9othercases unselected),2.33s, `/private/tmp/cycle5-uat151-created-deck-removal-corrected.log`. Immediate new-deck and same-owner controls remain for the final affected-suite run.

Final verdict remains held: the controller reports the explicit useQuery generic alone did not restore the90-diagnostic baseline. The author owns the pending return-type correction/refreeze. No new behavior finding from this rereview.
