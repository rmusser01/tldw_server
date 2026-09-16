# Independent review — UAT151 / TASK13260.90

## Final verdict

**Clear for the bounded repair; no remaining actionable finding.** Reviewed final freeze **2026-09-16T18:13:44.885Z** against baseline12b683dc49e853ee1731134c11ba36c0ecbe87c3. Independently ran **200 tests /14 suites,all passed,zero skips,exit0,36.18s**. The original unchanged private removal probe also passes1/1 on these bytes. All13 code/test hashes match after the tests. Native acceptance remains pending.

## Findings resolved

1. **Created-deck proof outlived authoritative deletion.** Initial code merged a locally created deck into every later successful list, overriding removal and membership validation. The independent actual Manager + QueryClient + real selector probe was RED1; the author retained permanent Generate and Occlusion RED2 controls. Final code captures each catalogue's latest successful `dataUpdatedAt` in a ref when the create acknowledgment arrives. Local proof applies only to that revision. A later successful list takes precedence, including an empty one; loading/error does not invent a successful list revision. Immediate new-deck save remains covered. The original private probe/config are unchanged and now pass.
2. **Query return typing widened existing consumers.** Controller full compiler initially found12 added diagnostics. The hook now supplies `useQuery<Deck[]>`, a typed local data field and one merged return shape, without consumer casts. The controller's final full comparison is90baseline/90current,0added/0removed. It remains a baseline comparison, not an error-free compiler claim.

Initial review history: `/private/tmp/cycle5-uat151-independent-initial-review.md`.

## Scope and reviewed contracts

Seven production paths: `useFlashcardQueries.ts`, `ImportExportTab.tsx`, `ImportExport/GeneratePanel.tsx`, `ImageOcclusionTransferPanel.tsx`, `services/flashcards.ts`, `services/flashcard-assets.ts`, and `services/tldw/service-prompt-scope-error.ts` under `apps/packages/ui/src`.

- The two creation panels use opt-in captured authority for deck queries. Separate `flashcards:decks:scoped` keys prevent the existing legacy numeric-ID updater from poisoning their caches. Null/aborted authority masks data and disables reads. Query and authority cancellation are combined; late results are rejected.
- Actual private state clearing correctly depends on the existing Manager `generationKey` remount, handoff hook revision and invalidation signal. Account/server changes clear scope/accepted intent and reset the ImportExport subtree. No second auth resolver or global cache architecture was introduced. Same-owner benign config changes preserve current draft/selection; delayed ABA and colliding numeric IDs have actual Manager coverage.
- Existing deck selection/save requires a successful current-owner list and valid membership. Create acknowledgment briefly supplies owned evidence before refetch; the corrected expiry respects later authoritative absence.
- Occlusion captures the same scope through canvas work, every asset upload, deck creation, bulk save, and retained Undo. Guards run before/after awaited boundaries and before publication. Asset upload checks cancellation on both sides of file.arrayBuffer. Undo reads the current version under the captured scope before deleting under that scope. Authority changes/unmount stop subsequent work rather than replaying or compensating under replacement credentials.
- Exact added policy routes are deck GET, asset/bulk POST and canonical UUID card GET/DELETE. Query visibility, expected_version, scope headers, array bodies and unscoped default callers are preserved. Unrelated methods/routes remain denied.

## Independent validation

Final affected run, working directory `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Flashcards/__tests__/FlashcardsManager.deck-authority.integration.test.tsx src/components/Flashcards/__tests__/FlashcardsManager.private-handoff.integration.test.tsx src/components/Flashcards/tabs/__tests__/ImageOcclusionTransferPanel.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx src/services/__tests__/flashcards.private-scope.test.ts src/services/__tests__/flashcard-assets.test.ts src/components/Flashcards/hooks/__tests__/useFlashcardQueries.deck-reference.test.tsx src/components/Flashcards/hooks/__tests__/useFlashcardQueries.generate.test.tsx src/services/__tests__/flashcards.decks.test.ts src/services/__tests__/flashcards.test.ts src/services/tldw/__tests__/service-prompt-scope-error.test.ts src/services/__tests__/flashcards-private-transfer.test.ts src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx --maxWorkers=1
```

Result **200/14 PASS**,zero skips,36.18s: `/private/tmp/cycle5-uat151-independent-final-green.log`.

Unchanged private regression:

```sh
./node_modules/.bin/vitest run --config /private/tmp/cycle5-uat151-created-deck-removal.config.mts src/components/Flashcards/__tests__/FlashcardsManager.deck-authority.integration.test.tsx -t 'independent fresh catalogue' --maxWorkers=1
```

Result1PASS/9unselected,2.52s: `/private/tmp/cycle5-uat151-created-deck-removal-final.log`. Original RED retained at `/private/tmp/cycle5-uat151-created-deck-removal.log`; unchanged injected fixture is `/private/tmp/cycle5-uat151-created-deck-removal-probe.txt`. The Vite transform adds a private test; no production substitution or repository test edit.

Earlier independent45/2 transport-only pass is overlapping evidence, not additional coverage to sum: `/private/tmp/cycle5-uat151-independent-transport.log`.

## Integrity and static checks

- All13 final code/test paths match `/private/tmp/cycle5-uat151-code-freeze.json`, verified before and after tests: `/private/tmp/cycle5-uat151-independent-final-hashes.json`.
- Inspected scoped author ESLint evidence:0errors,18retained warnings,0added/8removed against baseline. Existing diagnostics were not hidden.
- Inspected controller final compiler comparison `/private/tmp/cycle5-154-verified-typecheck-comparison.json`:90existing diagnostics,0added/removed. This reviewer did not duplicate the full compiler run.
- Bandit is not applicable to the TypeScript-only scope.

## Limits

The original native Alice→Bob selector/account-history scenario must still be repeated. Tests use actual UI/query/transport owners with external services controlled; they do not certify native permissions or live cross-account writes. Other Flashcards tabs and the Manager's initial summary keep their approved legacy query behavior and are not certified by this bounded repair. Some continuation tests isolate hooks; the actual Manager test supplies the real cache/remount boundary. Existing jsdom CSS/storage notices remain in logs.

No repository, source, test, task, runtime, browser, inference, staging or commit changes were made by this reviewer. Only private reports/probes/logs were written.
