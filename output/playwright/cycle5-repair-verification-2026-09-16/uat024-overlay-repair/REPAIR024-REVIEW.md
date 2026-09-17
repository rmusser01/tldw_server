# UAT024 author review packet

Task TASK13260.6 remains In Progress. Existing UAT024 reopened; this repair addresses generation error presentation only. Parent owns native reacceptance and restoration of the temporary provider/verifier configuration.

## Change and cause

GeneratePanel already catches a rejected generation and renders actionable source guidance. Its mutation also logged the HTTP Error through console.error. The installed Next Pages development handler dispatches that Error to the runtime overlay, including its raw verification JSON. This is a console bridge, not an escaped/unhandled promise rejection.

`useFlashcardQueries.ts` now routes generation errors through the existing UAT175 local expected-HTTP reporter. The helper was renamed to cover generation and saves; its integer HTTP400–599 Error classification and both create handlers retain their behavior. Expected errors still reject and remain available as a console warning with their original status/details. Unexpected TypeError still logs through console.error and reaches Next diagnostics. No verifier, service, panel, global console, source/default options, or retry behavior changed.

## Permanent regression and causal RED

Extended the existing `apps/tldw-frontend/__tests__/flashcards-generated-save-errors.test.tsx`. The mounted actual GeneratePanel, AntD controls, actual query mutation/service/request stack and installed Next Pages handler run together; fetch supplies the backend-shaped claim-verification422. Existing storage/auth/provider discovery boundaries remain the same as the UAT175 harness. The test checks:

- Source, non-default count/type/difficulty/focus/provider/model retained after rejection.
- Actionable source guidance; no raw error code/JSON or generated drafts/save controls.
- No process unhandled rejection or Next rejection dispatch; no Next runtime error dispatch.
- Original HTTP422 Error retained as the warning diagnostic.
- Subsequent actual Generate click succeeds with the same scoped request/options, creates reviewable draft, clears guidance, and makes no save request.
- Added actual generation TypeError negative control preserves rejection/Next diagnostics. Previous deck/card HTTP save and unexpected-error controls remain.

Before production edits, the new test failed precisely at Next runtime dispatcher receiving the raw HTTP422 Error; all prior assertions passed. Result `red-next.log`: 1 failed / 7 passed. Source baseline copy and optional nonmutating loader config retained. No harness errors or test relaxation were needed for this causal run.

## Verification

From `apps/tldw-frontend`, using its default config (extension config excludes the native Next test):

```sh
bunx vitest run __tests__/flashcards-generated-save-errors.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.deck-reference.test.tsx ../packages/ui/src/services/__tests__/flashcards.private-scope.test.ts
bunx vitest run ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.generate.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx ../packages/ui/src/components/Flashcards/utils/__tests__/error-taxonomy.test.ts
```

- 79/6 and 18/3 PASS: 97 tests / 9 suites, no skips.
- Scoped ESLint: 0 errors / 0 warnings. CLI emits its existing root-CWD missing Pages directory notice; this is separate from lint results.
- Full tsc: 90 baseline and 90 final diagnostics, 0 added/removed after location normalization, 0 in owned files. Both commands exit2 for the retained unrelated baseline; not a clean global compile claim.
- `git diff --check`: PASS.
- Required venv Bandit ran on both changed TS/TSX paths: 0 findings, 2 syntax/AST parse errors. Bandit provides no TypeScript security coverage. Scope review finds no new network/storage/authorization paths; existing request ownership assertions pass.

Optional reviewer baseline replay from `apps/tldw-frontend` (uses current permanent tests and private source loader, no source mutation):

```sh
bunx vitest run --config ../../.tmp/uat024-repair-20260916/vitest.baseline.config.ts __tests__/flashcards-generated-save-errors.test.tsx
```

## Bounds

Only two owned repository paths changed, recorded in `owned-manifest.json`, `owned.patch`, and byte-identical `review-snapshot/`. Private probe evidence is separate from committed/native evidence. This mounted regression observes the real installed Next overlay dispatch; it is not a headed native screenshot or an independent model-quality test. Parent's actual native422 on the deterministic provider fixture motivated it; no provider/model/browser/runtime calls were performed for this repair. Native reacceptance, subsequent real provider control and exact temporary config restore remain parent-owned. No tracker/task/git mutations by this agent.

The optional baseline-loader command was also verified after the freeze: `reviewer-baseline-loader-check.log` reproduces the same 1 expected overlay-dispatch failure / 7 controls PASS with no source mutation. Owned source/test hashes remain unchanged.
