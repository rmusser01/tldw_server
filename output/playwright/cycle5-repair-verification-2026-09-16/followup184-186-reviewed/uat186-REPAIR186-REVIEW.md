# UAT186 / TASK13260.123 review packet

## Result

The two observed Study HTTP failure handlers—assistant response and scheduled rating—now use the existing local recoverable Flashcard mutation-error reporter. HTTP Errors with integer status400…599 remain observable warnings while the original rejection reaches existing recovery UI. Unexpected failures retain console.error and Next diagnostics.

Production scope is exactly two onError call replacements and a generalization of the reporter comment. Its classifier is unchanged. Rating aborted-signal guard, request/version construction, cache updates, query invalidation, inline error handling and all other mutation handlers are unchanged. Backend185/187 causes are separate tasks.

## Permanent behavior evidence

The existing WebUI suite executes the installed Next Pages handler unchanged; only developer-log forwarding is stubbed. Real service/bgRequest and Error construction run against an in-memory HTTP boundary.

- Assistant: actual panel/query/mutation shows HTTP500 inline, retains the question and unchanged card context/messages, and keeps Ask assistant usable. A second response succeeds with the same question/card/expected version and existing bearer authentication, appends exactly one user/assistant pair, clears the error and question, and never dispatches a Next overlay or unhandled rejection.
- Scheduled rating: actual review-run, scope lease, mutation and transport preserve rejected rating/card/context/timing, install no unacknowledged session or success invalidation, then accept the same retry and install the acknowledged session with one invalidation. Existing full ReviewTab controls also pass, including visible retry and preserved answer timing.
- Deliberately thrown assistant/rating TypeErrors still reach the caller and Next diagnostic dispatcher without a network request. Prior creation/generation controls remain intact.

## RED / GREEN and static checks

- Original causal RED:2 failed/10 passed; both failures are exactly the installed Next dispatcher receiving the HTTP500 Error. Inline/caller-state checks preceding those assertions pass.
- Final frozen tests replayed against baseline production through a private nonmutating Vite loader:4 failed/17 passed across184/186, including both expected overlay failures. Current source was not replaced on disk.
- Final focused GREEN:12/12 Next-boundary cases. Adjacent:85/85 (assistant queries2, review-run45, ReviewTab assistant11, ReviewTab recovery25, existing deck creation1, generation1). **186 total97 tests/7 files pass, no skips.** Combined184/186137 tests/11 files pass.
- Correct root-CWD ESLint on four source/test files:0 errors,14 unchanged baseline warnings,0 added/removed. Frontend-CWD lint initially ignored shared package paths; that incomplete check is retained and superseded.
- Full compiler90 baseline/current diagnostics with0 additions/removals and none in owned tests. Diff check passes.
- Required venv Bandit attempt reports four TS/TSX parse errors; it provides no meaningful JS security coverage. TypeScript parsing, lint and behavior checks cover the changed source.

Two harness corrections are retained explicitly: the new review-run import required the existing safeStorageSerde export in its storage mock; the assistant's existing transport uses bearer authentication, so its successful-wire control was corrected from an inapplicable review-scope header expectation to the actual bearer contract. Neither required a production change or altered the causal overlay assertions.

## Reproduction

From apps/tldw-frontend, use default config so the actual Pages suite is included:

```sh
bunx vitest run __tests__/flashcards-generated-save-errors.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardAssistantQueries.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardReviewRun.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.generate.test.tsx
```

The actual shared runs are retained in green-focused-final.log and green-adjacent.log. Baseline replay uses `.tmp/uat184-186-ui-design-20260917/vitest.red.config.ts`; red-final-baseline-replay.log records its four intended failures. owned-manifest.json, owned.patch, baseline/ and review-snapshot/ isolate186.

## Remaining gate

Source frozen for independent review. Parent owns native real failure/success retry for assistant and rating after backend185/187 repairs. This author performs no browser/runtime/model/configuration/task/tracker/git mutation and makes no native186 acceptance claim.
