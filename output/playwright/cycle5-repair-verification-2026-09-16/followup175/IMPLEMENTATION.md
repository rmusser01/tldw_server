# UAT175 — contain HTTP save failures in the recovery UI

TASK13260.113. Source/tests are frozen for independent review. `owned-manifest.json`, `owned-paths.json`, `owned.patch` and `review-snapshot/` identify the exact three owned files. Parent owns native acceptance and tracking.

## Confirmed cause

The original native evidence contains one deck POST at 23:10:11.883Z and its HTTP500 response at 23:10:11.908Z, with `Failed to create deck`. The inline recovery alert and original draft remain present. The captured Runtime Error overlay uses the same error's creation stack in background-proxy. Those receipts alone do not establish an unhandled promise rejection.

The actual WebUI uses Next's **Pages Router**. Its installed `next/dist/next-devtools/userspace/pages/pages-dev-overlay-setup.js` takes the second argument to `console.error` and sends an Error directly to `dispatcher.onUnhandledError`. The deck mutation logs exactly that shape even when GeneratePanel catches `mutateAsync`. The card-create mutation uses the same pattern.

`probe-causal-red.log` exercises the real scoped createDeck service, background-proxy, request-core and TanStack mutation with an actual Fetch Response fixture. It runs the installed Next Pages handler unchanged in an isolated VM, replacing only the overlay dispatcher and developer-log forwarding boundary. It records one Fetch/500, the same Error caught by the caller and dispatched to Next, **zero process unhandled rejections**, and **zero Next rejection dispatches**. The actual POST bypasses the GET coalescing gates; no discarded cleanup promise was found. The cause is a console-driven overlay, not an escaped rejection.

## Minimal change

- In `useFlashcardQueries.ts`, a private reporter used only by `useCreateDeckMutation` and `useCreateFlashcardMutation` sends Error objects with integer HTTP status 400–599 to `console.warn`. The original error and request warning/storage diagnostics remain available. Other errors still use `console.error`.
- The rejected promise, response status/details, inline catches, draft state, retry handling and owner/target checks are unchanged. No GeneratePanel, transport, backend or global-console changes.
- New WebUI regression exercises the actual panel, AntD controls, real query/mutation hooks and actual service/request chain. Only Fetch, stored test credentials, unrelated provider discovery and message presentation are fixture boundaries.
- The existing scheduler-forwarding test had a stale one-argument expectation: the service already receives explicit `undefined` requestOptions. `existing-success-baseline.log` reproduces its failure with the frozen baseline hook. The correction only acknowledges that second argument and keeps all scheduler/body assertions.

## Verification

- `final-baseline-red.log`: final new tests replayed against the original hook through a private Vite loader, **4 expected failures / 2 passing unexpected-error controls**. Each failing HTTP case dispatches an overlay Error. Working source was never reverted.
- `final-verified-controls.log`: **77 tests / 6 files PASS**, no skips. Includes four mounted HTTP cases (deck409/500, card422/500), both genuine TypeError controls, scoped service/account/server/cancellation regressions, deck-reference authority, existing recovery, scheduler forwarding and deck creation.
- The new test verifies caught mutation status, draft retention, enabled Retry, functional picker opening/closing, editing and subsequent save. It confirms one generation, no duplicate acknowledged deck after a card failure, and retained source/account request attribution. Unexpected TypeErrors still reject and reach Next's error dispatcher. No unexpected rejection event is swallowed.
- ESLint: **0 errors / 0 warnings** on all three owned files. Root-cwd Next pages-directory advisory is retained separately in command output.
- TypeScript: **90 baseline / 90 current diagnostics**, no added/removed position-normalized messages, none in owned paths. Full-project compilation is not claimed to pass. `tsc-final.log` repeats the check after the final existing-test assertion correction.
- Bandit: **0 findings, 3 TypeScript/TSX parse errors**. Bandit cannot analyze these files; no TypeScript security assurance is claimed. Manual review confirms no auth/scope/response-data widening and no new network path.
- Owned tracked diff has no whitespace errors; the new test has no trailing whitespace.

## Intermediate harness results and limits

The initial private config used an unsupported import path, then accidentally merged the base test includes; that broad invocation was stopped with Ctrl-C. Its output is not acceptance evidence. The private window-listener wrapper initially used the wrong EventTarget receiver and was corrected before causal proof.

Initial mounted tests needed an unambiguous AntD option selector and the existing card-failure summary text. After the production change, Next assertions passed; subsequent failures were test interaction assertions: jsdom left the reopened AntD dropdown in its CSS preparation phase despite `aria-expanded=true`. After three attempts the retained `picker-diagnosis.log` established this limitation. The final test checks actual expanded state and keyboard close, awaiting the queued close update. Editing intentionally clears the Retry alert in existing UI, so the subsequent save uses its regular Save button. Native visibility/occlusion remains a parent acceptance responsibility. An intermediate stale-assertion edit hit the fixture rather than the expectation; it was corrected, and the final full suite passed.

The new test observes Next's dispatch boundary rather than mounting its entire developer overlay. The native screenshot shows the original overlay; this packet makes **no post-fix native acceptance claim**. It does not claim a clean console: expected HTTP warnings and unexpected-error diagnostics remain intentional. Only the two save mutations are changed; unrelated mutation logging and unrelated native findings remain outside this task.

## Reproduce

From `apps/tldw-frontend`:

```sh
bunx vitest run __tests__/flashcards-generated-save-errors.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.deck-reference.test.tsx ../packages/ui/src/services/__tests__/flashcards.private-scope.test.ts
bunx vitest run --config ../../.tmp/uat175-repair-20260916/baseline.config.ts __tests__/flashcards-generated-save-errors.test.tsx
bunx tsc --noEmit --incremental false
```

From repo root:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx apps/tldw-frontend/__tests__/flashcards-generated-save-errors.test.tsx -f json
source .venv/bin/activate
python -m bandit apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx apps/tldw-frontend/__tests__/flashcards-generated-save-errors.test.tsx -f json
```
