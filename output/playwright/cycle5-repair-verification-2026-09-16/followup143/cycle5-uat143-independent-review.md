# UAT143 / TASK13260.82 — independent test-repair review

## Verdict

**Clear. No actionable finding in the six scoped test corrections.** Existing behavior assertions are preserved, no tests are skipped, and the replacement llama.cpp guard checks an actual production preview payload. This is test-fixture/guard maintenance; it does not establish a new product repair or native pass.

Reviewed against base 3c30685611. Author freeze: 2026-09-16T15:55:09.441Z, `/private/tmp/cycle5-uat143-manifest.json`. All seven frozen paths match (six tests and official task82); `/private/tmp/cycle5-uat143-independent-hashes.json`.

## Scope assessment

- **Document processing:** The dependency fixture now supplies the required beginPromptAssistReset and markPromptAssistAttemptSaved callbacks used by the existing submit hook. All six original cases and their assertions are unchanged: explicit retrieval overrides, visible reserved turns, blocked preparation retaining attachments, duplicate-submit suppression, and attachment-change cancellation remain exercised. These additions restore the dependency contract; they do not bypass document preparation/send behavior.
- **Follow-up research / image refinement / voice:** Added mocks isolate the unrelated PromptAssistComposerAction and, where needed, Home milestone scope. These boundaries already exist in pinned-fallback/role-play/openui Form fixtures. The original query mocks return generic arrays and do not represent Prompt Assist's capability contract. The target Form/submit/research/refinement/preview/dictation components still execute. All five research, fifteen image-refine/dictation, and two voice/error-diagnostics cases and assertions remain intact. Home/Prompt Assist behavior is deliberately not certified by these suites.
- **Mobile composer guard:** The test identifies the mobile inline-send element and still requires every original token: col-span-2, flex, shrink-0, justify-end, self-end. It also retains the two-column input grid and forbidden three-column assertions. Class order and unrelated added tokens no longer break the guard. An absent target or required token still fails; this remains a static source guard, not geometry proof.
- **llama.cpp raw preview:** Replaces the stale Form-source-location assertion with the real extracted usePlaygroundRawPreview hook. It invokes refreshRawRequestSnapshot and checks the actual snapshot endpoint and exact thinking_budget_tokens, grammar_mode, grammar_id, grammar_inline, grammar_override values for library, inline, and none. Only provider resolution is controlled; the builder and snapshot capture remain real. Form still passes currentChatModelSettings into this hook. Removing/wrongly mapping any field fails the observable payload assertion. Three cases replace one, explaining the total increase from33 to35.

No production file is included in this unit's manifest or scoped correction diff. Unrelated UAT013/137/138/139–142 production and test work was excluded; this is not a claim that the entire shared working tree has no production changes.

## Independent verification

Installed local Vitest, six suites only: **35 passed / 6 suites**, exit0, no skipped tests or unhandled errors. `/private/tmp/cycle5-uat143-independent-tests.log`.

From `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.composer-options.guard.test.ts src/components/Option/Playground/__tests__/PlaygroundForm.llamacpp-controls.guard.test.ts src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.follow-up-research.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.image-refine.integration.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.voice-visibility.integration.test.tsx --maxWorkers=1 --no-file-parallelism
```

Scoped git diff --check is clean. No skip/todo/only markers were introduced. Reviewed author lint comparison: 0 errors / 73 unchanged baseline warnings, 0 added/removed signatures. No separate lint/full compiler run in this bounded read-only review.

Inspected retained author baseline evidence and read-only loader: unchanged tests on3c30685611 reproduce27failed/6passed plus3unhandled; corrected tests on that production baseline pass35/6. I did not repeat those baseline runs. They overlap the same test cases and must not be added together as distinct coverage. The loader substitutes baseline production/source-read bytes in memory and leaves the repository unchanged.

## Limits

No repository/source/test/task edits, browser, runtime, API/inference, dependency install, staging or commit. Private report/hash files only. Existing Node localStorage advisory remains. Raw-preview coverage is not live provider acceptance; fixture isolation is not Prompt Assist/Home integration coverage. Root owns the integrated150-suite run, compiler, retention and commit.
