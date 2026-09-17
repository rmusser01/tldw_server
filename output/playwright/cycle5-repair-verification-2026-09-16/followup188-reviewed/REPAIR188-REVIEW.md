# UAT188 / TASK13260.126 — Study streak grammar

Source and tests are frozen for independent review. A one-day streak now renders **1 day**; zero and multiple values render **0 days** and **6 days**. The component uses the supplied analytics value unchanged.

## Cause and minimal repair

Both the existing English `flashcards.studyStreakDays` resource and component fallback unconditionally said `{{count}} days`. Replace only those two messages with the existing ICU plural convention: `{count, plural, one {# day} other {# days}}`.

Independent scope checks prove the production diff is exactly those two strings and the parsed English resource is otherwise identical. No analytics calculation, endpoint, deck order/selection, other metrics, locale framework or generated extension locale changes. Source snapshots and exact hashes are in `owned-manifest.json` and `review-snapshot/`.

## Test evidence

- Initial adjacent baseline: 6 passed across ReviewTab analytics and ReviewProgress plurals.
- New permanent real-component/real-i18next/production-ICU tests on original source: **3 failed, 4 passed, zero skipped**. Resource singular, missing-key fallback singular and same-instance singular update failed with actual `1 days`; zero/multiple controls passed.
- After the two production strings: new7 and ReviewProgress3 passed; one existing ReviewTab assertion for `6 days` failed because its hand-written translation stub only substituted double-brace placeholders.
- The documented fixture correction replaces that stub's option-message formatting with the existing real ICU plugin. The mock retains its original key/string fallback behavior. Every byte outside that translation fixture, including all assertions, is unchanged.
- Final frozen run: **18 passed /4 files /zero skipped /2.40s**: new streak7, existing ReviewTab analytics3, ReviewProgress3 and existing ICU-format5.

The new7 tests are byte-identical to the causal RED. They exercise0/1/6 with the production English resource and absent-key component fallback, then update the same rendered component0→1→6 while retaining other numeric metrics. There are no class/string-mirroring assertions or fake HTTP responses.

## Verification

From `apps/tldw-frontend`, using its default Vitest config:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/components/__tests__/ReviewAnalyticsSummary.plurals.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.analytics-summary.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/ReviewProgress.plurals.test.tsx ../packages/ui/src/i18n/__tests__/icu-format.test.ts
```

Root-CWD ESLint using the frontend config and actual logical filenames: **0 errors,23 unchanged baseline warnings,0 added/removed messages** across component and two owned tests. New test has no warnings. JSON parse and owned diff whitespace checks pass. Vitest transforms the actual TSX; a full application TypeScript check was not repeated for this two-string repair.

Bandit was run from the project venv on all three owned TSX files. It reported three syntax parse errors and cannot assess TypeScript; zero findings is not a security pass. The change introduces no HTML rendering, transport, input conversion or permission behavior. Detailed receipts are retained privately.

## Handoff and limits

Root released the production hold after retaining the real Study exchange and reload. This packet certifies source/component behavior; root owns native final-source acceptance, task/tracker updates and integration. No browser, runtime, model, configuration, task/tracker or git mutations were made by this author.
