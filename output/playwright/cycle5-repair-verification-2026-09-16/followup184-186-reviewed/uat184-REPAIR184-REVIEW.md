# UAT184 / TASK13260.121 review packet

## Result

Manage now renders one full ICU count message: `1 card`, `0 cards`, or `2 cards`. The generic Cards tab label, computed totals, selection state and first-run visibility remain unchanged.

Production scope is one count expression in ManageTab and one English `flashcards.manageCardCount` resource. The permanent tests extend the existing Manage empty-state fixture, exercising the actual i18next/ICU plugin and production English resources for count cases. Existing conditional fallback-copy tests retain their original boundary. No runtime/browser/task/tracker/git changes were made.

## Evidence

- Causal RED on original source:2 failed/7 passed. Zero/multiple and existing empty-state controls pass; the two failures show original `1 Cards` wording.
- Final frozen-test nonmutating baseline replay:4 failed/17 passed across184 and186. Two failures are the same singular assertions; the other two are the independently scoped186 overlay cases. The private Vite loader returns baseline production source only; current files are unchanged.
- GREEN:9/9 actual Manage count/empty-state tests, including0/1/2, same-instance count update, selection/clear and genuine first-run hidden summary.
- Adjacent:31/31 across Manage scheduling metadata15, Manage undo/bulk selection13 and ReviewProgress real ICU3. **184 total40 tests/4 files pass, no skips.**
- Shared184/186 verification:137/137 tests across11 files. Correct root-based ESLint:0 errors,14 unchanged baseline warnings across four scoped files,0 added/removed messages. Full compiler:90 baseline/current diagnostics,0 added/removed and none in owned tests.
- Bandit was run through the project venv on the TS/TSX scope. It reports four parse errors and cannot assess these JavaScript/TypeScript files; zero findings is not meaningful security coverage. Actual TS parsing, ESLint and behavior tests passed. JSON resource parses and diff check passes.

Initial plural controls were unnecessarily case-sensitive at zero/multiple and were corrected before causal RED to test grammar rather than capitalization. Initial frontend-CWD lint ignored shared package paths; those incomplete results were replaced by the root-CWD stdin check using each actual logical filename and retained baseline input.

## Reproduction

From apps/tldw-frontend, use its default Vitest config:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/ReviewProgress.plurals.test.tsx
```

`owned-manifest.json`, `owned.patch`, `review-snapshot/` and `baseline/` isolate this task. The shared design/plan and baseline-replay config are under `.tmp/uat184-186-ui-design-20260917`. `green-focused-final.log` and `green-adjacent.log` contain the actual combined runs; counts above are explicitly attributed by file.

## Remaining gate

Source is frozen for independent review. Parent must recapture actual native Manage count1 from genuine response data. This packet does not certify native acceptance.
