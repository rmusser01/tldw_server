# Cycle 4 Task 9 — UAT058 titles and UAT067 Character form

## Scope and status

Tasks: TASK-13260.20 and TASK-13260.15, updated with the official Backlog CLI before edits. Approved by plan4 Task9 and the presentation section of `Docs/Design/2026-09-16-uat-cycle-4-repairs.md`.

Production/test edits are frozen and ready for independent review. Final combined verification: **135 tests PASS across 8 suites**, exit 0, 314.10 seconds. Parent owns independent review/native acceptance and Review Markdown triage. No Playground, PlaygroundForm, useChatActions, backend, runtime, browser, inference, subagents or commit work occurred.

Exact eight product/test files and SHA-256 hashes: `/private/tmp/cycle4-task9-owned-manifest.json`. Four production files only:

- `apps/tldw-frontend/pages/prompts.tsx`
- `apps/tldw-frontend/pages/characters.tsx`
- `apps/packages/ui/src/utils/update-page-title.ts`
- `apps/packages/ui/src/components/Option/Characters/Manager.tsx`

## UAT058 diagnosis and change

Both actual Next page wrappers lacked Head/title. The shared imperative Chat title helper also wrote document.title after asynchronous Chat persistence/loading completed, overriding whichever Next route had since mounted.

The wrappers now publish `Prompts | tldw` and `Characters | tldw`. In Next documents, the imperative helper leaves Head authoritative. It uses the existing app distinction `"__NEXT_DATA__" in window`, already used by the shared layouts. Chat's existing active-history title hook still feeds Chat's own Head. Extension documents retain imperative titles.

Permanent tests use the real Next Head manager and actual page wrappers. The delayed callback test runs real saveMessageOnSuccess with controlled title generation and storage adapters, starts on Chat, moves to actual ServerSettings page, releases the pending save, proves persistence still completes, and asserts Settings retains its title. A stale callback while a different Chat title is mounted is also rejected. Existing reactive Chat title/account/history and SSR tests remain in scope; extension and missing-document controls are included.

Evidence:

- `/private/tmp/cycle4-task9-title-red.log`: four expected failures (two missing titles, two imperative overwrite controls), ten existing controls pass.
- `/private/tmp/cycle4-task9-title-green.log`: initial five suites, 31 PASS (before adding two extra extension/SSR utility controls).

## UAT067 diagnosis and change

The Character manager had an outer Suspense boundary around dialogs plus an inner Suspense around its lazy CharacterEditorForm. The inner boundary allowed an interactive empty Create Drawer while the form module was still loading. Clicking Close called the real createForm.resetFields before any Form had connected, producing the exact AntD warning retained in native evidence.

The Manager-only correction removes the inner Suspense. A cold editor now suspends its owning dialog until its actual Form can render; reset/cancel controls do not appear ahead of that Form. Form values, form instances, create/edit mutations, reset logic and warning handling are unchanged.

Evidence:

- `/private/tmp/cycle4-task9-character-red.log`: actual Manager + AntD early-close reproduction fails with the exact `Instance created by useForm is not connected to any Form element` warning.
- `/private/tmp/cycle4-task9-character-lifecycle-initial.log`: ordinary loaded create → cancel → reopen → submit passes before the repair.
- `/private/tmp/cycle4-task9-character-boundary-red.log`: final permanent regression explicitly holds the real lazy module, waits for load start, and fails because an interactive Close button is available before any form exists.
- `/private/tmp/cycle4-task9-character-green.log`: cold loading boundary, warm cancel/reopen/submit, and disconnected Edit-form control: 3 PASS.

The cold test intentionally runs before any test warms the lazy editor module. Its async module mock delays loading only; the actual editor, Manager, Drawer, Form, reset handler and submit handling are used. An earlier component-wrapper fixture was reassessed after load-start assertions showed it could inspect the DOM too early; final RED/GREEN evidence comes from the explicit module-loading gate.

## Verification commands and limits

From `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run ../packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx ../packages/ui/src/components/Option/Characters/__tests__/Manager.crossFeatureStage1.test.tsx ../packages/ui/src/components/Option/Characters/__tests__/CharacterEditorForm.expression-validation.test.tsx __tests__/pages/core-route-titles.test.tsx __tests__/pages/chat-title-late-persistence.test.tsx __tests__/pages/chat-title.integration.test.tsx __tests__/pages/chat-title.ssr.test.tsx ../packages/ui/src/utils/__tests__/update-page-title.test.ts --maxWorkers=1 --no-file-parallelism
```

Log: `/private/tmp/cycle4-task9-final-tests.log`.

Scoped ESLint uses `apps/tldw-frontend/eslint.config.mjs` and all eight product/test paths. Baseline obtained without source swaps by passing `git show HEAD:path` to ESLint `--stdin --stdin-filename path`. Current and baseline: **0 errors, 167 warnings**, exact rule/message multiplicities unchanged. Logs: `/private/tmp/cycle4-task9-lint-baseline.json`, `cycle4-task9-lint-final.json`, `cycle4-task9-lint-comparison.json`. `git diff --check` passes for owned paths.

No full TypeScript run by parent instruction; this unit does not claim an independently revalidated 90-diagnostic global baseline. ESLint/Vitest provide syntax/module validation of the exercised files. Bandit is not applicable to this TS/TSX-only unit. No live native reconnect, runtime or inference acceptance is claimed. Native UAT retained only the warning text rather than a form method stack, so the deterministic early-close reproduction establishes a matching reachable lifecycle defect; final native warning absence remains parent-owned.

## Final verification

- Combined owned/adjacent frontend suite: **135 PASS / 8 suites**, no failed tests or unhandled errors, exit 0. Includes all 99 Character Manager first-use tests.
- Final owned-scope ESLint: **0 errors, 167 warnings**, identical baseline rule/message multiplicities; no added or removed warnings.
- Owned `git diff --check`: PASS.
- All eight product/test SHA-256 hashes match the frozen manifest after verification.
- Backlog .20/.15 updated; left In Progress for parent independent/native acceptance.
