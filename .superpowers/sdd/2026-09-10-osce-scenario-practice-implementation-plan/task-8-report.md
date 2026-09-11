# Task 8 Report: Shared WebUI Data Layer, Generate, Create, And Manage

## Status

Complete on `TASK-12102.3.5.4` from head `60382f75f0`.

## Baseline

- The supplied linked worktree and branch were clean at the requested prior head.
- `apps/node_modules` was absent, so the first neighboring Vitest command failed while loading `vitest/config` and `@vitejs/plugin-react`.
- Ran `bun install --frozen-lockfile` from `apps/`; no tracked lockfile change was produced.
- Neighboring service and Generate/Create/Manage baseline: 4 files, 31 tests passed. Existing Ant `Alert.message` deprecation and Node localStorage warnings remain.

## TDD Evidence

RED:

- Exact focused command: 4 files failed. Service, query-hook, and editor suites failed to resolve the intentionally absent Task 8 modules; all three Generate/Create tests failed because the OSCE profile and activity selector were unavailable.
- A first run also exposed JSX in the required `.ts` service-test filename. The test harness was corrected to use `React.createElement`, then RED was rerun so the recorded failures are attributable only to missing production behavior.

GREEN:

- Exact focused command: 4 files, 18 tests passed.
- Neighboring Quiz service and Generate/Create/Manage command: 16 files, 90 tests passed.
- QuizPlayground navigation regression: 1 file, 22 tests passed.

## Implementation

- Added the typed OSCE service and React Query hooks for compact station pages, authoring CRUD, discriminated candidate/revealed attempts, repeated state filters, optimistic-version mutations, stable keys, and affected cache invalidation.
- Added the shared full-width `OsceStationEditor` with candidate content, bounded duration, ordered checklist/rubric/key-point controls, citations, candidate preview, explicit save, dirty navigation signaling, and 409 reload/keep/confirm recovery.
- Added catalog-aware OSCE Generate controls, preserving source, difficulty, focus, generation-provider, and verification-provider inputs while suppressing question-only controls and routing successful generation to Manage.
- Added segmented Quiz/OSCE Create composition and the shared explicit-save station workflow. Question draft autosave remains isolated to question quizzes.
- Added compact OSCE station management with verification labels, explicit-save editing, dirty selection/close guards, OSCE-incompatible action suppression, and ordinary Quiz behavior retained.
- Added client portability that keeps question-only exports at v1, emits v2 for mixed/OSCE exports, recursively removes sensitive provenance keys, and passes v1/v2 imports to the Task 7 endpoint.

## Verification

- `bun run lint`: exit 0 with 0 errors and the existing 169-warning frontend baseline. The new Task 8 files also pass the frontend ESLint config with no file diagnostics.
- `bun run typecheck`: blocked by existing errors in Presentation Studio and skills-certification test files. No diagnostic references a Task 8 or Quiz service/component path.
- `git diff --check`: passed.
- Bandit: not applicable; this slice changes TypeScript/TSX and task documentation only, with no Python touched.
- Prettier check is not a passing repository baseline: an untouched Quiz file (`ResultsTab.tsx`) and the existing shared surface fail the configured check. No broad formatting rewrite was applied.
- Commit: `feat(webui): add OSCE authoring workspace` (this completed slice).

## Concerns

- Practice autosave, timer, Take, and Results remain deferred to Task 9.
- The OSCE fallback/catalog remains planned; Task 8 only exposes controls when a test or server catalog marks the profile available.
- Repository-wide typecheck remains blocked by unrelated baseline errors listed above.

## Review Fix Round 1

Status: complete from Task 8 head `58f0491602`; committed separately as `fix(webui): harden OSCE authoring workflows`.

RED evidence:

- Expanded Task 8 command: 6 files, 11 failed and 44 passed. Failures covered OSCE launch suppression, partial-create reuse, ambiguous response handling, conflict order preservation, pagination, citation/rubric validation, and Manage reset guarding.
- The first full Quiz run also identified five older CreateTab suites whose complete hook mocks needed the newly consumed OSCE mutation hook. After those fixtures were corrected, all 13 affected Create tests passed.

GREEN evidence:

- Exact Task 8 focused command: 4 files, 25 tests passed.
- Take and navigation regressions: 2 files, 31 tests passed.
- Full Quiz component suite with bounded concurrency: 38 files, 259 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors. Scoped shared UI lint also reports no errors.
- Frontend typecheck remains blocked by the existing Presentation Studio and skills-certification diagnostics. No diagnostic references a changed Task 8, Quiz, or OSCE path.
- `git diff --check`: passed after the final report/task updates.
- Bandit remains not applicable because this review fix changes TypeScript, TSX, tests, and task documentation only.

Fixes:

- Hid question-attempt Start, Practice, and Review entry points for OSCE rows and blocked direct OSCE auto-start until Task 9 adds its practice route.
- Retained a confirmed manual OSCE quiz shell ID across station-save retries, routed station creation through the shared mutation/invalidation contract, and locked shell metadata while retrying. Ambiguous shell-create responses now fail closed and direct the author to inspect Manage instead of risking an automatic duplicate; the current API has no create idempotency key that could safely resolve that ambiguity.
- Carried the latest server `order_index` and version together through conflict recovery.
- Propagated Manage editor dirtiness to the playground so global reset requires confirmation.
- Added complete bounded station pagination for Manage and v2 export, including non-advancing-offset and maximum-page failure behavior instead of silent truncation.
- Aligned citation URL, rubric-label uniqueness, and offset pagination types with the backend/OpenAPI contract.

Concerns:

- The OSCE generation profile remains planned/hidden unless the server catalog explicitly marks it available.
- Task 9 practice, autosave, and results behavior was not started.
- The first unconstrained full Quiz run produced load-related five-second timeouts. The timed-out tests passed in focused reruns, and the complete suite passed with `--maxWorkers=2`.
