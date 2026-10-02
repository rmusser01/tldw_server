# UAT Notes, Media and Settings repairs

Associated Backlog task: TASK13260.281.2 (umbrella TASK13260.281).

ADR check: ADR required: no. These repairs preserve existing editor ownership,
keyword API options, permalink hydration and settings navigation conventions.
No public API, persistence format or module boundary changes are planned.

## Stage 1: Regressions
**Goal**: Reproduce rapid note selection, missing export tags, pending Media permalink navigation and unsaved Server URL loss using actual components/hooks with held transport.
**Success Criteria**: Narrow assertions fail for the traced production defects.
**Tests**: Notes authority races and canonical page save; Notes export hook; Media permalink hydration; Settings form lifecycle.
**Status**: Complete

## Stage 2: Minimal repairs
**Goal**: Fence Notes mutations during detail selection, include export keywords, preserve incoming Media target until hydration resolves, and guide/guard unsaved settings navigation.
**Success Criteria**: Regressions pass without replacing authentication configuration or broad refactoring.
**Tests**: Same focused regressions, followed by existing touched-domain suites.
**Status**: Complete

## Stage 3: Verification and handoff
**Goal**: Review touched changes and run affected checks.
**Success Criteria**: Focused tests and suitable static checks pass; limitations reported to coordinator.
**Tests**: Vitest, touched TypeScript lint/format, coordinator consolidated typecheck and Bandit.
**Status**: Complete

## Handoff verification

- Red: pending Notes selection allowed old content/title mutation and Save;
  the real editor remained editable. Unfiltered JSON export omitted both pages'
  tags. Incoming Media URL 9 reverted to old selection URL 7 while GET 9 waited.
  Settings had neither unsaved guidance nor unload/navigation cancellation.
- Green: 61/61 tests pass across five focused suites, including normal settings
  links, cancellation, successful save, and failed-save retention.
- Affected existing suites: Media permalinks/handoffs, Notes editor reliability
  and followup, export progress, settings auth modes and timeout forms passed.
  The expanded run exposed two export-preflight fixtures missing canonical config
  and verified-owner I/O. Those fixtures now supply these prerequisites while
  retaining the real owner hooks and original export assertions; both pass.
- Run Vitest from `apps/tldw-frontend` with
  `NODE_OPTIONS=--no-experimental-webstorage`; the isolated temporary config
  owned by the Chat/Research repair agent aliases one existing React installation.
  It avoids mixed existing React copies without changing dependencies.
- Touched production/tests lint exits 0 using frontend config from package-ui
  cwd (`--quiet`); Next emits an advisory about its pages directory. New export
  test formatted; `git diff --check` passes. Coordinator owns consolidated
  typecheck, security validation, review, commit and PR integration.
- These are source regressions using controlled I/O. Live UAT remains paused.

Coordinator final review is clear. Programmatic Health navigation and browser Back now use the existing route-leave guard and pass actual-router regressions. Final integrated frontend checks pass241 cases; touched lint0errors and full types0touched diagnostics with inherited unrelated failures. No live UAT or auth-guard relaxation.
