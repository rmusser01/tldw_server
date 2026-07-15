# Skills Browser-Extension Parity Design

**Backlog task:** TASK-12970

**Status:** Ready for review

**Date:** 2026-07-15

## Summary

Certify the merged `/skills` experience through the production browser-extension
options shell. Add a deterministic Playwright suite that loads the built Chrome
extension, seeds its real storage and connection bootstrap, navigates through the
hash router to `#/skills`, and exercises the highest-risk beginner, power-user,
responsive, accessibility, persistence, and recovery seams.

The task is verification-led. Production code changes are allowed only when a
failure is reproduced in the built extension. It does not redesign Skills,
change REST or MCP behavior, add telemetry, or create a second Skills fixture
system.

## Current Evidence

- `OptionSkillsRoute` and `SkillsWorkspace` are shared by the WebUI and extension.
- The extension options entry is a thin import of the shared options application.
- The extension uses a hash router, while the WebUI workflow suite uses normal
  path routing.
- `SkillsManager` stores search, filters, sorting, pagination, and view state in
  router search parameters and navigates to `/chat` for Use in chat.
- WebUI Skills UAT has deterministic beginner, power-user, Trash, loading, and
  failure fixtures.
- The extension has built-package parity suites for other workspaces but no
  `/skills` Playwright coverage.
- A production Chrome extension build from `origin/dev` succeeds. Existing
  duplicate-import, circular-chunk, and chunk-size warnings are baseline and are
  outside this task.
- Focused Skills route and query-state tests pass on the starting revision.

The missing evidence is therefore the platform seam: extension storage and
authentication bootstrap, capability discovery, hash URL behavior, built-bundle
loading, extension dimensions, and extension navigation.

## Approaches Considered

### A. Built-extension deterministic contract

Launch the packaged extension and intercept the Skills API with deterministic
Playwright fixtures. Exercise the real extension router, storage, capability
gate, and shared production components.

This is the selected approach. It isolates extension behavior without requiring
a mutable backend and produces stable failures for CI and local review.

### B. Shared component tests only

Add more Vitest coverage around `SkillsManager` and the route wrapper.

Rejected as the primary approach because those tests cannot verify extension
storage, the hash router, packaged chunks, or extension navigation. Existing
component coverage is already extensive.

### C. Real-backend extension UAT only

Run every workflow against a live tldw_server instance.

Rejected as the required gate because setup, authentication, seeded data, and
cleanup would make routine parity checks slow and environment-dependent. A
small live smoke may be documented separately, but it cannot replace the
deterministic built-extension contract.

## Scope

In scope:

- a focused extension Skills parity Playwright spec;
- a strict package script that fails on test skips;
- reuse of the existing platform-neutral WebUI Skills API fixtures;
- production fixes only for extension-specific failures reproduced by the new
  suite;
- focused shared UI tests when a production fix changes shared behavior;
- screenshots and browser diagnostics for review, not tracked product assets.

Out of scope:

- backend, database, REST, MCP, or Skill execution changes;
- another Skills page redesign or new user-facing feature;
- duplicating the full WebUI Skills suite in the extension;
- moving or rewriting the existing fixture library solely for directory purity;
- Firefox/Edge certification in this first package-level gate;
- telemetry, analytics, visual-regression infrastructure, or a new test harness;
- unrelated extension build warnings or general bundle optimization.

## Test Architecture

Add `apps/extension/tests/e2e/skills.parity.spec.ts` and run it against the
existing `chromium-extension` Playwright project.

The suite will:

1. Use `launchWithBuiltExtension()` so launch/build failures fail the test rather
   than becoming conditional skips.
2. Seed first-run completion and a synthetic single-user server configuration
   through the existing extension storage helper.
3. Register deterministic health, OpenAPI capability, Skills, execution, and
   failure routes using the existing WebUI Skills fixtures. Direct test-only
   fixture reuse is preferred over copying or relocating the fixture library.
4. Force the existing connection store into its connected state only after its
   production bootstrap is present.
5. Navigate to `${optionsUrl}#/skills` and verify the route pathname/search state
   through the hash router.
6. Capture unexpected page errors, console errors, failed requests, and failed
   API assertions. Only narrowly documented startup failures may be ignored.
7. Close the persistent browser context in `finally` for every test.

Add package scripts parallel to the existing workspace-parity scripts:

- `test:e2e:skills-parity` for focused local execution;
- `test:e2e:skills-parity:strict` for JSON output plus the existing no-skips
  assertion.

No new Playwright configuration or launcher is required.

## Workflow Coverage

### Beginner contract

- Open `#/skills` through the built options shell.
- Confirm the capability gate resolves to the Skills manager.
- Start from the first-use empty state and seed built-in Skills.
- Open one Skill's details and confirm its description and runtime disclosure.
- Open Test run, enter arguments, and press Enter; verify this performs dry
  render only.
- Use the explicit Run test control and verify the request is not dry-run.
- Trigger Use in chat and verify the extension hash route becomes `/chat` with
  the expected invocation handed to the shared chat state.

### Power-user contract

- Load a deterministic multi-page library.
- Apply search, mode/tools/model filters, sorting, and a non-default page size.
- Verify those constraints are encoded inside the extension hash URL.
- Reload and verify the view is restored and the same API query is issued.
- Exercise one row-management action and bounded multi-page selection/export.
- Move one Skill to Trash, switch views, and restore it.

The contract tests platform seams rather than repeating every WebUI assertion.
Existing component and WebUI tests remain authoritative for detailed action
variants.

### Responsive and accessibility contract

- Set an extension-relevant compact viewport and assert no document-level
  horizontal overflow.
- Verify the primary toolbar, filters, row actions, details drawer, and test-run
  dialog remain reachable.
- Complete the principal workflow by keyboard.
- Verify focus returns to the invoking control after closing the drawer and
  dialog.
- Assert persistent interactive controls have usable accessible names and the
  loading/error states expose status or alert semantics.

### Recovery contract

- Fail the initial Skills list request and verify the recovery surface exposes a
  working retry without leaking raw secrets into primary copy.
- Start a delayed request, change route/query scope, and verify its stale result
  cannot replace the current view.
- Begin a dirty create/import draft, reload the extension page, and verify the
  session draft recovery prompt or restored draft appears.
- Verify an unreachable connection state blocks mutation while preserving a
  clear route back to connection recovery.

## Failure and Diagnostic Rules

- No unconditional or environment-based skips are permitted in the strict
  deterministic suite.
- Browser launch, missing build output, missing extension APIs, or absent Skills
  route support are test failures.
- Expected mocked API failures must be asserted in the UI and excluded narrowly
  from the request-failure collector by exact endpoint/status behavior.
- Unexpected page errors, console errors, and request failures fail the test and
  are reported with bounded URL/error context.
- Tests must not log API keys, Skill content, filesystem paths, or raw private
  response bodies.
- Browser contexts and delayed route handlers must be drained or closed during
  cleanup so cancellation does not leak across tests.

## Production-Fix Boundary

When the new suite fails:

1. Reproduce the failure on the built extension.
2. Identify whether the defect belongs to shared UI or the extension shell.
3. Add the smallest failing unit/integration assertion at the owning boundary
   when practical.
4. Apply the minimal production fix.
5. Rebuild and rerun the focused extension contract plus affected shared tests.

Do not refactor unrelated Skills or extension code. If no production defect is
reproduced, this task may correctly ship only the parity suite, scripts,
documentation, and task record.

## Verification

- Production Chrome extension build succeeds.
- Strict extension Skills parity suite passes with zero skipped tests.
- Existing WebUI Skills Playwright suite remains passing when shared test
  fixtures are touched.
- Focused Skills route, query-state, manager, preview, and drawer tests pass for
  any affected shared production scope.
- Extension TypeScript compile passes, or any unchanged repository baseline is
  recorded with touched-scope evidence.
- Locale JSON and i18n checks run only if copy changes.
- `git diff --check` passes.
- Bandit runs for touched Python paths; otherwise the frontend-only skip is
  recorded.

## Acceptance Criteria

1. The built extension options shell opens Skills through production routing,
   storage/auth bootstrap, and capability discovery.
2. Deterministic beginner and power-user contracts exercise the identified
   extension seams without a live backend.
3. Hash-backed state survives reload and Use in chat navigates through the
   extension router with the expected invocation state.
4. Compact-width, keyboard, focus-return, loading, and error semantics remain
   usable in the extension shell.
5. Failure, cancellation, retry, stale-result, and draft-recovery behavior is
   verified without lost user work or leaked sensitive diagnostics.
6. The strict extension suite contains no skips and fails on unexpected browser
   or network errors.
7. Any production change is tied to a reproduced extension failure and is
   covered at the narrowest owning boundary.
8. Focused build, tests, type checks, diff hygiene, and applicable security
   checks pass or record an unchanged external baseline precisely.

