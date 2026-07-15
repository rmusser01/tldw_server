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
failure is reproduced in the built extension. A reproduced defect in shared UI
code is in scope when it receives focused shared coverage; unrelated shared UI
refactors are not. The task does not redesign Skills, change REST or MCP
behavior, add telemetry, or create a second Skills fixture system.

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
- production fixes only for failures reproduced by the new extension suite,
  including defects owned by shared UI code;
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
3. Register deterministic `/api/v1/health/live`, OpenAPI capability, Skills,
   execution, binary export, and failure routes using the existing WebUI Skills
   fixtures. Extend those fixtures only for missing protocol behavior; do not
   copy or relocate the fixture library.
4. After launch, enable the same test-only direct-request fallback already used
   by extension E2E so Playwright can intercept requests that would otherwise
   originate in the background worker. Invoke the production connection
   store's `checkOnce({ force: true })` after routes are installed and wait for
   connected/capable state. This tests production bootstrap logic; it must not
   patch the store to connected. The existing connection-state mutation seam
   may be used only by the explicit unreachable-state recovery test.
5. Navigate to `${optionsUrl}#/skills` and verify the route pathname/search state
   through the hash router.
6. Capture unexpected page errors, console errors, failed requests, and failed
   API assertions. Only narrowly documented startup failures may be ignored.
7. Launch a fresh persistent browser context with a fresh storage seed and fresh
   mutable fixture state for every test. Close that context in `finally`; tests
   must not depend on ordering or share seeded Skills, routes, drafts, history,
   downloads, or connection state.

Add package scripts parallel to the existing workspace-parity scripts:

- `test:e2e:skills-parity` for focused local execution;
- `test:e2e:skills-parity:strict`, which removes the prior
  `.skills-parity-e2e-report.json`, runs Playwright with
  `PLAYWRIGHT_JSON_OUTPUT_NAME=.skills-parity-e2e-report.json`, calls the
  existing `scripts/assert-playwright-no-skips.mjs`, and copies the report to
  `test-results/skills-parity-e2e-report.json`.

No new Playwright configuration or launcher is required.

## Required Test Cases

The suite is capped at six deterministic tests. It does not duplicate every
WebUI action variant.

### 1. Bootstrap and beginner journey at 1280x900

- Open `#/skills` through the built options shell.
- Confirm the capability gate resolves to the Skills manager.
- Begin with no Skills and assert the first-use empty state. Activate the
  `Seed built-ins` button, assert one `POST /api/v1/skills/seed?overwrite=false`
  request, fulfill it with `{ seeded: ["summarize"], count: 1 }`, and assert the
  `Built-in skills seeded` confirmation and `summarize` row appear.
- Open one Skill's details and confirm its description and runtime disclosure.
- Open Test run, enter arguments, and press Enter; verify this performs dry
  render only.
- Use the explicit Run test control and verify the request is not dry-run.
- Trigger Use in chat and verify `window.location.hash` becomes `#/chat`, the
  chat surface renders, and the visible `#textarea-message` composer contains
  `/skill summarize`. The existing focused `SkillsManager` test remains
  authoritative for the exact store call; the extension assertion verifies the
  user-observable handoff rather than inspecting private Zustand state.

### 2. Hash-backed power-user state at 1280x900

- Load a deterministic multi-page library.
- Select exactly two visible Skills and trigger bulk export before narrowing the
  library. Assert one `.zip` download completes and the selection count remains
  truthful before and after export. Assert exactly two per-Skill
  `GET /api/v1/skills/{name}/export` requests; the shared fixture must return
  deterministic binary content and `content-disposition` filenames for each.
  The manager combines those responses client-side into the single downloaded
  archive.
- Apply `q=target`, `mode=fork`, `tools=with-tools`,
  `model=gpt-4.1-mini`, name
  descending sort, and page size 20.
- Verify the resulting hash search parameters exactly represent those values:
  `#/skills?q=target&mode=fork&tools=with-tools&model=gpt-4.1-mini&sort=name&order=desc&pageSize=20`.
- Assert the target fixture remains visible and the final list request contains
  the same server-side query, filter, sort, order, and limit values.
- Reload and verify the view is restored and the same API query is issued.
- Browser Back and Forward must restore the prior and committed filter states.

### 3. Trash management at 1280x900

- Using the deterministic Trash fixture, move `summarize` to Trash, assert the
  immediate Undo action, enter the Trash view, restore the Skill, and assert it
  returns to Library.

Existing component and WebUI tests remain authoritative for cross-page
selection, permanent purge, and other detailed action variants.

### 4. Compact keyboard and focus contract at 390x844

- Set the viewport to 390x844 and assert
  `document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1`.
- Assert the `Skills` level-one heading, Search skills textbox, `Skills view`
  radio group, New Skill button, and the target row's named details/test actions
  are discoverable by role or accessible name.
- Open details by keyboard, close with Escape, and assert focus returns to the
  named details trigger.
- Open Test run by keyboard, render once, close with Escape, and assert focus
  returns to the named test trigger.
- Re-run the overflow assertion while the details drawer and test dialog are
  open.

### 5. List failure, retry, and unreachable-state recovery

- Hold the first `GET /api/v1/skills` response behind a deterministic gate and
  assert the `Loading skills` `status` announcement before releasing it as HTTP
  503. Return a valid list on retry. Assert the shared recovery callout and its
  `Try again` action, and assert the row appears after retry.
- Primary copy must not contain the seeded API key, absolute paths, or the raw
  mocked response body.
- In the same fresh context, use the existing connection-state test seam to set
  the store unreachable. Assert the `Can't reach your tldw server right now.`
  state with `Health & diagnostics` and `Open Settings` actions. Assert New
  Skill, Seed built-ins, and Import actions are absent because the connection
  gate does not render the manager while unreachable.

### 6. Extension session draft recovery

- Open New Skill, enter a unique valid name and instructions, and leave the
  drawer dirty.
- Reload `options.html#/skills`, reopen New Skill, and assert the existing
  `tldw:skills:authoring-draft:v1:` session draft is reported as recovered with
  the entered values intact.
- Discard the recovered draft and assert a second reopen starts clean.

Stale-result suppression, retained-construction cancellation, import-draft
recovery, and every close-path confirmation remain covered by the existing
shared Skills tests and WebUI UAT. They are not repeated here unless the built
extension first reproduces a platform-specific failure.

## Failure and Diagnostic Rules

- No unconditional or environment-based skips are permitted in the strict
  deterministic suite.
- Browser launch, missing build output, missing extension APIs, or absent Skills
  route support are test failures.
- Expected HTTP 503 from the recovery test is a fulfilled response and is not a
  transport failure. If the client emits a console error, only the exact
  `GET /api/v1/skills` failure in that test may be excluded; broad message or
  network-error regexes are not allowed.
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
2. Six deterministic tests cover bootstrap/beginner, hash state/export, Trash,
   compact keyboard/focus, list/unreachable recovery, and session draft
   recovery without a live backend.
3. Hash-backed state survives reload and Use in chat navigates through the
   extension router with `/skill summarize` visible in the chat composer.
4. Compact-width, keyboard, focus-return, loading, and error semantics remain
   usable in the extension shell.
5. Extension list failure/retry, unreachable state, and session draft recovery
   are verified without lost user work or leaked sensitive diagnostics;
   existing shared tests remain authoritative for cancellation and stale-result
   suppression.
6. The strict extension suite contains no skips and fails on unexpected browser
   or network errors.
7. Any production change is tied to a reproduced extension failure and is
   covered at the narrowest owning boundary.
8. Focused build, tests, type checks, diff hygiene, and applicable security
   checks pass or record an unchanged external baseline precisely.
