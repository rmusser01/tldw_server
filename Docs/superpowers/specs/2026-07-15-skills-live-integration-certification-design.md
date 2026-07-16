# Skills Live Integration Certification Design

Date: 2026-07-15
Status: Approved design, pending written-spec review
Backlog: TASK-530.15

## Summary

The `/skills` workflow now has broad deterministic coverage, but it still lacks
one strict integration gate that proves the real backend contract and the
packaged browser extension transport.

The WebUI Skills suite currently provides 13 deterministic mocked workflows
plus optional live smoke checks. The packaged extension parity suite provides
six deterministic workflows, but deliberately disables `tldw:request` and
exercises the direct-fetch fallback. Those suites are valuable and remain
unchanged. They do not prove that a real Skills lifecycle works through the
packaged MV3 background service worker.

This design adds one explicit release-gate command. It starts a disposable
single-user backend, runs a bounded lifecycle in the WebUI, then runs the same
target-specific lifecycle in the packaged extension while proving service
worker ownership for every Skills API request. The gate is strict when invoked,
but it is not added to default pull-request CI in this task.

## Goals

1. Certify a real Skills create, discovery, dry-render, persistence, Trash,
   restore, and purge lifecycle in the WebUI.
2. Certify the same lifecycle in the packaged extension through the actual MV3
   `background.js` request relay.
3. Fail, rather than skip, when an explicitly invoked certification prerequisite
   or workflow is unavailable.
4. Isolate authentication and Skills data from the developer's normal runtime.
5. Attempt both client surfaces and report their outcomes independently.
6. Retain small, sanitized evidence while deleting disposable runtime data.
7. Reuse existing process, port, redaction, and Playwright patterns without
   building a general-purpose certification framework.

## Non-Goals

- Do not add this gate to default pull-request CI.
- Do not replace or rewrite the existing mocked Skills suites.
- Do not call a real model or execute declared tools.
- Do not certify multi-user authentication in this slice.
- Do not replay every filter, sort, bulk, import, export, accessibility, or
  responsive permutation against the live backend.
- Do not force MV3 suspension, browser restart, or background-worker recovery.
- Do not add telemetry, dashboards, artifact databases, or a generic test
  orchestration layer.
- Do not change product behavior unless the live gate reproduces a real defect.

## Selected Approach

Use a thin Node runner plus two dedicated Playwright specs:

- one real-backend WebUI spec;
- one real-backend packaged-extension spec;
- one runner that owns the backend profile, processes, result aggregation,
  cleanup, and evidence contract.

This is preferred over the alternatives:

1. A monolithic Node browser script would duplicate Playwright fixtures,
   assertions, screenshots, and test reporting.
2. Adding strict live behavior to the existing mocked specs would mix skip-based
   smoke tests with a no-skip release gate and make failures harder to classify.
3. Requiring an operator-managed backend would weaken isolation and make results
   depend on unknown data and configuration.
4. A generalized multi-product certification framework is unnecessary for this
   bounded Skills requirement.

## Architecture

The runner lives under `apps/tldw-frontend/scripts/skills-certification/`. The
command is owned by `apps/tldw-frontend/package.json` and is invoked from that
package directory:

```sh
cd apps/tldw-frontend
bun run e2e:skills:certify
```

The runner performs these high-level operations:

```text
create evidence and runtime roots
  -> build isolated Skills profile
  -> initialize single-user auth database
  -> allocate and lock backend/WebUI URLs
  -> start backend and WebUI
  -> verify empty Library and Trash through direct API calls
  -> run WebUI certification
  -> recheck backend health
  -> run packaged-extension certification
  -> run direct API postconditions
  -> stop and verify every child process
  -> delete disposable runtime data
  -> write and scan sanitized evidence
  -> return aggregate status
```

The WebUI and extension run sequentially against one disposable backend. They
use different skill names, supplied by the runner, so an incomplete WebUI
cleanup cannot make the extension target ambiguous. The WebUI is responsible
for proving the initial empty state. The extension is responsible only for its
own target-specific lifecycle.

### Reused Code

Reuse the generic behavior already present in the onboarding UAT helpers:

- loopback port reservation;
- logged process spawning;
- HTTP readiness polling;
- process-tree signaling;
- text redaction and final text-artifact leak scanning.

Do not reuse the onboarding profile. It intentionally marks setup incomplete,
adds mock provider configuration, and does not explicitly export the Skills
`USER_DB_BASE_DIR` contract.

Do not reuse the onboarding artifact factory. It hardcodes onboarding paths,
marker names, and mock-provider logs. The Skills runner needs only a small fixed
directory layout and can create it directly with Node standard-library calls.

The Skills runner adds strict verification around the shared process helper. It
must wait for child `close`, verify termination, and treat teardown errors as
gate failures. A broad shared process-helper refactor is not required.

## Disposable Backend Profile

The backend is the real FastAPI application, started through Uvicorn after the
normal AuthNZ initializer runs. The runner must not set `TESTING`, `TEST_MODE`,
or similar test-runtime flags because production startup correctly rejects
those flags outside pytest.

The profile has an isolated root containing:

- copied and patched `config.txt`;
- an explicit `.env` file;
- the AuthNZ SQLite database;
- `USER_DB_BASE_DIR` and per-user Skills storage;
- temporary `HOME` and `TMPDIR` directories;
- extension browser profile directories.

The backend environment is built from an allowlist, not by spreading the host
environment. It includes only runtime prerequisites and synthetic settings,
including:

- `AUTH_MODE=single_user`;
- a synthetic `SINGLE_USER_API_KEY` recognized by the existing redactor;
- an absolute temporary `DATABASE_URL`;
- an absolute temporary `USER_DB_BASE_DIR`;
- temporary allowed-root values;
- explicit `TLDW_CONFIG_FILE` and `TLDW_ENV_FILE` paths;
- temporary `HOME` and `TMPDIR` values;
- required executable and certificate environment values.

The copied configuration marks first-time setup complete and uses single-user
auth. No provider credential is inherited or added. Dry-render is the only
execution mode under test.

The AuthNZ initializer runs before Uvicorn with the same profile. Backend health
must pass with the synthetic API key before either browser suite starts.

## Port And Process Ownership

Port reservation has a time-of-check/time-of-use race because the reservation
socket closes before Uvicorn or the WebUI binds. Before browser execution, the
runner may retry a confirmed bind conflict with a fresh set of ports. The retry
is bounded to three startup attempts and must not classify configuration,
import, auth, or health failures as bind conflicts.

After either client starts, the backend and WebUI URLs are immutable. The
packaged extension is built with permission for the selected backend URL, and
the WebUI process is configured for the same URL.

Every long-lived child is added to one process registry, including backend,
WebUI, Playwright, and extension build/test processes. Signal handlers are
installed before the first child is spawned. Normal completion, exceptions,
`SIGINT`, and `SIGTERM` all enter the same idempotent teardown path.

Teardown sends a graceful signal, escalates after a bounded timeout, waits for
the child `close` event so logs are flushed, and verifies the process has
terminated. A process that cannot be confirmed closed makes the gate fail.

## Fixed Test Data

The runner owns and validates two stable names, then passes them through the
child environment:

- `skills-cert-web`;
- `skills-cert-extension`.

The dedicated specs do not generate names from clocks or randomness. The
backend is disposable, so stable names are clearer and cannot collide with
normal user data. Skill content and arguments are fixed synthetic fixtures and
must not contain credentials or private data.

## WebUI Certification Workflow

The WebUI uses one Playwright test with named `test.step` phases:

1. Open `/skills` against the disposable backend.
2. Confirm the beginner empty state is visible.
3. Open `New Skill` and create `skills-cert-web` through the normal form.
4. Confirm success and the new item in the Library.
5. Search for the exact name and confirm the list request carries the query.
6. Open the test-run surface and render fixed arguments with `dry_run: true`.
7. Confirm the response reports `dry_run: true` and the expected rendered
   prompt is visible.
8. Reload the page, repeat the exact search, and confirm persistence.
9. Move the skill to Trash through the confirmation flow.
10. Open Trash and restore the skill.
11. Return to Library and confirm the restored skill is available.
12. Move it to Trash a second time.
13. Permanently delete it through the destructive confirmation flow.
14. Outside the browser context, confirm detail returns `404` and Trash no
    longer contains the name.

The test must not use route interception or mocked API responses. It may inspect
the dry-render request and response in memory, but it must not write request
bodies or headers to evidence.

## Packaged Extension Certification Workflow

The extension uses one Playwright test with the same named lifecycle phases and
the distinct `skills-cert-extension` target.

The existing `launchWithBuiltExtension()` helper remains authoritative for the
packaged Chromium launch. Its existing `prepareOptionsPage` hook allows the
strict test to install `BrowserContext` request listeners before the final
navigation to the Skills route. The helper receives one new optional profile
root so its temporary home, user-data, and copied extension directories live
under the runner-owned runtime root.

The strict spec must not install the direct-request fallback used by
`skills.parity.spec.ts`. After launch it requires a service worker whose URL is
exactly:

```text
chrome-extension://<extension-id>/background.js
```

MV3 may suspend and recreate a worker, so ownership is matched by exact URL,
not by Playwright `Worker` object identity.

The extension does not require the whole backend to be empty. It searches and
asserts only its runner-provided target. This preserves second-surface evidence
when the WebUI failed before completing its own cleanup.

## Relay Evidence Contract

The extension test registers `BrowserContext` listeners for `request`,
`response`, and `requestfailed`. Every request whose pathname starts with
`/api/v1/skills` enters a sanitized in-memory ledger.

Each retained entry contains only:

- HTTP method;
- canonical Skills route label or normalized pathname;
- expected-worker ownership as a boolean;
- terminal outcome;
- HTTP status when available.

The ledger never contains headers, request bodies, response bodies, API keys,
or rendered skill content.

Acceptance rules:

- Every Skills request must have `request.serviceWorker()` with the exact
  expected worker URL.
- Any page-owned Skills request fails the gate, including a direct-fetch
  fallback after a worker transport error.
- Failed Skills requests fail the workflow.
- Request bodies may be inspected in memory only to prove `dry_run: true`.
- GET volume is variable because OpenAPI discovery, reloads, and query refreshes
  may add reads.
- Redirect requests remain visible in the raw ledger but are not duplicate
  terminal mutations.
- Successful terminal mutations must be exactly:
  - one create response with `201`;
  - one dry-execute response with `200`;
  - two move-to-Trash responses with `204`;
  - one restore response with `200`;
  - one purge response with `204`.
- No additional failed or successful terminal mutation is allowed.

After the browser context closes, direct API postcondition requests run outside
the relay ledger. They confirm detail returns `404` and Trash excludes the
extension target.

## Dry-Render Boundary

The browser tests can prove only observable behavior:

- the request contains `dry_run: true`;
- the response contains `dry_run: true`;
- a rendered prompt is returned and displayed;
- the spawned backend receives no provider credentials.

The browser gate must not claim to prove every internal executor branch. The
backend's focused unit and integration tests remain authoritative for the
guarantee that dry-run mode bypasses model, tool, and fork execution.

## Result Aggregation And Recovery

Each surface has one of three recorded states:

- `passed`;
- `failed`;
- `not_run_infrastructure`.

`not_run_infrastructure` is a failure, never a skip.

The runner attempts the extension after any WebUI-specific failure when the
shared backend remains usable. This includes WebUI startup, browser launch, and
workflow failures; the extension does not depend on the WebUI frontend process.
Before each surface the runner checks backend health. If the backend crashed
during the WebUI phase, the runner may restart it once on the same port with the
same runtime profile solely to collect extension evidence. The top-level result
remains failed even if that restart succeeds and the extension passes.

If the backend cannot be restored on the same URL, the extension is recorded as
`not_run_infrastructure`. Once browser execution begins, no fresh-port restart
is permitted because the WebUI configuration and extension host permission are
already bound to the selected URL.

There are no automatic workflow retries. Playwright runs with one worker and
`retries: 0` on every machine. Startup bind-conflict retries and the single
same-port evidence restart are infrastructure handling, not test retries.

## Failure Classification

The summary records one primary category and bounded redacted detail:

- `preflight`;
- `backend_startup`;
- `backend_health`;
- `webui_startup`;
- `webui_workflow`;
- `extension_build`;
- `extension_launch`;
- `extension_worker`;
- `extension_relay`;
- `postcondition`;
- `artifact_safety`;
- `cleanup`;
- `interrupted`.

Uncaught browser `pageerror` events and failed Skills requests fail the relevant
surface. Console errors are retained as bounded diagnostics but arbitrary
`console.error` text is not itself a release failure. This avoids a broad and
fragile console allowlist while keeping actionable evidence.

## Evidence And Cleanup Contract

Evidence is written beneath the already ignored frontend test-results root:

```text
apps/tldw-frontend/test-results/skills-certification/<run-id>/
  summary.json
  logs/
  webui/
  extension/
    relay-ledger.json
```

Retain by default:

- sanitized backend, WebUI, extension, and runner logs;
- final JSON summary and per-surface results;
- sanitized extension relay ledger;
- Playwright failure screenshots containing only synthetic fixture data.

Do not retain Playwright traces or video. Those binary formats may contain
request headers, bodies, browser storage, or the synthetic API key, while the
existing leak scanner intentionally scans only bounded text-like artifacts.

Finalization order is fixed:

1. stop and verify browser and child processes;
2. delete the disposable runtime root, including extension profiles;
3. write the final summary with cleanup outcomes;
4. scan all retained text artifacts with the existing redactor/leak checker and
   the synthetic API key as an additional exact secret;
5. retain evidence only if the scan passes;
6. compute and return the final exit status.

If the scan finds a credential, the evidence root is removed and only a generic
artifact-safety failure is written to stderr. Passing certification requires
successful runtime deletion. This task does not add a runtime-preservation
mode; developers can run the underlying focused specs separately when they need
interactive debugging.

The command exits zero only when both surfaces pass, all postconditions pass,
relay ownership is proven, evidence is safe, and cleanup succeeds.

## Acceptance Matrix

| Surface | Required behavior | Pass condition |
| --- | --- | --- |
| Runner preflight | Isolated profile, auth init, backend/WebUI startup, empty Library and Trash | All prerequisites succeed with no test-runtime flags or inherited credentials |
| WebUI | Empty state, create, exact search, dry render, reload, Trash, restore, Trash, purge | User-visible transitions and direct API postconditions succeed |
| Extension | Exact worker, target-specific lifecycle, request ledger | Same lifecycle succeeds and every Skills request is worker-owned |
| Aggregation | Sequentially attempt both surfaces | Both pass; no skip or infrastructure omission is accepted |
| Safety | Teardown, runtime deletion, evidence scan | No live process, disposable runtime, or credential leak remains |

## Supporting Automated Tests

Use the repository's existing Node/TypeScript test tooling; add no dependency.
Focused tests cover:

1. Profile creation sets setup complete, uses absolute temporary auth and user
   database paths, and excludes host provider credentials.
2. Command construction fixes URLs before browser execution and forces one
   worker, zero retries, and dedicated output paths.
3. Startup retries only confirmed bind conflicts and never changes URLs after
   the browser phase begins.
4. The extension still runs after a WebUI behavior failure.
5. A backend crash permits at most one same-port evidence restart and can never
   produce an overall pass.
6. Preflight and cleanup failures still produce a final summary.
7. Teardown waits for close, detects a surviving child, and handles
   `SIGINT`/`SIGTERM` through one idempotent path.
8. Text artifacts are redacted, exact synthetic credentials are detected, and
   contaminated evidence is removed.
9. The extension launcher places profile directories under the supplied root.
10. Relay normalization handles exact worker URLs, page-owned requests,
    responses, request failures, redirects, and terminal mutation counts.

## Existing Regression Gates

The implementation must leave these suites unchanged and run them as focused
regression verification:

- the 13 deterministic mocked workflows in
  `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`;
- the six deterministic packaged-extension workflows in
  `apps/extension/tests/e2e/skills.parity.spec.ts`.

Those suites remain responsible for keyboard and focus behavior, responsive
widths, filters, sorting, bulk actions, import/export, and deterministic failure
recovery. The new gate adds real integration evidence rather than duplicating
those permutations.

## Verification Requirements

Implementation verification must include:

- focused runner and relay unit tests;
- extension launcher unit tests for the profile-root addition;
- existing mocked WebUI Skills Playwright coverage;
- existing extension Skills parity Playwright coverage;
- the new strict live certification command;
- focused TypeScript lint/type checks for touched files;
- `git diff --check`;
- scoped Bandit only if Python files are changed, otherwise a documented
  frontend/docs-only skip.

The strict command's report must show zero skipped tests. Product code changes
require separate focused regression coverage for the reproduced defect.

## Risks And Mitigations

- **Full application startup is broader than Skills.** Use explicit config,
  env, auth DB, `USER_DB_BASE_DIR`, `HOME`, and `TMPDIR` paths. Keep the child
  environment allowlisted and verify the worktree remains clean.
- **Port reservation can race.** Retry only confirmed pre-browser bind conflicts
  and lock URLs once clients start.
- **A WebUI failure can leave its skill behind.** Use distinct fixed targets and
  make the extension target-specific.
- **MV3 can recreate its service worker.** Match exact worker URL rather than
  object identity.
- **Direct-fetch fallback can hide relay failure.** Fail any page-owned Skills
  request observed at `BrowserContext` scope.
- **Binary traces can retain credentials.** Disable traces and video; retain
  bounded text evidence and failure screenshots only.
- **Teardown can appear successful before logs close.** Await `close`, verify
  process termination, and scan evidence only after teardown.
- **The gate can become a second broad UAT suite.** Keep one bounded lifecycle
  per surface and leave permutations in existing deterministic tests.

## Implementation Boundary

Expected touched areas are limited to:

- a Skills certification runner and its focused tests under
  `apps/tldw-frontend/scripts/`;
- one dedicated real-backend WebUI Playwright spec and strict configuration;
- one dedicated real-backend packaged-extension Playwright spec and strict
  configuration;
- the extension launch helper's optional runner-owned profile root and its
  focused test;
- package scripts and short command documentation;
- `TASK-530.15`, the implementation plan, and verification notes.

Do not modify backend or Skills product code unless the strict gate first
reproduces and documents a genuine defect.
