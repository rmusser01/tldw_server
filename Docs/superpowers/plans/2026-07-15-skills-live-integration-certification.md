# Skills Live Integration Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit, strict release-gate command that certifies the real `/skills` lifecycle in the WebUI and packaged Chrome MV3 extension against one disposable FastAPI backend.

**Architecture:** A thin Node runner owns one isolated single-user backend profile, bounded process lifecycle, result aggregation, direct API postconditions, and sanitized evidence. Two dedicated Playwright specs reuse one small UI workflow helper; the extension spec additionally proves that every Skills request is owned by the expected MV3 `background.js` worker. Existing mocked WebUI and extension parity suites remain unchanged.

**Tech Stack:** Node.js ESM, TypeScript, Bun, Vitest, Playwright, Chromium MV3, Next.js, FastAPI/Uvicorn, SQLite.

**Spec:** `Docs/superpowers/specs/2026-07-15-skills-live-integration-certification-design.md`

**Backlog:** `TASK-530.15`

---

## Stage Overview

| Stage | Goal | Success Criteria | Status |
| --- | --- | --- | --- |
| 1 | Isolated and testable runner primitives | Profile, command, process, result, cleanup, and artifact tests pass | Not Started |
| 2 | Strict browser surfaces | WebUI and extension specs list exactly one test each; launcher and relay tests pass | Not Started |
| 3 | Thin end-to-end orchestration | `e2e:skills:certify` attempts both surfaces, aggregates failures, and always finalizes safely | Not Started |
| 4 | Release-gate verification | Existing Skills suites and the strict live command pass with zero skips and no retained runtime | Not Started |

## Scope Boundary

- Do not modify backend or Skills product behavior unless the strict command reproduces a genuine defect.
- Do not add this command to default pull-request CI.
- Do not add a model provider, mock provider, telemetry, runtime-preservation mode, generic UAT framework, or browser matrix.
- Do not rewrite the existing 13 mocked WebUI workflows or six packaged-extension parity workflows.
- Reuse `onboarding-uat/ports.mjs`, `spawnLoggedProcess()`, `waitForHttpOk()`, `stopProcessTree()`, `redactText()`, and `assertNoSecretLeaks()` as narrow primitives. Do not reuse the onboarding profile or artifact factory.
- Keep the runner's optional dependency overrides private to unit tests. They are concrete side-effect seams, not a plugin system.

## File Responsibilities

### Runner Primitives

- Create `apps/tldw-frontend/scripts/skills-certification/profile.mjs`: disposable profile creation, secret scrubbing, allowlisted environments, fixed command construction, and bind-conflict classification.
- Create `apps/tldw-frontend/scripts/skills-certification/lifecycle.mjs`: immediate child registration, close tracking, process-tree verification, signal wiring, and idempotent teardown.
- Create `apps/tldw-frontend/scripts/skills-certification/evidence.mjs`: fixed evidence layout, bounded sanitized JSON writing, safe runtime deletion, final leak scan, and contaminated-evidence removal.
- Create `apps/tldw-frontend/scripts/skills-certification/run.mjs`: the only orchestration entry point, direct Skills API checks, surface continuation/restart policy, failure aggregation, and exit status.
- Create focused tests under `apps/tldw-frontend/scripts/__tests__/skills-certification-*.test.ts`.

### Browser Surfaces

- Create `apps/tldw-frontend/e2e/utils/skills-live-certification.ts`: the shared UI lifecycle only; no process, profile, or evidence responsibilities.
- Create `apps/tldw-frontend/e2e/skills-certification/playwright.config.ts`: one-worker, zero-retry, trace-off, video-off WebUI configuration.
- Create `apps/tldw-frontend/e2e/skills-certification/skills.live.spec.ts`: one real-backend WebUI certification test.
- Modify `apps/extension/tests/e2e/utils/extension-build.ts`: accept one optional profile root.
- Modify `apps/extension/tests/e2e/utils/extension-build.test.ts`: prove every temporary extension directory stays under that root.
- Create `apps/extension/tests/e2e/utils/skills-certification-relay.ts`: sanitized request ledger and exact mutation assertions.
- Create `apps/extension/tests/e2e/utils/skills-certification-relay.test.ts`: relay ownership, redirect, failure, and count coverage.
- Create `apps/extension/playwright.skills-certification.config.ts`: one-worker, zero-retry, trace-off, video-off extension configuration.
- Create `apps/extension/tests/e2e/skills.live-certification.spec.ts`: one strict packaged-extension certification test.

### Command And Documentation

- Modify `apps/tldw-frontend/package.json`: add only `e2e:skills:certify`.
- Create `Docs/Development/Skills_Live_Integration_Certification.md`: prerequisites, command, evidence, exit contract, and non-CI status.
- Update `TASK-530.15` through Backlog MCP/CLI with plan, verification, commit, and PR references.

## Stage 0: Preflight

- [ ] Confirm the isolated worktree is clean and based on current `origin/dev`:

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin dev
git merge-base --is-ancestor origin/dev HEAD
git status --short --branch
git rev-parse origin/dev HEAD
```

Expected: the ancestor check exits 0 and only task-owned changes appear.

- [ ] Read `TASK-530.15` through the Backlog workflow and keep it `In Progress`. Record this plan path without manually editing the task file.

- [ ] Activate the shared project virtualenv and install the Bun workspace only
  when this fresh worktree has no ignored dependency tree:

```bash
repo_root="$(git rev-parse --show-toplevel)"
common_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  source "$common_root/.venv/bin/activate"
fi
test -x "$VIRTUAL_ENV/bin/python"
cd "$repo_root/apps"
if [ ! -d node_modules ]; then
  bun install --frozen-lockfile
fi
test -d node_modules
git diff --exit-code -- bun.lock
```

Expected: the shared Python environment is active, the worktree-local Bun
workspace is ready, and `bun.lock` is unchanged. A failed install or missing
Python environment is a preflight blocker, not a skipped certification.

## Task 1: Build The Disposable Profile And Fixed Commands

**Files:**
- Create: `apps/tldw-frontend/scripts/skills-certification/profile.mjs`
- Create: `apps/tldw-frontend/scripts/__tests__/skills-certification-profile.test.ts`

- [ ] **Step 1: Write failing profile-isolation tests**

Cover these contracts with temporary directories:

```ts
expect(path.isAbsolute(profile.usersDbPath)).toBe(true)
expect(path.isAbsolute(profile.userDatabasesDir)).toBe(true)
expect(configText).toContain("setup_completed = true")
expect(configText).toContain("auth_mode = single_user")
expect(configText).not.toContain("host-provider-secret")
expect(backendEnv.USER_DB_BASE_DIR).toBe(profile.userDatabasesDir)
expect(backendEnv.HOME).toBe(profile.homeDir)
expect(backendEnv.TMPDIR).toBe(profile.tmpDir)
expect(backendEnv).not.toHaveProperty("OPENAI_API_KEY")
expect(backendEnv).not.toHaveProperty("ANTHROPIC_API_KEY")
expect(backendEnv).not.toHaveProperty("TESTING")
expect(backendEnv).not.toHaveProperty("TEST_MODE")
for (const childEnv of [
  backendEnv,
  frontendEnv,
  webuiPlaywrightEnv,
  extensionBuildEnv,
  extensionPlaywrightEnv,
]) {
  expect(childEnv).not.toHaveProperty("OPENAI_API_KEY")
  expect(childEnv).not.toHaveProperty("ANTHROPIC_API_KEY")
  expect(childEnv).not.toHaveProperty("HOST_PROVIDER_SECRET")
}
```

Also assert:

- the copied INI blanks values whose normalized key ends in `api_key`, `token`, `secret`, or `password`, then writes only the synthetic single-user key;
- `.env` contains only single-user auth and isolated absolute paths;
- backend, frontend, WebUI Playwright, extension build, extension Playwright,
  and package-local Chromium probe commands use fixed working directories and
  explicit argument arrays;
- WebUI and extension Playwright commands point at their dedicated configs;
- the extension command sets `TLDW_E2E_SKIP_EXTENSION_BUILD=1`, because the runner tracks the production Chrome build separately;
- fixed names are exactly `skills-cert-web` and `skills-cert-extension`;
- `isConfirmedBindConflict()` accepts `EADDRINUSE`, `address already in use`, and matching Uvicorn errno text, but rejects import, auth, health, and configuration failures.

- [ ] **Step 2: Run the tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run scripts/__tests__/skills-certification-profile.test.ts
```

Expected: FAIL because the Skills profile module does not exist.

- [ ] **Step 3: Implement the minimal profile API**

Export only these stable entry points:

```js
export const SKILLS_CERT_API_KEY = "THIS-IS-A-SECURE-KEY-123-UAT"
export const SKILLS_CERT_NAMES = Object.freeze({
  webui: "skills-cert-web",
  extension: "skills-cert-extension",
})

export function createSkillsCertificationProfile(options) {}
export function buildSkillsCertificationEnvironments(options) {}
export function buildSkillsCertificationCommands(options) {}
export function isConfirmedBindConflict(text) {}
```

Implementation requirements:

1. Create one marker-protected runtime root beneath the supplied temporary base.
2. Create `Config_Files`, auth DB, `Databases/user_databases`, `home`, `tmp`, and extension-profile directories.
3. Copy the repository `config.txt`, scrub credential-valued INI keys, and patch `[Setup]` to enabled/completed plus `[AuthNZ]` to single-user.
4. Write an explicit `.env` with `AUTH_MODE`, `SINGLE_USER_API_KEY`, `DATABASE_URL`, `USER_DB_BASE_DIR`, and allowed-root variables. Add no provider configuration.
5. Start every child environment from one fixed safe-base allowlist containing
   only executable, locale, virtualenv, certificate, browser-runtime, and
   platform prerequisites. Never spread the runner's `process.env` into any
   auth-init, backend, frontend, build, probe, or Playwright child.
6. Build backend, frontend, build, probe, and browser-test environments
   separately. Replace backend `HOME`/`TMPDIR` with profile paths. Pass the
   synthetic key only where connection seeding requires it; never serialize it
   into command arguments or result JSON.
7. Resolve Python from the active `VIRTUAL_ENV`, then the worktree-local `.venv`
   as a fallback, and fail preflight if neither executable exists.
8. Build one finite probe command per frontend package. Each command imports
   that package's `@playwright/test`, launches headless Chromium without an app
   URL, closes it, and exits nonzero on failure. Never install or download a
   browser from the certification command.

- [ ] **Step 4: Run profile tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run scripts/__tests__/skills-certification-profile.test.ts
```

Expected: PASS with no host provider credential in any generated file or child environment.

- [ ] **Step 5: Commit the profile slice**

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/tldw-frontend/scripts/skills-certification/profile.mjs \
  apps/tldw-frontend/scripts/__tests__/skills-certification-profile.test.ts
git commit -m "test(skills): isolate live certification profile"
```

## Task 2: Add Strict Process, Result, And Evidence Finalization

**Files:**
- Create: `apps/tldw-frontend/scripts/skills-certification/lifecycle.mjs`
- Create: `apps/tldw-frontend/scripts/skills-certification/evidence.mjs`
- Create: `apps/tldw-frontend/scripts/__tests__/skills-certification-lifecycle.test.ts`
- Create: `apps/tldw-frontend/scripts/__tests__/skills-certification-evidence.test.ts`

- [ ] **Step 1: Write failing lifecycle tests**

Use `EventEmitter` child doubles and fake process-tree probes. Prove that:

- registration attaches a `close` listener immediately;
- a child `exit` event alone does not satisfy teardown;
- teardown calls the shared `stopProcessTree()`, awaits `close`, and probes the PID/process group afterward;
- a surviving child rejects teardown;
- reverse-order teardown is used;
- two teardown calls share one promise and do not signal twice;
- `SIGINT` and `SIGTERM` enter that same teardown path;
- every finite command remains registered until `close` and returns its code/signal.

The public surface stays small:

```js
export function createProcessRegistry(options) {}
export function installCertificationSignalHandlers(options) {}
```

- [ ] **Step 2: Write failing evidence and aggregation tests**

Cover:

- fixed evidence paths under `test-results/skills-certification/<run-id>`;
- separate evidence and runtime markers;
- bounded `redactText()` output before JSON/log writes;
- deterministic compaction of each retained `.log` file to at most 1 MiB by
  preserving bounded head and tail sections with a truncation marker;
- safe refusal to delete paths outside the expected roots;
- successful runtime deletion before summary writing and scanning;
- exact synthetic-secret detection through `assertNoSecretLeaks(..., { additionalSecrets: [...] })`;
- contaminated evidence removal with no replacement evidence file;
- cleanup failure retained in the summary and final status;
- combined workflow, artifact-safety, and cleanup failures retaining every category;
- `interrupted` overriding `primary_category`, otherwise the first category in the documented phase order;
- top-level `cleanup` and `artifact_safety` fields remaining visible regardless of primary category.

Use this summary shape; do not add headers, bodies, storage, or credentials:

```ts
type CertificationSummary = {
  run_id: string
  status: "passed" | "failed"
  primary_category: string | null
  failures: Array<{ category: string; surface?: "webui" | "extension"; detail?: string }>
  surfaces: {
    webui: { state: "passed" | "failed" | "not_run_infrastructure"; postcondition: boolean }
    extension: { state: "passed" | "failed" | "not_run_infrastructure"; postcondition: boolean }
  }
  cleanup: { children_closed: boolean; runtime_deleted: boolean }
  artifact_safety: { passed: boolean }
}
```

- [ ] **Step 3: Run the tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run \
  scripts/__tests__/skills-certification-lifecycle.test.ts \
  scripts/__tests__/skills-certification-evidence.test.ts
```

Expected: FAIL because the lifecycle and evidence modules do not exist.

- [ ] **Step 4: Implement strict lifecycle behavior**

Wrap the existing onboarding primitives rather than modifying them broadly:

```js
const registry = createProcessRegistry({
  spawnLoggedProcess,
  stopProcessTree,
  probeProcessTree,
})

const backend = registry.spawn(command, logPath)
const result = await registry.wait(finiteCommand)
await registry.teardown()
```

On POSIX, verify the detached process group is gone; on Windows, verify the child PID is gone. Timeouts and probe errors are cleanup failures. Signal handlers must be installed before the first child spawn and removed during finalization.

- [ ] **Step 5: Implement fixed evidence finalization**

Export only:

```js
export function createSkillsCertificationEvidence(options) {}
export function writeSanitizedJson(filePath, value) {}
export function buildCertificationSummary(input) {}
export function finalizeSkillsCertificationEvidence(options) {}
```

Finalization order must be:

1. receive the completed process-teardown outcome;
2. delete and verify the marker-protected runtime root;
3. compact retained `.log` files without truncating JSON reports;
4. write the final sanitized summary;
5. scan all retained text artifacts plus the exact synthetic key;
6. remove the entire marker-protected evidence root on any leak;
7. return a failing status for teardown, runtime deletion, scan, or artifact removal errors.

Do not add a preserve option.

- [ ] **Step 6: Run lifecycle/evidence tests GREEN and commit**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run \
  scripts/__tests__/skills-certification-lifecycle.test.ts \
  scripts/__tests__/skills-certification-evidence.test.ts
```

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/tldw-frontend/scripts/skills-certification/lifecycle.mjs \
  apps/tldw-frontend/scripts/skills-certification/evidence.mjs \
  apps/tldw-frontend/scripts/__tests__/skills-certification-lifecycle.test.ts \
  apps/tldw-frontend/scripts/__tests__/skills-certification-evidence.test.ts
git commit -m "test(skills): harden certification finalization"
```

## Task 3: Isolate Extension Profiles And Prove Relay Ownership

**Files:**
- Modify: `apps/extension/tests/e2e/utils/extension-build.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-build.test.ts`
- Create: `apps/extension/tests/e2e/utils/skills-certification-relay.ts`
- Create: `apps/extension/tests/e2e/utils/skills-certification-relay.test.ts`

- [ ] **Step 1: Add a failing launcher profile-root test**

Pass `profileRoot` to `launchWithBuiltExtension()` and assert that:

```ts
expect(userDataDir).toMatch(`${profileRoot}/user-data-`)
expect(browserHome).toMatch(`${profileRoot}/home-`)
expect(browserTmp).toBe(`${profileRoot}/tmp`)
expect(crashDumpsDir).toBe(`${profileRoot}/crash-dumps`)
expect(preparedExtensionRoot).toMatch(`${profileRoot}/user-data-`)
expect(chromiumEnv).not.toHaveProperty("OPENAI_API_KEY")
```

The default path and environment behavior remain unchanged for existing tests.
The isolated environment and crash path apply only when `profileRoot` is
supplied.

- [ ] **Step 2: Add failing relay-unit tests**

Test plain request/response doubles for:

1. canonical root and `:name` route normalization;
2. exact worker URL ownership by URL string, not object identity;
3. page-owned requests failing even after a worker request succeeded;
4. request failures and HTTP errors failing;
5. redirects retained but excluded from terminal mutation counts;
6. arbitrary GET counts being accepted;
7. exactly one `POST /api/v1/skills` `201`;
8. exactly one `POST /api/v1/skills/:name/execute` `200` with in-memory `dry_run: true`;
9. exactly two `DELETE /api/v1/skills/:name` `204`;
10. exactly one restore `200` and one purge `204`;
11. any extra successful mutation failing;
12. serialized ledger entries containing only method, normalized path, `worker_owned`, outcome, and optional status.

Explicitly assert the JSON ledger does not contain the API key, arguments, skill content, headers, full URL, or request/response body.

- [ ] **Step 3: Run both tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
bunx vitest run \
  tests/e2e/utils/extension-build.test.ts \
  tests/e2e/utils/skills-certification-relay.test.ts
```

Expected: FAIL for missing `profileRoot` and relay module.

- [ ] **Step 4: Implement the optional profile root**

Add only one option:

```ts
type LaunchOptions = {
  // existing fields
  profileRoot?: string
}
```

Change `makeTempProfileDirs()` to create its `home-*`, `user-data-*`, `tmp`, and
`crash-dumps` directories beneath `profileRoot` when supplied.
`prepareExtensionLaunchPath()` already places the copied extension beneath
`userDataDir`; do not add another copy layer.

For the strict `profileRoot` path only, build Chromium's environment from a
small browser-safe allowlist, override `HOME`, `TMPDIR`, `TMP`, and `TEMP` with
the isolated directories, and replace the literal `/tmp` crash argument with
`--crash-dumps-dir=<profileRoot>/crash-dumps`. This provides defense in depth
even though the runner also allowlists the extension Playwright child. Preserve
the existing environment behavior when `profileRoot` is omitted.

- [ ] **Step 5: Implement the relay observer**

Expose a test-focused API:

```ts
export function createSkillsRelayObserver(
  context: BrowserContext,
  expectedWorkerUrl: string,
): {
  entries: SkillsRelayEntry[]
  assertValid(): void
  dispose(): void
}
```

Use a `WeakMap<Request, SkillsRelayEntry>` to update one entry across `request`, `response`, and `requestfailed`. Inspect execute request JSON only long enough to set an internal dry-run assertion flag; never retain the body. Normalize dynamic names to `:name` labels.

- [ ] **Step 6: Run extension unit tests GREEN and commit**

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
bunx vitest run \
  tests/e2e/utils/extension-build.test.ts \
  tests/e2e/utils/skills-certification-relay.test.ts
```

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/extension/tests/e2e/utils/extension-build.ts \
  apps/extension/tests/e2e/utils/extension-build.test.ts \
  apps/extension/tests/e2e/utils/skills-certification-relay.ts \
  apps/extension/tests/e2e/utils/skills-certification-relay.test.ts
git commit -m "test(extension): prove Skills relay ownership"
```

## Task 4: Add The Shared Lifecycle And Strict WebUI Spec

**Files:**
- Create: `apps/tldw-frontend/e2e/utils/skills-live-certification.ts`
- Create: `apps/tldw-frontend/e2e/skills-certification/playwright.config.ts`
- Create: `apps/tldw-frontend/e2e/skills-certification/skills.live.spec.ts`

- [ ] **Step 1: Implement one shared UI lifecycle helper**

The helper accepts only a `Page`, the calling spec's Playwright `expect`, an
`initialExpectation` of `empty-library-and-trash` or `target-absent`, fixed
skill name, fixed arguments, expected rendered prompt, and a `step()` callback.
Import `Page` and assertion types with `import type`; do not import a runtime
`expect` from the frontend package because the extension owns a separate
Playwright dependency. The helper performs no API setup, route interception,
process work, or file cleanup.

Use these stable synthetic values:

```ts
export const SKILLS_CERT_DESCRIPTION = "Skills live certification fixture"
export const SKILLS_CERT_INSTRUCTIONS = "Organize $ARGUMENTS into verified notes."
export const SKILLS_CERT_ARGUMENTS = "bounded certification input"
export const SKILLS_CERT_RENDERED = "Organize bounded certification input into verified notes."
```

Named phases must cover:

1. for `empty-library-and-trash`, confirm Library empty, open Trash and confirm
   it is empty, then return to Library; for `target-absent`, exact-search the
   supplied name in Library, confirm that target is absent, open Trash and
   confirm that exact target is absent there, then return to Library and clear
   the search;
2. open `New Skill`, fill `Name`, `Description`, and `Instructions`, then save;
3. confirm `Skill created` and the exact name;
4. fill `Search skills` and await a GET list request whose `q` equals the exact name;
5. open `Test run`, submit `Render prompt only`, inspect request/response in memory for `dry_run: true`, and assert the rendered prompt;
6. close the dialog, reload, repeat the exact search, and confirm persistence;
7. move the skill to Trash through `Delete` and `Move to Trash`;
8. open Trash, restore, return to Library, and confirm the exact name;
9. move it to Trash again;
10. confirm `Delete permanently` and the empty Trash state.

- [ ] **Step 2: Add the dedicated WebUI config**

The config must require runner-owned paths from environment variables and set:

```ts
retries: 0
workers: 1
fullyParallel: false
forbidOnly: true
trace: "off"
video: "off"
screenshot: "only-on-failure"
```

Use both `line` and JSON reporters. Place screenshots, JSON, and the per-surface result beneath the runner-provided `webui` evidence directory. Do not declare `webServer`; the runner owns it.

- [ ] **Step 3: Add the one-test real-backend WebUI spec**

The test must:

- require `TLDW_SKILLS_CERT_SKILL_NAME`, server URL, API key, and result path;
- call existing `seedAuth()` before navigation with `allowOffline: false`;
- call the shared lifecycle with
  `initialExpectation: "empty-library-and-trash"`;
- never call `page.route()` or fulfill a response;
- capture uncaught `pageerror` events and failed Skills requests;
- write `{ status: "running" }` at test-body entry so a missing result distinguishes browser launch from workflow failure;
- call the shared lifecycle;
- write a sanitized per-surface result in `finally`;
- fail on page errors or failed Skills requests.

Direct API absence checks remain runner-owned and happen after the browser process exits.

- [ ] **Step 4: Verify the strict config discovers exactly one test**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
TLDW_WEB_URL=http://127.0.0.1:18991 \
TLDW_SERVER_URL=http://127.0.0.1:18992 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT \
TLDW_SKILLS_CERT_SKILL_NAME=skills-cert-web \
TLDW_SKILLS_CERT_WEB_RESULT=/tmp/skills-cert-web-result.json \
TLDW_SKILLS_CERT_WEB_REPORT=/tmp/skills-cert-web-report.json \
TLDW_SKILLS_CERT_WEB_OUTPUT=/tmp/skills-cert-web-output \
bunx playwright test -c e2e/skills-certification/playwright.config.ts --list
```

Expected: exactly one `skills.live.spec.ts` test is listed; no skip branch exists.

- [ ] **Step 5: Run focused lint/type validation and commit**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx eslint \
  e2e/utils/skills-live-certification.ts \
  e2e/skills-certification/playwright.config.ts \
  e2e/skills-certification/skills.live.spec.ts
```

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/tldw-frontend/e2e/utils/skills-live-certification.ts \
  apps/tldw-frontend/e2e/skills-certification/playwright.config.ts \
  apps/tldw-frontend/e2e/skills-certification/skills.live.spec.ts
git commit -m "test(skills): add strict WebUI lifecycle"
```

## Task 5: Add The Strict Packaged-Extension Spec

**Files:**
- Create: `apps/extension/playwright.skills-certification.config.ts`
- Create: `apps/extension/tests/e2e/skills.live-certification.spec.ts`

- [ ] **Step 1: Add the dedicated extension config**

Mirror the WebUI strict settings: one worker, no retries, no trace, no video, failure screenshots, line plus JSON report. Keep the existing extension global setup, but require `TLDW_E2E_SKIP_EXTENSION_BUILD=1` so it only adds the runner-selected backend origin to the built manifest and never starts an untracked build child.

- [ ] **Step 2: Add the one-test strict extension workflow**

The test must not use `installDirectRequestFallback()` from `skills.parity.spec.ts`.

Before final Skills navigation, use `prepareOptionsPage` to:

1. require exactly one worker matching `chrome-extension://<id>/background.js`;
2. create the context-level relay observer;
3. capture page errors;
4. derive the expected worker URL from the observed worker, then confirm it equals the launch result's `extensionId` URL.

Launch with:

```ts
await launchWithBuiltExtension({
  seedConfig: {
    serverUrl,
    authMode: "single-user",
    apiKey,
  },
  allowOffline: false,
  optionsTarget: "/skills",
  profileRoot,
  prepareOptionsPage,
})
```

Import the shared lifecycle from
`../../../tldw-frontend/e2e/utils/skills-live-certification`, pass the extension
package's own runtime `expect`, then run it with `skills-cert-extension`.
Use `initialExpectation: "target-absent"` so a leftover WebUI target cannot
block independent extension evidence.

- [ ] **Step 3: Preserve multiple extension failure categories**

Track the current phase as `extension_launch`, `extension_worker`, `extension_workflow`, or `extension_relay`. In `finally`:

- close the persistent context;
- dispose listeners;
- run relay validation even if the UI workflow failed;
- write the sanitized relay ledger;
- write all applicable categories to the per-surface result;
- throw the original error or an `AggregateError` so Playwright exits nonzero.

No headers, bodies, content, key, or full URL may enter either JSON artifact.

- [ ] **Step 4: Verify exactly one extension test is listed**

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
TLDW_E2E_SKIP_EXTENSION_BUILD=1 \
TLDW_E2E_SERVER_URL=http://127.0.0.1:18992 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT \
TLDW_SKILLS_CERT_SKILL_NAME=skills-cert-extension \
TLDW_SKILLS_CERT_EXTENSION_PROFILE_ROOT=/tmp/skills-cert-extension-profile \
TLDW_SKILLS_CERT_EXTENSION_RESULT=/tmp/skills-cert-extension-result.json \
TLDW_SKILLS_CERT_EXTENSION_REPORT=/tmp/skills-cert-extension-report.json \
TLDW_SKILLS_CERT_EXTENSION_LEDGER=/tmp/skills-cert-extension-ledger.json \
TLDW_SKILLS_CERT_EXTENSION_OUTPUT=/tmp/skills-cert-extension-output \
bunx playwright test -c playwright.skills-certification.config.ts --list
```

Expected: exactly one strict extension test is listed and no skip branch exists.

- [ ] **Step 5: Compile, lint, and commit**

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
bun run compile
cd ../tldw-frontend
bunx eslint \
  ../extension/playwright.skills-certification.config.ts \
  ../extension/tests/e2e/skills.live-certification.spec.ts \
  ../extension/tests/e2e/utils/skills-certification-relay.ts
```

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/extension/playwright.skills-certification.config.ts \
  apps/extension/tests/e2e/skills.live-certification.spec.ts
git commit -m "test(extension): certify live Skills lifecycle"
```

## Task 6: Implement The Thin Runner And Public Command

**Files:**
- Create: `apps/tldw-frontend/scripts/skills-certification/run.mjs`
- Create: `apps/tldw-frontend/scripts/__tests__/skills-certification-runner.test.ts`
- Modify: `apps/tldw-frontend/package.json`
- Create: `Docs/Development/Skills_Live_Integration_Certification.md`

- [ ] **Step 1: Write failing orchestration tests**

Inject a small `operations` object containing the concrete side effects used by `runSkillsCertification()`. Cover:

1. initial Library and Trash must both report `total: 0`;
2. both package-local Chromium probes are tracked, close successfully, and
   fail the run as `preflight` rather than skipping when unavailable;
3. confirmed startup bind conflicts allocate fresh backend/WebUI ports at most three times before browser execution;
4. non-bind startup failures are never retried;
5. URLs cannot change after the first browser child starts;
6. WebUI startup failure still attempts the extension when backend health passes;
7. missing WebUI surface result after a nonzero Playwright exit maps to `webui_launch` and still attempts the extension;
8. WebUI workflow failure maps to `webui_workflow` and still attempts the extension;
9. one same-port backend restart is allowed before extension evidence, but the overall result stays failed;
10. a second crash or failed same-port restart records extension `not_run_infrastructure`;
11. extension build failure maps to `extension_build`;
12. extension result categories are retained without being collapsed;
13. direct detail `404` and Trash exclusion postconditions run after each attempted surface;
14. skipped, flaky, zero-test, missing, or malformed Playwright JSON reports fail;
15. preflight, workflow, postcondition, cleanup, and artifact failures still reach final summary construction;
16. every spawned probe, auth-init, backend, frontend, build, and Playwright child passes through the process registry.

- [ ] **Step 2: Run the runner tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run scripts/__tests__/skills-certification-runner.test.ts
```

Expected: FAIL because `run.mjs` does not exist.

- [ ] **Step 3: Implement the exact orchestration sequence**

`runSkillsCertification()` must execute this bounded state machine:

```text
install SIGINT/SIGTERM handlers
  -> create evidence and disposable profile
  -> run tracked WebUI-package and extension-package Chromium launch probes
  -> run tracked AuthNZ initializer
  -> startup attempt (backend, health, frontend)
       -> retry only a confirmed bind conflict, max 3, before browser execution
  -> direct API check: Library empty and Trash empty
  -> WebUI surface if frontend is ready
  -> WebUI direct API postcondition
  -> backend health check
       -> one same-port evidence restart if the WebUI phase crashed backend
  -> tracked production Chrome build
  -> extension surface if backend is usable
  -> extension direct API postcondition
  -> aggregate both surface reports and Playwright no-skip reports
finally:
  -> stop and verify every registered child
  -> delete runtime root
  -> write summary
  -> scan retained text evidence
  -> delete contaminated evidence
  -> set exit code
```

The binary-only Chromium probes do not navigate to an application and do not
lock endpoint URLs. URL immutability begins when the first WebUI or extension
certification browser child starts.

The extension does not depend on the frontend process. A frontend startup or WebUI browser/workflow failure must not short-circuit extension build/run while backend health remains usable.

The CLI entrypoint must pass every caught error through `redactText()` and a
fixed diagnostic length bound before writing stderr. Artifact-safety failures
print only the generic category because their evidence root has been removed.

- [ ] **Step 4: Implement direct API assertions without sensitive diagnostics**

Use Node `fetch()` with the synthetic `X-API-KEY` header. Error detail contains only a route label and HTTP status.

Initial state:

```text
GET /api/v1/skills/?limit=1&offset=0 -> total === 0
GET /api/v1/skills/trash?limit=1&offset=0 -> total === 0
```

Postcondition for each fixed name:

```text
GET /api/v1/skills/<name> -> 404
GET /api/v1/skills/trash?limit=500&offset=0 -> skills excludes <name>
```

- [ ] **Step 5: Parse Playwright reports and surface results strictly**

Each JSON report must show exactly one executed test, `skipped === 0`, `flaky === 0`, and `unexpected === 0`. Do not shell out to another report parser. Missing result files classify launch failures; present `running`/failed results classify workflow/worker/relay failures according to the surface contract. Bound every retained failure `detail` to 500 redacted characters.

- [ ] **Step 6: Add the public command and short operator documentation**

Add to `apps/tldw-frontend/package.json`:

```json
"e2e:skills:certify": "node scripts/skills-certification/run.mjs"
```

Document only:

- invocation from `apps/tldw-frontend`;
- required local `.venv`, Bun dependencies, and Playwright Chromium;
- tracked package-local Chromium probes that fail preflight rather than skip;
- one disposable backend and production Chrome build;
- evidence location;
- zero-skip and cleanup exit contract;
- no real model/tool execution;
- not part of default PR CI.

- [ ] **Step 7: Run all runner-unit tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run \
  scripts/__tests__/skills-certification-profile.test.ts \
  scripts/__tests__/skills-certification-lifecycle.test.ts \
  scripts/__tests__/skills-certification-evidence.test.ts \
  scripts/__tests__/skills-certification-runner.test.ts
```

- [ ] **Step 8: Commit the runner and command**

```bash
cd "$(git rev-parse --show-toplevel)"
git add \
  apps/tldw-frontend/scripts/skills-certification/run.mjs \
  apps/tldw-frontend/scripts/__tests__/skills-certification-runner.test.ts \
  apps/tldw-frontend/package.json \
  Docs/Development/Skills_Live_Integration_Certification.md
git commit -m "test(skills): add live certification command"
```

## Task 7: Run The Complete Certification And Close The Task

**Files:**
- Update through Backlog MCP/CLI: `TASK-530.15`
- Update during execution: this plan's stage/task checkboxes
- Modify only if a defect is reproduced: the narrow product file and its focused regression test

- [ ] **Step 1: Run focused unit, lint, and compile gates**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx vitest run \
  scripts/__tests__/skills-certification-profile.test.ts \
  scripts/__tests__/skills-certification-lifecycle.test.ts \
  scripts/__tests__/skills-certification-evidence.test.ts \
  scripts/__tests__/skills-certification-runner.test.ts
bunx eslint \
  scripts/skills-certification \
  scripts/__tests__/skills-certification-profile.test.ts \
  scripts/__tests__/skills-certification-lifecycle.test.ts \
  scripts/__tests__/skills-certification-evidence.test.ts \
  scripts/__tests__/skills-certification-runner.test.ts \
  e2e/utils/skills-live-certification.ts \
  e2e/skills-certification
```

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
bunx vitest run \
  tests/e2e/utils/extension-build.test.ts \
  tests/e2e/utils/skills-certification-relay.test.ts
bun run compile
cd ../tldw-frontend
bunx eslint \
  ../extension/playwright.skills-certification.config.ts \
  ../extension/tests/e2e/skills.live-certification.spec.ts \
  ../extension/tests/e2e/utils/extension-build.ts \
  ../extension/tests/e2e/utils/skills-certification-relay.ts
```

- [ ] **Step 2: Run the existing 13 mocked WebUI workflows unchanged**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bunx playwright test \
  e2e/workflows/tier-5-specialized/skills.spec.ts \
  --project=tier-5 \
  --grep "\\(mocked\\)" \
  --workers=1 \
  --reporter=line
```

Expected: 13 passed, zero skipped.

- [ ] **Step 3: Run the existing six extension parity workflows unchanged**

```bash
cd "$(git rev-parse --show-toplevel)/apps/extension"
bun run test:e2e:skills-parity:strict
```

Expected: six passed, zero skipped.

- [ ] **Step 4: Run the new strict live certification**

```bash
cd "$(git rev-parse --show-toplevel)/apps/tldw-frontend"
bun run e2e:skills:certify
```

Expected:

- one WebUI test passed;
- one packaged-extension test passed;
- both direct postconditions passed;
- every extension Skills request was worker-owned;
- terminal mutation counts matched exactly;
- both Playwright reports contain zero skips/flakes/unexpected failures;
- all child processes closed;
- runtime root was deleted;
- retained evidence scan passed;
- command exited 0.

- [ ] **Step 5: Handle a reproduced defect without scope creep**

If the strict gate reproduces a product defect:

1. record the exact synthetic reproduction and category in `TASK-530.15`;
2. invoke `superpowers:systematic-debugging`;
3. add one focused failing regression test in the owning module;
4. implement the smallest product fix;
5. rerun Tasks 7.1 through 7.4;
6. commit the fix separately.

Do not preemptively modify product code or expand the certification lifecycle.

- [ ] **Step 6: Run repository quality gates**

```bash
cd "$(git rev-parse --show-toplevel)"
git diff --check
git status --short --branch
```

No Python files are expected to change, so record Bandit as a scoped frontend/docs-only skip. If any Python file changes due to a reproduced defect, activate `.venv` and run Bandit on exactly the touched Python paths before proceeding.

- [ ] **Step 7: Perform final review and Backlog finalization**

Invoke `superpowers:requesting-code-review`. Address correctness, reliability, security, and overengineering findings before opening the PR. Update `TASK-530.15` through Backlog MCP/CLI with:

- final commits and touched files;
- exact unit, mocked WebUI, extension parity, and strict live results;
- evidence root and sanitized summary outcome;
- Bandit disposition;
- PR URL;
- final summary and status.

The PR remains blocked on the repository's human-written AI `Change summary` policy.

- [ ] **Step 8: Remove this completed implementation plan and commit closeout**

After every stage is complete and its durable verification is recorded in `TASK-530.15`, remove only this task-owned plan file as required by `AGENTS.md`. Keep the approved design spec and operator documentation.

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A -- \
  Docs/superpowers/plans/2026-07-15-skills-live-integration-certification.md \
  backlog/tasks/task-530.15\ -\ Add-strict-Skills-live-integration-certification.md
git commit -m "docs(skills): finalize live certification task"
```

## Acceptance Checklist

- [ ] One explicit `bun run e2e:skills:certify` command owns the full gate.
- [ ] The command is strict when invoked and absent from default PR CI.
- [ ] The backend profile is disposable, single-user, provider-free, and outside normal user data.
- [ ] Initial WebUI Library and Trash are proven empty.
- [ ] Both surfaces complete create, exact search, dry render, reload, Trash, restore, Trash, and purge.
- [ ] Extension continuation is attempted after WebUI startup, browser-launch, and workflow failures whenever backend health allows it.
- [ ] One backend same-port evidence restart cannot turn a failed run into a pass.
- [ ] Every extension Skills request is attributed to the exact MV3 `background.js` URL.
- [ ] No direct page fallback, request failure, unexpected status, or extra terminal mutation is accepted.
- [ ] No trace or video is retained.
- [ ] Every child closes, runtime data is deleted, and retained text evidence passes the leak scan.
- [ ] Existing 13 mocked WebUI and six extension parity workflows remain unchanged and pass.
- [ ] No product behavior changes unless backed by a reproduced strict-gate defect and focused regression test.
