# Legacy API Key Refresh Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve eligible pre-metadata single-user API keys across `/media` refreshes in the WebUI and packaged browser extension without weakening hosted, quickstart, origin-binding, or replacement-auth safeguards.

**Architecture:** Keep the product change inside the shared `TldwApiClient.initialize()` compatibility branch used by both browser surfaces. Classify the exact legacy object shape with own-property checks, reject managed transports and higher-precedence auth, then atomically add the existing manual/device/origin metadata. Extend the current unit and persistence suites; add no new production helper, storage key, UI, or dependency.

**Tech Stack:** TypeScript, React browser clients, Vitest/jsdom, Playwright persistent browser contexts, Chrome MV3 extension storage, Bun.

**Backlog:** TASK-12950

**Design:** `Docs/superpowers/specs/2026-07-12-legacy-api-key-refresh-migration-design.md`

---

## File Map

- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`: tighten the existing legacy credential predicate and write complete current-format metadata.
- Modify `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`: prove the regression and all migration exclusions at the shared initializer boundary.
- Modify `apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts`: return a minimal valid media/OpenAPI response so `/media` can exercise a production-shaped authenticated request.
- Modify `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts`: seed a legacy WebUI record once, hard-refresh `/media`, and verify migration plus authenticated media traffic.
- Modify `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts`: seed extension device storage before first route load, hard-refresh `options.html#/media`, and verify the same behavior in the packaged build.
- Update `backlog/tasks/task-12950 - Preserve-legacy-single-user-API-key-across-media-page-refresh.md`: record touched files, verification, Bandit non-applicability, and final summary.
- Delete this task-specific plan after every stage is complete, as required by the repository instructions.

## Stage 1: Shared Migration Contract

**Goal:** Establish the failing regression and fail-closed boundary, then implement the smallest shared fix.

**Success Criteria:** An eligible bare remote record becomes a complete origin-bound device credential; malformed, placeholder, hosted, quickstart, cookie, environment, and runtime cases do not migrate.

**Tests:** `tldw-api-client.quickstart-auth.test.ts`

**Status:** Not Started

### Task 1: Write the shared initializer regression tests

**Files:**

- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts:1-340`

- [ ] **Step 1: Reset runtime auth state in the fixture lifecycle**

Extend the existing runtime-auth import and reset calls:

```ts
import {
  activateCookieSessionConfig,
  clearRuntimeAuthOverride,
  isCookieSessionConfigInvalidated,
  setRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"

beforeEach(() => {
  // existing resets
  clearRuntimeAuthOverride()
})

afterEach(() => {
  // existing resets
  clearRuntimeAuthOverride()
})
```

- [ ] **Step 2: Replace the incorrect bare-record scrub expectation**

Rename the existing ambiguous-record test and assert the complete migrated object, including `authSource`:

```ts
it("migrates an eligible pre-metadata key to an origin-bound device credential", async () => {
  mocks.storage.set("tldwConfig", {
    authMode: "single-user",
    serverUrl: "https://api.example.test/path",
    apiKey: "legacy-device-secret"
  })

  const client = new TldwApiClient()
  await client.initialize()

  expect(mocks.storage.get("tldwConfig")).toEqual({
    authMode: "single-user",
    authSource: "manual",
    serverUrl: "https://api.example.test/path",
    apiKey: "legacy-device-secret",
    credentialSource: "manual",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: "https://api.example.test"
  })
  await expect(client.ensureConfigForRequest(true)).resolves.toMatchObject({
    apiKey: "legacy-device-secret"
  })
})
```

- [ ] **Step 3: Add own-property and source rejection cases**

Add focused tests proving that `credentialSource: ""`, `apiKeyPersistence: ""`, `apiKeyServerOrigin: ""`, `authSource: ""`, and `authSource: "cookie-session"` are present malformed/contradictory fields, not absent legacy metadata. Each test initializes a record with `authSource: "manual"` where applicable and asserts the persisted/effective config has no `apiKey`.

Use a table for the three metadata properties:

```ts
it.each([
  ["credentialSource", { credentialSource: "" }],
  ["apiKeyPersistence", { apiKeyPersistence: "" }],
  ["apiKeyServerOrigin", { apiKeyServerOrigin: "" }]
])("scrubs a legacy key when %s is present but empty", async (_name, patch) => {
  mocks.storage.set("tldwConfig", {
    authMode: "single-user",
    authSource: "manual",
    serverUrl: "https://api.example.test",
    apiKey: "must-be-scrubbed",
    ...patch
  })

  const client = new TldwApiClient()
  await client.initialize()

  expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
})
```

- [ ] **Step 4: Add placeholder and transport exclusion cases**

Add one test each for:

- `apiKey: "REPLACE-ME"` with otherwise migratable fields.
- An unparseable `serverUrl` such as `"not a URL"` with otherwise migratable fields.
- `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=hosted` with `authSource: "manual"`.
- `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`, a same-origin `serverUrl`, no cookie marker, and `authSource: "manual"`.

Each must assert device storage and effective config contain no legacy key.

- [ ] **Step 5: Add replacement-auth precedence cases**

Cover the sources that must scrub rather than migrate the stored bare key:

```ts
// Environment replacement
process.env.NEXT_PUBLIC_X_API_KEY = "active-environment-key"

// Runtime replacement
setRuntimeSingleUserApiKeyOverride("active-runtime-key")

// Cookie replacement in quickstart mode
mocks.storage.set("tldwCookieSessionConfig", {
  authMode: "single-user",
  authSource: "cookie-session",
  serverUrl: window.location.origin
})
```

For environment/runtime cases, assert the persisted `tldwConfig` is scrubbed while `ensureConfigForRequest(true)` resolves with the active replacement. For the cookie case, assert the effective config is cookie-session auth and contains no API key.

- [ ] **Step 6: Run the focused test and verify RED**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
```

Expected: FAIL for the eligible bare record, present-empty metadata, placeholder, hosted, and quickstart cases because the current predicate either scrubs the real legacy record or migrates excluded records.

### Task 2: Implement the minimal shared classifier

**Files:**

- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts:1820-1862`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`

- [ ] **Step 1: Replace truthiness with exact legacy-shape checks**

Inside the existing `if (storedManual)` block, keep the logic local and derive:

```ts
const hasOwn = (key: keyof TldwConfig): boolean =>
  Object.prototype.hasOwnProperty.call(storedManual, key)
const hasNoCredentialMetadata =
  !hasOwn("credentialSource") &&
  !hasOwn("apiKeyPersistence") &&
  !hasOwn("apiKeyServerOrigin")
const hasLegacyAuthSource =
  !hasOwn("authSource") || storedManual.authSource === "manual"
```

Do not extract a new production helper: the predicate is used once and belongs beside the migration write.

- [ ] **Step 2: Tighten `hasLegacyManualKey`**

The predicate must require all reviewed guards:

```ts
const hasLegacyManualKey =
  !isHostedTldwDeployment() &&
  !quickstartWebUiServerUrl &&
  storedManual.authMode === "single-user" &&
  typeof storedManual.apiKey === "string" &&
  Boolean(storedManual.apiKey.trim()) &&
  !isPlaceholderApiKey(storedManual.apiKey) &&
  hasNoCredentialMetadata &&
  hasLegacyAuthSource &&
  Boolean(origin) &&
  !activeCookieSession &&
  !envApiKey &&
  !getRuntimeSingleUserApiKeyOverride()
```

- [ ] **Step 3: Complete the atomic metadata rewrite**

Add the missing `authSource` field to the existing rewrite:

```ts
persistedManual = {
  ...storedManual,
  authSource: "manual",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: origin
}
```

Leave the existing incomplete-key scrub branch and storage-failure behavior unchanged.

- [ ] **Step 4: Run the focused test and verify GREEN**

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
```

Expected: all tests pass.

- [ ] **Step 5: Run adjacent credential policy tests**

```bash
bunx vitest run \
  src/services/__tests__/tldw-api-client.connection-sync.test.ts \
  src/services/tldw/__tests__/single-user-credential.test.ts
```

Expected: all tests pass; sync-to-local migration and current-format credential rules remain unchanged.

- [ ] **Step 6: Commit Stage 1**

```bash
git add \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
git commit -m "fix(web): migrate eligible legacy API keys"
```

## Stage 2: WebUI Media Refresh Regression

**Goal:** Exercise the reported `/media` refresh flow in a real advanced-mode WebUI browser.

**Success Criteria:** A bare legacy record is migrated once, survives hard refresh as current-format metadata, and authenticates a media-list request.

**Tests:** `manual-api-key-persistence.spec.ts`

**Status:** Not Started

### Task 3: Extend the shared browser fixture and WebUI persistence suite

**Files:**

- Modify: `apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts:1-80`
- Modify: `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts:1-190`

- [ ] **Step 1: Give the fixture production-shaped media capabilities**

Keep auth enforcement unchanged. Add response bodies for:

```ts
pathname === "/openapi.json"
  ? {
      openapi: "3.1.0",
      paths: {
        "/api/v1/media/": { get: {} },
        "/api/v1/media/search": { post: {} }
      }
    }
  : pathname === "/api/v1/media"
    ? {
        items: [],
        pagination: {
          page: 1,
          results_per_page: 20,
          total_items: 0,
          total_pages: 0
        }
      }
    : /* existing bodies */
```

- [ ] **Step 2: Add a one-time legacy seed**

In a new persistent profile, use `page.addInitScript` with a sentinel so only the first document writes the old record and reload sees the migrated record:

```ts
await page.addInitScript(({ serverUrl, apiKey }) => {
  if (localStorage.getItem("__legacy_api_key_seeded")) return
  localStorage.setItem("__legacy_api_key_seeded", "true")
  localStorage.setItem("__tldw_first_run_complete", "true")
  localStorage.setItem("tldw_skip_landing_hub", "true")
  localStorage.setItem(
    "tldwConfig",
    JSON.stringify({
      authMode: "single-user",
      serverUrl,
      apiKey
    })
  )
}, { serverUrl: fixture.url, apiKey: MANUAL_API_KEY })
```

- [ ] **Step 3: Add the `/media` hard-refresh assertion**

Navigate to `${WEB_URL}/media`, wait for an authenticated request whose path begins `/api/v1/media`, capture a new request offset, hard reload, and require another authenticated media request after that offset. Assert the page does not show `Add your credentials to use Media` and persisted config exactly matches the complete manual/device/origin fields.

- [ ] **Step 4: Run the WebUI regression explicitly in advanced mode**

From `apps/tldw-frontend`:

```bash
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
TLDW_WEB_URL=http://localhost:18084 \
TLDW_WEB_CMD='bun run dev:webpack -- -p 18084' \
npx playwright test e2e/manual-api-key-persistence.spec.ts \
  --project=chromium --reporter=line
```

Expected: all device, session, and legacy `/media` persistence tests pass.

- [ ] **Step 5: Commit Stage 2**

```bash
git add \
  apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts \
  apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts
git commit -m "test(web): cover legacy key media refresh"
```

## Stage 3: Packaged Extension Media Refresh Regression

**Goal:** Prove the same migration through Chrome extension device storage and the packaged MV3 runtime.

**Success Criteria:** `options.html#/media` remains authenticated before and after reload, and extension local storage contains complete migrated metadata.

**Tests:** `extension-api-key-persistence.spec.ts`

**Status:** Not Started

### Task 4: Seed and verify legacy extension storage

**Files:**

- Modify: `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts:1-370`

- [ ] **Step 1: Add a worker-backed storage seed helper**

Use the already-running extension service worker before any options route is opened:

```ts
const seedLegacyDeviceConfig = async (
  context: BrowserContext,
  serverUrl: string
): Promise<void> => {
  const worker = context.serviceWorkers()[0]
  if (!worker) throw new Error("Extension service worker is unavailable")
  await worker.evaluate(
    ({ url, apiKey }) =>
      new Promise<void>((resolve) => {
        chrome.storage.local.set(
          {
            tldwConfig: {
              authMode: "single-user",
              serverUrl: url,
              apiKey
            }
          },
          () => resolve()
        )
      }),
    { url: serverUrl, apiKey: MANUAL_API_KEY }
  )
}
```

- [ ] **Step 2: Add the packaged `/media` reload test**

Launch a fresh extension/profile, seed storage, then navigate directly to `chrome-extension://${extensionId}/options.html#/media`. Wait for an authenticated `/api/v1/media` request, record `const requestOffset = fixture.requests().length` immediately before reload, reload, and require an authenticated media request only within `fixture.requests().slice(requestOffset)`. This prevents duplicate pre-reload requests from satisfying the post-refresh assertion. Assert the credential prompt is absent and `chrome.storage.local.tldwConfig` has `authSource: "manual"`, `credentialSource: "manual"`, `apiKeyPersistence: "device"`, the normalized fixture origin, and the original key.

- [ ] **Step 3: Build the extension explicitly for advanced/self-hosted transport**

From `apps/extension`:

```bash
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced bun run build:chrome:prod
```

Expected: Chrome MV3 production build succeeds. Existing duplicate-import/circular-chunk warnings may remain but no new error is allowed.

- [ ] **Step 4: Run the complete packaged-extension persistence suite**

From `apps/tldw-frontend`:

```bash
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
TLDW_EXTENSION_PATH=../extension/.output/chrome-mv3 \
npx playwright test e2e/extension-api-key-persistence.spec.ts \
  --project=chromium --reporter=line
```

Expected: device, session, and legacy `/media` tests pass.

- [ ] **Step 5: Commit Stage 3**

```bash
git add apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts
git commit -m "test(extension): cover legacy key media refresh"
```

## Stage 4: Verification, Review, and Task Finalization

**Goal:** Confirm no adjacent auth behavior regressed and leave the task/branch ready for human integration.

**Success Criteria:** Focused and adjacent tests, browser suites, type checks, lint, build, and diff checks pass; review finds no blocking issue; TASK-12950 is fully documented.

**Tests:** All commands below.

**Status:** Not Started

### Task 5: Run the release-quality verification matrix

**Files:**

- Verify all modified TypeScript and test files.
- Update: `backlog/tasks/task-12950 - Preserve-legacy-single-user-API-key-across-media-page-refresh.md`
- Delete after completion: `Docs/superpowers/plans/2026-07-12-legacy-api-key-refresh-migration-implementation-plan.md`

- [ ] **Step 1: Run the shared auth unit matrix**

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/tldw-api-client.quickstart-auth.test.ts \
  src/services/__tests__/tldw-api-client.connection-sync.test.ts \
  src/services/tldw/__tests__/single-user-credential.test.ts
```

- [ ] **Step 2: Run both complete persistence browser suites**

Run the Stage 2 WebUI command and Stage 3 extension build/test commands again without `--grep`.

- [ ] **Step 3: Run static validation**

```bash
cd apps/tldw-frontend
bun run typecheck

cd ../extension
bun run compile

cd ../..
bunx eslint \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts \
  apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts \
  apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts \
  apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts

git diff --check origin/dev...HEAD
```

Expected: every command exits zero. Fix only failures caused by this task; document unrelated baseline failures rather than broadening scope.

- [ ] **Step 4: Perform required code review and verification checks**

Invoke `superpowers:requesting-code-review`, address any correctness/security findings, then invoke `superpowers:verification-before-completion` and rerun any command affected by review changes.

- [ ] **Step 5: Record the security validation result**

Bandit is not applicable because the implementation and tests touch TypeScript only. Record the explicit skip in TASK-12950; do not run Python Bandit against unrelated code.

- [ ] **Step 6: Finalize TASK-12950**

Using the Backlog.md MCP workflow:

- Mark acceptance criteria complete only from fresh verification evidence.
- Add every touched file.
- Record exact test/build/lint commands and results.
- Add the final summary and Bandit non-applicability note.
- Set status to Done only when no required work remains.

- [ ] **Step 7: Remove this completed plan and commit final records**

Update all stage statuses before deleting this task-specific plan per `AGENTS.md`, then commit the task finalization and plan deletion:

```bash
git add \
  'backlog/tasks/task-12950 - Preserve-legacy-single-user-API-key-across-media-page-refresh.md' \
  Docs/superpowers/plans/2026-07-12-legacy-api-key-refresh-migration-implementation-plan.md
git commit -m "docs: finalize legacy API key migration"
```

Do not create or merge a PR without the requester’s separate authorization. If a materially AI-authored PR is later requested, remind the requester that the repository’s human-written `Change summary` merge gate applies.
