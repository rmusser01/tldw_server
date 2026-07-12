# Remote API-Key Device Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give manually configured remote single-user WebUI and extension users an explicit, origin-bound “Remember on this device” choice that survives the requested lifecycle without relying on a password manager.

**Architecture:** Keep explicit manual device credentials atomically in the existing `tldwConfig` record with complete source/persistence/origin metadata so current background and streaming consumers remain compatible. Store session-only credentials separately in browser session storage, hydrate them into transient request config, and never persist or reclassify runtime cookie-session credentials. Validate candidate origins with only the freshly submitted key before committing an ordered origin transition.

**Tech Stack:** TypeScript, React, Ant Design form controls, Plasmo/WXT storage adapters, Vitest, Playwright persistent contexts.

## Global Constraints

- Execute only after TASK-12108 is complete; cookie-session runtime auth always wins for loopback quickstart.
- `Remember on this device` is visible for manual single-user setup and defaults enabled only for new manual entries.
- A device key is valid only when `tldwConfig` atomically contains `source="manual"`, `persistence="device"`, and a normalized matching `serverOrigin`.
- A session key is never written to `tldwConfig`, local storage, or `browser.storage.local`; extension session storage uses `browser.storage.session`.
- Runtime/cookie credentials are never copied into manual storage.
- Candidate-server validation sends only the key typed for that candidate and never calls the shared stored-credential resolver.
- Origin transitions commit in this order: successful candidate probe, clear old device/session secrets, write the chosen scope, publish connection metadata/in-memory config.
- Device-write failure falls back to session storage, then memory, with accurate UI text; session writes never fall forward to persistent storage.
- Add no dependency and no client-side encryption wrapper.

---

### Task 1: Credential Metadata and Storage Policy

**Files:**
- Create: `apps/packages/ui/src/services/tldw/single-user-credential.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/tldw-frontend/extension/shims/plasmo-storage.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts`
- Test: `apps/tldw-frontend/__tests__/extension/plasmo-storage.test.ts`

**Interfaces:**
- Consumes: existing `TldwConfig`, `Storage`, and WebUI storage shim.
- Produces: `ApiKeyPersistence`, `ManualCredentialMetadata`, `normalizeServerOrigin()`, `resolveManualCredential()`, `toPersistedTldwConfig()`, `clearManualCredentials()`.

- [x] **Step 1: Write failing policy tests**

```ts
it("accepts a complete device credential only for its exact origin", async () => {
  const config = {
    authMode: "single-user",
    serverUrl: "https://api.example.test/v1",
    apiKey: "secret",
    credentialSource: "manual",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: "https://api.example.test"
  } satisfies TldwConfig
  expect(await resolveManualCredential(config, stores)).toBe("secret")
  expect(await resolveManualCredential({ ...config, serverUrl: "https://other.test" }, stores)).toBeNull()
})

it("strips runtime, session, and incomplete keys from persisted tldwConfig", () => {
  expect(toPersistedTldwConfig({ ...base, apiKey: "runtime", credentialSource: "cookie-session" })).not.toHaveProperty("apiKey")
  expect(toPersistedTldwConfig({ ...base, apiKey: "session", apiKeyPersistence: "session" })).not.toHaveProperty("apiKey")
  expect(toPersistedTldwConfig({ ...base, apiKey: "ambiguous" })).not.toHaveProperty("apiKey")
})

it("maps WebUI session storage to window.sessionStorage", async () => {
  const session = new Storage({ area: "session" })
  await session.set("tldwManualSessionApiKey", { apiKey: "secret" })
  expect(window.sessionStorage.getItem("tldwManualSessionApiKey")).toContain("secret")
  expect(window.localStorage.getItem("tldwManualSessionApiKey")).toBeNull()
})
```

- [x] **Step 2: Run tests and confirm failures**

Run: `cd apps && bunx vitest run packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts tldw-frontend/__tests__/extension/plasmo-storage.test.ts`

Expected: policy module is missing and the WebUI shim does not provide true session scope.

- [x] **Step 3: Implement the minimal policy**

```ts
export type ApiKeyPersistence = "device" | "session"
export type CredentialSource = "manual" | "cookie-session"

export const MANUAL_SESSION_KEY = "tldwManualSessionApiKey"

export const normalizeServerOrigin = (value: string): string | null => {
  try {
    const url = new URL(value)
    return /^https?:$/.test(url.protocol) ? url.origin : null
  } catch {
    return null
  }
}

export const isCompleteDeviceCredential = (config: TldwConfig): boolean =>
  config.credentialSource === "manual" &&
  config.apiKeyPersistence === "device" &&
  Boolean(config.apiKey) &&
  normalizeServerOrigin(config.serverUrl) === config.apiKeyServerOrigin

export const toPersistedTldwConfig = (config: TldwConfig): TldwConfig => {
  if (isCompleteDeviceCredential(config)) return { ...config }
  const { apiKey: _secret, ...safe } = config
  return safe as TldwConfig
}
```

Implement session-record validation with `credentialSource="manual"`, `apiKeyPersistence="session"`, and matching origin. Extend `TldwConfig` with the three optional metadata fields. Update the WebUI shim so `area: "session"` reads/writes/removes `window.sessionStorage`; preserve extension-native `browser.storage.session` behavior.

- [x] **Step 4: Run policy tests**

Run: `cd apps && bunx vitest run packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts tldw-frontend/__tests__/extension/plasmo-storage.test.ts`

Expected: all selected tests pass.

- [x] **Step 5: Commit the policy**

```bash
git add apps/packages/ui/src/services/tldw/single-user-credential.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/tldw-frontend/extension/shims/plasmo-storage.ts apps/packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts apps/tldw-frontend/__tests__/extension/plasmo-storage.test.ts
git commit -m "feat(web): define manual API-key persistence policy"
```

### Task 2: Migration, Hydration, Save, and Clear Semantics

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwAuth.ts`
- Modify: `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`
- Test: `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

**Interfaces:**
- Consumes: Task 1 storage policy.
- Produces: `saveManualSingleUserCredential()`, `hydrateManualSingleUserCredential()`, `clearManualSingleUserCredentials()`, idempotent legacy migration.

- [x] **Step 1: Write failing lifecycle and migration unit tests**

```ts
it("persists an explicit device choice atomically", async () => {
  await client.saveManualSingleUserCredential({
    serverUrl: "https://api.example.test",
    apiKey: "secret",
    persistence: "device"
  })
  expect(await local.get("tldwConfig")).toMatchObject({
    apiKey: "secret",
    credentialSource: "manual",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: "https://api.example.test"
  })
  expect(await session.get(MANUAL_SESSION_KEY)).toBeNull()
})

it("keeps a session choice out of persistent config", async () => {
  await client.saveManualSingleUserCredential({
    serverUrl: "https://api.example.test",
    apiKey: "secret",
    persistence: "session"
  })
  expect(await local.get("tldwConfig")).not.toHaveProperty("apiKey")
  expect(await session.get(MANUAL_SESSION_KEY)).toMatchObject({ apiKey: "secret" })
})

it("does not migrate an ambiguous legacy bridge", async () => {
  await seedLegacy({ configApiKey: "secret", ownership: null, serverUrl: "https://api.example.test" })
  await client.initialize()
  expect(await local.get("tldwConfig")).not.toHaveProperty("apiKey")
  expect(await session.get(MANUAL_SESSION_KEY)).toBeNull()
})
```

- [x] **Step 2: Run tests and confirm failures**

Run: `cd apps && bunx vitest run packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

Expected: current config writes cannot distinguish device/session/runtime ownership.

- [x] **Step 3: Implement ordered writes and hydration**

```ts
async saveManualSingleUserCredential(input: {
  serverUrl: string
  apiKey: string
  persistence: ApiKeyPersistence
}): Promise<"device" | "session" | "memory"> {
  const serverOrigin = normalizeServerOrigin(input.serverUrl)
  if (!serverOrigin) throw new Error("Invalid server URL")
  await clearManualSingleUserCredentials(this.storage, this.sessionStorage)
  if (input.persistence === "device") {
    try {
      await this.storage.set("tldwConfig", {
        ...toConnectionMetadata(input.serverUrl),
        apiKey: input.apiKey,
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: serverOrigin
      })
      return "device"
    } catch {
      return await this.writeSessionOrMemory(input, serverOrigin)
    }
  }
  return await this.writeSessionOrMemory(input, serverOrigin)
}
```

Hydrate the session record into `this.config` without persisting it. Migrate a legacy persistent key only when the stored config is single-user, URL origin is valid, runtime/cookie auth is absent, and ownership is confidently manual; otherwise delete the ambiguous key. Make logout, disconnect, reset, auth-mode change, and origin change clear device metadata, session storage, and in-memory overrides. Do not clear on network/5xx responses.

- [x] **Step 4: Run client persistence regression tests**

Run: `cd apps && bunx vitest run packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

Expected: all selected tests pass.

- [x] **Step 5: Commit persistence behavior**

```bash
git add apps/packages/ui/src/services/tldw apps/tldw-frontend/extension/shims/runtime-bootstrap.ts apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
git commit -m "feat(web): persist manual API keys by selected scope"
```

### Task 3: Candidate-Origin Probe and Transactional Transition

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwAuth.ts`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/validation.ts`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/tldw.tsx`
- Test: `apps/packages/ui/src/services/tldw/__tests__/TldwAuth.api-key-origin.test.ts`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

**Interfaces:**
- Consumes: Task 2 save/clear methods.
- Produces: `testApiKey(serverUrl, apiKey)` as an unaffiliated candidate probe and `commitManualServerTransition()`.

- [x] **Step 1: Write failing origin-transition tests**

```ts
it("candidate probe sends only the submitted key", async () => {
  await auth.testApiKey("https://new.example.test", "new-key")
  expect(fetch).toHaveBeenCalledWith(
    "https://new.example.test/api/v1/users/me/profile",
    expect.objectContaining({ headers: { "X-API-KEY": "new-key" } })
  )
  expect(fetch.mock.calls[0][1].headers).not.toHaveProperty("Authorization")
})

it("keeps the old configuration when the new-origin probe fails", async () => {
  await expect(connectForm.save(candidate)).rejects.toThrow()
  expect(await local.get("tldwConfig")).toEqual(oldConfig)
  expect(await session.get(MANUAL_SESSION_KEY)).toEqual(oldSession)
})
```

- [x] **Step 2: Run tests and confirm failures**

Run: `cd apps && bunx vitest run packages/ui/src/services/tldw/__tests__/TldwAuth.api-key-origin.test.ts packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

Expected: form save publishes config before a fully isolated origin transition is established.

- [x] **Step 3: Implement explicit probe then ordered commit**

```ts
async testApiKey(serverUrl: string, apiKey: string): Promise<boolean> {
  const validationUrl = new URL("/api/v1/users/me/profile", `${serverUrl.replace(/\/+$/, "")}/`)
  const response = await fetch(validationUrl, {
    method: "GET",
    headers: { "X-API-KEY": apiKey },
    credentials: "omit",
    signal: AbortSignal.timeout(API_KEY_VALIDATION_TIMEOUT_MS)
  })
  return response.ok
}

async function commitManualServerTransition(input: ManualConnectionInput) {
  if (!(await tldwAuth.testApiKey(input.serverUrl, input.apiKey))) {
    throw new Error("API key validation failed")
  }
  const achieved = await tldwClient.saveManualSingleUserCredential(input)
  await tldwClient.publishHydratedConfig()
  return achieved
}
```

When a populated form changes to a different valid origin, clear its visible key field. Preserve it for path/trailing-slash changes on the same origin. Keep invalid URLs from invoking a probe.

- [x] **Step 4: Run origin transition tests**

Run: `cd apps && bunx vitest run packages/ui/src/services/tldw/__tests__/TldwAuth.api-key-origin.test.ts packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

Expected: all selected tests pass.

- [x] **Step 5: Commit safe origin transitions**

```bash
git add apps/packages/ui/src/services/tldw/TldwAuth.ts apps/packages/ui/src/components/Option/Onboarding apps/packages/ui/src/components/Option/Settings
git commit -m "fix(web): bind manual API keys to server origin"
```

### Task 4: Remember-Control UX in Onboarding and Settings

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/TldwConnectionSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/tldw.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Modify: `apps/packages/ui/src/public/_locales/en/settings.json`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

**Interfaces:**
- Consumes: Task 2 persistence methods and Task 3 commit flow.
- Produces: visible accessible `rememberApiKey` form field and truthful fallback warning.

- [x] **Step 1: Write failing component tests**

```tsx
it("defaults remember on for a new manual single-user setup", async () => {
  render(<OnboardingConnectForm />)
  expect(await screen.findByRole("checkbox", { name: "Remember on this device" })).toBeChecked()
  expect(screen.getByText(/Stores this API key in this browser/)).toBeVisible()
})

it("renders session-only copy when unchecked", async () => {
  const user = userEvent.setup()
  render(<TldwConnectionSettings {...props} />)
  await user.click(screen.getByRole("checkbox", { name: "Remember on this device" }))
  expect(screen.getByText("Keep signed in until this browser closes.")).toBeVisible()
})

it("hides manual controls for cookie-session runtime auth", () => {
  render(<OnboardingConnectForm initialConfig={{ authSource: "cookie-session" }} />)
  expect(screen.queryByLabelText("Paste your API key")).not.toBeInTheDocument()
  expect(screen.queryByRole("checkbox", { name: "Remember on this device" })).not.toBeInTheDocument()
  expect(screen.getByText("Connected securely through this WebUI.")).toBeVisible()
})
```

- [x] **Step 2: Run component tests and confirm failures**

Run: `cd apps && bunx vitest run packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

Expected: remember checkbox and cookie-session state are absent.

- [x] **Step 3: Add the inline checkbox and fallback messaging**

```tsx
<Form.Item name="rememberApiKey" valuePropName="checked" initialValue>
  <Checkbox>Remember on this device</Checkbox>
</Form.Item>
<Typography.Text type="secondary">
  {rememberApiKey
    ? "Stores this API key in this browser until you disconnect or clear browser data. Turn this off on a shared device."
    : "Keep signed in until this browser closes."}
</Typography.Text>
```

Map checked to `persistence: "device"` and unchecked to `persistence: "session"`. When device storage falls back, show `Couldn’t remember the key on this device; it will be kept until this browser closes.` When session storage also fails, show `This key is available only on this page and will be lost on reload.` Use the existing checkbox, typography, alert, focus, and form-validation components.

- [x] **Step 4: Run component and accessibility tests**

Run: `cd apps && bunx vitest run packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx packages/ui/src/design-system/__tests__/proof-surface-static-guard.test.ts`

Expected: all selected tests pass.

- [x] **Step 5: Commit the UX**

```bash
git add apps/packages/ui/src/components/Option/Onboarding apps/packages/ui/src/components/Option/Settings apps/packages/ui/src/assets/locale/en/settings.json apps/packages/ui/src/public/_locales/en/settings.json
git commit -m "feat(web): add manual API-key remember choice"
```

### Task 5: Browser and Extension Relaunch Coverage

**Files:**
- Create: `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts`
- Create: `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts`
- Create: `apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts`
- Modify: `apps/tldw-frontend/playwright.config.ts`
- Modify: `apps/packages/ui/src/i18n/index.ts`
- Modify: `apps/packages/ui/src/routes/app-route.tsx`
- Modify: tldw config readers under `apps/packages/ui/src/` to use extension-local storage
- Modify: `Dockerfiles/README.md`
- Modify: `backlog/tasks/task-12106 - Add-explicit-single-user-API-key-device-persistence-and-relaunch-coverage.md`

**Interfaces:**
- Consumes: real onboarding/settings UI and persistence policy.
- Produces: hard-reload, browser-process relaunch, and extension-installation relaunch evidence.

- [x] **Step 1: Add failing persistent-profile tests**

```ts
test("manual device save survives hard reload and browser relaunch", async ({}, testInfo) => {
  const profile = testInfo.outputPath("web-profile")
  await withPersistentChromium(profile, async page => {
    await saveManualConnectionThroughUi(page, { remember: true })
    await page.reload({ waitUntil: "networkidle" })
    await expectAuthenticated(page)
  })
  await withPersistentChromium(profile, async page => {
    await page.goto(webUiUrl)
    await expectAuthenticated(page)
  })
})

test("manual session save survives reload but not browser relaunch", async ({}, testInfo) => {
  const profile = testInfo.outputPath("session-profile")
  await withPersistentChromium(profile, async page => {
    await saveManualConnectionThroughUi(page, { remember: false })
    await page.reload({ waitUntil: "networkidle" })
    await expectAuthenticated(page)
  })
  await withPersistentChromium(profile, async page => {
    await page.goto(webUiUrl)
    await expect(page.getByLabel("Paste your API key")).toHaveValue("")
  })
})
```

The extension test launches Chromium twice with the same `userDataDir` and unpacked extension path, saves through the real extension UI, records the extension ID, and verifies the same ID plus authenticated UI after relaunch. It separately confirms session storage is empty after relaunch.

- [x] **Step 2: Run lifecycle tests and confirm failures**

Run WebUI: `cd apps/tldw-frontend && bunx playwright test e2e/manual-api-key-persistence.spec.ts --reporter=line`

Run extension: `cd apps/tldw-frontend && bunx playwright test e2e/extension-api-key-persistence.spec.ts --reporter=line`

Expected: relaunch cases fail before the final persistence wiring is complete.

- [x] **Step 3: Add only the harness configuration required by the tests**

```ts
export const launchPersistentExtension = (userDataDir: string, extensionPath: string) =>
  chromium.launchPersistentContext(userDataDir, {
    headless: false,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`
    ]
  })
```

Keep the unpacked path identical across launches and derive the extension ID from the service worker URL instead of hard-coding it. Do not seed `tldwConfig`; exercise the UI save flow.

- [x] **Step 4: Run final verification**

Run unit/component: `cd apps && bunx vitest run packages/ui/src/services/tldw/__tests__ packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

Run WebUI lifecycle: `cd apps/tldw-frontend && bunx playwright test e2e/manual-api-key-persistence.spec.ts --reporter=line`

Run extension lifecycle: `cd apps/tldw-frontend && bunx playwright test e2e/extension-api-key-persistence.spec.ts --reporter=line`

Run tracked-secret check: `git diff --check && rg -n "secret-test-key|THIS-IS-A-SECURE-KEY" apps/tldw-frontend/test-results apps/tldw-frontend/playwright-report 2>/dev/null`

Expected: tests pass; the final grep produces no output from generated artifacts.

- [x] **Step 5: Finalize and commit TASK-12106**

Record exact commands/results, checked acceptance criteria, browser/extension versions, persistence fallbacks, known skips, and final summary in Backlog.md. Bandit is not applicable to the TypeScript-only task; record that scoped skip. Then run:

```bash
git add apps/tldw-frontend/e2e apps/tldw-frontend/playwright.config.ts Dockerfiles/README.md "backlog/tasks/task-12106 - Add-explicit-single-user-API-key-device-persistence-and-relaunch-coverage.md"
git commit -m "test(web): cover API-key persistence lifecycle"
```
