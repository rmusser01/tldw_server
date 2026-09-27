import {
  chromium,
  expect,
  test,
  type BrowserContext,
  type Page
} from "@playwright/test"
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync
} from "node:fs"
import path from "node:path"

import {
  MANUAL_API_KEY,
  startManualApiKeyFixture,
  type ManualApiKeyFixture
} from "./helpers/manual-api-key-fixture"

const API_PORT = Number(
  process.env.TLDW_EXTENSION_PERSISTENCE_API_PORT || "19042"
)
const extensionProject = path.resolve(__dirname, "..", "..", "extension")

const resolveBuiltExtension = (): string => {
  const explicit = String(process.env.TLDW_EXTENSION_PATH || "").trim()
  const candidates = [
    explicit,
    path.join(extensionProject, "build", "chrome-mv3"),
    path.join(extensionProject, ".output", "chrome-mv3")
  ].filter(Boolean)
  const found = candidates.find((candidate) =>
    existsSync(path.join(candidate, "manifest.json"))
  )
  if (!found) {
    throw new Error(
      `No built extension found. Run bun run build:chrome:prod in ${extensionProject}`
    )
  }
  return found
}

const prepareExtension = (
  source: string,
  destination: string,
  apiUrl: string
): string => {
  cpSync(source, destination, { recursive: true })
  const manifestPath = path.join(destination, "manifest.json")
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
    host_permissions?: string[]
  }
  manifest.host_permissions = Array.from(
    new Set([...(manifest.host_permissions || []), `${apiUrl}/*`])
  )
  writeFileSync(manifestPath, JSON.stringify(manifest))
  return destination
}

const getExtensionServiceWorker = async (context: BrowserContext) =>
  context.serviceWorkers()[0] ||
  (await context.waitForEvent("serviceworker", { timeout: 15_000 }))

const resolveExtensionId = async (context: BrowserContext): Promise<string> => {
  const target = await getExtensionServiceWorker(context)
  const match = target.url().match(/^chrome-extension:\/\/([a-p]{32})\//)
  if (!match) throw new Error(`Could not derive extension id from ${target.url()}`)
  return match[1]
}

const setExtensionStorage = async (
  context: BrowserContext,
  area: "local" | "sync",
  values: Record<string, unknown>
): Promise<void> => {
  const worker = await getExtensionServiceWorker(context)
  await worker.evaluate(
    ({ storageArea, storageValues }) =>
      new Promise<void>((resolve, reject) => {
        chrome.storage[storageArea].set(storageValues, () => {
          const error = chrome.runtime.lastError
          if (error) {
            reject(new Error(error.message))
            return
          }
          resolve()
        })
      }),
    { storageArea: area, storageValues: values }
  )
}

const launchExtension = async (
  userDataDir: string,
  extensionPath: string,
  seedStartupState = true
): Promise<{ context: BrowserContext; page: Page; extensionId: string }> => {
  mkdirSync(path.join(userDataDir, "home"), { recursive: true })
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chromium",
    headless: true,
    locale: "en-US",
    ignoreDefaultArgs: ["--disable-extensions"],
    env: {
      ...process.env,
      HOME: path.join(userDataDir, "home")
    },
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      "--no-crashpad",
      "--disable-crash-reporter",
      "--crash-dumps-dir=/tmp"
    ]
  })
  try {
    const extensionId = await resolveExtensionId(context)
    if (seedStartupState) {
      await setExtensionStorage(context, "local", {
        __e2eSeeded: true,
        __tldw_first_run_complete: true,
        tldw_skip_landing_hub: true
      })
    }
    const page = context.pages()[0] || (await context.newPage())
    return { context, page, extensionId }
  } catch (startupError) {
    try {
      await context.close()
    } catch (cleanupError) {
      throw new AggregateError(
        [startupError, cleanupError],
        "Extension startup and cleanup both failed"
      )
    }
    throw startupError
  }
}

const openSettings = async (page: Page, extensionId: string): Promise<void> => {
  const browserErrors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      browserErrors.push(`console:${message.type()}:${message.text()}`)
    }
  })
  page.on("pageerror", (error) => browserErrors.push(`pageerror:${error.message}`))
  page.on("requestfailed", (request) =>
    browserErrors.push(
      `requestfailed:${request.url()}:${request.failure()?.errorText || "unknown"}`
    )
  )
  await page.goto(
    `chrome-extension://${extensionId}/options.html#/settings/tldw`,
    { waitUntil: "domcontentloaded" }
  )
  try {
    await expect(page.getByText("tldw Server Configuration")).toBeVisible({
      timeout: 60_000
    })
  } catch (error) {
    const diagnostics = await page.evaluate(() => ({
      url: location.href,
      text: document.body.innerText.slice(0, 2_000),
      readyState: document.readyState,
      localStorageKeys: Object.keys(localStorage)
    }))
    throw new Error(
      `Extension settings startup failed: ${JSON.stringify({ diagnostics, browserErrors })}`,
      { cause: error }
    )
  }
}

const saveManualConnection = async (
  page: Page,
  serverUrl: string,
  remember: boolean
): Promise<void> => {
  await page.getByLabel("Server URL").fill(serverUrl)
  await page.getByRole("textbox", { name: /API Key$/ }).fill(MANUAL_API_KEY)
  const rememberControl = page.getByRole("checkbox", {
    name: "Remember on this device"
  })
  if ((await rememberControl.isChecked()) !== remember) {
    await rememberControl.click()
  }
  await page.getByRole("button", { name: /^save$/i }).click()
  const storageArea = remember ? "local" : "session"
  const storageKey = remember ? "tldwConfig" : "tldwManualSessionApiKey"
  await expect
    .poll(async () =>
      JSON.stringify(
        await page.evaluate(
          ({ area, key }) =>
            new Promise<unknown>((resolve) => {
              chrome.storage[area].get(key, (items) => resolve(items[key]))
            }),
          { area: storageArea, key: storageKey }
        )
      )
    )
    .toContain(MANUAL_API_KEY)
}

const extensionStorageValue = async (
  page: Page,
  area: "local" | "session" | "sync",
  key: string
): Promise<unknown> =>
  page.evaluate(
    ({ storageArea, storageKey }) =>
      new Promise<unknown>((resolve) => {
        chrome.storage[storageArea].get(storageKey, (items) =>
          resolve(items[storageKey])
        )
      }),
    { storageArea: area, storageKey: key }
  )

const normalizeExtensionStorageValue = (value: unknown): unknown =>
  typeof value === "string" ? JSON.parse(value) : value

const seedLegacyDeviceConfig = async (
  context: BrowserContext,
  serverUrl: string
): Promise<void> => {
  await setExtensionStorage(context, "sync", {
    tldwConfig: JSON.stringify({
      authMode: "single-user",
      serverUrl,
      apiKey: MANUAL_API_KEY
    })
  })
}

const hasAuthenticatedMediaListRequest = (
  fixture: ManualApiKeyFixture,
  offset: number
): boolean =>
  fixture
    .requests()
    .slice(offset)
    .some(
      (request) =>
        request.method === "GET" &&
        request.path === "/api/v1/media" &&
        request.authenticated === true
    )

const expectProductionRagRequest = async (
  page: Page,
  fixture: ManualApiKeyFixture,
  authenticated: boolean
): Promise<void> => {
  const requestOffset = fixture.requests().length
  await page.evaluate(
    async ({ origin }) => {
      await new Promise<void>((resolve) => {
        chrome.storage.local.set(
          {
            tldwCookieSessionConfig: {
              serverUrl: origin,
              authMode: "single-user",
              authSource: "cookie-session"
            }
          },
          () => resolve()
        )
      })
    },
    { origin: fixture.url }
  )
  await page
    .getByRole("button", { name: /^(Recheck|Test Connection)$/ })
    .first()
    .click()

  if (authenticated) {
    await expect
      .poll(() =>
        fixture
          .requests()
          .slice(requestOffset)
          .some(
            (request) =>
              request.path === "/api/v1/rag/health" && request.authenticated
          )
      )
      .toBe(true)
    return
  }

  await expect(
    page.getByText("RAG: needs attention", { exact: true })
  ).toBeVisible()
  expect(
    fixture
      .requests()
      .slice(requestOffset)
      .some(
        (request) =>
          request.path === "/api/v1/rag/health" && request.authenticated
      )
  ).toBe(false)
}

test.describe.serial("manual extension API-key persistence", () => {
  test.describe.configure({ timeout: 180_000 })
  let fixture: ManualApiKeyFixture

  test.beforeAll(async () => {
    fixture = await startManualApiKeyFixture(API_PORT)
  })

  test.afterAll(async () => {
    await fixture?.close()
  })

  test("device save survives reopening the same extension installation and profile", async ({ browserName: _browserName }, testInfo) => {
    const extensionPath = prepareExtension(
      resolveBuiltExtension(),
      testInfo.outputPath("extension-device"),
      fixture.url
    )
    const profile = testInfo.outputPath("device-profile")
    let originalExtensionId = ""

    {
      const { context, page, extensionId } = await launchExtension(
        profile,
        extensionPath
      )
      originalExtensionId = extensionId
      try {
        await openSettings(page, extensionId)
        await saveManualConnection(page, fixture.url, true)
        expect(
          JSON.stringify(
            await extensionStorageValue(page, "local", "tldwConfig")
          )
        ).toContain(MANUAL_API_KEY)
        await expectProductionRagRequest(page, fixture, true)
      } finally {
        await context.close()
      }
    }

    {
      const { context, page, extensionId } = await launchExtension(
        profile,
        extensionPath,
        false
      )
      try {
        expect(extensionId).toBe(originalExtensionId)
        await openSettings(page, extensionId)
        await expect(
          page.getByRole("textbox", { name: /API Key$/ })
        ).toHaveValue(MANUAL_API_KEY)
        await expect(
          page.getByRole("checkbox", { name: "Remember on this device" })
        ).toBeChecked()
        await expectProductionRagRequest(page, fixture, true)
      } finally {
        await context.close()
      }
    }
  })

  test("legacy device key authenticates media after extension reload", async ({
    browserName: _browserName
  }, testInfo) => {
    const extensionPath = prepareExtension(
      resolveBuiltExtension(),
      testInfo.outputPath("extension-legacy-media"),
      fixture.url
    )
    const profile = testInfo.outputPath("legacy-media-profile")
    const expectedConfig = {
      authMode: "single-user",
      authSource: "manual",
      serverUrl: fixture.url,
      apiKey: MANUAL_API_KEY,
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: fixture.url
    }
    const { context, page, extensionId } = await launchExtension(
      profile,
      extensionPath
    )

    try {
      await seedLegacyDeviceConfig(context, fixture.url)
      await page.addInitScript(() => {
        localStorage.setItem("assistant_setup_dismissed", "true")
      })

      const initialRequestOffset = fixture.requests().length
      await page.goto(`chrome-extension://${extensionId}/options.html#/media`, {
        waitUntil: "domcontentloaded"
      })
      await expect
        .poll(() =>
          hasAuthenticatedMediaListRequest(fixture, initialRequestOffset)
        )
        .toBe(true)
      await expect(
        page.getByText("Add your credentials to use Media", { exact: true })
      ).toHaveCount(0)
      expect(
        normalizeExtensionStorageValue(
          await extensionStorageValue(page, "local", "tldwConfig")
        )
      ).toEqual(expectedConfig)
      expect(
        await extensionStorageValue(page, "sync", "tldwConfig")
      ).toBeUndefined()

      const reloadRequestOffset = fixture.requests().length
      await page.reload({ waitUntil: "domcontentloaded" })
      await expect
        .poll(() =>
          hasAuthenticatedMediaListRequest(fixture, reloadRequestOffset)
        )
        .toBe(true)
      await expect(
        page.getByText("Add your credentials to use Media", { exact: true })
      ).toHaveCount(0)
      expect(
        normalizeExtensionStorageValue(
          await extensionStorageValue(page, "local", "tldwConfig")
        )
      ).toEqual(expectedConfig)
      expect(
        await extensionStorageValue(page, "sync", "tldwConfig")
      ).toBeUndefined()
    } finally {
      await context.close()
    }
  })

  test("session save is cleared when the extension browser session restarts", async ({ browserName: _browserName }, testInfo) => {
    const extensionPath = prepareExtension(
      resolveBuiltExtension(),
      testInfo.outputPath("extension-session"),
      fixture.url
    )
    const profile = testInfo.outputPath("session-profile")
    let originalExtensionId = ""

    {
      const { context, page, extensionId } = await launchExtension(
        profile,
        extensionPath
      )
      originalExtensionId = extensionId
      try {
        await openSettings(page, extensionId)
        await saveManualConnection(page, fixture.url, false)
        expect(
          JSON.stringify(
            await extensionStorageValue(page, "local", "tldwConfig")
          )
        ).not.toContain(MANUAL_API_KEY)
        expect(
          JSON.stringify(
            await extensionStorageValue(
              page,
              "session",
              "tldwManualSessionApiKey"
            )
          )
        ).toContain(MANUAL_API_KEY)
        await page.reload({ waitUntil: "domcontentloaded" })
        await expect(page.getByText("tldw Server Configuration")).toBeVisible({
          timeout: 60_000
        })
        await expectProductionRagRequest(page, fixture, true)
      } finally {
        await context.close()
      }
    }

    {
      const { context, page, extensionId } = await launchExtension(
        profile,
        extensionPath,
        false
      )
      try {
        expect(extensionId).toBe(originalExtensionId)
        await openSettings(page, extensionId)
        await expect(
          page.getByRole("textbox", { name: /API Key$/ })
        ).toHaveValue("")
        expect(
          await extensionStorageValue(
            page,
            "session",
            "tldwManualSessionApiKey"
          )
        ).toBeUndefined()
        await expectProductionRagRequest(page, fixture, false)
      } finally {
        await context.close()
      }
    }
  })
})
