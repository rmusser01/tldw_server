import {
  chromium,
  expect,
  test,
  type BrowserContext,
  type Page
} from "@playwright/test"

import {
  MANUAL_API_KEY,
  startManualApiKeyFixture,
  type ManualApiKeyFixture
} from "./helpers/manual-api-key-fixture"

const WEB_URL = (process.env.TLDW_WEB_URL || "http://localhost:8080").replace(
  /\/$/,
  ""
)
const API_PORT = Number(process.env.TLDW_MANUAL_PERSISTENCE_API_PORT || "19041")

const withPersistentBrowser = async <T>(
  userDataDir: string,
  callback: (context: BrowserContext, page: Page) => Promise<T>
): Promise<T> => {
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: true,
    baseURL: WEB_URL,
    locale: "en-US"
  })
  const page = context.pages()[0] || (await context.newPage())
  try {
    return await callback(context, page)
  } finally {
    await context.close()
  }
}

const openSettings = async (page: Page): Promise<void> => {
  await page.goto(`${WEB_URL}/login`, { waitUntil: "domcontentloaded" })
  await expect(page.getByText("tldw Server Configuration")).toBeVisible()
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
  await page.getByRole("button", { name: "Save", exact: true }).click()
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("tldwConfig")))
    .not.toBeNull()
}

const expectProductionRagRequest = async (
  page: Page,
  fixture: ManualApiKeyFixture,
  authenticated: boolean
): Promise<void> => {
  const requestOffset = fixture.requests().length
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

test.describe.serial("manual WebUI API-key persistence", () => {
  let fixture: ManualApiKeyFixture

  test.beforeAll(async () => {
    fixture = await startManualApiKeyFixture(API_PORT)
  })

  test.afterAll(async () => {
    await fixture?.close()
  })

  test("device save survives hard reload and reopening the same browser profile", async ({ browserName: _browserName }, testInfo) => {
    const profile = testInfo.outputPath("device-profile")

    await withPersistentBrowser(profile, async (_context, page) => {
      await openSettings(page)
      await saveManualConnection(page, fixture.url, true)
      await page.reload({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("textbox", { name: /API Key$/ })).toHaveValue(
        MANUAL_API_KEY
      )
      await expect(
        page.getByRole("checkbox", { name: "Remember on this device" })
      ).toBeChecked()
      await expectProductionRagRequest(page, fixture, true)
    })

    await withPersistentBrowser(profile, async (_context, page) => {
      await openSettings(page)
      await expect(page.getByRole("textbox", { name: /API Key$/ })).toHaveValue(
        MANUAL_API_KEY
      )
      const config = await page.evaluate(() =>
        JSON.parse(localStorage.getItem("tldwConfig") || "null")
      )
      expect(config).toMatchObject({
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKey: MANUAL_API_KEY,
        apiKeyServerOrigin: fixture.url
      })
      await expectProductionRagRequest(page, fixture, true)
    })
  })

  test("legacy device key authenticates media after hard reload", async ({ browserName: _browserName }, testInfo) => {
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

    await withPersistentBrowser(profile, async (_context, page) => {
      await page.addInitScript(
        ({ serverUrl, apiKey }) => {
          if (localStorage.getItem("__legacy_api_key_seeded")) return
          localStorage.setItem("__legacy_api_key_seeded", "true")
          localStorage.setItem("__tldw_first_run_complete", "true")
          localStorage.setItem("tldw_skip_landing_hub", "true")
          localStorage.setItem("assistant_setup_dismissed", "true")
          // Legacy WebUI saves mirrored the selected server for bootstrap.
          localStorage.setItem("tldw-api-host", serverUrl)
          localStorage.setItem(
            "tldwConfig",
            JSON.stringify({
              authMode: "single-user",
              serverUrl,
              apiKey
            })
          )
        },
        { serverUrl: fixture.url, apiKey: MANUAL_API_KEY }
      )

      await page.goto(`${WEB_URL}/media`, { waitUntil: "domcontentloaded" })
      await expect
        .poll(() =>
          fixture
            .requests()
            .some(
              (request) =>
                request.path.startsWith("/api/v1/media") &&
                request.authenticated
            )
        )
        .toBe(true)
      await expect(
        page.getByText("Add your credentials to use Media", { exact: true })
      ).toHaveCount(0)
      expect(
        await page.evaluate(() =>
          JSON.parse(localStorage.getItem("tldwConfig") || "null")
        )
      ).toEqual(expectedConfig)

      const requestOffset = fixture.requests().length
      await page.reload({ waitUntil: "domcontentloaded" })
      await expect
        .poll(() =>
          fixture
            .requests()
            .slice(requestOffset)
            .some(
              (request) =>
                request.path.startsWith("/api/v1/media") &&
                request.authenticated
            )
        )
        .toBe(true)
      await expect(
        page.getByText("Add your credentials to use Media", { exact: true })
      ).toHaveCount(0)
      expect(
        await page.evaluate(() =>
          JSON.parse(localStorage.getItem("tldwConfig") || "null")
        )
      ).toEqual(expectedConfig)
    })
  })

  test("session save survives reload but not reopening the browser profile", async ({ browserName: _browserName }, testInfo) => {
    const profile = testInfo.outputPath("session-profile")

    await withPersistentBrowser(profile, async (_context, page) => {
      await openSettings(page)
      await saveManualConnection(page, fixture.url, false)
      expect(
        await page.evaluate(() => localStorage.getItem("tldwConfig"))
      ).not.toContain(MANUAL_API_KEY)
      expect(
        await page.evaluate(() => sessionStorage.getItem("tldwManualSessionApiKey"))
      ).toContain(MANUAL_API_KEY)

      await page.reload({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("textbox", { name: /API Key$/ })).toHaveValue(
        MANUAL_API_KEY
      )
      await expect(
        page.getByRole("checkbox", { name: "Remember on this device" })
      ).not.toBeChecked()
      await expectProductionRagRequest(page, fixture, true)
    })

    await withPersistentBrowser(profile, async (_context, page) => {
      await openSettings(page)
      await expect(page.getByRole("textbox", { name: /API Key$/ })).toHaveValue("")
      expect(
        await page.evaluate(() => sessionStorage.getItem("tldwManualSessionApiKey"))
      ).toBeNull()
      expect(
        await page.evaluate(() => localStorage.getItem("tldwConfig"))
      ).not.toContain(MANUAL_API_KEY)
      await expectProductionRagRequest(page, fixture, false)
    })
  })
})
