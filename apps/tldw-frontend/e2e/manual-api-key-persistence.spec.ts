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

test.describe.serial("manual WebUI API-key persistence", () => {
  let fixture: ManualApiKeyFixture

  test.beforeAll(async () => {
    fixture = await startManualApiKeyFixture(API_PORT)
  })

  test.afterAll(async () => {
    await fixture?.close()
  })

  test("device save survives hard reload and reopening the same browser profile", async ({}, testInfo) => {
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
    })
  })

  test("session save survives reload but not reopening the browser profile", async ({}, testInfo) => {
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
    })
  })
})
