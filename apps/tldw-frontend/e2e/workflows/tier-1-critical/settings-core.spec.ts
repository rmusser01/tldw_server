import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { SettingsPage } from "../../utils/page-objects"

// Sections accepted by SettingsPage.gotoSection() (typed union)
const SETTINGS_SECTIONS_VIA_PAGE_OBJECT = [
  "tldw", "model", "chat", "ui", "splash", "quick-ingest",
  "image-generation", "image-gen", "guardian", "prompt", "knowledge",
  "rag", "speech", "evaluations", "characters",
] as const

// Settings pages that exist but are outside the page-object union — use direct URL
const SETTINGS_SECTIONS_DIRECT_NAV = [
  "chatbooks", "world-books", "prompt-studio",
  "share", "about", "family-guardrails",
] as const

const SETTINGS_STANDALONE_PAGES = [
  {
    section: "health",
    heading: /health status/i,
  },
  {
    section: "mcp-hub",
    heading: /^mcp hub$/i,
  },
  {
    section: "processed",
    heading: /processed items \(local\)/i,
  },
] as const

test.describe("Settings", () => {
  let settings: SettingsPage

  test.beforeEach(async ({ authedPage, serverInfo }) => {
    skipIfServerUnavailable(serverInfo)
    settings = new SettingsPage(authedPage)
  })

  // --- Section-load smoke tests (page-object navigation) ---
  for (const section of SETTINGS_SECTIONS_VIA_PAGE_OBJECT) {
    test(`settings/${section} loads without errors`, async ({ authedPage, diagnostics }) => {
      await settings.gotoSection(section)
      await settings.waitForReady()

      // At least one interactive element should be present
      const interactiveElements = authedPage.locator(
        "button, input, select, textarea, a[href]"
      )
      await expect(interactiveElements.first()).toBeVisible({ timeout: 15_000 })
      expect(await interactiveElements.count()).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })
  }

  // --- Section-load smoke tests (direct URL navigation) ---
  for (const section of SETTINGS_SECTIONS_DIRECT_NAV) {
    test(`settings/${section} loads without errors`, async ({ authedPage, diagnostics }) => {
      await authedPage.goto(`/settings/${section}`, { waitUntil: "domcontentloaded" })
      await settings.waitForReady()

      const interactiveElements = authedPage.locator(
        "button, input, select, textarea, a[href]"
      )
      await expect(interactiveElements.first()).toBeVisible({ timeout: 15_000 })
      expect(await interactiveElements.count()).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })
  }

  for (const page of SETTINGS_STANDALONE_PAGES) {
    test(`settings/${page.section} loads without errors`, async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.goto(`/settings/${page.section}`, {
        waitUntil: "domcontentloaded",
      })

      await expect(
        authedPage.getByRole("heading", { name: page.heading }).first()
      ).toBeVisible({ timeout: 20_000 })

      const interactiveElements = authedPage.locator(
        "button, input, select, textarea, a[href]"
      )
      await expect(interactiveElements.first()).toBeVisible({ timeout: 15_000 })
      expect(await interactiveElements.count()).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })
  }

  test.describe("Prompt route intent", () => {
    test("legacy prompt studio route redirects to the prompts studio tab", async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.goto("/prompt-studio", { waitUntil: "domcontentloaded" })

      await expect(authedPage).toHaveURL(/\/prompts\?tab=studio/)
      await expect(
        authedPage.getByRole("heading", { name: /^prompts$/i }).first()
      ).toBeVisible({ timeout: 20_000 })
      await expect(
        authedPage.getByText("Getting started with Prompt Studio")
      ).toBeVisible({ timeout: 20_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("prompt workspace link settings and prompt studio settings stay distinct", async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.goto("/settings/prompt-studio", {
        waitUntil: "domcontentloaded",
      })
      await settings.waitForReady()

      await expect(authedPage).toHaveURL(/\/settings\/prompt-studio/)
      await expect(
        authedPage.getByText(
          "Configure defaults and monitor Prompt Studio health."
        )
      ).toBeVisible({ timeout: 20_000 })
      await expect(
        authedPage.getByRole("button", { name: /test prompt studio/i })
      ).toBeVisible({ timeout: 20_000 })

      await authedPage.goto("/settings/prompt", {
        waitUntil: "domcontentloaded",
      })
      await settings.waitForReady()

      await expect(authedPage).toHaveURL(/\/settings\/prompt(?:\?.*)?$/)
      await expect(
        authedPage.getByRole("heading", { name: "Prompts workspace" })
      ).toBeVisible({ timeout: 20_000 })
      await expect(
        authedPage.getByRole("button", { name: /open prompts workspace/i })
      ).toBeVisible({ timeout: 20_000 })
      await expect(
        authedPage.getByRole("button", { name: /test prompt studio/i })
      ).toHaveCount(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("settings navigation uses different labels for prompt workspace and prompt studio settings", async ({
      diagnostics,
    }) => {
      await settings.goto()
      await settings.waitForReady()

      const promptSettingsLink = settings.page.getByTestId(
        "settings-nav-link--settings-prompt"
      )
      const promptStudioSettingsLink = settings.page.getByTestId(
        "settings-nav-link--settings-prompt-studio"
      )

      await expect(promptSettingsLink).toContainText(/manage prompts/i)
      await expect(promptSettingsLink).not.toContainText(/prompt studio/i)
      await expect(promptStudioSettingsLink).toContainText(/prompt studio/i)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // --- Save button fires an API call ---
  test("save settings fires API", async ({ authedPage, diagnostics }) => {
    await settings.goto()
    await settings.gotoSection("tldw")
    await settings.waitForReady()

    const saveBtn = authedPage.getByRole("button", { name: /save/i }).first()
    if (await saveBtn.isVisible().catch(() => false)) {
      const apiCall = expectApiCall(authedPage, {
        url: "/api/v1/",
      })

      await saveBtn.click()
      const { response } = await apiCall
      expect(response.status()).toBeLessThan(400)
    }

    await assertNoCriticalErrors(diagnostics)
  })
})
