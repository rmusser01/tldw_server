import type { Page } from "@playwright/test"
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from "./smoke.setup"
import { waitForAppShell } from "../utils/helpers"

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT
const UNRESOLVED_TEMPLATE_PATTERN = /\{\{[^{}\n]{1,120}\}\}/g

const prepareChatRoute = async (page: Page) => {
  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ providers: [], any_configured: false })
    })
  })
  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [], total: 0 })
    })
  })
  await seedAuth(page)
}

test.describe("Stage 6 interaction stage 1 defect closures", () => {
  test("chat route does not expose unresolved template placeholders", async ({
    page
  }) => {
    await prepareChatRoute(page)

    await page.goto("/chat", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const input = page.locator("#textarea-message, [data-testid='chat-input']").first()
    await expect(input).toBeVisible({ timeout: LOAD_TIMEOUT })

    const bodyText = await page.evaluate(() => document.body?.innerText || "")
    const unresolvedTemplates = Array.from(bodyText.matchAll(UNRESOLVED_TEMPLATE_PATTERN)).map(
      (match) => match[0]
    )
    const uniqueUnresolvedTemplates = Array.from(new Set(unresolvedTemplates))

    expect(
      uniqueUnresolvedTemplates,
      `Unresolved template placeholders on /chat: ${uniqueUnresolvedTemplates.join(" | ")}`
    ).toHaveLength(0)
    expect(bodyText).not.toContain("{{percentage}}")
  })

  test("home route exposes an explicit theme toggle control", async ({
    page
  }) => {
    await prepareChatRoute(page)

    await page.goto("/", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const toggle = page.getByTestId("chat-header-theme-toggle")
    await expect(toggle).toBeVisible({ timeout: LOAD_TIMEOUT })

    const initialTheme = await page.evaluate(() =>
      document.documentElement.classList.contains("dark") ? "dark" : "light"
    )

    await toggle.click()

    await expect
      .poll(
        async () =>
          page.evaluate(() =>
            document.documentElement.classList.contains("dark") ? "dark" : "light"
          ),
        { timeout: LOAD_TIMEOUT }
      )
      .not.toBe(initialTheme)
  })
})
