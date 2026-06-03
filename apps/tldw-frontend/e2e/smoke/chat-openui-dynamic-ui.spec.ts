import { expect, test, type Page } from "@playwright/test"

const bypassChatGates = async (page: Page) => {
  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [] }),
    })
  })
  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ providers: [] }),
    })
  })

  await page.addInitScript(() => {
    const authConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
    }

    window.localStorage.setItem("assistant_setup_dismissed", "true")
    window.localStorage.setItem("__tldw_first_run_complete", "true")
    window.localStorage.setItem("__tldw_test_bypass", "true")
    window.localStorage.setItem("tldwConfig", JSON.stringify(authConfig))
    window.localStorage.setItem("apiKey", authConfig.apiKey)
    window.localStorage.setItem("authMode", authConfig.authMode)
  })
}

test("chat exposes OpenUI dynamic UI request mode control", async ({ page }) => {
  test.setTimeout(90_000)

  await bypassChatGates(page)
  await page.goto("/chat", { waitUntil: "domcontentloaded" })

  await expect(page.getByRole("button", { name: /OpenUI/i })).toBeVisible({
    timeout: 30_000,
  })
})
