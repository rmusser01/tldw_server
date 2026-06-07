import { test, expect, type Page, type Route } from "@playwright/test"

import { TEST_CONFIG } from "../utils/helpers"

const seedConfiguredAuthWithoutReadinessBypass = async (page: Page) => {
  await page.addInitScript((cfg) => {
    const authConfig = {
      serverUrl: cfg.serverUrl,
      authMode: "single-user",
      apiKey: cfg.apiKey,
    }

    try {
      localStorage.setItem("tldwConfig", JSON.stringify(authConfig))
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("serverUrl", cfg.serverUrl)
      localStorage.setItem("tldwServerUrl", cfg.serverUrl)
      localStorage.setItem("tldw-api-host", cfg.serverUrl)
      localStorage.setItem("authMode", "single-user")
      localStorage.setItem("apiKey", cfg.apiKey)
      localStorage.removeItem("__tldw_allow_offline")
      localStorage.removeItem("__tldw_test_bypass")
    } catch {
      // Storage seeding is best-effort in browser contexts.
    }
  }, TEST_CONFIG)
}

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

const expectKnowledgeRecovery = async (page: Page) => {
  await expect(
    page.getByRole("heading", { name: /Backend readiness check failed/i })
  ).toBeVisible({ timeout: 20_000 })
  const recovery = page.getByTestId("server-readiness-recovery")
  await expect(recovery).toContainText(
    "http://127.0.0.1:8000/api/v1/health"
  )
  await expect(recovery.getByRole("button", { name: "Retry" })).toBeVisible()
  await expect(recovery.getByRole("button", { name: "Health & diagnostics" })).toBeVisible()
  await expect(recovery.getByRole("button", { name: "Server settings" })).toBeVisible()
  await expect(page.getByTestId("server-readiness-route-content")).toBeAttached()
  await expect
    .poll(async () => {
      const text = await recovery.textContent()
      return text?.trim().length ?? 0
    })
    .toBeGreaterThan(0)
}

test.describe("Knowledge QA readiness recovery", () => {
  test("WebUI /knowledge shows recovery after failed backend health", async ({ page }) => {
    await seedConfiguredAuthWithoutReadinessBypass(page)
    await page.route("**/api/v1/health**", async (route) => {
      await fulfillJson(route, { status: "unavailable" }, 503)
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })

    await expectKnowledgeRecovery(page)
  })

  test("WebUI /knowledge shows recovery after stalled backend health", async ({ page }) => {
    await seedConfiguredAuthWithoutReadinessBypass(page)
    await page.route("**/api/v1/health**", async () => {
      await new Promise(() => undefined)
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })

    await expectKnowledgeRecovery(page)
  })
})
