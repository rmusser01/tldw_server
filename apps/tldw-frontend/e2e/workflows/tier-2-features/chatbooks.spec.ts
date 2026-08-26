/**
 * Chatbooks Backup & Import E2E Tests (Tier 2)
 *
 * Tests the Chatbooks Backup & Import page lifecycle:
 * - Page loads with expected elements (heading, tabs, job tracker)
 * - Tab switching between Export, Import, and Jobs
 * - Export button fires POST /api/v1/chatbooks/export (requires server)
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/chatbooks.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { ChatbooksPage } from "../../utils/page-objects/ChatbooksPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

test.describe("Chatbooks Backup & Import", () => {
  let chatbooks: ChatbooksPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    chatbooks = new ChatbooksPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Chatbooks Backup & Import page with heading and tabs", async ({
      authedPage,
      diagnostics,
    }) => {
      chatbooks = new ChatbooksPage(authedPage)
      await chatbooks.goto()
      await chatbooks.assertPageReady()

      // Either the heading is visible (server online) or the offline message
      const headingVisible = await chatbooks.heading.isVisible().catch(() => false)
      const offlineVisible = await chatbooks.offlineMessage.isVisible().catch(() => false)

      expect(headingVisible || offlineVisible).toBe(true)

      // If online, tabs and job tracker should be present
      if (headingVisible) {
        await expect(chatbooks.exportTab).toBeVisible()
        await expect(chatbooks.importTab).toBeVisible()
        await expect(chatbooks.jobsTab).toBeVisible()
        await expect(chatbooks.jobTrackerCard).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch between tabs without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      chatbooks = new ChatbooksPage(authedPage)
      await chatbooks.goto()
      await chatbooks.assertPageReady()

      // Skip tab switching if page is in offline state
      const headingVisible = await chatbooks.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      for (const tab of ["import", "jobs", "export"] as const) {
        await chatbooks.switchToTab(tab)
        const tabLocator = {
          import: chatbooks.importTab,
          jobs: chatbooks.jobsTab,
          export: chatbooks.exportTab,
        }[tab]
        await expect(tabLocator).toHaveAttribute("aria-selected", "true")
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Import Tab
  // =========================================================================

  test.describe("Import Tab", () => {
    test("should show upload dropzone on Import tab", async ({
      authedPage,
      diagnostics,
    }) => {
      chatbooks = new ChatbooksPage(authedPage)
      await chatbooks.goto()
      await chatbooks.assertPageReady()

      const headingVisible = await chatbooks.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      await chatbooks.switchToTab("import")

      await expect(chatbooks.uploadDropzone).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Export API Integration (requires server)
  // =========================================================================

  test.describe("Export API", () => {
    test("should fire full-account POST /api/v1/chatbooks/export when Backup all is clicked", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      chatbooks = new ChatbooksPage(authedPage)
      await chatbooks.goto()
      await chatbooks.assertPageReady()

      await expect(chatbooks.exportTab).toBeVisible()
      await chatbooks.switchToTab("export")

      await authedPage
        .getByRole("textbox", { name: /^Name$/i })
        .fill(`E2E Backup All ${Date.now()}`)
      await authedPage
        .getByPlaceholder(/Description/i)
        .fill("E2E full-account Backup all export")

      await expect(authedPage.getByText(/Backup all scope/i)).toBeVisible({
        timeout: 15_000,
      })

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/chatbooks\/export/,
        method: "POST",
      }, 15_000)

      await authedPage.getByRole("button", { name: /^Backup all$/i }).click()

      const { request, response } = await apiCall
      const body = request.postDataJSON() as Record<string, unknown>
      expect(body).not.toHaveProperty("content_selections")
      expect(response.status()).toBeLessThan(500)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
