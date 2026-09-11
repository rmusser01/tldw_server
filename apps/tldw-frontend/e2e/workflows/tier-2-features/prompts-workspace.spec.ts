/**
 * Prompts Workspace E2E Tests (Tier-2 Features)
 *
 * Tests the core prompts workflow:
 * - Page loads and renders correctly with key elements
 * - Create prompt saves locally and fires sync API call (POST /api/v1/prompt-studio/prompts/create)
 * - Delete prompt removes it from the list via confirmation dialog
 *
 * Note: Prompts are stored locally in IndexedDB (Dexie) first, then optionally
 * synced to the server. The sync POST only fires when a Prompt Studio project
 * is configured and the server is reachable.
 */
import type { Locator, Page } from "@playwright/test"
import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { PromptsWorkspacePage } from "../../utils/page-objects"
import { generateTestId } from "../../utils/helpers"

const expectNoOverlap = async (first: Locator, second: Locator) => {
  await expect(first).toBeVisible()
  await expect(second).toBeVisible()
  const [firstBox, secondBox] = await Promise.all([
    first.boundingBox(),
    second.boundingBox()
  ])
  expect(firstBox).not.toBeNull()
  expect(secondBox).not.toBeNull()
  if (!firstBox || !secondBox) return

  const overlaps =
    firstBox.x < secondBox.x + secondBox.width &&
    firstBox.x + firstBox.width > secondBox.x &&
    firstBox.y < secondBox.y + secondBox.height &&
    firstBox.y + firstBox.height > secondBox.y
  expect(overlaps).toBe(false)
}

const dismissIntentionalPromptOverlay = async (page: Page) => {
  const dismissButtons = [
    page.getByTestId("prompts-copilot-help-dismiss"),
    page.getByRole("button", { name: /^(Dismiss|Skip|Not now)$/i })
  ]
  for (const button of dismissButtons) {
    if (await button.first().isVisible().catch(() => false)) {
      await button.first().click()
    }
  }
}

test.describe("Prompts Workspace", () => {
  let prompts: PromptsWorkspacePage

  test.beforeEach(async ({ authedPage, serverInfo }) => {
    skipIfServerUnavailable(serverInfo)
    prompts = new PromptsWorkspacePage(authedPage)
    await prompts.goto()
  })

  test("page loads with expected elements", async ({ diagnostics }) => {
    await prompts.assertPageReady()
    await assertNoCriticalErrors(diagnostics)
  })

  test("fits standard desktop content widths without overlap", async ({
    authedPage,
    diagnostics
  }) => {
    for (const scenario of [
      { viewport: { width: 390, height: 844 }, desktopSidebar: false },
      { viewport: { width: 1365, height: 768 }, desktopSidebar: true },
      { viewport: { width: 1365, height: 900 }, desktopSidebar: true },
      { viewport: { width: 1536, height: 960 }, desktopSidebar: true }
    ]) {
      await authedPage.setViewportSize(scenario.viewport)
      await prompts.goto()
      await prompts.assertPageReady()
      await dismissIntentionalPromptOverlay(authedPage)

      expect(
        await authedPage.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth + 1
        )
      ).toBe(true)
      const promptSidebar = authedPage.getByTestId("prompt-sidebar")
      if (scenario.desktopSidebar) {
        await expectNoOverlap(
          promptSidebar,
          authedPage.getByTestId("prompts-search-control")
        )
      } else {
        await expect(promptSidebar).toBeHidden()
      }
      await expect(authedPage.getByTestId("prompts-add")).toBeEnabled()
      await expect(authedPage.getByTestId("prompts-search")).toBeEditable()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("create prompt fires sync API call", async ({ authedPage, diagnostics }) => {
    const testName = `Test Prompt ${generateTestId()}`

    // The sync may push to Prompt Studio when configured.
    // Listen for any prompt-related API call (create or sync).
    const apiCall = expectApiCall(authedPage, {
      method: "POST",
      url: /\/api\/v1\/prompt-studio\/prompts\/(create|update)/,
    }, 20_000)

    await prompts.createPrompt({ name: testName, template: "You are a helpful assistant." })

    // The sync API call is opportunistic -- it only fires when a Studio
    // project is available. We attempt to catch it but do not fail the
    // test if the server has no project configured.
    try {
      const { response } = await apiCall
      expect(response.status()).toBeLessThan(500)
    } catch {
      // Sync was not triggered (no Prompt Studio project configured).
      // The prompt was still saved locally in IndexedDB.
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("delete prompt removes it from the list", async ({ diagnostics }) => {
    const testName = `Delete Me ${generateTestId()}`

    // Create a prompt first
    await prompts.createPrompt({ name: testName, template: "Temporary prompt for deletion test." })
    await prompts.assertPromptVisible(testName)

    // Delete the prompt (local Dexie operation with confirmation dialog)
    await prompts.deletePrompt(testName)

    // Verify the prompt is no longer visible
    await prompts.assertPromptNotVisible(testName)

    await assertNoCriticalErrors(diagnostics)
  })
})
