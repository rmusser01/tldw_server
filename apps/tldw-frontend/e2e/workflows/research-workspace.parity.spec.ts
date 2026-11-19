import { test, expect, assertNoCriticalErrors } from "../utils/fixtures"
import { runResearchWorkspaceParityContract } from "../../../test-utils/research-workspace"
import { ResearchWorkspaceParityPage } from "../../../test-utils/research-workspace/page"
import { ResearchWorkspacePage } from "../utils/page-objects"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

const EMPTY_LIST_RESPONSE = {
  status: 200,
  contentType: "application/json",
  body: JSON.stringify({ items: [], total: 0 })
}

test.describe("Research Workspace parity (WebUI)", () => {
  test.beforeEach(async ({ authedPage }) => {
    await authedPage.setViewportSize(DESKTOP_VIEWPORT)
    await authedPage.route("**/api/v1/llm/models/metadata**", route =>
      route.fulfill({ json: { models: [] } })
    )
    await authedPage.route("**/api/v1/llm/providers**", route =>
      route.fulfill({ json: { providers: [] } })
    )

    await authedPage.route("**/api/v1/chats**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chats/**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chat/conversations**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chat/conversations/**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })
  })

  test("passes baseline + deterministic studio parity contract", async ({
    authedPage,
    diagnostics
  }) => {
    await runResearchWorkspaceParityContract({
      platform: "web",
      page: authedPage
    })

    await assertNoCriticalErrors(diagnostics)
  })

  test("opens an output discussion when Studio is the first pane visited", async ({ authedPage }) => {
    const page = authedPage
    const notebook = new ResearchWorkspaceParityPage(page)
    await page.setViewportSize({ width: 900, height: 844 })
    await page.addInitScript(() => localStorage.setItem("assistant_setup_dismissed", "true"))
    await page.goto("/research-workspace?tab=studio", { waitUntil: "domcontentloaded" })
    await expect(notebook.workspacesButton).toBeVisible({ timeout: 45_000 })
    await expect(page.getByRole("tab", { name: /^Studio/i })).toHaveAttribute("aria-selected", "true")
    await expect(notebook.generatedOutputsToggle).toBeVisible()
    await notebook.seedDeterministicArtifact()
    await notebook.getParityArtifactCard().getByRole("button", { name: /discuss in chat/i }).click()
    await expect(page.getByRole("tab", { name: /^Chat/i })).toHaveAttribute("aria-selected", "true")
    await notebook.expectComposerVisibleWithoutPageScroll()
  })

  test("keeps notebook actions reachable as the workspace changes width", async ({ authedPage }, testInfo) => {
    const page = authedPage
    const notebook = new ResearchWorkspaceParityPage(page)
    const app = new ResearchWorkspacePage(page)
    await page.setViewportSize({ width: 390, height: 844 })
    await notebook.goto("web")
    await notebook.waitForReady()

    // Intake is owned by Sources, even when Chat is the only pane visited.
    await page.getByTestId("workspace-chat-add-source-button").click()
    await expect(app.addSourceModal).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(app.addSourceModal).toBeHidden()

    await app.seedSources([
      { mediaId: 27, title: "Field observations", type: "pdf" },
      { mediaId: 28, title: "Interview transcript", type: "text" }
    ])
    for (const width of [900, 390]) {
      await page.setViewportSize({ width, height: 844 })
      const sources = page.getByRole("tab", { name: /^Sources/i })
      await sources.click()
      await expect(app.getSourceRowByTitle("Field observations")).toBeVisible()
      await app.selectSourceByTitle("Field observations")
      const organizer = page.getByRole("button", { name: /organize sources/i })
      await organizer.click()
      await expect(organizer).toHaveAttribute("aria-expanded", "true")
      await organizer.click()
      await page.getByRole("tab", { name: /^Studio/i }).click()
      await expect(notebook.generatedOutputsToggle).toBeVisible()
      if (width === 900) await notebook.seedDeterministicArtifact()
      await notebook.expectParityArtifactVisible()
      await page.getByRole("tab", { name: /^Chat/i }).click()
      await notebook.expectComposerVisibleWithoutPageScroll()
      const fits = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
      expect(fits).toBe(true)
    }
    await page.screenshot({ path: testInfo.outputPath("phone.png") })
    await page.setViewportSize(DESKTOP_VIEWPORT)
    await notebook.assertBaselinePanesVisible()
    await page.getByTestId("workspace-server-context-indicator").click()
    await expect(page.getByText("Server Workspace", { exact: true })).toBeVisible()
    await notebook.headerTitle.click()
    await expect(page.getByText("Server Workspace", { exact: true })).toBeHidden()
    await page.screenshot({ path: testInfo.outputPath("desktop.png") })
  })
})
