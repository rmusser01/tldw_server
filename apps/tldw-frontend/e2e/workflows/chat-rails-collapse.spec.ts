import { expect, test, type Page } from "@playwright/test"

import { seedAuth } from "../smoke/smoke.setup"
import { waitForAppShell } from "../utils/helpers"

const artifactFixture = {
  id: "artifact-rail-e2e",
  title: "Rail artifact",
  content: "value",
  kind: "code",
  language: "text"
}

const prepareChatRailPage = async (page: Page) => {
  await seedAuth(page)
  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [] })
    })
  })
  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ providers: [] })
    })
  })
  await page.addInitScript(() => {
    window.localStorage.setItem("ff_chatSidebar", "true")
    window.localStorage.setItem("stickyChatInput", "true")
    window.localStorage.setItem("playgroundComposerOptionsExpanded", "false")
  })
}

const openArtifactPanel = async (page: Page) => {
  await page.waitForFunction(() =>
    Boolean((window as any).__tldw_useArtifactsStore),
  )
  await page.evaluate((artifact) => {
    ;(window as any).__tldw_useArtifactsStore.getState().openArtifact(artifact)
  }, artifactFixture)
}

const closeVisibleArtifactPanel = async (page: Page) => {
  const closeButton = page.locator('[data-testid="artifacts-panel-close"]:visible')
  await expect(closeButton).toHaveCount(1)
  await closeButton.click()
}

const visibleArtifactPanel = (page: Page) =>
  page.locator('[data-testid="artifacts-panel"]:visible')

test.describe("/chat siderail collapse", () => {
  test("desktop collapsed rails expose same-side edge buttons and release width", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 960 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)

    const chatShell = page.getByTestId("playground-chat-shell")
    const composer = page.getByTestId("playground-chat-composer-dock")
    await expect(chatShell).toBeVisible()
    await expect(composer).toBeVisible()

    const leftEdge = page.getByTestId("chat-sidebar-edge-expand")
    await expect(leftEdge).toBeVisible()
    const leftCollapsedBox = await chatShell.boundingBox()
    expect(leftCollapsedBox).not.toBeNull()

    await leftEdge.click()
    await expect(page.getByTestId("chat-sidebar")).toBeVisible()
    const leftExpandedBox = await chatShell.boundingBox()
    expect(leftExpandedBox).not.toBeNull()
    expect(leftExpandedBox!.width).toBeLessThan(leftCollapsedBox!.width)

    const expandedTop = leftExpandedBox!.y
    await page.getByTestId("chat-sidebar-toggle").click()
    await expect(leftEdge).toBeVisible()
    const leftRecollapsedBox = await chatShell.boundingBox()
    expect(leftRecollapsedBox).not.toBeNull()
    expect(leftRecollapsedBox!.width).toBeGreaterThan(leftExpandedBox!.width)
    expect(Math.abs(leftRecollapsedBox!.y - expandedTop)).toBeLessThanOrEqual(2)

    await openArtifactPanel(page)
    const artifactPanel = visibleArtifactPanel(page)
    await expect(artifactPanel).toHaveCount(1)
    await expect(artifactPanel).toBeVisible()
    const rightOpenBox = await chatShell.boundingBox()
    expect(rightOpenBox).not.toBeNull()
    await closeVisibleArtifactPanel(page)

    const rightEdge = page.getByTestId("playground-artifacts-edge-expand")
    await expect(rightEdge).toBeVisible()
    const rightClosedBox = await chatShell.boundingBox()
    expect(rightClosedBox).not.toBeNull()
    expect(rightClosedBox!.width).toBeGreaterThan(rightOpenBox!.width)

    const composerBox = await composer.boundingBox()
    expect(composerBox).not.toBeNull()
    expect(
      Math.abs(960 - (composerBox!.y + composerBox!.height))
    ).toBeLessThanOrEqual(12)
  })

  test("medium and mobile viewports do not expose desktop edge buttons", async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 900 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
    await expect(page.getByTestId("chat-sidebar-edge-expand")).toHaveCount(0)
    await openArtifactPanel(page)
    await closeVisibleArtifactPanel(page)
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)

    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
    await expect(page.getByTestId("chat-sidebar-edge-expand")).toHaveCount(0)
    await openArtifactPanel(page)
    await closeVisibleArtifactPanel(page)
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)
  })
})
