import { expect, test, type Locator, type Page } from "@playwright/test"

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
      body: JSON.stringify({
        models: [
          {
            id: "openai/gpt-4o",
            name: "gpt-4o",
            provider: "openai",
            type: "chat",
            is_configured: true,
            provider_enabled: true,
            availability: "available"
          }
        ]
      })
    })
  })
  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        providers: [
          {
            name: "openai",
            display_name: "OpenAI",
            is_configured: true,
            enabled: true,
            models: ["gpt-4o"]
          }
        ]
      })
    })
  })
  await page.addInitScript(() => {
    window.localStorage.setItem("ff_chatSidebar", "true")
    window.localStorage.setItem("selectedModel", JSON.stringify("openai/gpt-4o"))
    window.localStorage.removeItem("stickyChatInput")
    window.localStorage.removeItem("tldw:nextgenComposerEnabled")
    window.localStorage.removeItem("tldw:composerVariant")
    window.localStorage.removeItem("playgroundComposerOptionsExpanded")
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

const expectStableVerticalPosition = (
  actual: { y: number } | null,
  expected: { y: number },
) => {
  expect(actual).not.toBeNull()
  expect(Math.abs(actual!.y - expected.y)).toBeLessThanOrEqual(2)
}

const expectStableBottomPosition = (
  actual: { y: number; height: number } | null,
  expected: { y: number; height: number },
) => {
  expect(actual).not.toBeNull()
  const actualBottom = actual!.y + actual!.height
  const expectedBottom = expected.y + expected.height
  expect(Math.abs(actualBottom - expectedBottom)).toBeLessThanOrEqual(2)
}

const expectLeftEdgeHandle = async (button: Locator) => {
  const box = await button.boundingBox()
  expect(box).not.toBeNull()
  expect(box!.x).toBeLessThanOrEqual(1)
  expect(box!.height).toBeGreaterThan(box!.width * 2)
}

const expectRightEdgeHandle = async (page: Page, button: Locator) => {
  const box = await button.boundingBox()
  const viewport = page.viewportSize()
  expect(box).not.toBeNull()
  expect(viewport).not.toBeNull()
  expect(box!.x + box!.width).toBeGreaterThanOrEqual(viewport!.width - 1)
  expect(box!.height).toBeGreaterThan(box!.width * 2)
}

test.describe("/chat siderail collapse", () => {
  test("desktop cockpit restore stays clickable above the collapsed chat rail edge", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 960 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)

    const chatRailEdge = page.getByTestId("chat-sidebar-edge-expand")
    await expect(chatRailEdge).toBeVisible()

    const cockpitLeftRail = page.getByTestId("playground-cockpit-left-rail")
    await expect(cockpitLeftRail).toBeVisible()
    await cockpitLeftRail
      .getByRole("button", { name: "Collapse context sidechannel" })
      .click()

    const cockpitRestore = page.getByTestId(
      "playground-cockpit-left-rail-restore",
    )
    await expect(cockpitRestore).toBeVisible()
    await expect(chatRailEdge).toBeVisible()

    await cockpitRestore.click()
    await expect(cockpitLeftRail).toBeVisible()
    await expect(chatRailEdge).toBeVisible()
  })

  test("desktop default composer keeps collapsed rails recoverable from the same edge", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 960 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)

    const chatShell = page.getByTestId("playground-chat-shell")
    const composerRegion = page.getByTestId("playground-chat-composer-region")
    const composerInput = page.getByTestId("chat-input")
    await expect(chatShell).toBeVisible()
    await expect(composerRegion).toBeVisible()
    await expect(composerInput).toBeVisible()
    await expect(page.getByTestId("playground-chat-composer-dock")).toHaveCount(0)

    const leftEdge = page.getByTestId("chat-sidebar-edge-expand")
    await expect(leftEdge).toBeVisible()
    await expectLeftEdgeHandle(leftEdge)

    await leftEdge.click()
    await expect(page.getByTestId("chat-sidebar")).toBeVisible()
    await openArtifactPanel(page)
    const artifactPanel = visibleArtifactPanel(page)
    await expect(artifactPanel).toHaveCount(1)
    await expect(artifactPanel).toBeVisible()
    await expect(leftEdge).toHaveCount(0)
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)

    const bothOpenBox = await chatShell.boundingBox()
    const bothOpenComposerBox = await composerRegion.boundingBox()
    expect(bothOpenBox).not.toBeNull()
    expect(bothOpenComposerBox).not.toBeNull()

    await page.getByTestId("chat-sidebar-toggle").click()
    await expect(leftEdge).toBeVisible()
    await expectLeftEdgeHandle(leftEdge)
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)
    await expect(artifactPanel).toHaveCount(1)
    await expect(artifactPanel).toBeVisible()
    const leftCollapsedBox = await chatShell.boundingBox()
    const leftCollapsedComposerBox = await composerRegion.boundingBox()
    expect(leftCollapsedBox).not.toBeNull()
    expect(leftCollapsedBox!.width).toBeGreaterThan(bothOpenBox!.width)
    expectStableVerticalPosition(leftCollapsedBox, bothOpenBox!)
    expectStableBottomPosition(leftCollapsedComposerBox, bothOpenComposerBox!)

    await leftEdge.click()
    await expect(page.getByTestId("chat-sidebar")).toBeVisible()
    const rightOpenBox = await chatShell.boundingBox()
    const rightOpenComposerBox = await composerRegion.boundingBox()
    expect(rightOpenBox).not.toBeNull()
    expect(rightOpenComposerBox).not.toBeNull()
    await closeVisibleArtifactPanel(page)

    const rightEdge = page.getByTestId("playground-artifacts-edge-expand")
    await expect(rightEdge).toBeVisible()
    await expectRightEdgeHandle(page, rightEdge)
    await expect(page.getByTestId("chat-sidebar")).toBeVisible()
    await expect(leftEdge).toHaveCount(0)
    await expect(artifactPanel).toHaveCount(0)
    const rightClosedBox = await chatShell.boundingBox()
    const rightClosedComposerBox = await composerRegion.boundingBox()
    expect(rightClosedBox).not.toBeNull()
    expect(rightClosedBox!.width).toBeGreaterThan(rightOpenBox!.width)
    expectStableVerticalPosition(rightClosedBox, rightOpenBox!)
    expectStableBottomPosition(rightClosedComposerBox, rightOpenComposerBox!)
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
