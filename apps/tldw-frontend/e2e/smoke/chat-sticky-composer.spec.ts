import { expect, test, type Page } from "@playwright/test"

const bypassChatGates = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
      window.localStorage.setItem("__tldw_test_bypass", "true")
      window.localStorage.setItem("stickyChatInput", "true")
    } catch {
      /* ignore */
    }
  })
}

const waitForStickyChat = async (page: Page) => {
  await page.goto("/chat", { waitUntil: "domcontentloaded" })
  await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
  await expect(page.getByTestId("playground-chat-composer-dock")).toBeVisible({
    timeout: 30_000
  })
}

const expectDockWithinViewport = async (page: Page) => {
  const dockBox = await page.getByTestId("playground-chat-composer-dock").boundingBox()
  expect(dockBox).not.toBeNull()
  expect(dockBox!.y).toBeGreaterThanOrEqual(0)
  expect(dockBox!.y + dockBox!.height).toBeLessThanOrEqual(page.viewportSize()!.height)
}

test.describe("chat sticky composer dock", () => {
  test("desktop sticky /chat keeps the composer visible while the transcript scrolls", async ({
    page
  }) => {
    test.setTimeout(90_000)
    await bypassChatGates(page)
    await page.setViewportSize({ width: 1440, height: 960 })
    await waitForStickyChat(page)

    const transcript = page.getByTestId("playground-chat-transcript")
    await expect(transcript).toBeVisible()

    await expectDockWithinViewport(page)
  })

  test("mobile-sized sticky /chat keeps the composer visible after focusing the input", async ({
    page
  }) => {
    test.setTimeout(90_000)
    await bypassChatGates(page)
    await page.setViewportSize({ width: 390, height: 844 })
    await waitForStickyChat(page)

    const chatInput = page.getByTestId("chat-input")
    await expect(chatInput).toBeVisible()
    await chatInput.focus()

    await expectDockWithinViewport(page)
  })

  test("desktop sticky /chat keeps the dock scoped to the chat column when artifacts are open", async ({
    page
  }) => {
    test.setTimeout(90_000)
    await bypassChatGates(page)
    await page.setViewportSize({ width: 1440, height: 960 })
    await waitForStickyChat(page)

    await page.evaluate(() => {
      const artifactsStore = (window as Window & {
        __tldw_useArtifactsStore?: {
          getState: () => {
            openArtifact: (
              artifact: {
                id: string
                title: string
                content: string
                kind: "code"
                language?: string
              },
              options?: { auto?: boolean }
            ) => void
          }
        }
      }).__tldw_useArtifactsStore

      artifactsStore?.getState().openArtifact(
        {
          id: "dock-smoke-artifact",
          title: "Smoke artifact",
          content: "console.log('dock')",
          kind: "code",
          language: "ts"
        },
        { auto: false }
      )
    })

    const artifactsPanel = page.locator('[data-testid="artifacts-panel"]').first()
    await expect(artifactsPanel).toBeVisible({ timeout: 30_000 })
    await expectDockWithinViewport(page)

    const dockBox = await page.getByTestId("playground-chat-composer-dock").boundingBox()
    const panelBox = await artifactsPanel.boundingBox()
    expect(dockBox).not.toBeNull()
    expect(panelBox).not.toBeNull()
    expect(dockBox!.x + dockBox!.width).toBeLessThanOrEqual(panelBox!.x + 1)
  })
})
