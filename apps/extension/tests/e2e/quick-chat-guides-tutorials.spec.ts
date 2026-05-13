import { expect, test, type Locator, type Page } from "@playwright/test"
import { launchWithBuiltExtension } from "./utils/extension-build"
import { forceConnected, waitForConnectionStore } from "./utils/connection"

const DEFAULT_SERVER_CONFIG = {
  serverUrl: "http://dummy-tldw",
  apiKey: "test-key"
}

const seedTutorialLocalStorage = (completedTutorials: string[] = []) => ({
  "tldw-tutorials": JSON.stringify({
    state: {
      completedTutorials,
      seenPromptPages: ["/chat", "/research-studio"]
    },
    version: 0
  })
})

const isTimeoutError = (error: unknown): error is Error =>
  error instanceof Error && error.name === "TimeoutError"

async function dismissWelcomeOverlayIfPresent(page: Page) {
  const heading = page.getByText(/Welcome to tldw Assistant/i).first()
  let visible = false

  try {
    visible = await heading.isVisible({ timeout: 2_000 })
  } catch (error) {
    if (!isTimeoutError(error)) {
      throw new Error("Failed while checking welcome overlay visibility", {
        cause: error
      })
    }
  }

  if (!visible) return

  const dialog = page.locator('[role="dialog"]').filter({ has: heading }).first()
  const closeButton = dialog.getByRole("button", { name: /close/i }).first()

  try {
    if (await closeButton.isVisible({ timeout: 1_500 })) {
      await closeButton.click()
    } else {
      await page.keyboard.press("Escape")
    }
  } catch (error) {
    if (!isTimeoutError(error)) {
      throw new Error("Failed to dismiss welcome overlay", { cause: error })
    }
  }
}

async function recoverOptionsErrorState(page: Page, label: string) {
  const errorHeading = page.getByText(/Something went wrong/i).first()
  let hasError = false

  try {
    hasError = await errorHeading.isVisible({ timeout: 1_000 })
  } catch (error) {
    if (!isTimeoutError(error)) {
      throw new Error("Failed while checking options error state", { cause: error })
    }
  }

  if (!hasError) {
    return
  }

  const reloadButton = page.getByRole("button", { name: /Reload Options/i }).first()
  let reloaded = false
  try {
    if (await reloadButton.isVisible({ timeout: 2_000 })) {
      await reloadButton.click()
      reloaded = true
    }
  } catch (error) {
    if (!isTimeoutError(error)) {
      throw new Error("Failed while attempting options error reload", {
        cause: error
      })
    }
  }

  if (!reloaded) {
    await page.reload({ waitUntil: "domcontentloaded" })
  }

  await page.waitForLoadState("networkidle")
  await waitForConnectionStore(page, `${label}-recovered`)
  await forceConnected(
    page,
    { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
    `${label}-recovered`
  )
  await dismissWelcomeOverlayIfPresent(page)
}

async function openQuickChatBrowseGuides(page: Page, label: string): Promise<Locator> {
  await recoverOptionsErrorState(page, `${label}-before-open`)
  const openButton = page.getByTestId("quick-chat-helper-open-button").first()

  try {
    await expect(openButton).toBeVisible({ timeout: 15_000 })
  } catch (error) {
    await recoverOptionsErrorState(page, `${label}-button-retry`)
    await expect(openButton).toBeVisible({ timeout: 15_000 })
  }

  await openButton.click()

  const modal = page.locator(".quick-chat-helper-modal").first()
  try {
    await expect(modal).toBeVisible({ timeout: 10_000 })
  } catch (error) {
    await recoverOptionsErrorState(page, `${label}-modal-retry`)
    await expect(openButton).toBeVisible({ timeout: 10_000 })
    await openButton.click()
    await expect(modal).toBeVisible({ timeout: 10_000 })
  }

  const browseGuidesOption = modal.getByText(/^Browse Guides$/i).first()
  await expect(browseGuidesOption).toBeVisible({ timeout: 10_000 })
  await browseGuidesOption.click()

  await expect(
    modal.getByTestId("quick-chat-guides-tutorials-section")
  ).toBeVisible({ timeout: 10_000 })
  return modal
}

test.describe("Quick Chat Browse Guides tutorials validation", () => {
  test("shows Research Studio tutorial cards on /research-studio route", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: DEFAULT_SERVER_CONFIG,
      seedLocalStorage: seedTutorialLocalStorage()
    })

    try {
      await waitForConnectionStore(page, "quick-chat-guides-workspace")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "quick-chat-guides-workspace"
      )

      await page.goto(`${optionsUrl}#/research-studio`)
      await page.waitForLoadState("networkidle")
      await dismissWelcomeOverlayIfPresent(page)

      const modal = await openQuickChatBrowseGuides(page, "research-studio")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-workspace-playground-basics")
      ).toBeVisible()
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-workspace-playground-basics")
      ).toHaveText(/Start/i)
      await expect(
        modal.getByTestId("quick-chat-guides-workflow-section")
      ).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("shows locked tutorial prerequisites on /chat when no tutorials are completed", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: DEFAULT_SERVER_CONFIG,
      seedLocalStorage: seedTutorialLocalStorage()
    })

    try {
      await waitForConnectionStore(page, "quick-chat-guides-chat-locked")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "quick-chat-guides-chat-locked"
      )

      await page.goto(`${optionsUrl}#/chat?tab=casual`)
      await page.waitForLoadState("networkidle")
      await dismissWelcomeOverlayIfPresent(page)

      const modal = await openQuickChatBrowseGuides(page, "chat-locked")
      const basicsCard = modal.getByTestId("quick-chat-guides-tutorial-playground-basics")
      const toolsCard = modal.getByTestId("quick-chat-guides-tutorial-playground-tools")
      const voiceCard = modal.getByTestId("quick-chat-guides-tutorial-playground-voice")

      await expect(basicsCard).toHaveAttribute("data-locked", "false")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-basics")
      ).toHaveText(/Start/i)

      await expect(toolsCard).toHaveAttribute("data-locked", "true")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-tools")
      ).toHaveText(/Locked/i)

      await expect(voiceCard).toHaveAttribute("data-locked", "true")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-voice")
      ).toHaveText(/Locked/i)
    } finally {
      await context.close()
    }
  })

  test("unlocks prerequisites and marks replay state when playground basics is completed", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: DEFAULT_SERVER_CONFIG,
      seedLocalStorage: seedTutorialLocalStorage(["playground-basics"])
    })

    try {
      await waitForConnectionStore(page, "quick-chat-guides-chat-completed")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "quick-chat-guides-chat-completed"
      )

      await page.goto(`${optionsUrl}#/chat`)
      await page.waitForLoadState("networkidle")
      await dismissWelcomeOverlayIfPresent(page)

      const modal = await openQuickChatBrowseGuides(page, "chat-completed")
      const basicsCard = modal.getByTestId("quick-chat-guides-tutorial-playground-basics")
      const toolsCard = modal.getByTestId("quick-chat-guides-tutorial-playground-tools")
      const voiceCard = modal.getByTestId("quick-chat-guides-tutorial-playground-voice")

      await expect(basicsCard).toHaveAttribute("data-completed", "true")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-basics")
      ).toHaveText(/Replay/i)

      await expect(toolsCard).toHaveAttribute("data-locked", "false")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-tools")
      ).toHaveText(/Start/i)

      await expect(voiceCard).toHaveAttribute("data-locked", "false")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-playground-voice")
      ).toHaveText(/Start/i)
    } finally {
      await context.close()
    }
  })

  test("shows knowledge tutorial card on /knowledge/thread route", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: DEFAULT_SERVER_CONFIG,
      seedLocalStorage: seedTutorialLocalStorage()
    })

    try {
      await waitForConnectionStore(page, "quick-chat-guides-knowledge-thread")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "quick-chat-guides-knowledge-thread"
      )

      await page.goto(`${optionsUrl}#/knowledge/thread/thread-123`)
      await page.waitForLoadState("networkidle")
      await dismissWelcomeOverlayIfPresent(page)

      const modal = await openQuickChatBrowseGuides(page, "knowledge-thread")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-knowledge-basics")
      ).toBeVisible()
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-knowledge-basics")
      ).toHaveText(/Start/i)
    } finally {
      await context.close()
    }
  })

  test("shows knowledge tutorial card on /knowledge/shared route", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: DEFAULT_SERVER_CONFIG,
      seedLocalStorage: seedTutorialLocalStorage()
    })

    try {
      await waitForConnectionStore(page, "quick-chat-guides-knowledge-shared")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "quick-chat-guides-knowledge-shared"
      )

      await page.goto(`${optionsUrl}#/knowledge/shared/share-token-123`)
      await page.waitForLoadState("networkidle")
      await dismissWelcomeOverlayIfPresent(page)

      const modal = await openQuickChatBrowseGuides(page, "knowledge-shared")
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-knowledge-basics")
      ).toBeVisible()
      await expect(
        modal.getByTestId("quick-chat-guides-tutorial-action-knowledge-basics")
      ).toHaveText(/Start/i)
    } finally {
      await context.close()
    }
  })
})
