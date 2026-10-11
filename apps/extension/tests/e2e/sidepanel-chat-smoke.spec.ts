import { expect, test, type Page } from "@playwright/test"
import { launchWithExtensionOrSkip } from "./utils/real-server"
import path from "path"
import { launchWithExtension } from "./utils/extension"
import {
  forceConnected,
  setSelectedModel,
  waitForConnectionStore
} from "./utils/connection"
import { grantHostPermission } from "./utils/permissions"
import {
  startSidepanelMockServer,
  stopSidepanelMockServer
} from "./utils/sidepanel-mock-server"

const EXT_PATH = path.resolve(
  process.env.TLDW_E2E_EXTENSION_PATH || "build/chrome-mv3"
)
const MODEL_ID = "mock-model"
const MODEL_KEY = `tldw:${MODEL_ID}`
const CHAT_HANDOFF_DESCRIPTION =
  "Opens /chat in a new tab. Use Continue in WebUI from the composer tools to carry a draft or page context."

const buildSeedConfig = (
  baseUrl: string,
  extra: Record<string, unknown> = {}
) => ({
  __tldw_first_run_complete: true,
  __tldw_allow_offline: true,
  tldwConfig: {
    serverUrl: baseUrl,
    authMode: "single-user",
    apiKey: "test-key"
  },
  ...extra
})

const startChatMockServer = () =>
  startSidepanelMockServer({
    modelId: MODEL_ID,
    modelName: "Mock Model",
    handleChatCompletions: async ({ body, res, sendJson }) => {
      let stream = true
      try {
        const parsed = JSON.parse(body || "{}")
        stream = parsed?.stream !== false
      } catch {
        stream = true
      }

      if (!stream) {
        return sendJson(200, {
          choices: [
            {
              message: { role: "assistant", content: "Mock reply from Playwright" }
            }
          ]
        })
      }

      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive"
      })

      const chunks = ["Mock reply", " from Playwright"]
      chunks.forEach((chunk) => {
        res.write(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: chunk } }]
          })}\n\n`
        )
      })
      res.write("data: [DONE]\n\n")
      return res.end()
    }
  })

const ensureChatInput = async (page: Page) => {
  const startButton = page.getByRole("button", { name: /Start chatting/i })
  if ((await startButton.count()) > 0) {
    await startButton.first().click()
  }

  let input = page.getByTestId("chat-input")
  if ((await input.count()) === 0) {
    input = page.getByPlaceholder(/Type a message/i)
  }
  await expect(input).toBeVisible({ timeout: 15000 })
  await expect(input).toBeEditable({ timeout: 15000 })
  await input.click()
  return input
}

test.describe("Sidepanel chat smoke", () => {
  test("keeps the 390px sidepanel chat layout inside the viewport", async () => {
    test.setTimeout(90000)
    const { server, baseUrl } = await startChatMockServer()

    const { context, page, openSidepanel, extensionId } =
      (await launchWithExtensionOrSkip(test, EXT_PATH, {
        seedConfig: buildSeedConfig(baseUrl)
      })) as any

    try {
      const origin = new URL(baseUrl).origin + "/*"
      const granted = await grantHostPermission(
        context,
        extensionId,
        origin
      )
      expect(
        granted,
        "Host permission must be granted programmatically before sidepanel chat can reach the mock server."
      ).toBe(true)

      await setSelectedModel(page, MODEL_KEY)

      const sidepanel = await openSidepanel("/chat")
      await sidepanel.setViewportSize({ width: 390, height: 780 })
      await waitForConnectionStore(sidepanel, "sidepanel-chat:narrow-store")
      await forceConnected(
        sidepanel,
        { serverUrl: baseUrl },
        "sidepanel-chat:narrow-connected"
      )

      await ensureChatInput(sidepanel)
      await expect(sidepanel.getByTestId("chat-main")).toBeVisible()
      await expect(
        sidepanel.getByRole("button", { name: /Send message|Queue request/i })
      ).toBeVisible()

      const metrics = await sidepanel.evaluate(() => {
        const rectFor = (selector: string) => {
          const node = document.querySelector(selector)
          if (!node) return null
          const rect = node.getBoundingClientRect()
          return {
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            width: Math.round(rect.width)
          }
        }

        return {
          innerWidth: window.innerWidth,
          documentScrollWidth: document.documentElement.scrollWidth,
          bodyScrollWidth: document.body.scrollWidth,
          workspace: rectFor('[data-testid="chat-workspace"]'),
          main: rectFor('[data-testid="chat-main"]'),
          messages: rectFor('[data-testid="chat-messages"]'),
          input: rectFor('[data-testid="chat-input"]'),
          send: rectFor(
            '[data-testid="chat-send"], [aria-label="Send message"], [aria-label="Queue request"]'
          )
        }
      })

      expect(metrics.innerWidth).toBe(390)
      expect(metrics.documentScrollWidth).toBeLessThanOrEqual(390)
      expect(metrics.bodyScrollWidth).toBeLessThanOrEqual(390)
      expect(metrics.workspace?.width).toBeLessThanOrEqual(390)
      expect(metrics.main?.right).toBeLessThanOrEqual(390)
      expect(metrics.messages?.right).toBeLessThanOrEqual(390)
      expect(metrics.input?.right).toBeLessThanOrEqual(390)
      expect(metrics.send?.right).toBeLessThanOrEqual(390)
    } finally {
      await context.close()
      await stopSidepanelMockServer(server)
    }
  })

  test("keeps packaged /chat handoffs route-only and rail-safe", async ({}, testInfo) => {
    test.setTimeout(90000)
    const { server, baseUrl } = await startChatMockServer()

    const { context, page, openSidepanel, extensionId } =
      (await launchWithExtensionOrSkip(test, EXT_PATH, {
        seedConfig: buildSeedConfig(baseUrl)
      })) as any

    try {
      const origin = new URL(baseUrl).origin + "/*"
      const granted = await grantHostPermission(
        context,
        extensionId,
        origin
      )
      expect(
        granted,
        "Host permission must be granted programmatically before sidepanel chat can reach the mock server."
      ).toBe(true)

      await setSelectedModel(page, MODEL_KEY)

      const sidepanel = await openSidepanel("/chat")
      await sidepanel.setViewportSize({ width: 390, height: 780 })
      await waitForConnectionStore(sidepanel, "sidepanel-chat:handoff-store")
      await forceConnected(
        sidepanel,
        { serverUrl: baseUrl },
        "sidepanel-chat:handoff-connected"
      )

      await ensureChatInput(sidepanel)
      await expect(sidepanel.getByTestId("chat-main")).toBeVisible()
      await expect(sidepanel.locator("body")).not.toContainText(
        "CharacterControlRail"
      )

      const headerFullScreen = sidepanel.getByTestId("chat-open-full-screen")
      await expect(headerFullScreen).toHaveAttribute(
        "title",
        CHAT_HANDOFF_DESCRIPTION
      )
      await expect(headerFullScreen).toHaveAttribute(
        "aria-label",
        "Open full chat in WebUI"
      )

      await sidepanel.screenshot({
        path: testInfo.outputPath("packaged-sidepanel-chat-handoff.png"),
        fullPage: true
      })

      const [openedFullChatPage] = await Promise.all([
        context.waitForEvent("page"),
        headerFullScreen.click()
      ])
      await expect
        .poll(() => openedFullChatPage.url(), { timeout: 10_000 })
        .toContain(`${new URL(baseUrl).origin}/chat`)

      const openedUrl = new URL(openedFullChatPage.url())
      expect(openedUrl.href).not.toContain("/options.html")
      expect(openedUrl.searchParams.has("handoff")).toBe(false)
      expect(new URLSearchParams(openedUrl.hash.slice(1)).get("handoff")).toBeTruthy()
    } finally {
      await context.close()
      await stopSidepanelMockServer(server)
    }
  })

  test("sends and renders a reply", async () => {
    test.setTimeout(90000)
    const { server, baseUrl } = await startChatMockServer()

    const { context, page, openSidepanel, extensionId } =
      (await launchWithExtensionOrSkip(test, EXT_PATH, {
        seedConfig: buildSeedConfig(baseUrl)
      })) as any

    try {
      const origin = new URL(baseUrl).origin + "/*"
      const granted = await grantHostPermission(
        context,
        extensionId,
        origin
      )
      expect(
        granted,
        "Host permission must be granted programmatically before sidepanel chat can reach the mock server."
      ).toBe(true)

      await setSelectedModel(page, MODEL_KEY)

      const sidepanel = await openSidepanel("/chat")
      await waitForConnectionStore(sidepanel, "sidepanel-chat:store")
      await forceConnected(
        sidepanel,
        { serverUrl: baseUrl },
        "sidepanel-chat:connected"
      )

      const input = await ensureChatInput(sidepanel)
      const message = `Playwright smoke ${Date.now()}`
      await input.fill(message)

      const sendButton = sidepanel.locator('[data-testid="chat-send"]')
      if ((await sendButton.count()) > 0) {
        await expect(sendButton).toBeEnabled({ timeout: 15000 })
        await sendButton.click()
      } else {
        await input.press("Enter")
      }

      const userMessage = sidepanel
        .locator('[data-testid="chat-message"][data-role="user"]')
        .filter({ hasText: message })
        .first()
      await expect(userMessage).toBeVisible({ timeout: 15000 })

      const assistantMessage = sidepanel
        .locator('[data-testid="chat-message"][data-role="assistant"]')
        .filter({ hasText: "Mock reply from Playwright" })
        .first()
      await expect(assistantMessage).toBeVisible({ timeout: 20000 })
    } finally {
      await context.close()
      await stopSidepanelMockServer(server)
    }
  })
})
