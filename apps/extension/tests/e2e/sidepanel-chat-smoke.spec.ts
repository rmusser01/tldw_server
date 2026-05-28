import { expect, test, type BrowserContext, type Page } from "@playwright/test"
import { launchWithExtensionOrSkip } from "./utils/real-server"
import http from "node:http"
import { AddressInfo } from "node:net"
import path from "path"
import { launchWithExtension } from "./utils/extension"
import {
  forceConnected,
  setSelectedModel,
  waitForConnectionStore
} from "./utils/connection"
import { grantHostPermission } from "./utils/permissions"

const EXT_PATH = path.resolve(
  process.env.TLDW_E2E_EXTENSION_PATH || "build/chrome-mv3"
)
const MODEL_ID = "mock-model"
const MODEL_KEY = `tldw:${MODEL_ID}`
const CHAT_HANDOFF_DESCRIPTION =
  "Opens /chat in a new tab. Sidepanel draft, current page context, and unsaved chat state stay in the sidepanel."

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

const readBody = (req: http.IncomingMessage) =>
  new Promise<string>((resolve) => {
    let body = ""
    req.on("data", (chunk) => {
      body += chunk
    })
    req.on("end", () => resolve(body))
  })

const startChatMockServer = async () => {
  const server = http.createServer(async (req, res) => {
    const method = (req.method || "GET").toUpperCase()
    const url = req.url || "/"

    const sendJson = (code: number, payload: unknown) => {
      res.writeHead(code, {
        "content-type": "application/json",
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true"
      })
      res.end(JSON.stringify(payload))
    }

    if (method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true",
        "access-control-allow-headers":
          "content-type, x-api-key, authorization"
      })
      return res.end()
    }

    if (url === "/api/v1/health" && method === "GET") {
      return sendJson(200, { status: "ok" })
    }

    if (url === "/api/v1/llm/models/metadata" && method === "GET") {
      return sendJson(200, [
        {
          id: MODEL_ID,
          name: "Mock Model",
          provider: "mock",
          context_length: 4096,
          capabilities: ["chat"]
        }
      ])
    }

    if (url === "/api/v1/llm/models" && method === "GET") {
      return sendJson(200, [MODEL_ID])
    }

    if (url === "/openapi.json" && method === "GET") {
      return sendJson(200, {
        openapi: "3.0.0",
        info: { version: "mock" },
        paths: {
          "/api/v1/health": {},
          "/api/v1/chat/completions": {},
          "/api/v1/llm/models": {},
          "/api/v1/llm/models/metadata": {}
        }
      })
    }

    if (url === "/api/v1/chat/completions" && method === "POST") {
      const body = await readBody(req)
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

    return sendJson(404, { detail: "not found" })
  })

  await new Promise<void>((resolve) =>
    server.listen(0, "127.0.0.1", resolve)
  )
  const addr = server.address() as AddressInfo
  return { server, baseUrl: `http://127.0.0.1:${addr.port}` }
}

const stopChatMockServer = async (server: http.Server) => {
  await new Promise<void>((resolve) => {
    let settled = false
    const done = () => {
      if (settled) return
      settled = true
      resolve()
    }

    server.close(done)
    server.closeAllConnections?.()
    const fallback = setTimeout(done, 1000)
    fallback.unref?.()
  })
}

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

const waitForOpenedExtensionRoute = async (
  context: BrowserContext,
  expectedUrl: string,
  trigger: () => Promise<void>
) => {
  const popupPromise = context
    .waitForEvent("page", { timeout: 10000 })
    .catch(() => null)

  await trigger()

  const popup = await popupPromise
  if (popup) {
    await popup.waitForLoadState("domcontentloaded").catch(() => {})
  }

  await expect
    .poll(
      () => context.pages().some((page) => page.url() === expectedUrl),
      { timeout: 10000 }
    )
    .toBe(true)

  const openedPage = context
    .pages()
    .find((page) => page.url() === expectedUrl)
  expect(openedPage, `Expected ${expectedUrl} to open`).toBeTruthy()
  return openedPage as Page
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
      await stopChatMockServer(server)
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

      const expectedFullChatUrl = `chrome-extension://${extensionId}/options.html#/chat`
      const openedFullChatPage = await waitForOpenedExtensionRoute(
        context,
        expectedFullChatUrl,
        () => headerFullScreen.click()
      )
      await expect(openedFullChatPage.locator("body")).not.toContainText(
        "CharacterControlRail"
      )
    } finally {
      await context.close()
      await stopChatMockServer(server)
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
      await stopChatMockServer(server)
    }
  })
})
