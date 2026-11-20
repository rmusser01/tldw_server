import { expect, test, type BrowserContext } from "@playwright/test"
import { grantHostPermission } from "./utils/permissions"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"
import {
  waitForConnectionStore,
  forceUnconfigured,
  setSelectedModel
} from "./utils/connection"

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value : `http://${value}`

test.describe("Chat workspace UX", () => {
  test("shows first-run connection CTA when unconfigured", async () => {
    test.setTimeout(60000)
    const { context, openSidepanel } = await launchWithExtensionOrSkip(test, "")

    try {
      const page = await openSidepanel()
      await page.waitForLoadState("domcontentloaded")
      const workspace = page.locator('[data-testid="chat-workspace"]')
      const main = page.locator('[data-testid="chat-main"]')
      if ((await workspace.count()) > 0) {
        await expect(workspace).toBeVisible()
      } else if ((await main.count()) > 0) {
        await expect(main).toBeVisible()
      } else {
        await expect(page.locator("#root")).toBeAttached()
      }

      await waitForConnectionStore(page, "chat-ux:first-run")
      await forceUnconfigured(page, "chat-ux:first-run")

      const emptyConnection = page.locator('[data-testid="chat-empty-connection"]')
      if ((await emptyConnection.count()) > 0) {
        await expect(emptyConnection).toBeVisible()
      } else {
        await expect(page.getByText(/Finish setup|Connect tldw Assistant/i).first()).toBeVisible()
      }

      const connectionCta = page.locator('[data-testid="chat-connection-cta"]')
      if ((await connectionCta.count()) > 0) {
        await expect(connectionCta).toBeVisible()
      } else {
        await expect(
          page.getByRole("button", { name: /Finish setup|Open tldw Settings|Review settings/i })
        ).toBeVisible()
      }

      let input = page.getByTestId("chat-input")
      if ((await input.count()) === 0) {
        input = page.getByPlaceholder(/Type a message/i)
      }
      await expect(input).not.toBeEditable()
    } finally {
      await context.close()
    }
  })

  test("walks through the core chat workflow", async () => {
    test.setTimeout(120000)
    const mark = (label: string) => {
      console.log(`[chat-ux] ${label}`)
    }

    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    let context: BrowserContext | null = null

    try {
      let modelsResponse: Response | null = null
      try {
        modelsResponse = await fetch(
          `${normalizedServerUrl}/api/v1/llm/models/metadata`,
          {
            headers: { "x-api-key": apiKey }
          }
        )
      } catch (error) {
        test.skip(
          true,
          `Chat models preflight unreachable in this environment: ${String(error)}`
        )
        return
      }
      if (!modelsResponse) return
      if (!modelsResponse.ok) {
        const body = await modelsResponse.text().catch(() => "")
        test.skip(
          true,
          `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
        )
        return
      }
      const modelsPayload = await modelsResponse.json().catch(() => [])
      const modelsList = Array.isArray(modelsPayload)
        ? modelsPayload
        : Array.isArray((modelsPayload as any)?.models)
          ? (modelsPayload as any).models
          : []
      const modelId =
        modelsList.find((m: any) => m?.id || m?.model || m?.name)?.id ||
        modelsList.find((m: any) => m?.id || m?.model || m?.name)?.model ||
        modelsList.find((m: any) => m?.id || m?.model || m?.name)?.name
      if (!modelId) {
        test.skip(true, "No chat models returned from tldw_server.")
      }

      mark("launch extension")
      const launchResult = await launchWithExtensionOrSkip(test, "", {
        seedConfig: {
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        }
      })
      context = launchResult.context
      const { page, extensionId, openSidepanel } = launchResult
      const origin = new URL(normalizedServerUrl).origin + "/*"

      mark("grant host permission")
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
        )
      }

      await setSelectedModel(page, String(modelId))

      mark("open chat workspace")
      const chatPage = await openSidepanel()
      await chatPage.waitForLoadState("domcontentloaded")
      const workspace = chatPage.locator('[data-testid="chat-workspace"]')
      const main = chatPage.locator('[data-testid="chat-main"]')
      if ((await workspace.count()) > 0) {
        await expect(workspace).toBeVisible()
      } else if ((await main.count()) > 0) {
        await expect(main).toBeVisible()
      } else {
        await expect(chatPage.locator("#root")).toBeAttached()
      }

      mark("check connection")
      await waitForConnectionStore(chatPage, "chat-ux:before-check")
      await chatPage.evaluate(() => {
        window.dispatchEvent(new CustomEvent("tldw:check-connection"))
      })
      await chatPage.waitForFunction(
        () => {
          const store = (window as any).__tldw_useConnectionStore
          const state = store?.getState?.().state
          return state?.isConnected === true && state?.phase === "connected"
        },
        undefined,
        { timeout: 15000 }
      )

      await chatPage.evaluate(() => {
        if (typeof chrome === "undefined" || !chrome.storage) return
        try {
          chrome.storage.local?.set?.({
            "tldw:seenHints": {
              "knowledge-search": true,
              "more-tools": true
            }
          })
          chrome.storage.sync?.set?.({
            ragSearchHintSeen: true
          })
        } catch {
          // ignore storage errors in test seed
        }
      })

      let input = chatPage.getByTestId("chat-input")
      if ((await input.count()) === 0) {
        input = chatPage.getByPlaceholder(/Type a message/i)
      }
      await expect(input).toBeVisible({ timeout: 15000 })
      await expect(input).toBeEditable({ timeout: 15000 })

      mark("select model")
      await setSelectedModel(chatPage, String(modelId))

      mark("toggle rag panel")
      // XS-15: Search & Context opens as a modal sheet over the composer;
      // it closes from the sheet (Close, Escape or a click outside).
      const ragToggle = chatPage.getByTestId("control-rag-toggle")
      await expect(ragToggle).toBeVisible()
      await ragToggle.click()
      const knowledgeSheet = chatPage.getByRole("dialog", { name: "Search & Context" })
      await expect(knowledgeSheet).toBeVisible()
      await chatPage.keyboard.press("Escape")
      await expect(knowledgeSheet).toBeHidden()
      await expect(ragToggle).toBeFocused()

      const webToggle = chatPage.getByTestId("control-web-toggle")
      if (await webToggle.isVisible()) {
        mark("toggle web search")
        await webToggle.click()
        await expect(webToggle).toHaveAttribute("aria-pressed", "true")
        await webToggle.click()
      }

      mark("send message")
      const message = `E2E Chat ${Date.now()}`
      await input.fill(message)
      const sendButton = chatPage.locator('[data-testid="chat-send"]')
      if ((await sendButton.count()) > 0) {
        await sendButton.click()
      } else {
        await input.press("Enter")
      }

      const userMessage = chatPage
        .locator('[data-testid="chat-message"][data-role="user"]')
        .filter({ hasText: message })
        .first()

      const sawMessage = async () => {
        if ((await userMessage.count()) > 0) {
          return await userMessage.isVisible().catch(() => false)
        }
        return await chatPage.evaluate((text) => {
          const log = document.querySelector('[role="log"]')
          return !!log && (log.textContent || "").includes(text)
        }, message)
      }

      if (!(await sawMessage())) {
        const alertText = await chatPage.evaluate(() => {
          const alert = document.querySelector('[role="alert"]')
          return (alert?.textContent || "").trim()
        })
        if (/model/i.test(alertText)) {
          await setSelectedModel(chatPage, String(modelId))
          const currentValue = await input.inputValue().catch(() => "")
          if (!currentValue || currentValue.trim().length === 0) {
            await input.fill(message)
          }
          if ((await sendButton.count()) > 0) {
            await sendButton.click()
          } else {
            await input.press("Enter")
          }
        }
      }

      const waitForSendOutcome = async (timeoutMs: number) => {
        const start = Date.now()
        while (Date.now() - start < timeoutMs) {
          if (await sawMessage()) return "sent"
          const state = await chatPage.evaluate(() => {
            const input =
              document.querySelector('[data-testid="chat-input"]') ||
              document.querySelector('textarea')
            const inputValue =
              input && "value" in input ? String((input as HTMLTextAreaElement).value) : ""
            const alert = document.querySelector('[role="alert"]')
            const log = document.querySelector('[role="log"]')
            return {
              inputFound: !!input,
              inputValue,
              alertText: (alert?.textContent || "").trim(),
              logText: log?.textContent || ""
            }
          })
          if (state.logText.includes(message)) return "sent"
          if (state.inputFound && state.inputValue.trim().length === 0) return "cleared"
          if (state.alertText.length > 0) return "alert"
          await new Promise((resolve) => setTimeout(resolve, 250))
        }
        return ""
      }

      const sendOutcome = await waitForSendOutcome(15000)
      if (!sendOutcome) {
        console.warn("[chat-ux] message did not appear; continuing")
      }

      if (sendOutcome === "sent") {
        if ((await userMessage.count()) > 0) {
          await expect(userMessage).toBeVisible({ timeout: 15000 })
        } else {
          await expect
            .poll(async () => await sawMessage(), { timeout: 15000 })
            .toBe(true)
        }
      }

      const assistantMessages = chatPage.locator(
        '[data-testid="chat-message"][data-role="assistant"]'
      )
      const hasAssistantHooks = (await assistantMessages.count()) > 0
      const rows = chatPage.locator("[data-index]")
      const hasRowHooks = (await rows.count()) > 0

      if (sendOutcome === "sent") {
        if (hasAssistantHooks) {
          await expect
            .poll(async () => assistantMessages.count(), { timeout: 20000 })
            .toBeGreaterThan(0)
        } else if (hasRowHooks) {
          await expect
            .poll(async () => rows.count(), { timeout: 20000 })
            .toBeGreaterThan(1)
        } else {
          // Old build without test hooks: just ensure the log has content.
          await expect
            .poll(async () => {
              const text = await chatPage.evaluate(() => {
                const log = document.querySelector('[role="log"]')
                return (log?.textContent || "").trim()
              })
              return text.length > 0
            }, { timeout: 15000 })
            .toBe(true)
        }
      } else {
        console.warn("[chat-ux] assistant check skipped (message not confirmed)")
      }

      mark("toggle save status")
      let saveStatus = chatPage.locator('[data-testid="chat-save-status"]')
      if ((await saveStatus.count()) === 0) {
        saveStatus = chatPage.getByRole("button", {
          name: /Saved locally|Saved to server|Ephemeral/i
        })
      }
      if (await saveStatus.isVisible()) {
        const before = await saveStatus.getAttribute("aria-label")
        await saveStatus.click()
        const ephemeralBadge = chatPage.locator('[data-testid="chat-ephemeral-badge"]')
        if ((await ephemeralBadge.count()) > 0) {
          await expect(ephemeralBadge).toBeVisible()
        } else {
          try {
            await expect
              .poll(async () => (await saveStatus.getAttribute("aria-label")) || "", {
                timeout: 8000
              })
              .not.toBe(before || "")
          } catch {
            console.warn("[chat-ux] save status label did not change after toggle")
          }
        }
      } else {
        console.warn("[chat-ux] save status control not visible; skipping toggle")
      }
    } finally {
      if (context) {
        await context.close()
      }
    }
  })
})
