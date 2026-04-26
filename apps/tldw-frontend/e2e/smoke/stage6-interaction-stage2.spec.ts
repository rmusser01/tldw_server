import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from "./smoke.setup"
import type { Locator, Route } from "@playwright/test"
import { dispatchKeyboardShortcut, waitForAppShell } from "../utils/helpers"

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(data)
  })
}

const expectMobileTouchTarget = async (locator: Locator, name: string) => {
  const box = await locator.boundingBox()
  if (!box) {
    throw new Error(`${name} should have a measurable bounding box`)
  }
  expect(box.width, `${name} width should be >= 44px`).toBeGreaterThanOrEqual(44)
  expect(box.height, `${name} height should be >= 44px`).toBeGreaterThanOrEqual(44)
}

test.describe("Stage 6 interaction stage 2 positive regressions", () => {
  test("search typing and deterministic no-results answer remain functional", async ({
    page
  }) => {
    await seedAuth(page)
    await page.addInitScript(() => {
      try {
        localStorage.setItem("ff_knowledgeQaStreaming", "false")
      } catch {}
    })

    let ragSearchCalls = 0
    let lastRagQuery = ""
    let messageCounter = 0

    await page.route("**/api/v1/characters/search**", async (route) => {
      await fulfillJson(route, 200, [])
    })

    await page.route("**/api/v1/characters**", async (route) => {
      await fulfillJson(route, 200, [])
    })

    await page.route("**/api/v1/chats/**", async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const method = request.method().toUpperCase()

      if (method === "POST" && /^\/api\/v1\/chats\/?$/.test(url.pathname)) {
        await fulfillJson(route, 200, {
          id: "stage2-thread",
          title: "Stage 2 thread",
          state: "in-progress",
          version: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        return
      }

      if (method === "GET" && /^\/api\/v1\/chats\/?$/.test(url.pathname)) {
        await fulfillJson(route, 200, {
          chats: [],
          total: 0
        })
        return
      }

      if (method === "GET" && /^\/api\/v1\/chats\/[^/]+$/.test(url.pathname)) {
        await fulfillJson(route, 200, {
          id: "stage2-thread",
          title: "Stage 2 thread",
          state: "in-progress",
          version: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        return
      }

      if (method === "POST" && /\/api\/v1\/chats\/[^/]+\/messages$/.test(url.pathname)) {
        messageCounter += 1
        await fulfillJson(route, 200, {
          id: `stage2-message-${messageCounter}`,
          role: "assistant",
          content: "",
          created_at: new Date().toISOString()
        })
        return
      }

      await fulfillJson(route, 200, {})
    })

    await page.route("**/api/v1/chat/conversations**", async (route) => {
      const request = route.request()
      const method = request.method().toUpperCase()

      if (method === "GET") {
        await fulfillJson(route, 200, {
          items: []
        })
        return
      }

      if (method === "PATCH") {
        await fulfillJson(route, 200, { success: true })
        return
      }

      await fulfillJson(route, 200, {})
    })

    await page.route("**/api/v1/chat/conversations/**", async (route) => {
      const request = route.request()
      const method = request.method().toUpperCase()

      if (method === "GET") {
        await fulfillJson(route, 200, {
          id: "stage2-thread",
          version: 1,
          keywords: []
        })
        return
      }

      if (method === "PATCH") {
        await fulfillJson(route, 200, { success: true })
        return
      }

      await fulfillJson(route, 200, {})
    })

    await page.route("**/api/v1/chat/messages/**/rag-context**", async (route) => {
      await fulfillJson(route, 200, { success: true })
    })

    await page.route("**/api/v1/rag/search**", async (route) => {
      ragSearchCalls += 1
      const body = route.request().postDataJSON() as { query?: string } | null
      lastRagQuery = typeof body?.query === "string" ? body.query : ""

      await fulfillJson(route, 200, {
        results: [],
        generated_answer:
          "No relevant sources found for this query in your current knowledge base.",
        citations: []
      })
    })

    await page.goto("/search", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await page.waitForURL((url) => url.pathname === "/knowledge", {
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const knowledgeSearchInput = page.getByLabel("Search your knowledge base")
    await expect(knowledgeSearchInput).toBeVisible({ timeout: LOAD_TIMEOUT })

    const query = "stage2 deterministic no results query"
    await knowledgeSearchInput.fill(query)
    await expect(knowledgeSearchInput).toHaveValue(query)
    await knowledgeSearchInput.press("Enter")
    const askButton = page.getByRole("button", { name: /^Ask$/i })

    await expect
      .poll(
        async () =>
          ragSearchCalls > 0 || (await askButton.isVisible().catch(() => false)),
        {
          timeout: LOAD_TIMEOUT,
          message:
            "Expected pressing Enter to start search or leave the Ask button available as a fallback"
        }
      )
      .toBe(true)

    if (ragSearchCalls === 0) {
      if (await askButton.isVisible().catch(() => false)) {
        await askButton.click()
      }
    }

    await expect
      .poll(() => ragSearchCalls, {
        timeout: LOAD_TIMEOUT,
        message: "Expected deterministic /api/v1/rag/search stub to be called"
      })
      .toBeGreaterThan(0)

    expect(lastRagQuery).toBe(query)

    await expect(page.getByText("AI Answer")).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(page.getByTestId("knowledge-answer-content")).toContainText(
      "No relevant sources found for this query in your current knowledge base.",
      { timeout: LOAD_TIMEOUT }
    )
    await expect(page.locator("[id^='source-card-']")).toHaveCount(0)
  })

  test("command palette supports keyboard-only open, focus, and execute", async ({
    page
  }) => {
    await seedAuth(page)

    await page.goto("/", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const palette = page.getByRole("dialog", { name: /command palette/i })

    await expect
      .poll(
        async () => {
          await dispatchKeyboardShortcut(page, { key: "k", metaKey: true })
          if (await palette.isVisible().catch(() => false)) {
            return true
          }

          await dispatchKeyboardShortcut(page, { key: "k", ctrlKey: true })
          return await palette.isVisible().catch(() => false)
        },
        {
          timeout: LOAD_TIMEOUT,
          message: "Expected Cmd/Ctrl+K to open the command palette once the home shell shortcuts were active"
        }
      )
      .toBe(true)

    const paletteInput = palette.locator("input[type='text']").first()
    await expect(paletteInput).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(paletteInput).toBeFocused()

    await paletteInput.fill("go to settings")
    const settingsOption = palette.getByRole("option", { name: /go to settings/i })
    await expect(settingsOption).toBeVisible({ timeout: LOAD_TIMEOUT })

    for (let index = 0; index < 8; index += 1) {
      const isSelected = await settingsOption.getAttribute("aria-selected")
      if (isSelected === "true") {
        break
      }
      await paletteInput.press("ArrowDown")
    }

    await expect(settingsOption).toHaveAttribute("aria-selected", "true", {
      timeout: LOAD_TIMEOUT
    })

    await paletteInput.press("Enter")
    await page.waitForURL((url) => url.pathname.startsWith("/settings"), {
      timeout: LOAD_TIMEOUT
    })
  })

  test("chat mobile composer keeps send/attach/settings controls discoverable and touch-safe", async ({
    page
  }) => {
    await seedAuth(page)
    await page.setViewportSize({ width: 390, height: 844 })
    await page.addInitScript(() => {
      try {
        localStorage.setItem(
          "tldw-ui-mode",
          JSON.stringify({
            state: { mode: "pro" },
            version: 0
          })
        )
      } catch {}

      class MockSpeechRecognition {
        continuous = false
        interimResults = false
        lang = "en-US"
        onresult: ((event: unknown) => void) | null = null
        onerror: ((event: unknown) => void) | null = null
        onend: (() => void) | null = null
        start() {}
        stop() {}
        abort() {}
      }
      ;(window as unknown as { SpeechRecognition?: typeof MockSpeechRecognition }).SpeechRecognition =
        MockSpeechRecognition
      ;(
        window as unknown as {
          webkitSpeechRecognition?: typeof MockSpeechRecognition
        }
      ).webkitSpeechRecognition = MockSpeechRecognition
    })

    await page.goto("/chat", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const composerInput = page
      .locator("#textarea-message, [data-testid='chat-input']")
      .first()
    await expect(composerInput).toBeVisible({ timeout: LOAD_TIMEOUT })

    const attachButton = page.getByRole("button", { name: "Attach image" })
    const sendButton = page.getByRole("button", { name: /send/i }).first()
    const settingsButton = page.getByRole("button", { name: /send options/i })

    await expect(attachButton).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(sendButton).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(settingsButton).toBeVisible({ timeout: LOAD_TIMEOUT })

    await expectMobileTouchTarget(attachButton, "attach button")
    await expectMobileTouchTarget(sendButton, "send button")
    await expectMobileTouchTarget(settingsButton, "send options button")
  })

  test("settings sidebar navigation remains clickable and active-state accurate", async ({
    page
  }) => {
    await seedAuth(page)

    await page.goto("/settings/tldw", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    await expect(page.getByTestId("settings-navigation")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    const chatSettingsLink = page.getByTestId("settings-nav-link--settings-chat")
    await expect(chatSettingsLink).toBeVisible({ timeout: LOAD_TIMEOUT })

    const blockingModal = page.locator(".ant-modal-wrap").first()
    if (await blockingModal.isVisible().catch(() => false)) {
      const closeButton = blockingModal
        .getByRole("button", { name: /close|cancel|ok|got it|dismiss/i })
        .first()
      if (await closeButton.isVisible().catch(() => false)) {
        await closeButton.click().catch(() => {})
      } else {
        await page.keyboard.press("Escape").catch(() => {})
      }
      await blockingModal.waitFor({ state: "hidden", timeout: 3000 }).catch(() => {})
    }

    await chatSettingsLink.click()
    await page.waitForURL((url) => url.pathname === "/settings/chat", {
      timeout: LOAD_TIMEOUT
    })

    const activeSettingsLink = page.locator(
      '[data-testid="settings-nav-link--settings-chat"][aria-current="page"]'
    )
    await expect(activeSettingsLink).toBeVisible({ timeout: LOAD_TIMEOUT })
  })
})
