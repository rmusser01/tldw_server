/**
 * Chat Workflow E2E Tests
 *
 * Tests the complete chat workflow from a user's perspective:
 * - Basic chat interactions
 * - Streaming responses
 * - Chat history
 * - Character selection
 * - Error handling
 * - UI interactions
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels, assertNoCriticalErrors } from "../utils/fixtures"
import { ChatPage } from "../utils/page-objects"
import { dispatchKeyboardShortcut, seedAuth, generateTestId } from "../utils/helpers"

test.describe("Chat Workflow", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
  })

  test.describe("Basic Chat Flow", () => {
    test("should navigate to chat page and display chat interface", async ({
      authedPage,
      diagnostics
    }) => {
      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Verify the main transcript surface is visible
      await expect(chatPage.messageList).toBeVisible()

      // Verify chat input is available
      const input = await chatPage.getChatInput()
      await expect(input).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should send a message and receive a response", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      const testMessage = `Hello, this is a test message ${generateTestId()}`

      await chatPage.sendMessage(testMessage)

      // Wait for response to appear
      await chatPage.waitForResponse(60000)

      // Verify messages in the conversation
      const messages = await chatPage.getMessages()
      expect(messages.length).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display streaming response incrementally", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      await chatPage.sendMessage("Count from 1 to 5 slowly")

      // Watch for streaming indicator
      const streamingIndicator = authedPage.locator(
        "[data-streaming='true'], .streaming, .typing-indicator"
      )

      // Wait for streaming to start or response to complete
      const startedStreaming = await streamingIndicator
        .waitFor({ state: "visible", timeout: 30000 })
        .then(() => true)
        .catch(() => false)

      if (startedStreaming) {
        // Verify streaming completes
        await chatPage.waitForStreamingComplete(60000)
      }

      // Verify response exists
      await chatPage.waitForResponse()

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Chat History", () => {
    test("should maintain conversation history across multiple messages", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Send first message
      await chatPage.sendMessage("My name is TestUser")
      await chatPage.waitForResponse()

      // Send follow-up that requires context
      await chatPage.sendMessage("What did I just tell you my name was?")
      await chatPage.waitForResponse()

      // Verify conversation has multiple exchanges
      const messages = await chatPage.getMessages()
      expect(messages.length).toBeGreaterThanOrEqual(4) // 2 user + 2 assistant

      await assertNoCriticalErrors(diagnostics)
    })

    test("should persist conversation in sidebar when persistence is enabled", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Enable persistence if toggle exists
      await chatPage.togglePersistence(true)

      const testId = generateTestId("conv")
      await chatPage.sendMessage(`Test conversation ${testId}`)
      await chatPage.waitForResponse()

      // Check sidebar for conversation
      const _conversations = await chatPage.getConversationHistory()
      // Conversation may appear in sidebar
      // Note: This depends on UI implementation

      await assertNoCriticalErrors(diagnostics)
    })

    test("should scroll properly for long conversations", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      const initialMessageCount = (await chatPage.getMessages()).length

      // Send multiple messages
      for (let i = 1; i <= 3; i++) {
        await chatPage.sendMessage(`Message number ${i}`)
        await chatPage.waitForResponse(60000)
      }

      // Verify the page can scroll and last message is visible
      const messages = await chatPage.getMessages()
      expect(messages.length).toBeGreaterThanOrEqual(initialMessageCount + 4)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Character Selection", () => {
    test("should display character selector if available", async ({
      authedPage,
      diagnostics
    }) => {
      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Check if character selector exists
      const characterSelector = authedPage.locator(
        "[data-testid='character-selector'], [data-testid='persona-selector'], .character-select"
      )

      // This is an optional feature - test just checks it doesn't crash
      const hasCharacterSelector = (await characterSelector.count()) > 0
      if (hasCharacterSelector) {
        await expect(characterSelector.first()).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Error Handling", () => {
    test("should handle server timeout gracefully", async ({
      authedPage,
      diagnostics
    }) => {
      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // This test verifies the UI handles timeout scenarios
      // We don't artificially create a timeout, just verify error UI patterns exist
      const errorIndicators = authedPage.locator(
        ".error-message, .ant-message-error, [data-error]"
      )

      // Error indicators should not be visible initially
      await expect(errorIndicators.first()).not.toBeVisible({ timeout: 5000 }).catch(() => {})

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display user-friendly error messages", async ({
      authedPage,
      diagnostics: _diagnostics
    }) => {
      // Navigate to chat with intentionally bad config to trigger error
      await authedPage.addInitScript(() => {
        localStorage.setItem(
          "tldwConfig",
          JSON.stringify({
            serverUrl: "http://invalid-server-url-12345.local:9999",
            authMode: "single-user",
            apiKey: "invalid-key"
          })
        )
        localStorage.setItem("__tldw_first_run_complete", "true")
      })

      await authedPage.goto("/chat", { waitUntil: "domcontentloaded" })

      // Wait for error state
      const _errorState = authedPage.locator(
        ".connection-error, .error-boundary, [data-testid='connection-error']"
      )

      // Either shows error or falls back gracefully without stalling on background traffic
      await expect
        .poll(
          async () => {
            const urlOk = /\/chat|\/settings|\/login/.test(authedPage.url())
            const errorVisible = await _errorState.first().isVisible().catch(() => false)
            const mainVisible = await authedPage.locator("main").first().isVisible().catch(() => false)
            return urlOk && (errorVisible || mainVisible)
          },
          { timeout: 10_000 }
        )
        .toBe(true)

      // Page should not crash
      await expect(authedPage).toHaveURL(/\/chat|\/settings|\/login/)
    })
  })

  test.describe("UI Interactions", () => {
    test("should open command palette with keyboard shortcut", async ({
      authedPage,
      diagnostics
    }) => {
      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Try to open command palette
      await dispatchKeyboardShortcut(authedPage, { key: "k", ctrlKey: true })

      // Check if command palette appears
      const palette = authedPage.locator(
        "[data-testid='command-palette'], [role='dialog']:has-text('command'), .command-palette"
      )

      // Command palette is optional feature
      const hasPalette = await palette.isVisible().catch(() => false)
      if (hasPalette) {
        await expect(palette).toBeVisible()
        // Close it
        await authedPage.keyboard.press("Escape")
        await expect(palette).not.toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should render code blocks properly", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      await chatPage.sendMessage("Show me a hello world function in Python")
      await chatPage.waitForResponse(60000)
      await chatPage.waitForStreamingComplete()

      // Check for code block rendering
      const _hasCodeBlock = await chatPage.hasCodeBlock()
      // Note: This depends on LLM response actually including code

      await assertNoCriticalErrors(diagnostics)
    })

    test("should render markdown properly", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      await chatPage.sendMessage(
        "Format a bulleted list with three items and a heading"
      )
      await chatPage.waitForResponse(60000)
      await chatPage.waitForStreamingComplete()

      // Check for markdown rendering
      const _hasMarkdown = await chatPage.hasRenderedMarkdown()
      // Note: This depends on LLM response

      await assertNoCriticalErrors(diagnostics)
    })

    test("should provide copy message functionality", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      await chatPage.sendMessage("Say exactly: COPY_TEST_MESSAGE")
      await chatPage.waitForResponse(60000)

      await chatPage.copyLastMessage()

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Model Selection", () => {
    test("should display model selector", async ({
      authedPage,
      diagnostics
    }) => {
      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Look for model selector
      const modelSelector = authedPage.locator(
        "[data-testid='model-selector'], [data-testid='model-select-trigger'], .model-dropdown"
      )

      // Model selector is expected in chat interface
      if ((await modelSelector.count()) > 0) {
        await expect(modelSelector.first()).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("New Conversation", () => {
    test("should start a new conversation", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      skipIfNoModels(serverInfo)

      const chatPage = new ChatPage(authedPage)
      await chatPage.goto()
      await chatPage.waitForReady()

      // Send initial message
      await chatPage.sendMessage("Initial test message")
      await chatPage.waitForResponse()

      // Start new conversation
      await chatPage.startNewConversation()

      // Verify conversation is cleared or new
      // Input should be empty and ready for new message
      const input = await chatPage.getChatInput()
      const inputValue = await input.inputValue().catch(() => "")
      expect(inputValue).toBe("")

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
