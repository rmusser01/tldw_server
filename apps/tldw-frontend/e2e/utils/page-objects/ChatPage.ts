/**
 * Page Object for Chat functionality
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { waitForConnection } from "../helpers"

export class ChatPage {
  readonly page: Page
  readonly header: Locator
  readonly messageInput: Locator
  readonly sendButton: Locator
  readonly messageList: Locator
  readonly sidebar: Locator
  readonly newChatButton: Locator
  readonly modelSelector: Locator
  private responseBaseline: {
    totalMessages: number
    assistantCount: number
    lastAssistantText: string
  } | null = null

  private allMessages(): Locator {
    return this.page.locator(
      "article[aria-label*='message'], [data-role], [data-message-role], .message"
    )
  }

  private assistantMessages(): Locator {
    return this.page.locator(
      "article[aria-label*='Assistant message'], [data-role='assistant'], [data-message-role='assistant'], .assistant-message"
    )
  }

  private normalizeMessageText(text: string): string {
    return text.replace(/▋/g, "").replace(/\s+/g, " ").trim()
  }

  private isMessageChrome(text: string): boolean {
    return /^(Mood:|Response complete$|Loading content(?:\.{3}|…)?$)/i.test(text)
  }

  private async getMessageBodyText(message: Locator): Promise<string> {
    const contentCandidates = message.locator(
      "p, pre, li, blockquote, h1, h2, h3, h4, h5, h6"
    )
    const candidateTexts = (await contentCandidates.allTextContents().catch(() => []))
      .map((text) => this.normalizeMessageText(text))
      .filter((text) => Boolean(text) && !this.isMessageChrome(text))

    if (candidateTexts.length > 0) {
      return candidateTexts.join("\n")
    }

    return (((await message.textContent().catch(() => "")) || "")
      .split(/\n+/)
      .map((text) => this.normalizeMessageText(text))
      .filter((text) => Boolean(text) && !this.isMessageChrome(text))
      .join(" "))
  }

  private async getLastAssistantText(
    assistantMessages: Locator,
    assistantCount: number
  ): Promise<string> {
    if (assistantCount === 0) {
      return ""
    }

    return this.getMessageBodyText(assistantMessages.last())
  }

  private async captureResponseBaseline(): Promise<void> {
    const allMessages = this.allMessages()
    const assistantMessages = this.assistantMessages()
    const [totalMessages, assistantCount] = await Promise.all([
      allMessages.count(),
      assistantMessages.count()
    ])

    this.responseBaseline = {
      totalMessages,
      assistantCount,
      lastAssistantText: await this.getLastAssistantText(assistantMessages, assistantCount),
    }
  }

  private async ensureModelSelected(): Promise<void> {
    const modelChip = this.page.getByTestId("model-selector").first()
    const chipVisible = await modelChip.isVisible().catch(() => false)
    if (chipVisible) {
      const chipLabel =
        (await modelChip.getAttribute("aria-label").catch(() => null))
        || (await modelChip.getAttribute("title").catch(() => null))
        || (await modelChip.textContent().catch(() => null))
        || ""
      if (chipLabel.trim() && !/select a model/i.test(chipLabel)) {
        return
      }
    }

    const selectModelTrigger = chipVisible
      ? modelChip
      : this.page.getByRole("button", { name: /select a model/i }).first()

    if (!(await selectModelTrigger.isVisible().catch(() => false))) {
      return
    }

    await selectModelTrigger.click()

    const pickerSurfaceCandidates = [
      this.page.getByRole("listbox").first(),
      this.page.getByRole("menu").first(),
    ]

    let pickerSurface: Locator | null = null
    await expect
      .poll(
        async () => {
          for (const candidate of pickerSurfaceCandidates) {
            if (await candidate.isVisible().catch(() => false)) {
              pickerSurface = candidate
              return true
            }
          }
          return false
        },
        { timeout: 10_000 }
      )
      .toBe(true)

    if (!pickerSurface) {
      throw new Error("Model picker surface did not appear after opening Select a model")
    }

    const choiceCandidates = [
      pickerSurface.getByRole("option").first(),
      pickerSurface.getByRole("menuitem").first(),
      this.page.getByRole("menuitem").first(),
    ]

    let firstChoice: Locator | null = null
    await expect
      .poll(
        async () => {
          for (const candidate of choiceCandidates) {
            if (await candidate.isVisible().catch(() => false)) {
              firstChoice = candidate
              return true
            }
          }
          return false
        },
        { timeout: 10_000 }
      )
      .toBe(true)

    if (!firstChoice) {
      throw new Error("Model picker did not render a selectable option")
    }

    await firstChoice.click()

    await expect(selectModelTrigger).not.toBeVisible({ timeout: 10_000 }).catch(() => {})
  }

  constructor(page: Page) {
    this.page = page
    this.header = page.locator("h1, h2, [role='heading']").filter({
      hasText: /start a new chat/i
    }).first()
    this.messageInput = page.locator("#textarea-message, [data-testid='chat-input']")
    this.sendButton = page.getByRole("button", { name: /send message|send/i }).first()
    this.messageList = page.getByRole("log", { name: /chat messages/i })
    this.sidebar = page.getByTestId("chat-sidebar")
    this.newChatButton = page
      .getByTestId("new-chat-button")
      .or(page.getByRole("button", { name: /new (saved )?chat|start chatting/i }).first())
    this.modelSelector = page.locator(
      "[data-testid='model-selector'], [data-testid='model-select-trigger']"
    ).first()
  }

  /**
   * Navigate to the chat page
   */
  async goto(): Promise<void> {
    await this.page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  /**
   * Wait for the chat page to be ready
   */
  async waitForReady(): Promise<void> {
    const waitForSurface = async () => {
      await Promise.race([
        this.page
          .getByRole("button", { name: /start chatting/i })
          .waitFor({ state: "visible", timeout: 20_000 }),
        this.page
          .getByPlaceholder(/type a message/i)
          .waitFor({ state: "visible", timeout: 20_000 }),
        this.messageList.waitFor({ state: "visible", timeout: 20_000 }),
      ])
    }

    await waitForSurface().catch(() => {})
    // Check for Next.js error overlay (e.g. rate_limited) and reload if found
    const hasErrorOverlay = await this.page.locator("nextjs-portal").count().catch(() => 0)
    if (hasErrorOverlay > 0) {
      await this.page.reload({ waitUntil: "domcontentloaded" })
    }
    // Dismiss any blocking modals
    await this.page.evaluate(() => {
      document.querySelectorAll('.ant-modal-root, .ant-modal-wrap, .ant-modal-mask').forEach(el => el.remove());
      document.querySelectorAll('nextjs-portal').forEach(el => { if (el.children.length > 0) el.remove(); });
    }).catch(() => {})
    await waitForSurface()
  }

  /**
   * Get the chat input (handles multiple possible selectors)
   */
  async getChatInput(): Promise<Locator> {
    // Try primary selector
    let input = this.page.locator("#textarea-message")
    if ((await input.count()) > 0) return input

    // Try testid selector
    input = this.page.getByTestId("chat-input")
    if ((await input.count()) > 0) return input

    // Try placeholder-based selector (actual: "Type a message... (/ commands, @ mentions)")
    return this.page.getByPlaceholder(/Type a message/i)
  }

  /**
   * Send a message in the chat
   */
  async sendMessage(message: string): Promise<void> {
    const input = await this.getChatInput()
    await expect(input).toBeVisible({ timeout: 15000 })

    // Click "Start chatting" if visible
    const startChat = this.page.getByRole("button", { name: /Start chatting/i })
    if ((await startChat.count()) > 0 && (await startChat.isVisible())) {
      await startChat.click()
    }

    await this.ensureModelSelected()
    await this.captureResponseBaseline()

    await input.fill(message)

    // Find and click send button (data-testid="chat-send")
    const sendButton = this.page.getByTestId("chat-send")

    if ((await sendButton.count()) > 0 && (await sendButton.isVisible())) {
      await sendButton.click()
    } else {
      // Fallback to keyboard submit
      await input.press("Enter")
    }
  }

  async selectCharacter(name: string): Promise<void> {
    const triggerCandidates = [
      this.page.getByTestId("character-selector").first(),
      this.page.getByRole("button", { name: /character/i }).first(),
      this.page.locator(".character-select").first(),
    ]

    let trigger: Locator | null = null
    await expect
      .poll(
        async () => {
          for (const candidate of triggerCandidates) {
            if (await candidate.isVisible().catch(() => false)) {
              trigger = candidate
              return true
            }
          }
          return false
        },
        { timeout: 10_000, message: "Timed out waiting for character selector trigger" }
      )
      .toBe(true)

    if (!trigger) {
      throw new Error("Character selector trigger not found")
    }

    await trigger.click()

    const pickerSurfaceCandidates = [
      this.page.getByTestId("character-selector-menu").first(),
      this.page.getByRole("listbox").first(),
      this.page.getByRole("menu").first(),
      this.page.locator(".character-select__menu").first(),
    ]

    let pickerSurface: Locator | null = null
    await expect
      .poll(
        async () => {
          for (const candidate of pickerSurfaceCandidates) {
            if (await candidate.isVisible().catch(() => false)) {
              pickerSurface = candidate
              return true
            }
          }
          return false
        },
        { timeout: 10_000, message: "Timed out waiting for character picker surface" }
      )
      .toBe(true)

    if (!pickerSurface) {
      throw new Error("Character picker surface not found")
    }

    const optionCandidates = [
      pickerSurface.getByRole("option", { name }).first(),
      pickerSurface.getByRole("menuitem", { name }).first(),
      pickerSurface.getByText(name, { exact: true }).first(),
    ]

    let option: Locator | null = null
    await expect
      .poll(
        async () => {
          for (const candidate of optionCandidates) {
            if (await candidate.isVisible().catch(() => false)) {
              option = candidate
              return true
            }
          }
          return false
        },
        { timeout: 10_000, message: `Timed out waiting for character option ${name}` }
      )
      .toBe(true)

    if (!option) {
      throw new Error(`Character option not found: ${name}`)
    }

    await option.click()

    await expect
      .poll(
        async () => {
          const selectedDisplay = this.page.getByTestId("character-selector").first()
          return (await selectedDisplay.textContent())?.includes(name) ?? false
        },
        {
          timeout: 5_000,
          message: `Timed out waiting for selected character ${name} to settle`
        }
      )
      .toBe(true)
  }

  /**
   * Wait for a response to appear
   */
  async waitForResponse(timeoutMs = 60000): Promise<void> {
    const allMessages = this.allMessages()
    const assistantMessages = this.assistantMessages()
    const [totalMessages, assistantCount] = await Promise.all([
      allMessages.count(),
      assistantMessages.count()
    ])
    const baseline = this.responseBaseline ?? {
      totalMessages,
      assistantCount,
      lastAssistantText: await this.getLastAssistantText(assistantMessages, assistantCount),
    }

    await expect
      .poll(
        async () => {
          const totalMessages = await allMessages.count()
          const assistantCount = await assistantMessages.count()
          const lastAssistant = assistantMessages.last()
          const text = await this.getMessageBodyText(lastAssistant)
          const assistantAdvanced =
            assistantCount > baseline.assistantCount || text !== baseline.lastAssistantText
          if (
            !assistantAdvanced
            || totalMessages <= baseline.totalMessages
          ) {
            return false
          }

          const isGenerating = await lastAssistant
            .getByText(/Generating response/i)
            .isVisible()
            .catch(() => false)
          const hasStopStreaming = await lastAssistant
            .getByRole("button", { name: /Stop streaming response|Stop Streaming/i })
            .isVisible()
            .catch(() => false)

          return Boolean(text) && !isGenerating && !hasStopStreaming
        },
        { timeout: timeoutMs, message: "Timed out waiting for a completed assistant response" }
      )
      .toBe(true)

    this.responseBaseline = null
  }

  /**
   * Wait for streaming to complete
   */
  async waitForStreamingComplete(timeoutMs = 60000): Promise<void> {
    // Wait for streaming indicator to disappear
    const streamingIndicator = this.page.locator(
      "[data-streaming='true'], .streaming, .typing-indicator"
    )

    // First wait for streaming to start (or skip if message already complete)
    const streamingStarted = await streamingIndicator.isVisible().catch(() => false)

    if (streamingStarted) {
      // Wait for streaming to complete
      await expect(streamingIndicator).not.toBeVisible({ timeout: timeoutMs })
    }
  }

  /**
   * Get all messages in the conversation
   */
  async getMessages(): Promise<Array<{ role: string; content: string }>> {
    const messages: Array<{ role: string; content: string }> = []

    const messageElements = this.page.locator(
      "article[aria-label*='message'], [data-role], [data-message-role], .message"
    )
    const count = await messageElements.count()

    for (let i = 0; i < count; i++) {
      const el = messageElements.nth(i)
      const ariaLabel = (await el.getAttribute("aria-label")) || ""
      const role =
        /assistant message/i.test(ariaLabel)
          ? "assistant"
          : /user message/i.test(ariaLabel)
            ? "user"
            : ((await el.getAttribute("data-role")) ||
              (await el.getAttribute("data-message-role")) ||
        "unknown"
            )
      const content = await this.getMessageBodyText(el)
      messages.push({ role, content })
    }

    return messages
  }

  /**
   * Start a new conversation
   */
  async startNewConversation(): Promise<void> {
    const newChatBtn = this.page.getByRole("button", {
      name: /new chat|new conversation/i
    })

    if ((await newChatBtn.count()) > 0) {
      await newChatBtn.first().click()
    } else {
      // Try keyboard shortcut
      await this.page.keyboard.press("Meta+n")
    }
  }

  /**
   * Open the command palette
   */
  async openCommandPalette(): Promise<void> {
    await this.page.keyboard.press("Meta+k")
    const palette = this.page.getByRole("dialog", { name: /command/i })
    await expect(palette).toBeVisible({ timeout: 5000 })
  }

  /**
   * Close the command palette
   */
  async closeCommandPalette(): Promise<void> {
    await this.page.keyboard.press("Escape")
  }

  /**
   * Select a model from the model selector
   */
  async selectModel(modelId: string): Promise<void> {
    // Click model selector to open dropdown
    const selector =
      this.modelSelector || this.page.getByTestId("model-select-trigger")

    if ((await selector.count()) > 0) {
      await selector.click()

      const modelChoiceCandidates = [
        this.page.getByRole("option", { name: new RegExp(modelId, "i") }).first(),
        this.page.getByRole("menuitem", { name: new RegExp(modelId, "i") }).first(),
      ]

      for (const candidate of modelChoiceCandidates) {
        if (await candidate.isVisible().catch(() => false)) {
          await candidate.click()
          return
        }
      }
    }
  }

  /**
   * Toggle chat history persistence
   */
  async togglePersistence(enabled: boolean): Promise<void> {
    const toggle = this.page.getByRole("switch", {
      name: /Save chat to history|Temporary chat/i
    })

    if ((await toggle.count()) === 0) return

    const isChecked =
      (await toggle.getAttribute("aria-checked")) === "true"

    if ((enabled && !isChecked) || (!enabled && isChecked)) {
      await toggle.click()
    }
  }

  /**
   * Get conversation history from sidebar
   */
  async getConversationHistory(): Promise<string[]> {
    const sidebarItems = this.page.locator(
      "[data-testid='chat-history-item'], .conversation-item"
    )
    const count = await sidebarItems.count()
    const titles: string[] = []

    for (let i = 0; i < count; i++) {
      const title = await sidebarItems.nth(i).textContent()
      if (title) titles.push(title.trim())
    }

    return titles
  }

  /**
   * Copy the last message to clipboard
   */
  async copyLastMessage(): Promise<void> {
    const assistantMessages = this.page.locator(
      "article[aria-label*='Assistant message']"
    )
    const lastAssistant = assistantMessages.last()

    if ((await lastAssistant.count()) > 0) {
      await lastAssistant.hover().catch(() => {})
      const lastCopy = lastAssistant.getByRole("button", {
        name: /copy to clipboard|copy/i
      })
      if ((await lastCopy.count()) > 0) {
        await lastCopy.click()
        return
      }
    }

    const fallbackCopy = this.page.locator(
      "[data-testid='copy-message'], button[aria-label*='copy' i]"
    ).last()
    if ((await fallbackCopy.count()) > 0) {
      await fallbackCopy.click()
    }
  }

  /**
   * Check if code block is rendered properly
   */
  async hasCodeBlock(): Promise<boolean> {
    const codeBlock = this.page.locator("pre code, .code-block, [data-code-block]")
    return (await codeBlock.count()) > 0
  }

  /**
   * Check if markdown is rendered properly
   */
  async hasRenderedMarkdown(): Promise<boolean> {
    // Check for common rendered markdown elements
    const markdownElements = this.page.locator(
      ".prose h1, .prose h2, .prose ul, .prose ol, .markdown-content"
    )
    return (await markdownElements.count()) > 0
  }
}
