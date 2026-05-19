/**
 * Page Object for KnowledgeQA (RAG Search) workflow
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { dispatchKeyboardShortcut, waitForAppShell, waitForConnection } from "../helpers"

export class KnowledgeQAPage {
  readonly page: Page
  readonly searchShell: Locator
  readonly resultsShell: Locator

  constructor(page: Page) {
    this.page = page
    this.searchShell = page.getByTestId("knowledge-search-shell")
    this.resultsShell = page.getByTestId("knowledge-results-shell")
  }

  // ── Navigation ──────────────────────────────────────────────────────

  async goto(): Promise<void> {
    await this.page.goto("/knowledge", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async waitForReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    await this.searchShell.waitFor({ state: "visible", timeout: 20_000 })
    await expect(this.page.locator("#knowledge-search-input")).toBeVisible({
      timeout: 20_000
    })
  }

  // ── API Intercept ───────────────────────────────────────────────────

  async waitForRagSearch(): Promise<{
    status: number
    body: any
    requestBody: any
    url: string
  }> {
    const response = await this.page.waitForResponse(
      (res) =>
        res.request().method() === "POST" &&
        /\/api\/v1\/rag\/search(?:\/stream)?(?:\?|$)/i.test(res.url()),
      { timeout: 60_000 }
    )
    const body = await response.json().catch(() => null)
    let requestBody: any = null
    try {
      requestBody = response.request().postDataJSON()
    } catch {
      requestBody = null
    }

    return {
      status: response.status(),
      body,
      requestBody,
      url: response.url()
    }
  }

  // ── Search Bar ──────────────────────────────────────────────────────

  async getSearchInput(): Promise<Locator> {
    const exactInput = this.page.locator("#knowledge-search-input")
    if (await exactInput.isVisible().catch(() => false)) {
      return exactInput
    }

    const shellInput = this.searchShell.locator("input").first()
    await shellInput.waitFor({ state: "visible", timeout: 10_000 })
    return shellInput
  }

  async search(query: string): Promise<void> {
    const input = await this.getSearchInput()
    await input.fill(query)
    await input.press("Enter")
  }

  async focusSearchBar(): Promise<void> {
    const input = await this.getSearchInput()
    await input.focus()
  }

  // ── Results ─────────────────────────────────────────────────────────

  async waitForResults(timeoutMs = 30_000): Promise<void> {
    await this.resultsShell.waitFor({ state: "visible", timeout: timeoutMs })
    await expect
      .poll(
        async () => {
          const stopVisible = await this.page
            .getByRole("button", { name: /^Stop$/i })
            .isVisible()
            .catch(() => false)
          const hasAnswer = await this.page
            .getByTestId("knowledge-answer-content")
            .isVisible()
            .catch(() => false)
          const hasNoResults = await this.hasNoResults()
          const hasError = (await this.getErrorMessage()) !== null
          const hasSourceOnlyState = await this.hasSourceOnlyState()

          return !stopVisible && (hasAnswer || hasNoResults || hasError || hasSourceOnlyState)
        },
        { timeout: timeoutMs }
      )
      .toBe(true)
  }

  async getAnswerText(): Promise<string> {
    const answer = this.page.getByTestId("knowledge-answer-content")
    if (!(await answer.isVisible().catch(() => false))) {
      return ""
    }
    return ((await answer.textContent()) ?? "").trim()
  }

  async getSourceCount(): Promise<number> {
    const sources = this.getEvidencePanel().getByRole("listitem")
    return sources.count().catch(() => 0)
  }

  async clickCitation(index: number): Promise<void> {
    const citation = this.getCitationButtons().nth(index)
    await citation.click()
  }

  async hasNoResults(): Promise<boolean> {
    const recovery = this.page.getByRole("button", {
      name: /Broaden scope|Enable web|Show nearest matches/i
    })
    return recovery.first().isVisible().catch(() => false)
  }

  async hasSourceOnlyState(): Promise<boolean> {
    return this.page
      .getByText(/Found \d+ relevant source/i)
      .first()
      .isVisible()
      .catch(() => false)
  }

  // ── Settings Panel ──────────────────────────────────────────────────

  async openSettings(): Promise<void> {
    if (await this.getSettingsDialog().isVisible().catch(() => false)) {
      return
    }

    const searchShellSettings = this.searchShell.getByRole("button", {
      name: "Open settings"
    })
    if (await searchShellSettings.isVisible().catch(() => false)) {
      await searchShellSettings.click()
      return
    }

    const enableInSettings = this.page.getByRole("button", {
      name: /Enable in Settings/i
    })
    if (await enableInSettings.isVisible().catch(() => false)) {
      await enableInSettings.click()
      return
    }

    const fallbackSettings = this.page.getByRole("button", {
      name: "Open settings"
    })
    await fallbackSettings.last().click()
  }

  async selectPreset(preset: "fast" | "balanced" | "thorough"): Promise<void> {
    const presetRadio = this.getSettingsDialog().getByRole("radio", {
      name: new RegExp(`^${preset}\\b`, "i")
    })
    await presetRadio.click()
  }

  async toggleExpertMode(): Promise<void> {
    const toggle = this.getExpertModeToggle()
    await toggle.click()
  }

  async setSearchMode(mode: "fts" | "vector" | "hybrid"): Promise<void> {
    const select = this.getSettingsDialog().locator(
      "[data-testid*='search-mode'], [id*='search_mode'], select"
    )
    if (await select.first().isVisible().catch(() => false)) {
      await select.first().click()
      await this.page.locator(`.ant-select-item:has-text("${mode}")`).click()
    }
  }

  // ── Follow-Up Questions ─────────────────────────────────────────────

  async getFollowUpInput(): Promise<Locator> {
    const stickyInput = this.page.getByTestId("knowledge-followup-sticky")
    if (await stickyInput.isVisible().catch(() => false)) {
      return stickyInput
    }

    return this.page.locator(
      "input[placeholder*='follow-up' i], textarea[placeholder*='follow-up' i]"
    ).first()
  }

  async askFollowUp(question: string): Promise<void> {
    const input = await this.getFollowUpInput()
    await input.fill(question)
    await input.press("Enter")
  }

  async isFollowUpVisible(): Promise<boolean> {
    const input = await this.getFollowUpInput()
    return input.isVisible().catch(() => false)
  }

  // ── History Sidebar ─────────────────────────────────────────────────

  async toggleHistorySidebar(): Promise<void> {
    await this.ensureResearchLayout()

    if (await this.getHistorySidebar().isVisible().catch(() => false)) {
      return
    }

    const desktopExpand = this.page.getByRole("button", {
      name: /Expand history sidebar/i
    })
    if (await desktopExpand.isVisible().catch(() => false)) {
      await desktopExpand.click()
      return
    }

    const mobileOpen = this.page.getByTestId("knowledge-history-mobile-open")
    if (await mobileOpen.isVisible().catch(() => false)) {
      await mobileOpen.click()
    }
  }

  async getHistoryItems(): Promise<Locator> {
    return this.getHistorySidebar().locator(
      "[data-testid*='history-item'], [aria-current], button"
    )
  }

  async clickHistoryItem(index: number): Promise<void> {
    const items = await this.getHistoryItems()
    await items.nth(index).click()
  }

  // ── Keyboard Shortcuts ──────────────────────────────────────────────

  async pressNewSearch(): Promise<void> {
    await dispatchKeyboardShortcut(this.page, { key: "k", ctrlKey: true })
  }

  async pressSlashToFocus(): Promise<void> {
    await this.page.keyboard.press("/")
  }

  // ── Loading / Error States ──────────────────────────────────────────

  async isLoading(): Promise<boolean> {
    const loading = this.page.locator(
      "[data-testid='knowledge-search-loading-indicator'], .ant-spin-spinning, .searching"
    )
    return (await loading.count()) > 0 && (await loading.first().isVisible().catch(() => false))
  }

  async getErrorMessage(): Promise<string | null> {
    const error = this.resultsShell.getByText(
      /Search failed|Search timed out|Cannot reach server|No relevant documents found/i
    )
    if (await error.first().isVisible().catch(() => false)) {
      return (await error.first().textContent()) ?? null
    }
    return null
  }

  getSettingsDialog(): Locator {
    return this.page.getByRole("dialog", { name: /RAG Settings/i })
  }

  getHistorySidebar(): Locator {
    return this.page.locator(
      "[data-testid='knowledge-history-desktop-open'], [data-testid='knowledge-history-mobile-overlay']"
    )
  }

  getEvidencePanel(): Locator {
    return this.page.getByRole("complementary", { name: /Evidence panel/i })
  }

  getCitationButtons(): Locator {
    return this.page.locator("[data-knowledge-citation-index]")
  }

  getExpertModeToggle(): Locator {
    return this.getSettingsDialog().getByRole("switch", {
      name: /Expert Mode|Basic Mode/i
    })
  }

  async ensureResearchLayout(): Promise<void> {
    const openWorkspaceView = this.page.getByRole("button", {
      name: /Open workspace view/i
    })
    if (await openWorkspaceView.isVisible().catch(() => false)) {
      await openWorkspaceView.click()
    }
  }
}
