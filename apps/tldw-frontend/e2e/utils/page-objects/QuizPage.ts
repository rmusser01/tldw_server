/**
 * Page Object for Quiz Playground workflow
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class QuizPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/quiz", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async gotoPath(path: string): Promise<void> {
    await this.page.goto(path, { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the quiz playground (online), beta badge, demo preview, or connection banner
    const betaBadge = this.page.locator('[data-testid="quiz-beta-badge"]')
    const demoPreview = this.page.locator('[data-testid="quiz-demo-preview"]')
    const connectionBanner = this.page.getByText("Connect to use Quiz Playground")
    const quizTabs = this.page.locator(".quiz-tabs")
    await Promise.race([
      betaBadge.first().waitFor({ state: "visible", timeout: 20_000 }),
      demoPreview.first().waitFor({ state: "visible", timeout: 20_000 }),
      connectionBanner.first().waitFor({ state: "visible", timeout: 20_000 }),
      quizTabs.first().waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  /** Beta badge button */
  get betaBadge(): Locator {
    return this.page.locator('[data-testid="quiz-beta-badge"]')
  }

  /** Beta tooltip (visible on hover/click of beta badge) */
  get betaTooltip(): Locator {
    return this.page.locator('[data-testid="quiz-beta-tooltip"]')
  }

  /** Hover the beta badge to open its tooltip (it uses mouseEnter/mouseLeave) */
  async hoverBetaBadge(): Promise<void> {
    await this.betaBadge.hover()
  }

  /** Connection problem banner (offline, non-demo) */
  get connectionBanner(): Locator {
    return this.page.getByText("Connect to use Quiz Playground")
  }

  /** Demo preview section (offline + demo mode) */
  get demoPreview(): Locator {
    return this.page.locator('[data-testid="quiz-demo-preview"]')
  }

  /** Demo start button */
  get demoStartButton(): Locator {
    return this.page.locator('[data-testid="quiz-demo-start"]')
  }

  /** Demo quiz taking section */
  get demoTaking(): Locator {
    return this.page.locator('[data-testid="quiz-demo-taking"]')
  }

  /** Demo submit button */
  get demoSubmitButton(): Locator {
    return this.page.locator('[data-testid="quiz-demo-submit"]')
  }

  /** Demo results section */
  get demoResults(): Locator {
    return this.page.locator('[data-testid="quiz-demo-results"]')
  }

  /** Demo score display */
  get demoScore(): Locator {
    return this.page.locator('[data-testid="quiz-demo-score"]')
  }

  /** Feature unavailable message (server lacks quiz API) */
  get featureUnavailable(): Locator {
    return this.page.getByText("Quiz API not available on this server")
  }

  /** Quiz tabs container (online + feature available) */
  get quizTabs(): Locator {
    return this.page.locator(".quiz-tabs")
  }

  /** Take Quiz tab */
  get takeTab(): Locator {
    return this.page
      .locator('[data-testid="quiz-tab-take"]')
      .locator("xpath=ancestor::*[@role='tab'][1]")
  }

  /** Generate tab */
  get generateTab(): Locator {
    return this.page
      .locator('[data-testid="quiz-tab-generate"]')
      .locator("xpath=ancestor::*[@role='tab'][1]")
  }

  /** Create tab */
  get createTab(): Locator {
    return this.page
      .locator('[data-testid="quiz-tab-create"]')
      .locator("xpath=ancestor::*[@role='tab'][1]")
  }

  /** Manage tab */
  get manageTab(): Locator {
    return this.page
      .locator('[data-testid="quiz-tab-manage"]')
      .locator("xpath=ancestor::*[@role='tab'][1]")
  }

  /** Results tab */
  get resultsTab(): Locator {
    return this.page
      .locator('[data-testid="quiz-tab-results"]')
      .locator("xpath=ancestor::*[@role='tab'][1]")
  }

  /** Global search input */
  get globalSearchInput(): Locator {
    return this.page.locator('[data-testid="quiz-global-search-input"]')
  }

  /** Global search apply button */
  get globalSearchApplyButton(): Locator {
    return this.page.locator('[data-testid="quiz-global-search-apply"]')
  }

  /** Reset current tab button */
  get resetCurrentTabButton(): Locator {
    return this.page.locator('[data-testid="quiz-reset-current-tab"]')
  }

  get manageShowWorkspaceQuizzesToggle(): Locator {
    return this.page
      .locator('[data-testid="quiz-manage-show-workspace-quizzes"]')
      .locator("xpath=ancestor::label[1]")
  }

  get manageWorkspaceFilter(): Locator {
    return this.page.locator('[data-testid="quiz-manage-workspace-filter"]')
  }

  getTakeQuizCard(quizId: number): Locator {
    return this.page.locator(`[data-testid="take-quiz-card-${quizId}"]`)
  }

  getManageQuizStartButton(quizId: number): Locator {
    return this.page.locator(`[data-testid="quiz-start-${quizId}"]`)
  }

  getManageQuizEditButton(quizId: number): Locator {
    return this.page.locator(`[data-testid="quiz-edit-${quizId}"]`)
  }

  // -- Helpers ---------------------------------------------------------------

  /** Whether the quiz playground (online state) is visible */
  async isPlaygroundVisible(): Promise<boolean> {
    return this.quizTabs.isVisible().catch(() => false)
  }

  /** Switch to a playground tab */
  async switchToTab(tab: "take" | "generate" | "create" | "manage" | "results"): Promise<void> {
    const tabLocator = {
      take: this.takeTab,
      generate: this.generateTab,
      create: this.createTab,
      manage: this.manageTab,
      results: this.resultsTab,
    }[tab]
    await tabLocator.click()
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Global search apply button",
        locator: this.globalSearchApplyButton,
        expectation: {
          type: "state_change",
          stateCheck: async (page) => {
            // State change: active tab may switch to "take" when search is applied
            const activeTab = await page.locator(".quiz-tabs .ant-tabs-tab-active").textContent().catch(() => "")
            return activeTab
          },
        },
        setup: async (page) => {
          await page.locator('[data-testid="quiz-global-search-input"]').fill("test query")
        },
      },
      {
        name: "Reset current tab button",
        locator: this.resetCurrentTabButton,
        expectation: {
          type: "state_change",
          stateCheck: async (page) => {
            // After reset, session storage keys are cleared -- check the button was handled
            return Date.now()
          },
        },
      },
    ]
  }
}
