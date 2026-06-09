import { type Locator, type Page, expect } from "@playwright/test"
import { waitForConnection } from "../helpers"

export class ExplainerPage {
  readonly page: Page
  readonly heading: Locator
  readonly outline: Locator
  readonly detail: Locator
  readonly goalTab: Locator
  readonly sourcesTab: Locator

  constructor(page: Page) {
    this.page = page
    this.heading = page.getByRole("heading", { name: "Explainer", exact: true })
    this.outline = page.getByRole("tree", { name: "Explainer outline" })
    this.detail = page.getByRole("region", { name: "Explainer detail" })
    this.goalTab = page.getByRole("tab", { name: "Goal" })
    this.sourcesTab = page.getByRole("tab", { name: "Sources" })
  }

  async goto(): Promise<void> {
    await this.page.goto("/explainer", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page).catch(() => {})
    await this.waitForReady()
  }

  async waitForReady(): Promise<void> {
    await expect(this.heading).toBeVisible({ timeout: 30_000 })
    await expect(this.goalTab).toBeVisible()
    await expect(this.sourcesTab).toBeVisible()
    await expect(this.outline).toBeVisible()
  }

  async createGoalSession(goal: string): Promise<void> {
    await this.goalTab.click()
    await this.page.getByLabel("Learning goal").fill(goal)
    await this.page.getByRole("button", { name: "Create Explainer" }).click()
  }

  async openSourcesTab(): Promise<void> {
    await this.sourcesTab.click()
    await expect(this.page.getByRole("region", { name: "Source setup" })).toBeVisible()
  }

  async searchSource(query: string): Promise<void> {
    await this.page.getByPlaceholder("Search media and notes").fill(query)
    await this.page.getByRole("button", { name: "Search sources" }).click()
  }

  async selectFirstSource(): Promise<void> {
    await this.page.getByRole("button", { name: /^Add / }).first().click()
  }

  async createSourceSession(): Promise<void> {
    await this.page.getByRole("button", { name: "Create Explainer" }).click()
  }

  async expandSelectedNode(): Promise<void> {
    await this.detail.getByRole("button", { name: "Expand node" }).click()
  }

  async selectNode(title: string | RegExp): Promise<void> {
    await this.outline.getByText(title).click()
  }

  async exportToChatbook(): Promise<void> {
    await this.page.getByRole("button", { name: "Export to Chatbook" }).click()
  }

  async expectNodeStatus(status: RegExp): Promise<void> {
    await expect(this.outline.getByText(status).first()).toBeVisible()
  }

  async expectCitation(text: RegExp): Promise<void> {
    await expect(this.detail.getByText(text)).toBeVisible()
  }
}
