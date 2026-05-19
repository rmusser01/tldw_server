/**
 * Page Object for World Books workflow
 *
 * Reflects the two-panel layout:
 *   Left panel  – list table with Edit + overflow "More actions" per row
 *   Right panel – detail panel with tabs: Entries, Attachments, Stats, Settings
 *
 * Toolbar actions (Import, Export All, Test Matching, Relationship Matrix,
 * Global Statistics) are accessed through a "Tools" dropdown.
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { waitForAppShell, waitForConnection } from "../helpers"

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export class WorldBooksPage {
  readonly page: Page

  constructor(page: Page) {
    this.page = page
  }

  // ── Navigation ──────────────────────────────────────────────────────

  async goto(): Promise<void> {
    await this.page.goto("/world-books", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async waitForReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    const container = this.page.locator(
      "[data-testid='world-books-two-panel'], [data-testid='world-books-stacked'], [data-testid='world-books-mobile'], [data-testid='world-books-manager'], .ant-empty"
    )
    try {
      await container.first().waitFor({ state: "visible", timeout: 20_000 })
    } catch (error) {
      const details = error instanceof Error ? error.message : String(error)
      throw new Error(`WorldBooksPage.waitForReady timed out waiting for the world-books layout: ${details}`)
    }
  }

  // ── Layout Containers ───────────────────────────────────────────────

  twoPanelContainer(): Locator {
    return this.page.getByTestId("world-books-two-panel")
  }

  detailPanel(): Locator {
    return this.page.locator("main[aria-label='World book detail']")
  }

  // ── API Intercept ───────────────────────────────────────────────────

  async waitForApiCall(
    urlPattern: string | RegExp,
    method: string = "GET"
  ): Promise<{ status: number; body: any }> {
    const response = await this.page.waitForResponse(
      (res) =>
        (typeof urlPattern === "string"
          ? res.url().includes(urlPattern)
          : urlPattern.test(res.url())) &&
        res.request().method() === method,
      { timeout: 15_000 }
    )
    const body = await response.json().catch(() => null)
    return { status: response.status(), body }
  }

  // ── World Book CRUD ─────────────────────────────────────────────────

  async clickNewWorldBook(): Promise<void> {
    const btn = this.page.getByTestId("world-books-new-button").or(
      this.page.getByRole("button", { name: /new world book|create|add/i })
    )
    await btn.first().click()
  }

  private worldBookDialog(title: RegExp): Locator {
    return this.page.getByRole("dialog", { name: title })
  }

  private async expectWorldBookRowVisible(name: string): Promise<void> {
    const row = this.page
      .getByRole("row")
      .filter({ hasText: new RegExp(escapeRegex(name), "i") })

    await this.searchWorldBooks(name)
    try {
      await expect(row).toBeVisible({ timeout: 10_000 })
      return
    } catch {
      await this.page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(this.page)
      await this.waitForReady()
      await this.searchWorldBooks(name)
      await expect(row).toBeVisible({ timeout: 30_000 })
    }
  }

  async fillWorldBookForm(fields: {
    name: string
    description?: string
    scanDepth?: number
    tokenBudget?: number
  }): Promise<void> {
    const modal = this.worldBookDialog(/create world book/i)
    await expect(modal).toBeVisible({ timeout: 10_000 })
    const nameInput = modal.getByLabel(/name/i).first()
    await nameInput.fill(fields.name)

    if (fields.description) {
      const descInput = modal.getByLabel(/description/i).first()
      await descInput.fill(fields.description)
    }
    if (fields.scanDepth !== undefined) {
      const depthInput = modal.getByLabel(/scan.depth/i).first()
      if (await depthInput.isVisible().catch(() => false)) {
        await depthInput.fill(String(fields.scanDepth))
      }
    }
    if (fields.tokenBudget !== undefined) {
      const budgetInput = modal.getByLabel(/token.budget/i).first()
      if (await budgetInput.isVisible().catch(() => false)) {
        await budgetInput.fill(String(fields.tokenBudget))
      }
    }
  }

  async submitWorldBookForm(title: RegExp = /create world book/i): Promise<void> {
    const modal = this.worldBookDialog(title)
    await expect(modal).toBeVisible({ timeout: 10_000 })
    const btn = modal.getByRole("button", { name: /create|save|ok|submit/i })
    await expect(btn).toBeEnabled({ timeout: 10_000 })
    await btn.click()
  }

  async createWorldBook(name: string, description?: string): Promise<void> {
    await this.clickNewWorldBook()
    await this.fillWorldBookForm({ name, description })
    const dialogTitle = /create world book/i

    for (let attempt = 1; attempt <= 4; attempt += 1) {
      const [apiResult] = await Promise.all([
        this.waitForApiCall(/\/api\/v1\/characters\/world-books\/?$/, "POST"),
        this.submitWorldBookForm(dialogTitle)
      ])

      if (apiResult.status < 300) {
        await this.worldBookDialog(dialogTitle).waitFor({ state: "hidden", timeout: 15_000 })
        await this.expectWorldBookRowVisible(name)
        return
      }

      if (apiResult.status !== 429 || attempt === 4) {
        throw new Error(`World book creation failed with status ${apiResult.status}`)
      }

      await expect(this.worldBookDialog(dialogTitle)).toBeVisible({ timeout: 10_000 })
      await this.page.waitForTimeout(1_000 * attempt)
    }
  }

  async searchWorldBooks(query: string): Promise<void> {
    // Toolbar search uses aria-label rather than data-testid
    const searchInput = this.page.getByRole("textbox", { name: /search world books/i }).or(
      this.page.getByTestId("world-books-search-input")
    )
    await expect(searchInput.first()).toBeVisible({ timeout: 10_000 })
    await searchInput.first().fill(query)
    await expect(searchInput.first()).toHaveValue(query, { timeout: 5_000 })
  }

  async findWorldBookRow(name: string): Promise<Locator> {
    await this.searchWorldBooks(name)
    return this.page
      .getByRole("row")
      .filter({ hasText: new RegExp(escapeRegex(name), "i") })
  }

  // ── Row Selection (two-panel) ───────────────────────────────────────

  /** Click a world book row to select it and open the detail panel */
  async selectWorldBookRow(name: string): Promise<void> {
    const row = await this.findWorldBookRow(name)
    await row.click()
  }

  // ── Detail Panel ────────────────────────────────────────────────────

  /** Verify the detail panel shows the expected world book name */
  async expectDetailPanelTitle(name: string): Promise<void> {
    const heading = this.detailPanel().getByRole("heading", {
      name: new RegExp(escapeRegex(name), "i")
    })
    await expect(heading).toBeVisible({ timeout: 10_000 })
  }

  /** Click a tab in the detail panel */
  async clickDetailTab(tabName: "Entries" | "Attachments" | "Stats" | "Settings"): Promise<void> {
    const tab = this.detailPanel().getByRole("tab", { name: tabName })
    await tab.click()
  }

  /** Verify a specific tab is active / visible in the detail panel */
  async expectDetailTabActive(tabName: string): Promise<void> {
    const tab = this.detailPanel().getByRole("tab", { name: tabName })
    await expect(tab).toHaveAttribute("aria-selected", "true", { timeout: 5_000 })
  }

  // ── Edit via Settings Tab ───────────────────────────────────────────

  /** Open the Settings tab in the detail panel (replaces old edit modal) */
  async openSettingsTab(name: string): Promise<void> {
    await this.selectWorldBookRow(name)
    await this.expectDetailPanelTitle(name)
    await this.clickDetailTab("Settings")
  }

  /** Fill the description field in the Settings tab form */
  async fillSettingsDescription(description: string): Promise<void> {
    const panel = this.detailPanel()
    const descInput = panel.getByLabel(/description/i).first()
    await descInput.fill(description)
  }

  /** Submit the Settings tab form */
  async submitSettingsForm(): Promise<void> {
    const panel = this.detailPanel()
    const btn = panel.getByRole("button", { name: /save|update|submit/i })
    await btn.click()
    await expect(btn).not.toHaveClass(/ant-btn-loading/, { timeout: 30_000 })
  }

  // ── Row Edit Button (opens detail panel with edit focus) ────────────

  async clickEditOnRow(name: string): Promise<void> {
    const row = await this.findWorldBookRow(name)
    const editBtn = row.getByRole("button", { name: /edit/i }).first()
    await editBtn.click()
  }

  // ── Row Overflow Menu Actions ───────────────────────────────────────

  /** Open the "More actions" overflow dropdown for a specific row */
  private async openRowOverflowMenu(name: string): Promise<Locator> {
    const row = await this.findWorldBookRow(name)
    const overflowBtn = row.getByRole("button", { name: /more actions/i })
    await overflowBtn.scrollIntoViewIfNeeded()
    await overflowBtn.click({ force: true })
    // Wait for the dropdown menu to be visible
    const menu = this.page.locator(".ant-dropdown:visible .ant-dropdown-menu")
    await menu.waitFor({ state: "visible", timeout: 5_000 })
    return menu
  }

  /** Click an item in the row overflow menu by its label text */
  private async clickRowOverflowItem(name: string, itemLabel: RegExp): Promise<void> {
    const menu = await this.openRowOverflowMenu(name)
    const item = menu.locator(".ant-dropdown-menu-item").filter({ hasText: itemLabel })
    await item.click()
  }

  async clickDeleteOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /delete/i)
  }

  async clickEntriesOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /manage entries/i)
  }

  async clickExportOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /export json/i)
  }

  async clickDuplicateOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /duplicate/i)
  }

  async clickAttachOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /quick attach characters/i)
  }

  async clickStatsOnRow(name: string): Promise<void> {
    await this.clickRowOverflowItem(name, /statistics/i)
  }

  async confirmDeletion(): Promise<void> {
    const confirmBtn = this.page.getByRole("button", { name: /ok|confirm|yes|delete/i }).last()
    await confirmBtn.click()
  }

  async clickUndoDelete(): Promise<void> {
    const undoBtn = this.page.getByRole("button", { name: /undo/i }).or(
      this.page.locator(".ant-notification").getByText(/undo/i)
    )
    await undoBtn.first().click()
  }

  pendingDeleteBanner(): Locator {
    return this.page.getByTestId("world-book-pending-delete-banner")
  }

  async cancelPendingDelete(): Promise<void> {
    await this.pendingDeleteBanner().getByRole("button", { name: /cancel pending/i }).click()
  }

  async getWorldBookNames(): Promise<string[]> {
    const rows = this.page.getByRole("row")
    const count = await rows.count()
    const names: string[] = []
    for (let i = 0; i < count; i++) {
      const text = await rows.nth(i).getByRole("cell").first().textContent().catch(() => null)
      if (text) names.push(text.trim())
    }
    return names
  }

  // ── Toolbar Tools Dropdown ──────────────────────────────────────────

  /** Open the "Tools" dropdown in the toolbar */
  private async openToolsDropdown(): Promise<void> {
    const toolsBtn = this.page.getByRole("button", { name: /tools/i }).or(
      this.page.locator("[aria-label='Tools']")
    )
    await toolsBtn.first().click()
  }

  /** Click an item in the Tools dropdown by label text */
  private async clickToolsItem(itemLabel: RegExp): Promise<void> {
    await this.openToolsDropdown()
    const menuItem = this.page.locator(".ant-dropdown:visible .ant-dropdown-menu-item").filter({
      hasText: itemLabel
    })
    await menuItem.click()
  }

  async clickImport(): Promise<void> {
    await this.clickToolsItem(/import json/i)
  }

  async clickExportAll(): Promise<void> {
    await this.clickToolsItem(/export all/i)
  }

  async clickTestMatching(): Promise<void> {
    await this.clickToolsItem(/test matching/i)
  }

  async clickRelationshipMatrix(): Promise<void> {
    await this.clickToolsItem(/relationship matrix/i)
  }

  async clickGlobalStatistics(): Promise<void> {
    await this.clickToolsItem(/global statistics/i)
  }

  // ── Entry Management (now in detail panel Entries tab) ──────────────

  /** Navigate to entries for a world book via the detail panel */
  async openEntriesTab(name: string): Promise<void> {
    await this.selectWorldBookRow(name)
    await this.expectDetailPanelTitle(name)
    await this.clickDetailTab("Entries")
  }

  async fillEntryForm(keywords: string, content: string, priority?: number): Promise<void> {
    const panel = this.detailPanel()
    const keywordsInput = panel.getByRole("combobox", { name: /keywords/i }).first()
    await keywordsInput.click()
    await keywordsInput.fill(keywords)
    await keywordsInput.press("Enter")

    const contentInput = panel.locator("textarea").first().or(
      panel.getByLabel(/content/i).first()
    )
    await contentInput.fill(content)

    if (priority !== undefined) {
      const priorityInput = panel.getByRole("spinbutton", { name: /priority/i }).first()
      if (await priorityInput.isVisible().catch(() => false)) {
        await priorityInput.fill(String(priority))
      }
    }
  }

  async submitEntry(): Promise<void> {
    const panel = this.detailPanel()
    const singleEntryButton = panel.getByRole("button", { name: /^add entry$/i })
    if (await singleEntryButton.isVisible().catch(() => false)) {
      await singleEntryButton.click()
      return
    }

    const bulkEntryButton = panel.getByRole("button", { name: /^add entries$/i })
    await bulkEntryButton.click()
  }

  async getEntryCount(): Promise<number> {
    const panel = this.detailPanel()
    const rows = panel.locator(".ant-table-row")
    return rows.count()
  }

  async toggleBulkAddMode(): Promise<void> {
    await this.detailPanel().getByLabel(/toggle bulk add mode/i).click()
  }

  async fillBulkText(text: string): Promise<void> {
    const textarea = this.detailPanel().getByLabel(/bulk entry input/i)
    await expect(textarea).toBeVisible({ timeout: 10_000 })
    await textarea.fill(text)
  }

  // ── Bulk Operations ─────────────────────────────────────────────────

  async selectEntryCheckboxes(count: number): Promise<void> {
    const checkboxes = this.detailPanel().locator(".ant-checkbox")
    const total = await checkboxes.count()
    for (let i = 0; i < Math.min(count, total); i++) {
      await checkboxes.nth(i).click()
    }
  }

  async clickBulkAction(action: string): Promise<void> {
    const btn = this.page.getByRole("button", { name: new RegExp(action, "i") })
    await btn.click()
  }

  // ── Attachments Tab ─────────────────────────────────────────────────

  /** Open the Attachments tab in the detail panel */
  async openAttachmentsTab(name: string): Promise<void> {
    await this.selectWorldBookRow(name)
    await this.expectDetailPanelTitle(name)
    await this.clickDetailTab("Attachments")
  }

  async attachToCharacter(characterName: string): Promise<void> {
    const panel = this.detailPanel()
    const dropdown = panel.locator(".ant-select").last()
    await dropdown.click()
    await this.page.locator(`.ant-select-item:has-text("${characterName}")`).click()
    const attachBtn = panel.getByRole("button", { name: /attach/i }).last()
    await attachBtn.click()
  }

  // ── Import File Upload ──────────────────────────────────────────────

  async uploadImportFile(filePath: string): Promise<void> {
    const fileInput = this.page.locator("input[type='file']")
    await fileInput.setInputFiles(filePath)
  }

  async toggleMergeOnConflict(): Promise<void> {
    const toggle = this.page.getByLabel(/merge/i).or(
      this.page.locator("[data-testid*='merge']")
    )
    await toggle.first().click()
  }

  // ── Statistics (via overflow menu or detail panel Stats tab) ────────

  async openStatsTab(name: string): Promise<void> {
    await this.selectWorldBookRow(name)
    await this.expectDetailPanelTitle(name)
    await this.clickDetailTab("Stats")
  }

  async getStatsTabContent(): Promise<string> {
    const panel = this.detailPanel()
    const statsContent = panel.locator("[data-testid='stats-tab-content']").or(
      panel.getByRole("tabpanel")
    )
    await statsContent.first().waitFor({ state: "visible", timeout: 10_000 })
    await expect
      .poll(async () => ((await statsContent.first().textContent()) ?? "").trim(), {
        timeout: 10_000
      })
      .not.toBe("Loading statistics...")
    return (await statsContent.first().textContent()) ?? ""
  }
}
