/**
 * Page Object for Characters management workflow
 *
 * Key selectors discovered from Manager.tsx:
 *   data-testid="characters-page"          — root container
 *   data-testid="characters-new-button"    — "New character" button
 *   data-testid="characters-search-input"  — search field
 *   data-testid="characters-table-view"    — table view wrapper
 *   data-testid="characters-gallery-view"  — gallery view wrapper
 *   data-testid="characters-view-mode-segmented" — table/gallery toggle
 *   data-testid="characters-scope-segmented"     — active/deleted scope
 *
 * Create drawer uses Ant Design <Drawer> with <Form> fields:
 *   name, system_prompt, greeting, description, tags, avatar
 *   Submit button: htmlType="submit" with text "Create character"
 *
 * Delete: per-row button with aria-label "Delete character <name>",
 *   followed by a confirmDanger modal (OK button labelled "Delete").
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class CharactersPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // ── Navigation ──────────────────────────────────────────────────────

  async goto(): Promise<void> {
    await this.page.goto("/characters", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the characters page root or a feature-unavailable / empty state
    const ready = this.page.locator(
      '[data-testid="characters-page"], [data-testid="characters-new-button"], .ant-empty, .ant-result'
    )
    await ready.first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {})
  }

  // ── Locators ────────────────────────────────────────────────────────

  get newButton(): Locator {
    return this.page.locator('[data-testid="characters-new-button"]')
  }

  get searchInput(): Locator {
    return this.page.locator('[data-testid="characters-search-input"]')
  }

  get tableView(): Locator {
    return this.page.locator('[data-testid="characters-table-view"]')
  }

  get galleryView(): Locator {
    return this.page.locator('[data-testid="characters-gallery-view"]')
  }

  get createDrawer(): Locator {
    return this.page.locator(".ant-drawer")
  }

  // ── API Request Helpers ─────────────────────────────────────────────

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

  // ── Character CRUD ──────────────────────────────────────────────────

  /**
   * Open the "New character" drawer, fill the form, and submit.
   *
   * The form lives in an Ant Design <Drawer> and requires at minimum
   * `name` and `system_prompt` (labelled "Behavior / instructions").
   */
  async createCharacter(opts: {
    name: string
    description?: string
    systemPrompt?: string
  }): Promise<void> {
    // Open the create drawer
    await this.newButton.click()
    await this.createDrawer.waitFor({ state: "visible", timeout: 10_000 })

    // Fill name
    const nameInput = this.createDrawer.locator("#name, [id$='_name']").first()
    await nameInput.fill(opts.name)

    // Fill system prompt (required) — falls back to textarea by label
    const sysPrompt = this.createDrawer
      .locator("#system_prompt, [id$='_system_prompt']")
      .first()
    await sysPrompt.fill(
      opts.systemPrompt ?? `You are ${opts.name}. Be helpful and concise.`
    )

    // Fill description if provided
    if (opts.description) {
      const descInput = this.createDrawer
        .locator("#description, [id$='_description']")
        .first()
      await descInput.fill(opts.description)
    }

    // Wait for a stable, unobstructed submit button before clicking.
    const submitBtn = this.createDrawer.getByRole("button", {
      name: /create character/i
    })
    await submitBtn.click()

    // Creation is complete only when the drawer closes successfully.
    await this.createDrawer.waitFor({ state: "hidden", timeout: 10_000 })
  }

  /**
   * Delete a character by name from the table view.
   *
   * Each row has a delete button with aria-label="Delete character <name>".
   * Clicking it opens a confirm dialog; we click OK / "Delete".
   */
  async deleteCharacter(name: string): Promise<void> {
    // Click the delete button identified by its aria-label
    const deleteBtn = this.page.locator(
      `button[aria-label*="Delete character"][aria-label*="${name}"]`
    )
    await deleteBtn.first().click()

    // Confirm the danger modal
    const confirmBtn = this.page
      .locator(".ant-modal-confirm, .ant-modal")
      .getByRole("button", { name: /delete|ok|confirm/i })
    await confirmBtn.first().click()
  }

  /**
   * Search for a character by name.
   */
  async search(query: string): Promise<void> {
    await this.searchInput.fill(query)
    await expect(this.searchInput).toHaveValue(query, { timeout: 5_000 })
  }

  /**
   * Check if a character row (or card) with the given name is visible.
   */
  async isCharacterVisible(name: string): Promise<boolean> {
    const row = this.page.locator(
      `.ant-table-row:has-text("${name}"), [data-testid="characters-gallery-view"] :text("${name}")`
    )
    return row.first().isVisible().catch(() => false)
  }
}
