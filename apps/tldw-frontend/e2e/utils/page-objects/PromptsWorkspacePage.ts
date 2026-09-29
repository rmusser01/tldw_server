/**
 * Page Object for Prompts Workspace functionality
 *
 * Extends BasePage. Wraps the PromptsWorkspace component which includes:
 * - Segmented tab bar (Custom / Copilot / Studio / Trash)
 * - PromptListToolbar (search, new-prompt, import/export)
 * - PromptDrawer (create/edit form with name, system, user fields)
 * - PromptFullPageEditor (alternative editor for prompts)
 * - PromptActionsMenu (edit, delete, duplicate per-prompt)
 *
 * Note: Prompts are stored locally in IndexedDB (Dexie) first, then
 * optionally synced to the server via Prompt Studio endpoints.
 */
import { type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForConnection } from "../helpers"

export interface CreatePromptOptions {
  name: string
  template: string
}
export class PromptsWorkspacePage extends BasePage {
  /* ------------------------------------------------------------------ */
  /* Locators                                                            */
  /* ------------------------------------------------------------------ */

  /** Page heading */
  get heading(): Locator {
    return this.page.getByRole("heading", { name: /prompts/i }).first()
  }

  /** Segmented tab bar (Custom / Copilot / Studio / Trash) */
  get segmentedTabs(): Locator {
    return this.page.getByTestId("prompts-segmented")
  }

  /** "New prompt" button (data-testid="prompts-add") */
  get newPromptButton(): Locator {
    return this.page.getByTestId("prompts-add")
  }

  /** Search input (data-testid="prompts-search") */
  get searchInput(): Locator {
    return this.page.getByTestId("prompts-search")
  }

  /** Export button (data-testid="prompts-export") */
  get exportButton(): Locator {
    return this.page.getByTestId("prompts-export")
  }

  /** Import button (data-testid="prompts-import") */
  get importButton(): Locator {
    return this.page.getByTestId("prompts-import")
  }

  /* --- Drawer locators --- */

  /** Prompt name input inside the drawer */
  get drawerNameInput(): Locator {
    return this.page.getByTestId("prompt-drawer-name")
  }

  /** System prompt textarea inside the drawer */
  get drawerSystemInput(): Locator {
    return this.page.getByTestId("prompt-drawer-system")
  }

  /** User prompt textarea inside the drawer */
  get drawerUserInput(): Locator {
    return this.page.getByTestId("prompt-drawer-user")
  }

  /** Save button inside the drawer (ant Button type="primary") */
  get drawerSaveButton(): Locator {
    // The drawer footer has a primary button with text "Save" or "Saving..."
    return this.page.locator(".ant-drawer-footer").getByRole("button").filter({ hasText: /save|saving/i })
  }

  /** Cancel button inside the drawer */
  get drawerCancelButton(): Locator {
    return this.page.locator(".ant-drawer-footer").getByRole("button", { name: /cancel/i })
  }

  /* --- Full-page editor locators --- */

  /** Full-page editor container */
  get fullPageEditor(): Locator {
    return this.page.getByTestId("prompt-full-page-editor")
  }

  /* ------------------------------------------------------------------ */
  /* BasePage overrides                                                  */
  /* ------------------------------------------------------------------ */

  async goto(): Promise<void> {
    await this.page.goto("/prompts", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  /**
   * Navigate to the Studio tab within the Prompts page.
   */
  async gotoStudioTab(): Promise<void> {
    await this.page.goto("/prompts?tab=studio", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await expect(this.heading).toBeVisible({ timeout: 20_000 })
    // Either the new-prompt button or the segmented tabs should be visible
    const hasNewPrompt = await this.newPromptButton.isVisible().catch(() => false)
    const hasSegmented = await this.segmentedTabs.isVisible().catch(() => false)
    if (!hasNewPrompt && !hasSegmented) {
      // Wait for at least the search input
      await expect(this.searchInput).toBeVisible({ timeout: 10_000 })
    }
  }

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "New prompt",
        locator: this.newPromptButton,
        expectation: {
          type: "state_change",
          stateCheck: async () => {
            // Full-page editor or drawer should appear
            const editorVisible = await this.fullPageEditor.isVisible().catch(() => false)
            const drawerVisible = await this.drawerNameInput.isVisible().catch(() => false)
            return { editorVisible, drawerVisible }
          },
        },
      },
    ]
  }

  /* ------------------------------------------------------------------ */
  /* High-level actions                                                  */
  /* ------------------------------------------------------------------ */

  /**
   * Create a new prompt via the full-page editor or drawer.
   *
   * Clicks "New prompt", fills name + template in whichever editor opens,
   * then saves.
   */
  async createPrompt(opts: CreatePromptOptions): Promise<void> {
    await this.newPromptButton.click()

    // The full-page editor is lazy-loaded. Wait for either supported editor
    // instead of treating isVisible() as a wait and racing the chunk load.
    await expect
      .poll(
        async () => {
          const fullEditorVisible = await this.fullPageEditor.isVisible().catch(() => false)
          const drawerVisible = await this.drawerNameInput.isVisible().catch(() => false)
          return fullEditorVisible || drawerVisible
        },
        { timeout: 10_000 }
      )
      .toBe(true)

    // The full-page editor opens by default; fall back to the drawer.
    const fullEditorVisible = await this.fullPageEditor.isVisible().catch(() => false)

    if (fullEditorVisible) {
      // Full-page editor path
      const nameInput = this.page.getByTestId("prompt-full-page-editor").locator("input").first()
      await expect(nameInput).toBeVisible({ timeout: 5_000 })
      await nameInput.fill(opts.name)

      const systemTextarea = this.page.getByTestId("full-editor-system-prompt")
      if (await systemTextarea.isVisible().catch(() => false)) {
        await systemTextarea.fill(opts.template)
      } else {
        const userTextarea = this.page.getByTestId("full-editor-user-prompt")
        await userTextarea.fill(opts.template)
      }

      // Click save in the full-page editor
      const saveBtn = this.page.getByTestId("prompt-full-page-editor")
        .getByRole("button", { name: /save/i }).first()
      const promptAddedNotices = this.page
        .locator(".ant-notification-notice-success")
        .filter({ hasText: /Prompt Added/i })
      await expect(promptAddedNotices).toHaveCount(0, { timeout: 10_000 })
      await saveBtn.click()
      await expect(promptAddedNotices).toHaveCount(1, { timeout: 30_000 })

      await this.page.goto("/prompts", { waitUntil: "domcontentloaded" })
      await this.assertPageReady()
      await expect(this.fullPageEditor).toBeHidden({ timeout: 5_000 })
    } else {
      // Drawer path (fallback)
      await expect(this.drawerNameInput).toBeVisible({ timeout: 10_000 })
      await this.drawerNameInput.locator("input").fill(opts.name)

      const userInput = this.drawerUserInput
      if (await userInput.isVisible().catch(() => false)) {
        await userInput.locator("textarea").first().fill(opts.template)
      } else {
        const systemInput = this.drawerSystemInput
        await systemInput.locator("textarea").first().fill(opts.template)
      }

      const promptAddedNotices = this.page
        .locator(".ant-notification-notice-success")
        .filter({ hasText: /Prompt Added/i })
      await expect(promptAddedNotices).toHaveCount(0, { timeout: 10_000 })
      await this.drawerSaveButton.click()
      await expect(promptAddedNotices).toHaveCount(1, { timeout: 30_000 })
    }
  }

  /**
   * Delete a prompt by name via its actions menu.
   *
   * Finds the prompt row/card, opens the "more" menu, and clicks Delete.
   */
  async deletePrompt(name: string): Promise<void> {
    // Close the full-page editor if it is still open (e.g. after createPrompt)
    const editorOpen = await this.fullPageEditor.isVisible().catch(() => false)
    if (editorOpen) {
      // Navigate directly to the prompts list page to escape the editor
      await this.page.goto("/prompts", { waitUntil: "domcontentloaded" })
      await this.assertPageReady()
    }

    // Find the prompt text in the list to identify its row
    const promptText = this.page.getByText(name, { exact: false }).first()
    await expect(promptText).toBeVisible({ timeout: 10_000 })

    // Click the prompt row to open the details drawer/panel
    await promptText.click()
    await expect
      .poll(
        async () => {
          const detailDeleteVisible = await this.page.getByRole("button", { name: /^delete$/i }).first().isVisible().catch(() => false)
          const moreVisible = await this.page.locator("[data-testid^='prompt-more-']").first().isVisible().catch(() => false)
          return detailDeleteVisible || moreVisible
        },
        { timeout: 5_000 }
      )
      .toBe(true)

    // The "Prompt details" panel should open with a Delete button.
    // Try to find "Delete" in the details panel first.
    const detailsDeleteBtn = this.page.getByRole("button", { name: /^delete$/i })
    if (await detailsDeleteBtn.first().isVisible({ timeout: 3_000 }).catch(() => false)) {
      await detailsDeleteBtn.first().click()
    } else {
      // Fallback: look for the more-actions button
      const moreButton = this.page.locator("[data-testid^='prompt-more-']").first()
      if (await moreButton.isVisible({ timeout: 2_000 }).catch(() => false)) {
        await moreButton.click()
        const deleteItem = this.page.getByText(/^delete$/i).last()
        await expect(deleteItem).toBeVisible({ timeout: 5_000 })
        await deleteItem.click()
      }
    }

    // Handle confirmation modal (useConfirmDanger / ant-modal-confirm)
    const confirmModal = this.page.locator('.ant-modal-confirm')
    await confirmModal.waitFor({ state: 'visible', timeout: 5_000 }).catch(() => {})
    const confirmDeleteBtn = confirmModal.getByRole("button", { name: /^delete$/i })
    if (await confirmDeleteBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await confirmDeleteBtn.click({ force: true })
    } else {
      // Fallback: any confirm/ok button in the modal
      const fallbackBtn = confirmModal.getByRole("button", { name: /ok|confirm|yes/i })
      if (await fallbackBtn.first().isVisible({ timeout: 2_000 }).catch(() => false)) {
        await fallbackBtn.first().click({ force: true })
      }
    }
  }

  /**
   * Search prompts using the toolbar search input.
   */
  async searchPrompts(query: string): Promise<void> {
    await expect(this.searchInput).toBeVisible({ timeout: 10_000 })
    await this.searchInput.locator("input").fill(query)
    await expect(this.searchInput.locator("input")).toHaveValue(query, { timeout: 5_000 })
  }

  /**
   * Assert a prompt with the given name is visible in the list.
   */
  async assertPromptVisible(name: string): Promise<void> {
    const item = this.page.getByText(name, { exact: false })
    await expect(item.first()).toBeVisible({ timeout: 15_000 })
  }

  /**
   * Assert a prompt with the given name is NOT visible in the list.
   */
  async assertPromptNotVisible(name: string): Promise<void> {
    const item = this.page.getByText(name, { exact: false })
    await expect(item).toBeHidden({ timeout: 10_000 })
  }

  /**
   * Switch to a specific segment tab (custom, copilot, studio, trash).
   */
  async switchTab(tab: "custom" | "copilot" | "studio" | "trash"): Promise<void> {
    const labels: Record<string, RegExp> = {
      custom: /custom prompts/i,
      copilot: /copilot prompts/i,
      studio: /studio/i,
      trash: /trash/i,
    }
    const segmented = this.segmentedTabs
    await segmented.getByText(labels[tab]).click()
    await expect(segmented.getByText(labels[tab])).toBeVisible({ timeout: 5_000 })
  }
}
