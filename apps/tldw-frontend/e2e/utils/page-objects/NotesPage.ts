/**
 * Page Object for Notes Manager functionality
 *
 * Extends BasePage. Wraps the NotesManagerPage component which includes:
 * - NotesSidebar (with NotesToolbar search/filter and NotesListPanel)
 * - NotesEditorPane (title input, content textarea, save/delete actions)
 * - NotesEditorHeader (save button, overflow menu with delete)
 */
import { type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { expectApiCall } from "../api-assertions"
import { waitForConnection } from "../helpers"

export interface CreateNoteOptions {
  title: string
  content: string
}

export interface CompleteNotesStudioSetupOptions {
  template: "Cornell"
  handwriting: "Accented"
  excerptText: string
}

export class NotesPage extends BasePage {
  /* ------------------------------------------------------------------ */
  /* Locators                                                            */
  /* ------------------------------------------------------------------ */

  /** "New note" button (aria-label "New note") */
  get newNoteButton(): Locator {
    return this.page.getByRole("button", { name: /new note/i })
  }

  /** Search input in the sidebar (placeholder "Search titles & content...") */
  get searchInput(): Locator {
    return this.page.getByPlaceholder(/search titles/i)
  }

  /** Title input in the editor pane (placeholder "Title") */
  get titleInput(): Locator {
    return this.page.getByPlaceholder(/^title$/i)
  }

  /** Content textarea/editor in the editor pane */
  get contentTextarea(): Locator {
    return this.page.getByPlaceholder(/write your note here/i).or(
      this.page.getByLabel(/note content/i)
    )
  }

  /** Save button (data-testid "notes-save-button") */
  get saveButton(): Locator {
    return this.page.getByTestId("notes-save-button")
  }

  /** Markdown input mode toggle (data-testid "notes-input-mode-markdown") */
  get markdownModeButton(): Locator {
    return this.page.getByTestId("notes-input-mode-markdown")
  }

  /** Overflow / "More actions" button (data-testid "notes-overflow-menu-button") */
  get overflowMenuButton(): Locator {
    return this.page.getByTestId("notes-overflow-menu-button")
  }

  /** Editor region (data-testid "notes-editor-region") */
  get editorRegion(): Locator {
    return this.page.getByTestId("notes-editor-region")
  }

  /** Notes Studio create modal (data-testid "notes-studio-create-modal") */
  get notesStudioCreateModal(): Locator {
    return this.page.getByTestId("notes-studio-create-modal")
  }

  /** Notes Studio view shell (data-testid "notes-studio-view") */
  get notesStudioView(): Locator {
    return this.page.getByTestId("notes-studio-view")
  }

  /** Print / Save as PDF menu item in the export submenu. */
  get printExportMenuItem(): Locator {
    return this.page.getByRole("menuitem", { name: /print \/ save as pdf/i })
  }

  /* ------------------------------------------------------------------ */
  /* BasePage overrides                                                  */
  /* ------------------------------------------------------------------ */

  async goto(): Promise<void> {
    await this.page.goto("/notes", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    // Wait for the notes list region or the search input to appear
    const listRegion = this.page.getByTestId("notes-list-region")
    const searchInput = this.searchInput
    await Promise.race([
      listRegion.waitFor({ state: "visible", timeout: 20_000 }),
      searchInput.waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
    await expect
      .poll(
        async () => {
          const signals = await Promise.all([
            this.editorRegion.isVisible().catch(() => false),
            this.titleInput.isVisible().catch(() => false),
            this.page.getByText(/no notes yet|select a note|create your first note/i).first().isVisible().catch(() => false),
          ])
          return signals.some(Boolean)
        },
        { timeout: 10_000 }
      )
      .toBe(true)
  }

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "New note",
        locator: this.newNoteButton,
        expectation: {
          type: "state_change",
          stateCheck: async () => {
            // Title input should appear or change after clicking new note
            return this.titleInput.inputValue().catch(() => "__absent__")
          },
        },
      },
    ]
  }

  /* ------------------------------------------------------------------ */
  /* High-level actions                                                  */
  /* ------------------------------------------------------------------ */

  /**
   * Create a new note: click "New note", fill title + content, save.
   */
  async createNote(opts: CreateNoteOptions): Promise<void> {
    await this.newNoteButton.click()

    // Wait for the editor to be ready
    await expect(this.titleInput).toBeVisible({ timeout: 10_000 })

    await this.titleInput.fill(opts.title)

    // The content textarea may be a plain <textarea> or a contenteditable div (WYSIWYG).
    // Prefer the textarea; fall back to the contenteditable editor.
    const textarea = this.contentTextarea
    if ((await textarea.count()) > 0 && (await textarea.isVisible())) {
      await textarea.fill(opts.content)
    } else {
      // WYSIWYG fallback
      const wysiwyg = this.page.getByTestId("notes-wysiwyg-editor")
      await wysiwyg.click()
      await this.page.keyboard.type(opts.content)
    }

    const createApiCall = expectApiCall(this.page, {
      method: "POST",
      url: /\/api\/v1\/notes\/?$/,
      bodyContains: {
        title: opts.title,
        content: opts.content,
      },
    }, 30_000)

    await this.saveButton.click()

    // Wait for save to complete (button stops showing loading state)
    await expect(this.saveButton).toBeEnabled({ timeout: 15_000 })

    const { response } = await createApiCall
    expect(response.status()).toBeLessThan(400)
  }

  /**
   * Ensure the editor is in Markdown mode before selecting text.
   */
  async ensureMarkdownMode(): Promise<void> {
    const wysiwygEditorVisible = await this.page
      .getByTestId("notes-wysiwyg-editor")
      .isVisible()
      .catch(() => false)
    if (wysiwygEditorVisible) {
      await this.markdownModeButton.click()
      await expect(this.contentTextarea).toBeVisible({ timeout: 10_000 })
    }
  }

  /**
   * Select a substring inside the Markdown textarea and fail fast if it is missing.
   */
  async selectMarkdownExcerpt(excerpt: string): Promise<void> {
    await this.ensureMarkdownMode()
    await expect(this.contentTextarea).toBeVisible({ timeout: 10_000 })
    await this.contentTextarea.focus()

    await this.contentTextarea.evaluate((element, targetExcerpt) => {
      const textarea = element as HTMLTextAreaElement
      const start = textarea.value.indexOf(targetExcerpt)
      if (start < 0) {
        throw new Error(`Could not find excerpt "${targetExcerpt}" in the Markdown editor.`)
      }

      textarea.focus()
      textarea.setSelectionRange(start, start + targetExcerpt.length)
      textarea.dispatchEvent(new Event("select", { bubbles: true }))
    }, excerpt)
  }

  /**
   * Open the Notes Studio entry from the overflow menu.
   */
  async openNotesStudio(): Promise<void> {
    await this.overflowMenuButton.click()
    const menuItem = this.page.getByRole("menuitem", { name: /^notes studio$/i })
    await expect(menuItem).toBeVisible({ timeout: 10_000 })
    await menuItem.click()
    await expect(this.notesStudioCreateModal).toBeVisible({ timeout: 10_000 })
  }

  /**
   * Complete the Notes Studio modal and wait for the derived studio view.
   */
  async completeNotesStudioSetup(opts: CompleteNotesStudioSetupOptions): Promise<void> {
    const modal = this.notesStudioCreateModal
    await expect(modal).toBeVisible({ timeout: 10_000 })

    await modal.getByRole("radio", { name: opts.template }).click()
    await modal.getByRole("radio", { name: opts.handwriting }).click()
    const deriveApiCall = expectApiCall(this.page, {
      method: "POST",
      url: /\/api\/v1\/notes\/studio\/derive$/,
      bodyContains: {
        template_type: "cornell",
        handwriting_mode: "accented",
        excerpt_text: opts.excerptText,
      },
    }, 30_000)

    await modal.getByRole("button", { name: /create notes studio note/i }).click()

    const { response } = await deriveApiCall
    expect(response.status()).toBeLessThan(400)

    await expect(this.notesStudioView).toBeVisible({ timeout: 20_000 })
    await expect(this.page.getByTestId("notes-studio-template-cornell")).toBeVisible({
      timeout: 10_000
    })
    await expect(modal).toBeHidden({ timeout: 10_000 }).catch(() => {})
  }

  /**
   * Trigger the print export flow from the Notes overflow menu.
   */
  async triggerPrintExport(): Promise<void> {
    await this.overflowMenuButton.click()

    // The submenu's accessible name includes its expand icon ("Export right").
    const exportMenuItem = this.page.getByRole("menuitem", { name: /^export\b/i })
    await expect(exportMenuItem).toBeVisible({ timeout: 10_000 })
    await exportMenuItem.hover()

    await expect(this.printExportMenuItem).toBeVisible({ timeout: 10_000 })
    await this.printExportMenuItem.click()
  }

  /**
   * Type a search query in the toolbar search input and press Enter.
   */
  async searchNotes(query: string): Promise<void> {
    await expect(this.searchInput).toBeVisible({ timeout: 10_000 })
    await this.searchInput.fill(query)
    await this.searchInput.press("Enter")
    await expect(this.searchInput).toHaveValue(query, { timeout: 5_000 })
  }

  /**
   * Assert that a note with the given title is visible in the list panel.
   */
  async assertNoteVisible(title: string): Promise<void> {
    const noteItem = this.page.getByText(title, { exact: false })
    await expect(noteItem.first()).toBeVisible({ timeout: 15_000 })
  }

  /**
   * Assert that a note with the given title is NOT visible in the main notes list.
   * (It may still appear in "RECENT NOTES" section after soft-delete.)
   */
  async assertNoteNotVisible(title: string): Promise<void> {
    const noteDeleted = this.page.getByText(/note deleted/i)
    const noteHidden = this.page.getByText(title, { exact: false })
    await expect
      .poll(
        async () => {
          const toastVisible = await noteDeleted.isVisible().catch(() => false)
          const noteVisible = await noteHidden.isVisible().catch(() => false)
          return toastVisible || !noteVisible
        },
        { timeout: 10_000 }
      )
      .toBe(true)
  }

  /**
   * Select a note by clicking its title in the list, then delete it
   * via the overflow menu.
   */
  async deleteNote(title: string): Promise<void> {
    // Click on the note in the list to select it
    const noteItem = this.page.getByText(title, { exact: false }).first()
    await noteItem.click()

    // Wait for editor header to show the title
    await expect(this.overflowMenuButton).toBeVisible({ timeout: 10_000 })

    // Open overflow menu
    await this.overflowMenuButton.click()

    // Click "Delete" in the dropdown menu
    const deleteMenuItem = this.page.getByRole("menuitem", { name: /delete/i })
    await expect(deleteMenuItem).toBeVisible({ timeout: 5_000 })
    await deleteMenuItem.click()

    // Handle "Please confirm" modal (the component uses confirmDanger with "Delete" button)
    const confirmModal = this.page.locator('.ant-modal-confirm')
    await confirmModal.waitFor({ state: 'visible', timeout: 5_000 }).catch(() => {})
    const confirmDeleteBtn = confirmModal.getByRole("button", { name: /^delete$/i })
    if (await confirmDeleteBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await confirmDeleteBtn.click()
    }
  }
}
