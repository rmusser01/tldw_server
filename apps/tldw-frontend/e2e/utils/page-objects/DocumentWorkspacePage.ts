/**
 * Page Object for Document Workspace route
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class DocumentWorkspacePage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/document-workspace", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for heading "Document Workspace" or the offline/empty state
    const heading = this.page.getByText("Document Workspace")
    const offlineMsg = this.page.getByText("Connect to your tldw server")
    await Promise.race([
      heading.first().waitFor({ state: "visible", timeout: 20_000 }),
      offlineMsg.first().waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  /** Main heading (shows active document title or "Document Workspace") */
  get heading(): Locator {
    return this.page.getByRole("heading", { name: /document workspace/i })
  }

  /** Offline / disconnected message */
  get offlineMessage(): Locator {
    return this.page.getByText("Connect to your tldw server")
  }

  /** Toggle left sidebar button */
  get toggleLeftButton(): Locator {
    return this.page.getByTestId("document-workspace-toggle-left")
  }

  /** Toggle right panel button */
  get toggleRightButton(): Locator {
    return this.page.getByTestId("document-workspace-toggle-right")
  }

  /** Open document (plus) button in the header */
  get openDocumentButton(): Locator {
    return this.page.getByTestId("document-open-picker-button")
  }

  /** Keyboard shortcuts button in the header */
  get shortcutsButton(): Locator {
    return this.page.getByRole("button", { name: /keyboard shortcuts/i })
  }

  // -- Left sidebar tabs -----------------------------------------------------

  get insightsTab(): Locator {
    return this.page.getByRole("tab", { name: /insights/i })
  }

  get figuresTab(): Locator {
    return this.page.getByRole("tab", { name: /figures/i })
  }

  get contentsTab(): Locator {
    return this.page.getByRole("tab", { name: /contents/i })
  }

  get infoTab(): Locator {
    return this.page.getByRole("tab", { name: /^info$/i })
  }

  get referencesTab(): Locator {
    return this.page.getByRole("tab", { name: /references/i })
  }

  // -- Right panel tabs ------------------------------------------------------

  get chatTab(): Locator {
    return this.page.getByRole("tab", { name: /chat/i })
  }

  get notesTab(): Locator {
    return this.page.getByRole("tab", { name: /notes/i })
  }

  get citeTab(): Locator {
    return this.page.getByRole("tab", { name: /cite/i })
  }

  get quizTab(): Locator {
    return this.page.getByRole("tab", { name: /quiz/i })
  }

  // -- Document picker modal -------------------------------------------------

  get pickerModal(): Locator {
    return this.page.locator(".ant-modal").filter({ hasText: /open document|library|upload/i })
  }

  get pickerSearchInput(): Locator {
    return this.pickerModal.getByPlaceholder(/search/i)
  }

  // -- Helpers ---------------------------------------------------------------

  /** Open the document picker modal via the header "+" button */
  async openPicker(): Promise<void> {
    await this.openDocumentButton.click()
    await this.pickerModal.waitFor({ state: "visible", timeout: 5_000 }).catch(() => {})
  }

  /** Close the document picker modal */
  async closePicker(): Promise<void> {
    await this.page.keyboard.press("Escape")
    await this.pickerModal.waitFor({ state: "hidden", timeout: 3_000 }).catch(() => {})
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Toggle left sidebar",
        locator: this.toggleLeftButton,
        expectation: { type: "state_change" as const, stateCheck: async (page: Page) => {
          // Check if the left aside is visible
          return page.locator("aside").first().isVisible().catch(() => false)
        }},
      },
      {
        name: "Toggle right panel",
        locator: this.toggleRightButton,
        expectation: { type: "state_change" as const, stateCheck: async (page: Page) => {
          return page.locator("aside").last().isVisible().catch(() => false)
        }},
      },
      {
        name: "Open document button (picker modal)",
        locator: this.openDocumentButton,
        expectation: {
          type: "modal" as const,
          modalSelector: ".ant-modal",
        },
      },
    ]
  }
}
