/**
 * Page Object for Content Review workflow
 *
 * Content Review allows users to review, edit, and commit ingested drafts
 * stored locally in IndexedDB. Key interactions include:
 * - Selecting batches and drafts from the sidebar
 * - Editing draft content, title, keywords, and notes
 * - AI-powered corrections and template formatting (via /api/v1/chat/completions)
 * - Committing drafts to the server (via /api/v1/media/add)
 * - Marking drafts as reviewed, discarding, or resetting
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class ContentReviewPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/content-review", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // The heading is always rendered, including the empty state. Missing UI
    // must reject readiness instead of silently completing acceptance.
    await this.heading.waitFor({ state: "visible", timeout: 20_000 })
  }

  // -- Locators --------------------------------------------------------------

  /** Page heading */
  get heading(): Locator {
    return this.page.locator("h3").filter({ hasText: "Content Review" })
  }

  /** Empty state message when no drafts exist */
  get emptyState(): Locator {
    return this.page.getByText("No drafts yet")
  }

  /** "Open Quick Ingest" button shown in empty state */
  get openQuickIngestButton(): Locator {
    return this.page.getByRole("button", { name: /open quick ingest/i })
  }

  /** Batch selector dropdown */
  get batchSelect(): Locator {
    return this.page.locator(".ant-select").first()
  }

  /** Drafts list in the sidebar */
  get draftsList(): Locator {
    return this.page.locator(".ant-list")
  }

  /** "Commit All" button in the header */
  get commitAllButton(): Locator {
    return this.page.getByRole("button", { name: /commit all/i })
  }

  /** "Clear drafts" button in the header */
  get clearDraftsButton(): Locator {
    return this.page.getByRole("button", { name: /clear drafts/i })
  }

  /** Draft title input field */
  get titleInput(): Locator {
    return this.page.getByPlaceholder("Draft title")
  }

  /** "Reset" button to restore original content */
  get resetButton(): Locator {
    return this.page.getByRole("button", { name: /^reset$/i })
  }

  /** "Diff view" button to open diff modal */
  get diffViewButton(): Locator {
    return this.page.getByRole("button", { name: /diff view/i })
  }

  /** "Save draft" button for manual local save */
  get saveDraftButton(): Locator {
    return this.page.getByRole("button", { name: /save draft/i })
  }

  /** Content textarea editor */
  get contentTextarea(): Locator {
    return this.page.locator("textarea:visible").first()
  }

  /** "AI fix" button for AI corrections */
  get aiFixButton(): Locator {
    return this.page.getByRole("button", { name: /ai fix/i })
  }

  /** "Apply template" button */
  get applyTemplateButton(): Locator {
    return this.page.getByRole("button", { name: /apply template/i })
  }

  /** "Detect sections" button */
  get detectSectionsButton(): Locator {
    return this.page.getByRole("button", { name: /detect sections/i })
  }

  /** "Commit" button (single draft) in the Actions panel */
  get commitButton(): Locator {
    return this.page.getByRole("button", { name: /^commit$/i })
  }

  /** "Mark reviewed" button */
  get markReviewedButton(): Locator {
    return this.page.getByRole("button", { name: /mark reviewed/i })
  }

  /** "Skip" button to navigate to next draft */
  get skipButton(): Locator {
    return this.page.getByRole("button", { name: /^skip$/i })
  }

  /** "Discard" button */
  get discardButton(): Locator {
    return this.page.getByRole("button", { name: /^discard$/i })
  }

  /** "ready" tag showing count of reviewed drafts */
  get readyTag(): Locator {
    return this.page.locator(".ant-tag-blue")
  }

  /** "committed" tag showing count of committed drafts */
  get committedTag(): Locator {
    return this.page.locator(".ant-tag-green")
  }

  /** "Content" label in the editor panel */
  get editorLabel(): Locator {
    return this.page.getByText("Content", { exact: false }).filter({ hasText: /^Content$/ })
  }

  /** "Metadata" label in the sidebar panel */
  get metadataLabel(): Locator {
    return this.page.getByText("Metadata", { exact: true })
  }

  /** "Actions" label in the sidebar panel */
  get actionsLabel(): Locator {
    return this.page.getByText("Actions", { exact: true })
  }

  /** Keywords tag-mode select */
  get keywordsSelect(): Locator {
    return this.page.locator(".ant-select-selector").filter({ hasText: /add keywords/i }).first()
  }

  /** Review notes textarea */
  get reviewNotesTextarea(): Locator {
    return this.page.locator("textarea:visible").nth(1)
  }

  /** Select a draft prompt when no draft is selected */
  get selectDraftMessage(): Locator {
    return this.page.getByText("Select a draft to begin reviewing")
  }

  /** Diff view modal */
  get diffModal(): Locator {
    return this.page.locator(".ant-modal").filter({ hasText: /original|edited/i })
  }

  // -- Helper methods --------------------------------------------------------

  /** Check if the page is showing the empty state (no drafts) */
  async isEmptyState(): Promise<boolean> {
    return this.emptyState.isVisible().catch(() => false)
  }

  /** Check if a draft is currently loaded in the editor */
  async isDraftLoaded(): Promise<boolean> {
    return this.titleInput.isVisible().catch(() => false)
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "AI fix button",
        locator: this.aiFixButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/chat\/completions/,
          method: "POST",
        },
      },
      {
        name: "Apply template button",
        locator: this.applyTemplateButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/chat\/completions/,
          method: "POST",
        },
      },
      {
        name: "Commit button",
        locator: this.commitButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/media\/add/,
          method: "POST",
        },
      },
      {
        name: "Diff view button",
        locator: this.diffViewButton,
        expectation: {
          type: "modal",
          modalSelector: ".ant-modal",
        },
      },
    ]
  }
}
