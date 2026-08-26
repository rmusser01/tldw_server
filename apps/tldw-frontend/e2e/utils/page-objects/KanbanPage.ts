/**
 * Page Object for Kanban Playground workflow
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class KanbanPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/kanban", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    await expect(this.heading).toBeVisible({ timeout: 20_000 })
    // The heading renders before the initial boards request settles. Wait for
    // either the zero-board empty state or a populated gallery card so callers
    // do not sample the transient blank body.
    await expect(
      this.emptyStateMessage.or(this.boardGalleryCards.first())
    ).toBeVisible({ timeout: 20_000 })
  }

  // -- Locators --------------------------------------------------------------

  /** Main heading */
  get heading(): Locator {
    return this.page.getByRole("heading", { name: /kanban playground/i })
  }

  /** Empty state description shown when no boards exist */
  get emptyStateMessage(): Locator {
    return this.page.getByText(
      "Organize research tasks, track projects with boards and cards.",
      { exact: true }
    )
  }

  /** Board selector dropdown */
  get boardSelector(): Locator {
    return this.page.locator(".ant-select").filter({ hasText: /select a board/i }).first()
  }

  /** "New Board" button in the header toolbar */
  get newBoardButton(): Locator {
    return this.page.getByRole("button", { name: /new board/i }).first()
  }

  /** "Create Board" button in the empty-state gallery */
  get createBoardButton(): Locator {
    return this.page.getByRole("button", { name: /create board/i })
  }

  /** Refresh button (icon-only button with RefreshCw icon) */
  get refreshButton(): Locator {
    // The refresh button is the last icon-only button in the header toolbar
    return this.page.locator(".kanban-playground .flex.items-center.gap-2 button").last()
  }

  /** Actions dropdown trigger (MoreVertical icon button) */
  get actionsDropdownButton(): Locator {
    // The actions dropdown is a button with no text, second to last in the toolbar
    return this.page.locator(".kanban-playground").getByRole("button").filter({ hasNotText: /./i }).nth(0)
  }

  /** "Create New Board" modal */
  get createBoardModal(): Locator {
    return this.page.locator(".ant-modal").filter({ hasText: "Create New Board" })
  }

  /** Board name input inside the create modal */
  get boardNameInput(): Locator {
    return this.createBoardModal.getByPlaceholder("Enter board name")
  }

  /** Board description textarea inside the create modal */
  get boardDescriptionInput(): Locator {
    return this.createBoardModal.getByPlaceholder("What is this board for?")
  }

  /** OK/Create button in the create modal */
  get modalCreateButton(): Locator {
    return this.createBoardModal.getByRole("button", { name: /create/i }).last()
  }

  /** "Import Board..." menu item in the actions dropdown */
  get importBoardMenuItem(): Locator {
    return this.page.getByText("Import Board...")
  }

  /** "Export Board..." menu item in the actions dropdown */
  get exportBoardMenuItem(): Locator {
    return this.page.getByText("Export Board...")
  }

  /** "Archived Items" menu item in the actions dropdown */
  get archivedItemsMenuItem(): Locator {
    return this.page.getByText("Archived Items")
  }

  /** Boards count badge */
  get boardsCountBadge(): Locator {
    return this.page.locator(".kanban-playground").locator(".ant-badge")
  }

  /** Board gallery cards (the board selection buttons) */
  get boardGalleryCards(): Locator {
    return this.page.locator(".kanban-playground button.text-left")
  }

  /** "New Board" card in the gallery (dashed border) */
  get galleryNewBoardCard(): Locator {
    return this.page.locator(".kanban-playground button.border-dashed")
  }

  // -- Actions ---------------------------------------------------------------

  /** Open the Create New Board modal via the header button */
  async openCreateBoardModal(): Promise<void> {
    await this.newBoardButton.click()
    await expect(this.createBoardModal).toBeVisible({ timeout: 5_000 })
  }

  /** Fill and submit the Create New Board modal */
  async createBoard(name: string, description?: string): Promise<void> {
    await this.openCreateBoardModal()
    await this.boardNameInput.fill(name)
    if (description) {
      await this.boardDescriptionInput.fill(description)
    }
    await this.modalCreateButton.click()
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "New Board button (opens create modal)",
        locator: this.newBoardButton,
        expectation: {
          type: "modal" as const,
          modalSelector: ".ant-modal",
        },
      },
      {
        name: "Refresh button fetches boards list",
        locator: this.refreshButton,
        expectation: {
          type: "api_call" as const,
          apiPattern: /\/api\/v1\/kanban\/boards/,
          method: "GET",
        },
      },
    ]
  }
}
