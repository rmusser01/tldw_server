/**
 * Page Object for the Sources (Ingestion Sources) page
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class SourcesPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/sources", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for heading, offline state, unsupported state, unavailable state, or empty state
    const heading = this.page.getByText("Sources")
    const loading = this.loadingSpinner
    const offline = this.page.getByText("Server is offline. Connect to manage ingestion sources.")
    const unsupported = this.page.getByText(/does not advertise ingestion source/i)
    const unavailable = this.page.getByRole("heading", { name: /cannot reach sources/i })
    const empty = this.page.getByText("No ingestion sources yet.")
    await Promise.race([
      heading.first().waitFor({ state: "visible", timeout: 20_000 }),
      loading.first().waitFor({ state: "visible", timeout: 20_000 }),
      offline.first().waitFor({ state: "visible", timeout: 20_000 }),
      unsupported.first().waitFor({ state: "visible", timeout: 20_000 }),
      unavailable.first().waitFor({ state: "visible", timeout: 20_000 }),
      empty.first().waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  get heading(): Locator {
    return this.page.getByRole("heading", { name: /^sources$/i }).first()
  }

  get description(): Locator {
    return this.page.getByText("Manage local folders and archive snapshots that sync into notes or media.")
  }

  get offlineMessage(): Locator {
    return this.page.getByText("Server is offline. Connect to manage ingestion sources.")
  }

  get unsupportedMessage(): Locator {
    return this.page.getByText(/does not advertise ingestion source/i)
  }

  get unavailableMessage(): Locator {
    return this.page.getByRole("heading", { name: /cannot reach sources/i })
  }

  get emptyMessage(): Locator {
    return this.page.getByText("No ingestion sources yet.")
  }

  /** "New source" primary button */
  get newSourceButton(): Locator {
    return this.page.getByRole("button", { name: /new source/i })
  }

  /** Loading spinner */
  get loadingSpinner(): Locator {
    return this.page.locator("[data-testid='sources-loading-state']").first()
  }

  /** Error alert */
  get errorAlert(): Locator {
    return this.page.locator(".ant-alert-error")
  }

  /** Source list cards */
  get sourceCards(): Locator {
    return this.page.locator(".ant-card")
  }

  /** "Sync now" button on the first source card */
  get syncNowButton(): Locator {
    return this.page.getByRole("button", { name: /sync now/i }).first()
  }

  /** Enable/Disable toggle button on the first source card */
  get enableDisableButton(): Locator {
    return this.page.getByRole("button", { name: /^(enable|disable)$/i }).first()
  }

  /** "Open detail" button on the first source card */
  get openDetailButton(): Locator {
    return this.page.getByRole("button", { name: /open detail/i }).first()
  }

  // -- Helpers ---------------------------------------------------------------

  /** Whether the page is showing the online workspace (heading + new source button, no unsupported/offline banner) */
  async isOnlineWorkspace(): Promise<boolean> {
    const headingVisible = await this.heading.isVisible().catch(() => false)
    const newSourceVisible = await this.newSourceButton.isVisible().catch(() => false)
    const unsupportedVisible = await this.unsupportedMessage.isVisible().catch(() => false)
    const offlineVisible = await this.offlineMessage.isVisible().catch(() => false)
    const unavailableVisible = await this.unavailableMessage.isVisible().catch(() => false)
    return headingVisible && newSourceVisible && !unsupportedVisible && !offlineVisible && !unavailableVisible
  }

  /** Whether sources are listed */
  async hasSourceCards(): Promise<boolean> {
    const count = await this.sourceCards.count()
    return count > 0
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    const elements: InteractiveElement[] = []

    // "New source" navigates to /sources/new
    elements.push({
      name: "New source button",
      locator: this.newSourceButton,
      expectation: {
        type: "navigation",
        targetUrl: /\/sources\/new/,
      },
    })

    // "Sync now" fires POST /api/v1/ingestion-sources/{id}/sync
    elements.push({
      name: "Sync now button",
      locator: this.syncNowButton,
      expectation: {
        type: "api_call",
        apiPattern: /\/api\/v1\/ingestion-sources\/.*\/sync/,
        method: "POST",
      },
    })

    // "Enable/Disable" fires PUT /api/v1/ingestion-sources/{id}
    elements.push({
      name: "Enable/Disable button",
      locator: this.enableDisableButton,
      expectation: {
        type: "api_call",
        apiPattern: /\/api\/v1\/ingestion-sources\//,
        method: "PUT",
      },
    })

    return elements
  }
}
