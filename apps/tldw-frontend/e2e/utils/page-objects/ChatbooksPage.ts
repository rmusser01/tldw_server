/**
 * Page Object for Chatbooks Playground workflow
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class ChatbooksPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/chatbooks-playground", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for heading or the offline empty state
    const heading = this.page.getByText("Chatbooks Playground")
    const offline = this.page.getByText("Connect to your tldw server to use Chatbooks")
    await Promise.race([
      heading.first().waitFor({ state: "visible", timeout: 20_000 }),
      offline.first().waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  get heading(): Locator {
    return this.page.getByRole("heading", { name: /chatbooks playground/i })
  }

  get offlineMessage(): Locator {
    return this.page.getByText("Connect to your tldw server to use Chatbooks")
  }

  /** Export tab */
  get exportTab(): Locator {
    return this.page.getByRole("tab", { name: /export/i })
  }

  /** Import tab */
  get importTab(): Locator {
    return this.page.getByRole("tab", { name: /import/i })
  }

  /** Jobs tab */
  get jobsTab(): Locator {
    return this.page.getByRole("tab", { name: /jobs/i })
  }

  /** Export chatbook button */
  get exportButton(): Locator {
    return this.page.getByRole("button", { name: /export chatbook/i })
  }

  /** Import chatbook button */
  get importButton(): Locator {
    return this.page.getByRole("button", { name: /import chatbook/i })
  }

  /** File upload dragger area */
  get uploadDropzone(): Locator {
    return this.page.getByText(/drop a \.zip or \.chatbook archive/i)
  }

  /** Job tracker card */
  get jobTrackerCard(): Locator {
    return this.page.getByText("Job tracker")
  }

  /** Chatbooks unavailable alert */
  get unavailableAlert(): Locator {
    return this.page.getByText("Chatbooks is not available on this server")
  }

  // -- Tab Navigation --------------------------------------------------------

  async switchToTab(tab: "export" | "import" | "jobs"): Promise<void> {
    const tabLocator = {
      export: this.exportTab,
      import: this.importTab,
      jobs: this.jobsTab,
    }[tab]
    await tabLocator.click()
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Export chatbook button",
        locator: this.exportButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/chatbooks\/export/,
          method: "POST",
        },
      },
    ]
  }
}
