/**
 * Page Object for Audiobook Studio workflow
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class AudiobookStudioPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/audiobook-studio", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the page heading or the tabs container to appear
    const heading = this.page.getByText("Audiobook Studio")
    await heading.first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  get heading(): Locator {
    return this.page.getByRole("heading", { name: /audiobook studio/i })
  }

  get contentTab(): Locator {
    return this.page.getByRole("tab", { name: /content/i })
  }

  get chaptersTab(): Locator {
    return this.page.getByRole("tab", { name: /chapters/i })
  }

  get generateTab(): Locator {
    return this.page.getByRole("tab", { name: /generate/i })
  }

  get outputTab(): Locator {
    return this.page.getByRole("tab", { name: /output/i })
  }

  get myProjectsButton(): Locator {
    return this.page.getByRole("button", { name: /my projects/i })
  }

  get newProjectButton(): Locator {
    return this.page.getByRole("button", { name: /^new$/i })
  }

  get saveButton(): Locator {
    return this.page.getByRole("button", { name: /^save(?: project)?$/i })
  }

  get saveStatus(): Locator {
    return this.page.getByRole("status")
  }

  get projectTitleInput(): Locator {
    return this.page.locator("input").filter({ hasText: "" }).locator("xpath=//input[@placeholder]").first()
  }

  get generateAllButton(): Locator {
    return this.page.getByRole("button", { name: /generate all|generate remaining/i })
  }

  get cancelGenerationButton(): Locator {
    return this.page.getByRole("button", { name: /cancel/i })
  }

  get combineAllButton(): Locator {
    return this.page.getByRole("button", { name: /combine all/i })
  }

  get downloadAllButton(): Locator {
    return this.page.getByRole("button", { name: /download all chapters/i })
  }

  get subtitlesButton(): Locator {
    return this.page.getByRole("button", { name: /subtitles/i })
  }

  // -- Tab Navigation --------------------------------------------------------

  async switchToTab(tab: "content" | "chapters" | "generate" | "output"): Promise<void> {
    const tabLocator = {
      content: this.contentTab,
      chapters: this.chaptersTab,
      generate: this.generateTab,
      output: this.outputTab,
    }[tab]
    await tabLocator.click()
  }

  // -- API Request Helpers ---------------------------------------------------

  async waitForApiCall(
    urlPattern: string | RegExp,
    method: string = "GET"
  ): Promise<{ status: number; body: unknown }> {
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

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "My Projects button",
        locator: this.myProjectsButton,
        expectation: {
          type: "state_change",
          stateCheck: async (page: Page) => {
            // Clicking "My Projects" toggles the project list view
            return page.getByText("Back to Editor").isVisible().catch(() => false)
          },
        },
      },
    ]
  }
}
