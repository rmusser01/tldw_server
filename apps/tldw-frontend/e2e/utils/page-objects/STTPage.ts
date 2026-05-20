/**
 * Page Object for the STT (Speech-to-Text) Playground
 *
 * Wraps the SttPlaygroundPage component which includes:
 * - Recording strip (record, upload, clear, settings toggle)
 * - Comparison panel (model selection, transcription results)
 * - History panel (past comparison entries)
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class STTPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/stt", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the page heading to appear
    const heading = this.page.getByText("Speech to Text")
    await heading.first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {})
  }

  // -- Locators: Page-level ---------------------------------------------------

  get heading(): Locator {
    return this.page.getByRole("heading", { name: /speech to text/i })
  }

  get subtitle(): Locator {
    return this.page.getByText(/try out transcription models|record audio and compare transcription|select models and record audio/i).first()
  }

  // -- Locators: Recording strip ----------------------------------------------

  get recordButton(): Locator {
    return this.page.getByRole("button", { name: /start recording/i })
  }

  get stopButton(): Locator {
    return this.page.getByRole("button", { name: /stop recording/i })
  }

  get uploadButton(): Locator {
    return this.page.getByRole("button", { name: /upload audio file/i })
  }

  get settingsToggleButton(): Locator {
    return this.page.getByRole("button", { name: /toggle settings/i })
  }

  get clearRecordingButton(): Locator {
    return this.page.getByRole("button", { name: /clear recording/i })
  }

  get durationDisplay(): Locator {
    return this.page.locator("[aria-live='polite']").first()
  }

  // -- Locators: Comparison panel ---------------------------------------------

  get comparisonPanel(): Locator {
    return this.page.locator(".ant-card").nth(1)
  }

  // -- Locators: History panel ------------------------------------------------

  get historyPanel(): Locator {
    return this.page.getByText(/history/i).first()
  }

  // -- Interactive elements for assertAllButtonsWired() -----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Settings toggle button",
        locator: this.settingsToggleButton,
        expectation: {
          type: "state_change",
          stateCheck: async (page: Page) => {
            // Toggling settings shows/hides the inline settings panel
            return page.getByText(/language/i).isVisible().catch(() => false)
          },
        },
      },
    ]
  }
}
