/**
 * Page Object for the Text to Speech route
 *
 * Wraps the /tts route which renders SpeechPlaygroundPage in locked "listen" mode.
 * Key UI elements: title, text input area, Play/Stop/Download buttons, voice provider strip.
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export class TTSPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/tts", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the canonical Text to Speech route heading.
    const heading = this.page.getByText("Text to Speech")
    await heading.first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  /** Page heading ("Text to Speech") */
  get heading(): Locator {
    return this.page.getByRole("heading", { name: /text to speech/i })
  }

  /** Subtitle text */
  get subtitle(): Locator {
    return this.page.getByText(/draft text, choose a voice, and generate audio/i)
  }

  /** The main text input area for entering text to synthesize */
  get textInput(): Locator {
    return this.page.getByRole("textbox", { name: /enter some text to hear it spoken/i })
  }

  /** "Insert sample text" button */
  get sampleTextButton(): Locator {
    return this.page.getByRole("button", { name: /insert sample text|sample/i })
  }

  /** Play button in the sticky action bar */
  get playButton(): Locator {
    return this.page.getByRole("button", { name: /^play$/i })
  }

  /** Stop button in the sticky action bar */
  get stopButton(): Locator {
    return this.page.getByRole("button", { name: /^stop$/i })
  }

  /** Download button in the sticky action bar */
  get downloadButton(): Locator {
    return this.page.getByRole("button", { name: /^download$/i })
  }

  /** The playback controls toolbar */
  get playbackToolbar(): Locator {
    return this.page.getByRole("toolbar", { name: /playback controls/i })
  }

  /** TTS history section heading */
  get historyHeading(): Locator {
    return this.page.getByText("TTS history", { exact: true })
  }

  // -- Helpers ---------------------------------------------------------------

  /** Fill the text input with the given text */
  async enterText(text: string): Promise<void> {
    await this.textInput.fill(text)
  }

  /** Click the sample text button to populate the input */
  async insertSampleText(): Promise<void> {
    await this.sampleTextButton.click()
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Play button",
        locator: this.playButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/audio\/speech/,
          method: "POST",
        },
        setup: async (page: Page) => {
          // Ensure there is text in the input before clicking Play
          const input = page.getByRole("textbox", { name: /enter some text to hear it spoken/i })
          const currentValue = await input.inputValue().catch(() => "")
          if (!currentValue) {
            await input.fill("Hello, this is a test of text to speech synthesis.")
          }
        },
      },
    ]
  }
}
