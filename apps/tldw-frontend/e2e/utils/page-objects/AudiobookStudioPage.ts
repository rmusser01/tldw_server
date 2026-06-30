/**
 * Compatibility wrapper for the legacy Audiobook Studio route.
 */
import { type Locator, type Page } from "@playwright/test"
import { AudioStudioPage } from "./AudioStudioPage"

export class AudiobookStudioPage extends AudioStudioPage {
  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await this.gotoCompatibilityRoute()
  }

  get narrationTab(): Locator {
    return this.workflowTab("narration")
  }

  get contentTab(): Locator {
    return this.page.getByRole("tab", { name: "Content", exact: true })
  }

  get chaptersTab(): Locator {
    return this.page.getByRole("tab", { name: "Chapters", exact: true })
  }

  get voiceTab(): Locator {
    return this.page.getByRole("tab", { name: "Voice", exact: true })
  }

  get outputTab(): Locator {
    return this.page.getByRole("tab", { name: "Output", exact: true })
  }

  async switchToTab(tab: "content" | "chapters" | "voice" | "output"): Promise<void> {
    const tabLocator = {
      content: this.contentTab,
      chapters: this.chaptersTab,
      voice: this.voiceTab,
      output: this.outputTab,
    }[tab]
    await tabLocator.click()
  }
}
