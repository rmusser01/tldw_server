/**
 * Page Object for the Audio Studio workflow surface.
 */
import { type Locator, type Page, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export type AudioStudioWorkflow = "narration" | "podcast" | "briefing" | "music"

const WORKFLOW_LABELS: Record<AudioStudioWorkflow, string> = {
  narration: "Narration",
  podcast: "Podcast",
  briefing: "Briefing",
  music: "Music",
}

export class AudioStudioPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  async goto(workflow?: AudioStudioWorkflow): Promise<void> {
    const suffix = workflow ? `?workflow=${workflow}` : ""
    await this.page.goto(`/audio-studio${suffix}`, { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async gotoCompatibilityRoute(): Promise<void> {
    await this.page.goto("/audiobook-studio", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    await expect(this.heading).toBeVisible({ timeout: 20_000 })
  }

  get heading(): Locator {
    return this.page.getByRole("heading", { name: "Audio Studio" })
  }

  get projectTitleInput(): Locator {
    return this.page.getByRole("textbox", { name: "Project title" })
  }

  get newProjectButton(): Locator {
    return this.page.getByRole("button", { name: "New Audio Studio project" })
  }

  get saveButton(): Locator {
    return this.page.getByRole("button", { name: "Save", exact: true })
  }

  get generationPanel(): Locator {
    return this.page.getByText("Generation").first()
  }

  get renderExportPanel(): Locator {
    return this.page.getByText("Render & Export")
  }

  workflowTab(workflow: AudioStudioWorkflow): Locator {
    return this.page.getByRole("tab", { name: new RegExp(`^${WORKFLOW_LABELS[workflow]}\\b`, "i") })
  }

  async switchWorkflow(workflow: AudioStudioWorkflow): Promise<void> {
    await this.workflowTab(workflow).click()
    await expect(this.workflowTab(workflow)).toHaveAttribute("aria-selected", "true")
  }

  async expectWorkflowControls(workflow: AudioStudioWorkflow): Promise<void> {
    switch (workflow) {
      case "narration":
        await expect(this.page.getByText("Paste or type your content")).toBeVisible()
        await expect(this.page.getByRole("tab", { name: "Content", exact: true })).toBeVisible()
        await expect(this.page.getByRole("tab", { name: "Chapters", exact: true })).toBeVisible()
        await expect(this.page.getByRole("tab", { name: "Voice", exact: true })).toBeVisible()
        await expect(this.page.getByRole("tab", { name: "Output", exact: true })).toBeVisible()
        break
      case "podcast":
        await expect(this.page.getByRole("textbox", { name: "Podcast script" })).toBeVisible()
        await expect(this.page.getByRole("textbox", { name: "Host speaker" })).toBeVisible()
        await expect(this.page.getByRole("textbox", { name: "Guest speaker" })).toBeVisible()
        break
      case "briefing":
        await expect(this.page.getByRole("textbox", { name: "Briefing outline" })).toBeVisible()
        await expect(this.page.getByRole("textbox", { name: "Source notes" })).toBeVisible()
        break
      case "music":
        await expect(this.page.getByRole("textbox", { name: "Prompt" })).toBeVisible()
        await expect(this.page.getByRole("textbox", { name: "Lyrics" })).toBeVisible()
        await expect(this.page.getByRole("textbox", { name: "Style" })).toBeVisible()
        await expect(this.page.getByRole("combobox", { name: "Provider" })).toBeVisible()
        break
    }
  }

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "New Audio Studio project",
        locator: this.newProjectButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/audio-studio\/projects$/,
          method: "POST",
        },
      },
    ]
  }
}
