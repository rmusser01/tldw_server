import { expect, type Locator, type Page } from "@playwright/test"
import {
  PARITY_SUMMARY_ARTIFACT,
  PARITY_SUMMARY_ARTIFACT_ID
} from "./fixtures"
import type { WorkspacePlaygroundPlatform } from "./types"

export class WorkspacePlaygroundParityPage {
  readonly page: Page
  readonly headerTitle: Locator
  readonly workspacesButton: Locator
  readonly sourcesPanel: Locator
  readonly chatPanel: Locator
  readonly studioPanel: Locator
  readonly generatedOutputsToggle: Locator

  constructor(page: Page) {
    this.page = page
    this.headerTitle = page.locator("header h1").first()
    this.workspacesButton = page
      .getByTestId("workspace-workspaces-button")
      .or(page.getByRole("button", { name: /workspaces/i }))
      .first()
    this.sourcesPanel = page.locator("#workspace-sources-panel")
    this.chatPanel = page.locator("#workspace-main-content")
    this.studioPanel = page.locator("#workspace-studio-panel:visible").first()
    this.generatedOutputsToggle = this.studioPanel
      .locator('button[aria-controls="studio-generated-outputs-section"]')
      .first()
  }

  private async disablePortalPointerInterception(): Promise<void> {
    await this.page.evaluate(() => {
      const portals = document.querySelectorAll("nextjs-portal")
      portals.forEach((portal) => {
        ;(portal as HTMLElement).style.pointerEvents = "none"
      })
    })
  }

  private async dismissAssistantSetupBlockingModal(timeoutMs = 5_000): Promise<void> {
    const setupHeading = this.page.getByText(/build your assistant/i).first()
    const overlay = this.page.getByTestId("assistant-setup-overlay").first()
    const overlayVisible =
      (await overlay.isVisible({ timeout: 1_000 }).catch(() => false)) ||
      (await setupHeading.isVisible({ timeout: 1_000 }).catch(() => false))

    if (!overlayVisible) {
      return
    }

    const skipControl = this.page.getByRole("button", { name: /skip for now/i }).first()
    await expect(skipControl).toBeVisible({ timeout: timeoutMs })
    await skipControl.click()

    await expect
      .poll(
        async () =>
          !(await overlay.isVisible().catch(() => false)) &&
          !(await setupHeading.isVisible().catch(() => false)),
        {
          timeout: timeoutMs,
          message: "Timed out dismissing the assistant setup modal",
        }
      )
      .toBe(true)
  }

  async goto(platform: WorkspacePlaygroundPlatform, optionsUrl?: string): Promise<void> {
    await this.page.addInitScript(() => {
      try {
        window.localStorage.setItem("assistant_setup_dismissed", "true")
      } catch {
        // Ignore storage access issues in constrained environments.
      }
    })

    if (platform === "extension") {
      if (!optionsUrl) {
        throw new Error("optionsUrl is required for extension parity navigation")
      }
      await this.page.goto(`${optionsUrl}#/workspace-playground`, {
        waitUntil: "domcontentloaded"
      })
    } else {
      await this.page.goto("/workspace-playground", {
        waitUntil: "domcontentloaded"
      })
    }

    await this.disablePortalPointerInterception()
  }

  async waitForReady(): Promise<void> {
    await this.dismissAssistantSetupBlockingModal()
    await expect(this.workspacesButton).toBeVisible({ timeout: 30_000 })
    await expect(this.chatPanel).toBeVisible({ timeout: 30_000 })
    await this.disablePortalPointerInterception()
  }

  async assertBaselinePanesVisible(): Promise<void> {
    await expect(this.headerTitle).toBeVisible()
    await expect(this.sourcesPanel).toBeVisible()
    await expect(this.chatPanel).toBeVisible()
    await expect(this.studioPanel).toBeVisible()
  }

  async openOutputTypesSection(): Promise<void> {
    const toggle = this.studioPanel
      .locator('button[aria-controls="studio-output-types-section"]')
      .first()
    await expect(toggle).toBeVisible({ timeout: 15_000 })
    if ((await toggle.getAttribute("aria-expanded")) === "false") {
      await toggle.click({ force: true })
    }
  }

  async openGeneratedOutputsSection(): Promise<void> {
    await expect(this.generatedOutputsToggle).toBeVisible({ timeout: 15_000 })
    if ((await this.generatedOutputsToggle.getAttribute("aria-expanded")) === "false") {
      await this.generatedOutputsToggle.evaluate((node) => {
        ;(node as HTMLButtonElement).click()
      })
    }
  }

  async collapseGeneratedOutputsSection(): Promise<void> {
    await expect(this.generatedOutputsToggle).toBeVisible({ timeout: 15_000 })
    if ((await this.generatedOutputsToggle.getAttribute("aria-expanded")) === "true") {
      await this.generatedOutputsToggle.evaluate((node) => {
        ;(node as HTMLButtonElement).click()
      })
    }
  }

  async expectGeneratedOutputsSectionHidden(): Promise<void> {
    await expect(this.generatedOutputsToggle).toHaveAttribute("aria-expanded", "false")
  }

  async seedDeterministicArtifact(): Promise<void> {
    await this.page.evaluate((payload) => {
      const store = (window as any).__tldw_useWorkspaceStore
      if (!store?.getState || !store?.setState) {
        throw new Error("Workspace store is unavailable on window")
      }

      const state = store.getState()
      const currentArtifacts = Array.isArray(state.generatedArtifacts)
        ? state.generatedArtifacts
        : []
      const existingWithoutParity = currentArtifacts.filter(
        (artifact: { id?: string }) => artifact?.id !== payload.id
      )

      const nextArtifact = {
        id: payload.id,
        type: payload.type,
        title: payload.title,
        status: payload.status,
        content: payload.content,
        createdAt: new Date(payload.createdAtIso),
        completedAt: new Date(payload.createdAtIso)
      }

      store.setState({
        generatedArtifacts: [nextArtifact, ...existingWithoutParity]
      })
    }, PARITY_SUMMARY_ARTIFACT)
  }

  getParityArtifactCard(): Locator {
    return this.page
      .locator(
        `#workspace-studio-panel [data-testid="studio-artifact-card-${PARITY_SUMMARY_ARTIFACT_ID}"]:visible`
      )
      .first()
  }

  async expectParityArtifactVisible(): Promise<void> {
    await expect(this.getParityArtifactCard()).toBeVisible({ timeout: 10_000 })
  }

  async expectArtifactActionButtons(): Promise<void> {
    const primaryActions = this.studioPanel
      .locator(
        `[data-testid="studio-artifact-primary-actions-${PARITY_SUMMARY_ARTIFACT_ID}"]:visible`
      )
      .first()
    const secondaryActions = this.studioPanel
      .locator(
        `[data-testid="studio-artifact-secondary-actions-${PARITY_SUMMARY_ARTIFACT_ID}"]:visible`
      )
      .first()

    await expect(primaryActions).toBeVisible()
    await expect(secondaryActions).toBeVisible()

    await expect(primaryActions.getByRole("button", { name: /View/i })).toBeVisible()
    await expect(primaryActions.getByRole("button", { name: /Download/i })).toBeVisible()

    await expect(
      secondaryActions.getByRole("button", { name: /Regenerate options/i })
    ).toBeVisible()
    await expect(
      secondaryActions.getByRole("button", { name: /Discuss in chat/i })
    ).toBeVisible()
    await expect(secondaryActions.getByRole("button", { name: /^Delete$/i })).toBeVisible()
  }

}
