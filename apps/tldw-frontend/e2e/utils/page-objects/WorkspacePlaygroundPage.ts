/**
 * Page Object for Workspace Playground workflow coverage
 */
import { type Locator, type Page, expect } from "@playwright/test"
import { dispatchKeyboardShortcut, waitForConnection } from "../helpers"

type WorkspaceSeedSource = {
  mediaId: number
  title: string
  type?: "pdf" | "video" | "audio" | "website" | "document" | "text"
  status?: "processing" | "ready" | "error"
  url?: string
}

export class WorkspacePlaygroundPage {
  readonly page: Page
  readonly headerTitle: Locator
  readonly workspacesButton: Locator
  readonly sourcesPanel: Locator
  readonly chatPanel: Locator
  readonly studioPanel: Locator
  readonly globalSearchModal: Locator
  readonly globalSearchInput: Locator
  readonly addSourceModal: Locator

  constructor(page: Page) {
    this.page = page
    this.headerTitle = page.locator("header h1").first()
    this.workspacesButton = page
      .getByTestId("workspace-workspaces-button")
      .or(page.getByRole("button", { name: /workspaces/i }))
      .first()
    this.sourcesPanel = page.locator("#workspace-sources-panel")
    this.chatPanel = page.locator("#workspace-main-content")
    this.studioPanel = page.locator("#workspace-studio-panel")
    this.globalSearchModal = page
      .getByRole("dialog")
      .filter({ hasText: /search workspace/i })
      .first()
    this.globalSearchInput = this.globalSearchModal.getByPlaceholder(
      /search sources, chat, and notes/i
    )
    this.addSourceModal = page
      .getByRole("dialog")
      .filter({ hasText: /add sources/i })
      .first()
  }

  private async disableNextJsPortalPointerInterception(): Promise<void> {
    await this.page.evaluate(() => {
      const portals = document.querySelectorAll("nextjs-portal")
      portals.forEach((portal) => {
        ;(portal as HTMLElement).style.pointerEvents = "none"
      })
    })
  }

  private async waitForModalBackdropsToClear(): Promise<void> {
    await expect(
      this.page.locator(
        "div.fixed.inset-0.z-50.bg-black\\/50, div.fixed.inset-0.z-50.backdrop-blur-sm, .ant-modal-mask"
      )
    ).toHaveCount(0, { timeout: 10_000 })
  }

  private async clickWhenActionable(locator: Locator): Promise<void> {
    await expect(locator).toBeVisible({ timeout: 10_000 })
    await this.disableNextJsPortalPointerInterception()
    try {
      await locator.click({ trial: true, timeout: 3_000 })
    } catch {
      await this.disableNextJsPortalPointerInterception()
      await locator.click({ trial: true, timeout: 3_000 })
    }
    await locator.click()
  }

  async goto(): Promise<void> {
    await this.page.goto("/workspace-playground", {
      waitUntil: "domcontentloaded"
    })
    await waitForConnection(this.page).catch(() => {})
    await this.disableNextJsPortalPointerInterception()
  }

  async setStudyMaterialsPolicy(policy: "general" | "workspace"): Promise<void> {
    await this.page.evaluate((nextPolicy) => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            setState?: (state: { studyMaterialsPolicy: "general" | "workspace" }) => void
          }
        | undefined

      if (!store?.setState) {
        throw new Error("Workspace store is unavailable on window")
      }

      store.setState({ studyMaterialsPolicy: nextPolicy })
    }, policy)
  }

  async getWorkspaceId(): Promise<string | null> {
    return await this.page.evaluate(() => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            getState?: () => { workspaceId?: string | null }
          }
        | undefined

      return store?.getState?.().workspaceId ?? null
    })
  }

  async getGeneratedArtifactRecord(
    artifactType: "quiz" | "flashcards",
  ): Promise<{
    id: string
    title: string
    status: string
    serverId: number | string | null
    data: Record<string, unknown> | null
  } | null> {
    return await this.page.evaluate((nextType) => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            getState?: () => {
              generatedArtifacts?: Array<{
                id: string
                title: string
                status: string
                serverId?: number | string | null
                data?: Record<string, unknown> | null
                type?: string
              }>
            }
          }
        | undefined

      const artifact = store?.getState?.().generatedArtifacts?.find(
        (entry) => entry.type === nextType
      )
      if (!artifact) {
        return null
      }

      return {
        id: artifact.id,
        title: artifact.title,
        status: artifact.status,
        serverId: artifact.serverId ?? null,
        data: artifact.data ?? null,
      }
    }, artifactType)
  }

  async waitForReady(): Promise<void> {
    await expect(this.workspacesButton).toBeVisible({ timeout: 30_000 })
    await expect(this.chatPanel).toBeVisible({ timeout: 30_000 })
    await this.disableNextJsPortalPointerInterception()
  }

  async resetWorkspace(name = "Workspace E2E"): Promise<void> {
    await this.page.evaluate((workspaceName) => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            getState?: () => {
              initializeWorkspace?: (name?: string) => void
            }
          }
        | undefined

      if (!store?.getState) {
        throw new Error("Workspace store is unavailable on window")
      }

      store.getState().initializeWorkspace?.(workspaceName)
    }, name)
  }

  async openGlobalSearchWithShortcut(): Promise<void> {
    await this.page.locator("body").click()
    await expect
      .poll(
        async () => {
          await dispatchKeyboardShortcut(this.page, { key: "k", ctrlKey: true })
          if (await this.globalSearchInput.isVisible().catch(() => false)) {
            return true
          }

          await dispatchKeyboardShortcut(this.page, { key: "k", metaKey: true })
          return await this.globalSearchInput.isVisible().catch(() => false)
        },
        {
          timeout: 10_000,
          message: "Expected Cmd/Ctrl+K to open the workspace global search"
        }
      )
      .toBe(true)
    await expect(this.globalSearchModal).toBeVisible({ timeout: 10_000 })
    await expect(this.globalSearchInput).toBeVisible({ timeout: 10_000 })
  }

  async closeGlobalSearchWithEscape(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await expect(this.globalSearchInput).toBeVisible({ timeout: 10_000 })
    await this.globalSearchInput.click()
    await expect(this.globalSearchInput).toBeFocused({ timeout: 10_000 })
    await this.globalSearchInput.press("Escape")
    await expect(this.globalSearchModal).toBeHidden({ timeout: 10_000 })
    await this.waitForModalBackdropsToClear()
  }

  async searchWorkspace(query: string): Promise<void> {
    await expect(this.globalSearchInput).toBeVisible({ timeout: 10_000 })
    await this.globalSearchInput.fill(query)
  }

  async hideSourcesPane(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.sourcesPanel.getByRole("button", { name: /hide sources/i })
    )
    await expect(this.sourcesPanel).toBeHidden({ timeout: 10_000 })
  }

  async showSourcesPane(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.page.getByRole("button", { name: /show sources/i }).first()
    )
    await expect(this.sourcesPanel).toBeVisible({ timeout: 10_000 })
  }

  async hideStudioPane(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.studioPanel.getByRole("button", { name: /hide studio/i })
    )
    await expect(this.studioPanel).toBeHidden({ timeout: 10_000 })
  }

  async showStudioPane(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.page.getByRole("button", { name: /show studio/i }).first()
    )
    await expect(this.studioPanel).toBeVisible({ timeout: 10_000 })
  }

  async openAddSourcesModal(): Promise<void> {
    await expect(this.sourcesPanel).toBeVisible({ timeout: 10_000 })
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.sourcesPanel.getByRole("button", { name: /^add$/i })
    )
    await expect(this.addSourceModal).toBeVisible({ timeout: 10_000 })
  }

  async closeAddSourcesModal(): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.addSourceModal.locator("button.ant-modal-close").first()
    )
    await expect(this.addSourceModal).toBeHidden({ timeout: 10_000 })
    await this.waitForModalBackdropsToClear()
  }

  async seedSources(sources: WorkspaceSeedSource[]): Promise<void> {
    await this.page.evaluate((seed) => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            getState?: () => {
              workspaceId?: string
              initializeWorkspace?: (name?: string) => void
              addSources?: (
                sources: Array<{
                  mediaId: number
                  title: string
                  type:
                    | "pdf"
                    | "video"
                    | "audio"
                    | "website"
                    | "document"
                    | "text"
                  url: string
                  status: "processing" | "ready" | "error"
                }>
              ) => void
            }
          }
        | undefined
      if (!store?.getState) {
        throw new Error("Workspace store is unavailable on window")
      }
      const state = store.getState()
      if (!state.workspaceId) {
        state.initializeWorkspace?.("Workspace E2E")
      }
      state.addSources?.(
        seed.map((source) => ({
          mediaId: source.mediaId,
          title: source.title,
          type: source.type || "document",
          url: source.url || `https://example.com/source-${source.mediaId}`,
          status: source.status || "ready"
        }))
      )
    }, sources)
  }

  async getSourceIds(): Promise<string[]> {
    return await this.page
      .locator("[data-source-id]")
      .evaluateAll((rows) =>
        rows
          .map((row) => row.getAttribute("data-source-id"))
          .filter((id): id is string => Boolean(id))
      )
  }

  getSourceRowByTitle(title: string): Locator {
    return this.page.locator("[data-source-id]").filter({ hasText: title }).first()
  }

  async selectSourceById(sourceId: string): Promise<void> {
    await this.disableNextJsPortalPointerInterception()
    const checkbox = this.page.locator(
      `[data-source-id="${sourceId}"] input[type="checkbox"]`
    )
    if (await checkbox.isChecked().catch(() => false)) {
      return
    }

    const hitArea = this.page.getByTestId(`source-checkbox-hitarea-${sourceId}`)
    if (await hitArea.isVisible().catch(() => false)) {
      await hitArea.click({ force: true })
      if (await checkbox.isChecked().catch(() => false)) {
        return
      }
    }

    const antCheckbox = this.page.locator(
      `[data-source-id="${sourceId}"] .ant-checkbox`
    )
    if (await antCheckbox.isVisible().catch(() => false)) {
      await antCheckbox.click({ force: true })
      if (await checkbox.isChecked().catch(() => false)) {
        return
      }
    }

    if (await checkbox.isVisible().catch(() => false)) {
      await checkbox.click({ force: true })
      return
    }
    await hitArea.click({ force: true })
  }

  async expectSourceSelected(sourceId: string): Promise<void> {
    await expect(
      this.page.locator(`[data-source-id="${sourceId}"] input[type="checkbox"]`)
    ).toBeChecked()
  }

  async selectSourceByTitle(title: string): Promise<void> {
    const row = this.getSourceRowByTitle(title)
    await expect(row).toBeVisible({ timeout: 10_000 })
    const sourceId = await row.getAttribute("data-source-id")
    if (!sourceId) {
      throw new Error(`Unable to resolve workspace source id for "${title}"`)
    }
    await this.selectSourceById(sourceId)
  }

  async expectSourceSelectedByTitle(title: string): Promise<void> {
    const row = this.getSourceRowByTitle(title)
    await expect(row).toBeVisible({ timeout: 10_000 })
    await expect(row.locator('input[type="checkbox"]')).toBeChecked()
  }

  getSelectedSourceTag(title: string): Locator {
    return this.chatPanel.locator(".ant-tag").filter({ hasText: title }).first()
  }

  getChatInput(): Locator {
    return this.chatPanel.locator("textarea").first()
  }

  async sendChatMessage(message: string): Promise<void> {
    const input = this.getChatInput()
    await expect(input).toBeVisible({ timeout: 10_000 })
    await input.fill(message)
    await this.chatPanel.getByRole("button", { name: /send/i }).click()
  }

  getGlobalSearchResult(text: string): Locator {
    return this.globalSearchModal.getByRole("button").filter({ hasText: text }).first()
  }

  getStudioOutputButton(label: string): Locator {
    return this.studioPanel.getByRole("button", { name: label, exact: true })
  }

  getStudioArtifactCards(): Locator {
    return this.studioPanel.locator("[data-testid^='studio-artifact-card-']")
  }
}

export default WorkspacePlaygroundPage
