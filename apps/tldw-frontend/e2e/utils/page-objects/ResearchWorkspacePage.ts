/**
 * Page Object for Research Workspace workflow coverage
 */
import {
  type ConsoleMessage,
  type Locator,
  type Page,
  type Request,
  type Response,
  type TestInfo,
  expect
} from "@playwright/test"
import { dispatchKeyboardShortcut, waitForConnection } from "../helpers"

type WorkspaceSeedSource = {
  mediaId: number
  title: string
  type?: "pdf" | "video" | "audio" | "website" | "document" | "text"
  status?: "processing" | "ready" | "error"
  url?: string
}

export type ResearchWorkspaceUatPersona = "beginner-no-key" | "power-api-key"

export type ResearchWorkspaceRouteTiming = {
  route: "/research-workspace"
  url: string
  startedAt: string
  completedAt: string
  durationMs: number
  navigationDurationMs: number | null
  domContentLoadedMs: number | null
  loadEventMs: number | null
}

export type ResearchWorkspaceUatDiagnosticsSnapshot = {
  console: Array<{
    type: string
    text: string
    location?: {
      url: string
      lineNumber: number
    }
  }>
  pageErrors: Array<{
    message: string
    stack?: string
  }>
  requestFailures: Array<{
    url: string
    method: string
    errorText: string
  }>
  failedResponses: Array<{
    url: string
    method: string
    status: number
  }>
}

export type ResearchWorkspaceUatEvidence = {
  label: string
  persona: ResearchWorkspaceUatPersona
  url: string
  title: string
  capturedAt: string
  routeTiming: ResearchWorkspaceRouteTiming
  warmRouteTiming?: ResearchWorkspaceRouteTiming
  diagnostics: ResearchWorkspaceUatDiagnosticsSnapshot
}

export type ResearchWorkspaceUatDiagnosticsRecorder = {
  snapshot: () => ResearchWorkspaceUatDiagnosticsSnapshot
  dispose: () => void
}

const cloneDiagnostics = (
  data: ResearchWorkspaceUatDiagnosticsSnapshot
): ResearchWorkspaceUatDiagnosticsSnapshot => ({
  console: [...data.console],
  pageErrors: [...data.pageErrors],
  requestFailures: [...data.requestFailures],
  failedResponses: [...data.failedResponses]
})

const sanitizeEvidenceLabel = (label: string): string =>
  label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "research-workspace-uat"

export function startResearchWorkspaceUatDiagnostics(
  page: Page
): ResearchWorkspaceUatDiagnosticsRecorder {
  const data: ResearchWorkspaceUatDiagnosticsSnapshot = {
    console: [],
    pageErrors: [],
    requestFailures: [],
    failedResponses: []
  }

  const onConsole = (message: ConsoleMessage) => {
    const location = message.location()
    data.console.push({
      type: message.type(),
      text: message.text(),
      location: location.url
        ? {
            url: location.url,
            lineNumber: location.lineNumber
          }
        : undefined
    })
  }

  const onPageError = (error: Error) => {
    data.pageErrors.push({
      message: error.message,
      stack: error.stack
    })
  }

  const onRequestFailed = (request: Request) => {
    data.requestFailures.push({
      url: request.url(),
      method: request.method(),
      errorText: request.failure()?.errorText || "request failed"
    })
  }

  const onResponse = (response: Response) => {
    if (response.status() < 400) {
      return
    }
    const request = response.request()
    data.failedResponses.push({
      url: response.url(),
      method: request.method(),
      status: response.status()
    })
  }

  page.on("console", onConsole)
  page.on("pageerror", onPageError)
  page.on("requestfailed", onRequestFailed)
  page.on("response", onResponse)

  return {
    snapshot: () => cloneDiagnostics(data),
    dispose: () => {
      page.off("console", onConsole)
      page.off("pageerror", onPageError)
      page.off("requestfailed", onRequestFailed)
      page.off("response", onResponse)
    }
  }
}

export class ResearchWorkspacePage {
  readonly page: Page
  readonly headerTitle: Locator
  readonly workspacesButton: Locator
  readonly sourcesPanel: Locator
  readonly chatPanel: Locator
  readonly studioPanel: Locator
  readonly chatInput: Locator
  readonly restoreSourcesButton: Locator
  readonly restoreStudioButton: Locator
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
    this.chatInput = page.locator("#workspace-main-content textarea").first()
    this.restoreSourcesButton = page.getByTestId("workspace-restore-sources").first()
    this.restoreStudioButton = page.getByTestId("workspace-restore-studio").first()
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
      const styleId = "tldw-e2e-disable-nextjs-portal-pointer-events"
      if (!document.getElementById(styleId)) {
        const style = document.createElement("style")
        style.id = styleId
        style.textContent =
          "nextjs-portal, nextjs-portal * { pointer-events: none !important; }"
        document.head.appendChild(style)
      }

      const portals = document.querySelectorAll("nextjs-portal")
      portals.forEach((portal) => {
        ;(portal as HTMLElement).style.setProperty(
          "pointer-events",
          "none",
          "important"
        )
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
    } catch (error) {
      if (String(error).includes("nextjs-portal")) {
        await this.disableNextJsPortalPointerInterception()
        await locator.evaluate((node) => {
          ;(node as HTMLElement).click()
        })
        return
      }

      await this.disableNextJsPortalPointerInterception()
      await locator.click({ trial: true, timeout: 3_000 })
    }
    try {
      await locator.click({ timeout: 3_000 })
    } catch (error) {
      if (!String(error).includes("nextjs-portal")) {
        throw error
      }

      await this.disableNextJsPortalPointerInterception()
      await locator.evaluate((node) => {
        ;(node as HTMLElement).click()
      })
    }
  }

  async goto(): Promise<void> {
    await this.page.goto("/research-workspace", {
      waitUntil: "domcontentloaded"
    })
    await waitForConnection(this.page).catch(() => {})
    await this.disableNextJsPortalPointerInterception()
  }

  async gotoWithTiming(): Promise<ResearchWorkspaceRouteTiming> {
    const startedAtMs = Date.now()
    const startedAt = new Date(startedAtMs).toISOString()
    await this.goto()
    const completedAtMs = Date.now()
    const browserTiming = await this.page
      .evaluate(() => {
        const navigationEntry = performance.getEntriesByType("navigation")[0] as
          | PerformanceNavigationTiming
          | undefined
        if (!navigationEntry) {
          return null
        }
        return {
          navigationDurationMs: Math.round(navigationEntry.duration),
          domContentLoadedMs: Math.round(navigationEntry.domContentLoadedEventEnd),
          loadEventMs:
            navigationEntry.loadEventEnd > 0
              ? Math.round(navigationEntry.loadEventEnd)
              : null
        }
      })
      .catch(() => null)

    return {
      route: "/research-workspace",
      url: this.page.url(),
      startedAt,
      completedAt: new Date(completedAtMs).toISOString(),
      durationMs: completedAtMs - startedAtMs,
      navigationDurationMs: browserTiming?.navigationDurationMs ?? null,
      domContentLoadedMs: browserTiming?.domContentLoadedMs ?? null,
      loadEventMs: browserTiming?.loadEventMs ?? null
    }
  }

  async captureUatEvidence(
    label: string,
    testInfo: TestInfo,
    options: {
      persona: ResearchWorkspaceUatPersona
      routeTiming: ResearchWorkspaceRouteTiming
      warmRouteTiming?: ResearchWorkspaceRouteTiming
      diagnostics: ResearchWorkspaceUatDiagnosticsSnapshot
    }
  ): Promise<ResearchWorkspaceUatEvidence> {
    const safeLabel = sanitizeEvidenceLabel(label)
    const evidence: ResearchWorkspaceUatEvidence = {
      label,
      persona: options.persona,
      url: this.page.url(),
      title: await this.page.title().catch(() => ""),
      capturedAt: new Date().toISOString(),
      routeTiming: options.routeTiming,
      warmRouteTiming: options.warmRouteTiming,
      diagnostics: options.diagnostics
    }

    await testInfo.attach(`research-workspace-${safeLabel}.png`, {
      body: await this.page.screenshot({ fullPage: true }),
      contentType: "image/png"
    })
    await testInfo.attach(`research-workspace-${safeLabel}.json`, {
      body: JSON.stringify(evidence, null, 2),
      contentType: "application/json"
    })

    return evidence
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

  async expectComposerVisibleWithoutPageScroll(): Promise<void> {
    await expect(this.chatInput).toBeVisible({ timeout: 10_000 })
    await expect
      .poll(async () => await this.page.evaluate(() => window.scrollY), {
        timeout: 5_000,
        message: "Expected research workspace to avoid page-level scrolling"
      })
      .toBe(0)

    const box = await this.chatInput.boundingBox()
    const viewport = this.page.viewportSize()
    expect(box, "Expected workspace chat composer to have a bounding box").not.toBeNull()
    expect(viewport, "Expected workspace page to have a viewport").not.toBeNull()

    if (!box || !viewport) {
      return
    }

    expect(box.y).toBeGreaterThanOrEqual(0)
    expect(box.y + box.height).toBeLessThanOrEqual(viewport.height)
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
    await this.clickWhenActionable(this.restoreSourcesButton)
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
    await this.clickWhenActionable(this.restoreStudioButton)
    await expect(this.studioPanel).toBeVisible({ timeout: 10_000 })
  }

  async openAddSourcesModal(): Promise<void> {
    await expect(this.sourcesPanel).toBeVisible({ timeout: 10_000 })
    await this.disableNextJsPortalPointerInterception()
    await this.clickWhenActionable(
      this.sourcesPanel
        .getByRole("button", { name: /^add sources$/i })
        .or(this.sourcesPanel.getByRole("button", { name: /^add$/i }))
        .first()
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
    return this.chatInput.first()
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

export default ResearchWorkspacePage
