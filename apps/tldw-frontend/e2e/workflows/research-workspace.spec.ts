/**
 * Research Workspace Workflow E2E Tests
 *
 * Dedicated interaction coverage for /research-workspace beyond smoke checks.
 */
import { test, expect, assertNoCriticalErrors } from "../utils/fixtures"
import { seedAuth } from "../utils/helpers"
import { ResearchWorkspacePage } from "../utils/page-objects"
import type { Locator, Page } from "@playwright/test"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const SUMMARY_TEST_MODEL = "gpt-4o-mini"
const SUMMARY_SOURCE_TEXT =
  "Workspace studio source content that should keep generation busy until the test releases it."

const createDeferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

const setWorkspaceSelectedModel = async (page: Parameters<typeof seedAuth>[0]) => {
  await page.evaluate((modelId) => {
    const store = (window as { __tldw_useStoreMessageOption?: unknown })
      .__tldw_useStoreMessageOption as
        | {
            setState?: (nextState: Record<string, unknown>) => void
          }
        | undefined
    if (!store?.setState) {
      throw new Error("Message option store is unavailable on window")
    }
    store.setState({ selectedModel: modelId })
  }, SUMMARY_TEST_MODEL)
}

const locatorContainsFocus = async (locator: Locator): Promise<boolean> =>
  (await locator.count()) > 0
    ? locator
        .evaluate((element) => {
          const active = document.activeElement
          const activeIsDocumentRoot =
            active === document.body || active === document.documentElement
          const activeIsCheckboxWrapper =
            active instanceof HTMLElement &&
            active.matches("label, .ant-checkbox-wrapper, [role='checkbox']")
          return (
            active === element ||
            (active instanceof Node && element.contains(active)) ||
            (active instanceof HTMLElement &&
              !activeIsDocumentRoot &&
              activeIsCheckboxWrapper &&
              active.contains(element))
          )
        })
        .catch(() => false)
    : false

const describeActiveElement = async (page: Page): Promise<string> =>
  page.evaluate(() => {
    const active = document.activeElement as HTMLElement | null
    if (!active) return "none"
    const label = active.getAttribute("aria-label")
    const role = active.getAttribute("role")
    const testId = active.getAttribute("data-testid")
    const text = active.textContent?.replace(/\s+/g, " ").trim().slice(0, 80)
    return [
      active.tagName.toLowerCase(),
      role ? `role=${role}` : null,
      label ? `aria-label=${label}` : null,
      testId ? `data-testid=${testId}` : null,
      text ? `text=${text}` : null
    ]
      .filter(Boolean)
      .join(" ")
  })

const pressTabUntilFocused = async (
  page: Page,
  locator: Locator,
  maxTabs = 40
): Promise<void> => {
  for (let attempt = 0; attempt < maxTabs; attempt += 1) {
    if (await locatorContainsFocus(locator)) {
      return
    }
    await page.keyboard.press("Tab")
  }

  throw new Error(
    `Expected keyboard focus to reach target after ${maxTabs} Tab presses; active=${await describeActiveElement(
      page
    )}`
  )
}

test.describe("Research Workspace Workflow", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    await page.setViewportSize(DESKTOP_VIEWPORT)

    await page.route("**/api/v1/llm/models/metadata**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ models: [] })
      })
    })

    await page.route("**/api/v1/chat/commands**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ commands: [] })
      })
    })

    await page.route("**/api/v1/chats**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 })
      })
    })

    await page.route("**/api/v1/chats/**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 })
      })
    })

    await page.route("**/api/v1/chat/conversations**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 })
      })
    })

    await page.route("**/api/v1/chat/conversations/**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 })
      })
    })
  })

  test("loads research workspace and renders core panes", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await expect(workspacePage.headerTitle).toBeVisible()
    await expect(workspacePage.sourcesPanel).toBeVisible()
    await expect(workspacePage.chatPanel).toBeVisible()
    await expect(workspacePage.studioPanel).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("opens and closes workspace search modal with keyboard shortcut", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.openGlobalSearchWithShortcut()
    await expect(
      workspacePage.globalSearchModal.getByPlaceholder(
        /search sources, chat, and notes/i
      )
    ).toBeVisible()

    await workspacePage.closeGlobalSearchWithEscape()

    await assertNoCriticalErrors(diagnostics)
  })

  test("collapses and restores sources + studio panes", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.hideSourcesPane()
    await expect(workspacePage.restoreSourcesButton).toBeVisible()
    await workspacePage.showSourcesPane()
    await workspacePage.hideStudioPane()
    await expect(workspacePage.restoreStudioButton).toBeVisible()
    await workspacePage.showStudioPane()

    await assertNoCriticalErrors(diagnostics)
  })

  test("keeps the chat composer visible on first load", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.expectComposerVisibleWithoutPageScroll()

    await assertNoCriticalErrors(diagnostics)
  })

  test("opens and dismisses add-sources modal from sources pane", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.openAddSourcesModal()
    await workspacePage.closeAddSourcesModal()

    await assertNoCriticalErrors(diagnostics)
  })

  test("preserves advanced source filters and temporary sort across sources pane remounts", async ({
    authedPage,
    diagnostics
  }) => {
    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.seedSources([
      {
        mediaId: 8_833_001,
        title: "Zulu Ready",
        type: "pdf",
        status: "ready"
      },
      {
        mediaId: 8_833_002,
        title: "Alpha Error",
        type: "website",
        status: "error"
      },
      {
        mediaId: 8_833_003,
        title: "Bravo Ready",
        type: "document",
        status: "ready"
      }
    ])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(3)

    const getRenderedSourceTitles = async (): Promise<string[]> =>
      workspacePage.sourcesPanel
        .locator("[data-source-id]")
        .evaluateAll((rows) =>
          rows
            .map((row) => row.querySelector("p")?.textContent?.trim() || "")
            .filter(Boolean)
        )

    await workspacePage.sourcesPanel
      .getByRole("button", { name: "Advanced" })
      .click()
    await workspacePage.sourcesPanel
      .getByRole("checkbox", { name: "Status Ready" })
      .click()
    await workspacePage.sourcesPanel
      .getByRole("combobox", { name: "Sort by" })
      .selectOption("name_asc")

    await expect
      .poll(async () => await getRenderedSourceTitles(), { timeout: 10_000 })
      .toEqual(["Bravo Ready", "Zulu Ready"])
    await workspacePage.sourcesPanel
      .getByRole("button", { name: "Advanced" })
      .click()
    await expect(
      workspacePage.sourcesPanel.getByText("Status=Ready · Sort: Name (A-Z)")
    ).toBeVisible()

    await workspacePage.hideSourcesPane()
    await workspacePage.showSourcesPane()

    await expect(
      workspacePage.sourcesPanel.getByText("Status=Ready · Sort: Name (A-Z)")
    ).toBeVisible()
    await workspacePage.sourcesPanel
      .getByRole("button", { name: "Advanced" })
      .click()
    await expect(
      workspacePage.sourcesPanel.getByRole("checkbox", { name: "Status Ready" })
    ).toBeChecked()
    await expect(
      workspacePage.sourcesPanel.getByRole("combobox", { name: "Sort by" })
    ).toHaveValue("name_asc")
    await expect
      .poll(async () => await getRenderedSourceTitles(), { timeout: 10_000 })
      .toEqual(["Bravo Ready", "Zulu Ready"])

    await assertNoCriticalErrors(diagnostics)
  })

  test("adds a source through the URL tab and shows processing state in the sources pane", async ({
    authedPage,
    diagnostics
  }) => {
    const addedSource = {
      mediaId: 8_811_001,
      title: "Workspace URL Ingested Source",
      url: "https://example.com/workspace-url-ingested",
      mediaType: "website"
    }
    const keywordUpdates: Array<{
      keywords?: string[]
      mode?: string
    }> = []

    await authedPage.route("**/api/v1/media/add", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          results: [
            {
              media_id: addedSource.mediaId,
              title: addedSource.title,
              url: addedSource.url,
              media_type: addedSource.mediaType
            }
          ]
        })
      })
    })

    await authedPage.route("**/api/v1/media/*/keywords", async (route) => {
      keywordUpdates.push(route.request().postDataJSON() as { keywords?: string[]; mode?: string })
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          media_id: addedSource.mediaId,
          keywords: keywordUpdates.at(-1)?.keywords || []
        })
      })
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.openAddSourcesModal()
    await workspacePage.addSourceModal.getByRole("tab", { name: /^url$/i }).click()
    await workspacePage.addSourceModal
      .getByPlaceholder("https://example.com/article or YouTube URL")
      .fill(addedSource.url)
    await workspacePage.addSourceModal.getByRole("button", { name: /^add url$/i }).click()

    await expect(workspacePage.addSourceModal).toBeHidden({ timeout: 10_000 })

    const sourceRow = workspacePage.sourcesPanel
      .locator("[data-source-id]", { hasText: addedSource.title })
      .first()
    await expect(sourceRow).toBeVisible()
    await expect(
      sourceRow.locator("span").filter({ hasText: /^processing$/i }).first()
    ).toBeVisible()
    await expect(sourceRow.locator('input[type="checkbox"]')).toBeDisabled()
    await expect
      .poll(() => keywordUpdates.length, { timeout: 10_000 })
      .toBeGreaterThanOrEqual(1)
    await expect
      .poll(() => keywordUpdates[0]?.mode || null, { timeout: 10_000 })
      .toBe("add")
    expect(keywordUpdates[0]?.keywords?.[0]).toMatch(/^workspace:/)

    await assertNoCriticalErrors(diagnostics)
  })

  test("adds a ready source from My Media and allows selecting it without store seeding", async ({
    authedPage,
    diagnostics
  }) => {
    const librarySource = {
      mediaId: 8_822_002,
      title: "Workspace Library Source",
      type: "pdf",
      url: "https://example.com/workspace-library-source.pdf"
    }
    const keywordUpdates: Array<{
      keywords?: string[]
      mode?: string
    }> = []

    await authedPage.route("**/api/v1/media?*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          media: [
            {
              id: librarySource.mediaId,
              title: librarySource.title,
              type: librarySource.type,
              url: librarySource.url
            }
          ],
          total_count: 1
        })
      })
    })

    await authedPage.route("**/api/v1/media/*/keywords", async (route) => {
      keywordUpdates.push(route.request().postDataJSON() as { keywords?: string[]; mode?: string })
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          media_id: librarySource.mediaId,
          keywords: keywordUpdates.at(-1)?.keywords || []
        })
      })
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.openAddSourcesModal()
    await workspacePage.addSourceModal.getByRole("tab", { name: /my media/i }).click()
    await expect(
      workspacePage.addSourceModal.getByText(librarySource.title)
    ).toBeVisible()

    await workspacePage.addSourceModal.getByText(librarySource.title).click()
    await workspacePage.addSourceModal
      .getByRole("button", { name: /^add 1 selected$/i })
      .click()

    await expect(workspacePage.addSourceModal).toBeHidden({ timeout: 10_000 })

    const sourceRow = workspacePage.sourcesPanel
      .locator("[data-source-id]", { hasText: librarySource.title })
      .first()
    await expect(sourceRow).toBeVisible()

    const sourceId = await sourceRow.getAttribute("data-source-id")
    expect(sourceId).toBeTruthy()

    await workspacePage.selectSourceById(sourceId!)
    await workspacePage.expectSourceSelected(sourceId!)
    await expect(workspacePage.sourcesPanel.getByText(/^1 selected$/)).toBeVisible()
    await expect
      .poll(() => keywordUpdates[0]?.mode || null, { timeout: 10_000 })
      .toBe("add")
    expect(keywordUpdates[0]?.keywords?.[0]).toMatch(/^workspace:/)

    await assertNoCriticalErrors(diagnostics)
  })

  test("submits grounded chat for selected sources and reopens the matching assistant turn from workspace search", async ({
    authedPage,
    diagnostics
  }) => {
    const sourceMediaId = 8_844_003
    const userQuestion = "What does the grounded workspace source say about evidence handling?"
    const answerSearchToken = "workspace-chat-search-token"
    const groundedAnswer = `Grounded answer for ${answerSearchToken}`
    const ragRequests: Array<Record<string, unknown>> = []
    const chatCompletionRequests: Array<Record<string, unknown>> = []
    const streamChunk = (text: string) =>
      `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`

    await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "text/plain",
        body: ""
      })
    })

    await authedPage.route("**/api/v1/rag/search", async (route) => {
      ragRequests.push((route.request().postDataJSON() as Record<string, unknown>) || {})
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          results: [
            {
              id: "workspace-grounded-source-1",
              content: "A source excerpt about handling evidence carefully and citing your sources.",
              metadata: {
                title: "Workspace Grounded Source",
                source: "Workspace Grounded Source",
                source_type: "media_db",
                media_id: sourceMediaId,
                url: "https://example.com/workspace-grounded-source"
              },
              score: 0.93
            }
          ],
          generated_answer: groundedAnswer,
          answer: groundedAnswer,
          citations: [
            {
              source: "Workspace Grounded Source",
              media_id: sourceMediaId
            }
          ]
        })
      })
    })

    await authedPage.route("**/api/v1/chat/completions", async (route) => {
      chatCompletionRequests.push(
        (route.request().postDataJSON() as Record<string, unknown>) || {}
      )
      const midpoint = Math.max(1, Math.floor(groundedAnswer.length / 2))
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          Connection: "keep-alive"
        },
        body:
          streamChunk(groundedAnswer.slice(0, midpoint)) +
          streamChunk(groundedAnswer.slice(midpoint)) +
          "data: [DONE]\n\n"
      })
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()
    await setWorkspaceSelectedModel(authedPage)

    await workspacePage.seedSources([
      {
        mediaId: sourceMediaId,
        title: "Workspace Grounded Source",
        type: "document",
        status: "ready"
      }
    ])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(1)

    const [sourceId] = await workspacePage.getSourceIds()
    await workspacePage.selectSourceById(sourceId)
    await workspacePage.expectSourceSelected(sourceId)
    await expect(
      workspacePage.chatPanel.getByText("Answers will be grounded in your selected sources")
    ).toBeVisible()

    const chatInput = workspacePage.chatPanel.getByPlaceholder(/ask about your sources/i)
    await chatInput.fill(userQuestion)
    await chatInput.press("Enter")

    await expect
      .poll(() => ragRequests.length, { timeout: 10_000 })
      .toBe(1)
    expect(ragRequests[0]?.query).toBe(userQuestion)
    expect(ragRequests[0]?.include_media_ids).toEqual([sourceMediaId])
    await expect
      .poll(() => chatCompletionRequests.length, { timeout: 10_000 })
      .toBe(1)
    expect(chatCompletionRequests[0]?.model).toBe(SUMMARY_TEST_MODEL)

    await expect(workspacePage.chatPanel.getByText(groundedAnswer)).toBeVisible({
      timeout: 10_000
    })
    const citationsToggle = workspacePage.chatPanel
      .getByRole("button", { name: /citations/i })
      .first()
    await expect(citationsToggle).toBeVisible()
    await citationsToggle.click()
    await expect(citationsToggle).toHaveAttribute("aria-expanded", "true")

    const assistantMessage = workspacePage.chatPanel
      .locator("[data-chat-message-id]", { hasText: groundedAnswer })
      .first()
    await expect
      .poll(async () => (await assistantMessage.getAttribute("class")) || "")
      .not.toContain("ring-2")

    await workspacePage.openGlobalSearchWithShortcut()
    await workspacePage.globalSearchInput.fill(answerSearchToken)

    const assistantResult = workspacePage.globalSearchModal
      .getByRole("button", { name: /assistant message/i })
      .first()
    await expect(assistantResult).toBeVisible()
    await assistantResult.click()

    await expect(workspacePage.globalSearchModal).toBeHidden({ timeout: 10_000 })
    await expect
      .poll(async () => (await assistantMessage.getAttribute("class")) || "", {
        timeout: 10_000
      })
      .toContain("ring-2")

    await assertNoCriticalErrors(diagnostics)
  })

  test("supports keyboard-only source selection and grounded chat submission", async ({
    authedPage,
    diagnostics
  }) => {
    const sourceMediaId = 8_844_903
    const sourceTitle = "Keyboard Grounded Source"
    const userQuestion = "What evidence does the keyboard-selected source provide?"
    const groundedAnswer = "Keyboard selected source evidence is cited and ready."
    const ragRequests: Array<Record<string, unknown>> = []
    const streamChunk = (text: string) =>
      `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`

    await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "text/plain",
        body: ""
      })
    })

    await authedPage.route("**/api/v1/rag/search", async (route) => {
      ragRequests.push((route.request().postDataJSON() as Record<string, unknown>) || {})
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          results: [
            {
              id: "keyboard-source-result",
              content: "Evidence from the keyboard-selected source.",
              metadata: {
                title: sourceTitle,
                source: sourceTitle,
                source_type: "media_db",
                media_id: sourceMediaId
              },
              score: 0.92
            }
          ],
          generated_answer: groundedAnswer,
          answer: groundedAnswer,
          citations: [{ source: sourceTitle, media_id: sourceMediaId }]
        })
      })
    })

    await authedPage.route("**/api/v1/chat/completions", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          Connection: "keep-alive"
        },
        body: streamChunk(groundedAnswer) + "data: [DONE]\n\n"
      })
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()
    await setWorkspaceSelectedModel(authedPage)

    await workspacePage.seedSources([
      {
        mediaId: sourceMediaId,
        title: sourceTitle,
        type: "document",
        status: "ready"
      }
    ])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(1)
    const [sourceId] = await workspacePage.getSourceIds()
    const sourceCheckbox = workspacePage.sourcesPanel.getByRole("checkbox", {
      name: `Select ${sourceTitle}`
    })

    await expect(sourceCheckbox).toHaveCount(1, { timeout: 5_000 })
    const skipToSourcesLink = authedPage.getByRole("link", {
      name: /skip to sources panel/i
    })
    await pressTabUntilFocused(authedPage, skipToSourcesLink, 40)
    await skipToSourcesLink.press("Enter")
    await expect
      .poll(
        () =>
          workspacePage.sourcesPanel.evaluate((panel) =>
            panel.contains(document.activeElement)
          ),
        { timeout: 5_000 }
      )
      .toBe(true)
    await pressTabUntilFocused(authedPage, sourceCheckbox)
    await expect
      .poll(() => locatorContainsFocus(sourceCheckbox), { timeout: 5_000 })
      .toBe(true)
    await expect
      .poll(
        async () =>
          authedPage.evaluate(() => {
            const active = document.activeElement as HTMLElement | null
            return active
              ?.closest("[data-source-id]")
              ?.getAttribute("data-source-id") ?? null
          }),
        { timeout: 5_000 }
      )
      .toBe(sourceId)

    await authedPage.keyboard.press("Space")
    await workspacePage.expectSourceSelected(sourceId)
    await expect(workspacePage.getSelectedSourceTag(sourceTitle)).toBeVisible()

    const chatInput = workspacePage.chatPanel.getByPlaceholder(/ask about your sources/i)
    await pressTabUntilFocused(authedPage, chatInput, 80)
    await expect(chatInput).toBeFocused({ timeout: 5_000 })
    await authedPage.keyboard.type(userQuestion)
    await expect(chatInput).toHaveValue(userQuestion)
    await authedPage.keyboard.press("Enter")

    await expect
      .poll(() => ragRequests.length, { timeout: 10_000 })
      .toBe(1)
    expect(ragRequests[0]?.query).toBe(userQuestion)
    expect(ragRequests[0]?.include_media_ids).toEqual([sourceMediaId])
    await expect(workspacePage.chatPanel.getByText(groundedAnswer)).toBeVisible({
      timeout: 10_000
    })

    await assertNoCriticalErrors(diagnostics)
  })

  test("generates compare-sources output for two selected sources through the studio pane", async ({
    authedPage,
    diagnostics
  }) => {
    const sourceA = {
      mediaId: 8_844_101,
      title: "Workspace Comparison Source A",
      type: "document" as const
    }
    const sourceB = {
      mediaId: 8_844_102,
      title: "Workspace Comparison Source B",
      type: "website" as const
    }
    const comparisonToken = "workspace-compare-sources-token"
    const comparisonOutput = `## Agreements\n- Both sources reference ${comparisonToken}.\n\n## Disagreements\n- Source B adds extra caveats.\n\n## Evidence Strength\n- Evidence is moderate.\n\n## Open Questions\n- Verify the missing caveat.`
    const compareRequests: Array<Record<string, unknown>> = []

    await authedPage.route("**/api/v1/rag/search", async (route) => {
      compareRequests.push((route.request().postDataJSON() as Record<string, unknown>) || {})
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          generation: comparisonOutput,
          usage: {
            total_tokens: 321,
            total_cost_usd: 0.12
          },
          citations: [
            { source: sourceA.title, media_id: sourceA.mediaId },
            { source: sourceB.title, media_id: sourceB.mediaId }
          ],
          results: [
            {
              id: "compare-source-a",
              content: "Source A evidence",
              metadata: {
                title: sourceA.title,
                media_id: sourceA.mediaId
              },
              score: 0.91
            },
            {
              id: "compare-source-b",
              content: "Source B evidence",
              metadata: {
                title: sourceB.title,
                media_id: sourceB.mediaId
              },
              score: 0.88
            }
          ]
        })
      })
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()

    await workspacePage.seedSources([sourceA, sourceB])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(2)

    const sourceIds = await workspacePage.getSourceIds()
    const compareButton = workspacePage.studioPanel.getByRole("button", {
      name: /^Compare Sources$/i
    })

    await workspacePage.selectSourceById(sourceIds[0])
    await workspacePage.expectSourceSelected(sourceIds[0])
    await expect(compareButton).toBeDisabled()

    await workspacePage.selectSourceById(sourceIds[1])
    await workspacePage.expectSourceSelected(sourceIds[1])
    await expect(workspacePage.sourcesPanel.getByText(/^2 selected$/)).toBeVisible()
    await expect(compareButton).toBeEnabled()

    await compareButton.click()

    await expect
      .poll(() => compareRequests.length, { timeout: 10_000 })
      .toBe(1)
    expect(compareRequests[0]?.query).toBe("agreements disagreements claims evidence")
    expect(compareRequests[0]?.include_media_ids).toEqual([
      sourceA.mediaId,
      sourceB.mediaId
    ])
    expect(compareRequests[0]?.top_k).toBe(30)
    expect(compareRequests[0]?.enable_generation).toBe(true)
    expect(compareRequests[0]?.enable_citations).toBe(true)
    expect(String(compareRequests[0]?.generation_prompt || "")).toContain(
      "Compare the selected sources"
    )

    const artifactCard = workspacePage.studioPanel
      .locator('[data-testid^="studio-artifact-card-"]')
      .first()
    await expect(artifactCard).toBeVisible({ timeout: 10_000 })
    await expect(artifactCard).toContainText("Compare Sources")
    await expect(artifactCard).toContainText("Tokens: 321")
    await expect(artifactCard).toContainText("Cost: $0.120")
    await expect(workspacePage.studioPanel).toContainText(
      "Estimated workspace usage: 321 tokens • $0.120"
    )
    await artifactCard.getByRole("button", { name: /view/i }).click()
    await expect(authedPage.getByText(comparisonToken)).toBeVisible({ timeout: 10_000 })
    await expect(
      artifactCard.getByRole("button", { name: /download/i })
    ).toBeVisible({ timeout: 10_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("cancels in-flight summary generation and marks the artifact failed", async ({
    authedPage,
    diagnostics
  }) => {
    const mediaDetailsRequested = createDeferred()
    const releaseMediaDetails = createDeferred()

    await authedPage.route(/\/api\/v1\/media\/\d+\?.*include_content=true/i, async (route) => {
      mediaDetailsRequested.resolve()
      await releaseMediaDetails.promise
      try {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            source: { title: "Research Workspace Summary Source" },
            content: { text: SUMMARY_SOURCE_TEXT }
          })
        })
      } catch {
        await route.abort().catch(() => {})
      }
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()
    await setWorkspaceSelectedModel(authedPage)

    await workspacePage.seedSources([
      {
        mediaId: 8_844_001,
        title: "Research Workspace Summary Source",
        type: "document",
        status: "ready"
      }
    ])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(1)

    const [sourceId] = await workspacePage.getSourceIds()
    await workspacePage.selectSourceById(sourceId)
    await workspacePage.expectSourceSelected(sourceId)

    await workspacePage.studioPanel
      .getByRole("button", { name: /^Summary$/i })
      .click()

    await mediaDetailsRequested.promise
    await expect(
      workspacePage.studioPanel.getByRole("button", { name: /^Cancel$/i })
    ).toBeVisible()
    await expect(
      workspacePage.studioPanel.locator('[data-testid^="studio-artifact-card-"]')
    ).toHaveCount(1)

    await workspacePage.studioPanel
      .getByRole("button", { name: /^Cancel$/i })
      .click()
    releaseMediaDetails.resolve()

    await expect(
      workspacePage.studioPanel.getByText(
        "Generation canceled before completion."
      )
    ).toBeVisible({ timeout: 10_000 })
    await expect(
      workspacePage.studioPanel.locator('[data-testid^="studio-artifact-card-"]')
    ).toHaveCount(1)

    await assertNoCriticalErrors(diagnostics)
  })

  test("recovers interrupted summary generation as a failed artifact after reload", async ({
    authedPage,
    diagnostics
  }) => {
    const mediaDetailsRequested = createDeferred()
    const releaseMediaDetails = createDeferred()

    await authedPage.route(/\/api\/v1\/media\/\d+\?.*include_content=true/i, async (route) => {
      mediaDetailsRequested.resolve()
      await releaseMediaDetails.promise
      try {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            source: { title: "Research Workspace Reload Source" },
            content: { text: SUMMARY_SOURCE_TEXT }
          })
        })
      } catch {
        await route.abort().catch(() => {})
      }
    })

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()
    await setWorkspaceSelectedModel(authedPage)

    await workspacePage.seedSources([
      {
        mediaId: 8_844_002,
        title: "Research Workspace Reload Source",
        type: "document",
        status: "ready"
      }
    ])

    await expect
      .poll(async () => (await workspacePage.getSourceIds()).length, {
        timeout: 10_000
      })
      .toBe(1)

    const [sourceId] = await workspacePage.getSourceIds()
    await workspacePage.selectSourceById(sourceId)
    await workspacePage.expectSourceSelected(sourceId)

    await workspacePage.studioPanel
      .getByRole("button", { name: /^Summary$/i })
      .click()

    await mediaDetailsRequested.promise
    await expect(
      workspacePage.studioPanel.getByRole("button", { name: /^Cancel$/i })
    ).toBeVisible()
    await expect(
      workspacePage.studioPanel.locator('[data-testid^="studio-artifact-card-"]')
    ).toHaveCount(1)

    await authedPage.reload({ waitUntil: "domcontentloaded" })
    releaseMediaDetails.resolve()
    await workspacePage.waitForReady()

    await expect(
      workspacePage.studioPanel.getByText(
        "Generation was interrupted. Click regenerate to try again."
      )
    ).toBeVisible({ timeout: 10_000 })
    await expect(
      workspacePage.studioPanel.getByRole("button", { name: /^Cancel$/i })
    ).toHaveCount(0)

    await assertNoCriticalErrors(diagnostics)
  })
})
