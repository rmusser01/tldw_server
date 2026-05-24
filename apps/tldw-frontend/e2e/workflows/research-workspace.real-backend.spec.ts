/**
 * Research Workspace Real-Backend Workflow E2E Tests
 *
 * These scenarios intentionally avoid route stubbing to catch regressions
 * that only appear when the page boots against a live API server.
 */
import {
  type Page,
  type Response,
  type Request,
  type Locator
} from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import {
  seedAuth,
  fetchWithApiKey,
  TEST_CONFIG,
  generateTestId
} from "../utils/helpers"
import { ResearchWorkspacePage } from "../utils/page-objects/ResearchWorkspacePage"
import { QuizPage } from "../utils/page-objects/QuizPage"
import { FlashcardsPage } from "../utils/page-objects/FlashcardsPage"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const CHAT_BOOTSTRAP_ENDPOINT =
  /\/api\/v1\/(?:chats(?:\/|\?|$)|chat\/conversations(?:\/|\?|$))/i

type BootstrapResponse = {
  url: string
  status: number
}

type BootstrapRequestFailure = {
  url: string
  errorText: string
}

const trackChatBootstrapResponses = (page: Page) => {
  const responses: BootstrapResponse[] = []
  const failures: BootstrapRequestFailure[] = []
  const onResponse = (response: Response) => {
    if (response.request().method().toUpperCase() !== "GET") return
    if (!CHAT_BOOTSTRAP_ENDPOINT.test(response.url())) return
    responses.push({
      url: response.url(),
      status: response.status()
    })
  }
  const onRequestFailed = (request: Request) => {
    if (request.method().toUpperCase() !== "GET") return
    if (!CHAT_BOOTSTRAP_ENDPOINT.test(request.url())) return
    failures.push({
      url: request.url(),
      errorText: request.failure()?.errorText || "request failed"
    })
  }

  page.on("response", onResponse)
  page.on("requestfailed", onRequestFailed)

  return {
    responses,
    failures,
    dispose: () => {
      page.off("response", onResponse)
      page.off("requestfailed", onRequestFailed)
    }
  }
}

const assertChatBootstrapHealthy = async (
  responses: BootstrapResponse[],
  failures: BootstrapRequestFailure[]
): Promise<void> => {
  if (failures.length > 0) {
    throw new Error(
      `Chat bootstrap requests failed at network layer: ${JSON.stringify(failures)}`
    )
  }

  const failingResponses = responses.filter(({ status }) => status >= 400)
  expect(
    failingResponses,
    `Chat bootstrap endpoints returned non-success responses: ${JSON.stringify(
      failingResponses
    )}`
  ).toHaveLength(0)
}

const canReachChatBootstrapEndpoint = async (): Promise<{
  reachable: boolean
  reason?: string
}> => {
  const endpoint = `${TEST_CONFIG.serverUrl}/api/v1/chats/?limit=1&offset=0&ordering=-updated_at`
  try {
    const response = await fetchWithApiKey(endpoint)
    if (response.ok) {
      return { reachable: true }
    }
    return {
      reachable: false,
      reason: `GET /api/v1/chats preflight returned HTTP ${response.status}`
    }
  } catch (error) {
    return {
      reachable: false,
      reason:
        error instanceof Error ? error.message : "GET /api/v1/chats preflight failed"
    }
  }
}

const ensureNoServerReachabilityDialog = async (page: Page): Promise<void> => {
  const serverDialog = page
    .getByRole("dialog")
    .filter({ hasText: /can't reach your tldw server/i })
  await expect(serverDialog).toBeHidden({ timeout: 5_000 })
}

type LiveWorkspaceSource = {
  mediaId: number
  title: string
  type: "document"
  url: string
  content: string
}

type RagSearchCall = {
  requestBody: Record<string, unknown>
  responseBody: Record<string, unknown> | null
  status: number
}

type ChatCompletionCall = {
  requestBody: Record<string, unknown>
  responseBody: Record<string, unknown> | null
  status: number
}

type QuizListResponse = {
  items: Array<{
    id: number
    name: string
    workspace_id?: string | null
    workspace_tag?: string | null
    deleted?: boolean
  }>
}

type FlashcardListResponse = {
  items: Array<{
    uuid: string
    deck_id?: number | null
    front: string
    back: string
  }>
}

type DeckListItem = {
  id: number
  name: string
  workspace_id?: string | null
}

const normalizeWhitespace = (value: string): string =>
  value.replace(/\s+/g, " ").trim()

const normalizeAssistantMessageText = (value: string): string =>
  value
    .replace(/▋/g, "")
    .split(/\n+/)
    .map((line) => normalizeWhitespace(line))
    .filter(
      (line) =>
        line.length > 0 &&
        !/^(Mood:|Response complete$|Loading content(?:\.{3}|…)?)$/i.test(line)
    )
    .join(" ")

const waitForCompletedAssistantReply = async (
  workspacePage: ResearchWorkspacePage
): Promise<string> => {
  const assistantMessage = workspacePage.chatPanel.locator(
    "article[aria-label*='Assistant message'], [data-role='assistant'], [data-message-role='assistant'], .assistant-message"
  ).last()

  await expect(assistantMessage).toBeVisible({ timeout: 30_000 })

  const readCompletedReply = async (): Promise<string> => {
    const isGenerating = await assistantMessage
      .getByText(/Generating response/i)
      .isVisible()
      .catch(() => false)
    const hasStopStreaming = await assistantMessage
      .getByRole("button", {
        name: /Stop streaming response|Stop Streaming/i
      })
      .isVisible()
      .catch(() => false)
    const text = normalizeAssistantMessageText(
      (await assistantMessage.textContent().catch(() => "")) || ""
    )

    if (isGenerating || hasStopStreaming) {
      return ""
    }

    return text
  }

  await expect
    .poll(readCompletedReply, {
      timeout: 90_000,
      message: "Timed out waiting for the grounded workspace assistant reply"
    })
    .not.toBe("")

  return readCompletedReply()
}

const seedLiveWorkspaceDocument = async (
  title: string,
  content: string
): Promise<LiveWorkspaceSource> => {
  const fileName = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.txt`
  const body = new FormData()
  body.append("media_type", "document")
  body.append("title", title)
  body.append("perform_analysis", "false")
  body.append("perform_chunking", "false")
  body.append("files", new Blob([content], { type: "text/plain" }), fileName)

  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/add`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      body
    }
  )
  if (!response.ok) {
    throw new Error(
      `Failed to seed live workspace media "${title}": ${response.status} ${await response.text()}`
    )
  }

  const payload = await response.json().catch(() => ({}))
  const result = Array.isArray(payload?.results)
    ? payload.results[0]
    : payload?.result || payload
  const mediaId = Number(result?.db_id ?? result?.media_id ?? result?.id)
  if (!Number.isFinite(mediaId) || mediaId <= 0) {
    throw new Error(
      `Live workspace media seed for "${title}" returned no usable media id: ${JSON.stringify(
        payload
      )}`
    )
  }

  const expectedSnippet = normalizeWhitespace(content).slice(0, 48)
  await expect
    .poll(
      async () => {
        const details = await fetchWithApiKey(
          `${TEST_CONFIG.serverUrl}/api/v1/media/${mediaId}?include_content=true&include_versions=false&include_version_content=false`,
          TEST_CONFIG.apiKey
        )
        if (!details.ok) return ""
        const body = await details.json().catch(() => ({}))
        return normalizeWhitespace(
          String(
            body?.content?.text ??
              body?.content?.content ??
              body?.transcript ??
              ""
          )
        )
      },
      {
        timeout: 30_000,
        message: `Media ${mediaId} never exposed usable content for workspace grounding`
      }
    )
    .toContain(expectedSnippet)

  return {
    mediaId,
    title,
    type: "document",
    url: `file://${fileName}`,
    content
  }
}

const waitForRagSearchCall = async (
  page: Page,
  action: () => Promise<void>
): Promise<RagSearchCall> => {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method().toUpperCase() === "POST" &&
      /\/api\/v1\/rag\/search(?:\?|$)/i.test(response.url()),
    { timeout: 90_000 }
  )
  const [response] = await Promise.all([responsePromise, action()])
  return {
    requestBody:
      (response.request().postDataJSON() as Record<string, unknown>) || {},
    responseBody:
      (await response.json().catch(() => null)) as Record<string, unknown> | null,
    status: response.status()
  }
}

const waitForChatCompletionCall = async (
  page: Page,
  action: () => Promise<void>
): Promise<ChatCompletionCall> => {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method().toUpperCase() === "POST" &&
      /\/api\/v1\/chat\/completions(?:\?|$)/i.test(response.url()),
    { timeout: 90_000 }
  )
  const [response] = await Promise.all([responsePromise, action()])
  return {
    requestBody:
      (response.request().postDataJSON() as Record<string, unknown>) || {},
    responseBody:
      (await response.json().catch(() => null)) as Record<string, unknown> | null,
    status: response.status()
  }
}

const waitForStudioArtifactCompletion = async (artifactCard: Locator) => {
  await expect
    .poll(
      async () => {
        const downloadVisible = await artifactCard
          .getByRole("button", { name: /download/i })
          .isVisible()
          .catch(() => false)
        if (downloadVisible) {
          return "completed"
        }

        const cardText = (await artifactCard.textContent()) || ""
        if (/failed|encountered an error|no usable/i.test(cardText)) {
          return `failed:${cardText}`
        }

        return "pending"
      },
      {
        timeout: 180_000,
        message: "Studio artifact did not reach a completed state"
      }
    )
    .toBe("completed")
}

const buildSeedSources = () => {
  const base = Date.now() % 1_000_000
  return [
    {
      mediaId: 9_200_000 + base,
      title: "Workspace Real Backend Source A",
      type: "document" as const,
      url: "https://example.com/workspace-real-source-a"
    },
    {
      mediaId: 9_300_000 + base,
      title: "Workspace Real Backend Source B",
      type: "website" as const,
      url: "https://example.com/workspace-real-source-b"
    }
  ]
}

const fetchJsonWithApiKey = async <T>(path: string): Promise<T> => {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}${path}`,
    TEST_CONFIG.apiKey
  )
  if (!response.ok) {
    throw new Error(`GET ${path} failed with HTTP ${response.status}: ${await response.text()}`)
  }
  return (await response.json()) as T
}

const listQuizRecords = async (params: Record<string, string | number | boolean | undefined> = {}) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue
    search.set(key, String(value))
  }
  const suffix = search.toString().length > 0 ? `?${search.toString()}` : ""
  return await fetchJsonWithApiKey<QuizListResponse>(`/api/v1/quizzes${suffix}`)
}

const listDeckRecords = async (params: Record<string, string | number | boolean | undefined> = {}) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue
    search.set(key, String(value))
  }
  const suffix = search.toString().length > 0 ? `?${search.toString()}` : ""
  return await fetchJsonWithApiKey<DeckListItem[]>(`/api/v1/flashcards/decks${suffix}`)
}

const listFlashcardRecords = async (
  params: Record<string, string | number | boolean | undefined> = {},
) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue
    search.set(key, String(value))
  }
  const suffix = search.toString().length > 0 ? `?${search.toString()}` : ""
  return await fetchJsonWithApiKey<FlashcardListResponse>(`/api/v1/flashcards${suffix}`)
}

const waitForGeneratedArtifactRecord = async (
  workspacePage: ResearchWorkspacePage,
  artifactType: "quiz" | "flashcards"
) => {
  await expect
    .poll(async () => workspacePage.getGeneratedArtifactRecord(artifactType), {
      timeout: 120_000,
      message: `Workspace ${artifactType} artifact never exposed a persisted record`
    })
    .not.toBeNull()
  const artifact = await workspacePage.getGeneratedArtifactRecord(artifactType)
  if (!artifact) {
    throw new Error(`Workspace ${artifactType} artifact record missing after completion`)
  }
  return artifact
}

const waitForPersistedWorkspaceArtifact = async (
  workspacePage: ResearchWorkspacePage,
  artifactType: "quiz" | "flashcards"
) => {
  await expect
    .poll(
      async () => {
        const artifact = await workspacePage.getGeneratedArtifactRecord(artifactType)
        if (!artifact) {
          return "missing"
        }
        const normalizedStatus = String(artifact.status || "").toLowerCase()
        if (normalizedStatus === "failed") {
          return `failed:${artifact.status}`
        }
        const persistedId = Number(
          artifact.serverId ??
            artifact.data?.quizId ??
            artifact.data?.deckId ??
            Number.NaN
        )
        if (Number.isFinite(persistedId) && persistedId > 0) {
          return "persisted"
        }
        return "pending"
      },
      {
        timeout: 180_000,
        message: `Workspace ${artifactType} artifact never persisted a server record`
      }
    )
    .toBe("persisted")

  const artifact = await workspacePage.getGeneratedArtifactRecord(artifactType)
  if (!artifact) {
    throw new Error(`Workspace ${artifactType} artifact record missing after persistence`)
  }
  return artifact
}

const disableNextJsPortalPointerInterception = async (page: Page) => {
  await page.evaluate(() => {
    document.querySelectorAll("nextjs-portal").forEach((portal) => {
      ;(portal as HTMLElement).style.pointerEvents = "none"
    })
  })
}

const clickActionable = async (locator: Locator) => {
  try {
    await locator.click({ timeout: 5_000 })
  } catch (error) {
    if (!String(error).includes("nextjs-portal")) {
      throw error
    }
    await locator.focus()
    await expect(locator).toBeFocused({ timeout: 5_000 })
    await locator.press("Enter")
  }
}

test.describe("Research Workspace Workflow (Real Backend)", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    await page.setViewportSize(DESKTOP_VIEWPORT)
  })

  test("boots cleanly and keeps chat bootstrap endpoints healthy", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)
      await ensureNoServerReachabilityDialog(authedPage)
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("supports core workspace interactions with live API context", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.openGlobalSearchWithShortcut()
      await workspacePage.closeGlobalSearchWithEscape()
      await workspacePage.hideSourcesPane()
      await workspacePage.showSourcesPane()
      await workspacePage.hideStudioPane()
      await workspacePage.showStudioPane()
      await workspacePage.openAddSourcesModal()
      await workspacePage.closeAddSourcesModal()

      await workspacePage.seedSources(buildSeedSources())
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(2)
      const sourceIds = await workspacePage.getSourceIds()
      await workspacePage.selectSourceById(sourceIds[0])
      await workspacePage.expectSourceSelected(sourceIds[0])

      await ensureNoServerReachabilityDialog(authedPage)
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("grounds live chat requests on the selected source", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const fixtureId = generateTestId("workspace-chat-grounding")
    const probeToken = `${fixtureId}-beacon-fox`
    const selectedSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Alpha`,
      `Workspace chat grounding source. Token ${probeToken}. Deterministic verification requires stable include media ids.`
    )
    const unselectedSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Beta`,
      `Workspace chat distractor source. Token ${fixtureId}-distractor.`
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)
    const question = `What does the source say about ${probeToken}?`

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.seedSources([selectedSource, unselectedSource])
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(2)

      await workspacePage.selectSourceByTitle(selectedSource.title)
      await workspacePage.expectSourceSelectedByTitle(selectedSource.title)
      await expect(workspacePage.getSelectedSourceTag(selectedSource.title)).toBeVisible({
        timeout: 10_000
      })

      const ragCall = await waitForRagSearchCall(authedPage, async () => {
        await workspacePage.sendChatMessage(question)
      })

      expect(ragCall.status).toBe(200)
      expect(ragCall.requestBody.include_media_ids).toEqual([selectedSource.mediaId])
      expect(ragCall.requestBody.sources).toEqual(["media_db"])
      expect(String(ragCall.requestBody.query ?? "")).toContain(probeToken)
      await expect(workspacePage.chatPanel.getByText(question)).toBeVisible({
        timeout: 10_000
      })
      const groundedReply = await waitForCompletedAssistantReply(workspacePage)
      expect(groundedReply.length).toBeGreaterThan(0)
      expect(groundedReply).not.toMatch(
        /cannot reach server|unable to reach server|request failed|connection/i
      )

      await ensureNoServerReachabilityDialog(authedPage)
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("scopes studio compare-sources generation to the selected media ids", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    test.setTimeout(120_000)
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const fixtureId = generateTestId("workspace-studio-scope")
    const leftSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Left`,
      `Left comparison source. Token ${fixtureId}-left. Claim: alpha baseline improved by 11 percent.`
    )
    const rightSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Right`,
      `Right comparison source. Token ${fixtureId}-right. Claim: beta baseline improved by 19 percent.`
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.seedSources([leftSource, rightSource])
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(2)

      await workspacePage.selectSourceByTitle(leftSource.title)
      await workspacePage.selectSourceByTitle(rightSource.title)
      await workspacePage.expectSourceSelectedByTitle(leftSource.title)
      await workspacePage.expectSourceSelectedByTitle(rightSource.title)
      await expect(
        workspacePage.getStudioOutputButton("Compare Sources")
      ).toBeEnabled({ timeout: 10_000 })

      const beforeCount = await workspacePage.getStudioArtifactCards().count()
      const chatCall = await waitForChatCompletionCall(authedPage, async () => {
        await workspacePage.getStudioOutputButton("Compare Sources").click()
      })

      expect(chatCall.status).toBe(200)
      const messages = Array.isArray(chatCall.requestBody.messages)
        ? (chatCall.requestBody.messages as Array<Record<string, unknown>>)
        : []
      const requestText = messages
        .map((message) =>
          typeof message.content === "string" ? message.content : ""
        )
        .join("\n")
      expect(requestText).toContain(leftSource.title)
      expect(requestText).toContain(rightSource.title)
      expect(requestText).toContain(
        "Compare the selected sources and produce"
      )
      expect(
        String(
          ((chatCall.responseBody?.choices as Array<Record<string, unknown>> | undefined)?.[0]
            ?.message as Record<string, unknown> | undefined)?.content ?? ""
        ).trim().length
      ).toBeGreaterThan(0)

      await expect(workspacePage.getStudioArtifactCards()).toHaveCount(
        beforeCount + 1,
        { timeout: 90_000 }
      )
      const artifactCard = workspacePage.getStudioArtifactCards().first()
      await waitForStudioArtifactCompletion(artifactCard)
      await expect(artifactCard).toContainText(/Compare Sources/i)

      await ensureNoServerReachabilityDialog(authedPage)
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("searches live chat turns from the workspace global search surface", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const fixtureId = generateTestId("workspace-global-search")
    const probeToken = `${fixtureId}-search-token`
    const selectedSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Search`,
      `Workspace search source. Token ${probeToken}.`
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)
    const question = `What does the document say about ${probeToken}?`

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.seedSources([selectedSource])
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(1)

      await workspacePage.selectSourceByTitle(selectedSource.title)
      await workspacePage.expectSourceSelectedByTitle(selectedSource.title)

      const ragCall = await waitForRagSearchCall(authedPage, async () => {
        await workspacePage.sendChatMessage(question)
      })
      expect(ragCall.status).toBe(200)
      expect(ragCall.requestBody.include_media_ids).toEqual([selectedSource.mediaId])

      await workspacePage.openGlobalSearchWithShortcut()
      await workspacePage.searchWorkspace(`chat: ${probeToken}`)

      const chatResult = workspacePage.getGlobalSearchResult(probeToken)
      await expect(chatResult).toBeVisible({ timeout: 10_000 })
      await expect(chatResult).toContainText(/Chat/i)
      await workspacePage.globalSearchInput.press("Enter")
      await expect(workspacePage.globalSearchModal).toBeHidden({ timeout: 10_000 })

      await ensureNoServerReachabilityDialog(authedPage)
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("keeps a workspace-generated quiz hidden until forced visible, then moves it to general without changing its record id", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    test.setTimeout(300_000)
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const fixtureId = generateTestId("workspace-study-quiz")
    const probeToken = `${fixtureId}-quiz-token`
    const selectedSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Quiz`,
      `Workspace quiz source for ${probeToken}. The source is intentionally specific so the generated study artifact has stable content.`
    )
    const companionSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Quiz Companion`,
      `Workspace quiz companion source for ${probeToken}-companion. The second source keeps the generation path aligned with the existing workspace output matrix.`
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)
    const quizPage = new QuizPage(authedPage)

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.resetWorkspace(`Workspace ${fixtureId}`)
      await workspacePage.setStudyMaterialsPolicy("workspace")
      await workspacePage.seedSources([selectedSource, companionSource])
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(1)
      await workspacePage.selectSourceByTitle(selectedSource.title)
      await workspacePage.selectSourceByTitle(companionSource.title)
      await workspacePage.expectSourceSelectedByTitle(selectedSource.title)
      await workspacePage.expectSourceSelectedByTitle(companionSource.title)

      const beforeArtifacts = await workspacePage.getStudioArtifactCards().count()
      await disableNextJsPortalPointerInterception(authedPage)
      await clickActionable(workspacePage.getStudioOutputButton("Quiz"))

      await expect(workspacePage.getStudioArtifactCards()).toHaveCount(
        beforeArtifacts + 1,
        { timeout: 90_000 }
      )
      const quizArtifact = await waitForPersistedWorkspaceArtifact(
        workspacePage,
        "quiz"
      )
      const quizId = Number(quizArtifact.serverId ?? quizArtifact.data?.quizId)
      expect(Number.isFinite(quizId) && quizId > 0).toBe(true)
      const workspaceId = await workspacePage.getWorkspaceId()
      expect(workspaceId).toBeTruthy()

      const quizQuestions = await fetchJsonWithApiKey<{
        items: Array<{ id: number }>
      }>(`/api/v1/quizzes/${quizId}/questions?include_answers=true&limit=100&offset=0`)
      expect(quizQuestions.items.length).toBeGreaterThan(0)

      const persistedQuizList = await listQuizRecords({
        include_workspace_items: true,
        limit: 100,
        offset: 0
      })
      const persistedQuiz = persistedQuizList.items.find((item) => item.id === quizId)
      expect(persistedQuiz).toBeTruthy()
      expect(persistedQuiz).toMatchObject({
        id: quizId,
        workspace_id: workspaceId
      })

      await quizPage.goto()
      await quizPage.assertPageReady()
      await quizPage.switchToTab("manage")
      await expect(quizPage.getManageQuizStartButton(quizId)).toBeHidden({
        timeout: 10_000
      })

      await expect(quizPage.manageShowWorkspaceQuizzesToggle).toBeVisible({
        timeout: 10_000
      })
      await quizPage.manageShowWorkspaceQuizzesToggle.click()
      await expect(quizPage.getManageQuizStartButton(quizId)).toBeVisible({
        timeout: 10_000
      })

      await quizPage.gotoPath(
        `/quiz?tab=take&start_quiz_id=${quizId}&highlight_quiz_id=${quizId}&include_workspace_items=1`
      )
      await quizPage.assertPageReady()
      await expect(
        authedPage.getByRole("dialog").filter({ hasText: /Ready to begin\?/i })
      ).toBeVisible({
        timeout: 10_000
      })

      await quizPage.goto()
      await quizPage.assertPageReady()
      await quizPage.switchToTab("manage")
      await expect(quizPage.manageShowWorkspaceQuizzesToggle).toBeVisible({
        timeout: 10_000
      })
      await quizPage.manageShowWorkspaceQuizzesToggle.click()
      await quizPage.getManageQuizEditButton(quizId).click()
      const quizEditModal = authedPage.getByTestId("manage-edit-quiz-modal")
      const workspaceIdInput = quizEditModal.getByLabel("Workspace ID")
      await expect(workspaceIdInput).toBeVisible({ timeout: 10_000 })
      await workspaceIdInput.fill("")
      await quizEditModal.getByRole("button", { name: "Save" }).click()

      await expect
        .poll(
          async () => {
            const updatedQuiz = await fetchJsonWithApiKey<{
              id: number
              workspace_id?: string | null
            }>(`/api/v1/quizzes/${quizId}`)
            return updatedQuiz.workspace_id ?? null
          },
          {
            timeout: 30_000,
            message: "Quiz workspace scope never moved back to general"
          }
        )
        .toBeNull()

      const generalQuizList = await listQuizRecords({
        include_workspace_items: false,
        limit: 100,
        offset: 0
      })
      expect(generalQuizList.items.some((item) => item.id === quizId)).toBe(true)
      const scopedQuizList = await listQuizRecords({
        workspace_id: workspaceId || undefined,
        include_workspace_items: false,
        limit: 100,
        offset: 0
      })
      expect(scopedQuizList.items.some((item) => item.id === quizId)).toBe(false)

      await quizPage.goto()
      await quizPage.assertPageReady()
      await quizPage.switchToTab("manage")
      await expect(quizPage.getManageQuizStartButton(quizId)).toBeVisible({
        timeout: 10_000
      })
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("keeps a workspace-generated flashcards deck hidden until forced visible, then moves it to general without changing its record id", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    test.setTimeout(300_000)
    skipIfServerUnavailable(serverInfo)
    const chatBootstrapPreflight = await canReachChatBootstrapEndpoint()
    test.skip(
      !chatBootstrapPreflight.reachable,
      chatBootstrapPreflight.reason ||
        "Skipping real-backend workspace test: chat bootstrap endpoint unavailable"
    )

    const fixtureId = generateTestId("workspace-study-flashcards")
    const probeToken = `${fixtureId}-flashcards-token`
    const selectedSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Flashcards`,
      `Workspace flashcards source for ${probeToken}. The source is intentionally specific so the generated deck remains stable enough for native-page assertions.`
    )
    const companionSource = await seedLiveWorkspaceDocument(
      `WS ${fixtureId} Flashcards Companion`,
      `Workspace flashcards companion source for ${probeToken}-companion. The second source helps exercise the same multi-source path as the output matrix probe.`
    )

    const tracker = trackChatBootstrapResponses(authedPage)
    const workspacePage = new ResearchWorkspacePage(authedPage)
    const flashcardsPage = new FlashcardsPage(authedPage)

    try {
      await workspacePage.goto()
      await workspacePage.waitForReady()
      await assertChatBootstrapHealthy(tracker.responses, tracker.failures)

      await workspacePage.resetWorkspace(`Workspace ${fixtureId}`)
      await workspacePage.setStudyMaterialsPolicy("workspace")
      await workspacePage.seedSources([selectedSource, companionSource])
      await expect
        .poll(async () => (await workspacePage.getSourceIds()).length, {
          timeout: 10_000
        })
        .toBeGreaterThanOrEqual(1)
      await workspacePage.selectSourceByTitle(selectedSource.title)
      await workspacePage.selectSourceByTitle(companionSource.title)
      await workspacePage.expectSourceSelectedByTitle(selectedSource.title)
      await workspacePage.expectSourceSelectedByTitle(companionSource.title)

      const beforeArtifacts = await workspacePage.getStudioArtifactCards().count()
      await disableNextJsPortalPointerInterception(authedPage)
      await clickActionable(workspacePage.getStudioOutputButton("Flashcards"))

      await expect(workspacePage.getStudioArtifactCards()).toHaveCount(
        beforeArtifacts + 1,
        { timeout: 90_000 }
      )
      const flashcardArtifact = await waitForPersistedWorkspaceArtifact(
        workspacePage,
        "flashcards"
      )
      const deckId = Number(flashcardArtifact.serverId ?? flashcardArtifact.data?.deckId)
      expect(Number.isFinite(deckId) && deckId > 0).toBe(true)
      const workspaceId = await workspacePage.getWorkspaceId()
      expect(workspaceId).toBeTruthy()

      const flashcardList = await listFlashcardRecords({
        deck_id: deckId,
        include_workspace_items: true,
        limit: 100,
        offset: 0
      })
      expect(flashcardList.items.length).toBeGreaterThan(0)
      const firstCardUuid = flashcardList.items[0]?.uuid
      expect(firstCardUuid).toBeTruthy()

      const deckRecords = await listDeckRecords({
        include_workspace_items: true
      })
      const persistedDeck = deckRecords.find((deck) => deck.id === deckId)
      expect(persistedDeck).toBeTruthy()
      expect(persistedDeck).toMatchObject({
        id: deckId,
        workspace_id: workspaceId
      })
      const generalDecksBeforeMove = await listDeckRecords({
        include_workspace_items: false
      })
      expect(generalDecksBeforeMove.some((deck) => deck.id === deckId)).toBe(false)

      await flashcardsPage.gotoPath(
        `/flashcards?deck_id=${deckId}&include_workspace_items=1`
      )
      await flashcardsPage.assertPageReady()
      await expect(flashcardsPage.reviewDeckSelect).toBeVisible({
        timeout: 10_000
      })
      const deckName = persistedDeck?.name ?? `Deck ${deckId}`
      await expect(flashcardsPage.reviewDeckSelect.getByText(deckName, { exact: true })).toBeVisible({
        timeout: 10_000
      })
      await expect(flashcardsPage.reviewActiveCard).toBeVisible({ timeout: 10_000 })

      await flashcardsPage.gotoPath(
        `/flashcards?tab=manage&deck_id=${deckId}&include_workspace_items=1`
      )
      await flashcardsPage.assertPageReady()
      await expect(flashcardsPage.manageMoveScopeButton).toBeVisible({
        timeout: 10_000
      })
      await expect(flashcardsPage.getManageFlashcardRow(firstCardUuid)).toBeVisible({
        timeout: 10_000
      })
      await expect(flashcardsPage.manageMoveScopeButton).toBeEnabled({
        timeout: 10_000
      })
      await flashcardsPage.manageMoveScopeButton.click()
      const flashcardsMoveModal = authedPage
        .getByRole("dialog")
        .filter({ hasText: /Move deck scope/i })
      const workspaceIdInput = flashcardsMoveModal.getByLabel("Workspace ID")
      await expect(workspaceIdInput).toBeVisible({ timeout: 10_000 })
      await workspaceIdInput.fill("")
      await flashcardsMoveModal.getByRole("button", { name: "Save" }).click()

      await expect
        .poll(
          async () => {
            const updatedDecks = await listDeckRecords({
              include_workspace_items: false
            })
            return updatedDecks.some((deck) => deck.id === deckId)
          },
          {
            timeout: 30_000,
            message: "Flashcards deck never moved back to general scope"
          }
        )
        .toBe(true)

      const remainingWorkspaceDecks = await listDeckRecords({
        workspace_id: workspaceId || undefined,
        include_workspace_items: false
      })
      expect(remainingWorkspaceDecks.some((deck) => deck.id === deckId)).toBe(false)

      const updatedDecks = await listDeckRecords({
        include_workspace_items: false
      })
      expect(updatedDecks.some((deck) => deck.id === deckId)).toBe(true)
      const updatedFlashcardList = await listFlashcardRecords({
        deck_id: deckId,
        include_workspace_items: false,
        limit: 100,
        offset: 0
      })
      expect(updatedFlashcardList.items.length).toBeGreaterThan(0)

      await flashcardsPage.goto()
      await flashcardsPage.assertPageReady()
      await flashcardsPage.switchToTab("manage")
      await expect(flashcardsPage.getManageFlashcardRow(firstCardUuid)).toBeVisible({
        timeout: 10_000
      })
    } finally {
      tracker.dispose()
    }

    await assertNoCriticalErrors(diagnostics)
  })
})
