/**
 * Manual local verification for Research Workspace real provider generation.
 *
 * This file is intentionally opt-in. It requires a locally running API server,
 * WebUI, valid saved auth, and a configured real LLM provider. It must not be
 * enabled in CI.
 */
import os from "node:os"
import path from "node:path"
import { type Page } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import {
  fetchWithApiKey,
  generateTestId,
  seedAuth,
  TEST_CONFIG
} from "../utils/helpers"
import { ResearchWorkspacePage } from "../utils/page-objects/ResearchWorkspacePage"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const MOBILE_VIEWPORT = { width: 390, height: 844 }
const EXPECTED_CAPABILITIES = [
  "source_browse",
  "chat",
  "artifact_text_generation",
  "slides_generation",
  "audio_summary",
  "export_download",
  "sync_share"
]

type LiveWorkspaceSource = {
  mediaId: number
  title: string
  type: "document"
  url: string
}

type SummaryArtifactSnapshot = {
  id: string
  status: string
  type: string
  title: string
  contentLength: number
}

type ChatCompletionCapture = {
  status: number
  model: string | null
  provider: string | null
  responseContentLength: number
}

const normalizeWhitespace = (value: string): string =>
  value.replace(/\s+/g, " ").trim()

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
      `Failed to seed live Research Workspace media: HTTP ${response.status}: ${await response.text()}`
    )
  }

  const payload = await response.json().catch(() => ({}))
  const result = Array.isArray(payload?.results)
    ? payload.results[0]
    : payload?.result || payload
  const mediaId = Number(result?.db_id ?? result?.media_id ?? result?.id)
  if (!Number.isFinite(mediaId) || mediaId <= 0) {
    throw new Error(
      `Live media seed returned no usable media id: ${JSON.stringify(payload)}`
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
        const bodyJson = await details.json().catch(() => ({}))
        return normalizeWhitespace(
          String(
            bodyJson?.content?.text ??
              bodyJson?.content?.content ??
              bodyJson?.transcript ??
              ""
          )
        )
      },
      {
        timeout: 30_000,
        message: `Media ${mediaId} never exposed usable source content`
      }
    )
    .toContain(expectedSnippet)

  return {
    mediaId,
    title,
    type: "document",
    url: `file://${fileName}`
  }
}

const setOptionalGenerationRuntime = async (page: Page): Promise<void> => {
  const requestedModel = process.env.TLDW_RESEARCH_WORKSPACE_LIVE_MODEL?.trim()
  const requestedProvider =
    process.env.TLDW_RESEARCH_WORKSPACE_LIVE_PROVIDER?.trim().toLowerCase()

  if (!requestedModel && !requestedProvider) return

  await page.evaluate(
    ({ model, provider }) => {
      const messageStore = (window as { __tldw_useStoreMessageOption?: unknown })
        .__tldw_useStoreMessageOption as
        | { setState?: (nextState: Record<string, unknown>) => void }
        | undefined
      const modelSettingsStore = (window as {
        __tldw_useStoreChatModelSettings?: unknown
      }).__tldw_useStoreChatModelSettings as
        | { setState?: (nextState: Record<string, unknown>) => void }
        | undefined

      if (model && !messageStore?.setState) {
        throw new Error("Message option store is unavailable on window")
      }

      if (model) messageStore?.setState?.({ selectedModel: model })
      if (provider) modelSettingsStore?.setState?.({ apiProvider: provider })
    },
    {
      model: requestedModel || null,
      provider: requestedProvider || null
    }
  )
}

const waitForCompletedSummaryArtifact = async (
  page: Page
): Promise<SummaryArtifactSnapshot> => {
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const store = (window as { __tldw_useWorkspaceStore?: unknown })
            .__tldw_useWorkspaceStore as
            | {
                getState?: () => {
                  generatedArtifacts?: Array<{
                    id?: string
                    status?: string
                    type?: string
                    title?: string
                    content?: string
                    errorMessage?: string
                  }>
                }
              }
            | undefined
          const artifact = store
            ?.getState?.()
            .generatedArtifacts?.find((entry) => entry.type === "summary")
          if (!artifact) return null
          return {
            id: String(artifact.id || ""),
            status: String(artifact.status || ""),
            type: String(artifact.type || ""),
            title: String(artifact.title || ""),
            contentLength: String(artifact.content || "").trim().length,
            errorMessage: String(artifact.errorMessage || "")
          }
        }),
      {
        timeout: 180_000,
        message: "Summary artifact never reached a terminal state"
      }
    )
    .toMatchObject({ status: "completed", type: "summary" })

  const artifact = await page.evaluate(() => {
    const store = (window as { __tldw_useWorkspaceStore?: unknown })
      .__tldw_useWorkspaceStore as
      | {
          getState?: () => {
            generatedArtifacts?: Array<{
              id?: string
              status?: string
              type?: string
              title?: string
              content?: string
            }>
          }
        }
      | undefined
    const entry = store
      ?.getState?.()
      .generatedArtifacts?.find((candidate) => candidate.type === "summary")
    return {
      id: String(entry?.id || ""),
      status: String(entry?.status || ""),
      type: String(entry?.type || ""),
      title: String(entry?.title || ""),
      contentLength: String(entry?.content || "").trim().length
    }
  })

  expect(artifact.contentLength).toBeGreaterThan(0)
  return artifact
}

const waitForChatCompletionCall = async (
  page: Page,
  action: () => Promise<void>
): Promise<ChatCompletionCapture> => {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method().toUpperCase() === "POST" &&
      /\/api\/v1\/chat\/completions(?:\?|$)/i.test(response.url()),
    { timeout: 120_000 }
  )
  const [response] = await Promise.all([responsePromise, action()])
  const requestBody =
    (response.request().postDataJSON() as Record<string, unknown>) || {}
  const responseBody =
    (await response.json().catch(() => null)) as Record<string, unknown> | null
  const choices = Array.isArray(responseBody?.choices)
    ? (responseBody?.choices as Array<Record<string, unknown>>)
    : []
  const firstMessage = choices[0]?.message as Record<string, unknown> | undefined
  const responseContent =
    typeof firstMessage?.content === "string" ? firstMessage.content : ""

  return {
    status: response.status(),
    model:
      typeof requestBody.model === "string" && requestBody.model.trim().length > 0
        ? requestBody.model
        : null,
    provider:
      typeof requestBody.api_provider === "string" &&
      requestBody.api_provider.trim().length > 0
        ? requestBody.api_provider
        : null,
    responseContentLength: responseContent.trim().length
  }
}

const expectCapabilityEndpointReady = async (): Promise<void> => {
  const payload = await fetchJsonWithApiKey<{
    capabilities?: Record<string, unknown>
  }>("/api/v1/research-workspace/capabilities")

  for (const capabilityId of EXPECTED_CAPABILITIES) {
    expect(payload.capabilities, `Missing capability ${capabilityId}`).toHaveProperty(
      capabilityId
    )
  }
}

const expectNoBlockingServerDialog = async (page: Page): Promise<void> => {
  await expect(
    page.getByRole("dialog").filter({ hasText: /can't reach your tldw server/i })
  ).toBeHidden({ timeout: 5_000 })
}

test.describe("Research Workspace live provider generation (manual)", () => {
  test.skip(
    process.env.TLDW_RESEARCH_WORKSPACE_LIVE_GENERATION !== "1",
    "Manual local-only real LLM verification; not a CI gate."
  )

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    await page.setViewportSize(DESKTOP_VIEWPORT)
  })

  test("verifies routes, capabilities, and a real Summary artifact", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    test.setTimeout(300_000)
    skipIfServerUnavailable(serverInfo)

    await expectCapabilityEndpointReady()

    const fixtureId = generateTestId("research-workspace-summary")
    const source = await seedLiveWorkspaceDocument(
      `RS ${fixtureId} Source`,
      [
        `Research Workspace summary source for ${fixtureId}.`,
        "The team shipped a capability-aware health contract before enabling action-specific UI gates.",
        "The Summary artifact must be generated by the configured local real LLM provider, not by a mocked response."
      ].join(" ")
    )

    const workspacePage = new ResearchWorkspacePage(authedPage)
    await workspacePage.goto()
    await workspacePage.waitForReady()
    await setOptionalGenerationRuntime(authedPage)

    await workspacePage.resetWorkspace(`Workspace ${fixtureId}`)
    await workspacePage.seedSources([source])
    await workspacePage.selectSourceByTitle(source.title)
    await workspacePage.expectSourceSelectedByTitle(source.title)
    await expect(
      workspacePage.getStudioOutputButton("Summary")
    ).toBeEnabled({ timeout: 10_000 })

    const chatCall = await waitForChatCompletionCall(authedPage, async () => {
      await workspacePage.getStudioOutputButton("Summary").click()
    })
    expect(chatCall.status).toBe(200)
    expect(chatCall.model).toBeTruthy()
    expect(chatCall.responseContentLength).toBeGreaterThan(0)

    const artifact = await waitForCompletedSummaryArtifact(authedPage)
    const screenshotPath = path.join(
      os.tmpdir(),
      `research-workspace-live-summary-${fixtureId}.png`
    )
    await authedPage.screenshot({ path: screenshotPath, fullPage: true })

    console.log(
      "RESEARCH_WORKSPACE_LIVE_SUMMARY_EVIDENCE",
      JSON.stringify({
        provider: chatCall.provider || "auto",
        model: chatCall.model,
        sourceType: source.type,
        sourceTitle: source.title,
        artifactType: artifact.type,
        outputCharacterCount: artifact.contentLength,
        responseCharacterCount: chatCall.responseContentLength,
        screenshotPath
      })
    )

    await expectNoBlockingServerDialog(authedPage)
    await assertNoCriticalErrors(diagnostics)
  })

  test("verifies canonical Research Workspace route entry without legacy aliases", async ({
    page,
    serverInfo
  }) => {
    skipIfServerUnavailable(serverInfo)

    await page.goto("/research-workspace", { waitUntil: "domcontentloaded" })
    await expect(page.locator("#workspace-sources-panel")).toBeVisible({
      timeout: 30_000
    })
    await expect(page.locator("#workspace-studio-panel")).toBeVisible({
      timeout: 30_000
    })

    await page.setViewportSize(MOBILE_VIEWPORT)
    await page.goto("/research-workspace?tab=studio", {
      waitUntil: "domcontentloaded"
    })
    await expect(
      page.getByText("Generate outputs from your sources")
    ).toBeVisible({ timeout: 30_000 })
  })
})
