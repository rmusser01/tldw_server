import { type Page, type Route } from "@playwright/test"

import { expect, test } from "./smoke.setup"
import { seedAuth, waitForAppShell } from "../utils/helpers"

const API_SERVER_URL = "http://127.0.0.1:8000"
const API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const WORKSPACE_ID = "workspace-chat-live-smoke"
const WORKSPACE_NAME = "Chat Live Smoke Workspace"
const MODEL_ID = "e2e-chat-model"
const SELECTED_MODEL = `tldw:${MODEL_ID}`
const PERSONA_ID = "workspace_smoke_persona"
const PERSONA_NAME = "Workspace Smoke Persona"
const READY_SOURCE_ID = "workspace-smoke-ready-source"
const READY_SOURCE_TITLE = "Workspace Smoke Source"
const READY_SOURCE_MEDIA_ID = 24601
const FALLBACK_SOURCE_ID = "workspace-smoke-fallback-source"
const FALLBACK_SOURCE_TITLE = "Workspace Smoke Fallback Source"

type CapturedRequest = {
  method: string
  path: string
  query: Record<string, string>
  body: unknown
}

type BackendFixtureMode = "success" | "streaming" | "failure"

type BackendFixture = {
  chatCreates: CapturedRequest[]
  chatMessages: CapturedRequest[]
  chatCompletions: CapturedRequest[]
  ragSearches: CapturedRequest[]
}

type ConnectionStoreWindow = Window & {
  __tldw_useConnectionStore?: {
    getState: () => { state: Record<string, unknown> }
    setState: (value: { state: Record<string, unknown> }) => void
  }
}

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,PUT,DELETE,OPTIONS",
  "access-control-expose-headers": "*"
}

const nowIso = () => "2026-06-30T12:00:00.000Z"

const parseRequestJson = async (route: Route): Promise<unknown> => {
  const raw = route.request().postData()
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

const captureRequest = async (route: Route): Promise<CapturedRequest> => {
  const request = route.request()
  const url = new URL(request.url())
  return {
    method: request.method().toUpperCase(),
    path: url.pathname,
    query: Object.fromEntries(url.searchParams.entries()),
    body: await parseRequestJson(route)
  }
}

const fulfillJson = async (
  route: Route,
  status: number,
  body: unknown
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: CORS_HEADERS,
    body: JSON.stringify(body)
  })
}

const fulfillOptions = async (route: Route): Promise<void> => {
  await route.fulfill({
    status: 204,
    headers: CORS_HEADERS
  })
}

const makeChatSummary = (id: string, title = "Workspace smoke chat") => ({
  id,
  chat_id: id,
  title,
  state: "in-progress",
  version: 1,
  created_at: nowIso(),
  updated_at: nowIso(),
  assistant_kind: null,
  assistant_id: null,
  character_id: null,
  persona_memory_mode: null,
  scope_type: "workspace",
  workspace_id: WORKSPACE_ID
})

const installBackendFixture = async (
  page: Page,
  options: { mode?: BackendFixtureMode; streamDelayMs?: number } = {}
): Promise<BackendFixture> => {
  const mode = options.mode ?? "success"
  const streamDelayMs = options.streamDelayMs ?? 4_000
  const fixture: BackendFixture = {
    chatCreates: [],
    chatMessages: [],
    chatCompletions: [],
    ragSearches: []
  }

  await page.route("**/openapi.json", async (route) => {
    await fulfillJson(route, 200, {
      openapi: "3.1.0",
      info: { title: "tldw e2e fixture", version: "0.0.0" },
      paths: {
        "/api/v1/chats/": {},
        "/api/v1/chat/completions": {},
        "/api/v1/rag/search": {}
      }
    })
  })

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request()
    const method = request.method().toUpperCase()
    const url = new URL(request.url())
    const path = url.pathname

    if (method === "OPTIONS") {
      await fulfillOptions(route)
      return
    }

    if (path === "/api/v1/health" || path === "/api/v1/health/live") {
      await fulfillJson(route, 200, {
        status: "healthy",
        ok: true,
        version: "e2e-chat-workspace"
      })
      return
    }

    if (
      path === "/api/v1/setup/first-run/state" ||
      path === "/api/v1/setup/first-run/metadata"
    ) {
      await fulfillJson(route, 200, {
        status: "completed",
        setup_required: false,
        setup_completed: true,
        current_step: null,
        completed_steps: ["first_chat"],
        acknowledged_steps: ["first_chat"],
        connection: {
          frontend_origin: "http://localhost:8080",
          api_origin: API_SERVER_URL,
          browser_access: "local"
        }
      })
      return
    }

    if (path === "/api/v1/config/docs-info") {
      await fulfillJson(route, 200, {
        server_version: "e2e-chat-workspace",
        openapi_url: "/openapi.json",
        features: {
          rag: true,
          persona_live_control: true,
          chat_workspace: true
        }
      })
      return
    }

    if (path === "/api/v1/config/providers") {
      await fulfillJson(route, 200, {
        any_configured: true,
        providers: [
          {
            name: "openai",
            configured: true,
            requires_api_key: true,
            key_source: "e2e"
          }
        ]
      })
      return
    }

    if (path === "/api/v1/llm/providers") {
      await fulfillJson(route, 200, {
        providers: [
          {
            name: "openai",
            is_configured: true,
            configured: true,
            default_model: MODEL_ID,
            models: [MODEL_ID]
          }
        ],
        any_configured: true
      })
      return
    }

    if (path === "/api/v1/llm/models/metadata") {
      await fulfillJson(route, 200, {
        models: [
          {
            id: MODEL_ID,
            name: MODEL_ID,
            model: MODEL_ID,
            provider: "openai",
            type: "chat",
            capabilities: ["chat"],
            modalities: ["text"],
            is_configured: true,
            provider_is_configured: true,
            catalog_only: false
          }
        ],
        total: 1
      })
      return
    }

    if (path === "/api/v1/rag/health") {
      await fulfillJson(route, 200, { status: "healthy", ok: true })
      return
    }

    if (path === "/api/v1/rag/search" && method === "POST") {
      const captured = await captureRequest(route)
      fixture.ragSearches.push(captured)
      await fulfillJson(route, 200, {
        generated_answer: `Grounded answer for ${READY_SOURCE_TITLE}`,
        results: [
          {
            content: "The workspace smoke source contains a deterministic proof point.",
            metadata: {
              title: READY_SOURCE_TITLE,
              source: READY_SOURCE_TITLE,
              type: "pdf",
              url: "workspace://smoke-source",
              media_id: READY_SOURCE_MEDIA_ID
            }
          }
        ],
        citations: [],
        metadata: { fixture: "chat-workspace-live-backend" }
      })
      return
    }

    if ((path === "/api/v1/chats" || path === "/api/v1/chats/") && method === "GET") {
      await fulfillJson(route, 200, { items: [], chats: [], total: 0 })
      return
    }

    if ((path === "/api/v1/chats" || path === "/api/v1/chats/") && method === "POST") {
      const captured = await captureRequest(route)
      fixture.chatCreates.push(captured)
      const body = captured.body as Record<string, unknown> | null
      const id = `workspace-chat-smoke-server-${fixture.chatCreates.length}`
      await fulfillJson(
        route,
        200,
        makeChatSummary(
          id,
          typeof body?.title === "string" ? body.title : "Workspace smoke chat"
        )
      )
      return
    }

    const chatSettingsMatch = path.match(/^\/api\/v1\/chats\/([^/]+)\/settings\/?$/)
    if (chatSettingsMatch) {
      await fulfillJson(route, 200, {
        chat_id: chatSettingsMatch[1],
        settings: {},
        version: 1
      })
      return
    }

    const chatMessagesMatch = path.match(/^\/api\/v1\/chats\/([^/]+)\/messages\/?$/)
    if (chatMessagesMatch && method === "GET") {
      await fulfillJson(route, 200, { items: [], messages: [], total: 0 })
      return
    }
    if (chatMessagesMatch && method === "POST") {
      const captured = await captureRequest(route)
      fixture.chatMessages.push(captured)
      await fulfillJson(route, 200, {
        id: `workspace-message-${fixture.chatMessages.length}`,
        chat_id: chatMessagesMatch[1],
        role: (captured.body as Record<string, unknown> | null)?.role ?? "assistant",
        content: (captured.body as Record<string, unknown> | null)?.content ?? "",
        version: fixture.chatMessages.length,
        created_at: nowIso()
      })
      return
    }

    const chatMatch = path.match(/^\/api\/v1\/chats\/([^/]+)\/?$/)
    if (chatMatch && method === "GET") {
      await fulfillJson(route, 200, makeChatSummary(chatMatch[1]))
      return
    }

    if (path === "/api/v1/chat/completions" && method === "POST") {
      const captured = await captureRequest(route)
      fixture.chatCompletions.push(captured)

      if (mode === "failure") {
        await fulfillJson(route, 503, {
          detail: "Deterministic workspace failure"
        })
        return
      }

      if (mode === "streaming") {
        await new Promise((resolve) => setTimeout(resolve, streamDelayMs))
        try {
          await route.fulfill({
            status: 200,
            contentType: "text/event-stream",
            headers: {
              ...CORS_HEADERS,
              "cache-control": "no-cache",
              connection: "keep-alive"
            },
            body: [
              `data: ${JSON.stringify({
                id: "chatcmpl-workspace-smoke",
                object: "chat.completion.chunk",
                choices: [{ delta: { content: "streamed workspace reply" } }]
              })}`,
              "",
              "data: [DONE]",
              ""
            ].join("\n")
          })
        } catch {
          // The user-facing stop flow aborts the fetch before this delayed
          // fixture responds. The aborted request has already exercised the UI.
        }
        return
      }

      await fulfillJson(route, 200, {
        id: "chatcmpl-workspace-smoke",
        choices: [
          {
            message: {
              role: "assistant",
              content: "deterministic workspace response"
            }
          }
        ]
      })
      return
    }

    await fulfillJson(route, 200, { status: "ok", fixture: "chat-workspace-live-backend" })
  })

  return fixture
}

const seedChatWorkspaceState = async (
  page: Page,
  options: {
    sourceId?: string
    sourceTitle?: string
    mediaId?: number
  } = {}
): Promise<void> => {
  const sourceId = options.sourceId ?? READY_SOURCE_ID
  const sourceTitle = options.sourceTitle ?? READY_SOURCE_TITLE
  const mediaId = options.mediaId ?? READY_SOURCE_MEDIA_ID

  await seedAuth(page, {
    serverUrl: API_SERVER_URL,
    apiKey: API_KEY,
    allowOffline: false
  })

  await page.addInitScript(
    ({ mediaId, sourceId, sourceTitle }) => {
      const seedValue = (key: string, value: unknown) => {
        localStorage.setItem(key, JSON.stringify(value))
        localStorage.setItem(`plasmo-storage-${key}`, JSON.stringify(value))
      }
      const now = "2026-06-30T12:00:00.000Z"
      const source = {
        id: sourceId,
        mediaId,
        title: sourceTitle,
        type: "pdf",
        status: "ready",
        statusMessage: "Indexed for workspace smoke coverage.",
        addedAt: now
      }
      const snapshot = {
        workspaceId: "workspace-chat-live-smoke",
        workspaceName: "Chat Live Smoke Workspace",
        workspaceTag: "workspace:chat-live-smoke",
        workspaceCreatedAt: now,
        workspaceChatReferenceId: "workspace-chat-live-smoke",
        sources: [source],
        selectedSourceIds: [],
        sourceFolders: [],
        sourceFolderMemberships: [],
        selectedSourceFolderIds: [],
        activeFolderId: null,
        generatedArtifacts: [],
        notes: "",
        currentNote: {
          id: 1,
          title: "Chat smoke note",
          content: "",
          keywords: [],
          version: 1,
          isDirty: false
        },
        workspaceBanner: {
          title: "",
          subtitle: "",
          image: null
        },
        leftPaneCollapsed: false,
        rightPaneCollapsed: false,
        audioSettings: {
          provider: "openai",
          model: "gpt-4o-mini-tts",
          voice: "alloy",
          speed: 1,
          format: "mp3"
        }
      }

      seedValue("selectedModel", "tldw:e2e-chat-model")
      seedValue("defaultApiProvider", "openai")
      seedValue("selectedAssistant", {
        kind: "persona",
        id: "workspace_smoke_persona",
        name: "Workspace Smoke Persona",
        system_prompt: "Answer as the workspace smoke persona.",
        metadata: { selectionMode: "overlay" }
      })
      seedValue("tldwModelsCache", {
        version: 3,
        timestamp: Date.parse(now),
        scope: "http://127.0.0.1:8000|single-user|key|none",
        models: [
          {
            id: "e2e-chat-model",
            name: "e2e-chat-model",
            provider: "openai",
            type: "chat",
            capabilities: ["chat"],
            modalities: ["text"],
            isConfigured: true,
            providerIsConfigured: true,
            catalogOnly: false
          }
        ]
      })
      localStorage.setItem(
        "tldw-workspace",
        JSON.stringify({
          state: {
            workspaceId: "workspace-chat-live-smoke",
            workspaceName: "Chat Live Smoke Workspace",
            workspaceTag: "workspace:chat-live-smoke",
            workspaceCreatedAt: now,
            workspaceChatReferenceId: "workspace-chat-live-smoke",
            sources: [source],
            selectedSourceIds: [],
            sourceFolders: [],
            sourceFolderMemberships: [],
            selectedSourceFolderIds: [],
            activeFolderId: null,
            generatedArtifacts: [],
            notes: "",
            currentNote: snapshot.currentNote,
            workspaceBanner: snapshot.workspaceBanner,
            leftPaneCollapsed: false,
            rightPaneCollapsed: false,
            audioSettings: snapshot.audioSettings,
            savedWorkspaces: [
              {
                id: "workspace-chat-live-smoke",
                name: "Chat Live Smoke Workspace",
                tag: "workspace:chat-live-smoke",
                createdAt: now,
                lastAccessedAt: now,
                collectionId: null
              }
            ],
            archivedWorkspaces: [],
            workspaceCollections: [],
            workspaceSnapshots: {
              "workspace-chat-live-smoke": snapshot
            },
            workspaceChatSessions: {}
          },
          version: 1
        })
      )
      localStorage.setItem(
        "tldw-workspace:workspace:workspace-chat-live-smoke:snapshot",
        JSON.stringify(snapshot)
      )
      localStorage.setItem(
        "tldw-workspace",
        JSON.stringify({
          schema: "workspace_split_v1",
          splitVersion: 1,
          state: {
            workspaceId: "workspace-chat-live-smoke",
            savedWorkspaces: [
              {
                id: "workspace-chat-live-smoke",
                name: "Chat Live Smoke Workspace",
                tag: "workspace:chat-live-smoke",
                createdAt: now,
                lastAccessedAt: now,
                collectionId: null
              }
            ],
            archivedWorkspaces: [],
            workspaceCollections: [],
            workspaceIds: ["workspace-chat-live-smoke"],
            workspaceSnapshots: {
              "workspace-chat-live-smoke": snapshot
            },
            workspaceChatSessions: {}
          },
          version: 1
        })
      )
    },
    { mediaId, sourceId, sourceTitle }
  )
}

const openSeededChatWorkspace = async (page: Page): Promise<void> => {
  await page.goto("/chat-workspace", { waitUntil: "domcontentloaded" })
  await waitForAppShell(page)
  await expect(page.getByTestId("chat-workspace-page")).toBeVisible()
  await expect(
    page.getByRole("heading", { name: WORKSPACE_NAME, exact: true })
  ).toBeVisible()
  await expect(page.getByText(SELECTED_MODEL)).toBeVisible()
  await expect(page.getByText(PERSONA_NAME)).toBeVisible()
}

test.describe("Chat Workspace live-backend smoke coverage", () => {
  test("sends staged workspace media through scoped backend requests", async ({ page }) => {
    const fixture = await installBackendFixture(page)
    await seedChatWorkspaceState(page)
    await openSeededChatWorkspace(page)

    await page
      .getByRole("button", { name: `Stage ${READY_SOURCE_TITLE} for chat` })
      .click()
    await expect(
      page.getByRole("region", { name: "Staged context" })
    ).toContainText(READY_SOURCE_TITLE)

    const message = "Summarize the staged source for the release gate"
    await page.getByLabel("Chat workspace message").fill(message)
    await expect(page.getByRole("button", { name: "Send message" })).toBeEnabled()
    await page.getByRole("button", { name: "Send message" }).click()

    await expect(page.getByText("deterministic workspace response")).toBeVisible()
    await expect(page.getByRole("region", { name: "Staged context" })).toHaveCount(0)
    await expect(page.getByLabel("Chat workspace message")).toHaveValue("")

    expect(fixture.chatCreates).toHaveLength(1)
    expect(fixture.chatCreates[0]?.body).toMatchObject({
      scope_type: "workspace",
      workspace_id: WORKSPACE_ID
    })
    expect(fixture.ragSearches).toHaveLength(1)
    expect(fixture.ragSearches[0]?.body).toMatchObject({
      include_media_ids: [READY_SOURCE_MEDIA_ID],
      sources: ["media_db"]
    })
    const completionBody = fixture.chatCompletions[0]?.body as {
      messages?: Array<{ content?: unknown }>
    }
    expect(JSON.stringify(completionBody.messages)).toContain(
      "The workspace smoke source contains a deterministic proof point."
    )
    expect(fixture.chatMessages.map((entry) => entry.path)).toContain(
      "/api/v1/chats/workspace-chat-smoke-server-1/messages"
    )
  })

  test("browses, unstages, re-stages, and sends the selected workspace source", async ({ page }) => {
    const fixture = await installBackendFixture(page)
    await seedChatWorkspaceState(page)
    await openSeededChatWorkspace(page)

    await expect(page.getByRole("link", { name: "Add source" })).toHaveAttribute(
      "href",
      "/research-workspace?tab=sources"
    )
    await expect(page.getByRole("link", { name: "Open library" })).toHaveAttribute(
      "href",
      "/media"
    )

    await page.getByRole("button", { name: `Browse ${READY_SOURCE_TITLE}` }).click()
    await expect(page.getByText("Browsing")).toBeVisible()

    await page
      .getByRole("button", { name: `Stage ${READY_SOURCE_TITLE} for chat` })
      .click()
    await expect(
      page.getByRole("region", { name: "Staged context" })
    ).toContainText(READY_SOURCE_TITLE)

    await page
      .getByRole("button", { name: `Unstage ${READY_SOURCE_TITLE} from chat` })
      .click()
    await expect(page.getByRole("region", { name: "Staged context" })).toHaveCount(0)
    await expect(page.getByRole("button", { name: "Send message" })).toBeDisabled()

    await page
      .getByRole("button", { name: `Stage ${READY_SOURCE_TITLE} for chat` })
      .click()
    await page
      .getByLabel("Chat workspace message")
      .fill("Use the re-staged source after unstage")
    await page.getByRole("button", { name: "Send with staged context" }).click()

    await expect(page.getByText("deterministic workspace response")).toBeVisible()
    expect(fixture.ragSearches).toHaveLength(1)
    expect(fixture.ragSearches[0]?.body).toMatchObject({
      include_media_ids: [READY_SOURCE_MEDIA_ID],
      sources: ["media_db"]
    })
    const completionBody = fixture.chatCompletions[0]?.body as {
      messages?: Array<{ content?: unknown }>
    }
    expect(JSON.stringify(completionBody.messages)).toContain(
      "The workspace smoke source contains a deterministic proof point."
    )
  })

  test("shows stop generation while the workspace stream is active", async ({ page }) => {
    const streamDelayMs = 1_000
    const fixture = await installBackendFixture(page, {
      mode: "streaming",
      streamDelayMs
    })
    await seedChatWorkspaceState(page)
    await openSeededChatWorkspace(page)

    await page.getByLabel("Chat workspace message").fill("Stream long enough to stop")
    await expect(page.getByRole("button", { name: "Send message" })).toBeEnabled()
    await page.getByRole("button", { name: "Send message" }).click()

    const stopButton = page.getByRole("button", { name: "Stop generating" })
    await expect(stopButton).toBeVisible()
    await expect(
      page.getByLabel("Chat workspace status").getByText("Streaming")
    ).toBeVisible()
    await expect(
      page
        .getByRole("complementary", { name: "Chat workspace inspector" })
        .getByText("Streaming")
    ).toBeVisible()
    await expect.poll(() => fixture.chatCreates.length).toBeGreaterThan(0)
    expect(fixture.chatCreates[0]?.body).toMatchObject({
      scope_type: "workspace",
      workspace_id: WORKSPACE_ID
    })
    await expect.poll(() => fixture.chatCompletions.length).toBeGreaterThan(0)
    expect(fixture.chatCompletions[0]?.body).toMatchObject({
      stream: true,
      conversation_id: "workspace-chat-smoke-server-1"
    })

    await stopButton.click()
    await expect(stopButton).toBeHidden()
    await expect(page.getByText("streamed workspace reply")).toHaveCount(0)
    await page.waitForTimeout(streamDelayMs + 250)
    await expect(page.getByText("streamed workspace reply")).toHaveCount(0)
    await expect(
      page.getByLabel("Chat workspace status").getByText("Streaming")
    ).toHaveCount(0)
    await expect(
      page
        .getByRole("complementary", { name: "Chat workspace inspector" })
        .getByText("Streaming")
    ).toHaveCount(0)
  })

  test("switches active streaming rails to offline recovery state", async ({ page }) => {
    await installBackendFixture(page, {
      mode: "streaming",
      streamDelayMs: 4_000
    })
    await seedChatWorkspaceState(page)
    await openSeededChatWorkspace(page)

    await page
      .getByLabel("Chat workspace message")
      .fill("Show offline recovery while streaming")
    await page.getByRole("button", { name: "Send message" }).click()

    const statusStrip = page.getByLabel("Chat workspace status")
    const inspector = page.getByRole("complementary", {
      name: "Chat workspace inspector"
    })
    const stopButton = page.getByRole("button", { name: "Stop generating" })

    await expect(stopButton).toBeVisible()
    await expect(statusStrip.getByText("Streaming")).toBeVisible()
    await expect(inspector.getByText("Streaming")).toBeVisible()
    await page.waitForFunction(
      () =>
        typeof (window as ConnectionStoreWindow).__tldw_useConnectionStore
          ?.getState === "function"
    )
    await page.evaluate(() => {
      const store = (window as ConnectionStoreWindow).__tldw_useConnectionStore
      if (!store) throw new Error("Connection store did not initialize")
      const current = store.getState().state
      store.setState({
        state: {
          ...current,
          phase: "error",
          isConnected: false,
          isChecking: false,
          errorKind: "unreachable",
          consecutiveFailures: 3,
          lastError: "offline",
          lastStatusCode: 0,
          lastCheckedAt: Date.now()
        }
      })
    })

    await expect(statusStrip.getByText("Server unavailable")).toBeVisible()
    await expect(statusStrip.getByText("Reconnect server")).toBeVisible()
    await expect(inspector.getByText("Server unavailable")).toBeVisible()
    await expect(statusStrip.getByText("Streaming")).toHaveCount(0)
    await expect(inspector.getByText("Streaming")).toHaveCount(0)

    await stopButton.click()
    await expect(stopButton).toBeHidden()
  })

  test("preserves draft and staged fallback context after send failure", async ({ page }) => {
    const fixture = await installBackendFixture(page, { mode: "failure" })
    await seedChatWorkspaceState(page, {
      sourceId: FALLBACK_SOURCE_ID,
      sourceTitle: FALLBACK_SOURCE_TITLE,
      mediaId: 0
    })
    await openSeededChatWorkspace(page)

    await page
      .getByRole("button", { name: `Stage ${FALLBACK_SOURCE_TITLE} for chat` })
      .click()
    await expect(
      page.getByRole("region", { name: "Staged context" })
    ).toContainText(FALLBACK_SOURCE_TITLE)

    const draft = "Keep this draft when the backend fails"
    await page.getByLabel("Chat workspace message").fill(draft)
    await expect(page.getByRole("button", { name: "Send message" })).toBeEnabled()
    await page.getByRole("button", { name: "Send message" }).click()

    await expect(
      page.getByRole("alert").filter({ hasText: "Stream completion failed" })
    ).toBeVisible()
    await expect(
      page.getByLabel("Chat workspace status").getByText("Send failed")
    ).toBeVisible()
    await expect(
      page
        .getByRole("complementary", { name: "Chat workspace inspector" })
        .getByText("Draft and staged context were preserved for retry.")
    ).toBeVisible()
    await expect(page.getByLabel("Chat workspace message")).toHaveValue(draft)
    await expect(
      page.getByRole("region", { name: "Staged context" })
    ).toContainText(FALLBACK_SOURCE_TITLE)

    expect(fixture.ragSearches).toHaveLength(0)
    expect(fixture.chatCompletions).toHaveLength(1)
    const completionBody = fixture.chatCompletions[0]?.body as {
      messages?: Array<{ content?: unknown }>
    }
    expect(JSON.stringify(completionBody.messages)).toContain("Context sources:")
    expect(JSON.stringify(completionBody.messages)).toContain(FALLBACK_SOURCE_TITLE)
  })

})
