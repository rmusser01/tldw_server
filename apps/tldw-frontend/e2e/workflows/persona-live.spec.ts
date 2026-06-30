/**
 * Persona Live Workflow E2E Tests
 *
 * Includes one live-backend websocket proof plus mocked workflow coverage for
 * persona visual pack runtime state.
 */
import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable
} from "../utils/fixtures"
import { TEST_CONFIG, fetchWithApiKey, seedAuth } from "../utils/helpers"
import {
  buildPersonaVisualPackFixture,
  buildPortablePersonaVisualPackUpload,
  PERSONA_VISUAL_E2E_PERSONA_ID,
  PERSONA_VISUAL_E2E_SESSION_ID,
  PERSONA_VISUAL_E2E_STARTER_PACK
} from "../fixtures/persona-visual-packs"

type DocsInfoPayload = {
  capabilities?: Record<string, unknown> | null
  supported_features?: Record<string, unknown> | null
}

type PersonaCatalogEntry = {
  id?: string | null
}

type PersonaProfilePayload = {
  version?: number | null
}

const DEFAULT_PERSONA_ID = "research_assistant"
const VISUAL_PERSONA_ID = PERSONA_VISUAL_E2E_PERSONA_ID
const VISUAL_SESSION_ID = PERSONA_VISUAL_E2E_SESSION_ID
const COMPLETED_SETUP_STEPS = [
  "persona",
  "voice",
  "commands",
  "safety",
  "test",
] as const
const COMPLETED_SETUP = {
  status: "completed",
  version: 1,
  run_id: "e2e-visual-setup",
  current_step: "test",
  completed_steps: [...COMPLETED_SETUP_STEPS],
  completed_at: "2026-05-08T00:00:00.000Z",
  last_test_type: "live_session",
}
const VISUAL_BUDDY_SUMMARY = {
  has_buddy: true,
  persona_name: "Visual Persona",
  role_summary: "Visual runtime E2E assistant",
  visual: {
    species_id: "sprite",
    silhouette_id: "companion",
    palette_id: "studio"
  }
}

type PersonaVisualMockMode = "active-pack" | "broken-pack" | "empty-pack"

type PersonaMockWindow = Window & {
  __personaWsMock?: {
    getSentPayloads: () => unknown[]
    emitJson: (payload: unknown) => void
  }
}

const fulfillJson = async (
  route: Route,
  status: number,
  payload: unknown
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload)
  })
}

const extractProfilePersonaId = (route: Route): string => {
  const url = new URL(route.request().url())
  const parts = url.pathname.split("/").filter(Boolean)
  const profileIndex = parts.findIndex((part) => part === "profiles")
  return decodeURIComponent(parts[profileIndex + 1] || VISUAL_PERSONA_ID)
}

const installPersonaVisualApiMocks = async (
  page: Page,
  options: { visualPackMode?: PersonaVisualMockMode } = {}
): Promise<void> => {
  const visualPackMode = options.visualPackMode ?? "active-pack"
  let packs =
    visualPackMode === "empty-pack" || visualPackMode === "broken-pack"
      ? []
      : [buildPersonaVisualPackFixture(VISUAL_PERSONA_ID)]
  let previewCompleted = false
  let importCommitted = false

  await page.route(/\/api\/v1\/health(?:\/live)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      status: "ok"
    })
  })

  await page.route(/\/api\/v1\/rag\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      status: "healthy",
      components: {
        search_index: {
          status: "healthy",
          fts_table_count: 1
        }
      }
    })
  })

  await page.route("**/openapi.json", async (route) => {
    await fulfillJson(route, 200, {
      info: { version: "e2e-persona-visuals" },
      paths: {
        "/api/v1/persona/catalog": {},
        "/api/v1/persona/session": {},
        "/api/v1/persona/stream": {},
        "/api/v1/audio/speech": {},
        "/api/v1/audio/transcriptions": {},
        "/api/v1/personalization/profile": {}
      }
    })
  })

  await page.route(/\/api\/v1\/config\/docs-info(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      capabilities: {
        persona: true,
        personalization: true,
        hasAudio: true,
        hasStt: true,
        hasTts: true,
        hasVoiceChat: true
      },
      supported_features: {
        persona: true,
        personalization: true
      },
      ffmpeg_available: true
    })
  })

  await page.route(/\/api\/v1\/persona\/catalog(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, [
      {
        id: VISUAL_PERSONA_ID,
        name: "Visual Persona",
        description: "Persona visual workflow E2E fixture",
        buddy_summary: VISUAL_BUDDY_SUMMARY
      }
    ])
  })

  await page.route(/\/api\/v1\/persona\/profiles(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, [
      {
        id: VISUAL_PERSONA_ID,
        name: "Visual Persona",
        version: 1,
        setup: COMPLETED_SETUP,
        buddy_summary: VISUAL_BUDDY_SUMMARY
      }
    ])
  })

  await page.route(/\/api\/v1\/persona\/visual-starter-packs(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      starter_packs: [PERSONA_VISUAL_E2E_STARTER_PACK]
    })
  })

  await page.route(/\/api\/v1\/persona\/visual-starter-packs\/[^/]+\/copy(?:\?.*)?$/, async (route) => {
    const personaId = VISUAL_PERSONA_ID
    const pack = buildPersonaVisualPackFixture(personaId, {
      packId: "starter-copy-e2e",
      title: "Research Buddy Starter",
      status: "draft",
      provenance: "imported"
    })
    packs = [pack]
    await fulfillJson(route, 201, pack)
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/[^/]+\/activate(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url())
    const parts = url.pathname.split("/").filter(Boolean)
    const packId = decodeURIComponent(parts[parts.findIndex((part) => part === "visual-packs") + 1] || "")
    packs = packs.map((pack) => ({
      ...pack,
      status: pack.id === packId ? "active" : "draft",
      active_at:
        pack.id === packId ? "2026-05-08T00:00:00.000Z" : null
    }))
    await fulfillJson(
      route,
      200,
      packs.find((pack) => pack.id === packId) ?? packs[0]
    )
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/import-previews(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      preview_id: "preview-upload-e2e",
      job_id: "preview-upload-job-e2e",
      portability_job_id: "portability-preview-upload-e2e",
      operation: "import_preview",
      target_persona_id: VISUAL_PERSONA_ID,
      status: "queued",
      stage: "queued"
    })
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/import-previews\/preview-upload-e2e(?:\?.*)?$/, async (route) => {
    previewCompleted = true
    await fulfillJson(route, 200, {
      preview_id: "preview-upload-e2e",
      job_id: "preview-upload-job-e2e",
      portability_job_id: "portability-preview-upload-e2e",
      operation: "import_preview",
      target_persona_id: VISUAL_PERSONA_ID,
      status: "completed",
      visual_status: "completed",
      stage: "completed",
      archive_sha256: "sha-uploaded-e2e",
      canonical_payload_fingerprint: "fingerprint-uploaded-e2e",
      schema_version: "tldw.persona_visual_pack.v1",
      bundle_summary: {
        pack_title: "Uploaded Visual Pack",
        asset_count: 1,
        assets_with_bytes: 1
      },
      validation_warnings: [],
      conflicts: [],
      proposed_plan: {
        target_mode: "create_new",
        commit_eligible: true,
        activation_eligible: true,
        renderer_import_preview: {
          status: "supported",
          renderer_type: "sprite_frames",
          manifest_version: 1,
          renderer_contract_version: 1,
          can_commit: true,
          activation_eligible: true,
          blockers: [],
          warnings: [],
          normalized_role_categories: { frame: ["portable-upload-pack-frame-idle"] },
          setup_status: "supported",
          setup_blockers: []
        }
      },
      quota_estimate: { asset_bytes: 96 },
      required_choices: [],
      target_warnings: []
    })
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/import-previews\/preview-upload-e2e\/commit(?:\?.*)?$/, async (route) => {
    const pack = buildPersonaVisualPackFixture(VISUAL_PERSONA_ID, {
      packId: "uploaded-pack-e2e",
      title: "Uploaded Visual Pack",
      status: "draft",
      provenance: "imported"
    })
    importCommitted = true
    packs = [pack]
    await fulfillJson(route, 200, {
      job_id: "import-upload-job-e2e",
      portability_job_id: "portability-import-upload-e2e",
      operation: "import_commit",
      preview_id: "preview-upload-e2e",
      target_persona_id: VISUAL_PERSONA_ID,
      status: "queued",
      stage: "queued"
    })
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/imports\/import-upload-job-e2e(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      job_id: "import-upload-job-e2e",
      portability_job_id: "portability-import-upload-e2e",
      operation: "import_commit",
      persona_id: VISUAL_PERSONA_ID,
      pack_id: "uploaded-pack-e2e",
      status: importCommitted && previewCompleted ? "completed" : "processing",
      visual_status: importCommitted && previewCompleted ? "completed" : "processing",
      stage: importCommitted && previewCompleted ? "completed" : "commit",
      progress: { asset_count: 1 },
      warnings: []
    })
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs(?:\?.*)?$/, async (route) => {
    if (visualPackMode === "broken-pack") {
      await fulfillJson(route, 500, {
        detail: "visual pack fixture unavailable"
      })
      return
    }

    const activePack = packs.find((pack) => pack.status === "active") ?? null
    await fulfillJson(route, 200, {
      packs,
      active_pack: activePack
    })
  })

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+(?:\?.*)?$/, async (route) => {
    const personaId = extractProfilePersonaId(route)
    await fulfillJson(route, 200, {
      id: personaId,
      version: 1,
      voice_defaults: null,
      setup: COMPLETED_SETUP,
      buddy_summary: {
        ...VISUAL_BUDDY_SUMMARY,
        persona_name:
          personaId === VISUAL_PERSONA_ID ? "Visual Persona" : "Research Assistant"
      }
    })
  })

  await page.route(/\/api\/v1\/persona\/sessions(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, [])
  })

  await page.route(/\/api\/v1\/persona\/sessions\/[^/?]+(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      session_id: VISUAL_SESSION_ID,
      persona: { id: VISUAL_PERSONA_ID },
      preferences: null,
      turns: []
    })
  })

  await page.route(/\/api\/v1\/persona\/session(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      session_id: VISUAL_SESSION_ID,
      persona: { id: VISUAL_PERSONA_ID }
    })
  })

  await page.route(/\/api\/v1\/chats(?:\/.*)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      items: [],
      total: 0
    })
  })
}

const installMockPersonaWebSocket = async (page: Page): Promise<void> => {
  await page.addInitScript(() => {
    type MockSocket = {
      url: string
      readyState: number
      binaryType: string
      sent: string[]
      onopen: ((event: Event) => void) | null
      onmessage: ((event: MessageEvent) => void) | null
      onerror: ((event: Event) => void) | null
      onclose: ((event: CloseEvent) => void) | null
      dispatchEvent?: (event: Event) => boolean
    }

    const state: { sockets: MockSocket[] } = { sockets: [] }
    const personaWindow = window as PersonaMockWindow

    personaWindow.__personaWsMock = {
      getSentPayloads: () => {
        const socket = state.sockets[state.sockets.length - 1]
        if (!socket) return []
        return socket.sent.map((raw) => {
          try {
            return JSON.parse(raw)
          } catch {
            return raw
          }
        })
      },
      emitJson: (payload: unknown) => {
        const socket = state.sockets[state.sockets.length - 1]
        if (!socket) return
        const messageEvent = new MessageEvent("message", {
          data: JSON.stringify(payload)
        })
        socket.onmessage?.(messageEvent)
        socket.dispatchEvent?.(messageEvent)
      }
    }

    class MockWebSocket {
      static CONNECTING = 0
      static OPEN = 1
      static CLOSING = 2
      static CLOSED = 3

      url: string
      readyState = MockWebSocket.CONNECTING
      binaryType = "blob"
      onopen: ((event: Event) => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      onerror: ((event: Event) => void) | null = null
      onclose: ((event: CloseEvent) => void) | null = null
      sent: string[] = []
      listeners: Record<string, EventListenerOrEventListenerObject[]> = {}

      constructor(url: string | URL) {
        this.url = String(url)
        state.sockets.push(this as unknown as MockSocket)
        window.setTimeout(() => {
          this.readyState = MockWebSocket.OPEN
          this.emitEvent(new Event("open"))
        }, 0)
      }

      addEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject | null
      ) {
        if (!listener) return
        this.listeners[type] = [...(this.listeners[type] || []), listener]
      }

      removeEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject | null
      ) {
        if (!listener || !this.listeners[type]) return
        this.listeners[type] = this.listeners[type].filter(
          (entry) => entry !== listener
        )
      }

      dispatchEvent(event: Event): boolean {
        for (const listener of this.listeners[event.type] || []) {
          if (typeof listener === "function") {
            listener.call(this, event)
          } else {
            listener.handleEvent(event)
          }
        }
        return true
      }

      emitEvent(event: Event) {
        if (event.type === "open") this.onopen?.(event)
        if (event.type === "message") this.onmessage?.(event as MessageEvent)
        if (event.type === "error") this.onerror?.(event)
        if (event.type === "close") this.onclose?.(event as CloseEvent)
        this.dispatchEvent(event)
      }

      send(payload: string) {
        this.sent.push(String(payload))
      }

      close() {
        this.readyState = MockWebSocket.CLOSED
        this.emitEvent(new CloseEvent("close"))
      }
    }

    window.WebSocket = MockWebSocket as unknown as typeof WebSocket
  })
}

const emitPersonaVisualStateOverride = async (
  page: Page,
  state: "speaking" | "tool_running" | "error"
): Promise<void> => {
  await page.evaluate(
    ({ state: nextState, personaId, sessionId }) => {
      ;(window as PersonaMockWindow).__personaWsMock?.emitJson?.({
        event: "visual_state_override",
        persona_id: personaId,
        session_id: sessionId,
        state: nextState,
        duration_ms: 10_000,
        reason: "persona_visuals.trigger_state"
      })
    },
    {
      state,
      personaId: VISUAL_PERSONA_ID,
      sessionId: VISUAL_SESSION_ID
    }
  )
}

const openPersonaVisualsTab = async (page: Page): Promise<void> => {
  const currentUrl = new URL(page.url())
  currentUrl.searchParams.set("tab", "visuals")
  await page.goto(`${currentUrl.pathname}${currentUrl.search}`, {
    waitUntil: "domcontentloaded"
  })
  await expect(page.getByTestId("persona-visual-pack-editor")).toBeVisible({
    timeout: 15_000
  })
}

const parseBooleanish = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value
  if (typeof value === "number") return value !== 0
  if (typeof value !== "string") return null
  const normalized = value.trim().toLowerCase()
  if (!normalized) return null
  if (["true", "1", "yes", "on", "enabled"].includes(normalized)) return true
  if (["false", "0", "no", "off", "disabled"].includes(normalized)) return false
  return null
}

const isPersonaAdvertised = (docsInfo: DocsInfoPayload): boolean => {
  const maps = [docsInfo?.capabilities, docsInfo?.supported_features]
  for (const map of maps) {
    if (!map || typeof map !== "object" || !("persona" in map)) {
      continue
    }
    const parsed = parseBooleanish(map.persona)
    if (parsed !== null) return parsed
  }
  return false
}

const resolvePersonaIdForLiveProof = (
  catalog: PersonaCatalogEntry[] | null
): string => {
  const entries = Array.isArray(catalog) ? catalog : []
  const preferred = entries.find(
    (entry) => String(entry?.id || "").trim() === DEFAULT_PERSONA_ID
  )
  if (preferred) {
    return DEFAULT_PERSONA_ID
  }

  const fallback = entries.find((entry) => String(entry?.id || "").trim().length > 0)
  return String(fallback?.id || DEFAULT_PERSONA_ID).trim() || DEFAULT_PERSONA_ID
}

const ensurePersonaSetupCompleted = async (personaId: string): Promise<void> => {
  const normalizedPersonaId = String(personaId || "").trim()
  if (!normalizedPersonaId) {
    throw new Error("Persona live proof requires a persona id")
  }

  const profileUrl = `${TEST_CONFIG.serverUrl}/api/v1/persona/profiles/${encodeURIComponent(
    normalizedPersonaId
  )}`
  const profileResp = await fetchWithApiKey(profileUrl, TEST_CONFIG.apiKey).catch(
    () => null
  )

  if (!profileResp?.ok) {
    throw new Error(
      `Failed to load persona profile for live proof (${normalizedPersonaId})`
    )
  }

  const profilePayload = (await profileResp.json().catch(() => null)) as
    | PersonaProfilePayload
    | null
  const expectedVersion =
    typeof profilePayload?.version === "number" ? profilePayload.version : null
  const updateUrl = expectedVersion
    ? `${profileUrl}?expected_version=${encodeURIComponent(String(expectedVersion))}`
    : profileUrl
  const completedSetup = {
    status: "completed",
    version: 1,
    run_id: `e2e-live-${Date.now()}`,
    current_step: "test",
    completed_steps: [...COMPLETED_SETUP_STEPS],
    completed_at: new Date().toISOString(),
    last_test_type: "live_session",
  }
  const updateResp = await fetchWithApiKey(updateUrl, TEST_CONFIG.apiKey, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      setup: completedSetup,
    }),
  }).catch(() => null)

  if (!updateResp?.ok) {
    throw new Error(
      `Failed to mark persona setup complete for live proof (${normalizedPersonaId})`
    )
  }
}

test.describe("Persona Live Workflow", () => {
  test.beforeEach(async ({ authedPage }) => {
    await seedAuth(authedPage, {
      serverUrl: TEST_CONFIG.webUrl,
      webUrl: TEST_CONFIG.webUrl,
      apiKey: TEST_CONFIG.apiKey,
    })
    await authedPage.route("**/api/**", async (route, request) => {
      const headers = {
        ...request.headers(),
      }

      if (
        TEST_CONFIG.apiKey &&
        !headers["x-api-key"] &&
        !headers["authorization"]
      ) {
        headers["x-api-key"] = TEST_CONFIG.apiKey
      }

      await route.continue({ headers })
    })
    await authedPage.addInitScript(() => {
      const OriginalWebSocket = window.WebSocket
      const seen: string[] = []
      ;(window as Window & { __tldwSeenWsUrls?: string[] }).__tldwSeenWsUrls = seen
      window.WebSocket = class extends OriginalWebSocket {
        constructor(url: string | URL, protocols?: string | string[]) {
          seen.push(String(url))
          super(url, protocols)
        }
      } as typeof WebSocket
    })
  })

  test("renders active persona visual pack and follows visual state overrides", async ({
    authedPage,
    diagnostics
  }) => {
    await installPersonaVisualApiMocks(authedPage)
    await installMockPersonaWebSocket(authedPage)

    await authedPage.goto(
      `/persona?persona_id=${encodeURIComponent(VISUAL_PERSONA_ID)}`,
      { waitUntil: "domcontentloaded" }
    )

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    await expect(authedPage.getByTestId("persona-buddy-dock")).toContainText(
      "Visual Persona",
      { timeout: 15_000 }
    )

    const visualFrame = authedPage.getByTestId("persona-visual-frame").first()
    await expect(visualFrame).toHaveAttribute("data-visual-state", "idle", {
      timeout: 15_000
    })

    await authedPage
      .getByRole("button", { name: /^Connect$/ })
      .evaluate((el: HTMLElement) => el.click())
    await expect(
      authedPage.getByRole("button", { name: /^Disconnect$/ })
    ).toBeVisible({ timeout: 15_000 })
    await expect(
      authedPage.getByText(/^Persona stream connected$/)
    ).toBeVisible({ timeout: 15_000 })

    for (const state of ["speaking", "tool_running", "error"] as const) {
      await emitPersonaVisualStateOverride(authedPage, state)
      await expect(visualFrame).toHaveAttribute("data-visual-state", state, {
        timeout: 10_000
      })
    }

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("default starter setup path activates a draft and renders BuddyShell visual", async ({
    authedPage,
    diagnostics
  }) => {
    await installPersonaVisualApiMocks(authedPage, {
      visualPackMode: "empty-pack"
    })
    await installMockPersonaWebSocket(authedPage)

    await authedPage.goto(
      `/persona?persona_id=${encodeURIComponent(VISUAL_PERSONA_ID)}`,
      { waitUntil: "domcontentloaded" }
    )

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    await expect(authedPage.getByTestId("persona-buddy-dock")).toContainText(
      "Visual Persona",
      { timeout: 15_000 }
    )
    await openPersonaVisualsTab(authedPage)

    await expect(
      authedPage.getByTestId("persona-visual-starter-pack-select")
    ).toHaveValue(PERSONA_VISUAL_E2E_STARTER_PACK.id)

    await authedPage.getByTestId("persona-visual-starter-copy-button").click()
    await expect(authedPage.getByTestId("persona-visual-pack-select")).toHaveValue(
      "starter-copy-e2e",
      { timeout: 15_000 }
    )
    await expect(authedPage.getByTestId("persona-visual-pack-status")).toHaveText(
      "draft"
    )

    await authedPage.getByTestId("persona-visual-activate-button").click()
    await expect(authedPage.getByTestId("persona-visual-pack-status")).toHaveText(
      "active",
      { timeout: 15_000 }
    )
    await expect(authedPage.getByTestId("persona-visual-frame").first())
      .toHaveAttribute("data-visual-state", "idle", { timeout: 15_000 })

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("uploaded pack setup path imports, activates, and renders BuddyShell visual", async ({
    authedPage,
    diagnostics
  }) => {
    await installPersonaVisualApiMocks(authedPage, {
      visualPackMode: "empty-pack"
    })
    await installMockPersonaWebSocket(authedPage)

    await authedPage.goto(
      `/persona?persona_id=${encodeURIComponent(VISUAL_PERSONA_ID)}`,
      { waitUntil: "domcontentloaded" }
    )

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    await openPersonaVisualsTab(authedPage)

    const portableUpload = await buildPortablePersonaVisualPackUpload()
    await authedPage
      .getByTestId("persona-visual-import-preview-input")
      .setInputFiles(portableUpload)
    await authedPage.getByTestId("persona-visual-import-preview-button").click()
    await expect(
      authedPage.getByTestId("persona-visual-import-preview-status")
    ).toHaveText("queued")

    await authedPage
      .getByTestId("persona-visual-import-preview-refresh-button")
      .click()
    await expect(
      authedPage.getByTestId("persona-visual-import-preview-status")
    ).toHaveText("completed", { timeout: 15_000 })
    await expect(
      authedPage.getByTestId("persona-visual-import-preview-summary")
    ).toContainText("Uploaded Visual Pack")

    await authedPage.getByTestId("persona-visual-import-commit-button").click()
    await expect(
      authedPage.getByTestId("persona-visual-import-commit-status")
    ).toHaveText("queued")
    await authedPage
      .getByTestId("persona-visual-import-commit-refresh-button")
      .click()
    await expect(
      authedPage.getByTestId("persona-visual-import-commit-status")
    ).toHaveText("completed", { timeout: 15_000 })
    await expect(authedPage.getByTestId("persona-visual-pack-select")).toHaveValue(
      "uploaded-pack-e2e",
      { timeout: 15_000 }
    )

    await authedPage.getByTestId("persona-visual-activate-button").click()
    await expect(authedPage.getByTestId("persona-visual-pack-status")).toHaveText(
      "active",
      { timeout: 15_000 }
    )
    await expect(authedPage.getByTestId("persona-visual-frame").first())
      .toHaveAttribute("data-visual-state", "idle", { timeout: 15_000 })

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("keeps live controls usable when visual pack loading fails", async ({
    authedPage,
    diagnostics
  }) => {
    await installPersonaVisualApiMocks(authedPage, {
      visualPackMode: "broken-pack"
    })
    await installMockPersonaWebSocket(authedPage)

    await authedPage.goto(
      `/persona?persona_id=${encodeURIComponent(VISUAL_PERSONA_ID)}`,
      { waitUntil: "domcontentloaded" }
    )

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    await expect(authedPage.getByTestId("persona-buddy-dock")).toContainText(
      "Visual Persona",
      { timeout: 15_000 }
    )
    await expect(
      authedPage.getByTestId("persona-visual-frame")
    ).toHaveCount(0)

    await authedPage
      .getByRole("button", { name: /^Connect$/ })
      .evaluate((el: HTMLElement) => el.click())
    await expect(
      authedPage.getByRole("button", { name: /^Disconnect$/ })
    ).toBeVisible({ timeout: 15_000 })
    await expect(
      authedPage.getByText(/^Persona stream connected$/)
    ).toBeVisible({ timeout: 15_000 })

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("connects to live persona websocket and receives a plan/cancel notice", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)

    const docsInfoResp = await fetchWithApiKey(
      `${TEST_CONFIG.serverUrl}/api/v1/config/docs-info`,
      TEST_CONFIG.apiKey
    ).catch(() => null)

    if (!docsInfoResp?.ok) {
      test.skip(true, "docs-info unavailable; cannot verify persona capability")
    }

    const docsInfo = (await docsInfoResp?.json().catch(() => null)) as
      | DocsInfoPayload
      | null
    if (!docsInfo || !isPersonaAdvertised(docsInfo)) {
      test.skip(true, "persona capability disabled on backend")
    }

    const catalogResp = await fetchWithApiKey(
      `${TEST_CONFIG.serverUrl}/api/v1/persona/catalog`,
      TEST_CONFIG.apiKey
    ).catch(() => null)
    const personaCatalog = (await catalogResp?.json().catch(() => null)) as
      | PersonaCatalogEntry[]
      | null
    const livePersonaId = resolvePersonaIdForLiveProof(personaCatalog)

    await ensurePersonaSetupCompleted(livePersonaId)

    await authedPage.goto(
      `/persona?persona_id=${encodeURIComponent(livePersonaId)}`,
      { waitUntil: "domcontentloaded" }
    )

    const personaUnavailableVisible = await authedPage
      .getByText("Persona unavailable")
      .isVisible()
      .catch(() => false)
    if (personaUnavailableVisible) {
      test.skip(true, "webui marked persona unavailable")
    }

    const setupOverlay = authedPage.getByTestId("assistant-setup-overlay")
    const connectButton = authedPage.getByRole("button", { name: /^Connect$/ })
    const disconnectButton = authedPage.getByRole("button", { name: /^Disconnect$/ })

    await expect(setupOverlay).toBeHidden({ timeout: 15_000 })

    const liveControlsState = await Promise.any([
      connectButton.waitFor({ state: "visible", timeout: 30_000 }).then(
        () => "connect" as const
      ),
      disconnectButton.waitFor({ state: "visible", timeout: 30_000 }).then(
        () => "disconnect" as const
      ),
    ]).catch(() => {
      throw new Error("Persona live proof did not reach connect or disconnect controls")
    })

    if (liveControlsState === "connect") {
      await connectButton.evaluate((el: HTMLElement) => el.click())
    }

    await expect(disconnectButton).toBeVisible({ timeout: 30000 })
    await expect(
      authedPage.getByText(/^Persona stream connected$/)
    ).toBeVisible({ timeout: 30000 })

    await authedPage
      .getByPlaceholder("Ask Persona...")
      .fill(`live persona ws ${Date.now()}`)
    await authedPage.getByRole("button", { name: /^Send$/ }).click()

    await expect(
      authedPage.getByText("Pending tool plan")
    ).toBeVisible({ timeout: 45000 })

    await authedPage.getByRole("button", { name: /^Cancel$/ }).click()

    await Promise.any([
      authedPage.getByText(/^Cancelled pending plan$/).waitFor({
        state: "visible",
        timeout: 30_000
      }),
      authedPage.getByText(/^Cancelled pending work\b/i).waitFor({
        state: "visible",
        timeout: 30_000
      }),
    ]).catch(() => {
      throw new Error("Expected a cancelled pending-work notice")
    })

    const seenWsUrls = await authedPage.evaluate(
      () => (window as Window & { __tldwSeenWsUrls?: string[] }).__tldwSeenWsUrls || []
    )
    const webHost = new URL(TEST_CONFIG.webUrl).host
    const backendHost = new URL(TEST_CONFIG.serverUrl).host
    const personaWsUrls = seenWsUrls.filter((raw: string) =>
      raw.includes("/api/v1/persona/stream")
    )

    expect(personaWsUrls.length).toBeGreaterThan(0)
    expect(
      personaWsUrls.some((raw: string) => new URL(raw).host === webHost)
    ).toBe(true)
    expect(
      personaWsUrls.every((raw: string) => new URL(raw).host !== backendHost)
    ).toBe(true)
    expect(diagnostics.pageErrors).toHaveLength(0)
  })
})
