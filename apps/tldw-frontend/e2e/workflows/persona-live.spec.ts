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
const VISUAL_PERSONA_ID = "visual_persona_e2e"
const VISUAL_SESSION_ID = "sess-visual-e2e-001"
const VISUAL_FRAME_DATA_URI =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
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

type PersonaVisualMockMode = "active-pack" | "broken-pack"

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

const buildPersonaVisualPack = (personaId: string) => {
  const assetIds = {
    idle: "frame-idle",
    speaking: "frame-speaking",
    tool: "frame-tool",
    error: "frame-error"
  }
  const buildAsset = (id: string) => ({
    id,
    pack_id: "visual-pack-e2e",
    persona_id: personaId,
    asset_role: "frame",
    storage_key: `persona-visuals/${personaId}/${id}.png`,
    url: VISUAL_FRAME_DATA_URI,
    original_filename: `${id}.png`,
    mime_type: "image/png",
    byte_size: 96,
    checksum_sha256: `${id}-checksum`,
    width: 1,
    height: 1,
    provenance: "e2e_fixture",
    created_at: "2026-05-08T00:00:00.000Z",
    last_modified: "2026-05-08T00:00:00.000Z",
    version: 1
  })
  const assetsById = {
    [assetIds.idle]: buildAsset(assetIds.idle),
    [assetIds.speaking]: buildAsset(assetIds.speaking),
    [assetIds.tool]: buildAsset(assetIds.tool),
    [assetIds.error]: buildAsset(assetIds.error)
  }

  return {
    id: "visual-pack-e2e",
    persona_id: personaId,
    user_id: "e2e-user",
    title: "Visual Runtime Pack",
    renderer_type: "sprite_frames",
    status: "active",
    manifest_version: 1,
    manifest: {
      manifest_version: 1,
      renderer_type: "sprite_frames",
      states: {
        idle: { animation_id: "idle-loop" },
        speaking: { animation_id: "speaking-loop" },
        tool_running: { animation_id: "tool-loop" },
        error: { animation_id: "error-loop" }
      },
      animations: {
        "idle-loop": {
          frames: [{ asset_id: assetIds.idle, duration_ms: 250 }],
          loop: true
        },
        "speaking-loop": {
          frames: [{ asset_id: assetIds.speaking, duration_ms: 250 }],
          loop: true
        },
        "tool-loop": {
          frames: [{ asset_id: assetIds.tool, duration_ms: 250 }],
          loop: true
        },
        "error-loop": {
          frames: [{ asset_id: assetIds.error, duration_ms: 250 }],
          loop: false
        }
      },
      fallbacks: {
        speaking: ["idle"],
        tool_running: ["idle"],
        error: ["idle"]
      },
      authored_triggers: [
        {
          id: "mcp-runtime-override",
          source: "mcp_runtime",
          match: "persona_visuals.trigger_state",
          state: "speaking",
          duration_ms: 5000,
          priority: 10
        }
      ]
    },
    active_at: "2026-05-08T00:00:00.000Z",
    assets: Object.values(assetsById),
    assets_by_id: assetsById,
    created_at: "2026-05-08T00:00:00.000Z",
    last_modified: "2026-05-08T00:00:00.000Z",
    version: 1
  }
}

const installPersonaVisualApiMocks = async (
  page: Page,
  options: { visualPackMode?: PersonaVisualMockMode } = {}
): Promise<void> => {
  const visualPackMode = options.visualPackMode ?? "active-pack"

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

  await page.route(/\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs(?:\?.*)?$/, async (route) => {
    if (visualPackMode === "broken-pack") {
      await fulfillJson(route, 500, {
        detail: "visual pack fixture unavailable"
      })
      return
    }

    const pack = buildPersonaVisualPack(extractProfilePersonaId(route))
    await fulfillJson(route, 200, {
      packs: [pack],
      active_pack: pack
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
