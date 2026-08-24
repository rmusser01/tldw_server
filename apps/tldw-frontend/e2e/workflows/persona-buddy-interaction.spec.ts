/**
 * Persona Buddy Interaction E2E Tests
 *
 * Verifies the compact Buddy shell can use live-control session state while
 * sending text through the existing Persona stream transport.
 */
import { createHash } from "node:crypto"

import type { Page, Route } from "@playwright/test"

import {
  buildPersonaVisualPackFixture,
  PERSONA_VISUAL_E2E_FRAME_DATA_URI,
  PERSONA_VISUAL_E2E_PERSONA_ID,
  PERSONA_VISUAL_E2E_SESSION_ID
} from "../fixtures/persona-visual-packs"
import { test, expect } from "../utils/fixtures"

type PersonaMockWindow = Window & {
  __personaWsMock?: {
    getSentPayloads: () => unknown[]
    getSocketUrls: () => string[]
    emitJson: (payload: unknown) => void
  }
  __personaVisibilityMock?: {
    setHidden: (hidden: boolean) => void
  }
}

type AmbientMode = "off" | "expressive" | "roaming"

type BuddyInteractionMockOptions = {
  globalMode?: AmbientMode
  personaMode?: AmbientMode | null
  preferenceReadStatus?: number
  failAssetState?: "idle" | "speaking"
  focusedSession?: boolean
  ambientState?: "look" | "walk"
  includeSecondPersona?: boolean
}

const VISUAL_PERSONA_ID = PERSONA_VISUAL_E2E_PERSONA_ID
const VISUAL_SESSION_ID = PERSONA_VISUAL_E2E_SESSION_ID
const SECOND_PERSONA_ID = "visual_persona_e2e_second"
const USER_MESSAGE_TEXT = "hello from buddy shell"
const FRAME_BYTES = Buffer.from(
  PERSONA_VISUAL_E2E_FRAME_DATA_URI.split(",")[1] || "",
  "base64"
)
const FRAME_SHA256 = createHash("sha256").update(FRAME_BYTES).digest("hex")

const COMPLETED_SETUP = {
  status: "completed",
  version: 1,
  run_id: "e2e-buddy-interaction-setup",
  current_step: "test",
  completed_steps: ["persona", "voice", "commands", "safety", "test"],
  completed_at: "2026-05-20T00:00:00.000Z",
  last_test_type: "live_session"
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

const buildLiveSessionSummary = (overrides: Record<string, unknown> = {}) => ({
  session_id: VISUAL_SESSION_ID,
  persona_id: VISUAL_PERSONA_ID,
  persona_name: "Visual Persona",
  lifecycle: "connected",
  status: "active",
  is_focused: true,
  focused_at: "2026-05-20T00:00:00.000Z",
  focus_generation: 1,
  last_activity_at: "2026-05-20T00:00:00.000Z",
  pending_approval_count: 0,
  active_tool_name: null,
  error_state: null,
  recovery_hint: null,
  suggested_visual_state: "idle",
  allowed_actions: ["send_text_ws", "focus", "stop"],
  capabilities: {
    text: true,
    voice: false,
    browser_microphone_required: false
  },
  ...overrides
})

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

const buildAmbientVisualPack = (
  personaId: string,
  ambientState?: "look" | "walk"
) => {
  const base = buildPersonaVisualPackFixture(personaId)
  const assets = base.assets.map((asset) => ({
    ...asset,
    url: `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/visual-packs/${encodeURIComponent(base.id)}/assets/${encodeURIComponent(asset.id)}/content`,
    byte_size: FRAME_BYTES.byteLength,
    checksum_sha256: FRAME_SHA256,
    duration_ms: null
  }))
  const idleAssetId = assets.find((asset) =>
    asset.id.endsWith("-frame-idle")
  )?.id
  const speakingAssetId = assets.find((asset) =>
    asset.id.endsWith("-frame-speaking")
  )?.id
  const toolAssetId = assets.find((asset) =>
    asset.id.endsWith("-frame-tool")
  )?.id
  const reactionAssetId = assets.find((asset) =>
    asset.id.endsWith("-frame-error")
  )?.id
  if (!idleAssetId || !speakingAssetId || !toolAssetId || !reactionAssetId) {
    throw new Error("Persona visual fixture is missing a required frame")
  }
  const oneFrameAnimation = (assetId: string) => ({
    frames: [{ asset_id: assetId, duration_ms: 8_000 }],
    loop: false
  })

  const ambientEntries = [
    {
      state: "ambient.look",
      trigger: "ambient",
      category: "idle_variant",
      suggested_weight: 1,
      suggested_cooldown_ms: 0
    },
    {
      state: "ambient.walk",
      trigger: "ambient",
      category: "move",
      suggested_weight: 3,
      suggested_cooldown_ms: 0,
      movement: {
        direction: "horizontal",
        motion_start_ratio: 0.1,
        motion_end_ratio: 0.9
      }
    }
  ].filter(
    (entry) => !ambientState || entry.state === `ambient.${ambientState}`
  )

  return {
    ...base,
    assets,
    assets_by_id: Object.fromEntries(assets.map((asset) => [asset.id, asset])),
    companion_behavior: {
      schema_version: 1,
      entries: [
        ...ambientEntries,
        {
          state: "reaction.click",
          trigger: "click",
          category: "reaction"
        },
        {
          state: "reaction.drag",
          trigger: "drag",
          category: "reaction"
        }
      ]
    },
    review: null,
    parent_pack_id: null,
    revision_number: 1,
    manifest: {
      ...base.manifest,
      states: {
        ...base.manifest.states,
        "ambient.look": { animation_id: "ambient-look" },
        "ambient.walk": { animation_id: "ambient-walk" },
        "reaction.click": { animation_id: "reaction-click" },
        "reaction.drag": { animation_id: "reaction-drag" }
      },
      animations: {
        ...base.manifest.animations,
        "idle-loop": oneFrameAnimation(idleAssetId),
        "speaking-loop": oneFrameAnimation(speakingAssetId),
        "ambient-look": oneFrameAnimation(toolAssetId),
        "ambient-walk": oneFrameAnimation(toolAssetId),
        "reaction-click": oneFrameAnimation(reactionAssetId),
        "reaction-drag": oneFrameAnimation(reactionAssetId)
      }
    }
  }
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

    type LocalPersonaMockWindow = Window & {
      __personaWsMock?: {
        getSentPayloads: () => unknown[]
        getSocketUrls: () => string[]
        emitJson: (payload: unknown) => void
      }
    }

    const state: { sockets: MockSocket[] } = { sockets: [] }
    const personaWindow = window as LocalPersonaMockWindow

    personaWindow.__personaWsMock = {
      getSocketUrls: () => state.sockets.map((socket) => socket.url),
      getSentPayloads: () => {
        const socket = [...state.sockets]
          .reverse()
          .find((candidate) => candidate.url.includes("/persona/stream"))
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
        const socket = [...state.sockets]
          .reverse()
          .find((candidate) => candidate.url.includes("/persona/stream"))
        if (!socket) return
        const event = new MessageEvent("message", {
          data: JSON.stringify(payload)
        })
        socket.onmessage?.(event)
        socket.dispatchEvent?.(event)
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

const installVisibilityMock = async (page: Page): Promise<void> => {
  await page.addInitScript(() => {
    let hidden = false
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => (hidden ? "hidden" : "visible")
    })
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => hidden
    })
    ;(window as PersonaMockWindow).__personaVisibilityMock = {
      setHidden: (nextHidden) => {
        hidden = nextHidden
        document.dispatchEvent(new Event("visibilitychange"))
      }
    }
  })
}

const installDeterministicRandom = async (
  page: Page,
  values: number[]
): Promise<void> => {
  await page.addInitScript((sequence) => {
    let index = 0
    Math.random = () => sequence[index++] ?? sequence.at(-1) ?? 0
  }, values)
}

const installBuddyInteractionApiMocks = async (
  page: Page,
  options: BuddyInteractionMockOptions = {}
): Promise<void> => {
  const pack = buildAmbientVisualPack(VISUAL_PERSONA_ID, options.ambientState)
  const secondPack = buildAmbientVisualPack(
    SECOND_PERSONA_ID,
    options.ambientState
  )

  await page.route(/\/api\/v1\/health(?:\/live)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, { status: "ok" })
  })

  await page.route(/\/api\/v1\/rag\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      status: "healthy",
      components: {
        search_index: { status: "healthy", fts_table_count: 1 }
      }
    })
  })

  await page.route("**/openapi.json", async (route) => {
    await fulfillJson(route, 200, {
      info: { version: "e2e-persona-buddy-interaction" },
      paths: {
        "/api/v1/persona/catalog": {},
        "/api/v1/persona/profiles": {},
        "/api/v1/persona/profiles/{persona_id}": {},
        "/api/v1/persona/profiles/{persona_id}/visual-packs": {},
        "/api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets/{asset_id}/content":
          {},
        "/api/v1/persona/buddy/preferences": {},
        "/api/v1/persona/profiles/{persona_id}/buddy/preferences": {},
        "/api/v1/persona/session": {},
        "/api/v1/persona/sessions": {},
        "/api/v1/persona/live/sessions": {},
        "/api/v1/persona/live/sessions/{session_id}/focus": {},
        "/api/v1/persona/live/sessions/{session_id}/stop": {},
        "/api/v1/persona/stream": {},
        "/api/v1/personalization/profile": {}
      }
    })
  })

  await page.route(/\/api\/v1\/config\/docs-info(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, {
      capabilities: {
        persona: true,
        hasPersonaLiveControl: true,
        personalization: true
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
        description: "Persona Buddy interaction E2E fixture",
        buddy_summary: VISUAL_BUDDY_SUMMARY
      },
      ...(options.includeSecondPersona
        ? [
            {
              id: SECOND_PERSONA_ID,
              name: "Second Persona",
              description: "Focused Persona switch fixture",
              buddy_summary: {
                ...VISUAL_BUDDY_SUMMARY,
                persona_name: "Second Persona"
              }
            }
          ]
        : [])
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
      },
      ...(options.includeSecondPersona
        ? [
            {
              id: SECOND_PERSONA_ID,
              name: "Second Persona",
              version: 1,
              setup: COMPLETED_SETUP,
              buddy_summary: {
                ...VISUAL_BUDDY_SUMMARY,
                persona_name: "Second Persona"
              }
            }
          ]
        : [])
    ])
  })

  await page.route(
    /\/api\/v1\/persona\/buddy\/preferences(?:\?.*)?$/,
    async (route) => {
      if (options.preferenceReadStatus && route.request().method() === "GET") {
        await fulfillJson(route, options.preferenceReadStatus, {
          detail: "preference read failed"
        })
        return
      }
      await fulfillJson(route, 200, {
        ambient_mode: options.globalMode ?? "expressive",
        version: 1,
        stored: true
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+\/buddy\/preferences(?:\?.*)?$/,
    async (route) => {
      if (options.preferenceReadStatus && route.request().method() === "GET") {
        await fulfillJson(route, options.preferenceReadStatus, {
          detail: "preference read failed"
        })
        return
      }
      await fulfillJson(route, 200, {
        ambient_mode: options.personaMode ?? null,
        version: 1,
        stored: options.personaMode != null
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs(?:\?.*)?$/,
    async (route) => {
      const activePack =
        extractProfilePersonaId(route) === SECOND_PERSONA_ID ? secondPack : pack
      await fulfillJson(route, 200, {
        packs: [activePack],
        active_pack: activePack
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/[^/]+\/assets\/[^/]+\/content(?:\?.*)?$/,
    async (route) => {
      const assetId = decodeURIComponent(
        new URL(route.request().url()).pathname.split("/").at(-2) || ""
      )
      if (
        options.failAssetState &&
        assetId.endsWith(`-frame-${options.failAssetState}`)
      ) {
        await fulfillJson(route, 503, { detail: "asset unavailable" })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: "image/png",
        headers: { "Cache-Control": "private, no-store" },
        body: FRAME_BYTES
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+(?:\?.*)?$/,
    async (route) => {
      const personaId = extractProfilePersonaId(route)
      await fulfillJson(route, 200, {
        id: personaId,
        name:
          personaId === VISUAL_PERSONA_ID
            ? "Visual Persona"
            : personaId === SECOND_PERSONA_ID
              ? "Second Persona"
              : personaId,
        version: 1,
        voice_defaults: null,
        setup: COMPLETED_SETUP,
        buddy_summary: {
          ...VISUAL_BUDDY_SUMMARY,
          persona_name:
            personaId === VISUAL_PERSONA_ID
              ? "Visual Persona"
              : personaId === SECOND_PERSONA_ID
                ? "Second Persona"
                : personaId
        }
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/live\/sessions(?:\?.*)?$/,
    async (route) => {
      const method = route.request().method().toUpperCase()
      if (method === "POST") {
        await fulfillJson(route, 200, {
          session: buildLiveSessionSummary()
        })
        return
      }
      await fulfillJson(route, 200, {
        sessions:
          options.focusedSession === false ? [] : [buildLiveSessionSummary()],
        focused_session_id:
          options.focusedSession === false ? null : VISUAL_SESSION_ID
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/live\/sessions\/[^/]+\/focus(?:\?.*)?$/,
    async (route) => {
      await fulfillJson(route, 200, {
        session: buildLiveSessionSummary()
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/live\/sessions\/[^/]+\/stop(?:\?.*)?$/,
    async (route) => {
      await fulfillJson(route, 200, {
        session: buildLiveSessionSummary({
          lifecycle: "stopped",
          status: "closed",
          is_focused: false,
          allowed_actions: ["focus"]
        })
      })
    }
  )

  await page.route(/\/api\/v1\/persona\/sessions(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, 200, [])
  })

  await page.route(
    /\/api\/v1\/persona\/sessions\/[^/?]+(?:\?.*)?$/,
    async (route) => {
      await fulfillJson(route, 200, {
        session_id: VISUAL_SESSION_ID,
        persona: { id: VISUAL_PERSONA_ID },
        preferences: null,
        turns: []
      })
    }
  )

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

const gotoPersonaBuddy = async (
  page: Page,
  expectedMode: AmbientMode
): Promise<ReturnType<Page["getByTestId"]>> => {
  await page.goto(`/persona?persona_id=${VISUAL_PERSONA_ID}`, {
    waitUntil: "domcontentloaded"
  })
  await expect(page.getByTestId("assistant-setup-overlay")).toBeHidden({
    timeout: 15_000
  })
  const dock = page.getByTestId("persona-buddy-dock")
  await expect(dock).toBeVisible({ timeout: 15_000 })
  await expect(dock).toHaveAttribute(
    "data-companion-effective-mode",
    expectedMode
  )
  await expect(page.getByTestId("persona-visual-frame")).toHaveAttribute(
    "src",
    /^blob:/
  )
  return dock
}

const emitVisualState = async (
  page: Page,
  state: "idle" | "speaking"
): Promise<void> => {
  await page.evaluate(
    ({ nextState, personaId, sessionId }) => {
      ;(window as PersonaMockWindow).__personaWsMock?.emitJson({
        event: "visual_state_override",
        persona_id: personaId,
        session_id: sessionId,
        state: nextState,
        duration_ms: 30_000,
        reason:
          nextState === "speaking"
            ? "persona_visuals.trigger_state"
            : "test.semantic.resume"
      })
    },
    {
      nextState: state,
      personaId: VISUAL_PERSONA_ID,
      sessionId: VISUAL_SESSION_ID
    }
  )
}

const connectPersonaPageStream = async (page: Page): Promise<void> => {
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await page.clock.runFor(0)
  await expect
    .poll(
      async () =>
        await page.evaluate(
          () =>
            (window as PersonaMockWindow).__personaWsMock?.getSocketUrls() ?? []
        )
    )
    .toContainEqual(expect.stringContaining("/api/v1/persona/stream"))
}

const pauseBrowserClock = async (page: Page): Promise<void> => {
  const now = await page.evaluate(() => Date.now())
  await page.clock.pauseAt(now)
}

const setDocumentHidden = async (
  page: Page,
  hidden: boolean
): Promise<void> => {
  await page.evaluate((nextHidden) => {
    ;(window as PersonaMockWindow).__personaVisibilityMock?.setHidden(
      nextHidden
    )
  }, hidden)
}

const readPersistedPosition = async (
  page: Page,
  bucket: "web-desktop" | "sidepanel-desktop"
): Promise<{ x: number; y: number } | null> =>
  await page.evaluate((positionBucket) => {
    const raw = localStorage.getItem("tldw-persona-buddy-shell")
    if (!raw) return null
    const parsed = JSON.parse(raw) as {
      state?: {
        positions?: Partial<
          Record<"web-desktop" | "sidepanel-desktop", { x: number; y: number }>
        >
      }
    }
    return parsed.state?.positions?.[positionBucket] ?? null
  }, bucket)

test.describe("Persona Buddy interaction", () => {
  test.beforeEach(async ({ authedPage }) => {
    await installMockPersonaWebSocket(authedPage)
    await installVisibilityMock(authedPage)
  })

  test("sends Buddy popover text through Persona stream and routes to Visuals", async ({
    authedPage,
    diagnostics
  }) => {
    await installBuddyInteractionApiMocks(authedPage)
    const protectedAssetRequest = authedPage.waitForRequest(
      /\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs\/[^/]+\/assets\/[^/]+\/content/
    )
    await authedPage.goto(`/persona?persona_id=${VISUAL_PERSONA_ID}`, {
      waitUntil: "domcontentloaded"
    })

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    const buddyDock = authedPage.getByTestId("persona-buddy-dock")
    const buddyButton = authedPage.getByRole("button", {
      name: "Toggle buddy for Visual Persona"
    })
    await expect(buddyButton).toBeVisible({ timeout: 15_000 })
    await expect(buddyButton).toHaveCSS("width", "64px")
    await expect(buddyButton).toHaveCSS("height", "64px")
    await expect(
      authedPage.getByTestId("persona-visual-frame")
    ).toHaveAttribute("src", /^blob:/)
    expect((await protectedAssetRequest).headers()["x-api-key"]).toBeTruthy()
    await expect(
      authedPage.getByTestId("persona-buddy-live-status")
    ).toHaveCount(0)
    await expect(authedPage.getByTestId("persona-buddy-popover")).toHaveCount(0)
    await expect(buddyDock).toHaveAttribute("data-companion-phase", "idle")

    await authedPage
      .getByRole("button", { name: "Open Buddy controls" })
      .click()
    await expect(authedPage.getByTestId("persona-buddy-popover")).toBeVisible()
    await expect(
      authedPage.getByTestId("persona-buddy-text-input")
    ).toBeVisible()

    await authedPage
      .getByTestId("persona-buddy-text-input")
      .fill(USER_MESSAGE_TEXT)
    await authedPage
      .getByTestId("persona-buddy-popover")
      .getByRole("button", { name: "Send" })
      .evaluate((element: HTMLElement) => element.click())

    await expect
      .poll(
        async () =>
          await authedPage.evaluate(() => {
            return (
              (
                window as PersonaMockWindow
              ).__personaWsMock?.getSentPayloads() ?? []
            )
          })
      )
      .toContainEqual(
        expect.objectContaining({
          type: "user_message",
          session_id: VISUAL_SESSION_ID,
          text: USER_MESSAGE_TEXT,
          client_message_id: expect.stringMatching(/^persona-buddy-draft:/)
        })
      )

    await authedPage
      .getByTestId("persona-buddy-open-visuals-link")
      .evaluate((element: HTMLElement) => element.click())

    await expect
      .poll(async () => new URL(authedPage.url()).pathname)
      .toBe("/persona")
    await expect
      .poll(async () =>
        new URL(authedPage.url()).searchParams.get("persona_id")
      )
      .toBe(VISUAL_PERSONA_ID)
    await expect
      .poll(async () => new URL(authedPage.url()).searchParams.get("tab"))
      .toBe("visuals")

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("keeps Off idle while preserving an authored direct reaction", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, { globalMode: "off" })
    const dock = await gotoPersonaBuddy(authedPage, "off")
    await pauseBrowserClock(authedPage)

    await authedPage.clock.runFor(90_000)
    await expect(dock).toHaveAttribute("data-companion-phase", "idle")
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(dock).toHaveAttribute("data-companion-transient-offset-x", "0")

    await authedPage
      .getByRole("button", { name: "Toggle buddy for Visual Persona" })
      .click()
    await authedPage.clock.runFor(299)
    await expect(dock).toHaveAttribute("data-companion-phase", "idle")
    await authedPage.clock.runFor(1)
    await expect(dock).toHaveAttribute("data-companion-phase", "action")
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "reaction.click"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("opens controls by double-click", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, { globalMode: "off" })
    await gotoPersonaBuddy(authedPage, "off")
    await pauseBrowserClock(authedPage)

    const buddyButton = authedPage.getByRole("button", {
      name: "Toggle buddy for Visual Persona"
    })
    await buddyButton.dblclick()
    await expect(authedPage.getByTestId("persona-buddy-popover")).toBeVisible()

    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("opens controls with Enter", async ({ authedPage, diagnostics }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, { globalMode: "off" })
    await gotoPersonaBuddy(authedPage, "off")
    await pauseBrowserClock(authedPage)

    const keyboardBuddyButton = authedPage.getByRole("button", {
      name: "Toggle buddy for Visual Persona"
    })
    await keyboardBuddyButton.focus()
    await keyboardBuddyButton.press("Enter")
    await expect(authedPage.getByTestId("persona-buddy-popover")).toBeVisible()
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("reacts to Space and a touch tap through the same authored click state", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, { globalMode: "off" })
    const dock = await gotoPersonaBuddy(authedPage, "off")
    await pauseBrowserClock(authedPage)
    const buddyButton = authedPage.getByRole("button", {
      name: "Toggle buddy for Visual Persona"
    })

    await buddyButton.focus()
    await buddyButton.press("Space")
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "reaction.click"
    )
    await authedPage.clock.runFor(1_000)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")

    const box = await buddyButton.boundingBox()
    if (!box) throw new Error("expected visible Buddy button")
    const touch = {
      id: 73,
      x: box.x + box.width / 2,
      y: box.y + box.height / 2
    }
    await buddyButton.evaluate((element, pointer) => {
      element.setPointerCapture = () => undefined
      element.releasePointerCapture = () => undefined
      element.dispatchEvent(
        new PointerEvent("pointerdown", {
          bubbles: true,
          button: 0,
          clientX: pointer.x,
          clientY: pointer.y,
          isPrimary: true,
          pointerId: pointer.id,
          pointerType: "touch"
        })
      )
    }, touch)
    await authedPage.evaluate((pointer) => {
      window.dispatchEvent(
        new PointerEvent("pointerup", {
          bubbles: true,
          button: 0,
          clientX: pointer.x,
          clientY: pointer.y,
          isPrimary: true,
          pointerId: pointer.id,
          pointerType: "touch"
        })
      )
    }, touch)
    await authedPage.clock.runFor(299)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await authedPage.clock.runFor(1)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "reaction.click"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("restarts a full expressive interval after semantic suspension", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      globalMode: "expressive",
      ambientState: "look"
    })
    const dock = await gotoPersonaBuddy(authedPage, "expressive")
    await pauseBrowserClock(authedPage)
    await connectPersonaPageStream(authedPage)

    await emitVisualState(authedPage, "speaking")
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "speaking"
    )
    await expect(dock).toHaveAttribute("data-companion-suspension", "semantic")
    // Incoming overrides are capped at 30s and the Host clears expired values
    // every second. Resume explicitly before expiry to test the fresh interval.
    await authedPage.clock.runFor(29_999)
    await expect(dock).toHaveAttribute("data-companion-phase", "idle")
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "speaking"
    )
    await expect(dock).toHaveAttribute("data-companion-suspension", "semantic")

    await emitVisualState(authedPage, "idle")
    await expect(dock).toHaveAttribute("data-companion-suspension", "none")
    await expect(
      authedPage.getByTestId("persona-visual-frame")
    ).toHaveAttribute("data-visual-state", "idle")
    await authedPage.clock.runFor(0)
    const generationAtSemanticResume = Number(
      await dock.getAttribute("data-companion-generation")
    )
    await authedPage.clock.runFor(29_999)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(dock).toHaveAttribute(
      "data-companion-generation",
      String(generationAtSemanticResume)
    )
    await authedPage.clock.runFor(1)
    await expect
      .poll(async () =>
        Number(await dock.getAttribute("data-companion-generation"))
      )
      .toBeGreaterThan(generationAtSemanticResume)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "ambient.look"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("starts a fresh expressive interval after hidden-tab resume", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      globalMode: "expressive",
      ambientState: "look"
    })
    const dock = await gotoPersonaBuddy(authedPage, "expressive")
    await pauseBrowserClock(authedPage)

    await setDocumentHidden(authedPage, true)
    await expect(dock).toHaveAttribute("data-companion-suspension", "hidden")
    await authedPage.clock.runFor(120_000)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await setDocumentHidden(authedPage, false)
    await expect(dock).toHaveAttribute("data-companion-suspension", "none")
    const generationAtResume = Number(
      await dock.getAttribute("data-companion-generation")
    )
    await authedPage.clock.runFor(29_999)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(dock).toHaveAttribute(
      "data-companion-generation",
      String(generationAtResume)
    )
    await authedPage.clock.runFor(1)
    await expect
      .poll(async () =>
        Number(await dock.getAttribute("data-companion-generation"))
      )
      .toBeGreaterThan(generationAtResume)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "ambient.look"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("uses a static PNG and suppresses ambient motion for reduced motion", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.emulateMedia({ reducedMotion: "reduce" })
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      globalMode: "expressive",
      ambientState: "look"
    })
    const dock = await gotoPersonaBuddy(authedPage, "expressive")
    await pauseBrowserClock(authedPage)
    const frame = authedPage.getByTestId("persona-visual-frame")
    const idleBlobUrl = await frame.getAttribute("src")
    expect(idleBlobUrl).toMatch(/^blob:/)
    await expect(frame).toHaveAttribute("data-visual-state", "idle")
    await expect(dock).toHaveAttribute(
      "data-companion-suspension",
      "reduced_motion"
    )

    await authedPage.clock.runFor(90_000)
    await expect(dock).toHaveAttribute("data-companion-phase", "idle")
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(frame).toHaveAttribute("src", idleBlobUrl || "")
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("fails closed when ambient preferences cannot be read", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      globalMode: "roaming",
      preferenceReadStatus: 503,
      ambientState: "walk"
    })
    const dock = await gotoPersonaBuddy(authedPage, "off")
    await pauseBrowserClock(authedPage)
    await authedPage.clock.runFor(90_000)
    await expect(dock).toHaveAttribute("data-companion-phase", "idle")
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(dock).toHaveAttribute("data-companion-transient-offset-x", "0")
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("invalidates the old timer when the focused Persona changes", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      ambientState: "look",
      focusedSession: false,
      globalMode: "expressive",
      includeSecondPersona: true
    })
    const dock = await gotoPersonaBuddy(authedPage, "expressive")
    await pauseBrowserClock(authedPage)
    await authedPage
      .getByRole("button", { name: "Connect", exact: true })
      .click()
    await expect
      .poll(
        async () =>
          await authedPage.evaluate(
            () =>
              (window as PersonaMockWindow).__personaWsMock?.getSocketUrls() ??
              []
          )
      )
      .toContainEqual(expect.stringContaining("/api/v1/persona/stream"))
    await authedPage.clock.runFor(0)
    const disconnectButton = authedPage.getByRole("button", {
      name: /Disconnect/
    })
    await expect(disconnectButton).toBeVisible()
    await disconnectButton.click()
    await expect(authedPage.getByTitle("Visual Persona")).toBeVisible()
    await setDocumentHidden(authedPage, true)
    await expect(dock).toHaveAttribute("data-companion-suspension", "hidden")
    await setDocumentHidden(authedPage, false)
    await expect(dock).toHaveAttribute("data-companion-suspension", "none")
    await authedPage.clock.runFor(29_999)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    const generationBeforeSwitch = Number(
      await dock.getAttribute("data-companion-generation")
    )

    await authedPage.getByTitle("Visual Persona").click()
    await authedPage
      .locator(".ant-select-dropdown:visible .ant-select-item-option-content")
      .filter({ hasText: "Second Persona" })
      .click()
    await expect(
      authedPage.getByRole("button", {
        name: "Toggle buddy for Second Persona"
      })
    ).toBeVisible()
    await expect(
      authedPage.getByTestId("persona-visual-frame")
    ).toHaveAttribute("data-visual-state", "idle")
    await authedPage.clock.runFor(0)
    await expect
      .poll(async () =>
        Number(await dock.getAttribute("data-companion-generation"))
      )
      .toBeGreaterThan(generationBeforeSwitch)
    const switchedGeneration = Number(
      await dock.getAttribute("data-companion-generation")
    )

    await authedPage.clock.runFor(1)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await expect(dock).toHaveAttribute(
      "data-companion-generation",
      String(switchedGeneration)
    )
    await authedPage.clock.runFor(29_998)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    await authedPage.clock.runFor(1)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "ambient.look"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("retains the previous Blob when a protected semantic asset fails", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      failAssetState: "speaking",
      globalMode: "expressive"
    })
    const dock = await gotoPersonaBuddy(authedPage, "expressive")
    await pauseBrowserClock(authedPage)
    await connectPersonaPageStream(authedPage)
    const frame = authedPage.getByTestId("persona-visual-frame")
    const idleBlobUrl = await frame.getAttribute("src")
    expect(idleBlobUrl).toMatch(/^blob:/)

    await emitVisualState(authedPage, "speaking")
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "speaking"
    )
    await expect(frame).toHaveAttribute("src", idleBlobUrl || "")
    await expect(
      authedPage.getByTestId("persona-buddy-visual-diagnostic")
    ).toContainText("Visual asset did not load")
    expect(diagnostics.pageErrors).toHaveLength(0)
  })

  test("keeps roaming horizontal and ephemeral until an explicit drag persists", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.clock.install({
      time: new Date("2026-08-24T12:00:00.000Z")
    })
    await authedPage.addInitScript(() => {
      localStorage.setItem(
        "tldw-persona-buddy-shell",
        JSON.stringify({
          state: {
            firstUseHintDismissed: true,
            positions: {
              "web-desktop": { x: 320, y: 240 },
              "sidepanel-desktop": { x: 320, y: 240 }
            }
          },
          version: 0
        })
      )
    })
    await installDeterministicRandom(authedPage, [0])
    await installBuddyInteractionApiMocks(authedPage, {
      globalMode: "roaming",
      ambientState: "walk"
    })
    const dock = await gotoPersonaBuddy(authedPage, "roaming")
    await pauseBrowserClock(authedPage)
    const activePositionBucket = "sidepanel-desktop" as const
    const persistedBeforeAmbient = await readPersistedPosition(
      authedPage,
      activePositionBucket
    )
    const boxBeforeAmbient = await dock.boundingBox()
    expect(boxBeforeAmbient).not.toBeNull()

    await authedPage.clock.runFor(30_000)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "ambient.walk"
    )
    const transientOffset = Number(
      await dock.getAttribute("data-companion-transient-offset-x")
    )
    expect(Math.abs(transientOffset)).toBeGreaterThan(0)
    const boxDuringAmbient = await dock.boundingBox()
    expect(boxDuringAmbient).not.toBeNull()
    expect(boxDuringAmbient?.y).toBe(boxBeforeAmbient?.y)
    expect(boxDuringAmbient?.x ?? 0).toBeGreaterThanOrEqual(16)
    expect(
      (boxDuringAmbient?.x ?? 0) + (boxDuringAmbient?.width ?? 0)
    ).toBeLessThanOrEqual(
      await authedPage.evaluate(() => window.innerWidth - 16)
    )
    expect(
      await readPersistedPosition(authedPage, activePositionBucket)
    ).toEqual(persistedBeforeAmbient)

    await authedPage.clock.runFor(1_000)
    await expect(dock).toHaveAttribute("data-companion-requested-state", "idle")
    expect(
      await readPersistedPosition(authedPage, activePositionBucket)
    ).toEqual(persistedBeforeAmbient)

    const buddyButton = authedPage.getByRole("button", {
      name: "Toggle buddy for Visual Persona"
    })
    const dragBox = await buddyButton.boundingBox()
    if (!dragBox) throw new Error("expected visible Buddy button")
    const startX = dragBox.x + dragBox.width / 2
    const startY = dragBox.y + dragBox.height / 2
    const pointerId = 41
    await buddyButton.evaluate(
      (element, pointer) => {
        element.setPointerCapture = () => undefined
        element.releasePointerCapture = () => undefined
        element.dispatchEvent(
          new PointerEvent("pointerdown", {
            bubbles: true,
            button: 0,
            clientX: pointer.x,
            clientY: pointer.y,
            isPrimary: true,
            pointerId: pointer.id,
            pointerType: "mouse"
          })
        )
      },
      { id: pointerId, x: startX, y: startY }
    )
    await authedPage.evaluate(
      (pointer) => {
        window.dispatchEvent(
          new PointerEvent("pointermove", {
            bubbles: true,
            button: 0,
            clientX: pointer.x,
            clientY: pointer.y,
            isPrimary: true,
            pointerId: pointer.id,
            pointerType: "mouse"
          })
        )
      },
      { id: pointerId, x: startX + 96, y: startY + 32 }
    )
    await expect(dock).toHaveAttribute("data-companion-suspension", "drag")
    await authedPage.evaluate(
      (pointer) => {
        window.dispatchEvent(
          new PointerEvent("pointerup", {
            bubbles: true,
            button: 0,
            clientX: pointer.x,
            clientY: pointer.y,
            isPrimary: true,
            pointerId: pointer.id,
            pointerType: "mouse"
          })
        )
      },
      { id: pointerId, x: startX + 96, y: startY + 32 }
    )
    await expect
      .poll(
        async () =>
          await readPersistedPosition(authedPage, activePositionBucket)
      )
      .not.toEqual(persistedBeforeAmbient)
    await expect(dock).toHaveAttribute(
      "data-companion-requested-state",
      "reaction.drag"
    )
    expect(diagnostics.pageErrors).toHaveLength(0)
  })
})
