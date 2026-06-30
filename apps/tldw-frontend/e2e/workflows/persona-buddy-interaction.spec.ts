/**
 * Persona Buddy Interaction E2E Tests
 *
 * Verifies the compact Buddy shell can use live-control session state while
 * sending text through the existing Persona stream transport.
 */
import type { Page, Route } from "@playwright/test"

import {
  buildPersonaVisualPackFixture,
  PERSONA_VISUAL_E2E_PERSONA_ID,
  PERSONA_VISUAL_E2E_SESSION_ID
} from "../fixtures/persona-visual-packs"
import { test, expect } from "../utils/fixtures"

type PersonaMockWindow = Window & {
  __personaWsMock?: {
    getSentPayloads: () => unknown[]
    emitJson: (payload: unknown) => void
  }
}

const VISUAL_PERSONA_ID = PERSONA_VISUAL_E2E_PERSONA_ID
const VISUAL_SESSION_ID = PERSONA_VISUAL_E2E_SESSION_ID
const USER_MESSAGE_TEXT = "hello from buddy shell"

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
        emitJson: (payload: unknown) => void
      }
    }

    const state: { sockets: MockSocket[] } = { sockets: [] }
    const personaWindow = window as LocalPersonaMockWindow

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

const installBuddyInteractionApiMocks = async (page: Page): Promise<void> => {
  const pack = buildPersonaVisualPackFixture(VISUAL_PERSONA_ID)

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

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+\/visual-packs(?:\?.*)?$/,
    async (route) => {
      await fulfillJson(route, 200, {
        packs: [pack],
        active_pack: pack
      })
    }
  )

  await page.route(
    /\/api\/v1\/persona\/profiles\/[^/]+(?:\?.*)?$/,
    async (route) => {
      const personaId = extractProfilePersonaId(route)
      await fulfillJson(route, 200, {
        id: personaId,
        name: personaId === VISUAL_PERSONA_ID ? "Visual Persona" : personaId,
        version: 1,
        voice_defaults: null,
        setup: COMPLETED_SETUP,
        buddy_summary: {
          ...VISUAL_BUDDY_SUMMARY,
          persona_name:
            personaId === VISUAL_PERSONA_ID ? "Visual Persona" : personaId
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
        sessions: [buildLiveSessionSummary()],
        focused_session_id: VISUAL_SESSION_ID
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

test.describe("Persona Buddy interaction", () => {
  test.beforeEach(async ({ authedPage }) => {
    await installMockPersonaWebSocket(authedPage)
    await installBuddyInteractionApiMocks(authedPage)
  })

  test("sends Buddy popover text through Persona stream and routes to Visuals", async ({
    authedPage,
    diagnostics
  }) => {
    await authedPage.goto(`/persona?persona_id=${VISUAL_PERSONA_ID}`, {
      waitUntil: "domcontentloaded"
    })

    await expect(authedPage.getByTestId("assistant-setup-overlay")).toBeHidden({
      timeout: 15_000
    })
    await expect(authedPage.getByTestId("persona-buddy-dock")).toContainText(
      "Visual Persona",
      { timeout: 15_000 }
    )

    await authedPage
      .getByRole("button", { name: "Toggle buddy for Visual Persona" })
      .click()
    await expect(authedPage.getByTestId("persona-buddy-popover")).toBeVisible()
    await expect(authedPage.getByTestId("persona-buddy-text-input")).toBeVisible()

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
              (window as PersonaMockWindow).__personaWsMock?.getSentPayloads() ??
              []
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

    await expect.poll(async () => new URL(authedPage.url()).pathname).toBe("/persona")
    await expect
      .poll(async () => new URL(authedPage.url()).searchParams.get("persona_id"))
      .toBe(VISUAL_PERSONA_ID)
    await expect
      .poll(async () => new URL(authedPage.url()).searchParams.get("tab"))
      .toBe("visuals")

    expect(diagnostics.pageErrors).toHaveLength(0)
  })
})
