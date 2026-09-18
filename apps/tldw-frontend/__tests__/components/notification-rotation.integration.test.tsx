import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  REFRESH_ROTATION_KEY,
  invalidateRefreshSessionIfCurrent,
  storeRefreshRotationIfCurrent
} from "@/services/tldw/single-user-credential"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import {
  NotificationLifecycleProvider,
  buildWebNotificationScopeKey,
  useNotificationLifecycle
} from "@web/components/notifications/NotificationLifecycleProvider"
import api, { buildAuthHeaders } from "@web/lib/api"
import { markNotificationsRead } from "@web/lib/api/notifications"
import { clearRuntimeAuth, setRuntimeApiBearer, setRuntimeApiKey } from "@web/lib/authStorage"
import { AUTH_CREDENTIALS_CHANGED_EVENT } from "@web/lib/auth-events"

vi.hoisted(() => {
  process.env.NEXT_PUBLIC_API_URL = "https://api.example.test"
})

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    isConnected: true, phase: "connected", mode: "normal", offlineBypass: false
  })
}))

const jwt = (subject: string, revision: number): string =>
  `header.${btoa(JSON.stringify({ sub: subject, revision }))}.signature`
const original: TldwConfig = {
  serverUrl: "https://api.example.test",
  authMode: "multi-user",
  authSource: "manual",
  orgId: 7,
  accessToken: jwt("alice", 0),
  refreshToken: "refresh-alice-0"
}
const rotated = { accessToken: jwt("alice", 1), refreshToken: "refresh-alice-1" }
const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status, headers: { "Content-Type": "application/json" }
  })
const bearer = (init?: RequestInit): string | null =>
  new Headers(init?.headers).get("Authorization")

function Probe() {
  const value = useNotificationLifecycle()
  return <output data-testid="lifecycle" data-scope={value.scopeKey}>{value.state}:{value.unreadCount}</output>
}

describe("notification rotation through actual storage and WebUI transport", () => {
  let persistent: Storage
  let requests: Array<{ url: string; init?: RequestInit }>
  let unreadStatus: number
  const originalEnv = { ...process.env }

  beforeEach(async () => {
    localStorage.clear()
    sessionStorage.clear()
    clearRuntimeAuth()
    delete process.env.NEXT_PUBLIC_API_BEARER
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    persistent = new Storage({ area: "local" })
    await persistent.set("tldwConfig", original)
    api.defaults.baseURL = "https://api.example.test/api/v1"
    requests = []
    unreadStatus = 200
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      requests.push({ url: String(url), init })
      if (String(url).includes("/unread-count")) {
        return unreadStatus === 200
          ? json({ unread_count: 3 })
          : json({ detail: "Request denied" }, unreadStatus)
      }
      if (String(url).includes("/stream")) {
        return new Response(new ReadableStream({
          start(controller) {
            init?.signal?.addEventListener("abort", () => controller.close(), { once: true })
          }
        }), { headers: { "Content-Type": "text/event-stream" } })
      }
      return json({ items: [], total: 0 })
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    process.env = { ...originalEnv }
    clearRuntimeAuth()
  })

  const mount = () => render(
    <NotificationLifecycleProvider><Probe /></NotificationLifecycleProvider>
  )
  const expectState = (state: string) => waitFor(() =>
    expect(screen.getByTestId("lifecycle")).toHaveTextContent(state)
  )
  const rotate = async (channel: "same-tab" | "cross-tab") => {
    if (channel === "same-tab") {
      await storeRefreshRotationIfCurrent(persistent, original, original.refreshToken!, rotated)
      return
    }
    const record = {
      version: 1, ...original,
      sourceAccessToken: original.accessToken,
      sourceRefreshToken: original.refreshToken,
      ...rotated
    }
    const newValue = JSON.stringify(record)
    localStorage.setItem(REFRESH_ROTATION_KEY, newValue)
    window.dispatchEvent(new StorageEvent("storage", { key: REFRESH_ROTATION_KEY, newValue }))
  }

  it("uses matching stored rotation for unread, list, and stream headers", async () => {
    await rotate("same-tab")
    mount()
    await expectState("active:3")
    expect(requests).toHaveLength(3)
    expect(requests.every(({ init }) => bearer(init) === `Bearer ${rotated.accessToken}`)).toBe(true)
    expect(localStorage.getItem("access_token")).toBeNull()
  })

  it.each(["same-tab", "cross-tab"] as const)("recovers an expired notification read after %s rotation", async (channel) => {
    unreadStatus = 401
    mount()
    await expectState("auth-required")
    unreadStatus = 200
    await act(async () => rotate(channel))
    await expectState("active:3")
    expect(bearer(requests.at(-1)?.init)).toBe(`Bearer ${rotated.accessToken}`)
  })

  it.each(["same-tab", "cross-tab"] as const)("keeps permission denial terminal after %s rotation", async (channel) => {
    unreadStatus = 403
    mount()
    await expectState("unavailable")
    await act(async () => rotate(channel))
    expect(screen.getByTestId("lifecycle")).toHaveTextContent("unavailable")
    expect(requests).toHaveLength(1)
  })

  it("does not recover auth-required for another target's rotation", async () => {
    unreadStatus = 401
    mount()
    await expectState("auth-required")
    await act(async () => persistent.set(REFRESH_ROTATION_KEY, {
      version: 1, ...original, serverUrl: "https://other.example.test",
      sourceAccessToken: original.accessToken,
      sourceRefreshToken: original.refreshToken, ...rotated
    }))
    expect(screen.getByTestId("lifecycle")).toHaveTextContent("auth-required")
    expect(requests).toHaveLength(1)
  })

  it("withholds an invalidated effective pair even when legacy bearer stores remain", async () => {
    await rotate("same-tab")
    await invalidateRefreshSessionIfCurrent(persistent, { ...original, ...rotated })
    localStorage.setItem("access_token", "legacy-session-token")
    localStorage.setItem("accessToken", "legacy-api-token")
    expect(buildAuthHeaders()).not.toHaveProperty("Authorization")
  })

  it("stops active private work when its effective pair is invalidated", async () => {
    await rotate("same-tab")
    mount()
    await expectState("active:3")
    await act(async () => {
      await invalidateRefreshSessionIfCurrent(persistent, { ...original, ...rotated })
    })
    await expectState("auth-required:0")
    expect(requests).toHaveLength(3)
  })

  it("uses rotated credentials for mutations without replaying a rejected write", async () => {
    await rotate("same-tab")
    const fetch = vi.fn().mockResolvedValue(json({ detail: "Denied" }, 401))
    vi.stubGlobal("fetch", fetch)
    await expect(markNotificationsRead([1])).rejects.toMatchObject({ status: 401 })
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(bearer(fetch.mock.calls[0][1])).toBe(`Bearer ${rotated.accessToken}`)
  })

  it.each(["account", "server"] as const)("discards a delayed response across an A-to-B-to-A %s change", async (boundary) => {
    let finishOld!: (response: Response) => void
    const transport = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementationOnce(() => new Promise<Response>((resolve) => {
      finishOld = resolve
    })).mockImplementation(transport)
    mount()
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
    const oldSignal = vi.mocked(fetch).mock.calls[0][1]?.signal
    const next = boundary === "account"
      ? { ...original, accessToken: jwt("bob", 0), refreshToken: "refresh-bob" }
      : { ...original, serverUrl: "https://other.example.test" }
    await act(async () => {
      await persistent.set("tldwConfig", next)
      await persistent.set("tldwConfig", original)
    })
    await expectState("active:3")
    await act(async () => finishOld(json({ detail: "Old request denied" }, 401)))
    expect(oldSignal?.aborted).toBe(true)
    expect(screen.getByTestId("lifecycle")).toHaveTextContent("active:3")
    expect(screen.getByTestId("lifecycle")).toHaveAttribute("data-scope", buildWebNotificationScopeKey())
  })

  it("does not let a delayed legacy-session denial sign out current canonical auth", async () => {
    await persistent.remove("tldwConfig")
    localStorage.setItem("access_token", "legacy-session")
    let finishOld!: (response: Response) => void
    vi.mocked(fetch).mockImplementationOnce(() => new Promise<Response>((resolve) => {
      finishOld = resolve
    }))
    const event = vi.fn()
    window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, event)
    try {
      const oldRequest = markNotificationsRead([1])
      const rejected = expect(oldRequest).rejects.toMatchObject({ status: 401 })
      await persistent.set("tldwConfig", original)
      finishOld(json({ detail: "Old session denied" }, 401))
      await rejected
      expect(event).not.toHaveBeenCalled()
      expect(buildAuthHeaders().Authorization).toBe(`Bearer ${original.accessToken}`)
    } finally {
      window.removeEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, event)
    }
  })

  it("never sends another server's credentials to the environment notification endpoint", async () => {
    await persistent.set("tldwConfig", {
      ...original, serverUrl: "https://other.example.test", accessToken: jwt("other-user", 0)
    })
    mount()
    await expectState("unavailable")
    expect(fetch).not.toHaveBeenCalled()
    await expect(markNotificationsRead([1])).rejects.toMatchObject({ status: 412 })
    expect(fetch).not.toHaveBeenCalled()
  })

  it("does not dispatch a notification mutation after canonical session invalidation", async () => {
    await rotate("same-tab")
    await invalidateRefreshSessionIfCurrent(persistent, { ...original, ...rotated })
    await expect(markNotificationsRead([1])).rejects.toMatchObject({ status: 401 })
    expect(fetch).not.toHaveBeenCalled()
  })

  it("preserves the active same-origin cookie session over stale manual configuration", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    api.defaults.baseURL = "/api/v1"
    await persistent.set(COOKIE_SESSION_CONFIG_KEY, {
      serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session"
    })
    localStorage.setItem("access_token", "stale-legacy-session")
    mount()
    await expectState("active:3")
    expect(requests.every(({ init }) => bearer(init) === null && init?.credentials === "include")).toBe(true)
  })

  it("preserves legacy sessions when an inactive cookie config is present", async () => {
    await persistent.remove("tldwConfig")
    await persistent.set(COOKIE_SESSION_CONFIG_KEY, {})
    localStorage.setItem("access_token", "legacy-session")
    mount()
    await expectState("active:3")
    expect(requests.every(({ init }) => bearer(init) === "Bearer legacy-session")).toBe(true)
  })

  it("preserves configured manual API keys", async () => {
    await persistent.set("tldwConfig", {
      serverUrl: original.serverUrl, authMode: "single-user", authSource: "manual", apiKey: "manual-key"
    })
    mount()
    await expectState("active:3")
    expect(requests.every(({ init }) => new Headers(init?.headers).get("X-API-KEY") === "manual-key")).toBe(true)
  })

  it("ignores removal of an ineligible legacy token while canonical authentication is active", async () => {
    await rotate("same-tab")
    mount()
    await expectState("active:3")
    act(() => window.dispatchEvent(new StorageEvent("storage", { key: "access_token", newValue: null })))
    expect(screen.getByTestId("lifecycle")).toHaveTextContent("active:3")
    expect(requests).toHaveLength(3)
  })

  describe.each(["environment", "runtime"] as const)("with a %s bearer override", (source) => {
    const override = jwt("fallback-user", 0)
    beforeEach(() => {
      if (source === "environment") process.env.NEXT_PUBLIC_API_BEARER = override
      else setRuntimeApiBearer(override)
    })

    it("uses the live canonical rotated identity for reads, streams and mutations", async () => {
      await rotate("same-tab")
      mount()
      await expectState("active:3")
      await markNotificationsRead([1])
      expect(requests).toHaveLength(4)
      expect(requests.every(({ init }) => bearer(init) === `Bearer ${rotated.accessToken}`)).toBe(true)
    })

    it("does not substitute another identity after canonical session invalidation", async () => {
      await rotate("same-tab")
      await invalidateRefreshSessionIfCurrent(persistent, { ...original, ...rotated })
      mount()
      await expectState("auth-required:0")
      await expect(markNotificationsRead([1])).rejects.toMatchObject({ status: 401 })
      expect(buildAuthHeaders()).not.toHaveProperty("Authorization")
      expect(fetch).not.toHaveBeenCalled()
    })

    it("retains explicit overrides when no canonical configuration exists", async () => {
      await persistent.remove("tldwConfig")
      mount()
      await expectState("active:3")
      await markNotificationsRead([1])
      expect(requests).toHaveLength(4)
      expect(requests.every(({ init }) => bearer(init) === `Bearer ${override}`)).toBe(true)
    })
  })

  describe.each(["environment", "runtime"] as const)("with a %s API key override", (source) => {
    const override = "fallback-api-key"
    beforeEach(() => {
      if (source === "environment") process.env.NEXT_PUBLIC_X_API_KEY = override
      else setRuntimeApiKey(override)
    })

    it("sends only the canonical multi-user credential on reads, streams and mutations", async () => {
      await rotate("same-tab")
      mount()
      await expectState("active:3")
      await markNotificationsRead([1])
      expect(requests).toHaveLength(4)
      expect(requests.every(({ init }) => bearer(init) === `Bearer ${rotated.accessToken}` &&
        !new Headers(init?.headers).has("X-API-KEY"))).toBe(true)
    })

    it("withholds both credentials after canonical session invalidation", async () => {
      await rotate("same-tab")
      await invalidateRefreshSessionIfCurrent(persistent, { ...original, ...rotated })
      mount()
      await expectState("auth-required:0")
      await expect(markNotificationsRead([1])).rejects.toMatchObject({ status: 401 })
      expect(buildAuthHeaders()).not.toHaveProperty("X-API-KEY")
      expect(buildAuthHeaders()).not.toHaveProperty("Authorization")
      expect(fetch).not.toHaveBeenCalled()
    })

    it.each(["unconfigured", "single-user"] as const)("preserves the API key for %s requests", async (mode) => {
      if (mode === "unconfigured") await persistent.remove("tldwConfig")
      else await persistent.set("tldwConfig", {
        serverUrl: original.serverUrl, authMode: "single-user", authSource: "manual"
      })
      mount()
      await expectState("active:3")
      await markNotificationsRead([1])
      expect(requests).toHaveLength(4)
      expect(requests.every(({ init }) => new Headers(init?.headers).get("X-API-KEY") === override)).toBe(true)
    })
  })
})
