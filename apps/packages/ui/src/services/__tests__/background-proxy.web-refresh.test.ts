import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// This suite exercises the REAL request-core (no tldwRequest mock) through the
// web/direct fallback of background-proxy, to verify token refresh is wired in
// the browser and single-flighted. `runtime` is stubbed with no id/sendMessage
// so bgRequest skips the extension messaging path and uses the direct fallback.
const mocks = vi.hoisted(() => ({
  store: {} as Record<string, unknown>,
  sessionStore: {} as Record<string, unknown>,
  storageGet: vi.fn(),
  sessionStorageGet: vi.fn(),
  storageSet: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    // No runtime.id / sendMessage → hasRuntimeMessage is false → direct fallback.
    runtime: {}
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => ({
    get: (...args: unknown[]) =>
      options?.area === "session"
        ? (mocks.sessionStorageGet as any)(...args)
        : (mocks.storageGet as any)(...args),
    set: (...args: unknown[]) => (mocks.storageSet as any)(...args)
  })
}))

const importProxy = async () => import("@/services/background-proxy")

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

describe("background proxy web token refresh", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useRealTimers()
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    document.cookie = "csrf_token=; Max-Age=0; Path=/"
    mocks.store = {
      tldwConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        accessToken: "stale-access",
        refreshToken: "refresh-token"
      }
    }
    mocks.sessionStore = {}
    mocks.storageGet.mockReset()
    mocks.sessionStorageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageGet.mockImplementation(async (key: string) => mocks.store[key] ?? null)
    mocks.sessionStorageGet.mockImplementation(
      async (key: string) => mocks.sessionStore[key] ?? null
    )
    mocks.storageSet.mockImplementation(async (key: string, value: unknown) => {
      mocks.store[key] = value
    })
  })

  it.each(["POST", "PATCH"])(
    "prefers an active exact-origin cookie marker and adds current CSRF for %s",
    async (method) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      document.cookie = "csrf_token=fresh-csrf; Path=/"
      mocks.store = {
        tldwConfig: {
          serverUrl: "https://remote.example.test",
          authMode: "single-user",
          authSource: "manual",
          apiKey: "preserved-remote-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://remote.example.test"
        },
        tldwCookieSessionConfig: {
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "must-not-leak",
          accessToken: "must-not-leak"
        }
      }
      const fetchSpy = vi.fn(async () =>
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      )
      vi.stubGlobal("fetch", fetchSpy as any)

      const { bgRequest } = await importProxy()
      await bgRequest({
        path: "/api/v1/notes/search/" as unknown as `/${string}`,
        method: method as "POST",
        body: { q: "cookie" },
        preferDirect: true
      })

      const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
      expect(headers.get("X-CSRF-Token")).toBe("fresh-csrf")
      expect(headers.has("X-API-KEY")).toBe(false)
      expect(headers.has("Authorization")).toBe(false)
      expect(mocks.store.tldwConfig).toMatchObject({
        apiKey: "preserved-remote-key"
      })
    }
  )

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    vi.unstubAllGlobals()
  })

  it("refreshes the access token and retries after a 401 in the browser direct path", async () => {
    let refreshHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response(
          JSON.stringify({ access_token: "fresh-access", refresh_token: "rotated-refresh" }),
          { status: 200, headers: { "content-type": "application/json" } }
        )
      }
      const auth = new Headers(init?.headers).get("Authorization") ?? ""
      if (auth === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgRequest } = await importProxy()
    const result = await bgRequest<{ ok: boolean }>({
      path: "/api/v1/notes/search/" as unknown as `/${string}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { q: "hello" }
    })

    expect(result).toEqual({ ok: true })
    expect(refreshHits).toBe(1)
    // Rotated refresh token persisted to storage.
    expect((mocks.store.tldwConfig as Record<string, unknown>).accessToken).toBe(
      "fresh-access"
    )
    expect((mocks.store.tldwConfig as Record<string, unknown>).refreshToken).toBe(
      "rotated-refresh"
    )
  })

  it("signals refresh failure when /auth/refresh returns no access_token (no masking with stale token)", async () => {
    let refreshHits = 0
    let staleRetryHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        // Refresh "succeeds" at the HTTP level but returns no access_token.
        return new Response(JSON.stringify({ detail: "expired" }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const auth = new Headers(init?.headers).get("Authorization") ?? ""
      if (auth === "Bearer stale-access") {
        staleRetryHits += 1
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgRequest } = await importProxy()

    // Because refreshAuthDirect signals failure, request-core marks the refresh
    // as failed and the still-401 retry surfaces "Session expired" rather than
    // resolving as if the (stale-token) retry had succeeded.
    await expect(
      bgRequest<{ ok: boolean }>({
        path: "/api/v1/notes/search/" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { q: "hello" }
      })
    ).rejects.toThrow(/session expired/i)

    expect(refreshHits).toBe(1)
    // The retry ran with the stale token and 401'd; it must NOT have been
    // treated as a success, and no bogus token was persisted.
    expect(staleRetryHits).toBeGreaterThanOrEqual(1)
    expect((mocks.store.tldwConfig as Record<string, unknown>).accessToken).toBe(
      "stale-access"
    )
  })

  it("signals refresh failure when /auth/refresh itself returns 401", async () => {
    let refreshHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response("unauthorized", { status: 401 })
      }
      const auth = new Headers(init?.headers).get("Authorization") ?? ""
      if (auth === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest<{ ok: boolean }>({
        path: "/api/v1/notes/search/" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { q: "hello" }
      })
    ).rejects.toThrow(/session expired/i)

    expect(refreshHits).toBe(1)
    expect((mocks.store.tldwConfig as Record<string, unknown>).accessToken).toBe(
      "stale-access"
    )
  })

  it("single-flights concurrent 401 refreshes into one refresh call", async () => {
    let refreshHits = 0
    let releaseRefresh: () => void = () => {}
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        // Hold the refresh open so both callers are parked on the shared
        // in-flight promise before it resolves.
        await refreshGate
        return new Response(
          JSON.stringify({ access_token: "fresh-access", refresh_token: "rotated-refresh" }),
          { status: 200, headers: { "content-type": "application/json" } }
        )
      }
      const auth = new Headers(init?.headers).get("Authorization") ?? ""
      if (auth === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgRequest } = await importProxy()
    const call = () =>
      bgRequest<{ ok: boolean }>({
        path: "/api/v1/notes/search/" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { q: "hello" }
      })

    const a = call()
    const b = call()

    // Let both requests reach the (gated) refresh before releasing it.
    await new Promise((resolve) => setTimeout(resolve, 20))
    releaseRefresh()

    const [ra, rb] = await Promise.all([a, b])
    expect(ra).toEqual({ ok: true })
    expect(rb).toEqual({ ok: true })
    // Two concurrent 401s → exactly ONE refresh network call.
    expect(refreshHits).toBe(1)
  })
})
