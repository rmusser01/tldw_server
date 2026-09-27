import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const fixture = vi.hoisted(() => ({
  values: {} as Record<string, unknown>,
  beforeRead: null as null | ((key: string) => Promise<void>)
}))
vi.mock("wxt/browser", () => ({ browser: {
  runtime: {}, storage: { local: { get: async () => ({}) }, onChanged: { addListener: () => undefined } }
} }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: (options?: { area?: string }) => ({
  get: async <T,>(key: string): Promise<T | null> => {
    if (options?.area === "session") return null
    const value = fixture.values[key] as T ?? null
    await fixture.beforeRead?.(key)
    return value
  },
  set: async (key: string, value: unknown) => { fixture.values[key] = value },
  remove: async (key: string) => { delete fixture.values[key] }
}) }))
vi.mock("@/services/tldw-server-url", () => ({ getStoredTldwServerURL: async () => (fixture.values.tldwConfig as TldwConfig)?.serverUrl }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => undefined,
  getConfig: async () => {
    const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
    const { createSafeStorage } = await import("@/utils/safe-storage")
    return resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))
  },
  updateConfig: async (next: Partial<TldwConfig>) => { fixture.values.tldwConfig = { ...(fixture.values.tldwConfig as TldwConfig), ...next } },
  ragHealth: async () => ({ status: "ok" })
} }))

import { apiSend } from "@/services/api-send"
import { useConnectionStore } from "@/store/connection"
import { createSafeStorage } from "@/utils/safe-storage"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import { ConnectionPhase } from "@/types/connection"
import { REFRESH_SESSION_INVALIDATION_PREFIX } from "@/services/tldw/single-user-credential"

const jwt = (user: string, generation = 0) => `test.${btoa(JSON.stringify({ sub: user, generation }))}.signature`
const config = (user = "Alice", serverUrl = "https://auth.test"): TldwConfig => ({
  serverUrl, authMode: "multi-user", accessToken: jwt(user), refreshToken: `synthetic-${user}-refresh`,
})
const response = (status: number, data: unknown = {}) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } })
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { resolve, promise }
}
const baseline = useConnectionStore.getState().state
const currentConfig = () => resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))

beforeEach(() => {
  fixture.values = { tldwConfig: config() }
  fixture.beforeRead = null
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  useConnectionStore.setState({ state: { ...baseline, serverUrl: "https://auth.test", phase: ConnectionPhase.UNCONFIGURED, lastCheckedAt: null, isChecking: false } })
})
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe("connection readiness through actual apiSend and canonical refresh", () => {
  it("rotates successive expired pairs and projects connected from the actual readiness GET", async () => {
    let generation = 0
    let expectedAccess = "expired"
    const refreshBodies: unknown[] = []
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/auth/refresh")) {
        refreshBodies.push(JSON.parse(String(init?.body)))
        generation += 1
        expectedAccess = jwt("Alice", generation)
        return response(200, { access_token: expectedAccess, refresh_token: `rotated-${generation}` })
      }
      return response(new Headers(init?.headers).get("Authorization") === `Bearer ${expectedAccess}` ? 200 : 401)
    }))
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expectedAccess = "expired-again"
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expect(refreshBodies).toEqual([{ refresh_token: "synthetic-Alice-refresh" }, { refresh_token: "rotated-1" }])
    expect(await currentConfig()).toMatchObject({ accessToken: jwt("Alice", 2), refreshToken: "rotated-2" })
    expect(fixture.values.tldwConfig).toEqual(config())
  })

  it("terminal refresh401 invalidates the pair and later readiness checks do not poll it", async () => {
    const fetchSpy = vi.fn(async () => response(401))
    vi.stubGlobal("fetch", fetchSpy)
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(await currentConfig()).not.toHaveProperty("accessToken")
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
  })

  it.each(["target", "owner", "A-B-A"])("rejects a delayed readiness refresh after %s changes without writing rotated authority", async change => {
    const entered = deferred<void>(), release = deferred<Response>()
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) { entered.resolve(); return release.promise }
      return response(401)
    })
    vi.stubGlobal("fetch", fetchSpy)
    const pending = useConnectionStore.getState().checkOnce({ force: true })
    await entered.promise
    await useConnectionStore.getState().setConfigPartial(change === "target" ? config("Alice", "https://other.test") : config("Bob"))
    if (change === "A-B-A") await useConnectionStore.getState().setConfigPartial(config())
    release.resolve(response(200, { access_token: jwt("Alice", 1), refresh_token: "old-operation-rotation" }))
    await pending
    expect(fixture.values.tldwRefreshRotation).toBeUndefined()
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
    expect(fetchSpy.mock.calls.filter(([url]) => !String(url).endsWith("/auth/refresh"))).toHaveLength(1)
  })

  it("shares one canonical refresh with a concurrent scoped request", async () => {
    const release = deferred<Response>()
    let refreshCalls = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/auth/refresh")) { refreshCalls += 1; return release.promise }
      return response(new Headers(init?.headers).get("Authorization") === `Bearer ${jwt("Alice", 1)}` ? 200 : 401)
    })
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await import("@/services/background-proxy")
    const readiness = useConnectionStore.getState().checkOnce({ force: true })
    const scopedRead = bgRequest({ path: "/api/v1/notes/private-note", method: "GET", servicePromptConfig: { ...config(), expectedUserId: "Alice" } })
    await vi.waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(3))
    release.resolve(response(200, { access_token: jwt("Alice", 1), refresh_token: "rotated-once" }))
    await Promise.all([readiness, scopedRead])
    expect(refreshCalls).toBe(1)
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expect(await currentConfig()).toMatchObject({ refreshToken: "rotated-once" })
  })

  it("does not refresh a late401 from an account that has been replaced", async () => {
    const entered = deferred<void>(), release = deferred<Response>()
    const fetchSpy = vi.fn(async () => { entered.resolve(); return release.promise })
    vi.stubGlobal("fetch", fetchSpy)
    const pending = useConnectionStore.getState().checkOnce({ force: true })
    await entered.promise
    await useConnectionStore.getState().setConfigPartial(config("Bob", "https://other.test"))
    release.resolve(response(401))
    await pending
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fixture.values.tldwRefreshRotation).toBeUndefined()
    expect(await currentConfig()).toEqual(config("Bob", "https://other.test"))
  })

  it("fails closed when authority changes during a marker read even if the resolver catches its error", async () => {
    const entered = deferred<void>(), release = deferred<void>()
    let current = true
    fixture.beforeRead = async key => {
      if (!key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)) return
      fixture.beforeRead = null
      entered.resolve()
      await release.promise
    }
    const fetchSpy = vi.fn(async () => response(200))
    vi.stubGlobal("fetch", fetchSpy)
    const result = apiSend({ path: "/api/v1/auth/sessions", method: "GET" }, {
      readiness: { config: config(), isCurrent: () => current }
    }).catch(error => error)
    await entered.promise
    current = false
    release.resolve()
    expect(await result).toMatchObject({ status: 412, details: { detail: { code: "request_config_scope_changed" } } })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("keeps the current pair after a retryable refresh503 and does not replay the GET", async () => {
    const fetchSpy = vi.fn(async (url: RequestInfo | URL) => response(String(url).endsWith("/auth/refresh") ? 503 : 401))
    vi.stubGlobal("fetch", fetchSpy)
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(await currentConfig()).toMatchObject(config())
    expect(useConnectionStore.getState().state).toMatchObject({ isConnected: false, lastStatusCode: 503 })
  })

  it.each(["valid bearer", "single-user", "hosted cookies"])("keeps the %s readiness transport without refresh", async mode => {
    if (mode === "single-user") fixture.values.tldwConfig = {
      serverUrl: "https://auth.test", authMode: "single-user", apiKey: "synthetic-key",
      credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: "https://auth.test"
    }
    if (mode === "hosted cookies") {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
      fixture.values.tldwConfig = { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }
    }
    const fetchSpy = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response(200))
    vi.stubGlobal("fetch", fetchSpy)
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit | undefined
    if (mode === "single-user") expect(new Headers(init?.headers).get("X-API-KEY")).toBe("synthetic-key")
    if (mode === "hosted cookies") {
      // Fetch defaults to same-origin cookies when credentials is omitted.
      expect(init?.credentials ?? "same-origin").toBe("same-origin")
      expect(String(fetchSpy.mock.calls[0]?.[0])).toMatch(/^\//)
      expect(new Headers(init?.headers).has("Authorization")).toBe(false)
    }
  })

  it.each([403, 0])("retains credentials without refreshing for non-auth failure %s", async status => {
    const fetchSpy = vi.fn(async () => { if (!status) throw new TypeError("Failed to fetch"); return response(status) })
    vi.stubGlobal("fetch", fetchSpy)
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(await currentConfig()).toMatchObject(config())
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
  })

  it("does not add refresh or replay to an ordinary direct mutation", async () => {
    const fetchSpy = vi.fn(async () => response(401))
    vi.stubGlobal("fetch", fetchSpy)
    expect(await apiSend({ path: "/api/v1/prompts/", method: "POST", body: { name: "private draft" } }, {
      readiness: { config: config(), isCurrent: () => true }
    })).toMatchObject({ status: 401 })
    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })
})
