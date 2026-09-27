import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  fetch: vi.fn(),
  get: vi.fn(),
  watchers: new Set<Record<string, (change?: unknown) => void>>()
}))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(), remove: vi.fn(),
    watch: (handlers: Record<string, () => void>) => boundary.watchers.add(handlers),
    unwatch: (handlers: Record<string, () => void>) => boundary.watchers.delete(handlers)
  }),
  safeStorageSerde: { serialize: JSON.stringify, deserialize: JSON.parse }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))
vi.mock("@/store/connection", async () => {
  const { create } = await import("zustand")
  return { useConnectionStore: create(() => ({ state: { isConnected: true, mode: "normal" } })) }
})

import { useScheduledTaskHomeSignals } from "../hooks"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"

const token = (id: number) => `test.${btoa(JSON.stringify({ sub: String(id) }))}.signature`
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" }
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
let config: TldwConfig
let userId: number
let taskAccess: boolean
let notificationAccess: boolean
let discoveryStatus: number
const notification = (title = "Alice automation") => ({
  id: 91, kind: "job_completed", title, message: "Output ready", severity: "info",
  created_at: "2030-01-01T09:05:00Z", link_type: "scheduled_task_result", link_id: "202"
})
const protectedCalls = () => boundary.fetch.mock.calls.filter(([url]) =>
  /\/api\/(?:v1|proxy)\/(scheduled-tasks|notifications)(?:\?|$|\/)/.test(String(url)))
const changed = () => window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed"))

describe("Home automation through caller discovery, verified scope and protected transport", () => {
  beforeEach(() => {
    userId = 7
    taskAccess = true
    notificationAccess = true
    discoveryStatus = 200
    config = { serverUrl: window.location.origin, authMode: "multi-user", accessToken: token(userId) }
    boundary.watchers.clear()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? { ...config } : null)
    boundary.fetch.mockReset().mockImplementation(async (url: string) => {
      const path = new URL(String(url), window.location.origin).pathname
      if (path.endsWith("/users/me/capabilities")) return response({
        user_id: userId, can_read_scheduled_tasks: taskAccess,
        can_read_notifications: notificationAccess, can_read_monitoring_alerts: false
      }, discoveryStatus)
      if (path.endsWith("/auth/me")) return response({ id: userId, is_active: true })
      if (path === "/api/auth/session") return response({ authenticated: true, user: { id: userId, is_active: true } })
      return response({ items: [], total: 0, partial: false, errors: [] })
    })
    vi.stubGlobal("fetch", boundary.fetch)
    vi.spyOn(tldwClient, "initialize").mockResolvedValue()
    vi.spyOn(tldwClient, "getConfig").mockImplementation(async () => ({ ...config }))
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(async () => ({ ...config }))
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs() })

  it("does not issue any known-denied reads or describe denial as an outage", async () => {
    taskAccess = notificationAccess = false
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(protectedCalls()).toHaveLength(0)
    expect(view.result.current.sourceStates).toEqual({ tasks: "denied", results: "denied", notifications: "denied" })
    expect(view.result.current.error).toBeNull()
  })

  it("keeps permitted notifications while skipping both task reads", async () => {
    taskAccess = false
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).includes("/notifications?")
      ? Promise.resolve(response({ items: [notification()], total: 1 })) : base(url, init))
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.items[0]?.title).toBe("Alice automation"))
    expect(protectedCalls()).toHaveLength(1)
    expect(view.result.current.sourceStates).toEqual({ tasks: "denied", results: "denied", notifications: "ready" })
    const [, init] = protectedCalls()[0]
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
  })

  it("uses normal protected reads when discovery is unsupported, classifying their actual responses", async () => {
    discoveryStatus = 404
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).includes("/scheduled-tasks")
      ? Promise.resolve(response({ detail: "tasks.read required" }, 403)) : base(url, init))
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(protectedCalls()).toHaveLength(3)
    expect(view.result.current.sourceStates).toEqual({ tasks: "denied", results: "denied", notifications: "ready" })
    expect(protectedCalls().every(([, init]) => new Headers(init.headers).get("X-TLDW-Expected-User-ID") === "7")).toBe(true)
  })

  it("does not infer denial from discovery403 and distinguishes unsupported endpoints from failure", async () => {
    discoveryStatus = 403
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).includes("/scheduled-tasks/results")
      ? Promise.resolve(response({}, 404)) : String(url).includes("/scheduled-tasks")
        ? Promise.resolve(response({}, 503)) : base(url, init))
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(view.result.current.sourceStates).toEqual({ tasks: "error", results: "unsupported", notifications: "ready" })
    expect(view.result.current.error).toBeTruthy()
  })

  it("refreshes caller permissions before retrying previously denied sources", async () => {
    taskAccess = notificationAccess = false
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    taskAccess = notificationAccess = true
    act(() => view.result.current.refresh())
    await waitFor(() => expect(view.result.current.sourceStates).toEqual({ tasks: "ready", results: "ready", notifications: "ready" }))
    expect(protectedCalls()).toHaveLength(3)
  })

  it("supports a single-user scope without requiring a multi-user identity lookup", async () => {
    config = { serverUrl: window.location.origin, authMode: "single-user", apiKey: "synthetic-home-key",
      credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: window.location.origin }
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.sourceStates).toEqual({ tasks: "ready", results: "ready", notifications: "ready" }))
    expect(boundary.fetch.mock.calls.some(([url]) => String(url).endsWith("/auth/me"))).toBe(false)
    expect(protectedCalls().every(([, init]) => new Headers(init.headers).get("X-API-KEY") === "synthetic-home-key")).toBe(true)
  })

  it("rejects a changed single-user credential between discovery and the scope lease", async () => {
    config = { serverUrl: window.location.origin, authMode: "single-user", apiKey: "synthetic-key-A",
      credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: window.location.origin }
    let configReads = 0
    vi.mocked(tldwClient.ensureConfigForRequest).mockImplementation(async () => {
      if (++configReads >= 3) config = { ...config, apiKey: "synthetic-key-B" }
      return { ...config }
    })
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(protectedCalls()).toHaveLength(0)
    expect(view.result.current.sourceStates).toEqual({ tasks: "unknown", results: "unknown", notifications: "unknown" })
  })

  it("binds cookie-session reads to the independently verified user", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    config = { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.sourceStates).toEqual({ tasks: "ready", results: "ready", notifications: "ready" }))
    expect(protectedCalls()).toHaveLength(3)
    expect(protectedCalls().every(([, init]) => new Headers(init.headers).get("X-TLDW-Expected-User-ID") === "7")).toBe(true)
  })

  it("retains discovery's verified user guard for a quickstart cookie session", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
    boundary.get.mockImplementation(async key => key === "tldwCookieSessionConfig" || key === "tldwConfig" ? { ...config } : null)
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.sourceStates).toEqual({ tasks: "ready", results: "ready", notifications: "ready" }))
    expect(protectedCalls()).toHaveLength(3)
    expect(protectedCalls().every(([, init]) => new Headers(init.headers).get("X-TLDW-Expected-User-ID") === "7")).toBe(true)
    expect(protectedCalls().every(([, init]) => init.credentials === "same-origin")).toBe(true)
  })

  it("does not read protected sources when cookie identity disagrees with discovery", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    config = { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).endsWith("/api/auth/session")
      ? Promise.resolve(response({ authenticated: true, user: { id: 8, is_active: true } })) : base(url, init))
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(view.result.current.sourceStates).toEqual({ tasks: "unknown", results: "unknown", notifications: "unknown" })
    expect(protectedCalls()).toHaveLength(0)
  })

  it.each([401, 410])("does not claim denial or send unowned fallback reads when legacy identity returns %i", async status => {
    discoveryStatus = 404
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).endsWith("/auth/me")
      ? Promise.resolve(response({}, status)) : base(url, init))
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    expect(view.result.current.sourceStates).toEqual({ tasks: "unknown", results: "unknown", notifications: "unknown" })
    expect(protectedCalls()).toHaveLength(0)
  })

  it("masks already displayed private notifications immediately on disable", async () => {
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).includes("/notifications?")
      ? Promise.resolve(response({ items: [notification()], total: 1 })) : base(url, init))
    const view = renderHook(({ enabled }) => useScheduledTaskHomeSignals({ enabled }), { initialProps: { enabled: true } })
    await waitFor(() => expect(view.result.current.items[0]?.title).toBe("Alice automation"))
    view.rerender({ enabled: false })
    expect(view.result.current.items).toEqual([])
  })

  it("clears displayed items on disable and aborts a pending read without accepting its late result", async () => {
    const pending = deferred<Response>()
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => String(url).includes("/notifications?")
      ? pending.promise : base(url, init))
    const view = renderHook(({ enabled }) => useScheduledTaskHomeSignals({ enabled }), { initialProps: { enabled: true } })
    await waitFor(() => expect(protectedCalls()).toHaveLength(3))
    const [, init] = protectedCalls().find(([url]) => String(url).includes("/notifications?"))!
    view.rerender({ enabled: false })
    expect(view.result.current.items).toEqual([])
    expect(view.result.current.loading).toBe(false)
    expect(init.signal.aborted).toBe(true)
    await act(async () => pending.resolve(response({ items: [notification()], total: 1 })))
    expect(view.result.current.items).toEqual([])
  })

  it("rejects old results across A→B→A even when the final principal matches", async () => {
    const pending = deferred<Response>()
    let first = true
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url: string, init: RequestInit) => {
      if (String(url).includes("/notifications?")) {
        if (first) { first = false; return pending.promise }
        return Promise.resolve(response({ items: [notification(`Current ${userId}`)], total: 1 }))
      }
      return base(url, init)
    })
    const view = renderHook(() => useScheduledTaskHomeSignals({ enabled: true }))
    await waitFor(() => expect(protectedCalls()).toHaveLength(3))
    act(() => { userId = 8; config = { ...config, accessToken: token(8) }; changed() })
    expect(view.result.current.items).toEqual([])
    await waitFor(() => expect(view.result.current.items[0]?.title).toBe("Current 8"))
    act(() => { userId = 7; config = { ...config, accessToken: token(7) }; changed() })
    expect(view.result.current.items).toEqual([])
    await waitFor(() => expect(view.result.current.items[0]?.title).toBe("Current 7"))
    await act(async () => pending.resolve(response({ items: [notification("Old Alice")], total: 1 })))
    expect(view.result.current.items[0]?.title).toBe("Current 7")
  })
})
