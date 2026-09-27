import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  watchers: new Set<Record<string, () => void>>(),
  fetch: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(),
    remove: vi.fn(),
    watch: (watchers: Record<string, () => void>) =>
      boundary.watchers.add(watchers),
    unwatch: (watchers: Record<string, () => void>) =>
      boundary.watchers.delete(watchers)
  }),
  safeStorageSerde: { serialize: JSON.stringify, deserialize: JSON.parse }
}))
vi.mock("@/store/connection", async () => {
  const { create } = await import("zustand")
  return {
    useConnectionStore: create(() => ({
      state: { isConnected: true, mode: "normal" }
    }))
  }
})
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null
}))

import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { useConnectionStore } from "@/store/connection"
import { useCallerCapabilities } from "../useCallerCapabilities"

const token = (id: number, revision = "one") =>
  `test.${btoa(JSON.stringify({ sub: String(id), revision }))}.signature`
const capabilities = (id = 7, allowed = true) => ({
  user_id: id,
  can_read_scheduled_tasks: allowed,
  can_read_notifications: false,
  can_read_monitoring_alerts: allowed
})
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" }
  })
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
let config: TldwConfig
const online = (connected: boolean) =>
  useConnectionStore.setState(({ state }) => ({
    state: { ...state, isConnected: connected }
  }))
const authorityChanged = () =>
  window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed"))

describe("caller capabilities through the real client and request transport", () => {
  beforeEach(() => {
    config = {
      serverUrl: window.location.origin,
      authMode: "multi-user",
      accessToken: token(7),
      orgId: 1
    }
    boundary.fetch
      .mockReset()
      .mockImplementation(async () => response(capabilities()))
    boundary.watchers.clear()
    vi.stubGlobal("fetch", boundary.fetch)
    vi.spyOn(tldwClient, "initialize").mockResolvedValue()
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(
      async () => ({ ...config })
    )
    online(true)
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it("keeps pending access unknown and derives allowed/denied from the authenticated self response", async () => {
    const pending = deferred<Response>()
    boundary.fetch.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(useCallerCapabilities)
    expect(result.current.scheduledTasks).toBe("unknown")
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(url).toBe(`${window.location.origin}/api/v1/users/me/capabilities`)
    expect(init.headers).toMatchObject({
      Authorization: `Bearer ${token(7)}`,
      "X-TLDW-Org-Id": "1"
    })
    await act(async () => pending.resolve(response(capabilities())))
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    expect(result.current.notifications).toBe("denied")
    expect(result.current.monitoringAlerts).toBe("allowed")
    expect(result.current.scopeKey).toContain("user:7")
    expect(result.current.userId).toBe(7)
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })

  it.each([401, 403, 410, 500, 503])(
    "keeps discovery HTTP %s unknown, rather than declaring permissions denied",
    async (status) => {
      boundary.fetch.mockResolvedValue(
        response({ detail: "unavailable" }, status)
      )
      const { result } = renderHook(useCallerCapabilities)
      await waitFor(() => expect(result.current.loading).toBe(false))
      expect(result.current.scheduledTasks).toBe("unknown")
      expect(result.current.notifications).toBe("unknown")
      expect(result.current.scopeKey).toBeNull()
      expect(result.current.userId).toBeNull()
      expect(boundary.fetch).toHaveBeenCalledTimes(1)
    }
  )

  it.each([404, 405, 501])(
    "reports an unsupported discovery endpoint for HTTP %s",
    async (status) => {
      boundary.fetch.mockResolvedValue(response({}, status))
      const { result } = renderHook(useCallerCapabilities)
      await waitFor(() =>
        expect(result.current.scheduledTasks).toBe("unsupported")
      )
    }
  )

  it.each([
    { ...capabilities(), can_read_notifications: "false" },
    { ...capabilities(), user_id: null },
    capabilities(8)
  ])("rejects malformed or mismatched caller responses: %j", async (body) => {
    boundary.fetch.mockResolvedValue(response(body))
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.scheduledTasks).toBe("unknown")
    expect(result.current.scopeKey).toBeNull()
    expect(result.current.userId).toBeNull()
  })

  it("uses verified response identity for cookie sessions without calling profile or auth/me", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    }
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    expect(result.current.scopeKey).toContain("user:7")
    expect(boundary.fetch.mock.calls[0][1].credentials).toBe("same-origin")
    expect(boundary.fetch.mock.calls[0][1].headers).not.toHaveProperty(
      "Authorization"
    )
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })

  it("uses hosted cookie identity even when an unrelated legacy bearer remains configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    config = { ...config, accessToken: token(99) }
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    expect(result.current.scopeKey).toContain("user:7")
    expect(boundary.fetch.mock.calls[0][1].headers).not.toHaveProperty(
      "Authorization"
    )
  })

  it("authenticates single-user API keys and binds decisions to the returned owner", async () => {
    config = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      apiKey: "synthetic-capability-key"
    }
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    expect(result.current.scopeKey).toContain("user:7")
    expect(boundary.fetch.mock.calls[0][1].headers["X-API-KEY"]).toBe(
      "synthetic-capability-key"
    )
    expect(result.current.scopeKey).not.toContain(config.apiKey)
  })

  it("masks prior decisions while refreshing and after a failed refresh", async () => {
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    const pending = deferred<Response>()
    boundary.fetch.mockReturnValueOnce(pending.promise)
    act(() => {
      void result.current.refresh()
    })
    expect(result.current.scheduledTasks).toBe("unknown")
    expect(result.current.scopeKey).toBeNull()
    expect(result.current.userId).toBeNull()
    await act(async () => pending.resolve(response({}, 403)))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.scheduledTasks).toBe("unknown")
    expect(result.current.userId).toBeNull()
  })

  it("masks on disconnect and fetches fresh decisions on reconnect", async () => {
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    act(() => online(false))
    expect(result.current.scheduledTasks).toBe("unknown")
    expect(result.current.userId).toBeNull()
    expect(result.current.scopeKey).toBeNull()
    boundary.fetch.mockResolvedValue(response(capabilities(7, false)))
    act(() => online(true))
    await waitFor(() => expect(result.current.scheduledTasks).toBe("denied"))
    expect(result.current.userId).toBe(7)
  })

  it("rejects delayed A-to-B-to-A results even when the original request ignores abort", async () => {
    const old = deferred<Response>()
    boundary.fetch.mockReturnValueOnce(old.promise)
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    config = { ...config, accessToken: token(8) }
    boundary.fetch.mockResolvedValue(response(capabilities(8, false)))
    act(authorityChanged)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("denied"))
    expect(result.current.scopeKey).toContain("user:8")
    expect(result.current.userId).toBe(8)
    config = { ...config, accessToken: token(7) }
    boundary.fetch.mockResolvedValue(response(capabilities(7, false)))
    act(authorityChanged)
    await waitFor(() => expect(result.current.scopeKey).toContain("user:7"))
    await act(async () => old.resolve(response(capabilities(7, true))))
    expect(result.current.scheduledTasks).toBe("denied")
    expect(result.current.userId).toBe(7)
  })

  it("rejects a delayed effective-config read after logout", async () => {
    const oldConfig = deferred<TldwConfig>()
    vi.mocked(tldwClient.ensureConfigForRequest).mockReturnValueOnce(
      oldConfig.promise
    )
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() =>
      expect(tldwClient.ensureConfigForRequest).toHaveBeenCalled()
    )
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" }
        })
      )
    )
    await act(async () => oldConfig.resolve(config))
    expect(result.current.scheduledTasks).toBe("unknown")
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it("rechecks authority after a response even when no change event was delivered", async () => {
    const old = deferred<Response>()
    boundary.fetch.mockReturnValueOnce(old.promise)
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    config = { ...config, orgId: 2 }
    await act(async () => old.resolve(response(capabilities())))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.scheduledTasks).toBe("unknown")
  })

  it.each(["server", "credential"])(
    "rejects a delayed response after a silent %s replacement",
    async (replacement) => {
      const old = deferred<Response>()
      boundary.fetch.mockReturnValueOnce(old.promise)
      const { result } = renderHook(useCallerCapabilities)
      await waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
      config =
        replacement === "server"
          ? { ...config, serverUrl: "https://replacement.test" }
          : { ...config, accessToken: token(7, "replacement") }
      await act(async () => old.resolve(response(capabilities())))
      await waitFor(() => expect(result.current.loading).toBe(false))
      expect(result.current.scheduledTasks).toBe("unknown")
      expect(result.current.scopeKey).toBeNull()
    }
  )

  it("ignores an old account's protected 403 after the replacement account is ready", async () => {
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    const reportFromA = result.current.refreshAfterForbidden
    config = { ...config, accessToken: token(8) }
    boundary.fetch.mockResolvedValue(response(capabilities(8, false)))
    act(authorityChanged)
    await waitFor(() => expect(result.current.scopeKey).toContain("user:8"))
    const calls = boundary.fetch.mock.calls.length
    await act(async () => reportFromA({ status: 403 }))
    expect(boundary.fetch).toHaveBeenCalledTimes(calls)
    expect(result.current.scheduledTasks).toBe("denied")
  })

  it("invalidates on a cross-tab effective credential rotation notification", async () => {
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    config = { ...config, accessToken: token(7, "two") }
    boundary.fetch.mockResolvedValue(response(capabilities(7, false)))
    act(() =>
      boundary.watchers.forEach((watch) => watch.tldwRefreshRotation?.())
    )
    await waitFor(() => expect(result.current.scheduledTasks).toBe("denied"))
  })

  it("refreshes after a current protected 403, ignores other errors and old-account reports", async () => {
    const { result } = renderHook(useCallerCapabilities)
    await waitFor(() => expect(result.current.scheduledTasks).toBe("allowed"))
    const oldReport = result.current.refreshAfterForbidden
    await act(async () => result.current.refreshAfterForbidden({ status: 500 }))
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    boundary.fetch.mockResolvedValue(response(capabilities(7, false)))
    await act(async () => result.current.refreshAfterForbidden({ status: 403 }))
    await waitFor(() => expect(result.current.scheduledTasks).toBe("denied"))
    const calls = boundary.fetch.mock.calls.length
    await act(async () => oldReport({ status: 403 }))
    expect(boundary.fetch).toHaveBeenCalledTimes(calls)
  })
})
