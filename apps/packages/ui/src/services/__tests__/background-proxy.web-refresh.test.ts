import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

// This suite exercises the REAL request-core (no tldwRequest mock) through the
// web/direct fallback of background-proxy, to verify token refresh is wired in
// the browser and single-flighted. `runtime` is stubbed with no id/sendMessage
// so bgRequest skips the extension messaging path and uses the direct fallback.
const mocks = vi.hoisted(() => ({
  store: {} as Record<string, unknown>,
  sessionStore: {} as Record<string, unknown>,
  storageGet: vi.fn(),
  sessionStorageGet: vi.fn(),
  storageSet: vi.fn(),
  storageRemove: vi.fn(),
  runtimeApiKey: null as string | null,
  nextRuntimeApiKey: null as string | null,
  runtimeApiKeyReads: 0
}))

vi.mock("wxt/browser", () => ({
  browser: {
    // No runtime.id / sendMessage → hasRuntimeMessage is false → direct fallback.
    runtime: {}
  }
}))

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/utils/safe-storage")>(),
  createSafeStorage: (options?: { area?: string }) => ({
    get: (...args: unknown[]) =>
      options?.area === "session"
        ? (mocks.sessionStorageGet as any)(...args)
        : (mocks.storageGet as any)(...args),
    set: (...args: unknown[]) => (mocks.storageSet as any)(...args),
    remove: (...args: unknown[]) => (mocks.storageRemove as any)(...args)
  })
}))

vi.mock("@/services/tldw/runtime-auth-override", async (importOriginal) => ({
  ...await importOriginal<
    typeof import("@/services/tldw/runtime-auth-override")
  >(),
  getRuntimeSingleUserApiKeyOverride: () => {
    const value = mocks.runtimeApiKeyReads > 0 && mocks.nextRuntimeApiKey !== null
      ? mocks.nextRuntimeApiKey
      : mocks.runtimeApiKey
    mocks.runtimeApiKeyReads += 1
    return value
  }
}))

const importProxy = async () => import("@/services/background-proxy")

const collectStream = async (stream: AsyncGenerator<string>) => {
  const chunks: string[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const jwtForUser = (userId: string | number): string =>
  `header.${btoa(JSON.stringify({ sub: String(userId) }))}.signature`

describe("background proxy web token refresh", () => {
  it.each([false, true])("invalidates a revoked rotated session without replaying its stale raw JWT (scoped=%s)", async (scoped) => {
    const config = {
      serverUrl: "https://api.example.com", authMode: "multi-user" as const,
      accessToken: jwtForUser(42), refreshToken: "original-refresh"
    }
    mocks.store.tldwConfig = config
    const { createSafeStorage } = await import("@/utils/safe-storage")
    const { storeRefreshRotationIfCurrent } = await import("@/services/tldw/single-user-credential")
    const storage = createSafeStorage({ area: "local" })
    await storeRefreshRotationIfCurrent(storage, config, config.refreshToken, {
      accessToken: `${jwtForUser(42)}-rotated`, refreshToken: "rotated-refresh"
    })
    const fetchSpy = vi.fn<typeof fetch>(async () => new Response("unauthorized", { status: 401 }))
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/private-note", method: "GET",
      ...(scoped ? { servicePromptConfig: { serverUrl: config.serverUrl, authMode: config.authMode, expectedUserId: 42 } } : {})
    })).rejects.toMatchObject({ status: 401 })
    const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
    expect(await resolveDirectBrowserConfig(storage)).not.toHaveProperty("accessToken")
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(mocks.store.tldwConfig).toEqual(config)
  })

  it.each([false, true])("invalidates only the current session after an actual refresh401 (scoped=%s)", async (scoped) => {
    const config = {
      serverUrl: "https://api.example.com", authMode: "multi-user" as const,
      accessToken: jwtForUser(42), refreshToken: "expired-refresh"
    }
    mocks.store.tldwConfig = config
    vi.stubGlobal("fetch", vi.fn(async () => new Response("unauthorized", { status: 401 })))
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/private-note", method: "GET",
      ...(scoped ? { servicePromptConfig: { serverUrl: config.serverUrl, authMode: config.authMode, expectedUserId: 42 } } : {})
    })).rejects.toMatchObject({ status: 401 })
    const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
    const { createSafeStorage } = await import("@/utils/safe-storage")
    expect(await resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))).not.toHaveProperty("accessToken")
    expect(mocks.store.tldwConfig).toEqual(config)
  })

  it("keeps cancellation during refresh out of the backend-unreachable modal", async () => {
    let started!: () => void
    let release!: () => void
    const entered = new Promise<void>(resolve => { started = resolve })
    const gate = new Promise<void>(resolve => { release = resolve })
    const events: Event[] = []
    const listener = (event: Event) => { events.push(event) }
    window.addEventListener("tldw:backend-unreachable", listener)
    const abort = new AbortController()
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) {
        started()
        await gate
        return new Response(JSON.stringify({ access_token: "fresh-access", refresh_token: "fresh-refresh" }), {
          status: 200, headers: { "content-type": "application/json" }
        })
      }
      return new Response("unauthorized", { status: 401 })
    }))
    try {
      const { bgRequest } = await importProxy()
      const rejected = expect(bgRequest({
        path: "/api/v1/notes/", method: "GET", abortSignal: abort.signal
      })).rejects.toMatchObject({ status: 0, code: "REQUEST_ABORTED" })
      await entered
      abort.abort()
      release()
      await rejected
      expect(events).toHaveLength(0)
    } finally {
      release()
      window.removeEventListener("tldw:backend-unreachable", listener)
    }
  })

  it.each(["fetch", "body"])("propagates a stream refresh %s abort without reading the original unauthorized response", async (phase) => {
    const originalResponse = new Response("unauthorized", { status: 401 })
    const readResponse = vi.spyOn(originalResponse, "text")
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) {
        const error = new DOMException("Refresh cancelled", "AbortError")
        if (phase === "fetch") throw error
        const response = Response.json({})
        vi.spyOn(response, "json").mockRejectedValueOnce(error)
        return response
      }
      return originalResponse
    })
    vi.stubGlobal("fetch", fetchSpy)
    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions", method: "POST", body: { stream: true }
    }))).rejects.toMatchObject({ name: "AbortError" })
    expect(readResponse).not.toHaveBeenCalled()
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("still reports an actual request deadline as a connection failure", async () => {
    const { bgRequest } = await importProxy()
    const events: Event[] = []
    const listener = (event: Event) => { events.push(event) }
    window.addEventListener("tldw:backend-unreachable", listener)
    vi.useFakeTimers()
    vi.stubGlobal("fetch", vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
      await new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      })))
    try {
      const result = bgRequest({
        path: "/api/v1/notes/", method: "GET", timeoutMs: 10
      }).then(value => value, error => error)
      await vi.advanceTimersByTimeAsync(20)
      expect(await result).toMatchObject({ status: 0, code: "REQUEST_TIMEOUT" })
      expect(events).toHaveLength(1)
    } finally {
      vi.useRealTimers()
      window.removeEventListener("tldw:backend-unreachable", listener)
    }
  })

  it.each([false, true])("retains retryable refresh failure details and does not replay a write (scoped=%s)", async (scoped) => {
    const config = {
      serverUrl: "https://api.example.com", authMode: "multi-user",
      accessToken: jwtForUser(42), refreshToken: "valid-refresh"
    }
    mocks.store.tldwConfig = config
    const writes: RequestInit[] = []
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/auth/refresh")) {
        return new Response(JSON.stringify({ detail: "Authentication service is busy" }), {
          status: 503, headers: { "content-type": "application/json", "retry-after": "2" }
        })
      }
      writes.push(init ?? {})
      return new Response("unauthorized", { status: 401 })
    }))
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/", method: "POST", body: { content: "Private draft" },
      ...(scoped ? { servicePromptConfig: {
        serverUrl: config.serverUrl, authMode: "multi-user" as const, expectedUserId: 42
      } } : {})
    })).rejects.toMatchObject({ status: 503, retryAfterMs: 2000 })
    expect(writes).toHaveLength(1)
    expect(mocks.store.tldwConfig).toEqual(config)
  })

  it.each(["POST", "PUT"] as const)("does not dispatch a scoped Notes %s after credential loading switches accounts", async (method) => {
    let release!: () => void
    let started!: () => void
    const gate = new Promise<void>((resolve) => { release = resolve })
    const entered = new Promise<void>((resolve) => { started = resolve })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") { started(); await gate }
      return mocks.store[key] ?? null
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    const pending = expect(bgRequest({
      path: method === "POST" ? "/api/v1/notes/" : "/api/v1/notes/private",
      method, body: { content: "Alice private draft" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com", authMode: "multi-user", expectedUserId: 42
      }
    })).rejects.toMatchObject({ status: 412 })
    // An unrecognized scoped route fails before storage; avoid hanging the red run.
    await Promise.race([entered, pending])
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com", authMode: "multi-user", accessToken: jwtForUser(84)
    }
    release()
    await pending
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("never retries a scoped Notes write as the account selected during refresh", async () => {
    const original = { serverUrl: "https://api.example.com", authMode: "multi-user", accessToken: jwtForUser(42), refreshToken: "alice-refresh" }
    const replacement = { ...original, accessToken: jwtForUser(84), refreshToken: "bob-refresh" }
    mocks.store.tldwConfig = original
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/auth/refresh")) {
        mocks.store.tldwConfig = replacement
        return new Response(JSON.stringify({ access_token: `${jwtForUser(42)}-new`, refresh_token: "alice-rotated" }), {
          status: 200, headers: { "content-type": "application/json" }
        })
      }
      return new Response("unauthorized", { status: 401 })
    })
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/", method: "POST", body: { content: "Alice private draft" },
      servicePromptConfig: { serverUrl: original.serverUrl, authMode: "multi-user", expectedUserId: 42 }
    })).rejects.toMatchObject({ status: 412 })
    expect(fetchSpy.mock.calls.filter(([url]) => String(url).endsWith("/notes/"))).toHaveLength(1)
    expect(mocks.store.tldwConfig).toEqual(replacement)
  })

  it("retries a scoped Notes write with refreshed credentials for the same principal", async () => {
    const original = { serverUrl: "https://api.example.com", authMode: "multi-user", accessToken: jwtForUser(42), refreshToken: "alice-refresh" }
    mocks.store.tldwConfig = original
    const rotated = `${jwtForUser(42)}-new`
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/auth/refresh")) return new Response(JSON.stringify({ access_token: rotated, refresh_token: "alice-rotated" }), { status: 200, headers: { "content-type": "application/json" } })
      return new Headers(init?.headers).get("Authorization") === `Bearer ${rotated}`
        ? new Response(JSON.stringify({ id: "saved" }), { status: 200, headers: { "content-type": "application/json" } })
        : new Response("unauthorized", { status: 401 })
    })
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/", method: "POST", body: { content: "Alice private draft" },
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: { serverUrl: original.serverUrl, authMode: "multi-user", expectedUserId: 42 }
    })).resolves.toEqual({ id: "saved" })
    const writes = fetchSpy.mock.calls.filter(([url]) => String(url).endsWith("/notes/"))
    expect(writes).toHaveLength(2)
    expect(writes.every(([, init]) => new Headers(init?.headers).get("X-TLDW-Expected-User-ID") === "42")).toBe(true)
  })

  it("rejects a changed runtime key before a scoped Notes write", async () => {
    mocks.store.tldwConfig = { serverUrl: "https://api.example.com", authMode: "single-user", apiKey: "alice-key" }
    mocks.runtimeApiKey = "bob-key"
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/notes/", method: "POST", body: { content: "Alice private draft" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com", authMode: "single-user",
        expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", "alice-key")
      }
    })).rejects.toMatchObject({ status: 412 })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each([
    "/api/v1/writing/manuscripts/scenes/scene-a",
    "/api/v1/writing/manuscripts/projects/project-a/characters?role=protagonist",
    "/api/v1/writing/manuscripts/projects/project-a/world-info?kind=location",
  ])("dispatches the scoped manuscript read %s through the real transport", async (path) => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com", authMode: "multi-user", accessToken: jwtForUser(42)
    }
    const fetchSpy = vi.fn().mockResolvedValue(new Response(JSON.stringify({ context: "owner-42" }), {
      status: 200, headers: { "content-type": "application/json" }
    }))
    vi.stubGlobal("fetch", fetchSpy)
    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: path as `/api/v1/writing/manuscripts/scenes/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: { serverUrl: "https://api.example.com", authMode: "multi-user", expectedUserId: 42 }
    })).resolves.toEqual({ context: "owner-42" })
    expect(fetchSpy.mock.calls[0][0]).toBe(`https://api.example.com${path}`)
    expect(new Headers(fetchSpy.mock.calls[0][1].headers).get("X-TLDW-Expected-User-ID")).toBe("42")
    expect(new Headers(fetchSpy.mock.calls[0][1].headers).get("Authorization")).toBe(`Bearer ${jwtForUser(42)}`)
  })

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
    mocks.runtimeApiKey = null
    mocks.nextRuntimeApiKey = null
    mocks.runtimeApiKeyReads = 0
    mocks.storageGet.mockReset()
    mocks.sessionStorageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageRemove.mockReset()
    mocks.storageGet.mockImplementation(async (key: string) => mocks.store[key] ?? null)
    mocks.sessionStorageGet.mockImplementation(
      async (key: string) => mocks.sessionStore[key] ?? null
    )
    mocks.storageSet.mockImplementation(async (key: string, value: unknown) => {
      mocks.store[key] = value
    })
    mocks.storageRemove.mockImplementation(async (key: string) => {
      delete mocks.store[key]
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
      const fetchSpy = vi.fn<typeof fetch>(async () =>
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      )
      vi.stubGlobal("fetch", fetchSpy)

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

  it.each(["POST", "PATCH"])(
    "keeps exact page-origin absolute %s requests on cookie auth",
    async (method) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      document.cookie = "csrf_token=absolute-csrf; Path=/"
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
          authSource: "cookie-session"
        }
      }
      const fetchSpy = vi.fn<typeof fetch>(async () =>
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      )
      vi.stubGlobal("fetch", fetchSpy)

      const { bgRequest } = await importProxy()
      await bgRequest({
        path: `${window.location.origin}/api/v1/notes/search/` as any,
        method: method as "POST",
        headers: {
          "X-API-KEY": "stale-key",
          Authorization: "Bearer stale-token",
          "X-CSRF-Token": "stale-csrf"
        },
        body: { q: "absolute-cookie" },
        preferDirect: true
      })

      const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
      expect(headers.get("X-CSRF-Token")).toBe("absolute-csrf")
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
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    const result = await bgRequest<{ ok: boolean }>({
      path: "/api/v1/notes/search/" as unknown as `/${string}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { q: "hello" }
    })

    expect(result).toEqual({ ok: true })
    expect(refreshHits).toBe(1)
    expect(mocks.store.tldwConfig).toEqual({
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "stale-access",
      refreshToken: "refresh-token"
    })
    expect(mocks.store.tldwRefreshRotation).toMatchObject({
      sourceRefreshToken: "refresh-token",
      accessToken: "fresh-access",
      refreshToken: "rotated-refresh"
    })
  })

  it("replaces a prior scoped rotation during an ordinary direct refresh", async () => {
    mocks.store.tldwRefreshRotation = {
      version: 1,
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      sourceAccessToken: "stale-access",
      sourceRefreshToken: "refresh-token",
      accessToken: "scoped-access",
      refreshToken: "scoped-refresh"
    }
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        expect(JSON.parse(String(init?.body))).toEqual({
          refresh_token: "scoped-refresh"
        })
        return new Response(JSON.stringify({
          access_token: "ordinary-access",
          refresh_token: "ordinary-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const authorization = new Headers(init?.headers).get("Authorization")
      if (authorization === "Bearer scoped-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/notes/search/" as unknown as `/${string}`,
      method: "POST",
      body: { q: "hello" }
    })).resolves.toEqual({ ok: true })

    expect(mocks.store.tldwConfig).toEqual({
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "stale-access",
      refreshToken: "refresh-token"
    })
    expect(mocks.store.tldwRefreshRotation).toMatchObject({
      sourceRefreshToken: "refresh-token",
      accessToken: "ordinary-access",
      refreshToken: "ordinary-refresh"
    })
  })

  it("does not overwrite or retry under an account selected during ordinary refresh", async () => {
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    let requestHits = 0
    const replacement = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "other-account-access",
      refreshToken: "other-account-refresh"
    }
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      requestHits += 1
      const authorization = new Headers(init?.headers).get("Authorization")
      return authorization === "Bearer stale-access"
        ? new Response("unauthorized", { status: 401 })
        : new Response(JSON.stringify({ ok: true }), {
            status: 200,
            headers: { "content-type": "application/json" }
          })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    const result = expect(bgRequest({
      path: "/api/v1/notes/search/" as unknown as `/${string}`,
      method: "POST",
      body: { q: "hello" }
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })

    await refreshStarted
    mocks.store.tldwConfig = replacement
    releaseRefresh()
    await result

    expect(mocks.store.tldwConfig).toEqual(replacement)
    expect(requestHits).toBe(1)
  })

  it("does not dispatch a direct request after storage changes", async () => {
    const servicePromptConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user" as const
    }
    mocks.store.tldwConfig = {
      serverUrl: "https://other.example.com",
      authMode: "multi-user",
      accessToken: "other-access",
      refreshToken: "other-refresh"
    }
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "PUT",
      body: { parts: {}, expected_revision: null },
      servicePromptConfig
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("does not dispatch a legacy catalog request after a same-target account switch", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: jwtForUser(84),
      refreshToken: "other-account-refresh"
    }
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/service-prompts" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        expectedUserId: 42
      }
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each([
    "/api/v1/chat/completions",
    "/api/v1/chats/chat-123/messages",
    "/api/v1/rag/search",
    "/api/v1/research/websearch"
  ])("allows a checked target for the exact execution route %s", async (path) => {
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: path as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: {},
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).resolves.toEqual({ ok: true })

    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(`https://api.example.com${path}`)
    expect(
      new Headers(fetchSpy.mock.calls[0]?.[1]?.headers).get(
        "X-TLDW-Expected-User-ID"
      )
    ).toBe("42")
  })

  it.each([
    ["/api/v1/notes/search/", "POST"],
    ["/api/v1/chat/completions/extra", "POST"],
    ["/api/v1/chats//messages", "POST"],
    ["/api/v1/chats/%2e%2e/messages", "POST"],
    ["/api/v1/chats/chat%2fid/messages", "POST"],
    ["/api/v1/chats/chat%5cid/messages", "POST"],
    ["/api/v1/chats/chat-123/messages/search", "POST"],
    ["/api/v1/chats/chat-123/messages/extra", "GET"]
  ] as const)("rejects a checked target on non-allowlisted route %s %s", async (path, method) => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: path as `/${string}`,
      method,
      body: {},
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).rejects.toThrow(/Service Prompt config/i)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("does not dispatch a direct stream after the checked target changes", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://other.example.com",
      authMode: "multi-user",
      accessToken: "other-access",
      refreshToken: "other-refresh"
    }
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    }))).rejects.toMatchObject({ status: 412 })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("does not stream after a same-target single-user runtime key change", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "single-user",
      apiKey: "captured-account-key"
    }
    mocks.runtimeApiKey = "changed-account-key"
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "single-user",
        expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
          "single-user",
          "captured-account-key"
        )!
      }
    }))).rejects.toMatchObject({ status: 412 })

    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("streams with the validated runtime key without reading a later override", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "single-user",
      apiKey: "stored-key"
    }
    mocks.runtimeApiKey = "captured-runtime-key"
    mocks.nextRuntimeApiKey = "later-runtime-key"
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response('data: {"ok":true}\n\ndata: [DONE]\n\n', {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "single-user",
        expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
          "single-user",
          "captured-runtime-key"
        )!
      }
    }))).resolves.toEqual(['{"ok":true}'])

    expect(mocks.runtimeApiKeyReads).toBe(1)
    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("captured-runtime-key")
  })

  it.each([
    "/api/v1/chat/completions/extra",
    "/api/v1/chats/%2e%2e/messages",
    "/api/v1/chats/chat%2fid/messages"
  ])("rejects a checked target on non-allowlisted direct stream route %s", async (path) => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: path as `/${string}`,
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    }))).rejects.toThrow(/Service Prompt config/i)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("binds a direct stream to the checked target and expected user", async () => {
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response('data: {"ok":true}\n\ndata: [DONE]\n\n', {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    }))).resolves.toEqual(['{"ok":true}'])

    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(
      "https://api.example.com/api/v1/chat/completions"
    )
    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("Authorization")).toBe("Bearer stale-access")
    expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
  })

  it("refreshes a checked direct stream without dropping the expected user", async () => {
    const streamHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const requestHeaders = new Headers(init?.headers)
      streamHeaders.push(requestHeaders)
      if (requestHeaders.get("Authorization") === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response("data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgStream } = await importProxy()
    await expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    }))).resolves.toEqual([])

    expect(streamHeaders).toHaveLength(2)
    expect(streamHeaders.map((headers) =>
      headers.get("X-TLDW-Expected-User-ID")
    )).toEqual(["42", "42"])
    expect(streamHeaders[1]?.get("Authorization")).toBe("Bearer fresh-access")
    expect(mocks.store.tldwConfig).toEqual({
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "stale-access",
      refreshToken: "refresh-token"
    })
    expect(mocks.store.tldwRefreshRotation).toMatchObject({
      sourceRefreshToken: "refresh-token",
      accessToken: "fresh-access",
      refreshToken: "rotated-refresh"
    })
  })

  describe("TASK-13549 coalesced GET preparation lifetime", () => {
    const original = {
      serverUrl: "https://api.example.com", authMode: "multi-user" as const,
      authSource: "manual" as const, accessToken: jwtForUser(42), refreshToken: "original-get-refresh"
    }
    const replacement = { ...original, accessToken: jwtForUser(84), refreshToken: "replacement-get-refresh" }
    const publishAccount = (config: typeof original) => {
      const previous = mocks.store.tldwConfig
      mocks.store.tldwConfig = structuredClone(config)
      window.dispatchEvent(new StorageEvent("storage", {
        key: "tldwConfig", oldValue: JSON.stringify(previous), newValue: JSON.stringify(config)
      }))
    }
    it.each(["A to B to exact A", "uninterrupted concurrent GETs"] as const)(
      "guards the first coalesced config read across %s", async boundary => {
        mocks.store.tldwConfig = structuredClone(original)
        let releaseConfig!: () => void
        const heldConfig = new Promise<void>(resolve => { releaseConfig = resolve })
        let releaseResponse!: () => void
        const heldResponse = new Promise<void>(resolve => { releaseResponse = resolve })
        let configEntered = false
        mocks.storageGet.mockImplementation(async (key: string) => {
          const value = structuredClone(mocks.store[key] ?? null)
          if (key === "tldwConfig") {
            configEntered = true
            await heldConfig
          }
          return value
        })
        const requests: { url: string; method?: string; authorization: string | null }[] = []
        const answer = { note: "Original account coalesced result" }
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/notes/private-note") throw new Error("Undeclared coalesced GET: " + url)
          requests.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization") })
          await heldResponse
          return Response.json(answer)
        }))
        const { watchChatAccountChanges } = await import("@/services/chat-account-boundary")
        const observed: boolean[] = []
        const releaseObserver = watchChatAccountChanges(invalidated => observed.push(invalidated))
        const { bgRequest } = await importProxy()
        const pending = bgRequest({ path: "/api/v1/notes/private-note", method: "GET", preferDirect: true })
          .then(value => ({ value }), error => ({ error }))
        let concurrent: typeof pending | undefined
        try {
          await vi.waitFor(() => expect(configEntered).toBe(true))
          expect(requests).toEqual([])
          if (boundary === "A to B to exact A") {
            publishAccount(replacement)
            publishAccount(original)
            expect(observed).toEqual([true, true])
          } else {
            concurrent = bgRequest({ path: "/api/v1/notes/private-note", method: "GET", preferDirect: true })
              .then(value => ({ value }), error => ({ error }))
            await vi.waitFor(() => expect(mocks.storageGet.mock.calls.filter(([key]) => key === "tldwConfig").length).toBeGreaterThanOrEqual(2))
          }
          releaseConfig()
          if (boundary === "uninterrupted concurrent GETs") {
            await vi.waitFor(() => expect(requests).toHaveLength(1))
            // Let both actual config resolutions finish while the shared physical
            // GET remains held, then require exactly one wire operation.
            await new Promise(resolve => setTimeout(resolve, 0))
            expect(requests).toHaveLength(1)
          }
          releaseResponse()
          const outcome = await pending
          if (boundary === "A to B to exact A") {
            expect(outcome).toMatchObject({ error: { status: 412 } })
            expect(requests).toEqual([])
          } else {
            expect(outcome).toEqual({ value: answer })
            expect(await concurrent).toEqual({ value: answer })
            expect(observed).toEqual([])
            expect(requests).toEqual([{ url: original.serverUrl + "/api/v1/notes/private-note",
              method: "GET", authorization: "Bearer " + original.accessToken }])
          }
          expect(mocks.store.tldwConfig).toEqual(original)
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        } finally {
          releaseConfig()
          releaseResponse()
          await pending
          if (concurrent) await concurrent
          releaseObserver()
        }
      }
    )
  })

  describe("TASK-13549 actual apiSend recipe preparation lifetime", () => {
    const original = {
      serverUrl: "https://api.example.com", authMode: "single-user" as const, authSource: "manual" as const,
      apiKey: "original-recipe-key", credentialSource: "manual" as const, apiKeyPersistence: "device" as const,
      apiKeyServerOrigin: "https://api.example.com"
    }
    const replacement = { ...original, apiKey: "replacement-recipe-key" }
    it.each(["A to B to exact A", "uninterrupted"] as const)(
      "keeps real createPrompt tied to its original account across %s", async boundary => {
        mocks.store.tldwConfig = structuredClone(original)
        const { resolveRecipePersistenceOwnerView, acknowledgeRecipePersistenceReceipt, readRecipePersistenceUncertainty } =
          await import("@/services/recipe-persistence-uncertainty")
        const owner = await resolveRecipePersistenceOwnerView()
        expect(owner).toMatchObject({ ownerId: expect.stringMatching(/^recipe-owner:sha256:[0-9a-f]{64}$/) })
        if (!owner) throw new Error("Actual direct recipe owner unavailable")
        const { CLEAR_TASK_RECIPE } = await import("@/components/Common/PromptAssist/recipes/built-in-recipes")
        const payload = { project_id: 42, name: "Original account recipe", prompt_format: "structured" as const,
          prompt_schema_version: 2, prompt_definition: CLEAR_TASK_RECIPE.definition, change_description: "Original recipe command" }
        const localId = "original-direct-recipe-" + boundary
        let releaseConfig!: () => void
        const heldConfig = new Promise<void>(resolve => { releaseConfig = resolve })
        let configEntered = false
        mocks.storageGet.mockImplementation(async (key: string) => {
          const value = structuredClone(mocks.store[key] ?? null)
          if (key === "tldwConfig" && !configEntered) {
            configEntered = true
            await heldConfig
          }
          return value
        })
        const requests: { url: string; method?: string; apiKey: string | null; body: unknown }[] = []
        const answer = { data: { id: 101, ...payload } }
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/prompt-studio/prompts/create") throw new Error("Undeclared recipe transport: " + url)
          requests.push({ url, method: init?.method, apiKey: new Headers(init?.headers).get("X-API-KEY"), body: JSON.parse(String(init?.body)) })
          return Response.json(answer)
        }))
        const { watchChatAccountChanges } = await import("@/services/chat-account-boundary")
        const observed: boolean[] = []
        const releaseObserver = watchChatAccountChanges(invalidated => observed.push(invalidated))
        const { createPrompt } = await import("@/services/prompt-studio")
        // Use the public client called by prompt-sync; neither apiSend nor its
        // original owner/dispatch authority is substituted by this fixture.
        const pending = createPrompt(payload, { recipePersistence: { mode: "require", expectedOwnerId: owner.ownerId, localId } })
        try {
          await vi.waitFor(() => expect(configEntered).toBe(true))
          expect(requests).toEqual([])
          if (boundary === "A to B to exact A") {
            for (const config of [replacement, original]) {
              const previous = mocks.store.tldwConfig
              mocks.store.tldwConfig = structuredClone(config)
              window.dispatchEvent(new StorageEvent("storage", {
                key: "tldwConfig", oldValue: JSON.stringify(previous), newValue: JSON.stringify(config)
              }))
            }
            expect(observed).toEqual([true, true])
          }
          releaseConfig()
          const response = await pending
          if (boundary === "A to B to exact A") {
            expect(response).toMatchObject({ ok: false, status: 412,
              recipePersistence: { state: "not_dispatched", actualOwnerId: null } })
            expect(response.recipeDelivery).toBeUndefined()
            expect(requests).toEqual([])
            expect(await readRecipePersistenceUncertainty(localId, owner.ownerId)).toBe("clear")
          } else {
            expect(response).toMatchObject({ ok: true, status: 200, data: answer,
              recipePersistence: { state: "dispatched", actualOwnerId: owner.ownerId },
              recipeDelivery: { id: localId, ownerId: owner.ownerId, operationId: expect.any(String) } })
            expect(requests).toEqual([{ url: original.serverUrl + "/api/v1/prompt-studio/prompts/create",
              method: "POST", apiKey: original.apiKey, body: payload }])
            expect(await readRecipePersistenceUncertainty(localId, owner.ownerId)).toBe("unknown_owner")
            if (!response.recipeDelivery) throw new Error("Actual direct dispatch receipt unavailable")
            await expect(acknowledgeRecipePersistenceReceipt(response.recipeDelivery)).resolves.toBeUndefined()
            expect(await readRecipePersistenceUncertainty(localId, owner.ownerId)).toBe("scoped")
            expect(observed).toEqual([])
          }
          expect(mocks.store.tldwConfig).toEqual(original)
        } finally {
          releaseConfig()
          await pending
          releaseObserver()
        }
      }
    )
  })

  describe("TASK-13549 original unscoped direct stream refresh authority", () => {
    const original = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      accessToken: jwtForUser(42),
      refreshToken: "original-direct-stream-refresh"
    }
    const replacement = {
      ...original,
      accessToken: jwtForUser(84),
      refreshToken: "replacement-direct-stream-refresh"
    }
    const rotatedAccess = `header.${btoa(JSON.stringify({ sub: "42", nonce: "stream-rotation" }))}.signature`
    const rotatedRefresh = "rotated-direct-stream-refresh"
    const streamBody = {
      stream: true,
      model: "original-direct-model",
      messages: [{ role: "user", content: "Exact original direct stream question" }]
    }
    const publishAccount = (config: typeof original) => {
      const previous = mocks.store.tldwConfig
      mocks.store.tldwConfig = structuredClone(config)
      // Exercise the actual account observer and serde through physical storage.
      window.dispatchEvent(new StorageEvent("storage", {
        key: "tldwConfig",
        oldValue: JSON.stringify(previous),
        newValue: JSON.stringify(config)
      }))
    }
    describe("TASK-13549 direct stream refresh failure settlement", () => {
      it.each([401, 502] as const)("preserves the actual refresh%s failure and retry metadata without reading or replaying the original stream", async status => {
        mocks.store.tldwConfig = structuredClone(original)
        const expectedMessage = status === 401 ? "Session expired. Please log in again."
          : "Original direct authentication service unavailable"
        const initialResponse = Response.json({ detail: "Original main access token expired" }, {
          status: 401, headers: { "retry-after": "1" }
        })
        const readOriginal = vi.spyOn(initialResponse, "json")
        const refreshResponse = Response.json({ detail: status === 401 ? "Original refresh session revoked" : expectedMessage }, {
          status, headers: { "retry-after": "7", "x-original-refresh-response": "original-direct-refresh-response" }
        })
        const readRefresh = vi.spyOn(refreshResponse, "json")
        let releaseRefresh!: () => void
        const heldRefresh = new Promise<void>(resolve => { releaseRefresh = resolve })
        const physical: { url: string; method?: string; authorization: string | null;
          command: string | null; body: unknown; signal: AbortSignal | null }[] = []
        let mainRequests = 0
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input), path = new URL(url).pathname
          if (path !== "/api/v1/chat/completions" && path !== "/api/v1/auth/refresh") throw new Error("Undeclared direct refresh failure transport: " + url)
          const headers = new Headers(init?.headers)
          physical.push({ url, method: init?.method, authorization: headers.get("Authorization"),
            command: headers.get("X-Original-Command"), body: JSON.parse(String(init?.body)), signal: init?.signal ?? null })
          if (path === "/api/v1/auth/refresh") {
            await heldRefresh
            return refreshResponse
          }
          mainRequests += 1
          return mainRequests === 1 ? initialResponse : streamResponse()
        }))
        const onDispatch = vi.fn()
        const onOpen = vi.fn()
        const { bgStream } = await importProxy()
        const pending = collectStream(bgStream({ path: "/api/v1/chat/completions", method: "POST", body: streamBody,
          headers: { "X-Original-Command": "original-direct-refresh-failure-command" }, onDispatch, onOpen
        })).then(value => ({ value }), error => ({ error }))
        try {
          await vi.waitFor(() => expect(physical).toHaveLength(2))
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, command: "original-direct-refresh-failure-command", body: streamBody })
          expect(physical[1]).toMatchObject({ url: original.serverUrl + "/api/v1/auth/refresh", method: "POST",
            authorization: null, command: null, body: { refresh_token: original.refreshToken } })
          expect(physical[0].signal?.aborted).toBe(false)
          expect(physical[1].signal?.aborted).toBe(false)
          expect(readOriginal).not.toHaveBeenCalled()
          expect(readRefresh).not.toHaveBeenCalled()
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).not.toHaveBeenCalled()
          releaseRefresh()
          const settled = await pending
          expect(physical.map(request => new URL(request.url).pathname)).toEqual([
            "/api/v1/chat/completions", "/api/v1/auth/refresh"
          ])
          expect(readRefresh).toHaveBeenCalledTimes(1)
          expect(settled).toMatchObject({ error: { status, message: expectedMessage,
            retryAfterMs: 7_000, headers: { "retry-after": "7", "x-original-refresh-response": "original-direct-refresh-response" }
          } })
          expect(readOriginal).not.toHaveBeenCalled()
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).not.toHaveBeenCalled()
          expect(mocks.store.tldwConfig).toEqual(original)
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
          const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
          const marker = refreshSessionInvalidationKey(original)
          if (!marker) throw new Error("Actual original direct refresh marker unavailable")
          if (status === 401) expect(mocks.store[marker]).toBe(true)
          else expect(mocks.store[marker]).toBeUndefined()
        } finally {
          releaseRefresh()
          await pending
          readOriginal.mockRestore()
          readRefresh.mockRestore()
        }
      })
    })

    describe("TASK-13549 original non-stream request lifetime", () => {
      const requestBody = { stream: false, model: "original-nonstream-model",
        messages: [{ role: "user", content: "Exact original private nonstream question" }] }
      const answer = { choices: [{ message: { role: "assistant", content: "Original nonstream answer" } }] }
      it.each(["A to B to exact A", "uninterrupted"] as const)("guards direct initial 401 across %s", async boundary => {
        mocks.store.tldwConfig = structuredClone(original)
        let releaseUnauthorized!: () => void
        const heldUnauthorized = new Promise<void>(resolve => { releaseUnauthorized = resolve })
        const requests: { url: string; method?: string; authorization: string | null; command: string | null; body: unknown }[] = []
        const refreshes: { url: string; method?: string; body: unknown }[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input), path = new URL(url).pathname
          if (path === "/api/v1/auth/refresh") {
            refreshes.push({ url, method: init?.method, body: JSON.parse(String(init?.body)) })
            return Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
          }
          if (path !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + url)
          const headers = new Headers(init?.headers)
          requests.push({ url, method: init?.method, authorization: headers.get("Authorization"),
            command: headers.get("X-Original-Command"), body: JSON.parse(String(init?.body)) })
          if (requests.length === 1) {
            await heldUnauthorized
            return new Response("Original nonstream unauthorized", { status: 401 })
          }
          return Response.json(answer)
        }))
        const { watchChatAccountChanges } = await import("@/services/chat-account-boundary")
        const accountEvents: boolean[] = []
        // Observe the same real native storage events without substituting or
        // granting authority to the request's own missing lifetime guard.
        const releaseObserver = watchChatAccountChanges(invalidated => accountEvents.push(invalidated))
        const { bgRequest } = await importProxy()
        const pending = bgRequest({ path: "/api/v1/chat/completions", method: "POST", body: requestBody,
          headers: { "X-Original-Command": "original-command-a" }, preferDirect: true
        }).then(value => ({ value }), error => ({ error }))
        try {
          await vi.waitFor(() => expect(requests).toHaveLength(1))
          expect(requests[0]).toEqual({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, command: "original-command-a", body: requestBody })
          expect(refreshes).toEqual([])
          if (boundary === "A to B to exact A") {
            publishAccount(replacement)
            publishAccount(original)
            expect(accountEvents).toEqual([true, true])
            expect(mocks.store.tldwConfig).toEqual(original)
          } else expect(accountEvents).toEqual([])
          releaseUnauthorized()
          const outcome = await pending
          if (boundary === "A to B to exact A") {
            // Exact restored credentials do not authorize the original command
            // to start a refresh or replay under a later account lifetime.
            expect(refreshes).toEqual([])
            expect(requests).toHaveLength(1)
            expect(outcome).toMatchObject({ error: { status: 412 } })
            expect(mocks.store.tldwRefreshRotation).toBeUndefined()
          } else {
            expect(outcome).toEqual({ value: answer })
            expect(refreshes).toEqual([{ url: original.serverUrl + "/api/v1/auth/refresh", method: "POST",
              body: { refresh_token: original.refreshToken } }])
            expect(requests).toEqual([{ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
              authorization: "Bearer " + original.accessToken, command: "original-command-a", body: requestBody },
              { url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
                authorization: "Bearer " + rotatedAccess, command: "original-command-a", body: requestBody }])
            expect(mocks.store.tldwRefreshRotation).toMatchObject({ sourceAccessToken: original.accessToken,
              sourceRefreshToken: original.refreshToken, accessToken: rotatedAccess, refreshToken: rotatedRefresh })
          }
          expect(mocks.store.tldwConfig).toEqual(original)
        } finally {
          releaseUnauthorized()
          await pending
          releaseObserver()
        }
      })
    })

    describe("TASK-13549 original extension-port queued settlement", () => {
      const firstFrame = '{"choices":[{"delta":{"content":"First original port frame"}}]}'
      const secondFrame = '{"choices":[{"delta":{"content":"Second original port frame"}}]}'
      it.each(["Stop", "uninterrupted"] as const)("guards the next actual queued port frame after %s", async boundary => {
        mocks.store.tldwConfig = structuredClone(original)
        const { browser } = await import("wxt/browser")
        const runtime = browser.runtime as unknown as Record<string, unknown>
        const messageListeners = new Set<(message: unknown) => void>()
        const disconnectListeners = new Set<() => void>()
        const portPayloads: unknown[] = []
        const delivered: unknown[] = []
        let disconnected = false
        const deliver = (message: unknown) => {
          delivered.push(message)
          for (const listener of messageListeners) listener(message)
        }
        const port = {
          postMessage: vi.fn((payload: unknown) => {
            portPayloads.push(payload)
            // This is the physical extension-port boundary. The production
            // bgStream owns its queue and async-generator drain unchanged.
            deliver({ event: "open" })
            deliver({ event: "data", data: firstFrame })
            deliver({ event: "data", data: secondFrame })
            deliver({ event: "done" })
          }),
          disconnect: vi.fn(() => {
            disconnected = true
            for (const listener of disconnectListeners) listener()
          }),
          onMessage: {
            addListener: (listener: (message: unknown) => void) => messageListeners.add(listener),
            removeListener: (listener: (message: unknown) => void) => messageListeners.delete(listener)
          },
          onDisconnect: {
            addListener: (listener: () => void) => disconnectListeners.add(listener),
            removeListener: (listener: () => void) => disconnectListeners.delete(listener)
          }
        }
        const sendMessage = vi.fn(async (message: { type?: string }) => {
          if (message?.type !== "tldw:ping") throw new Error("Undeclared runtime transport")
          return { ok: true, pong: true }
        })
        const connect = vi.fn(() => port)
        runtime.id = "task-13549-physical-port"
        runtime.sendMessage = sendMessage
        runtime.connect = connect
        const fetchSpy = vi.fn(async () => { throw new Error("Queued extension stream must not replay through direct fetch") })
        vi.stubGlobal("fetch", fetchSpy)
        const controller = new AbortController()
        const onDispatch = vi.fn(), onOpen = vi.fn()
        let stream: AsyncGenerator<string> | undefined
        try {
          const { bgStream } = await importProxy()
          stream = bgStream({ path: "/api/v1/chat/completions", method: "POST", body: streamBody,
            abortSignal: controller.signal, onDispatch, onOpen })
          expect(await stream.next()).toEqual({ value: firstFrame, done: false })
          expect(connect).toHaveBeenCalledWith({ name: "tldw:stream" })
          expect(sendMessage).toHaveBeenCalledWith({ type: "tldw:ping" })
          expect(portPayloads).toEqual([{ path: "/api/v1/chat/completions", method: "POST", headers: {},
            body: streamBody, streamIdleTimeoutMs: undefined }])
          expect(delivered).toEqual([{ event: "open" }, { event: "data", data: firstFrame },
            { event: "data", data: secondFrame }, { event: "done" }])
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).toHaveBeenCalledTimes(1)
          if (boundary === "Stop") {
            controller.abort()
            expect(disconnected).toBe(true)
            await expect(stream.next()).rejects.toMatchObject({ message: expect.stringMatching(/abort/i) })
          } else {
            expect(await stream.next()).toEqual({ value: secondFrame, done: false })
            expect(await stream.next()).toEqual({ value: undefined, done: true })
          }
          expect(await stream.next()).toEqual({ value: undefined, done: true })
          expect(fetchSpy).not.toHaveBeenCalled()
          expect(port.postMessage).toHaveBeenCalledTimes(1)
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).toHaveBeenCalledTimes(1)
          expect(mocks.store.tldwConfig).toEqual(original)
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        } finally {
          controller.abort()
          await stream?.return(undefined)
          delete runtime.id
          delete runtime.sendMessage
          delete runtime.connect
        }
      })
    })

    const streamResponse = () => new Response(
      'data: {"choices":[{"delta":{"content":"Original direct answer"}}]}\n\ndata: [DONE]\n\n',
      { headers: { "content-type": "text/event-stream" } }
    )

    describe("TASK-13549 buffered original direct stream settlement", () => {
      const firstFrame = '{"choices":[{"delta":{"content":"First original frame"}}]}'
      const secondFrame = '{"choices":[{"delta":{"content":"Second original frame"}}]}'
      const installBufferedTransport = (includeSecond: boolean) => {
        const physical: { url: string; method?: string; authorization: string | null; body: unknown; signal?: AbortSignal | null }[] = []
        const wire = `data: ${firstFrame}\n\n${includeSecond ? `data: ${secondFrame}\n\n` : ""}data: [DONE]\n\n`
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + url)
          physical.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
            body: JSON.parse(String(init?.body)), signal: init?.signal })
          // One actual physical body buffer. Abort cannot retract text already
          // decoded and held across an async-generator yield.
          return new Response(new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(wire))
              controller.close()
            }
          }), { headers: { "content-type": "text/event-stream" } })
        }))
        return physical
      }
      const boundaries = (["replacement", "A to B to A", "Stop"] as const).flatMap(boundary => [
        { boundary, next: "buffered frame", includeSecond: true },
        { boundary, next: "buffered DONE", includeSecond: false }
      ])

      it.each(boundaries)("refuses $next after $boundary following the first original yield", async ({ boundary, includeSecond }) => {
        mocks.store.tldwConfig = structuredClone(original)
        const physical = installBufferedTransport(includeSecond)
        const controller = new AbortController()
        const onDispatch = vi.fn()
        const onOpen = vi.fn()
        const { bgStream } = await importProxy()
        const stream = bgStream({ path: "/api/v1/chat/completions", method: "POST", body: streamBody,
          abortSignal: controller.signal, onDispatch, onOpen })
        try {
          expect(await stream.next()).toEqual({ done: false, value: firstFrame })
          expect(physical).toHaveLength(1)
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).toHaveBeenCalledTimes(1)
          if (boundary === "Stop") controller.abort()
          else {
            publishAccount(replacement)
            if (boundary === "A to B to A") publishAccount(original)
          }
          expect(physical[0].signal?.aborted).toBe(true)
          const resumed = await stream.next().then(value => ({ value }), error => ({ error }))
          expect(resumed).toMatchObject({ error: boundary === "Stop"
            ? { name: "AbortError", code: "REQUEST_ABORTED" }
            : { status: 412, details: { detail: { code: "request_config_scope_changed" } } } })
          expect(physical).toHaveLength(1)
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).toHaveBeenCalledTimes(1)
          expect(mocks.store.tldwConfig).toEqual(boundary === "replacement" ? replacement : original)
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        } finally {
          controller.abort()
          await stream.return(undefined)
        }
      })

      it.each([true, false])("preserves uninterrupted original buffered sequence with second frame=%s", async includeSecond => {
        mocks.store.tldwConfig = structuredClone(original)
        const physical = installBufferedTransport(includeSecond)
        const onDispatch = vi.fn()
        const onOpen = vi.fn()
        const { bgStream } = await importProxy()
        const stream = bgStream({ path: "/api/v1/chat/completions", method: "POST", body: streamBody, onDispatch, onOpen })
        try {
          expect(await stream.next()).toEqual({ done: false, value: firstFrame })
          if (includeSecond) expect(await stream.next()).toEqual({ done: false, value: secondFrame })
          expect(await stream.next()).toEqual({ done: true, value: undefined })
          expect(await stream.next()).toEqual({ done: true, value: undefined })
          expect(physical).toHaveLength(1)
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(onDispatch).toHaveBeenCalledTimes(1)
          expect(onOpen).toHaveBeenCalledTimes(1)
          expect(mocks.store.tldwConfig).toEqual(original)
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        } finally {
          await stream.return(undefined)
        }
      })
    })

    const holdInitialUnauthorized = () => {
      let releaseInitial!: () => void
      const heldInitial = new Promise<void>(resolve => { releaseInitial = resolve })
      const streams: { url: string; authorization: string | null; body: unknown }[] = []
      const refreshes: { url: string; body: unknown }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input), path = new URL(url).pathname
        if (path === "/api/v1/auth/refresh") {
          refreshes.push({ url, body: JSON.parse(String(init?.body)) })
          return Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
        }
        if (path !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + path)
        streams.push({ url, authorization: new Headers(init?.headers).get("Authorization"), body: JSON.parse(String(init?.body)) })
        if (streams.length === 1) {
          await heldInitial
          return new Response("Original stream unauthorized", { status: 401 })
        }
        return streamResponse()
      }))
      return { streams, refreshes, releaseInitial }
    }

    it.each(["replacement", "A to B to A"] as const)(
      "refuses original unscoped direct refresh or replay after held initial 401 and %s",
      async boundary => {
        mocks.store.tldwConfig = structuredClone(original)
        const physical = holdInitialUnauthorized()
        const { bgStream } = await importProxy()
        const pending = collectStream(bgStream({
          path: "/api/v1/chat/completions", method: "POST", body: streamBody
        })).then(value => ({ value }), error => ({ error }))
        try {
          await vi.waitFor(() => expect(physical.streams).toHaveLength(1))
          expect(physical.streams[0]).toEqual({
            url: original.serverUrl + "/api/v1/chat/completions",
            authorization: "Bearer " + original.accessToken,
            body: streamBody
          })
          expect(physical.refreshes).toEqual([])
          publishAccount(replacement)
          if (boundary === "A to B to A") publishAccount(original)
          const current = boundary === "replacement" ? replacement : original
          physical.releaseInitial()
          const settled = await pending
          // The account boundary occurs after the original physical stream
          // entry, before refresh. Exact restored A cannot revive that stream.
          expect(physical.refreshes).toEqual([])
          expect(settled).toMatchObject({ error: {
            status: 412, details: { detail: { code: "request_config_scope_changed" } }
          } })
          expect(mocks.storageSet.mock.calls.filter(([key]) => key === "tldwRefreshRotation")).toEqual([])
          expect(mocks.store.tldwRefreshRotation).toBeUndefined()
          expect(mocks.store.tldwConfig).toEqual(current)
          expect(physical.streams).toHaveLength(1)
        } finally {
          physical.releaseInitial()
          await pending
        }
      }
    )

    it("reuses a same-principal winner after held initial 401 while preserving the original direct stream", async () => {
      mocks.store.tldwConfig = structuredClone(original)
      const physical = holdInitialUnauthorized()
      const { bgStream } = await importProxy()
      const pending = collectStream(bgStream({
        path: "/api/v1/chat/completions", method: "POST", body: streamBody
      })).then(value => ({ value }), error => ({ error }))
      const winner = { ...original, accessToken: rotatedAccess, refreshToken: rotatedRefresh }
      try {
        await vi.waitFor(() => expect(physical.streams).toHaveLength(1))
        expect(physical.streams[0]).toEqual({
          url: original.serverUrl + "/api/v1/chat/completions",
          authorization: "Bearer " + original.accessToken,
          body: streamBody
        })
        publishAccount(winner)
        physical.releaseInitial()
        expect(await pending).toEqual({ value: ['{"choices":[{"delta":{"content":"Original direct answer"}}]}'] })
        expect(physical.refreshes).toEqual([])
        expect(physical.streams).toHaveLength(2)
        expect(physical.streams[1]).toEqual({
          url: original.serverUrl + "/api/v1/chat/completions",
          authorization: "Bearer " + rotatedAccess,
          body: streamBody
        })
        expect(mocks.storageSet.mock.calls.filter(([key]) => key === "tldwRefreshRotation")).toEqual([])
        expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        expect(mocks.store.tldwConfig).toEqual(winner)
      } finally {
        physical.releaseInitial()
        await pending
      }
    })

    it("TASK-13549 detaches direct Stop before the held spent refresh completes and later preserves its original rotation", async () => {
      mocks.store.tldwConfig = structuredClone(original)
      let releaseRefresh!: () => void
      const heldRefresh = new Promise<void>(resolve => { releaseRefresh = resolve })
      const streams: { authorization: string | null; body: unknown }[] = []
      const refreshes: { body: unknown; signal?: AbortSignal | null }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = new URL(String(input)).pathname
        if (path === "/api/v1/auth/refresh") {
          refreshes.push({ body: JSON.parse(String(init?.body)), signal: init?.signal })
          await heldRefresh
          return Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
        }
        if (path !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + path)
        streams.push({ authorization: new Headers(init?.headers).get("Authorization"), body: JSON.parse(String(init?.body)) })
        return streams.length === 1 ? new Response("Original stream unauthorized", { status: 401 }) : streamResponse()
      }))
      const controller = new AbortController()
      const { bgStream } = await importProxy()
      let completion: { value?: string[]; error?: unknown } | undefined
      let rotationObserved = false
      const pending = collectStream(bgStream({
        path: "/api/v1/chat/completions", method: "POST", body: streamBody, abortSignal: controller.signal
      })).then(value => { completion = { value }; return completion }, error => { completion = { error }; return completion })
      try {
        await vi.waitFor(() => expect(refreshes).toHaveLength(1))
        expect(streams).toEqual([{ authorization: "Bearer " + original.accessToken, body: streamBody }])
        expect(refreshes[0].body).toEqual({ refresh_token: original.refreshToken })
        controller.abort()
        // The original iterator must settle while the declared physical refresh
        // still waits. The account, rather than this stopped caller, owns it.
        await vi.waitFor(() => expect(completion).toMatchObject({ error: { name: "AbortError", code: "REQUEST_ABORTED" } }))
        expect(refreshes[0].signal?.aborted).toBe(false)
        expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        expect(streams).toHaveLength(1)
        releaseRefresh()
        await pending
        await vi.waitFor(() => expect(mocks.store.tldwRefreshRotation).toMatchObject({
          sourceAccessToken: original.accessToken, sourceRefreshToken: original.refreshToken,
          accessToken: rotatedAccess, refreshToken: rotatedRefresh
        }))
        rotationObserved = true
        expect(mocks.store.tldwConfig).toEqual(original)
        expect(streams).toHaveLength(1)
      } finally {
        controller.abort()
        releaseRefresh()
        await pending
        // A detached caller can finish first. Settle its account-owned token
        // publication before this fixture's physical store is reset.
        if (refreshes.length && !rotationObserved) {
          await vi.waitFor(() => expect(mocks.store.tldwRefreshRotation).toMatchObject({
            sourceAccessToken: original.accessToken, sourceRefreshToken: original.refreshToken,
            accessToken: rotatedAccess, refreshToken: rotatedRefresh
          }))
        }
      }
    })

    it("publishes the spent original rotation after sole Stop without replaying the cancelled stream", async () => {
      mocks.store.tldwConfig = structuredClone(original)
      let releaseRefresh!: () => void
      const heldRefresh = new Promise<void>(resolve => { releaseRefresh = resolve })
      const streams: { authorization: string | null; body: unknown }[] = []
      const refreshes: { body: unknown; signal?: AbortSignal | null }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = new URL(String(input)).pathname
        if (path === "/api/v1/auth/refresh") {
          refreshes.push({ body: JSON.parse(String(init?.body)), signal: init?.signal })
          await heldRefresh
          return Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
        }
        if (path !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + path)
        streams.push({ authorization: new Headers(init?.headers).get("Authorization"), body: JSON.parse(String(init?.body)) })
        return streams.length === 1 ? new Response("Original stream unauthorized", { status: 401 }) : streamResponse()
      }))
      const controller = new AbortController()
      const { bgStream } = await importProxy()
      let rotationObserved = false
      const pending = collectStream(bgStream({
        path: "/api/v1/chat/completions", method: "POST", body: streamBody, abortSignal: controller.signal
      })).then(value => ({ value }), error => ({ error }))
      try {
        await vi.waitFor(() => expect(refreshes).toHaveLength(1))
        expect(streams).toEqual([{ authorization: "Bearer " + original.accessToken, body: streamBody }])
        expect(refreshes[0].body).toEqual({ refresh_token: original.refreshToken })
        expect(refreshes[0].signal?.aborted).toBe(false)
        controller.abort()
        const rotationWasAbortedByStop = refreshes[0].signal?.aborted
        releaseRefresh()
        expect(await pending).toMatchObject({ error: { name: "AbortError", code: "REQUEST_ABORTED" } })
        expect(rotationWasAbortedByStop).toBe(false)
        await vi.waitFor(() => expect(mocks.store.tldwRefreshRotation).toMatchObject({
          sourceAccessToken: original.accessToken,
          sourceRefreshToken: original.refreshToken,
          accessToken: rotatedAccess,
          refreshToken: rotatedRefresh
        }))
        rotationObserved = true
        expect(mocks.store.tldwConfig).toEqual(original)
        expect(streams).toHaveLength(1)
        const { createSafeStorage } = await import("@/utils/safe-storage")
        const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
        expect(await resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))).toMatchObject({
          ...original, accessToken: rotatedAccess, refreshToken: rotatedRefresh
        })
      } finally {
        controller.abort()
        releaseRefresh()
        await pending
        if (refreshes.length && !rotationObserved) {
          await vi.waitFor(() => expect(mocks.store.tldwRefreshRotation).toMatchObject({
            sourceAccessToken: original.accessToken,
            sourceRefreshToken: original.refreshToken,
            accessToken: rotatedAccess,
            refreshToken: rotatedRefresh
          }))
        }
      }
    })

    it("refuses original rotation publication and replay after account ABA during the final storage read", async () => {
      mocks.store.tldwConfig = structuredClone(original)
      let releaseFinalRead!: () => void
      const heldFinalRead = new Promise<void>(resolve => { releaseFinalRead = resolve })
      let refreshReturned = false
      let finalReadEntered = false
      mocks.storageGet.mockImplementation(async (key: string) => {
        const physicalValue = mocks.store[key] ?? null
        if (key === "tldwRefreshRotation" && refreshReturned && !finalReadEntered) {
          finalReadEntered = true
          // The actual helper has already captured tldwConfig. Hold its last
          // physical read before publication without substituting its guard.
          await heldFinalRead
        }
        return physicalValue
      })
      const streams: { authorization: string | null; body: unknown }[] = []
      const refreshes: { body: unknown; signal?: AbortSignal | null }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = new URL(String(input)).pathname
        if (path === "/api/v1/auth/refresh") {
          refreshes.push({ body: JSON.parse(String(init?.body)), signal: init?.signal })
          refreshReturned = true
          return Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
        }
        if (path !== "/api/v1/chat/completions") throw new Error("Undeclared direct transport: " + path)
        streams.push({ authorization: new Headers(init?.headers).get("Authorization"), body: JSON.parse(String(init?.body)) })
        return streams.length === 1 ? new Response("Original stream unauthorized", { status: 401 }) : streamResponse()
      }))
      const { bgStream } = await importProxy()
      const pending = collectStream(bgStream({
        path: "/api/v1/chat/completions", method: "POST", body: streamBody
      })).then(value => ({ value }), error => ({ error }))
      try {
        await vi.waitFor(() => expect(finalReadEntered).toBe(true))
        expect(refreshes).toHaveLength(1)
        expect(refreshes[0].body).toEqual({ refresh_token: original.refreshToken })
        expect(streams).toEqual([{ authorization: "Bearer " + original.accessToken, body: streamBody }])
        expect(mocks.storageSet.mock.calls.filter(([key]) => key === "tldwRefreshRotation")).toEqual([])
        publishAccount(replacement)
        publishAccount(original)
        releaseFinalRead()
        expect(await pending).toMatchObject({ error: {
          status: 412, details: { detail: { code: "request_config_scope_changed" } }
        } })
        expect(mocks.storageSet.mock.calls.filter(([key]) => key === "tldwRefreshRotation")).toEqual([])
        expect(mocks.store.tldwRefreshRotation).toBeUndefined()
        expect(mocks.store.tldwConfig).toEqual(original)
        expect(streams).toHaveLength(1)
      } finally {
        releaseFinalRead()
        await pending
      }
    })
  })

  it("discards a checked direct stream refresh after the account token changes", async () => {
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    const streamHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const requestHeaders = new Headers(init?.headers)
      streamHeaders.push(requestHeaders)
      return new Response("unauthorized", { status: 401 })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const replacement = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "other-account-access",
      refreshToken: "other-account-refresh"
    }
    const { bgStream } = await importProxy()
    const result = expect(collectStream(bgStream({
      path: "/api/v1/chat/completions" as `/${string}`,
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    }))).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })

    await refreshStarted
    mocks.store.tldwConfig = replacement
    releaseRefresh()
    await result

    expect(mocks.store.tldwConfig).toEqual(replacement)
    expect(streamHeaders).toHaveLength(1)
  })

  it("uses the current direct credential while preserving the checked user binding", async () => {
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).resolves.toEqual({ ok: true })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("Authorization")).toBe("Bearer stale-access")
    expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
  })

  it("refreshes a scoped direct request without dropping the checked user binding", async () => {
    const servicePromptHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        return new Response(
          JSON.stringify({
            access_token: "fresh-access",
            refresh_token: "rotated-refresh"
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        )
      }

      const headers = new Headers(init?.headers)
      servicePromptHeaders.push(headers)
      if (headers.get("Authorization") === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).resolves.toEqual({ ok: true })

    expect(servicePromptHeaders).toHaveLength(2)
    expect(
      servicePromptHeaders.map((headers) =>
        headers.get("X-TLDW-Expected-User-ID")
      )
    ).toEqual(["42", "42"])
    expect(servicePromptHeaders[1]?.get("Authorization")).toBe(
      "Bearer fresh-access"
    )
    expect(mocks.store.tldwConfig).toEqual({
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: "stale-access",
      refreshToken: "refresh-token"
    })
    expect(mocks.store.tldwRefreshRotation).toMatchObject({
      sourceRefreshToken: "refresh-token",
      accessToken: "fresh-access",
      refreshToken: "rotated-refresh"
    })
  })

  it("reuses a completed scoped rotation for a delayed same-user request", async () => {
    let staleRequestHits = 0
    let refreshHits = 0
    let releaseDelayed!: () => void
    const delayed = new Promise<void>((resolve) => {
      releaseDelayed = resolve
    })
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      if (headers.get("Authorization") === "Bearer stale-access") {
        staleRequestHits += 1
        if (staleRequestHits === 2) await delayed
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    const call = () => bgRequest<{ ok: boolean }>({
      path: "/api/v1/chats/chat-123/messages" as unknown as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { message: "hello" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })
    const first = call()
    const second = call()

    await expect(first).resolves.toEqual({ ok: true })
    releaseDelayed()
    await expect(second).resolves.toEqual({ ok: true })
    expect(refreshHits).toBe(1)
  })

  it("recovers when another context wins the rotating refresh token", async () => {
    let refreshHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        setTimeout(() => {
          mocks.store.tldwRefreshRotation = {
            version: 1,
            serverUrl: "https://api.example.com",
            authMode: "multi-user",
            sourceAccessToken: "stale-access",
            sourceRefreshToken: "refresh-token",
            accessToken: "winner-access",
            refreshToken: "winner-refresh"
          }
        }, 5)
        return new Response("invalid refresh token", { status: 401 })
      }
      const headers = new Headers(init?.headers)
      if (headers.get("Authorization") === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/chats/chat-123/messages" as unknown as `/${string}`,
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { message: "hello" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).resolves.toEqual({ ok: true })

    expect(refreshHits).toBe(1)
    const requestHeaders = fetchSpy.mock.calls
      .filter(([input]) => !String(input).endsWith("/api/v1/auth/refresh"))
      .map(([, init]) => new Headers(init?.headers).get("Authorization"))
    expect(requestHeaders).toEqual([
      "Bearer stale-access",
      "Bearer winner-access"
    ])
  })

  it("reuses an ordinary raw-config refresh that wins before scoped refresh starts", async () => {
    const staleAccess = jwtForUser(42)
    const winnerAccess = `${jwtForUser(42)}-rotated`
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: staleAccess,
      refreshToken: "refresh-token"
    }
    let refreshHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response("refresh should not run", { status: 500 })
      }
      const headers = new Headers(init?.headers)
      if (headers.get("Authorization") === `Bearer ${staleAccess}`) {
        mocks.store.tldwConfig = {
          serverUrl: "https://api.example.com",
          authMode: "multi-user",
          accessToken: winnerAccess,
          refreshToken: "winner-refresh"
        }
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/chats/chat-123/messages" as unknown as `/${string}`,
      method: "POST",
      body: { message: "hello" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).resolves.toEqual({ ok: true })

    expect(refreshHits).toBe(0)
    const requestHeaders = fetchSpy.mock.calls
      .filter(([input]) => !String(input).endsWith("/api/v1/auth/refresh"))
      .map(([, requestInit]) =>
        new Headers(requestInit?.headers).get("Authorization")
      )
    expect(requestHeaders).toEqual([
      `Bearer ${staleAccess}`,
      `Bearer ${winnerAccess}`
    ])
  })

  it.each([
    {
      name: "target",
      replacement: {
        serverUrl: "https://other.example.com",
        authMode: "multi-user",
        accessToken: "other-target-access",
        refreshToken: "other-target-refresh"
      }
    },
    {
      name: "account token",
      replacement: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        accessToken: "other-account-access",
        refreshToken: "other-account-refresh"
      }
    }
  ])("discards a scoped direct refresh after the $name changes", async ({ replacement }) => {
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    const requestHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      requestHeaders.push(headers)
      if (headers.get("Authorization") === "Bearer stale-access") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    const result = expect(bgRequest({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user"
      }
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })

    await refreshStarted
    mocks.store.tldwConfig = replacement
    releaseRefresh()
    await result

    expect(mocks.store.tldwConfig).toEqual(replacement)
    expect(requestHeaders).toHaveLength(1)
  })

  it("does not refresh a scoped direct request with a token changed by the initial request", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: jwtForUser(42),
      refreshToken: "refresh-token"
    }
    const replacement = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user",
      accessToken: jwtForUser(84),
      refreshToken: "other-account-refresh"
    }
    let refreshHits = 0
    let requestHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response(JSON.stringify({
          access_token: "fresh-access",
          refresh_token: "rotated-refresh"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      requestHits += 1
      if (requestHits === 1) {
        mocks.store.tldwConfig = replacement
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        expectedUserId: 42
      }
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })

    expect(refreshHits).toBe(0)
    expect(requestHits).toBe(1)
    expect(mocks.store.tldwConfig).toEqual(replacement)
  })

  it("uses the captured single-user runtime override when its scope still matches", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "single-user",
      apiKey: "stored-key"
    }
    mocks.runtimeApiKey = "conflicting-runtime-key"
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest<{ ok: boolean }>({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "single-user",
        expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
          "single-user",
          "conflicting-runtime-key"
        )!
      }
    })).resolves.toEqual({ ok: true })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("conflicting-runtime-key")
  })

  it("rejects a changed single-user runtime override before direct fetch", async () => {
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "single-user",
      apiKey: "captured-account-key"
    }
    mocks.runtimeApiKey = "changed-account-key"
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      servicePromptConfig: {
        serverUrl: "https://api.example.com",
        authMode: "single-user",
        expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
          "single-user",
          "captured-account-key"
        )!
      }
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })

    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("fails a scoped direct request closed after same-target logout", async () => {
    const checkedConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user" as const
    }
    mocks.store.tldwConfig = {
      serverUrl: "https://api.example.com",
      authMode: "multi-user"
    }
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()
    await expect(bgRequest({
      path: "/api/v1/service-prompts/chat.rag.answer" as unknown as `/${string}`,
      method: "GET",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      servicePromptConfig: checkedConfig
    })).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
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
    vi.stubGlobal("fetch", fetchSpy)

    const { bgRequest } = await importProxy()

    // A malformed success is a service failure, not proof of an expired session.
    await expect(
      bgRequest<{ ok: boolean }>({
        path: "/api/v1/notes/search/" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { q: "hello" }
      })
    ).rejects.toMatchObject({ status: 502 })

    expect(refreshHits).toBe(1)
    expect(staleRetryHits).toBe(1)
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
    vi.stubGlobal("fetch", fetchSpy)

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
    vi.stubGlobal("fetch", fetchSpy)

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
