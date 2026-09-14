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

vi.mock("@/utils/safe-storage", () => ({
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
      const fetchSpy = vi.fn(async () =>
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
      const fetchSpy = vi.fn(async () =>
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
    const fetchSpy = vi.fn(async () =>
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
    ["/api/v1/chats/chat-123/messages", "GET"]
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
    const fetchSpy = vi.fn(async () =>
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
    const fetchSpy = vi.fn(async () =>
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
    const fetchSpy = vi.fn(async () =>
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
    const fetchSpy = vi.fn(async () =>
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
