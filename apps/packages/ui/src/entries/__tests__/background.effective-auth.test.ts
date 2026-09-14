import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const connectListeners = new Set<(port: any) => void>()
const runtimeMessageListeners = new Set<(
  message: unknown,
  sender: unknown,
  sendResponse: (response: unknown) => void
) => unknown>()
const jwtForUser = (userId: string | number): string =>
  `header.${btoa(JSON.stringify({ sub: String(userId) }))}.signature`

const storageState = vi.hoisted(() => ({
  persistent: new Map<string, unknown>(),
  session: new Map<string, unknown>(),
  set: vi.fn(async () => undefined)
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: (options?: { area?: string }) => {
    const values =
      options?.area === "session" ? storageState.session : storageState.persistent
    return {
      get: async (key: string) => values.get(key),
      set: async (key: string, value: unknown) => {
        await storageState.set(key, value)
        values.set(key, value)
      },
      remove: vi.fn(async (key: string) => values.delete(key))
    }
  }
}))

vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "model-warm",
  initBackground: vi.fn(async () => {})
}))

vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: vi.fn(async () => {})
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "extension-id",
      getURL: (path: string) => `chrome-extension://extension-id${path}`,
      sendMessage: vi.fn(async () => ({ handled: true })),
      onConnect: {
        addListener: (listener: (port: any) => void) => connectListeners.add(listener)
      },
      onMessage: {
        addListener: (listener: any) => runtimeMessageListeners.add(listener)
      },
      onStartup: { addListener: vi.fn() }
    },
    storage: {
      local: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {})
      },
      session: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {})
      },
      onChanged: { addListener: vi.fn() }
    },
    alarms: {
      clear: vi.fn(async () => true),
      create: vi.fn(async () => {}),
      onAlarm: { addListener: vi.fn() }
    },
    tabs: {
      create: vi.fn(),
      query: vi.fn(async () => []),
      sendMessage: vi.fn(async () => undefined)
    },
    action: { onClicked: { addListener: vi.fn() } },
    contextMenus: {
      create: vi.fn(),
      removeAll: vi.fn(),
      onClicked: { addListener: vi.fn() }
    },
    i18n: { getMessage: (key: string) => key }
  }
}))

type RuntimePortHarness = {
  messages: unknown[]
  postMessage: (message: unknown) => Promise<void>
  disconnect: () => void
}

const connectRuntimePort = (name: string): RuntimePortHarness => {
  const inboundListeners = new Set<(message: unknown) => unknown>()
  const disconnectListeners = new Set<() => void>()
  const messages: unknown[] = []
  const port = {
    name,
    sender: { id: "extension-id" },
    postMessage: (message: unknown) => messages.push(message),
    onMessage: {
      addListener: (listener: (message: unknown) => unknown) =>
        inboundListeners.add(listener),
      removeListener: (listener: (message: unknown) => unknown) =>
        inboundListeners.delete(listener)
    },
    onDisconnect: {
      addListener: (listener: () => void) => disconnectListeners.add(listener)
    },
    disconnect: () => {
      for (const listener of disconnectListeners) listener()
    }
  }
  for (const listener of connectListeners) listener(port)
  return {
    messages,
    postMessage: async (message: unknown) => {
      await Promise.all(
        Array.from(inboundListeners, (listener) => listener(message))
      )
    },
    disconnect: port.disconnect
  }
}

const sendRuntimeMessage = async (message: unknown): Promise<any> =>
  await new Promise((resolve) => {
    const listener = [...runtimeMessageListeners][0]
    if (!listener) throw new Error("Background runtime listener missing")
    listener(message, { id: "extension-id" }, resolve)
  })

const flushPromises = async () => {
  await Promise.resolve()
  await Promise.resolve()
  await new Promise((resolve) => setTimeout(resolve, 0))
}

import background from "@/entries/background"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { tldwAuth } from "@/services/tldw/TldwAuth"

const WORKER_API_KEY_SCOPE = deriveSingleUserApiKeyCredentialScope(
  "single-user",
  "worker-session-key"
)!

describe("background effective extension auth", () => {
  let windowDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    connectListeners.clear()
    runtimeMessageListeners.clear()
    storageState.persistent.clear()
    storageState.session.clear()
    storageState.set.mockClear()
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })
    storageState.persistent.set("tldwCookieSessionConfig", {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    storageState.session.set("tldwManualSessionApiKey", {
      apiKey: "worker-session-key",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })
    windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, "window")
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: undefined
    })
    background.main()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (windowDescriptor) {
      Object.defineProperty(globalThis, "window", windowDescriptor)
    } else {
      delete (globalThis as any).window
    }
  })

  it("registers runtime messaging when the optional commands API is unavailable", async () => {
    await expect(sendRuntimeMessage({ type: "tldw:ping" })).resolves.toMatchObject({
      ok: true,
      pong: true
    })
  })

  it("authenticates ordinary worker requests with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: { path: "/api/v1/health", method: "GET" }
      })
    ).resolves.toMatchObject({ ok: true, status: 200 })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("worker-session-key")
    expect(storageState.persistent.get("tldwConfig")).not.toHaveProperty("apiKey")
  })

  it("rejects when extension storage changes before worker dispatch", async () => {
    const servicePromptConfig = {
      ...(storageState.persistent.get("tldwConfig") as Record<string, unknown>),
      apiKey: "worker-session-key"
    }
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://other.example.test",
      authMode: "single-user",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://other.example.test",
      apiKey: "other-key"
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "PUT",
          servicePromptConfig
        }
      })
    ).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("rejects a same-target single-user API-key change before worker dispatch", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ id: "wrong-account-write" }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "PUT",
          body: { parts: {}, expected_revision: null },
          servicePromptConfig: {
            serverUrl: "https://api.example.test",
            authMode: "single-user",
            authSource: "manual",
            expectedSingleUserApiKeyScope: "key:captured-account"
          }
        }
      })
    ).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("uses the current worker credential after the configured target matches", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "GET",
          servicePromptConfig: {
            serverUrl: "https://api.example.test",
            authMode: "single-user",
            authSource: "manual",
            expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
          }
        }
      })
    ).resolves.toMatchObject({ ok: true, status: 200 })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("worker-session-key")
  })

  it("uses the current same-target account credential with the checked user binding", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "current-account-token"
    })
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "GET",
          headers: { "X-TLDW-Expected-User-ID": "42" },
          servicePromptConfig: {
            serverUrl: "https://api.example.test",
            authMode: "multi-user",
            authSource: "manual",
            accessToken: "checked-account-token"
          }
        }
      })
    ).resolves.toMatchObject({ ok: true, status: 200 })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("Authorization")).toBe("Bearer current-account-token")
    expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
  })

  it("does not dispatch a legacy catalog request after a same-target account switch", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: jwtForUser(84),
      refreshToken: "other-account-refresh"
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/service-prompts",
        method: "GET",
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual",
          expectedUserId: 42
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each([
    {
      name: "target",
      current: {
        serverUrl: "https://other.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: jwtForUser(42),
        refreshToken: "old-refresh"
      }
    },
    {
      name: "refresh lineage",
      current: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: jwtForUser(42),
        refreshToken: "other-refresh"
      }
    }
  ])("does not dispatch a captured worker refresh after $name drift", async ({ current }) => {
    storageState.persistent.set("tldwConfig", current)
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/auth/refresh",
        method: "POST",
        body: { refresh_token: "old-refresh" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual",
          expectedUserId: 42,
          expectedRefreshToken: "old-refresh"
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })

    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("refreshes a scoped worker request without dropping the checked user binding", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    const requestHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      requestHeaders.push(headers)
      if (headers.get("Authorization") === "Bearer stale-account-token") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "GET",
          headers: { "X-TLDW-Expected-User-ID": "42" },
          servicePromptConfig: {
            serverUrl: "https://api.example.test",
            authMode: "multi-user",
            authSource: "manual"
          }
        }
      })
    ).resolves.toMatchObject({ ok: true, status: 200 })

    expect(requestHeaders).toHaveLength(2)
    expect(
      requestHeaders.map((headers) =>
        headers.get("X-TLDW-Expected-User-ID")
      )
    ).toEqual(["42", "42"])
    expect(requestHeaders[1]?.get("Authorization")).toBe(
      "Bearer fresh-account-token"
    )
    expect(storageState.persistent.get("tldwConfig")).toEqual({
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    expect(storageState.persistent.get("tldwRefreshRotation")).toMatchObject({
      sourceRefreshToken: "refresh-token",
      accessToken: "fresh-account-token",
      refreshToken: "rotated-refresh-token"
    })
  })

  it("reuses an ordinary raw-config refresh that wins before scoped worker refresh starts", async () => {
    const staleAccess = jwtForUser(42)
    const winnerAccess = `${jwtForUser(42)}-rotated`
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: staleAccess,
      refreshToken: "refresh-token"
    })
    let refreshHits = 0
    const requestHeaders: Array<string | null> = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response("refresh should not run", { status: 500 })
      }
      const authorization = new Headers(init?.headers).get("Authorization")
      requestHeaders.push(authorization)
      if (authorization === `Bearer ${staleAccess}`) {
        storageState.persistent.set("tldwConfig", {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual",
          accessToken: winnerAccess,
          refreshToken: "winner-refresh-token"
        })
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "GET",
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })).resolves.toMatchObject({ ok: true, status: 200 })

    expect(refreshHits).toBe(0)
    expect(requestHeaders).toEqual([
      `Bearer ${staleAccess}`,
      `Bearer ${winnerAccess}`
    ])
  })

  it.each([
    {
      name: "target",
      replacement: {
        serverUrl: "https://other.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: "other-target-access",
        refreshToken: "other-target-refresh"
      }
    },
    {
      name: "account token",
      replacement: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: "other-account-access",
        refreshToken: "other-account-refresh"
      }
    }
  ])("discards a scoped worker refresh after the $name changes", async ({ replacement }) => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    vi.spyOn(tldwAuth, "refreshToken").mockImplementation(async () => {
      signalRefreshStarted()
      await refreshGate
      storageState.persistent.set("tldwConfig", {
        ...storageState.persistent.get("tldwConfig") as Record<string, unknown>,
        accessToken: "fresh-account-token",
        refreshToken: "rotated-refresh-token"
      })
      return {
        access_token: "fresh-account-token",
        refresh_token: "rotated-refresh-token",
        token_type: "bearer"
      }
    })
    const requestHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      requestHeaders.push(headers)
      if (headers.get("Authorization") === "Bearer stale-account-token") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    const result = expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "GET",
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })

    await refreshStarted
    storageState.persistent.set("tldwConfig", replacement)
    releaseRefresh()
    await result

    expect(storageState.persistent.get("tldwConfig")).toEqual(replacement)
    expect(requestHeaders).toHaveLength(1)
  })

  it("does not refresh a scoped worker request with a token changed by the initial request", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: jwtForUser(42),
      refreshToken: "refresh-token"
    })
    const replacement = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: jwtForUser(84),
      refreshToken: "other-account-refresh"
    }
    let refreshHits = 0
    let requestHits = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        refreshHits += 1
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      requestHits += 1
      if (requestHits === 1) {
        storageState.persistent.set("tldwConfig", replacement)
        return new Response("unauthorized", { status: 401 })
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "GET",
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual",
          expectedUserId: 42
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })

    expect(refreshHits).toBe(0)
    expect(requestHits).toBe(1)
    expect(storageState.persistent.get("tldwConfig")).toEqual(replacement)
  })

  it("fails a scoped worker request closed after same-target logout", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual"
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:request",
        payload: {
          path: "/api/v1/service-prompts/chat.rag.answer",
          method: "GET",
          headers: { "X-TLDW-Expected-User-ID": "42" },
          servicePromptConfig: {
            serverUrl: "https://api.example.test",
            authMode: "multi-user",
            authSource: "manual",
            accessToken: "checked-account-token"
          }
        }
      })
    ).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("aborts a started scoped worker request after target drift", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "current-account-token",
      refreshToken: "refresh-token"
    })
    let activeSignal: AbortSignal | undefined
    let resolveFetch!: (response: Response) => void
    let signalFetchStarted!: () => void
    const fetchStarted = new Promise<void>((resolve) => {
      signalFetchStarted = resolve
    })
    const fetchSpy = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
      await new Promise<Response>((resolve) => {
        activeSignal = init?.signal as AbortSignal | undefined
        resolveFetch = resolve
        signalFetchStarted()
      })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const requestId = "scoped-request-abort"
    const pending = sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        requestId,
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "PUT",
        body: { parts: {}, expected_revision: null },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })

    await fetchStarted
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://other.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "other-account-token",
      refreshToken: "other-refresh-token"
    })
    const cancelResponse = await sendRuntimeMessage({
      type: "tldw:cancel-request",
      payload: { requestId }
    })
    const wasAborted = activeSignal?.aborted
    resolveFetch(new Response(JSON.stringify({ persisted: true }), {
      status: 200,
      headers: { "content-type": "application/json" }
    }))
    const response = await pending
    const secondCancel = await sendRuntimeMessage({
      type: "tldw:cancel-request",
      payload: { requestId }
    })

    expect(cancelResponse).toMatchObject({ ok: true, cancelled: true })
    expect(wasAborted).toBe(true)
    expect(response).toMatchObject({ ok: false })
    expect(response).not.toMatchObject({ data: { persisted: true } })
    expect(secondCancel).toMatchObject({ ok: true, cancelled: false })
  })

  it.each([
    "/api/v1/chat/completions",
    "/api/v1/chats/chat-123/messages",
    "/api/v1/rag/search",
    "/api/v1/research/websearch"
  ])("allows a checked target for the exact worker route %s", async (path) => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path,
        method: "POST",
        headers: { "X-TLDW-Expected-User-ID": "42" },
        body: {},
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
        }
      }
    })).resolves.toMatchObject({ ok: true, status: 200 })

    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(`https://api.example.test${path}`)
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
  ] as const)("rejects a checked target on non-allowlisted worker route %s %s", async (path, method) => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path,
        method,
        body: {},
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
        }
      }
    })).resolves.toMatchObject({ ok: false, status: 400 })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("authenticates worker uploads with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(
      sendRuntimeMessage({
        type: "tldw:upload",
        payload: {
          path: "/api/v1/media/ingest/jobs",
          method: "POST",
          fields: { media_type: "document" }
        }
      })
    ).resolves.toMatchObject({ ok: true, status: 200 })

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("worker-session-key")
    expect(storageState.persistent.get("tldwConfig")).not.toHaveProperty("apiKey")
  })

  it("binds a worker upload to the checked target and current credentials", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "current-account-token",
      refreshToken: "refresh-token"
    })
    storageState.persistent.delete("tldwCookieSessionConfig")
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:upload",
      payload: {
        path: "/api/v1/media/add",
        method: "POST",
        fields: { media_type: "document", urls: ["https://example.com"] },
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })).resolves.toMatchObject({ ok: true, status: 200 })

    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(
      "https://api.example.test/api/v1/media/add"
    )
    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("Authorization")).toBe("Bearer current-account-token")
    expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
  })

  it("does not dispatch a worker upload after the checked target changes", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://other.example.test",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "other-key"
    })
    storageState.persistent.delete("tldwCookieSessionConfig")
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:upload",
      payload: {
        path: "/api/v1/media/add",
        method: "POST",
        fields: { media_type: "document", urls: ["https://example.com"] },
        headers: { "X-TLDW-Expected-User-ID": "42" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("does not upload after a same-target single-user API-key change", async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:upload",
      payload: {
        path: "/api/v1/media/add",
        method: "POST",
        fields: { media_type: "document", urls: ["https://example.com"] },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: "key:captured-account"
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("preserves an expected-user rejection from a scoped worker upload", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({
        detail: {
          code: "request_config_scope_changed",
          message: "Authenticated account changed"
        }
      }), {
        status: 412,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:upload",
      payload: {
        path: "/api/v1/media/add",
        method: "POST",
        fields: { media_type: "document", urls: ["https://example.com"] },
        headers: { "X-TLDW-Expected-User-ID": "99" },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 412,
      data: { detail: { code: "request_config_scope_changed" } }
    })
    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })

  it("aborts a started scoped worker upload after target drift", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "current-account-token",
      refreshToken: "refresh-token"
    })
    storageState.persistent.delete("tldwCookieSessionConfig")
    let activeSignal: AbortSignal | undefined
    let resolveFetch!: (response: Response) => void
    let signalFetchStarted!: () => void
    const fetchStarted = new Promise<void>((resolve) => {
      signalFetchStarted = resolve
    })
    const fetchSpy = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
      await new Promise<Response>((resolve) => {
        activeSignal = init?.signal as AbortSignal | undefined
        resolveFetch = resolve
        signalFetchStarted()
      })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const requestId = "scoped-upload-abort"
    const pending = sendRuntimeMessage({
      type: "tldw:upload",
      payload: {
        requestId,
        path: "/api/v1/media/add",
        method: "POST",
        fields: { urls: ["https://example.com"] },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })

    await fetchStarted
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://other.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "other-account-token",
      refreshToken: "other-refresh-token"
    })
    const cancelResponse = await sendRuntimeMessage({
      type: "tldw:cancel-request",
      payload: { requestId }
    })
    const wasAborted = activeSignal?.aborted
    resolveFetch(new Response(JSON.stringify({ persisted: true }), {
      status: 200,
      headers: { "content-type": "application/json" }
    }))
    const response = await pending
    const secondCancel = await sendRuntimeMessage({
      type: "tldw:cancel-request",
      payload: { requestId }
    })

    expect(cancelResponse).toMatchObject({ ok: true, cancelled: true })
    expect(wasAborted).toBe(true)
    expect(response).toMatchObject({ ok: false })
    expect(response).not.toMatchObject({ data: { persisted: true } })
    expect(secondCancel).toMatchObject({ ok: true, cancelled: false })
  })

  it("cleans up a completed cancelable worker request", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const requestId = "scoped-request-complete"

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        requestId,
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "GET",
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "manual",
          expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
        }
      }
    })).resolves.toMatchObject({ ok: true, status: 200 })

    await expect(sendRuntimeMessage({
      type: "tldw:cancel-request",
      payload: { requestId }
    })).resolves.toMatchObject({ ok: true, cancelled: false })
  })

  it("authenticates worker HTTP streams with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response('data: {"ok":true}\n\ndata: [DONE]\n\n', {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    void port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { stream: true }
    })
    await flushPromises()

    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("X-API-KEY")).toBe("worker-session-key")
    expect(port.messages).toContainEqual({ event: "done" })
    expect(storageState.persistent.get("tldwConfig")).not.toHaveProperty("apiKey")
  })

  it("isolates opted-in RAG stream errors from a concurrent public stream", async () => {
    let releaseBarrier: (() => void) | undefined
    const barrier = new Promise<void>((resolve) => {
      releaseBarrier = resolve
    })
    let started = 0
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      started += 1
      if (started === 2) releaseBarrier?.()
      await barrier

      if (String(input).includes("/api/v1/rag/search/stream")) {
        return new Response(
          JSON.stringify({
            detail: {
              error_code: "credential_scope_revoked",
              message: "RAW_WORKER_RAG_BODY",
              api_key: "RAW_WORKER_RAG_KEY",
              debug_path: "/RAW_WORKER_RAG_PATH/provider.json"
            },
            upstream_url: "https://RAW_WORKER_RAG_URL.example/v1"
          }),
          {
            status: 503,
            headers: { "content-type": "application/json" }
          }
        )
      }

      return new Response(
        JSON.stringify({
          detail: {
            message: "PUBLIC_STREAM_MESSAGE",
            public_detail: "PUBLIC_STREAM_DETAIL"
          }
        }),
        {
          status: 418,
          headers: { "content-type": "application/json" }
        }
      )
    })
    vi.stubGlobal("fetch", fetchSpy as any)
    const ragPort = connectRuntimePort("tldw:stream")
    const publicPort = connectRuntimePort("tldw:stream")

    ragPort.postMessage({
      path: "/api/v1/rag/search/stream",
      method: "POST",
      body: { query: "provider failure" },
      sanitizeRagProviderStreamError: true
    })
    publicPort.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { stream: true }
    })

    await vi.waitFor(() => {
      expect(
        ragPort.messages.some(
          (message: any) => message?.event === "error"
        )
      ).toBe(true)
      expect(
        publicPort.messages.some(
          (message: any) => message?.event === "error"
        )
      ).toBe(true)
    })

    const ragError = ragPort.messages.find(
      (message: any) => message?.event === "error"
    )
    const publicError = publicPort.messages.find(
      (message: any) => message?.event === "error"
    )
    expect(ragError).toMatchObject({
      event: "error",
      status: 503,
      code: "credential_scope_revoked",
      message:
        "The selected provider credential scope is no longer available.",
      details: {
        detail: {
          error_code: "credential_scope_revoked",
          message:
            "The selected provider credential scope is no longer available."
        }
      }
    })
    expect(JSON.stringify(ragError)).not.toMatch(
      /RAW_WORKER_RAG_(?:BODY|KEY|PATH|URL)/
    )
    expect(publicError).toMatchObject({
      event: "error",
      status: 418,
      message: "PUBLIC_STREAM_MESSAGE",
      details: {
        detail: {
          message: "PUBLIC_STREAM_MESSAGE",
          public_detail: "PUBLIC_STREAM_DETAIL"
        }
      }
    })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("does not dispatch a worker stream after the checked target changes", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://other.example.test",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "other-key"
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        authSource: "manual"
      }
    })
    await flushPromises()

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(port.messages).toContainEqual(expect.objectContaining({
      event: "error",
      status: 412
    }))
  })

  it("does not stream after a same-target single-user API-key change", async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")
    const capturedScope = deriveSingleUserApiKeyCredentialScope(
      "single-user",
      "captured-account-key"
    )

    port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        authSource: "manual",
        expectedSingleUserApiKeyScope: capturedScope!
      }
    })

    await vi.waitFor(() => {
      expect(port.messages).toContainEqual(expect.objectContaining({
        event: "error",
        status: 412
      }))
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each([
    "/api/v1/chat/completions/extra",
    "/api/v1/chats/%2e%2e/messages",
    "/api/v1/chats/chat%2fid/messages"
  ])("rejects a checked target on non-allowlisted worker stream route %s", async (path) => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
      path,
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        authSource: "manual"
      }
    })
    await flushPromises()

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(port.messages).toContainEqual(expect.objectContaining({
      event: "error",
      status: 400
    }))
  })

  it("binds a worker stream to the checked target and expected user", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "current-account-token",
      refreshToken: "refresh-token"
    })
    const fetchSpy = vi.fn(async () =>
      new Response("data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual"
      }
    })
    await flushPromises()

    expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(
      "https://api.example.test/api/v1/chat/completions"
    )
    const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
    expect(headers.get("Authorization")).toBe("Bearer current-account-token")
    expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
    expect(port.messages).toContainEqual({ event: "done" })
  })

  it("refreshes a checked worker stream without dropping the expected user", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    const streamHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      streamHeaders.push(headers)
      if (headers.get("Authorization") === "Bearer stale-account-token") {
        return new Response("unauthorized", { status: 401 })
      }
      return new Response("data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      headers: { "X-TLDW-Expected-User-ID": "42" },
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual"
      }
    })
    await flushPromises()

    expect(streamHeaders).toHaveLength(2)
    expect(streamHeaders.map((headers) =>
      headers.get("X-TLDW-Expected-User-ID")
    )).toEqual(["42", "42"])
    expect(streamHeaders[1]?.get("Authorization")).toBe("Bearer fresh-account-token")
    expect(port.messages).toContainEqual({ event: "done" })
  })

  it("does not retry a scoped worker stream after disconnect during refresh", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    const streamSignals: AbortSignal[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      streamSignals.push(init?.signal as AbortSignal)
      return new Response("unauthorized", { status: 401 })
    })
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    const streamCompleted = port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual"
      }
    })

    await refreshStarted
    port.disconnect()
    expect(streamSignals[0]?.aborted).toBe(true)
    releaseRefresh()
    await streamCompleted

    expect(streamSignals).toHaveLength(1)
  })

  it("discards a checked worker stream refresh after the account token changes", async () => {
    storageState.persistent.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "stale-account-token",
      refreshToken: "refresh-token"
    })
    let signalRefreshStarted!: () => void
    let releaseRefresh!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    vi.spyOn(tldwAuth, "refreshToken").mockImplementation(async () => {
      signalRefreshStarted()
      await refreshGate
      storageState.persistent.set("tldwConfig", {
        ...storageState.persistent.get("tldwConfig") as Record<string, unknown>,
        accessToken: "fresh-account-token",
        refreshToken: "rotated-refresh-token"
      })
      return {
        access_token: "fresh-account-token",
        refresh_token: "rotated-refresh-token",
        token_type: "bearer"
      }
    })
    const streamHeaders: Headers[] = []
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/v1/auth/refresh")) {
        signalRefreshStarted()
        await refreshGate
        return new Response(JSON.stringify({
          access_token: "fresh-account-token",
          refresh_token: "rotated-refresh-token"
        }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      const headers = new Headers(init?.headers)
      streamHeaders.push(headers)
      return new Response("unauthorized", { status: 401 })
    })
    vi.stubGlobal("fetch", fetchSpy)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: { stream: true },
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual"
      }
    })
    await refreshStarted
    const replacement = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "other-account-access",
      refreshToken: "other-account-refresh"
    }
    storageState.persistent.set("tldwConfig", replacement)
    releaseRefresh()
    await flushPromises()

    expect(storageState.persistent.get("tldwConfig")).toEqual(replacement)
    expect(streamHeaders).toHaveLength(1)
    expect(port.messages).toContainEqual(expect.objectContaining({
      event: "error",
      status: 412,
      details: expect.objectContaining({
        detail: expect.objectContaining({
          code: "request_config_scope_changed"
        })
      })
    }))
  })
})
