import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const runtimeSendMessage = vi.hoisted(() => vi.fn(async () => ({ handled: true })))

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
  get: vi.fn(async (_key: string): Promise<void> => undefined),
  set: vi.fn(async (_key: string, _value: unknown): Promise<void> => undefined)
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
    deserialize: (value: unknown) => value,
    // The physical map stores structured values, as chrome.storage permits.
    // Account events invoke the production serde names, not the legacy aliases.
    serializer: (value: unknown) => value,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: (options?: { area?: string }) => {
    const values =
      options?.area === "session" ? storageState.session : storageState.persistent
    return {
      get: async (key: string) => {
        await storageState.get(key)
        return values.get(key)
      },
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
      sendMessage: runtimeSendMessage,
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

type RuntimeStreamMessage = Record<string, unknown>

type RuntimePortHarness = {
  messages: RuntimeStreamMessage[]
  postMessage: (message: unknown) => Promise<void>
  disconnect: () => void
}

const connectRuntimePort = (name: string): RuntimePortHarness => {
  const inboundListeners = new Set<(message: unknown) => unknown>()
  const disconnectListeners = new Set<() => void>()
  const messages: RuntimeStreamMessage[] = []
  const port = {
    name,
    sender: { id: "extension-id" },
    postMessage: (message: RuntimeStreamMessage) => messages.push(message),
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
import { deriveConnectionAuthorityId, deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { browser, type Browser } from "wxt/browser"

const contextMenuTab: Browser.tabs.Tab = {
  id: 8,
  index: 0,
  pinned: false,
  highlighted: true,
  windowId: 1,
  active: true,
  frozen: false,
  incognito: false,
  selected: true,
  discarded: false,
  autoDiscardable: true,
  groupId: -1
}

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
    storageState.get.mockReset().mockResolvedValue(undefined)
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
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (windowDescriptor) {
      Object.defineProperty(globalThis, "window", windowDescriptor)
    } else {
      delete (globalThis as any).window
    }
  })

  it.each([
    ["save-to-notes-pa", "save-to-notes"],
    ["save-to-companion-pa", "save-to-companion"],
    ["narrate-selection-pa", "narrate-selection"],
  ])(
    "opens %s during its menu gesture before delayed delivery",
    async (menuItemId, type) => {
      vi.useFakeTimers()
      const open = vi.fn().mockResolvedValue(undefined)
      vi.stubGlobal("chrome", {
        sidePanel: { setOptions: vi.fn().mockResolvedValue(undefined), open },
        notifications: { create: vi.fn() },
      })
      runtimeSendMessage
        .mockClear()
        .mockResolvedValue({ handled: true })
      const calls = vi.mocked(browser.contextMenus.onClicked.addListener).mock
        .calls
      const onClick = calls[calls.length - 1][0]

      await onClick(
        {
          menuItemId,
          editable: false,
          selectionText: "Selected source",
          pageUrl: "https://example.com/story",
        },
        { ...contextMenuTab, title: "Story" },
      )

      expect(open).toHaveBeenCalledWith({ tabId: 8 })
      expect(browser.runtime.sendMessage).not.toHaveBeenCalled()
      await vi.advanceTimersByTimeAsync(5000)
      expect(browser.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type, text: "Selected source" }),
      )
    },
  )

  it("handles sidebar denial before the delayed Notes delivery without an unhandled rejection", async () => {
    vi.useFakeTimers()
    const open = vi.fn().mockRejectedValue(new Error("gesture required"))
    const notify = vi.fn()
    vi.stubGlobal("chrome", {
      sidePanel: { open },
      notifications: { create: notify },
    })
    vi.mocked(browser.runtime.sendMessage).mockClear()
    const calls = vi.mocked(browser.contextMenus.onClicked.addListener).mock
      .calls

    await calls[calls.length - 1][0](
      { menuItemId: "save-to-notes-pa", editable: false, selectionText: "Selected source" },
      contextMenuTab,
    )

    expect(open).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(5000)
    expect(browser.runtime.sendMessage).not.toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ message: "contextSaveToNotesDeliveryFailed" }),
    )
  })

  it("registers runtime messaging when the optional commands API is unavailable", async () => {
    await expect(sendRuntimeMessage({ type: "tldw:ping" })).resolves.toMatchObject({
      ok: true,
      pong: true
    })
  })

  it.each(["request", "stream"])("refreshes an expired worker %s without messaging itself", async (transport) => {
    const accessToken = jwtForUser(42)
    const rotatedAccessToken = `header.${btoa(JSON.stringify({ sub: "42", nonce: 2 }))}.signature`
    const config = {
      serverUrl: "https://api.example.test", authMode: "multi-user", authSource: "manual",
      accessToken, refreshToken: "valid-refresh"
    }
    storageState.persistent.set("tldwConfig", config)
    storageState.persistent.delete("tldwCookieSessionConfig")
    // A runtime message does not loop back into its sending worker's listener.
    vi.mocked(browser.runtime.sendMessage).mockResolvedValue(undefined)
    const requests: { path: string; authorization: string | null; body?: BodyInit | null }[] = []
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname
      const authorization = new Headers(init?.headers).get("Authorization")
      requests.push({ path, authorization, body: init?.body })
      if (path === "/api/v1/auth/refresh") return new Response(JSON.stringify({
        access_token: rotatedAccessToken, refresh_token: "rotated-refresh", token_type: "bearer"
      }), { headers: { "content-type": "application/json" } })
      if (authorization !== `Bearer ${rotatedAccessToken}`) return new Response("expired", { status: 401 })
      return transport === "stream"
        ? new Response("data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } })
        : new Response(JSON.stringify({ id: 42 }), { headers: { "content-type": "application/json" } })
    }))
    if (transport === "request") {
      await expect(sendRuntimeMessage({
        type: "tldw:request", payload: { path: "/api/v1/auth/me", method: "GET" }
      })).resolves.toMatchObject({ ok: true, status: 200, data: { id: 42 } })
    } else {
      const port = connectRuntimePort("tldw:stream")
      await port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: { stream: true } })
      expect(port.messages).toContainEqual({ event: "done" })
    }
    expect(requests.map(({ path }) => path)).toEqual([
      transport === "stream" ? "/api/v1/chat/completions" : "/api/v1/auth/me",
      "/api/v1/auth/refresh",
      transport === "stream" ? "/api/v1/chat/completions" : "/api/v1/auth/me"
    ])
    expect(JSON.parse(String(requests[1].body))).toEqual({ refresh_token: "valid-refresh" })
    expect(requests[2].authorization).toBe(`Bearer ${rotatedAccessToken}`)
    // Rotation uses the existing separate record, leaving the connection intact.
    expect(storageState.persistent.get("tldwConfig")).toEqual(config)
    await tldwClient.initialize()
    expect(await tldwClient.getConfig()).toMatchObject({
      ...config, accessToken: rotatedAccessToken, refreshToken: "rotated-refresh"
    })
  })

  it.each([false, true])("preserves a transient worker refresh failure without replaying a write (scoped=%s)", async (scoped) => {
    const config = {
      serverUrl: "https://api.example.test", authMode: "multi-user", authSource: "manual",
      accessToken: jwtForUser(42), refreshToken: "valid-refresh"
    }
    storageState.persistent.set("tldwConfig", config)
    storageState.persistent.delete("tldwCookieSessionConfig")
    if (!scoped) {
      vi.spyOn(tldwAuth, "refreshToken").mockRejectedValue(Object.assign(
        new Error("Authentication service is busy"), { status: 503, retryAfterMs: 2000 }
      ))
    }
    const writes: string[] = []
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith("/auth/refresh")) return new Response(JSON.stringify({ detail: "Authentication service is busy" }), {
        status: 503, headers: { "content-type": "application/json", "retry-after": "2" }
      })
      writes.push(url)
      return new Response("unauthorized", { status: 401 })
    }))
    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/notes/", method: "POST", body: { content: "Private draft" },
        ...(scoped ? { servicePromptConfig: {
          serverUrl: config.serverUrl, authMode: "multi-user", authSource: "manual", expectedUserId: 42
        } } : {})
      }
    })).resolves.toMatchObject({ ok: false, status: 503, retryAfterMs: 2000 })
    expect(writes).toHaveLength(1)
    expect(storageState.persistent.get("tldwConfig")).toEqual(config)
  })

  it("preserves a rejected scoped refresh with a malformed matching org ID without invalidating credentials", async () => {
    const config = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      orgId: "7",
      accessToken: jwtForUser(42),
      refreshToken: "expired-refresh"
    }
    storageState.persistent.set("tldwConfig", config)
    storageState.persistent.delete("tldwCookieSessionConfig")
    const persistentBefore = new Map(storageState.persistent)
    const sessionBefore = new Map(storageState.session)
    const fetchSpy = vi.fn<typeof fetch>(async () =>
      new Response("unauthorized", { status: 401 })
    )
    vi.stubGlobal("fetch", fetchSpy)

    await expect(sendRuntimeMessage({
      type: "tldw:request",
      payload: {
        path: "/api/v1/service-prompts/chat.rag.answer",
        method: "GET",
        servicePromptConfig: {
          serverUrl: config.serverUrl,
          authMode: "multi-user",
          authSource: "manual",
          orgId: "7",
          expectedUserId: 42
        }
      }
    })).resolves.toMatchObject({
      ok: false,
      status: 401,
      error: "Session expired. Please log in again."
    })
    expect(fetchSpy.mock.calls.map(([input]) => new URL(String(input)).pathname)).toEqual([
      "/api/v1/service-prompts/chat.rag.answer",
      "/api/v1/auth/refresh"
    ])
    expect(storageState.persistent).toEqual(persistentBefore)
    expect(storageState.session).toEqual(sessionBefore)
    expect(storageState.set).not.toHaveBeenCalled()
  })

  it("invalidates a rejected scoped worker refresh in shared effective credentials", async () => {
    const config = {
      serverUrl: "https://api.example.test", authMode: "multi-user", authSource: "manual",
      accessToken: jwtForUser(42), refreshToken: "expired-refresh"
    }
    storageState.persistent.set("tldwConfig", config)
    storageState.persistent.delete("tldwCookieSessionConfig")
    vi.stubGlobal("fetch", vi.fn(async () => new Response("unauthorized", { status: 401 })))
    await expect(sendRuntimeMessage({
      type: "tldw:request", payload: {
        path: "/api/v1/notes/private-note", method: "GET",
        servicePromptConfig: { serverUrl: config.serverUrl, authMode: "multi-user", authSource: "manual", expectedUserId: 42 }
      }
    })).resolves.toMatchObject({ ok: false, status: 401 })
    const { resolveEffectiveTldwConfig } = await import("@/services/tldw/single-user-credential")
    const { createSafeStorage } = await import("@/utils/safe-storage")
    expect(await resolveEffectiveTldwConfig({
      persistent: createSafeStorage({ area: "local" }), session: createSafeStorage({ area: "session" })
    })).not.toHaveProperty("refreshToken")
    expect(storageState.persistent.get("tldwConfig")).toEqual(config)
  })

  it("authenticates ordinary worker requests with session credentials", async () => {
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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

  it.each([
    ["/api/v1/service-prompts/chat.rag.answer", "PUT"],
    ["/api/v1/media/ingest-web-content", "POST"]
  ])("aborts a started scoped worker request %s after target drift", async (path, method) => {
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
        path,
        method,
        body: { parts: {}, expected_revision: null },
        servicePromptConfig: {
          serverUrl: "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual"
        }
      }
    })

    await Promise.race([
      fetchStarted,
      pending.then((response) => {
        if (!response.ok) throw new Error(response.error)
      })
    ])
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    ["/api/v1/chats/chat-123/messages/search", "GET"]
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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
    const fetchSpy = vi.fn<typeof fetch>(async () =>
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

  describe("TASK-13549 worker original recipe settlement authority", () => {
    const original = {
      serverUrl: "https://api.example.test", authMode: "single-user" as const, authSource: "manual" as const,
      apiKey: "original-worker-recipe-key", credentialSource: "manual" as const, apiKeyPersistence: "device" as const,
      apiKeyServerOrigin: "https://api.example.test"
    }
    it.each(["retired account", "uninterrupted"] as const)(
      "retains the actual admitted recipe owner and exact ACK across %s", async boundary => {
        storageState.persistent.set("tldwConfig", structuredClone(original))
        storageState.persistent.delete("tldwCookieSessionConfig")
        storageState.session.clear()
        const registrations = vi.mocked(browser.storage.onChanged.addListener).mock.calls
        const accountChanged = registrations[registrations.length - 1]?.[0]
        if (!accountChanged) throw new Error("Actual worker account observer missing")
        const owner = await sendRuntimeMessage({ type: "tldw:recipe-owner:resolve" })
        expect(owner).toMatchObject({ ownerId: expect.stringMatching(/^recipe-owner:sha256:[0-9a-f]{64}$/) })
        const { CLEAR_TASK_RECIPE } = await import("@/components/Common/PromptAssist/recipes/built-in-recipes")
        const body = { project_id: 42, name: "Original worker recipe", prompt_format: "structured",
          prompt_schema_version: 2, prompt_definition: CLEAR_TASK_RECIPE.definition }
        const localId = "original-worker-recipe-" + boundary
        let releaseResponse!: () => void
        const heldResponse = new Promise<void>(resolve => { releaseResponse = resolve })
        const requests: { url: string; method?: string; apiKey: string | null; body: unknown; signal: AbortSignal | null }[] = []
        const answer = { data: { id: 101, ...body } }
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/prompt-studio/prompts/create") throw new Error("Undeclared worker recipe transport: " + url)
          requests.push({ url, method: init?.method, apiKey: new Headers(init?.headers).get("X-API-KEY"),
            body: JSON.parse(String(init?.body)), signal: init?.signal ?? null })
          await heldResponse
          return Response.json(answer)
        }))
        const pending = sendRuntimeMessage({ type: "tldw:request", payload: {
          path: "/api/v1/prompt-studio/prompts/create", method: "POST", body,
          recipePersistence: { mode: "require", expectedOwnerId: owner.ownerId, localId }
        } })
        try {
          await vi.waitFor(() => expect(requests).toHaveLength(1))
          expect(requests[0]).toMatchObject({ url: original.serverUrl + "/api/v1/prompt-studio/prompts/create",
            method: "POST", apiKey: original.apiKey, body })
          // Observe the real registry reservation before touching account state.
          expect(await sendRuntimeMessage({ type: "tldw:recipe-uncertainty:read", id: localId, ownerId: owner.ownerId })).toBe("unknown_owner")
          if (boundary === "retired account") {
            const replacement = { ...original, apiKey: "replacement-worker-recipe-key" }
            storageState.persistent.set("tldwConfig", replacement)
            accountChanged({ tldwConfig: { oldValue: original, newValue: replacement } }, "local")
            expect(requests[0].signal?.aborted).toBe(true)
          } else expect(requests[0].signal?.aborted).toBe(false)
          releaseResponse()
          const response = await pending
          expect(response).toMatchObject({
            ok: boundary === "uninterrupted", status: boundary === "uninterrupted" ? 200 : 412,
            recipePersistence: { state: "dispatched", actualOwnerId: owner.ownerId },
            recipeDelivery: { id: localId, ownerId: owner.ownerId, operationId: expect.any(String) }
          })
          if (boundary === "uninterrupted") expect(response.data).toEqual(answer)
          else expect(response.data).toMatchObject({ detail: { code: "request_config_scope_changed" } })
          const { isRecipeDeliveryReceipt } = await import("@/services/recipe-persistence-uncertainty")
          expect(isRecipeDeliveryReceipt(response.recipeDelivery)).toBe(true)
          // An invented UUID/owner cannot discharge this worker's exact operation.
          expect(await sendRuntimeMessage({ type: "tldw:recipe-uncertainty:acknowledge",
            ...response.recipeDelivery, operationId: crypto.randomUUID() })).toEqual({ ok: false })
          expect(await sendRuntimeMessage({ type: "tldw:recipe-uncertainty:read", id: localId, ownerId: owner.ownerId })).toBe("unknown_owner")
          expect(await sendRuntimeMessage({ type: "tldw:recipe-uncertainty:acknowledge", ...response.recipeDelivery })).toEqual({ ok: true })
          expect(await sendRuntimeMessage({ type: "tldw:recipe-uncertainty:read", id: localId, ownerId: owner.ownerId })).toBe("scoped")
          expect(requests).toHaveLength(1)
        } finally {
          releaseResponse()
          await pending
        }
      }
    )
  })

  describe("TASK-13549 worker terminal marker publication lifetime", () => {
    const original = {
      serverUrl: "https://api.example.test", authMode: "multi-user" as const, authSource: "manual" as const,
      accessToken: jwtForUser(42), refreshToken: "original-worker-expired-refresh"
    }
    const configureOriginal = async () => {
      storageState.persistent.set("tldwConfig", structuredClone(original))
      storageState.persistent.delete("tldwCookieSessionConfig")
      storageState.session.clear()
      const registrations = vi.mocked(browser.storage.onChanged.addListener).mock.calls
      const accountChanged = registrations[registrations.length - 1]?.[0]
      if (!accountChanged) throw new Error("Actual worker account observer missing")
      const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
      const markerKey = refreshSessionInvalidationKey(original)
      if (!markerKey) throw new Error("Actual original session marker unavailable")
      return { accountChanged, markerKey }
    }
    it.each(["actual storage event", "unreported storage write"] as const)(
      "preserves its own real terminal 401 with %s", async publication => {
        const { accountChanged, markerKey } = await configureOriginal()
        const initialSet = storageState.set.getMockImplementation()
        const published: { key: string; oldValue: unknown; newValue: unknown }[] = []
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          if (key !== markerKey) return
          const oldValue = storageState.persistent.get(key)
          // Model physical persistence followed by its real chrome storage event.
          storageState.persistent.set(key, value)
          published.push({ key, oldValue, newValue: value })
          if (publication === "actual storage event") accountChanged({ [key]: { oldValue, newValue: value } }, "local")
        })
        const requests: { url: string; method?: string; authorization: string | null; body: unknown }[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input), path = new URL(url).pathname
          if (path !== "/api/v1/notes/private-note" && path !== "/api/v1/auth/refresh") throw new Error("Undeclared terminal transport: " + url)
          requests.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
            body: init?.body === undefined ? undefined : JSON.parse(String(init.body)) })
          return Response.json({ detail: path === "/api/v1/auth/refresh" ? "Original refresh session revoked" : "Original access expired" }, { status: 401 })
        }))
        try {
          const response = await sendRuntimeMessage({ type: "tldw:request", payload: {
            path: "/api/v1/notes/private-note", method: "GET", servicePromptConfig: {
              serverUrl: original.serverUrl, authMode: original.authMode, authSource: original.authSource, expectedUserId: 42
            }
          } })
          expect(published).toEqual([{ key: markerKey, oldValue: undefined, newValue: true }])
          expect(storageState.persistent.get(markerKey)).toBe(true)
          expect(response).toMatchObject({ ok: false, status: 401, error: "Session expired. Please log in again." })
          expect(requests).toEqual([
            { url: original.serverUrl + "/api/v1/notes/private-note", method: "GET", authorization: "Bearer " + original.accessToken, body: undefined },
            { url: original.serverUrl + "/api/v1/auth/refresh", method: "POST", authorization: null,
              body: { refresh_token: original.refreshToken } }
          ])
          const { resolveEffectiveTldwConfig } = await import("@/services/tldw/single-user-credential")
          const { createSafeStorage } = await import("@/utils/safe-storage")
          const effective = await resolveEffectiveTldwConfig({
            persistent: createSafeStorage({ area: "local" }), session: createSafeStorage({ area: "session" })
          })
          expect(effective).not.toHaveProperty("accessToken")
          expect(effective).not.toHaveProperty("refreshToken")
          expect(storageState.persistent.get("tldwConfig")).toEqual(original)
        } finally {
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
        }
      }
    )
    it.each(["external terminal marker", "uninterrupted"] as const)(
      "guards an unrelated admitted request across %s", async boundary => {
        const { accountChanged, markerKey } = await configureOriginal()
        const initialSet = storageState.set.getMockImplementation()
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          const oldValue = storageState.persistent.get(key)
          storageState.persistent.set(key, value)
          if (key === markerKey) accountChanged({ [key]: { oldValue, newValue: value } }, "local")
        })
        let releaseResponse!: () => void
        const heldResponse = new Promise<void>(resolve => { releaseResponse = resolve })
        const requests: { url: string; signal: AbortSignal | null; authorization: string | null }[] = []
        const answer = { note: "Original admitted private note" }
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/notes/private-note") throw new Error("Undeclared unrelated transport: " + url)
          requests.push({ url, signal: init?.signal ?? null, authorization: new Headers(init?.headers).get("Authorization") })
          await heldResponse
          return Response.json(answer)
        }))
        const pending = sendRuntimeMessage({ type: "tldw:request", payload: { path: "/api/v1/notes/private-note", method: "GET" } })
        try {
          await vi.waitFor(() => expect(requests).toHaveLength(1))
          expect(requests[0].authorization).toBe("Bearer " + original.accessToken)
          if (boundary === "external terminal marker") {
            const { createSafeStorage } = await import("@/utils/safe-storage")
            await createSafeStorage({ area: "local" }).set(markerKey, true)
            expect(requests[0].signal?.aborted).toBe(true)
          } else expect(requests[0].signal?.aborted).toBe(false)
          releaseResponse()
          const response = await pending
          if (boundary === "external terminal marker") {
            expect(response).toMatchObject({ ok: false, status: 412, data: { detail: { code: "request_config_scope_changed" } } })
            expect(response.data).not.toEqual(answer)
          } else expect(response).toMatchObject({ ok: true, status: 200, data: answer })
          expect(requests).toHaveLength(1)
        } finally {
          releaseResponse()
          await pending
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
        }
      }
    )
  })

  describe("TASK-13549 worker terminal publication consumers", () => {
    const original = {
      serverUrl: "https://api.example.test", authMode: "multi-user" as const, authSource: "manual" as const,
      accessToken: jwtForUser(42), refreshToken: "original-terminal-consumer-refresh"
    }
    const streamBody = { stream: true, model: "original-terminal-model",
      messages: [{ role: "user", content: "Original private terminal stream question" }] }
    const terminalMessage = "Session expired. Please log in again."
    const configureOriginal = async () => {
      storageState.persistent.set("tldwConfig", structuredClone(original))
      storageState.persistent.delete("tldwCookieSessionConfig")
      storageState.session.clear()
      const registrations = vi.mocked(browser.storage.onChanged.addListener).mock.calls
      const accountChanged = registrations[registrations.length - 1]?.[0]
      if (!accountChanged) throw new Error("Actual worker account observer missing")
      const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
      const markerKey = refreshSessionInvalidationKey(original)
      if (!markerKey) throw new Error("Actual terminal consumer marker unavailable")
      return { accountChanged, markerKey }
    }
    it.each(["own terminal marker", "external terminal marker"] as const)(
      "publishes only the original stream outcome for %s", async boundary => {
        const { accountChanged, markerKey } = await configureOriginal()
        const initialSet = storageState.set.getMockImplementation()
        const writes: { key: string; value: unknown }[] = []
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          const oldValue = storageState.persistent.get(key)
          storageState.persistent.set(key, value)
          writes.push({ key, value })
          if (key === markerKey) accountChanged({ [key]: { oldValue, newValue: value } }, "local")
        })
        let releaseInitial!: () => void
        const heldInitial = new Promise<void>(resolve => { releaseInitial = resolve })
        const requests: { path: string; method?: string; authorization: string | null; body: unknown; signal: AbortSignal | null }[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const path = new URL(String(input)).pathname
          if (path !== "/api/v1/chat/completions" && path !== "/api/v1/auth/refresh") throw new Error("Undeclared terminal stream: " + path)
          requests.push({ path, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
            body: JSON.parse(String(init?.body)), signal: init?.signal ?? null })
          if (path === "/api/v1/chat/completions") await heldInitial
          return Response.json({ detail: path === "/api/v1/auth/refresh" ? "Original session revoked" : "Original access expired" }, { status: 401 })
        }))
        const port = connectRuntimePort("tldw:stream")
        const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
        try {
          await vi.waitFor(() => expect(requests).toHaveLength(1))
          expect(requests[0]).toMatchObject({ path: "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(port.messages).toEqual([])
          if (boundary === "external terminal marker") {
            const { createSafeStorage } = await import("@/utils/safe-storage")
            await createSafeStorage({ area: "local" }).set(markerKey, true)
            expect(requests[0].signal?.aborted).toBe(true)
          }
          releaseInitial()
          await pending
          if (boundary === "own terminal marker") {
            expect(port.messages).toEqual([expect.objectContaining({ event: "error", status: 401, message: terminalMessage })])
            expect(requests.map(request => request.path)).toEqual(["/api/v1/chat/completions", "/api/v1/auth/refresh"])
            expect(requests[1]).toMatchObject({ method: "POST", authorization: null,
              body: { refresh_token: original.refreshToken } })
          } else {
            expect(port.messages).toEqual([expect.objectContaining({ event: "error", status: 412,
              details: { detail: { code: "request_config_scope_changed", message: expect.any(String) } } })])
            expect(requests).toHaveLength(1)
          }
          expect(writes).toEqual([{ key: markerKey, value: true }])
          expect(port.messages.some((message) => message?.event === "open" || message?.event === "data" || message?.event === "done")).toBe(false)
          expect(storageState.persistent.get("tldwConfig")).toEqual(original)
        } finally {
          releaseInitial()
          await pending
          port.disconnect()
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
        }
      }
    )
    describe("TASK-13549 worker stream refresh settlement", () => {
      const rotatedAccess = `header.${btoa(JSON.stringify({ sub: "42", nonce: "original-terminal-consumer-rotation" }))}.signature`
      const rotatedRefresh = "original-terminal-consumer-rotated-refresh"
      const frame = '{"choices":[{"delta":{"content":"Original rotated stream answer"}}]}'
      const refreshFailure = "Original authentication service unavailable"
      const refreshRetryAfter = "7"
      it.each([
        { outcome: "terminal 401 with own event held through settlement", status: 401 },
        { outcome: "nonterminal 502 without replay", status: 502 },
        { outcome: "successful same-principal rotation", status: 200 }
      ] as const)("settles $outcome at the actual original stream boundary", async ({ status }) => {
        const { accountChanged, markerKey } = await configureOriginal()
        const epoch = await sendRuntimeMessage({ type: "tldw:connection-authority",
          payload: { expectedConnectionAuthority: deriveConnectionAuthorityId(original) } })
        expect(epoch).toMatchObject({ ok: true, epoch: expect.any(String) })
        const initialSet = storageState.set.getMockImplementation()
        let delayedChange: { oldValue: unknown; newValue: unknown } | undefined
        let delivered = false
        const writes: { key: string; value: unknown }[] = []
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          const oldValue = storageState.persistent.get(key)
          storageState.persistent.set(key, value)
          writes.push({ key, value })
          if (key === markerKey) {
            // The actual write resolves, but its real account event stays queued
            // until this port's original consumer has completely settled.
            delayedChange = { oldValue, newValue: value }
          } else accountChanged({ [key]: { oldValue, newValue: value } }, "local")
        })
        let releaseRefresh!: () => void
        const heldRefresh = new Promise<void>(resolve => { releaseRefresh = resolve })
        const physical: { url: string; method?: string; authorization: string | null;
          command: string | null; body: unknown; signal: AbortSignal | null }[] = []
        let mainRequests = 0
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input), path = new URL(url).pathname
          if (path !== "/api/v1/chat/completions" && path !== "/api/v1/auth/refresh") throw new Error("Undeclared refresh settlement transport: " + url)
          const headers = new Headers(init?.headers)
          physical.push({ url, method: init?.method, authorization: headers.get("Authorization"),
            command: headers.get("X-Original-Command"), body: JSON.parse(String(init?.body)), signal: init?.signal ?? null })
          if (path === "/api/v1/auth/refresh") {
            await heldRefresh
            return status === 200
              ? Response.json({ access_token: rotatedAccess, refresh_token: rotatedRefresh, token_type: "bearer" })
              : Response.json({ detail: status === 401 ? "Original refresh session revoked" : refreshFailure }, {
                status, ...(status === 502 ? { headers: { "retry-after": refreshRetryAfter } } : {})
              })
          }
          mainRequests += 1
          if (mainRequests === 1) return Response.json({ detail: "Original access expired" }, { status: 401 })
          // A replay is a declared physical boundary, so a broken consumer settles
          // and fails the no-replay assertion instead of hiding behind a timeout.
          return new Response("data: " + frame + "\n\ndata: [DONE]\n\n", {
            headers: { "content-type": "text/event-stream" }
          })
        }))
        const port = connectRuntimePort("tldw:stream")
        const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody,
          headers: { "X-Original-Command": "original-refresh-settlement-command" } })
        try {
          await vi.waitFor(() => expect(physical).toHaveLength(2))
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, command: "original-refresh-settlement-command", body: streamBody })
          expect(physical[1]).toMatchObject({ url: original.serverUrl + "/api/v1/auth/refresh", method: "POST",
            authorization: null, command: null, body: { refresh_token: original.refreshToken } })
          expect(physical[0].signal?.aborted).toBe(false)
          expect(physical[1].signal?.aborted).toBe(false)
          expect(writes).toEqual([])
          expect(port.messages).toEqual([])
          releaseRefresh()
          await pending
          const expectedPaths = ["/api/v1/chat/completions", "/api/v1/auth/refresh",
            ...(status === 200 ? ["/api/v1/chat/completions"] : [])]
          expect(physical.map(request => new URL(request.url).pathname)).toEqual(expectedPaths)
          if (status === 200) {
            expect(physical[2]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
              authorization: "Bearer " + rotatedAccess, command: "original-refresh-settlement-command", body: streamBody })
            expect(port.messages).toEqual([{ event: "open" }, { event: "data", data: frame }, { event: "done" }])
            expect(storageState.persistent.get("tldwRefreshRotation")).toMatchObject({
              sourceAccessToken: original.accessToken, sourceRefreshToken: original.refreshToken,
              accessToken: rotatedAccess, refreshToken: rotatedRefresh
            })
            expect(storageState.persistent.get(markerKey)).toBeUndefined()
            expect(delayedChange).toBeUndefined()
          } else {
            expect(port.messages).toEqual([expect.objectContaining({ event: "error", status,
              message: status === 401 ? terminalMessage : refreshFailure,
              ...(status === 502 ? { retryAfter: refreshRetryAfter } : {}) })])
            if (status === 502) expect(port.messages[0]).not.toHaveProperty("headers")
            expect(port.messages.some((message) => message?.event === "open" || message?.event === "data" || message?.event === "done")).toBe(false)
            expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
            if (status === 401) {
              expect(writes).toEqual([{ key: markerKey, value: true }])
              expect(storageState.persistent.get(markerKey)).toBe(true)
              if (!delayedChange) throw new Error("Actual delayed terminal marker write missing")
              expect(delivered).toBe(false)
              const settledMessages = [...port.messages]
              delivered = true
              accountChanged({ [markerKey]: delayedChange }, "local")
              expect(port.messages).toEqual(settledMessages)
            } else {
              expect(writes).toEqual([])
              expect(delayedChange).toBeUndefined()
              expect(storageState.persistent.get(markerKey)).toBeUndefined()
            }
          }
          expect(storageState.persistent.get("tldwConfig")).toEqual(original)
          if (status !== 401) {
            const currentEpoch = await sendRuntimeMessage({ type: "tldw:connection-authority",
              payload: { expectedConnectionAuthority: deriveConnectionAuthorityId(original) } })
            expect(currentEpoch).toMatchObject({ ok: true, epoch: epoch.epoch })
          }
        } finally {
          releaseRefresh()
          await pending
          if (delayedChange && !delivered) accountChanged({ [markerKey]: delayedChange }, "local")
          port.disconnect()
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
        }
      })
    })

    it.each(["uninterrupted joined initiators", "Stop joined request"] as const)(
      "keeps each real joined initiator's terminal authority across %s", async boundary => {
        const { accountChanged, markerKey } = await configureOriginal()
        const initialSet = storageState.set.getMockImplementation()
        const markerWrites: unknown[] = []
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          const oldValue = storageState.persistent.get(key)
          storageState.persistent.set(key, value)
          if (key === markerKey) {
            markerWrites.push(value)
            accountChanged({ [key]: { oldValue, newValue: value } }, "local")
          }
        })
        let releaseRefresh!: () => void
        const heldRefresh = new Promise<void>(resolve => { releaseRefresh = resolve })
        const requests: { path: string; authorization: string | null; body: unknown; signal: AbortSignal | null }[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const path = new URL(String(input)).pathname
          if (!["/api/v1/chat/completions", "/api/v1/notes/joined-private-note", "/api/v1/auth/refresh"].includes(path)) throw new Error("Undeclared joined terminal transport: " + path)
          requests.push({ path, authorization: new Headers(init?.headers).get("Authorization"),
            body: init?.body === undefined ? undefined : JSON.parse(String(init.body)), signal: init?.signal ?? null })
          if (path === "/api/v1/auth/refresh") await heldRefresh
          return Response.json({ detail: path === "/api/v1/auth/refresh" ? "Original joined session revoked" : "Original access expired" }, { status: 401 })
        }))
        const port = connectRuntimePort("tldw:stream")
        const pendingStream = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
        let pendingRequest: Promise<unknown> | undefined
        try {
          await vi.waitFor(() => expect(requests.filter(request => request.path === "/api/v1/auth/refresh")).toHaveLength(1))
          const requestId = "original-joined-terminal-request"
          pendingRequest = sendRuntimeMessage({ type: "tldw:request", payload: {
            path: "/api/v1/notes/joined-private-note", method: "GET", requestId
          } })
          await vi.waitFor(() => expect(requests.filter(request => request.path === "/api/v1/notes/joined-private-note")).toHaveLength(1))
          // Both real preparations finish their physical reads while the one
          // shared refresh is held; no flight map/helper is substituted.
          await flushPromises()
          expect(requests.filter(request => request.path === "/api/v1/auth/refresh")).toHaveLength(1)
          expect(markerWrites).toEqual([])
          expect(port.messages).toEqual([])
          const physicalRefresh = requests.find(request => request.path === "/api/v1/auth/refresh")!
          expect(physicalRefresh.body).toEqual({ refresh_token: original.refreshToken })
          expect(physicalRefresh.signal?.aborted).toBe(false)
          if (boundary === "Stop joined request") {
            expect(await sendRuntimeMessage({ type: "tldw:cancel-request", payload: { requestId } })).toEqual({ ok: true, cancelled: true })
            const stopped = await pendingRequest
            // Caller Stop settles before the spent account flight is released.
            expect(stopped).toMatchObject({ ok: false, status: 499, error: "Cancelled by user." })
            expect(physicalRefresh.signal?.aborted).toBe(false)
            expect(markerWrites).toEqual([])
          }
          releaseRefresh()
          await pendingStream
          const response = await pendingRequest
          expect(port.messages).toEqual([expect.objectContaining({ event: "error", status: 401, message: terminalMessage })])
          if (boundary === "uninterrupted joined initiators") expect(response).toMatchObject({ ok: false, status: 401, error: terminalMessage })
          else expect(response).toMatchObject({ ok: false, status: 499, error: "Cancelled by user." })
          expect(markerWrites).toEqual([true])
          expect(requests.map(request => request.path)).toEqual([
            "/api/v1/chat/completions", "/api/v1/auth/refresh", "/api/v1/notes/joined-private-note"
          ])
          expect(requests[0]).toMatchObject({ authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(requests[2]).toMatchObject({ authorization: "Bearer " + original.accessToken, body: undefined })
        } finally {
          releaseRefresh()
          await pendingStream
          if (pendingRequest) await pendingRequest
          port.disconnect()
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
        }
      }
    )
    it.each(["delayed own event", "account ABA before delayed event"] as const)(
      "bounds delayed physical marker delivery while the original caller is unsettled: %s", async boundary => {
        const { accountChanged, markerKey } = await configureOriginal()
        const initialSet = storageState.set.getMockImplementation()
        const initialGet = storageState.get.getMockImplementation()
        let delayedChange: { oldValue: unknown; newValue: unknown } | undefined
        storageState.set.mockImplementation(async (key: string, value: unknown) => {
          const oldValue = storageState.persistent.get(key)
          storageState.persistent.set(key, value)
          if (key === markerKey) delayedChange = { oldValue, newValue: value }
        })
        let releaseRead!: () => void
        const heldPostWriteRead = new Promise<void>(resolve => { releaseRead = resolve })
        let readEntered = false
        storageState.get.mockImplementation(async (key: string) => {
          if (key === "tldwConfig" && delayedChange && !readEntered) {
            // This is the real helper's currentKey read after physical set has
            // resolved. The refresh flight remains pending at this actual gate.
            readEntered = true
            await heldPostWriteRead
          }
        })
        const requests: string[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
          const path = new URL(String(input)).pathname
          if (path !== "/api/v1/notes/delayed-private-note" && path !== "/api/v1/auth/refresh") throw new Error("Undeclared delayed terminal transport: " + path)
          requests.push(path)
          return Response.json({ detail: "Original delayed session revoked" }, { status: 401 })
        }))
        let settled = false
        let settledResponse: unknown
        const pending = sendRuntimeMessage({ type: "tldw:request", payload: {
          path: "/api/v1/notes/delayed-private-note", method: "GET"
        } }).then(response => { settled = true; settledResponse = response; return response })
        try {
          // The actual refresh helper first waits up to 1,000 ms for a sibling's
          // successful rotation. Allow that real poll to finish before this gate.
          await vi.waitFor(() => expect({ readEntered, settled, settledResponse, requests,
            marker: storageState.persistent.get(markerKey) }).toMatchObject({
            readEntered: true, settled: false, settledResponse: undefined, marker: true,
            requests: ["/api/v1/notes/delayed-private-note", "/api/v1/auth/refresh"]
          }), { timeout: 2_500 })
          expect(settled).toBe(false)
          expect(storageState.persistent.get(markerKey)).toBe(true)
          expect(requests).toEqual(["/api/v1/notes/delayed-private-note", "/api/v1/auth/refresh"])
          if (!delayedChange) throw new Error("Actual physical marker write missing")
          if (boundary === "account ABA before delayed event") {
            const replacement = { ...original, accessToken: jwtForUser(84), refreshToken: "replacement-delayed-refresh" }
            for (const config of [replacement, original]) {
              const oldValue = storageState.persistent.get("tldwConfig")
              storageState.persistent.set("tldwConfig", structuredClone(config))
              accountChanged({ tldwConfig: { oldValue, newValue: config } }, "local")
            }
          }
          accountChanged({ [markerKey]: delayedChange }, "local")
          releaseRead()
          const response = await pending
          if (boundary === "delayed own event") expect(response).toMatchObject({ ok: false, status: 401, error: terminalMessage })
          else expect(response).toMatchObject({ ok: false, status: 412, data: { detail: { code: "request_config_scope_changed" } } })
          expect(requests).toEqual(["/api/v1/notes/delayed-private-note", "/api/v1/auth/refresh"])
          expect(storageState.persistent.get("tldwConfig")).toEqual(original)
        } finally {
          releaseRead()
          await pending
          storageState.set.mockImplementation(initialSet ?? (async () => undefined))
          storageState.get.mockImplementation(initialGet ?? (async () => undefined))
        }
      }
    )
  })

  describe("TASK-13549 original unscoped worker stream refresh authority", () => {
    const original = {
      serverUrl: "https://api.example.test", authMode: "multi-user" as const, authSource: "manual" as const,
      accessToken: jwtForUser(42), refreshToken: "original-stream-refresh"
    }
    const replacement = {
      ...original, accessToken: jwtForUser(84), refreshToken: "replacement-stream-refresh"
    }
    const rotatedAccess = `header.${btoa(JSON.stringify({ sub: "42", nonce: "rotated" }))}.signature`
    const rotated = {
      ...original, accessToken: rotatedAccess, refreshToken: "same-principal-stream-refresh"
    }
    const streamBody = { stream: true, model: "original-model", messages: [{ role: "user", content: "Exact original stream question" }] }
    const readActualEpoch = async (config: typeof original) => {
      const receipt = await sendRuntimeMessage({ type: "tldw:connection-authority",
        payload: { expectedConnectionAuthority: deriveConnectionAuthorityId(config) } })
      expect(receipt).toMatchObject({ ok: true, epoch: expect.any(String) })
      return receipt.epoch as string
    }
    const configureOriginal = async () => {
      storageState.persistent.set("tldwConfig", structuredClone(original))
      storageState.persistent.delete("tldwCookieSessionConfig")
      storageState.session.clear()
      const changes = vi.mocked(browser.storage.onChanged.addListener).mock.calls
      const accountChanged = changes[changes.length - 1]?.[0]
      if (!accountChanged) throw new Error("Actual worker account observer must be registered")
      // Publish physical storage changes through this worker's actual observer.
      // No epoch, auth helper, refresh helper, or stream handler is substituted.
      const publishConfig = (config: typeof original) => {
        const oldValue = storageState.persistent.get("tldwConfig")
        const newValue = structuredClone(config)
        storageState.persistent.set("tldwConfig", newValue)
        accountChanged({ tldwConfig: { oldValue, newValue } }, "local")
      }
      return { startingEpoch: await readActualEpoch(original), publishConfig }
    }
    describe("TASK-13549 original non-stream request lifetime", () => {
      const requestBody = { stream: false, model: "original-worker-nonstream-model",
        messages: [{ role: "user", content: "Exact original private worker nonstream question" }] }
      const answer = { choices: [{ message: { role: "assistant", content: "Original worker nonstream answer" } }] }
      it.each(["A to B to exact A", "uninterrupted"] as const)("guards worker initial 401 across %s", async boundary => {
        const authority = await configureOriginal()
        let releaseUnauthorized!: () => void
        const heldUnauthorized = new Promise<void>(resolve => { releaseUnauthorized = resolve })
        const requests: { url: string; method?: string; authorization: string | null; command: string | null; body: unknown }[] = []
        const refreshes: { url: string; method?: string; body: unknown }[] = []
        vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input), path = new URL(url).pathname
          if (path === "/api/v1/auth/refresh") {
            refreshes.push({ url, method: init?.method, body: JSON.parse(String(init?.body)) })
            return Response.json({ access_token: rotatedAccess, refresh_token: rotated.refreshToken, token_type: "bearer" })
          }
          if (path !== "/api/v1/chat/completions") throw new Error("Undeclared worker transport: " + url)
          const headers = new Headers(init?.headers)
          requests.push({ url, method: init?.method, authorization: headers.get("Authorization"),
            command: headers.get("X-Original-Command"), body: JSON.parse(String(init?.body)) })
          if (requests.length === 1) {
            await heldUnauthorized
            return new Response("Original worker nonstream unauthorized", { status: 401 })
          }
          return Response.json(answer)
        }))
        const pending = sendRuntimeMessage({ type: "tldw:request", payload: { path: "/api/v1/chat/completions",
          method: "POST", body: requestBody, headers: { "X-Original-Command": "original-worker-command-a" } } })
        try {
          await vi.waitFor(() => expect(requests).toHaveLength(1))
          expect(requests[0]).toEqual({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, command: "original-worker-command-a", body: requestBody })
          expect(refreshes).toEqual([])
          if (boundary === "A to B to exact A") {
            authority.publishConfig(replacement)
            authority.publishConfig(original)
            expect(await readActualEpoch(original)).not.toBe(authority.startingEpoch)
            expect(storageState.persistent.get("tldwConfig")).toEqual(original)
          } else expect(await readActualEpoch(original)).toBe(authority.startingEpoch)
          releaseUnauthorized()
          const outcome = await pending
          if (boundary === "A to B to exact A") {
            expect(refreshes).toEqual([])
            expect(requests).toHaveLength(1)
            expect(outcome).toMatchObject({ ok: false, status: 412 })
            expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
          } else {
            expect(outcome).toMatchObject({ ok: true, status: 200, data: answer })
            expect(refreshes).toEqual([{ url: original.serverUrl + "/api/v1/auth/refresh", method: "POST",
              body: { refresh_token: original.refreshToken } }])
            expect(requests).toEqual([{ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
              authorization: "Bearer " + original.accessToken, command: "original-worker-command-a", body: requestBody },
              { url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
                authorization: "Bearer " + rotatedAccess, command: "original-worker-command-a", body: requestBody }])
            expect(storageState.persistent.get("tldwRefreshRotation")).toMatchObject({ sourceAccessToken: original.accessToken,
              sourceRefreshToken: original.refreshToken, accessToken: rotatedAccess, refreshToken: rotated.refreshToken })
          }
          expect(storageState.persistent.get("tldwConfig")).toEqual(original)
        } finally {
          releaseUnauthorized()
          await pending
        }
      })
    })

    describe("TASK-13549 worker awaited publication authority", () => {
      const frame = '{"choices":[{"delta":{"content":"Original buffered worker answer"}}]}'
      const debugCases = (["SSE data", "JSON data", "DONE"] as const).flatMap(format =>
        (["account retirement", "idle abort", "uninterrupted"] as const).map(boundary => ({ format, boundary })))

      it.each(debugCases)("guards $format after held debug delivery and $boundary", async ({ format, boundary }) => {
        const authority = await configureOriginal()
        vi.mocked(browser.runtime.sendMessage).mockReset().mockResolvedValue({ handled: true })
        await expect(sendRuntimeMessage({ type: "tldw:debug", enable: true })).resolves.toEqual({ ok: true })
        let releaseDebug!: () => void
        const heldDebug = new Promise<void>(resolve => { releaseDebug = resolve })
        const targetData = format === "DONE" ? "[DONE]" : frame
        const debugEntries: unknown[] = []
        vi.mocked(browser.runtime.sendMessage).mockImplementation(async (message: { type?: string; payload?: { kind?: string; data?: unknown } }) => {
          if (message?.type === "tldw:stream-debug" && message.payload?.kind === "data" && message.payload.data === targetData) {
            debugEntries.push(message)
            await heldDebug
          }
          return { handled: true }
        })
        const responseBody = format === "DONE" ? "data: [DONE]\n\n"
          : (format === "SSE data" ? "data: " : "") + frame + "\n\ndata: [DONE]\n\n"
        const physical: { url: string; method?: string; authorization: string | null; body: unknown; signal?: AbortSignal | null }[] = []
        const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/chat/completions") throw new Error("Undeclared worker transport: " + url)
          physical.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
            body: JSON.parse(String(init?.body)), signal: init?.signal })
          return new Response(responseBody, { headers: { "content-type": "text/event-stream" } })
        })
        vi.stubGlobal("fetch", fetchSpy)
        if (boundary === "idle abort") vi.useFakeTimers()
        const port = connectRuntimePort("tldw:stream")
        const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody,
          streamIdleTimeoutMs: 10_000 })
        try {
          await vi.waitFor(() => expect(debugEntries).toHaveLength(1))
          expect(debugEntries[0]).toMatchObject({ type: "tldw:stream-debug", payload: { kind: "data", data: targetData } })
          expect(physical).toHaveLength(1)
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(physical[0].signal?.aborted).toBe(false)
          expect(port.messages).toEqual([{ event: "open" }])
          if (boundary === "account retirement") {
            authority.publishConfig(replacement)
            expect(await readActualEpoch(replacement)).not.toBe(authority.startingEpoch)
            expect(physical[0].signal?.aborted).toBe(true)
          } else if (boundary === "idle abort") {
            // The production idle timer aborts the physical request while its
            // debug delivery still waits; no guard or epoch is substituted.
            await vi.advanceTimersByTimeAsync(10_001)
            expect(physical[0].signal?.aborted).toBe(true)
            expect(port.messages).toContainEqual({ event: "error", message: "Stream timeout: no updates received" })
          }
          releaseDebug()
          await pending
          if (boundary === "uninterrupted") {
            expect(port.messages).toEqual([{ event: "open" },
              ...(format === "DONE" ? [] : [{ event: "data", data: frame }]), { event: "done" }])
          } else {
            expect(port.messages.some((message) => message?.event === "data" || message?.event === "done")).toBe(false)
            if (boundary === "account retirement") {
              expect(port.messages).toEqual([{ event: "open" }, expect.objectContaining({ event: "error", status: 412 })])
            }
          }
          expect(fetchSpy).toHaveBeenCalledTimes(1)
          expect(storageState.persistent.get("tldwConfig")).toEqual(boundary === "account retirement" ? replacement : original)
          expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
        } finally {
          releaseDebug()
          await pending
          port.disconnect()
          vi.useRealTimers()
          vi.mocked(browser.runtime.sendMessage).mockReset().mockResolvedValue({ handled: true })
        }
      })

      const errorCases = (["json", "text"] as const).flatMap(format =>
        (["account retirement", "uninterrupted"] as const).map(boundary => ({ format, boundary })))
      it.each(errorCases)("guards the original $format error after held body read and $boundary", async ({ format, boundary }) => {
        const authority = await configureOriginal()
        const message = "Original account private stream failure"
        const errorBody = { detail: message, request_id: "original-private-request" }
        const response = format === "json" ? Response.json(errorBody, { status: 503, headers: { "retry-after": "3" } })
          : new Response(message, { status: 503, headers: { "content-type": "text/plain", "retry-after": "3" } })
        let releaseBody!: () => void
        const heldBody = new Promise<void>(resolve => { releaseBody = resolve })
        let bodyEntered = false
        const readJson = response.json.bind(response)
        const readText = response.text.bind(response)
        const readBody = format === "json"
          ? vi.spyOn(response, "json").mockImplementation(async () => { bodyEntered = true; await heldBody; return readJson() })
          : vi.spyOn(response, "text").mockImplementation(async () => { bodyEntered = true; await heldBody; return readText() })
        const physical: { url: string; method?: string; authorization: string | null; body: unknown; signal?: AbortSignal | null }[] = []
        const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = String(input)
          if (new URL(url).pathname !== "/api/v1/chat/completions") throw new Error("Undeclared worker transport: " + url)
          physical.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
            body: JSON.parse(String(init?.body)), signal: init?.signal })
          return response
        })
        vi.stubGlobal("fetch", fetchSpy)
        const port = connectRuntimePort("tldw:stream")
        const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
        try {
          await vi.waitFor(() => expect(bodyEntered).toBe(true))
          expect(readBody).toHaveBeenCalledTimes(1)
          expect(physical).toHaveLength(1)
          expect(physical[0]).toMatchObject({ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody })
          expect(physical[0].signal?.aborted).toBe(false)
          expect(port.messages).toEqual([])
          if (boundary === "account retirement") {
            authority.publishConfig(replacement)
            expect(await readActualEpoch(replacement)).not.toBe(authority.startingEpoch)
            expect(physical[0].signal?.aborted).toBe(true)
          }
          releaseBody()
          await pending
          if (boundary === "account retirement") {
            expect(port.messages).toEqual([expect.objectContaining({ event: "error", status: 412 })])
            expect(port.messages.some((entry) => entry.status === 503 || entry.message === message ||
              (typeof entry.details === "object" && entry.details !== null &&
                "request_id" in entry.details && entry.details.request_id === errorBody.request_id))).toBe(false)
          } else {
            expect(port.messages).toEqual([{ event: "error", status: 503, message,
              details: format === "json" ? errorBody : undefined, retryAfter: "3" }])
          }
          expect(fetchSpy).toHaveBeenCalledTimes(1)
          expect(storageState.persistent.get("tldwConfig")).toEqual(boundary === "account retirement" ? replacement : original)
          expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
        } finally {
          releaseBody()
          await pending
          port.disconnect()
        }
      })
    })

    it.each([true, false])("TASK-13549 worker initial stream preparation respects disconnect=%s", async disconnect => {
      await configureOriginal()
      let releaseConfig!: () => void
      const heldConfig = new Promise<void>(resolve => { releaseConfig = resolve })
      let configEntered = false
      storageState.get.mockImplementation(async (key: string) => {
        if (key === "tldwConfig" && !configEntered) {
          configEntered = true
          await heldConfig
        }
      })
      const physical: { url: string; method?: string; authorization: string | null; body: unknown }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (new URL(url).pathname !== "/api/v1/chat/completions") throw new Error("Undeclared worker transport: " + url)
        physical.push({ url, method: init?.method, authorization: new Headers(init?.headers).get("Authorization"),
          body: JSON.parse(String(init?.body)) })
        return new Response('data: {"choices":[{"delta":{"content":"Prepared original answer"}}]}\n\ndata: [DONE]\n\n',
          { headers: { "content-type": "text/event-stream" } })
      }))
      const port = connectRuntimePort("tldw:stream")
      const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
      try {
        await vi.waitFor(() => expect(configEntered).toBe(true))
        expect(physical).toEqual([])
        if (disconnect) port.disconnect()
        releaseConfig()
        await pending
        if (disconnect) {
          // Stop disconnected this actual port while its first config read was
          // pending, before a request controller or transport existed.
          expect(physical).toEqual([])
          expect(port.messages).toEqual([])
        } else {
          expect(physical).toEqual([{ url: original.serverUrl + "/api/v1/chat/completions", method: "POST",
            authorization: "Bearer " + original.accessToken, body: streamBody }])
          expect(port.messages).toEqual([
            { event: "open" },
            { event: "data", data: '{"choices":[{"delta":{"content":"Prepared original answer"}}]}' },
            { event: "done" }
          ])
        }
        expect(storageState.persistent.get("tldwConfig")).toEqual(original)
        expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
      } finally {
        releaseConfig()
        await pending
        port.disconnect()
      }
    })

    const holdOriginalUnauthorized = () => {
      let releaseUnauthorized!: () => void
      const held = new Promise<void>(resolve => { releaseUnauthorized = resolve })
      const streams: { url: string; authorization: string | null; body: unknown; signal?: AbortSignal | null }[] = []
      const refreshes: { url: string; body: unknown }[] = []
      vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input), path = new URL(url).pathname
        if (path === "/api/v1/auth/refresh") {
          refreshes.push({ url, body: JSON.parse(String(init?.body)) })
          return Response.json({ access_token: rotatedAccess,
            refresh_token: rotated.refreshToken, token_type: "bearer" })
        }
        if (path !== "/api/v1/chat/completions") throw new Error("Undeclared worker transport: " + path)
        streams.push({ url, authorization: new Headers(init?.headers).get("Authorization"),
          body: JSON.parse(String(init?.body)), signal: init?.signal })
        if (streams.length === 1) {
          await held
          return new Response("Original stream unauthorized", { status: 401 })
        }
        return new Response('data: {"choices":[{"delta":{"content":"Original answer"}}]}\n\ndata: [DONE]\n\n',
          { headers: { "Content-Type": "text/event-stream" } })
      }))
      return { streams, refreshes, releaseUnauthorized }
    }

    it("TASK-13549 aborts the physical original worker stream on account retirement while its main fetch is held", async () => {
      const authority = await configureOriginal()
      const physical = holdOriginalUnauthorized()
      const port = connectRuntimePort("tldw:stream")
      const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
      try {
        await vi.waitFor(() => expect(physical.streams).toHaveLength(1))
        expect(physical.streams[0]).toMatchObject({
          url: original.serverUrl + "/api/v1/chat/completions", authorization: "Bearer " + original.accessToken, body: streamBody
        })
        expect(physical.streams[0].signal?.aborted).toBe(false)
        authority.publishConfig(replacement)
        expect(await readActualEpoch(replacement)).not.toBe(authority.startingEpoch)
        // The actual account observer must cancel physical old-A work now;
        // checking an epoch only after the held response settles is too late.
        expect(physical.streams[0].signal?.aborted).toBe(true)
        physical.releaseUnauthorized()
        await pending
        expect(physical.refreshes).toEqual([])
        expect(physical.streams).toHaveLength(1)
        expect(port.messages).toContainEqual(expect.objectContaining({ event: "error", status: 412 }))
        expect(port.messages.some((message) => message?.event === "done")).toBe(false)
        expect(storageState.persistent.get("tldwConfig")).toEqual(replacement)
        expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
      } finally {
        physical.releaseUnauthorized()
        await pending
        port.disconnect()
      }
    })

    it.each(["replacement", "A to B to A"] as const)(
      "refuses original unscoped worker refresh or replay after held initial 401 and %s",
      async boundary => {
        const authority = await configureOriginal()
        const physical = holdOriginalUnauthorized()
        const port = connectRuntimePort("tldw:stream")
        const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
        try {
          await vi.waitFor(() => expect(physical.streams).toHaveLength(1))
          expect(physical.streams[0]).toMatchObject({
            url: original.serverUrl + "/api/v1/chat/completions", authorization: "Bearer " + original.accessToken, body: streamBody
          })
          expect(physical.refreshes).toEqual([])
          authority.publishConfig(replacement)
          if (boundary === "A to B to A") authority.publishConfig(original)
          const current = boundary === "replacement" ? replacement : original
          expect(await readActualEpoch(current)).not.toBe(authority.startingEpoch)
          physical.releaseUnauthorized()
          await pending
          // Retired A must neither refresh current B nor regain its old command
          // merely because the exact A config was restored under a later epoch.
          expect(physical.refreshes).toEqual([])
          expect(physical.streams).toHaveLength(1)
          expect(storageState.persistent.get("tldwConfig")).toEqual(current)
          expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
          expect(port.messages).toContainEqual(expect.objectContaining({ event: "error", status: 412 }))
          expect(port.messages.some((message) => message?.event === "done")).toBe(false)
        } finally {
          physical.releaseUnauthorized()
          await pending
          port.disconnect()
        }
      }
    )

    it("reuses same-principal rotation for the held original unscoped worker stream without retiring its epoch", async () => {
      const authority = await configureOriginal()
      const physical = holdOriginalUnauthorized()
      const port = connectRuntimePort("tldw:stream")
      const pending = port.postMessage({ path: "/api/v1/chat/completions", method: "POST", body: streamBody })
      try {
        await vi.waitFor(() => expect(physical.streams).toHaveLength(1))
        expect(physical.streams[0]).toMatchObject({ authorization: "Bearer " + original.accessToken, body: streamBody })
        authority.publishConfig(rotated)
        expect(await readActualEpoch(rotated)).toBe(authority.startingEpoch)
        physical.releaseUnauthorized()
        await pending
        expect(physical.refreshes).toEqual([])
        expect(physical.streams).toHaveLength(2)
        expect(physical.streams[1]).toMatchObject({
          url: original.serverUrl + "/api/v1/chat/completions", authorization: "Bearer " + rotatedAccess, body: streamBody
        })
        expect(storageState.persistent.get("tldwConfig")).toEqual(rotated)
        expect(storageState.persistent.get("tldwRefreshRotation")).toBeUndefined()
        expect(port.messages).toContainEqual({ event: "data", data: '{"choices":[{"delta":{"content":"Original answer"}}]}' })
        expect(port.messages.filter((message) => message?.event === "done")).toEqual([{ event: "done" }])
        expect(port.messages.some((message) => message?.event === "error")).toBe(false)
      } finally {
        physical.releaseUnauthorized()
        await pending
        port.disconnect()
      }
    })
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

  it.each([
    ["/api/v1/media/ingest-web-content", "POST"],
    ["/api/v1/web-clipper/925871f6-30fa-470b-a2c4-2272051f2373", "GET"],
    ["/api/v1/web-clipper/clip%20with%20spaces/", "GET"],
    ["/api/v1/media/71/versions?include_content=true", "GET"],
    ["/api/v1/media/71/versions?include_content=true&limit=10&page=2", "GET"],
    ["/api/v1/media/71/versions/7?include_content=true", "GET"],
    ["/api/v1/workspaces/workspace%20with%20spaces/sources", "GET"],
    [
      "/api/v1/workspaces/ws/sources/web-clipper%3A925871f6-30fa-470b-a2c4-2272051f2373/preview?version_number=7",
      "GET"
    ]
  ])(
    "admits exact capture request %s %s through the real worker",
    async (path, method) => {
      const capturedConfig = Object.freeze({
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        expectedUserId: 42,
        expectedRefreshToken: "capture-refresh"
      })
      storageState.persistent.set("tldwConfig", {
        serverUrl: capturedConfig.serverUrl,
        authMode: "multi-user",
        authSource: "manual",
        accessToken: jwtForUser(42),
        refreshToken: "capture-refresh"
      })
      storageState.persistent.delete("tldwCookieSessionConfig")
      const body =
        method === "POST"
          ? { urls: ["https://example.com/story"], credential_free: true }
          : undefined
      const fetchSpy = vi.fn(
        async (_input: RequestInfo | URL, _init?: RequestInit) =>
          new Response(JSON.stringify({ capture: "canonical" }), {
            headers: { "content-type": "application/json" }
          })
      )
      vi.stubGlobal("fetch", fetchSpy)
      await expect(
        sendRuntimeMessage({
          type: "tldw:request",
          payload: {
            path,
            method,
            body,
            servicePromptConfig: capturedConfig,
            headers: { "X-TLDW-Expected-User-ID": "42" }
          }
        })
      ).resolves.toMatchObject({
        ok: true,
        status: 200,
        data: { capture: "canonical" }
      })
      expect(fetchSpy).toHaveBeenCalledTimes(1)
      const [url, init] = fetchSpy.mock.calls[0]
      expect(String(url)).toBe(`https://api.example.test${path}`)
      expect(init?.method).toBe(method)
      expect(new Headers(init?.headers).get("Authorization")).toBe(
        `Bearer ${jwtForUser(42)}`
      )
      expect(new Headers(init?.headers).get("X-TLDW-Expected-User-ID")).toBe("42")
      if (body) expect(JSON.parse(String(init?.body))).toEqual(body)
    }
  )

  it.each([
    ["/api/v1/media/ingest-web-content", "GET"],
    ["/api/v1/media/ingest-web-content/extra", "POST"],
    ["/api/v1/media/71/versions", "POST"],
    ["/api/v1/media/71/versions/0", "GET"],
    ["/api/v1/media/71/versions/advanced", "GET"],
    ["/api/v1/media/71/versions/7/metadata", "GET"],
    ["/api/v1/workspaces/ws/sources", "POST"],
    ["/api/v1/workspaces/ws/sources/src/preview", "DELETE"],
    ["/api/v1/workspaces/ws/sources/status", "GET"],
    ["/api/v1/workspaces/ws/sources/src/preview/extra", "GET"],
    ["/api/v1/workspaces/../ws/sources", "GET"],
    ["/api/v1/workspaces/%2e%2e/sources", "GET"],
    ["/api/v1/workspaces/ws%2fforeign/sources", "GET"],
    ["/api/v1/workspaces/ws/sources/a%5cb/preview", "GET"],
    ["/api/v1/workspaces/ws%zz/sources", "GET"],
    ["/api/v1/web-clipper/clip/enrichments", "GET"],
    ["https://api.example.test/api/v1/media/ingest-web-content", "POST"]
  ])(
    "rejects capture sibling or malformed request %s %s in the worker before HTTP",
    async (path, method) => {
      const fetchSpy = vi.fn()
      vi.stubGlobal("fetch", fetchSpy)
      await expect(
        sendRuntimeMessage({
          type: "tldw:request",
          payload: {
            path,
            method,
            servicePromptConfig: {
              serverUrl: "https://api.example.test",
              authMode: "single-user",
              authSource: "manual",
              expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
            }
          }
        })
      ).resolves.toMatchObject({ ok: false, status: 400 })
      expect(fetchSpy).not.toHaveBeenCalled()
    }
  )

  it.each(
    ["principal", "origin", "refresh", "key"].flatMap((change) =>
      ["extract", "paged versions"].map((operation) => ({ change, operation }))
    )
  )(
    "rejects changed capture $change in worker before $operation HTTP",
    async ({ change, operation }) => {
      const fetchSpy = vi.fn()
      vi.stubGlobal("fetch", fetchSpy)
      if (change !== "key") {
        storageState.persistent.set("tldwConfig", {
          serverUrl:
            change === "origin"
              ? "https://other.example.test"
              : "https://api.example.test",
          authMode: "multi-user",
          authSource: "manual",
          accessToken: jwtForUser(change === "principal" ? 84 : 42),
          refreshToken:
            change === "refresh" ? "changed-refresh" : "capture-refresh"
        })
        storageState.persistent.delete("tldwCookieSessionConfig")
      }
      await expect(
        sendRuntimeMessage({
          type: "tldw:request",
          payload: {
            path:
              operation === "extract"
                ? "/api/v1/media/ingest-web-content"
                : "/api/v1/media/71/versions?include_content=true&limit=10&page=2",
            method: operation === "extract" ? "POST" : "GET",
            body:
              operation === "extract"
                ? { urls: ["https://example.com"] }
                : undefined,
            servicePromptConfig:
              change === "key"
                ? {
                    serverUrl: "https://api.example.test",
                    authMode: "single-user",
                    authSource: "manual",
                    expectedSingleUserApiKeyScope:
                      deriveSingleUserApiKeyCredentialScope(
                        "single-user",
                        "captured-other-key"
                      )
                  }
                : {
                    serverUrl: "https://api.example.test",
                    authMode: "multi-user",
                    authSource: "manual",
                    expectedUserId: 42,
                    expectedRefreshToken: "capture-refresh"
                  }
          }
        })
      ).resolves.toMatchObject({
        ok: false,
        status: 412,
        data: { detail: { code: "request_config_scope_changed" } }
      })
      expect(fetchSpy).not.toHaveBeenCalled()
    }
  )

  it.each(["\t", "\r", "\n"])(
    "rejects raw URL-normalizing control %j before worker capture dispatch",
    async (control) => {
      for (const path of [
        `/api/v1/workspaces/${control}../sources`,
        `/api/v1/web-clipper/${control}..`,
        `/api/v1/workspaces/ws/sources/${control}../preview`
      ]) {
        const fetchSpy = vi.fn(
          async () =>
            new Response(JSON.stringify({ capture: "unexpected" }), {
              headers: { "content-type": "application/json" }
            })
        )
        vi.stubGlobal("fetch", fetchSpy)
        await expect(
          sendRuntimeMessage({
            type: "tldw:request",
            payload: {
              path,
              method: "GET",
              servicePromptConfig: {
                serverUrl: "https://api.example.test",
                authMode: "single-user",
                authSource: "manual",
                expectedSingleUserApiKeyScope: WORKER_API_KEY_SCOPE
              }
            }
          })
        ).resolves.toMatchObject({ ok: false, status: 400 })
        expect(fetchSpy).not.toHaveBeenCalled()
      }
    }
  )

})
