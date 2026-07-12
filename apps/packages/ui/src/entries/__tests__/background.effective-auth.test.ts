import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const connectListeners = new Set<(port: any) => void>()
const runtimeMessageListeners = new Set<(
  message: unknown,
  sender: unknown,
  sendResponse: (response: unknown) => void
) => unknown>()

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
      set: storageState.set,
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
    commands: { onCommand: { addListener: vi.fn() } },
    i18n: { getMessage: (key: string) => key }
  }
}))

type RuntimePortHarness = {
  messages: unknown[]
  postMessage: (message: unknown) => void
}

const connectRuntimePort = (name: string): RuntimePortHarness => {
  const inboundListeners = new Set<(message: unknown) => void>()
  const disconnectListeners = new Set<() => void>()
  const messages: unknown[] = []
  const port = {
    name,
    sender: { id: "extension-id" },
    postMessage: (message: unknown) => messages.push(message),
    onMessage: {
      addListener: (listener: (message: unknown) => void) =>
        inboundListeners.add(listener),
      removeListener: (listener: (message: unknown) => void) =>
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
    postMessage: (message: unknown) => {
      for (const listener of inboundListeners) listener(message)
    }
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
    if (windowDescriptor) {
      Object.defineProperty(globalThis, "window", windowDescriptor)
    } else {
      delete (globalThis as any).window
    }
  })

  it("authenticates ordinary worker requests with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)

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

  it("authenticates worker uploads with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)

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

  it("authenticates worker HTTP streams with session credentials", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response('data: {"ok":true}\n\ndata: [DONE]\n\n', {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)
    const port = connectRuntimePort("tldw:stream")

    port.postMessage({
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
})
