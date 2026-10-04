import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { getEventListeners } from "node:events"
import { transferableAbortController } from "node:util"

const harness = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    saved: {} as Record<string, unknown>,
    changes: new Set<(changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void>(),
    values: new Map<string, unknown>(),
    sessionValues: new Map<string, unknown>(),
    workerConfig: null as Record<string, unknown> | null,
    beforeRequest: null as (() => void) | null,
    authorityReplyGate: null as Promise<unknown> | null,
    authorityReplyReady: null as (() => void) | null,
    listeners: new Set<
      (
        message: unknown,
        sender: unknown,
        reply: (value: unknown) => void
      ) => unknown
    >(),
    sent: [] as unknown[]
  }
})
vi.mock("@/utils/safe-storage", async importOriginal => {
  const { safeStorageSerde } = await importOriginal<typeof import("@/utils/safe-storage")>()
  return {
  safeStorageSerde,
  createSafeStorage: ({ area = "local" } = {}) => {
    const values = area === "session" ? harness.sessionValues : harness.values
    return {
      get: async (key: string) => values.get(key),
      set: async (key: string, value: unknown) => {
        const oldValue = values.get(key)
        values.set(key, value)
        for (const listener of harness.changes) listener({ [key]: {
          oldValue: safeStorageSerde.serializer(oldValue),
          newValue: safeStorageSerde.serializer(value)
        } }, area)
      },
      remove: async (key: string) => {
        const oldValue = values.get(key)
        values.delete(key)
        if (oldValue !== undefined) {
          for (const listener of harness.changes) listener({ [key]: {
            oldValue: safeStorageSerde.serializer(oldValue), newValue: undefined
          } }, area)
        }
      }
    }
  }
  }
})
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn() })
  return {
    browser: {
      runtime: {
        id: "extension-id",
        getURL: (path: string) => `chrome-extension://extension-id${path}`,
        sendMessage: async (message: unknown) => {
          harness.sent.push(message)
          if ((message as { type?: string }).type === "tldw:request") harness.beforeRequest?.()
          const previous = harness.values.get("tldwConfig")
          if (harness.workerConfig) harness.values.set("tldwConfig", harness.workerConfig)
          try {
            const result = await send(message)
            if ((message as { type?: string }).type === "tldw:connection-authority") {
              harness.authorityReplyReady?.()
              await harness.authorityReplyGate
            }
            return result
          } finally {
            if (harness.workerConfig) harness.values.set("tldwConfig", previous)
          }
        },
        onConnect: event(),
        onStartup: event(),
        onMessage: {
          addListener: (
            listener: Parameters<typeof harness.listeners.add>[0]
          ) => harness.listeners.add(listener)
        }
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        session: { get: async () => harness.saved, set: async (items: Record<string, unknown>) => { Object.assign(harness.saved, items) } },
        onChanged: { addListener: (listener: Parameters<typeof harness.changes.add>[0]) => harness.changes.add(listener) }
      },
      alarms: {
        get: async () => undefined,
        clear: async () => true,
        create: async () => {},
        onAlarm: event()
      },
      tabs: {
        query: async () => [],
        create: vi.fn(),
        sendMessage: async () => {}
      },
      action: { onClicked: event() },
      contextMenus: { create: vi.fn(), removeAll: vi.fn(), onClicked: event() },
      i18n: { getMessage: (key: string) => key }
    }
  }
})
const send = (message: unknown): Promise<unknown> =>
  new Promise((resolve) => {
    const listener = [...harness.listeners][0]
    if (!listener) throw new Error("Missing background listener")
    listener(message, { id: "extension-id" }, resolve)
  })
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" }
  })

import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
const config = { serverUrl: "https://api.example.test/base", authMode: "single-user" as const, authSource: "manual" as const, apiKey: "worker-key", credentialSource: "manual" as const, apiKeyPersistence: "device" as const, apiKeyServerOrigin: "https://api.example.test" }
const requestScope = { config: { serverUrl: config.serverUrl, authMode: config.authMode, authSource: config.authSource, expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", "worker-key")! }, userId: null }
const batch = { requestScope, entries: [{ id: "private", url: "https://source.test/doc.pdf", type: "pdf" }], files: [], storeRemote: true, processOnly: false }
const switchConfig = (next: typeof config) => {
  const previous = harness.values.get("tldwConfig")
  harness.values.set("tldwConfig", next)
  for (const listener of harness.changes) listener({ tldwConfig: { oldValue: previous, newValue: next } }, "local")
}
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }

beforeEach(async () => {
  vi.resetModules(); harness.listeners.clear(); harness.changes.clear(); harness.values.clear(); harness.sessionValues.clear(); harness.sent.length = 0; harness.saved = {}; harness.workerConfig = null; harness.beforeRequest = null; harness.authorityReplyGate = null; harness.authorityReplyReady = null
  harness.values.set("tldwConfig", config)
  vi.stubGlobal("window", undefined)
  vi.stubGlobal("chrome", { storage: (await import("wxt/browser")).browser.storage })
  const background = (await import("@/entries/background")).default
  background.main()
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers() })

describe("domain cache extension message authority", () => {
  it.each(["before dispatch", "pending response"])("rejects refresh-session invalidation with a %s", async timing => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { createSafeStorage } = await import("@/utils/safe-storage")
    const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
    const checked = {
      serverUrl: config.serverUrl, authMode: "multi-user" as const,
      accessToken: `test.${btoa(JSON.stringify({ sub: "alice" }))}.signature`, refreshToken: "synthetic-refresh"
    }
    const persistent = createSafeStorage({ area: "local" })
    await persistent.set("tldwConfig", checked)
    const key = refreshSessionInvalidationKey(checked)!
    const pending = deferred<Response>()
    const fetcher = vi.fn(() => pending.promise)
    vi.stubGlobal("fetch", fetcher)
    // Body serialization happens after credential resolution but before fetch.
    const body = timing === "before dispatch" ? {
      toJSON: () => { void persistent.set(key, true); return {} }
    } : {}
    const read = bgRequest({ path: "/api/v1/characters/7", method: "POST", body, configSnapshot: checked }).catch(error => error)
    if (timing === "pending response") {
      await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
      await persistent.set(key, true)
    }
    pending.resolve(json({ id: 7, name: "Invalidated owner" }))
    expect(await read).toMatchObject({ status: 412 })
    if (timing === "before dispatch") expect(fetcher).not.toHaveBeenCalled()
  })

  it.each(["VITE_TLDW_API_KEY", "VITE_TLDW_DEFAULT_API_KEY", "NEXT_PUBLIC_X_API_KEY"])("hydrates %s identically for fenced client and worker reads", async env => {
    vi.stubEnv(env, " synthetic-env-key ")
    const { TldwApiClient, TldwApiClientBase } = await import("@/services/tldw/TldwApiClient")
    const { characterMethods } = await import("@/services/tldw/domains/characters")
    const { chatRagMethods } = await import("@/services/tldw/domains/chat-rag")
    const { apiKey: _apiKey, ...stored } = config
    harness.values.set("tldwConfig", stored)
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(new Headers(init?.headers).get("X-API-Key")).toBe("synthetic-env-key")
      return json(String(input).includes("/messages")
        ? [{ id: "7", sender: "user", content: "Alice" }]
        : { id: 7, name: "Alice" })
    })
    vi.stubGlobal("fetch", fetcher)
    for (const adapter of ["public", "base", "domain"]) {
      const client = new TldwApiClient()
      vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0] as `/${string}`)
      const character = adapter === "base" ? TldwApiClientBase.prototype.getCharacter
        : adapter === "domain" ? characterMethods.getCharacter : client.getCharacter
      const messages = adapter === "base" ? TldwApiClientBase.prototype.listChatMessages
        : adapter === "domain" ? chatRagMethods.listChatMessages : client.listChatMessages
      expect(await character.call(client, 7)).toMatchObject({ name: "Alice" })
      expect(await messages.call(client, "7")).toMatchObject([{ content: "Alice" }])
      const dispatches = fetcher.mock.calls.length
      expect(await character.call(client, 7)).toMatchObject({ name: "Alice" })
      expect(await messages.call(client, "7")).toMatchObject([{ content: "Alice" }])
      expect(fetcher).toHaveBeenCalledTimes(dispatches)
    }
    expect(JSON.stringify(harness.sent)).not.toContain("synthetic-env-key")
  })

  it.each(["GET", "POST"] as const)("promptly cancels an unresolved %s authority handshake without dispatch or fallback", async method => {
    const { bgRequest } = await import("@/services/background-proxy")
    const checked = deferred<unknown>()
    harness.authorityReplyGate = checked.promise
    const controller = transferableAbortController()
    const listeners = getEventListeners(controller.signal, "abort").length
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    vi.useFakeTimers()
    const timers = vi.getTimerCount()
    let outcome: unknown
    const read = bgRequest({ path: "/api/v1/characters/7", method, configSnapshot: config, abortSignal: controller.signal }).catch(error => { outcome = error; return error })
    expect(harness.sent).toHaveLength(1)
    controller.abort()
    await vi.advanceTimersByTimeAsync(0)
    expect(outcome).toMatchObject({ name: "AbortError" })
    expect(vi.getTimerCount()).toBe(timers)
    expect(getEventListeners(controller.signal, "abort")).toHaveLength(listeners)
    expect(await read).toBe(outcome)
    checked.resolve(null)
    await vi.advanceTimersByTimeAsync(0)
    expect(harness.sent).toHaveLength(1)
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each(["before request", "during messaging", "before listener registration"] as const)("does not miss handshake cancellation %s", async timing => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { browser } = await import("wxt/browser")
    const controller = new AbortController()
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    const sendMessage = vi.spyOn(browser.runtime, "sendMessage").mockImplementation(() => {
      if (timing === "during messaging") controller.abort()
      return new Promise(() => {})
    })
    if (timing === "before request") controller.abort()
    if (timing === "before listener registration") {
      const addListener = controller.signal.addEventListener.bind(controller.signal)
      vi.spyOn(controller.signal, "addEventListener").mockImplementation((...args) => {
        controller.abort()
        addListener(...args)
      })
    }
    vi.useFakeTimers()
    const timers = vi.getTimerCount()
    let outcome: unknown
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config, abortSignal: controller.signal }).catch(error => { outcome = error; return error })
    await vi.advanceTimersByTimeAsync(0)
    expect(outcome).toMatchObject({ name: "AbortError" })
    expect(await read).toBe(outcome)
    expect(vi.getTimerCount()).toBe(timers)
    expect(sendMessage.mock.calls.every(([message]) => (message as unknown as { type?: string }).type === "tldw:connection-authority")).toBe(true)
    if (timing === "before request") expect(sendMessage).not.toHaveBeenCalled()
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each(["success", "error", "timeout"] as const)("releases cancellation resources after handshake %s", async result => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { browser } = await import("wxt/browser")
    const controller = transferableAbortController()
    const listeners = getEventListeners(controller.signal, "abort").length
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    if (result === "error") vi.spyOn(browser.runtime, "sendMessage").mockRejectedValueOnce(new Error("Worker unavailable"))
    if (result === "timeout") vi.spyOn(browser.runtime, "sendMessage").mockImplementationOnce(() => new Promise(() => {}))
    vi.useFakeTimers()
    const timers = vi.getTimerCount()
    let outcome: unknown
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config, abortSignal: controller.signal }).catch(error => error).then(value => { outcome = value; return value })
    await vi.advanceTimersByTimeAsync(result === "timeout" ? 3000 : 0)
    expect(await read).toBe(outcome)
    if (result === "success") expect(outcome).toEqual({ id: 7 })
    else expect(outcome).toBeInstanceOf(Error)
    expect(getEventListeners(controller.signal, "abort")).toHaveLength(listeners)
    expect(vi.getTimerCount()).toBe(timers)
    const messages = harness.sent.length
    controller.abort()
    await vi.advanceTimersByTimeAsync(0)
    expect(harness.sent).toHaveLength(messages)
    expect(fetcher).toHaveBeenCalledTimes(result === "success" ? 1 : 0)
  })

  it("classifies an unanswered handshake as a no-fallback timeout rather than a scope change", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { browser } = await import("wxt/browser")
    const sendMessage = vi.spyOn(browser.runtime, "sendMessage").mockImplementationOnce(() => new Promise(() => {}))
    const fetcher = vi.fn()
    vi.stubGlobal("fetch", fetcher)
    vi.useFakeTimers()
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config }).catch(error => error)
    await vi.advanceTimersByTimeAsync(3000)
    const error = await read
    expect(error).toMatchObject({ message: "Extension messaging timeout", __tldwNoDirectFallback: true, __tldwExtensionTimeout: true })
    expect(error.status).not.toBe(412)
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(fetcher).not.toHaveBeenCalled()
  })

  describe("session manual credential authority", () => {
    const setup = async () => {
      const { createSafeStorage } = await import("@/utils/safe-storage")
      const { MANUAL_SESSION_KEY, resolveEffectiveTldwConfig } = await import("@/services/tldw/single-user-credential")
      const { apiKey, ...metadata } = { ...config, apiKeyPersistence: "session" as const }
      const persistent = createSafeStorage({ area: "local" })
      const session = createSafeStorage({ area: "session" })
      const record = { credentialSource: "manual" as const, apiKeyPersistence: "session" as const, apiKeyServerOrigin: config.apiKeyServerOrigin, apiKey }
      await persistent.set("tldwConfig", metadata)
      await session.set(MANUAL_SESSION_KEY, record)
      const checked = await resolveEffectiveTldwConfig({ persistent, session })
      expect(checked?.apiKey).toBe("worker-key")
      expect(harness.values.get("tldwConfig")).not.toHaveProperty("apiKey")
      return { persistent, session, record, checked, MANUAL_SESSION_KEY, resolveEffectiveTldwConfig }
    }

    it.each(["replacement", "removal", "remove and restore"] as const)("rejects session-key %s while a worker response is pending", async change => {
      const { bgRequest } = await import("@/services/background-proxy")
      const stores = await setup()
      const pending = deferred<Response>()
      const fetcher = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => pending.promise)
      vi.stubGlobal("fetch", fetcher)
      const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: stores.checked }).catch(error => error)
      await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
      expect(new Headers(fetcher.mock.calls[0][1]?.headers).get("X-API-Key")).toBe("worker-key")
      if (change === "replacement") await stores.session.set(stores.MANUAL_SESSION_KEY, { ...stores.record, apiKey: "different-key" })
      else {
        await stores.session.remove(stores.MANUAL_SESSION_KEY)
        if (change === "remove and restore") await stores.session.set(stores.MANUAL_SESSION_KEY, stores.record)
      }
      const current = await stores.resolveEffectiveTldwConfig(stores)
      expect(current?.apiKey).toBe(change === "replacement" ? "different-key" : change === "removal" ? undefined : "worker-key")
      pending.resolve(json({ id: 7, name: "Old owner" }))
      expect(await read).toMatchObject({ status: 412 })
      expect(fetcher).toHaveBeenCalledTimes(1)
    })

    it("rejects a session-key remove and restore after the handshake before dispatch", async () => {
      const { bgRequest } = await import("@/services/background-proxy")
      const stores = await setup()
      const gate = deferred<unknown>()
      const ready = deferred<unknown>()
      harness.authorityReplyGate = gate.promise
      harness.authorityReplyReady = () => ready.resolve(null)
      const fetcher = vi.fn(async () => json({ id: 7, name: "Unfenced owner" }))
      vi.stubGlobal("fetch", fetcher)
      const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: stores.checked }).catch(error => error)
      await ready.promise
      await stores.session.remove(stores.MANUAL_SESSION_KEY)
      await stores.session.set(stores.MANUAL_SESSION_KEY, stores.record)
      gate.resolve(null)
      expect(await read).toMatchObject({ status: 412 })
      expect(fetcher).not.toHaveBeenCalled()
    })

    it.each(["identical", "whitespace", "unrelated"] as const)("preserves a pending request across a %s session write", async change => {
      const { bgRequest } = await import("@/services/background-proxy")
      const stores = await setup()
      const pending = deferred<Response>()
      const fetcher = vi.fn(() => pending.promise)
      vi.stubGlobal("fetch", fetcher)
      const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: stores.checked })
      await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
      if (change === "unrelated") await stores.session.set("unrelated-setting", "value")
      else await stores.session.set(stores.MANUAL_SESSION_KEY, { ...stores.record, apiKey: change === "whitespace" ? " worker-key " : stores.record.apiKey })
      expect((await stores.resolveEffectiveTldwConfig(stores))?.apiKey).toBe("worker-key")
      pending.resolve(json({ id: 7, name: "Current owner" }))
      expect(await read).toMatchObject({ name: "Current owner" })
      expect(fetcher).toHaveBeenCalledTimes(1)
    })
  })

  it("preserves a pending request across a same-principal JWT storage refresh", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { createSafeStorage } = await import("@/utils/safe-storage")
    const persistent = createSafeStorage({ area: "local" })
    const jwt = (iat: number) => `test.${btoa(JSON.stringify({ sub: "alice", iat }))}.signature`
    const checked = { serverUrl: config.serverUrl, authMode: "multi-user" as const, accessToken: jwt(1) }
    await persistent.set("tldwConfig", checked)
    const pending = deferred<Response>()
    const fetcher = vi.fn(() => pending.promise)
    vi.stubGlobal("fetch", fetcher)
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: checked })
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    await persistent.set("tldwConfig", { ...checked, accessToken: jwt(2) })
    pending.resolve(json({ id: 7, name: "Alice" }))
    expect(await read).toMatchObject({ name: "Alice" })
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it("clears the handshake timer and does not dispatch after cancellation during the check", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const checked = deferred<unknown>()
    harness.authorityReplyGate = checked.promise
    const controller = new AbortController()
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    vi.useFakeTimers()
    const timers = vi.getTimerCount()
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config, abortSignal: controller.signal }).catch(error => error)
    expect(harness.sent).toHaveLength(1)
    controller.abort()
    checked.resolve(null)
    expect(await read).toMatchObject({ name: "AbortError" })
    expect(harness.sent).toHaveLength(1)
    expect(fetcher).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(timers)
  })

  it.each(["error", "timeout", "missing epoch"])("does not fall back to direct transport after a handshake %s", async kind => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { browser } = await import("wxt/browser")
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    const sendMessage = vi.spyOn(browser.runtime, "sendMessage")
    if (kind === "error") sendMessage.mockRejectedValueOnce(new Error("Worker unavailable"))
    else if (kind === "missing epoch") sendMessage.mockResolvedValueOnce({ ok: true } as never)
    else sendMessage.mockImplementationOnce(() => new Promise(() => {}))
    vi.useFakeTimers()
    const timers = vi.getTimerCount()
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config }).catch(error => error)
    if (kind === "timeout") await vi.runAllTimersAsync()
    expect(await read).toBeInstanceOf(Error)
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(fetcher).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(timers)
  })

  it.each(["error", "timeout"])("does not lose the checked worker epoch on a product-message %s", async kind => {
    const { bgRequest } = await import("@/services/background-proxy")
    const { browser } = await import("wxt/browser")
    const fetcher = vi.fn(async () => json({ id: 7, name: "Unchecked fallback" }))
    vi.stubGlobal("fetch", fetcher)
    const send = browser.runtime.sendMessage.bind(browser.runtime)
    const sendMessage = vi.spyOn(browser.runtime, "sendMessage").mockImplementationOnce(send)
    if (kind === "error") sendMessage.mockRejectedValueOnce(new Error("Worker unavailable"))
    else sendMessage.mockImplementationOnce(() => new Promise(() => {}))
    vi.useFakeTimers()
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: config }).catch(error => error)
    if (kind === "timeout") await vi.runAllTimersAsync()
    expect(await read).toBeInstanceOf(Error)
    expect(fetcher).not.toHaveBeenCalled()
  })

  it("rejects an identical cookie-config logout/login roundtrip after checking the worker epoch", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const cookie = { serverUrl: config.serverUrl, authMode: "single-user" as const, authSource: "cookie-session" as const }
    harness.values.set("tldwConfig", cookie)
    harness.beforeRequest = () => {
      for (const listener of harness.changes) {
        listener({ tldwCookieSessionConfig: { oldValue: cookie, newValue: undefined } }, "local")
        listener({ tldwCookieSessionConfig: { oldValue: undefined, newValue: cookie } }, "local")
      }
    }
    const fetcher = vi.fn(async () => json({ id: 7, name: "Wrong account" }))
    vi.stubGlobal("fetch", fetcher)
    await expect(bgRequest({ path: "/api/v1/characters/7", configSnapshot: cookie, noAuth: true })).rejects.toMatchObject({ status: 412 })
    expect(fetcher).not.toHaveBeenCalled()
  })

  it("rejects a cookie-config epoch change while the worker response is pending", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const cookie = { serverUrl: config.serverUrl, authMode: "single-user" as const, authSource: "cookie-session" as const }
    harness.values.set("tldwConfig", cookie)
    const pending = deferred<Response>()
    const fetcher = vi.fn(() => pending.promise)
    vi.stubGlobal("fetch", fetcher)
    const read = bgRequest({ path: "/api/v1/characters/7", configSnapshot: cookie, noAuth: true }).catch(error => error)
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    for (const listener of harness.changes) listener({ tldwCookieSessionConfig: { oldValue: cookie, newValue: undefined } }, "local")
    pending.resolve(json({ id: 7, name: "Old account" }))
    expect(await read).toMatchObject({ status: 412 })
  })

  it("does not allow a runtime API-key override to replace the fenced worker credential", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const runtime = await import("@/services/tldw/runtime-auth-override")
    vi.spyOn(runtime, "getRuntimeSingleUserApiKeyOverride").mockReturnValue("different-runtime-key")
    const fetcher = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => json({ id: 7, name: "Alice" }))
    vi.stubGlobal("fetch", fetcher)
    expect(await bgRequest({ path: "/api/v1/characters/7", configSnapshot: config })).toMatchObject({ name: "Alice" })
    expect(new Headers(fetcher.mock.calls[0][1]?.headers).get("X-API-Key")).toBe("worker-key")
    expect(JSON.stringify(harness.sent)).not.toContain("different-runtime-key")
  })

  it("fails closed for combined snapshot and Service Prompt scopes", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    await expect(bgRequest({ path: "/api/v1/characters/7", configSnapshot: config, servicePromptConfig: requestScope.config })).rejects.toMatchObject({ status: 412 })
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each(["public", "base", "domain"] as const)("%s rejects mismatched worker authority without populating shared caches", async adapter => {
    const { TldwApiClient, TldwApiClientBase } = await import("@/services/tldw/TldwApiClient")
    const { characterMethods } = await import("@/services/tldw/domains/characters")
    const { chatRagMethods } = await import("@/services/tldw/domains/chat-rag")
    const client = new TldwApiClient()
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0] as `/${string}`)
    harness.workerConfig = { ...config, apiKey: "different-worker-key" }
    const fetcher = vi.fn(async () => json({ id: 7, name: "Wrong account" }))
    vi.stubGlobal("fetch", fetcher)
    const character = adapter === "base" ? TldwApiClientBase.prototype.getCharacter
      : adapter === "domain" ? characterMethods.getCharacter : client.getCharacter
    const messages = adapter === "base" ? TldwApiClientBase.prototype.listChatMessages
      : adapter === "domain" ? chatRagMethods.listChatMessages : client.listChatMessages
    await expect(character.call(client, 7)).rejects.toMatchObject({ status: 412 })
    await expect(messages.call(client, "7")).rejects.toMatchObject({ status: 412 })
    expect(fetcher).not.toHaveBeenCalled()
    expect(client.characterCache.size).toBe(0)
    expect(client.chatMessagesCache.size).toBe(0)
    const wire = JSON.stringify(harness.sent)
    expect(wire).not.toContain("worker-key")
    expect(wire).not.toContain("apiKey")
    expect(wire).not.toContain("accessToken")
  })

  it.each([
    ["server", { ...config, serverUrl: "https://other.test" }],
    ["organization", { ...config, orgId: 2 }],
    ["source", { ...config, authSource: "env" }],
    ["removed credential", { ...config, apiKey: undefined }]
  ])("rejects a worker %s change before fetch", async (_kind, workerConfig) => {
    const { bgRequest } = await import("@/services/background-proxy")
    harness.workerConfig = workerConfig
    const fetcher = vi.fn(async () => json({ id: 7 }))
    vi.stubGlobal("fetch", fetcher)
    await expect(bgRequest({ path: "/api/v1/characters/7", configSnapshot: config })).rejects.toMatchObject({ status: 412 })
    expect(fetcher).not.toHaveBeenCalled()
  })

  it("allows a refreshed JWT for the same worker principal without sending credentials", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    const jwt = (iat: number) => `test.${btoa(JSON.stringify({ sub: "alice", iat }))}.signature`
    const checked = { serverUrl: config.serverUrl, authMode: "multi-user" as const, accessToken: jwt(1) }
    harness.workerConfig = { ...checked, accessToken: jwt(2) }
    const fetcher = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => json({ id: 7, name: "Alice" }))
    vi.stubGlobal("fetch", fetcher)
    expect(await bgRequest({ path: "/api/v1/characters/7", configSnapshot: checked })).toMatchObject({ name: "Alice" })
    expect(new Headers(fetcher.mock.calls[0][1]?.headers).get("Authorization")).toBe(`Bearer ${jwt(2)}`)
    expect(JSON.stringify(harness.sent)).not.toContain(jwt(1))
    expect(JSON.stringify(harness.sent)).not.toContain(jwt(2))
  })
})

describe("native Quick Ingest worker ownership", () => {
  it("keeps processing after the initiating UI message returns and no UI listener remains", async () => {
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async (url: string) => url.endsWith("/ingest/jobs") ? uploaded.promise : json({ status: "completed", result: { media_id: 7 } }))
    vi.stubGlobal("fetch", fetcher)
    const ack = await send({ type: "tldw:quick-ingest/start", payload: batch }) as { sessionId: string }
    expect(ack.sessionId).toBeTruthy()
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    // The UI is already gone: only the worker's runtime handler remains.
    uploaded.resolve(json({ batch_id: "owned-batch", jobs: [{ id: 7 }] }))
    await vi.waitFor(() => expect(harness.sent).toContainEqual(expect.objectContaining({ type: "tldw:quick-ingest/completed", payload: expect.objectContaining({ sessionId: ack.sessionId }) })))
    expect(fetcher.mock.calls.map(call => call[0])).toEqual(["https://api.example.test/base/api/v1/media/ingest/jobs", "https://api.example.test/base/api/v1/media/ingest/jobs/7"])
  })

  it("rejects a legacy unowned start and a foreign colliding cancellation before dispatch", async () => {
    const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher)
    expect(await send({ type: "tldw:quick-ingest/start", payload: { ...batch, requestScope: undefined } })).toMatchObject({ ok: false })
    expect(await send({ type: "tldw:quick-ingest/cancel", payload: { sessionId: "qi-old", requestScope: { ...requestScope, config: { ...requestScope.config, serverUrl: "https://foreign.test" } } } })).toMatchObject({ ok: false })
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each([false, true])("resumes saved worker jobs only for their exact stored authority (foreign=%s)", async (foreign) => {
    const fetcher = vi.fn(async () => json({ status: "completed", result: { media_id: 7 } })); vi.stubGlobal("fetch", fetcher)
    // A fresh worker hydrates only metadata, never raw credentials.
    harness.listeners.clear(); harness.changes.clear()
    harness.saved = { "tldw:backgroundSessionStateV1": {
      ingestSessions: {}, pendingAuthReplay: [],
      quickIngestSessions: [{ sessionId: "qi-restored", cancelled: false, requestScope }],
      quickIngestBatches: [{ sessionId: "qi-restored", requestScope, totalCount: 1, processedCount: 0, ingestTimeoutMs: 60000,
        remoteJobs: [{ jobId: 7, batchId: "saved-batch", meta: { id: "private", type: "pdf", fileName: "Owned.pdf" } }], collectedResults: [], plannedConferenceItems: [] }]
    } }
    if (foreign) harness.values.set("tldwConfig", { ...config, serverUrl: "https://foreign.test", apiKeyServerOrigin: "https://foreign.test" })
    const background = (await import("@/entries/background")).default
    background.main()
    if (foreign) {
      await new Promise(resolve => setTimeout(resolve, 30))
      expect(fetcher).not.toHaveBeenCalled()
      expect(harness.sent.filter((message) => typeof message === "object" && message !== null && "type" in message && typeof message.type === "string" && message.type.startsWith("tldw:quick-ingest/"))).toEqual([])
    } else {
      await vi.waitFor(() => expect(harness.sent).toContainEqual(expect.objectContaining({ type: "tldw:quick-ingest/completed", payload: expect.objectContaining({ sessionId: "qi-restored" }) })))
      expect(fetcher).toHaveBeenCalledTimes(1)
      expect(fetcher.mock.calls[0][0]).toBe("https://api.example.test/base/api/v1/media/ingest/jobs/7")
    }
  })

  it("drops an upload completion across A to B to A without polling or server-job cancellation", async () => {
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async () => uploaded.promise); vi.stubGlobal("fetch", fetcher)
    await send({ type: "tldw:quick-ingest/start", payload: batch })
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    switchConfig({ ...config, serverUrl: "https://other.test", apiKeyServerOrigin: "https://other.test" }); switchConfig(config)
    uploaded.resolve(json({ batch_id: "foreign-colliding-batch", jobs: [{ id: 7 }] }))
    await new Promise(resolve => setTimeout(resolve, 30))
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(harness.sent.filter((message) => typeof message === "object" && message !== null && "type" in message && typeof message.type === "string" && message.type.startsWith("tldw:quick-ingest/"))).toEqual([])
  })

  it('regression worker stale marker check preserves a valid rotated in-flight ingest', async () => {
    const {refreshSessionInvalidationKey,storeRefreshRotationIfCurrent}=await import('@/services/tldw/single-user-credential')
    const token=(revision:number)=>'test.'+btoa(JSON.stringify({sub:'1',revision}))+'.signature'
    const original={serverUrl:config.serverUrl,authMode:'multi-user' as const,authSource:'manual' as const,accessToken:token(1),refreshToken:'worker-source-refresh'}
    harness.values.set('tldwConfig',original)
    const ownScope={config:{serverUrl:original.serverUrl,authMode:original.authMode,authSource:original.authSource},userId:1}
    const uploaded=deferred<Response>()
    const fetcher=vi.fn(async(url:string)=>url.endsWith('/ingest/jobs')?uploaded.promise:json({status:'completed',result:{media_id:7}}))
    vi.stubGlobal('fetch',fetcher)
    const ack=await send({type:'tldw:quick-ingest/start',payload:{...batch,requestScope:ownScope}}) as {sessionId:string}
    await vi.waitFor(()=>expect(fetcher).toHaveBeenCalledTimes(1))
    const key=refreshSessionInvalidationKey(original)!
    harness.values.set(key,true)
    const pending=deferred<boolean>()
    let intercepted=false
    const rawGet=harness.values.get.bind(harness.values)
    const get=vi.spyOn(harness.values,'get').mockImplementation(keyName=>{
      if(keyName===key&&!intercepted){intercepted=true;return pending.promise}
      return rawGet(keyName)
    })
    try {
      for(const listener of harness.changes)listener({[key]:{newValue:true}},'local')
      await vi.waitFor(()=>expect(intercepted).toBe(true))
      const storage={get:async<T,>(key:string)=>harness.values.get(key) as T,set:async<T,>(key:string,value:T)=>{harness.values.set(key,value)},remove:async(key:string)=>{harness.values.delete(key)}}
      expect(await storeRefreshRotationIfCurrent(storage,original,original.refreshToken,{accessToken:token(2),refreshToken:'worker-rotated-refresh'})).toBe(true)
      for(const listener of harness.changes)listener({tldwRefreshRotation:{newValue:harness.values.get('tldwRefreshRotation')}},'local')
      pending.resolve(true)
      await new Promise(resolve=>setTimeout(resolve,0))
      uploaded.resolve(json({batch_id:'owned-batch',jobs:[{id:7}]}))
      await vi.waitFor(()=>expect(harness.sent).toContainEqual(expect.objectContaining({type:'tldw:quick-ingest/completed',payload:expect.objectContaining({sessionId:ack.sessionId})})))
      expect(fetcher).toHaveBeenCalledTimes(2)
    } finally {pending.resolve(true);uploaded.resolve(json({batch_id:'owned-batch',jobs:[{id:7}]}));get.mockRestore()}
  })

  it("abandons current terminal worker credentials without cancelling another owner's jobs", async () => {
    const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
    const original = { serverUrl: config.serverUrl, authMode: "multi-user" as const, authSource: "manual" as const, accessToken: "test." + btoa(JSON.stringify({ sub: "1" })) + ".signature", refreshToken: "terminal-refresh" }
    harness.values.set("tldwConfig", original)
    const ownScope = { config: { serverUrl: original.serverUrl, authMode: original.authMode, authSource: original.authSource }, userId: 1 }
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => uploaded.promise)
    vi.stubGlobal("fetch", fetcher)
    await send({ type: "tldw:quick-ingest/start", payload: { ...batch, requestScope: ownScope } })
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    const key = refreshSessionInvalidationKey(original)!
    harness.values.set(key, true)
    for (const listener of harness.changes) listener({ [key]: { newValue: true } }, "local")
    await vi.waitFor(() => expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true))
    uploaded.resolve(json({ batch_id: "old-owner", jobs: [{ id: 7 }] }))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
