// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import type {
  FirstRunMetadata,
  FirstRunState,
  SetupProviderSaveRequest
} from "@/types/setup-onboarding"

// Keep the actual hook, active Object.assign API domain, proxy, request-core and
// extension worker account/epoch/abort handling. Only physical storage/browser/
// fetch IO is declared. The two unrelated worker startup routines below are
// excluded so model warming and notification subscriptions cannot own the gate.
const physical = await vi.hoisted(async () => {
  const { AsyncLocalStorage } = await import("node:async_hooks")
  type Surface = "caller" | "worker"
  const surfaces = new AsyncLocalStorage<Surface>()
  const context = { enabled: false }
  const withSurface = <T,>(surface: Surface, run: () => T): T =>
    context.enabled ? surfaces.run(surface, run) : run()
  const currentSurface = (): Surface => surfaces.getStore() ?? "caller"
  type Change = { oldValue?: unknown; newValue?: unknown }
  type Watch = Record<string, (change: Change) => void>
  type Read = { area: string; key: string; value: unknown; stack: string }
  type RuntimeMessage = { type?: string; payload?: Record<string, unknown> }
  type Listener = (
    message: RuntimeMessage,
    sender: { id: string },
    reply: (value: unknown) => void
  ) => unknown
  const areas = new Map<string, Map<string, unknown>>()
  const watchers = new Map<string, Set<Watch>>()
  const storageListeners = new Set<
    (changes: Record<string, Change>, area: string) => void
  >()
  const watchSurfaces = new WeakMap<Watch, Surface>()
  const storageSurfaces = new WeakMap<
    (changes: Record<string, Change>, area: string) => void,
    Surface
  >()
  const runtimeListeners = new Set<Listener>()
  const control = {
    beforeRead: undefined as ((read: Read) => Promise<unknown>) | undefined
  }
  const runtime = { id: undefined as string | undefined }
  const sent: RuntimeMessage[] = []
  const writes: { area: string; key: string; value: unknown }[] = []
  const inFlightMessages = new Set<Promise<unknown>>()
  const areaFor = (area: string) => {
    if (!areas.has(area)) areas.set(area, new Map())
    return areas.get(area)!
  }
  const publish = (
    area: string,
    key: string,
    oldValue: unknown,
    newValue: unknown
  ) => {
    const change = {
      oldValue: structuredClone(oldValue),
      newValue: structuredClone(newValue)
    }
    for (const subscription of watchers.get(area) ?? [])
      withSurface(watchSurfaces.get(subscription) ?? "caller", () =>
        subscription[key]?.(change)
      )
    for (const listener of storageListeners)
      withSurface(storageSurfaces.get(listener) ?? "caller", () =>
        listener({ [key]: change }, area)
      )
  }
  const storage = (area = "local") => ({
    area,
    hasPersistentBackend: true,
    get: async (key: string) => {
      const value = structuredClone(areaFor(area).get(key))
      if (control.beforeRead)
        return control.beforeRead({
          area,
          key,
          value,
          stack: new Error().stack ?? ""
        })
      return value
    },
    set: async (key: string, value: unknown) => {
      const rows = areaFor(area),
        previous = rows.get(key)
      writes.push({ area, key, value: structuredClone(value) })
      rows.set(key, structuredClone(value))
      publish(area, key, previous, value)
    },
    remove: async (key: string) => {
      const rows = areaFor(area),
        previous = rows.get(key)
      rows.delete(key)
      publish(area, key, previous, undefined)
    },
    getAll: async () => structuredClone(Object.fromEntries(areaFor(area))),
    watch: (subscription: Watch) => {
      if (!watchers.has(area)) watchers.set(area, new Set())
      watchers.get(area)!.add(subscription)
      watchSurfaces.set(subscription, currentSurface())
      return () => {
        watchers.get(area)?.delete(subscription)
      }
    },
    unwatch: (subscription: Watch) => {
      watchers.get(area)?.delete(subscription)
    }
  })
  const browserArea = (area: string) => ({
    get: async (keys?: string | string[] | Record<string, unknown> | null) => {
      const names =
        typeof keys === "string"
          ? [keys]
          : Array.isArray(keys)
            ? keys
            : keys
              ? Object.keys(keys)
              : [...areaFor(area).keys()]
      return Object.fromEntries(
        names.map((key) => [key, structuredClone(areaFor(area).get(key))])
      )
    },
    set: async (values: Record<string, unknown>) => {
      for (const [key, value] of Object.entries(values))
        await storage(area).set(key, value)
    },
    remove: async (keys: string | string[]) => {
      for (const key of typeof keys === "string" ? [keys] : keys)
        await storage(area).remove(key)
    }
  })
  const sendMessage = vi.fn((message: RuntimeMessage) => {
    sent.push(structuredClone(message))
    const pending = new Promise<unknown>((resolve, reject) => {
      const listener = [...runtimeListeners][0]
      if (!listener) {
        reject(new Error("No real extension worker listener"))
        return
      }
      try {
        withSurface("worker", () =>
          listener(message, { id: "setup-transport-fixture" }, resolve)
        )
      } catch (error) {
        reject(error)
      }
    })
    inFlightMessages.add(pending)
    void pending
      .finally(() => {
        inFlightMessages.delete(pending)
      })
      .catch(() => undefined)
    return pending
  })
  return {
    areas,
    watchers,
    storageListeners,
    storageSurfaces,
    runtimeListeners,
    control,
    runtime,
    sent,
    writes,
    context,
    withSurface,
    currentSurface,
    inFlightMessages,
    areaFor,
    storage,
    browserArea,
    sendMessage,
    fetch: vi.fn()
  }
})
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: (options: { area?: string } = {}) =>
    physical.storage(options.area)
}))
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "tldw:model-warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn(), removeListener: vi.fn() })
  return {
    browser: {
      runtime: {
        get id() {
          return physical.runtime.id
        },
        getURL: (path: string) =>
          "chrome-extension://setup-transport-fixture" + path,
        sendMessage: physical.sendMessage,
        onMessage: {
          addListener: (
            listener: Parameters<typeof physical.runtimeListeners.add>[0]
          ) => physical.runtimeListeners.add(listener)
        },
        onConnect: event(),
        onStartup: event()
      },
      storage: {
        local: physical.browserArea("local"),
        session: physical.browserArea("session"),
        onChanged: {
          addListener: (
            listener: Parameters<typeof physical.storageListeners.add>[0]
          ) => {
            physical.storageSurfaces.set(listener, physical.currentSurface())
            physical.storageListeners.add(listener)
          },
          removeListener: (
            listener: Parameters<typeof physical.storageListeners.delete>[0]
          ) => physical.storageListeners.delete(listener)
        }
      },
      alarms: {
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

const originalConfig: TldwConfig = {
  serverUrl: "https://setup-original.example",
  authMode: "single-user",
  authSource: "manual",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://setup-original.example",
  apiKey: "declared-original-setup-server-key"
}
const laterConfig: TldwConfig = {
  ...originalConfig,
  serverUrl: "https://setup-later.example",
  apiKeyServerOrigin: "https://setup-later.example",
  apiKey: "declared-later-setup-server-key"
}
const accessFor = (user: number, rotation: number) =>
  `header.${btoa(JSON.stringify({ sub: String(user), nonce: rotation }))}.signature`
const originalMultiConfig: TldwConfig = {
  serverUrl: originalConfig.serverUrl,
  authMode: "multi-user",
  authSource: "manual",
  accessToken: accessFor(42, 1),
  refreshToken: "declared-original-setup-refresh-token"
}
const laterMultiConfig: TldwConfig = {
  serverUrl: laterConfig.serverUrl,
  authMode: "multi-user",
  authSource: "manual",
  accessToken: accessFor(84, 1),
  refreshToken: "declared-later-setup-refresh-token"
}
const originalRefreshResponse = () =>
  Response.json({
    access_token: accessFor(42, 2),
    refresh_token: "declared-original-rotated-refresh-token",
    token_type: "bearer"
  })
const payload: SetupProviderSaveRequest = {
  provider_key: "openai",
  api_key: "declared-original-provider-key",
  model: "original-model",
  make_default: true
}
const stateFor = (origin: string): FirstRunState => ({
  status: "in_progress",
  completed_steps: [],
  skipped_steps: [],
  acknowledged_steps: [],
  step_data: { transport_probe: { api_origin: origin } },
  first_chat: { completed: false }
})
const metadataFor = (
  origin: string,
  authMode: FirstRunMetadata["auth_mode"] = "single_user"
): FirstRunMetadata => ({
  auth_mode: authMode,
  bundled_single_user_auth_available: false,
  manual_auth_required: true,
  setup_required: true,
  setup_completed: false,
  remote_setup_enabled: false,
  connection: { api_origin: origin, browser_access: "remote" },
  setup_paths: [],
  multi_user_exit: { guide_path: "/docs/multi-user" }
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
type Wire = {
  url: string
  method: string
  headers: Headers
  body: Record<string, unknown> | null
  signal?: AbortSignal | null
}
let wires: Wire[]
let rejectAuthenticatedState: boolean
let heldProviderResponse: ReturnType<typeof deferred<Response>> | undefined
let rejectOriginalBearerState: boolean
let rejectOriginalBearerStateReads: number
let metadataAuthMode: FirstRunMetadata["auth_mode"]
let heldAuthResponse: ReturnType<typeof deferred<Response>> | undefined
let rejectOriginalAuthRefresh: boolean
const providerWires = () =>
  wires.filter(
    (wire) =>
      wire.method === "POST" &&
      new URL(wire.url).pathname === "/api/v1/setup/first-run/providers"
  )
const refreshWires = () =>
  wires.filter(
    (wire) =>
      wire.method === "POST" &&
      new URL(wire.url).pathname === "/api/v1/auth/refresh"
  )
const rotationWrites = () =>
  physical.writes.filter(
    (write) => write.area === "local" && write.key === "tldwRefreshRotation"
  )
const awaitPhysicalWorkerIdle = async () => {
  while (physical.inFlightMessages.size)
    await Promise.allSettled([...physical.inFlightMessages])
}
const holdPhysicalResponse = (
  signal: AbortSignal | null | undefined,
  response: Promise<Response>
) =>
  new Promise<Response>((resolve, reject) => {
    const onAbort = () =>
      reject(new DOMException("Original setup fetch aborted", "AbortError"))
    signal?.addEventListener("abort", onAbort, { once: true })
    if (signal?.aborted) onAbort()
    response.then(resolve, reject).finally(() => {
      signal?.removeEventListener("abort", onAbort)
    })
  })

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  physical.areas.clear()
  physical.watchers.clear()
  physical.storageListeners.clear()
  physical.runtimeListeners.clear()
  physical.sent.length = 0
  physical.writes.length = 0
  physical.inFlightMessages.clear()
  physical.control.beforeRead = undefined
  physical.context.enabled = false
  physical.runtime.id = undefined
  localStorage.clear()
  sessionStorage.clear()
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "self_host")
  vi.stubEnv("NEXT_PUBLIC_API_URL", "")
  vi.stubGlobal("chrome", undefined)
  physical.areaFor("local").set("tldwConfig", structuredClone(originalConfig))
  wires = []
  rejectAuthenticatedState = false
  heldProviderResponse = undefined
  rejectOriginalBearerState = false
  rejectOriginalBearerStateReads = 0
  metadataAuthMode = "single_user"
  heldAuthResponse = undefined
  rejectOriginalAuthRefresh = false
  physical.fetch.mockImplementation(
    async (url: string, init: RequestInit = {}) => {
      // Native fetch does not put an already-aborted request on the wire.
      if (init.signal?.aborted)
        throw new DOMException("Aborted before physical fetch", "AbortError")
      const target = new URL(url),
        headers = new Headers(init.headers),
        method = String(init.method || "GET")
      const body =
        init.body == null
          ? null
          : (JSON.parse(String(init.body)) as Record<string, unknown>)
      wires.push({ url, method, headers, body, signal: init.signal })
      if (
        target.pathname === "/api/v1/setup/first-run/state" &&
        method === "GET"
      ) {
        if (
          (rejectOriginalBearerState || rejectOriginalBearerStateReads > 0) &&
          target.origin === originalMultiConfig.serverUrl &&
          headers.get("Authorization") ===
            "Bearer " + originalMultiConfig.accessToken
        ) {
          if (rejectOriginalBearerStateReads > 0)
            rejectOriginalBearerStateReads -= 1
          else rejectOriginalBearerState = false
          return Response.json(
            { detail: "Declared authenticated setup state 401" },
            { status: 401 }
          )
        }
        if (rejectAuthenticatedState && headers.has("X-API-KEY")) {
          rejectAuthenticatedState = false
          return Response.json(
            { detail: "Declared pre-auth setup response" },
            { status: 401 }
          )
        }
        return Response.json(stateFor(target.origin))
      }
      if (target.pathname === "/api/v1/setup/first-run/state" && method === "POST") {
        return Response.json(stateFor(target.origin))
      }
      if (target.pathname === "/api/v1/setup/first-run/metadata")
        return Response.json(metadataFor(target.origin, metadataAuthMode))
      if (target.pathname === "/api/v1/auth/refresh" && method === "POST") {
        if (heldAuthResponse)
          return await holdPhysicalResponse(
            init.signal,
            heldAuthResponse.promise
          )
        if (
          rejectOriginalAuthRefresh &&
          target.origin === originalMultiConfig.serverUrl &&
          body?.refresh_token === originalMultiConfig.refreshToken
        ) {
          rejectOriginalAuthRefresh = false
          return Response.json(
            { detail: "Declared original refresh token revoked" },
            { status: 401 }
          )
        }
        if (
          target.origin === originalMultiConfig.serverUrl &&
          body?.refresh_token === originalMultiConfig.refreshToken
        )
          return originalRefreshResponse()
        if (
          target.origin === laterMultiConfig.serverUrl &&
          body?.refresh_token === laterMultiConfig.refreshToken
        )
          return Response.json({
            access_token: accessFor(84, 2),
            refresh_token: "declared-later-rotated-refresh-token",
            token_type: "bearer"
          })
        return Response.json(
          {
            detail:
              "Declared refresh target rejects the other connection's token"
          },
          { status: 401 }
        )
      }
      if (
        target.pathname === "/api/v1/setup/first-run/providers" &&
        method === "POST"
      ) {
        if (heldProviderResponse) {
          // A physical fetch stays cancellable while awaiting response headers.
          return await new Promise<Response>((resolve, reject) => {
            const onAbort = () =>
              reject(
                new DOMException("Original setup fetch aborted", "AbortError")
              )
            init.signal?.addEventListener("abort", onAbort, { once: true })
            if (init.signal?.aborted) onAbort()
            heldProviderResponse!.promise.then(resolve, reject).finally(() => {
              init.signal?.removeEventListener("abort", onAbort)
            })
          })
        }
        return Response.json({
          provider_key: body?.provider_key,
          status: "saved",
          credential_configured: true,
          masked_api_key: "declared-...key",
          model: body?.model,
          make_default: body?.make_default,
          api_key: body?.api_key,
          provider: { api_key: body?.api_key }
        })
      }
      throw new Error(
        "Unexpected physical setup request: " + method + " " + target.pathname
      )
    }
  )
  vi.stubGlobal("fetch", physical.fetch)
})
afterEach(async () => {
  cleanup()
  physical.control.beforeRead = undefined
  await awaitPhysicalWorkerIdle()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

const mountOriginal = async (
  transport: "direct" | "extension",
  config: TldwConfig = originalConfig
) => {
  physical.areaFor("local").set("tldwConfig", structuredClone(config))
  metadataAuthMode =
    config.authMode === "multi-user" ? "multi_user" : "single_user"
  if (transport === "extension") {
    physical.runtime.id = "setup-transport-fixture"
    vi.stubGlobal("defineBackground", (entry: unknown) => entry)
    // Use the existing real-worker fixture contract: startup has no window;
    // the caller keeps its real DOM and runtime messaging chooses the handoff.
    const callerWindow = globalThis.window
    if (config.authMode === "multi-user") {
      // A worker and its caller have distinct physical global surfaces. Keep
      // that distinction across real async refreshes without replacing any
      // API, ownership helper, worker handler or cancellation implementation.
      physical.context.enabled = true
      vi.stubGlobal("window", callerWindow)
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        get: () =>
          physical.currentSurface() === "worker" ? undefined : callerWindow
      })
      await physical.withSurface("worker", async () => {
        const background = (await import("@/entries/background")).default
        background.main()
      })
    } else {
      vi.stubGlobal("window", undefined)
      const background = (await import("@/entries/background")).default
      background.main()
      vi.stubGlobal("window", callerWindow)
    }
    expect(physical.runtimeListeners.size).toBe(1)
  }
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  await tldwClient.initialize()
  expect(await tldwClient.getConfig()).toMatchObject(config)
  const { useSetupOnboarding } = await import("../useSetupOnboarding")
  const mounted = renderHook(() => useSetupOnboarding())
  await waitFor(() =>
    expect(mounted.result.current.metadata).toEqual(
      metadataFor(config.serverUrl, metadataAuthMode)
    )
  )
  expect(mounted.result.current.state).toEqual(stateFor(config.serverUrl))
  expect(mounted.result.current.connectionConfigured).toBe(true)
  expect(mounted.result.current.error).toBeNull()
  await act(async () => {
    await Promise.resolve()
  })
  return mounted
}
const replaceConfig = (config: TldwConfig) => {
  // Deliver real physical account events to both the hook and the worker epoch.
  void physical.storage("local").set("tldwConfig", config)
}

describe.each(["direct", "extension"] as const)(
  "setup mutation ownership through actual %s transport",
  (transport) => {
    it("sends the original provider configuration through first-run noAuth and preserves a same-authority save event", async () => {
      const mounted = await mountOriginal(transport)
      let response!: Awaited<
        ReturnType<typeof mounted.result.current.saveProvider>
      >
      await act(async () => {
        response = await mounted.result.current.saveProvider(payload)
      })
      expect(providerWires()).toHaveLength(1)
      const wire = providerWires()[0]
      expect(wire.url).toBe(
        originalConfig.serverUrl + "/api/v1/setup/first-run/providers"
      )
      expect(wire.body).toEqual(payload)
      expect(wire.headers.has("X-API-KEY")).toBe(false)
      expect(wire.headers.has("X-TLDW-Expected-User-ID")).toBe(false)
      expect(response).toMatchObject({
        status: "saved",
        provider_key: "openai",
        credential_configured: true,
        model: "original-model"
      })
      expect(JSON.stringify(response)).not.toContain(payload.api_key)
      expect(mounted.result.current.state).toEqual(
        stateFor(originalConfig.serverUrl)
      )
      expect(mounted.result.current.error).toBeNull()
      if (transport === "extension") {
        const request = physical.sent.find(
          (message) =>
            message.type === "tldw:request" &&
            message.payload?.method === "POST"
        )
        expect(request?.payload).toMatchObject({
          path: "/api/v1/setup/first-run/providers",
          noAuth: true,
          body: payload
        })
        expect(request?.payload?.servicePromptConfig).toBeUndefined()
      }
    })

    it("retains the actual authenticated-state 401-only public setup fallback", async () => {
      rejectAuthenticatedState = true
      await mountOriginal(transport)
      const stateReads = wires.filter(
        (wire) => new URL(wire.url).pathname === "/api/v1/setup/first-run/state"
      )
      expect(stateReads).toHaveLength(2)
      expect(stateReads.map((wire) => wire.url)).toEqual([
        originalConfig.serverUrl + "/api/v1/setup/first-run/state",
        originalConfig.serverUrl + "/api/v1/setup/first-run/state"
      ])
      expect(stateReads[0].headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
      expect(stateReads[1].headers.has("X-API-KEY")).toBe(false)
      expect(providerWires()).toEqual([])
    })

    it("aborts the original physical mutation when its connection retires during an in-flight response", async () => {
      const mounted = await mountOriginal(transport)
      const response = deferred<Response>()
      heldProviderResponse = response
      let pending: Promise<unknown> | undefined
      try {
        pending = mounted.result.current
          .saveProvider(payload)
          .catch((error) => error)
        await waitFor(() => expect(providerWires()).toHaveLength(1))
        const wire = providerWires()[0]
        expect(wire.url).toBe(
          originalConfig.serverUrl + "/api/v1/setup/first-run/providers"
        )
        expect(wire.body).toEqual(payload)
        expect(wire.signal?.aborted).toBe(false)
        act(() => {
          replaceConfig(laterConfig)
        })
        await waitFor(() => expect(wire.signal?.aborted).toBe(true))
        let result: unknown
        await act(async () => {
          result = await pending
        })
        expect(result).toBeInstanceOf(Error)
        expect(providerWires()).toHaveLength(1)
        await waitFor(() =>
          expect(mounted.result.current.metadata).toEqual(
            metadataFor(laterConfig.serverUrl)
          )
        )
        expect(mounted.result.current.state).toEqual(
          stateFor(laterConfig.serverUrl)
        )
        expect(mounted.result.current.error).toBeNull()
        if (transport === "extension") {
          const request = physical.sent.find(
            (message) =>
              message.type === "tldw:request" &&
              message.payload?.method === "POST"
          )
          expect(request?.payload?.requestId).toEqual(expect.any(String))
          expect(physical.sent).toContainEqual({
            type: "tldw:cancel-request",
            payload: { requestId: request!.payload!.requestId }
          })
        }
      } finally {
        response.resolve(
          Response.json({ status: "saved", provider_key: payload.provider_key })
        )
        if (pending)
          await act(async () => {
            await pending
          })
      }
    })

    if (transport === "direct") {
      it.each(["replacement", "ABA"] as const)(
        "does not dispatch an old direct provider mutation after %s between original readiness and API handoff",
        async (change) => {
          const mounted = await mountOriginal(transport)
          const response = deferred<Response>()
          heldProviderResponse = response
          let pending: Promise<unknown> | undefined
          let settled = false
          try {
            pending = mounted.result.current
              .saveProvider(payload)
              .catch((error) => error)
              .then((result) => {
                settled = true
                return result
              })
            // The original direct transport storage-read RED is retired explicitly:
            // configSnapshot eliminates that read. Yield only the actual readiness
            // continuation, then retire A before its API handoff. Any regressed
            // physical provider fetch remains held and visible in providerWires.
            await Promise.resolve()
            expect(settled).toBe(false)
            expect(providerWires()).toEqual([])
            act(() => {
              replaceConfig(laterConfig)
              if (change === "ABA") replaceConfig(originalConfig)
            })
            const currentConfig =
              change === "ABA" ? originalConfig : laterConfig
            await waitFor(() =>
              expect(mounted.result.current.metadata).toEqual(
                metadataFor(currentConfig.serverUrl)
              )
            )
            expect(providerWires()).toEqual([])
            let result: unknown
            await act(async () => {
              result = await pending
            })
            expect(result).toBeInstanceOf(Error)
            expect(providerWires()).toEqual([])
            expect(mounted.result.current.state).toEqual(
              stateFor(currentConfig.serverUrl)
            )
            expect(mounted.result.current.error).toBeNull()
          } finally {
            response.resolve(
              Response.json({
                status: "saved",
                provider_key: payload.provider_key
              })
            )
            if (pending)
              await act(async () => {
                await pending
              })
          }
        }
      )
    } else {
      it.each(["replacement", "ABA"] as const)(
        "does not dispatch an old provider mutation after %s during a held physical configuration read",
        async (change) => {
          const mounted = await mountOriginal(transport)
          const entered = deferred<void>(),
            held = deferred<void>()
          let heldStack = ""
          physical.control.beforeRead = async (read) => {
            if (read.area !== "local" || read.key !== "tldwConfig")
              return structuredClone(physical.areaFor(read.area).get(read.key))
            physical.control.beforeRead = undefined
            heldStack = read.stack
            entered.resolve(undefined)
            await held.promise
            return structuredClone(physical.areaFor(read.area).get(read.key))
          }
          let pending: Promise<unknown> | undefined
          try {
            pending = mounted.result.current
              .saveProvider(payload)
              .catch((error) => error)
            await Promise.race([
              entered.promise,
              pending.then(() => {
                throw new Error(
                  "Mutation settled before the declared transport configuration gate"
                )
              })
            ])
            expect(heldStack).toContain("getEffectiveConfig")
            expect(providerWires()).toEqual([])
            act(() => {
              replaceConfig(laterConfig)
              if (change === "ABA") replaceConfig(originalConfig)
            })
            const currentConfig =
              change === "ABA" ? originalConfig : laterConfig
            await waitFor(() =>
              expect(mounted.result.current.metadata).toEqual(
                metadataFor(currentConfig.serverUrl)
              )
            )
            held.resolve(undefined)
            let result: unknown
            await act(async () => {
              result = await pending
            })
            expect(result).toBeInstanceOf(Error)
            expect(providerWires()).toEqual([])
            expect(mounted.result.current.state).toEqual(
              stateFor(currentConfig.serverUrl)
            )
            expect(mounted.result.current.error).toBeNull()
          } finally {
            held.resolve(undefined)
            if (pending)
              await act(async () => {
                await pending
              })
          }
        }
      )
    }

    it("keeps the original ready owner across the hook's first await when the connection changes before API dispatch", async () => {
      const mounted = await mountOriginal(transport)
      const pending = mounted.result.current
        .saveProvider(payload)
        .catch((error) => error)
      // The resolved readiness await captures A; the API continuation runs next.
      // Retire A in between without replacing requireOwner or its authority.
      await Promise.resolve()
      act(() => {
        replaceConfig(laterConfig)
      })
      let result: unknown
      await act(async () => {
        result = await pending
      })
      expect(result).toBeInstanceOf(Error)
      expect(providerWires()).toEqual([])
      await waitFor(() =>
        expect(mounted.result.current.metadata).toEqual(
          metadataFor(laterConfig.serverUrl)
        )
      )
      expect(mounted.result.current.state).toEqual(
        stateFor(laterConfig.serverUrl)
      )
    })
  }
)

describe.each(["direct", "extension"] as const)(
  "multi-user setup state 401 ownership through actual %s refresh",
  (transport) => {
    const stateReadsSince = (index: number) =>
      wires
        .slice(index)
        .filter(
          (wire) =>
            wire.method === "GET" &&
            new URL(wire.url).pathname === "/api/v1/setup/first-run/state"
        )

    it("refreshes the original authenticated setup state and retries with its actual committed rotation", async () => {
      const mounted = await mountOriginal(transport, originalMultiConfig)
      const startIndex = wires.length
      rejectOriginalBearerState = true
      let result: FirstRunState | null | undefined
      await act(async () => {
        result = await mounted.result.current.refresh()
      })
      await awaitPhysicalWorkerIdle()
      expect(result).toEqual(stateFor(originalMultiConfig.serverUrl))
      expect(refreshWires()).toHaveLength(1)
      expect(refreshWires()[0]).toMatchObject({
        url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
        body: { refresh_token: originalMultiConfig.refreshToken }
      })
      expect(refreshWires()[0].signal?.aborted).toBe(false)
      const reads = stateReadsSince(startIndex)
      expect(reads).toHaveLength(2)
      expect(reads.map((wire) => wire.url)).toEqual([
        originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state",
        originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state"
      ])
      expect(reads.map((wire) => wire.headers.get("Authorization"))).toEqual([
        "Bearer " + originalMultiConfig.accessToken,
        "Bearer " + accessFor(42, 2)
      ])
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      await tldwClient.initialize()
      expect(await tldwClient.getConfig()).toMatchObject({
        ...originalMultiConfig,
        accessToken: accessFor(42, 2),
        refreshToken: "declared-original-rotated-refresh-token"
      })
      expect(mounted.result.current.state).toEqual(
        stateFor(originalMultiConfig.serverUrl)
      )
      expect(mounted.result.current.error).toBeNull()
      expect(providerWires()).toEqual([])
      if (transport === "extension") {
        // A real worker's auth service uses direct IO; it cannot send this
        // refresh back through its own runtime message listener.
        expect(
          physical.sent.filter(
            (message) =>
              message.type === "tldw:request" &&
              message.payload?.path === "/api/v1/auth/refresh"
          )
        ).toEqual([])
      }
    })

    it.each(["replacement", "ABA"] as const)(
      "does not dispatch a retired setup refresh after %s during its actual physical config read",
      async (change) => {
        const mounted = await mountOriginal(transport, originalMultiConfig)
        const startIndex = wires.length
        const entered = deferred<void>(),
          held = deferred<void>()
        let heldStack = ""
        physical.control.beforeRead = async (read) => {
          // The worker's original performTokenRefresh RED gate is retired:
          // fenced setup now retains its checked config in refreshScopedAuth.
          // Hold that real helper's physical storage read, not an auth result.
          const expectedCaller =
            transport === "direct" ? "refreshAuthDirect" : "refreshScopedAuth"
          if (
            read.area !== "local" ||
            read.key !== "tldwConfig" ||
            !read.stack.includes(expectedCaller)
          )
            return read.value
          physical.control.beforeRead = undefined
          heldStack = read.stack
          entered.resolve(undefined)
          await held.promise
          // Direct IO returns the original A read that was held. The worker's
          // fresh effective-config read returns the actual current stored value.
          // Neither source is an invented owner or a semantic helper result.
          return transport === "direct"
            ? read.value
            : structuredClone(physical.areaFor(read.area).get(read.key))
        }
        rejectOriginalBearerState = true
        let pending: Promise<unknown> | undefined
        try {
          pending = mounted.result.current.refresh().catch((error) => error)
          await Promise.race([
            entered.promise,
            pending.then(() => {
              throw new Error(
                "Setup refresh settled before its declared physical refresh configuration gate"
              )
            })
          ])
          expect(heldStack).toContain(
            transport === "direct" ? "refreshAuthDirect" : "refreshScopedAuth"
          )
          const originalReads = stateReadsSince(startIndex)
          expect(originalReads).toHaveLength(1)
          expect(originalReads[0].url).toBe(
            originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state"
          )
          expect(originalReads[0].headers.get("Authorization")).toBe(
            "Bearer " + originalMultiConfig.accessToken
          )
          expect(originalReads[0].signal?.aborted).toBe(false)
          expect(refreshWires()).toEqual([])
          act(() => {
            replaceConfig(laterMultiConfig)
            if (change === "ABA") replaceConfig(originalMultiConfig)
          })
          if (transport === "extension")
            await waitFor(() =>
              expect(originalReads[0].signal?.aborted).toBe(true)
            )
          else expect(originalReads[0].signal?.aborted).toBe(true)
          const current =
            change === "ABA" ? originalMultiConfig : laterMultiConfig
          await waitFor(() =>
            expect(mounted.result.current.metadata).toEqual(
              metadataFor(current.serverUrl, "multi_user")
            )
          )
          held.resolve(undefined)
          let result: unknown
          await act(async () => {
            result = await pending
            // The original extension reply can outlive the already-cancelled
            // hook operation. Wait for actual lower transport completion too.
            await awaitPhysicalWorkerIdle()
          })
          expect(refreshWires()).toEqual([])
          expect(result).toBeNull()
          expect(mounted.result.current.state).toEqual(
            stateFor(current.serverUrl)
          )
          expect(mounted.result.current.error).toBeNull()
          expect(providerWires()).toEqual([])
        } finally {
          held.resolve(undefined)
          if (pending)
            await act(async () => {
              await pending
              await awaitPhysicalWorkerIdle()
            })
        }
      }
    )

    it.each(["replacement", "ABA"] as const)(
      "cancels the original actual auth refresh fetch after %s while awaiting its response",
      async (change) => {
        const mounted = await mountOriginal(transport, originalMultiConfig)
        const startIndex = wires.length
        const startSentIndex = physical.sent.length
        const response = deferred<Response>()
        heldAuthResponse = response
        rejectOriginalBearerState = true
        let pending: Promise<unknown> | undefined
        try {
          pending = mounted.result.current.refresh().catch((error) => error)
          await waitFor(() => expect(refreshWires()).toHaveLength(1))
          const refresh = refreshWires()[0]
          expect(refresh.url).toBe(
            originalMultiConfig.serverUrl + "/api/v1/auth/refresh"
          )
          expect(refresh.body).toEqual({
            refresh_token: originalMultiConfig.refreshToken
          })
          expect(refresh.signal?.aborted).toBe(false)
          const originalReads = stateReadsSince(startIndex)
          expect(originalReads).toHaveLength(1)
          expect(originalReads[0].headers.get("Authorization")).toBe(
            "Bearer " + originalMultiConfig.accessToken
          )
          expect(originalReads[0].signal?.aborted).toBe(false)
          act(() => {
            replaceConfig(laterMultiConfig)
            if (change === "ABA") replaceConfig(originalMultiConfig)
          })
          if (transport === "extension")
            await waitFor(() =>
              expect(originalReads[0].signal?.aborted).toBe(true)
            )
          else expect(originalReads[0].signal?.aborted).toBe(true)
          await waitFor(() => expect(refresh.signal?.aborted).toBe(true))
          let result: unknown
          await act(async () => {
            result = await pending
            await awaitPhysicalWorkerIdle()
          })
          expect(result).toBeNull()
          expect(refreshWires()).toHaveLength(1)
          const current =
            change === "ABA" ? originalMultiConfig : laterMultiConfig
          await waitFor(() =>
            expect(mounted.result.current.metadata).toEqual(
              metadataFor(current.serverUrl, "multi_user")
            )
          )
          expect(mounted.result.current.state).toEqual(
            stateFor(current.serverUrl)
          )
          expect(mounted.result.current.error).toBeNull()
          expect(providerWires()).toEqual([])
          if (transport === "extension") {
            const originalRequest = physical.sent
              .slice(startSentIndex)
              .find(
                (message) =>
                  message.type === "tldw:request" &&
                  message.payload?.path === "/api/v1/setup/first-run/state"
              )
            expect(originalRequest?.payload?.requestId).toEqual(
              expect.any(String)
            )
            expect(physical.sent).toContainEqual({
              type: "tldw:cancel-request",
              payload: { requestId: originalRequest!.payload!.requestId }
            })
          }
        } finally {
          response.resolve(originalRefreshResponse())
          if (pending)
            await act(async () => {
              await pending
              await awaitPhysicalWorkerIdle()
            })
        }
      }
    )

    it("keeps a sibling setup request's actual shared refresh alive when the first caller stops", async () => {
      const mounted = await mountOriginal(transport, originalMultiConfig)
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      const configSnapshot = await tldwClient.getConfig()
      expect(configSnapshot).toMatchObject(originalMultiConfig)
      const firstController = new AbortController(),
        siblingController = new AbortController()
      const response = deferred<Response>()
      heldAuthResponse = response
      rejectOriginalBearerStateReads = 2
      const startIndex = wires.length
      let first: Promise<unknown> | undefined,
        sibling: Promise<unknown> | undefined
      let firstSettled = false,
        siblingSettled = false
      try {
        first = tldwClient
          .getFirstRunState({
            configSnapshot,
            abortSignal: firstController.signal
          })
          .catch((error) => error)
          .then((result) => {
            firstSettled = true
            return result
          })
        await waitFor(() => expect(refreshWires()).toHaveLength(1))
        sibling = tldwClient
          .getFirstRunState({
            configSnapshot,
            abortSignal: siblingController.signal
          })
          .catch((error) => error)
          .then((result) => {
            siblingSettled = true
            return result
          })
        await waitFor(() => expect(stateReadsSince(startIndex)).toHaveLength(2))
        // Both real 401 continuations use immediately-resolving declared storage
        // IO and join the held refresh. Drain that actual microtask work before
        // stopping either request; no refresh or ownership helper is replaced.
        await act(async () => {
          await Promise.resolve()
        })
        expect(firstSettled).toBe(false)
        expect(siblingSettled).toBe(false)
        expect(refreshWires()).toHaveLength(1)
        const refresh = refreshWires()[0]
        expect(refresh).toMatchObject({
          url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
          body: { refresh_token: originalMultiConfig.refreshToken }
        })
        const originalReads = stateReadsSince(startIndex)
        expect(
          originalReads.map((wire) => wire.headers.get("Authorization"))
        ).toEqual([
          "Bearer " + originalMultiConfig.accessToken,
          "Bearer " + originalMultiConfig.accessToken
        ])
        expect(
          originalReads.every((wire) => wire.signal?.aborted === false)
        ).toBe(true)
        expect(refresh.signal?.aborted).toBe(false)
        act(() => {
          firstController.abort()
        })
        if (transport === "extension")
          await waitFor(() =>
            expect(originalReads[0].signal?.aborted).toBe(true)
          )
        else expect(originalReads[0].signal?.aborted).toBe(true)
        expect(originalReads[1].signal?.aborted).toBe(false)
        expect(siblingController.signal.aborted).toBe(false)
        expect(refresh.signal?.aborted).toBe(false)
        response.resolve(originalRefreshResponse())
        let firstResult: unknown, siblingResult: unknown
        await act(async () => {
          ;[firstResult, siblingResult] = await Promise.all([first, sibling])
          await awaitPhysicalWorkerIdle()
        })
        expect(firstResult).toBeInstanceOf(Error)
        expect(siblingResult).toEqual(stateFor(originalMultiConfig.serverUrl))
        expect(refreshWires()).toHaveLength(1)
        const reads = stateReadsSince(startIndex)
        expect(reads).toHaveLength(3)
        expect(reads[2].url).toBe(
          originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state"
        )
        expect(reads[2].headers.get("Authorization")).toBe(
          "Bearer " + accessFor(42, 2)
        )
        expect(reads[2].signal?.aborted).toBe(false)
        await tldwClient.initialize()
        expect(await tldwClient.getConfig()).toMatchObject({
          ...originalMultiConfig,
          accessToken: accessFor(42, 2),
          refreshToken: "declared-original-rotated-refresh-token"
        })
        expect(mounted.result.current.state).toEqual(
          stateFor(originalMultiConfig.serverUrl)
        )
        expect(mounted.result.current.error).toBeNull()
        expect(providerWires()).toEqual([])
      } finally {
        response.resolve(originalRefreshResponse())
        await act(async () => {
          await Promise.all([first, sibling])
          await awaitPhysicalWorkerIdle()
        })
      }
    })

    describe.each(["tldwConfig", "tldwRefreshRotation"] as const)(
      "final rotation storage read %s",
      (key) => {
        it.each(["replacement", "ABA"] as const)(
          "does not write or retry a retired setup rotation after %s during the actual commit read",
          async (change) => {
            const mounted = await mountOriginal(transport, originalMultiConfig)
            const startIndex = wires.length
            const entered = deferred<void>(),
              held = deferred<void>()
            let heldStack = "",
              capturedValue: unknown
            physical.control.beforeRead = async (read) => {
              if (
                read.area !== "local" ||
                read.key !== key ||
                !read.stack.includes("storeRefreshRotationIfCurrent")
              )
                return read.value
              physical.control.beforeRead = undefined
              heldStack = read.stack
              capturedValue = read.value
              entered.resolve(undefined)
              await held.promise
              // A real completed storage read can return the original captured
              // value after its await. Keep that physical result so equality with
              // A cannot manufacture a fresh account lifetime after replacement.
              return read.value
            }
            rejectOriginalBearerState = true
            let pending: Promise<unknown> | undefined
            try {
              pending = mounted.result.current.refresh().catch((error) => error)
              await Promise.race([
                entered.promise,
                pending.then(() => {
                  throw new Error(
                    "Setup refresh settled before its actual rotation commit storage gate"
                  )
                })
              ])
              expect(heldStack).toContain("storeRefreshRotationIfCurrent")
              if (key === "tldwConfig")
                expect(capturedValue).toMatchObject(originalMultiConfig)
              else expect(capturedValue).toBeUndefined()
              expect(refreshWires()).toHaveLength(1)
              expect(refreshWires()[0]).toMatchObject({
                url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
                body: { refresh_token: originalMultiConfig.refreshToken }
              })
              const originalReads = stateReadsSince(startIndex)
              expect(originalReads).toHaveLength(1)
              expect(originalReads[0].headers.get("Authorization")).toBe(
                "Bearer " + originalMultiConfig.accessToken
              )
              expect(originalReads[0].signal?.aborted).toBe(false)
              expect(rotationWrites()).toEqual([])
              act(() => {
                replaceConfig(laterMultiConfig)
                if (change === "ABA") replaceConfig(originalMultiConfig)
              })
              if (transport === "extension")
                await waitFor(() =>
                  expect(originalReads[0].signal?.aborted).toBe(true)
                )
              else expect(originalReads[0].signal?.aborted).toBe(true)
              const current =
                change === "ABA" ? originalMultiConfig : laterMultiConfig
              await waitFor(() =>
                expect(mounted.result.current.metadata).toEqual(
                  metadataFor(current.serverUrl, "multi_user")
                )
              )
              held.resolve(undefined)
              let result: unknown
              await act(async () => {
                result = await pending
                await awaitPhysicalWorkerIdle()
              })
              expect(result).toBeNull()
              expect(rotationWrites()).toEqual([])
              expect(
                physical.areaFor("local").get("tldwRefreshRotation")
              ).toBeUndefined()
              expect(refreshWires()).toHaveLength(1)
              expect(
                stateReadsSince(startIndex).filter(
                  (wire) =>
                    wire.headers.get("Authorization") ===
                    "Bearer " + accessFor(42, 2)
                )
              ).toEqual([])
              const { tldwClient } = await import(
                "@/services/tldw/TldwApiClient"
              )
              await tldwClient.initialize()
              expect(await tldwClient.getConfig()).toMatchObject(current)
              expect(mounted.result.current.state).toEqual(
                stateFor(current.serverUrl)
              )
              expect(mounted.result.current.error).toBeNull()
              expect(providerWires()).toEqual([])
            } finally {
              held.resolve(undefined)
              if (pending)
                await act(async () => {
                  await pending
                  await awaitPhysicalWorkerIdle()
                })
            }
          }
        )
      }
    )

    it("finishes the actual same-account rotation after its sole setup caller stops without retrying that caller", async () => {
      const mounted = await mountOriginal(transport, originalMultiConfig)
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      const configSnapshot = await tldwClient.getConfig()
      expect(configSnapshot).toMatchObject(originalMultiConfig)
      const callerController = new AbortController()
      const response = deferred<Response>()
      heldAuthResponse = response
      rejectOriginalBearerState = true
      const startIndex = wires.length
      let pending: Promise<unknown> | undefined
      try {
        pending = tldwClient
          .getFirstRunState({
            configSnapshot,
            abortSignal: callerController.signal
          })
          .catch((error) => error)
        await waitFor(() => expect(refreshWires()).toHaveLength(1))
        const refresh = refreshWires()[0]
        expect(refresh).toMatchObject({
          url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
          body: { refresh_token: originalMultiConfig.refreshToken }
        })
        const originalReads = stateReadsSince(startIndex)
        expect(originalReads).toHaveLength(1)
        expect(originalReads[0].headers.get("Authorization")).toBe(
          "Bearer " + originalMultiConfig.accessToken
        )
        expect(originalReads[0].signal?.aborted).toBe(false)
        expect(refresh.signal?.aborted).toBe(false)
        act(() => {
          callerController.abort()
        })
        if (transport === "extension")
          await waitFor(() =>
            expect(originalReads[0].signal?.aborted).toBe(true)
          )
        else expect(originalReads[0].signal?.aborted).toBe(true)
        expect(refresh.signal?.aborted).toBe(false)
        response.resolve(originalRefreshResponse())
        let result: unknown
        await act(async () => {
          result = await pending
          await awaitPhysicalWorkerIdle()
        })
        expect(result).toBeInstanceOf(Error)
        await waitFor(() => expect(rotationWrites()).toHaveLength(1))
        expect(rotationWrites()[0].value).toMatchObject({
          sourceAccessToken: originalMultiConfig.accessToken,
          sourceRefreshToken: originalMultiConfig.refreshToken,
          accessToken: accessFor(42, 2),
          refreshToken: "declared-original-rotated-refresh-token"
        })
        expect(stateReadsSince(startIndex)).toHaveLength(1)
        expect(refreshWires()).toHaveLength(1)
        await tldwClient.initialize()
        expect(await tldwClient.getConfig()).toMatchObject({
          ...originalMultiConfig,
          accessToken: accessFor(42, 2),
          refreshToken: "declared-original-rotated-refresh-token"
        })
        expect(mounted.result.current.state).toEqual(
          stateFor(originalMultiConfig.serverUrl)
        )
        expect(mounted.result.current.error).toBeNull()
        expect(providerWires()).toEqual([])
      } finally {
        response.resolve(originalRefreshResponse())
        if (pending)
          await act(async () => {
            await pending
            await awaitPhysicalWorkerIdle()
          })
      }
    })

    it("does not publish a retired setup refresh's terminal invalidation after ABA during its actual storage read", async () => {
      const mounted = await mountOriginal(transport, originalMultiConfig)
      const { refreshSessionInvalidationKey, REFRESH_ROTATION_KEY } =
        await import("@/services/tldw/single-user-credential")
      const markerKey = refreshSessionInvalidationKey(originalMultiConfig)
      expect(markerKey).toEqual(expect.any(String))
      const startIndex = wires.length
      const entered = deferred<void>(),
        held = deferred<void>()
      let heldStack = "",
        capturedValue: unknown
      physical.control.beforeRead = async (read) => {
        if (
          read.area !== "local" ||
          read.key !== REFRESH_ROTATION_KEY ||
          !read.stack.includes("invalidateRefreshSessionIfCurrent")
        )
          return read.value
        physical.control.beforeRead = undefined
        heldStack = read.stack
        capturedValue = read.value
        entered.resolve(undefined)
        await held.promise
        return read.value
      }
      rejectOriginalBearerState = true
      rejectOriginalAuthRefresh = true
      let pending: Promise<unknown> | undefined
      try {
        pending = mounted.result.current.refresh().catch((error) => error)
        await Promise.race([
          entered.promise,
          pending.then(() => {
            throw new Error(
              "Setup refresh settled before its actual terminal invalidation storage gate"
            )
          })
        ])
        expect(heldStack).toContain("invalidateRefreshSessionIfCurrent")
        expect(capturedValue).toBeUndefined()
        expect(rejectOriginalAuthRefresh).toBe(false)
        expect(refreshWires()).toHaveLength(1)
        expect(refreshWires()[0]).toMatchObject({
          url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
          body: { refresh_token: originalMultiConfig.refreshToken }
        })
        const originalReads = stateReadsSince(startIndex)
        expect(originalReads).toHaveLength(1)
        expect(originalReads[0].headers.get("Authorization")).toBe(
          "Bearer " + originalMultiConfig.accessToken
        )
        expect(originalReads[0].signal?.aborted).toBe(false)
        expect(
          physical.writes.filter((write) => write.key === markerKey)
        ).toEqual([])
        act(() => {
          replaceConfig(laterMultiConfig)
          replaceConfig(originalMultiConfig)
        })
        if (transport === "extension")
          await waitFor(() =>
            expect(originalReads[0].signal?.aborted).toBe(true)
          )
        else expect(originalReads[0].signal?.aborted).toBe(true)
        held.resolve(undefined)
        let result: unknown
        await act(async () => {
          result = await pending
          await awaitPhysicalWorkerIdle()
        })
        expect(result).toBeNull()
        expect(
          physical.writes.filter((write) => write.key === markerKey)
        ).toEqual([])
        expect(physical.areaFor("local").get(markerKey!)).toBeUndefined()
        expect(rotationWrites()).toEqual([])
        expect(refreshWires()).toHaveLength(1)
        const { tldwClient } = await import("@/services/tldw/TldwApiClient")
        await tldwClient.initialize()
        expect(await tldwClient.getConfig()).toMatchObject(originalMultiConfig)
        await waitFor(() =>
          expect(mounted.result.current.metadata).toEqual(
            metadataFor(originalMultiConfig.serverUrl, "multi_user")
          )
        )
        expect(mounted.result.current.state).toEqual(
          stateFor(originalMultiConfig.serverUrl)
        )
        expect(mounted.result.current.error).toBeNull()
        expect(providerWires()).toEqual([])
      } finally {
        held.resolve(undefined)
        if (pending)
          await act(async () => {
            await pending
            await awaitPhysicalWorkerIdle()
          })
      }
    })

    it("publishes the current account's real terminal refresh marker and resumes public setup through a fresh actual hook owner", async () => {
      const mounted = await mountOriginal(transport, originalMultiConfig)
      const { refreshSessionInvalidationKey } = await import(
        "@/services/tldw/single-user-credential"
      )
      const markerKey = refreshSessionInvalidationKey(originalMultiConfig)
      expect(markerKey).toEqual(expect.any(String))
      const startIndex = wires.length
      rejectOriginalBearerState = true
      rejectOriginalAuthRefresh = true
      await act(async () => {
        // Terminal credential invalidation can retire this authenticated setup
        // owner. Do not adapt its old snapshot into a public owner or demand an
        // old fenced noAuth fallback. Resume through actual resolveConnection.
        await mounted.result.current.refresh().catch((error) => error)
        await awaitPhysicalWorkerIdle()
      })
      expect(rejectOriginalAuthRefresh).toBe(false)
      expect(refreshWires()).toHaveLength(1)
      expect(refreshWires()[0]).toMatchObject({
        url: originalMultiConfig.serverUrl + "/api/v1/auth/refresh",
        body: { refresh_token: originalMultiConfig.refreshToken }
      })
      const initialReads = stateReadsSince(startIndex)
      expect(initialReads[0].url).toBe(
        originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state"
      )
      expect(initialReads[0].headers.get("Authorization")).toBe(
        "Bearer " + originalMultiConfig.accessToken
      )
      expect(
        physical.writes.filter((write) => write.key === markerKey)
      ).toHaveLength(1)
      expect(physical.areaFor("local").get(markerKey!)).toBe(true)
      expect(rotationWrites()).toEqual([])
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      await tldwClient.initialize()
      const current = await tldwClient.getConfig()
      expect(current).toMatchObject({
        serverUrl: originalMultiConfig.serverUrl,
        authMode: "multi-user",
        authSource: "manual"
      })
      expect(current).not.toHaveProperty("accessToken")
      expect(current).not.toHaveProperty("refreshToken")
      const publicStart = wires.length
      let result: FirstRunState | null | undefined
      await act(async () => {
        result = await mounted.result.current.refresh()
        await awaitPhysicalWorkerIdle()
      })
      expect(result).toEqual(stateFor(originalMultiConfig.serverUrl))
      const publicReads = stateReadsSince(publicStart)
      expect(publicReads).toHaveLength(1)
      expect(publicReads[0].url).toBe(
        originalMultiConfig.serverUrl + "/api/v1/setup/first-run/state"
      )
      expect(publicReads[0].headers.has("Authorization")).toBe(false)
      expect(publicReads[0].headers.has("X-API-KEY")).toBe(false)
      expect(refreshWires()).toHaveLength(1)
      expect(mounted.result.current.connectionConfigured).toBe(true)
      expect(mounted.result.current.state).toEqual(
        stateFor(originalMultiConfig.serverUrl)
      )
      expect(mounted.result.current.metadata).toEqual(
        metadataFor(originalMultiConfig.serverUrl, "multi_user")
      )
      expect(mounted.result.current.error).toBeNull()
      expect(providerWires()).toEqual([])
    })
  }
)
