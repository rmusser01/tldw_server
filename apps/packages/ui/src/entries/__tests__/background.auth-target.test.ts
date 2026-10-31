import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const connectListeners = new Set<(port: unknown) => void>()
const runtimeMessageListeners = new Set<
  (
    message: unknown,
    sender: unknown,
    sendResponse: (response: unknown) => void,
  ) => unknown
>()
const storageState = vi.hoisted(() => ({
  persistent: new Map<string, unknown>(),
  session: new Map<string, unknown>(),
  runtimeId: "extension-id" as string | null,
  set: vi.fn(async (_key: string, _value: unknown) => undefined),
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options,
  })
  return {}
})

vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: (value: unknown) => value,
    deserializer: (value: unknown) => value,
  },
  createSafeStorage: (options?: { area?: string }) => {
    const values =
      options?.area === "session"
        ? storageState.session
        : storageState.persistent
    return {
      get: async (key: string) => values.get(key),
      set: async (key: string, value: unknown) => {
        await storageState.set(key, value)
        values.set(key, value)
      },
      remove: vi.fn(async (key: string) => values.delete(key)),
    }
  },
}))

vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "model-warm",
  initBackground: vi.fn(async () => {}),
}))

vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: vi.fn(async () => {}),
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return storageState.runtimeId
      },
      getURL: (path: string) => `chrome-extension://extension-id${path}`,
      sendMessage: vi.fn(async () => ({ handled: true })),
      onConnect: {
        addListener: (listener: (port: unknown) => void) =>
          connectListeners.add(listener),
      },
      onMessage: {
        addListener: (
          listener: (
            message: unknown,
            sender: unknown,
            sendResponse: (response: unknown) => void,
          ) => unknown,
        ) => runtimeMessageListeners.add(listener),
      },
      onStartup: { addListener: vi.fn() },
    },
    storage: {
      local: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {}),
      },
      session: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {}),
      },
      onChanged: { addListener: vi.fn(), removeListener: vi.fn() },
    },
    alarms: {
      clear: vi.fn(async () => true),
      create: vi.fn(async () => {}),
      onAlarm: { addListener: vi.fn() },
    },
    tabs: {
      create: vi.fn(),
      query: vi.fn(async () => []),
      sendMessage: vi.fn(async () => undefined),
    },
    action: { onClicked: { addListener: vi.fn() } },
    contextMenus: {
      create: vi.fn(),
      removeAll: vi.fn(),
      onClicked: { addListener: vi.fn() },
    },
    i18n: { getMessage: (key: string) => key },
  },
}))

import background from "@/entries/background"
import { browser } from "wxt/browser"
import { TldwAuthService } from "@/services/tldw/TldwAuth"
import { tldwClient } from "@/services/tldw/TldwApiClient"
const target = {
  serverUrl: "https://visible.example.test/base",
  authMode: "multi-user" as const,
}
const other = { ...target, serverUrl: "https://visible.example.test/other" }
const tokens = { access_token: "synthetic-access", token_type: "bearer" }
const methods = [
  {
    name: "password",
    path: "/api/v1/auth/login",
    invoke: (auth: TldwAuthService) =>
      auth.login(
        { username: "alice", password: "synthetic-password" },
        { target },
      ),
  },
  {
    name: "request",
    path: "/api/v1/auth/magic-link/request",
    invoke: (auth: TldwAuthService) =>
      auth.requestMagicLink("alice@example.test", { target }),
  },
  {
    name: "verify",
    path: "/api/v1/auth/magic-link/verify",
    invoke: (auth: TldwAuthService) =>
      auth.verifyMagicLink("synthetic-magic", { target }),
  },
]
beforeEach(() => {
  connectListeners.clear()
  runtimeMessageListeners.clear()
  storageState.persistent.clear()
  storageState.session.clear()
  storageState.set.mockClear()
  storageState.runtimeId = "extension-id"
  storageState.persistent.set("tldwConfig", target)
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "window")!
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: undefined,
  })
  background.main()
  Object.defineProperty(globalThis, "window", descriptor)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})
describe.each(["direct", "extension"])(
  "actual %s auth transport",
  (transport) => {
    it.each(methods)(
      "keeps $name credentials on the visible base after transport re-resolution",
      async ({ invoke, path }) => {
        storageState.runtimeId = transport === "direct" ? null : "extension-id"
        // The service read authorizes A; the later transport sees committed B.
        vi.spyOn(tldwClient, "getConfig").mockImplementation(async () => {
          storageState.persistent.set("tldwConfig", other)
          return target
        })
        vi.spyOn(tldwClient, "updateConfig").mockImplementation(
          async (_update, assertCurrent) => {
            assertCurrent?.(other)
          },
        )
        vi.mocked(browser.runtime.sendMessage).mockImplementation(
          async (message) =>
            new Promise((resolve) => {
              const listener = [...runtimeMessageListeners][0]
              if (!listener) throw new Error("Actual worker listener missing")
              listener(message, { id: "extension-id" }, resolve)
            }),
        )
        const fetchMock = vi.fn<typeof fetch>(
          async () =>
            new Response(JSON.stringify(tokens), {
              status: 200,
              headers: { "content-type": "application/json" },
            }),
        )
        vi.stubGlobal("fetch", fetchMock)
        await invoke(new TldwAuthService()).catch(() => undefined)
        expect(fetchMock).toHaveBeenCalledOnce()
        expect(String(fetchMock.mock.calls[0][0])).toBe(target.serverUrl + path)
        expect(
          new Headers(fetchMock.mock.calls[0][1]?.headers).get("Authorization"),
        ).toBeNull()
        if (transport === "extension")
          expect(browser.runtime.sendMessage).toHaveBeenCalledWith(
            expect.objectContaining({
              type: "tldw:request",
              payload: expect.objectContaining({
                path: target.serverUrl + path,
                noAuth: true,
              }),
            }),
          )
        expect(storageState.set).not.toHaveBeenCalled()
      },
    )
  },
)

it.each(["direct", "extension"])(
  "fails closed when the %s transport sees a different saved origin",
  async (transport) => {
    storageState.runtimeId = transport === "direct" ? null : "extension-id"
    vi.spyOn(tldwClient, "getConfig").mockImplementation(async () => {
      storageState.persistent.set("tldwConfig", {
        ...other,
        serverUrl: "https://foreign.example.test",
      })
      return target
    })
    const publish = vi.spyOn(tldwClient, "updateConfig")
    vi.mocked(browser.runtime.sendMessage).mockImplementation(
      async (message) =>
        new Promise((resolve) => {
          ;[...runtimeMessageListeners][0](
            message,
            { id: "extension-id" },
            resolve,
          )
        }),
    )
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    await expect(
      new TldwAuthService().login(
        { username: "alice", password: "synthetic" },
        { target },
      ),
    ).rejects.toBeDefined()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(publish).not.toHaveBeenCalled()
  },
)

it.each(["direct", "extension"])(
  "preserves normal login and credential publication using actual %s transport",
  async (transport) => {
    storageState.runtimeId = transport === "direct" ? null : "extension-id"
    vi.spyOn(tldwClient, "getConfig").mockResolvedValue(target)
    // Use the real guarded updateConfig, storage adapter, and request receiver.
    vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
    vi.mocked(browser.runtime.sendMessage).mockImplementation(
      async (message) =>
        new Promise((resolve) => {
          ;[...runtimeMessageListeners][0](
            message,
            { id: "extension-id" },
            resolve,
          )
        }),
    )
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify(tokens), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    )
    vi.stubGlobal("fetch", fetchMock)
    await expect(
      new TldwAuthService().login(
        { username: "alice", password: "synthetic" },
        { target },
      ),
    ).resolves.toEqual(tokens)
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      target.serverUrl + "/api/v1/auth/login",
    )
    expect(storageState.persistent.get("tldwConfig")).toMatchObject({
      ...target,
      accessToken: tokens.access_token,
    })
  },
)
