import { waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "../tldw/TldwApiClient"

type StorageListener = (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void
const boundary = vi.hoisted(() => ({
  request: vi.fn(), ensureConfig: vi.fn(), user: vi.fn(),
  config: null as TldwConfig | null,
  storageListeners: new Set<StorageListener>()
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => {}, ensureConfigForRequest: boundary.ensureConfig
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: boundary.user } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw-server", () => ({
  LEGACY_SERVICE_PROMPT_DEFAULTS: {}, promptForRag: vi.fn(), getWebSearchPrompt: vi.fn()
}))
vi.mock("wxt/browser", () => ({ browser: { storage: { onChanged: {
  addListener: (listener: StorageListener) => boundary.storageListeners.add(listener),
  removeListener: (listener: StorageListener) => boundary.storageListeners.delete(listener)
} } } }))
vi.mock("@/utils/safe-storage", async original => ({
  ...await original<typeof import("@/utils/safe-storage")>(),
  createSafeStorage: () => ({
    get: async () => null, set: async () => {}, remove: async () => {}, watch: () => {}, unwatch: () => {}
  })
}))

const config = (user = "alice", revision = "A"): TldwConfig => ({
  serverUrl: "https://keywords.test", authMode: "multi-user", authSource: "manual",
  accessToken: `test.${btoa(JSON.stringify({ sub: user }))}.${revision}`,
  refreshToken: `refresh-${user}-${revision}`
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(yes => { resolve = yes })
  return { promise, resolve }
}
const notifications = ["config-updated", "browser storage", "extension storage"] as const
type Notification = typeof notifications[number]
const notify = (notification: Notification, previous: TldwConfig, current: TldwConfig, authorityChanged = false) => {
  boundary.config = current
  if (notification === "config-updated") {
    // This is the complete payload published by TldwApiClient.updateConfig.
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged } }))
  } else if (notification === "browser storage") {
    window.dispatchEvent(new StorageEvent("storage", {
      key: "tldwConfig", oldValue: JSON.stringify(previous), newValue: JSON.stringify(current)
    }))
  } else {
    for (const listener of [...boundary.storageListeners]) {
      listener({ tldwConfig: { oldValue: previous, newValue: current } }, "local")
    }
  }
}
const credentialABA = (notification: Notification) => {
  const first = config()
  const rotated = config("alice", "B")
  // No await: a later effective-config reread sees A, never the intermediate B.
  notify(notification, first, rotated)
  notify(notification, rotated, first)
}
let service: typeof import("../note-keywords")
let watch: typeof import("../chat-account-boundary").watchChatAccountChanges
const operations = [
  { name: "list", run: () => service.getNoteKeywords(200), result: ["current-alice"] },
  { name: "all keywords", run: () => service.getAllNoteKeywords(2), result: ["current-alice"] },
  { name: "all stats", run: () => service.getAllNoteKeywordStats(2), result: [{ keyword: "current-alice", noteCount: 4 }] },
  { name: "search", run: () => service.searchNoteKeywords(" private ", 10), result: ["current-alice"] }
]

describe("keyword reads retire credential ABA without changing shared account refresh semantics", () => {
  beforeEach(async () => {
    expect(boundary.storageListeners.size).toBe(0)
    vi.resetModules()
    vi.clearAllMocks()
    boundary.config = config()
    boundary.ensureConfig.mockImplementation(async () => boundary.config ? { ...boundary.config } : null)
    boundary.user.mockImplementation(async () => ({ id: JSON.parse(atob(boundary.config!.accessToken!.split(".")[1])).sub }))
    boundary.request.mockResolvedValue([])
    vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Unexpected network operation") }))
    service = await import("../note-keywords")
    watch = (await import("../chat-account-boundary")).watchChatAccountChanges
  })
  afterEach(() => {
    expect(boundary.storageListeners.size).toBe(0)
    expect(fetch).not.toHaveBeenCalled()
    vi.unstubAllGlobals()
  })

  for (const operation of operations) {
    it.each(notifications)(`${operation.name} rejects a late old reply after non-invalidating %s credential ABA`, async notification => {
      const gate = deferred<unknown>()
      boundary.request.mockImplementationOnce(() => gate.promise)
      const observed = vi.fn()
      const stop = watch(observed)
      const old = operation.run().then(value => ({ value }), error => ({ error }))
      try {
        await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
        credentialABA(notification)
        expect(observed.mock.calls.map(([invalidated]) => invalidated)).toEqual([false, false])
        expect(boundary.config).toEqual(config())
        gate.resolve([{ keyword: "retired-alice", note_count: 4 }])
        expect(await old).toMatchObject({ error: { status: 412 } })
        expect(boundary.request.mock.calls[0][0].abortSignal.aborted).toBe(true)
        boundary.request.mockResolvedValue([{ keyword: "current-alice", note_count: 4 }])
        expect(await operation.run()).toEqual(operation.result)
        expect(boundary.request).toHaveBeenCalledTimes(2)
        const fresh = boundary.request.mock.calls[1][0]
        expect(fresh.configSnapshot).toEqual(config())
        expect(fresh.headers).toEqual({ "X-TLDW-Expected-User-ID": "alice" })
        expect(fresh.abortSignal.aborted).toBe(false)
      } finally {
        gate.resolve([])
        await old
        stop()
      }
    })
  }

  for (const operation of operations.slice(1, 3)) {
    it.each(notifications)(`${operation.name} stops a full old page after non-invalidating %s credential ABA`, async notification => {
      const gate = deferred<unknown>()
      boundary.request.mockImplementationOnce(() => gate.promise)
      const old = operation.run().then(value => ({ value }), error => ({ error }))
      try {
        await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
        credentialABA(notification)
        gate.resolve([{ keyword: "old-one", note_count: 1 }, { keyword: "old-two", note_count: 2 }])
        expect(await old).toMatchObject({ error: { status: 412 } })
        expect(boundary.request).toHaveBeenCalledTimes(1)
        expect(boundary.request.mock.calls[0][0].path).toBe("/api/v1/notes/keywords/?limit=2&offset=0&include_note_counts=true")
      } finally {
        gate.resolve([])
        await old
      }
    })
  }

  it.each(notifications)("retires unresolved owner capture across non-invalidating %s credential ABA before any GET", async notification => {
    const principal = deferred<{ id: string }>()
    boundary.user.mockImplementationOnce(() => principal.promise)
    const observed = vi.fn()
    const stop = watch(observed)
    const old = service.getNoteKeywords().then(value => ({ value }), error => ({ error }))
    try {
      await waitFor(() => expect(boundary.user).toHaveBeenCalledTimes(1))
      expect(boundary.request).not.toHaveBeenCalled()
      credentialABA(notification)
      expect(observed.mock.calls.map(([invalidated]) => invalidated)).toEqual([false, false])
      principal.resolve({ id: "alice" })
      expect(await old).toMatchObject({ error: { name: "AbortError", message: "Service Prompt request was aborted." } })
      expect(boundary.request).not.toHaveBeenCalled()
    } finally {
      principal.resolve({ id: "alice" })
      await old
      stop()
    }
  })

  it.each(["browser storage", "extension storage"] as const)("accepts a current-owner reply after same-credential %s notification", async notification => {
    const gate = deferred<unknown>()
    boundary.request.mockImplementationOnce(() => gate.promise)
    const observed = vi.fn()
    const stop = watch(observed)
    const current = service.getNoteKeywords()
    try {
      await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
      notify(notification, config(), { ...config(), apiKeyPersistence: "device" })
      expect(observed.mock.calls.map(([invalidated]) => invalidated)).toEqual([false])
      gate.resolve(["current-alice"])
      expect(await current).toEqual(["current-alice"])
      expect(boundary.request.mock.calls[0][0].abortSignal.aborted).toBe(false)
      expect(boundary.request).toHaveBeenCalledTimes(1)
    } finally {
      gate.resolve([])
      await current.catch(() => {})
      stop()
    }
  })

  it.each(notifications)("keeps account-change retirement through %s and permits only a fresh Bob read", async notification => {
    const gate = deferred<unknown>()
    boundary.request.mockImplementationOnce(() => gate.promise)
    const observed = vi.fn()
    const stop = watch(observed)
    const old = service.getNoteKeywords().then(value => ({ value }), error => ({ error }))
    try {
      await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
      notify(notification, config(), config("bob"), true)
      expect(observed.mock.calls.map(([invalidated]) => invalidated)).toEqual([true])
      gate.resolve(["private-alice"])
      expect(await old).toMatchObject({ error: { status: 412 } })
      boundary.request.mockResolvedValue(["private-bob"])
      expect(await service.getNoteKeywords()).toEqual(["private-bob"])
      expect(boundary.request.mock.calls[1][0].headers).toEqual({ "X-TLDW-Expected-User-ID": "bob" })
      expect(boundary.request).toHaveBeenCalledTimes(2)
    } finally {
      gate.resolve([])
      await old
      stop()
    }
  })
})
