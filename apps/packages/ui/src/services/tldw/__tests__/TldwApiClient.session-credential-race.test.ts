import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { MANUAL_SESSION_KEY } from "@/services/tldw/single-user-credential"

const storages = vi.hoisted(() => {
  class ControlledStorage {
    values = new Map<string, Record<string, unknown>>()
    pauseFirstSessionRead = false
    firstSessionReadStarted: (() => void) | null = null
    resumeFirstSessionRead: (() => void) | null = null
    pauseFirstSessionRemove = false
    firstSessionRemoveStarted: (() => void) | null = null
    resumeFirstSessionRemove: (() => void) | null = null
    saverProgress: ((progress: "lock" | "remove") => void) | null = null
    private firstSessionRead = true
    private firstSessionRemove = true
    private sessionRemoveCount = 0

    reset(): void {
      this.values.clear()
      this.pauseFirstSessionRead = false
      this.firstSessionReadStarted = null
      this.resumeFirstSessionRead = null
      this.pauseFirstSessionRemove = false
      this.firstSessionRemoveStarted = null
      this.resumeFirstSessionRemove = null
      this.saverProgress = null
      this.firstSessionRead = true
      this.firstSessionRemove = true
      this.sessionRemoveCount = 0
    }

    async get<T>(key: string, area: "local" | "session"): Promise<T | undefined> {
      const value = this.values.get(key) as T | undefined
      if (area === "session" && key === "tldwManualSessionApiKey" && this.pauseFirstSessionRead && this.firstSessionRead) {
        this.firstSessionRead = false
        this.firstSessionReadStarted?.()
        await new Promise<void>((resolve) => {
          this.resumeFirstSessionRead = resolve
        })
      }
      return value
    }

    async set<T>(key: string, value: T): Promise<void> {
      this.values.set(key, value as Record<string, unknown>)
    }

    async remove(key: string): Promise<void> {
      if (key === MANUAL_SESSION_KEY) {
        this.sessionRemoveCount += 1
        if (this.sessionRemoveCount === 2) {
          this.saverProgress?.("remove")
        }
        if (this.pauseFirstSessionRemove && this.firstSessionRemove) {
          this.firstSessionRemove = false
          this.firstSessionRemoveStarted?.()
          await new Promise<void>((resolve) => {
            this.resumeFirstSessionRemove = resolve
          })
        }
      }
      this.values.delete(key)
    }
  }

  return {
    local: new ControlledStorage(),
    session: new ControlledStorage()
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: ({ area }: { area?: "local" | "session" } = {}) => {
    const storage = area === "session" ? storages.session : storages.local
    return {
      get: <T>(key: string) => storage.get<T>(key, area === "session" ? "session" : "local"),
      set: <T>(key: string, value: T) => storage.set(key, value),
      remove: (key: string) => storage.remove(key)
    }
  },
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: JSON.parse
  }
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false,
  invalidateCookieSessionConfig: vi.fn()
}))

import { TldwApiClientBase } from "@/services/tldw/TldwApiClient"

const sessionConfig = {
  authMode: "single-user" as const,
  authSource: "manual" as const,
  credentialSource: "manual" as const,
  apiKeyPersistence: "session" as const,
  apiKeyServerOrigin: "https://api.example.test",
  serverUrl: "https://api.example.test"
}

describe("TldwApiClient session credential cleanup", () => {
  beforeEach(() => {
    storages.local.reset()
    storages.session.reset()
    let tail = Promise.resolve()
    let lockRequestCount = 0
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: {
          value: {
            request: (_name: string, work: () => unknown) => {
              lockRequestCount += 1
              if (lockRequestCount === 2) {
                storages.session.saverProgress?.("lock")
              }
              const next = tail.then(work)
              tail = next.then(
                () => undefined,
                () => undefined
              )
              return next
            }
          }
        }
      })
    )
  })

  afterEach(() => vi.unstubAllGlobals())

  it("does not remove a valid session credential written after an initializer observed its absence", async () => {
    await storages.local.set("tldwConfig", sessionConfig)
    storages.session.pauseFirstSessionRead = true
    const firstReadStarted = new Promise<void>((resolve) => {
      storages.session.firstSessionReadStarted = resolve
    })
    const cleanupClient = new TldwApiClientBase()
    const initialize = cleanupClient.initialize()
    await firstReadStarted

    const writer = new TldwApiClientBase()
    await expect(
      writer.saveManualSingleUserCredential({
        serverUrl: sessionConfig.serverUrl,
        apiKey: "session-secret",
        persistence: "session"
      })
    ).resolves.toBe("session")

    storages.session.resumeFirstSessionRead?.()
    await initialize

    await expect(storages.session.get(MANUAL_SESSION_KEY, "session")).resolves.toMatchObject({
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: sessionConfig.apiKeyServerOrigin,
      apiKey: "session-secret"
    })
  })

  it("still removes a session credential whose origin does not match its metadata", async () => {
    await storages.local.set("tldwConfig", sessionConfig)
    await storages.session.set(MANUAL_SESSION_KEY, {
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://other.example.test",
      apiKey: "wrong-origin-secret"
    })

    await new TldwApiClientBase().initialize()

    await expect(storages.session.get(MANUAL_SESSION_KEY, "session")).resolves.toBeUndefined()
  })

  it("defers destructive cleanup when the browser does not support credential locks", async () => {
    await storages.local.set("tldwConfig", sessionConfig)
    const invalidRecord = {
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://other.example.test",
      apiKey: "wrong-origin-secret"
    }
    await storages.session.set(MANUAL_SESSION_KEY, invalidRecord)
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: { value: undefined }
      })
    )

    const client = new TldwApiClientBase()
    await client.initialize()

    await expect(storages.session.get(MANUAL_SESSION_KEY, "session")).resolves.toEqual(
      invalidRecord
    )
    expect((await client.getConfig())?.apiKey).toBeUndefined()
  })

  it("queues a second saver before an in-flight cleanup can remove credentials", async () => {
    await storages.local.set("tldwConfig", sessionConfig)
    await storages.session.set(MANUAL_SESSION_KEY, {
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: sessionConfig.apiKeyServerOrigin,
      apiKey: "existing-session-secret"
    })
    const writer = new TldwApiClientBase()
    await writer.initialize()
    await storages.session.set(MANUAL_SESSION_KEY, {
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://other.example.test",
      apiKey: "stale-session-secret"
    })
    storages.session.pauseFirstSessionRemove = true
    const firstRemoveStarted = new Promise<void>((resolve) => {
      storages.session.firstSessionRemoveStarted = resolve
    })
    const saverProgress = new Promise<"lock" | "remove">((resolve) => {
      storages.session.saverProgress = resolve
    })
    const cleanup = new TldwApiClientBase().initialize()
    await firstRemoveStarted

    const save = writer.saveManualSingleUserCredential({
      serverUrl: sessionConfig.serverUrl,
      apiKey: "replacement-session-secret",
      persistence: "session"
    })
    await expect(saverProgress).resolves.toBe("lock")
    storages.session.resumeFirstSessionRemove?.()
    await Promise.all([cleanup, save])

    await expect(storages.session.get(MANUAL_SESSION_KEY, "session")).resolves.toMatchObject({
      apiKey: "replacement-session-secret",
      apiKeyServerOrigin: sessionConfig.apiKeyServerOrigin
    })
  })
})
