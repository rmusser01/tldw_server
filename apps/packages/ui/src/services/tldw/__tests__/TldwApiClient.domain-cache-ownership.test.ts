import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const io = vi.hoisted(() => ({
  storage: new Map<string, unknown>(),
  get: vi.fn(),
  request: vi.fn(),
  owner: "Alice",
  extensionListeners: new Set<(changes: Record<string, unknown>, area: string) => void>()
}))

vi.mock("wxt/browser", () => ({ browser: {
  runtime: { id: null },
  storage: { onChanged: {
    addListener: (listener: (changes: Record<string, unknown>, area: string) => void) => io.extensionListeners.add(listener),
    removeListener: (listener: (changes: Record<string, unknown>, area: string) => void) => io.extensionListeners.delete(listener)
  } }
} }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (key: string) => io.get(key),
    set: async (key: string, value: unknown) => { io.storage.set(key, value) },
    remove: async (key: string) => { io.storage.delete(key) }
  }),
  safeStorageSerde: { serializer: (value: unknown) => value, deserializer: (value: unknown) => value }
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => io.request(...args),
  bgUpload: vi.fn(), bgStream: vi.fn()
}))

import { TldwApiClient, TldwApiClientBase, type TldwConfig } from "../TldwApiClient"
import { characterMethods } from "../domains/characters"
import { chatRagMethods } from "../domains/chat-rag"

const jwt = (sub: string, iat = 1) => `test.${btoa(JSON.stringify({ sub, iat }))}.signature`
const alice: TldwConfig = { serverUrl: "https://chat.test", authMode: "multi-user", accessToken: jwt("alice") }
const deviceKey: TldwConfig = {
  serverUrl: "https://chat.test", authMode: "single-user", apiKey: "synthetic-key-a",
  authSource: "manual", credentialSource: "manual", apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://chat.test"
}
const cookie: TldwConfig = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
const configure = (config: TldwConfig) => {
  io.storage.clear()
  io.storage.set(config.authSource === "cookie-session" ? "tldwCookieSessionConfig" : "tldwConfig", config)
}
const response = (path: string, owner: string) => path.includes("/messages")
  ? [{ id: "7", sender: "user", content: owner, created_at: "2026-10-04T00:00:00Z" }]
  : { id: 7, name: owner }
const deferred = () => {
  let resolve!: (value: unknown) => void
  const promise = new Promise(done => { resolve = done })
  return { promise, resolve }
}

const adapters = ["public", "base", "domain"] as const
const resources = ["character", "messages"] as const
describe.each(adapters)("%s domain-cache ownership", adapter => {
  let client: TldwApiClient
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "")
    io.owner = "Alice"
    configure(alice)
    io.get.mockReset().mockImplementation(async (key: string) => io.storage.get(key) ?? null)
    io.request.mockReset().mockImplementation(async ({ path }) => response(path, io.owner))
    client = new TldwApiClient()
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0] as `/${string}`)
  })
  afterEach(() => vi.unstubAllEnvs())

  const read = async (resource: typeof resources[number]): Promise<string> => {
    if (resource === "character") {
      const method = adapter === "base" ? TldwApiClientBase.prototype.getCharacter
        : adapter === "domain" ? characterMethods.getCharacter : client.getCharacter
      return (await method.call(client, 7)).name
    }
    const method = adapter === "base" ? TldwApiClientBase.prototype.listChatMessages
      : adapter === "domain" ? chatRagMethods.listChatMessages : client.listChatMessages
    return (await method.call(client, "7"))[0].content
  }

  it("does not overwrite shared config when an older direct initialize finishes last", async () => {
    configure(deviceKey)
    const storageRead = deferred()
    const started = deferred()
    io.get.mockImplementationOnce(() => { started.resolve(null); return storageRead.promise })
    const oldInitialize = client.initialize().catch(error => error)
    await started.promise
    const bob = { ...deviceKey, apiKey: "synthetic-key-b" }
    configure(bob)
    await client.initialize()
    storageRead.resolve(deviceKey)
    expect(await oldInitialize).toMatchObject({ status: 412 })
    expect(await client.getConfig()).toMatchObject(bob)
  })

  describe.each(resources)("%s", resource => {
    const joinPendingRead = async () => {
      const reached = deferred()
      const index = resource === "character" ? client.characterInFlight : client.chatMessagesInFlight
      const get = index.get.bind(index)
      vi.spyOn(index, "get").mockImplementationOnce(key => {
        reached.resolve(null)
        return get(key)
      })
      const joined = read(resource)
      await reached.promise
      return { joined }
    }
    it.each([
      ["JWT", alice, { ...alice, accessToken: jwt("bob") }],
      ["API key", deviceKey, { ...deviceKey, apiKey: "synthetic-key-b" }]
    ])("rejects an old %s storage read that completes after a newer owner is published", async (_kind, before, after) => {
      configure(before)
      const storageRead = deferred()
      const started = deferred()
      io.get.mockImplementationOnce(() => {
        started.resolve(null)
        return storageRead.promise
      })
      const oldRead = read(resource).catch(error => error)
      await started.promise
      configure(after)
      io.owner = "Bob"
      expect(await read(resource)).toBe("Bob")
      storageRead.resolve(before)
      expect(await oldRead).toMatchObject({ status: 412 })
      expect(await read(resource)).toBe("Bob")
      expect(io.request).toHaveBeenCalledTimes(1)
    })

    it.each([alice, deviceKey])("resolves storage once for a cache hit and at most twice for a fetch ($authMode)", async config => {
      configure(config)
      expect(await read(resource)).toBe("Alice")
      expect(io.get.mock.calls.filter(([key]) => key === "tldwConfig").length).toBeLessThanOrEqual(2)
      expect(io.get.mock.calls.length).toBeLessThanOrEqual(4)
      io.get.mockClear()
      expect(await read(resource)).toBe("Alice")
      expect(io.get.mock.calls.filter(([key]) => key === "tldwConfig").length).toBe(1)
      expect(io.get.mock.calls.length).toBeLessThanOrEqual(2)
    })
    it.each([
      ["JWT principal", alice, { ...alice, accessToken: jwt("bob") }],
      ["API key", deviceKey, { ...deviceKey, apiKey: "synthetic-key-b" }],
      ["server", alice, { ...alice, serverUrl: "https://other.test" }],
      ["organization", alice, { ...alice, orgId: 2 }],
      ["auth mode", alice, deviceKey],
      ["auth source", deviceKey, { ...deviceKey, authSource: "env" }],
      ["unknown JWT principal", { ...alice, accessToken: "opaque-a" }, { ...alice, accessToken: "opaque-b" }]
    ] as const)("does not reuse the same ID after a %s change without an event", async (_kind, before, after) => {
      configure(before as TldwConfig)
      expect(await read(resource)).toBe("Alice")
      configure(after as TldwConfig)
      io.owner = "Bob"
      expect(await read(resource)).toBe("Bob")
      expect(await read(resource)).toBe("Bob")
      expect(io.request).toHaveBeenCalledTimes(2)
    })

    it("keeps same-principal JWT refresh cached", async () => {
      expect(await read(resource)).toBe("Alice")
      configure({ ...alice, accessToken: jwt("alice", 2) })
      expect(await read(resource)).toBe("Alice")
      expect(io.request).toHaveBeenCalledTimes(1)
    })

    it("rejects a boundary between configuration verification and a cache hit", async () => {
      expect(await read(resource)).toBe("Alice")
      const capture = client.getDomainCacheRevision
      vi.spyOn(client, "getDomainCacheRevision").mockImplementationOnce(async () => {
        const revision = await capture.call(client)
        queueMicrotask(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
        return revision
      })
      await expect(read(resource)).rejects.toMatchObject({ status: 412 })
    })

    it("rejects a boundary after response verification before cache publication", async () => {
      const payload = response(resource === "messages" ? "/messages" : "/characters", "Alice")
      // Payload consumption is the publication milestone, independent of config helper calls.
      Object.defineProperty(payload, resource === "character" ? "then" : "map", {
        get: () => {
          window.dispatchEvent(new Event("tldw:auth-principal-changed"))
          return resource === "character" ? undefined : Array.prototype.map
        }
      })
      io.request.mockResolvedValueOnce(payload)
      await expect(read(resource)).rejects.toMatchObject({ status: 412 })
      io.owner = "Bob"
      expect(await read(resource)).toBe("Bob")
    })

    it("coalesces concurrent reads only within the current authority", async () => {
      const pending = deferred()
      io.request.mockImplementationOnce(() => pending.promise)
      const first = read(resource)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(1))
      const second = read(resource)
      pending.resolve(response(resource === "messages" ? "/messages" : "/characters", "Alice"))
      expect(await Promise.all([first, second])).toEqual(["Alice", "Alice"])
      expect(io.request).toHaveBeenCalledTimes(1)
    })

    it.each(["tldw:auth-principal-changed", "tldw:auth-credentials-changed"])("invalidates identical quickstart cookie config on %s", async event => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      configure(cookie)
      expect(await read(resource)).toBe("Alice")
      io.owner = "Bob"
      window.dispatchEvent(new Event(event))
      expect(await read(resource)).toBe("Bob")
    })

    it.each(["window", "extension"])("invalidates a logout/login roundtrip delivered through %s storage", async surface => {
      expect(await read(resource)).toBe("Alice")
      if (surface === "window") {
        window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(alice), newValue: null }))
      } else {
        for (const listener of io.extensionListeners) listener({ tldwConfig: { oldValue: alice, newValue: undefined } }, "local")
      }
      io.owner = "Bob"
      expect(await read(resource)).toBe("Bob")
    })

    it("rejects a late old-owner response without replacing or deleting the newer in-flight read", async () => {
      const oldResponse = deferred()
      const newResponse = deferred()
      io.request.mockImplementationOnce(() => oldResponse.promise).mockImplementationOnce(() => newResponse.promise)
      const oldRead = read(resource).catch(error => error)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(1))
      configure({ ...alice, accessToken: jwt("bob") })
      const newRead = read(resource)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(2))
      oldResponse.resolve(response(resource === "messages" ? "/messages" : "/characters", "Alice"))
      expect(await oldRead).toMatchObject({ status: 412 })
      const { joined } = await joinPendingRead()
      newResponse.resolve(response(resource === "messages" ? "/messages" : "/characters", "Bob"))
      expect(await Promise.all([newRead, joined])).toEqual(["Bob", "Bob"])
      expect(await read(resource)).toBe("Bob")
      expect(io.request).toHaveBeenCalledTimes(2)
    })

    it("rejects stale completion across an identical cookie-session logout/login epoch", async () => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      configure(cookie)
      const pending = deferred()
      io.request.mockImplementationOnce(() => pending.promise)
      const oldRead = read(resource).catch(error => error)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(1))
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      io.owner = "Bob"
      pending.resolve(response(resource === "messages" ? "/messages" : "/characters", "Alice"))
      expect(await oldRead).toMatchObject({ status: 412 })
      expect(await read(resource)).toBe("Bob")
    })

    it("does not let an older failed configuration check clear the newer owner's flight", async () => {
      const oldResponse = deferred()
      const newResponse = deferred()
      const checking = deferred()
      let failCheck!: (error: Error) => void
      const failedCheck = new Promise<void>((_resolve, reject) => { failCheck = reject })
      io.request.mockImplementationOnce(() => oldResponse.promise).mockImplementationOnce(() => newResponse.promise)
      const oldRead = read(resource).catch(error => error)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(1))
      io.get.mockImplementationOnce(() => {
        checking.resolve(null)
        return failedCheck
      })
      oldResponse.resolve(response(resource === "messages" ? "/messages" : "/characters", "Alice"))
      await checking.promise
      configure({ ...alice, accessToken: jwt("bob") })
      io.owner = "Bob"
      const newRead = read(resource)
      await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(2))
      failCheck(new Error("old configuration read failed"))
      expect(await oldRead).toMatchObject({ message: "old configuration read failed" })
      const { joined } = await joinPendingRead()
      newResponse.resolve(response(resource === "messages" ? "/messages" : "/characters", "Bob"))
      expect(await Promise.all([newRead, joined])).toEqual(["Bob", "Bob"])
      expect(io.request).toHaveBeenCalledTimes(2)
    })

    it("never serves a cache hit after credentials are removed", async () => {
      expect(await read(resource)).toBe("Alice")
      configure({ ...alice, accessToken: undefined })
      await expect(read(resource)).rejects.toThrow(/authenticated|log in/i)
    })
  })
})
