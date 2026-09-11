import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const harness = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    values: new Map<string, unknown>(),
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
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: async (key: string) => harness.values.get(key),
    set: async (key: string, value: unknown) => {
      harness.values.set(key, value)
    },
    remove: async (key: string) => {
      harness.values.delete(key)
    }
  })
}))
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
          return send(message)
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
        session: { get: async () => ({}), set: async () => {} },
        onChanged: event()
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
const ownerA = `recipe-owner:sha256:${"a".repeat(64)}`
const ownerB = `recipe-owner:sha256:${"b".repeat(64)}`

describe("background recipe authority protocol", () => {
  beforeEach(async () => {
    vi.resetModules()
    harness.listeners.clear()
    harness.values.clear()
    harness.sent.length = 0
    harness.values.set("tldwConfig", {
      serverUrl: "https://api.example.test/base",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "worker-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    })
    vi.stubGlobal("window", undefined)
    const background = (await import("@/entries/background")).default
    background.main()
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("shares sidepanel and pop-out state after sidepanel disposal, with independent unknown Forget", async () => {
    let sidepanel = await import("@/services/recipe-persistence-uncertainty")
    await sidepanel.markRecipePersistenceScoped("one", ownerA)
    // Reloading the page module does not reload the already-running worker.
    vi.resetModules()
    sidepanel = null as unknown as typeof sidepanel
    const popout = await import("@/services/recipe-persistence-uncertainty")
    expect(await popout.readRecipePersistenceUncertainty("one", ownerA)).toBe(
      "scoped"
    )
    await popout.clearRecipePersistenceScoped("one", ownerB)
    expect(await popout.readRecipePersistenceUncertainty("one", ownerA)).toBe(
      "scoped"
    )
    await popout.markRecipePersistenceUnknown("one")
    expect(await popout.readRecipePersistenceUncertainty("one", ownerB)).toBe(
      "unknown_owner"
    )
    await popout.forgetRecipePersistenceUnknown("one")
    expect(await popout.readRecipePersistenceUncertainty("one", ownerA)).toBe(
      "scoped"
    )
    expect(harness.sent).toContainEqual({
      type: "tldw:recipe-uncertainty:mark-scoped",
      id: "one",
      ownerId: ownerA
    })
    expect(JSON.stringify(harness.sent)).not.toMatch(
      /worker-key|headers|snapshot|accessToken/
    )
  })

  it("restarts with a clean registry without erasing durable sync error", async () => {
    harness.values.set("saved-recipe", { id: "one", syncStatus: "error" })
    await send({
      type: "tldw:recipe-uncertainty:mark-scoped",
      id: "one",
      ownerId: ownerA
    })
    harness.listeners.clear()
    ;(await import("@/entries/background")).default.main()
    expect(
      await send({
        type: "tldw:recipe-uncertainty:read",
        id: "one",
        ownerId: ownerA
      })
    ).toBe("clear")
    expect(harness.values.get("saved-recipe")).toEqual({
      id: "one",
      syncStatus: "error"
    })
  })

  it("returns only owner/revision and marks before create/update fetch without clearing success", async () => {
    const owner = (await send({ type: "tldw:recipe-owner:resolve" })) as {
      ownerId: string
      authorizationRevision: string
    }
    expect(Object.keys(owner).sort()).toEqual([
      "authorizationRevision",
      "ownerId"
    ])
    expect(owner.ownerId).toMatch(/^recipe-owner:sha256:[a-f0-9]{64}$/)
    const observations: unknown[] = []
    vi.stubGlobal("fetch", async (url: string) => {
      observations.push({
        url,
        state: await send({
          type: "tldw:recipe-uncertainty:read",
          id: "one",
          ownerId: owner.ownerId
        })
      })
      return json({ id: "remote" })
    })
    for (const [path, method] of [
      ["/api/v1/prompts/", "POST"],
      ["/api/v1/prompts/123", "PATCH"]
    ]) {
      expect(
        await send({
          type: "tldw:request",
          payload: {
            path,
            method,
            body: {},
            recipePersistence: {
              mode: "require",
              expectedOwnerId: owner.ownerId,
              localId: "one"
            }
          }
        })
      ).toMatchObject({
        ok: true,
        recipePersistence: { state: "dispatched", actualOwnerId: owner.ownerId }
      })
      if (method === "POST")
        await send({
          type: "tldw:recipe-uncertainty:clear-scoped",
          id: "one",
          ownerId: owner.ownerId
        })
    }
    expect(observations).toEqual([
      { url: "https://api.example.test/base/api/v1/prompts/", state: "scoped" },
      {
        url: "https://api.example.test/base/api/v1/prompts/123",
        state: "scoped"
      }
    ])
    expect(
      await send({
        type: "tldw:recipe-uncertainty:read",
        id: "one",
        ownerId: owner.ownerId
      })
    ).toBe("scoped")
  })

  it("rejects an oversized dispatch ID before mutation or marker creation", async () => {
    const { RecipePersistenceRegistry } = await import(
      "@/services/recipe-persistence-registry"
    )
    const markScoped = vi.spyOn(
      RecipePersistenceRegistry.prototype,
      "markScoped"
    )
    const owner = (await send({ type: "tldw:recipe-owner:resolve" })) as {
      ownerId: string
    }
    const fetchSpy = vi.fn(async () => json({ id: "remote" }))
    vi.stubGlobal("fetch", fetchSpy)
    const result = await send({
      type: "tldw:request",
      payload: {
        path: "/api/v1/prompts/",
        method: "POST",
        body: {},
        recipePersistence: {
          mode: "require",
          expectedOwnerId: owner.ownerId,
          localId: "x".repeat(513)
        }
      }
    })
    expect(result).toMatchObject({
      ok: false,
      recipePersistence: { state: "not_dispatched", actualOwnerId: null }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
    // Call-through spy: invalid IDs cannot be read via the public protocol, so
    // observe the real registry's mutation boundary without replacing it.
    expect(markScoped).not.toHaveBeenCalled()
  })

  it("keeps a maximum-length dispatched ID readable and clearable", async () => {
    const owner = (await send({ type: "tldw:recipe-owner:resolve" })) as {
      ownerId: string
    }
    const id = "x".repeat(512)
    vi.stubGlobal("fetch", async () => json({ id: "remote" }))
    expect(
      await send({
        type: "tldw:request",
        payload: {
          path: "/api/v1/prompts/",
          method: "POST",
          body: {},
          recipePersistence: {
            mode: "require",
            expectedOwnerId: owner.ownerId,
            localId: id
          }
        }
      })
    ).toMatchObject({
      ok: true,
      recipePersistence: { state: "dispatched", actualOwnerId: owner.ownerId }
    })
    expect(
      await send({
        type: "tldw:recipe-uncertainty:read",
        id,
        ownerId: owner.ownerId
      })
    ).toBe("scoped")
    expect(
      await send({
        type: "tldw:recipe-uncertainty:clear-scoped",
        id,
        ownerId: owner.ownerId
      })
    ).toEqual({ ok: true })
    expect(
      await send({
        type: "tldw:recipe-uncertainty:read",
        id,
        ownerId: owner.ownerId
      })
    ).toBe("clear")
  })

  it("resolves a bearer through the snapshot-bound local current-user endpoint, never messaging itself", async () => {
    harness.values.set("tldwConfig", {
      serverUrl: "https://api.example.test/base",
      authMode: "multi-user",
      accessToken: "alice-token"
    })
    const requests: unknown[] = []
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      requests.push({
        url,
        token: new Headers(init.headers).get("Authorization")
      })
      harness.values.set("tldwConfig", {
        serverUrl: "https://other.example.test",
        authMode: "multi-user",
        accessToken: "bob-token"
      })
      return json({ id: 42 })
    })
    expect(await send({ type: "tldw:recipe-owner:resolve" })).toMatchObject({
      ownerId: expect.stringMatching(/^recipe-owner:/)
    })
    expect(requests).toEqual([
      {
        url: "https://api.example.test/base/api/v1/auth/me",
        token: "Bearer alice-token"
      }
    ])
    expect(harness.sent).toEqual([])
  })

  it.each([401, 200])(
    "fails closed without an authoritative bearer user (%s)",
    async (status) => {
      harness.values.set("tldwConfig", {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        accessToken: `h.${btoa('{"sub":"42"}')}.s`
      })
      vi.stubGlobal("fetch", async () => json({}, status))
      expect(await send({ type: "tldw:recipe-owner:resolve" })).toBeNull()
    }
  )

  it.each([
    { type: "tldw:recipe-uncertainty:mark-scoped", id: "", ownerId: ownerA },
    {
      type: "tldw:recipe-uncertainty:mark-scoped",
      id: "x".repeat(513),
      ownerId: ownerA
    },
    {
      type: "tldw:recipe-uncertainty:clear-scoped",
      id: "one",
      ownerId: "user:anonymous"
    },
    { type: "tldw:recipe-uncertainty:read", id: "one", ownerId: 42 },
    { type: "tldw:recipe-owner:resolve", headers: { Authorization: "evil" } },
    { type: "tldw:recipe-uncertainty:mark-unknown", id: "one", snapshot: {} }
  ])(
    "rejects malformed or credential-bearing protocol input %#",
    async (message) => {
      expect(await send(message)).toMatchObject({ ok: false })
      expect(
        await send({
          type: "tldw:recipe-uncertainty:read",
          id: "one",
          ownerId: ownerA
        })
      ).toBe("clear")
    }
  )
})
