import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const state = vi.hoisted(() => ({
  runtime: {} as {
    id?: string
    sendMessage?: (message: unknown) => Promise<unknown>
  },
  config: {} as Record<string, unknown>
}))
vi.mock("wxt/browser", () => ({ browser: { runtime: state.runtime } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) =>
      key === "tldwConfig" ? state.config : undefined
  })
}))

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" }
  })

describe("direct recipe authority", () => {
  beforeEach(() => {
    vi.resetModules()
    delete state.runtime.id
    delete state.runtime.sendMessage
    state.config = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "real-direct-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "http://localhost:3000"
    }
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it("shares system/composer markers without deriving a stored-user hint", async () => {
    const system = await import("@/services/recipe-persistence-uncertainty")
    const composer = await import("@/services/recipe-persistence-uncertainty")
    const owner = await system.resolveRecipePersistenceOwnerView()
    expect(owner?.ownerId).toMatch(/^recipe-owner:sha256:[a-f0-9]{64}$/)
    await system.markRecipePersistenceScoped("recipe", owner!.ownerId)
    expect(
      await composer.readRecipePersistenceUncertainty("recipe", owner!.ownerId)
    ).toBe("scoped")
    await composer.clearRecipePersistenceScoped("recipe", owner!.ownerId)
    expect(
      await system.readRecipePersistenceUncertainty("recipe", owner!.ownerId)
    ).toBe("clear")
  })

  it("marks the direct singleton before create/update fetch and retains raw success", async () => {
    const authority = await import("@/services/recipe-persistence-uncertainty")
    const { apiSend } = await import("@/services/api-send")
    const owner = (await authority.resolveRecipePersistenceOwnerView())!
    const observed: string[] = []
    vi.stubGlobal("fetch", async () => {
      observed.push(
        await authority.readRecipePersistenceUncertainty("one", owner.ownerId)
      )
      return json({ id: "remote" })
    })
    for (const [path, method] of [
      ["/api/v1/prompts/", "POST"],
      ["/api/v1/prompts/123", "PATCH"]
    ]) {
      const result = await apiSend({
        path,
        method,
        body: {},
        recipePersistence: {
          mode: "require",
          expectedOwnerId: owner.ownerId,
          localId: "one"
        }
      } as Parameters<typeof apiSend>[0])
      expect(result.recipePersistence).toEqual({
        state: "dispatched",
        actualOwnerId: owner.ownerId
      })
    }
    expect(observed).toEqual(["scoped", "scoped"])
    expect(
      await authority.readRecipePersistenceUncertainty("one", owner.ownerId)
    ).toBe("scoped")
  })

  it("binds manual bearer principal lookup to the captured credentials despite config drift", async () => {
    state.config = {
      ...state.config,
      authMode: "multi-user",
      accessToken: "alice-token"
    }
    const calls: { url: string; token: string | null }[] = []
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      calls.push({ url, token: new Headers(init.headers).get("Authorization") })
      state.config = { ...state.config, accessToken: "bob-token" }
      return json({ id: 42 })
    })
    const authority = await import("@/services/recipe-persistence-uncertainty")
    const alice = await authority.resolveRecipePersistenceOwnerView()
    expect(alice).not.toBeNull()
    expect(calls).toEqual([
      {
        url: "http://localhost:3000/api/v1/auth/me",
        token: "Bearer alice-token"
      }
    ])
  })

  it("does not dispatch when the application marker cannot be written", async () => {
    const authority = await import("@/services/recipe-persistence-uncertainty")
    const { RecipePersistenceRegistry } = await import(
      "@/services/recipe-persistence-registry"
    )
    const { apiSend } = await import("@/services/api-send")
    const owner = (await authority.resolveRecipePersistenceOwnerView())!
    vi.spyOn(
      RecipePersistenceRegistry.prototype,
      "markScoped"
    ).mockImplementation(() => {
      throw new Error("marker unavailable")
    })
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const result = await apiSend({
      path: "/api/v1/prompts/",
      method: "POST",
      recipePersistence: {
        mode: "require",
        expectedOwnerId: owner.ownerId,
        localId: "one"
      }
    })
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("rejects an oversized dispatch ID before mutation or marker creation", async () => {
    const authority = await import("@/services/recipe-persistence-uncertainty")
    const { RecipePersistenceRegistry } = await import(
      "@/services/recipe-persistence-registry"
    )
    const { apiSend } = await import("@/services/api-send")
    const owner = (await authority.resolveRecipePersistenceOwnerView())!
    const markScoped = vi.spyOn(
      RecipePersistenceRegistry.prototype,
      "markScoped"
    )
    const fetchSpy = vi.fn(async () => json({ id: "remote" }))
    vi.stubGlobal("fetch", fetchSpy)
    const result = await apiSend({
      path: "/api/v1/prompts/",
      method: "POST",
      recipePersistence: {
        mode: "require",
        expectedOwnerId: owner.ownerId,
        localId: "x".repeat(513)
      }
    })
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(markScoped).not.toHaveBeenCalled()
  })

  it("does not use the direct mutation authority when extension messaging is missing", async () => {
    const authority = await import("@/services/recipe-persistence-uncertainty")
    const owner = (await authority.resolveRecipePersistenceOwnerView())!
    state.runtime.id = "real-extension"
    const fetchSpy = vi.fn(async () => json({ id: "remote" }))
    vi.stubGlobal("fetch", fetchSpy)
    const { apiSend } = await import("@/services/api-send")
    const result = await apiSend({
      path: "/api/v1/prompts/",
      method: "POST",
      recipePersistence: {
        mode: "require",
        expectedOwnerId: owner.ownerId,
        localId: "one"
      }
    })
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it.each([401, 200])(
    "fails closed for bearer with unauthoritative principal (%s)",
    async (status) => {
      state.config = {
        ...state.config,
        authMode: "multi-user",
        accessToken: `header.${btoa('{"sub":"42"}')}.signature`
      }
      vi.stubGlobal("fetch", async () => json({}, status))
      const authority = await import(
        "@/services/recipe-persistence-uncertainty"
      )
      expect(await authority.resolveRecipePersistenceOwnerView()).toBeNull()
    }
  )

  it("uses cookie credentials for an authoritative current-user lookup", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    state.config = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    }
    const calls: RequestInit[] = []
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      calls.push(init)
      return json({ id: "cookie-user" })
    })
    const authority = await import("@/services/recipe-persistence-uncertainty")
    expect(await authority.resolveRecipePersistenceOwnerView()).not.toBeNull()
    expect(calls[0].credentials).toBe("same-origin")
    expect(new Headers(calls[0].headers).get("Authorization")).toBeNull()
  })

  it.each([401, 200])(
    "fails closed for cookie sessions without an authoritative user (%s)",
    async (status) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      state.config = {
        serverUrl: "http://localhost:3000",
        authMode: "single-user",
        authSource: "cookie-session",
        userId: 42
      }
      vi.stubGlobal("fetch", async () => json({}, status))
      const authority = await import(
        "@/services/recipe-persistence-uncertainty"
      )
      expect(await authority.resolveRecipePersistenceOwnerView()).toBeNull()
    }
  )

  it.each(["absent", "reject", "malformed"])(
    "never uses page-local state when extension messaging is %s",
    async (mode) => {
      const authority = await import(
        "@/services/recipe-persistence-uncertainty"
      )
      const owner = (await authority.resolveRecipePersistenceOwnerView())!
      state.runtime.id = "real-extension"
      if (mode !== "absent")
        state.runtime.sendMessage = async () => {
          if (mode === "reject") throw new Error("context invalidated")
          return { ownerId: owner.ownerId, token: "must-not-accept" }
        }
      expect(await authority.resolveRecipePersistenceOwnerView()).toBeNull()
      await expect(
        authority.readRecipePersistenceUncertainty("one", owner.ownerId)
      ).rejects.toThrow()
      await expect(
        authority.markRecipePersistenceScoped("one", owner.ownerId)
      ).rejects.toThrow()
      delete state.runtime.id
      expect(
        await authority.readRecipePersistenceUncertainty("one", owner.ownerId)
      ).toBe("clear")
    }
  )
})
