import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>(),
  syncStorage: new Map<string, unknown>(),
  sessionStorage: new Map<string, unknown>(),
  failDeviceWrite: false,
  failSessionWrite: false,
  failClearWrite: false,
  beforeLocalConfigWrite: null as null | (() => void)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

vi.mock("@/services/tldw/request-core", () => ({
  tldwRequest: (...args: unknown[]) => mocks.tldwRequest(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => {
    const values =
      options?.area === "session"
        ? mocks.sessionStorage
        : options?.area === "local"
          ? mocks.storage
          : mocks.syncStorage
    return {
      get: vi.fn(async (key: string) => values.get(key)),
      set: vi.fn(async (key: string, value: unknown) => {
        if (
          options?.area === "session" &&
          mocks.failSessionWrite &&
          key === "tldwManualSessionApiKey"
        ) {
          throw new Error("session storage unavailable")
        }
        if (
          options?.area === "local" &&
          mocks.failDeviceWrite &&
          key === "tldwConfig" &&
          Boolean((value as { apiKey?: unknown })?.apiKey)
        ) {
          throw new Error("device storage unavailable")
        }
        if (
          options?.area === "local" &&
          mocks.failClearWrite &&
          key === "tldwConfig" &&
          !(value as { apiKey?: unknown })?.apiKey
        ) {
          throw new Error("persistent clear unavailable")
        }
        if (options?.area === "local" && key === "tldwConfig") {
          const beforeWrite = mocks.beforeLocalConfigWrite
          mocks.beforeLocalConfigWrite = null
          beforeWrite?.()
        }
        values.set(key, value)
      }),
      remove: vi.fn(async (key: string) => {
        values.delete(key)
      })
    }
  },
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import {
  TldwApiClient,
  tldwClient
} from "@/services/tldw/TldwApiClient"
import { TldwAuthService } from "@/services/tldw/TldwAuth"

describe("TldwApiClient connection storage sync", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
    mocks.bgStream.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storage.clear()
    mocks.syncStorage.clear()
    mocks.sessionStorage.clear()
    mocks.failDeviceWrite = false
    mocks.failSessionWrite = false
    mocks.failClearWrite = false
    mocks.beforeLocalConfigWrite = null
    window.localStorage.clear()
  })

  it("mirrors saved server URLs into the WebUI bootstrap host key", async () => {
    const client = new TldwApiClient()

    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "test-api-key"
    })

    expect(mocks.storage.get("tldwServerUrl")).toBe("http://127.0.0.1:8000")
    expect(window.localStorage.getItem("tldw-api-host")).toBe(
      "http://127.0.0.1:8000"
    )
    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    expect(mocks.syncStorage.has("tldwConfig")).toBe(false)
  })

  it("clears mirrored server URLs when the saved URL is removed", async () => {
    const client = new TldwApiClient()

    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    await client.updateConfig({ serverUrl: "" })

    expect(mocks.storage.has("tldwServerUrl")).toBe(false)
    expect(window.localStorage.getItem("tldw-api-host")).toBeNull()
  })

  it("makes a scoped rotation inert when logout clears tokens", async () => {
    mocks.storage.set("tldwConfig", {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      accessToken: "raw-access",
      refreshToken: "raw-refresh"
    })
    mocks.storage.set("tldwRefreshRotation", {
      version: 1,
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      sourceAccessToken: "raw-access",
      sourceRefreshToken: "raw-refresh",
      rotatedFromRefreshToken: "raw-refresh",
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })
    const client = new TldwApiClient()
    await client.initialize()

    await client.updateConfig({
      accessToken: undefined,
      refreshToken: undefined
    })

    expect(mocks.storage.has("tldwRefreshRotation")).toBe(true)
    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      accessToken: undefined,
      refreshToken: undefined
    })
    await expect(client.getConfig()).resolves.toMatchObject({
      accessToken: undefined,
      refreshToken: undefined
    })
  })

  it("consumes a scoped rotation before persisting an unrelated config update", async () => {
    const rawConfig = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      accessToken: "raw-access",
      refreshToken: "raw-refresh"
    }
    mocks.storage.set("tldwConfig", rawConfig)
    const client = new TldwApiClient()
    await client.initialize()
    mocks.storage.set("tldwRefreshRotation", {
      version: 1,
      serverUrl: rawConfig.serverUrl,
      authMode: rawConfig.authMode,
      authSource: rawConfig.authSource,
      sourceAccessToken: rawConfig.accessToken,
      sourceRefreshToken: rawConfig.refreshToken,
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })

    await client.updateConfig({ orgId: 9 })

    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      orgId: 9,
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })
    expect(mocks.storage.has("tldwRefreshRotation")).toBe(true)
  })

  it("preserves a concurrent rotation racing an unrelated config write", async () => {
    const rawConfig = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      orgId: 9,
      accessToken: "raw-access",
      refreshToken: "raw-refresh"
    }
    mocks.storage.set("tldwConfig", rawConfig)
    const client = new TldwApiClient()
    await client.initialize()
    mocks.beforeLocalConfigWrite = () => {
      mocks.storage.set("tldwRefreshRotation", {
        version: 1,
        serverUrl: rawConfig.serverUrl,
        authMode: rawConfig.authMode,
        authSource: rawConfig.authSource,
        orgId: rawConfig.orgId,
        sourceAccessToken: rawConfig.accessToken,
        sourceRefreshToken: rawConfig.refreshToken,
        accessToken: "rotated-access",
        refreshToken: "rotated-refresh"
      })
    }

    await client.updateConfig({
      serverUrl: rawConfig.serverUrl,
      authMode: rawConfig.authMode,
      orgId: rawConfig.orgId,
      requestTimeoutMs: 12_000
    } as Partial<Parameters<TldwApiClient["updateConfig"]>[0]>)

    expect(mocks.storage.has("tldwRefreshRotation")).toBe(true)
    await expect(client.getConfig()).resolves.toMatchObject({
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh",
      requestTimeoutMs: 12_000
    })
  })

  it("reloads a scoped rotation before returning current config", async () => {
    const rawConfig = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      accessToken: "raw-access",
      refreshToken: "raw-refresh"
    }
    mocks.storage.set("tldwConfig", rawConfig)
    const client = new TldwApiClient()
    await client.initialize()
    mocks.storage.set("tldwRefreshRotation", {
      version: 1,
      serverUrl: rawConfig.serverUrl,
      authMode: rawConfig.authMode,
      authSource: rawConfig.authSource,
      sourceAccessToken: rawConfig.accessToken,
      sourceRefreshToken: rawConfig.refreshToken,
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })
    await expect(client.getConfig()).resolves.toMatchObject({
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })
  })

  it("reloads a scoped rotation for cached auth-header consumers", async () => {
    const rawConfig = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      accessToken: "raw-access",
      refreshToken: "raw-refresh"
    }
    mocks.storage.set("tldwConfig", rawConfig)
    await tldwClient.initialize()
    await expect(tldwClient.getConfig()).resolves.toMatchObject({
      accessToken: "raw-access"
    })
    mocks.storage.set("tldwRefreshRotation", {
      version: 1,
      serverUrl: rawConfig.serverUrl,
      authMode: rawConfig.authMode,
      authSource: rawConfig.authSource,
      sourceAccessToken: rawConfig.accessToken,
      sourceRefreshToken: rawConfig.refreshToken,
      accessToken: "rotated-access",
      refreshToken: "rotated-refresh"
    })

    const headers = new Headers(await new TldwAuthService().getAuthHeaders())
    expect(headers.get("Authorization")).toBe("Bearer rotated-access")
    await expect(tldwClient.getConfig()).resolves.toMatchObject({
      refreshToken: "rotated-refresh"
    })
  })

  it("migrates legacy sync config into device-local storage", async () => {
    const legacyConfig = {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      apiKey: "legacy-secret",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    }
    mocks.syncStorage.set("tldwConfig", legacyConfig)

    const client = new TldwApiClient()
    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).toEqual(legacyConfig)
    expect(mocks.syncStorage.has("tldwConfig")).toBe(false)
    await expect(client.getConfig()).resolves.toMatchObject({
      apiKey: "legacy-secret"
    })
  })

  it("initializes from the effective cookie config without exposing preserved manual auth", async () => {
    const previousMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const preservedManual = {
      serverUrl: "https://remote.example.test",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "preserved-device-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", preservedManual)
    mocks.storage.set("tldwCookieSessionConfig", {
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session",
      apiKey: "must-not-leak",
      apiBearer: "must-not-leak",
      accessToken: "must-not-leak",
      refreshToken: "must-not-leak"
    })

    try {
      const client = new TldwApiClient()
      await client.initialize()

      const effective = await client.getConfig()
      expect(effective).toMatchObject({
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      })
      expect(effective).not.toHaveProperty("apiKey")
      expect(effective).not.toHaveProperty("apiBearer")
      expect(effective).not.toHaveProperty("accessToken")
      expect(effective).not.toHaveProperty("refreshToken")
      expect(mocks.storage.get("tldwConfig")).toEqual(preservedManual)
    } finally {
      if (previousMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = previousMode
      }
    }
  })

  it("uses the in-memory WebUI config for audio preset mutations", async () => {
    const client = new TldwApiClient()
    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 201,
      data: { id: "preset-1", kind: "tts", name: "Preset" }
    })

    await expect(
      client.createAudioPreset({
        kind: "tts",
        name: "Preset",
        config: { provider: "tldw", voice: "Bella" }
      })
    ).resolves.toMatchObject({ id: "preset-1" })

    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    const [request, runtime] = mocks.tldwRequest.mock.calls[0]
    expect(request).toMatchObject({
      path: "/api/v1/audio/presets",
      method: "POST"
    })
    await expect(runtime.getConfig()).resolves.toMatchObject({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
  })

  it("routes chat creation through the scoped request proxy", async () => {
    const client = new TldwApiClient()
    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    mocks.bgRequest.mockResolvedValue({
      id: "thread-1",
      title: "Knowledge thread"
    })

    await expect(
      client.createChat({ title: "Knowledge thread", source: "knowledge_qa" })
    ).resolves.toMatchObject({ id: "thread-1", title: "Knowledge thread" })

    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/chats/",
      method: "POST",
      body: expect.objectContaining({
        title: "Knowledge thread",
        source: "knowledge_qa"
      })
    }))
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("uses the in-memory WebUI config for RAG search", async () => {
    const client = new TldwApiClient()
    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        results: [],
        generated_answer: "No relevant sources found."
      }
    })

    await expect(
      client.ragSearch("What changed?", { sources: ["media_db"] })
    ).resolves.toMatchObject({
      generated_answer: "No relevant sources found."
    })

    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    const [request, runtime] = mocks.tldwRequest.mock.calls[0]
    expect(request).toMatchObject({
      path: "/api/v1/rag/search",
      method: "POST",
      body: { query: "What changed?", sources: ["media_db"] }
    })
    await expect(runtime.getConfig()).resolves.toMatchObject({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
  })

  it("uses the in-memory WebUI config for OpenAPI discovery", async () => {
    const client = new TldwApiClient()
    await client.updateConfig({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { openapi: "3.1.0", paths: { "/api/v1/audio/presets": {} } }
    })

    await expect(client.getOpenAPISpec()).resolves.toMatchObject({
      paths: { "/api/v1/audio/presets": {} }
    })

    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    const [request, runtime] = mocks.tldwRequest.mock.calls[0]
    expect(request).toMatchObject({
      path: "http://127.0.0.1:8000/openapi.json",
      method: "GET"
    })
    await expect(runtime.getConfig()).resolves.toMatchObject({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    })
  })

  it("skips same-origin quickstart OpenAPI discovery", async () => {
    const previousMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    try {
      const client = new TldwApiClient()
      await client.updateConfig({
        serverUrl: window.location.origin,
        authMode: "single-user",
        apiKey: "test-api-key"
      })

      await expect(client.getOpenAPISpec()).resolves.toBeNull()

      expect(mocks.bgRequest).not.toHaveBeenCalled()
      expect(mocks.tldwRequest).not.toHaveBeenCalled()
    } finally {
      if (previousMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = previousMode
      }
    }
  })

  it("marks missing per-chat settings as an expected 404", async () => {
    const client = new TldwApiClient()
    mocks.bgRequest.mockResolvedValue({ settings: null })

    await expect(client.getChatSettings("chat-1")).resolves.toEqual({
      settings: null
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chats/chat-1/settings?scope_type=global",
        method: "GET",
        expectedStatuses: [404]
      })
    )
  })

  it("persists an explicit device choice atomically", async () => {
    const client = new TldwApiClient()

    await expect(
      client.saveManualSingleUserCredential({
        serverUrl: "https://api.example.test/path",
        apiKey: "secret",
        persistence: "device"
      })
    ).resolves.toBe("device")

    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      serverUrl: "https://api.example.test/path",
      apiKey: "secret",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    })
    expect(mocks.sessionStorage.has("tldwManualSessionApiKey")).toBe(false)
    await expect(client.getConfig()).resolves.toMatchObject({ apiKey: "secret" })
  })

  it("keeps a session choice out of persistent config and hydrates a new client", async () => {
    const client = new TldwApiClient()

    await expect(
      client.saveManualSingleUserCredential({
        serverUrl: "https://api.example.test",
        apiKey: "session-secret",
        persistence: "session"
      })
    ).resolves.toBe("session")

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    expect(mocks.sessionStorage.get("tldwManualSessionApiKey")).toMatchObject({
      apiKey: "session-secret",
      apiKeyPersistence: "session"
    })
    const reloaded = new TldwApiClient()
    await reloaded.initialize()
    await expect(reloaded.getConfig()).resolves.toMatchObject({
      serverUrl: "https://api.example.test",
      apiKey: "session-secret",
      apiKeyPersistence: "session"
    })
  })

  it("falls back from a failed device write to session storage", async () => {
    mocks.failDeviceWrite = true
    const client = new TldwApiClient()

    await expect(
      client.saveManualSingleUserCredential({
        serverUrl: "https://api.example.test",
        apiKey: "fallback-secret",
        persistence: "device"
      })
    ).resolves.toBe("session")

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    expect(mocks.sessionStorage.get("tldwManualSessionApiKey")).toMatchObject({
      apiKey: "fallback-secret"
    })
  })

  it("falls back to memory when session storage is unavailable", async () => {
    mocks.failSessionWrite = true
    const client = new TldwApiClient()

    await expect(
      client.saveManualSingleUserCredential({
        serverUrl: "https://api.example.test",
        apiKey: "memory-secret",
        persistence: "session"
      })
    ).resolves.toBe("memory")

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty(
      "apiKeyPersistence"
    )
    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty(
      "credentialSource"
    )
    await expect(client.getConfig()).resolves.toMatchObject({
      apiKey: "memory-secret"
    })
  })

  it("clears persistent, session, and in-memory manual credentials", async () => {
    const client = new TldwApiClient()
    await client.saveManualSingleUserCredential({
      serverUrl: "https://api.example.test",
      apiKey: "secret",
      persistence: "session"
    })

    await client.clearManualSingleUserCredentials()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("credentialSource")
    expect(mocks.sessionStorage.has("tldwManualSessionApiKey")).toBe(false)
    await expect(client.getConfig()).resolves.not.toMatchObject({ apiKey: "secret" })
  })

  it("does not claim a device credential was cleared when persistence fails", async () => {
    const client = new TldwApiClient()
    await client.saveManualSingleUserCredential({
      serverUrl: "https://api.example.test",
      apiKey: "device-secret",
      persistence: "device"
    })
    mocks.failClearWrite = true

    await expect(client.clearManualSingleUserCredentials()).rejects.toThrow(
      "persistent clear unavailable"
    )

    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      apiKey: "device-secret",
      apiKeyPersistence: "device"
    })
    await expect(client.getConfig()).resolves.toMatchObject({
      apiKey: "device-secret"
    })
  })
})
