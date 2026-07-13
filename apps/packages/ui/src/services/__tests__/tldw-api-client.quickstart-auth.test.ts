import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>(),
  sessionStorage: new Map<string, unknown>(),
  storageRemoveError: null as Error | null,
  envApiKey: vi.fn<() => string | null>()
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
    const values = options?.area === "session" ? mocks.sessionStorage : mocks.storage
    return {
      get: vi.fn(async (key: string) => values.get(key)),
      set: vi.fn(async (key: string, value: unknown) => {
        values.set(key, value)
      }),
      remove: vi.fn(async (key: string) => {
        if (options?.area !== "session" && mocks.storageRemoveError) {
          throw mocks.storageRemoveError
        }
        values.delete(key)
      })
    }
  },
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import {
  activateCookieSessionConfig,
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride,
  isCookieSessionConfigInvalidated
} from "@/services/tldw/runtime-auth-override"

describe("TldwApiClient quickstart auth bootstrap", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", undefined)
    vi.stubEnv("NEXT_PUBLIC_API_URL", undefined)
    vi.stubEnv("NEXT_PUBLIC_X_API_KEY", undefined)
    vi.stubEnv("VITE_TLDW_API_KEY", undefined)
    vi.stubEnv("VITE_TLDW_DEFAULT_API_KEY", undefined)
    mocks.envApiKey.mockReset()
    mocks.envApiKey.mockReturnValue(null)
    vi.spyOn(
      TldwApiClient.prototype as any,
      "getEnvApiKey"
    ).mockImplementation(() => mocks.envApiKey())
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
    mocks.bgStream.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storage.clear()
    mocks.sessionStorage.clear()
    mocks.storageRemoveError = null
    window.localStorage.clear()
    activateCookieSessionConfig()
    clearRuntimeAuthOverride()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    window.localStorage.clear()
    activateCookieSessionConfig()
    clearRuntimeAuthOverride()
  })

  it("does not persist the public quickstart api key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "quickstart-api-key"
    delete process.env.NEXT_PUBLIC_API_URL

    const client = new TldwApiClient()

    await client.initialize()

    await expect(client.getConfig()).resolves.toBeNull()
    expect(mocks.storage.get("tldwConfig")).toBeUndefined()
  })

  it("accepts a cookie-session quickstart config without an api key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-public-key"
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await expect(client.ensureConfigForRequest(true)).resolves.toMatchObject({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
  })

  it("hydrates the active cookie connection without mutating the stored manual record", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      serverUrl: "https://remote.example.test",
      apiKey: "manual-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", manualConfig)
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await client.initialize()
    await expect(client.getConfig()).resolves.toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(mocks.storage.get("tldwConfig")).toEqual(manualConfig)

    await client.initialize()
    await client.getConfig()
    expect(mocks.storage.get("tldwConfig")).toEqual(manualConfig)
  })

  it("preserves a matching manual session credential while cookie auth is active", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      authSource: "manual" as const,
      serverUrl: "https://remote.example.test/path",
      credentialSource: "manual" as const,
      apiKeyPersistence: "session" as const,
      apiKeyServerOrigin: "https://remote.example.test"
    }
    const manualSession = {
      apiKey: "session-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", manualConfig)
    mocks.sessionStorage.set("tldwManualSessionApiKey", manualSession)
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await client.initialize()

    await expect(client.getConfig()).resolves.toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(mocks.storage.get("tldwConfig")).toEqual(manualConfig)
    expect(mocks.sessionStorage.get("tldwManualSessionApiKey")).toEqual(
      manualSession
    )
  })

  it("removes the cookie marker and rehydrates the preserved manual connection", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      authSource: "manual" as const,
      serverUrl: "https://remote.example.test/path",
      apiKey: "manual-key",
      credentialSource: "manual" as const,
      apiKeyPersistence: "device" as const,
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", manualConfig)
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()
    await client.initialize()

    await client.clearCookieSingleUserSession()

    expect(mocks.storage.has("tldwCookieSessionConfig")).toBe(false)
    expect(isCookieSessionConfigInvalidated()).toBe(true)
    await expect(client.getConfig()).resolves.toEqual(manualConfig)
    expect(mocks.storage.get("tldwConfig")).toEqual(manualConfig)
  })

  it("rehydrates the preserved manual connection when marker removal fails", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      authSource: "manual" as const,
      serverUrl: "https://remote.example.test/path",
      apiKey: "manual-key",
      credentialSource: "manual" as const,
      apiKeyPersistence: "device" as const,
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", manualConfig)
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()
    await client.initialize()
    mocks.storageRemoveError = new Error("marker removal unavailable")

    await expect(client.clearCookieSingleUserSession()).rejects.toThrow(
      "marker removal unavailable"
    )

    expect(isCookieSessionConfigInvalidated()).toBe(true)
    await expect(client.getConfig()).resolves.toEqual(manualConfig)
    expect(mocks.storage.get("tldwCookieSessionConfig")).toBeDefined()
  })

  it("removes a mismatched manual session credential while cookie auth is active", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      authSource: "manual" as const,
      serverUrl: "https://remote.example.test/path",
      credentialSource: "manual" as const,
      apiKeyPersistence: "session" as const,
      apiKeyServerOrigin: "https://remote.example.test"
    }
    mocks.storage.set("tldwConfig", manualConfig)
    mocks.sessionStorage.set("tldwManualSessionApiKey", {
      apiKey: "stale-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://other.example.test"
    })
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await client.initialize()

    await expect(client.getConfig()).resolves.toMatchObject({
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(mocks.storage.get("tldwConfig")).toEqual(manualConfig)
    expect(mocks.sessionStorage.has("tldwManualSessionApiKey")).toBe(false)
  })

  it("keeps the explicit missing-key error when quickstart config has no key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_API_URL
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await expect(client.ensureConfigForRequest(true)).rejects.toThrow(
      /API key is missing/i
    )
  })

  it("migrates an eligible pre-metadata key to an origin-bound device credential", async () => {
    const legacyConfig = {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-device-secret"
    }
    mocks.storage.set("tldwConfig", legacyConfig)
    const client = new TldwApiClient()

    await client.initialize()

    const migratedConfig = {
      ...legacyConfig,
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    }
    expect(mocks.storage.get("tldwConfig")).toEqual(migratedConfig)
    await client.initialize()
    expect(mocks.storage.get("tldwConfig")).toEqual(migratedConfig)
    await expect(client.ensureConfigForRequest(true)).resolves.toEqual(
      migratedConfig
    )
  })

  it.each([
    { property: "credentialSource" },
    { property: "apiKeyPersistence" },
    { property: "apiKeyServerOrigin" }
  ])("scrubs a legacy key when $property is present but empty", async ({
    property
  }) => {
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      authSource: "manual",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-device-secret",
      [property]: ""
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    await expect(client.getConfig()).resolves.not.toHaveProperty("apiKey")
  })

  it.each([
    {
      caseName: "an empty auth source",
      deploymentMode: undefined,
      config: {
        authSource: "",
        serverUrl: "https://api.example.test/path",
        apiKey: "legacy-device-secret"
      }
    },
    {
      caseName: "a cookie-session auth source",
      deploymentMode: undefined,
      config: {
        authSource: "cookie-session",
        serverUrl: "https://api.example.test/path",
        apiKey: "legacy-device-secret"
      }
    },
    {
      caseName: "a placeholder API key",
      deploymentMode: undefined,
      config: {
        authSource: "manual",
        serverUrl: "https://api.example.test/path",
        apiKey: "REPLACE-ME"
      }
    },
    {
      caseName: "an invalid server origin",
      deploymentMode: undefined,
      config: {
        authSource: "manual",
        serverUrl: "not a URL",
        apiKey: "legacy-device-secret"
      }
    },
    {
      caseName: "hosted mode",
      deploymentMode: "hosted",
      config: {
        authSource: "manual",
        serverUrl: "https://api.example.test/path",
        apiKey: "legacy-device-secret"
      }
    },
    {
      caseName: "same-origin quickstart mode",
      deploymentMode: "quickstart",
      config: {
        authSource: "manual",
        serverUrl: window.location.origin,
        apiKey: "legacy-device-secret"
      }
    }
  ])("scrubs a legacy key with $caseName", async ({
    deploymentMode,
    config
  }) => {
    if (deploymentMode) {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", deploymentMode)
    }
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      ...config
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    await expect(client.getConfig()).resolves.not.toHaveProperty("apiKey")
  })

  it("uses an active environment key after scrubbing a legacy key", async () => {
    mocks.envApiKey.mockReturnValue("active-environment-key")
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-device-secret"
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    await expect(client.ensureConfigForRequest(true)).resolves.toMatchObject({
      apiKey: "active-environment-key"
    })
  })

  it("uses an active runtime key after scrubbing a legacy key", async () => {
    setRuntimeSingleUserApiKeyOverride("active-runtime-key")
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-device-secret"
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    await expect(client.ensureConfigForRequest(true)).resolves.toMatchObject({
      apiKey: "active-runtime-key"
    })
  })

  it("uses an active quickstart cookie session after scrubbing a legacy key", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-device-secret"
    })
    mocks.storage.set("tldwCookieSessionConfig", {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    await expect(client.ensureConfigForRequest(true)).resolves.toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
  })

  it("migrates a confidently manual legacy key to complete device metadata", async () => {
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      authSource: "manual",
      serverUrl: "https://api.example.test/path",
      apiKey: "legacy-manual-secret"
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      apiKey: "legacy-manual-secret",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    })
  })
})
