import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>(),
  sessionStorage: new Map<string, unknown>()
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

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL
const originalPublicApiKey = process.env.NEXT_PUBLIC_X_API_KEY

const restoreEnv = () => {
  if (originalDeploymentMode === undefined) {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  } else {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
  }

  if (originalApiUrl === undefined) {
    delete process.env.NEXT_PUBLIC_API_URL
  } else {
    process.env.NEXT_PUBLIC_API_URL = originalApiUrl
  }

  if (originalPublicApiKey === undefined) {
    delete process.env.NEXT_PUBLIC_X_API_KEY
  } else {
    process.env.NEXT_PUBLIC_X_API_KEY = originalPublicApiKey
  }
}

describe("TldwApiClient quickstart auth bootstrap", () => {
  beforeEach(() => {
    restoreEnv()
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
    mocks.bgStream.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storage.clear()
    mocks.sessionStorage.clear()
    window.localStorage.clear()
  })

  afterEach(() => {
    restoreEnv()
    window.localStorage.clear()
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

  it("scrubs an ambiguous legacy persistent key instead of migrating it", async () => {
    mocks.storage.set("tldwConfig", {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      apiKey: "ambiguous-secret"
    })
    const client = new TldwApiClient()

    await client.initialize()

    expect(mocks.storage.get("tldwConfig")).not.toHaveProperty("apiKey")
    expect(mocks.sessionStorage.has("tldwManualSessionApiKey")).toBe(false)
    await expect(client.getConfig()).resolves.not.toHaveProperty("apiKey")
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
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    })
  })
})
