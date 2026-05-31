import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>()
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
  createSafeStorage: () => ({
    get: vi.fn(async (key: string) => mocks.storage.get(key)),
    set: vi.fn(async (key: string, value: unknown) => {
      mocks.storage.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      mocks.storage.delete(key)
    })
  }),
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
    window.localStorage.clear()
  })

  afterEach(() => {
    restoreEnv()
    window.localStorage.clear()
  })

  it("creates first-run quickstart config from the public single-user API key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "quickstart-api-key"
    delete process.env.NEXT_PUBLIC_API_URL

    const client = new TldwApiClient()

    await client.initialize()

    await expect(client.getConfig()).resolves.toMatchObject({
      authMode: "single-user",
      apiKey: "quickstart-api-key",
      serverUrl: window.location.origin
    })
    expect(mocks.storage.get("tldwConfig")).toMatchObject({
      authMode: "single-user",
      apiKey: "quickstart-api-key",
      serverUrl: window.location.origin
    })
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
})
