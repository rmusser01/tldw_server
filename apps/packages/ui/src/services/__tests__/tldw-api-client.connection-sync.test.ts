import { beforeEach, describe, expect, it, vi } from "vitest"

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

describe("TldwApiClient connection storage sync", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
    mocks.bgStream.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storage.clear()
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
})
