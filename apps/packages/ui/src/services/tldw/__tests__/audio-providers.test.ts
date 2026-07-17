import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequestClient: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequestClient: (...args: unknown[]) => mocks.bgRequestClient(...args)
}))

import { fetchTtsProviders } from "../audio-providers"

describe("fetchTtsProviders", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("preserves legacy provider and voice parsing without advertising explicit backends", async () => {
    mocks.bgRequestClient.mockResolvedValue({
      providers: {
        openai: { models: ["tts-1"], voices: [{ id: "alloy" }] }
      },
      voices: { kokoro: [{ id: "af" }] }
    })

    await expect(fetchTtsProviders()).resolves.toEqual({
      providers: {
        openai: { models: ["tts-1"], voices: [{ id: "alloy" }] }
      },
      voices: {
        openai: [{ id: "alloy" }],
        kokoro: [{ id: "af" }]
      },
      supports_explicit_backend: false
    })
  })

  it("exposes only an exact boolean explicit-backend support flag", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      providers: { "gateway:company": { models: ["Vendor/Exact"] } },
      voices: {},
      supports_explicit_backend: true
    })

    await expect(fetchTtsProviders()).resolves.toMatchObject({
      supports_explicit_backend: true
    })

    mocks.bgRequestClient.mockResolvedValueOnce({
      providers: { openai: { models: ["tts-1"] } },
      voices: {},
      supports_explicit_backend: "true"
    })

    await expect(fetchTtsProviders()).resolves.toMatchObject({
      supports_explicit_backend: false
    })
  })
})
