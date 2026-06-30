import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchTtsProviders: vi.fn(),
  getOpenAPISpec: vi.fn()
}))

vi.mock("../audio-providers", () => ({
  fetchTtsProviders: (...args: unknown[]) =>
    (mocks.fetchTtsProviders as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("../TldwApiClient", () => ({
  tldwClient: {
    getOpenAPISpec: (...args: unknown[]) =>
      (mocks.getOpenAPISpec as (...args: unknown[]) => unknown)(...args)
  }
}))

describe("fetchTldwTtsModels", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.fetchTtsProviders.mockReset()
    mocks.getOpenAPISpec.mockReset()
    mocks.fetchTtsProviders.mockResolvedValue(null)
    mocks.getOpenAPISpec.mockRejectedValue(new Error("offline"))
  })

  it("includes KittenTTS model variants in the fallback catalog", async () => {
    const { fetchTldwTtsModels } = await import("../audio-models")

    const models = await fetchTldwTtsModels()
    const ids = models.map((entry) => entry.id)

    expect(ids).toContain("kitten_tts")
    expect(ids).toContain("KittenML/kitten-tts-nano-0.8")
    expect(ids).toContain("KittenML/kitten-tts-nano-0.8-int8")
    expect(ids).toContain("KittenML/kitten-tts-micro-0.8")
    expect(ids).toContain("KittenML/kitten-tts-mini-0.8")
  })

  it("includes Chatterbox family variants in the fallback catalog", async () => {
    const { fetchTldwTtsModels } = await import("../audio-models")

    const models = await fetchTldwTtsModels()
    const ids = models.map((entry) => entry.id)

    expect(ids).toContain("chatterbox")
    expect(ids).toContain("chatterbox-emotion")
    expect(ids).toContain("chatterbox-multilingual")
    expect(ids).toContain("chatterbox-turbo")
  })

  it("normalizes OpenAPI description lists before returning models", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      components: {
        schemas: {
          OpenAISpeechRequest: {
            properties: {
              model: {
                description:
                  "Supported models: kokoro, kitten_tts, KittenML/kitten-tts-nano-0.8, and vibevoice."
              }
            }
          }
        }
      }
    })

    const { fetchTldwTtsModels } = await import("../audio-models")

    const models = await fetchTldwTtsModels()
    const ids = models.map((entry) => entry.id)

    expect(ids).toContain("kitten_tts")
    expect(ids).toContain("KittenML/kitten-tts-nano-0.8")
    expect(ids).toContain("vibevoice")
  })
})
