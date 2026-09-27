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

  it("normalizes malformed provider capability shapes at the API boundary", async () => {
    mocks.bgRequestClient.mockResolvedValue({
      providers: {
        "gateway:safe": {
          provider_name: "gateway:safe",
          display_name: { unsafe: true },
          models: ["Vendor/Good", null, 42, "   ", "Vendor/Other"],
          default_model: ["Vendor/Good"],
          formats: "mp3",
          languages: ["en", null, 7],
          supports_streaming: "true",
          model_capabilities: {
            "Vendor/Good": {
              formats: { mp3: true },
              native_formats: ["wav", null],
              converted_formats: ["mp3", false],
              default_format: "wav",
              voices: { Narrator: true },
              default_voice: 9,
              requires_freeform_voice: "false",
              base_url: "https://model-authority.invalid"
            },
            "Vendor/Other": null
          },
          fallback: {
            available: "yes",
            targets: ["openrouter", 12, "gateway:backup"],
            admin_url: "https://gateway-authority.invalid/admin"
          },
          voices: [
            { id: "Narrator", name: 7, language: "en" },
            "LegacyVoice",
            null
          ],
          base_url: "https://gateway-authority.invalid",
          credential_source: "user-api-key"
        },
        "gateway:broken": {
          display_name: 11,
          models: { unsafe: true },
          default_model: { unsafe: true },
          model_capabilities: "not-a-mapping",
          fallback: ["openrouter"]
        }
      },
      voices: {
        broken: { id: "not-an-array" }
      },
      supports_explicit_backend: true
    })

    const result = await fetchTtsProviders()

    expect(result).toEqual({
      providers: {
        "gateway:safe": {
          provider_name: "gateway:safe",
          models: ["Vendor/Good", "Vendor/Other"],
          formats: [],
          languages: ["en"],
          model_capabilities: {
            "Vendor/Good": {
              formats: [],
              native_formats: ["wav"],
              converted_formats: ["mp3"],
              default_format: "wav",
              voices: []
            }
          },
          fallback: {
            targets: ["openrouter", "gateway:backup"]
          },
          voices: [
            { id: "Narrator", language: "en" },
            { id: "LegacyVoice", name: "LegacyVoice" }
          ]
        },
        "gateway:broken": {
          models: [],
          model_capabilities: {}
        }
      },
      voices: {
        "gateway:safe": [
          { id: "Narrator", language: "en" },
          { id: "LegacyVoice", name: "LegacyVoice" }
        ],
        broken: []
      },
      supports_explicit_backend: true
    })
    expect(JSON.stringify(result)).not.toMatch(
      /gateway-authority|model-authority|credential_source|admin_url|base_url/
    )
  })

  it("bounds discovery arrays before exposing them to settings consumers", async () => {
    mocks.bgRequestClient.mockResolvedValue({
      providers: {
        "gateway:bounded": {
          models: Array.from({ length: 1_050 }, (_, index) => `Model/${index}`),
          model_capabilities: {
            "Model/0": {
              formats: Array.from(
                { length: 80 },
                (_, index) => `format-${index}`
              ),
              voices: Array.from(
                { length: 1_050 },
                (_, index) => `Voice/${index}`
              )
            }
          },
          fallback: {
            targets: Array.from(
              { length: 80 },
              (_, index) => `gateway:target-${index}`
            )
          }
        }
      },
      voices: {}
    })

    const result = await fetchTtsProviders()
    const provider = result?.providers["gateway:bounded"]
    const model = provider?.model_capabilities?.["Model/0"]

    expect(provider?.models).toHaveLength(1_000)
    expect(model?.formats).toHaveLength(64)
    expect(model?.voices).toHaveLength(1_000)
    expect(provider?.fallback?.targets).toHaveLength(64)
  })
})
