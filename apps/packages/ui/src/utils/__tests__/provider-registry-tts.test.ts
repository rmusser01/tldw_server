import { describe, expect, it } from "vitest"

import {
  getProvidersByCapability,
  getProviderLabel,
  inferProviderFromModel,
  normalizeProviderKey
} from "@/utils/provider-registry"

describe("provider-registry TTS inference", () => {
  it("infers KittenTTS from local aliases and repo ids", () => {
    expect(inferProviderFromModel("kitten_tts", "tts")).toBe("kitten_tts")
    expect(inferProviderFromModel("KittenTTS", "tts")).toBe("kitten_tts")
    expect(inferProviderFromModel("KittenML/kitten-tts-micro-0.8", "tts")).toBe(
      "kitten_tts"
    )
  })

  it("returns the KittenTTS label for tts-engine displays", () => {
    expect(getProviderLabel("kitten_tts", "tts-engine")).toBe("KittenTTS")
  })

  it("normalizes KittenTTS aliases to the canonical registry key", () => {
    expect(normalizeProviderKey("kittentts")).toBe("kitten_tts")
    expect(normalizeProviderKey("kitten-tts")).toBe("kitten_tts")
  })

  it("keeps KittenTTS unique in capability listings", () => {
    const providers = getProvidersByCapability("tts-engine")
    const kittenProviders = providers.filter(({ key }) => key === "kitten_tts")
    expect(kittenProviders).toHaveLength(1)
  })
})

describe("provider-registry LLM inference", () => {
  it.each([
    ["gpt-4o", "openai"],
    ["claude-3-7-sonnet", "anthropic"],
    ["meta-llama/Llama-3.3-70B-Instruct", "meta"],
    ["gemini-2.5-pro", "google"],
    ["mistral-large-latest", "mistral"]
  ])("infers %s as %s", (model, provider) => {
    expect(inferProviderFromModel(model)).toBe(provider)
  })
})
