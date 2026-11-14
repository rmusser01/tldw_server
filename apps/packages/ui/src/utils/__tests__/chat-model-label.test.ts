import { describe, expect, it } from "vitest"
import {
  formatChatMessageAuthorName,
  formatChatModelLabel,
  formatChatProviderName,
  getChatModelLabelParts
} from "../chat-model-label"

describe("formatChatProviderName (CC-06)", () => {
  it.each([
    ["custom_openai_api", "Custom OpenAI API"],
    ["custom-openai-api", "Custom OpenAI API"],
    ["ollama", "Ollama"],
    ["openai", "OpenAI"]
  ])("names %s as %s", (provider, expected) => {
    expect(formatChatProviderName(provider)).toBe(expected)
  })

  it("returns nothing for placeholder providers", () => {
    expect(formatChatProviderName("unknown")).toBe("")
    expect(formatChatProviderName(undefined)).toBe("")
  })
})

describe("getChatModelLabelParts (CC-06)", () => {
  it("puts the model first and drops the internal tldw prefix", () => {
    expect(getChatModelLabelParts({ modelId: "tldw:gemma3:1b", provider: "ollama" })).toEqual({
      modelName: "gemma3:1b",
      providerName: "Ollama"
    })
  })

  it("splits a provider-qualified id into model and provider", () => {
    expect(getChatModelLabelParts({ modelId: "custom_openai_api:mock-gpt-4o" })).toEqual({
      modelName: "mock-gpt-4o",
      providerName: "Custom OpenAI API"
    })
  })

  it("prefers a catalog nickname", () => {
    expect(
      getChatModelLabelParts({ model: { model: "gpt-4o-mini", nickname: "Fast GPT", provider: "openai" } })
    ).toEqual({ modelName: "Fast GPT", providerName: "OpenAI" })
  })

  it("keeps an Ollama tag that is not a provider prefix", () => {
    expect(getChatModelLabelParts({ modelId: "gemma3:1b" }).modelName).toBe("gemma3:1b")
  })
})

describe("formatChatModelLabel (CC-06)", () => {
  it("reads 'model (Provider)'", () => {
    expect(formatChatModelLabel({ modelId: "openai:gpt-4.1-mini" })).toBe("gpt-4.1-mini (OpenAI)")
    expect(formatChatModelLabel({ modelId: "gpt-4.1-mini", provider: "openai" })).toBe(
      "gpt-4.1-mini (OpenAI)"
    )
  })

  it("never shows a raw provider key or the tldw prefix", () => {
    const label = formatChatModelLabel({ modelId: "tldw:gemma3:1b", provider: "custom-openai-api" })
    expect(label).toBe("gemma3:1b (Custom OpenAI API)")
    expect(label).not.toMatch(/tldw:|custom[-_]openai/)
  })
})

describe("formatChatMessageAuthorName (CC-06)", () => {
  it.each([
    ["custom_openai_api:mock-gpt-4o", "mock-gpt-4o"],
    ["tldw:gemma3:1b", "gemma3:1b"],
    ["ollama:gemma3:1b", "gemma3:1b"],
    ["gemma3:1b", "gemma3:1b"],
    ["Sam", "Sam"],
    ["Mira: the Navigator", "Mira: the Navigator"]
  ])("shows %s as %s", (name, expected) => {
    expect(formatChatMessageAuthorName(name)).toBe(expected)
  })

  it("removes local-server model suffixes and account paths", () => {
    expect(
      formatChatMessageAuthorName("openrouter:accounts/acme/models/llama-v3")
    ).toBe("llama-v3")
  })
})

describe("tldw transport provider (CC-06)", () => {
  it("is not shown as a provider", () => {
    expect(formatChatModelLabel({ modelId: "tldw:gpt-4o", provider: "tldw" })).toBe("gpt-4o")
  })

  it("gives way to the provider in a qualified id", () => {
    expect(formatChatModelLabel({ modelId: "openai:gpt-4o", provider: "tldw" })).toBe(
      "gpt-4o (OpenAI)"
    )
  })
})

// Ollama families such as mistral and qwen share a name with a provider key.
// "mistral:latest" is a model, not provider "mistral" with model "latest".
describe("model ids whose family is also a provider (CC-06)", () => {
  it.each([
    ["mistral:latest", "mistral:latest"],
    ["qwen:7b", "qwen:7b"],
    ["mistral:7b-instruct-q4_K_M", "mistral:7b-instruct-q4_K_M"],
    ["mistral:instruct", "mistral:instruct"],
    ["tldw:mistral:latest", "mistral:latest"],
    ["ollama:mistral:latest", "mistral:latest"],
    ["mistral:mistral-large-latest", "mistral-large-latest"],
    ["Google: Gemini Buddy", "Google: Gemini Buddy"],
    ["Mistral:Bot", "Mistral:Bot"]
  ])("heads a message from %s as %s", (name, expected) => {
    expect(formatChatMessageAuthorName(name)).toBe(expected)
  })

  it("keeps the family in a bare model name", () => {
    expect(getChatModelLabelParts({ modelId: "mistral:latest" }).modelName).toBe("mistral:latest")
    expect(getChatModelLabelParts({ modelId: "qwen:7b" }).modelName).toBe("qwen:7b")
  })

  it("names the supplied provider, not the family", () => {
    expect(formatChatModelLabel({ modelId: "mistral:latest", provider: "ollama" })).toBe(
      "mistral:latest (Ollama)"
    )
    expect(formatChatModelLabel({ modelId: "qwen:7b", provider: "ollama" })).toBe(
      "qwen:7b (Ollama)"
    )
    expect(
      formatChatModelLabel({ model: { model: "tldw:mistral:latest", provider: "ollama" } })
    ).toBe("mistral:latest (Ollama)")
  })

  it("still drops a prefix that names the supplied provider", () => {
    expect(formatChatModelLabel({ modelId: "ollama:mistral:latest", provider: "ollama" })).toBe(
      "mistral:latest (Ollama)"
    )
    expect(
      formatChatModelLabel({ modelId: "custom-openai-api:mock-gpt-4o", provider: "custom_openai_api" })
    ).toBe("mock-gpt-4o (Custom OpenAI API)")
  })
})
