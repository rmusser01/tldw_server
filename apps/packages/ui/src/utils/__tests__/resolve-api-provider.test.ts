import { beforeEach, describe, expect, it, vi } from "vitest"
import { tldwModels } from "@/services/tldw"
import { inferProviderFromModel } from "@/utils/provider-registry"
import {
  isAutoModelId,
  parseProviderQualifiedModelSelection,
  resolveApiProviderForModel,
  resolveExplicitProviderForSelectedModel
} from "../resolve-api-provider"

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getModels: vi.fn()
  }
}))

vi.mock("@/utils/provider-registry", () => ({
  inferProviderFromModel: vi.fn()
}))

describe("resolveApiProviderForModel", () => {
  const getModelsMock = vi.mocked(tldwModels.getModels)
  const inferProviderFromModelMock = vi.mocked(inferProviderFromModel)

  beforeEach(() => {
    getModelsMock.mockReset()
    getModelsMock.mockResolvedValue([])
    inferProviderFromModelMock.mockReset()
    inferProviderFromModelMock.mockReturnValue(null)
  })

  it("uses the explicit provider as the primary selection", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "tldw:moonshot-v1",
        explicitProvider: " OpenAI ",
        providerHint: "moonshot"
      })
    ).resolves.toBe("openai")
  })

  it("uses official provider metadata when model exists in the server catalog", async () => {
    getModelsMock.mockResolvedValue([
      {
        id: "deepseek-chat",
        name: "DeepSeek Chat",
        provider: "DeepSeek",
        type: "chat"
      }
    ] as any)

    await expect(
      resolveApiProviderForModel({
        modelId: "tldw:deepseek-chat"
      })
    ).resolves.toBe("deepseek")
  })

  it("falls back to model-prefix inference for stale model ids", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "deepseek-chat"
      })
    ).resolves.toBe("deepseek")
  })

  it("prefers server catalog provider for tldw namespaced models", async () => {
    getModelsMock.mockResolvedValue([
      {
        id: "anthropic/claude-4.5-sonnet",
        name: "anthropic/claude-4.5-sonnet",
        provider: "openrouter",
        type: "chat"
      }
    ] as any)

    await expect(
      resolveApiProviderForModel({
        modelId: "tldw:anthropic/claude-4.5-sonnet"
      })
    ).resolves.toBe("openrouter")
  })

  it("keeps inline provider inference for non-tldw namespaced model ids", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "anthropic/claude-4.5-sonnet"
      })
    ).resolves.toBe("anthropic")
  })

  it("returns undefined when the provider cannot be inferred", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "custom-random-model-123"
      })
    ).resolves.toBeUndefined()
  })

  it("does not apply provider hints to auto model selections", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "auto",
        providerHint: "openai"
      })
    ).resolves.toBeUndefined()
  })

  it("normalizes prefixed mixed-case auto model selections", () => {
    expect(isAutoModelId(" tldw:Auto ")).toBe(true)
    expect(isAutoModelId("gpt-4.1")).toBe(false)
  })

  it("uses provider-qualified model selections before stale explicit providers", async () => {
    await expect(
      resolveApiProviderForModel({
        modelId: "anthropic:claude-3-5-sonnet",
        explicitProvider: "openai"
      })
    ).resolves.toBe("anthropic")
  })
})

describe("parseProviderQualifiedModelSelection", () => {
  it("splits known provider:model selections without changing model ids", () => {
    expect(parseProviderQualifiedModelSelection("OpenAI:gpt-4o")).toEqual({
      raw: "OpenAI:gpt-4o",
      modelId: "gpt-4o",
      provider: "openai",
      isProviderQualified: true
    })
    expect(parseProviderQualifiedModelSelection("anthropic:claude:3")).toEqual({
      raw: "anthropic:claude:3",
      modelId: "claude:3",
      provider: "anthropic",
      isProviderQualified: true
    })
  })

  it("splits known provider:model selections after the internal tldw namespace", () => {
    expect(parseProviderQualifiedModelSelection("tldw:openai:gpt-4o")).toEqual({
      raw: "tldw:openai:gpt-4o",
      modelId: "gpt-4o",
      provider: "openai",
      isProviderQualified: true
    })
  })

  it("leaves unknown colon-delimited model ids intact", () => {
    expect(parseProviderQualifiedModelSelection("family:model")).toEqual({
      raw: "family:model",
      modelId: "family:model",
      provider: undefined,
      isProviderQualified: false
    })
  })
})

describe("resolveExplicitProviderForSelectedModel", () => {
  it("keeps the explicit provider when there is no selected-model override", () => {
    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "tldw:anthropic/claude-4.5-sonnet",
        requestedSelectedModel: undefined,
        explicitProvider: "openrouter"
      })
    ).toBe("openrouter")
  })

  it("keeps the explicit provider when the override matches the current selected model", () => {
    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "tldw:anthropic/claude-4.5-sonnet",
        requestedSelectedModel: "anthropic/claude-4.5-sonnet",
        explicitProvider: "openrouter"
      })
    ).toBe("openrouter")
  })

  it("drops the explicit provider when the override switches to a different selected model", () => {
    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "tldw:anthropic/claude-4.5-sonnet",
        requestedSelectedModel: "tldw:deepseek-chat",
        explicitProvider: "openrouter"
      })
    ).toBeUndefined()
  })

  it("prefers the provider from provider-qualified selected models", () => {
    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "openai:gpt-4o",
        requestedSelectedModel: "gpt-4o",
        explicitProvider: "anthropic"
      })
    ).toBe("openai")

    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "openai:gpt-4o",
        requestedSelectedModel: "anthropic:gpt-4o",
        explicitProvider: "openai"
      })
    ).toBe("anthropic")
  })

  it("keeps the current provider-qualified provider when a tldw override matches the same model", () => {
    expect(
      resolveExplicitProviderForSelectedModel({
        currentSelectedModel: "openrouter:anthropic/claude-3.5-sonnet",
        requestedSelectedModel: "tldw:anthropic/claude-3.5-sonnet",
        explicitProvider: undefined
      })
    ).toBe("openrouter")
  })
})
