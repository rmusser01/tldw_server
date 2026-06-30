import { describe, expect, it } from "vitest"

import {
  filterModelsForScope,
  getCanonicalModelKey,
  getModelId,
  getModelProvider,
  isConfiguredUsableModel,
  sortModelsForSelector
} from "../modelSelectorUtils"

const configuredModel = (provider: string, model: string, extra: Record<string, unknown> = {}) => ({
  provider,
  model,
  nickname: `${provider} ${model}`,
  is_configured: true,
  ...extra
})

describe("model selector utilities", () => {
  it("uses provider:model keys so duplicate model ids remain distinct", () => {
    expect(getCanonicalModelKey(configuredModel("openai", "gpt-4o"))).toBe("openai:gpt-4o")
    expect(getCanonicalModelKey(configuredModel("anthropic", "gpt-4o"))).toBe("anthropic:gpt-4o")
  })

  it("normalizes internal tldw transport ids out of provider:model keys", () => {
    const model = {
      id: "gpt-4o-mini",
      model: "tldw:gpt-4o-mini",
      name: "GPT-4o mini",
      provider: "openai",
      is_configured: true
    }

    expect(getModelId(model)).toBe("gpt-4o-mini")
    expect(getCanonicalModelKey(model)).toBe("openai:gpt-4o-mini")
  })

  it("treats legacy tldw-prefixed favorite keys as favorites after provider-key migration", () => {
    const models = [
      configuredModel("anthropic", "claude-3-5-sonnet"),
      configuredModel("openai", "gpt-4o")
    ]

    expect(
      sortModelsForSelector(models, {
        favoriteKeys: new Set(["tldw:gpt-4o"]),
        sortMode: "provider"
      }).map(getCanonicalModelKey)
    ).toEqual(["openai:gpt-4o", "anthropic:claude-3-5-sonnet"])
  })

  it("normalizes provider and model identifiers from common server payload shapes", () => {
    const model = {
      id: "claude-3-5-haiku",
      provider_key: " Anthropic ",
      details: {
        provider: "ignored"
      }
    }

    expect(getModelId(model)).toBe("claude-3-5-haiku")
    expect(getModelProvider(model)).toBe("anthropic")
    expect(getCanonicalModelKey(model)).toBe("anthropic:claude-3-5-haiku")
  })

  it("excludes catalog-only, unconfigured, and unusable models from configured scope", () => {
    const models = [
      configuredModel("openai", "gpt-4o"),
      configuredModel("openrouter", "catalog-only", { catalog_only: true }),
      configuredModel("anthropic", "unconfigured", { is_configured: false }),
      configuredModel("google", "provider-unconfigured", {
        provider_is_configured: false
      }),
      configuredModel("ollama", "disabled", { enabled: false }),
      configuredModel("vllm", "not-usable", { usable: false })
    ]

    expect(models.map(isConfiguredUsableModel)).toEqual([
      true,
      false,
      false,
      false,
      false,
      false
    ])
    expect(filterModelsForScope(models, "configured").map(getCanonicalModelKey)).toEqual([
      "openai:gpt-4o"
    ])
  })

  it("keeps catalog-only models available only in explicit catalog scope", () => {
    const models = [
      configuredModel("openai", "gpt-4o"),
      configuredModel("openrouter", "known-catalog-model", { catalog_only: true })
    ]

    expect(filterModelsForScope(models, "configured").map(getModelId)).toEqual(["gpt-4o"])
    expect(filterModelsForScope(models, "catalog").map(getModelId)).toEqual([
      "gpt-4o",
      "known-catalog-model"
    ])
  })

  it("ranks current model first, then recent and frequent usage before provider grouping", () => {
    const models = [
      configuredModel("zai", "z-model"),
      configuredModel("openai", "current"),
      configuredModel("anthropic", "frequent", { nickname: "Frequent" }),
      configuredModel("google", "recent", { nickname: "Recent" }),
      configuredModel("bedrock", "unused")
    ]

    const sorted = sortModelsForSelector(models, {
      selectedModel: "current",
      selectedProvider: "openai",
      favoriteKeys: new Set(),
      usageByKey: {
        "anthropic:frequent": { selectedCount: 4, lastSelectedAt: 200 },
        "google:recent": { selectedCount: 1, lastSelectedAt: 900 }
      },
      sortMode: "provider"
    })

    expect(sorted.map(getCanonicalModelKey)).toEqual([
      "openai:current",
      "anthropic:frequent",
      "google:recent",
      "bedrock:unused",
      "zai:z-model"
    ])
  })
})
