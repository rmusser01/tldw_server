import { describe, expect, it } from "vitest"

import { resolveSamplingSupport } from "../chat-sampling-support"

const models = [
  { model: "tldw:or-model", provider: "openrouter", sampling_fields: ["top_k", "min_p"] },
  { model: "tldw:gpt-4o", provider: "openai", sampling_fields: [] },
  { model: "tldw:mistral-large", provider: "mistral", sampling_fields: ["top_k"] },
  { model: "tldw:old-model", provider: "ollama" }
]

// #3283: Top K and Min P are offered only where the provider accepts them.
describe("resolveSamplingSupport", () => {
  it("reads what the selected model's provider accepts", () => {
    expect(resolveSamplingSupport(models, "openrouter:or-model")).toEqual({
      topK: true, minP: true, providerName: "OpenRouter"
    })
    expect(resolveSamplingSupport(models, "openai:gpt-4o")).toEqual({
      topK: false, minP: false, providerName: "OpenAI"
    })
    expect(resolveSamplingSupport(models, "mistral:mistral-large")).toMatchObject({ topK: true, minP: false })
  })

  it("finds the model by its id when the selection has no provider", () => {
    expect(resolveSamplingSupport(models, "tldw:gpt-4o")).toMatchObject({ topK: false, minP: false })
  })

  it("says unknown when the server doesn't list it or the model isn't in the catalog", () => {
    expect(resolveSamplingSupport(models, "ollama:old-model")).toMatchObject({ topK: null, minP: null })
    expect(resolveSamplingSupport(models, "openai:not-listed")).toMatchObject({ topK: null, minP: null })
    expect(resolveSamplingSupport(undefined, "openai:gpt-4o")).toMatchObject({ topK: null, minP: null })
    expect(resolveSamplingSupport(models, null)).toMatchObject({ topK: null, minP: null })
  })
})
