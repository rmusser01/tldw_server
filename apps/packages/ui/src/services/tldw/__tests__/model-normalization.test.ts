import { describe, expect, it, vi } from "vitest"
import { getNormalizedTldwModels } from "../model-normalization"

describe("model normalization helpers", () => {
  it("normalizes models with inherited provider availability metadata", async () => {
    const options = { refreshOpenRouter: true }
    const client = {
      getModelsMetadata: vi.fn(async () => ({
        models: [
          {
            id: "openai/gpt-4o-mini",
            name: "GPT-4o Mini",
            provider: "OpenAI",
            type: "chat"
          },
          {
            id: "anthropic/claude-sonnet-4",
            provider: "anthropic",
            type: "chat",
            provider_enabled: true
          }
        ]
      })),
      getProviders: vi.fn(async () => ({
        providers: [
          {
            name: "openai",
            configured: true,
            enabled: false,
            availability: "needs-key"
          },
          {
            name: "anthropic",
            is_configured: false,
            enabled: false,
            availability: "disabled"
          }
        ]
      }))
    }

    const models = await getNormalizedTldwModels(client, options)

    expect(client.getModelsMetadata).toHaveBeenCalledWith(options)
    expect(client.getProviders).toHaveBeenCalledOnce()
    expect(models).toEqual([
      expect.objectContaining({
        id: "openai/gpt-4o-mini",
        name: "GPT-4o Mini (openai/gpt-4o-mini)",
        provider: "OpenAI",
        is_configured: true,
        provider_enabled: false,
        availability: "needs-key"
      }),
      expect.objectContaining({
        id: "anthropic/claude-sonnet-4",
        provider: "anthropic",
        is_configured: false,
        provider_enabled: true,
        availability: "disabled"
      })
    ])
  })
})
