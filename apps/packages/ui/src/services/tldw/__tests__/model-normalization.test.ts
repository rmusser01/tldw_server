import { describe, expect, it, vi } from "vitest"
import {
  getNormalizedTldwModels,
  normalizeTldwModels
} from "../model-normalization"

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

  it("treats array-valued modalities as invalid metadata records", () => {
    const models = normalizeTldwModels({
      models: [
        {
          id: "openai/gpt-4o-mini",
          provider: "openai",
          type: "chat",
          modalities: ["text"],
          input_modality: ["text"],
          output_modality: "text"
        }
      ]
    })

    expect(models[0]?.modalities).toEqual({
      input: ["text"],
      output: ["text"]
    })
  })

  it("derives capability flags from array-shaped capability metadata", () => {
    const models = normalizeTldwModels({
      models: [
        {
          id: "openai/gpt-4o-mini",
          provider: "openai",
          capabilities: ["vision", "tool_use", "json_mode"]
        },
        {
          id: "anthropic/claude-sonnet-4",
          provider: "anthropic",
          features: ["function_calling", "json_output"]
        }
      ]
    })

    expect(models[0]).toEqual(
      expect.objectContaining({
        vision: true,
        function_calling: true,
        json_output: true
      })
    )
    expect(models[1]).toEqual(
      expect.objectContaining({
        vision: false,
        function_calling: true,
        json_output: true
      })
    )
  })

  it("preserves explicit false capability flags over alternate capability keys", () => {
    const models = normalizeTldwModels({
      models: [
        {
          id: "local/tool-model",
          provider: "local",
          capabilities: {
            function_calling: false,
            tool_use: true
          }
        }
      ]
    })

    expect(models[0]).toEqual(
      expect.objectContaining({
        function_calling: false
      })
    )
  })
})
