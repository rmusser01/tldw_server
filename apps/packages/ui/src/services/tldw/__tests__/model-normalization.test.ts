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

  it("CC-01: carries the provider's observed health onto its models", async () => {
    const client = {
      getModelsMetadata: vi.fn(async () => ({
        models: [
          { id: "gemma3:1b", provider: "ollama", type: "chat" },
          { id: "mock-gpt-4o", provider: "custom_openai_api", type: "chat" },
          { id: "gpt-4o", provider: "openai", type: "chat" }
        ]
      })),
      getProviders: vi.fn(async () => ({
        providers: [
          {
            name: "ollama",
            health: { status: "unhealthy", success_count: 0, consecutive_failures: 4 }
          },
          {
            name: "custom-openai-api",
            health: { status: "healthy", success_count: 2, consecutive_failures: 0 }
          },
          { name: "openai", health: { status: "unknown", success_count: 0 } }
        ]
      }))
    }

    const models = await getNormalizedTldwModels(client)

    expect(models.map((model) => [model.id, model.provider_health])).toEqual([
      ["gemma3:1b", "unreachable"],
      ["mock-gpt-4o", "working"],
      ["gpt-4o", "unknown"]
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

  it("retains the exact provider default even when model availability is complete", async () => {
    const client = {
      getModelsMetadata: async () => ({
        models: [
          {
            id: "shared",
            provider: "ollama",
            type: "chat",
            is_configured: true,
            provider_enabled: true,
            availability: "available"
          },
          {
            id: "shared",
            provider: "custom_openai_api",
            type: "chat",
            is_configured: true,
            provider_enabled: true,
            availability: "available"
          },
          {
            id: "other",
            provider: "custom_openai_api",
            type: "chat",
            is_configured: true,
            provider_enabled: true,
            availability: "available"
          }
        ]
      }),
      getProviders: async () => ({
        default_provider: "custom-openai-api",
        providers: [
          { name: "ollama", default_model: "shared" },
          { name: "custom_openai_api", default_model: "shared" }
        ]
      })
    }
    const models = await getNormalizedTldwModels(client)
    expect(models.map((model) => model.is_server_default)).toEqual([
      false,
      true,
      false
    ])
  })

  it.each(
    [undefined, null, "missing", ["shared"], { id: "shared" }].map(
      (defaultModel) => ({ defaultModel })
    )
  )(
    "does not fabricate a default for malformed or unavailable model %j",
    async ({ defaultModel }) => {
      const models = await getNormalizedTldwModels({
        getModelsMetadata: async () => [
          {
            id: "shared",
            provider: "openai",
            type: "chat",
            is_server_default: true
          }
        ],
        getProviders: async () => ({
          default_provider: "openai",
          providers: [{ name: "openai", default_model: defaultModel }]
        })
      })
      expect(models[0].is_server_default).toBe(false)
    }
  )

  it("does not accept a metadata default assertion after provider discovery fails", async () => {
    const models = await getNormalizedTldwModels({
      getModelsMetadata: async () => [
        {
          id: "shared",
          provider: "openai",
          type: "chat",
          is_server_default: true
        }
      ],
      getProviders: async () => {
        throw new Error("Unavailable provider defaults")
      }
    })
    expect(models[0].is_server_default).toBe(false)
  })

  it("supports the deployed name-only API model identity", async () => {
    const models = await getNormalizedTldwModels({
      getModelsMetadata: async () => [
        { name: "Vendor/Model", provider: "openai", type: "chat" }
      ],
      getProviders: async () => ({
        default_provider: "openai",
        providers: [{ name: "openai", default_model: "Vendor/Model" }]
      })
    })
    expect(models[0]).toMatchObject({
      id: "Vendor/Model",
      is_server_default: true
    })
  })

  it("does not derive a server default from a friendly display label", async () => {
    const models = await getNormalizedTldwModels({
      getModelsMetadata: async () => [
        {
          id: "different",
          name: "Friendly default label",
          provider: "openai",
          type: "chat"
        }
      ],
      getProviders: async () => ({
        default_provider: "openai",
        providers: [{ name: "openai", default_model: "Vendor/Model" }]
      })
    })
    expect(models[0].is_server_default).toBe(false)
  })

  it("matches a numbered provider's configuration alias to its canonical metadata identity", async () => {
    const models = await getNormalizedTldwModels({
      getModelsMetadata: async () => [
        { name: "Vendor/Model", provider: "custom-openai-api-3", type: "chat" }
      ],
      getProviders: async () => ({
        default_provider: "custom_openai_api_3",
        providers: [
          { name: "custom-openai-api-3", default_model: "Vendor/Model" }
        ]
      })
    })
    expect(models[0].is_server_default).toBe(true)
  })
})
