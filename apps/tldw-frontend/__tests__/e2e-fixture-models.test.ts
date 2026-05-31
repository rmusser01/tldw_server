import { describe, expect, it } from "vitest"

import { extractModelIds, extractUsableModelIds } from "../e2e/utils/fixtures"

describe("e2e model preflight helpers", () => {
  it("filters metadata to configured chat text models", () => {
    expect(
      extractUsableModelIds({
        models: [
          {
            provider: "openai",
            id: "gpt-4o",
            is_configured: false,
            provider_is_configured: false,
            catalog_only: true,
            type: "chat",
            output_modality: "text",
          },
          {
            provider: "ollama",
            id: "gemma3:1b",
            is_configured: true,
            provider_is_configured: true,
            catalog_only: false,
            type: "chat",
            output_modality: "text",
          },
          {
            provider: "openai",
            id: "dall-e-3",
            is_configured: true,
            provider_is_configured: true,
            catalog_only: false,
            type: "image",
            output_modality: "image",
          },
          {
            provider: "mlx",
            id: "deprecated-local",
            is_configured: true,
            provider_is_configured: true,
            type: "chat",
            output_modality: "text",
            deprecated: true,
          },
          "bare-chat-model",
        ],
      })
    ).toEqual(["ollama:gemma3:1b", "bare-chat-model"])
  })

  it("preserves already provider-prefixed metadata IDs", () => {
    expect(
      extractUsableModelIds({
        models: [
          {
            provider: "ollama",
            id: "ollama:gemma3:1b",
            is_configured: true,
            provider_is_configured: true,
            type: "chat",
            output_modality: ["text"],
          },
        ],
      })
    ).toEqual(["ollama:gemma3:1b"])
  })

  it("filters and normalizes provider fallback entries", () => {
    expect(
      extractModelIds({
        providers: [
          {
            name: "openai",
            is_configured: false,
            models: ["gpt-4o"],
          },
          {
            name: "custom",
            is_configured: true,
            models: [
              { id: "configured-model", is_configured: true, type: "chat" },
              { id: "custom:configured-model", is_configured: true, type: "chat" },
              { id: "image-only", is_configured: true, type: "image" },
              { id: "catalog-only-model", catalog_only: true },
            ],
          },
        ],
      })
    ).toEqual(["custom:configured-model"])
  })
})
