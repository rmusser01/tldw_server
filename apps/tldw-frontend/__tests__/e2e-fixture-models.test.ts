import { describe, expect, it } from "vitest"

import { extractModelIds, extractUsableModelIds } from "../e2e/utils/fixtures"

describe("e2e model preflight helpers", () => {
  it("filters metadata to configured non-catalog models", () => {
    expect(
      extractUsableModelIds({
        models: [
          {
            provider: "openai",
            id: "gpt-4o",
            is_configured: false,
            provider_is_configured: false,
            catalog_only: true,
          },
          {
            provider: "ollama",
            id: "gemma3:1b",
            is_configured: true,
            provider_is_configured: true,
            catalog_only: false,
          },
          {
            provider: "mlx",
            id: "deprecated-local",
            is_configured: true,
            provider_is_configured: true,
            deprecated: true,
          },
        ],
      })
    ).toEqual(["ollama:gemma3:1b"])
  })

  it("filters provider fallback entries with explicit unconfigured flags", () => {
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
              { id: "configured-model", is_configured: true },
              { id: "catalog-only-model", catalog_only: true },
            ],
          },
        ],
      })
    ).toEqual(["configured-model"])
  })
})
