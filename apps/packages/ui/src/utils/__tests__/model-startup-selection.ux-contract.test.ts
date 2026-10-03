/**
 * UX review 2026-10 contract reproduction CC-01 (#3111).
 *
 * The `it.fails` test asserts the CORRECT behaviour and passes only while the
 * defect exists; when the fix lands, convert it to a plain `it(...)` in the
 * same change.
 */
import { describe, expect, it } from "vitest"
import { resolveStartupSelectedModel } from "../model-startup-selection"

/**
 * GET /api/v1/llm/providers as served with `default_api = custom-openai-api`
 * (endpoints/llm_providers.py:2156). The default is reported in config spelling
 * (hyphens) while provider names use underscores.
 */
const providersPayload = {
  default_provider: "custom-openai-api",
  providers: [
    { name: "ollama", models: ["llama3.2"] },
    { name: "custom_openai_api", models: ["qwen3-8b"] }
  ]
}

/** Composer catalogue entries as built by mapTldwModelToUi (services/tldw-server.ts:96-100). */
const composerModels = providersPayload.providers.flatMap((provider) =>
  provider.models.map((model) => ({
    model: `tldw:${provider.name}/${model}`,
    provider: provider.name
  }))
)

describe("resolveStartupSelectedModel UX contract reproductions (#3111)", () => {
  // CC-01 (#3111): utils/model-startup-selection.ts:19-61 has no server-default input and returns the first catalogue model (applied at PlaygroundForm.tsx:1530-1540).
  it.fails("CC-01 (#3111): first-run selection picks the server default_provider's model", () => {
    // `serverDefaultProvider` is the input the review's recommended fix adds; it is
    // passed through a widened object so this compiles against today's signature.
    const startupInputs = {
      currentModel: null,
      models: composerModels,
      preferredModelIds: [],
      serverDefaultProvider: providersPayload.default_provider
    }

    expect(resolveStartupSelectedModel(startupInputs)).toBe("tldw:custom_openai_api/qwen3-8b")
  })
})
