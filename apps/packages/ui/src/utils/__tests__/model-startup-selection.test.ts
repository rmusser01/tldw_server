import { describe, expect, it } from "vitest"
import { resolveStartupSelectedModel } from "../model-startup-selection"

describe("resolveStartupSelectedModel", () => {
  it("does not auto-select while current model is hydrating", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [{ model: "gpt-4o-mini" }],
        isCurrentModelHydrating: true
      })
    ).toBeNull()
  })

  it("does not auto-select while model preferences are hydrating", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [{ model: "gpt-4o-mini" }],
        preferredModelIds: ["gpt-4o-mini"],
        arePreferencesHydrating: true
      })
    ).toBeNull()
  })

  it("does not overwrite an already selected model", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: "claude-3-5-sonnet",
        models: [{ model: "gpt-4o-mini" }]
      })
    ).toBeNull()
  })

  it("prefers first available favorite model in catalog order", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [
          { model: "gpt-4o-mini" },
          { model: "claude-3-5-sonnet" },
          { model: "llama-3.1-70b" }
        ],
        preferredModelIds: ["claude-3-5-sonnet", "llama-3.1-70b"]
      })
    ).toBe("claude-3-5-sonnet")
  })

  it("falls back to first available model when no favorite matches", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [{ model: "gpt-4o-mini" }, { model: "claude-3-5-sonnet" }],
        preferredModelIds: ["missing-model"]
      })
    ).toBe("gpt-4o-mini")
  })

  it("uses the server default ahead of catalog order", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [
          { model: "tldw:first", provider: "ollama" },
          { model: "tldw:default", provider: "openai", is_server_default: true }
        ]
      })
    ).toBe("openai:default")
  })

  it("keeps available canonical favorites ahead of the server default", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [
          { model: "tldw:shared", provider: "ollama", is_server_default: true },
          { model: "tldw:shared", provider: "openai" }
        ],
        preferredModelIds: ["openai:shared"]
      })
    ).toBe("openai:shared")
  })

  it("qualifies a legacy favorite with the provider of its catalog choice", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [{ model: "tldw:shared", provider: "openai" }],
        preferredModelIds: ["tldw:shared"]
      })
    ).toBe("openai:shared")
  })

  it("uses the existing fallback when the default is absent or invalid", () => {
    expect(
      resolveStartupSelectedModel({
        currentModel: null,
        models: [
          { model: "first", provider: "ollama" },
          { model: "second", provider: "openai", is_server_default: "true" }
        ]
      })
    ).toBe("ollama:first")
  })

  it.each(["unknown", "default", "unrecognized-provider"])(
    "preserves the legacy raw fallback for placeholder provider %s",
    (provider) => {
      expect(
        resolveStartupSelectedModel({
          currentModel: null,
          models: [{ model: "tldw:legacy", provider }]
        })
      ).toBe("tldw:legacy")
    }
  )
})
