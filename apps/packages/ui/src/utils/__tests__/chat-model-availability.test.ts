import { describe, expect, it } from "vitest"
import {
  buildCharacterChatReadiness,
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  getCharacterChatReadinessCopy,
  normalizeChatModelId
} from "../chat-model-availability"

describe("chat model availability utilities", () => {
  it("normalizes prefixed model IDs", () => {
    expect(normalizeChatModelId(" tldw:gpt-4o-mini ")).toBe("gpt-4o-mini")
  })

  it("builds available IDs from model and name fields", () => {
    const ids = buildAvailableChatModelIds([
      { model: "tldw:gpt-4o-mini" },
      { name: "claude-3-5-sonnet" },
      { model: "gpt-4o-mini" }
    ])

    expect([...ids]).toEqual(["gpt-4o-mini", "claude-3-5-sonnet"])
  })

  it("includes provider-qualified IDs for cockpit model selections", () => {
    const ids = buildAvailableChatModelIds([
      {
        id: "gpt-4o-mini",
        model: "tldw:gpt-4o-mini",
        provider: "openai"
      },
      {
        id: "gpt-4o-mini",
        model: "tldw:gpt-4o-mini",
        provider: "anthropic"
      }
    ])

    expect(ids.has("gpt-4o-mini")).toBe(true)
    expect(ids.has("openai:gpt-4o-mini")).toBe(true)
    expect(ids.has("anthropic:gpt-4o-mini")).toBe(true)
    expect(findUnavailableChatModel(["openai:gpt-4o-mini"], ids)).toBeNull()
  })

  it("adds base IDs when backend descriptors expose provider-qualified model fields", () => {
    const ids = buildAvailableChatModelIds([
      {
        model: "openai:gpt-4.1-mini",
        provider: "openai",
        is_configured: true
      } as any
    ])

    expect(ids.has("openai:gpt-4.1-mini")).toBe(true)
    expect(ids.has("gpt-4.1-mini")).toBe(true)
    expect(findUnavailableChatModel(["gpt-4.1-mini"], ids)).toBeNull()
  })

  it("excludes catalog-only and unconfigured backend models from readiness availability", () => {
    const ids = buildAvailableChatModelIds([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        provider: "openai",
        is_configured: false,
        provider_is_configured: false,
        catalog_only: true
      } as any,
      {
        id: "gemma3:1b",
        model: "tldw:gemma3:1b",
        provider: "ollama",
        is_configured: true,
        provider_is_configured: true
      } as any
    ])

    expect(ids.has("gpt-4o")).toBe(false)
    expect(ids.has("openai:gpt-4o")).toBe(false)
    expect(ids.has("gemma3:1b")).toBe(true)
    expect(ids.has("ollama:gemma3:1b")).toBe(true)
  })

  it("can fail closed on descriptors without explicit configuration flags", () => {
    const ids = buildAvailableChatModelIds(
      [
        {
          id: "gpt-4o",
          model: "tldw:gpt-4o",
          provider: "openai"
        } as any,
        {
          id: "gemma3:1b",
          model: "tldw:gemma3:1b",
          provider: "ollama",
          is_configured: true,
          provider_is_configured: true
        } as any
      ],
      { requireConfiguredFlags: true }
    )

    expect(ids.has("gpt-4o")).toBe(false)
    expect(ids.has("openai:gpt-4o")).toBe(false)
    expect(ids.has("gemma3:1b")).toBe(true)
    expect(ids.has("ollama:gemma3:1b")).toBe(true)
  })

  it("accepts provider-qualified selections when the available catalog only exposes the base model ID", () => {
    const ids = buildAvailableChatModelIds([
      {
        id: "gpt-4o-mini",
        model: "tldw:gpt-4o-mini"
      }
    ])

    expect(ids.has("gpt-4o-mini")).toBe(true)
    expect(ids.has("openai:gpt-4o-mini")).toBe(false)
    expect(findUnavailableChatModel(["openai:gpt-4o-mini"], ids)).toBeNull()
  })

  it("accepts unknown provider-qualified selections when the base model is available", () => {
    const unavailable = findUnavailableChatModel(
      ["local:gpt-4o-mini"],
      new Set(["gpt-4o-mini"])
    )

    expect(unavailable).toBeNull()
  })

  it("keeps provider-qualified selections unavailable when neither qualified nor base IDs are available", () => {
    const unavailable = findUnavailableChatModel(
      ["openai:gpt-4o-mini"],
      new Set(["claude-3-5-sonnet"])
    )

    expect(unavailable).toBe("openai:gpt-4o-mini")
  })

  it("does not flag unavailable model when catalog is empty", () => {
    const unavailable = findUnavailableChatModel(["gpt-4o-mini"], new Set())
    expect(unavailable).toBeNull()
  })

  it("returns the first unavailable model ID", () => {
    const unavailable = findUnavailableChatModel(
      [" tldw:gpt-4o-mini ", "missing-model"],
      new Set(["gpt-4o-mini"])
    )
    expect(unavailable).toBe("missing-model")
  })

  it("treats auto as a valid sentinel during availability checks", () => {
    expect(normalizeChatModelId(" auto ")).toBe("auto")

    const unavailable = findUnavailableChatModel(
      ["auto"],
      new Set(["gpt-4o-mini"])
    )

    expect(unavailable).toBeNull()
  })

  it("treats mixed-case prefixed auto values as the sentinel", () => {
    const unavailable = findUnavailableChatModel(
      [" tldw:Auto "],
      new Set(["gpt-4o-mini"])
    )

    expect(unavailable).toBeNull()
  })
})

describe("character chat readiness", () => {
  const t = (
    _key: string,
    fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
  ) => {
    if (typeof fallbackOrOptions === "string") return fallbackOrOptions
    const template = fallbackOrOptions?.defaultValue || _key
    return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
      const value = fallbackOrOptions?.[token]
      return value == null ? `{{${token}}}` : String(value)
    })
  }

  it("blocks character chat before the server is connected", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: false,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "gpt-4o-mini"
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "server-connection",
      recommendedAction: "open-server-settings"
    })
  })

  it("blocks character chat until a character is selected", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: null,
        selectedModel: "gpt-4o-mini"
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "selected-character",
      recommendedAction: "choose-character"
    })
  })

  it("blocks character chat when no chat model is available", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "gpt-4o-mini",
        availableModels: []
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-model",
      reason: "no-models-available",
      recommendedAction: "open-model-settings"
    })
  })

  it("blocks stale selected models when the catalog is loaded", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "missing-model",
        availableModels: [{ model: "gpt-4o-mini", is_configured: true }]
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-model",
      reason: "selected-model-unavailable",
      recommendedAction: "open-model-settings"
    })
  })

  it("blocks selected catalog-only models from the real backend model catalog", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "tldw:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false,
            provider_is_configured: false,
            catalog_only: true
          } as any,
          {
            id: "gemma3:1b",
            model: "tldw:gemma3:1b",
            provider: "ollama",
            is_configured: true,
            provider_is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-model",
      reason: "selected-model-unavailable",
      recommendedAction: "open-model-settings"
    })
  })

  it("allows character chat when connection, character, and model are ready", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "gpt-4o-mini",
        availableModels: [{ model: "tldw:gpt-4o-mini", is_configured: true }]
      })
    ).toEqual({
      status: "ready",
      canStart: true,
      missingRequirement: null,
      recommendedAction: null,
      reason: null
    })
  })

  it("returns consistent in-context no-model copy for selected characters", () => {
    const readiness = buildCharacterChatReadiness({
      isServerConnected: true,
      selectedCharacter: { id: 1, name: "Ariadne" },
      selectedModel: null,
      availableModels: []
    })

    expect(
      getCharacterChatReadinessCopy(readiness, t, {
        characterName: "Ariadne"
      })
    ).toEqual({
      title: "Choose a chat model before chatting as Ariadne",
      description:
        "Saved characters are still available. Configure a chat model, then return here to continue with Ariadne.",
      actionLabel: "Open model settings"
    })
  })
})
