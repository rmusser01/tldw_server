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
        availableModels: [{ model: "gpt-4o-mini" }]
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
        availableModels: [{ model: "tldw:gpt-4o-mini" }]
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
