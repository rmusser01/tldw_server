import { describe, expect, it } from "vitest"
import {
  buildCharacterChatReadiness,
  buildAvailableChatModelIds,
  buildChatModelUsability,
  findUnavailableChatModel,
  getCharacterChatReadinessCopy,
  getMatchingCharacterChatModelUsabilityCopy,
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

  it("does not count catalog-only false as a configured model flag", () => {
    const ids = buildAvailableChatModelIds(
      [
        {
          id: "gpt-4o",
          model: "tldw:gpt-4o",
          provider: "openai",
          catalog_only: false
        } as any
      ],
      { requireConfiguredFlags: true }
    )

    expect(ids.has("gpt-4o")).toBe(false)
    expect(ids.has("openai:gpt-4o")).toBe(false)
  })

  it("treats any catalog-only true flag as unavailable", () => {
    const ids = buildAvailableChatModelIds([
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        provider: "openai",
        catalog_only: false,
        is_configured: true,
        details: {
          catalog_only: true
        }
      } as any
    ])

    expect(ids.has("gpt-4o")).toBe(false)
    expect(ids.has("openai:gpt-4o")).toBe(false)
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

describe("chat model usability", () => {
  it("reports loading while model catalog data is hydrating", () => {
    expect(
      buildChatModelUsability({
        isServerConnected: true,
        selectedModel: "gpt-4o",
        availableModels: null,
        modelsLoading: true
      })
    ).toMatchObject({
      status: "loading",
      canSend: false,
      selectedModelId: "gpt-4o",
      recommendedAction: "retry"
    })
  })

  it("reports no_server before evaluating model availability", () => {
    expect(
      buildChatModelUsability({
        isServerConnected: false,
        selectedModel: "gpt-4o",
        availableModels: [{ model: "gpt-4o", is_configured: true }]
      })
    ).toMatchObject({
      status: "no_server",
      canSend: false,
      selectedModelId: "gpt-4o",
      recommendedAction: "open-server-settings"
    })
  })

  it("reports no_selection for null and blank selected models", () => {
    expect(
      buildChatModelUsability({
        selectedModel: null,
        availableModels: [{ model: "gpt-4o", is_configured: true }]
      })
    ).toMatchObject({
      status: "no_selection",
      canSend: false,
      selectedModelId: null,
      recommendedAction: "open-model-settings"
    })

    expect(
      buildChatModelUsability({
        selectedModel: "   ",
        availableModels: [{ model: "gpt-4o", is_configured: true }]
      })
    ).toMatchObject({
      status: "no_selection",
      canSend: false,
      selectedModelId: null
    })
  })

  it("reports no_models when no callable model exists and the selected model has no specific descriptor", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gpt-4o",
        availableModels: [
          {
            id: "claude-3-5-sonnet",
            model: "tldw:claude-3-5-sonnet",
            provider: "anthropic",
            provider_is_configured: false
          } as any
        ]
      })
    ).toMatchObject({
      status: "no_models",
      canSend: false,
      matchedModelId: null,
      recommendedAction: "open-model-settings"
    })
  })

  it("reports selected_missing when callable models exist but the selected model is absent", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "missing-model",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "selected_missing",
      canSend: false,
      matchedModelId: null,
      recommendedAction: "open-model-settings"
    })
  })

  it("reports provider_unconfigured for a known selected model whose provider flags are false", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false,
            provider_is_configured: false,
            catalog_only: false
          } as any
        ]
      })
    ).toMatchObject({
      status: "provider_unconfigured",
      canSend: false,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai",
      recommendedAction: "open-model-settings"
    })
  })

  it("reports model_unavailable for a selected catalog-only model", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true,
            provider_is_configured: true,
            catalog_only: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "model_unavailable",
      canSend: false,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai",
      recommendedAction: "open-model-settings"
    })
  })

  it("reports ready for callable base and provider-qualified selections", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      selectedModelId: "gpt-4o",
      providerQualifiedModelId: null,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai",
      recommendedAction: null
    })

    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      selectedModelId: "openai:gpt-4o",
      providerQualifiedModelId: "openai:gpt-4o",
      matchedModelId: "gpt-4o",
      matchedProvider: "openai"
    })
  })

  it("prefers a callable duplicate base descriptor over an earlier unusable descriptor", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false
          } as any,
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "ollama",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gpt-4o",
      matchedProvider: "ollama"
    })
  })

  it("prefers a callable same-provider duplicate for provider-qualified selections", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false
          } as any,
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai"
    })
  })

  it("matches colon-tagged local model IDs without treating the family as a provider", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gemma3:1b",
        availableModels: [
          {
            id: "gemma3:1b",
            model: "tldw:gemma3:1b",
            provider: "ollama",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gemma3:1b",
      matchedProvider: "ollama"
    })

    expect(
      buildChatModelUsability({
        selectedModel: "ollama:gemma3:1b",
        availableModels: [
          {
            id: "gemma3:1b",
            model: "tldw:gemma3:1b",
            provider: "ollama",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gemma3:1b",
      matchedProvider: "ollama"
    })
  })

  it("does not strip an unknown colon prefix from local model tags", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gemma3:1b",
        availableModels: [
          {
            id: "1b",
            model: "tldw:1b",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "selected_missing",
      canSend: false,
      matchedModelId: null
    })
  })

  it("matches backend provider aliases against canonical provider-qualified selections", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "custom-openai-api:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "custom_openai_api",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gpt-4o",
      matchedProvider: "custom-openai-api"
    })
  })

  it("matches selected provider aliases without treating local model tags as providers", () => {
    const cases = [
      {
        selectedModel: "customopenai:gpt-4o",
        descriptorProvider: "custom_openai_api",
        expectedProvider: "custom-openai-api"
      },
      {
        selectedModel: "localllm:mistral-7b",
        descriptorProvider: "local_llm",
        expectedProvider: "local-llm"
      },
      {
        selectedModel: "llama_cpp:llama3.1:8b",
        descriptorProvider: "llama-cpp",
        expectedProvider: "llama.cpp"
      }
    ]

    for (const testCase of cases) {
      expect(
        buildChatModelUsability({
          selectedModel: testCase.selectedModel,
          availableModels: [
            {
              id: testCase.selectedModel.slice(
                testCase.selectedModel.indexOf(":") + 1
              ),
              provider: testCase.descriptorProvider,
              is_configured: true
            } as any
          ]
        })
      ).toMatchObject({
        status: "ready",
        canSend: true,
        matchedProvider: testCase.expectedProvider
      })
    }
  })

  it("allows unknown provider-qualified selections to match an unqualified base descriptor only when no provider-specific descriptor conflicts", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "local:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      matchedModelId: "gpt-4o",
      matchedProvider: null
    })

    expect(
      buildChatModelUsability({
        selectedModel: "local:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "selected_missing",
      canSend: false,
      matchedModelId: null
    })
  })

  it("preserves auto model sentinel readiness semantics", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "auto",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "ready",
      canSend: true,
      selectedModelId: "auto",
      matchedModelId: null
    })

    expect(
      buildChatModelUsability({
        selectedModel: "auto",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false
          } as any
        ]
      })
    ).toMatchObject({
      status: "no_models",
      canSend: false
    })
  })

  it("applies degraded send policy to auto model selection", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "auto",
        serverDegraded: true,
        allowDegradedSend: true,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "degraded",
      canSend: true,
      selectedModelId: "auto",
      detailReason: "server-degraded"
    })

    expect(
      buildChatModelUsability({
        selectedModel: "auto",
        serverDegraded: true,
        allowDegradedSend: false,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "degraded",
      canSend: false,
      selectedModelId: "auto",
      recommendedAction: "retry",
      detailReason: "server-degraded"
    })
  })

  it("only allows degraded sends when explicitly configured", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "gpt-4o",
        serverDegraded: true,
        allowDegradedSend: true,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "degraded",
      canSend: true,
      detailReason: "server-degraded"
    })

    expect(
      buildChatModelUsability({
        selectedModel: "gpt-4o",
        serverDegraded: true,
        allowDegradedSend: false,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "degraded",
      canSend: false,
      matchedModelId: "gpt-4o",
      recommendedAction: "retry",
      detailReason: "server-degraded"
    })
  })

  it("keeps model-specific blockers ahead of degraded server policy", () => {
    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        serverDegraded: true,
        allowDegradedSend: false,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false
          } as any
        ]
      })
    ).toMatchObject({
      status: "provider_unconfigured",
      canSend: false,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai",
      detailReason: "provider-unconfigured"
    })

    expect(
      buildChatModelUsability({
        selectedModel: "openai:gpt-4o",
        serverDegraded: true,
        allowDegradedSend: true,
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: true,
            catalog_only: true
          } as any
        ]
      })
    ).toMatchObject({
      status: "model_unavailable",
      canSend: false,
      matchedModelId: "gpt-4o",
      matchedProvider: "openai",
      detailReason: "catalog-only"
    })
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

  it("does not reuse character-selection copy as model usability detail", () => {
    const readiness = buildCharacterChatReadiness({
      isServerConnected: true,
      selectedCharacter: null,
      selectedModel: null,
      availableModels: null,
      modelsLoading: true
    })
    const readinessCopy = getCharacterChatReadinessCopy(readiness, t)
    const modelUsability = buildChatModelUsability({
      isServerConnected: true,
      selectedModel: null,
      availableModels: null,
      modelsLoading: true
    })

    expect(readiness).toMatchObject({
      missingRequirement: "selected-character"
    })
    expect(modelUsability).toMatchObject({
      status: "no_selection",
      canSend: false
    })
    expect(
      getMatchingCharacterChatModelUsabilityCopy({
        modelUsability,
        readiness,
        readinessTitle: readinessCopy.title
      })
    ).toBeNull()
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

  it("blocks character chat while model readiness is loading", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "gpt-4o-mini",
        availableModels: null,
        modelsLoading: true
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-model",
      reason: "models-loading",
      recommendedAction: "retry"
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
      reason: "selected-model-missing",
      recommendedAction: "open-model-settings"
    })
  })

  it("blocks provider-unconfigured selected models with model settings recovery", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "openai:gpt-4o",
        availableModels: [
          {
            id: "gpt-4o",
            model: "tldw:gpt-4o",
            provider: "openai",
            is_configured: false,
            provider_is_configured: false
          } as any
        ]
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-model",
      reason: "provider-unconfigured",
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
            is_configured: true,
            provider_is_configured: true,
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
      reason: "model-unavailable",
      recommendedAction: "open-model-settings"
    })
  })

  it("allows send-disabled to block only after model usability is ready", () => {
    expect(
      buildCharacterChatReadiness({
        isServerConnected: true,
        selectedCharacter: { id: 1, name: "Ariadne" },
        selectedModel: "gpt-4o-mini",
        availableModels: [{ model: "tldw:gpt-4o-mini", is_configured: true }],
        isSendBlocked: true
      })
    ).toMatchObject({
      status: "blocked",
      missingRequirement: "chat-send",
      reason: "send-disabled",
      recommendedAction: "retry"
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

  it("returns precise in-context copy for model readiness blockers", () => {
    const cases = [
      {
        readiness: buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: 1, name: "Ariadne" },
          selectedModel: "gpt-4o-mini",
          availableModels: null,
          modelsLoading: true
        }),
        title: "Checking chat model readiness",
        actionLabel: "Try again"
      },
      {
        readiness: buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: 1, name: "Ariadne" },
          selectedModel: "missing-model",
          availableModels: [{ model: "tldw:gpt-4o-mini", is_configured: true }]
        }),
        title: "Choose an available chat model before chatting as Ariadne",
        actionLabel: "Open model settings"
      },
      {
        readiness: buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: 1, name: "Ariadne" },
          selectedModel: "gpt-4o",
          availableModels: []
        }),
        title: "Configure a chat model before chatting as Ariadne",
        actionLabel: "Open model settings"
      },
      {
        readiness: buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: 1, name: "Ariadne" },
          selectedModel: "openai:gpt-4o",
          availableModels: [
            {
              id: "gpt-4o",
              model: "tldw:gpt-4o",
              provider: "openai",
              is_configured: false
            } as any
          ]
        }),
        title:
          "Configure the selected model provider before chatting as Ariadne",
        actionLabel: "Open model settings"
      },
      {
        readiness: buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: 1, name: "Ariadne" },
          selectedModel: "openai:gpt-4o",
          availableModels: [
            {
              id: "gpt-4o",
              model: "tldw:gpt-4o",
              provider: "openai",
              is_configured: true,
              catalog_only: true
            } as any
          ]
        }),
        title: "The selected chat model is not callable right now",
        actionLabel: "Open model settings"
      }
    ]

    for (const testCase of cases) {
      const copy = getCharacterChatReadinessCopy(testCase.readiness, t, {
        characterName: "Ariadne"
      })
      expect(copy.title).toBe(testCase.title)
      expect(copy.description).toContain("Ariadne")
      expect(copy.description).toContain("kept")
      expect(copy.actionLabel).toBe(testCase.actionLabel)
    }
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
        "Your character selection and draft are kept. Configure a chat model, then return here to continue with Ariadne.",
      actionLabel: "Open model settings"
    })
  })
})
