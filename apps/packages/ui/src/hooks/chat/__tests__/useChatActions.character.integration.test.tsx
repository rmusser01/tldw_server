// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"

const {
  createChatMock,
  detectCharacterMoodMock,
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock,
  addChatMessageMock,
  normalChatModeMock,
  resolveVisualIdentityBindingMock
} = vi.hoisted(() => ({
  createChatMock: vi.fn(),
  detectCharacterMoodMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn(async () => ({
    assistant_message_id: "assistant-server-1",
    version: 1
  })),
  addChatMessageMock: vi.fn(async () => ({ id: "user-server-1", version: 1 })),
  normalChatModeMock: vi.fn(),
  resolveVisualIdentityBindingMock: vi.fn(async () => ({
    actor_kind: "character",
    actor_id: 12,
    pack_id: 1,
    pack_version_id: 2,
    expression_key: "surprised",
    requested_expression_key: "surprised",
    asset_id: 9,
    storage_relpath: null,
    fallback_reason: "manual_override",
    is_animated: false,
    content_type: "image/png",
    asset_url: "/api/v1/visual-identities/packs/1/assets/9/content"
  }))
}))

const messageStoreState = vi.hoisted(() => ({
  value: {
    selectedModel: "deepseek-chat" as string | null,
    serverChatId: null as string | null,
    serverChatCharacterId: null as string | number | null,
    serverChatAssistantKind: null as "character" | "persona" | null,
    serverChatSource: null as string | null
  }
}))

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
  normalChatMode: normalChatModeMock
}))

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/tabChatMode", () => ({
  tabChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/documentChatMode", () => ({
  documentChatMode: vi.fn()
}))

vi.mock("@/hooks/utils/messageHelpers", () => ({
  validateBeforeSubmit: vi.fn(() => true),
  createSaveMessageOnSuccess: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-character"
  ),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-character"
  )
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createRegenerateLastMessage: vi.fn(() => vi.fn()),
  createEditMessage: vi.fn(() => vi.fn()),
  createStopStreamingRequest: vi.fn(() => vi.fn()),
  createBranchMessage: vi.fn(() => vi.fn())
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => "generated-id"),
  saveHistory: vi.fn(),
  saveMessage: vi.fn(),
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(async () => null),
  removeMessageByIndex: vi.fn(),
  formatToChatHistory: vi.fn((items: unknown) => items),
  formatToMessage: vi.fn((items: unknown) => items),
  getSessionFiles: vi.fn(async () => []),
  getPromptById: vi.fn(async () => null)
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null)
}))

vi.mock("@/db/dexie/branch", () => ({
  generateBranchFromMessageIds: vi.fn(async () => null)
}))

vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: vi.fn(async () => null)
}))

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: detectCharacterMoodMock
}))

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selected_character",
  selectedCharacterStorage: {
    get: vi.fn(async () => null),
    set: vi.fn(async () => null)
  },
  selectedCharacterSyncStorage: {
    get: vi.fn(async () => null)
  },
  parseSelectedCharacterValue: vi.fn((value: unknown) => value)
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: {},
    updateSettings: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => {
    const [value] = React.useState(defaultValue)
    return [value, vi.fn()] as const
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => messageStoreState.value
  }
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    addChatMessage: addChatMessageMock,
    getChatSettings: vi.fn(async () => ({ settings: null })),
    initialize: vi.fn(async () => null),
    resolveVisualIdentityBinding: resolveVisualIdentityBindingMock
  }
}))

const createHookOptions = () => ({
  t: (_key: string, fallback?: string) => fallback || _key,
  notification: {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  },
  abortController: null,
  setAbortController: vi.fn(),
  messages: [],
  setMessages: vi.fn(),
  history: [],
  setHistory: vi.fn(),
  historyId: "history-character",
  setHistoryId: vi.fn(),
  temporaryChat: false,
  selectedModel: "deepseek-chat",
  useOCR: false,
  selectedSystemPrompt: null,
  selectedKnowledge: null,
  toolChoice: "auto" as const,
  webSearch: false,
  currentChatModelSettings: {
    apiProvider: "openai",
    setSystemPrompt: vi.fn()
  },
  setIsSearchingInternet: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setActionInfo: vi.fn(),
  fileRetrievalEnabled: false,
  ragMediaIds: null,
  ragSearchMode: "hybrid" as const,
  ragTopK: 8,
  ragEnableGeneration: true,
  ragEnableCitations: true,
  ragSources: [],
  ragAdvancedOptions: {},
  serverChatId: "tracked-chat-1",
  serverChatTitle: "Tracked character chat",
  serverChatCharacterId: "char-tracked",
  serverChatAssistantKind: "character" as const,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
  serverChatState: "in-progress" as const,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  setServerChatId: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setServerChatState: vi.fn(),
  setServerChatVersion: vi.fn(),
  setServerChatTopic: vi.fn(),
  setServerChatClusterId: vi.fn(),
  setServerChatSource: vi.fn(),
  setServerChatExternalRef: vi.fn(),
  ensureServerChatHistoryId: vi.fn(async () => "history-character"),
  contextFiles: [],
  setContextFiles: vi.fn(),
  documentContext: null,
  setDocumentContext: vi.fn(),
  uploadedFiles: [],
  compareModeActive: false,
  compareSelectedModels: [],
  compareMaxModels: 3,
  compareFeatureEnabled: false,
  markCompareHistoryCreated: vi.fn(),
  replyTarget: null,
  clearReplyTarget: vi.fn(),
  messageSteeringPrompts: null,
  setSelectedQuickPrompt: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  invalidateServerChatHistory: vi.fn(),
  selectedCharacter: {
    id: "char-stale",
    name: "Stale Character",
    system_prompt: "Stale prompt"
  },
  selectedAssistant: null,
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

describe("useChatActions character integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    normalChatModeMock.mockResolvedValue(undefined)
    createChatMock.mockResolvedValue({
      id: "unexpected-new-chat",
      character_id: "char-stale",
      title: "Wrong chat"
    })
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield {
        choices: [
          {
            delta: {
              content: "Tracked reply"
            }
          }
        ]
      }
    })
    detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.2,
      topic: "reply"
    })
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatSource: null
    }
    resolveVisualIdentityBindingMock.mockClear()
  })

  it("keeps tracked character routing anchored to current chat metadata when global character state is stale", async () => {
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello tracked character",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1)
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "tracked-chat-1",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("character")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith("char-tracked")
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(options.setServerChatId).not.toHaveBeenCalledWith(null)
  })

  it("handles emote commands without sending chat", async () => {
    const options = {
      ...createHookOptions(),
      setVisualIdentityManualExpressionOverride: vi.fn()
    }
    const { result } = renderHook(() => useChatActions(options as any))

    let submitResult: unknown
    await act(async () => {
      submitResult = await result.current.onSubmit({
        message: "/emote surprised",
        image: ""
      })
    })

    expect(options.setVisualIdentityManualExpressionOverride).toHaveBeenCalledWith(
      "surprised"
    )
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(options.setStreaming).not.toHaveBeenCalledWith(true)
    expect(submitResult).toEqual({
      status: "skipped",
      reason: "Visual identity expression updated"
    })
  })

  it("uses manual visual identity override when resolving assistant message metadata", async () => {
    const options = {
      ...createHookOptions(),
      serverChatCharacterId: 12,
      selectedCharacter: {
        id: "12",
        name: "Numeric Character",
        system_prompt: "Prompt"
      },
      visualIdentityManualExpressionOverride: "surprised"
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "React to this",
        image: ""
      })
    })

    expect(resolveVisualIdentityBindingMock).toHaveBeenCalledWith(
      expect.objectContaining({
        actor_kind: "character",
        actor_id: 12,
        expression_key: "surprised",
        manual_override_expression_key: "surprised"
      })
    )
  })

  it("does not let stale overlay assistant state override the active character chat", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "other-character-chat",
      serverChatTitle: "Other character chat",
      serverChatCharacterId: 42,
      serverChatAssistantId: "42",
      selectedCharacter: {
        id: 99,
        name: "Miku",
        system_prompt: "Stale Miku prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Stale Miku prompt",
        metadata: { selectionMode: "overlay" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue the other character chat",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "other-character-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "other-character-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 42
      }),
      undefined
    )
    const lastPersistCall = persistCharacterCompletionMock.mock.calls.at(-1) as
      | unknown[]
      | undefined
    const lastPersistRequest = lastPersistCall?.[1] as
      | Record<string, unknown>
      | undefined
    expect(lastPersistRequest?.speaker_character_name).toBeUndefined()
    expect(options.setServerChatId).not.toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).not.toHaveBeenCalledWith(null)
  })

  it("honors an explicit tracked character switch over the current chat metadata", async () => {
    createChatMock.mockResolvedValueOnce({
      id: "new-miku-chat",
      character_id: 99,
      title: "Miku chat"
    })
    const options = {
      ...createHookOptions(),
      serverChatId: "other-character-chat",
      serverChatTitle: "Other character chat",
      serverChatCharacterId: 42,
      serverChatAssistantId: "42",
      selectedCharacter: {
        id: 42,
        name: "Other Character",
        system_prompt: "Other character prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Miku prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Start talking with Miku",
        image: ""
      })
    })

    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: "99"
      })
    )
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "new-miku-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "new-miku-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 99,
        speaker_character_name: "Miku"
      }),
      undefined
    )
  })

  it("reuses the latest store character chat when greeting persistence updates before the send closure", async () => {
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: "ashley-greeting-chat",
      serverChatCharacterId: 4,
      serverChatAssistantKind: "character",
      serverChatSource: "webui-character-chat"
    }
    const options = {
      ...createHookOptions(),
      serverChatId: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatSource: null,
      selectedCharacter: {
        id: 4,
        name: "Ashley",
        system_prompt: "Ashley prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "4",
        name: "Ashley",
        system_prompt: "Ashley prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue Ashley from her greeting",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "ashley-greeting-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "ashley-greeting-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 4,
        speaker_character_name: "Ashley"
      }),
      undefined
    )
  })

  it("uses the latest store character id when the hook prop still points at another character", async () => {
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: "miku-current-chat",
      serverChatCharacterId: 99,
      serverChatAssistantKind: "character",
      serverChatSource: "webui-character-chat"
    }
    const options = {
      ...createHookOptions(),
      serverChatId: "old-character-chat",
      serverChatTitle: "Old character chat",
      serverChatCharacterId: 42,
      serverChatAssistantKind: "character",
      serverChatAssistantId: "42",
      serverChatSource: "webui-character-chat",
      selectedCharacter: {
        id: 99,
        name: "Miku",
        system_prompt: "Miku prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Miku prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue Miku from the selected greeting",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "miku-current-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "miku-current-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 99,
        speaker_character_name: "Miku"
      }),
      undefined
    )
  })

  it("passes workspace scope through when creating a character-backed chat", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      selectedCharacter: null,
      selectedAssistant: {
        kind: "character" as const,
        id: "char-scoped",
        name: "Scoped Character",
        system_prompt: "Scoped prompt",
        metadata: {
          selectionMode: "tracked" as const
        }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello workspace character",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: "char-scoped",
        state: "in-progress"
      }),
      { scope }
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        role: "user",
        content: "Hello workspace character"
      }),
      { scope }
    )
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.objectContaining({ scope })
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply"
      }),
      { scope }
    )
    expect(normalChatModeMock).not.toHaveBeenCalled()
  })

  it("strips explicit streaming emote directives and persists emote events", async () => {
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () {
      yield "Em"
      yield "ote: smug\n"
      yield "Hello "
      yield "there.\n"
      yield "Emote: annoyed\n"
      yield "Fine."
    })
    detectCharacterMoodMock.mockReturnValueOnce({
      label: "happy",
      confidence: 0.9,
      topic: "classifier"
    })

    let messagesState: any[] = []
    const messageSnapshots: any[][] = []
    const moodLabels: unknown[] = []
    const options = {
      ...createHookOptions(),
      setMessages: vi.fn((next) => {
        messagesState = typeof next === "function" ? next(messagesState) : next
        messageSnapshots.push(messagesState)
        const assistant = messagesState.find((message) => message?.isBot)
        moodLabels.push(assistant?.moodLabel)
      })
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Test explicit emotes",
        image: ""
      })
    })

    const persistCall = persistCharacterCompletionMock.mock.calls.at(-1) as
      | unknown[]
      | undefined
    const persistPayload = persistCall?.[1] as
      | Record<string, unknown>
      | undefined
    expect(persistPayload).toMatchObject({
      assistant_content: "Hello there.\nFine.",
      mood_label: "annoyed",
      emote_events: [
        { state: "smug", at_char: 0 },
        { state: "annoyed", at_char: 13 }
      ]
    })
    expect(persistPayload).not.toHaveProperty("mood_confidence")
    expect(persistPayload).not.toHaveProperty("mood_topic")
    expect(detectCharacterMoodMock).not.toHaveBeenCalled()

    const renderedAssistantMessages = messageSnapshots
      .flatMap((snapshot) => snapshot.filter((message) => message?.isBot))
      .map((message) => String(message?.message ?? ""))
    expect(
      renderedAssistantMessages.some((message) => message.includes("Emote:"))
    ).toBe(false)
    expect(renderedAssistantMessages.at(-1)).toBe("Hello there.\nFine.")
    expect(moodLabels).toContain("smug")
    expect(moodLabels).toContain("annoyed")
  })
})
