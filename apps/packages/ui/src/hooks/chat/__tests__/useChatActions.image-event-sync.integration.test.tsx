// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_EVENT_MIRROR_PREFIX,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
  parseImageGenerationEventMirrorContent
} from "@/utils/image-generation-chat"

const {
  addChatMessageMock,
  createChatMock,
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock,
  normalChatModeMock,
  updateMessageMediaMock,
  chatSettingsState,
  storageValues,
  storeOptionState
} = vi.hoisted(() => ({
  addChatMessageMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn(async () => ({
    assistant_message_id: "assistant-server-1",
    version: 1
  })),
  createChatMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  updateMessageMediaMock: vi.fn(async (_messageId: string, _payload: any) => null),
  chatSettingsState: {
    value: { imageEventSyncMode: "off" as "off" | "on" }
  },
  storageValues: new Map<string, unknown>(),
  storeOptionState: {
    value: { selectedModel: "deepseek-chat" as string | null }
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
        "history-image-sync"
  ),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-image-sync"
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
  updateMessageMedia: updateMessageMediaMock,
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

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selected_character",
  selectedCharacterStorage: {
    get: vi.fn(async () => null),
    set: vi.fn(async () => null)
  },
  selectedCharacterSyncStorage: {
    get: vi.fn(async () => null)
  },
  parseSelectedCharacterValue: vi.fn(() => null)
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    chatSettings: chatSettingsState.value,
    updateChatSettings: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    const [value] = React.useState(
      storageValues.has(key) ? storageValues.get(key) : defaultValue
    )
    return [value, vi.fn()] as const
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => storeOptionState.value
  }
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    addChatMessage: addChatMessageMock,
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    initialize: vi.fn(async () => null),
    getMessage: vi.fn(async () => ({ version: 1 })),
    editMessage: vi.fn(async () => null)
  }
}))

const defaultInitialMessages: any[] = [
  {
    id: "assistant-image-1",
    role: "assistant",
    name: "Image backend",
    isBot: true,
    message: "",
    messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
    images: ["data:image/png;base64,AAAA"],
    generationInfo: {
      image_generation: {
        request: {
          prompt: "sunlit city skyline",
          backend: "comfyui"
        },
        source: "generate-modal",
        variant_count: 1,
        active_variant_index: 0,
        createdAt: 1700000000000
      }
    }
  }
]

const createHookOptions = (
  initialMessagesOrOverrides: any[] | Record<string, unknown> = defaultInitialMessages
) => {
  const initialMessages = Array.isArray(initialMessagesOrOverrides)
    ? initialMessagesOrOverrides
    : defaultInitialMessages
  const overrides = Array.isArray(initialMessagesOrOverrides)
    ? {}
    : initialMessagesOrOverrides
  let currentMessages: any[] = initialMessages

  const setMessages = vi.fn((next: any[] | ((prev: any[]) => any[])) => {
    currentMessages =
      typeof next === "function" ? (next as (prev: any[]) => any[])(currentMessages) : next
  })

  const options: any = {
    t: (_key: string, fallback?: string) => fallback || _key,
    notification: {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    },
    abortController: null,
    setAbortController: vi.fn(),
    messages: currentMessages,
    setMessages,
    history: [],
    setHistory: vi.fn(),
    historyId: "history-image-sync",
    setHistoryId: vi.fn(),
    temporaryChat: false,
    selectedModel: "deepseek-chat",
    useOCR: false,
    selectedSystemPrompt: null,
    selectedKnowledge: null,
    toolChoice: "auto",
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
    ragSearchMode: "hybrid",
    ragTopK: 8,
    ragEnableGeneration: true,
    ragEnableCitations: true,
    ragSources: [],
    ragAdvancedOptions: {},
    serverChatId: "server-chat-1",
    serverChatTitle: "Image Sync Chat",
    serverChatCharacterId: null,
    serverChatState: "in-progress",
    serverChatTopic: null,
    serverChatClusterId: null,
    serverChatSource: null,
    serverChatExternalRef: null,
    setServerChatId: vi.fn(),
    setServerChatTitle: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatMetaLoaded: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatClusterId: vi.fn(),
    setServerChatSource: vi.fn(),
    setServerChatExternalRef: vi.fn(),
    ensureServerChatHistoryId: vi.fn(async () => "history-image-sync"),
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
    selectedCharacter: null,
    messageSteeringMode: "none",
    messageSteeringForceNarrate: false,
    clearMessageSteering: vi.fn()
  }

  Object.assign(options, overrides)

  return {
    options,
    getCurrentMessages: () => currentMessages,
    setMessages
  }
}

const invokeImageSubmit = async (
  onSubmit: any,
  imageEventSyncPolicy: "inherit" | "on" | "off",
  referenceFileId?: number
) => {
  const imageGenerationRequest = {
    prompt: "sunlit city skyline",
    backend: "comfyui",
    ...(typeof referenceFileId === "number"
      ? { referenceFileId }
      : {})
  }
  await act(async () => {
    await onSubmit({
      message: "generate skyline",
      image: "",
      imageBackendOverride: "comfyui",
      userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
      assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
      imageGenerationRequest,
      imageGenerationSource: "generate-modal",
      imageEventSyncPolicy
    })
  })
}

describe("useChatActions image event sync integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageValues.clear()
    storageValues.set(PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY, "off")
    chatSettingsState.value = { imageEventSyncMode: "off" }
    storeOptionState.value = { selectedModel: "deepseek-chat" }

    normalChatModeMock.mockImplementation(
      async (
        _message: string,
        _image: string,
        _isRegenerate: boolean,
        _messages: any[],
        _history: any[],
        _signal: AbortSignal,
        params: any
      ) => {
        await params.saveMessageOnSuccess({
          historyId: "history-image-sync",
          conversationId: "server-chat-1",
          saveToDb: false,
          message: "sunlit city skyline",
          fullText: "",
          assistantMessageId: "assistant-image-1",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          assistantImages: ["data:image/png;base64,AAAA"],
          generationInfo: {
            image_generation: {
              request: params?.imageGenerationRequest,
              source: "generate-modal",
              variant_count: 1,
              active_variant_index: 0,
              createdAt: 1700000000000
            }
          },
          imageEventSyncPolicy: params.imageEventSyncPolicy
        })
      }
    )

    streamCharacterChatCompletionMock.mockImplementation(
      async function* () {
        yield "Character reply"
      }
    )
  })

  it("keeps image events local-only when sync mode resolves to off", async () => {
    const { options } = createHookOptions()
    const { result } = renderHook(() => useChatActions(options))

    await invokeImageSubmit(result.current.onSubmit, "inherit")

    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(updateMessageMediaMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).toHaveBeenCalledTimes(1)
  })

  it("mirrors image events to server and marks sync as synced when mode is on", async () => {
    addChatMessageMock.mockResolvedValueOnce({ id: "server-message-42" })
    const { options, getCurrentMessages } = createHookOptions()
    const { result } = renderHook(() => useChatActions(options))

    await invokeImageSubmit(result.current.onSubmit, "on", 77)

    expect(addChatMessageMock).toHaveBeenCalledTimes(1)
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "server-chat-1",
      expect.objectContaining({
        role: "assistant",
        content: expect.stringContaining(IMAGE_GENERATION_EVENT_MIRROR_PREFIX)
      })
    )
    const mirroredContent = addChatMessageMock.mock.calls[0]?.[1]?.content as
      | string
      | undefined
    const mirroredPayload = parseImageGenerationEventMirrorContent(mirroredContent)
    expect(mirroredPayload?.request.referenceFileId).toBe(77)
    expect(updateMessageMediaMock).toHaveBeenCalledTimes(2)
    const lastCall = updateMessageMediaMock.mock.calls.at(-1)
    if (!lastCall) {
      throw new Error("Expected updateMessageMedia call payload")
    }
    const syncMeta = lastCall[1]?.generationInfo?.image_generation?.sync as
      | Record<string, any>
      | undefined
    expect(lastCall[0]).toBe("assistant-image-1")
    expect(syncMeta?.status).toBe("synced")
    expect(syncMeta?.mode).toBe("on")
    expect(syncMeta?.policy).toBe("on")
    expect(syncMeta?.serverMessageId).toBe("server-message-42")
    expect(getCurrentMessages()[0]?.generationInfo?.image_generation?.sync?.status).toBe(
      "synced"
    )
  })

  it("records failed sync status when server mirroring fails", async () => {
    addChatMessageMock.mockRejectedValueOnce(new Error("mirror timeout"))
    const { options, getCurrentMessages } = createHookOptions()
    const { result } = renderHook(() => useChatActions(options))

    await invokeImageSubmit(result.current.onSubmit, "on")

    expect(addChatMessageMock).toHaveBeenCalledTimes(1)
    expect(updateMessageMediaMock).toHaveBeenCalledTimes(2)
    const lastCall = updateMessageMediaMock.mock.calls.at(-1)
    if (!lastCall) {
      throw new Error("Expected updateMessageMedia call payload")
    }
    const syncMeta = lastCall[1]?.generationInfo?.image_generation?.sync as
      | Record<string, any>
      | undefined
    expect(syncMeta?.status).toBe("failed")
    expect(syncMeta?.mode).toBe("on")
    expect(syncMeta?.policy).toBe("on")
    expect(String(syncMeta?.error || "")).toContain("mirror timeout")
    expect(getCurrentMessages()[0]?.generationInfo?.image_generation?.sync?.status).toBe(
      "failed"
    )
  })

  it("retries mirroring on a later generation and updates sync status to synced", async () => {
    addChatMessageMock
      .mockRejectedValueOnce(new Error("mirror timeout"))
      .mockResolvedValueOnce({ id: "server-message-99" })
    const { options, getCurrentMessages } = createHookOptions()
    const { result } = renderHook(() => useChatActions(options))

    await invokeImageSubmit(result.current.onSubmit, "on")
    await invokeImageSubmit(result.current.onSubmit, "on")

    expect(addChatMessageMock).toHaveBeenCalledTimes(2)
    expect(updateMessageMediaMock).toHaveBeenCalledTimes(4)

    const statuses = updateMessageMediaMock.mock.calls
      .map((call) => call?.[1]?.generationInfo?.image_generation?.sync?.status)
      .filter((value): value is string => typeof value === "string")
    expect(statuses).toEqual(["pending", "failed", "pending", "synced"])

    const lastCall = updateMessageMediaMock.mock.calls.at(-1)
    if (!lastCall) {
      throw new Error("Expected updateMessageMedia call payload")
    }
    const syncMeta = lastCall[1]?.generationInfo?.image_generation?.sync as
      | Record<string, any>
      | undefined
    expect(syncMeta?.status).toBe("synced")
    expect(syncMeta?.serverMessageId).toBe("server-message-99")
    expect(getCurrentMessages()[0]?.generationInfo?.image_generation?.sync?.status).toBe(
      "synced"
    )
  })

  it("preserves the current explicit provider when selectedModel falls back from store state", async () => {
    storeOptionState.value = {
      selectedModel: "anthropic/claude-4.5-sonnet"
    }
    const { options } = createHookOptions({
      selectedModel: null,
      currentChatModelSettings: {
        apiProvider: "openrouter",
        setSystemPrompt: vi.fn()
      },
      selectedCharacter: {
        id: 7,
        name: "Guide"
      },
      serverChatCharacterId: 7
    })
    const { result } = renderHook(() => useChatActions(options))

    await act(async () => {
      await result.current.onSubmit({
        message: "Stay in character",
        image: "",
        requestOverrides: {
          selectedModel: "anthropic/claude-4.5-sonnet"
        }
      })
    })

    expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1)
    expect(streamCharacterChatCompletionMock.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({
        model: "anthropic/claude-4.5-sonnet",
        provider: "openrouter"
      })
    )
  })
})

describe("useChatActions character stream throttling integration", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-03-07T00:00:00.000Z"))
    vi.clearAllMocks()
    storageValues.clear()
    storageValues.set(PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY, "off")
    chatSettingsState.value = { imageEventSyncMode: "off" }
    normalChatModeMock.mockResolvedValue(undefined)
    createChatMock.mockResolvedValue({
      id: "chat-character-1",
      title: "Character Chat",
      version: 1,
      state: "in-progress",
      character_id: 101
    })
    addChatMessageMock.mockResolvedValue({ id: "chat-message-1", version: 1 })
    persistCharacterCompletionMock.mockResolvedValue({
      assistant_message_id: "assistant-message-1",
      version: 1
    })
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      for (let i = 0; i < 180; i += 1) {
        yield {
          choices: [
            {
              delta: {
                content: "x"
              }
            }
          ]
        }
      }
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("coalesces rapid tiny character chunks into bounded setMessages updates", async () => {
    const { options, setMessages, getCurrentMessages } = createHookOptions([])
    options.serverChatId = null
    options.serverChatCharacterId = null
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }
    options.selectedModel = "openrouter/openai/gpt-4.1-mini"
    options.currentChatModelSettings.apiProvider = "openrouter"

    const { result } = renderHook(() => useChatActions(options))

    await act(async () => {
      await result.current.onSubmit({
        message: "hello there",
        image: ""
      })
    })

    // Fake timers freeze the throttle window so this bound stays deterministic in CI.
    expect(setMessages.mock.calls.length).toBeLessThan(40)
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1)
    expect(normalChatModeMock).not.toHaveBeenCalled()

    const finalAssistant = getCurrentMessages()
      .filter((message: any) => message.isBot)
      .at(-1)
    expect(finalAssistant?.message).toBe("x".repeat(180))
  })

  it("does not fall back to addChatMessage when persistCharacterCompletion reports saved degraded state", async () => {
    persistCharacterCompletionMock.mockRejectedValueOnce(
      Object.assign(new Error("degraded"), {
        status: 503,
        details: {
          detail: {
            code: "persist_validation_degraded",
            saved: true,
            assistant_message_id: "assistant-server-99"
          }
        }
      })
    )
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield {
        choices: [
          {
            delta: {
              content: "saved degraded reply"
            }
          }
        ]
      }
    })

    const { options } = createHookOptions([])
    options.serverChatId = "server-chat-1"
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }

    const { result } = renderHook(() => useChatActions(options))

    await act(async () => {
      await result.current.onSubmit({
        message: "persist but degrade",
        image: ""
      })
    })

    expect(persistCharacterCompletionMock).toHaveBeenCalledTimes(1)
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        assistant_message_id: expect.any(String)
      })
    )
    expect(
      addChatMessageMock.mock.calls.filter(
        ([, payload]) => payload?.role === "assistant"
      )
    ).toHaveLength(0)
  })

  it("does not fall back when a saved degraded error only exposes top-level detail", async () => {
    persistCharacterCompletionMock.mockRejectedValueOnce(
      Object.assign(new Error("degraded"), {
        status: 503,
        detail: {
          code: "persist_validation_degraded",
          saved: true
        }
      })
    )
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield {
        choices: [
          {
            delta: {
              content: "saved degraded reply without server id"
            }
          }
        ]
      }
    })

    const { options } = createHookOptions([])
    options.serverChatId = "server-chat-1"
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }

    const { result } = renderHook(() => useChatActions(options))

    await act(async () => {
      await result.current.onSubmit({
        message: "persist but degrade without id",
        image: ""
      })
    })

    expect(persistCharacterCompletionMock).toHaveBeenCalledTimes(1)
    expect(
      addChatMessageMock.mock.calls.filter(
        ([, payload]) => payload?.role === "assistant"
      )
    ).toHaveLength(0)
  })
})
