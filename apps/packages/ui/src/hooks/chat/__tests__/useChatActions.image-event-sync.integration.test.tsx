import "./native-plasmo-storage-fixture"
// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { MemoryRouter } from "react-router-dom"
import { PlaygroundChat } from "@/components/Option/Playground/PlaygroundChat"
import { useStoreMessageOption } from "@/store/option"
import { useOriginalSdkSceneFixture } from "./sdk-original-scene-fixture"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_EVENT_MIRROR_PREFIX,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
  parseImageGenerationEventMirrorContent
} from "@/utils/image-generation-chat"

const playgroundState = vi.hoisted(() => ({
  options: {} as Record<string, unknown>
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...playgroundState.options,
    messages: useStoreMessageOption.getState().messages,
    compareMode: false,
    compareSelectionByCluster: {},
    compareActiveModelsByCluster: {},
    compareCanonicalByCluster: {},
    compareContinuationModeByCluster: {},
    compareSplitChats: {}
  })
}))
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: [] }) }))
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    messageId?: string
    onSelectImageVariant?: (payload: {
      messageId?: string
      variantIndex: number
    }) => void
    onDeleteImageVariant?: (payload: {
      messageId?: string
      variantIndex: number
    }) => void
  }) => (
    <div>
      <button
        onClick={() =>
          props.onSelectImageVariant?.({
            messageId: props.messageId,
            variantIndex: 0
          })
        }>
        Select older variant
      </button>
      <button
        onClick={() =>
          props.onDeleteImageVariant?.({
            messageId: props.messageId,
            variantIndex: 1
          })
        }>
        Delete latest variant
      </button>
    </div>
  )
}))

const {
  addChatMessageMock,
  createChatMock,
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock,
  normalChatModeMock,
  savedSuccessPayloads,
  updateMessageMediaMock,
  chatSettingsState,
  scopeState,
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
  savedSuccessPayloads: [] as unknown[],
  updateMessageMediaMock: vi.fn(async (_messageId: string, _payload: any) => null),
  chatSettingsState: {
    value: { imageEventSyncMode: "off" as "off" | "on" }
  },
  scopeState: {
    invalidatedSignal: undefined as AbortSignal | undefined,
    requestScope: { config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" as "single-user" | "multi-user" }, userId: null as number | null },
    originalSignal: new AbortController().signal
  },
  storageValues: new Map<string, unknown>(),
  storeOptionState: {
    value: { selectedModel: "deepseek-chat" as string | null }
  }
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
    scopeKey: "scope:test-chat", requestScope: scopeState.requestScope,
    scopeSignal: signal, scopeInvalidatedSignal: scopeState.invalidatedSignal ?? scopeState.originalSignal, definitions: {}, capability: "unchecked", release: vi.fn()
  })
}))

vi.mock("@/hooks/chat-modes/normalChatMode", async original => ({
  ...await original<typeof import("@/hooks/chat-modes/normalChatMode")>(),
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
      async (payload?: unknown): Promise<string | null> => {
        savedSuccessPayloads.push(payload)
        return "history-image-sync"
      }
  ),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-image-sync"
  )
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createNativeDeleteMessage: vi.fn(() => vi.fn(async () => false)),
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

vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>()
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
    settings: chatSettingsState.value,
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

vi.mock("@/store/option", async (original) => {
  const actual = await original<typeof import("@/store/option")>()
  return {
    ...actual,
    useStoreMessageOption: Object.assign(
      (...args: Parameters<typeof actual.useStoreMessageOption>) =>
        actual.useStoreMessageOption(...args),
      {
        getState: () => ({
          ...actual.useStoreMessageOption.getState(),
          ...storeOptionState.value
        }),
        setState: actual.useStoreMessageOption.setState,
        subscribe: actual.useStoreMessageOption.subscribe
      }
    )
  }
})

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    addChatMessage: addChatMessageMock,
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    getChatSettings: vi.fn(async () => ({ settings: null })),
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
    useStoreMessageOption.setState({ messages: currentMessages })
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
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatPersonaMemoryMode: null,
    serverChatState: "in-progress",
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

const useChatActions = (options: Parameters<typeof useOriginalSdkSceneFixture>[0]) =>
  useOriginalSdkSceneFixture(options, () => ({ requestScope: scopeState.requestScope,
    scopeInvalidatedSignal: scopeState.invalidatedSignal ?? scopeState.originalSignal }))

describe("useChatActions image event sync integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageValues.clear()
    storageValues.set(PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY, "off")
    chatSettingsState.value = { imageEventSyncMode: "off" }
    scopeState.invalidatedSignal = undefined
    scopeState.originalSignal = new AbortController().signal
    scopeState.requestScope = { config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" }, userId: null }
    storeOptionState.value = { selectedModel: "deepseek-chat" }
    savedSuccessPayloads.length = 0

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
          requestScope: params.servicePromptSnapshot?.requestScope,
          scopeSignal: params.servicePromptSnapshot?.scopeSignal,
          scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
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

  it("captures owner scope for a first local image generation even when server mirroring is off", async () => {
    const { options } = createHookOptions({ historyId: null, serverChatId: null })
    const { result } = renderHook(() => useChatActions(options))
    await invokeImageSubmit(result.current.onSubmit, "off")
    expect(normalChatModeMock.mock.calls[0]?.[6].servicePromptSnapshot?.requestScope).toEqual({
      config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" }, userId: null
    })
  })

  it("does not restore the old image transcript after its account is invalidated", async () => {
    const owner = new AbortController()
    scopeState.invalidatedSignal = owner.signal
    let finish!: () => void
    const held = new Promise<void>(resolve => { finish = resolve })
    const { options, getCurrentMessages, setMessages } = createHookOptions()
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as { setMessages: (next: unknown) => void; setHistory: (next: unknown) => void; setAbortController: (next: null) => void; setIsProcessing: (next: boolean) => void; setStreaming: (next: boolean) => void }
      await held
      params.setMessages(options.messages)
      params.setHistory(options.history)
      params.setAbortController(null)
      params.setIsProcessing(false)
      params.setStreaming(false)
      return { status: "skipped", reason: "Request scope changed" }
    })
    const { result } = renderHook(() => useChatActions(options))
    const controller = new AbortController()
    let submission!: ReturnType<typeof result.current.onSubmit>
    act(() => { submission = result.current.onSubmit({ message: "Draw a private scene", image: "", imageBackendOverride: "comfyui", controller }) })
    await waitFor(() => expect(normalChatModeMock).toHaveBeenCalledOnce())
    const bobMessages = [{ message: "Bob replacement transcript" }]
    setMessages(bobMessages)
    options.setHistory.mockClear()
    options.setAbortController.mockClear()
    options.setIsProcessing.mockClear()
    options.setStreaming.mockClear()
    await act(async () => { owner.abort(); finish(); await submission })
    expect(getCurrentMessages()).toBe(bobMessages)
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(options.setAbortController).not.toHaveBeenCalled()
    expect(options.setIsProcessing).not.toHaveBeenCalled()
    expect(options.setStreaming).not.toHaveBeenCalled()
  })

  it("preserves server conversation ownership when saving an image history ID", async () => {
    const { options } = createHookOptions()
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as { setHistoryId: (id: string, options: { preserveServerChatId: boolean }) => void }
      params.setHistoryId("owned-image-history", { preserveServerChatId: true })
      return { status: "submitted" }
    })
    const { result } = renderHook(() => useChatActions(options))
    await invokeImageSubmit(result.current.onSubmit, "off")
    expect(options.setHistoryId).toHaveBeenCalledWith("owned-image-history", { preserveServerChatId: true })
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
      }),
      { requestScope: scopeState.requestScope, signal: normalChatModeMock.mock.calls[0][6].servicePromptSnapshot.scopeSignal }
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

  it.each([false, true])(
    "a held image mirror retains original metadata after same-ID reopen=%s",
    async (retired) => {
      let finish!: (value: { id: string }) => void
      const held = new Promise<{ id: string }>((resolve) => {
        finish = resolve
      })
      addChatMessageMock.mockReturnValueOnce(held)
      const { options, getCurrentMessages, setMessages } = createHookOptions()
      const hook = renderHook(() => useChatActions(options))
      let submission!: Promise<unknown>
      act(() => {
        submission = hook.result.current.onSubmit({
          message: "Original image question",
          image: "",
          imageBackendOverride: "comfyui",
          imageEventSyncPolicy: "on",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          imageGenerationRequest:
            defaultInitialMessages[0].generationInfo.image_generation.request,
          imageGenerationSource: "generate-modal"
        })
      })
      try {
        await waitFor(() => expect(addChatMessageMock).toHaveBeenCalledOnce())
        const params = normalChatModeMock.mock.calls[0][6]
        expect(params.actorSceneSend.isCurrent()).toBe(true)
        const replacement = [
          {
            ...defaultInitialMessages[0],
            message: "Replacement display",
            generationInfo: {
              image_generation: {
                request: { prompt: "Replacement prompt" },
                sync: { status: "replacement" }
              }
            }
          }
        ]
        if (retired) {
          useStoreMessageOption.getState().startNewChatDraft(false)
          useStoreMessageOption.setState({
            historyId: options.historyId,
            serverChatId: options.serverChatId
          })
          setMessages(replacement)
          expect(params.actorSceneSend.isCurrent()).toBe(false)
        }
        expect(scopeState.originalSignal.aborted).toBe(false)
        await act(async () => {
          finish({ id: "original-mirror-ack" })
          await submission
        })
        if (retired) expect(getCurrentMessages()).toEqual(replacement)
        else
          expect(
            getCurrentMessages()[0].generationInfo.image_generation.sync.status
          ).toBe("synced")
        const persisted = savedSuccessPayloads[0] as {
          assistantImages: string[]
          generationInfo: {
            image_generation: {
              request: unknown
              sync: { serverMessageId: string }
            }
          }
        }
        expect(persisted.generationInfo.image_generation.request).toEqual(
          defaultInitialMessages[0].generationInfo.image_generation.request
        )
        expect(
          persisted.generationInfo.image_generation.sync.serverMessageId
        ).toBe("original-mirror-ack")
      } finally {
        finish({ id: "original-mirror-ack" })
        await submission
        hook.unmount()
      }
    }
  )

  it.each([
    { edit: "select", retired: false },
    { edit: "delete", retired: false },
    { edit: "select", retired: true },
    { edit: "delete", retired: true }
  ])(
    "preserves a later owned image $edit while its mirror is held (retired=$retired)",
    async ({ edit, retired }) => {
      let finish!: (value: { id: string }) => void
      const held = new Promise<{ id: string }>((resolve) => {
        finish = resolve
      })
      addChatMessageMock.mockReturnValueOnce(held)
      const { options, getCurrentMessages, setMessages } = createHookOptions()
      const hook = renderHook(() => useChatActions(options))
      let submission!: Promise<unknown>
      act(() => {
        submission = hook.result.current.onSubmit({
          message: "Original image question",
          image: "",
          imageBackendOverride: "comfyui",
          imageEventSyncPolicy: "on",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          imageGenerationRequest:
            defaultInitialMessages[0].generationInfo.image_generation.request
        })
      })
      try {
        await waitFor(() => expect(addChatMessageMock).toHaveBeenCalledOnce())
        const params = normalChatModeMock.mock.calls[0][6]
        const images =
          edit === "delete"
            ? []
            : ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]
        const imageGeneration = {
          ...getCurrentMessages()[0].generationInfo.image_generation,
          variant_count: images.length,
          active_variant_index: images.length - 1
        }
        setMessages([
          {
            ...getCurrentMessages()[0],
            images,
            generationInfo: { image_generation: imageGeneration }
          }
        ])
        expect(params.actorSceneSend.isCurrent()).toBe(true)
        const replacement = [
          {
            ...defaultInitialMessages[0],
            images: ["replacement"],
            generationInfo: { replacement: true }
          }
        ]
        if (retired) {
          useStoreMessageOption.getState().startNewChatDraft(false)
          useStoreMessageOption.setState({
            historyId: options.historyId,
            serverChatId: options.serverChatId
          })
          setMessages(replacement)
        }
        await act(async () => {
          finish({ id: "edited-image-ack" })
          await submission
        })
        if (retired) expect(getCurrentMessages()).toEqual(replacement)
        else
          expect(
            getCurrentMessages()[0].generationInfo.image_generation
              .active_variant_index
          ).toBe(images.length - 1)
        expect(updateMessageMediaMock.mock.calls.at(-1)[1].images).toEqual(images)
        expect(
          (
            savedSuccessPayloads[0] as {
              assistantImages: string[]
              generationInfo: {
                image_generation: {
                  request: unknown
                  sync: { serverMessageId: string }
                }
              }
            }
          ).assistantImages
        ).toEqual(images)
        expect(
          (
            savedSuccessPayloads[0] as {
              assistantImages: string[]
              generationInfo: {
                image_generation: {
                  request: unknown
                  sync: { serverMessageId: string }
                }
              }
            }
          ).generationInfo.image_generation.sync.serverMessageId
        ).toBe("edited-image-ack")
      } finally {
        finish({ id: "edited-image-ack" })
        await submission
        hook.unmount()
      }
    }
  )

  it.each(
    ["select", "delete"].flatMap((edit) =>
      ["current", "new", "same-id"].map((boundary) => ({ edit, boundary }))
    )
  )(
    "retains an owned image $edit during the final media write across $boundary",
    async ({ edit, boundary }) => {
      let release!: () => void
      const held = new Promise<null>((resolve) => {
        release = () => resolve(null)
      })
      updateMessageMediaMock.mockResolvedValueOnce(null).mockReturnValueOnce(held)
      addChatMessageMock.mockResolvedValueOnce({ id: "final-media-ack" })
      const { options, setMessages } = createHookOptions()
      const hook = renderHook(() => useChatActions(options))
      let pending!: Promise<unknown>
      act(() => {
        pending = hook.result.current.onSubmit({
          message: "Original image question",
          image: "",
          imageBackendOverride: "comfyui",
          imageEventSyncPolicy: "on",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          imageGenerationRequest:
            defaultInitialMessages[0].generationInfo.image_generation.request
        })
      })
      try {
        await waitFor(() =>
          expect(updateMessageMediaMock).toHaveBeenCalledTimes(2)
        )
        const row = useStoreMessageOption.getState().messages[0]
        const images = edit === "delete" ? [] : ["data:image/png;base64,LATEST"]
        setMessages([
          {
            ...row,
            images,
            generationInfo: {
              image_generation: {
                ...row.generationInfo.image_generation,
                active_variant_index: images.length - 1,
                variant_count: images.length
              }
            }
          }
        ])
        const replacement = [
          {
            ...row,
            images: ["replacement-image"],
            generationInfo: { replacement: true }
          }
        ]
        if (boundary !== "current") {
          useStoreMessageOption.getState().startNewChatDraft(false)
          if (boundary === "same-id")
            useStoreMessageOption.setState({
              historyId: options.historyId,
              serverChatId: options.serverChatId
            })
          setMessages(replacement)
        }
        await act(async () => {
          release()
          await pending
        })
        if (boundary !== "current")
          expect(useStoreMessageOption.getState().messages).toEqual(replacement)
        const persisted = savedSuccessPayloads[0] as Record<string, unknown>
        expect(persisted.assistantImages).toEqual(images)
        expect(persisted.generationInfo).toMatchObject({
          image_generation: {
            active_variant_index: images.length - 1,
            sync: { serverMessageId: "final-media-ack" }
          }
        })
      } finally {
        release()
        await pending
        hook.unmount()
      }
    }
  )

  it.each(
    ["select", "delete"].flatMap((edit) =>
      ["sdk", "media"].flatMap((stage) =>
        [false, true].map((retired) => ({ edit, stage, retired }))
      )
    )
  )(
    "actual image $edit preserves its original event during $stage wait (retired=$retired)",
    async ({ edit, stage, retired }) => {
      let releaseSdk!: (value: { id: string }) => void
      let releaseMedia!: () => void
      const sdk = new Promise<{ id: string }>((resolve) => {
        releaseSdk = resolve
      })
      const media = new Promise<null>((resolve) => {
        releaseMedia = () => resolve(null)
      })
      if (stage === "sdk") addChatMessageMock.mockReturnValueOnce(sdk)
      else {
        addChatMessageMock.mockResolvedValueOnce({ id: "group-ack" })
        updateMessageMediaMock
          .mockResolvedValueOnce(null)
          .mockReturnValueOnce(media)
      }
      const original = {
        ...defaultInitialMessages[0],
        activeVariantIndex: 1,
        variants: [
          {
            id: "older-variant",
            message: "",
            images: ["data:image/png;base64,OLDER"],
            generationInfo: defaultInitialMessages[0].generationInfo
          },
          {
            id: "assistant-image-1",
            message: "",
            images: ["data:image/png;base64,AAAA"],
            generationInfo: defaultInitialMessages[0].generationInfo
          }
        ]
      }
      const { options } = createHookOptions([original])
      playgroundState.options = options
      const hook = renderHook(() => useChatActions(options))
      const display = render(
        <MemoryRouter>
          <PlaygroundChat />
        </MemoryRouter>
      )
      let pending!: Promise<unknown>
      act(() => {
        pending = hook.result.current.onSubmit({
          message: "Original image question",
          image: "",
          imageBackendOverride: "comfyui",
          imageEventSyncPolicy: "on",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          imageGenerationRequest:
            defaultInitialMessages[0].generationInfo.image_generation.request
        })
      })
      try {
        await waitFor(() =>
          expect(
            stage === "sdk"
              ? addChatMessageMock.mock.calls.length
              : updateMessageMediaMock.mock.calls.length
          ).toBe(stage === "sdk" ? 1 : 2)
        )
        fireEvent.click(
          screen.getByRole("button", {
            name:
              edit === "select" ? "Select older variant" : "Delete latest variant"
          })
        )
        const selected = useStoreMessageOption.getState().messages[0]
        expect(selected.id).toBe("older-variant")
        expect(selected.generationInfo.image_generation.event_id).toBe(
          "assistant-image-1"
        )
        if (retired) {
          useStoreMessageOption.getState().startNewChatDraft(false)
          options.setMessages([{ ...original, images: ["replacement"] }])
        }
        await act(async () => {
          releaseSdk({ id: "group-ack" })
          releaseMedia()
          await pending
        })
        const persisted = savedSuccessPayloads[0] as {
          assistantMessageId: string
          assistantImages: string[]
        }
        expect(persisted.assistantMessageId).toBe("assistant-image-1")
        expect(persisted.assistantImages).toEqual(["data:image/png;base64,OLDER"])
        if (retired)
          expect(useStoreMessageOption.getState().messages[0].images).toEqual([
            "replacement"
          ])
        else
          expect(
            useStoreMessageOption.getState().messages[0].generationInfo
              .image_generation.sync.status
          ).toBe("synced")
      } finally {
        releaseSdk({ id: "group-ack" })
        releaseMedia()
        await pending
        display.unmount()
        hook.unmount()
      }
    }
  )

  it("includes scoped image sync metadata in the deferred local save", async () => {
    addChatMessageMock.mockResolvedValueOnce({ id: "server-message-42" })
    const scopeController = new AbortController()
    const requestScope = Object.freeze({
      config: Object.freeze({
        serverUrl: "https://scope.example",
        authMode: "multi-user" as const
      }),
      userId: 7
    })
    scopeState.requestScope = requestScope
    scopeState.invalidatedSignal = scopeController.signal
    normalChatModeMock.mockImplementationOnce(
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
          requestScope: params.servicePromptSnapshot?.requestScope,
          scopeSignal: params.servicePromptSnapshot?.scopeSignal,
          scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
          message: "sunlit city skyline",
          fullText: "",
          assistantMessageId: "assistant-image-1",
          userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
          assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
          assistantImages: ["data:image/png;base64,AAAA"],
          generationInfo: defaultInitialMessages[0].generationInfo,
          imageEventSyncPolicy: params.imageEventSyncPolicy,
        })
      }
    )
    const { options } = createHookOptions()
    const { result } = renderHook(() => useChatActions(options))

    await invokeImageSubmit(result.current.onSubmit, "on")

    expect(savedSuccessPayloads).toHaveLength(1)
    const persisted = savedSuccessPayloads[0] as { assistantImages: string[]; generationInfo: { image_generation: { request: unknown; sync: { serverMessageId: string } } } }
    expect(persisted.generationInfo.image_generation.sync).toEqual(
      expect.objectContaining({
        status: "synced",
        mode: "on",
        policy: "on",
        serverMessageId: "server-message-42"
      })
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
      selectedAssistant: {
        kind: "character", id: "7", name: "Guide",
        metadata: { selectionMode: "tracked" }
      },
      serverChatCharacterId: 7,
      serverChatAssistantKind: "character",
      serverChatAssistantId: "7"
    })
    const { result } = renderHook(() => useChatActions(options))

    await act(async () => {
      const outcome = await result.current.onSubmit({
        message: "Stay in character",
        image: "",
        requestOverrides: {
          selectedModel: "anthropic/claude-4.5-sonnet"
        }
      })
      expect(outcome).toEqual({ status: "submitted" })
    })

    expect(options.notification.error.mock.calls).toEqual([])
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
    options.selectedAssistant = {
      kind: "character",
      id: "101",
      name: "Stream Character",
      metadata: { selectionMode: "tracked" }
    }
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }
    options.selectedAssistant = {
      kind: "character", id: "101", name: "Stream Character",
      metadata: { selectionMode: "tracked" }
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
    expect(options.notification.error.mock.calls).toEqual([])
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
    options.serverChatId = "chat-character-1"
    options.serverChatCharacterId = 101
    options.serverChatAssistantKind = "character"
    options.serverChatAssistantId = "101"
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }
    options.selectedAssistant = {
      kind: "character", id: "101", name: "Stream Character",
      metadata: { selectionMode: "tracked" }
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
      }),
      expect.objectContaining({
        requestScope: expect.objectContaining({
          config: expect.objectContaining({
            serverUrl: "http://127.0.0.1:8000",
            authMode: "single-user"
          }),
          userId: null
        }),
        signal: expect.any(AbortSignal)
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
    options.serverChatCharacterId = 101
    options.serverChatAssistantKind = "character"
    options.serverChatAssistantId = "101"
    options.selectedCharacter = {
      id: 101,
      name: "Stream Character",
      avatar_url: ""
    }
    options.selectedAssistant = {
      kind: "character", id: "101", name: "Stream Character",
      metadata: { selectionMode: "tracked" }
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
