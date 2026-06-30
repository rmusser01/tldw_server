// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"
import type { MessageMetadataExtra } from "@/store/option"

const {
  addChatMessageMock,
  createChatMock,
  normalChatModeMock,
  saveMessageOnSuccessMock,
  storageValues,
  storeOptionState
} = vi.hoisted(() => ({
  addChatMessageMock: vi.fn(async () => ({ id: "server-message-1" })),
  createChatMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  saveMessageOnSuccessMock: vi.fn(async () => "history-1"),
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
  createSaveMessageOnSuccess: vi.fn(() => saveMessageOnSuccessMock),
  createSaveMessageOnError: vi.fn(() => vi.fn(async () => "history-1"))
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

vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: vi.fn(() => [])
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
    chatSettings: {},
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
    initialize: vi.fn(async () => null),
    getMessage: vi.fn(async () => ({ version: 1 })),
    editMessage: vi.fn(async () => null)
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
  historyId: "history-1",
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
  serverChatId: "chat-1",
  serverChatTitle: "Dynamic UI Action Chat",
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
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
  ensureServerChatHistoryId: vi.fn(async () => "history-1"),
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
  selectedAssistant: null,
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

describe("useChatActions dynamic UI action integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageValues.clear()
    storeOptionState.value = { selectedModel: "deepseek-chat" }
    normalChatModeMock.mockImplementation(
      async (
        message: string,
        image: string,
        isRegenerate: boolean,
        _messages: any[],
        _history: any[],
        _signal: AbortSignal,
        params: any
      ) => {
        await params.saveMessageOnSuccess({
          historyId: "history-1",
          setHistoryId: params.setHistoryId,
          isRegenerate,
          selectedModel: params.selectedModel,
          message,
          image,
          fullText: "Action received.",
          source: [],
          assistantMessageId: "assistant-response-1",
          modelId: params.selectedModel,
          reasoning_time_taken: 0,
          saveToDb: false,
          conversationId: "chat-1",
          userMetadataExtra: params.userMetadataExtra
        })
      }
    )
  })

  it("persists dynamic UI action provenance through the normal submit path", async () => {
    const metadata: MessageMetadataExtra = {
      dynamic_ui_action: {
        renderer: "openui",
        sourceMessageId: "assistant-1",
        actionId: "survey",
        actionType: "submit",
        values: { answer: "yes" },
        submittedAt: "2026-06-01T00:00:00.000Z"
      }
    }
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "OpenUI action: submit survey\n\nSubmitted values:\n- answer: yes",
        image: "",
        userMetadataExtra: metadata
      })
    })

    expect(normalChatModeMock).toHaveBeenCalledWith(
      expect.any(String),
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        userMetadataExtra: metadata
      })
    )
    expect(saveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({ userMetadataExtra: metadata })
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "chat-1",
      expect.objectContaining({
        role: "user",
        metadata_extra: metadata
      })
    )
  })
})
