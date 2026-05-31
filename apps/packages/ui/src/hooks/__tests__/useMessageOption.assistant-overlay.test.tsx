import React from "react"
import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const {
  storeState,
  chatBaseState,
  selectedAssistantState,
  chatSettingsState,
  setSelectedAssistantSpy,
  lastUseChatActionsArgs,
  defaultRagSettings
} = vi.hoisted(() => {
  const state: Record<string, any> = {
    selectedModel: null,
    setSelectedModel: vi.fn(),
    webSearch: false,
    setWebSearch: vi.fn(),
    toolChoice: "none",
    setToolChoice: vi.fn(),
    isSearchingInternet: false,
    setIsSearchingInternet: vi.fn(),
    queuedMessages: [],
    addQueuedMessage: vi.fn(),
    setQueuedMessages: vi.fn(),
    clearQueuedMessages: vi.fn(),
    selectedKnowledge: null,
    setSelectedKnowledge: vi.fn(),
    temporaryChat: false,
    setTemporaryChat: vi.fn(),
    documentContext: null,
    setDocumentContext: vi.fn(),
    uploadedFiles: [],
    setUploadedFiles: vi.fn(),
    contextFiles: [],
    setContextFiles: vi.fn(),
    actionInfo: null,
    setActionInfo: vi.fn(),
    fileRetrievalEnabled: false,
    setFileRetrievalEnabled: vi.fn(),
    ragMediaIds: null,
    setRagMediaIds: vi.fn(),
    ragSearchMode: "hybrid",
    setRagSearchMode: vi.fn(),
    ragTopK: 8,
    setRagTopK: vi.fn(),
    ragEnableGeneration: true,
    setRagEnableGeneration: vi.fn(),
    ragEnableCitations: true,
    setRagEnableCitations: vi.fn(),
    ragSources: [],
    setRagSources: vi.fn(),
    ragAdvancedOptions: {},
    setRagAdvancedOptions: vi.fn(),
    ragPinnedResults: [],
    setRagPinnedResults: vi.fn(),
    serverChatId: "chat-1",
    setServerChatId: vi.fn(),
    serverChatTitle: "Tracked conversation",
    setServerChatTitle: vi.fn(),
    serverChatCharacterId: null,
    setServerChatCharacterId: vi.fn(),
    serverChatAssistantKind: null,
    setServerChatAssistantKind: vi.fn(),
    serverChatAssistantId: null,
    setServerChatAssistantId: vi.fn(),
    serverChatPersonaMemoryMode: null,
    setServerChatPersonaMemoryMode: vi.fn(),
    serverChatMetaLoaded: true,
    setServerChatMetaLoaded: vi.fn(),
    serverChatLoadState: "loaded",
    setServerChatLoadState: vi.fn(),
    serverChatLoadError: null,
    setServerChatLoadError: vi.fn(),
    serverChatState: null,
    setServerChatState: vi.fn(),
    serverChatVersion: 1,
    setServerChatVersion: vi.fn(),
    serverChatTopic: null,
    setServerChatTopic: vi.fn(),
    serverChatClusterId: null,
    setServerChatClusterId: vi.fn(),
    serverChatSource: null,
    setServerChatSource: vi.fn(),
    serverChatExternalRef: null,
    setServerChatExternalRef: vi.fn(),
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    replyTarget: null,
    clearReplyTarget: vi.fn()
  }

  const chatBaseState = {
    messages: [
      {
        id: "msg-1",
        role: "assistant",
        message: "Existing reply",
        isBot: true,
        sources: []
      }
    ],
    setMessages: vi.fn(),
    history: [
      {
        role: "assistant",
        content: "Existing reply"
      }
    ],
    setHistory: vi.fn(),
    streaming: false,
    setStreaming: vi.fn(),
    isFirstMessage: false,
    setIsFirstMessage: vi.fn(),
    historyId: "history-1",
    setHistoryId: vi.fn(),
    isLoading: false,
    setIsLoading: vi.fn(),
    isProcessing: false,
    setIsProcessing: vi.fn(),
    chatMode: "normal",
    setChatMode: vi.fn(),
    isEmbedding: false,
    setIsEmbedding: vi.fn(),
    selectedQuickPrompt: null,
    setSelectedQuickPrompt: vi.fn(),
    selectedSystemPrompt: null,
    setSelectedSystemPrompt: vi.fn(),
    useOCR: false,
    setUseOCR: vi.fn()
  }

  const selectedAssistantState = {
    current: {
      kind: "persona",
      id: "overlay-1",
      name: "Overlay One"
    } as Record<string, unknown> | null
  }
  const chatSettingsState = {
    current: {
      assistantOverlay: {
        kind: "persona",
        id: "overlay-1",
        name: "Overlay One",
        avatar_url: null,
        system_prompt_snapshot: "Snapshot one",
        updatedAt: "2026-05-22T12:00:00.000Z"
      }
    } as Record<string, unknown> | null
  }

  return {
    storeState: state,
    chatBaseState,
    selectedAssistantState,
    chatSettingsState,
    setSelectedAssistantSpy: vi.fn(),
    lastUseChatActionsArgs: { value: null as Record<string, unknown> | null },
    defaultRagSettings: {
      top_k: 8,
      min_score: 0.2,
      enable_reranking: true
    }
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: vi.fn()
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key
  })
}))

vi.mock("@/context", () => ({
  usePageAssist: () => ({
    controller: null,
    setController: vi.fn()
  })
}))

vi.mock("@/store/webui", () => ({
  useWebUI: () => ({
    ttsEnabled: false
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  })
}))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => chatBaseState
}))

vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => vi.fn()
}))

vi.mock("@/hooks/chat/useServerChatHistoryId", () => ({
  useServerChatHistoryId: () => ({
    ensureServerChatHistoryId: vi.fn()
  })
}))

vi.mock("@/hooks/chat/useServerChatLoader", () => ({
  useServerChatLoader: vi.fn()
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: chatSettingsState.current,
    updateSettings: vi.fn(),
    chatKey: "server:chat-1"
  })
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/hooks/chat/useCompareMode", () => ({
  useCompareMode: () => ({
    compareMode: false,
    setCompareMode: vi.fn(),
    compareFeatureEnabled: false,
    setCompareFeatureEnabled: vi.fn(),
    compareSelectedModels: [],
    setCompareSelectedModels: vi.fn(),
    compareSelectionByCluster: {},
    setCompareSelectionForCluster: vi.fn(),
    compareActiveModelsByCluster: {},
    setCompareActiveModelsForCluster: vi.fn(),
    compareParentByHistory: {},
    setCompareParentForHistory: vi.fn(),
    compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(),
    compareContinuationModeByCluster: {},
    setCompareContinuationModeForCluster: vi.fn(),
    compareSplitChats: {},
    setCompareSplitChat: vi.fn(),
    compareMaxModels: 4,
    setCompareMaxModels: vi.fn(),
    compareModeActive: false,
    markCompareHistoryCreated: vi.fn()
  })
}))

vi.mock("@/hooks/chat/useChatActions", () => ({
  useChatActions: (args: Record<string, unknown>) => {
    lastUseChatActionsArgs.value = args
    return {
      onSubmit: vi.fn(),
      sendPerModelReply: vi.fn(),
      regenerateLastMessage: vi.fn(),
      stopStreamingRequest: vi.fn(),
      editMessage: vi.fn(),
      deleteMessage: vi.fn(),
      toggleMessagePinned: vi.fn(),
      createChatBranch: vi.fn(),
      createCompareBranch: vi.fn()
    }
  }
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [selectedAssistantState.current, setSelectedAssistantSpy]
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [25]
}))

vi.mock("@/services/rag/unified-rag", () => ({
  DEFAULT_RAG_SETTINGS: defaultRagSettings,
  toRagAdvancedOptions: vi.fn((value) => value || {})
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ apiProvider: undefined })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: Record<string, unknown>) => unknown) =>
    selector ? selector(storeState) : storeState
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    [defaultValue, vi.fn(), { isLoading: false }] as const
}))

import { useMessageOption } from "@/hooks/useMessageOption"

describe("useMessageOption assistant overlay changes", () => {
  beforeEach(() => {
    storeState.serverChatId = "chat-1"
    storeState.serverChatAssistantKind = null
    storeState.serverChatAssistantId = null
    storeState.serverChatCharacterId = null
    selectedAssistantState.current = {
      kind: "persona",
      id: "overlay-1",
      name: "Overlay One"
    }
    chatSettingsState.current = {
      assistantOverlay: {
        kind: "persona",
        id: "overlay-1",
        name: "Overlay One",
        avatar_url: null,
        system_prompt_snapshot: "Snapshot one",
        updatedAt: "2026-05-22T12:00:00.000Z"
      }
    }
    chatBaseState.messages = [
      {
        id: "msg-1",
        role: "assistant",
        message: "Existing reply",
        isBot: true,
        sources: []
      }
    ]
    chatBaseState.history = [
      {
        role: "assistant",
        content: "Existing reply"
      }
    ]
    chatBaseState.historyId = "history-1"
    chatBaseState.setMessages.mockReset()
    chatBaseState.setHistory.mockReset()
    chatBaseState.setHistoryId.mockReset()
    storeState.setServerChatId.mockReset()
  })

  it("does not clear the loaded conversation when the overlay selection changes", () => {
    const { rerender } = renderHook(() => useMessageOption())

    selectedAssistantState.current = {
      kind: "persona",
      id: "overlay-2",
      name: "Overlay Two"
    }
    chatSettingsState.current = {
      assistantOverlay: {
        kind: "persona",
        id: "overlay-2",
        name: "Overlay Two",
        avatar_url: null,
        system_prompt_snapshot: "Snapshot two",
        updatedAt: "2026-05-22T12:01:00.000Z"
      }
    }
    rerender()

    expect(storeState.setServerChatId).not.toHaveBeenCalledWith(null)
    expect(chatBaseState.setMessages).not.toHaveBeenCalledWith([])
    expect(chatBaseState.setHistory).not.toHaveBeenCalledWith([])
    expect(chatBaseState.setHistoryId).not.toHaveBeenCalledWith(null)
  })
})
