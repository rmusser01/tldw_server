import React from "react"
import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const ownerMeta = vi.hoisted(() => ({ current: undefined as { assistantKey: string; isLoading: boolean; isCurrent: () => boolean } | undefined }))

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
  useSelectedAssistant: () => [selectedAssistantState.current, setSelectedAssistantSpy, ownerMeta.current]
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
  useStoreMessageOption: Object.assign((selector?: (state: Record<string, unknown>) => unknown) =>
    selector ? selector(storeState) : storeState, { getState: () => storeState })
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
    lastUseChatActionsArgs.value = null
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

  it("preserves workspace assistant provenance after server chat metadata arrives", () => {
    selectedAssistantState.current = null
    chatSettingsState.current = null
    storeState.serverChatId = null
    storeState.serverChatAssistantKind = null
    storeState.serverChatAssistantId = null
    storeState.serverChatPersonaMemoryMode = null
    chatBaseState.messages = []
    chatBaseState.history = []
    chatBaseState.historyId = null
    const inheritedAssistant = {
      kind: "persona" as const,
      id: "workspace-helper",
      name: "Workspace Helper",
      metadata: {
        selectionMode: "tracked",
        source: "workspace",
        personaMemoryMode: "read_write"
      }
    }

    const { result, rerender } = renderHook(() =>
      useMessageOption({
        inheritedAssistant,
        inheritedPersonaMemoryMode: "read_write"
      })
    )

    expect(result.current).toEqual(
      expect.objectContaining({
        selectedAssistant: expect.objectContaining({
          id: "workspace-helper",
          name: "Workspace Helper"
        }),
        selectedAssistantSource: "workspace"
      })
    )

    storeState.serverChatId = "workspace-chat-1"
    storeState.serverChatAssistantKind = "persona"
    storeState.serverChatAssistantId = "workspace-helper"
    storeState.serverChatPersonaMemoryMode = "read_write"
    chatBaseState.messages = [
      {
        id: "msg-1",
        role: "user",
        message: "Use workspace persona",
        isBot: false,
        sources: []
      }
    ]
    chatBaseState.history = [
      {
        role: "user",
        content: "Use workspace persona"
      }
    ]
    rerender()

    expect(result.current).toEqual(
      expect.objectContaining({
        selectedAssistant: expect.objectContaining({
          id: "workspace-helper",
          name: "Workspace Helper"
        }),
        selectedAssistantSource: "workspace"
      })
    )
  })
  const canonicalCard = (suffix = "") => ({
    kind: "character",
    id: "42",
    name: `Helpful AI Assistant${suffix}`,
    avatar_url: `https://example.test/assistant${suffix}.png`,
    system_prompt: `Owned prompt${suffix}`,
    greeting: `Owned greeting${suffix}`,
    extensions: { reviewOwned: suffix || "initial" }
  })
  const seedCanonicalCard = () => {
    const lease = { current: true }
    ownerMeta.current = {
      assistantKey: "owner:alice",
      isLoading: false,
      isCurrent: () => lease.current
    }
    storeState.serverChatId = "owned-chat"
    storeState.serverChatAssistantKind = "character"
    storeState.serverChatCharacterId = 42
    storeState.serverChatAssistantId = "42"
    selectedAssistantState.current = canonicalCard()
    chatSettingsState.current = null
    setSelectedAssistantSpy.mockClear()
    return lease
  }
  it.each(["cleared", "foreign"])(
    "keeps full canonical display after %s shared preference",
    (mirror) => {
      seedCanonicalCard()
      const view = renderHook(() => useMessageOption())
      try {
        expect(view.result.current.effectiveAssistantState.displayName).toBe(
          "Helpful AI Assistant"
        )
        selectedAssistantState.current =
          mirror === "cleared"
            ? null
            : { kind: "character", id: "99", name: "Foreign card" }
        view.rerender()
        expect
          .soft(view.result.current.effectiveAssistantState.displayName)
          .toBe("Helpful AI Assistant")
        expect
          .soft(view.result.current.selectedAssistant)
          .toMatchObject(canonicalCard())
        expect
          .soft(lastUseChatActionsArgs.value?.selectedAssistant)
          .toMatchObject(canonicalCard())
        expect(setSelectedAssistantSpy).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        ownerMeta.current = undefined
      }
    }
  )
  it("plain transition immediately drops canonical metadata", () => {
    seedCanonicalCard()
    const view = renderHook(() => useMessageOption())
    try {
      selectedAssistantState.current = null
      storeState.serverChatId = null
      storeState.serverChatCharacterId = null
      storeState.serverChatAssistantKind = null
      storeState.serverChatAssistantId = null
      view.rerender()
      expect(view.result.current.effectiveAssistantState.mode).toBe("plain")
      expect(view.result.current.selectedAssistant).toBeNull()
    } finally {
      view.unmount()
      ownerMeta.current = undefined
    }
  })
  it.each(["conversation", "canonical-id"])(
    "%s change and return does not restore stale metadata",
    (boundary) => {
      seedCanonicalCard()
      const view = renderHook(() => useMessageOption())
      try {
        selectedAssistantState.current = null
        if (boundary === "conversation")
          storeState.serverChatId = "another-chat"
        else {
          storeState.serverChatCharacterId = 99
          storeState.serverChatAssistantId = "99"
        }
        view.rerender()
        expect(
          view.result.current.effectiveAssistantState.displayName
        ).not.toBe("Helpful AI Assistant")
        storeState.serverChatId = "owned-chat"
        storeState.serverChatCharacterId = 42
        storeState.serverChatAssistantId = "42"
        view.rerender()
        expect(
          view.result.current.effectiveAssistantState.displayName
        ).not.toBe("Helpful AI Assistant")
      } finally {
        view.unmount()
        ownerMeta.current = undefined
      }
    }
  )
  it.each(["revoked", "owner-aba"])(
    "%s invalidates the captured lease even with the same key",
    (boundary) => {
      const lease = seedCanonicalCard()
      const view = renderHook(() => useMessageOption())
      try {
        lease.current = false
        selectedAssistantState.current = null
        if (boundary === "owner-aba")
          ownerMeta.current = {
            assistantKey: "owner:alice",
            isLoading: false,
            isCurrent: () => true
          }
        view.rerender()
        expect(
          view.result.current.effectiveAssistantState.displayName
        ).not.toBe("Helpful AI Assistant")
        expect(setSelectedAssistantSpy).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        ownerMeta.current = undefined
      }
    }
  )
  it("a matching card update replaces retained full metadata", () => {
    seedCanonicalCard()
    const view = renderHook(() => useMessageOption())
    try {
      selectedAssistantState.current = canonicalCard(" updated")
      view.rerender()
      expect(view.result.current.selectedAssistant).toMatchObject(
        canonicalCard(" updated")
      )
      selectedAssistantState.current = null
      view.rerender()
      expect(view.result.current.selectedAssistant).toMatchObject(
        canonicalCard(" updated")
      )
    } finally {
      view.unmount()
      ownerMeta.current = undefined
    }
  })

  it("keeps the retained card when another tab publishes a failed lookup placeholder", () => {
    seedCanonicalCard()
    const view = renderHook(() => useMessageOption())
    try {
      selectedAssistantState.current = null
      view.rerender()
      selectedAssistantState.current = {
        kind: "character", id: "42", name: "Assistant",
        avatar_url: null, system_prompt: null, greeting: null, extensions: null
      }
      view.rerender()
      expect(view.result.current.selectedAssistant).toMatchObject(canonicalCard())
      selectedAssistantState.current = null
      view.rerender()
      expect(view.result.current.selectedAssistant).toMatchObject(canonicalCard())
    } finally { view.unmount(); ownerMeta.current = undefined }
  })

  it("accepts an updated full Character card actually named Assistant", () => {
    seedCanonicalCard()
    const view = renderHook(() => useMessageOption())
    try {
      selectedAssistantState.current = { ...canonicalCard(" updated"), name: "Assistant" }
      view.rerender()
      selectedAssistantState.current = null
      view.rerender()
      expect(view.result.current.selectedAssistant).toMatchObject({
        ...canonicalCard(" updated"), name: "Assistant"
      })
    } finally { view.unmount(); ownerMeta.current = undefined }
  })

})
