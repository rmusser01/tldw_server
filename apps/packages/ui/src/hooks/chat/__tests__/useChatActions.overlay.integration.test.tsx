// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"

const {
  createChatMock,
  normalChatModeMock,
  streamCharacterChatCompletionMock,
  syncChatSettingsForServerChatMock
} =
  vi.hoisted(() => ({
    createChatMock: vi.fn(),
    normalChatModeMock: vi.fn(),
    streamCharacterChatCompletionMock: vi.fn(),
    syncChatSettingsForServerChatMock: vi.fn(async () => null)
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
        "history-overlay"
  ),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-overlay"
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
    getState: () => ({ selectedModel: "deepseek-chat" as string | null })
  }
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: syncChatSettingsForServerChatMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    initialize: vi.fn(async () => null)
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
  historyId: "history-overlay",
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
  serverChatId: null,
  serverChatTitle: null,
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
  serverChatMetaLoaded: false,
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
  ensureServerChatHistoryId: vi.fn(async () => "history-overlay"),
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
    id: "char-overlay",
    name: "Overlay Guide",
    system_prompt: "Overlay guide prompt"
  },
  selectedAssistant: {
    kind: "character" as const,
    id: "char-overlay",
    name: "Overlay Guide",
    system_prompt: "Overlay guide prompt",
    metadata: {
      selectionMode: "overlay"
    }
  },
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

describe("useChatActions overlay integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    normalChatModeMock.mockResolvedValue(undefined)
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
  })

  it("routes overlay character selections through normal chat mode without creating tracked chats", async () => {
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Overlay hello",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).toHaveBeenCalledTimes(1)
    expect(normalChatModeMock.mock.calls[0]?.[6]).toEqual(
      expect.objectContaining({
        assistantIdentity: {
          name: "Overlay Guide",
          avatarUrl: undefined
        },
        overlaySystemPrompt: "Overlay guide prompt"
      })
    )
  })

  it("reconciles scratch overlay settings only when a send links a new server conversation", async () => {
    const options = createHookOptions()
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
      }
      await params.saveMessageOnSuccess({
        historyId: null,
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Overlay hello",
        image: "",
        fullText: "Overlay reply",
        source: [],
        conversationId: "server-chat-42"
      })
    })

    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Overlay hello",
        image: ""
      })
    })

    expect(syncChatSettingsForServerChatMock).toHaveBeenCalledWith({
      historyId: "history-overlay",
      serverChatId: "server-chat-42",
      allowScratchFallback: true
    })
  })
})
