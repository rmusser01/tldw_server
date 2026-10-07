// @vitest-environment jsdom
import React from "react"
import { getLatestChatErrorBannerEntry } from "@/components/Option/Playground/PlaygroundChatErrorBanner"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"

const {
  addChatMessageMock,
  createChatMock,
  detectCharacterMoodMock,
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock,
  normalChatModeMock,
  resolveVisualIdentityBindingMock
} = vi.hoisted(() => ({
  addChatMessageMock: vi.fn<(...args: any[]) => Promise<any>>(async () => ({
    id: "user-server-1",
    version: 1
  })),
  createChatMock: vi.fn(),
  detectCharacterMoodMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn(async () => ({
    assistant_message_id: "assistant-server-1",
    version: 1
  })),
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
  draftFields: null as (() => Record<string, unknown>) | null,
  value: {
    selectedModel: "deepseek-chat" as string | null,
    toolChoice: "auto" as "auto" | "none" | "required",
    serverChatId: null as string | null,
    serverChatCharacterId: null as string | number | null,
    serverChatAssistantKind: null as "character" | "persona" | null,
    serverChatSource: null as string | null
  }
}))

vi.mock("@/hooks/chat-modes/normalChatMode", async original => ({
  ...await original<typeof import("@/hooks/chat-modes/normalChatMode")>(),
  normalChatMode: (...args: any[]) => normalChatModeMock(...args)
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

vi.mock("@/db/dexie/helpers", async (original) => ({
  formatSelectedHistory: (await original<any>()).formatSelectedHistory,
  generateID: vi.fn(() => crypto.randomUUID()),
  saveHistory: (...args: any[]) => h1.saveHistory(...args),
  saveMessage: vi.fn(),
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(async () => null),
  removeMessageByIndex: vi.fn(),
  removeMessageById: vi.fn(),
  updateMessageById: vi.fn(),
  formatToChatHistory: vi.fn((items: unknown) => items),
  formatToMessage: vi.fn((items: unknown) => items),
  getSessionFiles: vi.fn(async () => []),
  getPromptById: vi.fn(async () => null)
}))

vi.mock("@/db/dexie/server-chat-mirror", async (original) => ({
  ...(await original<any>()),
  removeAcknowledgedServerMirrorMessage: vi.fn(async () => undefined)
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null)
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

vi.mock("@/store/option", async original => {
  const actual = await original<typeof import("@/store/option")>()
  return { ...actual, useStoreMessageOption: Object.assign(
    (select: (state: ReturnType<typeof actual.useStoreMessageOption.getState>) => unknown) => actual.useStoreMessageOption(select),
    { getState: () => ({ ...actual.useStoreMessageOption.getState(), ...messageStoreState.value, ...messageStoreState.draftFields?.() }) }
  ) }
})

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    addChatMessage: addChatMessageMock,
    getChat: vi.fn(async (id: string) => ({ id, title: "Actual owned chat", character_id: "12" })),
    getChatSettings: vi.fn(async () => ({ settings: null })),
    initialize: vi.fn(async () => null),
    getCharacter: h1.live,
    captureHistorySelection: (...args: any[]) => h1.capture(...args),
    streamChatCompletion: (...args: any[]) => h1.wire(...args),
    getConfig: vi.fn(async () => null),
    deleteMessage: vi.fn(async () => undefined),
    getMessage: vi.fn(async () => ({ version: 1 })),
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

const h1 = vi.hoisted(() => ({
  connected: false,
  create: vi.fn(),
  live: vi.fn(() => {
    throw new Error("live card deleted")
  }),
  auth: new AbortController(),
  controller: null as any,
  capture: vi.fn(),
  wire: vi.fn(),
  recover: vi.fn(),
  dismiss: vi.fn(),
  saveHistory: vi.fn(),
  localCapture: vi.fn(),
  localAppend: vi.fn(),
  localSettle: vi.fn(),
  beforeModel: vi.fn(),
  lookupRecoveries: vi.fn(),
  lookupAdmission: vi.fn()
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: { getState: () => ({ state: { isConnected: h1.connected, phase: "connected", mode: "normal" } }) }
}))
vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<any>()),
  useHistorySelectionContext: () => h1.controller
}))
vi.mock("@/db/dexie/history-selection", async (original) => {
  const actual = await original<any>()
  return {
  ...actual,
  loadHistoryTurnRecoveries: (...args: unknown[]) => h1.lookupRecoveries(...args) ?? actual.loadHistoryTurnRecoveries(...args),
  loadLocalHistoryAdmission: (...args: unknown[]) => h1.lookupAdmission(...args) ?? actual.loadLocalHistoryAdmission(...args),
  saveHistoryTurnRecovery: h1.recover,
  dismissHistoryTurnRecovery: h1.dismiss,
  captureLocalHistorySnapshot: h1.localCapture,
  appendLocalSelectedUser: h1.localAppend,
  settleLocalAcceptedAssistant: h1.localSettle
  }
})
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<any>()),
  resolveServicePromptScope: async () => ({
    config: { serverUrl: "https://server.test", authMode: "multi-user" },
    userId: "alice"
  }),
  subscribeToServicePromptConfigChanges: () => () => {},
  loadServicePromptSnapshot: vi.fn(async () => ({
    requestScope: {
      config: { serverUrl: "https://server.test", authMode: "multi-user" },
      userId: "alice"
    },
    definitions: {},
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: h1.auth.signal,
    release: vi.fn()
  }))
}))
vi.mock("@/services/tldw-server", async (original) => ({
  ...(await original<any>()),
  systemPromptForNonRagOption: async () => "system",
  getWebSearchPrompt: vi.fn()
}))
vi.mock("@/utils/human-message", async () => {
  const { HumanMessage } = await import("@/types/messages")
  return {
    humanMessageFormatter: async ({ content }: any) =>
      new HumanMessage({ content })
  }
})
vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: async (rows: any) => rows
}))
vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return {
    pageAssistModel: async (options: any) => {
      await h1.beforeModel()
      return new ChatTldw({
        ...options,
        temperature: 0.23,
        supportsMultimodal: true,
        slashCommandInjectionMode: (
          await import("@/store/model")
        ).useStoreChatModelSettings.getState().slashCommandInjectionMode
      })
    }
  }
})
vi.mock("@/hooks/utils/messageHelpers", async (original) => original())

const captureFor = (view: any) => {
  const nodes = [
    {
      id: "u-old",
      revision: "r1",
      parent_id: null,
      role: "user",
      settled: true
    },
    {
      id: "a1",
      revision: "r2",
      parent_id: "u-old",
      role: "assistant",
      settled: true
    },
    {
      id: "a2",
      revision: "r3",
      parent_id: "u-old",
      role: "assistant",
      settled: true
    }
  ]
  const rows =
    view.cursor.kind === "empty"
      ? []
      : nodes.filter(
          (row) => row.id === "u-old" || row.id === view.cursor.message_id
        )
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      fences: { conversation: "1", history: "1", settings: "1" },
      nodes,
      source_digest: "source",
      storage_context_digest: "storage",
      interpretation_status: { kind: "parent_graph_v1" }
    },
    rows,
    selected_content: rows.map((row) => ({
      id: row.id,
      revision: row.revision,
      message: row.id === "u-old" ? "old question" : "same answer",
      images: []
    })),
    view,
    purpose: "send",
    storage_context_digest: "storage"
  }
}
const makeController = () => {
  let current: any = {
    owner: {
      kind: "native",
      conversation_id: "tracked-chat-1",
      request_scope: {
        config: { serverUrl: "https://server.test", authMode: "multi-user" },
        userId: "alice"
      },
      validate_lease: () => true
    },
    view: {
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      view_session_id: "origin",
      selection_revision: 1,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "a1" }
    },
    bookmarkScope: { profile_id: "profile", client_session_id: "client" },
    status: "ready"
  }
  current.capture = captureFor(current.view)
  return {
    getCurrent: () => current,
    fence: () => {
      const origin = current
      return () => current === origin
    },
    followResult: vi.fn(async () => true),
    refreshRecovery: vi.fn(async () => {}),
    navigate: () => {
      current = {
        ...current,
        view: {
          ...current.view,
          conversation_id: "other",
          view_session_id: "other"
        }
      }
    }
  }
}

afterEach(() => vi.restoreAllMocks())

beforeEach(async () => {
  h1.connected = false
  h1.lookupRecoveries.mockReset()
  h1.lookupAdmission.mockReset()
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  const lockTails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", { configurable: true, value: {
    request: (key: string, work: () => unknown) => {
      const next = (lockTails.get(key) ?? Promise.resolve()).then(work)
      lockTails.set(key, next.catch(() => undefined))
      return next
    }
  } })
  const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  draftStore.getState().startNewChatDraft()
  messageStoreState.draftFields = () => {
    const state = draftStore.getState()
    return { chatDraftAddress: state.chatDraftAddress, chatDraftRevision: state.chatDraftRevision, setChatDraftAddress: state.setChatDraftAddress }
  }
  h1.beforeModel.mockReset()
  messageStoreState.value.selectedModel = "deepseek-chat"
  messageStoreState.value.toolChoice = "auto"
  h1.auth = new AbortController()
  const actual = await vi.importActual<any>("@/hooks/chat-modes/normalChatMode")
  normalChatModeMock.mockImplementation(actual.normalChatMode)
  h1.controller = makeController()
  h1.capture.mockImplementation(async (_id, request) =>
    captureFor(request.view)
  )
  addChatMessageMock.mockImplementation(async (_id: any, body: any) =>
    body.role === "user"
      ? {
          id: body.id,
          tldw_history_admission_v1: {
            version: 1,
            owner_key: "native-key",
            conversation_id: "tracked-chat-1",
            input_message_id: body.id,
            input_message_revision: "accepted-r",
            selection_digest: body.tldw_history_selection_v1.selection_digest,
            messages: body.tldw_history_selection_v1.messages,
            originating_selection_revision: 1
          }
        }
      : { id: body.id }
  )
  h1.wire.mockImplementation(async function* () {
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
})
const ordinaryOptions = () => ({
  ...createHookOptions(),
  selectedCharacter: null,
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
  messages: [
    { id: "u-old", isBot: false, message: "old question", sources: [] },
    { id: "a2", isBot: true, message: "wrong latest", sources: [] }
  ],
  history: [
    { role: "user", content: "old question" },
    { role: "assistant", content: "wrong latest" }
  ]
})

it("edits an ordinary local message while the mounted history controller is idle", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.editMessage(0, "edited", false, false)
  })
  expect(helpers.updateMessageById).toHaveBeenCalledWith(
    "history-character", "plain-message", "edited"
  )
})

it("deletes an ordinary local message while the mounted history controller is idle", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.deleteMessage(0)
  })
  expect(helpers.removeMessageById).toHaveBeenCalledWith(
    "history-character", "plain-message"
  )
})

it("removes a server-backed message from its exact mirror after canonical deletion", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const mirror = await import("@/db/dexie/server-chat-mirror")
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  const current = h1.controller.getCurrent()
  current.capture = null
  const options = {
    ...ordinaryOptions(),
    serverChatId: "tracked-chat-1",
    messages: [{ id: "local-message", serverMessageId: "server-message", serverMessageVersion: 1, isBot: true, message: "answer", sources: [] }]
  }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => { await result.current.deleteMessage(0) })
  expect(api.deleteMessage).toHaveBeenCalledWith("server-message", 1, "tracked-chat-1")
  expect(mirror.removeAcknowledgedServerMirrorMessage).toHaveBeenCalledWith({
    historyId: "history-character", chatId: "tracked-chat-1",
    localMessageId: "local-message", serverMessageId: "server-message"
  })
  expect(helpers.removeMessageById).not.toHaveBeenCalled()
  expect(options.setMessages).toHaveBeenCalledWith([])
})

it("routes ordinary regeneration while the mounted history controller is idle", async () => {
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  normalChatModeMock.mockResolvedValue({ state: "submitted" } as any)
  const options = ordinaryOptions()
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "again", image: "", isRegenerate: true })
  })
  expect(options.notification.error).not.toHaveBeenCalled()
  expect(normalChatModeMock).toHaveBeenCalled()
})

it("mounted ordinary submit sends the selected A1 and settles once under its pre-admitted input", async () => {
  const options = ordinaryOptions()
  h1.wire.mockImplementation(async function* () {
    expect(
      addChatMessageMock.mock.calls.map(([, body]: any) => body.role)
    ).toEqual(["user"])
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  const request = h1.wire.mock.calls[0][0]
  expect(request.messages).toContainEqual({
    role: "assistant",
    content: "same answer"
  })
  expect(JSON.stringify(request.messages)).not.toContain("wrong latest")
  expect(request.save_to_db).toBe(false)
  expect(request.conversation_id).toBeUndefined()
  const [user, assistant] = addChatMessageMock.mock.calls.map(
    ([, body]: any) => body
  )
  expect(
    user.tldw_history_selection_v1.messages.map((row: any) => row.id)
  ).toEqual(["u-old", "a1"])
  expect(assistant.tldw_history_admission_v1.input_message_id).toBe(user.id)
  expect(assistant.parent_message_id).toBe(user.id)
  expect(addChatMessageMock).toHaveBeenCalledTimes(2)
})

it.each(["model", "tools"])("rejects an ordinary playground send after its global %s selector changes during preparation", async changed => {
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  h1.beforeModel.mockImplementation(() => {
    if (changed === "model") messageStoreState.value.selectedModel = "changed-model"
    else messageStoreState.value.toolChoice = "none"
  })
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it.each(["model", "tools"])("keeps an explicit playground %s override after its unrelated global selector changes", async overridden => {
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  h1.beforeModel.mockImplementation(() => {
    if (overridden === "model") messageStoreState.value.selectedModel = "changed-model"
    else messageStoreState.value.toolChoice = "required"
  })
  await act(async () => {
    await result.current.onSubmit({
      message: "next", image: "",
      requestOverrides: overridden === "model"
        ? { selectedModel: "fixed-model" }
        : { toolChoice: "none" }
    })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.wire.mock.calls[0][0].model).toBe(overridden === "model" ? "fixed-model" : "deepseek-chat")
  expect(addChatMessageMock).toHaveBeenCalledTimes(2)
})

it("mounted native character sends A1 selection and only new input, following the owner ACK", async () => {
  const options = createHookOptions()
  h1.wire.mockImplementation(async function* (request) {
    yield {
      tldw_history_admission_v1: {
        version: 1,
        owner_key: "native-key",
        conversation_id: "tracked-chat-1",
        input_message_id: "native-input",
        input_message_revision: "r",
        selection_digest: request.tldw_history_selection_v1.selection_digest,
        messages: request.tldw_history_selection_v1.messages,
        originating_selection_revision: 1
      }
    }
    yield { choices: [{ delta: { content: "native answer" } }] }
    yield {
      tldw_message_id: "native-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  const request = h1.wire.mock.calls[0][0]
  expect(request).toMatchObject({
    save_to_db: true,
    conversation_id: "tracked-chat-1",
    model: "deepseek-chat",
    api_provider: "openai"
  })
  expect(request.messages).toEqual([{ role: "user", content: "next" }])
  expect(
    request.tldw_history_selection_v1.messages.map((row: any) => row.id)
  ).toEqual(["u-old", "a1"])
  expect(h1.controller.followResult).toHaveBeenCalledWith(
    expect.objectContaining({ view_session_id: "origin" }),
    "native-result"
  )
  expect(createChatMock).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it("same-text A2 cannot substitute for the selected A1 identity", async () => {
  const options = ordinaryOptions()
  options.history[1].content = "same answer"
  options.messages[1].message = "same answer"
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(
    addChatMessageMock.mock.calls[0][1].tldw_history_selection_v1.messages.map(
      (row: any) => row.id
    )
  ).toEqual(["u-old", "a1"])
})

it("native owner capability survives display navigation while a held result settles in its origin", async () => {
  const options = ordinaryOptions()
  let displayed: any[] = options.messages
  options.setMessages.mockImplementation((next: any) => {
    displayed = typeof next === "function" ? next(displayed) : next
  })
  h1.wire.mockImplementation(async function* () {
    h1.controller.getCurrent().owner.validate_lease = () => false
    h1.controller.navigate()
    displayed = [{ id: "new-view", message: "keep" }]
    options.setHistory.mockClear()
    options.setIsProcessing.mockClear()
    yield { choices: [{ delta: { content: "old result" } }] }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(displayed).toEqual([{ id: "new-view", message: "keep" }])
  expect(addChatMessageMock.mock.calls[1].slice(0, 2)).toMatchObject([
    "tracked-chat-1",
    {
      content: "old result",
      tldw_history_admission_v1: {
        input_message_id: addChatMessageMock.mock.calls[0][1].id
      }
    }
  ])
  expect(options.setHistory).not.toHaveBeenCalled()
  expect(options.setIsProcessing).not.toHaveBeenCalledWith(true)
  expect(options.setStreaming).toHaveBeenLastCalledWith(false)
  expect(options.setAbortController).toHaveBeenLastCalledWith(null)
})

it("unknown admission after navigation retains immutable original intent and dispatches no provider", async () => {
  addChatMessageMock.mockImplementationOnce(async () => {
    h1.controller.navigate()
    throw new Error("connection lost")
  })
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    state: "unknown",
    origin_view: {
      conversation_id: "tracked-chat-1",
      view_session_id: "origin"
    },
    input_text: "next",
    selection_digest: expect.any(String),
    request_context_digest: expect.any(String)
  })
  expect(addChatMessageMock).toHaveBeenCalledTimes(1)
})

it.each(["direct", "clear cached target"])("connected fresh saved normal send adopts a verified native owner before admission (%s)", async target => {
  h1.connected = true
  const options = { ...ordinaryOptions(), historyId: target === "clear cached target" ? "cached-history" : null, serverChatId: null, messages: [], history: [] }
  let current: any = { owner: null, view: null, status: "idle" }
  h1.controller = {
    getCurrent: () => current,
    fence: () => { const origin = current; return () => current === origin },
    loadConversation: vi.fn(async (target, _reference, onLoaded) => {
      expect(target).toMatchObject({ serverChatId: "tracked-chat-1" })
      expect(options.setServerChatId).not.toHaveBeenCalled()
      current = makeController().getCurrent()
      current.view = { ...current.view, cursor: { kind: "empty" } }
      current.capture = captureFor(current.view)
      onLoaded({ owner: current.owner, view: current.view })
      return true
    }),
    followResult: vi.fn(async () => true)
  }
  createChatMock.mockResolvedValue({ id: "tracked-chat-1" })
  options.setServerChatId.mockImplementation(() => {
    expect(current.owner.kind).toBe("native")
  })
  h1.wire.mockImplementation(async function* (request) {
    expect(addChatMessageMock).toHaveBeenCalledOnce()
    expect(request.save_to_db).toBe(false)
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
  const hook = renderHook(() => useChatActions(options as any))
  await act(async () => { expect(await hook.result.current.onSubmit({ message: "first", image: "", ...(target === "clear cached target" ? { requestOverrides: { historyId: null, serverChatId: null } } : {}) })).toEqual({ status: "submitted" }) })
  const draftStore = (await vi.importActual<typeof import("@/store/option")>("@/store/option")).useStoreMessageOption
  expect(draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "tracked-chat-1" })
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(options.setServerChatId).toHaveBeenCalledWith("tracked-chat-1")
  expect(h1.saveHistory).not.toHaveBeenCalled()
  expect(addChatMessageMock).toHaveBeenCalledTimes(2)
  expect(h1.wire).toHaveBeenCalledOnce()
})

it("first durable ordinary send creates its local owner and admits before inference without a server conversation", async () => {
  const options = {
    ...ordinaryOptions(),
    historyId: null,
    serverChatId: null,
    messages: [],
    history: []
  }
  let current: any = { owner: null, view: null, status: "idle" }
  const load = vi.fn(
    async ({ historyId }: any, _reference: any, onLoaded: any) => {
      const view = {
        owner_key: "local-key",
        conversation_id: historyId,
        view_session_id: "new",
        selection_revision: 0,
        cursor: { kind: "empty" },
        interpretation: { kind: "parent_graph_v1" }
      }
      const capture = {
        ...captureFor(view),
        snapshot: {
          ...captureFor(view).snapshot,
          owner_key: "local-key",
          conversation_id: historyId,
          nodes: []
        }
      }
      current = {
        owner: {
          kind: "local",
          profile_id: "profile",
          owner_key: "local-key",
          conversation_id: historyId
        },
        view,
        status: "ready",
        capture,
        bookmarkScope: { profile_id: "profile", client_session_id: "client" }
      }
      h1.localCapture.mockResolvedValue(capture)
      onLoaded({ owner: current.owner, view: current.view })
      return true
    }
  )
  h1.controller = {
    getCurrent: () => current,
    fence: () => () => true,
    loadConversation: load,
    followResult: vi.fn(async () => true)
  }
  h1.saveHistory.mockResolvedValue({ id: "new-local" })
  h1.localAppend.mockImplementation(async (_owner, selection, input) => ({
    version: 1,
    owner_key: "local-key",
    conversation_id: "new-local",
    input_message_id: input.id,
    input_message_revision: "1",
    selection_digest: selection.selection_digest,
    messages: [],
    originating_selection_revision: 0
  }))
  h1.localSettle.mockImplementation(async (_owner, _admission, input) => input)
  h1.wire.mockImplementation(async function* () {
    expect(h1.localAppend).toHaveBeenCalledOnce()
    expect(h1.localSettle).not.toHaveBeenCalled()
    yield { choices: [{ delta: { content: "answer" } }] }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "first", image: "" })
  })
  expect(h1.localSettle).toHaveBeenCalledOnce()
  expect(h1.localSettle.mock.calls[0][2]).toMatchObject({
    history_id: "new-local",
    parent_message_id: h1.localAppend.mock.calls[0][2].id
  })
  expect(createChatMock).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
})

describe("fresh-chat submit with a selection another conversation left behind (CS-01, #3106)", () => {
  const freshOptions = () => ({
    ...ordinaryOptions(),
    historyId: null,
    serverChatId: null,
    messages: [],
    history: []
  })
  const renderActions = (options: object) =>
    renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )

  it("resets the stale selection before the turn's fence and sends only the new message", async () => {
    // The native tracked-chat-1 selection (cursor after A1) survives a New chat.
    let current: unknown = h1.controller.getCurrent()
    const order: string[] = []
    const load = vi.fn(async (
      { historyId }: { historyId: string; serverChatId?: string | null },
      _reference: unknown,
      onLoaded: (receipt: { owner: unknown; view: unknown }) => void
    ) => {
      const view = {
        owner_key: "local-key",
        conversation_id: historyId,
        view_session_id: "new",
        selection_revision: 0,
        cursor: { kind: "empty" },
        interpretation: { kind: "parent_graph_v1" }
      }
      const capture = {
        ...captureFor(view),
        snapshot: { ...captureFor(view).snapshot, owner_key: "local-key", conversation_id: historyId, nodes: [] }
      }
      const owner = { kind: "local", profile_id: "profile", owner_key: "local-key", conversation_id: historyId }
      current = {
        owner,
        view,
        status: "ready",
        capture,
        bookmarkScope: { profile_id: "profile", client_session_id: "client" }
      }
      h1.localCapture.mockResolvedValue(capture)
      onLoaded({ owner, view })
      return true
    })
    h1.controller = {
      getCurrent: () => current,
      reset: vi.fn(() => {
        order.push("reset")
        current = { owner: null, view: null, capture: null, bookmarkScope: null, status: "idle" }
      }),
      fence: () => {
        order.push("fence")
        return () => true
      },
      loadConversation: load,
      followResult: vi.fn(async () => true)
    }
    h1.saveHistory.mockResolvedValue({ id: "new-local" })
    h1.localAppend.mockImplementation(async (_owner, selection, input) => ({
      version: 1,
      owner_key: "local-key",
      conversation_id: "new-local",
      input_message_id: input.id,
      input_message_revision: "1",
      selection_digest: selection.selection_digest,
      messages: [],
      originating_selection_revision: 0
    }))
    h1.localSettle.mockImplementation(async (_owner, _admission, input) => input)
    h1.wire.mockImplementation(async function* () {
      yield { choices: [{ delta: { content: "answer" } }] }
    })
    const { result } = renderActions(freshOptions())
    // Only the new send allocates this fence; render also captures an inactive Retry source.
    order.length = 0
    await act(async () => {
      await result.current.onSubmit({ message: "unrelated", image: "" })
    })

    expect(order.slice(0, 2)).toEqual(["reset", "fence"])
    expect(h1.saveHistory).toHaveBeenCalledOnce()
    expect(load.mock.calls[0][0]).toMatchObject({ historyId: "new-local" })
    expect(load.mock.calls[0][0].serverChatId).toBeFalsy()
    expect(h1.wire).toHaveBeenCalledOnce()
    const sent = JSON.stringify(h1.wire.mock.calls[0][0].messages)
    expect(sent).toContain("unrelated")
    expect(sent).not.toContain("old question")
    expect(sent).not.toContain("same answer")
    // Nothing is admitted into the conversation the user left.
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(h1.capture).not.toHaveBeenCalled()
  })

  it("leaves a selection that is still loading in place and refuses the send without dispatching", async () => {
    const loading = { ...h1.controller.getCurrent(), capture: null, status: "loading" }
    const reset = vi.fn()
    h1.controller = {
      getCurrent: () => loading,
      reset,
      fence: () => () => true,
      loadConversation: vi.fn(),
      followResult: vi.fn(async () => true)
    }
    const { result } = renderActions(freshOptions())
    await act(async () => {
      await result.current.onSubmit({ message: "unrelated", image: "" })
    })

    expect(reset).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(h1.saveHistory).not.toHaveBeenCalled()
  })

  it("keeps the selection of a turn that addresses its conversation", async () => {
    const reset = vi.fn()
    h1.controller = { ...h1.controller, reset }
    const { result } = renderActions({ ...ordinaryOptions(), serverChatId: "tracked-chat-1" })
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })

    expect(reset).not.toHaveBeenCalled()
    expect(JSON.stringify(h1.wire.mock.calls[0][0].messages)).toContain("old question")
  })
})

it("a first-create acknowledgement after navigation never installs or dispatches into the new view", async () => {
  const options = {
    ...ordinaryOptions(),
    historyId: null,
    serverChatId: null,
    messages: [],
    history: []
  }
  let current = true
  const load = vi.fn()
  h1.controller = {
    getCurrent: () => ({ owner: null, view: null, status: "idle" }),
    fence: () => () => current,
    loadConversation: load
  }
  h1.saveHistory.mockImplementation(async () => {
    current = false
    return { id: "created-original" }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "first", image: "" })
  })
  expect(load).not.toHaveBeenCalled()
  expect(options.setHistoryId).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.localAppend).not.toHaveBeenCalled()
})

it("actual outbound history preserves ordered images and empty assistant tool calls with their result", async () => {
  const view = h1.controller.getCurrent().view
  view.cursor = { kind: "after_message", message_id: "tool-result" }
  const calls = [
    {
      id: "call-1",
      type: "function",
      function: { name: "lookup", arguments: "{}" }
    }
  ]
  h1.capture.mockImplementation(async (_id, request) => {
    const base = captureFor({
      ...request.view,
      cursor: { kind: "after_message", message_id: "a1" }
    })
    const tool = {
      id: "tool-result",
      revision: "r4",
      parent_id: "a1",
      role: "tool",
      settled: true
    }
    return {
      ...base,
      view: request.view,
      snapshot: { ...base.snapshot, nodes: [...base.snapshot.nodes, tool] },
      rows: [...base.rows, tool],
      selected_content: [
        {
          ...base.selected_content[0],
          images: [
            "data:image/png;base64,first",
            "data:image/png;base64,second"
          ]
        },
        { ...base.selected_content[1], message: "", tool_calls: calls },
        {
          id: "tool-result",
          revision: "r4",
          message: "found",
          images: [],
          extra_metadata: { tool_call_id: "call-1" }
        }
      ]
    }
  })
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][0].messages).toEqual([
    { role: "system", content: "system" },
    {
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: { url: "data:image/png;base64,first" }
        },
        {
          type: "image_url",
          image_url: { url: "data:image/png;base64,second" }
        },
        { type: "text", text: "old question" }
      ]
    },
    { role: "assistant", content: null, tool_calls: calls },
    { role: "tool", content: "found", tool_call_id: "call-1" },
    { role: "user", content: "next" }
  ])
})

it("a changed selection during prompt preparation makes zero owner admission requests", async () => {
  const { getPromptById } = await import("@/db/dexie/helpers")
  vi.mocked(getPromptById).mockImplementationOnce(async () => {
    const current = h1.controller.getCurrent()
    current.view = {
      ...current.view,
      selection_revision: 2,
      cursor: { kind: "after_message", message_id: "a2" }
    }
    return null
  })
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.capture).toHaveBeenCalledOnce()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it("a settings change during owner admission retains accepted input without provider dispatch", async () => {
  const { useStoreChatModelSettings } = await import("@/store/model")
  const append = addChatMessageMock.getMockImplementation()!
  addChatMessageMock.mockImplementationOnce(async (...args: any[]) => {
    useStoreChatModelSettings.getState().setTemperature(1.7)
    return append(...args)
  })
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(addChatMessageMock).toHaveBeenCalledTimes(1)
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    state: "accepted_unsent",
    admission: { input_message_id: addChatMessageMock.mock.calls[0][1].id }
  })
})

it.each([false, true])(
  "a deferred first owner load cannot adopt another ready conversation (load result %s)",
  async (loadResult) => {
    const options = {
      ...ordinaryOptions(),
      historyId: null,
      serverChatId: null,
      messages: [],
      history: []
    }
    const destination = makeController().getCurrent()
    let current: any = { owner: null, view: null, status: "idle" }
    let release!: (value: boolean) => void
    let started!: () => void
    const loading = new Promise<void>((resolve) => {
      started = resolve
    })
    const load = vi.fn((_target: any, _reference: any, onLoaded: any) => {
      started()
      return new Promise<boolean>((resolve) => {
        release = (value) => {
          if (value)
            onLoaded({ owner: destination.owner, view: destination.view })
          resolve(value)
        }
      })
    })
    h1.controller = {
      getCurrent: () => current,
      fence: () => () => true,
      loadConversation: load,
      followResult: vi.fn()
    }
    h1.saveHistory.mockResolvedValue({ id: "created-original" })
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      const pending = result.current.onSubmit({
        message: "old draft",
        image: ""
      })
      await loading
      current = destination
      options.setMessages.mockClear()
      options.setHistory.mockClear()
      options.setHistoryId.mockClear()
      release(loadResult)
      await pending
    })
    expect(h1.localAppend).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(options.setHistoryId).not.toHaveBeenCalled()
    expect(current).toBe(destination)
  }
)

it("before-first on a nonempty full-page source admits an empty prior path and sends only current input", async () => {
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  expect(current.capture.snapshot.nodes).toHaveLength(3)
  const { result } = renderHook(() => useChatActions(ordinaryOptions() as any))
  await act(async () => {
    await result.current.onSubmit({ message: "fresh question", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(
    h1.wire.mock.calls[0][0].messages.filter(
      (row: any) => row.role !== "system"
    )
  ).toEqual([{ role: "user", content: "fresh question" }])
  const user = addChatMessageMock.mock.calls[0][1]
  expect(user.tldw_history_selection_v1.messages).toEqual([])
  expect(user.tldw_history_selection_v1.cursor).toEqual({ kind: "empty" })
  expect(user.parent_message_id ?? null).toBeNull()
})

it("a mounted swipe during an accepted full-page send releases its activity while preserving selection", async () => {
  const options = ordinaryOptions()
  h1.wire.mockImplementation(async function* () {
    const current = h1.controller.getCurrent()
    current.view = {
      ...current.view,
      selection_revision: 2,
      cursor: { kind: "after_message", message_id: "a2" }
    }
    options.setHistory.mockClear()
    options.setMessages.mockClear()
    yield { choices: [{ delta: { content: "accepted answer" } }] }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(options.setIsProcessing).toHaveBeenLastCalledWith(false)
  expect(options.setStreaming).toHaveBeenLastCalledWith(false)
  expect(options.setAbortController).toHaveBeenLastCalledWith(null)
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(options.setHistory).not.toHaveBeenCalled()
  expect(h1.controller.getCurrent().view.cursor.message_id).toBe("a2")
  expect(addChatMessageMock.mock.calls[1][1].parent_message_id).toBe(
    addChatMessageMock.mock.calls[0][1].id
  )
})

it.each(["preface", "replace"])(
  "finalizes slash injection %s into the exact mounted provider body",
  async (mode) => {
    const { useStoreChatModelSettings } = await import("@/store/model")
    const { historyDigest } = await import("@/db/dexie/history-selection")
    useStoreChatModelSettings.setState({ slashCommandInjectionMode: mode })
    const { result } = renderHook(() =>
      useChatActions(ordinaryOptions() as any)
    )
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    const body = h1.wire.mock.calls[0][0]
    expect(body.slash_command_injection_mode).toBe(mode)
    expect(
      addChatMessageMock.mock.calls[0][1].tldw_history_selection_v1
        .request_context_digest
    ).toBe(historyDigest(body))
    expect(Object.isFrozen(body)).toBe(true)
    expect(body.conversation_id).toBeUndefined()
    expect(body.history_message_limit).toBeUndefined()
    expect(body.save_to_db).toBe(false)
  }
)

const nativeAdmission = (request: any) => ({
  version: 1,
  owner_key: "native-key",
  conversation_id: "tracked-chat-1",
  input_message_id: "native-input",
  input_message_revision: "native-r",
  selection_digest: request.tldw_history_selection_v1.selection_digest,
  messages: request.tldw_history_selection_v1.messages,
  originating_selection_revision:
    request.tldw_history_selection_v1.selection_revision
})

it.each(["http502", "sse-beforeadmission", "sse-afteradmission", "sse-afterpartial"])("native provider recovery for %s retains its receipt and exposes existing model settings", async failure => {
  const options = createHookOptions()
  const safeMessage = "The selected provider credentials could not be authenticated."
  h1.wire.mockImplementation(async function* (request) {
    if (failure === "http502") throw Object.assign(new Error(safeMessage), {
      status: 502, details: { detail: { error_code: "provider_authentication_failed", message: safeMessage } }
    })
    if (failure !== "sse-beforeadmission") yield { tldw_history_admission_v1: nativeAdmission(request) }
    if (failure === "sse-afterpartial") yield { choices: [{ delta: { content: "partial result" } }] }
    yield { error: { code: "provider_authentication_failed", message: safeMessage } }
  })
  const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  const shown = options.setMessages.mock.lastCall?.[0]
  const banner = getLatestChatErrorBannerEntry(shown)
  expect(banner).toMatchObject({ summary: "Character chat model setup needs attention.", recoveryAction: "open-model-settings", category: "character_chat.provider_unconfigured" })
  expect(banner!.detail).toContain("provider_authentication_failed")
  expect(banner!.detail).toContain(safeMessage)
  expect(shown.at(-1).id).toMatch(/^native-draft-/)
  expect(shown.at(-1).serverMessageId).toBeUndefined()
  expect(shown.filter((row: { isBot: boolean }) => !row.isBot).map((row: { message: string }) => row.message)).not.toContain("next")
  const retained = h1.recover.mock.lastCall?.[2]
  expect(retained).toMatchObject({ persistence: "server", state: failure === "sse-afterpartial" ? "generated_unsaved" : failure === "sse-afteradmission" ? "accepted_unsent" : "unknown", input_text: "next", result_text: failure === "sse-afterpartial" ? "partial result" : "" })
  expect(retained.assistant_id).toBeUndefined()
  if (failure === "http502" || failure === "sse-beforeadmission") expect(retained.admission).toBeUndefined()
  else expect(retained.admission.input_message_id).toBe("native-input")
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  await expect(result.current.regenerateLastMessage()).rejects.toThrow("unsupported_history_regeneration")
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it.each(["navigation", "account"])("native provider recovery cannot overwrite a changed %s", async boundary => {
  const options = createHookOptions()
  h1.wire.mockImplementation(async function* () {
    if (boundary === "navigation") h1.controller.navigate()
    else h1.auth.abort()
    options.setMessages.mockClear()
    yield { error: { code: "provider_authentication_failed", message: "Safe provider failure" } }
  })
  const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({ persistence: "server", state: "unknown", input_text: "next" })
  expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
})


it("native before-first sends no historical content and retains unacknowledged text separately", async () => {
  const options = createHookOptions()
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      choices: [{ delta: { content: "unsaved text" }, finish_reason: "stop" }]
    }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][0].messages).toEqual([
    { role: "user", content: "next" }
  ])
  expect(h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages).toEqual(
    []
  )
  expect(h1.recover.mock.calls[0][2]).toMatchObject({
    persistence: "server",
    state: "dispatching"
  })
  expect(h1.recover.mock.calls[0][2].input_id).toBeUndefined()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    persistence: "server",
    state: "generated_unsaved",
    owner_key: "native-key",
    conversation_id: "tracked-chat-1",
    origin_view: {
      conversation_id: "tracked-chat-1",
      view_session_id: "origin"
    },
    admission: {
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      input_message_id: "native-input"
    },
    input_id: "native-input",
    result_text: "unsaved text"
  })
  expect(h1.recover.mock.lastCall?.[0]).toEqual({
    profile_id: "profile",
    client_session_id: "client"
  })
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.controller.refreshRecovery).toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it.each(["wrong-selection", "result-before-admission", "wrong-conversation"])(
  "native %s cannot supply an owner result",
  async (failure) => {
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* (request) {
      if (failure !== "result-before-admission")
        yield {
          tldw_history_admission_v1: {
            ...nativeAdmission(request),
            ...(failure === "wrong-selection"
              ? { selection_digest: "forged" }
              : {})
          }
        }
      yield {
        tldw_message_id: "forged-result",
        tldw_conversation_id:
          failure === "wrong-conversation" ? "other" : "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(h1.dismiss).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      "completed"
    )
    expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
  }
)

it("native selection navigation preserves server ownership and never overwrites the new view", async () => {
  const options = createHookOptions()
  const { result } = renderHook(() => useChatActions(options as any))
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    h1.controller.navigate()
    options.setMessages.mockClear()
    yield { choices: [{ delta: { content: "old result" } }] }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.objectContaining({ conversation_id: "tracked-chat-1" }),
    expect.any(String),
    "completed"
  )
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it("native settings drift during capture prevents dispatch", async () => {
  const options = createHookOptions()
  const { useMcpToolsStore } = (await import("@/store/mcp-tools")) as any
  const previous = useMcpToolsStore.getState()
  h1.capture.mockImplementationOnce(async (_id, request) => {
    useMcpToolsStore.setState({ ...previous })
    return captureFor(request.view)
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it.each(["account change", "disconnect"])(
  "native %s retains accepted evidence under the origin without claiming response settlement",
  async (interruption) => {
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      options.setMessages.mockClear()
      if (interruption === "account change") h1.auth.abort()
      else throw new Error("connection lost after admission")
      yield { choices: [{ delta: { content: "new account must not see" } }] }
      yield {
        tldw_message_id: "late-result",
        tldw_conversation_id: "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    if (interruption === "account change")
      expect(options.setMessages).not.toHaveBeenCalled()
    else expect(h1.controller.refreshRecovery).toHaveBeenCalledOnce()
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      persistence: "server",
      state: "accepted_unsent",
      owner_key: "native-key",
      conversation_id: "tracked-chat-1",
      origin_view: {
        conversation_id: "tracked-chat-1",
        view_session_id: "origin"
      },
      admission: {
        owner_key: "native-key",
        conversation_id: "tracked-chat-1",
        input_message_id: "native-input"
      },
      input_id: "native-input",
      result_text: ""
    })
    expect(h1.recover.mock.lastCall?.[0]).toEqual({
      profile_id: "profile",
      client_session_id: "client"
    })
    expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(h1.wire).toHaveBeenCalledOnce()
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
  }
)

it("a disconnect after the native owner ACK preserves the completed outcome", async () => {
  const options = createHookOptions()
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
    throw new Error("late disconnect")
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.anything(),
    expect.any(String),
    "completed"
  )
  expect(h1.controller.followResult).toHaveBeenCalledWith(
    expect.anything(),
    "owner-result"
  )
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
})

it.each(["asset", "temporary", "local-owner"])(
  "native %s is visibly gated without dispatch",
  async (unsupported) => {
    const options = createHookOptions()
    if (unsupported === "local-owner")
      h1.controller.getCurrent().owner = {
        kind: "local",
        owner_key: "local",
        conversation_id: "local",
        profile_id: "profile"
      }
    options.temporaryChat = unsupported === "temporary"
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.onSubmit({
        message: "next",
        image: unsupported === "asset" ? "data:image/png;base64,asset" : ""
      })
    })
    expect(h1.wire).not.toHaveBeenCalled()
    expect(options.notification.error).toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
  }
)

it.each([
  { navigate: false, model: "deepseek-chat", provider: "openai", modelId: "deepseek-chat", defaults: false },
  { navigate: true, model: "deepseek-chat", provider: "openai", modelId: "deepseek-chat", defaults: false },
  { navigate: false, model: "ollama:gemma3:4b", provider: "ollama", modelId: "gemma3:4b", defaults: true }
])(
  "native first creation captures setup and fences its ACK (model=$model, navigation=$navigate)",
  async ({ navigate, model, provider, modelId, defaults }) => {
    const options = {
      ...createHookOptions(),
      historyId: null,
      serverChatId: null,
      serverChatCharacterId: null,
      selectedModel: model,
      currentChatModelSettings: defaults ? { apiProvider: "openai" } : { apiProvider: "openai", temperature: 0, topP: 0 },
      selectedAssistant: {
        kind: "character",
        id: "12",
        name: "Saved",
        metadata: { selectionMode: "tracked" }
      }
    }
    let origin = true
    let current: any = { owner: null, view: null, status: "idle" }
    const ready = makeController().getCurrent()
    ready.view.cursor = { kind: "empty" }
    ready.capture = captureFor(ready.view)
    const load = vi.fn(async (_target, _reference, onLoaded) => {
      current = ready
      onLoaded({ owner: current.owner, view: current.view })
      return true
    })
    h1.controller = {
      getCurrent: () => current,
      fence: () => () => origin,
      loadConversation: load,
      followResult: vi.fn(async () => true),
      refreshRecovery: vi.fn()
    }
    createChatMock.mockImplementation(async () => {
      if (navigate) origin = false
      return { id: "tracked-chat-1" }
    })
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      yield {
        tldw_message_id: "native-result",
        tldw_conversation_id: "tracked-chat-1"
      }
    })
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(createChatMock).toHaveBeenCalledWith(
      defaults
        ? { character_id: "12", provider, model: modelId }
        : { character_id: "12", provider, model: modelId, temperature: 0, top_p: 0 },
      expect.objectContaining({ requestScope: expect.objectContaining({ userId: "alice" }) })
    )
    if (navigate) {
      expect(h1.wire).not.toHaveBeenCalled()
      expect(load).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
    } else {
      expect(h1.wire).toHaveBeenCalledOnce()
      expect(h1.wire.mock.calls[0][0]).toMatchObject({ model: modelId, api_provider: provider })
      expect(
        h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages
      ).toEqual([])
      expect(options.setServerChatId).toHaveBeenCalledWith("tracked-chat-1")
    }
    expect(h1.live).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(persistCharacterCompletionMock).not.toHaveBeenCalled()
  }
)

it("native dispatch freezes workspace routing independently of the view owner", async () => {
  const options = createHookOptions()
  const workspace = { type: "workspace", workspaceId: "original-workspace" }
  h1.controller.getCurrent().owner.scope = workspace
  h1.wire.mockImplementation(async function* (request) {
    workspace.workspaceId = "new-workspace"
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][1].scope).toEqual({
    type: "workspace",
    workspaceId: "original-workspace"
  })
  expect(h1.live).not.toHaveBeenCalled()
})

it("an older native completion cannot release a newer operation's Stop control", async () => {
  const options = createHookOptions()
  let enterFirst!: () => void,
    enterSecond!: () => void,
    releaseFirst!: () => void,
    releaseSecond!: () => void
  const firstEntered = new Promise<void>((resolve) => {
    enterFirst = resolve
  })
  const secondEntered = new Promise<void>((resolve) => {
    enterSecond = resolve
  })
  const firstHeld = new Promise<void>((resolve) => {
    releaseFirst = resolve
  })
  const secondHeld = new Promise<void>((resolve) => {
    releaseSecond = resolve
  })
  h1.wire.mockImplementationOnce(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    enterFirst()
    await firstHeld
    yield {
      tldw_message_id: "first-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  h1.wire.mockImplementationOnce(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    enterSecond()
    await secondHeld
    yield {
      tldw_message_id: "second-result",
      tldw_conversation_id: "tracked-chat-1"
    }
  })
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    const first = result.current.onSubmit({ message: "first", image: "" })
    await firstEntered
    const second = result.current.onSubmit({ message: "second", image: "" })
    await secondEntered
    options.setStreaming.mockClear()
    options.setAbortController.mockClear()
    releaseFirst()
    await first
    expect(options.setStreaming).not.toHaveBeenCalledWith(false)
    expect(options.setAbortController).not.toHaveBeenCalledWith(null)
    releaseSecond()
    await second
    expect(options.setStreaming).toHaveBeenLastCalledWith(false)
    expect(options.setAbortController).toHaveBeenLastCalledWith(null)
  })
})

it("native required client tools are explicitly gated rather than silently discarded", async () => {
  const options = { ...createHookOptions(), toolChoice: "required" as const }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(options.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: "native_history_client_tools_unsupported"
    })
  )
})

const localMutationController = () => {
  const current = h1.controller.getCurrent()
  current.owner = {
    kind: "local",
    profile_id: "profile",
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  current.view = {
    ...current.view,
    owner_key: "local-key",
    conversation_id: "history-1"
  }
}
describe("selected-history mutations", () => {
  beforeEach(localMutationController)
  const visible = [
    {
      id: "greeting",
      isBot: true,
      name: "Assistant",
      message: "welcome",
      messageType: "character:greeting",
      sources: []
    },
    {
      id: "u-old",
      isBot: false,
      name: "You",
      message: "question",
      sources: []
    },
    { id: "a1", isBot: true, name: "Assistant", message: "same", sources: [] }
  ]
  it("editing and deleting A1 use its stable ID, preserving hidden same-text A2 and greeting", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const persisted = new Map([
      ["u-old", "question"],
      ["a1", "same"],
      ["a2", "same"]
    ])
    vi.mocked(helpers.updateMessageById).mockImplementation(
      async (_owner, id, text) => {
        persisted.set(id, text)
      }
    )
    vi.mocked(helpers.removeMessageById).mockImplementation(
      async (_owner, id) => {
        persisted.delete(id)
        return undefined as any
      }
    )
    const options = {
      ...ordinaryOptions(),
      serverChatId: null,
      messages: visible,
      historyId: "history-1"
    }
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.editMessage(2, "edited", false, false)
    })
    expect(persisted.get("a1")).toBe("edited")
    expect(persisted.get("a2")).toBe("same")
    expect(persisted.get("u-old")).toBe("question")
    await act(async () => {
      await result.current.deleteMessage(2)
    })
    expect([...persisted.keys()]).toEqual(["u-old", "a2"])
    expect(options.setMessages).toHaveBeenLastCalledWith(visible.slice(0, 2))
    expect(helpers.removeMessageByIndex).not.toHaveBeenCalled()
  })
  it("edit-and-send and regeneration reject without owner writes or display truncation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const options = {
      ...ordinaryOptions(),
      serverChatId: null,
      messages: visible,
      historyId: "history-1"
    }
    const { result } = renderHook(() => useChatActions(options as any))
    options.setMessages.mockClear()
    options.setHistory.mockClear()
    await expect(
      result.current.editMessage(1, "new question", true, true)
    ).rejects.toThrow("unsupported_history_edit_and_send:u-old")
    await expect(result.current.regenerateLastMessage()).rejects.toThrow(
      "unsupported_history_regeneration"
    )
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("a held successful edit does not replace a newly selected conversation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    let release!: () => void
    vi.mocked(helpers.updateMessageById).mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    let current = true
    h1.controller.fence = () => () => current
    const options = {
      ...ordinaryOptions(),
      serverChatId: null,
      messages: visible,
      historyId: "history-1"
    }
    const { result } = renderHook(() => useChatActions(options as any))
    options.setMessages.mockClear()
    const editing = result.current.editMessage(2, "edited", false, false)
    await vi.waitFor(() => expect(helpers.updateMessageById).toHaveBeenCalled())
    current = false
    release()
    await editing
    expect(options.setMessages).not.toHaveBeenCalled()
  })
})

it("deleting the selected local leaf follows only its captured parent", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const visible = [
    { id: "u-old", isBot: false, message: "question", sources: [] },
    { id: "a1", isBot: true, message: "same", sources: [] }
  ]
  const choose = vi.fn(async () => true)
  h1.controller.choose = choose
  vi.mocked(helpers.removeMessageById).mockResolvedValue({
    id: "a1",
    history_id: "history-1",
    parent_message_id: "u-old"
  } as any)
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: visible,
    historyId: "history-1"
  }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.deleteMessage(1)
  })
  expect(choose).toHaveBeenCalledWith({
    kind: "after_message",
    message_id: "u-old"
  })
})
it("late leaf deletion retains its storage result without moving a new view", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const visible = [{ id: "a1", isBot: true, message: "same", sources: [] }]
  const choose = vi.fn(async () => true)
  h1.controller.choose = choose
  let current = true
  h1.controller.fence = () => () => current
  let release!: (row: any) => void
  vi.mocked(helpers.removeMessageById).mockImplementation(
    () =>
      new Promise((resolve) => {
        release = resolve
      })
  )
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: visible,
    historyId: "history-1"
  }
  const { result } = renderHook(() => useChatActions(options as any))
  options.setMessages.mockClear()
  const pending = result.current.deleteMessage(0)
  await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
  current = false
  release({ id: "a1", parent_message_id: "u-old" })
  await pending
  expect(choose).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("native plain edit/delete reject before any mutation or display change", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const visible = [
    {
      id: "a1",
      serverMessageId: "a1",
      isBot: true,
      message: "same",
      sources: []
    }
  ]
  const options = {
    ...ordinaryOptions(),
    serverChatId: "tracked-chat-1",
    messages: visible
  }
  const { result } = renderHook(() => useChatActions(options as any))
  options.setMessages.mockClear()
  await expect(
    result.current.editMessage(0, "bad", false, false)
  ).rejects.toThrow("native_history_mutation_unavailable")
  await expect(result.current.deleteMessage(0)).rejects.toThrow(
    "native_history_mutation_unavailable"
  )
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(helpers.updateMessageById).not.toHaveBeenCalled()
  expect(helpers.removeMessageById).not.toHaveBeenCalled()
})

const localForks = vi.hoisted(() => ({
  capture: vi.fn(),
  prepare: vi.fn(),
  commit: vi.fn(),
  digest: (_request: any) => "fork-digest"
}))
vi.mock("@/db/dexie/branch", () => ({
  captureLocalForkSelection: localForks.capture,
  prepareLocalFork: localForks.prepare,
  commitLocalFork: localForks.commit,
  forkRequestDigest: (request: any) => localForks.digest(request)
}))
it("mounted branch action captures a stable boundary and returns the committed owner result", async () => {
  const {historyDigest} = await vi.importActual<any>("@/db/dexie/history-selection")
  localForks.digest = (value: any) => historyDigest({operation_id: value.operation_id, owner_key: value.owner_key, destination_owner_key: value.destination_owner_key, input: value.input})
  const current = h1.controller.getCurrent()
  current.owner = {
    kind: "local",
    profile_id: "profile",
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  current.view = {
    ...current.view,
    owner_key: "local-key",
    conversation_id: "history-1"
  }
  localForks.capture.mockResolvedValue({
    kind: "normal",
    selection: { conversation_id: "history-1" }
  })
  localForks.prepare.mockImplementation(async (request: any) => ({request,
    history: { id: "child" },
    messages: [],
    files: undefined
  }))
  localForks.commit.mockImplementation(async (prepared: any) => ({
    state: "committed",
    owner_key: "local-key",
    operation_id: prepared.request.operation_id,
    child_id: prepared.history.id,
    message_map: { a1: "child-a1" }
  }))
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    historyId: "history-1"
  }
  const { result } = renderHook(() => useChatActions(options as any))
  let outcome: any
  await act(async () => {
    outcome = await result.current.createChatBranch("a1")
  })
  expect(localForks.capture).toHaveBeenCalledWith(
    current.owner,
    expect.objectContaining({
      cursor: { kind: "after_message", message_id: "a1" }
    }),
    expect.any(Object)
  )
  expect(outcome).toMatchObject({
    state: "committed",
    owner_key: "local-key",
    child_id: "child",
    message_map: { a1: "child-a1" }
  })
})

vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory
}))

it.each(["normal", "bubble", "cluster"])(
  "real controller fork -> immediate send -> reopen keeps source unchanged (entry=%s)",
  async (entry) => {
    const actual = await vi.importActual<typeof import("@/store/option")>("@/store/option")
    actual.useStoreMessageOption.getState().startNewChatDraft()
    messageStoreState.draftFields = () => {
      const state = actual.useStoreMessageOption.getState()
      return { chatDraftAddress: state.chatDraftAddress, chatDraftRevision: state.chatDraftRevision, setChatDraftAddress: state.setChatDraftAddress }
    }
    const comparison = entry !== "normal"
    const { memory, seedLocalFork } = await import(
      "@/hooks/chat/__tests__/local-history-fixture"
    )
    const source = await seedLocalFork(comparison)
    const authority = await vi.importActual<any>("@/db/dexie/history-selection")
    const projector = await vi.importActual<any>("@/db/dexie/branch")
    localForks.capture.mockImplementation(projector.captureLocalForkSelection)
    localForks.prepare.mockImplementation(projector.prepareLocalFork)
    localForks.commit.mockImplementation(projector.commitLocalFork)
    // Bind the actual request digest rather than the older handler-only test stub.
    localForks.digest = projector.forkRequestDigest
    h1.localCapture.mockImplementation(authority.captureLocalHistorySnapshot)
    h1.localAppend.mockImplementation(authority.appendLocalSelectedUser)
    h1.localSettle.mockImplementation(authority.settleLocalAcceptedAssistant)
    const { useHistorySelection } = await import(
      "@/hooks/chat/useHistorySelection"
    )
    const controller = renderHook(() => useHistorySelection())
    await act(async () => {
      await controller.result.current.loadConversation({
        historyId: "history-1"
      })
    })
    if (!comparison)
      await act(async () => {
        await controller.result.current.choose({
          kind: "after_message",
          message_id: "a1"
        })
      })
    h1.controller = controller.result.current
    const options: any = {
      ...ordinaryOptions(),
      serverChatId: null,
      historyId: "history-1",
      compareModeActive: comparison,
      messages: (await import("@/db/dexie/helpers")).formatToMessage(source)
    }
    options.setHistoryId.mockImplementation((id: string) => {
      options.historyId = id
    })
    const hook = renderHook(() => useChatActions(options))
    let outcome: any
    const onOpened = vi.fn(() => {
      options.compareModeActive = false
    })
    await act(async () => {
      outcome =
        entry === "cluster"
          ? await hook.result.current.createCompareBranch({
              clusterId: "c1",
              modelId: "A",
              onOpened
            })
          : await hook.result.current.createChatBranch(
              "a1",
              comparison ? { model_id: "A", cluster_id: "c1" } : undefined,
              onOpened
            )
    })
    expect(onOpened).toHaveBeenCalledWith(outcome.child_id)
    expect(outcome.state, JSON.stringify(outcome)).toBe("committed")
    const childId = outcome.child_id
    expect(actual.useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: childId })
    expect(options.historyId).toBe(childId)
    expect(controller.result.current.getCurrent().owner).toMatchObject({
      kind: "local",
      conversation_id: childId
    })
    expect(controller.result.current.getReference()).toMatchObject({
      conversation_id: childId
    })
    h1.controller = controller.result.current
    hook.rerender()
    await act(async () => {
      await hook.result.current.onSubmit({ message: "child next", image: "" })
    })
    const childRows = await memory.messages
      .where("history_id")
      .equals(childId)
      .toArray()
    expect(childRows.map((row: any) => row.content)).toEqual([
      "question",
      "selected answer",
      "child next",
      "new answer"
    ])
    expect(
      await memory.messages.where("history_id").equals("history-1").toArray()
    ).toEqual(source)
    const reference = controller.result.current.getReference()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => {
      await reopened.result.current.loadConversation(
        { historyId: childId },
        reference
      )
    })
    expect(reopened.result.current.getCurrent().owner).toMatchObject({
      conversation_id: childId
    })
    const reopenedCapture = reopened.result.current.getCurrent().capture
    expect(reopenedCapture?.status).toBe("captured")
    if (reopenedCapture?.status !== "captured") throw new Error("reopen failed")
    expect(reopenedCapture.rows.map((row) => row.id)).toEqual(
      childRows.map((row: any) => row.id)
    )
    localForks.digest = () => "fork-digest"
  }
)

vi.mock("@/utils/safe-storage", async (original) => {
  const actual = await original<any>()
  const { settings } = await import(
    "@/hooks/chat/__tests__/local-history-fixture"
  )
  return {
    ...actual,
    createSafeStorage: (options: any) => {
      const storage = actual.createSafeStorage(options)
      const area = options?.area ?? "sync"
      return new Proxy(storage, {
        get(target, key) {
          if (key === "hasPersistentBackend") return true
          if (key === "get")
            return async (name: string) =>
              name.startsWith("chatSettings:")
                ? settings.get(area + name)
                : target.get(name)
          if (key === "set")
            return async (name: string, value: unknown) =>
              name.startsWith("chatSettings:")
                ? settings.set(area + name, value)
                : target.set(name, value)
          return Reflect.get(target, key)
        }
      })
    }
  }
})

it.each(["missing", "unavailable", "mismatch"])(
  "%s controller never gains mutation authority from ambient local IDs",
  async (kind) => {
    const helpers = await import("@/db/dexie/helpers")
    localMutationController()
    if (kind === "missing") h1.controller = null
    else if (kind === "unavailable")
      h1.controller.getCurrent().owner = {
        kind: "unavailable",
        code: "owner_unavailable"
      }
    else h1.controller.getCurrent().owner.conversation_id = "other"
    const options = {
      ...ordinaryOptions(),
      serverChatId: null,
      historyId: "history-1"
    }
    const hook = renderHook(() => useChatActions(options as any))
    options.setMessages.mockClear()
    await expect(
      hook.result.current.editMessage(0, "changed", false, false)
    ).rejects.toThrow()
    await expect(hook.result.current.deleteMessage(0)).rejects.toThrow()
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
  }
)
it("mounted native fork without local history retains an unknown create and never dispatches the same intent again", async () => {
  const {memory} = await import("./local-history-fixture")
  memory.forkOperations.rows.clear()
  const projector = await vi.importActual<any>("@/db/dexie/branch")
  localForks.digest = projector.forkRequestDigest
  h1.capture.mockImplementation(async (_id, request) => ({...captureFor(request.view), purpose: request.purpose,
    snapshot: {...captureFor(request.view).snapshot, native_fork_context: {policy: "plain_v1", supported: true, storage_context_digest: "storage"}}}))
  createChatMock.mockRejectedValue(new Error("creation response lost"))
  const {result} = renderHook(() => useChatActions({...ordinaryOptions(), historyId: null} as any))
  let first: any, second: any
  await act(async () => {first = await result.current.createChatBranch("a1")})
  await act(async () => {second = await result.current.createChatBranch("a1")})
  expect(first).toMatchObject({state: "unknown", owner_key: "native-key"})
  expect(second.operation_id).toBe(first.operation_id)
  expect(createChatMock).toHaveBeenCalledTimes(1)
  expect(localForks.commit).not.toHaveBeenCalled()
  expect([...memory.forkOperations.rows.values()][0].request.operation_id).toBe(first.operation_id)
})


const openingCapture = (id: string, request: import("@/types/history-selection").HistoryCaptureRequestV1) => {
  const view = { ...request.view, owner_key: "native-key" }
  const node = { id: "opening-" + id, revision: "opening-r", parent_id: null, role: "assistant", settled: true }
  const rows = view.cursor.kind === "empty" ? [] : [node]
  const base = captureFor(view)
  return { ...base, view, purpose: request.purpose, rows,
    selected_content: rows.map(row => ({ id: row.id, revision: row.revision, message: "Saved opening", images: [] })),
    snapshot: { ...base.snapshot, conversation_id: id, nodes: [node] } }
}

const mountedNativeOpening = async () => {
  const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  const { useStoreChatModelSettings: modelStore } = await import("@/store/model")
  const { useMcpToolsStore: toolsStore } = await import("@/store/mcp-tools")
  toolsStore.getState().setTools([])
  draftStore.getState().startNewChatDraft()
  const sourceDraft = draftStore.getState().chatDraftAddress
  messageStoreState.draftFields = () => {
    const state = draftStore.getState()
    return { chatDraftAddress: state.chatDraftAddress, chatDraftRevision: state.chatDraftRevision, setChatDraftAddress: state.setChatDraftAddress }
  }
  const { memory } = await import("./local-history-fixture")
  for (const name of ["forkOperations", "historySelections"]) memory[name].rows.clear()
  const { useHistorySelection } = await import("../useHistorySelection")
  const onCapture = vi.fn()
  const selection = renderHook(() => useHistorySelection({ onCapture }))
  h1.controller = selection.result.current
  const options = { ...createHookOptions(), historyId: null, serverChatId: null, serverChatCharacterId: null,
    selectedAssistant: { kind: "character", id: "12", name: "Saved", metadata: { selectionMode: "tracked" } } }
  createChatMock.mockResolvedValue({ id: "tracked-chat-1" })
  h1.capture.mockImplementation(async (id, request) => openingCapture(id, request))
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield { tldw_message_id: "native-result", tldw_conversation_id: "tracked-chat-1" }
  })
  const hook = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
  return { draftStore, modelStore, toolsStore, sourceDraft, memory, onCapture, selection, options, hook }
}

it.each(["capture", "bookmark"].flatMap(stage =>
  ["settings", "tools", "stop", "new"].map(boundary => ({ stage, boundary }))
))("native first-load $stage rejects $boundary drift before publication and retires only its candidate", async ({ stage, boundary }) => {
  const mounted = await mountedNativeOpening()
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  let bookmarkSpy: ReturnType<typeof vi.spyOn> | undefined
  if (stage === "capture") {
    h1.capture.mockImplementation(async (id, request) => {
      if (h1.capture.mock.calls.length === 1) { entered(); await held }
      return openingCapture(id, request)
    })
  } else {
    const original = mounted.memory.historySelections.put
    bookmarkSpy = vi.spyOn(mounted.memory.historySelections, "put").mockImplementation(async (...args: unknown[]) => {
      entered(); await held
      return original(...args)
    })
  }
  let pending!: Promise<unknown>
  await act(async () => {
    pending = mounted.hook.result.current.onSubmit({ message: "first", image: "" })
    await ready
  })
  act(() => {
    if (boundary === "settings") mounted.modelStore.setState({ temperature: 0.71 })
    else if (boundary === "tools") mounted.toolsStore.getState().setTools([])
    else if (boundary === "stop") mounted.options.setAbortController.mock.calls[0][0].abort()
    else mounted.draftStore.getState().startNewChatDraft()
  })
  await act(async () => { release(); await pending })
  bookmarkSpy?.mockRestore()
  expect(mounted.onCapture).not.toHaveBeenCalled()
  expect(mounted.selection.result.current.getCurrent()).toMatchObject({ owner: null, capture: null, status: "stale_selection", error: null })
  expect(mounted.selection.result.current.canAutomaticallyLoad()).toBe(false)
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  if (boundary === "new") expect(mounted.draftStore.getState().chatDraftAddress).not.toBe(mounted.sourceDraft)
  else expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  expect(mounted.draftStore.getState().chatDraftAddress?.kind).toBe("fresh")
  expect(h1.wire).not.toHaveBeenCalled()
  expect(createChatMock).toHaveBeenCalledOnce()
  mounted.hook.unmount()
  mounted.selection.unmount()
})

it.each(["tracked-chat-1", "newer-chat"])("late native first-load retirement preserves a newer load of %s", async nextId => {
  const mounted = await mountedNativeOpening()
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  h1.capture.mockImplementation(async (id, request) => {
    if (h1.capture.mock.calls.length === 1) { entered(); await held }
    return openingCapture(id, request)
  })
  let pending!: Promise<unknown>
  await act(async () => {
    pending = mounted.hook.result.current.onSubmit({ message: "first", image: "" })
    await ready
    expect(await mounted.selection.result.current.loadConversation({ serverChatId: nextId })).toBe(true)
  })
  const accepted = mounted.selection.result.current.getCurrent()
  await act(async () => { release(); await pending })
  expect(mounted.selection.result.current.getCurrent().owner).toBe(accepted.owner)
  expect(mounted.selection.result.current.getCurrent().view).toBe(accepted.view)
  expect(accepted.owner?.kind === "native" && accepted.owner.validate_lease()).toBe(true)
  expect(mounted.onCapture).toHaveBeenCalledOnce()
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  expect(h1.wire).not.toHaveBeenCalled()
  mounted.hook.unmount()
  mounted.selection.unmount()
})

it("an accepted native first load keeps its saved owner after later settings and completed-turn Stop", async () => {
  const mounted = await mountedNativeOpening()
  await act(async () => { await mounted.hook.result.current.onSubmit({ message: "first", image: "" }) })
  const accepted = mounted.selection.result.current.getCurrent().owner
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(mounted.options.setServerChatId).toHaveBeenCalledWith("tracked-chat-1")
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "tracked-chat-1" })
  act(() => {
    mounted.modelStore.setState({ temperature: 0.42 })
    mounted.toolsStore.getState().setTools([])
    mounted.options.setAbortController.mock.calls[0][0].abort()
  })
  expect(accepted?.kind === "native" && accepted.validate_lease()).toBe(true)
  mounted.hook.unmount()
  mounted.selection.unmount()
})


it("native first-create uncertainty retains original text and blocks repeat and cold attempts", async () => {
  const mounted = await mountedNativeOpening()
  createChatMock.mockRejectedValue(new Error("Creation response lost"))
  await act(async () => { await mounted.hook.result.current.onSubmit({ message: "ORIGINAL PRIVATE INPUT", image: "" }) })
  await act(async () => { await mounted.hook.result.current.onSubmit({ message: "changed input", image: "" }) })
  expect(createChatMock).toHaveBeenCalledOnce()
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) }
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "unconfirmed", inputText: "ORIGINAL PRIVATE INPUT", characterId: "12" })
  mounted.hook.unmount()
  const cold = renderHook(() => useChatActions(mounted.options as unknown as Parameters<typeof useChatActions>[0]))
  await act(async () => { await cold.result.current.onSubmit({ message: "cold input", image: "" }) })
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  cold.unmount(); mounted.selection.unmount()
})

it.each(["new", "account"])("native first-create keeps a late ACK as a private locator after %s without binding", async boundary => {
  const mounted = await mountedNativeOpening()
  let enter!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  createChatMock.mockImplementation(async () => { enter(); await held; return { id: "late-created-chat" } })
  let operation!: Promise<unknown>
  await act(async () => { operation = mounted.hook.result.current.onSubmit({ message: "RETAINED INPUT", image: "" }); await ready })
  act(() => { if (boundary === "new") mounted.draftStore.getState().startNewChatDraft(); else h1.auth.abort() })
  await act(async () => { release(); await operation })
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) }
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "acknowledged", chatId: "late-created-chat", inputText: "RETAINED INPUT" })
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  mounted.hook.unmount(); mounted.selection.unmount()
})

it.each(["storage", "locks"])("native first-create refuses %s unavailability before POST", async boundary => {
  const mounted = await mountedNativeOpening()
  const { Storage } = await import("@plasmohq/storage")
  const unavailable = boundary === "storage" ? vi.spyOn(Storage.prototype, "get").mockRejectedValue(new Error("storage read unavailable")) : null
  if (boundary === "locks") Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
  await act(async () => { await mounted.hook.result.current.onSubmit({ message: "KEPT INPUT", image: "" }) })
  expect(createChatMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  unavailable?.mockRestore()
  mounted.hook.unmount(); mounted.selection.unmount()
})


it.each(["accept", "new-during-metadata", "foreign-metadata"])("explicit known-ID review %s uses owned H1 without create or provider dispatch", async boundary => {
  const mounted = await mountedNativeOpening()
  const { createChatWithRecovery, readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "RETAINED UNSENT INPUT", setup: { characterId: "12" },
    isCurrent: () => true, create: async () => ({ id: "tracked-chat-1" }) })
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  const getChat = vi.mocked(tldwClient.getChat)
  if (boundary === "foreign-metadata") getChat.mockResolvedValueOnce({ id: "other-chat" } as Awaited<ReturnType<typeof tldwClient.getChat>>)
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  if (boundary === "new-during-metadata") getChat.mockImplementationOnce(async id => { enter(); await held; return { id } as Awaited<ReturnType<typeof tldwClient.getChat>> })
  let composerCurrent = true
  const onAccepted = vi.fn()
  const { reviewNativeChatCreation } = await import("../native-history-character-send")
  let accepted = false
  await act(async () => {
    const reviewing = reviewNativeChatCreation({ controller: mounted.selection.result.current, handle, isCurrent: () => composerCurrent, onAccepted })
    if (boundary === "new-during-metadata") { await ready; mounted.draftStore.getState().startNewChatDraft(); release() }
    accepted = await reviewing
  })
  expect(accepted).toBe(boundary === "accept")
  expect(onAccepted).toHaveBeenCalledTimes(boundary === "accept" ? 1 : 0)
  expect(createChatMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ inputText: "RETAINED UNSENT INPUT", state: boundary === "accept" ? "bound" : "acknowledged" })
  if (boundary === "accept") {
    expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "tracked-chat-1" })
    composerCurrent = false
    const owner = mounted.selection.result.current.getCurrent().owner
    expect(owner?.kind === "native" && owner.validate_lease()).toBe(true)
  }
  else expect(mounted.draftStore.getState().chatDraftAddress?.kind).toBe("fresh")
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


it("native first-create does not adopt a newer cancelled load epoch during bound-record persistence", async () => {
  const mounted = await mountedNativeOpening()
  const { Storage } = await import("@plasmohq/storage")
  const original = Storage.prototype.set
  let enter!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const write = vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
    await original.call(this, key, value)
    if (key.startsWith("registry:chat-creation:") && value.value?.state === "bound") { enter(); await held }
    return null
  })
  let pending!: Promise<unknown>
  await act(async () => { pending = mounted.hook.result.current.onSubmit({ message: "held input", image: "" }); await ready })
  await act(async () => {
    const before = mounted.selection.result.current.fence()
    let eligible = true
    const newer = mounted.selection.result.current.loadConversation({ serverChatId: "cancelled-newer", isCurrent: () => eligible })
    expect(before()).toBe(false)
    eligible = false
    expect(await newer).toBe(false)
  })
  await act(async () => { release(); await pending })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mounted.selection.result.current.getCurrent().owner?.conversation_id).toBe("tracked-chat-1")
  write.mockRestore(); mounted.hook.unmount(); mounted.selection.unmount()
})


it.each(["newer-H1", "selection-intent"])("held creation recovery GET does not supersede %s with the fresh address unchanged", async boundary => {
  const mounted = await mountedNativeOpening()
  const { createChatWithRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { usePlaygroundSessionStore: sessions } = await import("@/store/playground-session")
  sessions.setState({ serverChatSelectionIntent: null })
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "RETAINED INPUT", setup: {},
    isCurrent: () => true, create: async () => ({ id: "tracked-chat-1" }) })
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.mocked(tldwClient.getChat).mockImplementationOnce(async id => { enter(); await held; return { id } as Awaited<ReturnType<typeof tldwClient.getChat>> })
  const { reviewNativeChatCreation } = await import("../native-history-character-send")
  const onAccepted = vi.fn()
  let reviewing!: Promise<boolean>
  await act(async () => { reviewing = reviewNativeChatCreation({ controller: mounted.selection.result.current, handle, isCurrent: () => true, onAccepted }); await ready })
  let newer: ReturnType<typeof mounted.selection.result.current.getCurrent> | undefined
  await act(async () => {
    if (boundary === "newer-H1") {
      expect(await mounted.selection.result.current.loadConversation({ serverChatId: "newer-chat" })).toBe(true)
      newer = mounted.selection.result.current.getCurrent()
    } else {
      mounted.draftStore.getState().setServerChatId("newer-chat")
      sessions.getState().requestServerChatSelection("newer-chat")
    }
    expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
    release(); expect(await reviewing).toBe(false)
  })
  expect(onAccepted).not.toHaveBeenCalled()
  expect(h1.capture.mock.calls.some(([id]) => id === "tracked-chat-1")).toBe(false)
  if (newer) expect(mounted.selection.result.current.getCurrent().owner).toBe(newer.owner)
  else expect(sessions.getState().serverChatSelectionIntent?.chatId).toBe("newer-chat")
  expect(createChatMock).not.toHaveBeenCalled(); expect(h1.wire).not.toHaveBeenCalled()
  sessions.setState({ serverChatSelectionIntent: null })
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


it.each(["newer-H1", "source-intent"])("creation Review does not adopt %s during a held composer retention write", async boundary => {
  const mounted = await mountedNativeOpening()
  const { createChatWithRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { usePlaygroundSessionStore: sessions } = await import("@/store/playground-session")
  sessions.setState({ serverChatSelectionIntent: null })
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "ORIGINAL ATTEMPT", setup: {}, isCurrent: () => true, create: async () => ({ id: "tracked-chat-1" }) })
  const { useComposerText } = await import("@/components/Chat/composer/hooks/useComposerText")
  const composer = renderHook(() => useComposerText({ draftKey: "review-retain", draftAddress: mounted.sourceDraft!, textareaRef: React.createRef<HTMLTextAreaElement>() }))
  await waitFor(() => expect(composer.result.current.creationRecovery.active?.chatId).toBe("tracked-chat-1"))
  act(() => composer.result.current.setMessageValue("NEXT UNSENT INPUT"))
  const { Storage } = await import("@plasmohq/storage")
  const original = Storage.prototype.set
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const write = vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
    if (key.startsWith("registry:draft:review-retain:") && value.value?.content === "NEXT UNSENT INPUT") { enter(); await held }
    return original.call(this, key, value)
  })
  let reviewing!: Promise<boolean>
  await act(async () => { reviewing = composer.result.current.reviewCreatedChat(); await ready })
  let newer: ReturnType<typeof mounted.selection.result.current.getCurrent> | undefined
  await act(async () => {
    if (boundary === "newer-H1") { expect(await mounted.selection.result.current.loadConversation({ serverChatId: "newer-chat" })).toBe(true); newer = mounted.selection.result.current.getCurrent() }
    else sessions.getState().markSourceSelectionIntent()
    expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
    release(); expect(await reviewing).toBe(false)
  })
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  expect(tldwClient.getChat).not.toHaveBeenCalled()
  expect(h1.capture.mock.calls.some(([id]) => id === "tracked-chat-1")).toBe(false)
  if (newer) expect(mounted.selection.result.current.getCurrent().owner).toBe(newer.owner)
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  write.mockRestore(); composer.unmount(); snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


it.each(["scope", "actor"].flatMap(stage => ["model", "settings", "tools", "new", "ABA", "explicit", "background"].map(boundary => ({ stage, boundary }))))(
  "ordinary caller handles $boundary changes during held $stage setup before native create", async ({ stage, boundary }) => {
    h1.connected = true
    const modelStore = (await import("@/store/model")).useStoreChatModelSettings
    modelStore.getState().reset()
    const draftStore = (await vi.importActual<typeof import("@/store/option")>("@/store/option")).useStoreMessageOption
    const source = draftStore.getState().chatDraftAddress
    const options = { ...ordinaryOptions(), historyId: null, serverChatId: null, messages: [], history: [] }
    let current: ReturnType<ReturnType<typeof makeController>["getCurrent"]> | { owner: null; view: null; status: "idle" } = { owner: null, view: null, status: "idle" }
    h1.controller = {
      getCurrent: () => current,
      fence: () => { const origin = current; return () => current === origin },
      loadConversation: vi.fn(async (_target, _reference, onLoaded) => {
        current = makeController().getCurrent()
        current.view = { ...current.view, cursor: { kind: "empty" } }
        current.capture = captureFor(current.view)
        onLoaded({ owner: current.owner, view: current.view })
        return true
      }),
      followResult: vi.fn(async () => true)
    }
    createChatMock.mockResolvedValue({ id: "tracked-chat-1" })
    let enter!: () => void, release!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    if (stage === "scope") {
      const scope = vi.mocked((await import("@/services/service-prompts")).loadServicePromptSnapshot)
      const original = scope.getMockImplementation()!
      scope.mockImplementationOnce(async (...args) => { enter(); await held; return original(...args) })
    } else {
      const actor = vi.mocked((await import("@/services/actor-settings")).getActorSettingsForChat)
      actor.mockImplementationOnce(async () => { enter(); await held; return null })
    }
    const hook = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    let pending!: Promise<unknown>
    act(() => { pending = hook.result.current.onSubmit({ message: "first", image: "",
      ...(boundary === "explicit" ? { requestOverrides: { selectedModel: "explicit-model", toolChoice: "none" as const } } : {})
    }) })
    await entered
    act(() => {
      if (boundary === "explicit") { messageStoreState.value.selectedModel = "changed-model"; messageStoreState.value.toolChoice = "none" }
      else if (boundary === "background") modelStore.getState().updateScopedSetting("unused:other", "temperature", 0.41)
      else if (boundary === "model") messageStoreState.value.selectedModel = "changed-model"
      else if (boundary === "settings") modelStore.getState().setTemperature(0.73)
      else if (boundary === "tools") messageStoreState.value.toolChoice = "none"
      else {
        draftStore.getState().startNewChatDraft()
        if (boundary === "ABA") draftStore.getState().setChatDraftAddress(source)
      }
    })
    let outcome: unknown
    await act(async () => { release(); outcome = await pending })
    if (boundary === "explicit" || boundary === "background") {
      expect(outcome).toEqual({ status: "submitted" })
      expect(createChatMock).toHaveBeenCalledOnce()
      expect(addChatMessageMock).toHaveBeenCalledTimes(2)
      expect(h1.wire).toHaveBeenCalledOnce()
    } else {
      expect(createChatMock).not.toHaveBeenCalled()
      expect(h1.controller.loadConversation).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
      expect(addChatMessageMock).not.toHaveBeenCalled()
      expect(h1.wire).not.toHaveBeenCalled()
    }
    hook.unmount()
  }
)


const startGreetingOnMountedNative = async (mounted: Awaited<ReturnType<typeof mountedNativeOpening>>, seed: import("@/utils/character-greetings").GreetingSeed, surface: "playground" | "sidepanel" = "playground", greetingIsCurrent?: () => boolean) => {
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const controller = mounted.selection.result.current
  const args = { controller, surface, originIsCurrent: controller.fence(), signal: new AbortController().signal,
    temporary: false, historyId: null, serverChatId: null, characterId: "12", model: "deepseek:deepseek-chat", currentModel: "deepseek:deepseek-chat",
    settings: { apiProvider: "deepseek", temperature: 0, topP: 0 }, message: "", image: "", greetingSeed: seed, greetingIsCurrent,
    t: mounted.options.t, setServerChatId: mounted.options.setServerChatId, setMessages: mounted.options.setMessages, releaseActivity: vi.fn() }
  return sendNativeHistoryCharacter(args)
}

it.each(["playground", "sidepanel"] as const)("explicit greeting startup on %s binds one owned assistant opening without a user turn or provider", async surface => {
  const mounted = await mountedNativeOpening()
  await act(async () => { await startGreetingOnMountedNative(mounted, { greetingStrategy: "default", rawText: "Saved opening" }, surface) })
  expect(createChatMock).toHaveBeenCalledWith(expect.objectContaining({ character_id: "12", provider: "deepseek", model: "deepseek-chat", temperature: 0, top_p: 0 }), expect.objectContaining({ seedFirstMessage: true, greetingStrategy: "default", requestScope: expect.any(Object) }))
  expect(mounted.onCapture).toHaveBeenCalledOnce()
  expect(mounted.selection.result.current.capture).toMatchObject({ status: "captured", rows: [{ id: "opening-tracked-chat-1", role: "assistant", settled: true }] })
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "tracked-chat-1" })
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const attempt = await readChatCreationRecovery({ address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) })
  expect(attempt).toMatchObject({ state: "settled", inputText: "" })
  mounted.selection.unmount(); mounted.hook.unmount()
})

it("explicit greeting startup preserves raw alternate index zero", async () => {
  const mounted = await mountedNativeOpening()
  await act(async () => { await startGreetingOnMountedNative(mounted, { greetingStrategy: "alternate_index", alternateIndex: 0, rawText: "Saved opening" }) })
  expect(createChatMock.mock.calls[0][1]).toMatchObject({ seedFirstMessage: true, greetingStrategy: "alternate_index", alternateIndex: 0 })
  expect(h1.wire).not.toHaveBeenCalled()
  mounted.selection.unmount(); mounted.hook.unmount()
})

it("explicit greeting startup rejects a missing stored seed before capture publication and keeps known-ID recovery", async () => {
  const mounted = await mountedNativeOpening()
  h1.capture.mockImplementation(async (id, request) => { const value = openingCapture(id, request); return { ...value, rows: [], selected_content: [], snapshot: { ...value.snapshot, nodes: [] } } })
  await act(async () => { await expect(startGreetingOnMountedNative(mounted, { greetingStrategy: "default", rawText: "Saved opening" })).rejects.toThrow() })
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(mounted.onCapture).not.toHaveBeenCalled()
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  expect(h1.wire).not.toHaveBeenCalled()
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  expect(await readChatCreationRecovery({ address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) })).toMatchObject({ state: "acknowledged", chatId: "tracked-chat-1", inputText: "" })
  mounted.selection.unmount(); mounted.hook.unmount()
})


it.each(["capture", "bookmark"])("greeting startup rejects changed UI intent during %s before publication", async stage => {
  const mounted = await mountedNativeOpening()
  let current = true, enter!: () => void, release!: () => void
  const entered = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  let spy: ReturnType<typeof vi.spyOn> | undefined
  if (stage === "capture") {
    h1.capture.mockImplementation(async (id, request) => { enter(); await held; return openingCapture(id, request) })
  } else {
    const original = mounted.memory.historySelections.put
    spy = vi.spyOn(mounted.memory.historySelections, "put").mockImplementation(async (...args: unknown[]) => { enter(); await held; return original(...args) })
  }
  let pending!: Promise<unknown>
  act(() => { pending = startGreetingOnMountedNative(mounted, { greetingStrategy: "default", rawText: "Saved opening" }, "playground", () => current) })
  await entered
  current = false
  await act(async () => { release(); await expect(pending).rejects.toThrow() })
  spy?.mockRestore()
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(mounted.onCapture).not.toHaveBeenCalled()
  expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  expect(h1.wire).not.toHaveBeenCalled()
  mounted.selection.unmount(); mounted.hook.unmount()
})

it("the actual chat submit action honors assistant-only greeting intent", async () => {
  const mounted = await mountedNativeOpening()
  let result: unknown
  await act(async () => { result = await mounted.hook.result.current.onSubmit({ message: "", image: "", greetingSeed: { greetingStrategy: "default", rawText: "Saved opening" }, greetingIsCurrent: () => true }) })
  expect(result).toEqual({ status: "submitted" })
  expect(createChatMock.mock.calls[0][1]).toMatchObject({ seedFirstMessage: true, greetingStrategy: "default" })
  expect(mounted.onCapture).toHaveBeenCalledOnce()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  mounted.selection.unmount(); mounted.hook.unmount()
})

it.each(["newer-H1", "source-intent", "reconnect"])("explicit greeting captures %s before a held durable composer ACK", async boundary => {
  const mounted = await mountedNativeOpening()
  const { useComposerText } = await import("@/components/Chat/composer/hooks/useComposerText")
  const { useCharacterGreetingStart } = await import("@/hooks/useCharacterGreetingStart")
  const { usePlaygroundSessionStore: sessions } = await import("@/store/playground-session")
  sessions.setState({ serverChatSelectionIntent: null })
  const composer = renderHook(() => useComposerText({ draftKey: "greeting-retain", draftAddress: mounted.sourceDraft!, textareaRef: React.createRef<HTMLTextAreaElement>() }))
  await waitFor(() => expect(composer.result.current.draftReady).toBe(true))
  act(() => composer.result.current.setMessageValue("UNSENT GREETING REVIEW\n  exact  "))
  const { Storage } = await import("@plasmohq/storage")
  const original = Storage.prototype.set
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const write = vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
    if (key.startsWith("registry:draft:greeting-retain:") && value.value?.content === "UNSENT GREETING REVIEW\n  exact  ") { enter(); await held }
    return original.call(this, key, value)
  })
  const submit = vi.fn(mounted.hook.result.current.onSubmit)
  const starter = renderHook(({ connected }) => useCharacterGreetingStart({ form: composer.result.current.form,
    retainCurrentDraft: composer.result.current.retainCurrentDraft, submit, ready: true, isConnectionReady: connected, busy: false,
    controller: mounted.selection.result.current, readContext: () => [], preflight: () => {}, focusComposer: vi.fn() }), { initialProps: { connected: true } })
  let pending!: Promise<void>
  act(() => { pending = starter.result.current.start({ id: "default", text: "Saved opening", index: 0, seed: { greetingStrategy: "default", rawText: "Saved opening" } }, () => true) })
  await ready
  if (boundary === "reconnect") { starter.rerender({ connected: false }); starter.rerender({ connected: true }) }
  await act(async () => {
    if (boundary === "newer-H1") expect(await mounted.selection.result.current.loadConversation({ serverChatId: "newer-chat" })).toBe(true)
    else if (boundary === "source-intent") sessions.getState().markSourceSelectionIntent()

    release(); await expect(pending).rejects.toThrow()
  })
  expect(submit).not.toHaveBeenCalled()
  expect(createChatMock).not.toHaveBeenCalled()
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(mounted.sourceDraft)
  expect(composer.result.current.form.getValues().message).toBe("UNSENT GREETING REVIEW\n  exact  ")
  expect(h1.wire).not.toHaveBeenCalled()
  write.mockRestore(); starter.unmount(); composer.unmount(); mounted.hook.unmount(); mounted.selection.unmount()
})

it.each(["owned", "outside"])("assistant-only bind respects %s keyboard focus", async focus => {
  const mounted = await mountedNativeOpening()
  const { useCharacterGreetingStart } = await import("@/hooks/useCharacterGreetingStart")
  const { useSimpleForm } = await import("@/hooks/useSimpleForm")
  const { usePlaygroundSessionStore: sessions } = await import("@/store/playground-session")
  sessions.setState({ serverChatSelectionIntent: null })
  const button = document.createElement("button"), textarea = document.createElement("textarea"), other = document.createElement("button")
  document.body.append(button, textarea, other); button.focus()
  const focusComposer = vi.fn(() => textarea.focus())
  const starter = renderHook(() => {
    const form = useSimpleForm({ initialValues: { message: "UNSENT", image: "" } })
    return useCharacterGreetingStart({ form, retainCurrentDraft: async () => true,
      submit: async input => {
        if (focus === "outside") other.focus()
        const result = await mounted.hook.result.current.onSubmit(input)
        button.remove()
        return result
      }, ready: true, isConnectionReady: true, busy: false, controller: mounted.selection.result.current,
      readContext: () => [], preflight: () => {}, focusComposer })
  })
  await act(async () => { await starter.result.current.start({ id: "default", text: "Saved opening", index: 0, seed: { greetingStrategy: "default", rawText: "Saved opening" } }, () => true) })
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "tracked-chat-1" })
  expect(focusComposer).toHaveBeenCalledTimes(focus === "owned" ? 1 : 0)
  expect(document.activeElement).toBe(focus === "owned" ? textarea : other)
  expect(h1.wire).not.toHaveBeenCalled()
  starter.unmount(); mounted.hook.unmount(); mounted.selection.unmount(); button.remove(); textarea.remove(); other.remove()
})


it.each(["bound-record", "send-capture"])("private native popup %s retires changed preparation while retaining its accepted owner", async stage => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { Storage } = await import("@plasmohq/storage")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  const before = mounted.draftStore.getState()
  let preparationCurrent = true
  let enter!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  let write: ReturnType<typeof vi.spyOn> | undefined
  if (stage === "bound-record") {
    const original = Storage.prototype.set
    write = vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
      await original.call(this, key, value)
      if (key.startsWith("registry:chat-creation:") && value.value?.state === "bound") { enter(); await held }
      return null
    })
  } else h1.capture.mockImplementation(async (id, request) => {
    // Opening qualifies its initial leaf with two captures; the third is send preparation.
    if (h1.capture.mock.calls.length === 3) { enter(); await held }
    return openingCapture(id, request)
  })
  let pending!: Promise<unknown>
  await act(async () => {
    pending = sendNativeHistoryCharacter({ controller: mounted.selection.result.current,
      privateDraft: { handle, isCurrent: () => true }, surface: "playground",
      originIsCurrent: () => preparationCurrent, signal: new AbortController().signal,
      temporary: false, scope: { type: "global" }, characterId: "12", model: "deepseek:deepseek-chat",
      settings: {}, toolChoice: "none", message: "private popup input", image: "", t: ((key: string) => key) as import("i18next").TFunction,
      setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
    }).catch(error => error)
    await ready
  })
  preparationCurrent = false
  await act(async () => { release(); await pending })
  const owner = mounted.selection.result.current.getCurrent().owner
  const acceptedOwnerValid = owner?.kind === "native" && owner.validate_lease()
  write?.mockRestore()
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(acceptedOwnerValid).toBe(true)
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(before.chatDraftAddress)
  expect(mounted.draftStore.getState().chatDraftRevision).toBe(before.chatDraftRevision)
})


it.each(["accepted", "bound-next", "image", "clipboard-denied", "newer-input", "address-ABA"])("settled private draft Review preserves destination text and private retention: %s", async boundary => {
  const mounted = await mountedNativeOpening()
  const { createChatWithRecovery, markChatCreationBound, markChatCreationSettled } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { usePlaygroundSessionStore: sessions } = await import("@/store/playground-session")
  const { useComposerText } = await import("@/components/Chat/composer/hooks/useComposerText")
  const { conversationDraftKey } = await import("@/utils/chat-draft-address")
  const { createLocalRegistryBucket } = await import("@/services/settings/local-bucket")
  sessions.setState({ serverChatSelectionIntent: null })
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "FIRST SENT", setup: {}, isCurrent: () => true, create: async () => ({ id: "tracked-chat-1" }) })
  await markChatCreationBound(handle, "tracked-chat-1")
  if (boundary !== "bound-next") await markChatCreationSettled(handle, "tracked-chat-1")
  const root = "tldw:playgroundChatDraft"
  const sourceKey = `${conversationDraftKey(root, handle.address)}:owner:${handle.ownerHash}`
  const destination = { kind: "native" as const, id: "tracked-chat-1", ownerKey: "native-key", scope: { type: "global" as const } }
  const destinationKey = `${conversationDraftKey(root, destination)}:owner:${handle.ownerHash}`
  const bucket = createLocalRegistryBucket<{ content: string; metadata: Record<string, unknown> }>({ prefix: "registry:draft:" })
  await bucket.set(sourceKey, { content: "  NEXT UNSENT\nExact spacing  ", metadata: { chatDraftHandle: handle } })
  const destinationPayload = { content: "DESTINATION DRAFT", metadata: { wasExpanded: true, collapsedRange: null } }
  await bucket.set(destinationKey, destinationPayload)
  sessionStorage.clear()
  const restored = vi.fn()
  const mountComposer = () => renderHook(() => {
    const address = mounted.draftStore(state => state.chatDraftAddress)
    const transition = mounted.draftStore(state => state.chatDraftTransition)
    return useComposerText({ draftKey: address ? conversationDraftKey(root, address) : root,
      draftAddress: address ?? undefined, tabScopedDraft: true,
      bindFromDraftKey: transition?.kind === "bind" && transition.sourceAddress ? conversationDraftKey(root, transition.sourceAddress) : undefined,
      textareaRef: React.createRef<HTMLTextAreaElement>(), restoreWithMetadata: restored,
      getDraftMetadata: () => destinationPayload.metadata })
  })
  const composer = mountComposer()
  await waitFor(() => expect(composer.result.current.creationRecovery.active?.state).toBe(boundary === "bound-next" ? "bound" : "settled"))
  await waitFor(() => expect(composer.result.current.form.values.message).toBe("  NEXT UNSENT\nExact spacing  "))
  expect(composer.result.current.creationRecovery.copyInputRequired).toBe(true)
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const copy = vi.fn(async () => { enter(); await held; if (boundary === "clipboard-denied") throw new Error("denied") })
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } })
  if (boundary === "image") act(() => composer.result.current.form.setValues({ image: "data:image/png;base64,SOURCE" }))
  h1.capture.mockClear()
  let reviewing!: Promise<boolean>
  await act(async () => { reviewing = composer.result.current.reviewCreatedChat().catch(() => false); if (boundary !== "image" && boundary !== "bound-next") await ready })
  if (boundary === "newer-input") act(() => composer.result.current.setMessageValue("NEWER EXACT INPUT  "))
  if (boundary === "address-ABA") act(() => {
    mounted.draftStore.getState().startNewChatDraft()
    mounted.draftStore.getState().setChatDraftAddress(handle.address)
  })
  let reviewed = false
  await act(async () => { release(); reviewed = await reviewing })
  const accepted = boundary === "accepted" || boundary === "bound-next"
  if (accepted) {
    await waitFor(() => expect(composer.result.current.form.values.message).toBe("DESTINATION DRAFT"))
    expect(mounted.draftStore.getState().chatDraftTransition?.kind).toBe("switch")
    expect(restored).toHaveBeenCalledWith("DESTINATION DRAFT", destinationPayload.metadata)
    composer.unmount(); sessionStorage.clear()
    const cold = mountComposer()
    await waitFor(() => expect(cold.result.current.form.values.message).toBe("DESTINATION DRAFT"))
    cold.unmount()
  } else {
    expect(mounted.draftStore.getState().chatDraftAddress).toBe(handle.address)
    if (boundary === "newer-input") expect(composer.result.current.form.values.message).toBe("NEWER EXACT INPUT  ")
    if (boundary === "image") {
      expect(composer.result.current.form.values).toMatchObject({ message: "  NEXT UNSENT\nExact spacing  ", image: "data:image/png;base64,SOURCE" })
      expect(h1.capture).not.toHaveBeenCalled()
    }
    composer.unmount()
  }
  expect(reviewed).toBe(accepted)
  if (boundary === "image") expect(copy).not.toHaveBeenCalled()
  else expect(copy).toHaveBeenCalledWith("  NEXT UNSENT\nExact spacing  ")
  expect((await bucket.get(destinationKey))?.value).toEqual(destinationPayload)
  if (boundary !== "newer-input") expect((await bucket.get(sourceKey))?.value.content).toBe("  NEXT UNSENT\nExact spacing  ")
  expect(createChatMock).not.toHaveBeenCalled(); expect(h1.wire).not.toHaveBeenCalled()
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


it.each([false, true])("private bound retry settles only the exact original input (changed %s)", async changed => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  let preparationCurrent = true
  h1.capture.mockImplementation(async (id, request) => {
    if (h1.capture.mock.calls.length === 3) preparationCurrent = false
    return openingCapture(id, request)
  })
  const send = (message: string, serverChatId?: string) => sendNativeHistoryCharacter({
    controller: mounted.selection.result.current, privateDraft: { handle, isCurrent: () => true }, surface: "playground",
    originIsCurrent: () => preparationCurrent, signal: new AbortController().signal, temporary: false,
    scope: { type: "global" }, characterId: "12", model: "deepseek:deepseek-chat", settings: {}, toolChoice: "none",
    message, image: "", serverChatId, t: ((key: string) => key) as import("i18next").TFunction,
    setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
  })
  await act(async () => { await send("  ORIGINAL INPUT  ").catch(() => {}) })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "bound", inputText: "  ORIGINAL INPUT  ", chatId: "tracked-chat-1" })
  preparationCurrent = true
  h1.capture.mockImplementation(async (id, request) => openingCapture(id, request))
  await act(async () => { await send(changed ? "DIFFERENT INPUT" : "  ORIGINAL INPUT  ", "tracked-chat-1") })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: changed ? "bound" : "settled", inputText: changed ? "  ORIGINAL INPUT  " : "" })
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


it("private first-create reports the actual qualification read failure and keeps the known receipt", async () => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  h1.capture.mockRejectedValue(new Error("owned_read_unavailable"))
  let error: unknown
  await act(async () => { error = await sendNativeHistoryCharacter({ controller: mounted.selection.result.current,
    privateDraft: { handle, isCurrent: () => true }, surface: "playground", originIsCurrent: () => true,
    signal: new AbortController().signal, temporary: false, scope: { type: "global" }, characterId: "12",
    model: "deepseek:deepseek-chat", settings: {}, toolChoice: "none", message: "  KEPT EXACT INPUT  ", image: "",
    t: ((key: string) => key) as import("i18next").TFunction, setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
  }).catch(cause => cause) })
  expect((error as Error).message).toBe("owned_read_unavailable")
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "acknowledged", chatId: "tracked-chat-1", inputText: "  KEPT EXACT INPUT  " })
  expect(h1.wire).not.toHaveBeenCalled()
  snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
})


// Actual H1/root action, with declared recovery storage and model I/O boundaries.
const mountInterruptedRetry = async (local = false, invalidCompare = false) => {
  const { useHistorySelection } = await import("../useHistorySelection")
  const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  h1.capture.mockImplementation(async (_id, request) => {
    const captured = captureFor({ ...request.view, owner_key: "native-key" })
    return { ...captured, purpose: request.purpose, ...(request.view.cursor.kind === "before_message" && request.view.cursor.message_id === "u-old" ? { rows: [], selected_content: [] } : {}) }
  })
  if (local) {
    const { seedLocalFork } = await import("./local-history-fixture")
    await seedLocalFork(false)
    const authority = await vi.importActual<typeof import("@/db/dexie/history-selection")>("@/db/dexie/history-selection")
    h1.localCapture.mockImplementation(authority.captureLocalHistorySnapshot)
  }
  const selection = renderHook(() => useHistorySelection())
  await act(async () => { expect(await selection.result.current.loadConversation(local ? { historyId: "history-1" } : { serverChatId: "tracked-chat-1" })).toBe(true) })
  h1.controller = selection.result.current
  const current = selection.result.current.getCurrent()
  const admission = { version: 1, owner_key: current.view!.owner_key, conversation_id: current.view!.conversation_id, input_message_id: "u-old", input_message_revision: "r1", selection_digest: "accepted", messages: [], originating_selection_revision: 1 }
  const entry = { scope: current.bookmarkScope!, turn: { version: 1, operation_id: "interrupted-retry", owner_key: current.view!.owner_key, conversation_id: current.view!.conversation_id, origin_view: current.view, input_id: "u-old", input_text: "old question", result_text: "partial", state: "generated_unsaved", persistence: "server", admission } }
  const options = { ...ordinaryOptions(), ...(invalidCompare ? { compareModeActive: true, compareFeatureEnabled: true, compareSelectedModels: [] } : {}), ...(local ? { serverChatId: null, historyId: "history-1" } : {}), messages: [ { id: "u-old", isBot: false, message: "old question", sources: [] }, { id: "a1", isBot: true, message: "partial", generationInfo: { interrupted: true }, sources: [] } ] }
  const hook = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
  const choose = vi.spyOn(selection.result.current, "choose")
  const dismiss = vi.spyOn(selection.result.current, "dismissRecovery")
  normalChatModeMock.mockResolvedValue({ status: "submitted" })
  h1.lookupRecoveries.mockResolvedValue([entry])
  return { selection, hook, options, draftStore, entry, admission, choose, dismiss }
}

it.each(["view", "new", "account"])("interrupted retry cannot replace a later %s after held recovery lookup", async boundary => {
  const mounted = await mountInterruptedRetry()
  let finish!: (value: unknown) => void
  h1.lookupRecoveries.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  let pending!: Promise<unknown>
  await act(async () => { pending = mounted.hook.result.current.regenerateLastMessage(); await Promise.resolve() })
  if (boundary === "view") await act(async () => { await mounted.selection.result.current.choose({ kind: "after_message", message_id: "a2" }) })
  else if (boundary === "new") mounted.draftStore.getState().startNewChatDraft()
  else window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
  const view = mounted.selection.result.current.getCurrent().view
  const draft = mounted.draftStore.getState().chatDraftAddress
  mounted.choose.mockClear()
  let outcome: unknown
  await act(async () => { finish([mounted.entry]); outcome = await pending })
  expect(outcome).toMatchObject({ status: "skipped" })
  expect(mounted.choose).not.toHaveBeenCalled()
  expect(mounted.dismiss).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
  expect(mounted.selection.result.current.getCurrent().view).toBe(view)
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(draft)
  expect(mounted.options.notification.error).not.toHaveBeenCalled()
})

it("current interrupted retry reuses its existing admission and submits once", async () => {
  const mounted = await mountInterruptedRetry()
  let outcome: unknown
  await act(async () => { outcome = await mounted.hook.result.current.regenerateLastMessage() })
  expect(outcome).toMatchObject({ status: "submitted" })
  expect(mounted.choose).toHaveBeenCalledExactlyOnceWith({ kind: "before_message", message_id: "u-old" })
  expect(mounted.dismiss).toHaveBeenCalledExactlyOnceWith(mounted.entry)
  expect(normalChatModeMock).toHaveBeenCalledOnce()
  expect(normalChatModeMock.mock.calls[0][6].historySelection.retryAdmission).toEqual(mounted.admission)
})

it("interrupted retry cannot submit after its recovery dismissal outlives its selected view", async () => {
  const mounted = await mountInterruptedRetry()
  let finish!: () => void
  const entered = new Promise<void>(resolve => { mounted.dismiss.mockImplementationOnce(async () => { resolve(); await new Promise<void>(release => { finish = release }) }) })
  let pending!: Promise<unknown>
  await act(async () => { pending = mounted.hook.result.current.regenerateLastMessage(); await entered })
  await act(async () => { await mounted.selection.result.current.choose({ kind: "after_message", message_id: "a2" }) })
  mounted.choose.mockClear()
  let outcome: unknown
  await act(async () => { finish(); outcome = await pending })
  expect(outcome).toMatchObject({ status: "skipped" })
  expect(normalChatModeMock).not.toHaveBeenCalled()
  expect(mounted.choose).not.toHaveBeenCalled()
})


it("interrupted retry does not choose after a held local admission read and New", async () => {
  const mounted = await mountInterruptedRetry(true)
  h1.lookupRecoveries.mockResolvedValue([])
  let finish!: (value: unknown) => void
  let entered!: () => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  h1.lookupAdmission.mockImplementationOnce(() => { entered(); return new Promise(resolve => { finish = resolve }) })
  let pending!: Promise<unknown>
  await act(async () => { pending = mounted.hook.result.current.regenerateLastMessage(); await ready })
  mounted.draftStore.getState().startNewChatDraft()
  let outcome: unknown
  await act(async () => { finish(null); outcome = await pending })
  expect(outcome).toMatchObject({ status: "skipped" })
  expect(mounted.choose).not.toHaveBeenCalled()
  expect(mounted.dismiss).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
})

it("interrupted retry retains its own choice during a held capture without adopting a newer choice", async () => {
  const mounted = await mountInterruptedRetry()
  const capture = h1.capture.getMockImplementation()!
  let finish!: () => void
  let entered!: () => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  h1.capture.mockImplementationOnce(async (...args) => { entered(); await new Promise<void>(resolve => { finish = resolve }); return capture(...args) })
  let pending!: Promise<unknown>
  await act(async () => { pending = mounted.hook.result.current.regenerateLastMessage(); await ready })
  await act(async () => { await mounted.selection.result.current.choose({ kind: "after_message", message_id: "a2" }) })
  const view = mounted.selection.result.current.getCurrent().view
  let outcome: unknown
  await act(async () => { finish(); outcome = await pending })
  expect(outcome).toMatchObject({ status: "skipped" })
  expect(mounted.dismiss).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
  expect(mounted.options.notification.error).not.toHaveBeenCalled()
  expect(mounted.selection.result.current.getCurrent().view).toBe(view)
})


it.each(["view", "new", "account"])("an old rendered interrupted retry cannot adopt a later %s at invocation", async boundary => {
  const mounted = await mountInterruptedRetry()
  const oldRetry = mounted.hook.result.current.regenerateLastMessage
  let pending!: Promise<unknown>
  let changed: Promise<unknown> | undefined
  act(() => {
    if (boundary === "view") changed = mounted.selection.result.current.choose({ kind: "after_message", message_id: "a2" })
    else if (boundary === "new") mounted.draftStore.getState().startNewChatDraft()
    else window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    pending = oldRetry()
  })
  let outcome: unknown
  await act(async () => { await changed; outcome = await pending })
  expect(outcome).toMatchObject({ status: "skipped" })
  expect(mounted.choose).toHaveBeenCalledTimes(boundary === "view" ? 1 : 0)
  expect(mounted.dismiss).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
  expect(mounted.options.notification.error).not.toHaveBeenCalled()
})


it("an old rendered interrupted Retry does not report obsolete Compare validation after New", async () => {
  const mounted = await mountInterruptedRetry(false, true)
  const oldRetry = mounted.hook.result.current.regenerateLastMessage
  let pending!: Promise<unknown>
  act(() => { mounted.draftStore.getState().startNewChatDraft(); pending = oldRetry() })
  await act(async () => { expect(await pending).toMatchObject({ status: "skipped" }) })
  expect(mounted.options.notification.error).not.toHaveBeenCalled()
  expect(mounted.choose).not.toHaveBeenCalled()
  expect(mounted.dismiss).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
})
