// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
  value: {
    selectedModel: "deepseek-chat" as string | null,
    serverChatId: null as string | null,
    serverChatCharacterId: null as string | number | null,
    serverChatAssistantKind: null as "character" | "persona" | null,
    serverChatSource: null as string | null
  }
}))

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
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

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createRegenerateLastMessage: vi.fn(() => vi.fn()),
  createEditMessage: vi.fn(() => vi.fn()),
  createStopStreamingRequest: vi.fn(() => vi.fn()),
  createBranchMessage: vi.fn(() => vi.fn())
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
    captureHistorySelection: (...args: any[]) => h1.capture(...args),
    streamChatCompletion: (...args: any[]) => h1.wire(...args),
    getConfig: vi.fn(async () => null),
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
  controller: null as any,
  capture: vi.fn(),
  wire: vi.fn(),
  recover: vi.fn(),
  dismiss: vi.fn(),
  saveHistory: vi.fn(),
  localCapture: vi.fn(),
  localAppend: vi.fn(),
  localSettle: vi.fn()
}))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => h1.controller
}))
vi.mock("@/db/dexie/history-selection", async (original) => ({
  ...(await original<any>()),
  saveHistoryTurnRecovery: h1.recover,
  dismissHistoryTurnRecovery: h1.dismiss,
  captureLocalHistorySnapshot: h1.localCapture,
  appendLocalSelectedUser: h1.localAppend,
  settleLocalAcceptedAssistant: h1.localSettle
}))
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<any>()),
  loadServicePromptSnapshot: vi.fn(async () => ({
    requestScope: {
      config: { serverUrl: "https://server.test", authMode: "multi-user" },
      userId: "alice"
    },
    definitions: {},
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
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
    pageAssistModel: async (options: any) =>
      new ChatTldw({ ...options, temperature: 0.23, supportsMultimodal: true })
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

beforeEach(async () => {
  vi.clearAllMocks()
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

it("mounted selected-history character submit blocks the unversioned timestamp route before writes", async () => {
  const options = createHookOptions()
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(createChatMock).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
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

it("first durable ordinary send creates its local owner and admits before inference without a server conversation", async () => {
  const options = {
    ...ordinaryOptions(),
    historyId: null,
    serverChatId: null,
    messages: [],
    history: []
  }
  let current: any = { owner: null, view: null, status: "idle" }
  const load = vi.fn(async ({ historyId }: any) => {
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
    return true
  })
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
