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
    getCharacter: h1.live,
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
    pageAssistModel: async (options: any) =>
      new ChatTldw({
        ...options,
        temperature: 0.23,
        supportsMultimodal: true,
        slashCommandInjectionMode: (
          await import("@/store/model")
        ).useStoreChatModelSettings.getState().slashCommandInjectionMode
      })
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

it.each([false, true])(
  "native first creation is fenced around its ACK (navigation=%s)",
  async (navigate) => {
    const options = {
      ...createHookOptions(),
      historyId: null,
      serverChatId: null,
      serverChatCharacterId: null,
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
    if (navigate) {
      expect(h1.wire).not.toHaveBeenCalled()
      expect(load).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
    } else {
      expect(h1.wire).toHaveBeenCalledOnce()
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

describe("selected-history mutations", () => {
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

const localForks = vi.hoisted(() => ({capture:vi.fn(),prepare:vi.fn(),commit:vi.fn()}))
vi.mock("@/db/dexie/branch", () => ({
  captureLocalForkSelection:localForks.capture,
  prepareLocalFork:localForks.prepare,
  commitLocalFork:localForks.commit,
  forkRequestDigest:()=>"fork-digest"
}))
it("mounted branch action captures a stable boundary and returns the committed owner result", async () => {
  const current = h1.controller.getCurrent()
  current.owner = {kind:"local",profile_id:"profile",owner_key:"local-key",conversation_id:"history-1"}
  current.view = {...current.view,owner_key:"local-key",conversation_id:"history-1"}
  localForks.capture.mockResolvedValue({kind:"normal",selection:{conversation_id:"history-1"}})
  localForks.prepare.mockResolvedValue({history:{id:"child"},messages:[],files:undefined})
  localForks.commit.mockImplementation(async (prepared:any) => ({state:"committed",owner_key:"local-key",operation_id:"op",child_id:prepared.history.id,message_map:{a1:"child-a1"}}))
  const options = {...ordinaryOptions(),serverChatId:null,historyId:"history-1"}
  const {result} = renderHook(() => useChatActions(options as any))
  let outcome: any
  await act(async () => {outcome = await result.current.createChatBranch("a1")})
  expect(localForks.capture).toHaveBeenCalledWith(current.owner,expect.objectContaining({cursor:{kind:"after_message",message_id:"a1"}}),expect.any(Object))
  expect(outcome).toMatchObject({state:"committed",owner_key:"local-key",child_id:"child",message_map:{a1:"child-a1"}})
})
