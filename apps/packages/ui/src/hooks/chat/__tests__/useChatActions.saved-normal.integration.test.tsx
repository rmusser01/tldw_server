import React from "react"
import i18n from "i18next"
import { ConfigProvider, notification } from "antd"
import {
  act,
  fireEvent,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ChatTldw } from "@/models/ChatTldw"
import { useChatActions } from "../useChatActions"
import { useServerChatLoader } from "../useServerChatLoader"
import { usePlaygroundPersistence } from "@/components/Option/Playground/hooks/usePlaygroundPersistence"
import { useStoreMessageOption } from "@/store/option"
import { reconcileServerChatMessages, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { useComposerQueue } from "@/components/Chat/composer/hooks/useComposerQueue"

const mocks = vi.hoisted(() => ({
  removeMessageById: vi.fn(),
  deleteMessage: vi.fn(),
  createChat: vi.fn(),
  getChat: vi.fn(),
  addChatMessage: vi.fn(),
  listChatMessages: vi.fn(),
  initialize: vi.fn(),
  pageAssistModel: vi.fn(),
  streamMessage: vi.fn(),
  saveHistory: vi.fn(),
  saveMessage: vi.fn(),
  ensureHistory: vi.fn(),
  getActorSettings: vi.fn(),
  notifyError: vi.fn(),
  rows: [] as Array<Record<string, unknown>>,
  mirrorHistories: new Map<string, Record<string, unknown>>(),
  withLoader: false,
  selectionRevision: 0,
  serverRows: new Map<
    string,
    Array<{ id?: string; role: string; content: string; metadata_extra?: Record<string, unknown> }>
  >(),
  watches: new Set<{ tldwConfig: (change: { newValue: unknown }) => void }>(),
  config: {
    serverUrl: "https://chat.test",
    authMode: "single-user",
    apiKey: "synthetic-a"
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async () => undefined,
    set: vi.fn(),
    remove: vi.fn(),
    watch: (watch: { tldwConfig: (change: { newValue: unknown }) => void }) =>
      mocks.watches.add(watch),
    unwatch: (watch: { tldwConfig: (change: { newValue: unknown }) => void }) =>
      mocks.watches.delete(watch)
  })
}))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: vi.fn() }
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    deleteMessage: mocks.deleteMessage,
    initialize: mocks.initialize,
    getConfig: async () => mocks.config,
    ensureConfigForRequest: async () => mocks.config,
    createChat: mocks.createChat,
    getChat: mocks.getChat,
    addChatMessage: mocks.addChatMessage,
    listChatMessages: mocks.listChatMessages
  }
}))
vi.mock("@/services/tldw", async () => {
  const actual = await vi.importActual<typeof import("@/services/tldw")>("@/services/tldw")
  return { ...actual, tldwChat: { ...actual.tldwChat, streamMessage: mocks.streamMessage } }
})
vi.mock("@/models", () => ({ pageAssistModel: mocks.pageAssistModel }))
vi.mock("@/services/title", () => ({
  generateTitle: async () => "First question"
}))
vi.mock("@/services/tldw-server", () => ({
  systemPromptForNonRagOption: async () => ""
}))
vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: async ({
    content
  }: {
    content: Array<{ text: string }>
  }) => ({ role: "user", content: content[0].text })
}))
vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: async (history: unknown[]) => history
}))
vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: mocks.getActorSettings
}))
vi.mock("@/db/dexie/schema", () => ({ db: {
  chatHistories: { get: async (id: string) => mocks.mirrorHistories.get(id) },
  messages: {
    get: async (id: string) => mocks.rows.find(row => row.id === id),
    where: (field: string) => ({ equals: (value: unknown) => ({ toArray: async () => mocks.rows.filter(row => row[field] === value) }) }),
    add: async (row: Record<string, unknown>) => { mocks.rows.push(row); return row.id },
    put: async (row: Record<string, unknown>) => {
      const index = mocks.rows.findIndex(item => item.id === row.id)
      if (index < 0) mocks.rows.push(row)
      else mocks.rows[index] = row
      return row.id
    }
  }
} }))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  getSelectedAssistantOperationRevision: () => mocks.selectionRevision,
  useSelectedAssistant: () => [null, setLoaderSelection]
}))
function setLoaderSelection() { mocks.selectionRevision++ }

vi.mock("@/db/dexie/helpers", () => ({
  acknowledgeSavedUserMessage: async (historyId: string, id: string, serverMessageId: string) => {
    const row = mocks.rows.find(row => row.history_id === historyId && row.id === id && row.role === "user")
    if (row) {
      if (row.serverMessageId && row.serverMessageId !== serverMessageId) throw new Error("The saved user message changed. Reload the conversation before retrying.")
      row.serverMessageId = serverMessageId
    }
  },
  generateID: () => crypto.randomUUID(),
  saveHistory: mocks.saveHistory,
  saveMessage: mocks.saveMessage,
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(),
  removeMessageByIndex: vi.fn(),
  removeMessageById: mocks.removeMessageById,
  formatToChatHistory: (items: unknown) => items,
  formatToMessage: (items: Array<Record<string, unknown>>) => mocks.withLoader ? items.map(row => ({
    ...row, id: row.id, serverMessageId: row.serverMessageId, message: row.content,
    isBot: row.role !== "user", name: row.name || "You", sources: row.sources || [],
    parentMessageId: row.parent_message_id
  })) : items,
  getSessionFiles: async () => [],
  getPromptById: async () => null,
  updateLastUsedModel: vi.fn(),
  updateLastUsedPrompt: vi.fn(),
  updateChatHistoryCreatedAt: vi.fn(),
  addFileToSession: vi.fn()
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: async (
    signal: AbortSignal | undefined,
    work: () => Promise<unknown>
  ) => {
    if (signal?.aborted)
      throw new DOMException("Request scope changed", "AbortError")
    const result = await work()
    if (signal?.aborted)
      throw new DOMException("Request scope changed", "AbortError")
    return result
  }
}))
vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: async () => null
}))
vi.mock("@/db/dexie/branch", () => ({ generateBranchFromMessageIds: vi.fn() }))
vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: vi.fn()
}))
vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: async () => ({ hasChatSaveToDb: true })
}))
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: {}, updateSettings: vi.fn() })
}))
vi.mock("@/hooks/playground", () => ({
  usePersistenceMode: () => ({
    persistenceTooltip: "Saved",
    focusConnectionCard: vi.fn(),
    getPersistenceModeLabel: () => "Saved"
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })
  return { promise, resolve, reject }
}

const createHookOptions = () => ({
  t: (_key: string, fallback?: string) => fallback || _key,
  notification: {
    error: mocks.notifyError,
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
  historyId: "history-persona",
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
  ensureServerChatHistoryId: vi.fn(async () => "history-persona"),
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

const NotificationContext = React.createContext<
  ReturnType<typeof notification.useNotification>[0] | null
>(null)
const NotificationHost = ({ children }: React.PropsWithChildren) => {
  const [api, holder] = notification.useNotification()
  return (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <NotificationContext.Provider value={api}>
        {holder}
        {children}
      </NotificationContext.Provider>
    </ConfigProvider>
  )
}

const loaderTranslate = ((_key: string, options?: { defaultValue?: string }) => options?.defaultValue || _key) as Parameters<typeof useServerChatLoader>[0]["t"]
const loaderNotification = { error: mocks.notifyError }
const ServerLoaderHost = ({ children }: React.PropsWithChildren) => {
  useServerChatLoader({ ensureServerChatHistoryId: mocks.ensureHistory, notification: loaderNotification, t: loaderTranslate })
  return <>{children}</>
}

const renderWorkspace = (realNotifications = false, withLoader = false) =>
  renderHook(
    ({ ready }) => {
      const notificationApi = React.useContext(NotificationContext)
      const state = useStoreMessageOption()
      const actions = useChatActions({
        ...createHookOptions(),
        ...state,
        ensureServerChatHistoryId: mocks.ensureHistory,
        selectedCharacter: null,
        selectedAssistant: null
      } as Parameters<typeof useChatActions>[0])
      const persistence = usePlaygroundPersistence({
        ...state,
        isConnectionReady: ready,
        isFireFoxPrivateMode: false,
        setTemporaryChat: (temporaryChat) =>
          useStoreMessageOption.setState({ temporaryChat }),
        clearChat: vi.fn(),
        selectedCharacter: null,
        selectedAssistantMode: null,
        assistantOverlayActive: false,
        serverPersistenceHintSeen: true,
        setServerPersistenceHintSeen: vi.fn(),
        invalidateServerChatHistory: vi.fn(),
        navigate: vi.fn(),
        notificationApi: notificationApi || createHookOptions().notification,
        t: createHookOptions().t
      } as Parameters<typeof usePlaygroundPersistence>[0])
      const queue = useComposerQueue({
        isConnectionReady: ready,
        isStreaming: state.streaming,
        queuedMessages: state.queuedMessages,
        setQueuedMessages: state.setQueuedMessages,
        sendQueuedRequest: async (item) => {
          await actions.onSubmit({
            message: item.promptText,
            image: item.image
          })
        },
        stopStreamingRequest: vi.fn(),
        resolveConversationId: () => state.historyId,
        buildQueuedDocuments: () => [],
        buildQueuedRequestSnapshot: () => ({
          selectedModel: state.selectedModel
        }),
        isQueuedDispatchBlocked: false,
        cancelCurrentAndRunDisabledReasonText: null
      })
      return { actions, persistence, state, queue }
    },
    {
      initialProps: { ready: true },
      wrapper: withLoader ? ServerLoaderHost : realNotifications ? NotificationHost : undefined
    }
  )

const replaceAuthority = (apiKey: string) => {
  mocks.config = { ...mocks.config, apiKey }
  window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
  for (const watch of mocks.watches)
    watch.tldwConfig({ newValue: mocks.config })
}

const seedLocalDraft = () =>
  useStoreMessageOption.setState({
    historyId: "local-draft",
    history: [
      { role: "user", content: "Own draft" },
      { role: "assistant", content: "Own answer" }
    ],
    messages: [
      {
        id: "draft-user",
        isBot: false,
        name: "You",
        message: "Own draft",
        sources: []
      },
      {
        id: "draft-assistant",
        isBot: true,
        name: "Assistant",
        message: "Own answer",
        sources: []
      }
    ]
  })

describe("saved normal Chat pipeline with autosave", () => {
  it.each(["local", "server"])("deletes a qualified mirror row and clears its %s reply target using the canonical request ID", async replyKind => {
    const localId = "history-A:server:answer"
    const serverId = "canonical-answer"
    useStoreMessageOption.setState({ historyId: "mirror", serverChatId: "cedar", temporaryChat: false,
      messages: [{ id: localId, serverMessageId: serverId, serverMessageVersion: 2, role: "assistant", isBot: true, name: "Cedar", message: "Cedar reply" }],
      history: [{ role: "assistant", content: "Cedar reply" }],
      replyTarget: { id: replyKind === "local" ? localId : serverId, role: "assistant", text: "Cedar reply" }
    })
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.deleteMessage(0) })
    expect(mocks.deleteMessage).toHaveBeenCalledWith(serverId, 2, "cedar")
    expect(mocks.removeMessageById).toHaveBeenCalledWith("mirror", localId)
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreMessageOption.getState().replyTarget).toBeNull()
  })

  it.each([0, 1])(
    "retains incomplete promotion across same-authority reconnect after %s copies",
    async (copied) => {
      const originalAdd = mocks.addChatMessage.getMockImplementation()!
      if (copied) mocks.addChatMessage.mockImplementationOnce(originalAdd)
      mocks.addChatMessage.mockRejectedValueOnce(new Error("Copy interrupted"))
      seedLocalDraft()
      const { result, rerender } = renderWorkspace(true)
      await screen.findByRole("button", { name: "Retry saving chat" })
      rerender({ ready: false })
      rerender({ ready: true })
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Blocked follow-up",
          image: ""
        })
      })
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      expect(mocks.watches.size).toBe(1)
      fireEvent.click(screen.getByRole("button", { name: "Retry saving chat" }))
      await waitFor(() =>
        expect(mocks.serverRows.get("chat-1")).toHaveLength(2)
      )
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Follow-up question",
          image: ""
        })
      })
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual(
        [
          "Own draft",
          "Own answer",
          "Follow-up question",
          "Answer: Follow-up question"
        ]
      )
    }
  )

  it.each(["create", "first copy"])(
    "pauses rather than completes when readiness is lost during %s",
    async (boundary) => {
      const blocked = deferred<void>()
      if (boundary === "create") {
        mocks.createChat.mockImplementationOnce(async () => {
          await blocked.promise
          mocks.serverRows.set("chat-1", [])
          return { id: "chat-1" }
        })
      } else {
        const originalAdd = mocks.addChatMessage.getMockImplementation()!
        mocks.addChatMessage.mockImplementationOnce(async (...args) => {
          await blocked.promise
          return originalAdd(...args)
        })
      }
      seedLocalDraft()
      const { result, rerender } = renderWorkspace(true)
      await waitFor(() =>
        expect(
          boundary === "create" ? mocks.createChat : mocks.addChatMessage
        ).toHaveBeenCalledTimes(1)
      )
      rerender({ ready: false })
      await act(async () => {
        blocked.resolve()
      })
      await screen.findByRole("button", { name: "Retry saving chat" })
      expect(result.current.state.serverChatId).toBe("chat-1")
      expect(mocks.serverRows.get("chat-1")).toHaveLength(
        boundary === "create" ? 0 : 1
      )
      expect(mocks.listChatMessages).not.toHaveBeenCalled()
      rerender({ ready: true })
      fireEvent.click(screen.getByRole("button", { name: "Retry saving chat" }))
      await waitFor(() =>
        expect(mocks.serverRows.get("chat-1")).toHaveLength(2)
      )
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.listChatMessages).toHaveBeenCalledWith(
        "chat-1",
        { limit: 200, offset: 0, render_placeholders: false },
        expect.objectContaining({ fresh: true })
      )
    }
  )

  it.each(["create", "acknowledged copy"])(
    "blocks changed server history after a readiness pause during %s",
    async (boundary) => {
      const blocked = deferred<void>()
      if (boundary === "create") {
        mocks.createChat.mockImplementationOnce(async () => {
          await blocked.promise
          mocks.serverRows.set("chat-1", [])
          return { id: "chat-1" }
        })
      } else {
        const originalAdd = mocks.addChatMessage.getMockImplementation()!
        mocks.addChatMessage.mockImplementationOnce(async (...args) => {
          await blocked.promise
          return originalAdd(...args)
        })
      }
      seedLocalDraft()
      const { result, rerender } = renderWorkspace(true)
      await waitFor(() =>
        expect(
          boundary === "create" ? mocks.createChat : mocks.addChatMessage
        ).toHaveBeenCalledTimes(1)
      )
      rerender({ ready: false })
      await act(async () => blocked.resolve())
      await screen.findByRole("button", { name: "Retry saving chat" })
      const stored = mocks.serverRows.get("chat-1")!
      if (boundary === "create") {
        stored.push({
          id: "another-tab-row",
          role: "user",
          content: "Another tab changed this history"
        })
      } else {
        stored[0].id = "replaced-by-another-tab"
      }
      rerender({ ready: true })
      fireEvent.click(screen.getByRole("button", { name: "Retry saving chat" }))
      await act(async () => {})
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Must remain blocked",
          image: ""
        })
      })
      expect(mocks.addChatMessage).toHaveBeenCalledTimes(
        boundary === "create" ? 0 : 1
      )
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(
        screen.getByRole("button", { name: "Retry saving chat" })
      ).toBeInTheDocument()
    }
  )

  it("reconciles literal placeholder text using raw listing content", async () => {
    const originalAdd = mocks.addChatMessage.getMockImplementation()!
    mocks.addChatMessage.mockImplementationOnce(originalAdd)
    mocks.addChatMessage.mockRejectedValueOnce(new Error("Copy interrupted"))
    mocks.listChatMessages.mockImplementation(async (id, params) =>
      mocks.serverRows.get(id)!.map((row) => ({
        ...row,
        content:
          params.render_placeholders === false
            ? row.content
            : row.content
                .replace("{{user}}", "User")
                .replace("<CHAR>", "Assistant")
      }))
    )
    seedLocalDraft()
    const text = "Explain {{user}} and <CHAR>"
    useStoreMessageOption.setState((state) => ({
      messages: state.messages.map((row, index) =>
        index === 0 ? { ...row, message: text } : row
      ),
      history: [
        { role: "user", content: text },
        { role: "assistant", content: "Own answer" }
      ]
    }))
    const { result } = renderWorkspace(true)
    fireEvent.click(
      await screen.findByRole("button", { name: "Retry saving chat" })
    )
    await waitFor(() => expect(mocks.serverRows.get("chat-1")).toHaveLength(2))
    expect(result.current.state.history[0].content).toBe(text)
    expect(mocks.serverRows.get("chat-1")?.[0].content).toBe(text)
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.rows.length = 0
    mocks.mirrorHistories.clear()
    mocks.withLoader = false
    mocks.selectionRevision = 0
    mocks.serverRows.clear()
    mocks.watches.clear()
    mocks.config = {
      serverUrl: "https://chat.test",
      authMode: "single-user",
      apiKey: "synthetic-a"
    }
    useStoreMessageOption.setState(
      {
        ...useStoreMessageOption.getInitialState(),
        temporaryChat: false,
        selectedModel: "llama/test-model"
      },
      true
    )
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getActorSettings.mockResolvedValue(null)
    mocks.createChat.mockImplementation(async (payload) => {
      const id = `chat-${mocks.serverRows.size + 1}`
      mocks.serverRows.set(id, [])
      return { id, title: payload.title, source: payload.source, version: 1 }
    })
    mocks.getChat.mockImplementation(async (id) => ({
      id,
      scope_type: "global"
    }))
    mocks.addChatMessage.mockImplementation(async (id, row) => {
      const stored = { ...row, id: crypto.randomUUID(), version: 1 }
      mocks.serverRows.get(id)!.push(stored)
      return stored
    })
    mocks.listChatMessages.mockImplementation(async (id, params) =>
      mocks.serverRows
        .get(id)!
        .slice(params.offset, params.offset + params.limit)
    )
    mocks.saveHistory.mockResolvedValue({ id: "local-1" })
    mocks.saveMessage.mockImplementation(async (row) => {
      mocks.rows.push(row)
    })
    mocks.ensureHistory.mockImplementation(async (chatId, title, _signal, snapshot) => {
      const localId = useStoreMessageOption.getState().historyId || "local-1"
      useStoreMessageOption
        .getState()
        .setHistoryId(localId, { preserveServerChatId: true })
      if (snapshot) mocks.mirrorHistories.set(localId, { id: localId, title, server_chat_id: chatId, server_scope_key: serverChatMirrorOwnerKey(snapshot) })
      return localId
    })
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => {
      const saved = !useStoreMessageOption.getState().temporaryChat
      // The transport creates an implicit default-character chat only when the
      // application fails to provide the canonical saved conversation.
      const id = saved ? (conversationId ?? "implicit-chat") : undefined
      if (id && !mocks.serverRows.has(id)) mocks.serverRows.set(id, [])
      return {
        saveToDb: saved,
        conversationId: id,
        stream: async function* (
          messages: Array<{ role: string; content: string }>
        ) {
          const question = messages[messages.length - 1].content
          if (id)
            mocks.serverRows
              .get(id)!
              .push(
                { role: "user", content: question },
                { role: "assistant", content: `Answer: ${question}` }
              )
          yield `Answer: ${question}`
        }
      }
    })
  })

  it("hydrates a failed saved turn before Retry without duplicating the canonical user", async () => {
    mocks.withLoader = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const projected: unknown[][] = []
    mocks.pageAssistModel.mockImplementation(async ({ conversationId, clientMessageId, retryFailedTurn }) =>
      new ChatTldw({ model: "test", saveToDb: true, conversationId, clientMessageId, retryFailedTurn }))
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      projected.push(messages)
      if (projected.length === 1) {
        mocks.serverRows.get(options.conversationId)!.push({ id: "saved-failed-user", role: "user", content: "Retry question", metadata_extra: { client_message_id: options.clientMessageId } })
        throw new Error("Provider failed before response metadata")
      }
      mocks.serverRows.get(options.conversationId)!.push({ id: "saved-answer", role: "assistant", content: "Recovered final answer" })
      onChunk({ tldw_user_message_id: "saved-failed-user", tldw_message_id: "saved-answer" })
      yield "Recovered final answer"
    })
    let view = renderWorkspace(false, true)
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Retry question", image: "" }) })
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => !row.isBot)).toHaveLength(1)
    expect(view.result.current.state.messages.find(row => !row.isBot)?.serverMessageId).toBe("saved-failed-user")
    expect(mocks.rows.filter(row => row.role === "user")).toHaveLength(1)
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(JSON.stringify(projected[1]).match(/Retry question/g)).toHaveLength(1)
    expect(view.result.current.state.messages.at(-1)?.message).toBe("Recovered final answer")
    const localId = view.result.current.state.messages.find(row => !row.isBot)?.id
    view.unmount()
    act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => !row.isBot).map(row => row.id)).toEqual([localId])
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "saved-answer")).toMatchObject([{ message: "Recovered final answer" }])
    view.unmount()
  })

  it.each(["B", "A"])("does not apply delayed failed-user correlation after A to B to %s", async destination => {
    mocks.withLoader = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const response = deferred<Array<{ id: string; role: string; content: string; metadata_extra: Record<string, unknown> }>>()
    mocks.listChatMessages.mockReturnValueOnce(response.promise)
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => ({
      saveToDb: true, conversationId, stream: async function* () { yield await Promise.reject(new Error("Provider failed")) }
    }))
    const view = renderWorkspace(false, true)
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Alice pending question", image: "" }) })
    const original = view.result.current.state.messages.find(row => !row.isBot)!
    await waitFor(() => expect(mocks.listChatMessages).toHaveBeenCalled())
    const draft = { id: "new-owner-draft", isBot: false, name: "You", message: "New owner work", sources: [] }
    await act(async () => {
      replaceAuthority("synthetic-b")
      if (destination === "A") replaceAuthority("synthetic-a")
      useStoreMessageOption.setState({ serverChatId: null, historyId: "new-owned-history", messages: [draft], history: [{ role: "user", content: draft.message }] })
      response.resolve([{ id: "late-canonical-user", role: "user", content: original.message, metadata_extra: { client_message_id: original.id } }])
      await response.promise
    })
    expect(view.result.current.state.messages).toEqual([draft])
    expect(mocks.rows.find(row => row.id === original.id)?.serverMessageId).toBeUndefined()
    view.unmount()
  })

  it("successful regeneration does not re-ACK the already acknowledged user", async () => {
    let calls = 0
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => ({
      saveToDb: true,
      conversationId,
      userServerMessageId: `server-user-${++calls}`,
      stream: async function* () { yield "Ordinary final answer" }
    }))
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.onSubmit({ message: "Ordinary question", image: "" }) })
    const original = result.current.state.messages.find(row => !row.isBot)?.serverMessageId
    expect(original).toBe("server-user-1")
    await act(async () => { await result.current.actions.regenerateLastMessage() })
    expect(mocks.pageAssistModel.mock.calls.map(([options]) => options.retryFailedTurn)).toEqual([false, false])
    expect(result.current.state.messages.find(row => !row.isBot)?.serverMessageId).toBe(original)
    expect(mocks.rows.filter(row => row.role === "user").map(row => row.serverMessageId)).toEqual([original])
  })

  it("real failed-turn regeneration sends the intended user once without its display error", async () => {
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const projected: unknown[][] = []
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => ({
      saveToDb: true, conversationId,
      userServerMessageId: projected.length ? "retry-user" : undefined,
      stream: async function* (messages: unknown[]) {
        projected.push(messages)
        if (projected.length === 1) throw new Error("Provider failed")
        yield "Recovered final answer"
      }
    }))
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.onSubmit({ message: "Retry question", image: "" }) })
    expect(result.current.state.messages.at(-1)?.message).toContain("__tldw_error__:")
    await act(async () => { await result.current.actions.regenerateLastMessage() })
    const request = JSON.stringify(projected[1])
    expect(request).not.toContain("__tldw_error__:")
    expect(request.match(/Retry question/g)).toHaveLength(1)
    expect(result.current.state.messages.filter(row => !row.isBot)).toHaveLength(1)
    expect(result.current.state.messages.at(-1)?.message).toBe("Recovered final answer")
    expect(mocks.pageAssistModel.mock.calls.map(([options]) => options.retryFailedTurn)).toEqual([false, true])
    expect(result.current.state.messages.find(row => !row.isBot)?.serverMessageId).toBe("retry-user")
    expect(mocks.rows.filter(row => row.role === "user").map(row => row.serverMessageId)).toEqual(["retry-user"])
  })

  it("persists a real acknowledged reasoning-only stream as one recoverable saved pair", async () => {
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => new ChatTldw({ model: "test", saveToDb: true, conversationId }))
    mocks.streamMessage.mockImplementation(async function* (_messages, options, onChunk) {
      onChunk({ tldw_conversation_id: options.conversationId, tldw_user_message_id: "reasoning-user" })
      yield "<think>Still working</think>"
      onChunk({ tldw_message_id: "reasoning-assistant" })
    })
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.onSubmit({ message: "Question", image: "" }) })
    expect(mocks.rows.map(row => [row.role, row.serverMessageId, row.content])).toEqual([
      ["user", "reasoning-user", "Question"],
      ["assistant", "reasoning-assistant", "<think>Still working</think>"]
    ])
    expect(result.current.state.messages.at(-1)).toMatchObject({
      serverMessageId: "reasoning-assistant",
      generationInfo: { interrupted: true, interruptionReason: expect.stringContaining("final answer") }
    })
    expect(result.current.state.history.map(row => row.content)).toEqual(["Question", "<think>Still working</think>"])
    expect(mocks.addChatMessage).not.toHaveBeenCalled()
  })

  it("acknowledges both saved turns before backlink eligibility and reload without consuming an identical unsent draft", async () => {
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => new ChatTldw({
      model: "test", saveToDb: true, conversationId
    }))
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      const rows = mocks.serverRows.get(options.conversationId)!
      if (!rows.length) rows.push({ id: "system", role: "system", content: "System" })
      const userId = `user-${rows.length}`
      const assistantId = `assistant-${rows.length}`
      const content = messages.at(-1)!.content
      rows.push({ id: userId, role: "user", content },
        { id: assistantId, role: "assistant", content: `Answer: ${content}` })
      onChunk({ tldw_conversation_id: options.conversationId, tldw_user_message_id: userId })
      yield `Answer: ${content}`
      onChunk({ tldw_message_id: assistantId })
    })
    const { result } = renderWorkspace()
    for (const message of ["Same question", "Same question"]) {
      await act(async () => { await result.current.actions.onSubmit({ message, image: "" }) })
    }
    const visible = useStoreMessageOption.getState().messages
    expect(visible.filter(row => row.message?.trim() && !row.serverMessageId)).toEqual([])
    const local = mocks.rows.map(row => ({
      id: String(row.id), serverMessageId: row.serverMessageId as string | undefined,
      message: String(row.content), isBot: row.role !== "user", name: String(row.name || ""),
      sources: [], parentMessageId: row.parent_message_id as string | null
    }))
    expect(local.map(row => row.serverMessageId)).toEqual(["user-1", "assistant-1", "user-3", "assistant-3"])
    const remote = mocks.serverRows.get("chat-1")!.map(row => ({
      id: row.id!, serverMessageId: row.id!, message: row.content,
      isBot: row.role !== "user", role: row.role, name: "", sources: []
    }))
    expect(reconcileServerChatMessages(local, remote)).toHaveLength(5)
    const draft = { id: "unsent", message: "Same question", isBot: false, name: "You", sources: [] }
    expect(reconcileServerChatMessages([...local, draft], remote)).toEqual([
      ...reconcileServerChatMessages(local, remote), draft
    ])
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
  })

  it.each(["B", "A"])("does not apply late stream acknowledgements after switching A to B to %s", async (destination) => {
    const held = deferred<void>()
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => new ChatTldw({ model: "test", saveToDb: true, conversationId }))
    mocks.streamMessage.mockImplementation(async function* (_messages, options, onChunk) {
      await held.promise
      onChunk({ tldw_conversation_id: options.conversationId, tldw_user_message_id: "old-user", tldw_message_id: "old-assistant" })
      yield "Old reply"
    })
    const { result } = renderWorkspace()
    let pending!: Promise<unknown>
    act(() => { pending = result.current.actions.onSubmit({ message: "Old question", image: "" }) })
    await waitFor(() => expect(mocks.streamMessage).toHaveBeenCalled())
    const draft = { id: "new-draft", isBot: false, name: "You", message: "New draft" }
    await act(async () => {
      replaceAuthority("synthetic-b")
      if (destination === "A") replaceAuthority("synthetic-a")
      useStoreMessageOption.setState({ historyId: "new-local", serverChatId: "new-chat", messages: [draft], history: [] })
      held.resolve()
      await pending
    })
    expect(result.current.state.messages).toEqual([draft])
    expect(mocks.rows).toEqual([])
    expect(result.current.state.serverChatId).toBe("new-chat")
  })

  it("establishes one neutral conversation and local history before inference, including a queued second turn", async () => {
    const linking = deferred<string>()
    mocks.ensureHistory.mockImplementationOnce(async () => {
      const id = await linking.promise
      useStoreMessageOption
        .getState()
        .setHistoryId(id, { preserveServerChatId: true })
      return id
    })
    const { result } = renderWorkspace()
    let first!: Promise<unknown>
    act(() => {
      first = result.current.actions.onSubmit({
        message: "First question",
        image: ""
      })
    })
    await waitFor(() => expect(mocks.ensureHistory).toHaveBeenCalledTimes(1))
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    act(() => {
      result.current.queue.enqueue({ promptText: "Second question", image: "" })
    })
    expect(result.current.state.queuedMessages).toHaveLength(1)
    await act(async () => {
      linking.resolve("local-1")
      await first
    })
    await waitFor(() =>
      expect(result.current.state.queuedMessages).toHaveLength(0)
    )
    await waitFor(() => {
      expect(mocks.notifyError.mock.calls).toEqual([])
      expect(mocks.pageAssistModel).toHaveBeenCalledTimes(2)
    })
    expect(mocks.serverRows.size).toBe(1)
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
    expect(mocks.createChat.mock.calls[0][0]).not.toHaveProperty("character_id")
    expect(result.current.state.serverChatCharacterId).toBeNull()
    expect(result.current.state.serverChatAssistantKind).toBeNull()
    expect(
      mocks.pageAssistModel.mock.calls.map(
        ([request]) => request.conversationId
      )
    ).toEqual(["chat-1", "chat-1"])
    expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual([
      "First question",
      "Answer: First question",
      "Second question",
      "Answer: Second question"
    ])
    expect(mocks.rows.map((row) => row.history_id)).toEqual([
      "local-1",
      "local-1",
      "local-1",
      "local-1"
    ])
  })

  it("cannot autosave a second conversation while the completed first pair is waiting for local persistence", async () => {
    const saving = deferred<void>()
    mocks.saveMessage.mockImplementationOnce(async (row) => {
      await saving.promise
      mocks.rows.push(row)
    })
    const { result } = renderWorkspace()
    let first!: Promise<unknown>
    act(() => {
      first = result.current.actions.onSubmit({
        message: "First question",
        image: ""
      })
    })
    await waitFor(() => expect(result.current.state.history).toHaveLength(2))
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.serverRows.size).toBe(1)
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
    await act(async () => {
      saving.resolve()
      await first
    })
    expect(result.current.state.serverChatId).toBe("chat-1")
  })

  it("keeps temporary inference unsaved and promotes its existing pair only once", async () => {
    useStoreMessageOption.setState({ temporaryChat: true })
    const { result } = renderWorkspace()
    await act(async () => {
      await result.current.actions.onSubmit({
        message: "Temporary question",
        image: ""
      })
    })
    expect(mocks.serverRows.size).toBe(0)
    expect(mocks.createChat).not.toHaveBeenCalled()
    act(() => useStoreMessageOption.setState({ temporaryChat: false }))
    await waitFor(() => expect(mocks.serverRows.get("chat-1")).toHaveLength(2))
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
    expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual([
      "Temporary question",
      "Answer: Temporary question"
    ])
  })

  it("returns a recoverable creation failure and retries without a duplicate pair", async () => {
    mocks.createChat.mockRejectedValueOnce(new Error("Creation unavailable"))
    const { result } = renderWorkspace()
    let outcome!: Awaited<ReturnType<typeof result.current.actions.onSubmit>>
    await act(async () => {
      outcome = await result.current.actions.onSubmit({
        message: "First question",
        image: ""
      })
    })
    expect(outcome.status).toBe("failed")
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(result.current.state.serverChatId).toBeNull()
    expect(result.current.state.streaming).toBe(false)
    await act(async () => {
      await result.current.actions.onSubmit({
        message: "First question",
        image: ""
      })
    })
    expect(mocks.serverRows.get("chat-1")).toHaveLength(2)
    expect(mocks.serverRows.size).toBe(1)
  })

  it.each(["create", "message"])(
    "waits for the entire local promotion during %s before a follow-up and queued turn",
    async (boundary) => {
      const blocked = deferred<void>()
      if (boundary === "create") {
        mocks.createChat.mockImplementationOnce(async () => {
          mocks.serverRows.set("promoted-chat", [])
          await blocked.promise
          return { id: "promoted-chat" }
        })
      } else {
        mocks.addChatMessage.mockImplementationOnce(async (id, row) => {
          await blocked.promise
          mocks.serverRows.get(id)!.push(row)
          return { id: crypto.randomUUID(), version: 1 }
        })
      }
      seedLocalDraft()
      const { result } = renderWorkspace()
      await waitFor(() =>
        expect(
          boundary === "create" ? mocks.createChat : mocks.addChatMessage
        ).toHaveBeenCalledTimes(1)
      )
      let followUp!: Promise<unknown>
      act(() => {
        followUp = result.current.actions.onSubmit({
          message: "Follow-up question",
          image: ""
        })
      })
      await waitFor(() => expect(mocks.initialize).toHaveBeenCalledTimes(2))
      act(() => {
        result.current.queue.enqueue({
          promptText: "Queued question",
          image: ""
        })
      })
      await act(async () => {
        await Promise.resolve()
      })
      const inferredBeforePromotion = mocks.pageAssistModel.mock.calls.length
      await act(async () => {
        blocked.resolve()
        await followUp
      })
      await waitFor(() =>
        expect(result.current.state.queuedMessages).toHaveLength(0)
      )
      expect(inferredBeforePromotion).toBe(0)
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.serverRows.size).toBe(1)
      expect(
        mocks.serverRows
          .get(result.current.state.serverChatId!)
          ?.map((row) => row.content)
      ).toEqual([
        "Own draft",
        "Own answer",
        "Follow-up question",
        "Answer: Follow-up question",
        "Queued question",
        "Answer: Queued question"
      ])
      expect(result.current.state.historyId).toBe("local-draft")
    }
  )

  it("rechecks the current conversation after promotion finishes during send initialization", async () => {
    const promotion = deferred<void>()
    const sending = deferred<void>()
    mocks.addChatMessage.mockImplementationOnce(async (id, row) => {
      await promotion.promise
      mocks.serverRows.get(id)!.push(row)
      return { id: crypto.randomUUID(), version: 1 }
    })
    seedLocalDraft()
    const { result } = renderWorkspace()
    // The send captures the initial null ID before the autosave creates its chat.
    mocks.initialize.mockReturnValueOnce(sending.promise)
    let followUp!: Promise<unknown>
    act(() => {
      followUp = result.current.actions.onSubmit({
        message: "Follow-up question",
        image: ""
      })
    })
    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalledTimes(1))
    await act(async () => {
      promotion.resolve()
    })
    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalledTimes(2))
    await act(async () => {
      sending.resolve()
      await followUp
    })
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
    expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual([
      "Own draft",
      "Own answer",
      "Follow-up question",
      "Answer: Follow-up question"
    ])
  })

  it("cancels a waiting follow-up without cancelling the owning promotion", async () => {
    const creating = deferred<void>()
    mocks.createChat.mockImplementationOnce(async () => {
      mocks.serverRows.set("promoted-chat", [])
      await creating.promise
      return { id: "promoted-chat" }
    })
    seedLocalDraft()
    const { result } = renderWorkspace()
    await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
    const controller = new AbortController()
    let followUp!: Promise<unknown>
    act(() => {
      followUp = result.current.actions.onSubmit({
        message: "Cancelled follow-up",
        image: "",
        controller
      })
    })
    await waitFor(() => expect(mocks.initialize).toHaveBeenCalledTimes(2))
    await act(async () => {
      controller.abort()
      await followUp
    })
    expect(mocks.createChat.mock.calls[0][1].signal.aborted).toBe(false)
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    await act(async () => {
      creating.resolve()
    })
    await waitFor(() =>
      expect(mocks.serverRows.get("promoted-chat")).toHaveLength(2)
    )
    await act(async () => {
      await result.current.actions.onSubmit({
        message: "Retry follow-up",
        image: ""
      })
    })
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
    expect(
      mocks.serverRows.get("promoted-chat")?.map((row) => row.content)
    ).toEqual([
      "Own draft",
      "Own answer",
      "Retry follow-up",
      "Answer: Retry follow-up"
    ])
  })

  it("cleans up a failed promotion so retry saves the original pair before the next turn", async () => {
    const creating = deferred<{ id: string }>()
    mocks.createChat.mockReturnValueOnce(creating.promise)
    seedLocalDraft()
    const { result } = renderWorkspace()
    await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
    let followUp!: Promise<unknown>
    act(() => {
      followUp = result.current.actions.onSubmit({
        message: "Follow-up question",
        image: ""
      })
    })
    await waitFor(() => expect(mocks.initialize).toHaveBeenCalledTimes(2))
    await act(async () => {
      creating.reject(new Error("Creation unavailable"))
      await followUp
    })
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(result.current.state.history.map((row) => row.content)).toEqual([
      "Own draft",
      "Own answer"
    ])
    await act(async () => {
      await result.current.persistence.handleSaveChatToServer()
    })
    await act(async () => {
      await result.current.actions.onSubmit({
        message: "Follow-up question",
        image: ""
      })
    })
    expect(mocks.serverRows.size).toBe(1)
    expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual([
      "Own draft",
      "Own answer",
      "Follow-up question",
      "Answer: Follow-up question"
    ])
    expect(mocks.watches.size).toBe(0)
  })

  it.each([0, 1, "ambiguous commit"])(
    "retries incomplete promotion through the visible notification after %s acknowledged copies",
    async (copied) => {
      const originalAdd = mocks.addChatMessage.getMockImplementation()!
      if (copied === 1) mocks.addChatMessage.mockImplementationOnce(originalAdd)
      mocks.addChatMessage.mockImplementationOnce(async (id, row) => {
        if (copied === "ambiguous commit") await originalAdd(id, row)
        throw new Error("Copy temporarily unavailable")
      })
      seedLocalDraft()
      const { result } = renderWorkspace(true)
      await screen.findByRole("button", { name: "Retry saving chat" })
      const acknowledgedIds = mocks.serverRows
        .get("chat-1")!
        .map((row) => row.id)
      // Dismissal must not strand recovery: a blocked send reopens the action.
      fireEvent.click(screen.getByRole("button", { name: "Close" }))
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Blocked follow-up",
          image: ""
        })
      })
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      const retry = await screen.findByRole("button", {
        name: "Retry saving chat"
      })
      const reading = deferred<void>()
      const originalRead = mocks.listChatMessages.getMockImplementation()!
      mocks.listChatMessages.mockImplementationOnce(async (...args) => {
        await reading.promise
        return originalRead(...args)
      })
      fireEvent.click(retry)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Retry saving chat" })
        ).toBeDisabled()
      )
      await act(async () => {
        reading.resolve()
      })
      await waitFor(() =>
        expect(mocks.serverRows.get("chat-1")).toHaveLength(2)
      )
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: "Retry saving chat" })
        ).toBeNull()
      )
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Follow-up question",
          image: ""
        })
      })
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.serverRows.get("chat-1")?.map((row) => row.content)).toEqual(
        [
          "Own draft",
          "Own answer",
          "Follow-up question",
          "Answer: Follow-up question"
        ]
      )
      expect(
        mocks.serverRows
          .get("chat-1")!
          .slice(0, acknowledgedIds.length)
          .map((row) => row.id)
      ).toEqual(acknowledgedIds)
      expect(mocks.listChatMessages).toHaveBeenCalledWith(
        "chat-1",
        expect.any(Object),
        expect.objectContaining({
          fresh: true,
          requestScope: expect.any(Object)
        })
      )
      expect(mocks.watches.size).toBe(0)
    }
  )

  it.each([
    "changed identity",
    "additional row",
    "reordered rows",
    "deleted acknowledgement",
    "read failure"
  ])(
    "keeps incomplete promotion blocked when reconciliation finds %s",
    async (conflict) => {
      mocks.addChatMessage.mockImplementationOnce(
        mocks.addChatMessage.getMockImplementation()!
      )
      mocks.addChatMessage.mockRejectedValueOnce(
        new Error("Copy temporarily unavailable")
      )
      seedLocalDraft()
      const { result } = renderWorkspace()
      await waitFor(() => expect(mocks.notifyError).toHaveBeenCalled())
      const rows = mocks.serverRows.get("chat-1")!
      if (conflict === "changed identity") rows[0].id = "different-id"
      if (conflict === "additional row")
        rows.push({ id: "extra", role: "user", content: "Other writer" })
      if (conflict === "reordered rows")
        rows.unshift({ id: "answer", role: "assistant", content: "Own answer" })
      if (conflict === "deleted acknowledgement") rows.length = 0
      if (conflict === "read failure")
        mocks.listChatMessages.mockRejectedValueOnce(
          new Error("Read unavailable")
        )
      await act(async () => {
        await result.current.persistence.handleSaveChatToServer()
      })
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "Blocked follow-up",
          image: ""
        })
      })
      expect(mocks.addChatMessage).toHaveBeenCalledTimes(2)
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      expect(result.current.state.history.map((row) => row.content)).toEqual([
        "Own draft",
        "Own answer"
      ])
      expect(mocks.notifyError.mock.calls.at(-1)?.[0].description).toMatch(
        /not fully saved/
      )
    }
  )

  it.each(["new history", "A to B to A", "unmount"])(
    "releases incomplete promotion and removes its retry action on %s",
    async (boundary) => {
      mocks.addChatMessage.mockRejectedValueOnce(
        new Error("Copy temporarily unavailable")
      )
      seedLocalDraft()
      const { result, unmount } = renderWorkspace(true)
      await screen.findByRole("button", { name: "Retry saving chat" })
      expect(mocks.watches.size).toBe(1)
      act(() => {
        if (boundary === "unmount") unmount()
        else if (boundary === "A to B to A") {
          replaceAuthority("synthetic-b")
          replaceAuthority("synthetic-a")
        } else
          useStoreMessageOption.setState({
            historyId: "other-local",
            serverChatId: "other-chat",
            messages: [],
            history: []
          })
      })
      await waitFor(() => expect(mocks.watches.size).toBe(0))
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: "Retry saving chat" })
        ).toBeNull()
      )
      expect(mocks.addChatMessage).toHaveBeenCalledTimes(1)
      if (boundary !== "unmount")
        expect(result.current.state.serverChatId).not.toBeNull()
    }
  )

  it("discards a delayed retry read after an A to B to A boundary", async () => {
    mocks.addChatMessage.mockRejectedValueOnce(
      new Error("Copy temporarily unavailable")
    )
    seedLocalDraft()
    const { result } = renderWorkspace()
    await waitFor(() => expect(mocks.notifyError).toHaveBeenCalled())
    const reading = deferred<unknown[]>()
    mocks.listChatMessages.mockReturnValueOnce(reading.promise)
    let retry!: Promise<void>
    act(() => {
      retry = result.current.persistence.handleSaveChatToServer()
    })
    await waitFor(() => expect(mocks.listChatMessages).toHaveBeenCalled())
    await act(async () => {
      replaceAuthority("synthetic-b")
      replaceAuthority("synthetic-a")
      useStoreMessageOption.setState({
        historyId: "new-local",
        serverChatId: "new-chat",
        history: [],
        messages: []
      })
      reading.resolve([])
      await retry
    })
    expect(mocks.addChatMessage).toHaveBeenCalledTimes(1)
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(result.current.state.serverChatId).toBe("new-chat")
    expect(mocks.watches.size).toBe(0)
  })

  it.each(["new history", "A to B to A"])(
    "does not join or publish an older promotion after %s on the same surface",
    async (boundary) => {
      const creating = deferred<void>()
      mocks.createChat.mockImplementationOnce(async () => {
        mocks.serverRows.set("old-chat", [])
        await creating.promise
        return { id: "old-chat" }
      })
      seedLocalDraft()
      const { result } = renderWorkspace()
      await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
      act(() => {
        if (boundary === "A to B to A") {
          replaceAuthority("synthetic-b")
          replaceAuthority("synthetic-a")
        }
        useStoreMessageOption.setState({
          historyId: boundary === "new history" ? "new-local" : "local-draft",
          messages: [],
          history: [],
          serverChatId: null
        })
      })
      await act(async () => {
        await result.current.actions.onSubmit({
          message: "New conversation question",
          image: ""
        })
      })
      expect(result.current.state.serverChatId).toBe("chat-2")
      await act(async () => {
        creating.resolve()
      })
      await waitFor(() => expect(mocks.watches.size).toBe(0))
      expect(result.current.state.serverChatId).toBe("chat-2")
      expect(mocks.serverRows.get("old-chat")).toEqual([])
      expect(mocks.serverRows.get("chat-2")?.map((row) => row.content)).toEqual(
        ["New conversation question", "Answer: New conversation question"]
      )
    }
  )

  it("does not publish a cancelled creation response or invoke inference", async () => {
    const creating = deferred<{ id: string }>()
    mocks.createChat.mockReturnValueOnce(creating.promise)
    const { result } = renderWorkspace()
    const controller = new AbortController()
    let pending!: Promise<unknown>
    act(() => {
      pending = result.current.actions.onSubmit({
        message: "First question",
        image: "",
        controller
      })
    })
    await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
    await act(async () => {
      controller.abort()
      creating.resolve({ id: "late-chat" })
      await pending
    })
    expect(result.current.state.serverChatId).toBeNull()
    expect(mocks.ensureHistory).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(mocks.watches.size).toBe(0)
  })

  it("does not attach a pending first-turn creation to a different local history", async () => {
    const creating = deferred<{ id: string }>()
    mocks.createChat.mockReturnValueOnce(creating.promise)
    const { result } = renderWorkspace()
    let pending!: Promise<unknown>
    act(() => {
      pending = result.current.actions.onSubmit({
        message: "Old question",
        image: ""
      })
    })
    await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
    await act(async () => {
      useStoreMessageOption.setState({
        historyId: "other-local",
        serverChatId: "other-chat"
      })
      creating.resolve({ id: "late-chat" })
      await pending
    })
    expect(mocks.ensureHistory).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(result.current.state.serverChatId).toBe("other-chat")
    expect(result.current.state.historyId).toBe("other-local")
  })

  it.each(["initialize", "create", "link"])(
    "rejects an A to B to A response during %s before inference",
    async (boundary) => {
      const blocked = deferred<unknown>()
      if (boundary === "initialize")
        mocks.initialize.mockReturnValueOnce(blocked.promise)
      if (boundary === "create")
        mocks.createChat.mockReturnValueOnce(blocked.promise)
      if (boundary === "link")
        mocks.ensureHistory.mockReturnValueOnce(blocked.promise)
      const { result } = renderWorkspace()
      let pending!: Promise<unknown>
      act(() => {
        pending = result.current.actions.onSubmit({
          message: "First question",
          image: ""
        })
      })
      const target =
        boundary === "initialize"
          ? mocks.initialize
          : boundary === "create"
            ? mocks.createChat
            : mocks.ensureHistory
      await waitFor(() => expect(target).toHaveBeenCalled())
      await act(async () => {
        replaceAuthority("synthetic-b")
        replaceAuthority("synthetic-a")
        blocked.resolve(
          boundary === "create" ? { id: "late-chat" } : "late-history"
        )
        await pending
      })
      expect(result.current.state.serverChatId).toBeNull()
      expect(result.current.state.messages).toHaveLength(0)
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      expect(mocks.watches.size).toBe(0)
    }
  )

  it("rechecks an autosave after initialization when the send path has established a conversation", async () => {
    const initializing = deferred<void>()
    mocks.initialize.mockReturnValueOnce(initializing.promise)
    seedLocalDraft()
    const { result } = renderWorkspace()
    await waitFor(() => expect(mocks.initialize).toHaveBeenCalledTimes(1))
    await act(async () => {
      useStoreMessageOption.getState().setServerChatId("canonical-chat")
      initializing.resolve()
    })
    expect(mocks.createChat).not.toHaveBeenCalled()
    expect(result.current.state.serverChatId).toBe("canonical-chat")
  })

  it.each(["stream", "local persistence"])(
    "keeps the new account's state when an old %s completes",
    async (boundary) => {
      const blocked = deferred<void>()
      if (boundary === "stream") {
        mocks.pageAssistModel.mockImplementationOnce(
          async ({ conversationId }) => ({
            saveToDb: true,
            conversationId,
            stream: async function* () {
              await blocked.promise
              yield "Old account reply"
            }
          })
        )
      } else {
        mocks.saveMessage.mockImplementationOnce(async () => {
          await blocked.promise
        })
      }
      const { result } = renderWorkspace()
      let pending!: Promise<unknown>
      act(() => {
        pending = result.current.actions.onSubmit({
          message: "First question",
          image: ""
        })
      })
      await waitFor(() =>
        expect(
          boundary === "stream" ? mocks.pageAssistModel : mocks.saveMessage
        ).toHaveBeenCalled()
      )
      const bobMessage = {
        id: "bob-message",
        isBot: false,
        name: "You",
        message: "Bob's own draft",
        sources: []
      }
      await act(async () => {
        replaceAuthority("synthetic-b")
        useStoreMessageOption.setState({
          historyId: "bob-local",
          serverChatId: "bob-chat",
          messages: [bobMessage],
          history: [{ role: "user", content: "Bob's own draft" }]
        })
        blocked.resolve()
        await pending
      })
      expect(result.current.state.messages).toEqual([bobMessage])
      expect(result.current.state.history.map((row) => row.content)).toEqual([
        "Bob's own draft"
      ])
      expect(result.current.state.serverChatId).toBe("bob-chat")
    }
  )

  it.each(["initialize", "create", "message"])(
    "stops old autosave writes after A to B to A during %s",
    async (boundary) => {
      const blocked = deferred<unknown>()
      if (boundary === "initialize")
        mocks.initialize.mockReturnValueOnce(blocked.promise)
      if (boundary === "create")
        mocks.createChat.mockReturnValueOnce(blocked.promise)
      if (boundary === "message")
        mocks.addChatMessage.mockReturnValueOnce(blocked.promise)
      seedLocalDraft()
      const { result } = renderWorkspace()
      const target =
        boundary === "initialize"
          ? mocks.initialize
          : boundary === "create"
            ? mocks.createChat
            : mocks.addChatMessage
      await waitFor(() => {
        expect(mocks.notifyError.mock.calls).toEqual([])
        expect(target).toHaveBeenCalled()
      })
      await act(async () => {
        replaceAuthority("synthetic-b")
        useStoreMessageOption.setState({
          historyId: null,
          history: [],
          serverChatId: null
        })
        replaceAuthority("synthetic-a")
        blocked.resolve(boundary === "create" ? { id: "late-chat" } : undefined)
      })
      await waitFor(() => expect(mocks.watches.size).toBe(0))
      expect(mocks.addChatMessage).toHaveBeenCalledTimes(
        boundary === "message" ? 1 : 0
      )
      expect(result.current.state.serverChatId).toBeNull()
    }
  )
})
