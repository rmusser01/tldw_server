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
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"
import { useChatActions } from "../useChatActions"
import { useServerChatLoader } from "../useServerChatLoader"
import { usePlaygroundPersistence } from "@/components/Option/Playground/hooks/usePlaygroundPersistence"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { reconcileServerChatMessages, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { useComposerQueue } from "@/components/Chat/composer/hooks/useComposerQueue"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"

const mocks = vi.hoisted(() => ({
  capability: true as boolean | "error",
  bgRequest: vi.fn(),
  removeMessageById: vi.fn(),
  deleteMessage: vi.fn(),
  createChat: vi.fn(),
  getChat: vi.fn(),
  addChatMessage: vi.fn(),
  listChatMessages: vi.fn(),
  initialize: vi.fn(),
  pageAssistModel: vi.fn(),
  getModel: vi.fn(),
  ocr: vi.fn(),
  realFormatter: false,
  realPersistence: false,
  ragSearch: vi.fn(),
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
    Array<{ id?: string; role: string; content: string; metadata_extra?: Record<string, unknown>; images?: string[]; version?: number }>
  >(),
  watches: new Set<{ tldwConfig: (change: { newValue: unknown }) => void }>(),
  config: {
    serverUrl: "https://chat.test",
    authMode: "single-user",
    apiKey: "synthetic-a"
  }
}))

vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => mocks.bgRequest(...args), bgStream: vi.fn(), bgUpload: vi.fn() }))

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
    listChatMessages: mocks.listChatMessages,
    ragSearch: mocks.ragSearch,
    listServicePrompts: async () => {
      const { ServicePromptApiError } = await import("@/services/tldw/domains/service-prompts")
      throw new ServicePromptApiError("Legacy server", { status: 404 })
    }
  }
}))
vi.mock("@/services/tldw", async () => {
  const actual = await vi.importActual<typeof import("@/services/tldw")>("@/services/tldw")
  return { ...actual, tldwModels: { ...actual.tldwModels, getModel: mocks.getModel }, tldwChat: { ...actual.tldwChat, streamMessage: mocks.streamMessage } }
})
vi.mock("@/models", () => ({ pageAssistModel: mocks.pageAssistModel }))
vi.mock("@/services/title", () => ({
  generateTitle: async () => "First question"
}))
vi.mock("@/services/tldw-server", () => ({
  LEGACY_SERVICE_PROMPT_DEFAULTS: {
    "chat.rag.answer": { template: "Context: {context}\nQuestion: {question}" },
    "chat.rag.question_rewrite": { template: "History: {chat_history}\nQuestion: {question}" }
  },
  promptForRag: async () => ({ ragPrompt: "Context: {context}\nQuestion: {question}", ragQuestionPrompt: "History: {chat_history}\nQuestion: {question}" }),
  systemPromptForNonRagOption: async () => "",
  getDefaultApiProvider: async () => "openai"
}))
vi.mock("@/services/model-settings", async () => ({
  ...await vi.importActual<typeof import("@/services/model-settings")>("@/services/model-settings"),
  getAllDefaultModelSettings: async () => ({}),
  getModelSettings: async () => ({})
}))
vi.mock("@/utils/resolve-api-provider", async () => ({
  ...await vi.importActual<typeof import("@/utils/resolve-api-provider")>("@/utils/resolve-api-provider"),
  resolveApiProviderForModel: async () => "openai"
}))
vi.mock("@/utils/ocr", () => ({ processImageForOCR: mocks.ocr }))
// Existing transport-independent cases retain their simple fixture; image-input
// controls use the real formatter and model factory, with only OCR disabled.
vi.mock("@/utils/human-message", async () => {
  const { HumanMessage } = await vi.importActual<typeof import("@/types/messages")>("@/types/messages")
  const actual = await vi.importActual<typeof import("@/utils/human-message")>("@/utils/human-message")
  return { humanMessageFormatter: async (options: Parameters<typeof actual.humanMessageFormatter>[0]) =>
    mocks.realFormatter ? actual.humanMessageFormatter(options) : Array.isArray(options.content) && options.content.some(part => part.type === "image_url")
      ? new HumanMessage({ content: options.content })
      : { role: "user", content: Array.isArray(options.content) ? options.content[0]?.type === "text" ? options.content[0].text : "" : options.content }
  }
})
vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: async (history: unknown[]) => history
}))
vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: mocks.getActorSettings
}))
vi.mock("@/db/dexie/schema", () => ({ db: {
  chatHistories: {
    get: async (id: string) => mocks.mirrorHistories.get(id),
    update: async (id: string, patch: Record<string, unknown>) => {
      const row = mocks.mirrorHistories.get(id)
      if (row) mocks.mirrorHistories.set(id, { ...row, ...patch })
    }
  },
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

vi.mock("@/db/dexie/helpers", async () => {
  const actual = await vi.importActual<typeof import("@/db/dexie/helpers")>("@/db/dexie/helpers")
  return {
  ...actual,
  acknowledgeSavedUserMessage: async (historyId: string, id: string, serverMessageId: string) => {
    const row = mocks.rows.find(row => row.history_id === historyId && row.id === id && row.role === "user")
    if (row) {
      if (row.serverMessageId && row.serverMessageId !== serverMessageId) throw new Error("The saved user message changed. Reload the conversation before retrying.")
      row.serverMessageId = serverMessageId
    }
  },
  generateID: () => crypto.randomUUID(),
  saveHistory: mocks.saveHistory,
  saveMessage: (...args: Parameters<typeof actual.saveMessage>) => mocks.realPersistence ? actual.saveMessage(...args) : mocks.saveMessage(...args),
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(),
  removeMessageByIndex: vi.fn(),
  removeMessageById: mocks.removeMessageById,
  formatToChatHistory: (items: Parameters<typeof actual.formatToChatHistory>[0]) => mocks.realPersistence ? actual.formatToChatHistory(items) : items,
  formatToMessage: (items: Array<Record<string, unknown>>) => mocks.realPersistence ? actual.formatToMessage(items as Parameters<typeof actual.formatToMessage>[0]) : mocks.withLoader ? items.map(row => ({
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
} })
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
  getServerCapabilities: async () => {
    if (mocks.capability === "error") throw new Error("Capabilities unavailable")
    return { hasChatSaveToDb: mocks.capability }
  }
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

const renderWorkspace = (realNotifications = false, withLoader = false, options: Partial<ReturnType<typeof createHookOptions>> = {}) =>
  renderHook(
    ({ ready }) => {
      const notificationApi = React.useContext(NotificationContext)
      const state = useStoreMessageOption()
      const actions = useChatActions({
        ...createHookOptions(),
        ...state,
        ...options,
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
  it.each([
    { name: "both ACKs / capability false", ack: "both", capability: false, expectedFallback: [] },
    { name: "both ACKs / capability error", ack: "both", capability: "error", expectedFallback: [] },
    { name: "both ACKs / capability true control", ack: "both", capability: true, expectedFallback: [] },
    { name: "user ACK only / persist genuinely unsaved assistant", ack: "user", capability: false, expectedFallback: ["assistant"] },
    { name: "assistant ACK only / persist genuinely unsaved user", ack: "assistant", capability: false, expectedFallback: ["user"] },
    { name: "no ACK / ordinary unpersisted text fallback", ack: "none", capability: false, expectedFallback: ["user", "assistant"] },
    { name: "no ACK / ordinary fallback on capability error", ack: "none", capability: "error", expectedFallback: ["user", "assistant"] }
  ])("UAT157 $name", async ({ ack, capability, expectedFallback }) => {
    mocks.capability = capability as boolean | "error"
    mocks.realFormatter = true
    mocks.realPersistence = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const image = ack === "none" ? "" : "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    const actualModels = await vi.importActual<typeof import("@/models")>("@/models")
    mocks.pageAssistModel.mockImplementation(actualModels.pageAssistModel)
    mocks.getModel.mockResolvedValue({ id: "vision-test", name: "Vision model", capabilities: ["vision"] })
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      expect(options.saveToDb).toBe(true)
      expect(options.clientMessageId).toBeTruthy()
      const stored = mocks.serverRows.get(options.conversationId)!
      const chunk: Record<string, unknown> = { tldw_conversation_id: options.conversationId }
      if (ack === "both" || ack === "user") {
        stored.push({ id: "canonical-user", role: "user", content: "Image question", images: image ? [image] : [], metadata_extra: { client_message_id: options.clientMessageId } })
        chunk.tldw_user_message_id = "canonical-user"
      }
      if (ack === "both" || ack === "assistant") {
        stored.push({ id: "canonical-answer", role: "assistant", content: "A red square." })
        chunk.tldw_message_id = "canonical-answer"
      }
      onChunk(chunk)
      yield "A red square."
    })
    const view = renderWorkspace()
    act(() => useStoreMessageOption.setState({ selectedModel: "vision-test" }))
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Image question", image }) })
    // These are current stream ACKs; no server loader runs in this fixture.
    if (ack === "both" || ack === "user") {
      expect(mocks.rows.find(row => row.role === "user")?.serverMessageId).toBe("canonical-user")
    }
    if (ack === "both" || ack === "assistant") {
      expect(mocks.rows.find(row => row.role === "assistant")?.serverMessageId).toBe("canonical-answer")
    }
    const roles = mocks.addChatMessage.mock.calls.map(call => call[1].role)
    view.unmount()
    expect(roles).toEqual(expectedFallback)
    expect([...mocks.serverRows.values()].flat().map(row => row.role).sort()).toEqual(["assistant", "user"])
  })

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

  it.each(["immediate", "delayed ACK", "ambiguous ACK"])("UAT131 promotes exact greeting identity through loader, send and reload: %s", async boundary => {
    mocks.withLoader = true
    const greeting = { id: "original-greeting", role: "assistant" as const, isBot: true, name: "Default Assistant", message: "Hello! How can I help you today?", messageType: "character:greeting", images: [], sources: [], createdAt: 1 }
    useStoreMessageOption.setState({ historyId: null, messages: [greeting], history: [{ role: "system", content: "Speak like a pirate. Say ARRR." }, { role: "assistant", content: greeting.message, messageType: greeting.messageType }] })
    const ack = deferred<void>()
    const originalAdd = mocks.addChatMessage.getMockImplementation()!
    mocks.addChatMessage.mockImplementationOnce(async (...args) => {
      const row = await originalAdd(...args)
      if (boundary !== "immediate") await ack.promise
      if (boundary === "ambiguous ACK") throw new Error("Acknowledgement lost after commit")
      return row
    })
    let view = renderWorkspace(false, true)
    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalled())
    if (boundary !== "immediate") {
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 300)) })
      expect(mocks.listChatMessages).not.toHaveBeenCalled()
      expect(view.result.current.state.messages.map(row => row.id)).toEqual([greeting.id])
      await act(async () => ack.resolve())
    }
    if (boundary === "ambiguous ACK") {
      await waitFor(() => expect(mocks.notifyError).toHaveBeenCalled())
      await act(async () => view.result.current.persistence.handleSaveChatToServer())
    }
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    const savedSystem = mocks.serverRows.get("chat-1")![0]
    const savedGreeting = mocks.serverRows.get("chat-1")![1]
    expect(view.result.current.state.messages.filter(row => row.role === "assistant")).toMatchObject([{ id: greeting.id, serverMessageId: savedGreeting.id, message: greeting.message }])
    expect(view.result.current.state.messages).toHaveLength(2)
    expect(savedSystem.content).toBe("Speak like a pirate. Say ARRR.")
    expect(mocks.rows.filter(row => row.serverMessageId === savedGreeting.id)).toMatchObject([{ id: greeting.id, content: greeting.message }])
    const sent: Array<Array<{ content: string }>> = []
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => new ChatTldw({ model: "test", saveToDb: true, conversationId }))
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      sent.push(messages)
      const stored = mocks.serverRows.get(options.conversationId)!
      stored.push({ id: "next-user", role: "user", content: "Weather?", version: 1 }, { id: "next-reply", role: "assistant", content: "ARRR clear skies", version: 1 })
      onChunk({ tldw_user_message_id: "next-user", tldw_message_id: "next-reply" })
      yield "ARRR clear skies"
    })
    await act(async () => view.result.current.actions.onSubmit({ message: "Weather?", image: "" }))
    expect(sent[0].filter(row => (typeof row.content === "string" ? row.content : JSON.stringify(row.content)).includes(greeting.message))).toHaveLength(1)
    expect(mocks.serverRows.get("chat-1")!.map(row => row.id)).toEqual([savedSystem.id, savedGreeting.id, "next-user", "next-reply"])
    view.unmount()
    act(() => useStoreMessageOption.setState({ messages: [], history: [], serverChatMetaLoaded: false, serverChatLoadState: "idle" }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === savedGreeting.id)).toMatchObject([{ id: greeting.id }])
    expect(view.result.current.state.messages.map(row => row.serverMessageId)).toEqual([savedSystem.id, savedGreeting.id, "next-user", "next-reply"])
    expect(mocks.addChatMessage).toHaveBeenCalledTimes(2)
    view.unmount()
  })

  it.each(["late edit", "equal rows", "authority", "new history", "route roundtrip"])("UAT131 preserves captured source identity across %s", async boundary => {
    mocks.withLoader = true
    const original = { id: "source-one", role: "assistant" as const, isBot: true, name: "Assistant", message: "Same greeting", images: [], sources: [] }
    const rows = boundary === "equal rows" ? [original, { ...original, id: "source-two" }] : [original]
    useStoreMessageOption.setState({ historyId: "local-draft", messages: rows, history: rows.map(row => ({ role: row.role, content: row.message })) })
    mocks.mirrorHistories.set("local-draft", { id: "local-draft" })
    mocks.rows.push(...rows.map(row => ({ id: row.id, role: row.role, content: row.message, history_id: "local-draft", images: [] })))
    const ack = deferred<void>()
    const originalAdd = mocks.addChatMessage.getMockImplementation()!
    mocks.addChatMessage.mockImplementationOnce(async (...args) => { const saved = await originalAdd(...args); await ack.promise; return saved })
    const view = renderWorkspace(false, true)
    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalledTimes(1))
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 300)) })
    expect(mocks.listChatMessages).not.toHaveBeenCalled()
    await act(async () => {
      if (boundary === "late edit") {
        useStoreMessageOption.getState().setMessages(current => current.map(row => ({ ...row, message: "New local edit" })))
        mocks.rows[0].content = "New local edit"
      } else if (boundary === "authority" || boundary === "new history" || boundary === "route roundtrip") {
        if (boundary === "authority") { replaceAuthority("synthetic-b"); replaceAuthority("synthetic-a") }
        if (boundary === "route roundtrip") {
          usePlaygroundSessionStore.getState().cancelPendingRestore()
          usePlaygroundSessionStore.getState().cancelPendingRestore()
          useStoreMessageOption.setState({ serverChatId: null, messages: [], history: [{ role: "assistant", content: "Replacement context" }] })
        } else useStoreMessageOption.setState({ historyId: "replacement", serverChatId: null, messages: [], history: [] })
      }
      ack.resolve()
    })
    if (boundary === "authority" || boundary === "new history" || boundary === "route roundtrip") {
      expect(mocks.rows[0].serverMessageId).toBeUndefined()
      expect(view.result.current.state.messages).toEqual([])
    } else {
      await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
      const canonical = mocks.serverRows.get("chat-1")!
      expect(view.result.current.state.messages.map(row => [row.id, row.serverMessageId])).toEqual(rows.map((row, index) => [row.id, canonical[index].id]))
      expect(mocks.rows.map(row => [row.id, row.serverMessageId])).toEqual(rows.map((row, index) => [row.id, canonical[index].id]))
      if (boundary === "late edit") {
        expect(view.result.current.state.messages[0].message).toBe("New local edit")
        expect(mocks.rows[0].content).toBe("New local edit")
        expect(canonical[0].content).toBe("Same greeting")
      }
    }
    view.unmount()
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
    mocks.capability = true
    vi.clearAllMocks()
    mocks.rows.length = 0
    mocks.mirrorHistories.clear()
    mocks.withLoader = false
    mocks.realFormatter = false
    mocks.realPersistence = false
    mocks.ragSearch.mockReset()
    mocks.getModel.mockResolvedValue({ id: "test-model", name: "Test model", capabilities: [] })
    mocks.ocr.mockResolvedValue("Explicit OCR text")
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

  it("UAT103 deleting a local diagnostic does not remove equal genuine prompt text", async () => {
    const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    const messages = [
      { id: "diagnostic-user", isBot: false, name: "You", message: "Repeated question", sources: [], generationInfo },
      { id: "diagnostic-answer", isBot: true, name: "Assistant", message: "Repeated answer", sources: [], generationInfo, parentMessageId: "diagnostic-user" },
      { id: "real-user", isBot: false, name: "You", message: "Repeated question", sources: [] },
      { id: "real-answer", isBot: true, name: "Assistant", message: "Repeated answer", sources: [], parentMessageId: "real-user" }
    ]
    useStoreMessageOption.setState({ temporaryChat: true, messages,
      history: [{ role: "user", content: "Repeated question" }, { role: "assistant", content: "Repeated answer" }] })
    const view = renderWorkspace()
    await act(async () => { await view.result.current.actions.deleteMessage(0) })
    expect(view.result.current.state.history).toEqual([{ role: "user", content: "Repeated question" }, { role: "assistant", content: "Repeated answer" }])
    expect(view.result.current.state.messages.map(row => row.id)).toEqual(["diagnostic-answer", "real-user", "real-answer"])
    view.unmount()
  })

  it("UAT103 diagnostic deletion preserves a canonical system row's prompt role", async () => {
    const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    useStoreMessageOption.setState({ temporaryChat: true })
    const view = renderWorkspace()
    act(() => useStoreMessageOption.setState({ messages: [
      { id: "system", serverMessageId: "server-system", role: "system", isBot: false, name: "System", message: "Speak like a pirate.", sources: [] },
      { id: "diagnostic-user", role: "user", isBot: false, name: "You", message: "Local question", sources: [], generationInfo },
      { id: "diagnostic-answer", role: "assistant", isBot: true, name: "Assistant", message: "Local diagnostic", sources: [], generationInfo, parentMessageId: "diagnostic-user" }
    ], history: [{ role: "system", content: "Speak like a pirate." }] }))
    await act(async () => { await view.result.current.actions.deleteMessage(1) })
    expect(view.result.current.state.messages.find(row => row.id === "system")).toMatchObject({ serverMessageId: "server-system", role: "system" })
    expect(view.result.current.state.history).toEqual([{ role: "system", content: "Speak like a pirate." }])
    view.unmount()
  })

  it("UAT103 deleting only the diagnostic assistant keeps its exact draft eligible immediately and after reload", async () => {
    mocks.realPersistence = true
    const { formatToMessage, formatToChatHistory } = await import("@/db/dexie/helpers")
    const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    const rows = [
      { id: "diagnostic-user", role: "user", content: "Repeated question", history_id: "local", images: [], generationInfo, createdAt: 1 },
      { id: "diagnostic-answer", role: "assistant", content: "Repeated answer", history_id: "local", images: [], generationInfo, parent_message_id: "diagnostic-user", createdAt: 2 },
      { id: "real-user", role: "user", content: "Repeated question", history_id: "local", images: [], createdAt: 3 },
      { id: "real-answer", role: "assistant", content: "Repeated answer", history_id: "local", images: [], parent_message_id: "real-user", createdAt: 4 }
    ]
    mocks.rows.push(...rows)
    mocks.removeMessageById.mockImplementationOnce(async (historyId, id) => {
      mocks.rows = mocks.rows.filter(row => row.history_id !== historyId || row.id !== id)
    })
    useStoreMessageOption.setState({ temporaryChat: true, historyId: "local", messages: formatToMessage(rows), history: formatToChatHistory(rows) })
    const view = renderWorkspace()
    await act(async () => { await view.result.current.actions.deleteMessage(1) })
    const reloadMessages = formatToMessage(mocks.rows)
    const reloadHistory = formatToChatHistory(mocks.rows)
    expect(mocks.removeMessageById).toHaveBeenCalledWith("local", "diagnostic-answer")
    expect(view.result.current.state.messages.map(row => row.id)).toEqual(["diagnostic-user", "real-user", "real-answer"])
    expect(reloadMessages.map(row => row.id)).toEqual(["diagnostic-user", "real-user", "real-answer"])
    const promptRows = (history: typeof reloadHistory) => history.map(({ role, content }) => ({ role, content }))
    expect(promptRows(view.result.current.state.history)).toEqual(promptRows(reloadHistory))
    expect(reloadHistory.map(row => row.content)).toEqual(["Repeated question", "Repeated question", "Repeated answer"])
    view.unmount()
  })

  it("UAT103 excludes the local diagnostic from a later global RAG question rewrite", async () => {
    useStoreMessageOption.setState({ temporaryChat: true,
      messages: [
        { id: "prior-user", isBot: false, name: "You", message: "Earlier question", sources: [] },
        { id: "prior-answer", isBot: true, name: "Assistant", message: "Earlier answer", sources: [] }
      ], history: [{ role: "user", content: "Earlier question" }, { role: "assistant", content: "Earlier answer" }] })
    mocks.ragSearch.mockResolvedValue({ documents: [] })
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Local failed source question", image: "" }) })
    const invoke = vi.fn(async () => ({ content: "Rewritten real question" }))
    mocks.pageAssistModel.mockImplementation(async () => ({ invoke, saveToDb: false, stream: async function* () { yield "Real global answer" } }))
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Real evidence", metadata: {} }] })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Follow-up question", image: "",
      requestOverrides: { ragMediaIds: [], selectedKnowledge: { id: "knowledge", title: "All sources" } } }) })
    expect(invoke).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(invoke.mock.calls[0])).toContain("Earlier answer")
    expect(JSON.stringify(invoke.mock.calls[0])).not.toContain("Local failed source question")
    expect(JSON.stringify(invoke.mock.calls[0])).not.toContain("did not send this as general chat")
    expect(view.result.current.state.messages.at(-1)?.message).toBe("Real global answer")
    view.unmount()
  })

  it("UAT103 Continue retries an exact pending source diagnostic instead of extending its prose", async () => {
    useStoreMessageOption.setState({ temporaryChat: true,
      messages: [{ id: "prior-user", isBot: false, name: "You", message: "Earlier question", sources: [] }, { id: "prior-answer", isBot: true, name: "Assistant", message: "Earlier answer", sources: [] }],
      history: [{ role: "user", content: "Earlier question" }, { role: "assistant", content: "Earlier answer" }] })
    mocks.ragSearch.mockResolvedValue({ documents: [] })
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Retry source question", image: "" }) })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "", image: "", isContinue: true }) })
    expect(mocks.ragSearch.mock.calls.map(call => call[0])).toEqual(["Retry source question", "Retry source question"])
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(view.result.current.state.history.map(row => row.content)).toEqual(["Earlier question", "Earlier answer"])
    expect(view.result.current.state.messages.filter(row => !row.isBot && row.message === "Retry source question")).toHaveLength(1)
    view.unmount()
  })

  it.each(["validation skipped", "provider failed"])("UAT103 Continue reports the exact diagnostic Retry outcome: %s", async outcome => {
    const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    const messages = [
      { id: "diagnostic-user", isBot: false, name: "You", message: "Source question", sources: [], generationInfo },
      { id: "diagnostic-answer", isBot: true, name: "Assistant", message: "Diagnostic", sources: [], generationInfo, parentMessageId: "diagnostic-user" }
    ]
    useStoreMessageOption.setState({ temporaryChat: true, messages, history: [],
      ...(outcome === "validation skipped" ? { selectedModel: "" } : {}) })
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Actual evidence", metadata: {} }] })
    mocks.pageAssistModel.mockRejectedValue(new Error("Provider unavailable"))
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true,
      ...(outcome === "validation skipped" ? { selectedModel: "" } : {}) })
    let result: Awaited<ReturnType<typeof view.result.current.actions.onSubmit>> | undefined
    await act(async () => { result = await view.result.current.actions.onSubmit({ message: "", image: "", isContinue: true }) })
    expect(result).toMatchObject({ status: outcome === "validation skipped" ? "skipped" : "failed" })
    if (outcome === "validation skipped") {
      expect(mocks.ragSearch).not.toHaveBeenCalled()
      expect(view.result.current.state.messages).toEqual(messages)
    } else {
      expect(mocks.pageAssistModel).toHaveBeenCalledTimes(1)
    }
    view.unmount()
  })

  it.each(["Complete answer", "Partial answer"])("UAT103 preserves ordinary Continue of %s", async answer => {
    useStoreMessageOption.setState({ temporaryChat: true,
      messages: [{ id: "user", isBot: false, name: "You", message: "Question", sources: [] }, { id: "answer", isBot: true, name: "Assistant", message: answer, sources: [], generationInfo: { interrupted: answer.startsWith("Partial") } }],
      history: [{ role: "user", content: "Question" }, { role: "assistant", content: answer }] })
    mocks.pageAssistModel.mockResolvedValue({ saveToDb: false, stream: async function* () { yield " continued" } })
    const view = renderWorkspace()
    await act(async () => { await view.result.current.actions.onSubmit({ message: "", image: "", isContinue: true }) })
    expect(mocks.ragSearch).not.toHaveBeenCalled()
    expect(view.result.current.state.messages.at(-1)?.message).toBe(answer + " continued")
    expect(view.result.current.state.history.at(-1)?.content).toBe(answer + " continued")
    view.unmount()
  })

  it.each(["retrieval failure", "no evidence"])("UAT103 keeps a %s local through actual action, reload and next completion", async failure => {
    mocks.realPersistence = true
    mocks.withLoader = true
    const { formatToMessage, formatToChatHistory } = await import("@/db/dexie/helpers")
    const initial = [
      { id: "earlier-user", serverMessageId: "earlier-user", role: "user", content: "Earlier real question", history_id: "local-1", name: "You", images: [], createdAt: 1 },
      { id: "earlier-answer", serverMessageId: "earlier-answer", role: "assistant", content: "Earlier real answer", history_id: "local-1", name: "Assistant", images: [], createdAt: 2 }
    ]
    mocks.rows.push(...initial)
    mocks.serverRows.set("source-chat", initial)
    useStoreMessageOption.setState({ historyId: "local-1", serverChatId: "source-chat", serverChatMetaLoaded: true,
      messages: formatToMessage(initial), history: formatToChatHistory(initial) })
    mocks.ragSearch.mockImplementation(async () => {
      if (failure === "retrieval failure") throw new Error("Retrieval unavailable")
      return { documents: [] }
    })
    let view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Local failed source question", image: "" }) })
    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(mocks.addChatMessage).not.toHaveBeenCalled()
    const diagnosticIds = view.result.current.state.messages.slice(2).map(row => row.id)
    expect(diagnosticIds).toHaveLength(2)
    expect(mocks.rows.slice(2)).toMatchObject([
      { role: "user", generationInfo: { mode: "rag", grounded: false } },
      { role: "assistant", generationInfo: { mode: "rag", grounded: false } }
    ])
    view.unmount()
    act(() => useStoreMessageOption.setState({ messages: [], history: [], serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => diagnosticIds.includes(row.id))).toHaveLength(2)
    const projected: unknown[][] = []
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Rowan opens east on Friday.", metadata: { title: "Rowan" } }] })
    mocks.pageAssistModel.mockImplementation(async ({ conversationId, clientMessageId, retryFailedTurn }) =>
      new ChatTldw({ model: "test", saveToDb: true, conversationId: conversationId ?? useStoreMessageOption.getState().serverChatId, clientMessageId, retryFailedTurn }))
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      projected.push(messages)
      mocks.serverRows.get(options.conversationId)!.push(
        { id: "source-user", role: "user", content: "Real source question" },
        { id: "source-answer", role: "assistant", content: "Rowan opens east on Friday." })
      onChunk({ tldw_user_message_id: "source-user", tldw_message_id: "source-answer" })
      yield "Rowan opens east on Friday."
    })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Real source question", image: "" }) })
    expect(view.result.current.state.messages.at(-1)?.message).toBe("Rowan opens east on Friday.")
    expect(mocks.ragSearch).toHaveBeenCalledTimes(2)
    expect(projected).toHaveLength(1)
    expect(JSON.stringify(projected[0])).toContain("Earlier real answer")
    expect(JSON.stringify(projected[0])).not.toContain("Local failed source question")
    expect(JSON.stringify(projected[0])).not.toContain("did not send this as general chat")
    view.unmount()
    act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages).toHaveLength(6)
    expect(mocks.serverRows.get("source-chat")).toHaveLength(4)
    expect(view.result.current.state.messages.filter(row => diagnosticIds.includes(row.id))).toHaveLength(2)
    view.unmount()
  })

  it.each([false, true])("UAT103 retries the exact local source question and restores real context (temporary %s)", async temporary => {
    mocks.realPersistence = true
    mocks.serverRows.set("source-chat", [])
    useStoreMessageOption.setState({ temporaryChat: temporary, historyId: temporary ? "temp" : "local-1",
      serverChatId: temporary ? null : "source-chat", serverChatMetaLoaded: true,
      history: [{ role: "user", content: "Earlier question" }, { role: "assistant", content: "Earlier answer" }],
      messages: [
        { id: "older-user", isBot: false, name: "You", message: "Earlier question", sources: [] },
        { id: "older-answer", isBot: true, name: "Assistant", message: "Earlier answer", sources: [] }
      ] })
    mocks.ragSearch.mockResolvedValue({ documents: [] })
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Retry this exact question", image: "" }) })
    const userId = view.result.current.state.messages.find(row => row.message === "Retry this exact question")!.id
    for (let attempt = 0; attempt < 2; attempt++) {
      await act(async () => { await view.result.current.actions.regenerateLastMessage() })
      expect(mocks.ragSearch.mock.calls.at(-1)?.[0]).toBe("Retry this exact question")
      expect(view.result.current.state.history.map(row => row.content)).toEqual(["Earlier question", "Earlier answer"])
    }
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Evidence", metadata: {} }] })
    const projected: unknown[][] = []
    mocks.pageAssistModel.mockImplementation(async options => ({
      saveToDb: !temporary, conversationId: options.conversationId,
      userServerMessageId: temporary ? undefined : "retried-user",
      serverMessageId: temporary ? undefined : "retried-answer",
      stream: async function* (messages: unknown[]) { projected.push(messages); yield "Real answer" }
    }))
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(mocks.pageAssistModel.mock.calls.at(-1)?.[0]).toMatchObject({ clientMessageId: userId, retryFailedTurn: false })
    expect(JSON.stringify(projected[0])).toContain("Earlier answer")
    expect(view.result.current.state.messages.filter(row => row.id === userId)).toHaveLength(1)
    expect(view.result.current.state.history.map(row => row.content)).toEqual(["Earlier question", "Earlier answer", "Retry this exact question", "Real answer"])
    view.unmount()
  })

  it.each([undefined, { mode: "rag", grounded: true, reason: "selected_source_retrieval_failed" }, { mode: "normal", grounded: false, reason: "selected_source_retrieval_failed" }])("UAT103 preserves ordinary handled text fallback: %j", async generationInfo => {
    const { __testing__ } = await import("@/hooks/chat-modes/ragMode")
    const preflight = vi.spyOn(__testing__.ragModeDefinition, "preflight").mockResolvedValue({ handled: true, fullText: "Same plain response", saveToDb: false, generationInfo })
    mocks.serverRows.set("source-chat", [])
    useStoreMessageOption.setState({ historyId: "local-1", serverChatId: "source-chat", serverChatMetaLoaded: true })
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    try {
      for (let turn = 0; turn < 2; turn++) {
        await act(async () => { await view.result.current.actions.onSubmit({ message: "Same plain question", image: "" }) })
      }
      expect(mocks.serverRows.get("source-chat")?.map(row => row.content)).toEqual(["Same plain question", "Same plain response", "Same plain question", "Same plain response"])
      expect(view.result.current.state.history.map(row => row.content)).toEqual(["Same plain question", "Same plain response", "Same plain question", "Same plain response"])
      expect(view.result.current.state.messages).toHaveLength(4)
    } finally {
      preflight.mockRestore()
      view.unmount()
    }
  })

  it("UAT103 promotes genuine temporary history with exact receipts while keeping its local diagnostic", async () => {
    mocks.ragSearch.mockResolvedValue({ documents: [] })
    useStoreMessageOption.setState({ temporaryChat: true })
    const view = renderWorkspace(false, false, { ragMediaIds: [42], fileRetrievalEnabled: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Unanswered source question", image: "" }) })
    const diagnosticIds = view.result.current.state.messages.map(row => row.id)
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Evidence", metadata: {} }] })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Genuine answered question", image: "" }) })
    const realIds = view.result.current.state.messages.filter(row => !diagnosticIds.includes(row.id)).map(row => row.id)
    await act(async () => { useStoreMessageOption.setState({ temporaryChat: false }) })
    await waitFor(() => expect(view.result.current.state.serverChatId).toBe("chat-1"))
    expect(mocks.serverRows.get("chat-1")?.map(row => row.role)).toEqual(["user", "assistant"])
    expect(mocks.serverRows.get("chat-1")?.some(row => row.content.includes("Unanswered source question"))).toBe(false)
    expect(view.result.current.state.messages.filter(row => diagnosticIds.includes(row.id))).toHaveLength(2)
    expect(view.result.current.state.messages.filter(row => realIds.includes(row.id)).every(row => Boolean(row.serverMessageId))).toBe(true)
    view.unmount()
  })

  it.each([false, true].flatMap(prior => ["", "  Keep this image  "].map(text => ({ prior, text }))))("keeps unsupported image work through blocked Retry, actual vision-model selection and canonical remount: $text / prior $prior", async ({ text, prior }) => {
    mocks.withLoader = true
    mocks.realFormatter = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    const actualModels = await vi.importActual<typeof import("@/models")>("@/models")
    mocks.pageAssistModel.mockImplementation(actualModels.pageAssistModel)
    mocks.getModel.mockImplementation(async model => ({ id: model, name: model, capabilities: model === "vision-test" ? ["vision"] : [] }))
    let accepted = 0
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      if (options.model !== "vision-test") throw new Error("Unsupported model reached transport")
      const users = messages.filter((row: { role: string }) => row.role === "user")
      expect(users).toHaveLength(prior && accepted ? 2 : 1)
      expect(users.at(-1)).toEqual({ role: "user", content: [{ type: "text", text }, { type: "image_url", image_url: { url: image } }] })
      // This is a new, never-dispatched user even if an earlier answered user
      // happens to have identical text and image. Backend Retry=true rejects it.
      expect(options.retryFailedTurn).toBe(false)
      const userId = prior && accepted === 0 ? "prior-image-user" : "accepted-image-user"
      const answerId = prior && accepted === 0 ? "prior-image-answer" : "accepted-image-answer"
      accepted++
      mocks.serverRows.get(options.conversationId)!.push(
        { id: userId, role: "user", content: text || "<Image attachment x1>", images: [image], version: 1, metadata_extra: { client_message_id: options.clientMessageId, ...(!text ? { content_placeholder_reason: "image_attachment" } : {}) } },
        { id: answerId, role: "assistant", content: "Image accepted", images: [], version: 1 })
      onChunk({ tldw_user_message_id: userId, tldw_message_id: answerId })
      yield "Image accepted"
    })
    let view = renderWorkspace(false, true)
    if (prior) {
      act(() => useStoreMessageOption.setState({ selectedModel: "vision-test" }))
      await act(async () => { await view.result.current.actions.onSubmit({ message: text, image }) })
      await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
      act(() => useStoreMessageOption.setState({ selectedModel: "unconfirmed-test" }))
    }
    await act(async () => { await view.result.current.actions.onSubmit({ message: text, image }) })
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(mocks.streamMessage).toHaveBeenCalledTimes(prior ? 1 : 0)
    const user = view.result.current.state.messages.findLast(row => !row.isBot)!
    expect(user).toMatchObject({ message: text, images: [image] })
    expect(user.serverMessageId).toBeUndefined()
    expect(mocks.serverRows.get(view.result.current.state.serverChatId!)?.filter(row => row.role === "user")).toHaveLength(prior ? 1 : 0)
    expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)).toMatchObject({ recoveryAction: "open-model-selector", summary: "Image support is not confirmed for this model.", serverRetryRequired: false })
    view.unmount()
    act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(mocks.streamMessage).toHaveBeenCalledTimes(prior ? 1 : 0)
    expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)).toMatchObject({ serverRetryRequired: false })
    expect(view.result.current.state.messages.find(row => row.id === user.id)).toMatchObject({ id: user.id, message: text, images: [image] })
    expect(mocks.rows.filter(row => row.role === "user")).toHaveLength(prior ? 2 : 1)
    act(() => useStoreMessageOption.setState({ selectedModel: "vision-test" }))
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(mocks.streamMessage).toHaveBeenCalledTimes(prior ? 2 : 1)
    expect(mocks.streamMessage.mock.calls.at(-1)![1]).toMatchObject({ retryFailedTurn: false, clientMessageId: user.id })
    expect(view.result.current.state.messages.find(row => row.id === user.id)).toMatchObject({ id: user.id, serverMessageId: "accepted-image-user", images: [image] })
    view.unmount()
    act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "accepted-image-user")).toMatchObject([{ id: user.id, message: text, images: [image], serverMessageId: "accepted-image-user" }])
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "accepted-image-answer")).toMatchObject([{ message: "Image accepted" }])
    view.unmount()
  })

  it.each([false, true])("retains server Retry after an ambiguous transport failure and a later capability refusal: initial refusal %s", async initialRefusal => {
    mocks.realFormatter = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const actualModels = await vi.importActual<typeof import("@/models")>("@/models")
    mocks.pageAssistModel.mockImplementation(actualModels.pageAssistModel)
    mocks.getModel.mockImplementation(async model => ({ id: model, capabilities: model === "vision-test" ? ["vision"] : [] }))
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    const attempts: boolean[] = []
    mocks.streamMessage.mockImplementation(async function* (_messages, options, onChunk) {
      attempts.push(options.retryFailedTurn)
      if (attempts.length === 1) {
        mocks.serverRows.get(options.conversationId)!.push({ id: "ambiguous-image-user", role: "user", content: "Image question", images: [image], metadata_extra: { client_message_id: options.clientMessageId } })
        throw new Error("Provider failed before acknowledgement")
      }
      mocks.serverRows.get(options.conversationId)!.push({ id: "recovered-image-answer", role: "assistant", content: "Recovered" })
      onChunk({ tldw_user_message_id: "ambiguous-image-user", tldw_message_id: "recovered-image-answer" })
      yield "Recovered"
    })
    act(() => useStoreMessageOption.setState({ selectedModel: initialRefusal ? "unconfirmed" : "vision-test" }))
    const view = renderWorkspace()
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Image question", image }) })
    if (initialRefusal) {
      expect(attempts).toEqual([])
      expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)?.serverRetryRequired).toBe(false)
      act(() => useStoreMessageOption.setState({ selectedModel: "vision-test" }))
      await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    }
    expect(attempts).toEqual([false])
    const user = view.result.current.state.messages.find(row => !row.isBot)!
    expect(user.serverMessageId).toBeUndefined()
    expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)?.serverRetryRequired).toBeUndefined()
    act(() => useStoreMessageOption.setState({ selectedModel: "unconfirmed" }))
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(attempts).toEqual([false])
    expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)?.serverRetryRequired).toBe(true)
    act(() => useStoreMessageOption.setState({ selectedModel: "vision-test" }))
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(attempts).toEqual([false, true])
    expect(mocks.serverRows.get(view.result.current.state.serverChatId!)?.filter(row => row.role === "user")).toHaveLength(1)
    expect(view.result.current.state.messages.filter(row => !row.isBot)).toMatchObject([{ id: user.id, images: [image], serverMessageId: "ambiguous-image-user" }])
    expect(mocks.rows.filter(row => row.role === "user")).toHaveLength(1)
    view.unmount()
  })

  it("allows explicit current-image OCR but refuses unconverted prior images on the next text-only turn", async () => {
    mocks.realFormatter = true
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const actualModels = await vi.importActual<typeof import("@/models")>("@/models")
    mocks.pageAssistModel.mockImplementation(actualModels.pageAssistModel)
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    mocks.streamMessage.mockImplementation(async function* (_messages, _options, onChunk) {
      onChunk({ tldw_user_message_id: "ocr-user", tldw_message_id: "ocr-answer" })
      yield "OCR answer"
    })
    const view = renderWorkspace(false, false, { useOCR: true })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Read receipt", image }) })
    expect(mocks.streamMessage).toHaveBeenCalledTimes(1)
    expect(mocks.streamMessage.mock.calls[0][0].at(-1)?.content).toBe("Read receipt\n\n[IMAGE OCR TEXT]\nExplicit OCR text")
    expect(view.result.current.state.history[0]).toMatchObject({ content: "Read receipt", image })
    await act(async () => { await view.result.current.actions.onSubmit({ message: "Follow-up", image: "" }) })
    expect(mocks.streamMessage).toHaveBeenCalledTimes(1)
    expect(mocks.ocr).toHaveBeenCalledTimes(1)
    expect(view.result.current.state.messages.find(row => row.serverMessageId === "ocr-user")?.images).toEqual([image])
    expect(decodeChatErrorPayload(view.result.current.state.messages.at(-1)!.message)).toMatchObject({
      summary: "Image support is not confirmed for this model.",
      hint: "Choose a model that supports images, or start a new text-only conversation."
    })
    view.unmount()
  })

  it.each(["B", "A"])("does not publish delayed image capability errors after A to B to %s", async destination => {
    mocks.realFormatter = true
    const modelInfo = deferred<{ capabilities: string[] }>()
    const actualModels = await vi.importActual<typeof import("@/models")>("@/models")
    mocks.pageAssistModel.mockImplementation(actualModels.pageAssistModel)
    mocks.getModel.mockReturnValue(modelInfo.promise)
    const view = renderWorkspace()
    let pending!: Promise<unknown>
    act(() => { pending = view.result.current.actions.onSubmit({ message: "Alice image", image: "data:image/png;base64,aW1hZ2U=" }) })
    await waitFor(() => expect(mocks.getModel).toHaveBeenCalled())
    const draft = { id: "new-owner-draft", isBot: false, name: "You", message: "New owner work", sources: [] }
    await act(async () => {
      replaceAuthority("synthetic-b")
      if (destination === "A") replaceAuthority("synthetic-a")
      useStoreMessageOption.setState({ messages: [draft], history: [], historyId: null, serverChatId: null })
      modelInfo.resolve({ capabilities: [] })
      await pending
    })
    expect(mocks.streamMessage).not.toHaveBeenCalled()
    expect(view.result.current.state.messages).toEqual([draft])
    expect(mocks.rows).toEqual([])
    view.unmount()
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

  it("rejects a multi-attachment user before changing the active draft or local mirror", async () => {
    mocks.withLoader = true
    seedLocalDraft()
    const original = useStoreMessageOption.getState()
    const messages = original.messages
    const history = original.history
    mocks.serverRows.set("chat-multi-image", [{ id: "multi-image-user", role: "user", content: "Two images", images: ["data:image/png;base64,aW1hZ2U=", "data:image/png;base64,b3RoZXI="] }])
    useStoreMessageOption.setState({ serverChatId: "chat-multi-image", serverChatMetaLoaded: false })
    const view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("failed"))
    expect(view.result.current.state.serverChatLoadError).toMatch(/multiple.*image|one image/i)
    expect(view.result.current.state.messages).toEqual(messages)
    expect(view.result.current.state.history).toEqual(history)
    expect(mocks.ensureHistory).not.toHaveBeenCalled()
    expect(mocks.saveMessage).not.toHaveBeenCalled()
    expect(mocks.rows).toEqual([])
    view.unmount()
  })

  it.each([false, true].flatMap(prior => ["Image question", ""].map(text => ({ prior, text }))))("recovers actual image transport through the domain adapter, mounted mirror, Retry and remount: $text / prior $prior", async ({ text, prior }) => {
    mocks.withLoader = true
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    vi.spyOn(i18n, "t").mockImplementation((key, fallback) => typeof fallback === "string" ? fallback : String(key))
    const projected: Array<Array<{ role: string; content: unknown }>> = []
    mocks.listChatMessages.mockImplementation((id, params, options) => chatRagMethods.listChatMessages.call(
      { getChatMessagesCacheKey: (key: string, query: string) => key + query } as never, id, params, options))
    mocks.bgRequest.mockImplementation(async request => {
      const url = new URL(request.path, "https://chat.test")
      expect(url.searchParams.get("include_images")).toBe("true")
      expect(request.servicePromptConfig).toMatchObject({ serverUrl: "https://chat.test", authMode: "single-user" })
      const rows = mocks.serverRows.get(url.pathname.split("/")[4]) || []
      const offset = Number(url.searchParams.get("offset"))
      const limit = Number(url.searchParams.get("limit"))
      return { messages: rows.slice(offset, offset + limit).map(row => ({ ...row, sender: row.role, timestamp: "2026-09-16T00:00:00Z" })) }
    })
    mocks.pageAssistModel.mockImplementation(async ({ conversationId, clientMessageId, retryFailedTurn }) =>
      new ChatTldw({ model: "test", supportsMultimodal: true, saveToDb: true, conversationId, clientMessageId, retryFailedTurn }))
    mocks.streamMessage.mockImplementation(async function* (messages, options, onChunk) {
      projected.push(messages)
      if (prior && projected.length === 1) {
        mocks.serverRows.get(options.conversationId)!.push(
          { id: "prior-image-user", role: "user", content: "Prior image", images: [image], version: 1, metadata_extra: { client_message_id: options.clientMessageId } },
          { id: "prior-image-answer", role: "assistant", content: "Prior answer", images: [], version: 1 })
        onChunk({ tldw_user_message_id: "prior-image-user", tldw_message_id: "prior-image-answer" })
        yield "Prior answer"
        return
      }
      if (projected.length === (prior ? 2 : 1)) {
        mocks.serverRows.get(options.conversationId)!.push({ id: "image-user", role: "user", content: text || "<Image attachment x1>", images: [image], version: 1,
          metadata_extra: { client_message_id: options.clientMessageId, ...(!text ? { content_placeholder_reason: "image_attachment" } : {}) } })
        throw new Error("Provider failed before response metadata")
      }
      mocks.serverRows.get(options.conversationId)!.push({ id: "image-answer", role: "assistant", content: "Recovered image answer", images: [], version: 1 })
      onChunk({ tldw_user_message_id: "image-user", tldw_message_id: "image-answer" })
      yield "Recovered image answer"
    })
    let view = renderWorkspace(false, true)
    if (prior) {
      await act(async () => { await view.result.current.actions.onSubmit({ message: "Prior image", image }) })
      await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    }
    await act(async () => { await view.result.current.actions.onSubmit({ message: text, image }) })
    if (prior) {
      // Existing loaded chats fetch new canonical rows on the next normal reload.
      view.unmount()
      act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
      view = renderWorkspace(false, true)
    }
    await waitFor(() => {
      expect(view.result.current.state.serverChatLoadError).toBeNull()
      expect(view.result.current.state.serverChatLoadState).toBe("loaded")
    })
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "image-user")).toMatchObject([{ message: text, images: [image], serverMessageId: "image-user" }])
    expect(mocks.rows.filter(row => row.role === "user")).toHaveLength(prior ? 2 : 1)
    await act(async () => { await view.result.current.actions.regenerateLastMessage() })
    expect(projected).toHaveLength(prior ? 3 : 2)
    const retryUsers = projected.at(-1)!.filter(row => row.role === "user")
    expect(retryUsers).toHaveLength(prior ? 2 : 1)
    for (const user of retryUsers) {
      expect(user.content).toEqual(expect.arrayContaining([{ type: "image_url", image_url: { url: image } }]))
    }
    if (prior) expect(retryUsers[0].content).toEqual([{ type: "image_url", image_url: { url: image } }, { type: "text", text: "Prior image" }])
    const localId = view.result.current.state.messages.find(row => row.serverMessageId === "image-user")?.id
    view.unmount()
    act(() => useStoreMessageOption.setState({ serverChatLoadState: "idle", serverChatMetaLoaded: false }))
    view = renderWorkspace(false, true)
    await waitFor(() => expect(view.result.current.state.serverChatLoadState).toBe("loaded"))
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "image-user")).toMatchObject([{ id: localId, serverMessageId: "image-user", message: text, images: [image] }])
    expect(view.result.current.state.messages.filter(row => row.serverMessageId === "image-answer")).toMatchObject([{ message: "Recovered image answer" }])
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
    expect(view.result.current.state.messages).toMatchObject([draft])
    expect(view.result.current.state.messages).toHaveLength(1)
    const ownReceipt = mocks.serverRows.get(view.result.current.state.serverChatId!)?.find(row => row.content === draft.message)
    expect(view.result.current.state.messages[0].serverMessageId).toBe(ownReceipt?.id)
    expect(view.result.current.state.messages[0].serverMessageId).not.toBe("late-canonical-user")
    expect(mocks.rows.find(row => row.id === original.id)?.serverMessageId).toBeUndefined()
    view.unmount()
  })

  it("successful regeneration does not re-ACK the already acknowledged user", async () => {
    let calls = 0
    mocks.pageAssistModel.mockImplementation(async ({ conversationId }) => ({
      saveToDb: true,
      conversationId,
      userServerMessageId: `server-user-${++calls}`,
      serverMessageId: `server-answer-${calls}`,
      stream: async function* () { yield "Ordinary final answer" }
    }))
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.onSubmit({ message: "Ordinary question", image: "" }) })
    const original = result.current.state.messages.find(row => !row.isBot)?.serverMessageId
    expect(original).toBe("server-user-1")
    await act(async () => { await result.current.actions.regenerateLastMessage() })
    expect(mocks.pageAssistModel.mock.calls.map(([options]) => options.retryFailedTurn)).toEqual([false, false])
    expect(mocks.pageAssistModel.mock.calls.map(([options]) => options.regenerateFromMessageId)).toEqual([undefined, "server-answer-1"])
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

  it("keeps resolved ordinary metadata after saving the first completed turn", async () => {
    const { result } = renderWorkspace()
    await act(async () => { await result.current.actions.onSubmit({ message: "Bob's ordinary question", image: "" }) })
    expect(result.current.state.serverChatId).toBe("chat-1")
    expect(result.current.state.serverChatMetaLoaded).toBe(true)
    expect(result.current.state.serverChatCharacterId).toBeNull()
    expect(result.current.state.serverChatAssistantKind).toBeNull()
    expect(mocks.createChat).toHaveBeenCalledTimes(1)
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
      if (boundary === "A to B to A")
        expect(result.current.state.serverChatId).toBeNull()
      else if (boundary === "new history")
        expect(result.current.state.serverChatId).toBe("other-chat")
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
