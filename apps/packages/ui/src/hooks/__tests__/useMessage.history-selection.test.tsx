import React from "react"
import { protectedEditRequestSchema } from "@/services/__tests__/history-edit-fixture"
import { roleplayDetail, roleplayAppend, roleplayCompletion, captureRoleplayFixtureRows } from "@/hooks/chat/__tests__/roleplay-resume-fixture"
import { createDefaultActorSettings } from "@/types/actor"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { chatDraftAddressForReceipt } from "@/hooks/chat/chat-draft-transition"
import { captureCurrentActorSceneTarget } from "@/hooks/chat/actor-scene-source"
import { normalizeChatSubmitResult as toChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import type { HistoryLoadReceipt, HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { State } from "@/store/option/types"
import type { Message as StoredMessage } from "@/db/dexie/types"
import { getLatestChatErrorBannerEntry } from "@/components/Option/Playground/PlaygroundChatErrorBanner"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const setMessages = vi.fn()
  const setHistory = vi.fn()
  const setAbortController = vi.fn()
  const setEmbeddingController = vi.fn()
  const setIsEmbedding = vi.fn()
  const setIsLoading = vi.fn()
  const setIsProcessing = vi.fn()
  const setStreaming = vi.fn()
  const resetChatLoopState = vi.fn()
  const notification = {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  }
  const saveMessageOnSuccess = vi.fn<
    (payload: any) => Promise<string | undefined>
  >(async () => undefined)
  const saveMessageOnError = vi.fn<(payload: any) => Promise<string | false>>(
    async () => false
  )
  const finalModel = {
    stream: vi.fn<(...args: any[]) => Promise<any>>(async () => []),
    invoke: vi.fn()
  }
  const rewriteModel = {
    stream: vi.fn(),
    invoke: vi.fn(async () => ({ content: "standalone retrieval query" }))
  }

  const storeState: Record<string, unknown> = {
    messages: [
      {
        id: "user-old",
        isBot: false,
        name: "You",
        message: "Earlier question",
        images: [],
        sources: [],
        createdAt: 1,
        parentMessageId: null
      },
      {
        id: "assistant-old",
        isBot: true,
        name: "model-1",
        message: "Earlier answer",
        images: [],
        sources: [],
        createdAt: 2,
        parentMessageId: "user-old"
      }
    ],
    setMessages,
    webSearch: false,
    setWebSearch: vi.fn(),
    toolChoice: "required",
    setToolChoice: vi.fn(),
    isSearchingInternet: false,
    setIsSearchingInternet: vi.fn(),
    temporaryChat: false,
    setTemporaryChat: vi.fn(),
    queuedMessages: [],
    addQueuedMessage: vi.fn(),
    setQueuedMessages: vi.fn(),
    clearQueuedMessages: vi.fn(),
    fileRetrievalEnabled: true,
    setActionInfo: vi.fn(),
    replyTarget: null as import("@/store/option/types").ReplyTarget | null,
    clearReplyTarget: vi.fn(),
    serverChatId: null,
    setServerChatId: vi.fn(),
    serverChatTitle: "",
    setServerChatTitle: vi.fn(),
    setServerChatRoleplaySetup: vi.fn(),
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
    serverChatState: null,
    setServerChatState: vi.fn(),
    setServerChatVersion: vi.fn(),
    serverChatTopic: null,
    setServerChatTopic: vi.fn(),
    serverChatClusterId: null,
    setServerChatClusterId: vi.fn(),
    serverChatSource: null,
    setServerChatSource: vi.fn(),
    serverChatExternalRef: null,
    setServerChatExternalRef: vi.fn()
  }

  const history = [
    { role: "user", content: "Earlier question" },
    { role: "assistant", content: "Earlier answer" }
  ]
  const chatBaseState = {
    history,
    setHistory,
    streaming: false,
    setStreaming,
    isFirstMessage: false,
    setIsFirstMessage: vi.fn(),
    historyId: "history-1" as string | null,
    setHistoryId: vi.fn(),
    isLoading: false,
    setIsLoading,
    isProcessing: false,
    setIsProcessing,
    chatMode: "rag",
    setChatMode: vi.fn(),
    isEmbedding: false,
    setIsEmbedding,
    selectedQuickPrompt: null,
    setSelectedQuickPrompt: vi.fn(),
    selectedSystemPrompt: "Keep this system prompt",
    setSelectedSystemPrompt: vi.fn(),
    useOCR: true,
    setUseOCR: vi.fn()
  }

  const definition = (
    id: "chat.rag.answer" | "chat.rag.question_rewrite",
    requiredVariables: readonly string[]
  ) =>
    Object.freeze({
      id,
      parts: Object.freeze([
        Object.freeze({
          key: "template",
          mode: "template" as const,
          required_variables: Object.freeze([...requiredVariables])
        })
      ])
    })
  const answerDefinition = definition("chat.rag.answer", [
    "context",
    "question"
  ])
  const rewriteDefinition = definition("chat.rag.question_rewrite", [
    "chat_history",
    "question"
  ])
  const makeSnapshot = (
    capability: "supported" | "legacy-404" = "supported",
    answerTemplate = "custom answer {context} :: {question}",
    rewriteTemplate = "custom rewrite {chat_history} :: {question}",
    lifetime?: {
      scopeSignal?: AbortSignal
      scopeInvalidatedSignal?: AbortSignal
      release?: () => void
    }
  ) =>
    Object.freeze({
      scopeKey: "scope:user-1",
      requestScope: Object.freeze({
        config: Object.freeze({
          serverUrl: "https://example.test",
          authMode: "multi-user" as const,
          authSource: "manual" as const,
          orgId: 9
        }),
        userId: 42
      }),
      capability,
      definitions: Object.freeze({
        "chat.rag.answer": Object.freeze({
          definition: answerDefinition,
          parts: Object.freeze({ template: answerTemplate }),
          source: "user" as const,
          revision: capability === "supported" ? "answer-revision" : null
        }),
        "chat.rag.question_rewrite": Object.freeze({
          definition: rewriteDefinition,
          parts: Object.freeze({ template: rewriteTemplate }),
          source: "user" as const,
          revision: capability === "supported" ? "rewrite-revision" : null
        })
      }),
      scopeSignal: lifetime?.scopeSignal ?? new AbortController().signal,
      scopeInvalidatedSignal:
        lifetime?.scopeInvalidatedSignal ?? new AbortController().signal,
      release: lifetime?.release ?? vi.fn()
    })

  return {
    getActorScene: vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>(),
    useRealPageAssist: false,
    controllerReader: null as (() => AbortController | null) | null,
    draftFields: null as (() => Record<string, unknown>) | null,
    addMedia: vi.fn(),
    answerDefinition,
    chatBaseState,
    finalModel,
    history,
    humanMessageFormatter: vi.fn(),
    loadServicePromptSnapshot: vi.fn(),
    makeSnapshot,
    notification,
    pageAssistModel: vi.fn(),
    promptForRag: vi.fn(),
    ragSearch: vi.fn(),
    renderServicePromptPart: vi.fn(),
    resetChatLoopState,
    rewriteDefinition,
    rewriteModel,
    saveMessageOnError,
    saveMessageOnSuccess,
    setAbortController,
    setEmbeddingController,
    setHistory,
    setIsEmbedding,
    setIsLoading,
    setIsProcessing,
    setMessages,
    setStreaming,
    storeState
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("~/services/tldw-server", async (original) => ({
  ...(await original<any>()),
  systemPromptForNonRagOption: async () => "system",
  promptForRag: (...args: unknown[]) => mocks.promptForRag(...args),
  systemPromptForNonRag: vi.fn(async () => "system")
}))

vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => mocks.makeSnapshot("supported").requestScope,
  subscribeToServicePromptConfigChanges: () => () => {},
  loadServicePromptSnapshot: (...args: unknown[]) =>
    mocks.loadServicePromptSnapshot(...args),
  renderServicePromptPart: (...args: unknown[]) =>
    mocks.renderServicePromptPart(...args)
}))

vi.mock("~/store/option", () => ({
  useStoreMessageOption: Object.assign(
    (selector?: (state: Record<string, unknown>) => unknown) =>
      selector ? selector({ ...mocks.chatBaseState, ...mocks.storeState, ...mocks.draftFields?.() }) : { ...mocks.chatBaseState, ...mocks.storeState, ...mocks.draftFields?.() },
    {
      getState: () => ({ ...mocks.chatBaseState, ...mocks.storeState, ...mocks.draftFields?.() }),
      setState: (update: Record<string, unknown> | ((state: typeof mocks.storeState) => Record<string, unknown>)) => {
        Object.assign(mocks.storeState, typeof update === "function" ? update(mocks.storeState) : update)
      },
      subscribe: () => () => {}
    }
  )
}))

vi.mock("~/store", () => ({
  useStoreMessage: () => ({ currentURL: "", setCurrentURL: vi.fn() })
}))

vi.mock("@/context", async original => {
  const actual = await original<typeof import("@/context")>()
  return { ...actual, usePageAssist: () => mocks.useRealPageAssist ? actual.usePageAssist() : ({
    controller: null,
    ...(mocks.controllerReader ? { getController: mocks.controllerReader } : {}),
    setController: mocks.setAbortController,
    embeddingController: null,
    setEmbeddingController: mocks.setEmbeddingController
  }) }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    key === "chatWithWebsiteEmbedding" ? true : fallback,
    vi.fn()
  ]
}))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => ({ ...mocks.chatBaseState, ...mocks.draftFields?.() })
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: mocks.storeState.selectedModel,
    setSelectedModel: vi.fn()
  })
}))

vi.mock("@/store/model", () => {
  const state = {
    apiProvider: "provider-1", reset: vi.fn(),
    activeSettingsScope: "global",
    getEffectiveSettings: () => ({ apiProvider: "provider-1" })
  }
  return {
    useStoreChatModelSettings: Object.assign(() => state, {
      getState: () => state
    })
  }
})

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn()]
}))

vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>(),
  getActorSceneForChat: mocks.getActorScene
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => mocks.notification
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null })
}))

vi.mock("@/hooks/chat/effective-assistant-state", () => ({
  resolveEffectiveAssistantState: () => ({
    mode: h1.persona
      ? "tracked_persona"
      : h1.native
        ? "tracked_character"
        : "plain",
    kind: h1.persona ? "persona" : h1.native ? "character" : null,
    id: h1.native || h1.persona ? "12" : null,
    displayName: null,
    avatarUrl: null,
    systemPromptSnapshot: null
  })
}))

vi.mock("@/services/chat-loop/hooks", () => ({
  useChatLoopState: () => ({
    state: {},
    dispatch: vi.fn(),
    reset: mocks.resetChatLoopState
  })
}))

vi.mock("@/services/chat-loop/bridge", () => ({
  subscribeChatLoopEvents: () => vi.fn(),
  publishChatLoopEvent: vi.fn()
}))

vi.mock("@/utils/chat-model-validation", () => ({
  validateSelectedChatModelAvailability: vi.fn(async () => ({
    status: "valid"
  }))
}))

vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: () => []
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({}))
}))

vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return {
    pageAssistModel: async (options: any) => {
      mocks.pageAssistModel(options)
      await h1.beforeModel()
      return new ChatTldw({ ...options, temperature: 0.23, supportsMultimodal: h1.supportsMultimodal })
    }
  }
})

vi.mock("@/libs/get-html", () => ({
  getContentFromCurrentTab: vi.fn(async () => ({
    content: "raw page content",
    url: "https://source.example/page",
    type: "html",
    pdf: []
  }))
}))

vi.mock("@/services/tldw/server-capabilities", async original => ({
  ...await original<typeof import("@/services/tldw/server-capabilities")>(),
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false, hasRoleplayResumeV1: h1.roleplayEnabled }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    ensureConfigForRequest: vi.fn(async () => h1.cookieConfig),
    getOpenAPISpec: (...args: unknown[]) => h1.spec(...args),
    getCharacter: h1.live,
    createChat: (...args: any[]) => h1.create(...args),
    getChat: vi.fn(async (id: string, options?: { requireRoleplayResume?: boolean }) => options?.requireRoleplayResume ? { ...roleplayDetail(id, h1.roleplayState.get(id)!.tail), history_version: h1.roleplayState.get(id)!.historyVersion } : { id, title: "Owned opening", character_id: "12" }),
    appendRoleplayResumeInput: (...args: unknown[]) => h1.roleplayAppend(...args),
    completeCharacterChatTurn: (...args: unknown[]) => h1.roleplayComplete(...args),
    getChatSettings: vi.fn(async (id: string) => ({ conversation_id: id, settings: {} })),
    captureHistorySelection: async (id: string, request: import("@/types/history-selection").HistoryCaptureRequestV1, options: unknown) => {
      const capture = await h1.capture(id, request, options)
      const state = h1.roleplayState.get(id)
      return state ? captureRoleplayFixtureRows(capture, request, state.rows, state.content) : capture
    },
    addChatMessage: (...args: any[]) => h1.append(...args),
    streamChatCompletion: (...args: any[]) => h1.wire(...args),
    getConfig: async () => null,
    getMessage: h1.read,
    editMessage: h1.write,
    deleteMessage: vi.fn(async () => undefined),
    addMedia: (...args: unknown[]) => mocks.addMedia(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: () => "grounded context"
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) =>
    mocks.humanMessageFormatter(...args)
}))

vi.mock("@/libs/reasoning", () => ({
  MISSING_FINAL_ANSWER_MESSAGE: "Final answer unavailable",
  isReasoningOnlyResponse: () => false,
  isReasoningEnded: () => false,
  isReasoningStarted: () => false,
  mergeReasoningContent: (value: string) => value,
  removeReasoning: (value: string) => value
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => ({
    model_avatar: "avatar.png",
    model_name: "Model One"
  }))
}))

vi.mock("@/db/dexie/helpers", async (original) => {
  const actual = await original<typeof import("@/db/dexie/helpers")>()
  let id = 0
  return {
    ...actual,
    formatSelectedHistory: actual.formatSelectedHistory,
    formatToChatHistory: (items: any[]) => items,
    formatToMessage: (items: any[]) => items,
    saveMessage: vi.fn(actual.saveMessage),
    updateLastUsedModel: vi.fn(async () => {}),
    updateLastUsedPrompt: vi.fn(async () => {}),
    updateChatHistoryCreatedAt: vi.fn(async () => {}),
    deleteChatForEdit: vi.fn(),
    deleteChatAfterMessageId: vi.fn(),
    generateID: () => `generated-${++id}`,
    getPromptById: vi.fn(async () => null),
    removeMessageByIndex: vi.fn(),
    removeMessageById: vi.fn(),
    updateMessageByIndex: vi.fn(),
    updateMessageById: vi.fn()
  }
})

vi.mock("../chat-modes/tabChatMode", () => ({ tabChatMode: vi.fn() }))
vi.mock("../chat-modes/documentChatMode", () => ({ documentChatMode: vi.fn() }))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

import { useMessage } from "../useMessage"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { usePageAssist } from "@/context"
import {
  SidepanelSendGateContext,
  type SidepanelSendGate
} from "@/hooks/chat/sidepanel-send-gate"

const h1 = vi.hoisted(() => ({
  roleplayEnabled: false,
  roleplayState: new Map<string, { tail: string | null; historyVersion: number; rows: import("@/hooks/chat/__tests__/roleplay-resume-fixture").RoleplayFixtureRow[]; content: Map<string, import("@/types/history-selection").HistorySelectionCaptureV1["selected_content"][number]> }>(),
  roleplayAppend: vi.fn(),
  roleplayComplete: vi.fn(),
  spec: vi.fn(),
  cookieConfig: null as { serverUrl: string; authMode: "single-user"; authSource: "cookie-session" } | null,
  native: false,
  persona: false,
  supportsMultimodal: false,
  create: vi.fn(),
  live: vi.fn(() => {
    throw new Error("live card deleted")
  }),
  auth: new AbortController(),
  controller: null as any,
  capture: vi.fn(),
  append: vi.fn(),
  wire: vi.fn(),
  recover: vi.fn(),
  dismiss: vi.fn(),
  beforeModel: vi.fn(),
  read: vi.fn(),
  write: vi.fn()
}))
vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<any>()),
  useHistorySelectionContext: () => h1.controller
}))
vi.mock("@/db/dexie/history-selection", async (original) => ({
  ...(await original<any>()),
  saveHistoryTurnRecovery: h1.recover,
  dismissHistoryTurnRecovery: h1.dismiss
}))
let nativeChatId = "tracked-chat-1"
const createdNativeChat = (body: { id?: string; character_id?: number | string }) => {
  if (!body.id) throw new Error("fixture requires the reserved client chat ID")
  nativeChatId = body.id
  if (body.character_id != null) {
    h1.roleplayEnabled = true
    h1.roleplayState.set(body.id, { tail: null, historyVersion: 9, rows: [], content: new Map() })
  }
  return { id: body.id }
}

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
      : view.cursor.kind === "before_message"
        ? nodes.filter(
            (row) => row.id === "u-old" && view.cursor.message_id !== "u-old"
          )
        : nodes.filter(
            (row) => row.id === "u-old" || row.id === view.cursor.message_id
          )
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: "native-key",
      conversation_id: nativeChatId,
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
const makeDraftAccount = (originIsCurrent: () => boolean = () => true): NonNullable<HistorySelectionController["draftAccount"]> => {
  const auth = h1.auth
  return {
    requestScope: mocks.makeSnapshot().requestScope,
    isCurrent: () => !auth.signal.aborted && originIsCurrent(),
    retryVerification: vi.fn()
  }
}
const installFixtureReceipt = (
  receipt: HistoryLoadReceipt,
  store?: typeof import("@/store/option").useStoreMessageOption,
  historyId: string | null = receipt.owner.kind === "local" ? receipt.owner.conversation_id : null
) => {
  const address = chatDraftAddressForReceipt(receipt)
  if (!address) throw new Error("fixture requires a qualified original receipt")
  const serverChatId = receipt.owner.kind === "native" ? receipt.owner.conversation_id : null
  mocks.chatBaseState.historyId = historyId
  mocks.storeState.serverChatId = serverChatId
  delete mocks.storeState.historyId
  if (store) {
    store.getState().setChatDraftAddress(address)
    store.setState({ historyId, serverChatId })
  } else {
    mocks.storeState.chatDraftAddress = address
  }
}
const bridgeDraftStore = (store: typeof import("@/store/option").useStoreMessageOption) => {
  const setHistoryId: State["setHistoryId"] = (id, options) => {
    store.getState().setHistoryId(id, options)
    mocks.chatBaseState.historyId = id
    mocks.chatBaseState.setHistoryId(id, options)
  }
  const setServerChatId: State["setServerChatId"] = id => {
    store.getState().setServerChatId(id)
    mocks.storeState.serverChatId = id
    mocks.storeState.setServerChatId(id)
  }
  mocks.draftFields = () => {
    const state = store.getState()
    return {
      chatDraftAddress: state.chatDraftAddress,
      chatDraftRevision: state.chatDraftRevision,
      chatDraftTransition: state.chatDraftTransition,
      historyId: state.historyId,
      serverChatId: state.serverChatId,
      setChatDraftAddress: state.setChatDraftAddress,
      startNewChatDraft: state.startNewChatDraft,
      setHistoryId,
      setServerChatId,
      setServerChatRoleplaySetup: state.setServerChatRoleplaySetup
    }
  }
}
const startFixtureDraft = async (temporary = false) => {
  const { useStoreMessageOption: store } =
    await vi.importActual<typeof import("@/store/option")>("@/store/option")
  store.getState().startNewChatDraft(temporary)
  store.setState({ historyId: temporary ? "temp" : null, serverChatId: null })
  mocks.chatBaseState.historyId = temporary ? "temp" : null
  mocks.storeState.serverChatId = null
  mocks.storeState.temporaryChat = temporary
  bridgeDraftStore(store)
  return store
}
const makeController = () => {
  const draftAccount = makeDraftAccount()
  let current: any = {
    owner: {
      kind: "native",
      conversation_id: nativeChatId,
      request_scope: mocks.makeSnapshot().requestScope,
      owner_key: "native-key",
      validate_lease: draftAccount.isCurrent
    },
    view: {
      owner_key: "native-key",
      conversation_id: nativeChatId,
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
    draftAccount,
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
  h1.recover.mockReset()
  h1.dismiss.mockReset()
  const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory.historySelections.rows.clear()
  await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
  nativeChatId = "tracked-chat-1"
  h1.roleplayEnabled = false
  h1.roleplayState.clear()
  h1.roleplayAppend.mockReset().mockImplementation(async (id, input, options) => {
    options.onDispatch?.()
    const state = h1.roleplayState.get(id)!
    const receipt = roleplayAppend(id, input)
    state.tail = receipt.fence.tail.message_id
    state.historyVersion = receipt.fence.history_version
    state.rows.push({ id: input.message_id, revision: "1", parent_id: input.expected.tail.message_id, role: "user", settled: true, message: input.content })
    return receipt
  })
  h1.roleplayComplete.mockReset().mockImplementation(async (id, body, options) => {
    options.onDispatch?.()
    const state = h1.roleplayState.get(id)!
    const reply = roleplayCompletion(id, body)
    state.tail = reply.tldw_roleplay_resume_v1.fence.tail.message_id
    state.historyVersion = reply.tldw_roleplay_resume_v1.fence.history_version
    state.rows.push({ id: reply.assistant_message_id, revision: "1", parent_id: reply.user_message_id, role: "assistant", settled: true, message: reply.assistant_content })
    return reply
  })
  h1.create.mockReset()
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  vi.mocked(api.getChat).mockReset().mockImplementation(async (id, options) => options?.requireRoleplayResume ? { ...roleplayDetail(id, h1.roleplayState.get(id)!.tail), history_version: h1.roleplayState.get(id)!.historyVersion } : { id, title: "Owned opening", character_id: "12" })
  vi.mocked(api.getChatSettings).mockReset().mockImplementation(async id => ({ conversation_id: id, settings: {} }))
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
  mocks.draftFields = null
  mocks.controllerReader = null
  mocks.useRealPageAssist = false
  mocks.storeState.detachedHistoryResponse = null
  mocks.storeState.replyTarget = null
  h1.auth = new AbortController()
  h1.cookieConfig = null
  const strictAdmission = {
    post: {
      requestBody: {
        content: {
          "application/json": {
            schema: protectedEditRequestSchema()
          }
        }
      }
    }
  }
  h1.spec
    .mockReset()
    .mockResolvedValue({
      paths: {
        "/api/v1/chat/completions": strictAdmission,
        "/api/v1/chats/{chat_id}/messages": strictAdmission
      }
    })

  h1.native = false
  h1.persona = false
  h1.beforeModel.mockReset()
  h1.supportsMultimodal = false
  mocks.chatBaseState.historyId = "history-1"
  mocks.storeState.temporaryChat = false
  mocks.storeState.serverChatCharacterId = null
  mocks.storeState.serverChatAssistantKind = null
  h1.controller = makeController()
  delete mocks.storeState.historyId
  installFixtureReceipt(h1.controller.getCurrent(), undefined, "history-1")
  mocks.storeState.chatDraftRevision = 0
  mocks.storeState.chatDraftTransition = null
  mocks.getActorScene.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
  mocks.chatBaseState.chatMode = "normal"
  mocks.chatBaseState.useOCR = false
  mocks.chatBaseState.selectedSystemPrompt = ""
  mocks.storeState.fileRetrievalEnabled = false
  mocks.storeState.toolChoice = "auto"
  mocks.storeState.selectedModel = "model-1"
  mocks.storeState.serverChatId = nativeChatId
  mocks.chatBaseState.history = [
    { role: "user", content: "old question" },
    { role: "assistant", content: "wrong latest" }
  ]
  mocks.loadServicePromptSnapshot.mockResolvedValue(
    mocks.makeSnapshot("supported", undefined, undefined, {
      scopeInvalidatedSignal: h1.auth.signal
    })
  )
  const { HumanMessage } = await import("@/types/messages")
  mocks.humanMessageFormatter.mockImplementation(
    async ({ content }: any) => new HumanMessage({ content })
  )
  h1.capture.mockImplementation(async (_id, request) =>
    captureFor(request.view)
  )
  h1.append.mockImplementation(async (_id: any, body: any) =>
    body.role === "user"
      ? {
          id: body.id,
          tldw_history_admission_v1: {
            version: 1,
            owner_key: "native-key",
            conversation_id: nativeChatId,
            input_message_id: body.id,
            input_message_revision: "accepted-r",
            selection_digest: body.tldw_history_selection_v1.selection_digest,
            messages: body.tldw_history_selection_v1.messages,
            originating_selection_revision:
              body.tldw_history_selection_v1.selection_revision
          }
        }
      : { id: body.id }
  )
  h1.wire.mockImplementation(async function* () {
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
})

it.each(
  [false, true].flatMap((cached) =>
    ["current", "new", "same-id", "account", "stop"].map((boundary) => ({
      cached,
      boundary
    }))
  )
)(
  "compact SDK persona keeps exact creation ownership across $boundary (cached=$cached)",
  async ({ cached, boundary }) => {
    h1.persona = true
    h1.controller = null
    mocks.storeState.fileRetrievalEnabled = false
    const store = await startFixtureDraft(false)
    mocks.storeState.serverChatMetaLoaded = cached
    if (cached) store.setState({ serverChatId: "compact-plain-before-persona" })
    const original = store.getState()
    const account = makeDraftAccount()
    const commandIsCurrent = () => {
      const latest = store.getState()
      return (
        account.isCurrent() &&
        latest.chatDraftAddress === original.chatDraftAddress &&
        latest.chatDraftRevision === original.chatDraftRevision
      )
    }
    const { captureActorSettingsTarget } = await import(
      "@/utils/actor-settings-target"
    )
    const { captureChatDraftSource } = await import(
      "@/hooks/chat/chat-draft-transition"
    )
    const { captureActorSceneSendForTarget } = await import(
      "@/hooks/chat/actor-scene-source"
    )
    const source = captureChatDraftSource(commandIsCurrent)
    const target = captureActorSettingsTarget({
      address: original.chatDraftAddress,
      requestScope: account.requestScope,
      accountIsCurrent: account.isCurrent,
      isCurrent: source.isCurrent
    })
    const scene = await captureActorSceneSendForTarget(
      target,
      account,
      source.isCurrent,
      commandIsCurrent
    )!.read()
    let release!: (value: {
      id: string
      assistant_kind: string
      assistant_id: string
    }) => void
    const created = new Promise<{
      id: string
      assistant_kind: string
      assistant_id: string
    }>((resolve) => {
      release = resolve
    })
    h1.create.mockReturnValueOnce(created)
    const normal = await import("@/hooks/chat-modes/normalChatMode")
    const dispatch = vi
      .spyOn(normal, "normalChatMode")
      .mockImplementationOnce(
        async (
          _message,
          _image,
          _regenerate,
          _rows,
          _history,
          _signal,
          params
        ) => {
          expect(params.actorSceneSend?.isCurrent()).toBe(true)
          return { status: "submitted" }
        }
      )
    const hook = renderHook(() => useMessage())
    const controller = new AbortController()
    let pending!: Promise<unknown>
    act(() => {
      pending = hook.result.current
        .onSubmit({
          message: "Original compact persona question",
          image: "",
          controller,
          actorSceneSend: scene
        })
        .catch((error) => {
          if (boundary !== "stop") throw error
          expect(error).toMatchObject({ name: "AbortError" })
          return { status: "skipped", reason: "aborted" }
        })
    })
    try {
      await waitFor(() => expect(h1.create).toHaveBeenCalledOnce())
      mocks.storeState.setServerChatMetaLoaded.mockClear()
      if (boundary === "new" || boundary === "same-id") {
        store.getState().startNewChatDraft(false)
        if (boundary === "same-id")
          store.setState({
            historyId: original.historyId,
            serverChatId: original.serverChatId
          })
      }
      if (boundary === "account") h1.auth.abort()
      if (boundary === "stop") controller.abort()
      await act(async () => {
        release({
          id: "compact-persona-created",
          assistant_kind: "persona",
          assistant_id: "12"
        })
        await pending
      })
      if (boundary === "current") {
        expect(dispatch).toHaveBeenCalledOnce()
        expect(store.getState().serverChatId).toBe("compact-persona-created")
        expect(mocks.storeState.setServerChatMetaLoaded).toHaveBeenCalledWith(
          true
        )
      } else {
        expect(dispatch).not.toHaveBeenCalled()
        expect(mocks.storeState.setServerChatMetaLoaded).not.toHaveBeenCalled()
      }
    } finally {
      release({
        id: "compact-persona-created",
        assistant_kind: "persona",
        assistant_id: "12"
      })
      await pending
      hook.unmount()
      dispatch.mockRestore()
    }
  }
)

it.each(["current", "model", "tool", "settings"])(
  "compact reused Persona stops metadata publication after %s changes",
  async (boundary) => {
    h1.persona = true
    h1.controller = null
    mocks.storeState.fileRetrievalEnabled = false
    const store = await startFixtureDraft(false)
    store.setState({ serverChatId: "existing-persona" })
    Object.assign(mocks.storeState, {
      serverChatMetaLoaded: true,
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "12"
    })
    const original = store.getState()
    const account = makeDraftAccount()
    const commandIsCurrent = () =>
      account.isCurrent() &&
      store.getState().chatDraftAddress === original.chatDraftAddress &&
      store.getState().chatDraftRevision === original.chatDraftRevision
    const { captureActorSettingsTarget } = await import(
      "@/utils/actor-settings-target"
    )
    const { captureChatDraftSource } = await import(
      "@/hooks/chat/chat-draft-transition"
    )
    const { captureActorSceneSendForTarget } = await import(
      "@/hooks/chat/actor-scene-source"
    )
    const source = captureChatDraftSource(commandIsCurrent)
    const target = captureActorSettingsTarget({
      address: original.chatDraftAddress,
      requestScope: account.requestScope,
      accountIsCurrent: account.isCurrent,
      isCurrent: source.isCurrent
    })
    const scene = await captureActorSceneSendForTarget(
      target,
      account,
      source.isCurrent,
      commandIsCurrent
    )!.read()
    const normal = await import("@/hooks/chat-modes/normalChatMode")
    const dispatch = vi
      .spyOn(normal, "normalChatMode")
      .mockResolvedValueOnce({ status: "submitted" })
    const { useStoreChatModelSettings } = await import("@/store/model")
    const settings = vi.spyOn(
      useStoreChatModelSettings.getState(),
      "getEffectiveSettings"
    )
    mocks.storeState.setServerChatAssistantKind.mockImplementationOnce(() => {
      if (boundary === "model")
        mocks.storeState.selectedModel = "replacement-model"
      if (boundary === "tool") mocks.storeState.toolChoice = "none"
      if (boundary === "settings")
        settings.mockReturnValue({ apiProvider: "replacement-provider" })
    })
    const hook = renderHook(() => useMessage())
    try {
      await act(async () => {
        const send = hook.result.current.onSubmit({
          message: "Existing compact Persona",
          image: "",
          actorSceneSend: scene
        })
        if (boundary === "current") await send
        else await expect(send).rejects.toMatchObject({ status: 412 })
      })
      expect(h1.create).not.toHaveBeenCalled()
      expect(mocks.storeState.setServerChatAssistantKind).toHaveBeenCalledWith(
        "persona"
      )
      expect(mocks.storeState.setServerChatAssistantId).toHaveBeenCalledTimes(
        boundary === "current" ? 1 : 0
      )
      expect(dispatch).toHaveBeenCalledTimes(boundary === "current" ? 1 : 0)
      expect(
        mocks.storeState.setServerChatPersonaMemoryMode
      ).toHaveBeenCalledTimes(boundary === "current" ? 1 : 0)
      expect(store.getState()).toMatchObject({
        historyId: original.historyId,
        serverChatId: original.serverChatId,
        chatDraftAddress: original.chatDraftAddress,
        chatDraftRevision: original.chatDraftRevision
      })
      expect(account.isCurrent()).toBe(true)
    } finally {
      hook.unmount()
      dispatch.mockRestore()
      settings.mockRestore()
    }
  }
)

it("edits an ordinary local sidepanel message while history selection is idle", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = [
    { id: "plain-message", isBot: false, message: "old", sources: [] }
  ]
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    expect(await result.current.editMessage(0, "edited", false, false)).toEqual(
      { status: "submitted" }
    )
  })
  expect(mocks.setMessages).toHaveBeenCalledWith([
    expect.objectContaining({ id: "plain-message", message: "edited" })
  ])
  expect(
    mocks.setHistory.mock.calls
      .at(-1)![0]
      .map((row: { content: string }) => row.content)
  ).toEqual(["edited"])
  expect(helpers.updateMessageById).toHaveBeenCalledWith(
    "history-1",
    "plain-message",
    "edited"
  )
})

it("deletes an ordinary local sidepanel message while history selection is idle", async () => {
  localMutationController()
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.deleteMessage(0)
  })
  expect(helpers.removeMessageById).toHaveBeenCalledWith(
    "history-1", "plain-message", undefined, { validate_lease: expect.any(Function) }
  )
})

it("routes ordinary sidepanel regeneration while history selection is idle", async () => {
  await startFixtureDraft()
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  mocks.storeState.serverChatId = null
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "again", image: "", isRegenerate: true })
  })
  expect(mocks.setMessages).toHaveBeenCalled()
})

it("mounted sidepanel normal submit uses canonical A1 selection and pre-admits its input", async () => {
  const { result } = renderHook(() => useMessage())
  h1.wire.mockImplementation(async function* () {
    expect(h1.append.mock.calls.map(([, body]: any) => body.role)).toEqual([
      "user"
    ])
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
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
  const [user, assistant] = h1.append.mock.calls.map(([, body]: any) => body)
  expect(
    user.tldw_history_selection_v1.messages.map((row: any) => row.id)
  ).toEqual(["u-old", "a1"])
  expect(assistant.tldw_history_admission_v1.input_message_id).toBe(user.id)
  expect(h1.append).toHaveBeenCalledTimes(2)
})

it.each(["model", "tools"])("rejects an ordinary sidepanel send after its global %s selector changes during preparation", async changed => {
  const { result } = renderHook(() => useMessage())
  h1.beforeModel.mockImplementation(() => {
    if (changed === "model") mocks.storeState.selectedModel = "changed-model"
    else mocks.storeState.toolChoice = "none"
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.append).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it.each(["model", "tools"])("keeps an explicit sidepanel %s override after its unrelated global selector changes", async overridden => {
  const { result } = renderHook(() => useMessage())
  h1.beforeModel.mockImplementation(() => {
    if (overridden === "model") mocks.storeState.selectedModel = "changed-model"
    else mocks.storeState.toolChoice = "required"
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
  expect(h1.wire.mock.calls[0][0].model).toBe(overridden === "model" ? "fixed-model" : "model-1")
  expect(h1.append).toHaveBeenCalledTimes(2)
})

it("before-first on a nonempty sidepanel source admits an empty prior path and sends only current input", async () => {
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  expect(current.capture.snapshot.nodes).toHaveLength(3)
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "fresh question", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(
    h1.wire.mock.calls[0][0].messages.filter(
      (row: any) => row.role !== "system"
    )
  ).toEqual([{ role: "user", content: "fresh question" }])
  const user = h1.append.mock.calls[0][1]
  expect(user.tldw_history_selection_v1.messages).toEqual([])
  expect(user.tldw_history_selection_v1.cursor).toEqual({ kind: "empty" })
  expect(user.parent_message_id ?? null).toBeNull()
})

it("an older sidepanel send cannot clear the newer send's activity or Stop controller", async () => {
  let entered!: () => void,
    releaseFirst!: () => void,
    releaseSecond!: () => void
  const started = new Promise<void>((resolve) => {
    entered = resolve
  })
  const first = new Promise<void>((resolve) => {
    releaseFirst = resolve
  })
  const second = new Promise<void>((resolve) => {
    releaseSecond = resolve
  })
  let secondEntered!: () => void
  const secondStarted = new Promise<void>((resolve) => {
    secondEntered = resolve
  })
  h1.wire.mockImplementationOnce(async function* () {
    entered()
    await first
    yield { choices: [{ delta: { content: "first" } }] }
  })
  h1.wire.mockImplementationOnce(async function* () {
    secondEntered()
    await second
    yield { choices: [{ delta: { content: "second" } }] }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    const old = result.current.onSubmit({ message: "first", image: "" })
    await started
    const newer = result.current.onSubmit({ message: "second", image: "" })
    await secondStarted
    mocks.setStreaming.mockClear()
    mocks.setIsProcessing.mockClear()
    mocks.setAbortController.mockClear()
    releaseFirst()
    await old
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(false)
    expect(mocks.setIsProcessing).not.toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
    releaseSecond()
    await newer
    expect(mocks.setStreaming).toHaveBeenLastCalledWith(false)
    expect(mocks.setAbortController).toHaveBeenLastCalledWith(null)
  })
})

const installWorkspaceCharacterFixture = () => {
  const current = h1.controller.getCurrent()
  if (current.owner?.kind !== "native") throw new Error("workspace fixture requires its accepted native owner")
  current.owner.scope = { type: "workspace", workspaceId: "original-compact-workspace" }
  installFixtureReceipt(current, undefined, mocks.chatBaseState.historyId)
}
const installSavedRoleplayFixture = () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  h1.roleplayEnabled = true
  h1.roleplayState.set(nativeChatId, { tail: "a1", historyVersion: 9, rows: [], content: new Map() })
}

it("workspace H1: mounted sidepanel native character uses owner selection and ACK without legacy persistence", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  h1.wire.mockImplementation(async function* (request) {
    yield {
      tldw_history_admission_v1: {
        version: 1,
        owner_key: "native-key",
        conversation_id: nativeChatId,
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
      tldw_conversation_id: nativeChatId
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.wire.mock.calls[0][0]).toMatchObject({
    save_to_db: true,
    conversation_id: nativeChatId,
    tldw_history_branch: false,
    messages: [{ role: "user", content: "next" }]
  })
  expect(
    h1.wire.mock.calls[0][0].tldw_history_selection_v1.messages.map(
      (row: any) => row.id
    )
  ).toEqual(["u-old", "a1"])
  expect(h1.controller.followResult).toHaveBeenCalledWith(
    expect.objectContaining({ view_session_id: "origin" }),
    "native-result"
  )
  expect(h1.append).not.toHaveBeenCalled()
})

const nativeAdmission = (request: any) => ({
  version: 1,
  owner_key: "native-key",
  conversation_id: nativeChatId,
  input_message_id: "native-input",
  input_message_revision: "native-r",
  selection_digest: request.tldw_history_selection_v1.selection_digest,
  messages: request.tldw_history_selection_v1.messages,
  originating_selection_revision:
    request.tldw_history_selection_v1.selection_revision
})

it.each(["current", "navigation", "account", "owner lease"])("compact native settings-read feedback belongs to its %s source", async boundary => {
  installSavedRoleplayFixture()
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  let rejectRead!: (error: Error) => void
  vi.mocked(api.getChat).mockImplementationOnce(() => new Promise((_, reject) => { rejectRead = reject }))
  const { result } = renderHook(() => useMessage())
  let sending!: Promise<unknown>
  await act(async () => {
    sending = result.current.onSubmit({ message: "  exact input  ", image: "" })
    await (await import("@testing-library/react")).waitFor(() => expect(api.getChat).toHaveBeenCalledOnce())
  })
  if (boundary === "navigation") h1.controller.navigate()
  else if (boundary === "account") h1.auth.abort()
  else if (boundary === "owner lease") h1.controller.getCurrent().owner.validate_lease = () => false
  let submitted: unknown
  await act(async () => { rejectRead(new Error("failed read")); submitted = await sending })
  expect(submitted).toMatchObject({ status: "failed", errorMessage: "native_history_settings_unverified" })
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.roleplayAppend).not.toHaveBeenCalled()
  expect(h1.roleplayComplete).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  if (boundary === "current") expect(mocks.notification.error.mock.lastCall?.[0].description).toContain("Your message has not been sent")
  else expect(mocks.notification.error).not.toHaveBeenCalled()
})

it("workspace H1: compact native admitted transport text cannot certify a failed settings read", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield await Promise.reject(new Error("native_history_settings_unverified"))
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({ state: "accepted_unsent", input_text: "next" })
  expect(mocks.notification.error.mock.lastCall?.[0].description).not.toContain("Your message has not been sent")
})

it("workspace H1: compact native confirmed context refusal retains raw input and shows a failed result without a transcript row", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  h1.wire.mockImplementation(async function* () {
    yield await Promise.reject(
      Object.assign(new Error("HTTP 409"), {
        status: 409,
        details: {
          detail: {
            status: "unsupported_history_capability",
            code: "unsupported_history_context_world_books"
          }
        }
      })
    )
  })
  const { result } = renderHook(() => useMessage())
  let submitted: unknown
  await act(async () => {
    submitted = await result.current.onSubmit({
      message: "  compact exact input  ",
      image: ""
    })
  })
  expect(submitted).toMatchObject({
    status: "failed",
    errorMessage: "native_history_context_refused"
  })
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    state: "refused",
    input_text: "  compact exact input  ",
    result_text: ""
  })
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(mocks.setMessages).not.toHaveBeenCalled()
  expect(h1.wire).toHaveBeenCalledOnce()
})

it.each(["http502", "sse-afteradmission"])("workspace H1: native provider recovery for %s retains its receipt and exposes existing model settings", async failure => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  const safeMessage = "The selected provider credentials could not be authenticated."
  h1.wire.mockImplementation(async function* (request) {
    if (failure === "http502") throw Object.assign(new Error(safeMessage), {
      status: 502, details: { detail: { error_code: "provider_authentication_failed", message: safeMessage } }
    })
    if (failure !== "sse-beforeadmission") yield { tldw_history_admission_v1: nativeAdmission(request) }
    if (failure === "sse-afterpartial") yield { choices: [{ delta: { content: "partial result" } }] }
    yield { error: { code: "provider_authentication_failed", message: safeMessage } }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  const shown = mocks.setMessages.mock.lastCall?.[0]
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
  expect(await result.current.regenerateLastMessage()).toEqual({ status: "skipped", reason: "message_changed" })
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.append).not.toHaveBeenCalled()
  expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
  expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
})

it.each(["navigation", "account"])("workspace H1: native provider recovery cannot overwrite a changed %s", async boundary => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  h1.wire.mockImplementation(async function* () {
    if (boundary === "navigation") h1.controller.navigate()
    else h1.auth.abort()
    mocks.setMessages.mockClear()
    yield { error: { code: "provider_authentication_failed", message: "Safe provider failure" } }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => { await result.current.onSubmit({ message: "next", image: "" }) })
  expect(mocks.setMessages).not.toHaveBeenCalled()
  expect(h1.wire).toHaveBeenCalledOnce()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({ persistence: "server", state: "unknown", input_text: "next" })
  expect(h1.recover.mock.lastCall?.[2].assistant_id).toBeUndefined()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
})


it("workspace H1: native before-first sends no historical content and retains unacknowledged text separately", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      choices: [{ delta: { content: "unsaved text" }, finish_reason: "stop" }]
    }
  })
  const { result } = renderHook(() => useMessage())
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
    conversation_id: nativeChatId,
    origin_view: {
      conversation_id: nativeChatId,
      view_session_id: "origin"
    },
    admission: {
      owner_key: "native-key",
      conversation_id: nativeChatId,
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
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["wrong-selection", "result-before-admission", "wrong-conversation"])(
  "workspace H1: native %s cannot supply an owner result",
  async (failure) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
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
          failure === "wrong-conversation" ? "other" : nativeChatId
      }
    })
    const { result } = renderHook(() => useMessage())
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
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("workspace H1: native selection navigation preserves server ownership and never overwrites the new view", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  const { result } = renderHook(() => useMessage())
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    h1.controller.navigate()
    mocks.setMessages.mockClear()
    yield { choices: [{ delta: { content: "old result" } }] }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: nativeChatId
    }
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(mocks.setMessages).not.toHaveBeenCalled()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.objectContaining({ conversation_id: nativeChatId }),
    expect.any(String),
    "completed"
  )
  expect(h1.append).not.toHaveBeenCalled()
})

it("workspace H1: native settings drift during capture prevents dispatch", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  const { useMcpToolsStore } = (await import("@/store/mcp-tools")) as any
  const previous = useMcpToolsStore.getState()
  h1.capture.mockImplementationOnce(async (_id, request) => {
    useMcpToolsStore.setState({ ...previous })
    return captureFor(request.view)
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["account change", "disconnect"])(
  "workspace H1: native %s retains accepted evidence under the origin without claiming response settlement",
  async (interruption) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      mocks.setMessages.mockClear()
      if (interruption === "account change") h1.auth.abort()
      else throw new Error("connection lost after admission")
      yield { choices: [{ delta: { content: "new account must not see" } }] }
      yield {
        tldw_message_id: "late-result",
        tldw_conversation_id: nativeChatId
      }
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    if (interruption === "account change")
      expect(mocks.setMessages).not.toHaveBeenCalled()
    else expect(h1.controller.refreshRecovery).toHaveBeenCalledOnce()
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      persistence: "server",
      state: "accepted_unsent",
      owner_key: "native-key",
      conversation_id: nativeChatId,
      origin_view: {
        conversation_id: nativeChatId,
        view_session_id: "origin"
      },
      admission: {
        owner_key: "native-key",
        conversation_id: nativeChatId,
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
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("workspace H1: a disconnect after the native owner ACK preserves the completed outcome", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: nativeChatId
    }
    throw new Error("late disconnect")
  })
  const { result } = renderHook(() => useMessage())
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
  expect(h1.append).not.toHaveBeenCalled()
})

it.each(["asset", "temporary", "local-owner"])(
  "native %s is visibly gated without dispatch",
  async (unsupported) => {
    h1.native = true
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    if (unsupported === "local-owner") localMutationController()
    mocks.storeState.temporaryChat = unsupported === "temporary"
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({
        message: "next",
        image: unsupported === "asset" ? "data:image/png;base64,asset" : ""
      })
    })
    expect(h1.wire).not.toHaveBeenCalled()
    expect(mocks.notification.error).toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it.each([false, true])(
  "native first creation is fenced around its ACK (navigation=%s)",
  async (navigate) => {
    const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
    draftStore.getState().startNewChatDraft(false)
    draftStore.setState({ historyId: null, serverChatId: null })
    const sourceDraft = draftStore.getState().chatDraftAddress
    bridgeDraftStore(draftStore)
    h1.native = true
    mocks.chatBaseState.historyId = null as any
    mocks.storeState.serverChatId = null
    mocks.storeState.serverChatCharacterId = null
    let origin = true
    let current: any = { owner: null, view: null, status: "idle" }
    const ready = makeController().getCurrent()
    ready.view.cursor = { kind: "empty" }
    ready.capture = captureFor(ready.view)
    const load = vi.fn(async (target, _reference, onLoaded) => {
      ready.owner.conversation_id = target.serverChatId
      ready.view.conversation_id = target.serverChatId
      ready.capture = captureFor(ready.view)
      current = ready
      onLoaded({ owner: current.owner, view: current.view })
      return true
    })
    h1.controller = {
      draftAccount: makeDraftAccount(() => origin),
      getCurrent: () => current,
      fence: () => () => origin,
      loadConversation: load,
      followResult: vi.fn(async () => true),
      refreshRecovery: vi.fn()
    }
    h1.create.mockImplementation(async body => {
      if (navigate) origin = false
      return createdNativeChat(body)
    })
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
      yield {
        tldw_message_id: "native-result",
        tldw_conversation_id: nativeChatId
      }
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(h1.create).toHaveBeenCalledOnce()
    if (navigate) {
      expect(draftStore.getState().chatDraftAddress).toBe(sourceDraft)
      expect(h1.wire).not.toHaveBeenCalled()
      expect(load).not.toHaveBeenCalled()
      expect(mocks.storeState.setServerChatId).not.toHaveBeenCalled()
    } else {
      expect(draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId, ownerKey: "native-key" })
      expect(draftStore.getState().chatDraftTransition).toMatchObject({ kind: "bind", sourceAddress: sourceDraft })
      expect(h1.roleplayAppend).toHaveBeenCalledWith(nativeChatId, expect.objectContaining({ content: "next", expected: expect.objectContaining({ tail: { message_id: null, message_version: null } }) }), expect.objectContaining({ requestScope: mocks.makeSnapshot().requestScope }))
      expect(h1.roleplayComplete).toHaveBeenCalledOnce()
      expect(h1.wire).not.toHaveBeenCalled()
      expect(mocks.storeState.setServerChatId).toHaveBeenCalledWith(
        nativeChatId
      )
    }
    expect(h1.live).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  }
)

it("dispatches first native creation through a real controller epoch and accepted bind", async () => {
  const actual = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  actual.useStoreMessageOption.getState().startNewChatDraft(false)
  actual.useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  bridgeDraftStore(actual.useStoreMessageOption)
  const { useHistorySelection } = await import("@/hooks/chat/useHistorySelection")
  const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory.forkOperations.rows.clear()
  const selection = renderHook(() => useHistorySelection())
  h1.controller = { ...selection.result.current, draftAccount: makeDraftAccount() }
  const sourceCurrent = h1.controller.fence()
  const service = await vi.importActual<typeof import("@/services/actor-settings")>("@/services/actor-settings")
  mocks.getActorScene.mockImplementation(service.getActorSceneForChat)
  const originalScene = { ...createDefaultActorSettings(), isEnabled: false, notes: "SAVED FRESH COMPACT SCENE" }
  const sourceTarget = captureCurrentActorSceneTarget(h1.controller)
  expect(sourceTarget).not.toBeNull()
  expect(await service.saveActorSettingsForChat({ target: sourceTarget!, expectedRecord: null, settings: originalScene })).toBe(true)
  const sourceRecord = (await service.getActorSceneForChat({ target: sourceTarget! })).originalRecord
  expect(sourceRecord?.value).toEqual(originalScene)
  h1.native = true
  mocks.chatBaseState.historyId = null
  mocks.storeState.serverChatId = null
  h1.create.mockImplementation(createdNativeChat)
  h1.capture.mockImplementation(async (_id, request) => {
    const result = captureFor({ ...request.view, owner_key: "native-key" })
    return { ...result, rows: [], selected_content: [], snapshot: { ...result.snapshot, nodes: [] } }
  })
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield { tldw_message_id: "native-result", tldw_conversation_id: nativeChatId }
  })
  const hook = renderHook(() => useMessage())
  await act(async () => { await hook.result.current.onSubmit({ message: "first", image: "" }) })
  expect(sourceCurrent()).toBe(false)
  expect(h1.create).toHaveBeenCalledOnce()
  expect(h1.roleplayAppend).toHaveBeenCalledOnce()
  expect(h1.roleplayComplete).toHaveBeenCalledOnce()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(actual.useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
  const boundTarget = captureCurrentActorSceneTarget(h1.controller)
  expect(boundTarget).not.toBeNull()
  expect(boundTarget!.key).not.toBe(sourceTarget!.key)
  const boundScene = await service.getActorSceneForChat({ target: boundTarget! })
  expect(boundScene.settings).toEqual(originalScene)
  expect(boundScene.originalRecord?.value).toEqual(sourceRecord!.value)
  expect(mocks.getActorScene).toHaveBeenCalledOnce()
  expect(JSON.stringify(h1.roleplayComplete.mock.calls[0][1])).not.toContain(originalScene.notes)
  hook.unmount()
  selection.unmount()
})

it("native dispatch freezes workspace routing independently of the view owner", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const workspace = { type: "workspace", workspaceId: "original-workspace" }
  h1.controller.getCurrent().owner.scope = workspace
  installFixtureReceipt(h1.controller.getCurrent(), undefined, "history-1")
  h1.wire.mockImplementation(async function* (request) {
    workspace.workspaceId = "new-workspace"
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: nativeChatId
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire.mock.calls[0][1].scope).toEqual({
    type: "workspace",
    workspaceId: "original-workspace"
  })
  expect(h1.live).not.toHaveBeenCalled()
})

it("workspace H1: an older native completion cannot release a newer operation's Stop control", async () => {
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  installWorkspaceCharacterFixture()
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
      tldw_conversation_id: nativeChatId
    }
  })
  h1.wire.mockImplementationOnce(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    enterSecond()
    await secondHeld
    yield {
      tldw_message_id: "second-result",
      tldw_conversation_id: nativeChatId
    }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    const first = result.current.onSubmit({ message: "first", image: "" })
    await firstEntered
    const second = result.current.onSubmit({ message: "second", image: "" })
    await secondEntered
    mocks.setStreaming.mockClear()
    mocks.setAbortController.mockClear()
    releaseFirst()
    await first
    expect(mocks.setStreaming).not.toHaveBeenCalledWith(false)
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
    releaseSecond()
    await second
    expect(mocks.setStreaming).toHaveBeenLastCalledWith(false)
    expect(mocks.setAbortController).toHaveBeenLastCalledWith(null)
  })
})

it("native required client tools are explicitly gated rather than silently discarded", async () => {
  h1.native = true
  mocks.storeState.toolChoice = "required"
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mocks.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: "Client tools are not supported in saved character chat. Set tools to None before starting."
    })
  )
})

it("native persona limitations remain visible before any owner operation", async () => {
  h1.persona = true
  const { result } = renderHook(() => useMessage())
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(mocks.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: expect.stringContaining("Persona conversations are not supported")
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
  installFixtureReceipt(current)
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
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
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
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
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
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = visible
    const options = {
      setMessages: mocks.setMessages,
      setHistory: mocks.setHistory
    }
    const { result } = renderHook(() => useMessage())
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
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
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
  mocks.storeState.serverChatId = null
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
  options.setMessages.mockClear()
  const pending = result.current.deleteMessage(0)
  await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
  current = false
  release({ id: "a1", parent_message_id: "u-old" })
  await pending
  expect(choose).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})
it("unqualified native edits and unsupported deletes cannot mutate or change the display", async () => {
  h1.controller.getCurrent().status = "unsupported_history_capability"
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
  mocks.storeState.serverChatId = nativeChatId
  mocks.storeState.messages = visible
  const options = {
    setMessages: mocks.setMessages,
    setHistory: mocks.setHistory
  }
  const { result } = renderHook(() => useMessage())
  options.setMessages.mockClear()
  await expect(
    result.current.editMessage(0, "bad", false, false)
  ).resolves.toEqual({ status: "skipped", reason: "stale_selection" })
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
  installFixtureReceipt(current)
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
  mocks.storeState.serverChatId = null
  const { result } = renderHook(() => useMessage())
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

it.each([false, true])(
  "real controller fork -> immediate send -> reopen keeps source unchanged (comparison=%s)",
  async (comparison) => {
    const actual = await vi.importActual<typeof import("@/store/option")>("@/store/option")
    actual.useStoreMessageOption.getState().startNewChatDraft(false)
    actual.useStoreMessageOption.setState({ historyId: null, serverChatId: null })
    bridgeDraftStore(actual.useStoreMessageOption)
    const { memory, seedLocalFork } = await import(
      "@/hooks/chat/__tests__/local-history-fixture"
    )
    const source = await seedLocalFork(comparison)
    const authority = await vi.importActual<any>("@/db/dexie/history-selection")
    if (comparison) {
      // Comparison opens for review without a ready-send callback. Start from
      // its actual profile-owned record; the real controller qualifies the view.
      const sourceOwner = await authority.getLocalHistoryOwner("history-1")
      actual.useStoreMessageOption.getState().setChatDraftAddress({
        kind: "local", id: sourceOwner.conversation_id, profileId: sourceOwner.profile_id
      })
      actual.useStoreMessageOption.getState().setHistoryId(sourceOwner.conversation_id)
    }
    const projector = await vi.importActual<any>("@/db/dexie/branch")
    localForks.capture.mockImplementation(projector.captureLocalForkSelection)
    localForks.prepare.mockImplementation(projector.prepareLocalFork)
    localForks.commit.mockImplementation(projector.commitLocalFork)
    // Bind the actual request digest rather than the older handler-only test stub.
    localForks.digest = projector.forkRequestDigest

    const { useHistorySelection } = await import(
      "@/hooks/chat/useHistorySelection"
    )
    const controller = renderHook(() => useHistorySelection())
    await act(async () => {
      await controller.result.current.loadConversation(
        { historyId: "history-1" },
        undefined,
        receipt => installFixtureReceipt(receipt, actual.useStoreMessageOption)
      )
    })
    if (comparison)
      expect(controller.result.current.getCurrent()).toMatchObject({
        status: "unsupported_history_capability",
        error: "unsupported_comparison_history",
        owner: { kind: "local", conversation_id: "history-1", profile_id: "profile" },
        view: { conversation_id: "history-1", owner_key: "local-history-v1:profile" }
      })
    if (!comparison)
      await act(async () => {
        await controller.result.current.choose({
          kind: "after_message",
          message_id: "a1"
        })
      })
    h1.controller = { ...controller.result.current, draftAccount: makeDraftAccount() }
    mocks.storeState.serverChatId = null
    mocks.storeState.fileRetrievalEnabled = false
    mocks.chatBaseState.historyId = "history-1"
    mocks.chatBaseState.setHistoryId.mockImplementation((id: string) => {
      mocks.chatBaseState.historyId = id
    })
    const hook = renderHook(() => useMessage())
    let outcome: any
    await act(async () => {
      outcome = await hook.result.current.createChatBranch(
        "a1",
        comparison ? { model_id: "A", cluster_id: "c1" } : undefined
      )
    })
    expect(outcome.state, JSON.stringify(outcome)).toBe("committed")
    const childId = outcome.child_id
    expect(actual.useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: childId })
    expect(mocks.chatBaseState.historyId).toBe(childId)
    expect(controller.result.current.getCurrent().owner).toMatchObject({
      kind: "local",
      conversation_id: childId
    })
    expect(controller.result.current.getReference()).toMatchObject({
      conversation_id: childId
    })
    h1.controller = { ...controller.result.current, draftAccount: makeDraftAccount() }
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
    mocks.storeState.serverChatId = null
    const hook = renderHook(() => useMessage())
    mocks.setMessages.mockClear()
    await expect(
      hook.result.current.editMessage(0, "changed", false, false)
    ).rejects.toThrow()
    await expect(hook.result.current.deleteMessage(0)).rejects.toThrow()
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenCalled()
  }
)
it("mounted native fork without local history retains an unknown create and never dispatches the same intent again", async () => {
  const {memory} = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory.forkOperations.rows.clear()
  const projector = await vi.importActual<any>("@/db/dexie/branch")
  localForks.digest = projector.forkRequestDigest
  h1.capture.mockImplementation(async (_id, request) => ({...captureFor(request.view), purpose: request.purpose,
    snapshot: {...captureFor(request.view).snapshot, native_fork_context: {policy: "plain_v1", supported: true, storage_context_digest: "storage"}}}))
  h1.create.mockRejectedValue(new Error("creation response lost"))
  mocks.chatBaseState.historyId = null
  const {result} = renderHook(() => useMessage())
  let first: any, second: any
  await act(async () => {first = await result.current.createChatBranch("a1")})
  await act(async () => {second = await result.current.createChatBranch("a1")})
  expect(first).toMatchObject({state: "unknown", owner_key: "native-key"})
  expect(second.operation_id).toBe(first.operation_id)
  expect(h1.create).toHaveBeenCalledTimes(1)
  expect(localForks.commit).not.toHaveBeenCalled()
  expect([...memory.forkOperations.rows.values()][0].request.operation_id).toBe(first.operation_id)
})


describe("compact original scene preparation", () => {
  it("freezes one scene before model discovery and uses its original scope and notes", async () => {
    const original = { ...createDefaultActorSettings(), isEnabled: true, notes: "ORIGINAL COMPACT SCENE" }
    let stored = original
    mocks.getActorScene.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored), originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
    const validation = await import("@/utils/chat-model-validation")
    let release!: (value: { status: "valid" }) => void
    const modelRead = vi.spyOn(validation, "validateSelectedChatModelAvailability").mockReturnValueOnce(new Promise(resolve => { release = resolve }))
    const view = renderHook(() => useMessage())
    let send!: ReturnType<typeof view.result.current.onSubmit>
    try {
      act(() => { send = view.result.current.onSubmit({ message: "Compact original question", image: "" }) })
      await waitFor(() => expect(modelRead).toHaveBeenCalled())
      const readsBeforeModel = mocks.getActorScene.mock.calls.length
      stored = { ...original, notes: "NEWER COMPACT SCENE" }
      await act(async () => { release({ status: "valid" }); await send })
      expect(readsBeforeModel).toBe(1)
      expect(mocks.getActorScene).toHaveBeenCalledOnce()
      expect(JSON.stringify(h1.wire.mock.calls[0][0])).toContain("ORIGINAL COMPACT SCENE")
      expect(JSON.stringify(h1.wire.mock.calls[0][0])).not.toContain("NEWER COMPACT SCENE")
      expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith([], expect.objectContaining({ requestScope: mocks.makeSnapshot().requestScope }))
    } finally { modelRead.mockRestore(); view.unmount() }
  })

  it("refuses replacement scene and provider dispatch after New during held model discovery", async () => {
    const validation = await import("@/utils/chat-model-validation")
    let release!: (value: { status: "valid" }) => void
    const modelRead = vi.spyOn(validation, "validateSelectedChatModelAvailability").mockReturnValueOnce(new Promise(resolve => { release = resolve }))
    const view = renderHook(() => useMessage())
    let send!: ReturnType<typeof view.result.current.onSubmit>
    try {
      act(() => { send = view.result.current.onSubmit({ message: "Compact original question", image: "" }) })
      await waitFor(() => expect(modelRead).toHaveBeenCalled())
      mocks.storeState.chatDraftAddress = freshChatDraftAddress(false)
      mocks.storeState.chatDraftRevision = 1
      h1.controller.navigate()
      await act(async () => { release({ status: "valid" }); await send })
      expect(mocks.getActorScene).toHaveBeenCalledOnce()
      expect(h1.append).not.toHaveBeenCalled()
      expect(h1.wire).not.toHaveBeenCalled()
      expect(mocks.notification.error).not.toHaveBeenCalled()
    } finally { modelRead.mockRestore(); view.unmount() }
  })
})

describe("compact alternate-mode preparation ownership", () => {
  type Mode = "preset" | "vision" | "website"
  const mountMode = async (mode: Mode) => {
    h1.supportsMultimodal = mode === "vision"
    const store = await startFixtureDraft()
    const current = h1.controller.getCurrent()
    current.owner = null
    current.view = null
    current.capture = null
    current.status = "idle"
    mocks.storeState.messages = []
    mocks.chatBaseState.history = []
    mocks.chatBaseState.chatMode = mode === "vision" ? "vision" : mode === "website" ? "rag" : "normal"
    mocks.chatBaseState.useOCR = false
    const originalMessages = mocks.setMessages.getMockImplementation()
    const originalHistory = mocks.setHistory.getMockImplementation()
    mocks.setMessages.mockImplementation(update => {
      mocks.storeState.messages = typeof update === "function" ? update(mocks.storeState.messages) : update
    })
    mocks.setHistory.mockImplementation(update => {
      mocks.chatBaseState.history = typeof update === "function" ? update(mocks.chatBaseState.history) : update
    })
    let selection: {
      unmount: () => void
      result: { current: HistorySelectionController }
    } | undefined
    if (mode === "website") {
      // The Website wrapper now creates and binds genuine local H1 ownership
      // before auxiliary preparation. IDs alone cannot stand in for this receipt.
      const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
      memory.chatHistories.rows.clear()
      memory.messages.rows.clear()
      memory.historySelections.rows.clear()
      await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
      mocks.chatBaseState.streaming = false
      mocks.chatBaseState.isProcessing = false
      mocks.chatBaseState.isEmbedding = false
      delete mocks.storeState.streaming
      delete mocks.storeState.isProcessing
      store.setState({ messages: [], history: [], streaming: false, isProcessing: false, isEmbedding: false })
      const { useHistorySelection } = await vi.importActual<typeof import("@/hooks/chat/useHistorySelection")>("@/hooks/chat/useHistorySelection")
      const { formatSelectedHistory } = await vi.importActual<typeof import("@/db/dexie/helpers")>("@/db/dexie/helpers")
      selection = renderHook(() => useHistorySelection({
        onCapture: capture => {
          const projection = formatSelectedHistory(capture)
          store.setState(projection)
          mocks.storeState.messages = projection.messages
          mocks.chatBaseState.history = projection.history
        }
      }))
      h1.controller = { ...selection.result.current, draftAccount: makeDraftAccount() }
    }
    const application = await import("@/services/application")
    const screenshots = await import("@/libs/get-screenshot")
    const { ChatTldw } = await import("@/models/ChatTldw")
    const prompt = vi.spyOn(application, "getPrompt").mockResolvedValue("Preset {text}")
    const screenshot = vi.spyOn(screenshots, "getScreenshotFromCurrentTab").mockResolvedValue({
      success: true, screenshot: "data:image/png;base64,original-screenshot", error: null
    })
    const stream = vi.spyOn(ChatTldw.prototype, "stream")
    const invoke = vi.spyOn(ChatTldw.prototype, "invoke")
    mocks.addMedia.mockResolvedValue({ media_id: 37 })
    mocks.ragSearch.mockResolvedValue({ results: [{ content: "Original website context", metadata: { title: "Source" } }] })
    mocks.renderServicePromptPart.mockReturnValue("Original website answer prompt")
    const hook = renderHook(() => useMessage())
    const input = { message: "Original compact mode question", image: "", ...(mode === "preset" ? { messageType: "summary" } : {}) }
    return { store, hook, input, prompt, screenshot, stream, invoke, selection,
      cleanup: () => {
        hook.unmount()
        selection?.unmount()
        prompt.mockRestore(); screenshot.mockRestore(); stream.mockRestore(); invoke.mockRestore()
        mocks.setMessages.mockReset(); mocks.setHistory.mockReset()
        if (originalMessages) mocks.setMessages.mockImplementation(originalMessages)
        if (originalHistory) mocks.setHistory.mockImplementation(originalHistory)
        mocks.addMedia.mockReset(); mocks.ragSearch.mockReset(); mocks.renderServicePromptPart.mockReset()
      }
    }
  }
  const mountOwnedEmissionMode = async (mode: Mode) => {
    const mounted = await mountMode(mode)
    mounted.hook.unmount()
    const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
    memory.chatHistories.rows.clear()
    memory.messages.rows.clear()
    const originalStreaming = mocks.setStreaming.getMockImplementation()
    const originalProcessing = mocks.setIsProcessing.getMockImplementation()
    const originalEmbedding = mocks.setIsEmbedding.getMockImplementation()
    mocks.setMessages.mockImplementation(update => {
      const messages = typeof update === "function" ? update(mocks.storeState.messages) : update
      mocks.storeState.messages = messages
      mounted.store.setState({ messages })
    })
    mocks.setHistory.mockImplementation(update => {
      const history = typeof update === "function" ? update(mocks.chatBaseState.history) : update
      mocks.chatBaseState.history = history
      mounted.store.setState({ history })
    })
    mocks.setStreaming.mockImplementation(streaming => {
      mocks.chatBaseState.streaming = streaming
      mounted.store.setState({ streaming })
    })
    mocks.setIsProcessing.mockImplementation(isProcessing => {
      mocks.chatBaseState.isProcessing = isProcessing
      mounted.store.setState({ isProcessing })
    })
    mocks.setIsEmbedding.mockImplementation(isEmbedding => {
      mocks.chatBaseState.isEmbedding = isEmbedding
      mounted.store.setState({ isEmbedding })
    })
    mocks.chatBaseState.streaming = false
    mocks.chatBaseState.isProcessing = false
    mounted.store.setState({ messages: [], history: [], streaming: false, isProcessing: false })
    const account = h1.controller.draftAccount as NonNullable<HistorySelectionController["draftAccount"]>
    const originalScope = account.requestScope
    if (!originalScope) throw new Error("fixture requires the original verified account")
    const service = await vi.importActual<typeof import("@/services/actor-settings")>("@/services/actor-settings")
    const sourceTarget = captureCurrentActorSceneTarget(h1.controller)
    if (!sourceTarget) throw new Error("fixture requires the original typed fresh target")
    const originalScene = { ...createDefaultActorSettings(), isEnabled: false, notes: `Retained original ${mode} scene` }
    expect(await service.saveActorSettingsForChat({ target: sourceTarget, expectedRecord: null, settings: originalScene })).toBe(true)
    const sourceRecord = (await service.getActorSceneForChat({ target: sourceTarget })).originalRecord
    expect(sourceRecord?.value).toEqual(originalScene)
    mocks.getActorScene.mockImplementation(service.getActorSceneForChat)
    const { createLocalRegistryBucket } = await import("@/services/settings/local-bucket")
    const sceneBucket = createLocalRegistryBucket<import("@/types/actor").ActorSettings>({ prefix: "actorSettings:owned:" })
    const originalScenes = await sceneBucket.list()
    expect(originalScenes.unavailable).toBe(false)
    const session = (await import("@/store/playground-session")).usePlaygroundSessionStore.getState()
    const originalSession = { restoreRevision: session.restoreRevision, sourceSelectionRevision: session.sourceSelectionRevision }
    mocks.useRealPageAssist = true
    const hook = renderHook(() => ({ ...useMessage(), context: usePageAssist() }), { wrapper: PageAssistProvider })
    return { ...mounted, hook, memory, account, originalScope, sourceTarget, originalScene, sceneBucket, originalScenes, originalSession,
      cleanup: () => {
        hook.unmount()
        mounted.cleanup()
        mocks.useRealPageAssist = false
        mocks.chatBaseState.streaming = false
        mocks.chatBaseState.isProcessing = false
        mocks.chatBaseState.isEmbedding = false
        mocks.setStreaming.mockReset(); mocks.setIsProcessing.mockReset(); mocks.setIsEmbedding.mockReset()
        if (originalStreaming) mocks.setStreaming.mockImplementation(originalStreaming)
        if (originalProcessing) mocks.setIsProcessing.mockImplementation(originalProcessing)
        if (originalEmbedding) mocks.setIsEmbedding.mockImplementation(originalEmbedding)
      }
    }
  }
  const boundaries = [
    { mode: "preset" as const, stage: "nickname" },
    { mode: "preset" as const, stage: "prompt" },
    { mode: "vision" as const, stage: "nickname" },
    { mode: "vision" as const, stage: "prompt" },
    { mode: "vision" as const, stage: "screenshot" },
    { mode: "website" as const, stage: "nickname" },
    { mode: "website" as const, stage: "prompt" },
    { mode: "website" as const, stage: "retrieval" }
  ].flatMap(boundary => ["New", "replacement"].map(retirement => ({ ...boundary, retirement })))
  it.each(boundaries)("compact $mode refuses provider admission after $retirement during held $stage", async ({ mode, stage, retirement }) => {
    const mounted = await mountMode(mode)
    let release!: () => void, reached = false
    const held = new Promise<void>(resolve => { release = resolve })
    const nickname = vi.mocked((await import("@/db/dexie/nickname")).getModelNicknameByID)
    const systemPrompt = vi.mocked((await import("~/services/tldw-server")).systemPromptForNonRag)
    if (stage === "nickname") nickname.mockImplementationOnce(async () => {
      reached = true; await held
      return { model_avatar: "avatar.png", model_name: "Model One" } as Awaited<ReturnType<typeof nickname>>
    })
    else if (stage === "prompt" && mode === "preset") mounted.prompt.mockImplementationOnce(async () => { reached = true; await held; return "Preset {text}" })
    else if (stage === "prompt" && mode === "vision") systemPrompt.mockImplementationOnce(async () => { reached = true; await held; return "Original vision system" })
    else if (stage === "prompt") mocks.loadServicePromptSnapshot.mockImplementationOnce(async () => { reached = true; await held; return mocks.makeSnapshot() })
    else if (stage === "screenshot") mounted.screenshot.mockImplementationOnce(async () => {
      reached = true; await held
      return { success: true, screenshot: "data:image/png;base64,original-screenshot", error: null }
    })
    else mocks.ragSearch.mockImplementationOnce(async () => { reached = true; await held; return { results: [{ content: "Original website context" }] } })
    let pending!: ReturnType<typeof mounted.hook.result.current.onSubmit>
    const replacement = [{ id: "replacement-row", name: "You", isBot: false, message: "Replacement conversation must stay", sources: [] }]
    const replacementHistory = [{ role: "user" as const, content: "Replacement conversation must stay" }]
    try {
      act(() => { pending = mounted.hook.result.current.onSubmit(mounted.input) })
      void pending.catch(() => {})
      await waitFor(() => expect(reached).toBe(true))
      if (mode === "website" && retirement === "replacement") {
        const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
        await memory.chatHistories.put({ id: "replacement-history", title: "Replacement conversation", createdAt: 2, is_rag: false, local_owner_key: "local-history-v1:profile" })
        await memory.messages.put({ id: replacement[0].id, history_id: "replacement-history", role: "user", name: "You", content: replacement[0].message, images: [], createdAt: 2, parent_message_id: null })
        await act(async () => {
          expect(await mounted.selection!.result.current.loadConversation(
            { historyId: "replacement-history" }, undefined,
            receipt => installFixtureReceipt(receipt, mounted.store)
          )).toBe(true)
        })
      }
      act(() => {
        if (retirement === "New") {
          mounted.store.getState().startNewChatDraft(false)
          mounted.store.setState({ historyId: null, serverChatId: null })
        } else if (mode !== "website") {
          h1.controller.navigate()
          const destination = makeController()
          destination.getCurrent().owner.conversation_id = "replacement-chat"
          destination.getCurrent().view.conversation_id = "replacement-chat"
          h1.controller = destination
          installFixtureReceipt(destination.getCurrent(), mounted.store, "replacement-history")
        }
        mocks.storeState.messages = replacement
        mocks.chatBaseState.history = replacementHistory
        mounted.store.setState({ messages: replacement, history: replacementHistory })
        mounted.hook.rerender()
      })
      await act(async () => { release(); await pending })
      expect(mounted.stream).not.toHaveBeenCalled()
      expect(mounted.invoke).not.toHaveBeenCalled()
      expect(h1.wire).not.toHaveBeenCalled()
      expect(h1.append).not.toHaveBeenCalled()
      expect(mocks.storeState.messages).toEqual(replacement)
      expect(mocks.chatBaseState.history).toEqual(replacementHistory)
      expect(mounted.store.getState().messages).toEqual(replacement)
      expect(mounted.store.getState().history).toEqual(replacementHistory)
      expect(mocks.notification.error).not.toHaveBeenCalled()
      expect(mocks.notification.warning).not.toHaveBeenCalled()
      expect(mocks.notification.info).not.toHaveBeenCalled()
    } finally {
      release()
      try { await pending } finally { mounted.cleanup() }
    }
  })
  it("fresh compact H1 Persona refuses its unsupported capability before any dispatch", async () => {
    h1.persona = true
    h1.create.mockResolvedValueOnce({ id: "persona-created", assistant_kind: "persona", assistant_id: "12", persona_memory_mode: "read_only" })
    const mounted = await mountMode("preset")
    try {
      let outcome: unknown
      await act(async () => {
        outcome = await mounted.hook.result.current.onSubmit({ message: "Keep this exact Persona input", image: "" })
      })
      expect(outcome).toMatchObject({ status: "skipped", reason: "native_history_persona_unsupported" })
      expect(mocks.notification.error).toHaveBeenCalledOnce()
      const description = mocks.notification.error.mock.calls[0][0].description
      expect(description).toMatch(/persona/i)
      expect(description).toMatch(/not supported|unsupported|unavailable/i)
      expect(description).not.toContain("native_history_persona_unsupported")
      expect(h1.create).not.toHaveBeenCalled()
      expect(h1.capture).not.toHaveBeenCalled()
      expect(h1.append).not.toHaveBeenCalled()
      expect(mocks.pageAssistModel).not.toHaveBeenCalled()
      expect(mounted.stream).not.toHaveBeenCalled()
      expect(mounted.invoke).not.toHaveBeenCalled()
      expect(h1.wire).not.toHaveBeenCalled()
    } finally { mounted.cleanup() }
  })
  it.each(["preset", "vision", "website"] as const)("compact $mode forwards the original scope and explicit model/OCR choices", async mode => {
    const mounted = await mountMode(mode)
    const originalScope = h1.controller.draftAccount.requestScope
    try {
      await act(async () => {
        await mounted.hook.result.current.onSubmit({
          ...mounted.input,
          requestOverrides: { selectedModel: "explicit-mode-model", useOCR: true }
        })
      })
      expect(mocks.pageAssistModel).toHaveBeenCalledWith(expect.objectContaining({ model: "explicit-mode-model", requestScope: originalScope }))
      expect(mocks.humanMessageFormatter).toHaveBeenCalledWith(expect.objectContaining({ model: "explicit-mode-model", useOCR: true }))
      expect(mounted.stream).toHaveBeenCalledOnce()
      expect(h1.wire).toHaveBeenCalledOnce()
      expect(h1.wire.mock.calls[0][0].model).toBe("explicit-mode-model")
    } finally { mounted.cleanup() }
  })
  it.each([
    { mode: "document" as const, retirement: "raw IDs" },
    { mode: "tab" as const, retirement: "account" }
  ])("compact $mode late mode emission preserves the replacement after $retirement retirement", async ({ mode, retirement }) => {
    const mounted = await mountOwnedEmissionMode("preset")
    const document = await import("../chat-modes/documentChatMode")
    const tab = await import("../chat-modes/tabChatMode")
    const modeCall = vi.mocked(mode === "document" ? document.documentChatMode : tab.tabChatMode)
    const originalRows = [{ id: "original-mode-row", name: "You", isBot: false, message: "Original mode emission", sources: [] }]
    const originalHistory = [{ role: "user" as const, content: "Original mode emission" }]
    const replacementRows = [{ id: "replacement-mode-row", name: "You", isBot: false, message: "Keep replacement mode rows", sources: [] }]
    const replacementHistory = [{ role: "user" as const, content: "Keep replacement mode history" }]
    let release!: () => void, reached = false, emittedLate = false
    const held = new Promise<void>(resolve => { release = resolve })
    modeCall.mockImplementationOnce(async (...args: Parameters<typeof document.documentChatMode> | Parameters<typeof tab.tabChatMode>) => {
      const params = args.at(-1) as Parameters<typeof document.documentChatMode>[7] | Parameters<typeof tab.tabChatMode>[7]
      expect(params.actorSceneSend?.requestScope).toEqual(mounted.originalScope)
      expect(params.actorScene?.target.key).toBe(mounted.sourceTarget.key)
      expect(params.actorScene?.originalRecord?.value).toEqual(mounted.originalScene)
      params.setMessages(originalRows)
      params.setHistory(originalHistory)
      params.setStreaming(true)
      params.setIsProcessing(true)
      reached = true
      await held
      emittedLate = true
      params.setMessages([{ id: "late-mode-row", name: "Assistant", isBot: true, message: "Obsolete mode emission", sources: [] }])
      params.setHistory([{ role: "assistant", content: "Obsolete mode history" }])
      params.setHistoryId("late-mode-history")
      params.setIsProcessing(false)
      params.setStreaming(false)
      params.setAbortController(null)
      return { status: "submitted" as const }
    })
    const originalAddress = mounted.store.getState().chatDraftAddress
    const originalRevision = mounted.store.getState().chatDraftRevision
    const input: Parameters<typeof mounted.hook.result.current.onSubmit>[0] = {
      message: "Original raw mode question", image: "",
      ...(mode === "document"
        ? { uploadedFiles: [{ id: "original-file", filename: "original.txt", type: "text/plain", content: "Original document", size: 17, uploadedAt: 1, processed: true }] }
        : { docs: [{ type: "tab" as const, tabId: 1, title: "Original tab", url: "https://original.example" }] })
    }
    let pending!: ReturnType<typeof mounted.hook.result.current.onSubmit>
    try {
      act(() => { pending = mounted.hook.result.current.onSubmit(input) })
      await waitFor(() => expect(reached).toBe(true))
      expect(mounted.store.getState().messages).toEqual(originalRows)
      expect(mounted.store.getState().history).toEqual(originalHistory)
      expect(mounted.hook.result.current.context.getController!()).not.toBeNull()
      act(() => {
        if (retirement === "raw IDs") {
          mounted.store.getState().setHistoryId("replacement-local", { preserveServerChatId: true })
          mounted.store.getState().setServerChatId("replacement-chat")
        } else h1.auth.abort()
        mocks.storeState.messages = replacementRows
        mocks.chatBaseState.history = replacementHistory
        mounted.store.setState({ messages: replacementRows, history: replacementHistory })
        mounted.hook.rerender()
      })
      expect(mounted.store.getState().chatDraftAddress).toBe(originalAddress)
      expect(mounted.store.getState().chatDraftRevision).toBe(originalRevision)
      expect(mounted.account.isCurrent()).toBe(retirement === "raw IDs")
      await act(async () => { release(); await pending })
      expect(emittedLate).toBe(true)
      expect(modeCall).toHaveBeenCalledOnce()
      expect(mocks.getActorScene).toHaveBeenCalledOnce()
      expect(mounted.store.getState().messages).toEqual(replacementRows)
      expect(mounted.store.getState().history).toEqual(replacementHistory)
      expect(mocks.storeState.messages).toEqual(replacementRows)
      expect(mocks.chatBaseState.history).toEqual(replacementHistory)
      expect(mounted.store.getState().historyId).toBe(retirement === "raw IDs" ? "replacement-local" : null)
      expect(mounted.store.getState().serverChatId).toBe(retirement === "raw IDs" ? "replacement-chat" : null)
      expect((await import("@/store/playground-session")).usePlaygroundSessionStore.getState()).toMatchObject(mounted.originalSession)
      expect(mocks.notification.error).not.toHaveBeenCalled()
    } finally {
      release()
      try { await pending } finally { modeCall.mockReset(); mounted.cleanup() }
    }
  })

  it.each(["New", "cookie principal"] as const)("compact Website's real admitted snapshot distinguishes %s retirement", async boundary => {
    const originalSnapshotFactory = mocks.makeSnapshot
    const cookieConfig = { serverUrl: "https://cookie-website.example", authMode: "single-user" as const, authSource: "cookie-session" as const }
    h1.cookieConfig = cookieConfig
    const { deriveSingleUserApiKeyCredentialScope } = await import("@/services/chat-surface-scope")
    const requestConfig = { ...cookieConfig,
      expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(cookieConfig.authMode, undefined)! }
    mocks.makeSnapshot = ((...args: Parameters<typeof originalSnapshotFactory>) => ({
      ...originalSnapshotFactory(...args), requestScope: { config: requestConfig, userId: null }
    })) as typeof originalSnapshotFactory
    const deploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "self_host")
    let mounted: Awaited<ReturnType<typeof mountOwnedEmissionMode>> | undefined
    let actualSnapshot: import("@/services/service-prompts").ServicePromptSnapshot | undefined
    let release!: () => void, entered = false, enter!: () => void
    const opened = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    let pending: ReturnType<ReturnType<typeof useMessage>["onSubmit"]> | undefined
    try {
      const service = await vi.importActual<typeof import("@/services/service-prompts")>("@/services/service-prompts")
      const { createSafeStorage } = await import("@/utils/safe-storage")
      const storage = createSafeStorage({ area: "local" })
      await storage.set("tldwCookieSessionConfig", cookieConfig)
      mounted = await mountOwnedEmissionMode("website")
      mocks.loadServicePromptSnapshot.mockImplementationOnce(async (_ids, options) => {
        // Keep the existing declared catalog definitions, but use the real
        // service's retained request/account signals and release lifecycle.
        actualSnapshot = await service.loadServicePromptSnapshot([], options)
        return { ...mocks.makeSnapshot(), ...actualSnapshot, definitions: mocks.makeSnapshot().definitions }
      })
      h1.wire.mockImplementationOnce(async function* () {
        yield { choices: [{ delta: { content: "Partial cookie-owned answer" } }] }
        entered = true
        enter()
        await held
        yield { choices: [{ delta: { content: " completed" } }] }
      })
      act(() => { pending = mounted!.hook.result.current.onSubmit(mounted!.input) })
      void pending!.catch(() => {})
      await Promise.race([opened, pending!.then(outcome => { throw new Error(`Cookie Website settled before admission/provider: ${JSON.stringify(outcome)}`) })])
      expect(entered).toBe(true)
      expect(actualSnapshot!.requestScope.config).toMatchObject(requestConfig)
      expect(actualSnapshot!.requestScope.userId).toBeNull()
      const accepted = mounted.selection!.result.current.getCurrent()
      expect(accepted.status).toBe("ready")
      expect(accepted.owner).toMatchObject({ kind: "local", profile_id: "profile" })
      const historyId = accepted.owner!.conversation_id
      expect((await mounted.memory.messages.where("history_id").equals(historyId).toArray()).map((row: StoredMessage) => [row.role, row.content]))
        .toEqual([["user", mounted.input.message]])
      expect(actualSnapshot?.scopeInvalidatedSignal.aborted).toBe(false)
      const response = mounted.hook.result.current.context.getController!()!
      if (boundary === "cookie principal") {
        h1.cookieConfig = null
        await storage.remove("tldwCookieSessionConfig")
        act(() => { window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })) })
      } else act(() => {
        mounted!.store.getState().startNewChatDraft()
        mounted!.store.setState({ messages: [{ id: "new-cookie-draft", isBot: false, message: "Keep same-account New", sources: [] }] })
        mounted!.hook.rerender()
      })
      expect(response.signal.aborted).toBe(false)
      let outcome: unknown
      await act(async () => { release(); outcome = toChatSubmitResult(await pending!) })
      expect(outcome).toEqual(boundary === "New" ? { status: "submitted" } : { status: "failed", errorMessage: "The server or authenticated account changed." })
      const retained = await mounted.memory.messages.where("history_id").equals(historyId).toArray()
      expect(retained.map((row: StoredMessage) => [row.role, row.content])).toEqual(boundary === "New"
        ? [["user", mounted.input.message], ["assistant", "Partial cookie-owned answer completed"]]
        : [["user", mounted.input.message]])
      expect(mounted.hook.result.current.context.getController!()).toBeNull()
      expect(h1.wire).toHaveBeenCalledOnce()
    } finally {
      release?.()
      try { await pending?.catch(() => {}) } finally {
        actualSnapshot?.release()
        mounted?.cleanup()
        mocks.makeSnapshot = originalSnapshotFactory
        h1.cookieConfig = null
        vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", deploymentMode)
      }
    }
  })

  it("compact Website refuses local settlement after the admitted provider's original account retires", async () => {
    const mounted = await mountOwnedEmissionMode("website")
    let release!: () => void, entered = false
    const held = new Promise<void>(resolve => { release = resolve })
    h1.wire.mockImplementationOnce(async function* () {
      yield { choices: [{ delta: { content: "Partial original answer" } }] }
      entered = true
      await held
      yield { choices: [{ delta: { content: " completed" } }] }
    })
    let pending!: ReturnType<typeof mounted.hook.result.current.onSubmit>
    try {
      act(() => { pending = mounted.hook.result.current.onSubmit(mounted.input) })
      void pending.catch(() => {})
      await waitFor(() => expect(entered).toBe(true))
      const accepted = mounted.selection!.result.current.getCurrent()
      expect(accepted.status).toBe("ready")
      expect(accepted.owner).toMatchObject({ kind: "local", profile_id: "profile" })
      const historyId = accepted.owner!.conversation_id
      const admitted = await mounted.memory.messages.where("history_id").equals(historyId).toArray()
      expect(admitted.map((row: StoredMessage) => [row.role, row.content])).toEqual([["user", mounted.input.message]])
      expect(mounted.account.isCurrent()).toBe(true)
      // Retire the actual captured snapshot/account without replacing the
      // local profile or aborting the response signal as if Stop were pressed.
      act(() => { h1.auth.abort(); mounted.hook.rerender() })
      expect(mounted.account.isCurrent()).toBe(false)
      await act(async () => {
        release()
        expect(toChatSubmitResult(await pending)).toEqual({ status: "failed", errorMessage: "The server or authenticated account changed." })
      })
      expect(h1.wire).toHaveBeenCalledOnce()
      const retained = await mounted.memory.messages.where("history_id").equals(historyId).toArray()
      expect(retained.map((row: StoredMessage) => [row.role, row.content])).toEqual([["user", mounted.input.message]])
      expect(mounted.hook.result.current.context.getController!()).toBeNull()
      expect(mocks.chatBaseState.streaming).toBe(false)
      expect((await mounted.sceneBucket.list()).records[mounted.sourceTarget.key]).toEqual(mounted.originalScenes.records[mounted.sourceTarget.key])
    } finally {
      release()
      try { await pending } finally { mounted.cleanup() }
    }
  })

  it.each(["preset", "vision", "website"] as const)("compact $mode keeps original scoped persistence and releases its controller after New during an admitted provider", async mode => {
    const mounted = await mountOwnedEmissionMode(mode)
    const helpers = await import("@/db/dexie/helpers")
    const saveHistory = vi.spyOn(helpers, "saveHistory")
    const title = vi.spyOn(await import("@/services/title"), "generateTitle").mockResolvedValue(`Original ${mode} history`)
    const syncSettings = vi.spyOn(await import("@/services/chat-settings"), "syncChatSettingsForServerChat")
    const replacementRows = [{ id: "replacement-provider-row", name: "You", isBot: false, message: "Keep replacement provider rows", sources: [] }]
    const replacementHistory = [{ role: "user" as const, content: "Keep replacement provider history" }]
    const replacementSettings = { selectedModel: "replacement-model", selectedSystemPrompt: "replacement-prompt", useOCR: false, toolChoice: "none" as const }
    let release!: () => void, entered = false, emittedLate = false
    const held = new Promise<void>(resolve => { release = resolve })
    h1.wire.mockImplementationOnce(async function* () {
      yield { choices: [{ delta: { content: "Partial " } }] }
      entered = true
      await held
      emittedLate = true
      yield { choices: [{ delta: { content: "original final answer" } }] }
    })
    let pending!: ReturnType<typeof mounted.hook.result.current.onSubmit>
    let replacementAddress: State["chatDraftAddress"]
    let acceptedWebsiteSceneKey: string | undefined
    let acceptedWebsiteScenes: typeof mounted.originalScenes.records | undefined
    try {
      act(() => { pending = mounted.hook.result.current.onSubmit({ ...mounted.input, requestOverrides: { selectedModel: "original-finish-model", useOCR: true } }) })
      void pending.catch(() => {})
      await waitFor(() => expect(entered).toBe(true))
      expect(mounted.stream).toHaveBeenCalledOnce()
      expect(h1.wire).toHaveBeenCalledOnce()
      expect(mocks.chatBaseState.streaming).toBe(true)
      expect(mocks.chatBaseState.isProcessing).toBe(true)
      const ownedController = mounted.hook.result.current.context.getController!()
      expect(ownedController).not.toBeNull()
      if (mode === "website") {
        const accepted = mounted.selection!.result.current.getCurrent()
        expect(accepted.status).toBe("ready")
        expect(accepted.owner).toMatchObject({ kind: "local", profile_id: "profile" })
        expect(mounted.store.getState().chatDraftAddress).toMatchObject({ kind: "local", id: accepted.owner!.conversation_id, profileId: "profile" })
        expect(saveHistory).toHaveBeenCalledExactlyOnceWith(mounted.input.message, false, "web-ui", undefined, undefined, mounted.originalScope)
        const boundTarget = captureCurrentActorSceneTarget(h1.controller)
        expect(boundTarget).not.toBeNull()
        acceptedWebsiteSceneKey = boundTarget!.key
        expect(acceptedWebsiteSceneKey).not.toBe(mounted.sourceTarget.key)
        const actorService = await vi.importActual<typeof import("@/services/actor-settings")>("@/services/actor-settings")
        const boundScene = await actorService.getActorSceneForChat({ target: boundTarget! })
        expect(boundScene.settings).toEqual(mounted.originalScene)
        expect(boundScene.originalRecord?.value).toEqual(mounted.originalScene)
        acceptedWebsiteScenes = (await mounted.sceneBucket.list()).records
      } else expect(saveHistory).not.toHaveBeenCalled()
      act(() => {
        mounted.store.getState().startNewChatDraft(false)
        replacementAddress = mounted.store.getState().chatDraftAddress
        mounted.store.setState({ historyId: null, serverChatId: null, messages: replacementRows, history: replacementHistory, ...replacementSettings })
        mocks.storeState.messages = replacementRows
        mocks.chatBaseState.history = replacementHistory
        mocks.storeState.selectedModel = replacementSettings.selectedModel
        mocks.storeState.toolChoice = replacementSettings.toolChoice
        mocks.chatBaseState.selectedSystemPrompt = replacementSettings.selectedSystemPrompt
        mocks.chatBaseState.useOCR = replacementSettings.useOCR
        mounted.hook.rerender()
      })
      expect(mounted.hook.result.current.context.getController!()).toBe(ownedController)
      expect(ownedController!.signal.aborted).toBe(false)
      await act(async () => { release(); expect(toChatSubmitResult(await pending)).toEqual({ status: "submitted" }) })
      expect(emittedLate).toBe(true)
      expect(mocks.pageAssistModel).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ model: "original-finish-model", requestScope: mounted.originalScope }))
      expect(h1.wire).toHaveBeenCalledOnce()
      expect(h1.wire.mock.calls[0][1]).toMatchObject({ requestScope: mounted.originalScope })
      expect(mounted.invoke).not.toHaveBeenCalled()
      expect(mocks.getActorScene).toHaveBeenCalledOnce()
      if (mode === "website") {
        expect(saveHistory).toHaveBeenCalledExactlyOnceWith(mounted.input.message, false, "web-ui", undefined, undefined, mounted.originalScope)
      } else expect(saveHistory).toHaveBeenCalledExactlyOnceWith(`Original ${mode} history`, false, "copilot", undefined, undefined, mounted.originalScope)
      const histories = [...mounted.memory.chatHistories.rows.values()]
      expect(histories).toHaveLength(1)
      const { serverChatMirrorOwnerKey } = await import("@/db/dexie/server-chat-mirror")
      expect(histories[0].server_scope_key).toBe(serverChatMirrorOwnerKey({ requestScope: mounted.originalScope }))
      const rows = await mounted.memory.messages.where("history_id").equals(histories[0].id).toArray()
      expect(rows.map((row: StoredMessage) => [row.role, row.content])).toEqual([["user", mounted.input.message], ["assistant", "Partial original final answer"]])
      expect(mounted.store.getState()).toMatchObject({ messages: replacementRows, history: replacementHistory, historyId: null, serverChatId: null, ...replacementSettings })
      expect(mounted.store.getState().chatDraftAddress).toBe(replacementAddress!)
      expect(mocks.storeState.messages).toEqual(replacementRows)
      expect(mocks.chatBaseState.history).toEqual(replacementHistory)
      expect(mocks.storeState.selectedModel).toBe(replacementSettings.selectedModel)
      expect(mocks.chatBaseState.selectedSystemPrompt).toBe(replacementSettings.selectedSystemPrompt)
      expect(mocks.chatBaseState.useOCR).toBe(replacementSettings.useOCR)
      const retainedScenes = (await mounted.sceneBucket.list()).records
      if (mode === "website") {
        expect(retainedScenes).toEqual(acceptedWebsiteScenes)
        expect(retainedScenes[mounted.sourceTarget.key]).toEqual(mounted.originalScenes.records[mounted.sourceTarget.key])
        expect(retainedScenes[acceptedWebsiteSceneKey!]?.value).toEqual(mounted.originalScene)
      } else expect(retainedScenes).toEqual(mounted.originalScenes.records)
      expect(syncSettings).not.toHaveBeenCalled()
      expect(mocks.chatBaseState.streaming).toBe(false)
      expect(mocks.chatBaseState.isProcessing).toBe(false)
      expect(mounted.store.getState().streaming).toBe(false)
      expect(mounted.store.getState().isProcessing).toBe(false)
      expect(mounted.hook.result.current.context.getController!()).toBeNull()
      expect(mounted.hook.result.current.context.embeddingController).toBeNull()
      expect(mocks.notification.error).not.toHaveBeenCalled()
      expect(mocks.notification.warning).not.toHaveBeenCalled()
    } finally {
      release()
      try { await pending } finally {
        saveHistory.mockRestore(); title.mockRestore(); syncSettings.mockRestore()
        mounted.cleanup()
      }
    }
  })

})

describe("compact submission outcomes", () => {
  it("returns the normal helper's completed result", async () => {
    const { result } = renderHook(() => useMessage())
    let outcome: unknown
    await act(async () => {
      outcome = await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(outcome).toEqual({ status: "submitted" })
    expect(h1.append).toHaveBeenCalledTimes(2)
  })

  it("returns a failed normal outcome while retaining its accepted input", async () => {
    h1.wire.mockImplementation(async function* () {
      yield {}
      throw new Error("provider refused this request")
    })
    const { result } = renderHook(() => useMessage())
    let outcome: unknown
    await act(async () => {
      outcome = await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(outcome).toMatchObject({ status: "failed" })
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      input_text: "next",
      state: "accepted_unsent"
    })
    expect(h1.controller.followResult).not.toHaveBeenCalled()
  })

  it.each([false, true])(
    "workspace H1: returns submitted after a verified native settlement (late disconnect=%s)",
    async (lateDisconnect) => {
      h1.native = true
    installWorkspaceCharacterFixture()
      mocks.storeState.serverChatCharacterId = "12"
      mocks.storeState.serverChatAssistantKind = "character"
      h1.wire.mockImplementation(async function* (request) {
        yield { tldw_history_admission_v1: nativeAdmission(request) }
        yield {
          tldw_message_id: "owner-result",
          tldw_conversation_id: nativeChatId
        }
        if (lateDisconnect) throw new Error("late disconnect")
      })
      const { result } = renderHook(() => useMessage())
      let outcome: unknown
      await act(async () => {
        outcome = await result.current.onSubmit({ message: "next", image: "" })
      })
      expect(outcome).toEqual({ status: "submitted" })
      expect(h1.dismiss).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        expect.any(String),
        "completed"
      )
    }
  )

  it("workspace H1: returns failed when native admission has no settlement, keeping the recovery receipt", async () => {
    h1.native = true
    installWorkspaceCharacterFixture()
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    h1.wire.mockImplementation(async function* (request) {
      yield { tldw_history_admission_v1: nativeAdmission(request) }
    })
    const { result } = renderHook(() => useMessage())
    let outcome: unknown
    await act(async () => {
      outcome = await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(outcome).toMatchObject({
      status: "failed",
      errorMessage: "native_history_persistence_unacknowledged"
    })
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      state: "accepted_unsent",
      input_text: "next"
    })
    expect(h1.dismiss).not.toHaveBeenCalled()
  })

  it("returns an explicit skipped outcome when model selection blocks preparation", async () => {
    mocks.storeState.selectedModel = ""
    const { result } = renderHook(() => useMessage())
    let outcome: unknown
    await act(async () => {
      outcome = await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(outcome).toMatchObject({ status: "skipped" })
    expect(h1.wire).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  })

  it("returns skipped for an unsupported native persona before dispatch", async () => {
    h1.persona = true
    const { result } = renderHook(() => useMessage())
    let outcome: unknown
    await act(async () => {
      outcome = await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(outcome).toMatchObject({
      status: "skipped",
      reason: "native_history_persona_unsupported"
    })
    expect(h1.wire).not.toHaveBeenCalled()
  })
})

it.each(["default", "alternate_index"] as const)("actual compact submit binds the %s opening without adding or generating a turn", async strategy => {
  const actual = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  actual.useStoreMessageOption.getState().startNewChatDraft(false)
  actual.useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  bridgeDraftStore(actual.useStoreMessageOption)
  const { useHistorySelection } = await import("@/hooks/chat/useHistorySelection")
  const { memory } = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory.forkOperations.rows.clear(); memory.historySelections.rows.clear()
  const onCapture = vi.fn()
  const selection = renderHook(() => useHistorySelection({ onCapture }))
  h1.controller = { ...selection.result.current, draftAccount: makeDraftAccount() }
  h1.native = true
  mocks.chatBaseState.historyId = null
  mocks.storeState.serverChatId = null
  h1.create.mockImplementation(createdNativeChat)
  h1.capture.mockImplementation(async (id, request) => {
    const result = captureFor({ ...request.view, owner_key: "native-key" })
    const node = { id: "saved-root", revision: "root-revision", role: "assistant", parent_id: null, settled: true }
    const rows = request.view.cursor.kind === "empty" ? [] : [node]
    return { ...result, purpose: request.purpose, rows,
      selected_content: rows.map(row => ({ id: row.id, revision: row.revision, message: "Raw {{user}} meets {{char}}", images: [] })),
      snapshot: { ...result.snapshot, conversation_id: id, nodes: [node] } }
  })
  const hook = renderHook(() => useMessage())
  const seed = strategy === "default" ? { greetingStrategy: strategy, rawText: "Raw {{user}} meets {{char}}" } :
    { greetingStrategy: strategy, alternateIndex: 0, rawText: "Raw {{user}} meets {{char}}" }
  let result: unknown
  await act(async () => { result = await hook.result.current.onSubmit({ message: "", image: "", greetingSeed: seed, greetingIsCurrent: () => true }) })
  expect(result).toEqual({ status: "submitted" })
  expect(h1.create).toHaveBeenCalledOnce()
  expect(h1.create.mock.calls[0][1]).toMatchObject({ seedFirstMessage: true, greetingStrategy: strategy, ...(strategy === "alternate_index" ? { alternateIndex: 0 } : {}) })
  expect(onCapture).toHaveBeenCalledOnce()
  expect(actual.useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(h1.live).not.toHaveBeenCalled()
  expect(mocks.finalModel.stream).not.toHaveBeenCalled()
  hook.unmount(); selection.unmount()
})

describe("side-panel send gate (XP-08)", () => {
  const withGate = (gate: SidepanelSendGate) => {
    const ref = { current: gate }
    return ({ children }: { children: React.ReactNode }) => (
      <SidepanelSendGateContext.Provider value={ref}>
        {children}
      </SidepanelSendGateContext.Provider>
    )
  }

  it("holds a send the gate refuses before anything reaches the server", async () => {
    const gate = vi.fn<SidepanelSendGate>(async () => ({ proceed: false, refreshed: false }))
    localMutationController()
    const current = {}
    mocks.storeState.detachedHistoryResponse = current
    const { result } = renderHook(() => useMessage(), { wrapper: withGate(gate) })
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(mocks.storeState.detachedHistoryResponse).toBe(current)
    expect(gate).toHaveBeenCalledWith({ message: "next" })
    expect(h1.capture).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
  })

  it("native admission preserves its selected point instead of invoking the legacy auto-refresh gate", async () => {
    // The server gained a newer answer (a2); the gate moves the tab onto it.
    const gate = vi.fn<SidepanelSendGate>(async () => {
      const current = h1.controller.getCurrent()
      current.view = { ...current.view, cursor: { kind: "after_message", message_id: "a2" } }
      current.capture = captureFor(current.view)
      return { proceed: true, refreshed: true }
    })
    const { result } = renderHook(() => useMessage(), { wrapper: withGate(gate) })
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(gate).not.toHaveBeenCalled()
    const user = h1.append.mock.calls[0][1]
    expect(
      user.tldw_history_selection_v1.messages.map((row: any) => row.id)
    ).toEqual(["u-old", "a1"])
    expect(user.tldw_history_branch).toBe(false)
  })

  it("does not ask the gate before regenerating an answer", async () => {
    await startFixtureDraft()
    const gate = vi.fn<SidepanelSendGate>(async () => ({ proceed: false, refreshed: false }))
    const current = h1.controller.getCurrent()
    current.status = "idle"
    current.owner = null
    current.view = null
    mocks.storeState.serverChatId = null
    const { result } = renderHook(() => useMessage(), { wrapper: withGate(gate) })
    await act(async () => {
      await result.current.onSubmit({ message: "again", image: "", isRegenerate: true })
    })
    expect(gate).not.toHaveBeenCalled()
  })
})


it("workspace H1: keeps a failed actual scene read ahead of compact native admission and shows recovery guidance", async () => {
  const { Storage } = await import("@plasmohq/storage")
  const service = await vi.importActual<
    typeof import("@/services/actor-settings")
  >("@/services/actor-settings")
  mocks.getActorScene.mockImplementationOnce(service.getActorSceneForChat)
  installWorkspaceCharacterFixture()
  const target = captureCurrentActorSceneTarget(h1.controller)
  expect(target).not.toBeNull()
  const read = vi
    .spyOn(Storage.prototype, "get")
    .mockRejectedValueOnce(new Error("scene read unavailable"))
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  const { result } = renderHook(() => useMessage())
  let outcome: unknown
  await act(async () => {
    outcome = await result.current.onSubmit({
      message: "Keep this exact input",
      image: ""
    })
  })
  expect(outcome).toMatchObject({
    status: "failed",
    errorMessage: "actor_settings_unavailable"
  })
  expect(mocks.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description:
        "Scene settings could not be loaded. Existing chat setup was left unchanged."
    })
  )
  expect(read).toHaveBeenCalledWith(`actorSettings:owned:${target!.key}`)
  expect(h1.create).not.toHaveBeenCalled()
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mocks.storeState.setServerChatId).not.toHaveBeenCalled()
  read.mockRestore()
})


it.each(["history-epoch", "new-draft", "same-id-restore"])(
  "a retained compact idle edit refuses %s before another render",
  async (retirement) => {
    localMutationController()
    const helpers = await import("@/db/dexie/helpers")
    const sessions = await import("@/store/playground-session")
    const state = h1.controller.getCurrent()
    state.status = "idle"
    state.owner = null
    state.view = null
    let epoch = 0
    h1.controller.fence = () => {
      const token = epoch
      return () => epoch === token
    }
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = [
      {
        id: "idle",
        name: "You",
        isBot: false,
        message: "Original compact row",
        sources: []
      }
    ]
    const hook = renderHook(() => useMessage())
    const retained = hook.result.current.editMessage
    mocks.setMessages.mockClear()
    mocks.setHistory.mockClear()
    if (retirement === "history-epoch") epoch++
    else if (retirement === "new-draft")
      mocks.storeState.chatDraftRevision =
        Number(mocks.storeState.chatDraftRevision ?? 0) + 1
    else
      sessions.usePlaygroundSessionStore.setState({
        restoreRevision:
          sessions.usePlaygroundSessionStore.getState().restoreRevision + 1
      })
    expect(await retained(0, "Obsolete compact edit", true, false)).toEqual({
      status: "skipped",
      reason: "stale_selection"
    })
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
  }
)


it.each(
  ["normal", "character"].flatMap((mode) =>
    ["current", "newer edit", "interrupted"].map((boundary) => ({
      mode,
      boundary
    }))
  )
)(
  "compact $mode edit-resend preserves originals through $boundary",
  async ({ mode, boundary }) => {
    h1.native = mode === "character"
    mocks.storeState.messages = [
      {
        id: "u-old",
        name: "You",
        isBot: false,
        message: "old question",
        sources: []
      },
      {
        id: "a1",
        name: "Assistant",
        isBot: true,
        message: "same answer",
        sources: []
      }
    ]
    const original = structuredClone(mocks.storeState.messages)
    const origin = h1.controller.getCurrent().view
    let editorCurrent = true
    const isCurrent = () => editorCurrent
    h1.controller.followResult.mockImplementation(
      async (_origin: unknown, _id: unknown, current?: () => boolean) =>
        current?.() !== false
    )
    h1.wire.mockImplementation(async function* (request) {
      expect(mocks.storeState.detachedHistoryResponse).toBeTruthy()
      if (mode === "character") {
        const selection = request.tldw_history_selection_v1
        yield {
          tldw_history_admission_v1: {
            version: 1,
            owner_key: origin.owner_key,
            conversation_id: nativeChatId,
            input_message_id: "edited-input",
            input_message_revision: "input-r",
            selection_digest: selection.selection_digest,
            messages: selection.messages,
            originating_selection_revision: selection.selection_revision
          }
        }
      }
      yield { choices: [{ delta: { content: "Edited answer" } }] }
      if (boundary === "interrupted") throw new Error("disconnected")
      if (boundary !== "current") editorCurrent = false
      if (mode === "character")
        yield {
          tldw_message_id: "edited-answer",
          tldw_conversation_id: nativeChatId
        }
    })
    const hook = renderHook(() => useMessage())
    mocks.setMessages.mockClear()
    mocks.setHistory.mockClear()
    let outcome: unknown
    await act(async () => {
      outcome = await hook.result.current.editMessage(
        0,
        "Edited question  \n",
        true,
        true,
        isCurrent
      )
    })
    expect(mocks.storeState.detachedHistoryResponse).toBeNull()
    expect(h1.wire).toHaveBeenCalledOnce()
    const branch =
      mode === "character"
        ? h1.wire.mock.calls[0][0]
        : h1.append.mock.calls[0][1]
    expect(branch).toMatchObject({
      tldw_history_branch: true,
      tldw_history_selection_v1: {
        cursor: { kind: "before_message", message_id: "u-old" },
        messages: [],
        selection_revision: origin.selection_revision + 1
      }
    })
    expect(mocks.storeState.messages).toEqual(original)
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(h1.controller.getCurrent().view).toBe(origin)
    if (boundary === "current") {
      expect(outcome).toEqual({ status: "submitted" })
      expect(h1.controller.followResult).toHaveBeenCalledWith(
        origin,
        expect.any(String),
        expect.any(Function)
      )
      expect(h1.controller.followResult.mock.calls[0][2]()).toBe(true)
      expect(h1.dismiss).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        expect.anything(),
        "completed"
      )
    } else {
      expect(h1.dismiss).not.toHaveBeenCalled()
      expect(h1.recover).toHaveBeenLastCalledWith(
        expect.anything(),
        expect.anything(),
        expect.objectContaining({
          input_text: "Edited question  \n",
          result_text: "Edited answer",
          state: "generated_unsaved",
          ...(boundary === "interrupted"
            ? {}
            : { outcome: "complete", settled_message_id: expect.any(String) })
        })
      )
    }
  }
)


it.each([false, true])(
  "a rejected compact callback edited=%s cannot replace a held response marker",
  async (edited) => {
    const current = {}
    mocks.storeState.detachedHistoryResponse = current
    const hook = renderHook(() => useMessage())
    await act(async () => {
      await expect(
        hook.result.current.onSubmit({
          message: "Rejected callback",
          image: "",
          ...(edited ? { historyEditResend: {} as any } : {}),
          assertCurrent: () => {
            throw new Error("retired source")
          }
        })
      ).rejects.toThrow("retired source")
    })
    expect(mocks.storeState.detachedHistoryResponse).toBe(current)
    expect(mocks.setStreaming).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
  }
)


it.each([false, true])(
  "held compact character edit survives a rejected second callback edited=%s",
  async (edited) => {
    h1.native = true
    mocks.storeState.messages = [
      {
        id: "u-old",
        isBot: false,
        name: "You",
        message: "old question",
        sources: []
      },
      {
        id: "a1",
        isBot: true,
        name: "Assistant",
        message: "same answer",
        sources: []
      }
    ]
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    h1.wire.mockImplementation(async function* (request) {
      await gate
      const selection = request.tldw_history_selection_v1
      yield {
        tldw_history_admission_v1: {
          version: 1,
          owner_key: h1.controller.getCurrent().view.owner_key,
          conversation_id: nativeChatId,
          input_message_id: "edited-input",
          input_message_revision: "input-r",
          selection_digest: selection.selection_digest,
          messages: selection.messages,
          originating_selection_revision: selection.selection_revision
        }
      }
      yield { choices: [{ delta: { content: "Edited answer" } }] }
      yield {
        tldw_message_id: "edited-answer",
        tldw_conversation_id: nativeChatId
      }
    })
    const hook = renderHook(() => useMessage())
    let pending!: ReturnType<typeof hook.result.current.editMessage>
    try {
      act(() => {
        pending = hook.result.current.editMessage(
          0,
          "Edited question",
          true,
          true,
          () => true
        )
      })
      await waitFor(() => expect(h1.wire).toHaveBeenCalledOnce())
      const marker = mocks.storeState.detachedHistoryResponse
      const controllerCalls = mocks.setAbortController.mock.calls.length
      expect(marker).toBeTruthy()
      await act(async () => {
        await expect(
          hook.result.current.onSubmit({
            message: "Rejected",
            image: "",
            ...(edited ? { historyEditResend: {} as any } : {}),
            assertCurrent: () => {
              throw new Error("retired source")
            }
          })
        ).rejects.toThrow("retired source")
      })
      expect(mocks.storeState.detachedHistoryResponse).toBe(marker)
      expect(mocks.setAbortController).toHaveBeenCalledTimes(controllerCalls)
      expect(h1.wire).toHaveBeenCalledOnce()
    } finally {
      release()
      await act(async () => {
        await pending
      })
    }
    expect(mocks.storeState.detachedHistoryResponse).toBeNull()
  }
)


const mountNativePlainSave = async (
  role: "user" | "assistant",
  character = false
) => {
  const { useHistorySelection } = await vi.importActual<
    typeof import("@/hooks/chat/useHistorySelection")
  >("@/hooks/chat/useHistorySelection")
  const { formatSelectedHistory } =
    await vi.importActual<typeof import("@/db/dexie/helpers")>(
      "@/db/dexie/helpers"
    )
  const { useStoreMessageOption: draftStore } =
    await vi.importActual<typeof import("@/store/option")>("@/store/option")
  bridgeDraftStore(draftStore)
  const messageId = role === "user" ? "u-old" : "a1"
  const server = { version: 1, text: "Original {{char}}  \n" }
  h1.capture.mockImplementation(async (_id, request) => {
    const captured = captureFor({ ...request.view, owner_key: "native-key" })
    captured.snapshot.nodes.forEach((node) => {
      node.revision =
        node.id === messageId
          ? `opaque-target-${server.version}`
          : "opaque-other"
      if (node.id === messageId)
        Object.assign(node, {
          metadata: [
            { id: messageId, kind: "message_metadata", revision: "preserved" }
          ]
        })
    })
    captured.selected_content.forEach((row) => {
      row.revision =
        row.id === messageId
          ? `opaque-target-${server.version}`
          : "opaque-other"
      if (row.id === messageId)
        Object.assign(row, {
          message: server.text,
          extra_metadata: {
            sender_role: role,
            pinned: true,
            nested: { value: 1 }
          }
        })
    })
    return captured
  })
  const publish = vi.fn((capture: any) => {
    const projection = formatSelectedHistory(capture)
    draftStore.setState(projection)
    mocks.storeState.messages = projection.messages
    mocks.chatBaseState.history = projection.history
  })
  const selection = renderHook(() =>
    useHistorySelection({ onCapture: publish })
  )
  await act(async () => {
    expect(
      await selection.result.current.loadConversation(
        { serverChatId: nativeChatId },
        undefined,
        receipt => installFixtureReceipt(receipt, draftStore, mocks.chatBaseState.historyId)
      )
    ).toBe(true)
  })
  h1.controller = { ...selection.result.current, draftAccount: makeDraftAccount() }
  await act(async () => {
    expect(
      await selection.result.current.choose({
        kind: "after_message",
        message_id: "a1"
      })
    ).toBe(true)
  })
  const origin = selection.result.current.getCurrent().capture!
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  const read = vi.mocked(api.getMessage),
    write = vi.mocked(api.editMessage)
  const wire = () => ({
    id: messageId,
    conversation_id: nativeChatId,
    parent_message_id: role === "user" ? null : "u-old",
    sender: role,
    content: server.text,
    has_image: false,
    version: server.version
  })
  read.mockReset().mockImplementation(async () => wire())
  write.mockReset().mockImplementation(async (_id, text) => {
    server.version++
    server.text = text
    return { ...wire(), content: text.replace("{{user}}", "Alice") }
  })
  mocks.storeState.serverChatId = nativeChatId
  mocks.storeState.serverChatCharacterId = character ? "12" : null
  mocks.storeState.serverChatAssistantKind = character ? "character" : null
  const hook = renderHook(() => useMessage())
  mocks.setMessages.mockClear()
  mocks.setHistory.mockClear()
  const index = role === "user" ? 0 : 1
  publish.mockClear()
  return {
    hook,
    selection,
    publish,
    origin,
    owner: selection.result.current.getCurrent().owner!,
    server,
    read,
    write,
    index,
    role,
    draftStore,
    wire,
    cleanup: () => {
      hook.unmount()
      selection.unmount()
    }
  }
}

it.each([
  { role: "user" as const, character: false },
  { role: "assistant" as const, character: false },
  { role: "user" as const, character: true },
  { role: "assistant" as const, character: true }
])(
  "native plain Save $role character=$character acknowledges only canonical raw text",
  async ({ role, character }) => {
    const mounted = await mountNativePlainSave(role, character)
    const helpers = await import("@/db/dexie/helpers")
    const original = structuredClone(mounted.origin)
    let outcome: unknown
    try {
      await act(async () => {
        outcome = await mounted.hook.result.current.editMessage(
          mounted.index,
          "Edited {{user}}  \n",
          role === "user",
          false,
          () => true
        )
      })
      expect(outcome).toEqual({ status: "submitted" })
      expect(mounted.write).toHaveBeenCalledExactlyOnceWith(
        role === "user" ? "u-old" : "a1",
        "Edited {{user}}  \n",
        1,
        nativeChatId,
        expect.objectContaining({
          requestScope: mounted.owner.request_scope,
          scope: mounted.owner.scope,
          signal: expect.any(AbortSignal)
        })
      )
      expect(
        mounted.selection.result.current.getCurrent().view?.cursor
      ).toEqual(original.view.cursor)
      const capture = mounted.selection.result.current.getCurrent().capture
      expect(capture?.status).toBe("captured")
      if (capture?.status !== "captured")
        throw new Error("missing canonical capture")
      expect(capture.selected_content[mounted.index]).toMatchObject({
        message: "Edited {{user}}  \n",
        revision: "opaque-target-2",
        extra_metadata: original.selected_content[mounted.index].extra_metadata
      })
      expect(
        capture.snapshot.nodes.map((row) => ({
          id: row.id,
          parent: row.parent_id
        }))
      ).toEqual(
        original.snapshot.nodes.map((row) => ({
          id: row.id,
          parent: row.parent_id
        }))
      )
      expect(mounted.publish).toHaveBeenCalledOnce()
      expect(helpers.updateMessageById).not.toHaveBeenCalled()
      expect(h1.wire).not.toHaveBeenCalled()
      expect(mocks.setMessages).not.toHaveBeenCalled()
      expect(mocks.setHistory).not.toHaveBeenCalled()
    } finally {
      mounted.cleanup()
    }
  }
)

it.each(
  ["read", "write", "refresh", "final-read"].flatMap((stage) =>
    [
      "editor",
      "new-draft",
      "same-id-epoch",
      "account",
      "streaming",
      "isProcessing"
    ].map((boundary) => ({ stage, boundary }))
  )
)(
  "native plain Save retains the current source after $boundary during held $stage",
  async ({ stage, boundary }) => {
    const mounted = await mountNativePlainSave("user")
    let current = true,
      reached = false,
      release!: () => void
    const held = new Promise<void>((accept) => {
      release = accept
    })
    if (stage === "final-read") {
      const original = mounted.read.getMockImplementation()!
      mounted.read.mockImplementation(async (...args: any[]) => {
        if (mounted.server.version === 2 && !reached) {
          reached = true
          await held
        }
        return (original as any)(...args)
      })
    } else if (stage !== "refresh") {
      const boundary = stage === "read" ? mounted.read : mounted.write
      const original = boundary.getMockImplementation()!
      boundary.mockImplementationOnce(async (...args: any[]) => {
        reached = true
        await held
        return (original as any)(...args)
      })
    } else {
      const original = h1.capture.getMockImplementation()!
      h1.capture.mockImplementation(async (...args) => {
        if (mounted.server.version === 2 && !reached) {
          reached = true
          await held
        }
        return original(...args)
      })
    }
    let pending!: Promise<unknown>
    try {
      act(() => {
        pending = mounted.hook.result.current.editMessage(
          0,
          "Kept authored edit",
          true,
          false,
          () => current
        )
        void pending.catch(() => {})
      })
      await waitFor(() => expect(reached).toBe(true))
      if (boundary === "editor") current = false
      else if (boundary === "streaming" || boundary === "isProcessing")
        Object.assign(mocks.chatBaseState, { [boundary]: true })
      else if (boundary === "new-draft")
        mounted.draftStore.getState().startNewChatDraft()
      else if (boundary === "same-id-epoch")
        act(() => {
          mounted.selection.result.current.beginLoad()
        })
      else
        act(() => {
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        })
      const replacement = mounted.selection.result.current.getCurrent().capture
      let outcome: unknown
      await act(async () => {
        release()
        outcome = await pending
      })
      expect(outcome).toMatchObject({ status: "skipped" })
      expect(mounted.write).toHaveBeenCalledTimes(stage === "read" ? 0 : 1)
      expect(mounted.publish).not.toHaveBeenCalled()
      expect(mounted.selection.result.current.getCurrent().capture).toBe(
        replacement
      )
    } finally {
      release?.()
      await pending?.catch(() => {})
      Object.assign(mocks.chatBaseState, {
        streaming: false,
        isProcessing: false
      })
      mounted.cleanup()
    }
  }
)


it.each(["streaming", "isProcessing"])(
  "an open native editor refuses Save during live %s",
  async (flag) => {
    const mounted = await mountNativePlainSave("user")
    const signal = mounted.selection.result.current.getSignal()
    try {
      Object.assign(mocks.chatBaseState, { [flag]: true })
      let outcome: unknown
      await act(async () => {
        outcome = await mounted.hook.result.current.editMessage(
          0,
          "Held authored text",
          true,
          false,
          () => true
        )
      })
      expect(outcome).toMatchObject({ status: "skipped" })
      expect(mounted.read).not.toHaveBeenCalled()
      expect(mounted.write).not.toHaveBeenCalled()
      expect(mounted.publish).not.toHaveBeenCalled()
      expect(mounted.selection.result.current.getSignal()).toBe(signal)
      expect(signal?.aborted).toBe(false)
    } finally {
      Object.assign(mocks.chatBaseState, {
        streaming: false,
        isProcessing: false
      })
      mounted.cleanup()
    }
  }
)


it.each(["read", "write", "refresh"])(
  "the mounted native editor keeps newer text saveable after typing during %s",
  async (stage) => {
    const mounted = await mountNativePlainSave("user")
    const { EditMessageForm } = await import(
      "@/components/Common/Playground/EditMessageForm"
    )
    const close = vi.fn()
    function Editor() {
      const rows = mounted.draftStore((state) => state.messages)
      const actions = useMessage()
      return (
        <EditMessageForm
          value={rows[0].message}
          isBot={false}
          onClose={close}
          onSumbit={(value, send, current, origin) =>
            (actions.editMessage as any)(0, value, true, send, current, origin)
          }
        />
      )
    }
    let reached = false,
      release!: () => void
    const held = new Promise<void>((accept) => {
      release = accept
    })
    if (stage !== "refresh") {
      const boundary = stage === "read" ? mounted.read : mounted.write
      const original = boundary.getMockImplementation()!
      boundary.mockImplementationOnce(async (...args: any[]) => {
        reached = true
        await held
        return (original as any)(...args)
      })
    } else {
      const original = h1.capture.getMockImplementation()!
      h1.capture.mockImplementation(async (...args) => {
        if (mounted.server.version === 2 && !reached) {
          reached = true
          await held
        }
        return original(...args)
      })
    }
    const view = render(<Editor />)
    try {
      const textarea = screen.getByRole("textbox")
      fireEvent.change(textarea, { target: { value: "First {{user}}  \n" } })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(reached).toBe(true))
      fireEvent.change(textarea, {
        target: { value: "Newer authored edit  \n" }
      })
      await act(async () => {
        release()
        await held
      })
      await waitFor(() =>
        expect(screen.getByRole("button", { name: /^save$/i })).toBeEnabled()
      )
      expect(close).not.toHaveBeenCalled()
      expect(textarea).toHaveValue("Newer authored edit  \n")
      expect(mounted.write).toHaveBeenCalledTimes(stage === "read" ? 0 : 1)
      if (stage !== "read")
        expect(mounted.draftStore.getState().messages[0].message).toBe(
          "First {{user}}  \n"
        )
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(close).toHaveBeenCalledOnce())
      expect(
        mounted.write.mock.calls.map(([, text, version]) => [text, version])
      ).toEqual(
        stage === "read"
          ? [["Newer authored edit  \n", 1]]
          : [
              ["First {{user}}  \n", 1],
              ["Newer authored edit  \n", 2]
            ]
      )
    } finally {
      release?.()
      view.unmount()
      mounted.cleanup()
    }
  }
)


it.each(["write", "refresh", "final-read"])(
  "Cancel retires the actual native editor during %s",
  async (stage) => {
    const mounted = await mountNativePlainSave("user")
    const { EditMessageForm } = await import(
      "@/components/Common/Playground/EditMessageForm"
    )
    const close = vi.fn()
    function Editor() {
      const rows = mounted.draftStore((state) => state.messages)
      const actions = useMessage()
      return (
        <EditMessageForm
          value={rows[0].message}
          isBot={false}
          onClose={close}
          onSumbit={(value, send, current, origin) =>
            actions.editMessage(0, value, true, send, current, origin)
          }
        />
      )
    }
    let reached = false,
      release!: () => void
    const held = new Promise<void>((accept) => {
      release = accept
    })
    if (stage === "write") {
      const original = mounted.write.getMockImplementation()!
      mounted.write.mockImplementationOnce(async (...args: any[]) => {
        reached = true
        await held
        return (original as any)(...args)
      })
    } else if (stage === "final-read") {
      const original = mounted.read.getMockImplementation()!
      mounted.read.mockImplementation(async (...args: any[]) => {
        if (mounted.server.version === 2 && !reached) {
          reached = true
          await held
        }
        return (original as any)(...args)
      })
    } else {
      const original = h1.capture.getMockImplementation()!
      h1.capture.mockImplementation(async (...args) => {
        if (mounted.server.version === 2 && !reached) {
          reached = true
          await held
        }
        return original(...args)
      })
    }
    const view = render(<Editor />)
    try {
      fireEvent.change(screen.getByRole("textbox"), {
        target: { value: "Cancelled authored edit" }
      })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(reached).toBe(true))
      fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }))
      await act(async () => {
        release()
        await held
      })
      await waitFor(() =>
        expect(screen.getByRole("button", { name: /^save$/i })).toBeEnabled()
      )
      expect(close).toHaveBeenCalledOnce()
      expect(mounted.publish).not.toHaveBeenCalled()
      expect(mounted.selection.result.current.getCurrent().capture).toBe(
        mounted.origin
      )
    } finally {
      release?.()
      view.unmount()
      mounted.cleanup()
    }
  }
)


it.each(["normal", "character"].flatMap(mode => ["current", "interrupted", "old result", "wrong admission"].map(boundary => ({ mode, boundary }))))(
  "compact native owned regeneration through $mode / $boundary keeps the canonical input and original reply",
  async ({ mode, boundary }) => {
    h1.native = mode === "character"
    mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
      { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
    const original = structuredClone(mocks.storeState.messages)
    const hook = renderHook(() => useMessage())
    mocks.setMessages.mockClear(); mocks.setHistory.mockClear()
    const origin = h1.controller.getCurrent().view
    const admission = { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
      input_message_id: "u-old", input_message_revision: "1", selection_digest: "ORIGINAL INPUT ADMISSION",
      originating_selection_revision: 0, messages: [] }
    h1.read.mockResolvedValue({ id: "u-old", conversation_id: nativeChatId, sender: "user", parent_message_id: null,
      content: "old question", has_image: false, version: 1 })
    h1.wire.mockImplementation(async function* () {
      expect(mocks.storeState.detachedHistoryResponse).toBeTruthy()
      yield { tldw_history_admission_v1: { ...admission, ...(boundary === "wrong admission" ? { input_message_id: "foreign" } : {}) } }
      yield { choices: [{ delta: { content: "New alternative reply" } }] }
      if (boundary === "interrupted") throw new Error("Response lost")
      yield { tldw_message_id: boundary === "old result" ? "a2" : "new-alternative", tldw_conversation_id: nativeChatId }
    })
    let outcome: unknown
    await act(async () => { outcome = await hook.result.current.regenerateLastMessage() })
    expect(h1.wire).toHaveBeenCalledOnce()
    expect(h1.read).toHaveBeenCalledWith("u-old", expect.objectContaining({ requestScope: expect.any(Object) }))
    const request = h1.wire.mock.calls[0][0]
    expect(request).toMatchObject({ save_to_db: true, tldw_history_branch: true,
      metadata: { tldw_regenerate_from_message_id: "a1" }, messages: [{ role: "user", content: "old question" }],
      tldw_history_selection_v1: { cursor: { kind: "after_message", message_id: "a1" }, selection_revision: origin.selection_revision + 1,
        messages: [{ id: "u-old", revision: "r1" }, { id: "a1", revision: "r2" }] } })
    expect(mocks.storeState.messages).toEqual(original)
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(mocks.storeState.detachedHistoryResponse).toBeNull()
    expect(h1.append).not.toHaveBeenCalled()
    expect(h1.create).not.toHaveBeenCalled()
    expect(mocks.finalModel.stream).not.toHaveBeenCalled()
    if (boundary === "current") {
      expect(outcome).toEqual({ status: "submitted" })
      expect(h1.controller.followResult).toHaveBeenCalledWith(origin, "new-alternative", expect.any(Function))
      expect(h1.dismiss).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), "completed")
      expect(h1.recover).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.objectContaining({ action: "regenerate",
        admission: { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId, input_message_id: "u-old",
          input_message_revision: "1", selection_digest: "ORIGINAL INPUT ADMISSION" } }))
    } else {
      expect(outcome).toMatchObject({ status: "failed" })
      expect(h1.controller.followResult).not.toHaveBeenCalled()
      expect(h1.dismiss).not.toHaveBeenCalled()
    }
    hook.unmount()
  }
)


it.each(["stream", "model"])("compact native Regenerate acquires live activity once across held %s preparation", async stage => {
  mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
  mocks.setStreaming.mockImplementation(value => { mocks.chatBaseState.streaming = value })
  mocks.setIsProcessing.mockImplementation(value => { mocks.chatBaseState.isProcessing = value })
  let releaseModel!: () => void, releaseStream!: () => void, enter!: () => void
  const modelGate = new Promise<void>(resolve => { releaseModel = resolve }),
    streamGate = new Promise<void>(resolve => { releaseStream = resolve }), entered = new Promise<void>(resolve => { enter = resolve })
  const validation = await import("@/utils/chat-model-validation")
  const modelRead = vi.spyOn(validation, "validateSelectedChatModelAvailability").mockImplementation(async () => {
    if (stage === "model") await modelGate
    return { status: "valid" }
  })
  h1.read.mockResolvedValue({ id: "u-old", sender: "user", conversation_id: nativeChatId, parent_message_id: null,
    content: "old question", has_image: false, version: 1 })
  const origin = h1.controller.getCurrent().view
  h1.wire.mockImplementation(async function* () {
    enter(); await streamGate
    yield { tldw_history_admission_v1: { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
      input_message_id: "u-old", input_message_revision: "1", selection_digest: "original", messages: [], originating_selection_revision: 0 } }
    yield { tldw_message_id: "new-alternative", tldw_conversation_id: nativeChatId }
  })
  const hook = renderHook(() => useMessage())
  let first!: Promise<unknown>, second!: Promise<unknown>
  try {
    act(() => {
      first = hook.result.current.regenerateLastMessage()
      if (stage === "model") second = hook.result.current.regenerateLastMessage()
    })
    releaseModel(); await entered
    expect(mocks.chatBaseState.streaming).toBe(true)
    expect(mocks.chatBaseState.isProcessing).toBe(true)
    const marker = mocks.storeState.detachedHistoryResponse
    if (stage === "stream") second = hook.result.current.regenerateLastMessage()
    expect(await second).toEqual({ status: "skipped", reason: "response_in_progress" })
    expect(mocks.storeState.detachedHistoryResponse).toBe(marker)
    expect(await hook.result.current.editMessage(0, "Obsolete concurrent edit", true, false)).toEqual({ status: "skipped", reason: "response_in_progress" })
    expect(h1.write).not.toHaveBeenCalled()
    expect(h1.wire).toHaveBeenCalledOnce()
    releaseStream(); await act(async () => { expect(await first).toEqual({ status: "submitted" }) })
    expect(mocks.chatBaseState.streaming).toBe(false)
    expect(mocks.chatBaseState.isProcessing).toBe(false)
    expect(mocks.storeState.detachedHistoryResponse).toBeNull()
  } finally {
    releaseModel(); releaseStream(); await first; await second; hook.unmount(); modelRead.mockRestore()
    mocks.chatBaseState.streaming = false; mocks.chatBaseState.isProcessing = false
    mocks.setStreaming.mockReset(); mocks.setIsProcessing.mockReset()
  }
})


it.each(["normal", "character"])("compact $mode native regeneration preserves literal whitespace with every image through final request serialization", async mode => {
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    const decorate = (captured: ReturnType<typeof captureFor>) => ({ ...captured,
      selected_content: captured.selected_content.map(row => row.id === "u-old" ? { ...row, message: "  \n", images: [image, image],
        extra_metadata: { sender_role: "user", image_details: ["high", "low"] } } : row) })
    h1.controller.getCurrent().capture = decorate(h1.controller.getCurrent().capture)
    h1.capture.mockImplementation(async (_id, request) => decorate(captureFor(request.view)))
    h1.native = mode === "character"
    mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "  \n", images: [image, image], sources: [] },
      { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
    const hook = renderHook(() => useMessage())
    h1.read.mockResolvedValue({ id: "u-old", sender: "user", conversation_id: nativeChatId, parent_message_id: null,
      content: "  \n", has_image: true, version: 1 })
    const origin = h1.controller.getCurrent().view
    h1.wire.mockImplementation(async function* (request) {
      expect(request.messages).toEqual([{ role: "user", content: [{ type: "text", text: "  \n" },
        { type: "image_url", image_url: { url: image, detail: "high" } }, { type: "image_url", image_url: { url: image, detail: "low" } }] }])
      yield { tldw_history_admission_v1: { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
        input_message_id: "u-old", input_message_revision: "1", selection_digest: "original", originating_selection_revision: 0, messages: [] } }
      yield { tldw_message_id: "new-alternative", tldw_conversation_id: nativeChatId }
    })
    await act(async () => { expect(await hook.result.current.regenerateLastMessage()).toEqual({ status: "submitted" }) })
    expect(h1.wire).toHaveBeenCalledOnce()
    hook.unmount()
  })


it("compact native regeneration cannot follow an unmounted source even when its outer H1 owner survives", async () => {
  mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
  const hook = renderHook(() => useMessage())
  const origin = h1.controller.getCurrent().view
  h1.read.mockResolvedValue({ id: "u-old", sender: "user", conversation_id: nativeChatId, parent_message_id: null,
    content: "old question", has_image: false, version: 1 })
  let release!: () => void, enter!: () => void
  const gate = new Promise<void>(resolve => { release = resolve }), entered = new Promise<void>(resolve => { enter = resolve })
  h1.wire.mockImplementation(async function* () {
    yield { tldw_history_admission_v1: { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
      input_message_id: "u-old", input_message_revision: "1", selection_digest: "original", originating_selection_revision: 0, messages: [] } }
    enter(); await gate
    yield { tldw_message_id: "unmounted-alternative", tldw_conversation_id: nativeChatId }
  })
  let pending!: Promise<unknown>
  act(() => { pending = hook.result.current.regenerateLastMessage() })
  await entered; hook.unmount(); release()
  expect(await pending).toEqual({ status: "submitted" })
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(h1.recover).toHaveBeenLastCalledWith(expect.anything(), expect.anything(), expect.objectContaining({
    action: "regenerate", outcome: "complete", settled_message_id: "unmounted-alternative" }))
})


it.each(["regenerate", "send"])("compact an older regeneration cannot clear a newer request through another mounted hook: %s", async kind => {
    let sharedController: AbortController | null = null
    mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
      { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
    h1.native = kind === "send"
    mocks.controllerReader = () => sharedController
    mocks.setAbortController.mockImplementation(value => { sharedController = typeof value === "function" ? value(sharedController) : value })
    mocks.setStreaming.mockImplementation(value => { mocks.chatBaseState.streaming = value })
    mocks.setIsProcessing.mockImplementation(value => { mocks.chatBaseState.isProcessing = value })
    const firstHook = renderHook(() => useMessage()), secondHook = renderHook(() => useMessage())
    const origin = h1.controller.getCurrent().view
    h1.read.mockResolvedValue({ id: "u-old", sender: "user", conversation_id: nativeChatId, parent_message_id: null,
      content: "old question", has_image: false, version: 1 })
    let releaseA!: () => void, releaseB!: () => void, enterA!: () => void, enterB!: () => void, dispatched = 0
    const a = new Promise<void>(resolve => { releaseA = resolve }), b = new Promise<void>(resolve => { releaseB = resolve }),
      enteredA = new Promise<void>(resolve => { enterA = resolve }), enteredB = new Promise<void>(resolve => { enterB = resolve })
    h1.wire.mockImplementation(async function* (request) {
      const first = dispatched++ === 0, selection = request.tldw_history_selection_v1
      yield { tldw_history_admission_v1: request.metadata?.tldw_regenerate_from_message_id ? { version: 1,
        owner_key: origin.owner_key, conversation_id: nativeChatId, input_message_id: "u-old", input_message_revision: "1",
        selection_digest: "original", originating_selection_revision: 0, messages: [] } : { version: 1, owner_key: origin.owner_key,
        conversation_id: nativeChatId, input_message_id: "new-input-B", input_message_revision: "1", selection_digest: selection.selection_digest,
        originating_selection_revision: selection.selection_revision, messages: selection.messages } }
      if (first) { enterA(); await a } else { enterB(); await b }
      yield { tldw_message_id: first ? "reply-A" : "reply-B", tldw_conversation_id: nativeChatId }
    })
    let first!: Promise<unknown>, second!: Promise<unknown>
    try {
      act(() => { first = firstHook.result.current.regenerateLastMessage() })
      await enteredA
      mocks.chatBaseState.streaming = false; mocks.chatBaseState.isProcessing = false; mocks.storeState.chatDraftRevision = Number(mocks.storeState.chatDraftRevision ?? 0) + 1
      secondHook.rerender()
      act(() => { second = kind === "send" ? secondHook.result.current.onSubmit({ message: "Next B", image: "" }) : secondHook.result.current.regenerateLastMessage() })
      await enteredB
      const controllerB = sharedController, markerB = mocks.storeState.detachedHistoryResponse
      releaseA(); await act(async () => { await first })
      expect(sharedController).toBe(controllerB)
      expect(mocks.chatBaseState.streaming).toBe(true)
      expect(mocks.chatBaseState.isProcessing).toBe(true)
      expect(mocks.storeState.detachedHistoryResponse).toBe(markerB)
      releaseB(); await act(async () => { expect(await second).toEqual({ status: "submitted" }) })
      expect(mocks.chatBaseState.streaming).toBe(false)
    } finally {
      releaseA(); releaseB(); await first; await second; firstHook.unmount(); secondHook.unmount()
      mocks.chatBaseState.streaming = false; mocks.chatBaseState.isProcessing = false; mocks.setStreaming.mockReset(); mocks.setIsProcessing.mockReset(); mocks.setAbortController.mockReset()
    }
  })


it.each([false, true])(
  "compact ordinary response exposes Stop until its owned stream settles (Temporary=%s)",
  async (temporary) => {
    mocks.useRealPageAssist = true
    mocks.storeState.temporaryChat = temporary
    if (temporary) {
      await startFixtureDraft(true)
      const current = h1.controller.getCurrent()
      current.status = "idle"
      current.owner = null
      current.view = null
      mocks.storeState.serverChatId = null
      mocks.chatBaseState.historyId = "temp"
    }
    mocks.setStreaming.mockImplementation((value) => {
      mocks.chatBaseState.streaming = value
    })
    mocks.setIsProcessing.mockImplementation((value) => {
      mocks.chatBaseState.isProcessing = value
    })
    let release!: () => void, enter!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const entered = new Promise<void>((resolve) => {
      enter = resolve
    })
    h1.wire.mockImplementation(async function* () {
      yield { choices: [{ delta: { content: "partial answer" } }] }
      enter()
      await gate
    })
    const hook = renderHook(
      () => ({ context: usePageAssist(), actions: useMessage() }),
      { wrapper: PageAssistProvider }
    )
    let pending!: Promise<unknown>
    try {
      act(() => {
        pending = hook.result.current.actions.onSubmit({
          message: "Exact compact question",
          image: ""
        })
      })
      await act(async () => {
        await entered
      })
      expect(mocks.chatBaseState.streaming).toBe(true)
      const owned = hook.result.current.context.getController!()!
      act(() => {
        hook.result.current.actions.stopStreamingRequest()
      })
      expect(owned.signal.aborted).toBe(true)
      expect(mocks.chatBaseState.streaming).toBe(true)
      release()
      await act(async () => {
        await pending
      })
      expect(mocks.chatBaseState.streaming).toBe(false)
      expect(mocks.chatBaseState.isProcessing).toBe(false)
      expect(hook.result.current.context.getController!()).toBeNull()
    } finally {
      release()
      await pending
      hook.unmount()
      mocks.chatBaseState.streaming = false
      mocks.chatBaseState.isProcessing = false
      mocks.setStreaming.mockReset()
      mocks.setIsProcessing.mockReset()
      mocks.useRealPageAssist = false
    }
  }
)

it("compact Stop releases native busy state through the actual shared provider after the aborted turn finishes", async () => {
  mocks.useRealPageAssist = true
  mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
  mocks.setStreaming.mockImplementation(value => { mocks.chatBaseState.streaming = value })
  mocks.setIsProcessing.mockImplementation(value => { mocks.chatBaseState.isProcessing = value })
  const hook = renderHook(() => ({ context: usePageAssist(), actions: useMessage() }), { wrapper: PageAssistProvider })
  const origin = h1.controller.getCurrent().view
  h1.read.mockResolvedValue({ id: "u-old", sender: "user", conversation_id: nativeChatId, parent_message_id: null,
    content: "old question", has_image: false, version: 1 })
  let release!: () => void, enter!: () => void
  const gate = new Promise<void>(resolve => { release = resolve }), entered = new Promise<void>(resolve => { enter = resolve })
  h1.wire.mockImplementation(async function* () {
    yield { tldw_history_admission_v1: { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
      input_message_id: "u-old", input_message_revision: "1", selection_digest: "original", originating_selection_revision: 0, messages: [] } }
    enter(); await gate
    throw new DOMException("Stopped", "AbortError")
  })
  const focusRequests = vi.fn()
  window.addEventListener("tldw:focus-composer", focusRequests)
  let pending!: Promise<unknown>
  try {
    act(() => { pending = hook.result.current.actions.regenerateLastMessage() })
    await act(async () => { await entered })
    const owned = hook.result.current.context.getController!()!
    act(() => { hook.result.current.actions.stopStreamingRequest() })
    expect(owned.signal.aborted).toBe(true)
    release(); await act(async () => { await pending })
    expect(hook.result.current.context.getController!()).toBeNull()
    expect(mocks.chatBaseState.streaming).toBe(false)
    expect(mocks.chatBaseState.isProcessing).toBe(false)
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(focusRequests).not.toHaveBeenCalled()
  } finally { window.removeEventListener("tldw:focus-composer", focusRequests); release(); await pending; mocks.chatBaseState.streaming = false; mocks.chatBaseState.isProcessing = false; hook.unmount()
    mocks.setStreaming.mockReset(); mocks.setIsProcessing.mockReset(); mocks.useRealPageAssist = false }
})


it.each(["marker", "busy"])("compact a synchronous replacement during %s publication keeps the newer provider owner", async publication => {
  mocks.useRealPageAssist = true
  mocks.storeState.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }]
  mocks.setStreaming.mockImplementation(value => { mocks.chatBaseState.streaming = value })
  mocks.setIsProcessing.mockImplementation(value => { mocks.chatBaseState.isProcessing = value })
  const hook = renderHook(() => ({ context: usePageAssist(), actions: useMessage() }), { wrapper: PageAssistProvider })
  const newer = new AbortController(), marker = { originalReply: { id: "newer", isBot: true, message: "B" } }
  let replaced = false
  const store = (await import("@/store/option")).useStoreMessageOption
  const originalSetState = store.setState
  const replace = () => {
    if (replaced) return
    replaced = true
    hook.result.current.context.setController(newer)
    mocks.storeState.chatDraftRevision = Number(mocks.storeState.chatDraftRevision ?? 0) + 1
    mocks.chatBaseState.streaming = true; mocks.chatBaseState.isProcessing = true
    mocks.storeState.detachedHistoryResponse = marker
  }
  const spy = vi.spyOn(store, "setState").mockImplementation(update => {
    originalSetState(update)
    if (publication === "marker" && mocks.storeState.detachedHistoryResponse) replace()
  })
  mocks.setIsProcessing.mockImplementation(value => {
    mocks.chatBaseState.isProcessing = value
    if (publication === "busy" && value) replace()
  })
  try {
    await act(async () => { await hook.result.current.actions.regenerateLastMessage().catch(() => undefined) })
    expect(replaced).toBe(true)
    expect(hook.result.current.context.getController!()).toBe(newer)
    expect(mocks.chatBaseState.streaming).toBe(true)
    expect(mocks.chatBaseState.isProcessing).toBe(true)
    expect(mocks.storeState.detachedHistoryResponse).toBe(marker)
    expect(h1.wire).not.toHaveBeenCalled()
  } finally { spy.mockRestore(); mocks.chatBaseState.streaming = false; mocks.chatBaseState.isProcessing = false; hook.unmount()
    mocks.setStreaming.mockReset(); mocks.setIsProcessing.mockReset(); mocks.useRealPageAssist = false }
})


const mountNativeLeafDelete = async (character = false, mirror = false) => {
  // The native selector clears historyId unless an owned device mirror is linked.
  if (!mirror) mocks.chatBaseState.historyId = null
  const mounted = await mountNativePlainSave("assistant", character)
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  let deleted = false
  const capture = h1.capture.getMockImplementation()!
  h1.capture.mockImplementation(async (...args) => {
    const result = (await capture(...args)) as ReturnType<typeof captureFor>
    if (deleted) {
      result.snapshot.nodes = result.snapshot.nodes.filter(
        (row) => row.id !== "a1"
      )
      result.rows = result.rows.filter((row) => row.id !== "a1")
      result.selected_content = result.selected_content.filter(
        (row) => row.id !== "a1"
      )
    }
    return result
  })
  const remove = vi.mocked(api.deleteMessage)
  remove.mockReset().mockImplementation(async () => {
    deleted = true
  })
  if (mounted.owner.kind !== "native")
    throw new Error("Expected native fixture ownership")
  return {
    ...mounted,
    owner: mounted.owner,
    remove,
    markDeleted: () => {
      deleted = true
    }
  }
}

it.each([false, true])(
  "native leaf Delete recaptures its parent and keeps the other branch character=%s",
  async (character) => {
    const mounted = await mountNativeLeafDelete(character)
    try {
      let result: unknown
      await act(async () => {
        result = await mounted.hook.result.current.deleteMessage(mounted.index)
      })
      expect(result).toBe(true)
      expect(mounted.remove).toHaveBeenCalledExactlyOnceWith(
        "a1",
        1,
        nativeChatId,
        expect.objectContaining({
          requireLeaf: true,
          expectedRevision: "opaque-target-1",
          requestScope: mounted.owner.request_scope,
          signal: expect.any(AbortSignal)
        })
      )
      expect(mounted.selection.result.current.view?.cursor).toEqual({
        kind: "after_message",
        message_id: "u-old"
      })
      expect(mounted.publish).toHaveBeenCalledOnce()
      expect(
        mounted.selection.result.current
          .getCurrent()
          .capture?.snapshot.nodes.map((row) => row.id)
      ).toEqual(["u-old", "a2"])
      expect(
        mounted.draftStore.getState().messages.map((row) => row.id)
      ).toEqual(["u-old"])
      expect(mounted.write).not.toHaveBeenCalled()
    } finally {
      mounted.cleanup()
    }
  }
)

it("native Delete never removes the mounted reply before acknowledgement", async () => {
  const mounted = await mountNativeLeafDelete()
  try {
    let release!: () => void
    mounted.remove.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => {
            mounted.markDeleted()
            resolve()
          }
        })
    )
    let pending!: Promise<unknown>
    act(() => {
      pending = mounted.hook.result.current.deleteMessage(mounted.index)
    })
    await vi.waitFor(() => expect(mounted.remove).toHaveBeenCalledOnce())
    expect(mounted.publish).not.toHaveBeenCalled()
    expect(mounted.draftStore.getState().messages.map((row) => row.id)).toEqual(
      ["u-old", "a1"]
    )
    await act(async () => {
      release()
      expect(await pending).toBe(true)
    })
  } finally {
    mounted.cleanup()
  }
})

it("a retained Delete confirmation refuses a same-ID H1 replacement before reading the target", async () => {
  const mounted = await mountNativeLeafDelete()
  try {
    const retained = mounted.hook.result.current.deleteMessage
    await act(async () => {
      expect(await mounted.selection.result.current.refresh()).toBe(true)
    })
    mounted.read.mockClear()
    await act(async () => {
      expect(await retained(mounted.index)).toBe(false)
    })
    expect(mounted.read).not.toHaveBeenCalled()
    expect(mounted.remove).not.toHaveBeenCalled()
  } finally {
    mounted.cleanup()
  }
})

it("a late acknowledged Delete preserves a newer draft and publishes no old capture", async () => {
  const mounted = await mountNativeLeafDelete()
  try {
    let release!: () => void
    mounted.remove.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => {
            mounted.markDeleted()
            resolve()
          }
        })
    )
    let pending!: Promise<unknown>
    act(() => {
      pending = mounted.hook.result.current.deleteMessage(mounted.index)
    })
    await vi.waitFor(() => expect(mounted.remove).toHaveBeenCalledOnce())
    const address = {
      kind: "fresh" as const,
      id: "newer-draft",
      scope: { type: "global" as const },
      temporary: false
    }
    expect(mounted.draftStore.getState().setChatDraftAddress(address)).toBe(
      true
    )
    mounted.draftStore.setState({
      messages: [
        {
          id: "newer",
          message: "Keep my newer draft",
          isBot: false,
          sources: []
        }
      ]
    })
    await act(async () => {
      release()
      expect(await pending).toBe(false)
    })
    expect(mounted.publish).not.toHaveBeenCalled()
    expect(mounted.draftStore.getState().messages[0].message).toBe(
      "Keep my newer draft"
    )
    expect(mounted.remove).toHaveBeenCalledOnce()
  } finally {
    mounted.cleanup()
  }
})

it("an acknowledged Delete with a failed refresh remains recoverable without false success", async () => {
  const mounted = await mountNativeLeafDelete()
  try {
    const capture = h1.capture.getMockImplementation()!
    mounted.remove.mockImplementationOnce(async () => {
      mounted.markDeleted()
      h1.capture.mockRejectedValueOnce(new Error("response lost"))
    })
    await act(async () => {
      expect(
        await mounted.hook.result.current.deleteMessage(mounted.index)
      ).toBe(false)
    })
    expect(mounted.publish).not.toHaveBeenCalled()
    expect(mounted.selection.result.current.status).toBe("stale_selection")
    expect(mounted.selection.result.current.view?.cursor).toEqual({
      kind: "after_message",
      message_id: "u-old"
    })
    h1.capture.mockImplementation(capture)
    await act(async () => {
      expect(await mounted.selection.result.current.refresh()).toBe(true)
    })
    expect(mounted.publish).toHaveBeenCalledOnce()
    expect(mounted.remove).toHaveBeenCalledOnce()
  } finally {
    mounted.cleanup()
  }
})

it("a retained native Delete cannot downgrade to legacy after New resets H1", async () => {
  const mounted = await mountNativeLeafDelete()
  try {
    const retained = mounted.hook.result.current.deleteMessage
    act(() => mounted.selection.result.current.reset())
    mounted.read.mockClear()
    await act(async () => {
      expect(await retained(mounted.index)).toBe(false)
    })
    expect(mounted.read).not.toHaveBeenCalled()
    expect(mounted.remove).not.toHaveBeenCalled()
    expect(mounted.publish).not.toHaveBeenCalled()
  } finally {
    mounted.cleanup()
  }
})

it.each(["choose", "follow"])(
  "native Delete reserves the writer revision before %s",
  async (action) => {
    const mounted = await mountNativeLeafDelete()
    try {
      await act(async () => {
        expect(
          await mounted.hook.result.current.deleteMessage(mounted.index)
        ).toBe(true)
      })
      const deletedView = mounted.selection.result.current.getCurrent().view!
      await act(async () => {
        const accepted =
          action === "choose"
            ? await mounted.selection.result.current.choose({ kind: "empty" })
            : await mounted.selection.result.current.followResult(
                deletedView,
                "a2",
                () => true
              )
        expect(accepted).toBe(true)
      })
      expect(
        mounted.selection.result.current.view!.selection_revision
      ).toBeGreaterThan(deletedView.selection_revision)
    } finally {
      mounted.cleanup()
    }
  }
)

it.each(["mirror", "refresh"])(
  "Delete publishes no obsolete feedback after H1 retires during %s",
  async (stage) => {
    const mounted = await mountNativeLeafDelete(false, true)
    const mirror = await import("@/db/dexie/server-chat-mirror")
    const cleanupMirror = vi.spyOn(
      mirror,
      "removeAcknowledgedServerMirrorMessage"
    )
    let release!: () => void,
      reached = false
    const originalCapture = h1.capture.getMockImplementation()!
    try {
      if (stage === "mirror")
        cleanupMirror.mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              reached = true
              release = resolve
            })
        )
      else {
        cleanupMirror.mockResolvedValue(undefined)
        mounted.remove.mockImplementationOnce(async () => {
          mounted.markDeleted()
          h1.capture.mockImplementationOnce(async (...args) => {
            reached = true
            await new Promise<void>((resolve) => {
              release = resolve
            })
            return originalCapture(...args)
          })
        })
      }
      let pending!: Promise<unknown>
      act(() => {
        pending = mounted.hook.result.current.deleteMessage(mounted.index)
      })
      await vi.waitFor(() => expect(reached).toBe(true))
      await act(async () => {
        expect(
          await mounted.selection.result.current.choose({
            kind: "after_message",
            message_id: "u-old"
          })
        ).toBe(true)
      })
      mounted.publish.mockClear()
      mocks.notification.warning.mockClear()
      await act(async () => {
        release()
        expect(await pending).toBe(false)
      })
      expect(mocks.notification.warning).not.toHaveBeenCalled()
      expect(mounted.publish).not.toHaveBeenCalled()
      expect(mounted.remove).toHaveBeenCalledOnce()
    } finally {
      cleanupMirror.mockRestore()
      mounted.cleanup()
    }
  }
)


describe("rendered Delete ownership", () => {
  const localDelete = async () => {
    const helpers = await import("@/db/dexie/helpers")
    vi.mocked(helpers.removeMessageById).mockReset()
    vi.mocked(helpers.removeMessageById).mockResolvedValue(undefined)
    const store =
      await vi.importActual<typeof import("@/store/option")>("@/store/option")
    localMutationController()
    const localReceipt = h1.controller.getCurrent()
    localReceipt.owner.conversation_id = "local-delete"
    localReceipt.view.conversation_id = "local-delete"
    installFixtureReceipt(localReceipt, store.useStoreMessageOption)
    bridgeDraftStore(store.useStoreMessageOption)
    store.useStoreMessageOption.setState({
      streaming: false,
      isProcessing: false
    })
    mocks.chatBaseState.streaming = false
    mocks.chatBaseState.isProcessing = false
    const origin = h1.controller.getCurrent()
    origin.status = "idle"
    origin.owner = null
    origin.capture = null
    origin.view = null
    mocks.storeState.historyId = "local-delete"
    mocks.chatBaseState.historyId = "local-delete"
    mocks.storeState.serverChatId = null
    mocks.storeState.messages = [
      {
        id: "local-leaf",
        isBot: true,
        message: "saved answer",
        sources: [],
        serverMessageId: undefined as string | undefined
      }
    ]
    const mounted = renderHook(() => useMessage())
    const setters = { messages: mocks.setMessages, history: mocks.setHistory }

    setters.messages.mockReset()
    setters.history.mockReset()
    return { ...mounted, setters, store }
  }
  it.each(["new draft", "H1 replacement", "unmount"])(
    "retires a retained local Delete after %s before storage",
    async (kind) => {
      const helpers = await import("@/db/dexie/helpers")
      const mounted = await localDelete()
      const retained = mounted.result.current.deleteMessage
      if (kind === "new draft")
        mounted.store.useStoreMessageOption.getState().startNewChatDraft()
      else if (kind === "H1 replacement") h1.controller.navigate()
      else mounted.unmount()
      expect(await retained(0)).toBe(false)
      expect(helpers.removeMessageById).not.toHaveBeenCalled()
      expect(mounted.setters.messages).not.toHaveBeenCalled()
      expect(mounted.setters.history).not.toHaveBeenCalled()
    }
  )
  it("refuses a retained local Delete while a reply owns activity", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    mocks.chatBaseState.streaming = true
    await expect(mounted.result.current.deleteMessage(0)).rejects.toThrow(
      "response_in_progress"
    )
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it.each(["acknowledged", "rejected"])(
    "does not publish %s local storage outcome after New",
    async (outcome) => {
      const helpers = await import("@/db/dexie/helpers")
      const mounted = await localDelete()
      let release!: () => void
      let reject!: (error: Error) => void
      vi.mocked(helpers.removeMessageById).mockImplementation(
        () =>
          new Promise<Awaited<ReturnType<typeof helpers.removeMessageById>>>(
            (resolve, fail) => {
              release = () => resolve(undefined)
              reject = fail
            }
          )
      )
      const pending = mounted.result.current.deleteMessage(0)
      await vi.waitFor(() =>
        expect(helpers.removeMessageById).toHaveBeenCalled()
      )
      mounted.store.useStoreMessageOption.getState().startNewChatDraft()
      if (outcome === "acknowledged") release()
      else reject(new Error("storage rejected"))
      expect(await pending).toBe(false)
      expect(mounted.setters.messages).not.toHaveBeenCalled()
      expect(mounted.setters.history).not.toHaveBeenCalled()
    }
  )
  it("deletes only a qualified local copy with an inherited server message alias", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
    const mounted = await localDelete()
    mocks.storeState.messages[0].serverMessageId = "server-leaf"
    const origin = h1.controller.getCurrent()
    origin.status = "ready"
    origin.owner = {
      kind: "local",
      profile_id: "profile",
      owner_key: "local-key",
      conversation_id: "local-delete"
    }
    origin.view = {
      ...captureFor(makeController().getCurrent().view).view,
      owner_key: "local-key",
      conversation_id: "local-delete"
    }
    mounted.rerender()
    expect(await mounted.result.current.deleteMessage(0)).toBe(true)
    expect(api.deleteMessage).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).toHaveBeenCalled()
  })
  it("refuses legacy server deletion without canonical owner authority", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mirror = await import("@/db/dexie/server-chat-mirror")
    const mirrorCleanup = vi.spyOn(
      mirror,
      "removeAcknowledgedServerMirrorMessage"
    )
    const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
    const mounted = await localDelete()
    mocks.storeState.serverChatId = nativeChatId
    mounted.store.useStoreMessageOption.setState({ serverChatId: nativeChatId })
    mocks.storeState.messages[0].serverMessageId = "server-leaf"
    mounted.rerender()
    await expect(mounted.result.current.deleteMessage(0)).rejects.toThrow(
      "native_history_mutation_unavailable"
    )
    expect(api.getMessage).not.toHaveBeenCalled()
    expect(api.deleteMessage).not.toHaveBeenCalled()
    expect(mirrorCleanup).not.toHaveBeenCalled()
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("rejects a changed live row from a retained Delete confirmation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    const retained = mounted.result.current.deleteMessage
    mocks.storeState.messages = [
      { ...mocks.storeState.messages[0], message: "newer content" }
    ]
    mounted.rerender()
    await expect(retained(0)).rejects.toThrow("message_changed")
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("keeps Temporary Delete usable without a canonical controller", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    h1.controller = null
    mocks.storeState.temporaryChat = true
    mocks.chatBaseState.historyId = "temp"
    mounted.store.useStoreMessageOption.getState().startNewChatDraft(true)
    mounted.store.useStoreMessageOption.setState({ historyId: "temp", serverChatId: null })
    mounted.rerender()
    expect(await mounted.result.current.deleteMessage(0)).toBe(true)
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
    expect(mounted.setters.messages).toHaveBeenCalled()
  })
  it("does not publish old history after the message setter synchronously starts New", async () => {
    const mounted = await localDelete()
    mounted.setters.messages.mockImplementation(() =>
      mounted.store.useStoreMessageOption.getState().startNewChatDraft()
    )
    expect(await mounted.result.current.deleteMessage(0)).toBe(false)
    expect(mounted.setters.history).not.toHaveBeenCalled()
  })
  it("checks the live target before React has rerendered", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    const retained = mounted.result.current.deleteMessage
    mocks.storeState.messages = [
      { ...mocks.storeState.messages[0], message: "newer content" }
    ]
    await expect(retained(0)).rejects.toThrow("message_changed")
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("retains exact target validity during a held local mutation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    let release!: () => void
    vi.mocked(helpers.removeMessageById).mockImplementation(
      () =>
        new Promise((resolve) => {
          release = () => resolve(undefined)
        })
    )
    const pending = mounted.result.current.deleteMessage(0)
    await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
    const lease = vi.mocked(helpers.removeMessageById).mock.calls[0][3]!
      .validate_lease!
    mocks.storeState.messages = [
      { ...mocks.storeState.messages[0], message: "newer content" }
    ]
    expect(lease()).toBe(false)
    release()
    expect(await pending).toBe(false)
    expect(mounted.setters.messages).not.toHaveBeenCalled()
  })
  it.each([false, true])(
    "qualifies only its owned parent-follow epoch (retired=%s)",
    async (retired) => {
      const helpers = await import("@/db/dexie/helpers")
      const mounted = await localDelete()
      const origin = h1.controller.getCurrent()
      origin.status = "ready"
      origin.owner = {
        kind: "local",
        profile_id: "profile",
        owner_key: "local-key",
        conversation_id: "local-delete"
      }
      origin.view = {
        ...captureFor(makeController().getCurrent().view).view,
        owner_key: "local-key",
        conversation_id: "local-delete",
        cursor: { kind: "after_message", message_id: "local-leaf" }
      }
      let epoch = 0
      h1.controller.fence = () => {
        const captured = epoch
        return () => captured === epoch
      }
      h1.controller.choose = vi.fn(() => {
        epoch++
        if (retired)
          queueMicrotask(() => {
            epoch++
          })
        return Promise.resolve(true)
      })
      vi.mocked(helpers.removeMessageById).mockResolvedValue({
        id: "local-leaf",
        history_id: "local-delete",
        parent_message_id: null
      } as Awaited<ReturnType<typeof helpers.removeMessageById>>)
      mounted.rerender()
      expect(await mounted.result.current.deleteMessage(0)).toBe(!retired)
    }
  )
  it.each([false, true])(
    "deletes the exact repeated reply from prompt history (Temporary=%s)",
    async (temporary) => {
      const mounted = await localDelete()
      const rows = [
        {
          id: "question-1",
          isBot: false,
          message: "First question",
          sources: [],
          serverMessageId: undefined
        },
        {
          id: "answer-1",
          isBot: true,
          message: "Sure.",
          sources: [],
          serverMessageId: undefined
        },
        {
          id: "question-2",
          isBot: false,
          message: "Second question",
          sources: [],
          serverMessageId: undefined
        },
        {
          id: "answer-2",
          isBot: true,
          message: "Sure.",
          sources: [],
          serverMessageId: undefined
        }
      ]
      mocks.storeState.messages = rows
      if (temporary) {
        h1.controller = null
        mocks.storeState.temporaryChat = true
        mocks.chatBaseState.historyId = "temp"
        mounted.store.useStoreMessageOption.getState().startNewChatDraft(true)
        mounted.store.useStoreMessageOption.setState({ historyId: "temp", serverChatId: null })
      }
      mounted.rerender()
      expect(await mounted.result.current.deleteMessage(3)).toBe(true)
      const published = mounted.setters.messages.mock.calls.at(-1)![0]
      expect(
        typeof published === "function" ? published(rows) : published
      ).toEqual(rows.slice(0, 3))
      const previous = rows.map((row) => ({
        role: row.isBot ? "assistant" : "user",
        content: row.message
      }))
      const history = mounted.setters.history.mock.calls.at(-1)![0]
      const next = typeof history === "function" ? history(previous) : history
      expect(next.map(({ role, content }) => ({ role, content }))).toEqual(
        previous.slice(0, 3)
      )
    }
  )
  it("preserves a newer Reply target and does not clear Reply before acknowledgement", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    mocks.storeState.replyTarget = { id: "local-leaf", preview: "saved answer" }
    mounted.rerender()
    let release!: () => void
    vi.mocked(helpers.removeMessageById).mockImplementation(
      () =>
        new Promise((resolve) => {
          release = () => resolve(undefined)
        })
    )
    const pending = mounted.result.current.deleteMessage(0)
    await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
    const clearedBeforeAck = mocks.storeState.clearReplyTarget.mock.calls.length
    mocks.storeState.replyTarget = { id: "other-reply", preview: "new reply" }
    release()
    await pending
    expect(clearedBeforeAck).toBe(0)
    expect(mocks.storeState.clearReplyTarget).not.toHaveBeenCalled()
  })
  it("preserves a surviving message edited during held Delete", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    const rows = [
      mocks.storeState.messages[0],
      {
        id: "other-leaf",
        isBot: true,
        message: "original sibling",
        sources: []
      }
    ]
    mocks.storeState.messages = rows
    mounted.rerender()
    let release!: () => void
    vi.mocked(helpers.removeMessageById).mockImplementation(
      () =>
        new Promise((resolve) => {
          release = () => resolve(undefined)
        })
    )
    const pending = mounted.result.current.deleteMessage(0)
    await vi.waitFor(() => expect(helpers.removeMessageById).toHaveBeenCalled())
    const updated = [rows[0], { ...rows[1], message: "edited sibling" }]
    mocks.storeState.messages = updated
    release()
    await pending
    const published = mounted.setters.messages.mock.calls.at(-1)![0]
    expect(
      typeof published === "function" ? published(updated) : published
    ).toEqual([updated[1]])
    const history = mounted.setters.history.mock.calls.at(-1)![0]
    expect(
      typeof history === "function"
        ? history(
            updated.map((row) => ({ role: "assistant", content: row.message }))
          )
        : history
    ).toEqual([{ role: "assistant", content: "edited sibling" }])
  })
})

it.each(["confirmed", "wrong-parent", "stream-evidence", "explicit-branch"])(
  "workspace H1: native branch refusal distinguishes %s before any observed turn",
  async (kind) => {
    h1.native = true
    installWorkspaceCharacterFixture()
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    h1.controller.holdStaleSend = vi.fn(() => true)
    h1.controller.takeBranchIntent = vi.fn(() => kind === "explicit-branch")
    h1.wire.mockImplementation(async function* () {
      if (kind === "stream-evidence")
        yield { choices: [{ delta: { role: "assistant" } }] }
      yield await Promise.reject(
        Object.assign(new Error("HTTP 409"), {
          status: 409,
          details: {
            detail: {
              status: "stale_selection",
              code: "history_branch_changed",
              conversation_id: nativeChatId,
              parent_message_id: kind === "wrong-parent" ? "foreign" : "a1",
              leaf_ids: ["elsewhere"],
              history_version: 2
            }
          }
        })
      )
    })
    const { result } = renderHook(() => useMessage())
    let submitted: unknown
    await act(async () => {
      submitted = await result.current.onSubmit({
        message: "  exact branch input\nspaces  ",
        image: ""
      })
    })
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      state: kind === "confirmed" ? "refused" : "unknown",
      input_text: "  exact branch input\nspaces  "
    })
    expect(h1.wire.mock.lastCall?.[0].tldw_history_branch).toBe(
      kind === "explicit-branch"
    )
    expect(h1.dismiss).not.toHaveBeenCalled()
    if (kind === "confirmed") {
      expect(submitted).toMatchObject({
        status: "failed",
        errorMessage: "native_history_branch_changed"
      })
      expect(mocks.setMessages).not.toHaveBeenCalled()
      expect(h1.controller.holdStaleSend).toHaveBeenCalledOnce()
    } else expect(h1.controller.holdStaleSend).not.toHaveBeenCalled()
  }
)

it.each(["H1", "account", "draft", "journal failure"])(
  "workspace H1: native confirmed refusal owns its recovery through %s",
  async (boundary) => {
    h1.native = true
    installWorkspaceCharacterFixture()
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    const actualStore =
      await vi.importActual<typeof import("@/store/option")>("@/store/option")
    installFixtureReceipt(h1.controller.getCurrent(), actualStore.useStoreMessageOption, "history-1")
    bridgeDraftStore(actualStore.useStoreMessageOption)
    h1.controller.holdStaleSend = vi.fn(() => true)
    if (boundary === "journal failure")
      h1.recover
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error("Local storage unavailable"))
    h1.wire.mockImplementation(async function* () {
      if (boundary === "H1") h1.controller.navigate()
      if (boundary === "account") h1.auth.abort()
      if (boundary === "draft")
        (
          await vi.importActual<typeof import("@/store/option")>(
            "@/store/option"
          )
        ).useStoreMessageOption
          .getState()
          .startNewChatDraft(false)
      yield await Promise.reject(
        Object.assign(new Error("HTTP 409"), {
          status: 409,
          details: {
            detail: {
              status: "stale_selection",
              code: "history_branch_changed",
              conversation_id: nativeChatId,
              parent_message_id: "a1",
              leaf_ids: ["elsewhere"],
              history_version: 2
            }
          }
        })
      )
    })
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({
        message: "Original refused input",
        image: ""
      })
    })
    expect(h1.controller.holdStaleSend).toHaveBeenCalledTimes(
      boundary === "journal failure" ? 1 : 0
    )
    if (boundary !== "journal failure")
      expect(mocks.notification.error).not.toHaveBeenCalled()
    expect(h1.dismiss).not.toHaveBeenCalled()
  }
)

it("workspace H1: mounted Character Send holds an older-server continuation locally, before any dispatch record or generation", async () => {
  h1.native = true
    installWorkspaceCharacterFixture()
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  h1.spec.mockResolvedValue(null)
  h1.controller.holdStaleSend = vi.fn(() => true)
  h1.capture.mockImplementation(async (_id, request) => {
    const result = captureFor(request.view)
    return {
      ...result,
      snapshot: {
        ...result.snapshot,
        nodes: [
          ...result.snapshot.nodes,
          {
            id: "other-input",
            revision: "1",
            parent_id: "a1",
            role: "user",
            settled: true
          }
        ]
      }
    }
  })
  const mounted = renderHook(() => useMessage())
  let outcome: unknown
  await act(async () => {
    outcome = await mounted.result.current.onSubmit({
      message: "  Character input\nkept exactly.  ",
      image: ""
    })
  })
  expect(outcome).toMatchObject({
    status: "failed",
    errorMessage: "native_history_branch_choice_required"
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(h1.recover.mock.calls.map((call) => call[2].state)).toEqual([
    "refused"
  ])
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    input_text: "  Character input\nkept exactly.  ",
    input_images: [],
    result_text: ""
  })
  expect(h1.controller.holdStaleSend).toHaveBeenCalledOnce()
  expect(h1.dismiss).not.toHaveBeenCalled()
  mounted.unmount()
})

it.each(["H1", "account", "New", "unmount"])(
  "workspace H1: held Character contract discovery cannot restore authority after %s",
  async (boundary) => {
    h1.native = true
    installWorkspaceCharacterFixture()
    mocks.storeState.serverChatCharacterId = "12"
    mocks.storeState.serverChatAssistantKind = "character"
    h1.controller.holdStaleSend = vi.fn(() => true)
    const actualStore =
      await vi.importActual<typeof import("@/store/option")>("@/store/option")
    installFixtureReceipt(h1.controller.getCurrent(), actualStore.useStoreMessageOption, "history-1")
    bridgeDraftStore(actualStore.useStoreMessageOption)
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    h1.spec.mockImplementation(async () => {
      await held
      return null
    })
    const mounted = renderHook(() => useMessage())
    let pending!: Promise<unknown>
    act(() => {
      pending = mounted.result.current.onSubmit({
        message: "Original unsent input",
        image: ""
      })
    })
    await waitFor(() => expect(h1.spec).toHaveBeenCalledOnce())
    if (boundary === "H1") h1.controller.navigate()
    if (boundary === "account") h1.auth.abort()
    if (boundary === "New")
      actualStore.useStoreMessageOption.getState().startNewChatDraft(false)
    if (boundary === "unmount") mounted.unmount()
    await act(async () => {
      release()
      await pending
    })
    expect(h1.wire).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
    expect(h1.recover).not.toHaveBeenCalled()
    expect(h1.controller.holdStaleSend).not.toHaveBeenCalled()
    if (boundary !== "unmount") mounted.unmount()
  }
)


it("mounted Compact saved Roleplay dispatch bypasses mutable live Scene and model defaults", async () => {
  const sender = await import("../chat/roleplay-resume-send")
  const send = vi
    .spyOn(sender, "sendSavedRoleplay")
    .mockResolvedValue(true)
  h1.native = true
  mocks.storeState.serverChatCharacterId = "12"
  mocks.storeState.serverChatAssistantKind = "character"
  mocks.storeState.selectedModel = ""
  mocks.getActorScene.mockRejectedValue(new Error("live Scene forbidden"))
  const hook = renderHook(() => useMessage())
  try {
    await act(async () => {
      await hook.result.current.onSubmit({
        message: "  Exact Compact literal\n",
        image: ""
      })
    })
    expect(send).toHaveBeenCalledWith(
      expect.objectContaining({
        controller: h1.controller,
        message: "  Exact Compact literal\n",
        serverChatId: nativeChatId
      })
    )
    expect(mocks.getActorScene).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(h1.append).not.toHaveBeenCalled()
  } finally {
    hook.unmount()
    send.mockRestore()
  }
})


it("mounted Compact global saved Roleplay submits exact literal input with original protected receipts", async () => {
  installSavedRoleplayFixture()
  const { result } = renderHook(() => useMessage())
  await act(async () => { await result.current.onSubmit({ message: "  Exact Compact literal\nSecond line.  ", image: "" }) })
  expect(h1.roleplayAppend).toHaveBeenCalledWith(nativeChatId, expect.objectContaining({ content: "  Exact Compact literal\nSecond line.  ", expected: expect.objectContaining({ tail: { message_id: "a1", message_version: 1 } }) }), expect.objectContaining({ requestScope: mocks.makeSnapshot().requestScope }))
  expect(h1.roleplayComplete).toHaveBeenCalledOnce()
  expect(h1.controller.followResult).toHaveBeenCalledWith(expect.objectContaining({ conversation_id: nativeChatId, view_session_id: "origin" }), h1.roleplayComplete.mock.calls[0][1].tldw_roleplay_resume_v1.assistant_message_id, expect.any(Function))
  expect(mocks.getActorScene).not.toHaveBeenCalled()
  expect(h1.append).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it("mounted Compact global saved Roleplay refuses a before-first cursor without changing its canonical tail", async () => {
  installSavedRoleplayFixture()
  const current = h1.controller.getCurrent()
  current.view.cursor = { kind: "empty" }
  current.capture = captureFor(current.view)
  const { result } = renderHook(() => useMessage())
  let submitted: unknown
  await act(async () => { submitted = await result.current.onSubmit({ message: "  Keep original branch input  ", image: "" }) })
  expect(submitted).toMatchObject({ status: "failed" })
  expect(h1.roleplayState.get(nativeChatId)!.tail).toBe("a1")
  expect(h1.roleplayAppend).not.toHaveBeenCalled()
  expect(h1.roleplayComplete).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it.each(["append", "completion"])("mounted Compact global saved Roleplay rejects a foreign %s receipt", async boundary => {
  installSavedRoleplayFixture()
  if (boundary === "append") h1.roleplayAppend.mockImplementationOnce(async (id, input, options) => {
    options.onDispatch?.(); return { ...roleplayAppend(id, input), conversation_id: "foreign-chat" }
  })
  else h1.roleplayComplete.mockImplementationOnce(async (id, body, options) => {
    options.onDispatch?.(); return { ...roleplayCompletion(id, body), chat_id: "foreign-chat" }
  })
  const { result } = renderHook(() => useMessage())
  await act(async () => { await result.current.onSubmit({ message: "  Original owned input  ", image: "" }) })
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({ conversation_id: nativeChatId, input_text: "  Original owned input  ", state: boundary === "append" ? "unknown" : "generated_unsaved" })
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})
