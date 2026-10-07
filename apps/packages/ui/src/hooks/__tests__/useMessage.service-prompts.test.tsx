import React from "react"
import type { Message } from "@/store/option"
import type { HistoryInfo, Message as StoredMessage } from "@/db/dexie/types"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import type { AssistantSelection } from "@/types/assistant-selection"
import type { EffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import type { SelectedChatModelValidationResult } from "@/utils/chat-model-validation"
import { act, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const fixtureResolvedSnapshot = vi.hoisted(() => ({ current: null as import("@/services/service-prompts").ServicePromptSnapshot | null }))

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
  const saveMessageOnError = vi.fn<
    (payload: any) => Promise<string | false>
  >(async () => false)
  const finalModel = {
    stream: vi.fn<(...args: any[]) => Promise<any>>(async () => []),
    invoke: vi.fn()
  }
  const rewriteModel = {
    stream: vi.fn(),
    invoke: vi.fn(async () => ({ content: "standalone retrieval query" }))
  }

  const storeState: Record<string, unknown> & {
    setServerChatId: ReturnType<typeof vi.fn<(id: string | null) => void>>
    setServerChatMetaLoaded: ReturnType<typeof vi.fn<(ready: boolean) => void>>
  } = {
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
    replyTarget: null,
    clearReplyTarget: vi.fn(),
    serverChatId: null,
    setServerChatId: vi.fn(),
    serverChatTitle: "",
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
    serverChatLoadState: "loading",
    setServerChatLoadState: vi.fn(),
    setServerChatLoadError: vi.fn(),
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
    historyId: "history-1",
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
  ) => Object.freeze({
    id,
    parts: Object.freeze([
      Object.freeze({
        key: "template",
        mode: "template" as const,
        required_variables: Object.freeze([...requiredVariables])
      })
    ])
  })
  const answerDefinition = definition("chat.rag.answer", ["context", "question"])
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
  ) => Object.freeze({
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
    useRealQueueState: false,
    selectedCharacter: null as { id: number; name: string; greeting: string } | null,
    selectedAssistant: null as AssistantSelection | null,
    effectiveAssistantState: null as EffectiveAssistantState | null,
    createChat: vi.fn(),
    getChat: vi.fn(),
    deleteMessage: vi.fn(),
    removeMessageById: vi.fn(),
    mirrorHistories: new Map<string, HistoryInfo>(),
    mirrorMessages: new Map<string, StoredMessage>(),
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

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ cancelQueries: vi.fn(), invalidateQueries: vi.fn() }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("~/services/tldw-server", () => ({
  promptForRag: (...args: unknown[]) => mocks.promptForRag(...args),
  systemPromptForNonRag: vi.fn(async () => "system")
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (...args: unknown[]) => {
    const snapshot = await mocks.loadServicePromptSnapshot(...args)
    if (snapshot) fixtureResolvedSnapshot.current = snapshot
    return snapshot
  },
  renderServicePromptPart: (...args: unknown[]) =>
    mocks.renderServicePromptPart(...args)
}))

vi.mock("~/store/option", async (importOriginal) => {
  const actual = await importOriginal<typeof import("~/store/option")>()
  return {
    useStoreMessageOption: Object.assign(
      (
        selector?: (state: Record<string, unknown>) => unknown,
        equalityFn?: (a: unknown, b: unknown) => boolean
      ) =>
        mocks.useRealQueueState
          ? actual.useStoreMessageOption(
              selector ?? ((state) => state),
              equalityFn
            )
          : selector
            ? selector(mocks.storeState)
            : mocks.storeState,
      {
        setState: (
          update:
            | Record<string, unknown>
            | ((state: typeof mocks.storeState) => Record<string, unknown>)
        ) => {
          if (mocks.useRealQueueState)
            actual.useStoreMessageOption.setState(update)
          else
            Object.assign(
              mocks.storeState,
              typeof update === "function" ? update(mocks.storeState) : update
            )
        },
        getState: () =>
          mocks.useRealQueueState
            ? actual.useStoreMessageOption.getState()
            : mocks.storeState,
        subscribe: (
          listener: Parameters<typeof actual.useStoreMessageOption.subscribe>[0]
        ) =>
          mocks.useRealQueueState
            ? actual.useStoreMessageOption.subscribe(listener)
            : () => {}
      }
    )
  }
})

vi.mock("~/store", () => ({
  useStoreMessage: () => ({ currentURL: "", setCurrentURL: vi.fn() })
}))

vi.mock("@/context", () => ({
  usePageAssist: () => ({
    controller: null,
    setController: mocks.setAbortController,
    embeddingController: null,
    setEmbeddingController: mocks.setEmbeddingController
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    key === "chatWithWebsiteEmbedding" ? true : fallback,
    vi.fn()
  ]
}))

vi.mock("@/hooks/chat/useChatBaseState", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/chat/useChatBaseState")>()
  return {
    useChatBaseState: (...args: Parameters<typeof actual.useChatBaseState>) =>
      mocks.useRealQueueState ? actual.useChatBaseState(...args) : mocks.chatBaseState
  }
})

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({ selectedModel: "model-1", setSelectedModel: vi.fn() })
}))

vi.mock("@/store/model", () => {
  const settings = {
    apiProvider: "provider-1",
    reset: vi.fn(),
    activeSettingsScope: "global",
    getEffectiveSettings: () => ({ apiProvider: "provider-1" })
  }
  return {
    useStoreChatModelSettings: Object.assign(
      (selector?: (state: typeof settings) => unknown) => selector ? selector(settings) : settings,
      { getState: () => settings }
    )
  }
})

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [mocks.selectedCharacter, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  getSelectedAssistantOperationRevision: () => 0,
  useSelectedAssistant: () => [mocks.selectedAssistant, vi.fn()]
}))

vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>(),
  getActorSceneForChat: vi.fn(async ({ target }) => ({ target, settings: (await import("@/types/actor")).createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => mocks.notification
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null })
}))

vi.mock("@/hooks/chat/effective-assistant-state", () => ({
  resolveEffectiveAssistantState: () => mocks.effectiveAssistantState ?? ({
    mode: "plain",
    kind: null,
    id: null,
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
  subscribeChatLoopEvents: () => vi.fn()
}))

vi.mock("@/utils/chat-model-validation", () => ({
  validateSelectedChatModelAvailability: vi.fn(async () => ({ status: "valid" }))
}))

vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: () => []
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({}))
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

vi.mock("@/libs/get-html", () => ({
  getContentFromCurrentTab: vi.fn(async () => ({
    content: "raw page content",
    url: "https://source.example/page",
    type: "html",
    pdf: []
  }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: (...args: unknown[]) => mocks.getChat(...args),
    createChat: (...args: unknown[]) => mocks.createChat(...args),
    addChatMessage: vi.fn(async () => ({ id: "saved-user", version: 1 })),
    streamCharacterChatCompletion: vi.fn(async function* () { yield { delta: "Image received" } }),
    persistCharacterCompletion: vi.fn(async () => ({ assistant_message_id: "saved-answer", version: 1 })),
    getMessage: vi.fn(),
    deleteMessage: mocks.deleteMessage,
    addMedia: (...args: unknown[]) => mocks.addMedia(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: () => "grounded context"
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) => mocks.humanMessageFormatter(...args)
}))

vi.mock("@/utils/generate-history", () => ({
  generateHistory: () => []
}))

vi.mock("@/libs/reasoning", () => ({
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

vi.mock("@/db/dexie/helpers", () => {
  let id = 0
  return {
    deleteChatForEdit: vi.fn(),
    deleteChatAfterMessageId: vi.fn(),
    generateID: () => `generated-${++id}`,
    getPromptById: vi.fn(),
    removeMessageByIndex: vi.fn(),
    removeMessageById: mocks.removeMessageById,
    updateMessageByIndex: vi.fn(),
    updateMessageById: vi.fn()
  }
})

// Keep mirror ownership checks and transaction orchestration real; replace IndexedDB storage.
vi.mock("@/db/dexie/schema", () => ({ db: {
  chatHistories: { get: async (id: string) => mocks.mirrorHistories.get(id) },
  messages: {
    get: async (id: string) => mocks.mirrorMessages.get(id),
    where: (field: keyof StoredMessage) => ({ equals: (value: unknown) => ({
      toArray: async () => [...mocks.mirrorMessages.values()].filter(row => row[field] === value)
    }) }),
    delete: async (id: string) => { mocks.mirrorMessages.delete(id) }
  },
  modelNickname: {}, sessionFiles: {},
  transaction: async (_mode: unknown, _tables: unknown, operation: (transaction: { abort: () => void }) => Promise<unknown>) =>
    operation({ abort: vi.fn() })
} }))

vi.mock("@/hooks/utils/messageHelpers", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/utils/messageHelpers")>()
  return {
    createSaveMessageOnError: (...args: Parameters<typeof actual.createSaveMessageOnError>) =>
      mocks.useRealQueueState ? actual.createSaveMessageOnError(...args) : mocks.saveMessageOnError,
    createSaveMessageOnSuccess: (...args: Parameters<typeof actual.createSaveMessageOnSuccess>) =>
      mocks.useRealQueueState ? actual.createSaveMessageOnSuccess(...args) : mocks.saveMessageOnSuccess,
    validateBeforeSubmit: actual.validateBeforeSubmit
  }
})

vi.mock("@/hooks/chat-helper", () => ({
  saveMessageOnSuccess: (...args: Parameters<typeof mocks.saveMessageOnSuccess>) => mocks.saveMessageOnSuccess(...args),
  saveMessageOnError: (...args: Parameters<typeof mocks.saveMessageOnError>) => mocks.saveMessageOnError(...args)
}))
vi.mock("@/libs/get-screenshot", () => ({
  getScreenshotFromCurrentTab: vi.fn(async () => ({ screenshot: "data:image/png;base64,synthetic" }))
}))
vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: vi.fn(async () => undefined)
}))
vi.mock("@/services/application", () => ({
  getPrompt: vi.fn(async () => "Please summarize {text}")
}))

vi.mock("@/hooks/handlers/messageHandlers", async importOriginal => ({
  ...await importOriginal<typeof import("@/hooks/handlers/messageHandlers")>(),
  createBranchMessage: () => vi.fn(),
  createRegenerateLastMessage: () => vi.fn()
}))

vi.mock("../chat-modes/normalChatMode", async importOriginal => ({
  ...await importOriginal<typeof import("../chat-modes/normalChatMode")>(),
  normalChatMode: vi.fn()
}))
vi.mock("../chat-modes/tabChatMode", () => ({ tabChatMode: vi.fn() }))
vi.mock("../chat-modes/documentChatMode", () => ({ documentChatMode: vi.fn() }))

vi.mock("@/utils/mcp-disclosure", () => ({
  applyMcpModuleDisclosureFromToolCalls: vi.fn()
}))

import { useMessage as useRawMessage } from "../useMessage"
import { validateSelectedChatModelAvailability } from "@/utils/chat-model-validation"
import { normalChatMode } from "../chat-modes/normalChatMode"

import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { freshChatDraftAddress, type ChatDraftAddress } from "@/utils/chat-draft-address"
import { captureActorSceneSendForTarget } from "@/hooks/chat/actor-scene-source"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { captureChatDraftSource, chatDraftAddressForReceipt } from "@/hooks/chat/chat-draft-transition"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"

beforeEach(() => { fixtureResolvedSnapshot.current = null })

// Model/transport mocks remain legacy. Only the caller's original private scene
// read is qualified, using the same real capture helpers as the composer.
const useMessage = () => {
  const lifetime = React.useRef({ active: true, accountRetired: false })
  const requestScope = React.useRef(structuredClone(mocks.makeSnapshot().requestScope)).current
  const accountIsCurrent = () => {
    const observed = fixtureResolvedSnapshot.current
    return !lifetime.current.accountRetired && (!observed ||
      chatDraftOwnerKey(observed.requestScope) === chatDraftOwnerKey(requestScope))
  }
  const seeded = React.useRef(false)
  if (!seeded.current) {
    const state = useStoreMessageOption.getState()
    const historyId = mocks.useRealQueueState ? state.historyId : mocks.chatBaseState.historyId
    const owner = state.serverChatId ? { kind: "native" as const, conversation_id: state.serverChatId, owner_key: "legacy-compact-scene-owner",
      request_scope: requestScope, validate_lease: accountIsCurrent } : historyId && historyId !== "temp" ? {
      kind: "local" as const, conversation_id: historyId, owner_key: "legacy-compact-local-owner", profile_id: "legacy-compact-profile",
      validate_lease: accountIsCurrent
    } : null
    const view = owner ? { owner_key: owner.owner_key, conversation_id: owner.conversation_id, view_session_id: "legacy-compact-scene-view",
      selection_revision: 0, interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "empty" as const } } : null
    const currentAddress = state.chatDraftAddress
    const addressMatches = currentAddress && (owner ? currentAddress.kind !== "fresh" && currentAddress.id === owner.conversation_id : currentAddress.kind === "fresh")
    useStoreMessageOption.setState({ historyId, selectedModel: "model-1", selectedModelRevision: state.selectedModelRevision ?? 0,
      toolChoice: state.toolChoice ?? "required", chatDraftRevision: state.chatDraftRevision ?? 0,
      chatDraftAddress: addressMatches ? currentAddress : owner && view ? chatDraftAddressForReceipt({ owner, view }) : freshChatDraftAddress(state.temporaryChat) })
    seeded.current = true
  }
  React.useLayoutEffect(() => {
    const original = lifetime.current
    original.active = true
    const release = watchChatAccountChanges(invalidated => { if (invalidated) original.accountRetired = true })
    return () => { original.active = false; release() }
  }, [])
  const actions = useRawMessage()
  const renderedSource = captureChatDraftSource(() => lifetime.current.active)
  const renderedAddress = useStoreMessageOption.getState().chatDraftAddress as ChatDraftAddress
  const renderedRevision = useStoreMessageOption.getState().chatDraftRevision
  const renderedSession = usePlaygroundSessionStore.getState()
  const read = async (input?: Parameters<typeof actions.onSubmit>[0]) => {
    const address = renderedAddress
    const queued = input?.assertQueuedDispatchCurrent
    const admittedQueue = queued?.isSourceCurrent?.() === true
    const sourceIsCurrent = () => renderedSource.isCurrent() || (admittedQueue && queued?.isSourceCurrent?.() === true && lifetime.current.active &&
      useStoreMessageOption.getState().chatDraftAddress === renderedAddress && useStoreMessageOption.getState().chatDraftRevision === renderedRevision &&
      usePlaygroundSessionStore.getState().restoreRevision === renderedSession.restoreRevision &&
      usePlaygroundSessionStore.getState().sourceSelectionRevision === renderedSession.sourceSelectionRevision)
    const owner = address.kind === "native" ? { kind: "native" as const, conversation_id: address.id, owner_key: address.ownerKey,
      request_scope: requestScope, scope: address.scope, validate_lease: accountIsCurrent } : address.kind === "local" ? {
      kind: "local" as const, conversation_id: address.id, owner_key: "legacy-compact-local-owner", profile_id: address.profileId, validate_lease: accountIsCurrent
    } : null
    const view = owner ? { owner_key: owner.owner_key, conversation_id: owner.conversation_id, view_session_id: "legacy-compact-scene-view",
      selection_revision: 0, interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "empty" as const } } : null
    const target = captureActorSettingsTarget({ address, owner, view, requestScope, accountIsCurrent, isCurrent: sourceIsCurrent })
    const dispatchIsCurrent = captureChatDispatchSource({
      selectedModel: input?.requestOverrides?.selectedModel || "model-1",
      toolChoice: input?.requestOverrides?.toolChoice,
      selectionSource: {
        model: input?.requestOverrides?.selectedModel ? "explicit" : "global",
        toolChoice: input?.requestOverrides?.toolChoice ? "explicit" : "global"
      }
    })
    const originalIntentIsCurrent = () => lifetime.current.active && accountIsCurrent() && dispatchIsCurrent() &&
      useStoreMessageOption.getState().chatDraftAddress === renderedAddress &&
      useStoreMessageOption.getState().chatDraftRevision === renderedRevision &&
      usePlaygroundSessionStore.getState().restoreRevision === renderedSession.restoreRevision &&
      usePlaygroundSessionStore.getState().sourceSelectionRevision === renderedSession.sourceSelectionRevision
    const capture = captureActorSceneSendForTarget(target, { requestScope, isCurrent: accountIsCurrent }, sourceIsCurrent, originalIntentIsCurrent)
    if (!capture) throw new Error("Legacy compact fixture has no qualified original scene")
    return capture.read()
  }
  return { ...actions, onSubmit: async (input: Parameters<typeof actions.onSubmit>[0]) => actions.onSubmit({ ...input,
    actorSceneSend: input.actorSceneSend ?? await read(input) }) }
}

describe("sidepanel saved conversation metadata lifetime", () => {
  it.each([false, true])("uses the real Persona creation helper without adopting a replacement (replaced=%s)", async (replaced) => {
    vi.clearAllMocks()
    mocks.storeState.serverChatId = null
    mocks.storeState.serverChatMetaLoaded = false
    mocks.selectedAssistant = { kind: "persona", id: "p1", name: "Persona", metadata: { selectionMode: "tracked" } }
    mocks.effectiveAssistantState = { mode: "tracked_persona", kind: "persona", id: "p1", displayName: "Persona", avatarUrl: null, systemPromptSnapshot: null, source: "tracked" }
    const idSetter = mocks.storeState.setServerChatId
    idSetter.mockImplementation((id) => { mocks.storeState.serverChatId = id })
    const readySetter = mocks.storeState.setServerChatMetaLoaded
    readySetter.mockImplementation((ready) => { mocks.storeState.serverChatMetaLoaded = ready })
    let resolveCreation!: (value: unknown) => void
    mocks.createChat.mockReturnValue(new Promise(resolve => { resolveCreation = resolve }))
    const hook = renderHook(() => useMessage())
    const guard = hook.result.current.conversationMetadata.captureQueuedDispatchGuard()
    const submission = hook.result.current.onSubmit({
      message: "First queued Persona turn", image: "", requestOverrides: { chatMode: "normal" },
      assertQueuedDispatchCurrent: guard
    })
    try {
      await waitFor(() => expect(mocks.createChat).toHaveBeenCalledTimes(1))
      if (replaced) mocks.storeState.serverChatId = "foreign-replacement"
      await act(async () => { resolveCreation({ id: "created-persona", assistant_kind: "persona", assistant_id: "p1" }) })
      const result = await submission
      if (replaced) {
        expect(mocks.storeState.serverChatId).toBe("foreign-replacement")
        expect(result).toMatchObject({ status: "skipped" })
        expect(normalChatMode).not.toHaveBeenCalled()
      } else {
        expect(result).toBeUndefined()
        expect(mocks.storeState.serverChatId).toBe("created-persona")
        expect(mocks.storeState.serverChatMetaLoaded).toBe(true)
        expect(normalChatMode).toHaveBeenCalledTimes(1)
      }
    } finally {
      hook.unmount()
      mocks.selectedAssistant = null
      mocks.effectiveAssistantState = null
      mocks.storeState.serverChatId = null
      mocks.storeState.serverChatMetaLoaded = true
      idSetter.mockImplementation(() => {})
      readySetter.mockImplementation(() => {})
    }
  })

  it("rejects a queued request when model preflight finishes after conversation replacement", async () => {
    vi.clearAllMocks()
    mocks.storeState.serverChatId = "saved-chat"
    mocks.storeState.serverChatMetaLoaded = true
    let resolveModel!: (value: SelectedChatModelValidationResult) => void
    vi.mocked(validateSelectedChatModelAvailability).mockReturnValueOnce(new Promise(resolve => { resolveModel = resolve }))
    const hook = renderHook(() => useMessage())
    const assertCurrent = hook.result.current.conversationMetadata.captureQueuedDispatchGuard()
    const submission = hook.result.current.onSubmit({
      message: "Queued private request", image: "", requestOverrides: { chatMode: "normal" },
      assertQueuedDispatchCurrent: assertCurrent
    }).then(() => null, (error: unknown) => error)
    await waitFor(() => expect(validateSelectedChatModelAvailability).toHaveBeenCalled())
    mocks.storeState.serverChatId = "replacement-chat"
    await act(async () => { resolveModel({ status: "valid" }) })
    expect(await submission).toMatchObject({ message: "Conversation changed. The queued request was not sent." })
    expect(normalChatMode).not.toHaveBeenCalled()
    hook.unmount()
    mocks.storeState.serverChatId = null
  })

  it.each(["unmount", "account", "conversation"])("does not publish a held read after %s", async (boundary) => {
    vi.clearAllMocks()
    mocks.storeState.serverChatId = "saved-chat"
    mocks.storeState.serverChatMetaLoaded = false
    const scope = new AbortController()
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot("supported", undefined, undefined, {
      scopeInvalidatedSignal: scope.signal
    }))
    let resolveChat!: (value: unknown) => void
    mocks.getChat.mockReturnValue(new Promise((resolve) => { resolveChat = resolve }))
    const hook = renderHook(() => useMessage())
    await waitFor(() => expect(mocks.getChat).toHaveBeenCalled())
    if (boundary === "unmount") hook.unmount()
    if (boundary === "account") scope.abort()
    if (boundary === "conversation") mocks.storeState.serverChatId = "replacement-chat"
    await act(async () => {
      resolveChat({ id: "saved-chat", title: "Old private title" })
    })
    expect(mocks.storeState.setServerChatTitle).not.toHaveBeenCalled()
    expect(mocks.storeState.setServerChatMetaLoaded).not.toHaveBeenCalled()
    hook.unmount()
    mocks.storeState.serverChatId = null
    mocks.storeState.serverChatMetaLoaded = true
  })

  it("exposes metadata failure for a retained queue instead of silently hanging", async () => {
    vi.clearAllMocks()
    mocks.storeState.serverChatId = "saved-chat"
    mocks.storeState.serverChatMetaLoaded = false
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    mocks.getChat.mockRejectedValue(new Error("private transport detail"))
    const hook = renderHook(() => useMessage())
    await waitFor(() => expect(mocks.storeState.setServerChatLoadState).toHaveBeenCalledWith("failed"))
    expect(mocks.storeState.setServerChatMetaLoaded).not.toHaveBeenCalled()
    hook.unmount()
    mocks.storeState.serverChatId = null
    mocks.storeState.serverChatMetaLoaded = true
  })
})

describe("useMessage legacy Sidepanel Service Prompts", () => {
  let restoreProjectionMocks: (() => void) | undefined
  const projectLegacyChatState = () => {
    const previousMessages = mocks.setMessages.getMockImplementation()
    const previousHistory = mocks.setHistory.getMockImplementation()
    let messages = structuredClone(mocks.storeState.messages) as Message[]
    let history = structuredClone(mocks.history)
    mocks.setMessages.mockImplementation((next) => {
      messages = typeof next === "function" ? next(messages) : next
    })
    mocks.setHistory.mockImplementation((next) => {
      history = typeof next === "function" ? next(history) : next
    })
    restoreProjectionMocks = () => {
      mocks.setMessages.mockImplementation(previousMessages ?? (() => {}))
      mocks.setHistory.mockImplementation(previousHistory ?? (() => {}))
    }
    return { get messages() { return messages }, get history() { return history } }
  }
  afterEach(() => {
    restoreProjectionMocks?.()
    restoreProjectionMocks = undefined
  })
  const expectNoMessageMutation = () => {
    expect(mocks.resetChatLoopState).not.toHaveBeenCalled()
    expect(mocks.setEmbeddingController).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(mocks.setStreaming).not.toHaveBeenCalled()
    expect(mocks.setIsProcessing).not.toHaveBeenCalled()
    expect(mocks.setIsEmbedding).not.toHaveBeenCalled()
    expect(mocks.setIsLoading).not.toHaveBeenCalled()
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.mirrorHistories.clear()
    mocks.mirrorMessages.clear()
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: "legacy answer {context} {question}",
      ragQuestionPrompt: "legacy rewrite {chat_history} {question}"
    })
    mocks.pageAssistModel.mockImplementation(async (options: unknown) => {
      const modelOptions = options as { toolChoice?: string }
      return modelOptions.toolChoice === "none"
        ? mocks.rewriteModel
        : mocks.finalModel
    })
    mocks.ragSearch.mockResolvedValue({
      results: [
        {
          content: "retrieved chunk",
          metadata: {
            source: "Source title",
            type: "html",
            url: "https://source.example/page"
          }
        }
      ]
    })
    mocks.addMedia.mockResolvedValue({
      results: [{ status: "Success", db_id: 321 }]
    })
    mocks.humanMessageFormatter.mockImplementation(async (input: unknown) => input)
    mocks.renderServicePromptPart.mockImplementation(
      (_definition: unknown, _partKey: string, authored: string, values: Record<string, string>) =>
        authored.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_match, key: string) =>
          Object.prototype.hasOwnProperty.call(values, key) ? values[key] : _match
        )
    )
  })

  it.each([false, true])("sends preset requests with the resolved model and OCR override (%s)", async (useOCR) => {
    const { result } = renderHook(() => useMessage())
    await act(async () => {
      await result.current.onSubmit({
        message: "Summarize this page",
        image: "",
        messageType: "summary",
        requestOverrides: { chatMode: "normal", selectedModel: "override-model", useOCR }
      })
    })
    expect(mocks.pageAssistModel).toHaveBeenCalledWith(
      expect.objectContaining({ model: "override-model" })
    )
    expect(mocks.humanMessageFormatter).toHaveBeenCalledWith(
      expect.objectContaining({ model: "override-model", useOCR })
    )
    expect(mocks.finalModel.stream).toHaveBeenCalled()
  })

  it.each([
    ...(["rag", "vision", "preset", "normal"] as const).flatMap(mode =>
      [false, true].map(temporary => ({ mode, temporary, server: false }))
    ),
    { mode: "normal" as const, temporary: false, server: true }
  ])("keeps queued $mode completion owned after its own promotion (temporary=$temporary, server=$server)", async ({ mode, temporary, server }) => {
    const { useStoreMessageOption: realStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
    const previous = realStore.getState()
    const chatMode = mode === "preset" ? "normal" : mode
    mocks.useRealQueueState = true
    realStore.setState({ historyId: null, serverChatId: null, serverChatMetaLoaded: false, messages: [], history: [], temporaryChat: temporary, chatMode, streaming: false, selectedSystemPrompt: null })
    mocks.saveMessageOnSuccess.mockImplementation(async payload => {
      payload.setHistoryId("created-local-history")
      return "created-local-history"
    })
    vi.mocked(normalChatMode).mockImplementationOnce(async (...args) => {
      const options = args[6]
      await options.saveMessageOnSuccess({
        historyId: null,
        setHistoryId: options.setHistoryId,
        selectedModel: "model-1",
        modelId: "model-1",
        assistantMessageId: "queued-answer",
        isRegenerate: args[2],
        message: args[0],
        image: args[1],
        fullText: "Queued reply",
        source: [],
        reasoning_time_taken: 0,
        ...(server ? { conversationId: "created-server" } : {})
      })
      return { status: "submitted" }
    })
    const hook = renderHook(() => useMessage())
    const guard = hook.result.current.conversationMetadata.captureQueuedDispatchGuard(null)
    const completion = hook.result.current.conversationMetadata.isQueuedCompletionCurrent
    try {
      await act(async () => {
        await hook.result.current.onSubmit({ message: "First queued request", image: "", controller: new AbortController(), requestOverrides: { chatMode }, ...(mode === "preset" ? { messageType: "summary" } : {}), assertQueuedDispatchCurrent: guard })
      })
      if (!temporary) expect(mocks.saveMessageOnSuccess).toHaveBeenCalled()
      expect(realStore.getState().historyId).toBe(temporary ? "temp" : "created-local-history")
      if (server) expect(realStore.getState().serverChatId).toBe("created-server")
      expect(completion()).toBe(true)
    } finally {
      hook.unmount()
      realStore.setState(previous, true)
      mocks.useRealQueueState = false
      mocks.saveMessageOnSuccess.mockReset()
      vi.mocked(normalChatMode).mockReset()
    }
  })

  it.each([
    ...(["rag", "vision", "preset", "character"] as const).flatMap((mode) =>
      (["current", "source", "stop", "newer-controller", "unmount"] as const).map(
        (boundary) => ({ mode, boundary })
      )
    ),
    { mode: "rag" as const, boundary: "account" as const }
  ])(
    "deferred legacy $mode text respects its $boundary boundary",
    async ({ mode, boundary }) => {
      const { useStoreMessageOption: realStore } =
        await vi.importActual<typeof import("@/store/option")>("@/store/option")
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      const previous = realStore.getState()
      const oldCharacter = mocks.selectedCharacter
      const oldAssistant = mocks.selectedAssistant
      const oldEffective = mocks.effectiveAssistantState
      const controller = new AbortController()
      const account = new AbortController()
      let resume!: () => void
      let entered!: () => void
      const held = new Promise<void>((resolve) => {
        resume = resolve
      })
      const started = new Promise<void>((resolve) => {
        entered = resolve
      })
      const stream = async function* () {
        yield "Deferred answer"
        entered()
        await held
      }
      mocks.useRealQueueState = true
      realStore.getState().startNewChatDraft()
      realStore.setState({
        historyId: null,
        serverChatId: mode === "character" ? "character-chat" : null,
        messages: [],
        history: [],
        chatMode: mode === "character" || mode === "preset" ? "normal" : mode,
        temporaryChat: false,
        selectedSystemPrompt: null,
        streaming: false,
        isProcessing: false,
        serverChatMetaLoaded: true
      })
      if (mode === "character") {
        mocks.selectedCharacter = { id: 42, name: "Mira", greeting: "" }
        mocks.effectiveAssistantState = {
          mode: "tracked_character",
          kind: "character",
          id: "42",
          displayName: "Mira",
          avatarUrl: null,
          systemPromptSnapshot: null,
          source: "tracked"
        }
        vi.mocked(
          tldwClient.streamCharacterChatCompletion
        ).mockImplementationOnce(stream)
      } else {
        mocks.finalModel.stream.mockResolvedValueOnce(stream())
      }
      mocks.loadServicePromptSnapshot.mockResolvedValue(
        mocks.makeSnapshot("supported", undefined, undefined, {
          scopeInvalidatedSignal: account.signal
        })
      )
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] })
      const hook = renderHook(() => useMessage())
      let submission!: ReturnType<typeof hook.result.current.onSubmit>
      try {
        await act(async () => {
          submission = hook.result.current.onSubmit({
            message: "Current request",
            image: "",
            controller,
            ...(mode === "preset" ? { messageType: "summary" } : {})
          })
          await Promise.race([started, submission.then(outcome => { throw new Error(`Legacy compact turn ended before its held provider: ${JSON.stringify(outcome)}`) })])
        })
        const before = realStore.getState().messages
        expect(before.at(-1)?.message).toBe("▋")
        if (boundary === "source")
          act(() => realStore.getState().startNewChatDraft())
        if (boundary === "stop") controller.abort()
        if (boundary === "unmount") hook.unmount()
        if (boundary === "account") account.abort()
        if (boundary === "newer-controller") {
          mocks.selectedCharacter = null
          mocks.selectedAssistant = {
            kind: "character",
            id: "overlay-current",
            name: "Overlay",
            metadata: { selectionMode: "overlay" }
          }
          mocks.effectiveAssistantState = null
          hook.rerender()
          vi.mocked(normalChatMode).mockImplementationOnce(async () => ({
            status: "submitted"
          }))
          await act(async () => {
            await hook.result.current.onSubmit({
              message: "Newer request",
              image: "",
              controller: new AbortController(),
              requestOverrides: { chatMode: "normal" }
            })
          })
        }
        await act(async () => {
          await vi.advanceTimersByTimeAsync(80)
        })
        if (boundary === "current")
          expect(realStore.getState().messages.at(-1)?.message).toBe(
            "Deferred answer▋"
          )
        else expect(realStore.getState().messages).toBe(before)
        await act(async () => {
          resume()
          await submission
        })
        if (boundary === "current")
          expect(realStore.getState().messages.at(-1)?.message).toBe(
            "Deferred answer"
          )
      } finally {
        resume()
        try { if (submission) await submission.catch(() => {}) } finally {
          hook.unmount()
          vi.useRealTimers()
          realStore.setState(previous, true)
          mocks.useRealQueueState = false
          mocks.selectedCharacter = oldCharacter
          mocks.selectedAssistant = oldAssistant
          mocks.effectiveAssistantState = oldEffective
          vi.mocked(normalChatMode).mockReset()
        }
      }
    }
  )

  it("keeps a fresh compact settings publisher usable after an address-only draft change", async () => {
    const { useStoreMessageOption: realStore } =
      await vi.importActual<typeof import("@/store/option")>("@/store/option")
    const previous = realStore.getState()
    mocks.useRealQueueState = true
    realStore.getState().startNewChatDraft()
    realStore.setState({
      historyId: null,
      serverChatId: null,
      messages: [],
      history: [],
      chatMode: "normal",
      temporaryChat: false,
      selectedSystemPrompt: null,
      streaming: false,
      isProcessing: false
    })
    const hook = renderHook(() => useMessage())
    try {
      act(() => realStore.getState().startNewChatDraft())
      vi.mocked(normalChatMode).mockImplementationOnce(async (...args) => {
        args[6].setHistoryId("fresh-current-history")
        return { status: "submitted" }
      })
      await act(async () => {
        await hook.result.current.onSubmit({
          message: "First current request",
          image: ""
        })
      })
      expect(realStore.getState().historyId).toBe("fresh-current-history")
    } finally {
      hook.unmount()
      realStore.setState(previous, true)
      mocks.useRealQueueState = false
      vi.mocked(normalChatMode).mockReset()
    }
  })

  it.each(["current", "new", "same-id", "unmount", "account", "history-new", "history-aba", "server-new", "server-aba"] as const)(
    "keeps scratch-to-server save publication and its settings lease owned across %s",
    async (boundary) => {
      const { useStoreMessageOption: realStore } =
        await vi.importActual<typeof import("@/store/option")>("@/store/option")
      const { syncChatSettingsForServerChat } = await import(
        "@/services/chat-settings"
      )
      const previous = realStore.getState()
      const previousAssistant = mocks.selectedAssistant
      const previousEffectiveAssistant = mocks.effectiveAssistantState
      mocks.selectedAssistant = null
      mocks.effectiveAssistantState = null
      mocks.useRealQueueState = true
      realStore.getState().startNewChatDraft()
      realStore.setState({
        historyId: null,
        serverChatId: null,
        serverChatMetaLoaded: false,
        messages: [],
        history: [],
        temporaryChat: false,
        chatMode: "normal",
        streaming: false,
        isProcessing: false,
        selectedSystemPrompt: null
      })
      const source = realStore.getState()
      const scope = new AbortController()
      let stopSubscriber: (() => void) | undefined
      let subscriberReplacement: ReturnType<typeof realStore.getState> | undefined
      let finishPersistence!: () => void
      const held = new Promise<void>((resolve) => {
        finishPersistence = resolve
      })
      mocks.saveMessageOnSuccess.mockImplementation(async (payload) => {
        await held
        payload.setHistoryId("created-local-history")
        return "created-local-history"
      })
      vi.mocked(normalChatMode).mockImplementationOnce(async (...args) => {
        await args[6].saveMessageOnSuccess({
          historyId: null,
          setHistoryId: args[6].setHistoryId,
          conversationId: "created-server",
          saveToDb: true,
          scopeSignal: scope.signal,
          message: "Original request",
          fullText: "Saved reply"
        })
        return { status: "submitted" }
      })
      const hook = renderHook(() => useMessage())
      const submission = hook.result.current.onSubmit({
        message: "Original request",
        image: "",
        controller: new AbortController(),
        requestOverrides: { chatMode: "normal" }
      })
      try {
        await waitFor(() =>
          expect(mocks.saveMessageOnSuccess).toHaveBeenCalledOnce()
        )
        if (boundary === "new" || boundary === "same-id") {
          act(() => {
            realStore.getState().startNewChatDraft()
            realStore.setState({
              historyId: boundary === "same-id" ? "reopened-local-history" : null,
              serverChatId: boundary === "same-id" ? "created-server" : null,
              serverChatMetaLoaded: true
            })
          })
        }
        if (boundary === "unmount") hook.unmount()
        if (boundary === "account") scope.abort()
        if (boundary.startsWith("history-") || boundary.startsWith("server-")) {
          const field = boundary.startsWith("history-") ? "historyId" : "serverChatId"
          const issuedId = field === "historyId" ? "created-local-history" : "created-server"
          stopSubscriber = realStore.subscribe((state, previousState) => {
            if (subscriberReplacement || state[field] !== issuedId || previousState[field] === issuedId) return
            // Retire during the actual guarded setter, before it can acknowledge its ID.
            stopSubscriber?.()
            realStore.getState().startNewChatDraft()
            realStore.setState({
              historyId: boundary.endsWith("aba") ? "created-local-history" : null,
              serverChatId: boundary === "server-aba" ? "created-server" : null,
              serverChatMetaLoaded: true,
              serverChatTitle: "Replacement owns these details"
            })
            subscriberReplacement = realStore.getState()
          })
        }
        const currentView = realStore.getState()
        await act(async () => {
          finishPersistence()
          await submission
        })
        if (boundary === "current") {
          expect(realStore.getState().historyId).toBe("created-local-history")
          expect(realStore.getState().serverChatId).toBe("created-server")
          expect(syncChatSettingsForServerChat).toHaveBeenCalledOnce()
          const options = vi.mocked(syncChatSettingsForServerChat).mock
            .calls[0][0]
          expect(options.sourceDraftAddress).toBe(source.chatDraftAddress)
          expect(options.validate_lease?.()).toBe(true)
          act(() => {
            realStore.getState().startNewChatDraft()
            realStore.setState({
              historyId: "created-local-history",
              serverChatId: "created-server"
            })
          })
          expect(options.validate_lease?.()).toBe(false)
        } else {
          const expectedView = subscriberReplacement ?? currentView
          if (boundary.startsWith("history-") || boundary.startsWith("server-")) {
            expect(subscriberReplacement).toBeDefined()
            expect(realStore.getState().chatDraftAddress).toBe(expectedView.chatDraftAddress)
            expect(realStore.getState().chatDraftRevision).toBe(expectedView.chatDraftRevision)
            expect(realStore.getState().serverChatMetaLoaded).toBe(true)
            expect(realStore.getState().serverChatTitle).toBe("Replacement owns these details")
          }
          expect(realStore.getState().serverChatId).toBe(expectedView.serverChatId)
          expect(syncChatSettingsForServerChat).not.toHaveBeenCalled()
          if (boundary !== "account")
            expect(realStore.getState().historyId).toBe(expectedView.historyId)
        }
        expect(mocks.saveMessageOnSuccess).toHaveBeenCalledOnce()
      } finally {
        stopSubscriber?.()
        finishPersistence()
        await submission
        hook.unmount()
        realStore.setState(previous, true)
        mocks.useRealQueueState = false
        mocks.selectedAssistant = previousAssistant
        mocks.effectiveAssistantState = previousEffectiveAssistant
        mocks.saveMessageOnSuccess.mockReset()
        vi.mocked(normalChatMode).mockReset()
      }
    }
  )

  it.each(["local", "server"])(
    "refuses server mirror deletion without canonical ownership for the %s Reply target",
    async (replyKind) => {
      const previous = { ...mocks.storeState }
      const localId = "history-A:server:answer"
      const canonicalId = "canonical-answer"
      try {
        mocks.mirrorHistories.set("history-1", {
          id: "history-1",
          title: "Cedar",
          createdAt: 1,
          is_rag: false,
          message_source: "server",
          server_chat_id: "cedar",
          server_scope_key: serverChatMirrorOwnerKey(mocks.makeSnapshot())
        })
        const stored: StoredMessage = {
          id: localId,
          history_id: "history-1",
          name: "Cedar",
          role: "assistant",
          content: "Cedar reply",
          createdAt: 1,
          serverMessageId: canonicalId,
          serverMessageVersion: 2
        }
        const draft = {
          ...stored,
          id: "same-text-draft",
          serverMessageId: undefined
        }
        const otherHistory = {
          ...stored,
          id: "other-history:server:answer",
          history_id: "other-history"
        }
        for (const row of [stored, draft, otherHistory])
          mocks.mirrorMessages.set(row.id, row)
        Object.assign(mocks.storeState, {
          serverChatId: "cedar",
          replyTarget: { id: replyKind === "local" ? localId : canonicalId },
          messages: [
            {
              id: localId,
              serverMessageId: canonicalId,
              serverMessageVersion: 2,
              isBot: true,
              name: "Cedar",
              message: "Cedar reply"
            }
          ]
        })
        const { result } = renderHook(() => useMessage())
        const original = { ...mocks.storeState }
        mocks.setMessages.mockClear()
        mocks.setHistory.mockClear()
        await act(async () => {
          await expect(result.current.deleteMessage(0)).rejects.toThrow(
            "native_history_mutation_unavailable"
          )
        })
        expect(mocks.deleteMessage).not.toHaveBeenCalled()
        expect(
          (await import("@/services/tldw/TldwApiClient")).tldwClient.getMessage
        ).not.toHaveBeenCalled()
        expect(mocks.removeMessageById).not.toHaveBeenCalled()
        expect([...mocks.mirrorMessages.values()]).toEqual([
          stored,
          draft,
          otherHistory
        ])
        expect(mocks.setMessages).not.toHaveBeenCalled()
        expect(mocks.setHistory).not.toHaveBeenCalled()
        expect(mocks.storeState.replyTarget).toEqual(original.replyTarget)
        expect(mocks.storeState.clearReplyTarget).not.toHaveBeenCalled()
      } finally {
        Object.assign(mocks.storeState, previous)
      }
    }
  )

  it("loads one immutable RAG snapshot before the first message mutation", async () => {
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })
    })

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledTimes(1)
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      { signal: controller.signal, requestScope: mocks.makeSnapshot().requestScope }
    )
    expect(mocks.loadServicePromptSnapshot.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.setMessages.mock.invocationCallOrder[0]
    )
    const snapshot = await mocks.loadServicePromptSnapshot.mock.results[0].value
    expect(Object.isFrozen(snapshot)).toBe(true)
    expect(Object.isFrozen(snapshot.definitions)).toBe(true)
  })

  it("blocks unresolved supported-server migration with a Workflow prompts link", async () => {
    const migrationError = Object.assign(
      new Error("Review workflow prompts before continuing"),
      { code: "service_prompt_migration_required" }
    )
    mocks.loadServicePromptSnapshot.mockRejectedValue(migrationError)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Blocked follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expectNoMessageMutation()
    expect(mocks.notification.warning).toHaveBeenCalledTimes(1)
    const notice = mocks.notification.warning.mock.calls[0][0]
    render(<>{notice.description}</>)
    expect(screen.getByRole("link", { name: "Review workflow prompts" })).toHaveAttribute(
      "href",
      "/options.html#/settings/prompt"
    )
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("keeps catalog-404 local prompt behavior at the snapshot boundary", async () => {
    mocks.loadServicePromptSnapshot.mockResolvedValue(
      mocks.makeSnapshot(
        "legacy-404",
        "local answer {context} -> {question}",
        "local rewrite {chat_history} -> {question}"
      )
    )
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      1,
      mocks.rewriteDefinition,
      "template",
      "local rewrite {chat_history} -> {question}",
      expect.objectContaining({ question: "Current follow-up" })
    )
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      2,
      mocks.answerDefinition,
      "template",
      "local answer {context} -> {question}",
      {
        context: "grounded context",
        question: "standalone retrieval query"
      }
    )
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("does not fall back to local prompts after a supported-server failure", async () => {
    const secret = "PRIVATE TEMPLATE BODY"
    const supportedFailure = Object.assign(
      new Error(`Supported server detail failed: ${secret}`),
      { detail: { body: secret } }
    )
    mocks.loadServicePromptSnapshot.mockRejectedValue(supportedFailure)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expectNoMessageMutation()
    expect(mocks.notification.error).toHaveBeenCalledTimes(1)
    const notice = mocks.notification.error.mock.calls[0][0]
    const copy = `${String(notice.message)} ${String(notice.description)}`
    expect(copy).toMatch(/connection/i)
    expect(copy).toMatch(/try again/i)
    expect(copy).not.toContain(secret)
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.renderServicePromptPart).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.ragSearch).not.toHaveBeenCalled()
  })

  it("registers the snapshot controller while resolution is pending", async () => {
    let resolveSnapshot!: (snapshot: ReturnType<typeof mocks.makeSnapshot>) => void
    const pendingSnapshot = new Promise<ReturnType<typeof mocks.makeSnapshot>>(
      (resolve) => {
        resolveSnapshot = resolve
      }
    )
    mocks.loadServicePromptSnapshot.mockReturnValue(pendingSnapshot)
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller
    })
    await vi.waitFor(() => {
      expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledTimes(1)
    })

    expectNoMessageMutation()
    expect(mocks.setAbortController.mock.calls).toEqual([[controller]])

    resolveSnapshot(mocks.makeSnapshot())
    await act(async () => {
      await submission
    })
    expect(mocks.setAbortController.mock.calls[0]).toEqual([controller])
    expect(mocks.setAbortController).not.toHaveBeenCalledWith(null)
    const releaseController = mocks.setAbortController.mock.calls[1]?.[0]
    expect(releaseController).toEqual(expect.any(Function))
    const newerController = new AbortController()
    expect(releaseController(newerController)).toBe(newerController)
  })

  it("silently stops an aborted snapshot preflight and releases controller ownership", async () => {
    mocks.loadServicePromptSnapshot.mockImplementation(
      (_ids: unknown, options: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          options.signal.addEventListener("abort", () => {
            const abortError = new Error("Service Prompt request was aborted")
            abortError.name = "AbortError"
            reject(abortError)
          }, { once: true })
        })
    )
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller
    })
    await vi.waitFor(() => {
      expect(mocks.setAbortController).toHaveBeenCalledWith(controller)
    })
    controller.abort()
    await act(async () => {
      await submission
    })

    expectNoMessageMutation()
    const releaseController = mocks.setAbortController.mock.calls[1]?.[0]
    expect(releaseController).toEqual(expect.any(Function))
    expect(releaseController(controller)).toBeNull()
    const newerController = new AbortController()
    expect(releaseController(newerController)).toBe(newerController)
    expect(mocks.notification.error).not.toHaveBeenCalled()
    expect(mocks.notification.warning).not.toHaveBeenCalled()
  })

  it("releases controller ownership when the Sidepanel preamble fails", async () => {
    const controller = new AbortController()
    const release = vi.fn()
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(
      mocks.makeSnapshot(
        "supported",
        "custom answer {context} :: {question}",
        "custom rewrite {chat_history} :: {question}",
        { release }
      )
    )
    mocks.pageAssistModel.mockRejectedValueOnce(new Error("preamble failed"))
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await expect(result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })).rejects.toThrow("preamble failed")
    })

    expect(mocks.setAbortController).toHaveBeenCalledWith(controller)
    const releaseController = mocks.setAbortController.mock.calls.at(-1)?.[0]
    expect(releaseController).toEqual(expect.any(Function))
    expect(releaseController(controller)).toBeNull()
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("releases the prompt scope when pre-dispatch setup throws", async () => {
    const controller = new AbortController()
    const release = vi.fn()
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(
      mocks.makeSnapshot(
        "supported",
        "custom answer {context} :: {question}",
        "custom rewrite {chat_history} :: {question}",
        { release }
      )
    )
    mocks.resetChatLoopState.mockImplementationOnce(() => {
      throw new Error("pre-dispatch setup failed")
    })
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await expect(result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })).rejects.toThrow("pre-dispatch setup failed")
    })

    expect(release).toHaveBeenCalledTimes(1)
    const releaseController = mocks.setAbortController.mock.calls.at(-1)?.[0]
    expect(releaseController).toEqual(expect.any(Function))
    expect(releaseController(controller)).toBeNull()
  })

  it("reuses the snapshot for a tool-disabled rewrite and the final Sidepanel answer", async () => {
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      { scopeSignal: scopeController.signal, release }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const controller = new AbortController()
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller
      })
    })

    expect(mocks.pageAssistModel).toHaveBeenNthCalledWith(1, {
      model: "model-1",
      requestScope: snapshot.requestScope
    })
    expect(mocks.pageAssistModel).toHaveBeenNthCalledWith(2, {
      model: "model-1",
      toolChoice: "none",
      tools: [],
      saveToDb: false,
      requestScope: snapshot.requestScope
    })
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      1,
      snapshot.definitions["chat.rag.question_rewrite"]?.definition,
      "template",
      "custom rewrite {chat_history} :: {question}",
      {
        chat_history: [
          "Human: Earlier question",
          "Assistant: Earlier answer",
          "Human: Current follow-up"
        ].join("\n"),
        question: "Current follow-up"
      }
    )
    expect(mocks.ragSearch).toHaveBeenCalledWith("standalone retrieval query", {
      top_k: 4,
      sources: ["media_db"],
      include_media_ids: [321],
      signal: scopeController.signal,
      requestScope: snapshot.requestScope
    })
    expect(mocks.addMedia).toHaveBeenCalledWith("https://source.example/page", {
      signal: scopeController.signal,
      requestScope: snapshot.requestScope
    })
    expect(mocks.addMedia.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.ragSearch.mock.invocationCallOrder[0]
    )
    expect(mocks.renderServicePromptPart).toHaveBeenNthCalledWith(
      2,
      snapshot.definitions["chat.rag.answer"]?.definition,
      "template",
      "custom answer {context} :: {question}",
      {
        context: "grounded context",
        question: "standalone retrieval query"
      }
    )
    expect(mocks.humanMessageFormatter).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ model: "model-1", useOCR: true })
    )
    expect(mocks.humanMessageFormatter).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ model: "model-1", useOCR: true })
    )
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedModel: "model-1",
        source: [
          expect.objectContaining({
            name: "Source title",
            type: "html",
            url: "https://source.example/page",
            pageContent: "retrieved chunk"
          })
        ]
      })
    )
    const rewriteMessage = mocks.humanMessageFormatter.mock.results[0].value
    const finalMessage = mocks.humanMessageFormatter.mock.results[1].value
    await expect(rewriteMessage).resolves.toEqual({
      content: [
        {
          text: [
            "custom rewrite Human: Earlier question",
            "Assistant: Earlier answer",
            "Human: Current follow-up :: Current follow-up"
          ].join("\n"),
          type: "text"
        }
      ],
      model: "model-1",
      useOCR: true
    })
    await expect(finalMessage).resolves.toEqual({
      content: [
        {
          text: "custom answer grounded context :: standalone retrieval query",
          type: "text"
        }
      ],
      model: "model-1",
      useOCR: true
    })
    expect(mocks.rewriteModel.invoke).toHaveBeenCalledWith(
      [await rewriteMessage],
      { signal: scopeController.signal }
    )
    expect(mocks.finalModel.stream).toHaveBeenCalledWith(
      [await finalMessage],
      {
        signal: scopeController.signal,
        callbacks: [
          {
            handleLLMEnd: expect.any(Function)
          }
        ]
      }
    )
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("falls back to inline context when best-effort media ingest fails normally", async () => {
    mocks.addMedia.mockRejectedValueOnce(new Error("ingest unavailable"))
    const snapshot = mocks.makeSnapshot()
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.addMedia).toHaveBeenCalledWith("https://source.example/page", {
      signal: snapshot.scopeSignal,
      requestScope: snapshot.requestScope
    })
    expect(mocks.ragSearch).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledTimes(1)
  })

  it("stops Sidepanel retrieval when media ingest rejects the captured account", async () => {
    mocks.addMedia.mockRejectedValueOnce(Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    }))
    const snapshot = mocks.makeSnapshot()
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.ragSearch).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(mocks.setMessages).toHaveBeenLastCalledWith(mocks.storeState.messages)
    expect(mocks.setHistory).toHaveBeenLastCalledWith(mocks.history)
  })

  it("discards partial Sidepanel output when the prompt scope changes", async () => {
    const scopeController = new AbortController()
    const release = vi.fn()
    mocks.loadServicePromptSnapshot.mockResolvedValue(
      mocks.makeSnapshot(
        "supported",
        "custom answer {context} :: {question}",
        "custom rewrite {chat_history} :: {question}",
        {
          scopeSignal: scopeController.signal,
          scopeInvalidatedSignal: scopeController.signal,
          release
        }
      )
    )
    mocks.finalModel.stream.mockResolvedValueOnce(
      (async function* () {
        yield "partial old-scope answer"
        scopeController.abort()
      })()
    )
    const projection = projectLegacyChatState()
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(projection.messages).toEqual(mocks.storeState.messages)
    expect(projection.history).toEqual(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("discards a server-rejected Sidepanel turn before the scope signal aborts", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    const release = vi.fn()
    mocks.loadServicePromptSnapshot.mockResolvedValue(
      mocks.makeSnapshot(
        "supported",
        "custom answer {context} :: {question}",
        "custom rewrite {chat_history} :: {question}",
        { release }
      )
    )
    mocks.finalModel.stream.mockResolvedValueOnce(
      (async function* () {
        yield "partial old-scope answer"
        throw scopeChangedError
      })()
    )
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: new AbortController()
      })
    })

    expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(mocks.setMessages).toHaveBeenLastCalledWith(mocks.storeState.messages)
    expect(mocks.setHistory).toHaveBeenLastCalledWith(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("discards a Sidepanel turn when its prompt scope changes during final persistence", async () => {
    let resolveSave!: () => void
    const pendingSave = new Promise<string | undefined>((resolve) => {
      resolveSave = () => resolve(undefined)
    })
    mocks.saveMessageOnSuccess.mockReturnValueOnce(pendingSave)
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      {
        scopeSignal: scopeController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        release
      }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const projection = projectLegacyChatState()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller: new AbortController()
    })
    await vi.waitFor(() => {
      expect(mocks.saveMessageOnSuccess).toHaveBeenCalledTimes(1)
    })
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        requestScope: snapshot.requestScope,
        scopeSignal: scopeController.signal
      })
    )
    scopeController.abort()
    resolveSave()

    await act(async () => {
      await submission
    })

    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(projection.messages).toEqual(mocks.storeState.messages)
    expect(projection.history).toEqual(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("does not duplicate a completed Sidepanel turn when Stop fires during final persistence", async () => {
    let resolveSave!: () => void
    const pendingSave = new Promise<string | undefined>((resolve) => {
      resolveSave = () => resolve(undefined)
    })
    mocks.saveMessageOnSuccess.mockReturnValueOnce(pendingSave)
    const userController = new AbortController()
    const executionController = new AbortController()
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      {
        scopeSignal: executionController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        release
      }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller: userController
    })
    await vi.waitFor(() => {
      expect(mocks.saveMessageOnSuccess).toHaveBeenCalledTimes(1)
    })

    userController.abort()
    executionController.abort()
    resolveSave()

    await act(async () => {
      await submission
    })

    expect(scopeController.signal.aborted).toBe(false)
    expect(mocks.saveMessageOnError).not.toHaveBeenCalled()
    expect(mocks.setMessages).not.toHaveBeenLastCalledWith(
      mocks.storeState.messages
    )
    expect(mocks.setHistory).not.toHaveBeenLastCalledWith(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("persists a manual-stop Sidepanel partial under a scoped lease", async () => {
    const userController = new AbortController()
    const executionController = new AbortController()
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      {
        scopeSignal: executionController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        release
      }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    mocks.finalModel.stream.mockResolvedValueOnce(
      (async function* () {
        yield "partial answer"
        userController.abort()
        executionController.abort()
      })()
    )
    const { result } = renderHook(() => useMessage())

    await act(async () => {
      await result.current.onSubmit({
        message: "Current follow-up",
        image: "",
        controller: userController
      })
    })

    expect(mocks.saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: expect.stringContaining("partial answer"),
        requestScope: snapshot.requestScope,
        scopeSignal: executionController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        shouldAbortForScopeChange: expect.any(Function)
      })
    )
    const [savePayload] = mocks.saveMessageOnError.mock.calls.at(-1)!
    expect(savePayload.shouldAbortForScopeChange()).toBe(false)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("discards a Sidepanel turn when its prompt scope changes during error persistence", async () => {
    let resolveSave!: (value: string) => void
    const pendingSave = new Promise<string>((resolve) => {
      resolveSave = resolve
    })
    mocks.saveMessageOnError.mockReturnValueOnce(pendingSave)
    const executionController = new AbortController()
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      {
        scopeSignal: executionController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        release
      }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    mocks.finalModel.stream.mockResolvedValueOnce(
      (async function* () {
        yield "partial answer"
        throw new Error("provider failed")
      })()
    )
    const projection = projectLegacyChatState()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller: new AbortController()
    })
    await vi.waitFor(() => {
      expect(mocks.saveMessageOnError).toHaveBeenCalledTimes(1)
    })
    const [savePayload] = mocks.saveMessageOnError.mock.calls[0]!
    expect(savePayload).toMatchObject({
      requestScope: snapshot.requestScope,
      scopeSignal: executionController.signal,
      scopeInvalidatedSignal: scopeController.signal
    })
    expect(savePayload.shouldAbortForScopeChange).toEqual(expect.any(Function))
    scopeController.abort()
    expect(savePayload.shouldAbortForScopeChange()).toBe(true)
    resolveSave("history-1")

    await act(async () => {
      await submission
    })

    expect(projection.messages).toEqual(mocks.storeState.messages)
    expect(projection.history).toEqual(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })

  it("discards a stopped Sidepanel turn if its account changes during error persistence", async () => {
    let resolveSave!: (value: string) => void
    const pendingSave = new Promise<string>((resolve) => {
      resolveSave = resolve
    })
    mocks.saveMessageOnError.mockReturnValueOnce(pendingSave)
    const userController = new AbortController()
    const executionController = new AbortController()
    const scopeController = new AbortController()
    const release = vi.fn()
    const snapshot = mocks.makeSnapshot(
      "supported",
      "custom answer {context} :: {question}",
      "custom rewrite {chat_history} :: {question}",
      {
        scopeSignal: executionController.signal,
        scopeInvalidatedSignal: scopeController.signal,
        release
      }
    )
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshot)
    mocks.finalModel.stream.mockResolvedValueOnce(
      (async function* () {
        yield "partial answer"
        throw new Error("provider failed")
      })()
    )
    const projection = projectLegacyChatState()
    const { result } = renderHook(() => useMessage())

    const submission = result.current.onSubmit({
      message: "Current follow-up",
      image: "",
      controller: userController
    })
    await vi.waitFor(() => {
      expect(mocks.saveMessageOnError).toHaveBeenCalledTimes(1)
    })
    const [savePayload] = mocks.saveMessageOnError.mock.calls[0]!

    userController.abort()
    executionController.abort()
    expect(savePayload.shouldAbortForScopeChange()).toBe(false)

    scopeController.abort()
    expect(savePayload.shouldAbortForScopeChange()).toBe(true)
    resolveSave("history-1")

    await act(async () => {
      await submission
    })

    expect(projection.messages).toEqual(mocks.storeState.messages)
    expect(projection.history).toEqual(mocks.history)
    expect(release).toHaveBeenCalledTimes(1)
  })
})


describe("Character live attachments in the legacy caller", () => {
  it.each(["", "Describe this image", null])("preserves the submitted image for text %s", async text => {
    vi.clearAllMocks()
    const previous = { ...mocks.storeState }
    const image = text === null ? "" : "data:image/png;base64,aW1hZ2U="
    let visible: Message[] = []
    mocks.effectiveAssistantState = { mode: "tracked_character", kind: "character", id: "42", displayName: "Mira", avatarUrl: null, systemPromptSnapshot: null, source: "tracked" }
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    Object.assign(mocks.storeState, { messages: [], serverChatId: "character-chat", serverChatCharacterId: 42, serverChatMetaLoaded: true })
    mocks.setMessages.mockImplementation(next => { visible = typeof next === "function" ? next(visible) : next })
    const hook = renderHook(() => useMessage())
    try {
      await act(async () => { await hook.result.current.onSubmit({ message: text ?? "Text-only control", image, requestOverrides: { chatMode: "normal" } }) })
      const user = visible.find(row => !row.isBot)
      expect(user).toBeDefined()
      expect(user.images).toEqual(image ? [image] : [])
    } finally {
      hook.unmount()
      Object.assign(mocks.storeState, previous)
      mocks.effectiveAssistantState = null
      mocks.setMessages.mockImplementation(() => {})
    }
  })
})


describe("Character greeting continuity in the sidebar caller", () => {
  it.each(["messages", "history", "saved-greeting", "new-chat"] as const)("preserves greeting provenance for %s", async kind => {
    vi.clearAllMocks()
    const previousStore = { ...mocks.storeState }
    const previousBase = { ...mocks.chatBaseState }
    const greeting = "Welcome from the Character default"
    const user = { id: "earlier-user", isBot: false, name: "You", message: "Earlier question", images: [], sources: [], createdAt: 1 }
    const savedGreeting = { id: "saved-greeting", isBot: true, name: "Mira", message: greeting, messageType: "character:greeting", images: [], sources: [], createdAt: 0 }
    const initial = kind === "new-chat" || kind === "history" ? [] : kind === "saved-greeting" ? [savedGreeting, user] : [user]
    let visible: Message[] = initial
    mocks.selectedCharacter = { id: 42, name: "Mira", greeting }
    mocks.effectiveAssistantState = { mode: "tracked_character", kind: "character", id: "42", displayName: "Mira", avatarUrl: null, systemPromptSnapshot: null, source: "tracked" }
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    mocks.createChat.mockResolvedValue({ id: "created-character-chat", character_id: 42 })
    Object.assign(mocks.storeState, { messages: initial, serverChatId: kind === "new-chat" ? null : "character-chat", serverChatCharacterId: 42, serverChatMetaLoaded: true })
    mocks.chatBaseState.history = kind === "history" ? [{ role: "user", content: "Earlier question" }] : []
    mocks.setMessages.mockImplementation(next => { visible = typeof next === "function" ? next(visible) : next })
    const hook = renderHook(() => useMessage())
    try {
      await act(async () => { await hook.result.current.onSubmit({ message: "Follow-up", image: "", requestOverrides: { chatMode: "normal" } }) })
      const expected = kind === "saved-greeting" || kind === "new-chat" ? 1 : 0
      expect(visible.filter(row => row.messageType === "character:greeting")).toHaveLength(expected)
      expect(mocks.saveMessageOnSuccess).toHaveBeenCalled()
      const savedHistory = mocks.setHistory.mock.calls.at(-1)?.[0]
      expect(savedHistory.filter((row: { content: string }) => row.content === greeting)).toHaveLength(expected)
    } finally {
      hook.unmount()
      Object.assign(mocks.storeState, previousStore)
      Object.assign(mocks.chatBaseState, previousBase)
      mocks.selectedCharacter = null
      mocks.effectiveAssistantState = null
      mocks.setMessages.mockImplementation(() => {})
    }
  })
})


describe("compact legacy character reasoning progress", () => {
  it("publishes held Thinking progress before any answer or persistence", async () => {
    vi.clearAllMocks()
    const previousStore = { ...mocks.storeState }
    const previousBase = { ...mocks.chatBaseState }
    const previousEffective = mocks.effectiveAssistantState
    const originalMessages = mocks.setMessages.getMockImplementation()
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const stream = vi.mocked(tldwClient.streamCharacterChatCompletion)
    const originalStream = stream.getMockImplementation()
    let visible: Message[] = [], entered = false, release!: () => void, enter!: () => void
    const opened = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    stream.mockImplementationOnce(async function* () {
      yield { choices: [{ delta: { reasoning_content: "Plan the compact reply" } }] }
      entered = true
      enter()
      await held
      yield { choices: [{ delta: { content: "Compact answer" } }] }
    })
    mocks.effectiveAssistantState = { mode: "tracked_character", kind: "character", id: "42", displayName: "Mira", avatarUrl: null, systemPromptSnapshot: null, source: "tracked" }
    mocks.loadServicePromptSnapshot.mockResolvedValue(mocks.makeSnapshot())
    Object.assign(mocks.storeState, { messages: [], serverChatId: "character-chat", serverChatCharacterId: 42, serverChatMetaLoaded: true })
    mocks.chatBaseState.history = []
    mocks.setMessages.mockImplementation(next => { visible = typeof next === "function" ? next(visible) : next; mocks.storeState.messages = visible })
    const { captureActorSettingsTarget } = await import("@/utils/actor-settings-target")
    const { captureActorSceneSendForTarget } = await import("@/hooks/chat/actor-scene-source")
    const { captureChatDraftSource, chatDraftAddressForReceipt } = await import("@/hooks/chat/chat-draft-transition")
    const requestScope = mocks.makeSnapshot().requestScope
    const owner = { kind: "native" as const, conversation_id: "character-chat", owner_key: "legacy-compact-scene-owner", request_scope: requestScope,
      validate_lease: () => true }
    const selection = { owner_key: owner.owner_key, conversation_id: owner.conversation_id, view_session_id: "legacy-compact-scene-view",
      selection_revision: 0, interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "empty" as const } }
    // Prepared scene ownership is explicit; the legacy SDK transport is not
    // silently replaced by the native H1 sender in this publisher control.
    Object.assign(mocks.storeState, { chatDraftAddress: chatDraftAddressForReceipt({ owner, view: selection }), chatDraftRevision: 0,
      historyId: mocks.chatBaseState.historyId, selectedModel: "model-1", toolChoice: "auto", temporaryChat: false })
    const source = captureChatDraftSource()
    const target = captureActorSettingsTarget({ address: mocks.storeState.chatDraftAddress as import("@/utils/chat-draft-address").ChatDraftAddress,
      owner, view: selection, requestScope, accountIsCurrent: owner.validate_lease, isCurrent: source.isCurrent })
    expect(target).not.toBeNull()
    const actorSceneSend = await captureActorSceneSendForTarget(target, target!.account, source.isCurrent)!.read()
    const hook = renderHook(() => useMessage())
    let pending!: ReturnType<typeof hook.result.current.onSubmit>
    try {
      act(() => { pending = hook.result.current.onSubmit({ message: "One compact question", image: "", requestOverrides: { chatMode: "normal" }, actorSceneSend }) })
      void pending.catch(() => {})
      await Promise.race([opened, pending.then(outcome => { throw new Error(`Compact legacy turn settled before its declared provider: ${JSON.stringify(outcome)}`) })])
      expect(entered).toBe(true)
      await waitFor(() => expect(visible.find(row => row.isBot)?.message).toContain("<think>Plan the compact reply"))
      expect(tldwClient.persistCharacterCompletion).not.toHaveBeenCalled()
      expect(mocks.saveMessageOnSuccess).not.toHaveBeenCalled()
      await act(async () => { release(); await pending })
      expect(visible.find(row => row.isBot)?.message).toBe("<think>Plan the compact reply</think>Compact answer")
      expect(tldwClient.persistCharacterCompletion).toHaveBeenCalledOnce()
    } finally {
      release?.()
      try { await pending?.catch(() => {}) } finally {
        hook.unmount()
        Object.assign(mocks.storeState, previousStore)
        Object.assign(mocks.chatBaseState, previousBase)
        mocks.effectiveAssistantState = previousEffective
        mocks.setMessages.mockReset()
        if (originalMessages) mocks.setMessages.mockImplementation(originalMessages)
        stream.mockReset()
        if (originalStream) stream.mockImplementation(originalStream)
      }
    }
  })
})
