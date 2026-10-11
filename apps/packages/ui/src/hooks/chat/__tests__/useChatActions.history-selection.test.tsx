// @vitest-environment jsdom
import React from "react"
import { getLatestChatErrorBannerEntry } from "@/components/Option/Playground/PlaygroundChatErrorBanner"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"
import { protectedEditRequestSchema } from "@/services/__tests__/history-edit-fixture"
import { roleplayDetail, roleplayAppend, roleplayCompletion, captureRoleplayFixtureRows } from "./roleplay-resume-fixture"
import { captureChatDraftSource, chatDraftAddressForReceipt } from "../chat-draft-transition"
import type { HistoryLoadReceipt } from "../useHistorySelection"
import { captureActorSceneSend, captureActorSceneSendForTarget } from "../actor-scene-source"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { usePageAssist } from "@/context"

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
  editRows: null as (() => unknown[]) | null,
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

vi.mock("@/services/actor-settings", async original => {
  const actual = await original<typeof import("@/services/actor-settings")>()
  return {
    ...actual,
    getActorSceneForChat: vi.fn(actual.getActorSceneForChat)
  }
})

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

vi.mock("@/store/option", async (original) => {
  const actual = await original<typeof import("@/store/option")>()
  return {
    ...actual,
    useStoreMessageOption: Object.assign(
      (
        select: (
          state: ReturnType<typeof actual.useStoreMessageOption.getState>
        ) => unknown
      ) => actual.useStoreMessageOption(select),
      {
        setState: actual.useStoreMessageOption.setState,
        getState: () => ({
          ...actual.useStoreMessageOption.getState(),
          ...messageStoreState.value,
          ...messageStoreState.draftFields?.(),
          ...(messageStoreState.editRows
            ? { messages: messageStoreState.editRows() }
            : {})
        })
      }
    )
  }
})

vi.mock("@/services/tldw/server-capabilities", async original => ({
  ...(await original<typeof import("@/services/tldw/server-capabilities")>()),
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false, hasRoleplayResumeV1: h1.roleplayEnabled }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    addChatMessage: addChatMessageMock,
    getChat: vi.fn(async (id: string, options?: { requireRoleplayResume?: boolean }) => options?.requireRoleplayResume ? { ...roleplayDetail(id, h1.roleplayState.get(id)!.tail), history_version: h1.roleplayState.get(id)!.historyVersion } : { id, title: "Actual owned chat", character_id: "12" }),
    appendRoleplayResumeInput: (...args: unknown[]) => h1.roleplayAppend(...args),
    completeCharacterChatTurn: (...args: unknown[]) => h1.roleplayComplete(...args),
    getChatSettings: vi.fn(async (id: string) => ({ conversation_id: id, settings: {} })),
    initialize: vi.fn(async () => null),
    getOpenAPISpec: (...args: unknown[]) => h1.spec(...args),
    getCharacter: h1.live,
    captureHistorySelection: async (id: string, request: import("@/types/history-selection").HistoryCaptureRequestV1, options: unknown) => {
      const capture = await h1.capture(id, request, options)
      const state = h1.roleplayState.get(id)
      return state ? captureRoleplayFixtureRows(capture, request, state.rows, state.content) : capture
    },
    streamChatCompletion: (...args: any[]) => h1.wire(...args),
    getConfig: vi.fn(async () => null),
    deleteMessage: vi.fn(async () => undefined),
    getMessage: h1.read,
    editMessage: h1.write,
    resolveVisualIdentityBinding: resolveVisualIdentityBindingMock
  }
}))

let nativeChatId = "tracked-chat-1"
const createdNativeChat = (body: { id?: string; character_id?: number | string }) => {
  if (!body.id) throw new Error("fixture requires the reserved client chat ID")
  nativeChatId = body.id
  if (body.character_id != null) {
    h1.roleplayEnabled = true
    h1.roleplayState.set(body.id, { tail: h1.roleplayGreeting ? "opening-" + body.id : null, historyVersion: 9, rows: [], content: new Map() })
  }
  return { id: body.id }
}
const creationReviewIdentity = async (
  handle: import("@/utils/chat-draft-address").ChatDraftHandle
) => {
  const { readChatCreationRecovery } =
    await import("@/services/pending-chat-promotion")
  const attempt = await readChatCreationRecovery(handle)
  if (!attempt) throw new Error("expected creation fixture")
  return {
    expectedOperationId: attempt.operationId,
    expectedClientChatId: attempt.clientChatId
  }
}

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
  setHistoryId: vi.fn((...args: Parameters<ReturnType<typeof actualDraftStore.getState>["setHistoryId"]>) =>
    actualDraftStore.getState().setHistoryId(...args)
  ),
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
  serverChatId: nativeChatId,
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
  setServerChatId: vi.fn((id: string | null) =>
    actualDraftStore.getState().setServerChatId(id)
  ),
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
  replyTarget: null as import("@/store/option/types").ReplyTarget | null,
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
  roleplayEnabled: false,
  roleplayGreeting: false,
  roleplayState: new Map<string, { tail: string | null; historyVersion: number; rows: import("./roleplay-resume-fixture").RoleplayFixtureRow[]; content: Map<string, import("@/types/history-selection").HistorySelectionCaptureV1["selected_content"][number]> }>(),
  roleplayAppend: vi.fn(),
  roleplayComplete: vi.fn(),
  spec: vi.fn(),
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
  lookupAdmission: vi.fn(),
  read: vi.fn(),
  write: vi.fn()
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
let actualDraftStore: typeof import("@/store/option").useStoreMessageOption

const declaredDraftAccount = (originIsCurrent: () => boolean = () => true) => {
  const signal = h1.auth.signal
  return {
    requestScope: {
      config: { serverUrl: "https://server.test", authMode: "multi-user" as const },
      userId: "alice"
    },
    isCurrent: () => !signal.aborted && originIsCurrent(),
    retryVerification: vi.fn()
  }
}

const installFreshSource = (model = "deepseek-chat") => {
  actualDraftStore.getState().startNewChatDraft(false)
  actualDraftStore.setState({ historyId: null, serverChatId: null })
  messageStoreState.value.selectedModel = model
}

const installAcceptedSource = (
  receipt: HistoryLoadReceipt,
  historyId: string | null = receipt.owner.kind === "local"
    ? receipt.owner.conversation_id
    : "history-character"
) => {
  const address = chatDraftAddressForReceipt(receipt)
  if (!address || !actualDraftStore.getState().setChatDraftAddress(address))
    throw new Error("fixture requires an accepted draft address")
  actualDraftStore.setState({
    historyId,
    serverChatId: address.kind === "native" ? address.id : null
  })
}

const installLocalSource = (id: string, profileId = "profile") => {
  if (!actualDraftStore.getState().setChatDraftAddress({ kind: "local", id, profileId }))
    throw new Error("fixture requires an accepted local draft address")
  actualDraftStore.setState({ historyId: id, serverChatId: null })
}

const readPrivateActorScene = async (
  handle: import("@/utils/chat-draft-address").ChatDraftHandle,
  snapshot: import("@/services/service-prompts").ServicePromptSnapshot
) => {
  const accountIsCurrent = () => !snapshot.scopeInvalidatedSignal.aborted
  const target = captureActorSettingsTarget({
    address: handle.address,
    requestScope: snapshot.requestScope,
    accountIsCurrent,
    isCurrent: accountIsCurrent
  })
  const captured = captureActorSceneSendForTarget(target, target?.account)
  if (!captured) throw new Error("fixture requires a qualified private scene")
  return captured.read()
}

const bridgeDraftFields = () => {
  const state = actualDraftStore.getState()
  return {
    chatDraftAddress: state.chatDraftAddress,
    chatDraftRevision: state.chatDraftRevision,
    chatDraftTransition: state.chatDraftTransition,
    setChatDraftAddress: state.setChatDraftAddress,
    historyId: state.historyId,
    serverChatId: state.serverChatId,
    setHistoryId: state.setHistoryId,
    setServerChatId: state.setServerChatId
  }
}

const makeController = () => {
  const account = declaredDraftAccount()
  let current: any = {
    owner: {
      kind: "native",
      conversation_id: nativeChatId,
      request_scope: {
        config: { serverUrl: "https://server.test", authMode: "multi-user" },
        userId: "alice"
      },
      validate_lease: account.isCurrent
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
    draftAccount: account,
    getCurrent: () => current,
    fence: () => {
      const origin = current
      return () => current === origin
    },
    followResult: vi.fn(async () => true),
    refreshRecovery: vi.fn(async () => {}),
    retireSameTuple: () => {
      current = { ...current, owner: { ...current.owner }, view: { ...current.view } }
    },
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

  messageStoreState.editRows = null
  nativeChatId = "tracked-chat-1"
  h1.roleplayEnabled = false
  h1.roleplayGreeting = false
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
    const reply = roleplayCompletion(id, body)
    const state = h1.roleplayState.get(id)!
    state.tail = reply.tldw_roleplay_resume_v1.fence.tail.message_id
    state.historyVersion = reply.tldw_roleplay_resume_v1.fence.history_version
    state.rows.push({ id: reply.assistant_message_id, revision: "1", parent_id: reply.user_message_id, role: "assistant", settled: true, message: reply.assistant_content })
    return reply
  })
  h1.connected = false
  h1.recover.mockReset()
  h1.dismiss.mockReset().mockResolvedValue(true)
  h1.lookupRecoveries.mockReset()
  h1.lookupAdmission.mockReset()
  vi.clearAllMocks()
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  vi.mocked(api.getChat).mockReset().mockImplementation(async (id, options) => options?.requireRoleplayResume ? { ...roleplayDetail(id, h1.roleplayState.get(id)!.tail), history_version: h1.roleplayState.get(id)!.historyVersion } : { id, title: "Actual owned chat", character_id: "12" })
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
  const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  actualDraftStore = draftStore
  draftStore.getState().startNewChatDraft(false)
  messageStoreState.draftFields = bridgeDraftFields
  const { memory } = await import("./local-history-fixture")
  await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
  const actualScenes = await vi.importActual<typeof import("@/services/actor-settings")>("@/services/actor-settings")
  vi.mocked((await import("@/services/actor-settings")).getActorSceneForChat)
    .mockReset().mockImplementation(actualScenes.getActorSceneForChat)
  h1.beforeModel.mockReset()
  messageStoreState.value.selectedModel = "deepseek-chat"
  messageStoreState.value.toolChoice = "auto"
  h1.auth = new AbortController()
  const actual = await vi.importActual<any>("@/hooks/chat-modes/normalChatMode")
  normalChatModeMock.mockImplementation(actual.normalChatMode)
  h1.controller = makeController()
  installAcceptedSource(h1.controller.getCurrent())
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
            conversation_id: nativeChatId,
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
  installLocalSource("history-character")
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: [
      { id: "plain-message", isBot: false, message: "old", sources: [] }
    ]
  }
  messageStoreState.editRows = () => options.messages
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    expect(await result.current.editMessage(0, "edited", false, false)).toEqual(
      { status: "submitted" }
    )
  })
  expect(options.setMessages).toHaveBeenCalledWith([
    expect.objectContaining({ id: "plain-message", message: "edited" })
  ])
  expect(
    options.setHistory.mock.calls
      .at(-1)![0]
      .map((row: { content: string }) => row.content)
  ).toEqual(["edited"])
  expect(helpers.updateMessageById).toHaveBeenCalledWith(
    "history-character",
    "plain-message",
    "edited"
  )
})

it("deletes an ordinary local message while the mounted history controller is idle", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const current = h1.controller.getCurrent()
  current.status = "idle"
  current.owner = null
  current.view = null
  installLocalSource("history-character")
  const options = {
    ...ordinaryOptions(),
    serverChatId: null,
    messages: [{ id: "plain-message", isBot: false, message: "old", sources: [] }]
  }
  messageStoreState.editRows = () => options.messages
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.deleteMessage(0)
  })
  expect(helpers.removeMessageById).toHaveBeenCalledWith(
    "history-character", "plain-message", undefined, { validate_lease: expect.any(Function) }
  )
})

it("refuses legacy server deletion without canonical owner authority", async () => {
  const helpers = await import("@/db/dexie/helpers")
  const mirror = await import("@/db/dexie/server-chat-mirror")
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  const current = h1.controller.getCurrent()
  current.capture = null
  current.owner = null
  const options = { ...ordinaryOptions(), serverChatId: nativeChatId,
    messages: [{ id: "local-message", serverMessageId: "server-message", serverMessageVersion: 1, isBot: true, message: "answer", sources: [] }] }
  const { result } = renderHook(() => useChatActions(options as any))
  await expect(result.current.deleteMessage(0)).rejects.toThrow("native_history_mutation_unavailable")
  expect(api.getMessage).not.toHaveBeenCalled()
  expect(api.deleteMessage).not.toHaveBeenCalled()
  expect(mirror.removeAcknowledgedServerMirrorMessage).not.toHaveBeenCalled()
  expect(helpers.removeMessageById).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
})

it("routes ordinary regeneration while the mounted history controller is idle", async () => {
  const controller = h1.controller
  const current = controller.getCurrent()
  // Retain the saved scene's locator without granting this idle controller H1 authority.
  const receipt: HistoryLoadReceipt = { owner: current.owner, view: current.view }
  if (receipt.owner.kind !== "native")
    throw new Error("fixture requires the original native receipt")
  current.status = "idle"
  current.owner = null
  current.view = null
  current.capture = null
  const historyIsCurrent = controller.fence()
  const source = captureChatDraftSource(() =>
    h1.controller === controller && historyIsCurrent()
  )
  const target = captureActorSettingsTarget({
    address: actualDraftStore.getState().chatDraftAddress,
    ...receipt,
    requestScope: receipt.owner.request_scope,
    accountIsCurrent: receipt.owner.validate_lease,
    isCurrent: source.isCurrent
  })
  const captured = captureActorSceneSendForTarget(target, target?.account, source.isCurrent)
  if (!captured) throw new Error("fixture requires the original saved scene")
  const actorSceneSend = await captured.read()
  normalChatModeMock.mockResolvedValue({ state: "submitted" } as any)
  const options = ordinaryOptions()
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "again", image: "", isRegenerate: true, actorSceneSend })
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

it.each([
  { label: "active zero", saved: { chatGenerationOverride: { enabled: true, temperature: 0, top_p: 0 } }, omitTemperature: true, omitTopP: true },
  { label: "active finite", saved: { chatGenerationOverride: { temperature: 0.71, top_p: 0.45 } }, omitTemperature: true, omitTopP: true },
  { label: "null inherits model", saved: { chatGenerationOverride: { enabled: true, temperature: null, top_p: null } }, omitTemperature: false, omitTopP: false },
  { label: "disabled inherits model", saved: { chatGenerationOverride: { enabled: false, temperature: 0, top_p: 0 } }, omitTemperature: false, omitTopP: false },
  { label: "empty preferred wins legacy", saved: { chatGenerationOverride: {}, generationOverrides: { temperature: 0, top_p: 0 } }, omitTemperature: false, omitTopP: false },
  { label: "disabled preferred wins legacy", saved: { chatGenerationOverride: { enabled: false }, generationOverrides: { temperature: 0, top_p: 0 } }, omitTemperature: false, omitTopP: false },
  { label: "legacy active", saved: { generationOverrides: { temperature: 0, top_p: 0.6 } }, omitTemperature: true, omitTopP: true },
  { label: "one field inherits", saved: { chatGenerationOverride: { temperature: 0.4, top_p: null } }, omitTemperature: true, omitTopP: false },
  { label: "no override", saved: {}, omitTemperature: false, omitTopP: false }
])("native sampler uses the saved chat independently of model defaults: $label", async ({ saved, omitTemperature, omitTopP }) => {
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  vi.mocked(api.getChatSettings).mockResolvedValueOnce({ conversation_id: nativeChatId, settings: saved })
  const options = { ...createHookOptions(), currentChatModelSettings: { apiProvider: "openai", temperature: 0.23, topP: 0.9 } }
  h1.controller.getCurrent().owner.scope = { type: "workspace", workspaceId: "workspace-a" }
  installAcceptedSource(h1.controller.getCurrent())
  const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  await act(async () => { await result.current.onSubmit({ message: "saved sampler", image: "" }) })
  expect(h1.wire).toHaveBeenCalledOnce()
  const request = h1.wire.mock.calls[0][0]
  expect(Object.hasOwn(request, "temperature")).toBe(!omitTemperature)
  expect(Object.hasOwn(request, "top_p")).toBe(!omitTopP)
  if (!omitTemperature) expect(request.temperature).toBe(0.23)
  if (!omitTopP) expect(request.top_p).toBe(0.9)
  const { tldw_history_selection_v1: selection, tldw_history_branch, ...preparedPayload } = request
  expect(tldw_history_branch).toBe(false)
  const { historyDigest } = await import("@/db/dexie/history-selection")
  expect(selection.request_context_digest).toBe(historyDigest(preparedPayload))
  expect(api.getChatSettings).toHaveBeenCalledWith(nativeChatId, expect.objectContaining({
    scope: { type: "workspace", workspaceId: "workspace-a" },
    requestScope: h1.controller.getCurrent().owner.request_scope,
    signal: expect.any(AbortSignal)
  }))
})

it.each([
  { label: "foreign owner", raw: { conversation_id: "foreign", settings: {} } },
  { label: "missing ID", raw: { settings: {} } },
  { label: "null settings", raw: { conversation_id: "tracked-chat-1", settings: null } },
  { label: "array settings", raw: { conversation_id: "tracked-chat-1", settings: [] } },
  { label: "read failure", raw: undefined }
])("native sampler refuses $label before history capture or provider dispatch", async ({ raw }) => {
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  vi.mocked(api.getChatSettings).mockImplementationOnce(async () => {
    if (raw === undefined) throw new Error("settings unavailable")
    return raw as Awaited<ReturnType<typeof api.getChatSettings>>
  })
  const options = createHookOptions()
  const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  await act(async () => { await result.current.onSubmit({ message: "keep this input", image: "" }) })
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(options.notification.error).toHaveBeenCalled()
})

it.each(["view", "account", "owner lease"])("native sampler does not continue a held settings read after %s retirement", async boundary => {
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  let release!: () => void
  vi.mocked(api.getChatSettings).mockImplementationOnce(() => new Promise(resolve => {
    release = () => resolve({ conversation_id: nativeChatId, settings: { chatGenerationOverride: { temperature: 0 } } })
  }))
  const options = createHookOptions()
  let ownerLive = true
  h1.controller.getCurrent().owner.validate_lease = () => ownerLive
  const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  await act(async () => {
    const sending = result.current.onSubmit({ message: "held input", image: "" })
    await waitFor(() => expect(api.getChatSettings).toHaveBeenCalledOnce())
    if (boundary === "view") h1.controller.navigate()
    else if (boundary === "account") h1.auth.abort()
    else ownerLive = false
    release()
    await sending
  })
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
})

it.each(["view", "account", "owner lease"])("native sampler suppresses a failed held read after %s retirement", async boundary => {
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  let rejectRead!: (error: Error) => void
  vi.mocked(api.getChatSettings).mockImplementationOnce(() => new Promise((_, reject) => { rejectRead = reject }))
  const options = createHookOptions()
  const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  let sending!: Promise<unknown>
  await act(async () => {
    sending = result.current.onSubmit({ message: "retained input", image: "" })
    await waitFor(() => expect(api.getChatSettings).toHaveBeenCalledOnce())
  })
  if (boundary === "view") h1.controller.retireSameTuple()
  else if (boundary === "account") h1.auth.abort()
  else h1.controller.getCurrent().owner.validate_lease = () => false
  await act(async () => { rejectRead(new Error("read failed")); await sending })
  expect(h1.capture).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(options.notification.error).not.toHaveBeenCalled()
})

it.each(["capture", "dispatch journal"])("native sampler refuses an identical H1 replacement during its %s", async stage => {
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  if (stage === "capture") h1.capture.mockImplementationOnce(async (_id, request) => {
    entered(); await held; return captureFor(request.view)
  })
  else h1.recover.mockImplementationOnce(async () => { entered(); await held })
  const options = createHookOptions()
  const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  let sending!: Promise<unknown>
  await act(async () => {
    sending = result.current.onSubmit({ message: "original input", image: "" })
    await Promise.race([ready, sending.then(outcome => { throw new Error(`Sender finished before the ${stage} hold: ${JSON.stringify(outcome)}`) })])
  })
  h1.controller.retireSameTuple()
  await act(async () => { release(); await sending })
  expect(h1.wire).not.toHaveBeenCalled()
})

it("mounted native character sends A1 selection and only new input, following the owner ACK", async () => {
  const options = createHookOptions()
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
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).toHaveBeenCalledOnce()
  const request = h1.wire.mock.calls[0][0]
  expect(request).toMatchObject({
    save_to_db: true,
    conversation_id: nativeChatId,
    tldw_history_branch: false,
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
    nativeChatId,
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
      conversation_id: nativeChatId,
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
  installFreshSource()
  const options = { ...ordinaryOptions(), historyId: target === "clear cached target" ? "cached-history" : null, serverChatId: null, messages: [], history: [] }
  let current: any = { owner: null, view: null, status: "idle" }
  h1.controller = {
    draftAccount: declaredDraftAccount(),
    getCurrent: () => current,
    fence: () => { const origin = current; return () => current === origin },
    loadConversation: vi.fn(async (target, _reference, onLoaded) => {
      expect(target).toMatchObject({ serverChatId: nativeChatId })
      expect(options.setServerChatId).not.toHaveBeenCalled()
      current = makeController().getCurrent()
      current.view = { ...current.view, cursor: { kind: "empty" } }
      current.capture = captureFor(current.view)
      onLoaded({ owner: current.owner, view: current.view })
      return true
    }),
    followResult: vi.fn(async () => true)
  }
  createChatMock.mockImplementation(createdNativeChat)
  options.setServerChatId.mockImplementation(() => {
    expect(current.owner.kind).toBe("native")
    actualDraftStore.getState().setServerChatId(nativeChatId)
  })
  h1.wire.mockImplementation(async function* (request) {
    expect(addChatMessageMock).toHaveBeenCalledOnce()
    expect(request.save_to_db).toBe(false)
    yield { choices: [{ delta: { content: "new answer" } }] }
  })
  const hook = renderHook(() => useChatActions(options as any))
  await act(async () => { expect(await hook.result.current.onSubmit({ message: "first", image: "", ...(target === "clear cached target" ? { requestOverrides: { historyId: null, serverChatId: null } } : {}) })).toEqual({ status: "submitted" }) })
  const draftStore = (await vi.importActual<typeof import("@/store/option")>("@/store/option")).useStoreMessageOption
  expect(draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(options.setServerChatId).toHaveBeenCalledWith(nativeChatId)
  expect(h1.saveHistory).not.toHaveBeenCalled()
  expect(addChatMessageMock).toHaveBeenCalledTimes(2)
  expect(h1.wire).toHaveBeenCalledOnce()
})

it("fresh H1 tracked Persona refuses before creation, mirror, admission, or provider dispatch", async () => {
  installFreshSource()
  const originalAddress = actualDraftStore.getState().chatDraftAddress
  const current = { owner: null, view: null, capture: null, bookmarkScope: null, status: "idle" }
  h1.controller = {
    draftAccount: declaredDraftAccount(),
    getCurrent: () => current,
    fence: () => () => true,
    loadConversation: vi.fn(),
    followResult: vi.fn()
  }
  const options = {
    ...ordinaryOptions(),
    historyId: null,
    serverChatId: null,
    messages: [],
    history: [],
    selectedAssistant: {
      kind: "persona" as const,
      id: "garden-helper",
      name: "Garden Helper",
      metadata: { selectionMode: "tracked" as const }
    }
  }
  createChatMock.mockResolvedValue({
    id: "persona-created",
    title: "Garden Helper",
    assistant_kind: "persona",
    assistant_id: "garden-helper",
    persona_memory_mode: "read_only"
  })
  normalChatModeMock.mockResolvedValue({ status: "submitted" })
  const hook = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))
  let outcome: unknown
  await act(async () => {
    outcome = await hook.result.current.onSubmit({ message: "Keep this Persona input", image: "" })
  })
  expect(outcome).toMatchObject({ status: "failed", errorMessage: "native_history_persona_unsupported" })
  expect(options.notification.error).toHaveBeenCalledWith(expect.objectContaining({
    description: expect.stringMatching(/persona/i)
  }))
  expect(options.notification.error.mock.lastCall?.[0].description).not.toContain("native_history_persona_unsupported")
  expect(createChatMock).not.toHaveBeenCalled()
  expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled()
  expect(h1.saveHistory).not.toHaveBeenCalled()
  expect(h1.controller.loadConversation).not.toHaveBeenCalled()
  expect(h1.capture).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.localAppend).not.toHaveBeenCalled()
  expect(normalChatModeMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(options.setHistoryId).not.toHaveBeenCalled()
  expect(options.setServerChatId).not.toHaveBeenCalled()
  expect(actualDraftStore.getState()).toMatchObject({ historyId: null, serverChatId: null })
  expect(actualDraftStore.getState().chatDraftAddress).toBe(originalAddress)
  hook.unmount()
})

it("first durable ordinary send creates its local owner and admits before inference without a server conversation", async () => {
  installFreshSource()
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
    draftAccount: declaredDraftAccount(),
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
  const freshOptions = () => {
    installFreshSource()
    return {
      ...ordinaryOptions(),
      historyId: null,
      serverChatId: null,
      messages: [],
      history: []
    }
  }
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
      draftAccount: declaredDraftAccount(),
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
      draftAccount: declaredDraftAccount(),
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
    const { result } = renderActions({ ...ordinaryOptions(), serverChatId: nativeChatId })
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })

    expect(reset).not.toHaveBeenCalled()
    expect(JSON.stringify(h1.wire.mock.calls[0][0].messages)).toContain("old question")
  })
})

it("a first-create acknowledgement after navigation never installs or dispatches into the new view", async () => {
  installFreshSource()
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
    draftAccount: declaredDraftAccount(() => current),
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
    installFreshSource()
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
      draftAccount: declaredDraftAccount(),
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
  conversation_id: nativeChatId,
  input_message_id: "native-input",
  input_message_revision: "native-r",
  selection_digest: request.tldw_history_selection_v1.selection_digest,
  messages: request.tldw_history_selection_v1.messages,
  originating_selection_revision:
    request.tldw_history_selection_v1.selection_revision
})

const savedContextRefusal = (
  code = "unsupported_history_context_saved_settings"
) =>
  Object.assign(new Error("HTTP 409"), {
    status: 409,
    details: { detail: { status: "unsupported_history_capability", code } }
  })

it.each([
  "saved_settings",
  "world_books",
  "skills_registry",
  "snapshot_invalid",
  "prompt_template"
])(
  "native confirmed context refusal %s keeps exact input without an uncertain send or transcript row",
  async (suffix) => {
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* () {
      yield await Promise.reject(
        savedContextRefusal(`unsupported_history_context_${suffix}`)
      )
    })
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    let submitted: unknown
    await act(async () => {
      submitted = await result.current.onSubmit({
        message: "  exact question\ntrailing spaces  ",
        image: ""
      })
    })
    expect(submitted).toMatchObject({
      status: "failed",
      errorMessage: "native_history_context_refused"
    })
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      persistence: "server",
      state: "refused",
      input_text: "  exact question\ntrailing spaces  ",
      result_text: ""
    })
    const retained = h1.recover.mock.lastCall?.[2]
    expect(retained.admission).toBeUndefined()
    expect(retained.outcome).toBeUndefined()
    expect(retained.input_id).toBeUndefined()
    expect(retained.assistant_id).toBeUndefined()
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.notification.error.mock.lastCall?.[0].description).toContain(
      "Your message was not added"
    )
    expect(h1.controller.refreshRecovery).toHaveBeenCalledOnce()
    expect(h1.wire).toHaveBeenCalledOnce()
    expect(createChatMock).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
  }
)

it.each([
  {
    label: "general conflict",
    status: 409,
    detail: { status: "stale_selection", code: "stale_source" }
  },
  {
    label: "another capability",
    status: 409,
    detail: {
      status: "unsupported_history_capability",
      code: "sync_owner_unsupported"
    }
  },
  {
    label: "future context",
    status: 409,
    detail: {
      status: "unsupported_history_capability",
      code: "unsupported_history_context_future_effect"
    }
  },
  {
    label: "wrong HTTP status",
    status: 502,
    detail: {
      status: "unsupported_history_capability",
      code: "unsupported_history_context_world_books"
    }
  },
  {
    label: "text HTTP status",
    status: "409",
    detail: {
      status: "unsupported_history_capability",
      code: "unsupported_history_context_world_books"
    }
  },
  {
    label: "missing capability",
    status: 409,
    detail: { code: "unsupported_history_context_world_books" }
  },
  {
    label: "text-only envelope",
    status: 409,
    detail: "unsupported_history_context_world_books"
  }
])("native $label retains uncertain input", async ({ status, detail }) => {
  const options = createHookOptions()
  h1.wire.mockImplementation(async function* () {
    yield await Promise.reject(
      Object.assign(new Error("unconfirmed transport"), {
        status,
        details: { detail }
      })
    )
  })
  const { result } = renderHook(() =>
    useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
  )
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
    state: "unknown",
    input_text: "next"
  })
  expect(h1.dismiss).not.toHaveBeenCalled()
  expect(
    options.notification.error.mock.lastCall?.[0].description
  ).not.toContain("Your message was not added")
  expect(h1.wire).toHaveBeenCalledOnce()
})

it.each([
  "admission",
  "content",
  "admission-and-content",
  "empty-content",
  "role-only",
  "admission-name-collision",
  "settings-read-name-collision"
])(
  "a context-shaped error after %s cannot erase observed native evidence",
  async (observed) => {
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* (request) {
      if (
        observed === "admission" ||
        observed === "admission-and-content" ||
        observed === "admission-name-collision" ||
        observed === "settings-read-name-collision"
      )
        yield { tldw_history_admission_v1: nativeAdmission(request) }
      if (observed === "content" || observed === "admission-and-content")
        yield { choices: [{ delta: { content: "kept partial" } }] }
      if (observed === "empty-content")
        yield { choices: [{ delta: { content: "" } }] }
      if (observed === "role-only")
        yield { choices: [{ delta: { role: "assistant" } }] }
      if (observed === "admission-name-collision")
        yield await Promise.reject(new Error("native_history_context_refused"))
      if (observed === "settings-read-name-collision")
        yield await Promise.reject(new Error("native_history_settings_unverified"))
      yield await Promise.reject(savedContextRefusal())
    })
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(h1.recover.mock.lastCall?.[2]).toMatchObject({
      state:
        observed === "admission" || observed === "admission-name-collision" || observed === "settings-read-name-collision"
          ? "accepted_unsent"
          : observed === "empty-content" || observed === "role-only"
            ? "unknown"
            : "generated_unsaved",
      input_text: "next"
    })
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(
      options.notification.error.mock.lastCall?.[0].description
    ).not.toMatch(/Your message (?:was not added|has not been sent)/)
    expect(h1.wire).toHaveBeenCalledOnce()
  }
)

it.each(["current", "navigation", "account"])(
  "a confirmed native refusal keeps exact input across a cold %s view and preserves other pending turns",
  async (boundary) => {
    const { memory } = await import("./local-history-fixture")
    for (const table of Object.values(memory))
      if (table?.rows) table.rows.clear()
    await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
    const actual = await vi.importActual<
      typeof import("@/db/dexie/history-selection")
    >("@/db/dexie/history-selection")
    const current = h1.controller.getCurrent()
    await actual.saveHistoryTurnRecovery(current.bookmarkScope, current.view, {
      operation_id: "other-uncertain-turn",
      persistence: "server",
      origin_view: current.view,
      owner_key: current.view.owner_key,
      conversation_id: current.view.conversation_id,
      selection_digest: "other-selection",
      request_context_digest: "other-context",
      created_at: 1,
      input_text: "earlier uncertain question",
      input_images: [],
      result_text: "",
      state: "unknown"
    })
    h1.recover.mockImplementation(actual.saveHistoryTurnRecovery)
    h1.dismiss.mockImplementation(actual.dismissHistoryTurnRecovery)
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* () {
      if (boundary === "navigation") h1.controller.navigate()
      if (boundary === "account") h1.auth.abort()
      options.setMessages.mockClear()
      yield await Promise.reject(savedContextRefusal())
    })
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    await act(async () => {
      await result.current.onSubmit({
        message: "  cold exact input  ",
        image: ""
      })
    })
    const reopened = await actual.loadHistoryTurnRecoveries(
      { ...current.bookmarkScope, client_session_id: "cold-reopen" },
      current.view
    )
    expect(reopened).toHaveLength(2)
    expect(
      reopened.find(
        (entry) => entry.turn.operation_id !== "other-uncertain-turn"
      )?.turn
    ).toMatchObject({
      state: "refused",
      input_text: "  cold exact input  ",
      result_text: ""
    })
    expect(
      reopened.find(
        (entry) => entry.turn.operation_id === "other-uncertain-turn"
      )?.turn
    ).toMatchObject({
      state: "unknown",
      input_text: "earlier uncertain question"
    })
    expect(
      await actual.loadHistoryTurnRecoveries(current.bookmarkScope, {
        ...current.view,
        owner_key: "other-account"
      })
    ).toEqual([])
    expect(options.setMessages).not.toHaveBeenCalled()
    if (boundary !== "current")
      expect(options.notification.error).not.toHaveBeenCalled()
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(h1.wire).toHaveBeenCalledOnce()
  }
)

it.each(["journal write", "recovery refresh"])(
  "a failed %s preserves the confirmed native error and original transport cause",
  async (failure) => {
    const { sendNativeHistoryCharacter } =
      await import("../native-history-character-send")
    const original = savedContextRefusal()
    const options = createHookOptions()
    h1.wire.mockImplementation(async function* () {
      yield await Promise.reject(original)
    })
    if (failure === "journal write")
      h1.recover.mockImplementation(async (_scope, _view, turn) => {
        if (turn.state === "refused") throw new Error("journal unavailable")
      })
    else
      h1.controller.refreshRecovery.mockRejectedValueOnce(
        new Error("refresh unavailable")
      )
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const caught = await sendNativeHistoryCharacter({
      controller: h1.controller,
      surface: "playground",
      originIsCurrent: () => true,
      signal: new AbortController().signal,
      temporary: false,
      historyId: options.historyId,
      serverChatId: options.serverChatId,
      model: options.selectedModel,
      settings: { apiProvider: "openai" } as Parameters<
        typeof sendNativeHistoryCharacter
      >[0]["settings"],
      message: "private input never logged",
      image: "",
      t: options.t as import("i18next").TFunction,
      setServerChatId: options.setServerChatId,
      setMessages: options.setMessages,
      releaseActivity: vi.fn()
    }).catch((error) => error)
    expect(caught).toMatchObject({
      message: "native_history_context_refused",
      cause: original
    })
    expect(warn).toHaveBeenCalledOnce()
    expect(warn.mock.calls.flat().join(" ")).not.toContain("private input")
    expect(h1.dismiss).not.toHaveBeenCalled()
    expect(h1.wire).toHaveBeenCalledOnce()
  }
)

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
  expect(await result.current.regenerateLastMessage()).toEqual({ status: "skipped", reason: "message_changed" })
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
          failure === "wrong-conversation" ? "other" : nativeChatId
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
      tldw_conversation_id: nativeChatId
    }
  })
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(h1.controller.followResult).not.toHaveBeenCalled()
  expect(h1.dismiss).toHaveBeenCalledWith(
    expect.anything(),
    expect.objectContaining({ conversation_id: nativeChatId }),
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
        tldw_conversation_id: nativeChatId
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
      tldw_conversation_id: nativeChatId
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
    installFreshSource(model)
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
    const load = vi.fn(async (_target, _reference, onLoaded) => {
      const ready = makeController().getCurrent()
      ready.view.cursor = { kind: "empty" }
      ready.capture = captureFor(ready.view)
      current = ready
      onLoaded({ owner: current.owner, view: current.view })
      return true
    })
    h1.controller = {
      draftAccount: declaredDraftAccount(() => origin),
      getCurrent: () => current,
      fence: () => () => origin,
      loadConversation: load,
      followResult: vi.fn(async () => true),
      refreshRecovery: vi.fn()
    }
    createChatMock.mockImplementation(async body => {
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
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => {
      await result.current.onSubmit({ message: "next", image: "" })
    })
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(createChatMock).toHaveBeenCalledWith(
      defaults
        ? { id: nativeChatId, character_id: "12", provider, model: modelId }
        : { id: nativeChatId, character_id: "12", provider, model: modelId, temperature: 0, top_p: 0 },
      expect.objectContaining({ requestScope: expect.objectContaining({ userId: "alice" }) })
    )
    if (navigate) {
      expect(h1.wire).not.toHaveBeenCalled()
      expect(load).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
    } else {
      expect(h1.roleplayAppend).toHaveBeenCalledWith(nativeChatId, expect.objectContaining({ content: "next", expected: expect.objectContaining({ tail: { message_id: null, message_version: null } }) }), expect.objectContaining({ requestScope: expect.objectContaining({ userId: "alice" }) }))
      expect(h1.roleplayComplete).toHaveBeenCalledOnce()
      expect(h1.wire).not.toHaveBeenCalled()
      expect(options.setServerChatId).toHaveBeenCalledWith(nativeChatId)
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
  installAcceptedSource(h1.controller.getCurrent())
  h1.wire.mockImplementation(async function* (request) {
    workspace.workspaceId = "new-workspace"
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield {
      tldw_message_id: "owner-result",
      tldw_conversation_id: nativeChatId
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
  messageStoreState.value.toolChoice = "required"
  const options = { ...createHookOptions(), toolChoice: "required" as const }
  const { result } = renderHook(() => useChatActions(options as any))
  await act(async () => {
    await result.current.onSubmit({ message: "next", image: "" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(options.notification.error).toHaveBeenCalledWith(
    expect.objectContaining({
      description: "Client tools are not supported in saved character chat. Set tools to None before starting."
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
  installAcceptedSource(current)
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
    messageStoreState.editRows = () => options.messages
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
    messageStoreState.editRows = () => options.messages
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
  messageStoreState.editRows = () => options.messages
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
  messageStoreState.editRows = () => options.messages
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
  const options = {
    ...ordinaryOptions(),
    serverChatId: nativeChatId,
    messages: visible
  }
  messageStoreState.editRows = () => options.messages
  const { result } = renderHook(() => useChatActions(options as any))
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
  installAcceptedSource(current)
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
    messageStoreState.draftFields = bridgeDraftFields
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
    h1.controller = Object.assign(controller.result.current, { draftAccount: declaredDraftAccount() })
    installAcceptedSource(controller.result.current.getCurrent() as HistoryLoadReceipt)
    const options: any = {
      ...ordinaryOptions(),
      serverChatId: null,
      historyId: "history-1",
      compareModeActive: comparison,
      messages: (await import("@/db/dexie/helpers")).formatToMessage(source)
    }
    options.setHistoryId.mockImplementation((id: string) => {
      options.historyId = id
      actualDraftStore.getState().setHistoryId(id)
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
    h1.controller = Object.assign(controller.result.current, { draftAccount: declaredDraftAccount() })
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
    messageStoreState.editRows = () => options.messages
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
  installAcceptedSource(h1.controller.getCurrent(), null)
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
  h1.roleplayGreeting = true
  const { useStoreMessageOption: draftStore } = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  const { useStoreChatModelSettings: modelStore } = await import("@/store/model")
  const { useMcpToolsStore: toolsStore } = await import("@/store/mcp-tools")
  toolsStore.getState().setTools([])
  installFreshSource()
  const sourceDraft = draftStore.getState().chatDraftAddress
  messageStoreState.draftFields = bridgeDraftFields
  const { memory } = await import("./local-history-fixture")
  for (const name of ["forkOperations", "historySelections"]) memory[name].rows.clear()
  const { useHistorySelection } = await import("../useHistorySelection")
  const onCapture = vi.fn()
  const selection = renderHook(() => useHistorySelection({ onCapture }))
  h1.controller = Object.assign(selection.result.current, { draftAccount: declaredDraftAccount() })
  const options = { ...createHookOptions(), historyId: null, serverChatId: null, serverChatCharacterId: null,
    selectedAssistant: { kind: "character", id: "12", name: "Saved", metadata: { selectionMode: "tracked" } } }
  createChatMock.mockImplementation(createdNativeChat)
  h1.capture.mockImplementation(async (id, request) => openingCapture(id, request))
  h1.wire.mockImplementation(async function* (request) {
    yield { tldw_history_admission_v1: nativeAdmission(request) }
    yield { tldw_message_id: "native-result", tldw_conversation_id: nativeChatId }
  })
  const hook = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
  return { draftStore, modelStore, toolsStore, sourceDraft, memory, onCapture, selection, options, hook }
}

it("a refused first native turn preserves acknowledged bound creation and its exact original input", async () => {
  const mounted = await mountedNativeOpening()
  const { readChatCreationRecovery } =
    await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const actual = await vi.importActual<
    typeof import("@/db/dexie/history-selection")
  >("@/db/dexie/history-selection")
  h1.recover.mockImplementation(actual.saveHistoryTurnRecovery)
  h1.roleplayAppend.mockRejectedValueOnce(Object.assign(new Error("HTTP 409"), {
    status: 409, details: { detail: { status: "resume_conflict", code: "settings_version_changed" } }
  }))
  let submitted: unknown
  await act(async () => {
    submitted = await mounted.hook.result.current.onSubmit({
      message: "  FIRST ORIGINAL INPUT  ",
      image: ""
    })
  })
  const selected = mounted.selection.result.current.getCurrent()
  const owner = selected.owner
  expect(submitted).toMatchObject({
    status: "failed",
    errorMessage: "native_history_context_refused"
  })
  expect(owner?.kind).toBe("native")
  if (
    owner?.kind !== "native" ||
    !mounted.sourceDraft ||
    !selected.bookmarkScope ||
    !selected.view
  )
    throw new Error("expected accepted native owner")
  const creation = await readChatCreationRecovery({
    address: mounted.sourceDraft,
    ownerHash: chatDraftOwnerKey(owner.request_scope)
  })
  expect(creation).toMatchObject({
    state: "bound",
    chatId: nativeChatId,
    inputText: "  FIRST ORIGINAL INPUT  "
  })
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({
    kind: "native",
    id: nativeChatId
  })
  const [recovered] = await actual.loadHistoryTurnRecoveries(
    { ...selected.bookmarkScope, client_session_id: "reopened" },
    selected.view
  )
  expect(recovered.turn).toMatchObject({
    state: "refused",
    input_text: "  FIRST ORIGINAL INPUT  "
  })
  expect(
    mounted.options.notification.error.mock.lastCall?.[0].description
  ).toContain("Your message was not added")
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(h1.roleplayAppend).toHaveBeenCalledOnce()
  expect(h1.roleplayComplete).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.dismiss).not.toHaveBeenCalled()
  mounted.hook.unmount()
  mounted.selection.unmount()
})

it("native sampler retires an actual same-ID H1 read without recapture or dispatch", async () => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const controller = mounted.selection.result.current
  await act(async () => { expect(await controller.loadConversation({ serverChatId: nativeChatId })).toBe(true) })
  const original = controller.getCurrent()
  const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
  let release!: () => void
  vi.mocked(api.getChatSettings).mockImplementationOnce(() => new Promise(resolve => {
    release = () => resolve({ conversation_id: nativeChatId, settings: { chatGenerationOverride: { temperature: 0 } } })
  }))
  h1.capture.mockClear(); h1.wire.mockClear(); h1.recover.mockClear()
  let sending!: Promise<unknown>
  await act(async () => {
    sending = sendNativeHistoryCharacter({
      controller, surface: "playground", originIsCurrent: controller.fence(),
      signal: new AbortController().signal, temporary: false,
      serverChatId: nativeChatId, characterId: "12", model: "deepseek:deepseek-chat", currentModel: "deepseek:deepseek-chat",
      settings: { apiProvider: "deepseek", temperature: 0.23, topP: 0.9 },
      message: "held next", image: "", t: mounted.options.t,
      setServerChatId: mounted.options.setServerChatId, setMessages: mounted.options.setMessages, releaseActivity: vi.fn()
    }).catch(error => error)
    await waitFor(() => expect(api.getChatSettings).toHaveBeenCalledOnce())
  })
  expect(h1.capture).not.toHaveBeenCalled()
  await act(async () => { expect(await controller.loadConversation({ serverChatId: nativeChatId })).toBe(true) })
  const next = controller.getCurrent()
  expect(next.owner).not.toBe(original.owner)
  expect(original.owner?.kind === "native" && original.owner.validate_lease()).toBe(false)
  const loadCaptureCount = h1.capture.mock.calls.length
  let error: unknown
  await act(async () => { release(); error = await sending })
  expect(error).toMatchObject({ message: "stale_selection" })
  expect(h1.capture).toHaveBeenCalledTimes(loadCaptureCount)
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
  expect(controller.getCurrent().owner).toBe(next.owner)
  mounted.hook.unmount(); mounted.selection.unmount()
})

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

it.each(["same-chat", "newer-chat"])("late native first-load retirement preserves a newer load of %s", async nextChoice => {
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
    expect(await mounted.selection.result.current.loadConversation({ serverChatId: nextChoice === "same-chat" ? nativeChatId : nextChoice })).toBe(true)
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
  expect(h1.roleplayAppend).toHaveBeenCalledOnce()
  expect(h1.roleplayComplete).toHaveBeenCalledOnce()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(mounted.options.setServerChatId).toHaveBeenCalledWith(nativeChatId)
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
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
  createChatMock.mockImplementation(async body => { enter(); await held; return createdNativeChat(body) })
  let operation!: Promise<unknown>
  await act(async () => { operation = mounted.hook.result.current.onSubmit({ message: "RETAINED INPUT", image: "" }); await ready })
  act(() => { if (boundary === "new") mounted.draftStore.getState().startNewChatDraft(); else h1.auth.abort() })
  await act(async () => { release(); await operation })
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const handle = { address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) }
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "acknowledged", chatId: nativeChatId, inputText: "RETAINED INPUT" })
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
    isCurrent: () => true, create: async id => createdNativeChat({ id }) })
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
    const reviewing = reviewNativeChatCreation({ controller: mounted.selection.result.current, handle, ...(await creationReviewIdentity(handle)), isCurrent: () => composerCurrent, onAccepted })
    if (boundary === "new-during-metadata") { await ready; mounted.draftStore.getState().startNewChatDraft(); release() }
    accepted = await reviewing
  })
  expect(accepted).toBe(boundary === "accept")
  expect(onAccepted).toHaveBeenCalledTimes(boundary === "accept" ? 1 : 0)
  expect(createChatMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ inputText: "RETAINED UNSENT INPUT", state: boundary === "accept" ? "bound" : "acknowledged" })
  if (boundary === "accept") {
    expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
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
  expect(mounted.selection.result.current.getCurrent().owner?.conversation_id).toBe(nativeChatId)
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
    isCurrent: () => true, create: async id => createdNativeChat({ id }) })
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  let release!: () => void, enter!: () => void
  const ready = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.mocked(tldwClient.getChat).mockImplementationOnce(async id => { enter(); await held; return { id } as Awaited<ReturnType<typeof tldwClient.getChat>> })
  const { reviewNativeChatCreation } = await import("../native-history-character-send")
  const onAccepted = vi.fn()
  let reviewing!: Promise<boolean>
  await act(async () => { reviewing = reviewNativeChatCreation({ controller: mounted.selection.result.current, handle, ...(await creationReviewIdentity(handle)), isCurrent: () => true, onAccepted }); await ready })
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
  expect(h1.capture.mock.calls.some(([id]) => id === nativeChatId)).toBe(false)
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
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "ORIGINAL ATTEMPT", setup: {}, isCurrent: () => true, create: async id => createdNativeChat({ id }) })
  const { useComposerText } = await import("@/components/Chat/composer/hooks/useComposerText")
  const composer = renderHook(() => useComposerText({ draftKey: "review-retain", draftAddress: mounted.sourceDraft!, textareaRef: React.createRef<HTMLTextAreaElement>() }))
  await waitFor(() => expect(composer.result.current.creationRecovery.active?.chatId).toBe(nativeChatId))
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
  expect(h1.capture.mock.calls.some(([id]) => id === nativeChatId)).toBe(false)
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
    installFreshSource()
    const source = draftStore.getState().chatDraftAddress
    const options = { ...ordinaryOptions(), historyId: null, serverChatId: null, messages: [], history: [] }
    let current: ReturnType<ReturnType<typeof makeController>["getCurrent"]> | { owner: null; view: null; status: "idle" } = { owner: null, view: null, status: "idle" }
    h1.controller = {
      draftAccount: declaredDraftAccount(),
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
    createChatMock.mockImplementation(createdNativeChat)
    let enter!: () => void, release!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    if (stage === "scope") {
      const scope = vi.mocked((await import("@/services/service-prompts")).loadServicePromptSnapshot)
      const original = scope.getMockImplementation()!
      scope.mockImplementationOnce(async (...args) => { enter(); await held; return original(...args) })
    } else {
      const actor = vi.mocked((await import("@/services/actor-settings")).getActorSceneForChat)
      const original = actor.getMockImplementation()!
      actor.mockImplementationOnce(async (...args) => { enter(); await held; return original(...args) })
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
  const captured = captureActorSceneSend(h1.controller)
  if (!captured) throw new Error("fixture requires a qualified greeting scene")
  const actorSceneSend = await captured.read()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const controller = mounted.selection.result.current
  const args = { controller, actorSceneSend, surface, originIsCurrent: controller.fence(), signal: new AbortController().signal,
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
  expect(mounted.selection.result.current.capture).toMatchObject({ status: "captured", rows: [{ id: "opening-" + nativeChatId, role: "assistant", settled: true }] })
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
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
  expect(await readChatCreationRecovery({ address: mounted.sourceDraft!, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "https://server.test", authMode: "multi-user" }, userId: "alice" }) })).toMatchObject({ state: "acknowledged", chatId: nativeChatId, inputText: "" })
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
  expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({ kind: "native", id: nativeChatId })
  expect(focusComposer).toHaveBeenCalledTimes(focus === "owned" ? 1 : 0)
  expect(document.activeElement).toBe(focus === "owned" ? textarea : other)
  expect(h1.wire).not.toHaveBeenCalled()
  starter.unmount(); mounted.hook.unmount(); mounted.selection.unmount(); button.remove(); textarea.remove(); other.remove()
})


it.each(["bound-record", "resume-detail", "resume-detail-current"])("private native popup %s retires changed preparation while retaining its accepted owner", async stage => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { Storage } = await import("@plasmohq/storage")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  const actorSceneSend = await readPrivateActorScene(handle, snapshot)
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
  } else {
    const api = (await import("@/services/tldw/TldwApiClient")).tldwClient
    const original = vi.mocked(api.getChat).getMockImplementation()!
    vi.mocked(api.getChat).mockImplementation(async (id, options) => {
      if (options?.requireRoleplayResume) { enter(); await held }
      return original(id, options)
    })
  }
  let pending!: Promise<unknown>
  try {
    await act(async () => {
    pending = sendNativeHistoryCharacter({ controller: mounted.selection.result.current,
      privateDraft: { handle, requestScope: snapshot.requestScope, isCurrent: () => true }, actorSceneSend, surface: "playground",
      originIsCurrent: () => preparationCurrent, signal: new AbortController().signal,
      temporary: false, scope: { type: "global" }, characterId: "12", model: "deepseek:deepseek-chat",
      settings: {}, toolChoice: "none", message: "private popup input", image: "", t: ((key: string) => key) as import("i18next").TFunction,
      setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
    }).catch(error => error)
    await Promise.race([ready, pending.then(outcome => { throw new Error(`Sender finished before ${stage}: ${String(outcome)}`) })])
  })
  preparationCurrent = stage === "resume-detail-current"
  await act(async () => { release(); await pending })
  const owner = mounted.selection.result.current.getCurrent().owner
  const acceptedOwnerValid = owner?.kind === "native" && owner.validate_lease()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(h1.roleplayAppend).toHaveBeenCalledTimes(stage === "resume-detail-current" ? 1 : 0)
  expect(h1.roleplayComplete).toHaveBeenCalledTimes(stage === "resume-detail-current" ? 1 : 0)
  expect(acceptedOwnerValid).toBe(true)
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(before.chatDraftAddress)
  expect(mounted.draftStore.getState().chatDraftRevision).toBe(before.chatDraftRevision)
  } finally {
    release()
    if (pending) await act(async () => { await pending })
    write?.mockRestore()
    snapshot.release(); mounted.hook.unmount(); mounted.selection.unmount()
  }
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
  await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "FIRST SENT", setup: {}, isCurrent: () => true, create: async id => createdNativeChat({ id }) })
  await markChatCreationBound(handle, nativeChatId)
  if (boundary !== "bound-next") await markChatCreationSettled(handle, nativeChatId)
  const root = "tldw:playgroundChatDraft"
  const sourceKey = `${conversationDraftKey(root, handle.address)}:owner:${handle.ownerHash}`
  const destination = { kind: "native" as const, id: nativeChatId, ownerKey: "native-key", scope: { type: "global" as const } }
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


it.each([false, true].flatMap(changed => [false, true].map(saved => ({ changed, saved }))))("private bound retry settles only saved exact original input (changed $changed, saved $saved)", async ({ changed, saved }) => {
  const mounted = await mountedNativeOpening()
  const { sendNativeHistoryCharacter } = await import("../native-history-character-send")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { loadServicePromptSnapshot } = await import("@/services/service-prompts")
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const snapshot = await loadServicePromptSnapshot([])
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
  const actorSceneSend = await readPrivateActorScene(handle, snapshot)
  let preparationCurrent = true
  h1.capture.mockImplementation(async (id, request) => {
    if (h1.capture.mock.calls.length === 3) preparationCurrent = false
    return openingCapture(id, request)
  })
  const send = (message: string, serverChatId?: string) => sendNativeHistoryCharacter({
    controller: mounted.selection.result.current, privateDraft: { handle, requestScope: snapshot.requestScope, isCurrent: () => true }, actorSceneSend, surface: "playground",
    originIsCurrent: () => preparationCurrent, signal: new AbortController().signal, temporary: false,
    scope: { type: "global" }, characterId: "12", model: "deepseek:deepseek-chat", settings: {}, toolChoice: "none",
    message, image: "", serverChatId, t: ((key: string) => key) as import("i18next").TFunction,
    setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
  })
  await act(async () => { await send("  ORIGINAL INPUT  ").catch(() => {}) })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "bound", inputText: "  ORIGINAL INPUT  ", chatId: nativeChatId })
  preparationCurrent = true
  h1.capture.mockImplementation(async (id, request) => openingCapture(id, request))
  if (!saved) h1.roleplayComplete.mockImplementationOnce(async (id, body, options) => {
    options.onDispatch?.()
    const reply = roleplayCompletion(id, body)
    return { ...reply, saved: false, assistant_message_id: null,
      tldw_roleplay_resume_v1: { ...reply.tldw_roleplay_resume_v1, assistant_message_version: null, fence: null, unsaved_reason: "history_conflict" } }
  })
  await act(async () => { await send(changed ? "DIFFERENT INPUT" : "  ORIGINAL INPUT  ", nativeChatId) })
  expect(h1.roleplayAppend).toHaveBeenCalledOnce()
  expect(h1.roleplayComplete).toHaveBeenCalledOnce()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(createChatMock).toHaveBeenCalledOnce()
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: changed || !saved ? "bound" : "settled", inputText: changed || !saved ? "  ORIGINAL INPUT  " : "" })
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
  const actorSceneSend = await readPrivateActorScene(handle, snapshot)
  h1.capture.mockRejectedValue(new Error("owned_read_unavailable"))
  let error: unknown
  await act(async () => { error = await sendNativeHistoryCharacter({ controller: mounted.selection.result.current,
    privateDraft: { handle, requestScope: snapshot.requestScope, isCurrent: () => true }, actorSceneSend, surface: "playground", originIsCurrent: () => true,
    signal: new AbortController().signal, temporary: false, scope: { type: "global" }, characterId: "12",
    model: "deepseek:deepseek-chat", settings: {}, toolChoice: "none", message: "  KEPT EXACT INPUT  ", image: "",
    t: ((key: string) => key) as import("i18next").TFunction, setServerChatId: vi.fn(), setMessages: vi.fn(), releaseActivity: vi.fn()
  }).catch(cause => cause) })
  expect((error as Error).message).toBe("owned_read_unavailable")
  expect(await readChatCreationRecovery(handle)).toMatchObject({ state: "acknowledged", chatId: nativeChatId, inputText: "  KEPT EXACT INPUT  " })
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
  await act(async () => { expect(await selection.result.current.loadConversation(local ? { historyId: "history-1" } : { serverChatId: nativeChatId })).toBe(true) })
  h1.controller = Object.assign(selection.result.current, { draftAccount: declaredDraftAccount() })
  const current = selection.result.current.getCurrent()
  installAcceptedSource(current as HistoryLoadReceipt)
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

it.each(["refused", "failed"])(
  "interrupted retry retains recovery and sends nothing after %s dismissal",
  async (mode) => {
    const mounted = await mountInterruptedRetry()
    if (mode === "failed")
      h1.dismiss.mockRejectedValueOnce(new Error("journal unavailable"))
    else h1.dismiss.mockResolvedValueOnce(false)
    let outcome: unknown
    await act(async () => {
      outcome = await mounted.hook.result.current.regenerateLastMessage()
    })
    expect(outcome).toMatchObject({
      status: "failed",
      errorMessage: "recovery_dismissal_not_committed"
    })
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    await expect(
      h1.lookupRecoveries.mock.results.at(-1)?.value
    ).resolves.toEqual([mounted.entry])
    expect(mounted.options.notification.error).toHaveBeenCalledOnce()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)

it("current interrupted retry reuses its existing admission and submits once", async () => {
  const mounted = await mountInterruptedRetry()
  let outcome: unknown
  await act(async () => { outcome = await mounted.hook.result.current.regenerateLastMessage() })
  expect(outcome).toMatchObject({ status: "submitted" })
  expect(mounted.choose).toHaveBeenCalledExactlyOnceWith({ kind: "before_message", message_id: "u-old" })
  expect(mounted.dismiss).toHaveBeenCalledExactlyOnceWith(mounted.entry, expect.any(Function))
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


it.each([
  ["capture", "abort"],
  ["capture", "source"],
  ["dismissal", "abort"],
  ["dismissal", "source"]
])(
  "interrupted retry preserves remaining recoveries when dispatch retires during %s (%s)",
  async (held, boundary) => {
    const mounted = await mountInterruptedRetry()
    const second = {
      ...mounted.entry,
      turn: { ...mounted.entry.turn, operation_id: "second-interrupted-retry" }
    }
    h1.lookupRecoveries.mockResolvedValue([mounted.entry, second])
    let finish!: () => void
    let entered!: () => void
    const ready = new Promise<void>((resolve) => {
      entered = resolve
    })
    if (held === "capture") {
      const capture = h1.capture.getMockImplementation()!
      h1.capture.mockImplementationOnce(async (...args) => {
        entered()
        await new Promise<void>((resolve) => {
          finish = resolve
        })
        return capture(...args)
      })
    } else {
      mounted.dismiss.mockImplementationOnce(async () => {
        entered()
        await new Promise<void>((resolve) => {
          finish = resolve
        })
      })
    }
    const controller = new AbortController()
    let sourceCurrent = true
    const assertCurrent = () => {
      if (!sourceCurrent) throw new Error("Research source retired")
    }
    let pending!: Promise<unknown>
    await act(async () => {
      pending = mounted.hook.result.current
        .regenerateLastMessage({ controller, assertCurrent })
        .catch((error) => error)
      await ready
    })
    if (boundary === "abort") controller.abort()
    else sourceCurrent = false
    let outcome: unknown
    await act(async () => {
      finish()
      outcome = await pending
    })
    expect(mounted.dismiss).toHaveBeenCalledTimes(held === "capture" ? 0 : 1)
    expect(mounted.dismiss).not.toHaveBeenCalledWith(second)
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(outcome).toBeInstanceOf(Error)
    expect(mounted.options.notification.error).not.toHaveBeenCalled()
  }
)


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


it.each([false, true])(
  "keeps actual unread scenes ahead of native creation and retains a stable failure code (greeting=%s)",
  async (greeting) => {
    const mounted = await mountedNativeOpening()
    const { Storage } = await import("@plasmohq/storage")
    const service = await vi.importActual<
      typeof import("@/services/actor-settings")
    >("@/services/actor-settings")
    const reader = vi.mocked(
      (await import("@/services/actor-settings")).getActorSceneForChat
    )
    reader.mockImplementationOnce(service.getActorSceneForChat)
    const originalRead = Storage.prototype.get
    let enteredRead!: () => void, releaseRead!: () => void
    const entered = new Promise<void>(resolve => { enteredRead = resolve })
    const held = new Promise<void>(resolve => { releaseRead = resolve })
    const read = vi.spyOn(Storage.prototype, "get").mockImplementation(async function (this: Storage, key) {
      if (key.startsWith("actorSettings:owned:")) {
        enteredRead()
        await held
        throw new Error("scene read unavailable")
      }
      return originalRead.call(this, key)
    })
    let outcome: unknown, pending!: Promise<unknown>
    act(() => {
      pending = mounted.hook.result.current.onSubmit({
        message: greeting ? "" : "Keep this exact input",
        image: "",
        ...(greeting
          ? {
              greetingSeed: {
                greetingStrategy: "default" as const,
                rawText: "Saved opening"
              },
              greetingIsCurrent: () => true
            }
          : {})
      })
    })
    await act(async () => {
      await Promise.race([entered, pending.then(() => {
        throw new Error("Sender finished before the qualified scene storage read")
      })])
      releaseRead()
      outcome = await pending
    })
    expect(read).toHaveBeenCalledWith(`actorSettings:owned:${reader.mock.calls[0][0].target.key}`)
    expect(reader.mock.calls[0][0].target.address).toEqual(mounted.sourceDraft)
    expect(outcome).toMatchObject({
      status: "failed",
      errorMessage: "actor_settings_unavailable"
    })
    if (!greeting)
      expect(mounted.options.notification.error).toHaveBeenCalledWith(
        expect.objectContaining({
          description:
            "Scene settings could not be loaded. Existing chat setup was left unchanged."
        })
      )
    expect(createChatMock).not.toHaveBeenCalled()
    expect(h1.capture).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(mounted.options.setServerChatId).not.toHaveBeenCalled()
    read.mockRestore()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)


it.each(["playground", "sidepanel"] as const)(
  "client UUID character greeting %s sends the durable ID without replay after a lost response",
  async (surface) => {
    const mounted = await mountedNativeOpening()
    createChatMock.mockRejectedValue(new Error("Response lost"))
    await act(async () => {
      await expect(
        startGreetingOnMountedNative(
          mounted,
          {
            greetingStrategy: "alternate_index",
            alternateIndex: 0,
            rawText: "Saved opening"
          },
          surface
        )
      ).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    })
    const { readChatCreationRecovery } =
      await import("@/services/pending-chat-promotion")
    const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
    const attempt = await readChatCreationRecovery({
      address: mounted.sourceDraft!,
      ownerHash: chatDraftOwnerKey({
        config: { serverUrl: "https://server.test", authMode: "multi-user" },
        userId: "alice"
      })
    })
    expect(createChatMock.mock.calls[0][0]).toMatchObject({
      id: attempt?.operationId,
      character_id: "12",
      provider: "deepseek",
      model: "deepseek-chat",
      temperature: 0,
      top_p: 0
    })
    expect(createChatMock.mock.calls[0][1]).toMatchObject({
      seedFirstMessage: true,
      greetingStrategy: "alternate_index",
      alternateIndex: 0
    })
    expect(attempt).toMatchObject({
      clientChatId: attempt?.operationId,
      state: "unconfirmed",
      inputText: ""
    })
    expect(createChatMock.mock.calls[0][0].id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    )
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(h1.wire).not.toHaveBeenCalled()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)

const mountClientUUIDRecovery = async (workspace = false) => {
  const mounted = await mountedNativeOpening()
  const { Storage } = await import("@plasmohq/storage")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const { chatDraftAddressKey } = await import("@/utils/chat-draft-address")
  const { loadServicePromptSnapshot } =
    await import("@/services/service-prompts")
  const { usePlaygroundSessionStore: sessions } =
    await import("@/store/playground-session")
  sessions.setState({ serverChatSelectionIntent: null })
  if (workspace)
    mounted.draftStore.getState().setChatDraftAddress({
      ...mounted.sourceDraft!,
      scope: { type: "workspace", workspaceId: "owned-workspace" }
    })
  const snapshot = await loadServicePromptSnapshot([])
  const handle = {
    address: mounted.draftStore.getState().chatDraftAddress!,
    ownerHash: chatDraftOwnerKey(snapshot.requestScope)
  }
  const operationId = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
  const attempt = {
    version: 1 as const,
    operationId,
    clientChatId: operationId,
    handle,
    surface: "playground" as const,
    createdAt: 1,
    state: "unconfirmed" as const,
    inputText: "  ORIGINAL ATTEMPT  "
  }
  const storage = new Storage({ area: "local" })
  const key =
    "registry:chat-creation:" +
    handle.ownerHash +
    ":" +
    chatDraftAddressKey(handle.address)
  await storage.set(key, { value: attempt, updatedAt: 1 })
  const { reviewNativeChatCreation } =
    await import("../native-history-character-send")
  const onAccepted = vi.fn()
  const review = (
    extra: Partial<Parameters<typeof reviewNativeChatCreation>[0]> = {}
  ) =>
    reviewNativeChatCreation({
      controller: mounted.selection.result.current,
      handle,
      expectedOperationId: operationId,
      expectedClientChatId: operationId,
      inputText: attempt.inputText,
      isCurrent: () => true,
      onAccepted,
      ...extra
    })
  return {
    ...mounted,
    handle,
    attempt,
    snapshot,
    storage,
    key,
    review,
    onAccepted
  }
}

it.each([false, true])(
  "client UUID explicit lookup binds the exact owned chat in workspace %s without create or provider replay",
  async (workspace) => {
    const mounted = await mountClientUUIDRecovery(workspace)
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { readChatCreationRecovery } =
      await import("@/services/pending-chat-promotion")
    await act(async () => {
      expect(await mounted.review()).toBe(true)
    })
    expect(tldwClient.getChat).toHaveBeenCalledWith(
      mounted.attempt.clientChatId,
      expect.objectContaining({
        scope: mounted.handle.address.scope,
        requestScope: mounted.snapshot.requestScope,
        signal: expect.any(AbortSignal)
      })
    )
    expect(mounted.onAccepted).toHaveBeenCalledWith(
      expect.objectContaining({ id: mounted.attempt.clientChatId })
    )
    expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({
      kind: "native",
      id: mounted.attempt.clientChatId
    })
    expect(await readChatCreationRecovery(mounted.handle)).toMatchObject({
      state: "bound",
      inputText: mounted.attempt.inputText,
      chatId: mounted.attempt.clientChatId
    })
    expect(createChatMock).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    mounted.snapshot.release()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)

it.each([
  "404",
  "410",
  "transport",
  "mismatched",
  "new",
  "account",
  "input",
  "newer-H1",
  "record",
  "storage",
  "locks",
  "copy-failed",
  "copy"
])("client UUID lookup preserves recovery across %s", async (boundary) => {
  const mounted = await mountClientUUIDRecovery()
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  const { readChatCreationRecovery } =
    await import("@/services/pending-chat-promotion")
  const getChat = vi.mocked(tldwClient.getChat)
  let current = true
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>((resolve) => {
    entered = resolve
  })
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  const delayed = ["new", "account", "input", "newer-H1", "record"].includes(
    boundary
  )
  getChat.mockImplementationOnce(async (id) => {
    if (delayed) {
      entered()
      await held
    }
    if (["404", "410", "transport"].includes(boundary))
      throw Object.assign(new Error("lookup failed"), {
        status: Number(boundary) || undefined
      })
    return {
      id: boundary === "mismatched" ? "foreign-chat" : id,
      title: "Actual owned chat"
    } as Awaited<ReturnType<typeof tldwClient.getChat>>
  })
  const { Storage } = await import("@plasmohq/storage")
  const originalSet = Storage.prototype.set
  if (boundary === "storage")
    vi.spyOn(Storage.prototype, "set").mockImplementation(async function (
      this: Storage,
      key,
      value
    ) {
      if (key === mounted.key && value.value?.state === "acknowledged")
        throw new Error("ACK storage failed")
      return originalSet.call(this, key, value)
    })
  if (boundary === "locks")
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: undefined
    })
  const copyInput = vi.fn(async () => boundary !== "copy-failed")
  let result = false
  await act(async () => {
    const reviewing = mounted
      .review({
        isCurrent: () => current,
        ...(["copy", "copy-failed"].includes(boundary)
          ? { inputText: "  DIFFERENT UNSENT TEXT  ", copyInput }
          : {})
      })
      .catch((error) => {
        if (!["404", "410", "transport"].includes(boundary)) throw error
        return false
      })
    if (delayed) {
      await ready
      if (boundary === "new") mounted.draftStore.getState().startNewChatDraft()
      else if (boundary === "account") h1.auth.abort()
      else if (boundary === "input") current = false
      else if (boundary === "newer-H1")
        await mounted.selection.result.current.loadConversation({
          serverChatId: "newer-chat"
        })
      else {
        const operationId = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a15"
        await mounted.storage.set(mounted.key, {
          value: {
            ...mounted.attempt,
            operationId,
            clientChatId: operationId,
            inputText: "NEWER ORIGINAL"
          },
          updatedAt: 2
        })
      }
      release()
    }
    result = await reviewing
  })
  expect(result).toBe(boundary === "copy")
  expect(mounted.onAccepted).toHaveBeenCalledTimes(boundary === "copy" ? 1 : 0)
  expect(createChatMock).not.toHaveBeenCalled()
  expect(h1.wire).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  const record = await readChatCreationRecovery(mounted.handle)
  expect(record).toMatchObject({
    state: boundary === "copy" ? "bound" : "unconfirmed",
    inputText:
      boundary === "record" ? "NEWER ORIGINAL" : mounted.attempt.inputText
  })
  if (boundary === "copy-failed") expect(getChat).not.toHaveBeenCalled()
  if (["copy", "copy-failed"].includes(boundary))
    expect(copyInput).toHaveBeenCalledOnce()
  if (boundary === "newer-H1")
    expect(
      mounted.selection.result.current.getCurrent().owner?.conversation_id
    ).toBe("newer-chat")
  mounted.snapshot.release()
  mounted.hook.unmount()
  mounted.selection.unmount()
})

it("client UUID retained composer Check cannot adopt a replacement operation at the same address", async () => {
  const mounted = await mountClientUUIDRecovery()
  const { useComposerText } =
    await import("@/components/Chat/composer/hooks/useComposerText")
  const composer = renderHook(() =>
    useComposerText({
      draftKey: "uuid-operation",
      draftAddress: mounted.handle.address,
      textareaRef: React.createRef<HTMLTextAreaElement>()
    })
  )
  await waitFor(() =>
    expect(composer.result.current.creationRecovery.active?.operationId).toBe(
      mounted.attempt.operationId
    )
  )
  const review = composer.result.current.reviewCreatedChat
  const operationId = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a15"
  await mounted.storage.set(mounted.key, {
    value: {
      ...mounted.attempt,
      operationId,
      clientChatId: operationId,
      chatId: operationId,
      state: "acknowledged",
      inputText: "NEWER ORIGINAL"
    },
    updatedAt: 2
  })
  await act(async () => {
    expect(await review()).toBe(false)
  })
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  expect(tldwClient.getChat).not.toHaveBeenCalled()
  expect(mounted.onAccepted).not.toHaveBeenCalled()
  expect(mounted.draftStore.getState().chatDraftAddress).toBe(
    mounted.handle.address
  )
  composer.unmount()
  mounted.snapshot.release()
  mounted.hook.unmount()
  mounted.selection.unmount()
})

it.each([false, true])(
  "client UUID alias review wins before original POST resolves and preserves settlement %s",
  async (settle) => {
    const mounted = await mountedNativeOpening()
    let entered!: () => void, release!: () => void
    const ready = new Promise<void>((resolve) => {
      entered = resolve
    })
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    createChatMock.mockImplementation(async (body) => {
      entered()
      await held
      return createdNativeChat(body)
    })
    let original!: Promise<unknown>
    await act(async () => {
      original = mounted.hook.result.current.onSubmit({
        message: "ORIGINAL PRIVATE INPUT",
        image: ""
      })
      await ready
    })
    const {
      readChatCreationRecovery,
      copyChatCreationRecovery,
      markChatCreationSettled
    } = await import("@/services/pending-chat-promotion")
    const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
    const ownerHash = chatDraftOwnerKey({
      config: { serverUrl: "https://server.test", authMode: "multi-user" },
      userId: "alice"
    })
    const source = { address: mounted.sourceDraft!, ownerHash }
    const alias = {
      address: { ...mounted.sourceDraft!, id: "copied-uuid-recovery" },
      ownerHash
    }
    expect(await copyChatCreationRecovery(source, alias)).toBe(true)
    act(() => mounted.draftStore.getState().setChatDraftAddress(alias.address))
    const attempt = (await readChatCreationRecovery(alias))!
    const { reviewNativeChatCreation } =
      await import("../native-history-character-send")
    let reviewed = false
    await act(async () => {
      reviewed = await reviewNativeChatCreation({
        controller: mounted.selection.result.current,
        handle: alias,
        expectedOperationId: attempt.operationId,
        expectedClientChatId: attempt.clientChatId,
        inputText: attempt.inputText,
        isCurrent: () => true,
        onAccepted: vi.fn()
      })
      if (reviewed && settle)
        expect(
          await markChatCreationSettled(alias, attempt.clientChatId!)
        ).toBe(true)
      release()
      await original
    })
    expect(reviewed).toBe(true)
    expect(await readChatCreationRecovery(source)).toMatchObject({
      state: settle ? "settled" : "bound",
      chatId: attempt.clientChatId,
      inputText: settle ? "" : "ORIGINAL PRIVATE INPUT"
    })
    expect(await readChatCreationRecovery(alias)).toEqual(
      await readChatCreationRecovery(source)
    )
    expect(mounted.draftStore.getState().chatDraftAddress).toMatchObject({
      kind: "native",
      id: attempt.clientChatId
    })
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)

it.each(["H1", "input", "edit-undo"])(
  "client UUID rendered Check rejects %s replacement before invocation",
  async (boundary) => {
    const mounted = await mountClientUUIDRecovery()
    const { useComposerText } =
      await import("@/components/Chat/composer/hooks/useComposerText")
    const clipboard = vi.fn(async () => undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboard }
    })
    const composer = renderHook(() =>
      useComposerText({
        draftKey: "uuid-rendered",
        draftAddress: mounted.handle.address,
        textareaRef: React.createRef<HTMLTextAreaElement>()
      })
    )
    await waitFor(() =>
      expect(composer.result.current.creationRecovery.active?.operationId).toBe(
        mounted.attempt.operationId
      )
    )
    act(() =>
      composer.result.current.setMessageValue(mounted.attempt.inputText)
    )
    const review = composer.result.current.reviewCreatedChat
    let pending!: Promise<boolean>
    act(() => {
      if (boundary === "H1") mounted.selection.result.current.beginLoad()
      else {
        composer.result.current.setMessageValue("NEWER UNSENT INPUT")
        if (boundary === "edit-undo")
          composer.result.current.setMessageValue(mounted.attempt.inputText)
      }
      pending = review()
    })
    await act(async () => {
      expect(await pending).toBe(false)
    })
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { readChatCreationRecovery } =
      await import("@/services/pending-chat-promotion")
    expect(tldwClient.getChat).not.toHaveBeenCalled()
    expect(clipboard).not.toHaveBeenCalled()
    expect(h1.capture).not.toHaveBeenCalled()
    expect(mounted.draftStore.getState().chatDraftAddress).toBe(
      mounted.handle.address
    )
    expect(await readChatCreationRecovery(mounted.handle)).toMatchObject({
      state: "unconfirmed",
      inputText: mounted.attempt.inputText
    })
    composer.unmount()
    mounted.snapshot.release()
    mounted.hook.unmount()
    mounted.selection.unmount()
  }
)


it.each(["history-epoch", "new-draft", "same-id-restore"])(
  "a retained ordinary idle edit refuses %s before another render",
  async (retirement) => {
    const helpers = await import("@/db/dexie/helpers")
    const actual =
      await vi.importActual<typeof import("@/store/option")>("@/store/option")
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
    const options = {
      ...ordinaryOptions(),
      serverChatId: null,
      messages: [
        {
          id: "idle",
          name: "You",
          isBot: false,
          message: "Original idle row",
          sources: []
        }
      ]
    }
    installLocalSource("history-character")
    messageStoreState.editRows = () => options.messages
    const hook = renderHook(() => useChatActions(options as any))
    const retained = hook.result.current.editMessage
    options.setMessages.mockClear()
    options.setHistory.mockClear()
    if (retirement === "history-epoch") epoch++
    else if (retirement === "new-draft")
      actual.useStoreMessageOption.setState({
        chatDraftRevision:
          actual.useStoreMessageOption.getState().chatDraftRevision + 1
      })
    else
      sessions.usePlaygroundSessionStore.setState({
        restoreRevision:
          sessions.usePlaygroundSessionStore.getState().restoreRevision + 1
      })
    expect(await retained(0, "Obsolete edit", true, false)).toEqual({
      status: "skipped",
      reason: "stale_selection"
    })
    expect(helpers.updateMessageById).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
  }
)


it.each(["current", "newer edit", "Cancel", "interrupted"])(
  "mounted native character edit-resend preserves originals through %s",
  async (boundary) => {
    const options = createHookOptions()
    // The real native character capture includes backend structural role metadata.
    const decorate = (capture: any) => ({
      ...capture,
      snapshot: {
        ...capture.snapshot,
        nodes: capture.snapshot.nodes.map((node: any) =>
          node.id === "u-old"
            ? {
                ...node,
                metadata: [
                  { id: "u-old", revision: "m1", kind: "message_metadata" }
                ]
              }
            : node
        )
      },
      selected_content: capture.selected_content.map((row: any) =>
        row.id === "u-old"
          ? { ...row, extra_metadata: { sender_role: "user" } }
          : row
      )
    })
    h1.controller.getCurrent().capture = decorate(
      h1.controller.getCurrent().capture
    )
    const capture = h1.capture.getMockImplementation()!
    h1.capture.mockImplementation(async (...args) =>
      decorate(await capture(...args))
    )
    options.messages = [
      {
        id: "u-old",
        isBot: false,
        message: "old question",
        sources: [],
        metadataExtra: { sender_role: "user" }
      },
      { id: "a1", isBot: true, message: "same answer", sources: [] }
    ] as any
    messageStoreState.editRows = () => options.messages
    const original = structuredClone(options.messages)
    const origin = h1.controller.getCurrent().view
    let editorCurrent = true
    const isCurrent = () => editorCurrent
    const store = (await import("@/store/option")).useStoreMessageOption
    h1.wire.mockImplementation(async function* (request) {
      expect(store.getState().detachedHistoryResponse).toBeTruthy()
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
      yield { choices: [{ delta: { content: "Edited answer" } }] }
      if (boundary === "interrupted") throw new Error("disconnected")
      if (boundary !== "current") editorCurrent = false
      yield {
        tldw_message_id: "edited-answer",
        tldw_conversation_id: nativeChatId
      }
    })
    const hook = renderHook(() => useChatActions(options as any))
    options.setMessages.mockClear()
    options.setHistory.mockClear()
    let result: unknown
    await act(async () => {
      result = await hook.result.current.editMessage(
        0,
        "Edited question  \n",
        true,
        true,
        isCurrent
      )
    })
    expect(store.getState().detachedHistoryResponse).toBeNull()
    expect(h1.wire).toHaveBeenCalledOnce()
    const request = h1.wire.mock.calls[0][0]
    expect(request).toMatchObject({
      tldw_history_branch: true,
      messages: [{ role: "user", content: "Edited question  \n" }],
      tldw_history_selection_v1: {
        cursor: { kind: "before_message", message_id: "u-old" },
        messages: [],
        selection_revision: origin.selection_revision + 1
      }
    })
    expect(options.messages).toEqual(original)
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(h1.controller.getCurrent().view).toBe(origin)
    if (boundary === "current") {
      expect(result).toEqual({ status: "submitted" })
      expect(h1.controller.followResult).toHaveBeenCalledExactlyOnceWith(
        origin,
        "edited-answer",
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
      expect(h1.controller.followResult).not.toHaveBeenCalled()
      expect(h1.dismiss).not.toHaveBeenCalled()
      expect(h1.recover).toHaveBeenLastCalledWith(
        expect.anything(),
        expect.anything(),
        expect.objectContaining({
          input_text: "Edited question  \n",
          input_id: "edited-input",
          result_text: "Edited answer",
          state: "generated_unsaved",
          ...(boundary === "interrupted"
            ? {}
            : { outcome: "complete", settled_message_id: "edited-answer" })
        })
      )
    }
  }
)

it("native edit-resend refuses Compare without local or provider writes", async () => {
  const options = ordinaryOptions()
  options.compareModeActive = true
  options.compareSelectedModels = ["deepseek-chat", "another-model"] as never
  messageStoreState.editRows = () => options.messages
  const original = structuredClone(options.messages)
  const hook = renderHook(() => useChatActions(options as any))
  options.setMessages.mockClear()
  options.setHistory.mockClear()
  await act(async () => {
    expect(
      await hook.result.current.editMessage(0, "edited", true, true, () => true)
    ).toMatchObject({ status: "failed" })
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.saveHistory).not.toHaveBeenCalled()
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(options.setHistory).not.toHaveBeenCalled()
  expect(options.messages).toEqual(original)
})


it.each([false, true])(
  "a rejected full callback edited=%s cannot replace a held response marker",
  async (edited) => {
    const store = (await import("@/store/option")).useStoreMessageOption
    const current = {}
    store.setState({ detachedHistoryResponse: current })
    const options = ordinaryOptions()
    const hook = renderHook(() => useChatActions(options as any))
    options.setStreaming.mockClear()
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
    expect(store.getState().detachedHistoryResponse).toBe(current)
    expect(options.setStreaming).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    store.setState({ detachedHistoryResponse: null })
  }
)


it.each([false, true])(
  "held full character edit survives a rejected second callback edited=%s",
  async (edited) => {
    const store = (await import("@/store/option")).useStoreMessageOption
    const options = createHookOptions()
    options.messages = [
      { id: "u-old", isBot: false, message: "old question", sources: [] },
      { id: "a1", isBot: true, message: "same answer", sources: [] }
    ] as any
    messageStoreState.editRows = () => options.messages
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
    const hook = renderHook(() => useChatActions(options as any))
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
      const marker = store.getState().detachedHistoryResponse
      const controllerCalls = options.setAbortController.mock.calls.length
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
      expect(store.getState().detachedHistoryResponse).toBe(marker)
      expect(options.setAbortController).toHaveBeenCalledTimes(controllerCalls)
      expect(h1.wire).toHaveBeenCalledOnce()
    } finally {
      release()
      await act(async () => {
        await pending
      })
    }
    expect(store.getState().detachedHistoryResponse).toBeNull()
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
    messageStoreState.editRows = () => draftStore.getState().messages
  })
  const selection = renderHook(() =>
    useHistorySelection({ onCapture: publish })
  )
  await act(async () => {
    expect(
      await selection.result.current.loadConversation({
        serverChatId: nativeChatId
      })
    ).toBe(true)
  })
  h1.controller = Object.assign(selection.result.current, { draftAccount: declaredDraftAccount() })
  await act(async () => {
    expect(
      await selection.result.current.choose({
        kind: "after_message",
        message_id: "a1"
      })
    ).toBe(true)
  })
  const origin = selection.result.current.getCurrent().capture!
  installAcceptedSource(selection.result.current.getCurrent() as HistoryLoadReceipt)
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
  const projection = formatSelectedHistory(origin as any)
  const options = {
    ...ordinaryOptions(),
    ...projection,
    serverChatId: nativeChatId,
    selectedCharacter: character ? { id: 12, name: "Mira" } : null,
    serverChatCharacterId: character ? "12" : null,
    serverChatAssistantKind: character ? "character" : null
  }
  const hook = renderHook(() => useChatActions(options as any))
  ;(hook as any).flat = options
  options.setMessages.mockClear()
  options.setHistory.mockClear()
  const index = role === "user" ? 0 : 1
  publish.mockClear()
  return {
    options,
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
      expect((mounted.hook as any).flat.setMessages).not.toHaveBeenCalled()
      expect((mounted.hook as any).flat.setHistory).not.toHaveBeenCalled()
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
        mounted.draftStore.setState({ [boundary]: true })
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
      mounted.draftStore.setState({ streaming: false, isProcessing: false })
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
      mounted.draftStore.setState({ [flag]: true })
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
      mounted.draftStore.setState({ streaming: false, isProcessing: false })
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
      const actions = useChatActions({
        ...(mounted.hook as any).flat,
        messages: rows,
        history: mounted.draftStore.getState().history
      })
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
      const actions = useChatActions({
        ...(mounted.hook as any).flat,
        messages: rows,
        history: mounted.draftStore.getState().history
      })
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
  "full native owned regeneration through $mode / $boundary keeps the canonical input and original reply",
  async ({ mode, boundary }) => {
    const options = mode === "character" ? createHookOptions() : ordinaryOptions()
    options.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
      { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
    messageStoreState.editRows = () => options.messages
    const original = structuredClone(options.messages)
    const hook = renderHook(() => useChatActions(options as never))
    options.setMessages.mockClear(); options.setHistory.mockClear()
    const store = (await import("@/store/option")).useStoreMessageOption
    const origin = h1.controller.getCurrent().view
    const admission = { version: 1, owner_key: origin.owner_key, conversation_id: nativeChatId,
      input_message_id: "u-old", input_message_revision: "1", selection_digest: "ORIGINAL INPUT ADMISSION",
      originating_selection_revision: 0, messages: [] }
    h1.read.mockResolvedValue({ id: "u-old", conversation_id: nativeChatId, sender: "user", parent_message_id: null,
      content: "old question", has_image: false, version: 1 })
    h1.wire.mockImplementation(async function* () {
      expect(store.getState().detachedHistoryResponse).toBeTruthy()
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
    expect(options.messages).toEqual(original)
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(options.setHistory).not.toHaveBeenCalled()
    expect(store.getState().detachedHistoryResponse).toBeNull()
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalled()
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


it.each(["normal", "character"])("full $mode native regeneration preserves literal whitespace with every image through final request serialization", async mode => {
    const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    const decorate = (captured: ReturnType<typeof captureFor>) => ({ ...captured,
      selected_content: captured.selected_content.map(row => row.id === "u-old" ? { ...row, message: "  \n", images: [image, image],
        extra_metadata: { sender_role: "user", image_details: ["high", "low"] } } : row) })
    h1.controller.getCurrent().capture = decorate(h1.controller.getCurrent().capture)
    h1.capture.mockImplementation(async (_id, request) => decorate(captureFor(request.view)))
    const options = mode === "character" ? createHookOptions() : ordinaryOptions()
    options.messages = [{ id: "u-old", name: "You", isBot: false, message: "  \n", images: [image, image], sources: [] },
      { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
    messageStoreState.editRows = () => options.messages
    const hook = renderHook(() => useChatActions(options as never))
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


it("full native regeneration cannot follow an unmounted source even when its outer H1 owner survives", async () => {
  const options = ordinaryOptions()
  options.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
  messageStoreState.editRows = () => options.messages
  const hook = renderHook(() => useChatActions(options as never))
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


it.each(["regenerate", "send"])("full an older regeneration cannot clear a newer request through another mounted hook: %s", async kind => {
    let sharedController: AbortController | null = null
    const store = (await import("@/store/option")).useStoreMessageOption
    const optionsFor = () => {
      const options = kind === "send" ? createHookOptions() : ordinaryOptions()
      options.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
        { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
      options.setStreaming.mockImplementation(value => store.setState({ streaming: value }))
      options.setIsProcessing.mockImplementation(value => store.setState({ isProcessing: value }))
      options.setAbortController.mockImplementation(value => { sharedController = value })
      return { ...options, getAbortController: () => sharedController }
    }
    const firstOptions = optionsFor(), secondOptions = optionsFor()
    messageStoreState.editRows = () => firstOptions.messages
    const firstHook = renderHook(() => useChatActions(firstOptions as never))
    const secondHook = renderHook(() => useChatActions(secondOptions as never))
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
      store.setState({ streaming: false, isProcessing: false, chatDraftRevision: store.getState().chatDraftRevision + 1 })
      secondHook.rerender()
      act(() => { second = kind === "send" ? secondHook.result.current.onSubmit({ message: "Next B", image: "" }) : secondHook.result.current.regenerateLastMessage() })
      await enteredB
      const controllerB = sharedController, markerB = store.getState().detachedHistoryResponse
      const processingB = store.getState().isProcessing
      releaseA(); await act(async () => { await first })
      expect(sharedController).toBe(controllerB)
      expect(store.getState().streaming).toBe(true)
      expect(store.getState().isProcessing).toBe(processingB)
      expect(store.getState().detachedHistoryResponse).toBe(markerB)
      releaseB(); await act(async () => { expect(await second).toEqual({ status: "submitted" }) })
      expect(store.getState().streaming).toBe(false)
    } finally {
      releaseA(); releaseB(); await first; await second; firstHook.unmount(); secondHook.unmount()
      store.setState({ streaming: false, isProcessing: false })
    }
  })


it("full Stop releases native busy state through the actual shared provider after the aborted turn finishes", async () => {
  const store = (await import("@/store/option")).useStoreMessageOption
  const options = ordinaryOptions()
  options.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
  messageStoreState.editRows = () => options.messages
  options.setStreaming.mockImplementation(value => store.setState({ streaming: value }))
  options.setIsProcessing.mockImplementation(value => store.setState({ isProcessing: value }))
  const hook = renderHook(() => {
    const context = usePageAssist()
    return { context, actions: useChatActions({ ...options, abortController: context.controller,
      setAbortController: context.setController, getAbortController: context.getController } as never) }
  }, { wrapper: PageAssistProvider })
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
    expect(store.getState().streaming).toBe(false)
    expect(store.getState().isProcessing).toBe(false)
    expect(h1.controller.followResult).not.toHaveBeenCalled()
    expect(focusRequests).not.toHaveBeenCalled()
  } finally { window.removeEventListener("tldw:focus-composer", focusRequests); release(); await pending; store.setState({ streaming: false, isProcessing: false }); hook.unmount() }
})


it.each(["marker", "busy"])("full a synchronous replacement during %s publication keeps the newer provider owner", async publication => {
  const store = (await import("@/store/option")).useStoreMessageOption
  const options = ordinaryOptions()
  options.messages = [{ id: "u-old", name: "You", isBot: false, message: "old question", sources: [] },
    { id: "a1", name: "Assistant", isBot: true, message: "same answer", sources: [] }] as never
  messageStoreState.editRows = () => options.messages
  options.setStreaming.mockImplementation(value => store.setState({ streaming: value }))
  options.setIsProcessing.mockImplementation(value => store.setState({ isProcessing: value }))
  const hook = renderHook(() => {
    const context = usePageAssist()
    return { context, actions: useChatActions({ ...options, abortController: context.controller,
      setAbortController: context.setController, getAbortController: context.getController } as never) }
  }, { wrapper: PageAssistProvider })
  const newer = new AbortController(), marker = { originalReply: { id: "newer", isBot: true, message: "B" } }
  let replaced = false
  const actualStore = (await vi.importActual<typeof import("@/store/option")>("@/store/option")).useStoreMessageOption
  const unsubscribe = actualStore.subscribe((state, previous) => {
    if (!replaced && (publication === "marker"
      ? state.detachedHistoryResponse && state.detachedHistoryResponse !== previous.detachedHistoryResponse
      : state.streaming && !previous.streaming)) {
      replaced = true
      hook.result.current.context.setController(newer)
      store.setState({ chatDraftRevision: state.chatDraftRevision + 1, streaming: true, isProcessing: true, detachedHistoryResponse: marker })
    }
  })
  try {
    await act(async () => { await hook.result.current.actions.regenerateLastMessage().catch(() => undefined) })
    expect(replaced).toBe(true)
    expect(hook.result.current.context.getController!()).toBe(newer)
    expect(store.getState().streaming).toBe(true)
    expect(store.getState().isProcessing).toBe(true)
    expect(store.getState().detachedHistoryResponse).toBe(marker)
    expect(h1.wire).not.toHaveBeenCalled()
  } finally { unsubscribe(); store.setState({ streaming: false, isProcessing: false }); hook.unmount() }
})


const mountNativeLeafDelete = async (character = false) => {
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
    const mounted = await mountNativeLeafDelete()
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
      const notification = mounted.options.notification
      mounted.publish.mockClear()
      notification.warning.mockClear()
      await act(async () => {
        release()
        expect(await pending).toBe(false)
      })
      expect(notification.warning).not.toHaveBeenCalled()
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
    store.useStoreMessageOption.getState().startNewChatDraft()
    messageStoreState.draftFields = bridgeDraftFields
    store.useStoreMessageOption.setState({
      streaming: false,
      isProcessing: false
    })
    const origin = h1.controller.getCurrent()
    origin.status = "idle"
    origin.owner = null
    origin.capture = null
    origin.view = null
    const options = {
      ...ordinaryOptions(),
      historyId: "local-delete",
      serverChatId: null,
      messages: [
        {
          id: "local-leaf",
          isBot: true,
          message: "saved answer",
          sources: [],
          serverMessageId: undefined as string | undefined
        }
      ]
    }
    installLocalSource("local-delete")
    messageStoreState.editRows = () => options.messages
    const mounted = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    const setters = {
      messages: options.setMessages,
      history: options.setHistory
    }

    setters.messages.mockReset()
    setters.history.mockReset()
    return { ...mounted, setters, store, options }
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
    mounted.store.useStoreMessageOption.setState({ streaming: true })
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
    mounted.options.messages[0].serverMessageId = "server-leaf"
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
  it("rejects a changed live row from a retained Delete confirmation", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    const retained = mounted.result.current.deleteMessage
    mounted.options.messages = [
      { ...mounted.options.messages[0], message: "newer content" }
    ]
    mounted.rerender()
    await expect(retained(0)).rejects.toThrow("message_changed")
    expect(helpers.removeMessageById).not.toHaveBeenCalled()
  })
  it("keeps Temporary Delete usable without a canonical controller", async () => {
    const helpers = await import("@/db/dexie/helpers")
    const mounted = await localDelete()
    h1.controller = null
    mounted.options.temporaryChat = true
    mounted.options.historyId = "temp"
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
    messageStoreState.editRows = () => [
      { ...mounted.options.messages[0], message: "newer content" }
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
    messageStoreState.editRows = () => [
      { ...mounted.options.messages[0], message: "newer content" }
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
      mounted.options.messages = rows
      if (temporary) {
        h1.controller = null
        mounted.options.temporaryChat = true
        mounted.options.historyId = "temp"
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
    mounted.options.replyTarget = { id: "local-leaf", preview: "saved answer" }
    mounted.store.useStoreMessageOption
      .getState()
      .setReplyTarget(mounted.options.replyTarget)
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
    const clearedBeforeAck = mounted.options.clearReplyTarget.mock.calls.length
    mounted.store.useStoreMessageOption
      .getState()
      .setReplyTarget({ id: "other-reply", preview: "new reply" })
    release()
    await pending
    expect(clearedBeforeAck).toBe(0)
    expect(mounted.options.clearReplyTarget).not.toHaveBeenCalled()
  })
  it.each([false, true])(
    "preserves a surviving message edited during held Delete (diagnostic=%s)",
    async (diagnostic) => {
      const helpers = await import("@/db/dexie/helpers")
      const mounted = await localDelete()
      const rows = [
        {
          ...mounted.options.messages[0],
          generationInfo: diagnostic
            ? {
                mode: "rag",
                grounded: false,
                reason: "selected_source_retrieval_failed"
              }
            : undefined
        },
        {
          id: "other-leaf",
          isBot: true,
          message: "original sibling",
          sources: [],
          serverMessageId: undefined
        }
      ]
      mounted.options.messages = rows
      mounted.rerender()
      let release!: () => void
      vi.mocked(helpers.removeMessageById).mockImplementation(
        () =>
          new Promise((resolve) => {
            release = () => resolve(undefined)
          })
      )
      const pending = mounted.result.current.deleteMessage(0)
      await vi.waitFor(() =>
        expect(helpers.removeMessageById).toHaveBeenCalled()
      )
      const updated = [rows[0], { ...rows[1], message: "edited sibling" }]
      mounted.options.messages = updated
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
              updated.map((row) => ({
                role: "assistant",
                content: row.message
              }))
            )
          : history
      ).toEqual([{ role: "assistant", content: "edited sibling" }])
    }
  )
})

it.each(["confirmed", "wrong-parent", "stream-evidence", "explicit-branch"])(
  "native branch refusal distinguishes %s before any observed turn",
  async (kind) => {
    const options = createHookOptions()
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
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
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
      expect(options.setMessages).not.toHaveBeenCalled()
      expect(h1.controller.holdStaleSend).toHaveBeenCalledOnce()
    } else expect(h1.controller.holdStaleSend).not.toHaveBeenCalled()
  }
)

it.each(["H1", "account", "draft", "journal failure"])(
  "native confirmed refusal owns its recovery through %s",
  async (boundary) => {
    const options = createHookOptions()
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
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
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
      expect(options.notification.error).not.toHaveBeenCalled()
    expect(h1.dismiss).not.toHaveBeenCalled()
  }
)

it("held full Character contract discovery cannot dispatch from an unmounted composer whose outer H1 survives", async () => {
  const options = createHookOptions()
  let release!: () => void
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  h1.spec.mockImplementation(async () => {
    await held
    return null
  })
  const hook = renderHook(() =>
    useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
  )
  let pending!: Promise<unknown>
  act(() => {
    pending = hook.result.current.onSubmit({
      message: "Original unsent full input",
      image: ""
    })
  })
  await waitFor(() => expect(h1.spec).toHaveBeenCalledOnce())
  hook.unmount()
  await act(async () => {
    release()
    await pending
  })
  expect(h1.wire).not.toHaveBeenCalled()
  expect(addChatMessageMock).not.toHaveBeenCalled()
  expect(h1.recover).not.toHaveBeenCalled()
})


it("an acknowledged first character creation uses saved Roleplay before UI metadata arrives", async () => {
  const mounted = await mountedNativeOpening()
  const sender = await import("../roleplay-resume-send")
  const send = vi
    .spyOn(sender, "sendSavedRoleplay")
    .mockResolvedValue(true)
  try {
    await act(async () => {
      await mounted.hook.result.current.onSubmit({
        message: "Exact first character input",
        image: ""
      })
    })
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(send).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Exact first character input",
        serverChatId: nativeChatId
      })
    )
    expect(h1.wire).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
  } finally {
    mounted.hook.unmount()
    mounted.selection.unmount()
    send.mockRestore()
  }
})


it("mounted Full saved Roleplay dispatch bypasses mutable live Scene and model defaults", async () => {
  const sender = await import("../roleplay-resume-send")
  const send = vi
    .spyOn(sender, "sendSavedRoleplay")
    .mockResolvedValue(true)
  const scenes = await import("@/services/actor-settings")
  vi.mocked(scenes.getActorSceneForChat).mockRejectedValue(
    new Error("live Scene forbidden")
  )
  messageStoreState.value.serverChatCharacterId = "12"
  messageStoreState.value.serverChatAssistantKind = "character"
  messageStoreState.value.selectedModel = ""
  const options = { ...createHookOptions(), selectedModel: "" }
  const hook = renderHook(() =>
    useChatActions(options as Parameters<typeof useChatActions>[0])
  )
  try {
    await act(async () => {
      await hook.result.current.onSubmit({
        message: "  Exact Full literal\n",
        image: ""
      })
    })
    expect(send).toHaveBeenCalledWith(
      expect.objectContaining({
        controller: h1.controller,
        message: "  Exact Full literal\n",
        serverChatId: nativeChatId
      })
    )
    expect(scenes.getActorSceneForChat).not.toHaveBeenCalled()
    expect(h1.wire).not.toHaveBeenCalled()
    expect(addChatMessageMock).not.toHaveBeenCalled()
  } finally {
    hook.unmount()
    send.mockRestore()
  }
})


it("an acknowledged workspace character creation keeps its existing native path", async () => {
  const mounted = await mountedNativeOpening()
  const sender = await import("../roleplay-resume-send")
  const send = vi
    .spyOn(sender, "sendSavedRoleplay")
    .mockResolvedValue(true)
  const { sendNativeHistoryCharacter } = await import(
    "../native-history-character-send"
  )
  const workspace = {
    type: "workspace" as const,
    workspaceId: "owned-workspace"
  }
  const address = mounted.draftStore.getState().chatDraftAddress!
  expect(
    mounted.draftStore
      .getState()
      .setChatDraftAddress({ ...address, scope: workspace })
  ).toBe(true)
  createChatMock.mockImplementation((body) => ({
    ...createdNativeChat(body),
    scope_type: "workspace",
    workspace_id: workspace.workspaceId
  }))
  try {
    await act(async () => {
      await sendNativeHistoryCharacter({
        controller: mounted.selection.result.current,
        surface: "playground",
        originIsCurrent: () => true,
        sourceIsCurrent: () => true,
        signal: new AbortController().signal,
        temporary: false,
        scope: workspace,
        characterId: "12",
        model: "deepseek-chat",
        settings: { apiProvider: "openai" },
        message: "Workspace first character input",
        image: "",
        t: ((key: string) => key) as Parameters<
          typeof sendNativeHistoryCharacter
        >[0]["t"],
        setServerChatId: vi.fn(),
        setMessages: vi.fn(),
        releaseActivity: vi.fn()
      })
    })
    expect(createChatMock).toHaveBeenCalledOnce()
    expect(mounted.selection.result.current.getCurrent().owner?.scope).toEqual(
      workspace
    )
    expect(send).not.toHaveBeenCalled()
  } finally {
    mounted.hook.unmount()
    mounted.selection.unmount()
    send.mockRestore()
  }
})
