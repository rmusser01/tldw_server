import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, useEffect, useLayoutEffect, useMemo, useState, type PropsWithChildren } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { App } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatTldwOptions } from "@/models/ChatTldw"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import * as servicePrompts from "@/services/service-prompts"
import type { HistoryInfo, Message } from "@/db/dexie/types"

// Table-engine double only: creation, public sanitization, rereads, mounted
// selection, capture, admission and settlement use production implementations.
// The Critical browser journeys separately qualify native IndexedDB/reload.
const storage = vi.hoisted(() => {
  type Row = Record<string, unknown>
  type Collection = {
    toArray: () => Promise<Row[]>
    and: (predicate: (row: Row) => boolean) => Collection
    delete: () => Promise<void>
  }
  const tables: Record<string, {
    rows: Map<string, Row>
    get: (id: unknown) => Promise<Row | undefined>
    add: (row: Row) => Promise<void>
    put: (row: Row) => Promise<void>
    bulkPut: (rows: Row[]) => Promise<void>
    update: (id: unknown, patch: Row) => Promise<void>
    delete: (id: unknown) => Promise<void>
    toArray: () => Promise<Row[]>
    where: (field: string) => { equals: (value: unknown) => Collection }
  }> = {}
  const control = { beforeHistoryAdd: undefined as (() => Promise<void>) | undefined }
  for (const name of ["chatHistories", "messages", "userSettings", "sessionFiles", "compareStates", "historySelections", "historyProjections", "forkOperations", "modelNickname", "prompts"]) {
    const rows = new Map<string, Row>()
    const key = (row: Row) => JSON.stringify(name === "historySelections"
      ? [row.profile_id, row.client_session_id, row.owner_key, row.conversation_id]
      : row.id)
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: predicate => collection(row => matches(row) && predicate(row)),
      delete: async () => {
        for (const [id, row] of rows) if (matches(row)) rows.delete(id)
      }
    })
    tables[name] = {
      rows,
      get: async id => structuredClone(rows.get(JSON.stringify(id))),
      add: async row => {
        if (name === "chatHistories") await control.beforeHistoryAdd?.()
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
      },
      put: async row => { rows.set(key(row), structuredClone(row)) },
      bulkPut: async values => { values.forEach(row => rows.set(key(row), structuredClone(row))) },
      update: async (id, patch) => { rows.set(JSON.stringify(id), { ...rows.get(JSON.stringify(id)), ...structuredClone(patch) }) },
      delete: async id => { rows.delete(JSON.stringify(id)) },
      toArray: async () => structuredClone([...rows.values()]),
      where: field => ({ equals: value => collection(row => row[field] === value) })
    }
  }
  let tail = Promise.resolve()
  return {
    tables,
    control,
    db: {
      ...tables,
      transaction: (_mode: unknown, _tables: unknown, operation: (tx: { abort: () => void }) => Promise<unknown>) => {
        const result = tail.then(async () => {
          const copies = Object.values(tables).map(table => new Map(table.rows))
          try { return await operation({ abort: () => { throw new Error("aborted") } }) }
          catch (error) {
            Object.values(tables).forEach((table, index) => {
              table.rows.clear()
              copies[index].forEach((row, key) => table.rows.set(key, row))
            })
            throw error
          }
        })
        tail = result.then(() => {}, () => {})
        return result
      }
    }
  }
})
vi.mock("@/db/dexie/schema", () => ({ db: storage.db }))
const prompt = vi.hoisted(() => ({ scope: {
  config: { serverUrl: "https://chat.test", authMode: "multi-user" as const },
  userId: "alice"
} }))
vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => prompt.scope,
  subscribeToServicePromptConfigChanges: () => () => {}
}))
vi.mock("~/services/tldw-server", async original => ({
  ...await original<typeof import("~/services/tldw-server")>(),
  systemPromptForNonRagOption: async () => ""
}))
vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return { pageAssistModel: async (options: ChatTldwOptions) => {
    await preparation.beforeModel()
    if (preparation.realFactory) {
      const actual = await vi.importActual<typeof import("@/models")>("@/models")
      return actual.pageAssistModel(options as unknown as Parameters<typeof actual.pageAssistModel>[0])
    }
    return new ChatTldw({ ...options, streaming: true, apiProvider: "openai", supportsMultimodal: preparation.multimodal })
  } }
})

const preparation = vi.hoisted(() => ({ beforeModel: vi.fn(), multimodal: false, realFactory: false }))

import { normalChatMode } from "../normalChatMode"
import { protectedEditRequestSchema } from "@/services/__tests__/history-edit-fixture"
import { useChatActions } from "@/hooks/chat/useChatActions"
import { useServerChatHistoryId } from "@/hooks/chat/useServerChatHistoryId"
import { useKeepInterruptedHistoryTurns } from "@/hooks/chat/useKeepInterruptedHistoryTurns"
import { useMessage } from "@/hooks/useMessage"
import { useMessageOption } from "@/hooks/useMessageOption"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { normalizeChatSubmitResult, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import * as tldwServer from "@/services/tldw-server"
import type { ChatDocuments } from "@/models/ChatTypes"
import type { UploadedFile } from "@/db/dexie/types"
import { HistorySelectionContext, useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { keepHistoryTurnRecovery } from "@/services/history-turn-keep"
import { loadHistoryTurnRecoveries } from "@/db/dexie/history-selection"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { formatSelectedHistory, saveHistory, saveMessage } from "@/db/dexie/helpers"
import { saveMessageOnSuccess, saveMessageOnError } from "@/hooks/chat-helper"
import { captureHistorySnapshot } from "@/services/chat-history-selection"
import { assertHistoryConversationTarget, type NativeHistoryOwnerV1 } from "@/services/chat-history-selection"
import { linkServerChatMirror } from "@/db/dexie/server-chat-mirror"
import { tldwChat, tldwModels } from "@/services/tldw"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { HistoryLoadReceipt, HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreMessageOption } from "@/store/option"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase } from "@/types/connection"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { Storage } from "@plasmohq/storage"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { readChatCreationRecovery } from "@/services/pending-chat-promotion"
import { conversationDraftKey, type ChatDraftHandle } from "@/utils/chat-draft-address"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import {
  getActorSceneForChat,
  saveActorSettingsForChat,
  type ActorSceneRead
} from "@/services/actor-settings"
import { createDefaultActorSettings } from "@/types/actor"
import { useActorStore } from "@/store/actor"
import { captureCurrentActorSceneTarget } from "@/hooks/chat/actor-scene-source"
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import { resolveHistorySelection } from "@/utils/history-selection"
import { setTitleGenEnabled } from "@/services/title"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

let newServerId = "new-server"
const createdServerChat = async (body: { id?: string }) => {
  if (!body.id) throw new Error("fixture requires the reserved client chat ID")
  newServerId = body.id
  return { id: body.id } as never
}

const ACCOUNT = '["https://chat.test","multi-user","manual",null,"alice",null]'
const snapshot = (): ServicePromptSnapshot => ({
  scopeKey: "alice-scope", requestScope: structuredClone(prompt.scope),
  capability: "supported", definitions: {},
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: () => {}
})
// Issued IDs use the same physical setters as the mounted callers. Spies below
// retain observation without replacing the store publication with a no-op.
const publishFixtureHistoryId = (id: string) =>
  useStoreMessageOption.getState().setHistoryId(id)
const publishFixtureServerChatId = (id: string) =>
  useStoreMessageOption.getState().setServerChatId(id)
const send = (
  controller: HistorySelectionController,
  options: {
    historyId?: string
    serverChatId?: string
    actorScene?: ActorSceneRead
    message?: string
    image?: string
    snapshot?: ServicePromptSnapshot | null
    originIsCurrent?: () => boolean
    sourceIsCurrent?: () => boolean
    setHistoryId?: (id: string) => void
    toolChoice?: Parameters<typeof normalChatMode>[6]["toolChoice"]
    fixedOverrides?: boolean
    setServerChatId?: (id: string) => void
    saveOnSuccess?: typeof saveMessageOnSuccess
    surface?: "playground" | "sidepanel"
  } = {}
) => {
  const params: Parameters<typeof normalChatMode>[6] = {
    actorScene: options.actorScene,
    actorSettings: options.actorScene?.settings,
    selectedModel: "test",
    selectedSystemPrompt: "",
    useOCR: false,
    currentChatModelSettings: {},
    servicePromptSnapshot:
      options.snapshot === null ? undefined : options.snapshot ?? snapshot(),
    toolChoice:
      options.toolChoice ?? useStoreMessageOption.getState().toolChoice,
    selectionSource: {
      model: options.fixedOverrides ? "explicit" : "global",
      toolChoice: options.fixedOverrides ? "explicit" : "global"
    },
    historySelection: {
      controller,
      originIsCurrent: options.originIsCurrent ?? (() => true),
      ...(options.sourceIsCurrent
        ? { sourceIsCurrent: options.sourceIsCurrent }
        : {}),
      createServerChat: Boolean(options.setServerChatId)
    },
    historyId: options.historyId ?? null,
    serverChatId: options.serverChatId,
    setServerChatId: options.setServerChatId,
    ...{ surface: options.surface ?? "playground" },
    setHistoryId: options.setHistoryId ?? publishFixtureHistoryId,
    setMessages: () => {},
    setHistory: () => {},
    setIsProcessing: () => {},
    setStreaming: () => {},
    setAbortController: () => {},
    saveMessageOnSuccess: options.saveOnSuccess ?? saveMessageOnSuccess,
    saveMessageOnError,
    userMessageId: "first-user",
    assistantMessageId: "first-answer"
  }
  return normalChatMode(
    options.message ?? "First question",
    options.image ?? "",
    false,
    [],
    [],
    new AbortController().signal,
    params
  )
}
const publicHistory = (id: string): HistoryInfo => ({ id, title: "Imported", createdAt: 1, is_rag: false, message_source: "web-ui" })
const savedMessages = async (): Promise<Message[]> => await storage.tables.messages.toArray() as Message[]
const emptyServerCapture = (request: Parameters<typeof tldwClient.captureHistorySelection>[1], id = newServerId) => {
  const view = { ...request.view, owner_key: "server-owner" }
  return {
    status: "captured", purpose: request.purpose, view, rows: [], selected_content: [],
    snapshot: { version: 1, owner_key: view.owner_key, conversation_id: id, nodes: [],
      fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source",
      storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } },
    storage_context_digest: "storage"
  } as never
}

beforeEach(() => {
  newServerId = "new-server"
  vi.restoreAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  useStoreMessageOption.getState().setServerChatId(null)
  useStoreMessageOption.setState({
    historyId: null,
    messages: [],
    history: [],
    temporaryChat: false,
    streaming: false,
    isProcessing: false,
    isLoading: false,
    detachedHistoryResponse: null
  })
  useStoreMessageOption.getState().startNewChatDraft(false)
  const tails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: (key: string, work: () => unknown) => {
        const next = (tails.get(key) ?? Promise.resolve()).then(work)
        tails.set(
          key,
          next.catch(() => undefined)
        )
        return next
      }
    }
  })
  Object.values(storage.tables).forEach((table) => table.rows.clear())
  storage.control.beforeHistoryAdd = undefined
  prompt.scope.userId = "alice"
  preparation.beforeModel.mockReset()
  preparation.multimodal = false
  preparation.realFactory = false
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  useStoreMessageOption.setState({ selectedModel: "test", toolChoice: "none" })
  vi.spyOn(tldwClient, "getOpenAPISpec").mockResolvedValue({
    paths: Object.fromEntries(
      ["/api/v1/chats/{chat_id}/messages", "/api/v1/chat/completions"].map(
        (path) => [
          path,
          {
            post: {
              requestBody: {
                content: {
                  "application/json": {
                    schema: {
                      ...protectedEditRequestSchema()
                    }
                  }
                }
              }
            }
          }
        ]
      )
    )
  })

  // This IO double declares successful transport entry. Separate SDK/proxy
  // integration cases qualify the real waits and guards before onDispatch.
  vi.spyOn(tldwChat, "streamMessage").mockImplementation(
    async function* (_messages, options) {
      options.onDispatch?.()
      yield "First answer"
    }
  )
})

afterEach(() => { vi.restoreAllMocks() })


describe("fresh Compare original scene through real H1 first bind", () => {
  it.each([
    { name: "projects one shared question and dispatches both models after its original scene is bound", rejectSecondBranch: false },
    { name: "rolls back only a scoped rejected Compare cluster while retaining its accepted chat and original scene", rejectSecondBranch: true }
  ])("$name", async ({ rejectSecondBranch }) => {
    const initialCore = useStoreMessageOption.getState()
    const question = "One shared Compare question"
    const models = ["model-a", "model-b"]
    const scoped = snapshot()
    const originalScope = structuredClone(scoped.requestScope)
    let originMounted = true
    const draftAccount = Object.freeze({
      requestScope: originalScope,
      isCurrent: () =>
        originMounted &&
        !scoped.scopeInvalidatedSignal.aborted &&
        chatDraftOwnerKey(prompt.scope) === chatDraftOwnerKey(originalScope),
      retryVerification: vi.fn()
    })
    useStoreMessageOption.getState().setServerChatId(null)
    useStoreMessageOption.setState({
      historyId: null,
      messages: [],
      history: [],
      selectedModel: models[0],
      chatMode: "normal",
      toolChoice: "none",
      webSearch: false,
      useOCR: false,
      selectedSystemPrompt: null,
      selectedQuickPrompt: null,
      selectedKnowledge: null,
      fileRetrievalEnabled: false,
      ragMediaIds: null,
      contextFiles: [],
      uploadedFiles: [],
      documentContext: null,
      replyTarget: null,
      messageSteeringMode: "none",
      messageSteeringForceNarrate: false,
      compareMode: true,
      compareSelectedModels: models
    })
    useStoreMessageOption.getState().startNewChatDraft(false)
    useStoreChatModelSettings.getState().setApiProvider("openai")
    useActorStore.getState().reset()
    await setTitleGenEnabled(false)
    vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(scoped)
    const createChat = vi.spyOn(tldwClient, "createChat")
    const admission = vi.spyOn(tldwClient, "addChatMessage")

    const source = useStoreMessageOption.getState()
    expect(source.chatDraftAddress?.kind).toBe("fresh")
    expect(source.historyId).toBeNull()
    expect(source.serverChatId).toBeNull()
    const sourceTarget = captureActorSettingsTarget({
      address: source.chatDraftAddress,
      requestScope: originalScope,
      accountIsCurrent: draftAccount.isCurrent,
      isCurrent: () => {
        const current = useStoreMessageOption.getState()
        return (
          current.chatDraftAddress === source.chatDraftAddress &&
          current.chatDraftRevision === source.chatDraftRevision &&
          current.historyId === null &&
          current.serverChatId === null
        )
      }
    })!
    const scene = {
      ...createDefaultActorSettings(),
      isEnabled: true,
      notes: "Saved original fresh Compare scene"
    }
    expect(
      await saveActorSettingsForChat({
        historyId: null,
        serverChatId: null,
        target: sourceTarget,
        expectedRecord: null,
        settings: scene
      })
    ).toBe(true)
    const originalScene = await getActorSceneForChat({ target: sourceTarget })
    expect(originalScene.originalRecord).not.toBeNull()
    expect(useActorStore.getState().applied).toBeNull()

    let selection!: HistorySelectionController
    const captures: HistorySelectionCaptureV1[] = []
    const projectedRows: ReturnType<typeof formatSelectedHistory>[] = []
    const projectCapture = vi.fn((capture: HistorySelectionCaptureV1) => {
      captures.push(capture)
      const display = formatSelectedHistory(capture)
      // The production Playground publishes both row arrays in one core update.
      useStoreMessageOption.setState({
        history: display.history,
        messages: display.messages
      })
      const current = useStoreMessageOption.getState()
      projectedRows.push({
        history: current.history,
        messages: current.messages
      })
    })
    const Wrapper = ({ children }: PropsWithChildren) => {
      const current = useHistorySelection({ onCapture: projectCapture })
      const value = useMemo(() => ({ ...current, draftAccount }), [current])
      useLayoutEffect(() => { selection = value }, [value])
      return createElement(
        HistorySelectionContext.Provider,
        { value },
        children
      )
    }
    type ActionOptions = Parameters<typeof useChatActions>[0]
    const notification = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    } as unknown as ActionOptions["notification"]
    const markCompareHistoryCreated = vi.fn()
    const mounted = renderHook(() => {
      const core = useStoreMessageOption()
      const settings = useStoreChatModelSettings()
      const [abortController, setAbortController] =
        useState<AbortController | null>(null)
      return useChatActions({
        ...core,
        t: ((_key: string, fallback?: unknown) =>
          typeof fallback === "string" ? fallback : _key) as ActionOptions["t"],
        notification,
        abortController,
        setAbortController,
        currentChatModelSettings: settings,
        compareModeActive: core.compareMode,
        compareMaxModels: 3,
        compareFeatureEnabled: true,
        markCompareHistoryCreated,
        ensureServerChatHistoryId: vi.fn(async () => null),
        messageSteeringPrompts: null,
        invalidateServerChatHistory: vi.fn(),
        selectedCharacter: null,
        selectedAssistant: null
      })
    }, { wrapper: Wrapper })
    expect(selection.getCurrent()).toMatchObject({
      status: "idle",
      owner: null,
      view: null,
      capture: null
    })

    let loadedReceipt!: HistoryLoadReceipt
    const loadConversation = selection.loadConversation
    const load = vi.spyOn(selection, "loadConversation").mockImplementation(
      (target, reference, onLoaded) =>
        loadConversation(target, reference, receipt => {
          loadedReceipt = receipt
          onLoaded?.(receipt)
        })
    )
    const applied = vi.spyOn(useActorStore.getState(), "applyScene")
    applied.mockClear()
    const dispatchStates: {
      address: typeof source.chatDraftAddress
      appliedKey: string | undefined
    }[] = []
    let bothBranchesEntered!: () => void
    let releaseProviderResults!: () => void
    const providersEntered = new Promise<void>(resolve => { bothBranchesEntered = resolve })
    const providerResultsHeld = new Promise<void>(resolve => { releaseProviderResults = resolve })
    vi.mocked(tldwChat.streamMessage).mockImplementation(
      async function* (_messages, options) {
        options.onDispatch?.()
        dispatchStates.push({
          address: useStoreMessageOption.getState().chatDraftAddress,
          appliedKey: useActorStore.getState().applied?.target.key
        })
        if (dispatchStates.length === models.length) bothBranchesEntered()
        if (rejectSecondBranch) {
          await providerResultsHeld
          if (options?.model === "model-b") throw createServicePromptScopeChangedError()
        }
        yield "Compare answer from " + options?.model
      }
    )
    const sourceKey = "actorSettings:owned:" + sourceTarget.key
    let destinationKey: string | undefined
    let pendingRecord: unknown
    let writeEntered!: () => void
    let releaseWrite!: () => void
    const entered = new Promise<void>(resolve => { writeEntered = resolve })
    const held = new Promise<void>(resolve => { releaseWrite = resolve })
    const originalSet = Storage.prototype.set
    vi.spyOn(Storage.prototype, "set").mockImplementation(
      async function (key, value) {
        if (
          key.startsWith("actorSettings:owned:") &&
          key !== sourceKey &&
          !destinationKey
        ) {
          destinationKey = key
          pendingRecord = value
          writeEntered()
          await held
        }
        return originalSet.call(this, key, value)
      }
    )
    let attempt: ReturnType<typeof mounted.result.current.onSubmit> | undefined
    try {
      await act(async () => {
        attempt = mounted.result.current.onSubmit({ message: question, image: "" })
        await Promise.race([
          entered,
          attempt.then(result => {
            throw new Error(
              "Compare finished before its real destination Actor write: " +
                JSON.stringify(result)
            )
          })
        ])
      })
      expect(load).toHaveBeenCalledOnce()
      expect(loadedReceipt.owner.kind).toBe("local")
      const acceptedAddress = useStoreMessageOption.getState().chatDraftAddress
      if (loadedReceipt.owner.kind !== "local" || acceptedAddress?.kind !== "local")
        throw new Error("Expected the actual accepted local Compare bind")
      const historyId = loadedReceipt.owner.conversation_id
      const acceptedProfileId = loadedReceipt.owner.profile_id
      const acceptedOwnerKey = loadedReceipt.view.owner_key
      expect(acceptedAddress).toEqual({
        kind: "local",
        id: historyId,
        profileId: loadedReceipt.owner.profile_id
      })
      expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({
        kind: "bind",
        sourceAddress: source.chatDraftAddress
      })
      expect(selection.getCurrent().owner).toBe(loadedReceipt.owner)
      expect(selection.getCurrent().view).toBe(loadedReceipt.view)
      expect(selection.getCurrent().status).toBe("ready")
      expect(projectCapture).toHaveBeenCalledOnce()
      expect(captures[0].view).toBe(loadedReceipt.view)
      expect(captures[0].selected_content).toHaveLength(1)
      const persistedUser = (await savedMessages()).filter(row => row.role === "user")
      expect(persistedUser).toHaveLength(1)
      const sharedUser = persistedUser[0]
      expect(sharedUser).toMatchObject({
        history_id: historyId,
        content: question,
        messageType: "compare:user"
      })
      expect(typeof sharedUser.id).toBe("string")
      expect(typeof sharedUser.clusterId).toBe("string")
      expect(captures[0].rows.map(row => row.id)).toEqual([sharedUser.id])
      expect(projectedRows[0].messages.map(row => row.id)).toEqual([sharedUser.id])
      expect(projectedRows[0].history.filter(row => row.role === "user")).toEqual([
        expect.objectContaining({ content: question })
      ])
      const boundTarget = captureActorSettingsTarget({
        address: acceptedAddress,
        ...loadedReceipt,
        requestScope: originalScope,
        accountIsCurrent: draftAccount.isCurrent,
        isCurrent: () =>
          useStoreMessageOption.getState().chatDraftAddress === acceptedAddress &&
          selection.getCurrent().owner === loadedReceipt.owner &&
          selection.getCurrent().view === loadedReceipt.view
      })!
      expect(destinationKey).toBe("actorSettings:owned:" + boundTarget.key)
      expect(pendingRecord).toEqual(originalScene.originalRecord)
      expect(await new Storage({ area: "local" }).get(destinationKey!)).toBeUndefined()
      expect(useActorStore.getState().applied).toBeNull()
      expect(applied).not.toHaveBeenCalled()
      expect(tldwChat.streamMessage).not.toHaveBeenCalled()

      await act(async () => {
        releaseWrite()
        if (rejectSecondBranch) {
          await Promise.race([
            providersEntered,
            attempt!.then(result => {
              throw new Error("Compare finished before both held provider branches entered: " + JSON.stringify(result))
            })
          ])
        } else {
          expect(await attempt).toEqual({ status: "submitted" })
        }
      })
      expect(useStoreMessageOption.getState().historyId).toBe(historyId)
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(acceptedAddress)
      expect(markCompareHistoryCreated).toHaveBeenCalledExactlyOnceWith(historyId)
      expect((await savedMessages()).filter(row => row.role === "user").map(row => row.id))
        .toEqual([sharedUser.id])
      expect(useStoreMessageOption.getState().messages.filter(row => !row.isBot).map(row => row.id))
        .toEqual([sharedUser.id])
      if (!rejectSecondBranch) {
        expect(useStoreMessageOption.getState().history.filter(row => row.role === "user"))
          .toEqual([
            expect.objectContaining({ content: question, messageType: "compare:user" })
          ])
      }
      expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
      const wires = vi.mocked(tldwChat.streamMessage).mock.calls
      expect(wires.map(([, options]) => options?.model).sort()).toEqual(models)
      for (const [messages, options] of wires) {
        expect(messages.filter(row => row.role === "user")).toEqual([
          expect.objectContaining({ content: question })
        ])
        expect(options?.requestScope).toEqual(originalScope)
        expect(JSON.stringify(messages)).toContain(scene.notes)
      }
      expect(dispatchStates).toEqual(models.map(() => ({
        address: acceptedAddress,
        appliedKey: boundTarget.key
      })))
      expect(applied).toHaveBeenCalledOnce()
      expect(useActorStore.getState().applied?.target.key).toBe(boundTarget.key)
      expect(useActorStore.getState().applied?.settings).toEqual(scene)
      expect((await getActorSceneForChat({ target: boundTarget })).originalRecord)
        .toEqual(originalScene.originalRecord)
      expect(await new Storage({ area: "local" }).get(sourceKey))
        .toEqual(originalScene.originalRecord)
      expect(createChat).not.toHaveBeenCalled()
      expect(admission).not.toHaveBeenCalled()
      expect(notification.error).not.toHaveBeenCalled()

      if (rejectSecondBranch) {
        // A different cluster in this same accepted chat must survive compensation.
        await saveMessage({
          id: "retained-user", history_id: historyId, name: "You", role: "user",
          content: "Retained other Compare question", images: [],
          message_type: "compare:user", clusterId: "retained-cluster", parent_message_id: null
        })
        await saveMessage({
          id: "retained-answer", history_id: historyId, name: "model-a", role: "assistant",
          content: "Retained other Compare answer", images: [],
          message_type: "compare:reply", clusterId: "retained-cluster", parent_message_id: "retained-user"
        })
        const persistedBranchWrites = vi.spyOn(storage.tables.messages, "add")
        await act(async () => {
          releaseProviderResults()
          expect(await attempt).toEqual({ status: "skipped", reason: "Request scope changed" })
        })
        expect(persistedBranchWrites.mock.calls.some(([row]) =>
          row.role === "assistant" && row.modelId === "model-a" && row.clusterId === sharedUser.clusterId
        )).toBe(true)
        expect(await storage.tables.chatHistories.get(historyId)).toMatchObject({
          id: historyId, server_scope_key: ACCOUNT
        })
        const remaining = await savedMessages()
        expect(remaining.map(row => row.id).sort()).toEqual(["retained-answer", "retained-user"])
        expect(remaining.every(row => row.history_id === historyId && row.clusterId === "retained-cluster")).toBe(true)
        expect(remaining.some(row => row.id === sharedUser.id || row.clusterId === sharedUser.clusterId)).toBe(false)
        const current = selection.getCurrent()
        expect(current.status).toBe("ready")
        expect(current.owner).toMatchObject({
          kind: "local", conversation_id: historyId, profile_id: acceptedProfileId
        })
        expect(current.view).toMatchObject({
          owner_key: acceptedOwnerKey, conversation_id: historyId
        })
        expect(current.capture?.status).toBe("captured")
        if (current.capture?.status !== "captured")
          throw new Error("Expected the actual H1 recapture after Compare compensation")
        expect(current.capture.rows.map(row => row.id)).not.toContain(sharedUser.id)
        expect(current.capture.selected_content.map(row => row.message)).not.toContain(question)
        const core = useStoreMessageOption.getState()
        expect(core.historyId).toBe(historyId)
        expect(core.serverChatId).toBeNull()
        expect(core.chatDraftAddress).toBe(acceptedAddress)
        expect(core.messages.map(row => row.id)).not.toContain(sharedUser.id)
        expect(core.history.map(row => row.content)).not.toContain(question)
        expect(projectedRows.at(-1)?.messages.map(row => row.id)).not.toContain(sharedUser.id)
        expect(await new Storage({ area: "local" }).get(destinationKey!)).toEqual(originalScene.originalRecord)
        expect(await new Storage({ area: "local" }).get(sourceKey)).toEqual(originalScene.originalRecord)
        expect(useActorStore.getState().applied?.target.key).toBe(boundTarget.key)
        expect(useActorStore.getState().applied?.settings).toEqual(scene)
        expect(applied).toHaveBeenCalledOnce()
        expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
        expect(notification.error).not.toHaveBeenCalled()
      }
    } finally {
      releaseWrite()
      releaseProviderResults()
      if (attempt) await act(async () => { await attempt })
      originMounted = false
      mounted.unmount()
      useStoreMessageOption.setState(initialCore)
      useActorStore.getState().reset()
    }
  })
})

describe("normal Chat owned history creation and H1 admission", () => {
  it.each(["native", "local"] as const)(
    "carries the original saved fresh scene through its accepted %s first bind",
    async (kind) => {
      const source = useStoreMessageOption.getState()
      const target = captureActorSettingsTarget({
        address: source.chatDraftAddress,
        requestScope: snapshot().requestScope,
        accountIsCurrent: () => true,
        isCurrent: () =>
          useStoreMessageOption.getState().chatDraftAddress ===
          source.chatDraftAddress
      })!
      const scene = {
        ...createDefaultActorSettings(),
        isEnabled: true,
        notes: "Exact original first-send scene"
      }
      expect(
        await saveActorSettingsForChat({
          historyId: null,
          serverChatId: null,
          target,
          expectedRecord: null,
          settings: scene
        })
      ).toBe(true)
      const read = await getActorSceneForChat({ target })
      const mounted = renderHook(() => useHistorySelection())
      useActorStore.getState().reset()
      if (kind === "native") {
        vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
        vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
          async (_id, request) => emptyServerCapture(request)
        )
        vi.spyOn(tldwClient, "addChatMessage").mockImplementation(
          async (_id, body) =>
            ({
              id: body.id,
              ...(body.tldw_history_selection_v1
                ? {
                    tldw_history_admission_v1: {
                      version: 1,
                      owner_key: "server-owner",
                      conversation_id: newServerId,
                      input_message_id: body.id,
                      input_message_revision: "1",
                      selection_digest:
                        body.tldw_history_selection_v1.selection_digest,
                      messages: body.tldw_history_selection_v1.messages,
                      originating_selection_revision:
                        body.tldw_history_selection_v1.selection_revision
                    }
                  }
                : {})
            }) as never
        )
      }
      await act(async () => {
        expect(
          await send(mounted.result.current, {
            actorScene: read,
            setHistoryId: (id) =>
              useStoreMessageOption.getState().setHistoryId(id),
            ...(kind === "native"
              ? {
                  setServerChatId: (id) =>
                    useStoreMessageOption.getState().setServerChatId(id)
                }
              : {})
          })
        ).toEqual({ status: "submitted" })
      })
      const selected = mounted.result.current.getCurrent()
      const bound = captureActorSettingsTarget({
        address: useStoreMessageOption.getState().chatDraftAddress,
        owner: selected.owner,
        view: selected.view,
        requestScope: snapshot().requestScope,
        accountIsCurrent: () => true,
        isCurrent: () => true
      })!
      expect(
        (await getActorSceneForChat({ target: bound })).originalRecord
      ).toEqual(read.originalRecord)
      expect(
        await new Storage({ area: "local" }).get(
          `actorSettings:owned:${target.key}`
        )
      ).toEqual(read.originalRecord)
      expect(useActorStore.getState().applied?.target.key).toBe(bound.key)
      expect(useActorStore.getState().applied?.settings).toEqual(scene)
      expect(
        JSON.stringify(vi.mocked(tldwChat.streamMessage).mock.calls)
      ).toContain(scene.notes)
      mounted.unmount()
    }
  )
  it("sends the original genuine-local scene without requiring unrelated backend-default verification", async () => {
    const created = await saveHistory(
      "Local scene",
      false,
      "web-ui",
      undefined,
      undefined,
      snapshot().requestScope
    )
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => {
      expect(
        await mounted.result.current.loadConversation({ historyId: created.id })
      ).toBe(true)
    })
    const selected = mounted.result.current.getCurrent()
    if (!selected.owner || !selected.view) throw new Error("fixture_receipt")
    expect(
      captureChatDraftLoad("switch")({
        owner: selected.owner,
        view: selected.view
      })
    ).toBe(true)
    useStoreMessageOption.getState().setHistoryId(created.id)
    const target = captureCurrentActorSceneTarget(mounted.result.current)!
    expect(target.account).toBeUndefined()
    const scene = {
      ...createDefaultActorSettings(),
      isEnabled: true,
      notes: "Owned local scene remains usable"
    }
    expect(
      await saveActorSettingsForChat({
        historyId: created.id,
        serverChatId: null,
        target,
        expectedRecord: null,
        settings: scene
      })
    ).toBe(true)
    const read = await getActorSceneForChat({ target })
    await act(async () => {
      expect(
        await send(mounted.result.current, {
          actorScene: read,
          historyId: created.id
        })
      ).toEqual({ status: "submitted" })
    })
    expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
    mounted.unmount()
  })

  it.each(["same-ID reload", "destination conflict"] as const)(
    "retains the accepted identity bind and prevents scene dispatch after %s",
    async (boundary) => {
      const handle = creationSource()
      const state = useStoreMessageOption.getState()
      const target = captureActorSettingsTarget({
        address: state.chatDraftAddress,
        requestScope: snapshot().requestScope,
        accountIsCurrent: () => true,
        isCurrent: () =>
          useStoreMessageOption.getState().chatDraftAddress ===
          state.chatDraftAddress
      })!
      const scene = {
        ...createDefaultActorSettings(),
        notes: "Saved original scene before native bind"
      }
      expect(
        await saveActorSettingsForChat({
          historyId: null,
          serverChatId: null,
          target,
          expectedRecord: null,
          settings: scene
        })
      ).toBe(true)
      const read = await getActorSceneForChat({ target })
      const mounted = renderHook(() => useHistorySelection())
      useActorStore.getState().reset()
      vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
      vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
        async (_id, request) => emptyServerCapture(request)
      )
      vi.spyOn(tldwClient, "addChatMessage")
      const sourceKey = `actorSettings:owned:${target.key}`
      let release!: () => void
      let writeEntered!: () => void
      const entered = new Promise<void>((resolve) => {
        writeEntered = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      const originalSet = Storage.prototype.set
      const originalGet = Storage.prototype.get
      const competing = {
        value: { ...scene, notes: "Independent destination scene" },
        updatedAt: 999
      }
      let destinationKey: string | undefined
      if (boundary === "same-ID reload") {
        vi.spyOn(Storage.prototype, "set").mockImplementation(
          async function (key, value) {
            if (key.startsWith("actorSettings:owned:") && key !== sourceKey) {
              destinationKey = key
              writeEntered()
              await held
            }
            return originalSet.call(this, key, value)
          }
        )
      } else {
        vi.spyOn(Storage.prototype, "get").mockImplementation(
          async function (key) {
            if (
              key.startsWith("actorSettings:owned:") &&
              key !== sourceKey &&
              !destinationKey
            ) {
              destinationKey = key
              await originalSet.call(this, key, competing)
            }
            return originalGet.call(this, key)
          }
        )
      }
      let attempt: Promise<unknown> | undefined
      try {
        await act(async () => {
          attempt = send(mounted.result.current, {
            actorScene: read,
            setServerChatId: publishFixtureServerChatId
          }).catch((error) => error)
          if (boundary === "same-ID reload") {
            await Promise.race([
              entered,
              attempt.then((result) => {
                throw result instanceof Error
                  ? result
                  : new Error("fixture_scene_copy_not_entered")
              })
            ])
            mounted.result.current.beginLoad()
            release()
          }
          expect(await attempt).toBeInstanceOf(Error)
        })
        expect(useActorStore.getState().applied).toBeNull()
        expect(await readChatCreationRecovery(handle)).toMatchObject({
          state: "bound",
          chatId: newServerId,
          inputText: "First question"
        })
        expect(tldwClient.addChatMessage).not.toHaveBeenCalled()
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(await new Storage({ area: "local" }).get(sourceKey)).toEqual(
          read.originalRecord
        )
        if (boundary === "destination conflict")
          expect(await new Storage({ area: "local" }).get(destinationKey!)).toEqual(
            competing
          )
      } finally {
        release()
        if (attempt) await act(async () => { await attempt })
        mounted.unmount()
      }
    }
  )

  it.each(["owned", "foreign", "other conversation"])("checks a historyId-only native mirror against its captured owner (%s)", async kind => {
    const historyId = await linkServerChatMirror({ chatId: kind === "other conversation" ? "other" : "native",
      title: "Mirror", ownerKey: kind === "foreign" ? "other-account" : ACCOUNT })
    const owner: NativeHistoryOwnerV1 = { kind: "native", conversation_id: "native", request_scope: snapshot().requestScope, validate_lease: () => true }
    const check = assertHistoryConversationTarget(owner, { historyId })
    if (kind === "owned") await expect(check).resolves.toBeUndefined()
    else await expect(check).rejects.toThrow("owner_conversation_mismatch")
  })

  it("refuses a ready selection belonging to a different displayed conversation", async () => {
    const source = await saveHistory("Source", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const target = await saveHistory("Target", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: target.id }) })
    await act(async () => {
      await expect(send(mounted.result.current, { historyId: source.id })).rejects.toThrow("owner_conversation_mismatch")
    })
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it.each(["Stop", "navigation", "account"])("does not adopt a native load completing after %s", async boundary => {
    const mounted = renderHook(() => useHistorySelection())
    const scoped = snapshot()
    const stopped = new AbortController()
    const auth = new AbortController()
    Object.assign(scoped, { scopeSignal: stopped.signal, scopeInvalidatedSignal: auth.signal })
    vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    let finish!: () => void
    const waiting = new Promise<void>(resolve => { finish = resolve })
    const capture = vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      await waiting
      return emptyServerCapture(request)
    })
    const adopt = vi.fn(publishFixtureServerChatId)
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { snapshot: scoped, setServerChatId: adopt, originIsCurrent: mounted.result.current.fence() }) })
    await waitFor(() => expect(capture).toHaveBeenCalledOnce())
    await act(async () => {
      if (boundary === "Stop") stopped.abort()
      if (boundary === "account") auth.abort()
      if (boundary === "navigation") mounted.result.current.reset()
      finish()
      await expect(pending).rejects.toThrow("stale_selection")
    })
    expect(adopt).not.toHaveBeenCalled()
    expect(mounted.result.current.status).toBe(boundary === "navigation" ? "idle" : "stale_selection")
    const owner = mounted.result.current.getCurrent().owner
    if (owner?.kind === "native") expect(owner.validate_lease()).toBe(false)
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it("does not admit after a qualified native receipt is followed by an epoch-only load change", async () => {
    const mounted = renderHook(() => useHistorySelection())
    vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    const capture = vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
    const admission = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, value) => {
      const selection = value.tldw_history_selection_v1
      return { id: value.id, ...(selection ? { tldw_history_admission_v1: {
        version: 1, owner_key: "server-owner", conversation_id: newServerId, input_message_id: value.id,
        input_message_revision: "1", selection_digest: selection.selection_digest,
        messages: selection.messages, originating_selection_revision: selection.selection_revision
      } } : {}) } as never
    })
    const controller = mounted.result.current
    const load = controller.loadConversation
    const changed = { ...controller, loadConversation: async (...args: Parameters<typeof load>) => {
      const loaded = await load(...args)
      controller.beginLoad()
      return loaded
    } }
    const adopt = vi.fn(publishFixtureServerChatId)
    await act(async () => { await expect(send(changed, { setServerChatId: adopt, originIsCurrent: controller.fence() })).rejects.toThrow("stale_selection") })
    expect(adopt).not.toHaveBeenCalled()
    expect(await storage.tables.chatHistories.toArray()).toEqual([])
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    const owner = mounted.result.current.getCurrent().owner
    expect(owner?.kind === "native" && owner.validate_lease()).toBe(false)
    capture.mockClear()
    await act(async () => { await expect(send(controller)).rejects.toThrow("stale_selection") })
    expect(capture).not.toHaveBeenCalled()
    expect(admission).not.toHaveBeenCalled()
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it("keeps an enabled-tool first send local without creating an empty native chat", async () => {
    const mounted = renderHook(() => useHistorySelection())
    useStoreMessageOption.setState({ toolChoice: "auto" })
    useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{
      name: "lookup", canExecute: true, inputSchema: { type: "object", properties: {} }
    }] }).chatTools })
    const create = vi.spyOn(tldwClient, "createChat")
    await act(async () => { expect(await send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId) })).toEqual({ status: "submitted" }) })
    expect(create).not.toHaveBeenCalled()
    expect(mounted.result.current.owner?.kind).toBe("local")
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

  it.each(["navigation", "account", "Stop", "missing ACK", "creation failure"])(
    "never adopts or generates after native creation crosses %s",
    async boundary => {
      const mounted = renderHook(() => useHistorySelection())
      const scoped = snapshot()
      const cancelled = new AbortController()
      Object.assign(scoped, { scopeSignal: cancelled.signal, scopeInvalidatedSignal: cancelled.signal })
      let finish!: () => void
      const waiting = new Promise<void>(resolve => { finish = resolve })
      let current = true
      const create = vi.spyOn(tldwClient, "createChat").mockImplementation(async body => {
        await waiting
        if (boundary === "creation failure") throw new Error("create unavailable")
        return boundary === "missing ACK" ? { id: "" } as never : createdServerChat(body)
      })
      const adopt = vi.fn(publishFixtureServerChatId)
      let pending!: ReturnType<typeof send>
      act(() => { pending = send(mounted.result.current, { snapshot: scoped, setServerChatId: adopt, originIsCurrent: () => current }) })
      await waitFor(() => expect(create).toHaveBeenCalledOnce())
      await act(async () => {
        if (boundary === "navigation") { current = false; mounted.result.current.reset() }
        if (boundary === "account" || boundary === "Stop") cancelled.abort()
        finish()
        await expect(pending).rejects.toThrow()
      })
      expect(adopt).not.toHaveBeenCalled()
      expect(mounted.result.current.getCurrent().owner).toBeNull()
      expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
      expect(await savedMessages()).toEqual([])
      expect(await storage.tables.chatHistories.toArray()).toEqual([])
      mounted.unmount()
    }
  )

  it.each(["playground", "sidepanel"].flatMap(surface =>
    ["First answer", "<think>Reasoning text</think>First answer"].map(answer => ({ surface, answer }))
  ))("$surface creates a native owner and reopens its server pair, preserving $answer", async ({ surface, answer }) => {
    const source = creationSource()
    const sourceAddress = useStoreMessageOption.getState().chatDraftAddress
    const stopped = new AbortController()
    const scoped = { ...snapshot(), scopeSignal: stopped.signal }
    const rows: Array<{ id: string; role: string; content: string; parent_message_id: string | null }> = []
    const create = vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      const view = { ...request.view, owner_key: "server-owner" }
      const nodes = rows.map(row => ({ id: row.id, role: row.role, parent_id: row.parent_message_id, revision: "1", settled: true }))
      return {
        status: "captured", purpose: request.purpose, view,
        snapshot: { version: 1, owner_key: view.owner_key, conversation_id: newServerId, nodes,
          fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source",
          storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } },
        rows: view.cursor.kind === "empty" ? [] : nodes,
        selected_content: view.cursor.kind === "empty" ? [] : rows.map(row => ({ id: row.id, revision: "1", message: row.content, images: [] })),
        storage_context_digest: "storage"
      } as never
    })
    const append = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, body) => {
      rows.push({ id: body.id!, role: body.role, content: body.content!, parent_message_id: body.parent_message_id ?? null })
      const selection = body.tldw_history_selection_v1
      return { id: body.id, ...(selection ? { tldw_history_admission_v1: {
        version: 1, owner_key: "server-owner", conversation_id: newServerId, input_message_id: body.id,
        input_message_revision: "1", selection_digest: selection.selection_digest,
        messages: selection.messages, originating_selection_revision: selection.selection_revision
      } } : {}) } as never
    })
    if (answer.startsWith("<think>")) {
      vi.mocked(tldwChat.streamMessage).mockImplementation(async function* (_messages, options) {
        options.onDispatch?.()
        yield "<think>Reasoning text"
        await new Promise(resolve => setTimeout(resolve, 10))
        yield "</think>First answer"
      })
    }
    const mounted = renderHook(() => useHistorySelection())
    const adopt = vi.fn(publishFixtureServerChatId)
    await act(async () => {
      expect(await send(mounted.result.current, { setServerChatId: adopt, snapshot: scoped, surface: surface as "playground" | "sidepanel" })).toEqual({ status: "submitted" })
    })
    expect(create).toHaveBeenCalledOnce()
    expect(create.mock.calls[0][1]).toMatchObject({ requestScope: prompt.scope, signal: expect.any(AbortSignal) })
    expect(adopt).toHaveBeenCalledWith(newServerId)
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "native", id: newServerId, ownerKey: "server-owner" })
    expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({ kind: "bind", sourceAddress })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "settled", chatId: newServerId, inputText: "" })
    expect(mounted.result.current.owner).toMatchObject({ kind: "native", conversation_id: newServerId })
    stopped.abort()
    const owner = mounted.result.current.getCurrent().owner
    expect(owner?.kind === "native" && owner.validate_lease()).toBe(true)
    expect(append).toHaveBeenCalledTimes(2)
    expect(append.mock.calls[0][1]).toMatchObject({ tldw_history_branch: false })
    expect(append.mock.calls[1][1]).not.toHaveProperty("tldw_history_branch")
    expect(rows.map(row => [row.role, row.content, row.parent_message_id])).toEqual([
      ["user", "First question", null], ["assistant", answer, "first-user"]
    ])
    expect(await savedMessages()).toEqual([])
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).toMatchObject({ save_to_db: false })
    const reference = mounted.result.current.getReference()!
    mounted.unmount()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => { await reopened.result.current.loadConversation({ serverChatId: newServerId }, reference) })
    expect(reopened.result.current.status).toBe("ready")
    expect(reopened.result.current.capture?.status === "captured" && reopened.result.current.capture.selected_content.map(row => row.message)).toEqual(["First question", answer])
    reopened.unmount()
  })

  it("dispatches explicit model and tool overrides despite unrelated global selector changes", async () => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    preparation.beforeModel.mockImplementation(() => {
      useStoreMessageOption.setState({ selectedModel: "unrelated-model", toolChoice: "auto" })
    })
    await act(async () => {
      expect(await send(mounted.result.current, { historyId: created.id, toolChoice: "none", fixedOverrides: true })).toEqual({ status: "submitted" })
    })
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).toMatchObject({ model: "test" })
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).not.toHaveProperty("tools")
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

  it.each(["loading", "inactive settings", "MCP health"])("allows a saved send after unrelated %s publication", async publication => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    preparation.beforeModel.mockImplementation(() => {
      if (publication === "loading") useMcpToolsStore.getState().setToolsLoading(true)
      else if (publication === "inactive settings") useStoreChatModelSettings.getState().updateScopedSetting("unused:other", "temperature", 0.8)
      else useMcpToolsStore.setState({ healthState: "healthy", toolCatalog: "background refresh", toolModules: ["module-updated"] })
    })
    await act(async () => {
      expect(await send(mounted.result.current, { historyId: created.id })).toEqual({ status: "submitted" })
    })
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

  it.each(["settings", "model", "tools", "enabled tools", "account"])("rejects a saved send when its %s changes during preparation", async changed => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    const scoped = snapshot()
    if (changed === "enabled tools") {
      useStoreMessageOption.setState({ toolChoice: "auto" })
      useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{
        name: "lookup", canExecute: true, inputSchema: { type: "object", properties: {} }
      }] }).chatTools })
    }
    preparation.beforeModel.mockImplementation(() => {
      if (changed === "settings") useStoreChatModelSettings.getState().setTemperature(0.8)
      if (changed === "model") useStoreMessageOption.getState().setSelectedModel("changed-model")
      if (changed === "tools") useStoreMessageOption.getState().setToolChoice("auto")
      if (changed === "enabled tools") useMcpToolsStore.setState({ chatTools: [] })
      if (changed === "account") {
        const controller = new AbortController()
        controller.abort()
        Object.assign(scoped, { scopeInvalidatedSignal: controller.signal })
      }
    })
    await act(async () => {
      expect((await send(mounted.result.current, { historyId: created.id, snapshot: scoped })).status).not.toBe("submitted")
    })
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it("persists the captured account before first local admission and reopens the exact selected turn", async () => {
    useStoreMessageOption.getState().startNewChatDraft();
    const sourceDraft = useStoreMessageOption.getState().chatDraftAddress;
    const mounted = renderHook(() => useHistorySelection())
    let historyId = ""
    await act(async () => {
      expect(await send(mounted.result.current, { originIsCurrent: mounted.result.current.fence(), setHistoryId: id => { publishFixtureHistoryId(id); historyId = id } })).toEqual({ status: "submitted" })
    })
    const persisted = await new PageAssistDatabase().getHistoryInfo(historyId)
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: historyId });
    expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({ kind: "bind", sourceAddress: sourceDraft });
    expect(persisted).toMatchObject({ server_scope_key: ACCOUNT, local_owner_key: expect.stringMatching(/^local-history-v1:/) })
    expect(persisted.server_chat_id).toBeUndefined()
    expect((await savedMessages()).map(({ id, history_id, role, content, parent_message_id }) => ({ id, history_id, role, content, parent_message_id }))).toEqual([
      { id: "first-user", history_id: historyId, role: "user", content: "First question", parent_message_id: null },
      { id: "first-answer", history_id: historyId, role: "assistant", content: "First answer", parent_message_id: "first-user" }
    ])
    expect(mounted.result.current.capture?.status).toBe("captured")
    const reference = mounted.result.current.getReference()!
    expect(mounted.result.current.view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
    mounted.unmount()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => { await reopened.result.current.loadConversation({ historyId }, reference) })
    expect(reopened.result.current.status).toBe("ready")
    expect(reopened.result.current.capture?.status === "captured" && reopened.result.current.capture.rows.map(row => row.id)).toEqual(["first-user", "first-answer"])
    expect(await savedMessages()).toHaveLength(2)
    const dispatched = vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest
    expect(dispatched).toMatchObject({ model: "test", save_to_db: false, messages: [{ role: "user", content: "First question" }] })
    expect(dispatched).not.toHaveProperty("conversation_id")
  })

  it("rereads separately supplied ownership while public and import writes discard supplied authority", async () => {
    const created = await saveHistory("Scoped", false, "web-ui", undefined, undefined, snapshot().requestScope)
    expect((await new PageAssistDatabase().getHistoryInfo(created.id)).server_scope_key).toBe(ACCOUNT)
    const db = new PageAssistDatabase()
    await db.addChatHistory({ ...publicHistory("public"), server_scope_key: ACCOUNT, local_owner_key: "forged" })
    expect(await db.getHistoryInfo("public")).toEqual(publicHistory("public"))
    await db.importChatHistoryV2([{ history: { ...publicHistory("import"), server_scope_key: ACCOUNT, local_owner_key: "forged" }, messages: [] }])
    expect(await db.getHistoryInfo("import")).toMatchObject(publicHistory("import"))
    expect((await db.getHistoryInfo("import")).server_scope_key).toBeUndefined()
    expect((await db.getHistoryInfo("import")).local_owner_key).toBeUndefined()
  })

  it.each(["missing", "foreign"])("does not admit a %s account history", async ownership => {
    const row = ownership === "missing"
      ? await saveHistory("Unowned", false, "web-ui")
      : await saveHistory("Foreign", false, "web-ui", undefined, undefined, { ...snapshot().requestScope, userId: "bob" })
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: row.id }) })
    expect(mounted.result.current.error).toBe("owner_conversation_mismatch")
    await act(async () => { await expect(send(mounted.result.current, { historyId: row.id })).rejects.toThrow("owner_conversation_mismatch") })
    expect(await savedMessages()).toEqual([])
  })

  it("rejects an expired captured account before admitting or generating a user row", async () => {
    const invalidated = new AbortController()
    invalidated.abort()
    const expired = { ...snapshot(), scopeInvalidatedSignal: invalidated.signal }
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { expect(await send(mounted.result.current, { snapshot: expired })).toEqual({ status: "failed", errorMessage: "Request cancelled" }) })
    expect(await savedMessages()).toEqual([])
  })

  it("does not adopt a creation that finishes after navigation", async () => {
    useStoreMessageOption.getState().startNewChatDraft();
    let finish!: () => void
    const gate = new Promise<void>(resolve => { finish = resolve })
    let entered = false
    storage.control.beforeHistoryAdd = async () => { entered = true; await gate }
    const mounted = renderHook(() => useHistorySelection())
    let adopted = ""
    let current = true
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setHistoryId: id => { publishFixtureHistoryId(id); adopted = id }, originIsCurrent: () => current }) })
    await waitFor(() => expect(entered).toBe(true))
    await act(async () => {
      current = false
      useStoreMessageOption.getState().startNewChatDraft();
      mounted.result.current.reset()
      finish()
      await expect(pending).rejects.toThrow("stale_selection")
    })
    expect(adopted).toBe("")
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    expect(mounted.result.current.getCurrent().owner).toBeNull()
    expect(await savedMessages()).toEqual([])
  })

  it("cannot reuse local account authority after an A to B to A boundary", async () => {
    const row = await saveHistory("Owned", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: row.id }) })
    expect(mounted.result.current.status).toBe("ready")
    const owner = mounted.result.current.owner!
    const view = mounted.result.current.view!
    act(() => {
      prompt.scope.userId = "bob"
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      prompt.scope.userId = "alice"
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    expect(mounted.result.current.error).toBe("request_config_scope_changed")
    await expect(captureHistorySnapshot(owner, view, "send")).rejects.toMatchObject({ code: "request_config_scope_changed" })
    await act(async () => { await expect(send(mounted.result.current, { historyId: row.id })).rejects.toThrow("request_config_scope_changed") })
    expect(await savedMessages()).toEqual([])
  })
})


const creationSource = (): ChatDraftHandle => ({
  address: structuredClone(useStoreMessageOption.getState().chatDraftAddress!),
  ownerHash: chatDraftOwnerKey(snapshot().requestScope)
})

describe("ordinary native creation recovery and preparation", () => {
  it.each(["playground", "sidepanel"] as const)("%s keeps unknown create input and blocks repeated and cold dispatch", async surface => {
    const source = creationSource()
    const mounted = renderHook(() => useHistorySelection())
    const create = vi.spyOn(tldwClient, "createChat").mockRejectedValue(new Error("Response lost"))
    const adopt = vi.fn(publishFixtureServerChatId)
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: adopt, surface })).rejects.toThrow() })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "unconfirmed", inputText: "First question", surface })
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: adopt, surface })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" }) })
    mounted.unmount()
    const cold = renderHook(() => useHistorySelection())
    await act(async () => { await expect(send(cold.result.current, { setServerChatId: adopt, surface })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" }) })
    expect(create).toHaveBeenCalledOnce()
    expect(adopt).not.toHaveBeenCalled()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    cold.unmount()
  })

  it.each(["new", "ABA", "account", "Stop"])("keeps a late ordinary create ACK private after %s", async boundary => {
    const source = creationSource()
    const originalAddress = useStoreMessageOption.getState().chatDraftAddress
    const mounted = renderHook(() => useHistorySelection())
    const scoped = snapshot()
    const abort = new AbortController()
    Object.assign(scoped, boundary === "account" ? { scopeInvalidatedSignal: abort.signal } : { scopeSignal: abort.signal })
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const create = vi.spyOn(tldwClient, "createChat").mockImplementation(async body => { await held; return createdServerChat(body) })
    const adopt = vi.fn(publishFixtureServerChatId)
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setServerChatId: adopt, snapshot: scoped, originIsCurrent: mounted.result.current.fence() }) })
    await waitFor(() => expect(create).toHaveBeenCalledOnce())
    act(() => {
      if (boundary === "new" || boundary === "ABA") {
        useStoreMessageOption.getState().startNewChatDraft()
        if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(originalAddress)
      } else abort.abort()
    })
    await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "acknowledged", chatId: newServerId, inputText: "First question" })
    expect(adopt).not.toHaveBeenCalled()
    expect(mounted.result.current.getCurrent().owner).toBeNull()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it.each(["create", "capture", "bookmark"].flatMap(stage => ["model", "settings", "tools"].map(boundary => ({ stage, boundary }))))("rejects $boundary drift during held first native $stage", async ({ stage, boundary }) => {
    const source = useStoreMessageOption.getState().chatDraftAddress
    const mounted = renderHook(() => useHistorySelection())
    let enter!: () => void, release!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(tldwClient, "createChat").mockImplementation(async body => {
      if (stage === "create") { enter(); await held }
      return createdServerChat(body)
    })
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      if (stage === "capture") { enter(); await held }
      return emptyServerCapture(request)
    })
    if (stage === "bookmark") {
      const put = storage.tables.historySelections.put
      vi.spyOn(storage.tables.historySelections, "put").mockImplementation(async value => { enter(); await held; return put(value) })
    }
    const adopt = vi.fn(publishFixtureServerChatId)
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setServerChatId: adopt }) })
    await entered
    act(() => {
      if (boundary === "model") useStoreMessageOption.getState().setSelectedModel("new-model")
      else if (boundary === "settings") useStoreChatModelSettings.getState().setTemperature(0.73)
      else useStoreMessageOption.getState().setToolChoice("auto")
    })
    await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
    expect(adopt).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it.each(["storage", "locks"])("refuses ordinary %s unavailability before creating", async boundary => {
    const mounted = renderHook(() => useHistorySelection())
    const create = vi.spyOn(tldwClient, "createChat")
    if (boundary === "storage") vi.spyOn(Storage.prototype, "get").mockRejectedValue(new Error("Unavailable"))
    else Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId) })).rejects.toMatchObject({ code: "chat_creation_recovery_unavailable" }) })
    expect(create).not.toHaveBeenCalled()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })
})


it.each(["new", "ABA"])("captures the source before awaited service setup crosses %s", async boundary => {
  const source = useStoreMessageOption.getState().chatDraftAddress
  const mounted = renderHook(() => useHistorySelection())
  let enter!: () => void, release!: () => void
  const entered = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockImplementation(async () => { enter(); await held; return snapshot() })
  const create = vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
  let pending!: Promise<unknown>
  act(() => { pending = send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId), snapshot: null }).catch(error => error) })
  await entered
  act(() => {
    useStoreMessageOption.getState().startNewChatDraft()
    if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(source)
  })
  let outcome: unknown
  await act(async () => { release(); outcome = await pending })
  expect(create).not.toHaveBeenCalled()
  expect(outcome).toMatchObject({ message: "stale_selection" })
  expect(tldwChat.streamMessage).not.toHaveBeenCalled()
  mounted.unmount()
})


it.each([false, true])("keeps original ordinary input bound when the assistant save acknowledgement is lost (persisted=%s)", async persisted => {
  const serverReplies: unknown[] = []
  const source = creationSource()
  const mounted = renderHook(() => useHistorySelection())
  vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
  const append = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, body) => {
    if (body.role === "assistant") {
      if (persisted) serverReplies.push(structuredClone(body))
      throw new Error("Assistant save response lost")
    }
    const selection = body.tldw_history_selection_v1!
    return { id: body.id, tldw_history_admission_v1: {
      version: 1, owner_key: "server-owner", conversation_id: newServerId, input_message_id: body.id,
      input_message_revision: "1", selection_digest: selection.selection_digest,
      messages: selection.messages, originating_selection_revision: selection.selection_revision
    } } as never
  })
  await act(async () => { expect((await send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId) })).status).toBe("failed") })
  expect(append).toHaveBeenCalledTimes(2)
  expect(serverReplies).toHaveLength(persisted ? 1 : 0)
  const selected = mounted.result.current.getCurrent()
  const retained = await loadHistoryTurnRecoveries(selected.bookmarkScope!, selected.view!)
  expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({
    state: "generated_unsaved", input_text: "First question", result_text: "First answer"
  }) })])
  expect(retained[0].turn.outcome).toBeUndefined()
  expect(await readChatCreationRecovery(source)).toMatchObject({ state: "bound", chatId: newServerId, inputText: "First question" })
  mounted.unmount()
})

it.each(["same-chat", "newer-server"])("a held ordinary bound-record write cannot dispatch after a newer H1 load of %s", async nextChoice => {
  const source = creationSource()
  const mounted = renderHook(() => useHistorySelection())
  vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (id, request) => {
    return emptyServerCapture(request, id)
  })
  const original = Storage.prototype.set
  let enter!: () => void, release!: () => void
  const entered = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
    const result = await original.call(this, key, value)
    if (key.startsWith("registry:chat-creation:") && value?.value?.state === "bound") { enter(); await held }
    return result
  })
  let pending!: ReturnType<typeof send>
  act(() => { pending = send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId) }) })
  await entered
  await act(async () => { await mounted.result.current.loadConversation({ serverChatId: nextChoice === "same-chat" ? newServerId : nextChoice }) })
  const newer = mounted.result.current.getCurrent()
  await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
  expect(mounted.result.current.getCurrent().owner).toBe(newer.owner)
  expect(mounted.result.current.getCurrent().view).toBe(newer.view)
  expect(newer.owner?.kind === "native" && newer.owner.validate_lease()).toBe(true)
  expect(tldwChat.streamMessage).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(source)).toMatchObject({ state: "bound", inputText: "First question" })
  mounted.unmount()
})


it.each(["cleanup", "navigation", "account"])("retains an acknowledged local assistant through %s after saving", async boundary => {
  const source = await saveHistory("Source", false, "web-ui", undefined, undefined, snapshot().requestScope)
  const other = await saveHistory("Other", false, "web-ui", undefined, undefined, snapshot().requestScope)
  const mounted = renderHook(() => useHistorySelection())
  await act(async () => { await mounted.result.current.loadConversation({ historyId: source.id }) })
  const origin = mounted.result.current.getCurrent()
  const scoped = snapshot()
  const invalidation = new AbortController()
  // Explicit account-signal I/O double; real event/controller retirement is covered by root ownership checks.
  scoped.scopeInvalidatedSignal = invalidation.signal
  let acknowledgedId: string | undefined
  const save = vi.fn(async (data: Parameters<typeof saveMessageOnSuccess>[0]) => {
    const turn = data.historyTurn!
    const complete = turn.complete!
    turn.complete = async () => {
      await complete()
      acknowledgedId = turn.resultId
      if (boundary !== "cleanup") {
        if (boundary === "account") invalidation.abort()
        await mounted.result.current.loadConversation({ historyId: other.id })
        useStoreMessageOption.getState().startNewChatDraft()
      }
      throw new Error("Ancillary cleanup failed")
    }
    return saveMessageOnSuccess(data)
  })
  await act(async () => {
    expect((await send(mounted.result.current, { historyId: source.id, saveOnSuccess: save, snapshot: scoped })).status).toBe(boundary === "account" ? "failed" : "submitted")
  })
  expect(acknowledgedId).toBe("first-answer")
  expect(save).toHaveBeenCalledOnce()
  expect((await savedMessages()).filter(row => row.role === "assistant")).toHaveLength(1)
  if (boundary === "cleanup") {
    expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
  } else {
    expect(mounted.result.current.getCurrent().owner?.conversation_id).toBe(other.id)
    expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "empty" })
    const retained = await loadHistoryTurnRecoveries(origin.bookmarkScope!, origin.view!)
    if (boundary === "navigation") {
      expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({ settled_message_id: "first-answer", outcome: "complete" }) })])
      await act(async () => {
        await mounted.result.current.loadConversation({ historyId: source.id })
        expect((await keepHistoryTurnRecovery(mounted.result.current, retained[0])).status).toBe("kept")
      })
      expect((await savedMessages()).filter(row => row.role === "assistant")).toHaveLength(1)
      expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
    } else expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({ settled_message_id: "first-answer", outcome: "complete" }) })])
  }
  mounted.unmount()
})


it.each(["playground", "sidepanel"] as const)(
  "client UUID ordinary %s create sends its durable journal ID even if the response is lost",
  async (surface) => {
    const source = creationSource()
    const mounted = renderHook(() => useHistorySelection())
    const create = vi
      .spyOn(tldwClient, "createChat")
      .mockRejectedValue(new Error("Response lost"))
    await act(async () => {
      await expect(
        send(mounted.result.current, { setServerChatId: vi.fn(publishFixtureServerChatId), surface })
      ).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    })
    const attempt = await readChatCreationRecovery(source)
    expect(create.mock.calls[0][0]).toMatchObject({
      id: attempt?.operationId,
      title: "First question",
      source: "webui-chat"
    })
    expect(attempt).toMatchObject({
      clientChatId: attempt?.operationId,
      state: "unconfirmed",
      inputText: "First question"
    })
    expect(create.mock.calls[0][0].id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    )
    expect(create).toHaveBeenCalledOnce()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  }
)


it.each(
  ["capture", "dispatch journal", "model"].flatMap((stage) =>
    ["edit", "ordinary", "branch"].map((action) => ({ stage, action }))
  )
)(
  "native $action refuses an actual same-ID H1 refresh during held $stage before admitting input",
  async ({ stage, action }) => {
    const graph = [
      {
        id: "original-user",
        revision: "u1",
        parent_id: null,
        role: "user",
        settled: true,
        preview: "Original"
      },
      {
        id: "original-answer",
        revision: "a1",
        parent_id: "original-user",
        role: "assistant",
        settled: true,
        preview: "Original answer"
      }
    ]
    const serverCapture = async (_id: unknown, request: any) => {
      const view = { ...request.view, owner_key: "server-owner" }
      const snapshot = {
        version: 1 as const,
        owner_key: "server-owner",
        conversation_id: newServerId,
        nodes: graph,
        fences: { conversation: "1", history: "1", settings: "1" },
        source_digest: "source",
        storage_context_digest: "storage",
        interpretation_status: { kind: "parent_graph_v1" as const }
      }
      const { resolveHistorySelection } = await import(
        "@/utils/history-selection"
      )
      const selected = resolveHistorySelection(
        snapshot,
        view,
        request.purpose,
        ""
      )
      if (selected.status !== "ready") throw new Error(selected.code)
      return {
        status: "captured",
        view,
        snapshot,
        rows: selected.rows,
        purpose: request.purpose,
        selected_content: selected.rows.map((row) => ({
          id: row.id,
          revision: row.revision,
          message: graph.find((item) => item.id === row.id)!.preview,
          images: []
        })),
        storage_context_digest: "storage"
      } as never
    }
    const capture = vi
      .spyOn(tldwClient, "captureHistorySelection")
      .mockImplementation(serverCapture)
    const append = vi.spyOn(tldwClient, "addChatMessage")
    const mounted = renderHook(() => useHistorySelection())
    const native: NativeHistoryOwnerV1 = {
      kind: "native",
      conversation_id: newServerId,
      request_scope: snapshot().requestScope,
      validate_lease: () => true
    }
    await act(async () => {
      expect(await mounted.result.current.open(native)).toBe(true)
    })
    expect(mounted.result.current.status).toBe("ready")
    const origin = mounted.result.current.view!
    if (action === "branch")
      act(() => {
        mounted.result.current.holdStaleSend(origin, () => true)
        mounted.result.current.branchHere(origin, () => true)
      })
    const source = mounted.result.current.fence()
    let entered!: () => void, release!: () => void
    const ready = new Promise<void>((done) => {
      entered = done
    })
    const held = new Promise<void>((done) => {
      release = done
    })
    if (stage === "capture")
      capture.mockImplementationOnce(async (...args) => {
        entered()
        await held
        return serverCapture(...args)
      })
    if (stage === "model")
      preparation.beforeModel.mockImplementationOnce(async () => {
        entered()
        await held
      })
    if (stage === "dispatch journal") {
      const put = storage.tables.historySelections.put
      vi.spyOn(storage.tables.historySelections, "put").mockImplementationOnce(
        async (row) => {
          entered()
          await held
          return put(row)
        }
      )
    }
    const params: Parameters<typeof normalChatMode>[6] = {
      selectedModel: "test",
      selectedSystemPrompt: "",
      useOCR: false,
      currentChatModelSettings: {},
      toolChoice: "none",
      servicePromptSnapshot: snapshot(),
      historyId: null,
      serverChatId: newServerId,
      historySelection: {
        controller: mounted.result.current,
        originIsCurrent: source,
        editResend:
          action === "edit"
            ? {
                owner_key: origin.owner_key,
                conversation_id: origin.conversation_id,
                message_id: "original-user",
                message_revision: "u1",
                isCurrent: () => true,
                isOriginCurrent: source
              }
            : undefined
      },
      setHistoryId: () => {},
      setMessages: vi.fn(),
      setHistory: vi.fn(),
      setIsProcessing: vi.fn(),
      setStreaming: vi.fn(),
      setAbortController: vi.fn(),
      saveMessageOnSuccess,
      saveMessageOnError
    }
    let attempt!: Promise<unknown>
    await act(async () => {
      attempt = normalChatMode(
        "Edited",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        params
      )
      void attempt.catch(() => {})
      await Promise.race([
        ready,
        attempt.then(() => {
          throw new Error("send finished before held stage")
        })
      ])
    })
    vi.mocked(params.setMessages).mockClear()
    vi.mocked(params.setHistory).mockClear()
    let refreshing!: Promise<boolean>
    act(() => {
      if (action === "ordinary") {
        mounted.result.current.beginLoad()
        refreshing = Promise.resolve(false)
      } else refreshing = mounted.result.current.refresh()
    })
    expect(source()).toBe(false)
    await act(async () => {
      release()
      await Promise.allSettled([attempt, refreshing])
    })
    expect(mounted.result.current.view).toMatchObject({
      conversation_id: origin.conversation_id,
      view_session_id: origin.view_session_id,
      selection_revision: origin.selection_revision
    })
    expect(append).not.toHaveBeenCalled()
    if (stage === "dispatch journal") {
      const original = mounted.result.current.getCurrent()
      const kept = await loadHistoryTurnRecoveries(
        original.bookmarkScope!,
        origin
      )
      expect(kept[0]?.turn).toMatchObject({
        state: "refused",
        input_text: "Edited",
        result_text: ""
      })
    }
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    expect(params.setMessages).not.toHaveBeenCalled()
    expect(params.setHistory).not.toHaveBeenCalled()
    expect(graph.map((row) => row.preview)).toEqual([
      "Original",
      "Original answer"
    ])
    mounted.unmount()
  }
)

describe("native latest intent and owned refusal", () => {
  it.each(
    ["playground", "sidepanel"].flatMap((surface) =>
      [
        "",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6VbMAAAAASUVORK5CYII="
      ].map((image) => ({
        surface: surface as "playground" | "sidepanel",
        image
      }))
    )
  )(
    "keeps rejected ordinary input and sends an explicit branch from the same empty view ($surface $image)",
    async ({ surface, image }) => {
      vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
      vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
        async (_id, request) => emptyServerCapture(request)
      )
      const append = vi
        .spyOn(tldwClient, "addChatMessage")
        .mockImplementation(async (_id, _body) => {
          throw Object.assign(new Error("HTTP 409"), {
            status: 409,
            details: {
              detail: {
                status: "stale_selection",
                code: "history_branch_changed",
                conversation_id: newServerId,
                parent_message_id: null,
                leaf_ids: ["elsewhere"],
                history_version: 2
              }
            }
          })
        })
      const hook = renderHook(() => useHistorySelection())
      preparation.multimodal = Boolean(image)
      const input = "  exact ordinary input\nspaces  "
      let outcome: unknown
      await act(async () => {
        outcome = await send(hook.result.current, {
          surface,
          image,
          message: input,
          setServerChatId: publishFixtureServerChatId
        })
      })
      expect(outcome).toMatchObject({
        status: "failed",
        errorMessage: "native_history_branch_changed"
      })
      const current = hook.result.current.getCurrent()
      const recoveries = await loadHistoryTurnRecoveries(
        current.bookmarkScope!,
        current.view!
      )
      expect(recoveries[0].turn).toMatchObject({
        state: "refused",
        input_text: input,
        persistence: "client",
        input_id: "first-user",
        assistant_id: "first-answer",
        result_text: ""
      })
      expect(recoveries[0].turn.input_images).toEqual(image ? [image] : [])
      expect(recoveries[0].turn.admission).toBeUndefined()
      expect(hook.result.current.sendConflict).toBe(true)
      expect(hook.result.current.status).toBe("stale_selection")
      expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
      expect(await savedMessages()).toEqual([])
      act(() => {
        expect(hook.result.current.branchHere(current.view!, () => true)).toBe(
          true
        )
      })
      append.mockRejectedValueOnce(new Error("Unknown transport"))
      await act(async () => {
        await send(hook.result.current, { surface, serverChatId: newServerId })
      })
      expect(append.mock.calls[1][1].tldw_history_branch).toBe(true)
      expect(append.mock.calls[1][1].parent_message_id ?? null).toBeNull()
      expect(hook.result.current.branchPending).toBe(false)
      hook.unmount()
    }
  )
})

it.each(["playground", "sidepanel"] as const)(
  "holds an unknown older-server continuation before admission in %s and keeps an explicit Branch choice",
  async (surface) => {
    vi.spyOn(tldwClient, "getOpenAPISpec").mockResolvedValue(null)
    vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    let captureReads = 0
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
      async (_id, request) => {
        const result = emptyServerCapture(
          request
        ) as unknown as import("@/types/history-selection").HistorySelectionCaptureV1
        if (++captureReads === 1) return result as never
        return {
          ...result,
          snapshot: {
            ...result.snapshot,
            nodes: [
              {
                id: "elsewhere",
                revision: "1",
                parent_id: null,
                role: "user",
                settled: true
              }
            ]
          }
        } as never
      }
    )
    const append = vi
      .spyOn(tldwClient, "addChatMessage")
      .mockRejectedValue(new Error("Unexpected admission"))
    const hook = renderHook(() => useHistorySelection())
    preparation.multimodal = true
    const image =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6VbMAAAAASUVORK5CYII="
    const message = "  Older-server input retained exactly.\nSecond line.  "
    let outcome: unknown
    await act(async () => {
      outcome = await send(hook.result.current, {
        surface,
        message,
        image,
        setServerChatId: publishFixtureServerChatId
      })
    })
    expect(outcome).toMatchObject({
      status: "failed",
      errorMessage: "native_history_branch_choice_required"
    })
    expect(append).not.toHaveBeenCalled()
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    const original = hook.result.current.getCurrent()
    const recoveries = await loadHistoryTurnRecoveries(
      original.bookmarkScope!,
      original.view!
    )
    expect(recoveries[0].turn).toMatchObject({
      state: "refused",
      input_text: message,
      input_images: [image],
      result_text: ""
    })
    expect(recoveries[0].turn.admission).toBeUndefined()
    expect(hook.result.current.sendConflict).toBe(true)
    act(() => {
      expect(hook.result.current.branchHere(original.view!, () => true)).toBe(
        true
      )
    })
    await act(async () => {
      await send(hook.result.current, {
        surface,
        message,
        image,
        serverChatId: newServerId
      })
    })
    expect(append).toHaveBeenCalledOnce()
    expect(append.mock.calls[0][1]).toMatchObject({
      content: message,
      tldw_history_branch: true
    })
    hook.unmount()
  }
)

it.each(["playground", "sidepanel"] as const)(
  "native first bind keeps its independent %s source lease during held contract discovery",
  async (surface) => {
    vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
      async (_id, request) => emptyServerCapture(request)
    )
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    vi.mocked(tldwClient.getOpenAPISpec).mockImplementation(async () => {
      await held
      return null
    })
    const append = vi
      .spyOn(tldwClient, "addChatMessage")
      .mockRejectedValue(new Error("Unexpected late admission"))
    const hook = renderHook(() => useHistorySelection())
    let sourceCurrent = true
    let pending!: Promise<unknown>
    act(() => {
      pending = send(hook.result.current, {
        surface,
        setServerChatId: publishFixtureServerChatId,
        sourceIsCurrent: () => sourceCurrent
      })
    })
    await waitFor(() =>
      expect(tldwClient.getOpenAPISpec).toHaveBeenCalledOnce()
    )
    expect(hook.result.current.getCurrent().owner).toMatchObject({
      kind: "native",
      conversation_id: newServerId
    })
    sourceCurrent = false
    await act(async () => {
      release()
      await pending
    })
    expect(append).not.toHaveBeenCalled()
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    hook.unmount()
  }
)

// These consumers share the retaining table engine above. Only the external
// prompt/catalog/provider/tab APIs are doubles; H1, Scene storage, shared save,
// projection, and the compact controller provider remain production code.
describe("real full and compact Scene consumers across persistence and steering", () => {
  type Surface = "full" | "compact"
  type AlternateMode = "document" | "tab" | "local RAG"
  type FullActions = ReturnType<typeof useChatActions>
  type Consumer = {
    onSubmit: (options: {
      message: string
      image: string
      docs?: ChatDocuments
      uploadedFiles?: UploadedFile[]
    }) => Promise<ChatSubmitResult | void>
    sendPerModelReply?: FullActions["sendPerModelReply"]
    regenerateLastMessage?: FullActions["regenerateLastMessage"]
    stopStreamingRequest: ReturnType<typeof useMessage>["stopStreamingRequest"]
    clearChat?: ReturnType<typeof useMessage>["clearChat"]
    composer?: ReturnType<typeof useComposerText>
  }
  let initialCore: ReturnType<typeof useStoreMessageOption.getState>

  // ChatTldw passes an empty positional array after H1 prepares the actual body.
  const wireMessages = (call: Parameters<typeof tldwChat.streamMessage>) =>
    call[1]?.preparedRequest?.messages ?? call[0]

  const files: UploadedFile[] = [
    {
      id: "original-upload",
      filename: "original.txt",
      content: "Original document context",
      type: "text/plain",
      size: 25,
      uploadedAt: 1,
      processed: true
    }
  ]
  const tabs: ChatDocuments = [
    {
      type: "tab",
      tabId: 23,
      title: "Original tab",
      url: "https://source.test/original"
    }
  ]
  const answerSnapshot = (): ServicePromptSnapshot => ({
    ...snapshot(),
    definitions: {
      "chat.rag.answer": {
        definition: {
          id: "chat.rag.answer",
          parts: [
            {
              key: "template",
              mode: "template",
              required_variables: ["context", "question"]
            }
          ]
        },
        parts: {
          template: "Original context:\n{context}\nQuestion: {question}"
        },
        source: "packaged",
        revision: null
      },
      "chat.rag.question_rewrite": {
        definition: {
          id: "chat.rag.question_rewrite",
          parts: [
            {
              key: "template",
              mode: "template",
              required_variables: ["chat_history", "question"]
            }
          ]
        },
        parts: { template: "History: {chat_history}\nQuestion: {question}" },
        source: "packaged",
        revision: null
      }
    }
  })

  beforeEach(async () => {
    preparation.realFactory = true
    initialCore = useStoreMessageOption.getState()
    initialCore.setServerChatId(null)
    useStoreMessageOption.setState({
      historyId: null,
      messages: [],
      history: [],
      selectedModel: "test",
      selectedModelStorageHandled: true,
      chatMode: "normal",
      toolChoice: "none",
      webSearch: false,
      useOCR: false,
      temporaryChat: false,
      streaming: false,
      isProcessing: false,
      isLoading: false,
      selectedSystemPrompt: null,
      selectedQuickPrompt: null,
      selectedKnowledge: null,
      fileRetrievalEnabled: false,
      ragMediaIds: null,
      contextFiles: [],
      uploadedFiles: [],
      documentContext: null,
      replyTarget: null,
      compareMode: false,
      compareSelectedModels: [],
      messageSteeringMode: "none",
      messageSteeringForceNarrate: false,
      serverChatCharacterId: null,
      serverChatAssistantId: null,
      serverChatAssistantKind: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      serverChatState: null,
      serverChatVersion: null
    })
    useStoreMessageOption.getState().startNewChatDraft(false)
    useStoreChatModelSettings.getState().setApiProvider("openai")
    useActorStore.getState().reset()
    await setTitleGenEnabled(false)
    const catalog = ["test", "second-model"].map((id) => ({
      id,
      name: id,
      provider: "openai",
      type: "chat" as const,
      capabilities: []
    }))
    vi.spyOn(tldwModels, "getModel").mockImplementation(
      async (id) => catalog.find((model) => model.id === id) ?? null
    )
    vi.spyOn(tldwModels, "getModels").mockResolvedValue(catalog)
    vi.spyOn(tldwServer, "fetchChatModels").mockResolvedValue([
      { id: "test", model: "test", name: "test", provider: "openai" }
    ] as never)
    vi.spyOn(tldwClient, "ragSearch").mockResolvedValue({
      results: [
        {
          content: "Original document context",
          metadata: { filename: "original.txt" }
        }
      ]
    } as never)
    // Execute the production tab extractor against a declared browser response.
    vi.stubGlobal("browser", {
      ...(typeof browser === "undefined" ? {} : browser),
      scripting: {
        executeScript: vi.fn(async () => [
          {
            result: {
              html: "<html><body><article><h1>Original tab</h1><p>Original tab context.</p></article></body></html>",
              title: "Original tab",
              url: tabs[0].url,
              isPDF: false
            }
          }
        ])
      }
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    useStoreMessageOption.setState(initialCore)
    useActorStore.getState().reset()
  })

  const clearAlternateContext = () => {
    const core = useStoreMessageOption.getState()
    core.setContextFiles([])
    core.setUploadedFiles([])
    core.setDocumentContext(null)
    core.setFileRetrievalEnabled(false)
    core.setRagMediaIds(null)
    core.setSelectedKnowledge(null)
    core.setChatMode("normal")
  }

  const mountConsumer = (
    surface: Surface,
    scoped: ServicePromptSnapshot,
    options: {
      keepInterruptedTurns?: boolean
      actualOption?: boolean
      withComposer?: boolean
    } = {}
  ) => {
    let mountedOrigin = true
    const originalScope = structuredClone(scoped.requestScope)
    const draftAccount = Object.freeze({
      requestScope: originalScope,
      isCurrent: () =>
        mountedOrigin &&
        !scoped.scopeInvalidatedSignal.aborted &&
        chatDraftOwnerKey(prompt.scope) === chatDraftOwnerKey(originalScope),
      retryVerification: vi.fn()
    })
    let selection!: HistorySelectionController
    const receipts: HistoryLoadReceipt[] = []
    const acceptedLoads: {
      receipt: HistoryLoadReceipt
      address: ReturnType<
        typeof useStoreMessageOption.getState
      >["chatDraftAddress"]
      transition: ReturnType<
        typeof useStoreMessageOption.getState
      >["chatDraftTransition"]
    }[] = []
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const Wrapper = ({ children }: PropsWithChildren) => {
      useEffect(
        () => () => {
          mountedOrigin = false
        },
        []
      )
      const controller = useHistorySelection({
        onCapture: (capture) => {
          const display = formatSelectedHistory(capture)
          useStoreMessageOption.setState({
            history: display.history,
            messages: display.messages
          })
        }
      })
      const value = useMemo<HistorySelectionController>(
        () => ({
          ...controller,
          draftAccount,
          loadConversation: (target, reference, onLoaded) =>
            controller.loadConversation(target, reference, (receipt) => {
              receipts.push(receipt)
              expect(controller.getCurrent().owner).toBe(receipt.owner)
              expect(controller.getCurrent().view).toBe(receipt.view)
              onLoaded?.(receipt)
              const core = useStoreMessageOption.getState()
              acceptedLoads.push({
                receipt,
                address: core.chatDraftAddress,
                transition: core.chatDraftTransition
              })
            })
        }),
        [controller]
      )
      useLayoutEffect(() => {
        selection = value
      }, [value])
      const wrapped = createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(
          App,
          null,
          createElement(
            PageAssistProvider,
            null,
            createElement(HistorySelectionContext.Provider, { value }, children)
          )
        )
      )
      return options.actualOption || options.withComposer
        ? createElement(MemoryRouter, null, wrapped)
        : wrapped
    }
    type ActionOptions = Parameters<typeof useChatActions>[0]
    const notification = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    } as unknown as ActionOptions["notification"]
    const useFullConsumer = (): Consumer => {
      const core = useStoreMessageOption()
      const settings = useStoreChatModelSettings()
      // Public Playground presentation is opted in only by the native Retry
      // controls below; recovery creation, keeping and projection remain real.
      useKeepInterruptedHistoryTurns({
        selection: options.keepInterruptedTurns ? selection : null,
        messages: core.messages,
        setMessages: core.setMessages
      })
      const [abortController, setAbortController] =
        useState<AbortController | null>(null)
      const t = ((_key: string, fallback?: unknown) =>
        typeof fallback === "string" ? fallback : _key) as ActionOptions["t"]
      const { ensureServerChatHistoryId } = useServerChatHistoryId({
        serverChatId: core.serverChatId,
        historyId: core.historyId,
        setHistoryId: core.setHistoryId,
        temporaryChat: core.temporaryChat,
        t
      })
      const actions = useChatActions({
        ...core,
        t,
        notification,
        abortController,
        setAbortController,
        currentChatModelSettings: settings,
        compareModeActive: core.compareMode,
        compareMaxModels: 3,
        compareFeatureEnabled: true,
        markCompareHistoryCreated: vi.fn(),
        ensureServerChatHistoryId,
        messageSteeringPrompts: null,
        invalidateServerChatHistory: vi.fn(),
        selectedCharacter: null,
        selectedAssistant: null
      })
      return {
        onSubmit: actions.onSubmit,
        sendPerModelReply: actions.sendPerModelReply,
        regenerateLastMessage: actions.regenerateLastMessage,
        stopStreamingRequest: actions.stopStreamingRequest
      }
    }
    const useCompactConsumer = (): Consumer => {
      const compact = useMessage()
      return {
        onSubmit: compact.onSubmit,
        stopStreamingRequest: compact.stopStreamingRequest,
        clearChat: compact.clearChat,
        sendPerModelReply: undefined as
          | FullActions["sendPerModelReply"]
          | undefined
      }
    }
    const useOptionConsumer = (): Consumer => {
      const option = useMessageOption()
      return {
        onSubmit: option.onSubmit,
        sendPerModelReply: option.sendPerModelReply,
        regenerateLastMessage: option.regenerateLastMessage,
        stopStreamingRequest: option.stopStreamingRequest,
        clearChat: option.clearChat
      }
    }
    const useSurfaceConsumer = options.actualOption
      ? useOptionConsumer
      : surface === "full"
        ? useFullConsumer
        : useCompactConsumer
    const useRecoveryConsumer = (): Consumer => {
      const consumer = useSurfaceConsumer()
      const core = useStoreMessageOption()
      const textareaRef = useMemo(
        () => ({ current: null as HTMLTextAreaElement | null }),
        []
      )
      const root =
        surface === "full"
          ? "tldw:playgroundChatDraft"
          : "tldw:sidepanelChatDraft"
      const address = core.chatDraftAddress
      const composer = useComposerText({
        draftKey: address ? conversationDraftKey(root, address) : root,
        draftAddress: address ?? undefined,
        draftEnabled: address !== null,
        bindFromDraftKey:
          core.chatDraftTransition?.kind === "bind" &&
          core.chatDraftTransition.sourceAddress
            ? conversationDraftKey(root, core.chatDraftTransition.sourceAddress)
            : undefined,
        recoverySurface: root,
        tabScopedDraft: true,
        textareaRef
      })
      return { ...consumer, composer }
    }
    const mounted = renderHook(
      options.withComposer ? useRecoveryConsumer : useSurfaceConsumer,
      { wrapper: Wrapper }
    )
    return {
      mounted,
      originalScope,
      draftAccount,
      receipts,
      acceptedLoads,
      get selection() {
        return selection
      },
      unmount: () => {
        mounted.unmount()
        queryClient.clear()
      }
    }
  }

  const declareNativeMirrorTransport = () => {
    const rows: Array<{
      id: string
      role: string
      content: string
      parent_message_id: string | null
    }> = []
    vi.spyOn(tldwClient, "createChat").mockImplementation(createdServerChat)
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
      async (_id, request) => {
        const view = { ...request.view, owner_key: "server-owner" }
        const nodes = rows.map((row) => ({
          id: row.id,
          role: row.role,
          parent_id: row.parent_message_id,
          revision: "1",
          settled: true
        }))
        return {
          status: "captured",
          purpose: request.purpose,
          view,
          snapshot: {
            version: 1,
            owner_key: view.owner_key,
            conversation_id: newServerId,
            nodes,
            fences: { conversation: "1", history: "1", settings: "1" },
            source_digest: "source",
            storage_context_digest: "storage",
            interpretation_status: { kind: "parent_graph_v1" }
          },
          rows: view.cursor.kind === "empty" ? [] : nodes,
          selected_content:
            view.cursor.kind === "empty"
              ? []
              : rows.map((row) => ({
                  id: row.id,
                  revision: "1",
                  message: row.content,
                  images: []
                })),
          storage_context_digest: "storage"
        } as never
      }
    )
    const append = vi
      .spyOn(tldwClient, "addChatMessage")
      .mockImplementation(async (_id, body) => {
        rows.push({
          id: body.id!,
          role: body.role,
          content: body.content!,
          parent_message_id: body.parent_message_id ?? null
        })
        const selection = body.tldw_history_selection_v1
        return {
          id: body.id,
          ...(selection
            ? {
                tldw_history_admission_v1: {
                  version: 1,
                  owner_key: "server-owner",
                  conversation_id: newServerId,
                  input_message_id: body.id,
                  input_message_revision: "1",
                  selection_digest: selection.selection_digest,
                  messages: selection.messages,
                  originating_selection_revision: selection.selection_revision
                }
              }
            : {})
        } as never
      })
    return { rows, append }
  }

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (
        [
          "unchanged",
          "nested unrelated",
          "historyId ABA",
          "serverChatId ABA",
          "historyId subscriber ABA",
          "serverChatId subscriber ABA"
        ] as const
      ).map((boundary) => ({ surface, boundary }))
    )
  )(
    "actual mounted $surface native mirror checks $boundary through its held bound record",
    async ({ surface, boundary }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const { rows, append } = declareNativeMirrorTransport()
      const initialConnection = useConnectionStore.getState().state
      useConnectionStore.setState({
        state: {
          ...initialConnection,
          isConnected: true,
          phase: ConnectionPhase.CONNECTED,
          mode: "normal",
          offlineBypass: false
        }
      })
      const consumer = mountConsumer(surface, scoped, {
        actualOption: surface === "full"
      })
      let unrelatedObserved = false
      const restoreFromEarlierSubscriber = useStoreMessageOption.subscribe(
        (next, previous) => {
          if (
            boundary === "nested unrelated" &&
            !unrelatedObserved &&
            next.historyId &&
            !previous.historyId
          ) {
            unrelatedObserved = true
            next.setServerChatTitle("Unrelated nested display title")
          }
          if (!boundary.includes("subscriber")) return
          const field = boundary.startsWith("historyId")
            ? "historyId"
            : "serverChatId"
          if (
            next[field] !==
            (field === "historyId" ? "foreign-mirror" : "foreign-conversation")
          )
            return
          if (field === "historyId")
            next.setHistoryId(previous[field], { preserveServerChatId: true })
          else next.setServerChatId(previous[field])
        }
      )
      const original = Storage.prototype.set
      let enter!: () => void, release!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      vi.spyOn(Storage.prototype, "set").mockImplementation(async function (
        this: Storage,
        key,
        value
      ) {
        const result = await original.call(this, key, value)
        if (
          key.startsWith("registry:chat-creation:") &&
          value?.value?.state === "bound"
        ) {
          enter()
          await held
        }
        return result
      })
      let pending: Promise<ChatSubmitResult | void> | undefined
      try {
        act(() => {
          pending = consumer.mounted.result.current.onSubmit({
            message: "Original first native mirror question",
            image: ""
          })
        })
        await Promise.race([
          entered,
          pending!.then((result) => {
            throw new Error(
              "First Send settled before bound record: " + JSON.stringify(result)
            )
          })
        ])
        await waitFor(() =>
          expect(useStoreMessageOption.getState().historyId).not.toBeNull()
        )
        const core = useStoreMessageOption.getState()
        const receipt = consumer.receipts[0]
        expect(core.serverChatId).toBe(newServerId)
        expect(receipt.owner.kind).toBe("native")
        expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
        const mirrors = await storage.tables.chatHistories.toArray()
        expect(mirrors).toEqual([
          expect.objectContaining({
            id: core.historyId,
            server_chat_id: newServerId,
            server_scope_key: ACCOUNT
          })
        ])
        expect(append).not.toHaveBeenCalled()
        await act(async () => {
          if (boundary.includes("ABA")) {
            const field = boundary.startsWith("historyId")
              ? "historyId"
              : "serverChatId"
            const id = core[field]
            if (field === "historyId") {
              core.setHistoryId("foreign-mirror", { preserveServerChatId: true })
              core.setHistoryId(id, { preserveServerChatId: true })
            } else {
              core.setServerChatId("foreign-conversation")
              core.setServerChatId(id)
            }
          }
          release()
          const result = normalizeChatSubmitResult(await pending)
          if (!boundary.includes("ABA"))
            expect(result).toEqual({ status: "submitted" })
          else expect(result.status).not.toBe("submitted")
        })
        if (!boundary.includes("ABA")) {
          expect(rows.map((row) => [row.role, row.content])).toEqual([
            ["user", "Original first native mirror question"],
            ["assistant", "First answer"]
          ])
          expect(append).toHaveBeenCalledTimes(2)
          expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        } else {
          expect(rows).toEqual([])
          expect(append).not.toHaveBeenCalled()
          expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        }
      } finally {
        release?.()
        if (pending)
          await act(async () => {
            await pending
          })
        restoreFromEarlierSubscriber()
        consumer.unmount()
        useConnectionStore.setState({ state: initialConnection })
      }
    }
  )

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (
        [
          "link failure",
          "New",
          "foreign alias",
          "same-CID reload",
          "account ABA",
          "Stop",
          "model",
          "tools"
        ] as const
      ).map((boundary) => ({ surface, boundary }))
    )
  )(
    "actual mounted $surface native mirror refuses $boundary during its original link",
    async ({ surface, boundary }) => {
      const scoped = answerSnapshot()
      const auth = new AbortController()
      scoped.scopeInvalidatedSignal = auth.signal
      // Declared snapshot IO retains its parent's Stop, as the production lease does.
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockImplementation(
        async (_ids, options) => ({
          ...scoped,
          scopeSignal: options?.signal ?? scoped.scopeSignal
        })
      )
      const { rows, append } = declareNativeMirrorTransport()
      const initialConnection = useConnectionStore.getState().state
      useConnectionStore.setState({
        state: {
          ...initialConnection,
          isConnected: true,
          phase: ConnectionPhase.CONNECTED,
          mode: "normal",
          offlineBypass: false
        }
      })
      const consumer = mountConsumer(surface, scoped, {
        actualOption: surface === "full"
      })
      const source = creationSource()
      const originalAddress = useStoreMessageOption.getState().chatDraftAddress
      let enter!: () => void, release!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      storage.control.beforeHistoryAdd = async () => {
        enter()
        await held
        if (boundary === "link failure")
          throw new Error("Declared mirror storage failure")
      }
      let pending: Promise<ChatSubmitResult | void> | undefined
      let replacement: Promise<unknown> | undefined
      try {
        act(() => {
          pending = consumer.mounted.result.current.onSubmit({
            message: "Retained original native mirror question",
            image: ""
          })
        })
        await Promise.race([
          entered,
          pending!.then((result) => {
            throw new Error(
              "First Send settled before mirror: " + JSON.stringify(result)
            )
          })
        ])
        expect(consumer.selection.getCurrent().owner).toMatchObject({
          kind: "native",
          conversation_id: newServerId
        })
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          originalAddress
        )
        expect(useStoreMessageOption.getState().serverChatId).toBeNull()
        await act(async () => {
          if (boundary === "New")
            useStoreMessageOption.getState().startNewChatDraft(false)
          if (boundary === "foreign alias")
            useStoreMessageOption.setState({ historyId: "foreign-mirror" })
          if (boundary === "same-CID reload")
            replacement = consumer.selection.loadConversation({
              serverChatId: newServerId
            })
          if (boundary === "account ABA") {
            prompt.scope.userId = "bob"
            auth.abort()
            prompt.scope.userId = "alice"
          }
          if (boundary === "Stop")
            consumer.mounted.result.current.stopStreamingRequest()
          if (boundary === "model")
            useStoreMessageOption.getState().setSelectedModel("second-model")
          if (boundary === "tools")
            useStoreMessageOption.getState().setToolChoice("auto")
          release()
          expect(normalizeChatSubmitResult(await pending).status).not.toBe(
            "submitted"
          )
          if (replacement) await replacement
        })
        expect(append).not.toHaveBeenCalled()
        expect(rows).toEqual([])
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(await storage.tables.chatHistories.toArray()).toEqual([])
        expect(await readChatCreationRecovery(source)).toMatchObject({
          state: "acknowledged",
          chatId: newServerId,
          inputText: "Retained original native mirror question"
        })
        if (boundary === "link failure")
          expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
            originalAddress
          )
      } finally {
        release?.()
        if (pending)
          await act(async () => {
            await pending
            if (replacement) await replacement
          })
        storage.control.beforeHistoryAdd = undefined
        consumer.unmount()
        useConnectionStore.setState({ state: initialConnection })
      }
    }
  )

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (["historyId", "serverChatId"] as const).flatMap((field) =>
        (
          [
            "New",
            "foreign ID ABA",
            "original ID ABA",
            "recursive original ID ABA"
          ] as const
        ).map((boundary) => ({
          surface,
          field,
          boundary
        }))
      )
    )
  )(
    "actual mounted $surface native mirror retires $boundary inside its $field setter",
    async ({ surface, field, boundary }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const { rows, append } = declareNativeMirrorTransport()
      const initialConnection = useConnectionStore.getState().state
      useConnectionStore.setState({
        state: {
          ...initialConnection,
          isConnected: true,
          phase: ConnectionPhase.CONNECTED,
          mode: "normal",
          offlineBypass: false
        }
      })
      const consumer = mountConsumer(surface, scoped, {
        actualOption: surface === "full"
      })
      const source = creationSource()
      let observed = false
      let candidate: string | null = null
      const unsubscribe = useStoreMessageOption.subscribe((next, previous) => {
        if (
          boundary === "recursive original ID ABA" &&
          observed &&
          next[field] === null &&
          previous[field] === candidate
        ) {
          useStoreMessageOption.setState({ [field]: candidate })
          return
        }
        if (observed || !next[field] || previous[field]) return
        observed = true
        candidate = next[field]
        if (boundary === "New") next.startNewChatDraft(false)
        else {
          const original = next[field]
          useStoreMessageOption.setState({
            [field]:
              boundary === "original ID ABA" ||
              boundary === "recursive original ID ABA"
                ? previous[field]
                : "foreign-native-alias"
          })
          useStoreMessageOption.setState({ [field]: original })
        }
      })
      try {
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit({
                message: "Original native publication question",
                image: ""
              })
            ).status
          ).not.toBe("submitted")
        })
        expect(observed).toBe(true)
        expect(append).not.toHaveBeenCalled()
        expect(rows).toEqual([])
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(await readChatCreationRecovery(source)).toMatchObject({
          state: "acknowledged",
          chatId: newServerId,
          inputText: "Original native publication question"
        })
        if (field === "historyId")
          expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      } finally {
        unsubscribe()
        consumer.unmount()
        useConnectionStore.setState({ state: initialConnection })
      }
    }
  )

  it.each(["full", "compact"] as const)(
    "actual mounted %s failed mirror link can Review its original created chat before a later explicit Send",
    async (surface) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const { rows, append } = declareNativeMirrorTransport()
      vi.spyOn(tldwClient, "getChat").mockImplementation(
        async (id) => ({ id, title: "Recovered original chat" }) as never
      )
      const initialConnection = useConnectionStore.getState().state
      useConnectionStore.setState({
        state: {
          ...initialConnection,
          isConnected: true,
          phase: ConnectionPhase.CONNECTED,
          mode: "normal",
          offlineBypass: false
        }
      })
      const consumer = mountConsumer(surface, scoped, {
        actualOption: surface === "full",
        withComposer: true
      })
      const input = "  Original failed mirror input.\nRetain this literal.  "
      const originalAddress = useStoreMessageOption.getState().chatDraftAddress
      const source = creationSource()
      try {
        await waitFor(() =>
          expect(consumer.mounted.result.current.composer!.draftReady).toBe(true)
        )
        act(() =>
          consumer.mounted.result.current.composer!.setMessageValue(input)
        )
        storage.control.beforeHistoryAdd = async () => {
          throw new Error("Declared mirror storage failure")
        }
        await act(async () => {
          const result = await consumer.mounted.result.current.onSubmit({
            message: input,
            image: ""
          })
          expect(normalizeChatSubmitResult(result).status).not.toBe("submitted")
        })
        storage.control.beforeHistoryAdd = undefined
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          originalAddress
        )
        expect(
          consumer.mounted.result.current.composer!.form.values.message
        ).toBe(input)
        await waitFor(() =>
          expect(
            consumer.mounted.result.current.composer!.creationRecovery.active
          ).toMatchObject({ state: "acknowledged", chatId: newServerId })
        )
        const originalReceipt = consumer.selection.getCurrent().owner
        await act(async () => {
          expect(
            await consumer.mounted.result.current.composer!.reviewCreatedChat()
          ).toBe(true)
        })
        expect(consumer.selection.getCurrent().owner).not.toBe(originalReceipt)
        expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({
          kind: "native",
          id: newServerId
        })
        expect(
          consumer.mounted.result.current.composer!.form.values.message
        ).toBe(input)
        expect(await readChatCreationRecovery(source)).toMatchObject({
          state: "bound",
          chatId: newServerId
        })
        expect(rows).toEqual([])
        expect(append).not.toHaveBeenCalled()
        expect(tldwClient.createChat).toHaveBeenCalledOnce()
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit({
                message: input,
                image: ""
              })
            )
          ).toEqual({ status: "submitted" })
        })
        expect(rows.map((row) => [row.role, row.content])).toEqual([
          ["user", input],
          ["assistant", "First answer"]
        ])
        expect(append).toHaveBeenCalledTimes(2)
        expect(tldwClient.createChat).toHaveBeenCalledOnce()
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
      } finally {
        storage.control.beforeHistoryAdd = undefined
        consumer.unmount()
        useConnectionStore.setState({ state: initialConnection })
      }
    }
  )

  const alternateCases = (["full", "compact"] as const).flatMap((surface) =>
    (["document", "tab", "local RAG"] as const).flatMap((mode) =>
      [true, false].map((enabled) => ({ surface, mode, enabled }))
    )
  )

  const configureAlternateContext = (surface: Surface, mode: AlternateMode) => {
    const core = useStoreMessageOption.getState()
    if (mode === "document") {
      if (surface === "full") core.setContextFiles(files)
      else core.setUploadedFiles(files)
    }
    if (mode === "tab") core.setDocumentContext(tabs)
    if (mode === "local RAG") {
      core.setChatMode("rag")
      if (surface === "full") {
        core.setFileRetrievalEnabled(true)
        core.setRagMediaIds([73])
      }
    }
  }

  const alternateRequest = (
    surface: Surface,
    mode: AlternateMode,
    message: string
  ) => ({
    message,
    image: "",
    ...(mode === "tab" ? { docs: tabs } : {}),
    ...(mode === "document" && surface === "compact"
      ? { uploadedFiles: files }
      : {})
  })

  const declareRetainedContextIO = async (
    surface: Surface,
    mode: AlternateMode
  ) => {
    const rewrite = vi
      .spyOn(tldwChat, "sendMessage")
      .mockResolvedValue("Rewritten original context question")
    const settingsStorage = new Storage()
    const previousEmbedding = await settingsStorage.get(
      "chatWithWebsiteEmbedding"
    )
    if (surface === "compact" && mode === "local RAG") {
      await settingsStorage.set("chatWithWebsiteEmbedding", true)
      const capturedTab = { id: 23, url: tabs[0].url }
      const tabResponse = {
        html: "<html><body><article><h1>Original tab</h1><p>Original tab context.</p></article></body></html>",
        content:
          "<html><body><article><h1>Original tab</h1><p>Original tab context.</p></article></body></html>",
        title: "Original tab",
        url: tabs[0].url,
        type: "html",
        isPDF: false
      }
      const tabApi = {
        query: vi.fn(
          (
            _query: unknown,
            callback?: (rows: (typeof capturedTab)[]) => void
          ) => {
            callback?.([capturedTab])
            return Promise.resolve([capturedTab])
          }
        )
      }
      const scripting = {
        executeScript: vi.fn(async () => [{ result: tabResponse }])
      }
      vi.stubGlobal("chrome", {
        ...(typeof chrome === "undefined" ? {} : chrome),
        tabs: tabApi,
        scripting
      })
      vi.stubGlobal("browser", {
        ...(typeof browser === "undefined" ? {} : browser),
        tabs: tabApi,
        scripting
      })
      vi.spyOn(tldwClient, "initialize").mockResolvedValue()
      vi.spyOn(tldwClient, "addMedia").mockResolvedValue({
        media_id: 73
      } as never)
    }
    return {
      rewrite,
      restore: async () => {
        if (surface !== "compact" || mode !== "local RAG") return
        if (previousEmbedding === undefined)
          await settingsStorage.remove("chatWithWebsiteEmbedding")
        else
          await settingsStorage.set(
            "chatWithWebsiteEmbedding",
            previousEmbedding
          )
      }
    }
  }

  it.each(alternateCases)(
    "$surface fresh $mode retains its saved enabled=$enabled Scene through accepted H1 bind, consecutive Send, and reopen",
    async ({ surface, mode, enabled }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer(surface, scoped)
      try {
        const originalTarget = captureCurrentActorSceneTarget(
          consumer.selection
        )!
        expect(originalTarget.address?.kind).toBe("fresh")
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: enabled,
          notes: "Original qualified " + surface + " " + mode + " Scene"
        }
        expect(
          await saveActorSettingsForChat({
            target: originalTarget,
            historyId: null,
            serverChatId: null,
            expectedRecord: null,
            settings
          })
        ).toBe(true)
        const originalScene = await getActorSceneForChat({
          target: originalTarget
        })
        expect(originalScene.originalRecord).not.toBeNull()
        const createChat = vi.spyOn(tldwClient, "createChat")
        await act(async () => {
          if (mode === "document" && surface === "full")
            useStoreMessageOption.getState().setContextFiles(files)
          if (mode === "local RAG") {
            useStoreMessageOption.getState().setChatMode("rag")
            if (surface === "full") {
              useStoreMessageOption.getState().setFileRetrievalEnabled(true)
              useStoreMessageOption.getState().setRagMediaIds([73])
            }
          }
        })
        await act(async () => {
          const result = await consumer.mounted.result.current.onSubmit({
            message: "Original alternate-mode question",
            image: "",
            ...(mode === "tab" ? { docs: tabs } : {}),
            ...(mode === "document" && surface === "compact"
              ? { uploadedFiles: files }
              : {})
          })
          expect(normalizeChatSubmitResult(result)).toEqual({
            status: "submitted"
          })
        })
        // Production must obtain and accept the H1 receipt. The test only observes it.
        const firstCore = useStoreMessageOption.getState()
        const historyId = firstCore.historyId!
        expect(consumer.receipts).toHaveLength(1)
        const firstReceipt = consumer.receipts[0]
        expect(consumer.selection.getCurrent().owner).toBe(firstReceipt.owner)
        expect(consumer.selection.getCurrent().view).toMatchObject({
          owner_key: firstReceipt.view.owner_key,
          view_session_id: firstReceipt.view.view_session_id,
          conversation_id: firstReceipt.view.conversation_id
        })
        expect(firstReceipt.owner.kind).toBe("local")
        expect(consumer.acceptedLoads[0].receipt).toBe(firstReceipt)
        expect(consumer.acceptedLoads[0].transition).toMatchObject({
          kind: "bind",
          sourceAddress: originalTarget.address
        })
        expect(consumer.acceptedLoads[0].address).toMatchObject({
          kind: "local",
          id: historyId,
          profileId:
            firstReceipt.owner.kind === "local"
              ? firstReceipt.owner.profile_id
              : undefined
        })
        expect(firstCore.chatDraftTransition?.kind).toBe("bind")
        expect(firstCore.chatDraftAddress).toMatchObject({
          kind: "local",
          id: historyId,
          profileId:
            firstReceipt.owner.kind === "local"
              ? firstReceipt.owner.profile_id
              : undefined
        })
        const storedHistory = await storage.tables.chatHistories.get(historyId)
        expect(storedHistory).toMatchObject({
          id: historyId,
          server_scope_key: ACCOUNT
        })
        expect(
          (await savedMessages()).map((row) => [
            row.history_id,
            row.role,
            row.content
          ])
        ).toEqual([
          [historyId, "user", "Original alternate-mode question"],
          [historyId, "assistant", "First answer"]
        ])
        const boundTarget = captureCurrentActorSceneTarget(consumer.selection)!
        expect(boundTarget.key).not.toBe(originalTarget.key)
        const boundScene = await getActorSceneForChat({ target: boundTarget })
        expect(boundScene.originalRecord).toEqual(originalScene.originalRecord)
        expect(boundScene.settings).toEqual(originalScene.settings)
        const firstWire = vi.mocked(tldwChat.streamMessage).mock.calls[0]
        expect(firstWire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(
          JSON.stringify(wireMessages(firstWire)).includes(settings.notes)
        ).toBe(enabled)
        expect(createChat).not.toHaveBeenCalled()

        await act(async () => {
          clearAlternateContext()
        })
        await act(async () => {
          const result = await consumer.mounted.result.current.onSubmit({
            message: "Immediate consecutive question",
            image: ""
          })
          expect(normalizeChatSubmitResult(result)).toEqual({
            status: "submitted"
          })
        })
        expect(useStoreMessageOption.getState().historyId).toBe(historyId)
        expect(
          (await savedMessages()).filter((row) => row.history_id === historyId)
        ).toHaveLength(4)
        const secondWire = vi.mocked(tldwChat.streamMessage).mock.calls[1]
        expect(secondWire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(JSON.stringify(wireMessages(secondWire))).toContain(
          "Original alternate-mode question"
        )
        expect(
          JSON.stringify(wireMessages(secondWire)).includes(settings.notes)
        ).toBe(enabled)

        await act(async () => {
          consumer.selection.reset()
          const core = useStoreMessageOption.getState()
          core.startNewChatDraft(false)
          core.setHistoryId(null)
          core.setServerChatId(null)
          core.setMessages([])
          core.setHistory([])
          useActorStore.getState().reset()
        })
        const acceptReopen = captureChatDraftLoad("switch")
        let reopenedReceipt!: HistoryLoadReceipt
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { historyId },
              null,
              (receipt) => {
                reopenedReceipt = receipt
                expect(acceptReopen(receipt)).toBe(true)
                // Publish the display alias only after the actual H1 receipt was accepted.
                useStoreMessageOption.getState().setHistoryId(historyId)
              }
            )
          ).toBe(true)
        })
        expect(consumer.selection.getCurrent().owner).toBe(
          reopenedReceipt.owner
        )
        expect(consumer.selection.getCurrent().view).toBe(reopenedReceipt.view)
        const reopenedTarget = captureCurrentActorSceneTarget(
          consumer.selection
        )!
        expect(reopenedTarget.key).toBe(boundTarget.key)
        expect(
          (await getActorSceneForChat({ target: reopenedTarget }))
            .originalRecord
        ).toEqual(originalScene.originalRecord)
        await act(async () => {
          const result = await consumer.mounted.result.current.onSubmit({
            message: "Question after reopen",
            image: ""
          })
          expect(normalizeChatSubmitResult(result)).toEqual({
            status: "submitted"
          })
        })
        const reopenedWire = vi.mocked(tldwChat.streamMessage).mock.calls[2]
        expect(reopenedWire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(JSON.stringify(wireMessages(reopenedWire))).toContain(
          "Immediate consecutive question"
        )
        expect(
          JSON.stringify(wireMessages(reopenedWire)).includes(settings.notes)
        ).toBe(enabled)
        expect((await savedMessages()).map((row) => row.history_id)).toEqual(
          Array(6).fill(historyId)
        )
      } finally {
        consumer.unmount()
      }
    }
  )

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (
        [{ type: "global" }, { type: "workspace", workspaceId: "17" }] as const
      ).map((scope) => ({ surface, scope }))
    )
  )(
    "$surface preserves the alternate-context refusal for a real accepted native $scope.type owner",
    async ({ surface, scope }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer(surface, scoped)
      const serverChatId = "unsupported-alternate-native"
      vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
        async (_id, request) => emptyServerCapture(request, serverChatId)
      )
      try {
        const accept = captureChatDraftLoad("switch")
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { serverChatId, scope },
              null,
              (receipt) => {
                expect(accept(receipt)).toBe(true)
                const core = useStoreMessageOption.getState()
                core.setServerChatId(serverChatId)
                core.setServerChatMetaLoaded(true)
              }
            )
          ).toBe(true)
        })
        expect(consumer.receipts).toHaveLength(1)
        const receipt = consumer.receipts[0]
        expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
        expect(receipt.owner).toMatchObject({
          kind: "native",
          conversation_id: serverChatId,
          scope
        })
        const acceptedAddress =
          useStoreMessageOption.getState().chatDraftAddress
        expect(acceptedAddress).toMatchObject({
          kind: "native",
          id: serverChatId,
          scope
        })
        for (const mode of [
          "document",
          "tab",
          "local RAG"
        ] as AlternateMode[]) {
          await act(async () => {
            clearAlternateContext()
            if (mode === "document" && surface === "full")
              useStoreMessageOption.getState().setContextFiles(files)
            if (mode === "local RAG") {
              useStoreMessageOption.getState().setChatMode("rag")
              if (surface === "full") {
                useStoreMessageOption.getState().setFileRetrievalEnabled(true)
                useStoreMessageOption.getState().setRagMediaIds([73])
              }
            }
          })
          await act(async () => {
            const result = await consumer.mounted.result.current
              .onSubmit({
                message: "Unsupported native alternate context",
                image: "",
                ...(mode === "tab" ? { docs: tabs } : {}),
                ...(mode === "document" && surface === "compact"
                  ? { uploadedFiles: files }
                  : {})
              })
              .catch((error) => ({
                status: "failed" as const,
                errorMessage: error.message
              }))
            expect(normalizeChatSubmitResult(result)).toMatchObject({
              status: "failed",
              errorMessage: "unsupported_history_action_context"
            })
          })
          expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
            acceptedAddress
          )
          expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
          expect(tldwChat.streamMessage).not.toHaveBeenCalled()
          expect(tldwClient.ragSearch).not.toHaveBeenCalled()
          expect(await savedMessages()).toEqual([])
        }
      } finally {
        consumer.unmount()
      }
    }
  )

  it.each(
    [true, false].flatMap((enabled) =>
      (["no evidence", "retrieval refusal"] as const).map((outcome) => ({
        enabled,
        outcome
      }))
    )
  )(
    "full fresh local RAG $outcome retains its enabled=$enabled Scene after a diagnostic bind and plain reopen",
    async ({ enabled, outcome }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const retrieval = vi.mocked(tldwClient.ragSearch)
      if (outcome === "no evidence")
        retrieval.mockResolvedValue({ results: [] } as never)
      else
        retrieval.mockRejectedValue(
          new Error("Selected-source retrieval was refused")
        )
      const consumer = mountConsumer("full", scoped)
      try {
        const sourceTarget = captureCurrentActorSceneTarget(consumer.selection)!
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: enabled,
          notes: "Original diagnostic Scene " + enabled + " " + outcome
        }
        expect(
          await saveActorSettingsForChat({
            target: sourceTarget,
            expectedRecord: null,
            settings,
            historyId: null,
            serverChatId: null
          })
        ).toBe(true)
        const original = await getActorSceneForChat({ target: sourceTarget })
        await act(async () => {
          const core = useStoreMessageOption.getState()
          core.setChatMode("rag")
          core.setFileRetrievalEnabled(true)
          core.setRagMediaIds([73])
          core.setMessageSteeringMode("continue_as_user")
          core.setMessageSteeringForceNarrate(false)
        })
        const question = "Selected-source question without usable evidence"
        await act(async () => {
          expect(
            await consumer.mounted.result.current.onSubmit({
              message: question,
              image: ""
            })
          ).toEqual({ status: "submitted" })
        })
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(retrieval).toHaveBeenCalledExactlyOnceWith(
          question,
          expect.objectContaining({
            include_media_ids: [73],
            sources: ["media_db"],
            enable_generation: false,
            enable_intent_routing: false,
            requestScope: consumer.originalScope
          })
        )
        const core = useStoreMessageOption.getState()
        const historyId = core.historyId!
        expect(consumer.receipts).toHaveLength(1)
        const receipt = consumer.receipts[0]
        expect(consumer.acceptedLoads[0]).toMatchObject({
          receipt,
          address: { kind: "local", id: historyId },
          transition: { kind: "bind", sourceAddress: sourceTarget.address }
        })
        expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
        expect(core.messageSteeringMode).toBe("continue_as_user")
        const boundTarget = captureCurrentActorSceneTarget(consumer.selection)!
        expect(
          (await getActorSceneForChat({ target: boundTarget })).originalRecord
        ).toEqual(original.originalRecord)
        expect(await storage.tables.chatHistories.get(historyId)).toMatchObject(
          { id: historyId, server_scope_key: ACCOUNT }
        )
        const diagnosticRows = await savedMessages()
        expect(diagnosticRows).toHaveLength(2)
        expect(diagnosticRows.map((row) => [row.history_id, row.role])).toEqual(
          [
            [historyId, "user"],
            [historyId, "assistant"]
          ]
        )
        expect(diagnosticRows[0].content).toBe(question)
        for (const row of diagnosticRows)
          expect(row.generationInfo).toMatchObject({
            mode: "rag",
            grounded: false,
            reason:
              outcome === "no evidence"
                ? "selected_source_evidence_not_found"
                : "selected_source_retrieval_failed"
          })
        expect(diagnosticRows[1].content).toContain(
          outcome === "no evidence"
            ? "I couldn't find supporting evidence"
            : "I couldn't retrieve evidence"
        )
        if (outcome === "retrieval refusal")
          expect(diagnosticRows[1].content).toContain(
            "Selected-source retrieval was refused"
          )

        await act(async () => {
          clearAlternateContext()
          consumer.selection.reset()
          core.startNewChatDraft(false)
          core.setHistoryId(null)
          core.setServerChatId(null)
          core.setMessages([])
          core.setHistory([])
          useActorStore.getState().reset()
        })
        const accept = captureChatDraftLoad("switch")
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { historyId },
              null,
              (loaded) => {
                expect(accept(loaded)).toBe(true)
                useStoreMessageOption.getState().setHistoryId(historyId)
              }
            )
          ).toBe(true)
        })
        const reopened = captureCurrentActorSceneTarget(consumer.selection)!
        expect(reopened.key).toBe(boundTarget.key)
        expect(
          (await getActorSceneForChat({ target: reopened })).originalRecord
        ).toEqual(original.originalRecord)
        await act(async () => {
          expect(
            await consumer.mounted.result.current.onSubmit({
              message: "Explicit plain question after reopen",
              image: ""
            })
          ).toEqual({ status: "submitted" })
        })
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        const wire = vi.mocked(tldwChat.streamMessage).mock.calls[0]
        expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(
          JSON.stringify(wireMessages(wire)).includes(settings.notes)
        ).toBe(enabled)
        expect(JSON.stringify(wireMessages(wire))).toContain(
          "Explicit plain question after reopen"
        )
        expect(JSON.stringify(wireMessages(wire))).not.toContain(question)
        expect(JSON.stringify(wireMessages(wire))).not.toContain(
          diagnosticRows[1].content
        )
        expect((await savedMessages()).map((row) => row.history_id)).toEqual(
          Array(4).fill(historyId)
        )
      } finally {
        consumer.unmount()
      }
    }
  )

  const immediateSteeringActions = [
    "full Send",
    "Compare Send",
    "Compare reply"
  ] as const

  it.each(immediateSteeringActions)(
    "$action retains current steering when its actual catalogue refuses images before provider dispatch",
    async (action) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer("full", scoped)
      const image = "data:image/png;base64,original-image"
      const catalog = (vision: boolean) =>
        ["test", "second-model"].map((id) => ({
          id,
          name: id,
          provider: "openai",
          type: "chat" as const,
          capabilities: vision ? ["vision"] : []
        }))
      const declareVision = (vision: boolean) => {
        const models = catalog(vision)
        vi.mocked(tldwModels.getModel).mockImplementation(
          async (id) => models.find((model) => model.id === id) ?? null
        )
        vi.mocked(tldwModels.getModels).mockResolvedValue(models)
      }
      try {
        if (action !== "full Send")
          await act(async () => {
            useStoreMessageOption.getState().setCompareMode(true)
            useStoreMessageOption
              .getState()
              .setCompareSelectedModels(["test", "second-model"])
          })
        let clusterId: string | undefined
        if (action === "Compare reply") {
          declareVision(true)
          await act(async () => {
            expect(
              await consumer.mounted.result.current.onSubmit({
                message: "Seed image Compare question",
                image
              })
            ).toEqual({ status: "submitted" })
          })
          expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
          for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls) {
            expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
            // The legacy fallback preserves this declared payload while using
            // JPEG because the marker is not a valid PNG data URL/signature.
            expect(JSON.stringify(wireMessages(wire))).toContain(
              "data:image/jpeg;base64,original-image"
            )
          }
          expect(
            (await savedMessages()).find((row) => row.role === "user")?.images
          ).toEqual([image])
          clusterId = useStoreMessageOption
            .getState()
            .messages.find((row) => !row.isBot)?.clusterId
          expect(clusterId).toBeTruthy()
          vi.mocked(tldwChat.streamMessage).mockClear()
        }
        declareVision(false)
        vi.mocked(tldwModels.getModel).mockClear()
        vi.mocked(tldwModels.getModels).mockClear()
        await act(async () => {
          useStoreMessageOption
            .getState()
            .setMessageSteeringMode("continue_as_user")
          useStoreMessageOption.getState().setMessageSteeringForceNarrate(true)
        })
        await act(async () => {
          if (action === "Compare reply")
            await consumer.mounted.result.current.sendPerModelReply!({
              clusterId: clusterId!,
              modelId: "test",
              message: "Reply after catalogue image refusal"
            })
          else
            expect(
              normalizeChatSubmitResult(
                await consumer.mounted.result.current.onSubmit({
                  message: "Image question after catalogue refusal",
                  image
                })
              )
            ).toMatchObject({ status: "failed" })
        })
        expect(tldwModels.getModel).toHaveBeenCalledWith("test")
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(useStoreMessageOption.getState()).toMatchObject({
          messageSteeringMode: "continue_as_user",
          messageSteeringForceNarrate: true
        })
      } finally {
        consumer.unmount()
      }
    }
  )

  it.each(immediateSteeringActions)(
    "$action captures the actual steering tuple when setters and Send run before React commits",
    async (action) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer("full", scoped)
      try {
        if (action !== "full Send")
          await act(async () => {
            useStoreMessageOption.getState().setCompareMode(true)
            useStoreMessageOption
              .getState()
              .setCompareSelectedModels(["test", "second-model"])
          })
        let clusterId: string | undefined
        if (action === "Compare reply") {
          await act(async () => {
            expect(
              await consumer.mounted.result.current.onSubmit({
                message: "Seed immediate steering Compare",
                image: ""
              })
            ).toEqual({ status: "submitted" })
          })
          clusterId = useStoreMessageOption
            .getState()
            .messages.find((row) => !row.isBot)?.clusterId
          expect(clusterId).toBeTruthy()
          vi.mocked(tldwChat.streamMessage).mockClear()
        }
        await act(async () => {
          const core = useStoreMessageOption.getState()
          core.setMessageSteeringMode("continue_as_user")
          core.setMessageSteeringForceNarrate(true)
          // Invoke the existing render's handler synchronously after both actual setters.
          if (action === "Compare reply")
            await consumer.mounted.result.current.sendPerModelReply!({
              clusterId: clusterId!,
              modelId: "test",
              message: "Immediate steering tuple question"
            })
          else
            expect(
              await consumer.mounted.result.current.onSubmit({
                message: "Immediate steering tuple question",
                image: ""
              })
            ).toEqual({ status: "submitted" })
        })
        expect(tldwChat.streamMessage).toHaveBeenCalledTimes(
          action === "Compare Send" ? 2 : 1
        )
        for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls) {
          expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
          expect(JSON.stringify(wireMessages(wire))).toContain(
            "Continue the user's current thought"
          )
          expect(JSON.stringify(wireMessages(wire))).toContain(
            "Use narrative prose style for this reply"
          )
        }
        expect(useStoreMessageOption.getState()).toMatchObject({
          messageSteeringMode: "none",
          messageSteeringForceNarrate: false
        })
      } finally {
        consumer.unmount()
      }
    }
  )

  it("saved normal full Send consumes its current steering after its own H1 follow advances the view", async () => {
    const scoped = answerSnapshot()
    vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
      scoped
    )
    const consumer = mountConsumer("full", scoped)
    try {
      await act(async () => {
        expect(
          await consumer.mounted.result.current.onSubmit({
            message: "Seed saved normal question",
            image: ""
          })
        ).toEqual({ status: "submitted" })
      })
      const initial = consumer.selection.getCurrent()
      expect(initial.owner?.kind).toBe("local")
      const owner = initial.owner!
      const view = initial.view!
      await act(async () => {
        useStoreMessageOption
          .getState()
          .setMessageSteeringMode("continue_as_user")
      })
      await act(async () => {
        expect(
          await consumer.mounted.result.current.onSubmit({
            message: "Current saved normal steering question",
            image: ""
          })
        ).toEqual({ status: "submitted" })
      })
      const current = consumer.selection.getCurrent()
      expect(current.owner).toBe(owner)
      expect(current.view?.view_session_id).toBe(view.view_session_id)
      expect(current.view?.selection_revision).toBeGreaterThan(
        view.selection_revision
      )
      const wire = vi.mocked(tldwChat.streamMessage).mock.calls[1]
      expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
      expect(JSON.stringify(wireMessages(wire))).toContain(
        "Continue the user's current thought"
      )
      expect(useStoreMessageOption.getState()).toMatchObject({
        messageSteeringMode: "none",
        messageSteeringForceNarrate: false
      })
      expect((await savedMessages()).map((row) => row.history_id)).toEqual(
        Array(4).fill(owner.conversation_id)
      )
    } finally {
      consumer.unmount()
    }
  })

  it.each(alternateCases)(
    "$surface retained $mode context keeps enabled=$enabled Scene on the second Send and accepted reopen",
    async ({ surface, mode, enabled }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const external = await declareRetainedContextIO(surface, mode)
      const consumer = mountConsumer(surface, scoped)
      try {
        const originalTarget = captureCurrentActorSceneTarget(
          consumer.selection
        )!
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: enabled,
          notes: "Retained qualified " + surface + " " + mode + " Scene"
        }
        expect(
          await saveActorSettingsForChat({
            target: originalTarget,
            historyId: null,
            serverChatId: null,
            expectedRecord: null,
            settings
          })
        ).toBe(true)
        const originalScene = await getActorSceneForChat({
          target: originalTarget
        })
        expect(originalScene.originalRecord).not.toBeNull()
        await act(async () => {
          configureAlternateContext(surface, mode)
        })
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit(
                alternateRequest(
                  surface,
                  mode,
                  "First retained context question"
                )
              )
            )
          ).toEqual({ status: "submitted" })
        })
        const core = useStoreMessageOption.getState()
        const historyId = core.historyId!
        const acceptedAddress = core.chatDraftAddress
        const receipt = consumer.receipts[0]
        expect(consumer.receipts).toHaveLength(1)
        expect(receipt.owner).toMatchObject({
          kind: "local",
          conversation_id: historyId
        })
        expect(consumer.acceptedLoads[0]).toMatchObject({
          receipt,
          address: acceptedAddress,
          transition: { kind: "bind", sourceAddress: originalTarget.address }
        })
        expect(await storage.tables.chatHistories.get(historyId)).toMatchObject(
          { id: historyId, server_scope_key: ACCOUNT }
        )
        const boundTarget = captureCurrentActorSceneTarget(consumer.selection)!
        expect(
          (await getActorSceneForChat({ target: boundTarget })).originalRecord
        ).toEqual(originalScene.originalRecord)

        // Keep all real context selectors and request documents in place.
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit(
                alternateRequest(
                  surface,
                  mode,
                  "Second retained context question"
                )
              )
            )
          ).toEqual({ status: "submitted" })
        })
        expect(useStoreMessageOption.getState().historyId).toBe(historyId)
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          acceptedAddress
        )
        expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
        expect(consumer.receipts).toHaveLength(1)
        expect(
          (await savedMessages()).map((row) => [
            row.history_id,
            row.role,
            row.content
          ])
        ).toEqual([
          [historyId, "user", "First retained context question"],
          [historyId, "assistant", "First answer"],
          [historyId, "user", "Second retained context question"],
          [historyId, "assistant", "First answer"]
        ])

        await act(async () => {
          consumer.selection.reset()
          const next = useStoreMessageOption.getState()
          next.startNewChatDraft(false)
          next.setHistoryId(null)
          next.setServerChatId(null)
          next.setMessages([])
          next.setHistory([])
          useActorStore.getState().reset()
        })
        const accept = captureChatDraftLoad("switch")
        let reopened!: HistoryLoadReceipt
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { historyId },
              null,
              (loaded) => {
                reopened = loaded
                expect(accept(loaded)).toBe(true)
                useStoreMessageOption.getState().setHistoryId(historyId)
              }
            )
          ).toBe(true)
        })
        expect(consumer.selection.getCurrent().owner).toBe(reopened.owner)
        expect(consumer.selection.getCurrent().view).toBe(reopened.view)
        const reopenedTarget = captureCurrentActorSceneTarget(
          consumer.selection
        )!
        const reopenedScene = await getActorSceneForChat({
          target: reopenedTarget
        })
        expect(reopenedTarget.key).toBe(boundTarget.key)
        expect(reopenedScene.originalRecord).toEqual(
          originalScene.originalRecord
        )
        expect(reopenedScene.settings).toEqual(originalScene.settings)
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit(
                alternateRequest(
                  surface,
                  mode,
                  "Reopened retained context question"
                )
              )
            )
          ).toEqual({ status: "submitted" })
        })
        expect(consumer.receipts).toHaveLength(2)
        expect((await savedMessages()).map((row) => row.history_id)).toEqual(
          Array(6).fill(historyId)
        )
        expect(
          (await savedMessages())
            .filter((row) => row.role === "user")
            .map((row) => row.content)
        ).toEqual([
          "First retained context question",
          "Second retained context question",
          "Reopened retained context question"
        ])
        expect(tldwChat.streamMessage).toHaveBeenCalledTimes(3)
        for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls) {
          expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
          expect(
            JSON.stringify(wireMessages(wire)).includes(settings.notes)
          ).toBe(enabled)
          expect(JSON.stringify(wireMessages(wire))).toContain(
            mode === "tab"
              ? "Original tab context"
              : "Original document context"
          )
        }
        if (
          mode === "document" ||
          (mode === "local RAG" && surface === "compact")
        ) {
          expect(external.rewrite).toHaveBeenCalledTimes(2)
          for (const request of external.rewrite.mock.calls) {
            expect(request[1]).toMatchObject({
              saveToDb: false,
              requestScope: consumer.originalScope,
              signal: expect.any(AbortSignal)
            })
            // The real factory resolves toolChoice:none to omitted request tools.
            expect(request[1]?.tools).toBeUndefined()
            expect(request[1]?.toolChoice).toBeUndefined()
          }
        } else expect(external.rewrite).not.toHaveBeenCalled()
        if (mode === "local RAG") {
          expect(tldwClient.ragSearch).toHaveBeenCalledTimes(3)
          for (const request of vi.mocked(tldwClient.ragSearch).mock.calls)
            expect(request[1]).toMatchObject({
              include_media_ids: [73],
              requestScope: consumer.originalScope
            })
        }
      } finally {
        consumer.unmount()
        await external.restore()
      }
    }
  )

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (["document", "tab", "local RAG"] as const).flatMap((mode) =>
        (
          [
            { type: "global" },
            { type: "workspace", workspaceId: "17" }
          ] as const
        ).map((scope) => ({ surface, mode, scope }))
      )
    )
  )(
    "$surface retained $mode context still refuses an accepted native $scope.type destination after its local bind",
    async ({ surface, mode, scope }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const external = await declareRetainedContextIO(surface, mode)
      const consumer = mountConsumer(surface, scoped)
      const serverChatId = "native-after-local-context"
      try {
        await act(async () => {
          configureAlternateContext(surface, mode)
        })
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit(
                alternateRequest(
                  surface,
                  mode,
                  "Original local context before native Open"
                )
              )
            )
          ).toEqual({ status: "submitted" })
        })
        expect(consumer.receipts[0].owner.kind).toBe("local")
        const originalRows = await savedMessages()
        expect(originalRows).toHaveLength(2)
        vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
          async (_id, request) => emptyServerCapture(request, serverChatId)
        )
        const accept = captureChatDraftLoad("switch")
        let nativeReceipt!: HistoryLoadReceipt
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { serverChatId, scope },
              null,
              (receipt) => {
                nativeReceipt = receipt
                expect(accept(receipt)).toBe(true)
                const core = useStoreMessageOption.getState()
                core.setHistoryId(null)
                core.setServerChatId(serverChatId)
                core.setServerChatMetaLoaded(true)
              }
            )
          ).toBe(true)
        })
        const acceptedAddress =
          useStoreMessageOption.getState().chatDraftAddress
        expect(nativeReceipt.owner).toMatchObject({
          kind: "native",
          conversation_id: serverChatId,
          scope
        })
        expect(acceptedAddress).toMatchObject({
          kind: "native",
          id: serverChatId,
          scope
        })
        expect(consumer.selection.getCurrent().owner).toBe(nativeReceipt.owner)
        vi.mocked(tldwChat.streamMessage).mockClear()
        vi.mocked(tldwClient.ragSearch).mockClear()
        external.rewrite.mockClear()
        await act(async () => {
          const result = await consumer.mounted.result.current
            .onSubmit(
              alternateRequest(
                surface,
                mode,
                "Refused native retained context question"
              )
            )
            .catch((error) => ({
              status: "failed" as const,
              errorMessage: error.message
            }))
          expect(normalizeChatSubmitResult(result)).toMatchObject({
            status: "failed",
            errorMessage: "unsupported_history_action_context"
          })
        })
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          acceptedAddress
        )
        expect(consumer.selection.getCurrent().owner).toBe(nativeReceipt.owner)
        expect(consumer.selection.getCurrent().view).toBe(nativeReceipt.view)
        expect(tldwChat.streamMessage).not.toHaveBeenCalled()
        expect(tldwClient.ragSearch).not.toHaveBeenCalled()
        expect(external.rewrite).not.toHaveBeenCalled()
        expect(await savedMessages()).toEqual(originalRows)
      } finally {
        consumer.unmount()
        await external.restore()
      }
    }
  )

  it.each(
    (["document", "tab"] as const).flatMap((mode) =>
      (["failed", "skipped"] as const).map((outcome) => ({ mode, outcome }))
    )
  )(
    "compact $mode propagates the actual shared $outcome result after provider failure or Stop",
    async ({ mode, outcome }) => {
      const scoped = answerSnapshot()
      let originalCallerSignal: AbortSignal | undefined
      let requestSignal: AbortSignal | undefined
      // The declared prompt IO result keeps the production lease contract:
      // Stop aborts the request signal without invalidating the account scope.
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockImplementation(
        async (_ids, options) => {
          if (outcome !== "skipped") return scoped
          originalCallerSignal = options?.signal
          if (!originalCallerSignal)
            throw new Error(
              "The actual compact request did not supply its Stop signal"
            )
          const lease = new AbortController()
          if (originalCallerSignal.aborted) lease.abort()
          else
            originalCallerSignal.addEventListener(
              "abort",
              () => lease.abort(),
              { once: true }
            )
          requestSignal = lease.signal
          return { ...scoped, scopeSignal: requestSignal }
        }
      )
      const consumer = mountConsumer("compact", scoped)
      let release!: () => void
      let entered!: () => void
      const providerEntered = new Promise<void>((resolve) => {
        entered = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      vi.mocked(tldwChat.streamMessage).mockImplementation(
        async function* (_messages, options) {
          options.onDispatch?.()
          entered()
          if (outcome === "failed")
            throw new Error("Declared original provider refusal")
          await held
          yield* []
          throw new DOMException(
            "Declared original request stopped",
            "AbortError"
          )
        }
      )
      let pending!: Promise<ChatSubmitResult | void>
      try {
        await act(async () => {
          configureAlternateContext("compact", mode)
        })
        act(() => {
          pending = consumer.mounted.result.current.onSubmit(
            alternateRequest(
              "compact",
              mode,
              "Original context result question"
            )
          )
        })
        await Promise.race([
          providerEntered,
          pending.then((result) => {
            throw new Error(
              "The actual provider was not entered: " + JSON.stringify(result)
            )
          })
        ])
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        expect(
          vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.requestScope
        ).toEqual(consumer.originalScope)
        expect(consumer.receipts).toHaveLength(1)
        expect(consumer.selection.getCurrent().owner).toMatchObject({
          kind: "local"
        })
        if (outcome === "skipped")
          await act(async () => {
            consumer.mounted.result.current.stopStreamingRequest()
            expect(originalCallerSignal?.aborted).toBe(true)
            expect(requestSignal?.aborted).toBe(true)
            expect(
              vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.signal
                ?.aborted
            ).toBe(true)
            expect(scoped.scopeInvalidatedSignal.aborted).toBe(false)
            release()
          })
        let result!: ChatSubmitResult | void
        await act(async () => {
          result = await pending
        })
        expect(normalizeChatSubmitResult(result)).toMatchObject({
          status: outcome
        })
        expect(
          (await savedMessages())
            .filter((row) => row.role === "user")
            .map((row) => row.content)
        ).toEqual(["Original context result question"])
      } finally {
        release()
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  it("compact held original reply cannot clear a later accepted conversation's reply target after New", async () => {
    const scoped = answerSnapshot()
    vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
      scoped
    )
    const consumer = mountConsumer("compact", scoped)
    let release!: () => void
    let pending: Promise<ChatSubmitResult | void> | undefined
    try {
      await act(async () => {
        expect(
          await consumer.mounted.result.current.onSubmit({
            message: "Seed original reply owner",
            image: ""
          })
        ).toEqual({ status: "submitted" })
      })
      const originalOwner = consumer.selection.getCurrent().owner!
      const originalReplyRow = useStoreMessageOption.getState().messages.at(-1)!
      await act(async () => {
        useStoreMessageOption.getState().setReplyTarget({
          id: originalReplyRow.id!,
          preview: originalReplyRow.message,
          isBot: true
        })
      })
      let enter!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      const question = "Held original reply question"
      vi.mocked(tldwChat.streamMessage).mockImplementation(
        async function* (messages, options) {
          options.onDispatch?.()
          if (
            JSON.stringify(
              options.preparedRequest?.messages ?? messages
            ).includes(question)
          ) {
            enter()
            await held
            yield "Original late reply answer"
          } else yield "Later conversation answer"
        }
      )
      act(() => {
        pending = consumer.mounted.result.current.onSubmit({
          message: question,
          image: ""
        })
      })
      await Promise.race([
        entered,
        pending!.then((result) => {
          throw new Error(
            "The reply completed before its original provider entered: " +
              JSON.stringify(result)
          )
        })
      ])
      expect(
        (await savedMessages())
          .filter(
            (row) =>
              row.history_id === originalOwner.conversation_id &&
              row.role === "user"
          )
          .map((row) => row.content)
      ).toEqual(["Seed original reply owner", question])
      await act(async () => {
        expect(consumer.mounted.result.current.clearChat!()).toBe(true)
      })
      expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe(
        "fresh"
      )
      expect(useStoreMessageOption.getState().replyTarget).toBeNull()
      await act(async () => {
        expect(
          await consumer.mounted.result.current.onSubmit({
            message: "Seed later accepted reply owner",
            image: ""
          })
        ).toEqual({ status: "submitted" })
      })
      const laterOwner = consumer.selection.getCurrent().owner!
      expect(laterOwner.kind).toBe("local")
      expect(laterOwner).not.toBe(originalOwner)
      const laterAddress = useStoreMessageOption.getState().chatDraftAddress
      const laterReplyRow = useStoreMessageOption.getState().messages.at(-1)!
      const laterTarget = {
        id: laterReplyRow.id!,
        preview: laterReplyRow.message,
        isBot: true
      }
      await act(async () => {
        useStoreMessageOption.getState().setReplyTarget(laterTarget)
      })
      await act(async () => {
        release()
        await pending
      })
      expect(useStoreMessageOption.getState().replyTarget).toBe(laterTarget)
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
        laterAddress
      )
      expect(consumer.selection.getCurrent().owner).toBe(laterOwner)
      expect(
        (await savedMessages())
          .filter((row) => row.history_id === laterOwner.conversation_id)
          .map((row) => row.content)
      ).toEqual([
        "Seed later accepted reply owner",
        "Later conversation answer"
      ])
      expect(
        useStoreMessageOption.getState().messages.map((row) => row.message)
      ).toEqual([
        "Seed later accepted reply owner",
        "Later conversation answer"
      ])
      for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls)
        expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
    } finally {
      release?.()
      if (pending)
        await act(async () => {
          await pending
        })
      consumer.unmount()
    }
  })

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (["current", "reselected target", "New"] as const).map((boundary) => ({
        surface,
        boundary
      }))
    )
  )(
    "$surface saved H1 Reply clears only its original target after $boundary",
    async ({ surface, boundary }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer(surface, scoped)
      let release!: () => void, enter!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      let pending: Promise<ChatSubmitResult | void> | undefined
      try {
        await act(async () => {
          expect(
            normalizeChatSubmitResult(
              await consumer.mounted.result.current.onSubmit({
                message: "Seed saved Reply owner",
                image: ""
              })
            )
          ).toEqual({ status: "submitted" })
        })
        const original = useStoreMessageOption.getState()
        const originalOwner = consumer.selection.getCurrent().owner!
        expect(originalOwner.kind).toBe("local")
        const last = original.messages.at(-1)!
        expect(last.isBot).toBe(true)
        const originalTarget = {
          id: last.id!,
          preview: last.message,
          isBot: true
        }
        await act(async () => {
          original.setReplyTarget(originalTarget)
        })
        vi.mocked(tldwChat.streamMessage).mockImplementationOnce(
          async function* (_messages, options) {
            options.onDispatch?.()
            enter()
            await held
            yield "Successful saved Reply answer"
          }
        )
        act(() => {
          pending = consumer.mounted.result.current.onSubmit({
            message: "Saved Reply question",
            image: ""
          })
        })
        await Promise.race([
          entered,
          pending!.then((result) => {
            throw new Error(
              "Saved Reply missed real provider admission: " +
                JSON.stringify(result)
            )
          })
        ])
        const laterTarget = {
          ...originalTarget,
          preview: "Later explicit reply selection"
        }
        await act(async () => {
          const core = useStoreMessageOption.getState()
          if (boundary === "New") {
            core.startNewChatDraft(false)
            consumer.selection.reset()
            core.setHistoryId(null)
            core.setServerChatId(null)
            core.setMessages([])
            core.setHistory([])
          }
          if (boundary !== "current") core.setReplyTarget(laterTarget)
        })
        const replacementAddress =
          useStoreMessageOption.getState().chatDraftAddress
        await act(async () => {
          release()
          expect(normalizeChatSubmitResult(await pending)).toEqual({
            status: "submitted"
          })
        })
        expect(
          (await savedMessages())
            .filter((row) => row.history_id === original.historyId)
            .map((row) => row.content)
        ).toEqual([
          "Seed saved Reply owner",
          "First answer",
          "Saved Reply question",
          "Successful saved Reply answer"
        ])
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          replacementAddress
        )
        if (boundary === "current") {
          expect(useStoreMessageOption.getState().replyTarget).toBeNull()
          expect(consumer.selection.getCurrent().owner).toBe(originalOwner)
          await act(async () => {
            expect(
              normalizeChatSubmitResult(
                await consumer.mounted.result.current.onSubmit({
                  message: "Immediate plain Send after Reply",
                  image: ""
                })
              )
            ).toEqual({ status: "submitted" })
          })
          expect(tldwChat.streamMessage).toHaveBeenCalledTimes(3)
          expect(
            (await savedMessages())
              .filter((row) => row.history_id === original.historyId)
              .map((row) => row.content)
          ).toEqual([
            "Seed saved Reply owner",
            "First answer",
            "Saved Reply question",
            "Successful saved Reply answer",
            "Immediate plain Send after Reply",
            "First answer"
          ])
        } else {
          expect(useStoreMessageOption.getState().replyTarget).toBe(laterTarget)
          expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
          if (boundary === "New") {
            expect(replacementAddress).not.toBe(original.chatDraftAddress)
            expect(useStoreMessageOption.getState().messages).toEqual([])
            expect(useStoreMessageOption.getState().historyId).toBeNull()
          }
        }
        for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls)
          expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
      } finally {
        release?.()
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  const steeringCases = (
    ["full Send", "Compare Send", "Compare reply"] as const
  ).flatMap((action) =>
    (
      [
        "current completion",
        "New",
        "later same-source edit",
        "later same-source edit back to original flags"
      ] as const
    ).map((boundary) => ({ action, boundary }))
  )

  it.each(steeringCases)(
    "$action consumes only its original steering across $boundary",
    async ({ action, boundary }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer("full", scoped)
      let release!: () => void
      try {
        const models = ["test", "second-model"]
        if (action !== "full Send") {
          await act(async () => {
            useStoreMessageOption.getState().setCompareMode(true)
            useStoreMessageOption.getState().setCompareSelectedModels(models)
          })
        }
        let clusterId: string | undefined
        if (action === "Compare reply") {
          await act(async () => {
            expect(
              await consumer.mounted.result.current.onSubmit({
                message: "Seed Compare question",
                image: ""
              })
            ).toEqual({ status: "submitted" })
          })
          clusterId = useStoreMessageOption
            .getState()
            .messages.find((row) => !row.isBot)?.clusterId
          expect(clusterId).toBeTruthy()
        }
        await act(async () => {
          useStoreMessageOption
            .getState()
            .setMessageSteeringMode("continue_as_user")
          useStoreMessageOption.getState().setMessageSteeringForceNarrate(false)
        })
        const oldQuestion = "Question with original single-response steering"
        let enter!: () => void
        const entered = new Promise<void>((resolve) => {
          enter = resolve
        })
        const held = new Promise<void>((resolve) => {
          release = resolve
        })
        const wireOffset = vi.mocked(tldwChat.streamMessage).mock.calls.length
        let enteredCount = 0
        vi.mocked(tldwChat.streamMessage).mockImplementation(
          async function* (_messages, options) {
            options.onDispatch?.()
            if (++enteredCount === (action === "Compare Send" ? 2 : 1)) enter()
            await held
            yield "Original scoped steering answer"
          }
        )
        let pending!: Promise<unknown>
        act(() => {
          pending =
            action === "Compare reply"
              ? consumer.mounted.result.current.sendPerModelReply!({
                  clusterId: clusterId!,
                  modelId: models[0],
                  message: oldQuestion
                })
              : consumer.mounted.result.current.onSubmit({
                  message: oldQuestion,
                  image: ""
                })
        })
        await Promise.race([
          entered,
          pending.then((result) => {
            throw new Error(
              "The turn completed before its held providers entered: " +
                JSON.stringify(result)
            )
          })
        ])
        const originalHistoryId = useStoreMessageOption.getState().historyId!
        expect(originalHistoryId).toBeTruthy()
        const originalAddress =
          useStoreMessageOption.getState().chatDraftAddress
        expect(originalAddress?.kind).toBe("local")
        const originalWire = vi.mocked(tldwChat.streamMessage).mock.calls[
          wireOffset
        ]
        expect(originalWire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(JSON.stringify(wireMessages(originalWire))).toContain(
          "Continue the user's current thought"
        )

        await act(async () => {
          const core = useStoreMessageOption.getState()
          if (boundary === "New") {
            core.startNewChatDraft(false)
            consumer.selection.reset()
            core.setHistoryId(null)
            core.setServerChatId(null)
            core.setMessages([])
            core.setHistory([])
            core.setStreaming(false)
            core.setIsProcessing(false)
          }
          if (boundary !== "current completion") {
            core.setMessageSteeringMode("impersonate_user")
            core.setMessageSteeringForceNarrate(true)
            if (boundary === "later same-source edit back to original flags") {
              core.setMessageSteeringMode("continue_as_user")
              core.setMessageSteeringForceNarrate(false)
            }
          }
        })
        const replacementAddress =
          useStoreMessageOption.getState().chatDraftAddress
        const expectedSteering =
          boundary === "current completion"
            ? {
                messageSteeringMode: "none",
                messageSteeringForceNarrate: false
              }
            : boundary === "later same-source edit back to original flags"
              ? {
                  messageSteeringMode: "continue_as_user",
                  messageSteeringForceNarrate: false
                }
              : {
                  messageSteeringMode: "impersonate_user",
                  messageSteeringForceNarrate: true
                }
        await act(async () => {
          release()
          await pending
        })
        expect(useStoreMessageOption.getState()).toMatchObject(expectedSteering)
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          replacementAddress
        )
        const rows = await savedMessages()
        expect(rows.every((row) => row.history_id === originalHistoryId)).toBe(
          true
        )
        expect(rows).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              history_id: originalHistoryId,
              role: "user",
              content: oldQuestion
            })
          ])
        )
        if (boundary === "New") {
          expect(replacementAddress).not.toBe(originalAddress)
          expect(useStoreMessageOption.getState().historyId).toBeNull()
          expect(useStoreMessageOption.getState().messages).toEqual([])
        } else {
          expect(replacementAddress).toBe(originalAddress)
          expect(rows).toEqual(
            expect.arrayContaining([
              expect.objectContaining({
                history_id: originalHistoryId,
                role: "user",
                content: oldQuestion
              }),
              expect.objectContaining({
                history_id: originalHistoryId,
                role: "assistant",
                content: "Original scoped steering answer"
              })
            ])
          )
        }
      } finally {
        release?.()
        consumer.unmount()
      }
    }
  )

  it.each(
    (["full", "compact"] as const).flatMap((surface) =>
      (
        [
          "current",
          "raw metadata replacement",
          "source selection intent",
          "restore intent"
        ] as const
      ).map((boundary) => ({ surface, boundary }))
    )
  )(
    "$surface result follow retains original command permission after $boundary",
    async ({ surface, boundary }) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer(surface, scoped)
      let release!: () => void, enter!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      let pending: Promise<ChatSubmitResult | void> | undefined
      try {
        const originalSource = captureCurrentActorSceneTarget(
          consumer.selection
        )!
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Original admitted follow Scene"
        }
        expect(
          await saveActorSettingsForChat({
            historyId: null,
            serverChatId: null,
            target: originalSource,
            expectedRecord: null,
            settings
          })
        ).toBe(true)
        vi.mocked(tldwChat.streamMessage).mockImplementationOnce(
          async function* (_messages, options) {
            options.onDispatch?.()
            yield "Original partial reply"
            enter()
            await held
            yield " completed"
          }
        )
        act(() => {
          pending = consumer.mounted.result.current.onSubmit({
            message: "Original admitted follow question",
            image: ""
          })
        })
        await Promise.race([
          entered,
          pending!.then((result) => {
            throw new Error(
              "Follow control missed original provider admission: " +
                JSON.stringify(result)
            )
          })
        ])
        const source = useStoreMessageOption.getState()
        const selected = consumer.selection.getCurrent()
        expect(selected.status).toBe("ready")
        expect(selected.owner).toMatchObject({
          kind: "local",
          conversation_id: source.historyId
        })
        expect(consumer.receipts).toHaveLength(1)
        expect(source.chatDraftAddress).toMatchObject({
          kind: "local",
          id: source.historyId
        })
        expect(
          (await savedMessages()).map((row) => [row.role, row.content])
        ).toEqual([["user", "Original admitted follow question"]])
        const foreignMessages = [
          {
            id: "foreign-follow-row",
            name: "You",
            isBot: false,
            message: "Keep the replacement presentation",
            images: [],
            sources: []
          }
        ]
        const foreignHistory = [
          {
            role: "user" as const,
            content: "Keep the replacement presentation"
          }
        ]
        if (boundary !== "current")
          await act(async () => {
            const core = useStoreMessageOption.getState()
            if (boundary === "raw metadata replacement")
              core.setHistoryId("foreign-follow-history")
            if (boundary === "source selection intent")
              usePlaygroundSessionStore.getState().markSourceSelectionIntent()
            if (boundary === "restore intent")
              usePlaygroundSessionStore.getState().cancelPendingRestore()
            core.setMessages(foreignMessages)
            core.setHistory(foreignHistory)
          })
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          source.chatDraftAddress
        )
        expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
          source.chatDraftRevision
        )
        expect(consumer.selection.getCurrent().owner).toBe(selected.owner)
        expect(consumer.selection.getCurrent().view).toBe(selected.view)
        expect(consumer.draftAccount.isCurrent()).toBe(true)
        await act(async () => {
          release()
          expect(normalizeChatSubmitResult(await pending)).toEqual({
            status: "submitted"
          })
        })
        const saved = await savedMessages()
        expect(
          saved.map((row) => [row.history_id, row.role, row.content])
        ).toEqual([
          [source.historyId, "user", "Original admitted follow question"],
          [source.historyId, "assistant", "Original partial reply completed"]
        ])
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        expect(
          vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.requestScope
        ).toEqual(consumer.originalScope)
        expect(
          JSON.stringify(
            wireMessages(vi.mocked(tldwChat.streamMessage).mock.calls[0])
          )
        ).toContain(settings.notes)
        expect(consumer.selection.getCurrent().owner).toBe(selected.owner)
        expect(consumer.selection.getCurrent().status).toBe("ready")
        if (boundary === "current") {
          expect(useStoreMessageOption.getState().historyId).toBe(
            source.historyId
          )
          expect(
            useStoreMessageOption.getState().messages.map((row) => row.message)
          ).toEqual([
            "Original admitted follow question",
            "Original partial reply completed"
          ])
          expect(consumer.selection.getCurrent().view?.cursor).toEqual({
            kind: "after_message",
            message_id: saved[1].id
          })
          expect(
            await loadHistoryTurnRecoveries(
              selected.bookmarkScope!,
              selected.view!
            )
          ).toEqual([])
        } else {
          expect(useStoreMessageOption.getState().historyId).toBe(
            boundary === "raw metadata replacement"
              ? "foreign-follow-history"
              : source.historyId
          )
          expect(useStoreMessageOption.getState().serverChatId).toBe(
            source.serverChatId
          )
          expect(useStoreMessageOption.getState().messages).toEqual(
            foreignMessages
          )
          expect(useStoreMessageOption.getState().history).toEqual(
            foreignHistory
          )
          expect(consumer.selection.getCurrent().view).toBe(selected.view)
          expect(
            await loadHistoryTurnRecoveries(
              selected.bookmarkScope!,
              selected.view!
            )
          ).toEqual([
            expect.objectContaining({
              turn: expect.objectContaining({
                outcome: "complete",
                settled_message_id: saved[1].id,
                result_text: "Original partial reply completed"
              })
            })
          ])
        }
      } finally {
        release?.()
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  it.each([
    "current",
    "typed bind raw-ID replacement",
    "own history setter raw-ID replacement"
  ] as const)(
    "full fresh Compare qualifies exact issued metadata after %s",
    async (boundary) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      const consumer = mountConsumer("full", scoped)
      let unsubscribe: (() => void) | undefined
      try {
        await act(async () => {
          const core = useStoreMessageOption.getState()
          core.setCompareMode(true)
          core.setCompareSelectedModels(["test", "second-model"])
        })
        const original = useStoreMessageOption.getState()
        const sourceTarget = captureCurrentActorSceneTarget(consumer.selection)!
        expect(sourceTarget.address).toEqual(original.chatDraftAddress)
        expect(sourceTarget.isCurrent()).toBe(true)
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Exact original Compare metadata Scene"
        }
        expect(
          await saveActorSettingsForChat({
            historyId: null,
            serverChatId: null,
            target: sourceTarget,
            expectedRecord: null,
            settings
          })
        ).toBe(true)
        const originalScene = await getActorSceneForChat({
          target: sourceTarget
        })
        expect(originalScene.originalRecord).not.toBeNull()
        let observedAddress: typeof original.chatDraftAddress | undefined
        let observedRevision: number | undefined
        let replacement:
          | ReturnType<typeof useStoreMessageOption.getState>
          | undefined
        let replacementApplied = false
        unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
          const typedBind =
            state.chatDraftAddress?.kind === "local" &&
            previous.chatDraftAddress?.kind === "fresh"
          const ownHistorySetter =
            state.chatDraftAddress?.kind === "local" &&
            state.historyId === state.chatDraftAddress.id &&
            previous.historyId !== state.historyId
          if (
            !typedBind &&
            !(
              boundary === "own history setter raw-ID replacement" &&
              ownHistorySetter
            )
          )
            return
          observedAddress = state.chatDraftAddress
          observedRevision = state.chatDraftRevision
          if (
            boundary === "current" ||
            replacementApplied ||
            (boundary === "own history setter raw-ID replacement" &&
              !ownHistorySetter)
          )
            return
          replacementApplied = true
          // Only real raw metadata setters run here. The accepted typed address,
          // H1 owner/view, revisions and account deliberately stay unchanged.
          state.setHistoryId("replacement-Compare-raw-history")
          state.setMessages([
            {
              id: "replacement-Compare-row",
              isBot: false,
              name: "You",
              message: "Replacement Compare view owns this row",
              images: [],
              sources: []
            }
          ])
          state.setHistory([
            { role: "user", content: "Replacement Compare view owns this row" }
          ])
          state.setServerChatTitle("Replacement Compare metadata")
          state.setServerChatMetaLoaded(true)
          replacement = useStoreMessageOption.getState()
        })
        let result!: ChatSubmitResult | void
        await act(async () => {
          result = await consumer.mounted.result.current.onSubmit({
            message: "Original fresh Compare raw-ID question",
            image: ""
          })
        })
        expect(observedAddress).toMatchObject({ kind: "local" })
        expect(consumer.receipts).toHaveLength(1)
        const receipt = consumer.receipts[0]
        expect(receipt.owner).toMatchObject({
          kind: "local",
          conversation_id: observedAddress!.id
        })
        expect(consumer.selection.getCurrent().owner).toBe(receipt.owner)
        expect(consumer.selection.getCurrent().view).toBe(receipt.view)
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          observedAddress
        )
        expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
          observedRevision
        )
        expect(consumer.draftAccount.isCurrent()).toBe(true)
        expect(
          await new Storage({ area: "local" }).get(
            "actorSettings:owned:" + sourceTarget.key
          )
        ).toEqual(originalScene.originalRecord)
        if (boundary === "current") {
          expect(normalizeChatSubmitResult(result)).toEqual({
            status: "submitted"
          })
          expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
          expect(useStoreMessageOption.getState().historyId).toBe(
            observedAddress!.id
          )
          expect((await savedMessages()).map((row) => row.history_id)).toEqual(
            Array(3).fill(observedAddress!.id)
          )
          expect(
            (await savedMessages())
              .filter((row) => row.role === "user")
              .map((row) => row.content)
          ).toEqual(["Original fresh Compare raw-ID question"])
          for (const wire of vi.mocked(tldwChat.streamMessage).mock.calls) {
            expect(wire[1]?.requestScope).toEqual(consumer.originalScope)
            expect(JSON.stringify(wireMessages(wire))).toContain(settings.notes)
          }
          const boundTarget = captureCurrentActorSceneTarget(
            consumer.selection
          )!
          expect(boundTarget.key).not.toBe(sourceTarget.key)
          expect(
            (await getActorSceneForChat({ target: boundTarget })).originalRecord
          ).toEqual(originalScene.originalRecord)
          expect(useActorStore.getState().applied?.target.key).toBe(
            boundTarget.key
          )
        } else {
          expect(replacementApplied).toBe(true)
          expect(replacement).toBeDefined()
          expect(tldwChat.streamMessage).not.toHaveBeenCalled()
          expect(normalizeChatSubmitResult(result).status).not.toBe("submitted")
          expect(useStoreMessageOption.getState().historyId).toBe(
            "replacement-Compare-raw-history"
          )
          expect(useStoreMessageOption.getState().serverChatId).toBe(
            replacement!.serverChatId
          )
          expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(
            true
          )
          expect(useStoreMessageOption.getState().serverChatTitle).toBe(
            "Replacement Compare metadata"
          )
          expect(useStoreMessageOption.getState().messages).toEqual(
            replacement!.messages
          )
          expect(useStoreMessageOption.getState().history).toEqual(
            replacement!.history
          )
          // The original shared question was already durably saved before H1
          // bound; retiring presentation does not erase that accepted write.
          expect(
            (await savedMessages()).map((row) => [row.role, row.content])
          ).toEqual([["user", "Original fresh Compare raw-ID question"]])
        }
      } finally {
        unsubscribe?.()
        consumer.unmount()
      }
    }
  )

  it.each(["current", "model", "tool", "effective settings"] as const)(
    "public native interrupted Retry retains entry intent across its real recovery read (%s)",
    async (boundary) => {
      const scoped = answerSnapshot()
      vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockResolvedValue(
        scoped
      )
      useStoreChatModelSettings.getState().setTemperature(0.24)
      const consumer = mountConsumer("full", scoped, {
        keepInterruptedTurns: true
      })
      const serverChatId = "native-public-Retry-entry"
      const question = "Original admitted native Retry question"
      const partial = "Original physically interrupted native answer"
      const rows: Array<{
        id: string
        role: string
        content: string
        parent_message_id: string | null
      }> = []
      // Declared native API I/O retains the actual request's view and admission
      // fields. The controller validates these responses and issues every H1
      // receipt; no receipt, admission helper, Scene or save helper is replaced.
      const capture = vi
        .spyOn(tldwClient, "captureHistorySelection")
        .mockImplementation(async (_id, request, options) => {
          expect(_id).toBe(serverChatId)
          expect(options?.requestScope).toEqual(consumer.originalScope)
          const view = { ...request.view, owner_key: "server-owner" }
          const nativeSnapshot = {
            version: 1 as const,
            owner_key: view.owner_key,
            conversation_id: serverChatId,
            nodes: rows.map((row) => ({
              id: row.id,
              role: row.role,
              parent_id: row.parent_message_id,
              revision: "1",
              settled: true
            })),
            fences: { conversation: "1", history: "1", settings: "1" },
            source_digest: "native-Retry-source",
            storage_context_digest: "native-Retry-storage",
            interpretation_status: { kind: "parent_graph_v1" as const }
          }
          const resolved = resolveHistorySelection(
            nativeSnapshot,
            view,
            request.purpose,
            ""
          )
          if (resolved.status !== "ready")
            throw new Error("Native Retry capture contract: " + resolved.code)
          return {
            status: "captured",
            purpose: request.purpose,
            view,
            snapshot: nativeSnapshot,
            rows: resolved.rows,
            selected_content: resolved.rows.map((node) => ({
              id: node.id,
              revision: node.revision,
              message: rows.find((row) => row.id === node.id)!.content,
              images: []
            })),
            storage_context_digest: nativeSnapshot.storage_context_digest
          } as never
        })
      vi.spyOn(tldwClient, "getChat").mockResolvedValue({
        id: serverChatId,
        version: 1,
        history_version: 1,
        title: "Original native Retry",
        character_id: null,
        assistant_id: null
      } as never)
      const append = vi
        .spyOn(tldwClient, "addChatMessage")
        .mockImplementation(async (_id, body, options) => {
          expect(_id).toBe(serverChatId)
          expect(options?.requestScope).toEqual(consumer.originalScope)
          rows.push({
            id: body.id!,
            role: body.role,
            content: body.content!,
            parent_message_id: body.parent_message_id ?? null
          })
          const selection = body.tldw_history_selection_v1
          return {
            id: body.id,
            ...(selection
              ? {
                  tldw_history_admission_v1: {
                    version: 1,
                    owner_key: "server-owner",
                    conversation_id: serverChatId,
                    input_message_id: body.id,
                    input_message_revision: "1",
                    selection_digest: selection.selection_digest,
                    messages: selection.messages,
                    originating_selection_revision: selection.selection_revision
                  }
                }
              : {})
          } as never
        })
      let release!: () => void
      let pending:
        | Promise<{ result?: ChatSubmitResult | void; error?: unknown }>
        | undefined
      let heldRows: Record<string, unknown>[] | undefined
      try {
        const accept = captureChatDraftLoad("switch")
        await act(async () => {
          expect(
            await consumer.selection.loadConversation(
              { serverChatId },
              null,
              (receipt) => {
                expect(accept(receipt)).toBe(true)
                const core = useStoreMessageOption.getState()
                core.setServerChatId(serverChatId)
                core.setServerChatMetaLoaded(true)
              }
            )
          ).toBe(true)
        })
        expect(consumer.receipts).toHaveLength(1)
        expect(consumer.receipts[0].owner.kind).toBe("native")
        const sourceTarget = captureCurrentActorSceneTarget(consumer.selection)!
        const settings = {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Original public native Retry Scene"
        }
        expect(
          await saveActorSettingsForChat({
            historyId: null,
            serverChatId,
            target: sourceTarget,
            expectedRecord: null,
            settings
          })
        ).toBe(true)
        const scene = await getActorSceneForChat({ target: sourceTarget })
        expect(scene.originalRecord).not.toBeNull()
        vi.mocked(tldwChat.streamMessage).mockImplementationOnce(
          async function* (_messages, options) {
            options.onDispatch?.()
            yield partial
            throw new Error("Declared native provider transport interruption")
          }
        )
        let seedResult!: ChatSubmitResult | void
        await act(async () => {
          seedResult = await consumer.mounted.result.current.onSubmit({
            message: question,
            image: ""
          })
        })
        expect(normalizeChatSubmitResult(seedResult)).toMatchObject({
          status: "failed",
          errorMessage: "Declared native provider transport interruption"
        })
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        expect(rows.map((row) => [row.role, row.content])).toEqual([
          ["user", question]
        ])
        await waitFor(() => {
          expect(consumer.selection.getCurrent().status).toBe("ready")
          expect(
            useStoreMessageOption.getState().messages.map((row) => row.message)
          ).toEqual([question, partial])
          expect(
            useStoreMessageOption.getState().messages.at(-1)?.generationInfo
          ).toMatchObject({ interrupted: true })
          expect(consumer.selection.recoveries).toHaveLength(1)
        })
        await act(async () => {
          await consumer.selection.refreshRecovery()
        })
        const original = useStoreMessageOption.getState()
        const selected = consumer.selection.getCurrent()
        const retained = await loadHistoryTurnRecoveries(
          selected.bookmarkScope!,
          selected.view!
        )
        expect(retained).toHaveLength(1)
        expect(retained[0].turn).toMatchObject({
          input_text: question,
          result_text: partial,
          outcome: "interrupted",
          admission: {
            owner_key: selected.view!.owner_key,
            conversation_id: serverChatId,
            input_message_id: rows[0].id
          }
        })
        expect(original.chatDraftAddress).toMatchObject({
          kind: "native",
          id: serverChatId
        })
        expect(selected.owner).toBe(consumer.receipts[0].owner)
        expect(selected.view?.cursor).toEqual({
          kind: "after_message",
          message_id: rows[0].id
        })
        const originalWire = vi.mocked(tldwChat.streamMessage).mock.calls[0]
        expect(originalWire[1]?.requestScope).toEqual(consumer.originalScope)
        expect(originalWire[1]?.preparedRequest).toMatchObject({
          model: "test",
          temperature: 0.24
        })
        expect(JSON.stringify(wireMessages(originalWire))).toContain(
          settings.notes
        )
        expect(original.selectedModel).toBe("test")
        expect(original.toolChoice).toBe("none")
        let enter!: () => void
        const entered = new Promise<void>((resolve) => {
          enter = resolve
        })
        const held = new Promise<void>((resolve) => {
          release = resolve
        })
        let armed = true
        const physicalWhere = storage.tables.historySelections.where
        vi.spyOn(storage.tables.historySelections, "where").mockImplementation(
          (field) => {
            const where = physicalWhere(field)
            return {
              equals: (value) => {
                const collection = where.equals(value)
                return {
                  ...collection,
                  toArray: async () => {
                    const result = await collection.toArray()
                    if (
                      armed &&
                      field === "owner_key" &&
                      value === selected.view!.owner_key
                    ) {
                      armed = false
                      heldRows = result
                      enter()
                      await held
                    }
                    return result
                  }
                }
              }
            }
          }
        )
        const captureOffset = capture.mock.calls.length
        const appendOffset = append.mock.calls.length
        vi.mocked(tldwChat.streamMessage).mockImplementation(
          async function* (_messages, options) {
            options.onDispatch?.()
            yield "Original native Retry final answer"
          }
        )
        act(() => {
          // This is Playground's public no-dispatch entry. It must retain T0
          // model/tool/settings even though its legitimate choose moves H1.
          pending = consumer.mounted.result.current
            .regenerateLastMessage!().then(
            (result) => ({ result }),
            (error) => ({ error })
          )
        })
        await Promise.race([
          entered,
          pending!.then((settled) => {
            throw new Error(
              "Native Retry missed its physical recovery gate: " +
                JSON.stringify(settled)
            )
          })
        ])
        expect(heldRows).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              owner_key: selected.view!.owner_key,
              conversation_id: serverChatId,
              pending_turns: expect.objectContaining({
                [retained[0].turn.operation_id]: retained[0].turn
              })
            })
          ])
        )
        expect(consumer.selection.getCurrent().owner).toBe(selected.owner)
        expect(consumer.selection.getCurrent().view).toBe(selected.view)
        expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
        await act(async () => {
          if (boundary === "model")
            useStoreMessageOption.getState().setSelectedModel("second-model")
          if (boundary === "tool")
            useStoreMessageOption.getState().setToolChoice("auto")
          if (boundary === "effective settings")
            useStoreChatModelSettings.getState().setTemperature(0.81)
        })
        expect(useStoreMessageOption.getState().selectedModel).toBe(
          boundary === "model" ? "second-model" : "test"
        )
        expect(useStoreMessageOption.getState().selectedModelRevision).toBe(
          original.selectedModelRevision + (boundary === "model" ? 1 : 0)
        )
        expect(useStoreMessageOption.getState().toolChoice).toBe(
          boundary === "tool" ? "auto" : "none"
        )
        expect(useStoreChatModelSettings.getState().temperature).toBe(
          boundary === "effective settings" ? 0.81 : 0.24
        )
        expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
          original.chatDraftAddress
        )
        expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
          original.chatDraftRevision
        )
        expect(consumer.selection.getCurrent().owner).toBe(selected.owner)
        expect(consumer.selection.getCurrent().view).toBe(selected.view)
        expect(consumer.draftAccount.isCurrent()).toBe(true)
        expect(
          selected.owner?.kind === "native" && selected.owner.validate_lease()
        ).toBe(true)
        let settled!: { result?: ChatSubmitResult | void; error?: unknown }
        await act(async () => {
          release()
          settled = await pending!
        })
        if (boundary === "current") {
          expect(settled.error).toBeUndefined()
          expect(normalizeChatSubmitResult(settled.result)).toEqual({
            status: "submitted"
          })
          expect(tldwChat.streamMessage).toHaveBeenCalledTimes(2)
          const retryWire = vi.mocked(tldwChat.streamMessage).mock.calls[1]
          expect(retryWire[1]?.requestScope).toEqual(consumer.originalScope)
          expect(retryWire[1]?.preparedRequest).toMatchObject({
            model: "test",
            temperature: 0.24,
            save_to_db: false
          })
          expect(JSON.stringify(wireMessages(retryWire))).toContain(
            settings.notes
          )
          expect(JSON.stringify(wireMessages(retryWire))).toContain(question)
          expect(
            capture.mock.calls
              .slice(captureOffset)
              .some(
                (call) =>
                  call[1].view.cursor.kind === "before_message" &&
                  call[1].view.cursor.message_id === rows[0].id
              )
          ).toBe(true)
          expect(
            rows.map((row) => [row.role, row.content, row.parent_message_id])
          ).toEqual([
            ["user", question, null],
            ["assistant", "Original native Retry final answer", rows[0].id]
          ])
          expect(
            append.mock.calls.slice(appendOffset).map((call) => call[1].role)
          ).toEqual(["assistant"])
          expect(
            await loadHistoryTurnRecoveries(
              selected.bookmarkScope!,
              selected.view!
            )
          ).toEqual([])
          expect(consumer.selection.getCurrent().view?.cursor).toEqual({
            kind: "after_message",
            message_id: rows[1].id
          })
          const currentScene = await getActorSceneForChat({
            target: captureCurrentActorSceneTarget(consumer.selection)!
          })
          expect(currentScene.originalRecord).toEqual(scene.originalRecord)
        } else {
          // The entry was retired before its own chooser or dismissal. A late
          // onSubmit refusal cannot recover the removed partial/recovery record.
          expect(capture.mock.calls).toHaveLength(captureOffset)
          expect(append.mock.calls).toHaveLength(appendOffset)
          expect(tldwChat.streamMessage).toHaveBeenCalledOnce()
          expect(consumer.selection.getCurrent().view).toBe(selected.view)
          expect(useStoreMessageOption.getState().messages).toEqual(
            original.messages
          )
          expect(useStoreMessageOption.getState().history).toEqual(
            original.history
          )
          expect(
            await loadHistoryTurnRecoveries(
              selected.bookmarkScope!,
              selected.view!
            )
          ).toEqual(retained)
          expect(rows.map((row) => [row.role, row.content])).toEqual([
            ["user", question]
          ])
        }
      } finally {
        release?.()
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )
})


it.each(["unsupported", "changed", "uncertain"])(
  "ordinary protected Edit keeps original graph on %s admission",
  async (outcome) => {
    const graph = [
      {
        id: "original-user",
        revision: "u1",
        parent_id: null,
        role: "user",
        settled: true,
        preview: "Original"
      },
      {
        id: "original-answer",
        revision: "a1",
        parent_id: "original-user",
        role: "assistant",
        settled: true,
        preview: "Original answer"
      }
    ]
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(
      async (
        _id,
        request: {
          tldw_history_selection_v1: {
            conversation_id: string
            selection_digest: string
          }
        }
      ) => {
        const view = { ...request.view, owner_key: "server-owner" }
        const state = {
          version: 1 as const,
          owner_key: "server-owner",
          conversation_id: newServerId,
          nodes: graph,
          fences: { conversation: "1", history: "1", settings: "1" },
          source_digest: "source",
          storage_context_digest: "storage",
          interpretation_status: { kind: "parent_graph_v1" as const }
        }
        const { resolveHistorySelection } = await import(
          "@/utils/history-selection"
        )
        const result = resolveHistorySelection(state, view, request.purpose, "")
        if (result.status !== "ready") throw new Error(result.code)
        return {
          status: "captured",
          view,
          snapshot: state,
          rows: result.rows,
          purpose: request.purpose,
          storage_context_digest: "storage",
          selected_content: result.rows.map((row) => ({
            id: row.id,
            revision: row.revision,
            message: graph.find((item) => item.id === row.id)!.preview,
            images: []
          }))
        } as never
      }
    )
    if (outcome === "unsupported")
      vi.mocked(tldwClient.getOpenAPISpec).mockResolvedValue({ paths: {} })
    const append = vi
      .spyOn(tldwClient, "addChatMessage")
      .mockImplementation(async (_id, body) => {
        const selection = body.tldw_history_selection_v1
        throw outcome === "uncertain"
          ? new Error("transport lost")
          : Object.assign(new Error("target changed"), {
              status: 409,
              details: {
                detail: {
                  status: "stale_selection",
                  code: "history_edit_target_changed",
                  conversation_id: selection.conversation_id,
                  message_id: "original-user",
                  message_revision: "u1",
                  selection_digest: selection.selection_digest
                }
              }
            })
      })
    const mounted = renderHook(() => useHistorySelection())
    const native: NativeHistoryOwnerV1 = {
      kind: "native",
      conversation_id: newServerId,
      request_scope: snapshot().requestScope,
      validate_lease: () => true
    }
    await act(async () => {
      expect(await mounted.result.current.open(native)).toBe(true)
    })
    const origin = mounted.result.current.getCurrent()
    const source = mounted.result.current.fence()
    const setMessages = vi.fn(),
      setHistory = vi.fn()
    const params: Parameters<typeof normalChatMode>[6] = {
      selectedModel: "test",
      selectedSystemPrompt: "",
      useOCR: false,
      currentChatModelSettings: {},
      toolChoice: "none",
      servicePromptSnapshot: snapshot(),
      historyId: null,
      serverChatId: newServerId,
      historySelection: {
        controller: mounted.result.current,
        originIsCurrent: source,
        editResend: {
          owner_key: origin.view!.owner_key,
          conversation_id: origin.view!.conversation_id,
          message_id: "original-user",
          message_revision: "u1",
          isCurrent: () => true,
          isOriginCurrent: source
        }
      },
      setHistoryId: () => {},
      setMessages,
      setHistory,
      setIsProcessing: vi.fn(),
      setStreaming: vi.fn(),
      setAbortController: vi.fn(),
      saveMessageOnSuccess,
      saveMessageOnError
    }
    try {
      let error: unknown
      await act(async () => {
        error = await normalChatMode(
          "  Edited\nExact proposal.  ",
          "",
          false,
          [],
          [],
          new AbortController().signal,
          params
        ).catch((value) => value)
      })
      expect(error).toMatchObject({
        status: "failed",
        errorMessage:
          outcome === "unsupported"
            ? "native_history_edit_unsupported"
            : outcome === "changed"
              ? "native_history_edit_target_changed"
              : "transport lost"
      })
      const kept = await loadHistoryTurnRecoveries(
        origin.bookmarkScope!,
        origin.view!
      )
      if (outcome === "unsupported") {
        expect(append).not.toHaveBeenCalled()
        expect(kept).toEqual([])
      } else {
        expect(append.mock.calls[0][1].tldw_history_edit_v1).toEqual({
          version: 1,
          message_id: "original-user",
          message_revision: "u1"
        })
        expect(kept[0]?.turn).toMatchObject({
          state: outcome === "changed" ? "refused" : "unknown",
          input_text: "  Edited\nExact proposal.  ",
          result_text: ""
        })
      }
      expect(tldwChat.streamMessage).not.toHaveBeenCalled()
      expect(setMessages).not.toHaveBeenCalled()
      expect(setHistory).not.toHaveBeenCalled()
      expect(graph.map((row) => row.preview)).toEqual([
        "Original",
        "Original answer"
      ])
    } finally {
      mounted.unmount()
    }
  }
)
