import "./native-plasmo-storage-fixture"
// @vitest-environment jsdom
import React from "react"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useOriginalSdkSceneFixture } from "./sdk-original-scene-fixture"
import { useChatActions as useRawChatActions } from "../useChatActions"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftLoad, captureChatDraftSource } from "../chat-draft-transition"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"

const sceneSelection = vi.hoisted(() => ({ controller: null as unknown as import("../useHistorySelection").HistorySelectionController | null }))

vi.mock("@/hooks/chat/useHistorySelection", async original => ({
  ...await original<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => sceneSelection.controller
}))

const {
  createChatMock,
  getChatMock,
  addChatMessageMock,
  normalChatModeMock,
  ragModeMock,
  streamCharacterChatCompletionMock,
  baseSaveMessageOnSuccessMock,
  syncChatSettingsForServerChatMock,
  getConfigMock,
  savePlaygroundSessionMock,
  buildChatSurfaceScopeKeyFromConfigMock,
  loadServicePromptSnapshotMock,
  releaseServicePromptSnapshotMock
} = vi.hoisted(() => ({
  createChatMock: vi.fn(),
  getChatMock: vi.fn(),
  addChatMessageMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  ragModeMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  baseSaveMessageOnSuccessMock: vi.fn(
    async (_payload?: unknown): Promise<string | null> => "history-persona"
  ),
  syncChatSettingsForServerChatMock: vi.fn(async () => null),
  getConfigMock: vi.fn(),
  savePlaygroundSessionMock: vi.fn(),
  buildChatSurfaceScopeKeyFromConfigMock: vi.fn(),
  loadServicePromptSnapshotMock: vi.fn(),
  releaseServicePromptSnapshotMock: vi.fn()
}))

vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  loadServicePromptSnapshot: loadServicePromptSnapshotMock
}))

vi.mock("@/hooks/chat-modes/normalChatMode", async original => ({
  ...await original<typeof import("@/hooks/chat-modes/normalChatMode")>(),
  normalChatMode: normalChatModeMock
}))

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: ragModeMock
}))

vi.mock("@/hooks/chat-modes/tabChatMode", () => ({
  tabChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/documentChatMode", () => ({
  documentChatMode: vi.fn()
}))

vi.mock("@/hooks/utils/messageHelpers", () => ({
  validateBeforeSubmit: vi.fn(() => true),
  createSaveMessageOnSuccess: vi.fn(() => baseSaveMessageOnSuccessMock),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-persona"
  )
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createNativeDeleteMessage: vi.fn(() => vi.fn(async () => false)),
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

vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>()
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

vi.mock("@/store/option", async original => {
  const actual = await original<typeof import("@/store/option")>()
  return {
    ...actual,
    useStoreMessageOption: {
      getState: actual.useStoreMessageOption.getState,
      setState: actual.useStoreMessageOption.setState
    }
  }
})

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: syncChatSettingsForServerChatMock
}))

vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: buildChatSurfaceScopeKeyFromConfigMock
}))

vi.mock("@/store/playground-session", async original => {
  const actual = await original<typeof import("@/store/playground-session")>()
  return { ...actual, usePlaygroundSessionStore: {
    ...actual.usePlaygroundSessionStore,
    getState: () => ({ ...actual.usePlaygroundSessionStore.getState(), saveSession: savePlaygroundSessionMock })
  } }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    getChat: getChatMock,
    addChatMessage: addChatMessageMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    initialize: vi.fn(async () => null),
    getConfig: getConfigMock
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
  ensureServerChatHistoryId: vi.fn(async (_id, _title, _signal, _snapshot, scene) => {
    if (scene) {
      if (!scene.acceptHistoryMirror?.("history-persona")) throw new Error("Original mirror source changed")
      useStoreMessageOption.getState().setHistoryId("history-persona", { preserveServerChatId: true })
    }
    return "history-persona"
  }),
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
  selectedAssistant: {
    kind: "persona" as const,
    id: "garden-helper",
    name: "Garden Helper",
    metadata: {
      selectionMode: "tracked"
    }
  },
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

const servicePromptSnapshot = {
  scopeKey: "scope:captured-user",
  requestScope: {
    config: {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user" as const,
      authSource: "manual" as const
    },
    userId: 42
  },
  capability: "supported" as const,
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  definitions: {
    "chat.rag.answer": {
      definition: { id: "chat.rag.answer", parts: [] },
      parts: { template: "Answer" },
      source: "packaged" as const,
      revision: null
    },
    "chat.rag.question_rewrite": {
      definition: { id: "chat.rag.question_rewrite", parts: [] },
      parts: { template: "Rewrite" },
      source: "packaged" as const,
      revision: null
    },
    "chat.web_search.answer": {
      definition: { id: "chat.web_search.answer", parts: [] },
      parts: { template: "Web answer" },
      source: "packaged" as const,
      revision: null
    }
  },
  release: releaseServicePromptSnapshotMock
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

let entrySnapshot = servicePromptSnapshot
const useChatActions = (options: Parameters<typeof useOriginalSdkSceneFixture>[0]) =>
  useOriginalSdkSceneFixture(options, () => entrySnapshot)

describe("useChatActions persona integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sceneSelection.controller = null
    entrySnapshot = servicePromptSnapshot
    syncChatSettingsForServerChatMock.mockResolvedValue(null)
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      authSource: "manual"
    })
    buildChatSurfaceScopeKeyFromConfigMock.mockReturnValue("scope:chat")
    loadServicePromptSnapshotMock.mockResolvedValue(servicePromptSnapshot)
    createChatMock.mockResolvedValue({
      id: "persona-chat-1",
      title: "Persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only"
    })
    getChatMock.mockResolvedValue({
      id: "workspace-existing-chat",
      scope_type: "workspace",
      workspace_id: "workspace-plain"
    })
    addChatMessageMock.mockImplementation(async (_chatId, payload) => ({
      id: `server-message-${payload?.role ?? "message"}`,
      version: 1
    }))
    normalChatModeMock.mockResolvedValue(undefined)
  })

  it.each([false, true])("creates a persona-backed chat with assistant_kind=persona (temporary=%s)", async temporaryChat => {
    const options = { ...createHookOptions(), temporaryChat }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: expect.any(AbortSignal) })
    expect(options.setServerChatId).toHaveBeenCalledWith("persona-chat-1", { preserveTemporaryChat: temporaryChat })
    expect(useStoreMessageOption.getState().temporaryChat).toBe(temporaryChat)
    expect(normalChatModeMock).toHaveBeenCalledOnce()
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith("garden-helper")
    expect(options.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      "read_only"
    )
    expect(
      options.setServerChatId.mock.invocationCallOrder[0]
    ).toBeLessThan(
      options.setServerChatAssistantKind.mock.invocationCallOrder.at(-1) ?? 0
    )
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(true)
    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "persona-chat-1",
        trackedAssistantKind: "persona",
        trackedAssistantId: "garden-helper",
        trackedCharacterId: null,
        trackedAssistantDisplayName: "Garden Helper",
        trackedAssistantAvatarUrl: null,
        serverChatPersonaMemoryMode: "read_only",
        scopeKey: temporaryChat ? "scope:chat" : servicePromptSnapshot.scopeKey,
        trackedAssistantSelection: expect.objectContaining({
          kind: "persona",
          id: "garden-helper",
          name: "Garden Helper",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      })
    )
    expect(
      savePlaygroundSessionMock.mock.invocationCallOrder[0]
    ).toBeLessThan(normalChatModeMock.mock.invocationCallOrder[0])
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        historyId: "history-persona",
        serverChatId: "persona-chat-1"
      })
    )
  })

  it.each(["current", "new", "same-id", "account", "stop"])(
    "retains exact persona creation ownership across %s",
    async (boundary) => {
      const created = deferred<{
        id: string
        assistant_kind: string
        assistant_id: string
      }>()
      createChatMock.mockReturnValueOnce(created.promise)
      const owner = new AbortController()
      const snapshot = {
        ...servicePromptSnapshot,
        scopeInvalidatedSignal: owner.signal
      }
      entrySnapshot = snapshot
      loadServicePromptSnapshotMock.mockResolvedValueOnce(snapshot)
      const options = {
        ...createHookOptions(),
        historyId: null,
        serverChatId: null
      }
      const hook = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )
      const controller = new AbortController()
      let pending!: Promise<unknown>
      act(() => {
        pending = hook.result.current.onSubmit({
          message: "Original persona question",
          image: "",
          controller
        })
      })
      try {
        await vi.waitFor(() => expect(createChatMock).toHaveBeenCalledOnce())
        if (boundary === "new" || boundary === "same-id") {
          useStoreMessageOption.getState().startNewChatDraft(false)
          if (boundary === "same-id")
            useStoreMessageOption.setState({
              historyId: null,
              serverChatId: null
            })
        }
        if (boundary === "account") owner.abort()
        if (boundary === "stop") controller.abort()
        await act(async () => {
          created.resolve({
            id: "owned-persona-created",
            assistant_kind: "persona",
            assistant_id: "garden-helper"
          })
          await pending
        })
        if (boundary === "current") {
          expect(normalChatModeMock).toHaveBeenCalledOnce()
          expect(useStoreMessageOption.getState()).toMatchObject({
            historyId: "history-persona",
            serverChatId: "owned-persona-created"
          })
          expect(
            normalChatModeMock.mock.calls[0][6].actorSceneSend.isCurrent()
          ).toBe(true)
        } else {
          expect(normalChatModeMock).not.toHaveBeenCalled()
          expect(options.setServerChatId).not.toHaveBeenCalled()
          expect(options.setServerChatMetaLoaded).not.toHaveBeenCalled()
        }
      } finally {
        created.resolve({
          id: "owned-persona-created",
          assistant_kind: "persona",
          assistant_id: "garden-helper"
        })
        await pending
        hook.unmount()
      }
    }
  )

  it.each(["current", "new", "same-id"])(
    "an existing Persona owns its first mirror publication across %s",
    async (boundary) => {
      const linking = deferred<void>()
      const options = {
        ...createHookOptions(),
        historyId: null,
        serverChatId: "existing-persona",
        serverChatMetaLoaded: true,
        serverChatAssistantKind: "persona" as const,
        serverChatAssistantId: "garden-helper"
      }
      options.ensureServerChatHistoryId.mockImplementationOnce(
        async (_id, _title, signal, _snapshot, scene) => {
          await linking.promise
          signal?.throwIfAborted()
          if (scene && !scene.acceptHistoryMirror?.("first-persona-mirror"))
            throw new Error("Original mirror source changed")
          // The real legacy helper publishes too; omitting the fifth argument must not hide it.
          useStoreMessageOption
            .getState()
            .setHistoryId("first-persona-mirror", { preserveServerChatId: true })
          return "first-persona-mirror"
        }
      )
      const hook = renderHook(() => useChatActions(options))
      let pending!: Promise<unknown>
      act(() => {
        pending = hook.result.current.onSubmit({
          message: "Existing Persona question",
          image: ""
        })
      })
      try {
        await vi.waitFor(() =>
          expect(options.ensureServerChatHistoryId).toHaveBeenCalledOnce()
        )
        if (boundary !== "current") {
          useStoreMessageOption.getState().startNewChatDraft(false)
          if (boundary === "same-id")
            useStoreMessageOption.setState({
              historyId: null,
              serverChatId: "existing-persona"
            })
        }
        await act(async () => {
          linking.resolve()
          await pending
        })
        expect(createChatMock).not.toHaveBeenCalled()
        if (boundary === "current") {
          expect(normalChatModeMock).toHaveBeenCalledOnce()
          expect(normalChatModeMock.mock.calls[0][6]).toMatchObject({
            historyId: "first-persona-mirror",
            serverChatId: "existing-persona"
          })
          expect(
            normalChatModeMock.mock.calls[0][6].actorSceneSend.isCurrent()
          ).toBe(true)
        } else {
          expect(normalChatModeMock).not.toHaveBeenCalled()
          expect(useStoreMessageOption.getState().historyId).toBeNull()
        }
      } finally {
        linking.resolve()
        await pending
        hook.unmount()
      }
    }
  )

  it.each([false, true])(
    "Temporary persona session configuration retains its original owner (retired=%s)",
    async (retired) => {
      const configuration = deferred<{ serverUrl: string; authMode: string }>()
      getConfigMock.mockReturnValueOnce(configuration.promise)
      const options = {
        ...createHookOptions(),
        temporaryChat: true,
        historyId: null,
        serverChatId: null
      }
      const hook = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )
      let pending!: Promise<unknown>
      act(() => {
        pending = hook.result.current.onSubmit({
          message: "Temporary persona question",
          image: ""
        })
      })
      try {
        await vi.waitFor(() => expect(getConfigMock).toHaveBeenCalledOnce())
        expect(loadServicePromptSnapshotMock).not.toHaveBeenCalled()
        if (retired) useStoreMessageOption.getState().startNewChatDraft(true)
        await act(async () => {
          configuration.resolve({
            serverUrl: "http://127.0.0.1:8000",
            authMode: "multi-user"
          })
          await pending
        })
        expect(savePlaygroundSessionMock).toHaveBeenCalledTimes(retired ? 0 : 1)
        expect(normalChatModeMock).toHaveBeenCalledTimes(retired ? 0 : 1)
        expect(createChatMock.mock.calls[0][1]).toMatchObject({
          requestScope: servicePromptSnapshot.requestScope,
          signal: expect.any(AbortSignal)
        })
      } finally {
        configuration.resolve({
          serverUrl: "http://127.0.0.1:8000",
          authMode: "multi-user"
        })
        await pending
        hook.unmount()
      }
    }
  )

  it.each(["turn", "reply"])(
    "captures the tracked %s owner before mirror linking and releases it",
    async (operation) => {
      const pending = deferred<typeof servicePromptSnapshot>()
      loadServicePromptSnapshotMock.mockReturnValueOnce(pending.promise)
      const options = {
        ...createHookOptions(),
        serverChatId: "persona-chat-1",
        serverChatMetaLoaded: true,
        serverChatAssistantKind: "persona",
        serverChatAssistantId: "garden-helper",
        compareFeatureEnabled: true
      }
      if (operation === "reply") {
        // This declares only a private Scene locator, not H1 send/admission authority.
        const account = {
          requestScope: servicePromptSnapshot.requestScope,
          isCurrent: () => !servicePromptSnapshot.scopeInvalidatedSignal.aborted
        }
        const ownerKey = chatDraftOwnerKey(account.requestScope)
        const owner = {
          kind: "native" as const,
          owner_key: ownerKey,
          conversation_id: "persona-chat-1",
          request_scope: account.requestScope,
          scope: { type: "global" as const },
          validate_lease: account.isCurrent
        }
        const view = { owner_key: ownerKey, conversation_id: "persona-chat-1" }
        const store = useStoreMessageOption
        store.getState().startNewChatDraft(false)
        store.setState({
          historyId: options.historyId,
          serverChatId: options.serverChatId,
          selectedModel: options.selectedModel,
          toolChoice: options.toolChoice
        })
        expect(
          captureChatDraftLoad("switch")({ owner, view } as Parameters<
            ReturnType<typeof captureChatDraftLoad>
          >[0])
        ).toBe(true)
        const source = captureChatDraftSource(account.isCurrent)
        sceneSelection.controller = {
          draftAccount: account,
          getCurrent: () => ({ owner, view, status: "ready" }),
          fence: () => source.isCurrent
        } as unknown as import("../useHistorySelection").HistorySelectionController
      }
      const hook = renderHook(() =>
        operation === "turn"
          ? useChatActions(
              options as unknown as Parameters<typeof useChatActions>[0]
            )
          : useRawChatActions(
              options as unknown as Parameters<typeof useRawChatActions>[0]
            )
      )
      let request!: Promise<unknown>
      act(() => {
        request =
          operation === "turn"
            ? hook.result.current.onSubmit({
                message: "Owned follow-up",
                image: ""
              })
            : hook.result.current.sendPerModelReply({
                clusterId: "cluster",
                modelId: "openai:gpt-4.1",
                message: "Owned reply"
              })
      })
      try {
        await vi.waitFor(() =>
          expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1)
        )
        expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled()
        await act(async () => {
          pending.resolve(servicePromptSnapshot)
          await request
        })
        expect(
          options.ensureServerChatHistoryId.mock.calls[0].slice(0, 4)
        ).toEqual([
          "persona-chat-1",
          options.serverChatTitle || undefined,
          servicePromptSnapshot.scopeInvalidatedSignal,
          servicePromptSnapshot
        ])
        expect(servicePromptSnapshot.release).toHaveBeenCalledTimes(1)
      } finally {
        pending.resolve(servicePromptSnapshot)
        await request
        hook.unmount()
        sceneSelection.controller = null
      }
    }
  )

  it("persists the canonical captured scope for a prompt-backed persona turn", async () => {
    const options = {
      ...createHookOptions(),
      webSearch: true
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Search with this persona",
        image: ""
      })
    })

    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({ scopeKey: "scope:captured-user" })
    )
    expect(buildChatSurfaceScopeKeyFromConfigMock).not.toHaveBeenCalled()
    expect(releaseServicePromptSnapshotMock).toHaveBeenCalledTimes(1)
  })

  it("passes workspace scope through when creating a persona-backed chat", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const options = {
      ...createHookOptions(),
      scope
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello scoped persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      {
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        state: "in-progress",
        topic_label: undefined,
        cluster_id: undefined,
        source: undefined,
        external_ref: undefined
      },
      { scope, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal }
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello scoped persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "persona-chat-1"
      })
    )
  })

  it("routes an inherited workspace persona default when no explicit assistant is selected", async () => {
    const inheritedWorkspaceAssistant = {
      kind: "persona" as const,
      id: "workspace-helper",
      name: "Workspace Helper",
      metadata: {
        selectionMode: "tracked",
        source: "workspace",
        personaMemoryMode: "read_write"
      }
    }
    const options = {
      ...createHookOptions(),
      selectedAssistant: inheritedWorkspaceAssistant,
      inheritedAssistant: inheritedWorkspaceAssistant,
      inheritedPersonaMemoryMode: "read_write" as const
    }
    createChatMock.mockResolvedValueOnce({
      id: "workspace-persona-chat",
      title: "Workspace persona chat",
      assistant_kind: "persona",
      assistant_id: "workspace-helper",
      persona_memory_mode: "read_write"
    })
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use workspace persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "workspace-helper",
      persona_memory_mode: "read_write",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal })
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(
      "workspace-helper"
    )
    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({
        serverChatId: "workspace-persona-chat",
        trackedAssistantKind: "persona",
        trackedAssistantId: "workspace-helper",
        trackedAssistantDisplayName: "Workspace Helper",
        serverChatPersonaMemoryMode: "read_write"
      })
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Use workspace persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Workspace Helper",
          avatarUrl: undefined
        },
        serverChatId: "workspace-persona-chat"
      })
    )
  })

  it("creates a workspace-scoped server chat for plain normal sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-plain" } as const
    createChatMock.mockResolvedValueOnce({
      id: "workspace-plain-chat",
      title: "Workspace plain chat",
      state: "in-progress",
      version: 3
    })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        historyId: string | null
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
        serverChatId?: string | null
      }
      expect(params.historyId).toBe("history-persona")
      expect(params.serverChatId).toBe("workspace-plain-chat")
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Plain workspace hello",
        image: "",
        fullText: "Plain workspace reply",
        source: []
      })
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Plain workspace hello",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({
        scope
      })
    )
    expect(options.setServerChatId).toHaveBeenCalledWith("workspace-plain-chat")
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(null)
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Plain workspace hello",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "workspace-plain-chat"
      })
    )
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "workspace-plain-chat"
      })
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "workspace-plain-chat",
      expect.objectContaining({
        role: "user",
        content: "Plain workspace hello"
      }),
      { scope, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal }
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "workspace-plain-chat",
      expect.objectContaining({
        role: "assistant",
        content: "Plain workspace reply"
      }),
      { scope, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal }
    )
  })

  it("adopts the first persisted plain chat so server message actions become available", async () => {
    createChatMock.mockResolvedValueOnce({ id: "first-saved-chat", title: "Hello" })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as { saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null> }
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: null, selectedModel: "deepseek-chat", message: "Hello", image: "",
        fullText: "Saved answer", source: [], saveToDb: true, conversationId: "first-saved-chat"
      })
    })
    const options = {
      ...createHookOptions(), serverChatId: null, serverChatTitle: null,
      serverChatAssistantKind: null, serverChatAssistantId: null,
      serverChatCharacterId: null, selectedAssistant: null, historyId: null
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    await act(async () => { await result.current.onSubmit({ message: "Hello", image: "" }) })
    expect(options.setServerChatId).toHaveBeenCalledWith("first-saved-chat")
  })

  it("rejects a stale server chat id from another scope before workspace sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-fresh" } as const
    getChatMock.mockResolvedValueOnce({
      id: "stale-global-chat",
      scope_type: "global",
      workspace_id: null
    })
    createChatMock.mockResolvedValueOnce({
      id: "workspace-fresh-chat",
      title: "Workspace fresh chat",
      state: "in-progress"
    })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        serverChatId?: string | null
        conversationId?: string | null
      }
      expect(params.serverChatId).toBe("workspace-fresh-chat")
      expect(params.conversationId).toBe("workspace-fresh-chat")
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: "stale-global-chat",
      serverChatTitle: "Stale global chat",
      serverChatMetaLoaded: true,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use the active workspace",
        image: ""
      })
    })

    expect(getChatMock).toHaveBeenCalledWith("stale-global-chat", { scope, requestScope: servicePromptSnapshot.requestScope, signal: expect.any(AbortSignal) })
    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({ scope })
    )
    expect(options.setServerChatId).toHaveBeenLastCalledWith(
      "workspace-fresh-chat"
    )
  })

  it("creates a workspace-scoped server chat for staged RAG sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-rag" } as const
    createChatMock.mockResolvedValueOnce({
      id: "workspace-rag-chat",
      title: "Workspace RAG chat",
      state: "in-progress"
    })
    ragModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        historyId: string | null
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
        serverChatId?: string | null
      }
      expect(params.historyId).toBe("history-persona")
      expect(params.serverChatId).toBe("workspace-rag-chat")
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Use staged source",
        image: "",
        fullText: "RAG reply",
        source: [],
        saveToDb: false
      })
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use staged source",
        image: "",
        requestOverrides: {
          fileRetrievalEnabled: true,
          ragMediaIds: [101]
        }
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({
        scope,
        requestScope: servicePromptSnapshot.requestScope,
        signal: servicePromptSnapshot.scopeSignal
      })
    )
    expect(ragModeMock).toHaveBeenCalledWith(
      "Use staged source",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "workspace-rag-chat",
        ragMediaIds: [101]
      })
    )
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "workspace-rag-chat"
      })
    )
  })

  it("does not publish workspace chat metadata when scope changes during history linking", async () => {
    const scope = {
      type: "workspace",
      workspaceId: "workspace-scope-race"
    } as const
    const scopeController = new AbortController()
    const scopedSnapshot = {
      ...servicePromptSnapshot,
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: scopeController.signal
    }
    entrySnapshot = scopedSnapshot
    loadServicePromptSnapshotMock.mockResolvedValueOnce(scopedSnapshot)
    createChatMock.mockResolvedValueOnce({
      id: "workspace-scope-race-chat",
      title: "Workspace scope race",
      state: "resolved",
      version: 5,
      topic_label: "Race topic",
      cluster_id: "race-cluster",
      source: "workspace",
      external_ref: "race-ref"
    })
    const historyLink = deferred<string | null>()
    const options = {
      ...createHookOptions(),
      scope,
      selectedAssistant: null,
      ensureServerChatHistoryId: vi.fn(() => historyLink.promise)
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    let submission!: ReturnType<typeof result.current.onSubmit>
    act(() => {
      submission = result.current.onSubmit({
        message: "Keep workspace state scoped",
        image: "",
        requestOverrides: {
          fileRetrievalEnabled: true,
          ragMediaIds: [101]
        }
      })
    })
    await vi.waitFor(() =>
      expect(options.ensureServerChatHistoryId).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    historyLink.resolve("history-stale")

    await act(async () => {
      await submission
    })

    expect(options.ensureServerChatHistoryId).toHaveBeenCalledWith(
      "workspace-scope-race-chat",
      "Workspace scope race",
      scopeController.signal,
      scopedSnapshot,
      expect.objectContaining({ acceptHistoryMirror: expect.any(Function), isCurrent: expect.any(Function) })
    )
    expect(options.setServerChatId).not.toHaveBeenCalled()
    expect(options.setServerChatTitle).not.toHaveBeenCalled()
    expect(options.setServerChatCharacterId).not.toHaveBeenCalled()
    expect(options.setServerChatAssistantKind).not.toHaveBeenCalled()
    expect(options.setServerChatAssistantId).not.toHaveBeenCalled()
    expect(options.setServerChatPersonaMemoryMode).not.toHaveBeenCalled()
    expect(options.setServerChatMetaLoaded).not.toHaveBeenCalled()
    expect(options.setServerChatState).not.toHaveBeenCalled()
    expect(options.setServerChatVersion).not.toHaveBeenCalled()
    expect(options.setServerChatTopic).not.toHaveBeenCalled()
    expect(options.setServerChatClusterId).not.toHaveBeenCalled()
    expect(options.setServerChatSource).not.toHaveBeenCalled()
    expect(options.setServerChatExternalRef).not.toHaveBeenCalled()
    expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    expect(ragModeMock).not.toHaveBeenCalled()
  })

  it("reuses an existing global conversation through inference and local persistence", async () => {
    let capturedParams:
      | {
          conversationId?: string | null
          serverChatId?: string | null
        }
      | null = null
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        conversationId?: string | null
        saveMessageOnSuccess: (
          payload: Record<string, unknown>
        ) => Promise<string | null>
        serverChatId?: string | null
      }
      capturedParams = params
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Plain global hello",
        image: "",
        fullText: "Plain global reply",
        source: []
      })
    })
    const options = {
      ...createHookOptions(),
      selectedAssistant: null,
      serverChatId: "existing-global-chat",
      serverChatTitle: "Existing global chat"
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Plain global hello",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(capturedParams?.serverChatId).toBe("existing-global-chat")
    expect(capturedParams?.conversationId).toBe("existing-global-chat")
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "existing-global-chat"
      })
    )
  })

  it("does not forward stale character state into the first persona send", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "character-chat-1",
      serverChatTitle: "Old character chat",
      serverChatCharacterId: 42,
      serverChatAssistantKind: "character" as const,
      serverChatAssistantId: "42",
      serverChatPersonaMemoryMode: "read_write" as const,
      serverChatMetaLoaded: true,
      serverChatState: "resolved" as const,
      serverChatTopic: "Old topic",
      serverChatClusterId: "old-cluster",
      serverChatSource: "old-source",
      serverChatExternalRef: "old-ref"
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(null)
    expect(options.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(null)
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(options.setServerChatTitle).toHaveBeenCalledWith(null)
    expect(options.setServerChatState).toHaveBeenCalledWith("in-progress")
    expect(options.setServerChatVersion).toHaveBeenCalledWith(null)
    expect(options.setServerChatTopic).toHaveBeenCalledWith(null)
    expect(options.setServerChatClusterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatSource).toHaveBeenCalledWith(null)
    expect(options.setServerChatExternalRef).toHaveBeenCalledWith(null)
    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal })
    expect(options.setServerChatAssistantKind).toHaveBeenLastCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenLastCalledWith(
      "garden-helper"
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        serverChatId: "persona-chat-1"
      })
    )
  })

  it("keeps persona routing ahead of character fallback when persona chats carry a character id", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "persona-chat-existing",
      serverChatAssistantKind: "persona" as const,
      serverChatAssistantId: "garden-helper",
      serverChatCharacterId: "char-shadow",
      selectedCharacter: {
        id: "char-stale",
        name: "Stale Character",
        system_prompt: "Stale prompt"
      }
    }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Stay persona",
        image: ""
      })
    })

    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: expect.anything()
      })
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Stay persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        serverChatId: "persona-chat-existing"
      })
    )
  })

  it("preserves tracked persona server linkage when local history is assigned", async () => {
    const setHistoryId = vi.fn()
    const options = {
      ...createHookOptions(),
      historyId: null,
      setHistoryId,
      serverChatId: null
    }
    baseSaveMessageOnSuccessMock.mockImplementationOnce(
      async (payload?: { setHistoryId?: (id: string) => void }) => {
        payload?.setHistoryId?.("history-persona")
        return "history-persona"
      }
    )
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        servicePromptSnapshot?: ServicePromptSnapshot
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
      }
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: null,
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Hello persona",
        image: "",
        fullText: "Persona reply",
        source: [],
        assistantMessageId: "assistant-persona-1",
        reasoning_time_taken: 0,
        conversationId: "persona-chat-1"
      })
    })

    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(setHistoryId).toHaveBeenCalledWith("history-persona", {
      preserveServerChatId: true
    })
  })
})
