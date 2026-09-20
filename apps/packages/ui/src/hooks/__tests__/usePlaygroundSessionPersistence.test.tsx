// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  scopeSignal: new AbortController(),
  loadSnapshot: vi.fn(),
  assistantLoading: false,
  getConfig: vi.fn(),
  getFullChatData: vi.fn(),
  getPromptById: vi.fn(),
  setSystemPrompt: vi.fn(),
  setSelectedAssistant: vi.fn()
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => mocks.loadSnapshot(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig
  }
}))

vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: vi.fn(() => []),
  formatToMessage: vi.fn(() => []),
  getFullChatData: (...args: unknown[]) => mocks.getFullChatData(...args),
  getPromptById: (...args: unknown[]) => mocks.getPromptById(...args)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({
    setSystemPrompt: mocks.setSystemPrompt
  })
}))

vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: () => "global"
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    serverUrl: "http://127.0.0.1:8000",
    lastConfigUpdatedAt: 0
  })
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, mocks.setSelectedAssistant, { isLoading: mocks.assistantLoading }]
}))

import { usePlaygroundSessionPersistence } from "../usePlaygroundSessionPersistence"
import { useSelectServerChat } from "../chat/useSelectServerChat"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { buildQueuedRequest } from "@/utils/chat-request-queue"

describe("usePlaygroundSessionPersistence", () => {
  beforeEach(() => {
    mocks.scopeSignal = new AbortController()
    mocks.assistantLoading = false
    localStorage.clear()
    vi.clearAllMocks()
    mocks.loadSnapshot.mockReset().mockImplementation(async () => ({
      requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "bob" },
      scopeSignal: mocks.scopeSignal.signal, scopeInvalidatedSignal: mocks.scopeSignal.signal,
      release: vi.fn()
    }))
    mocks.getConfig.mockResolvedValue(null)
    mocks.getFullChatData.mockResolvedValue(null)
    mocks.setSelectedAssistant.mockReset()
    useStoreMessageOption.setState({
      history: [],
      messages: [],
      historyId: null,
      serverChatId: null,
      serverChatTitle: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatCharacterId: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
      fileRetrievalEnabled: false,
      ragMediaIds: null,
      ragSearchMode: "hybrid",
      ragTopK: null,
      ragEnableGeneration: true,
      ragEnableCitations: true,
      queuedMessages: [],
      temporaryChat: false
    })
    usePlaygroundSessionStore.getState().clearSession()
  })

  it("preserves a saved selection and queue when identity verification is unavailable, then retries", async () => {
    const saved = { historyId: "bob-history", serverChatId: "bob-chat", scopeKey: "global",
      queuedMessages: [buildQueuedRequest({ id: "queued-bob", clientRequestId: "request-bob", message: "Private queued question" })] }
    usePlaygroundSessionStore.getState().saveSession(saved)
    mocks.loadSnapshot.mockRejectedValueOnce(new TypeError("Failed to fetch"))
    mocks.getFullChatData.mockResolvedValue({ historyInfo: {
      id: saved.historyId, server_chat_id: saved.serverChatId,
      server_scope_key: '["http://chat.test","multi-user","manual",null,"bob",null]'
    }, messages: [] })
    const view = renderHook(() => usePlaygroundSessionPersistence())
    await waitFor(() => expect(view.result.current.sessionScopeReady).toBe(true))
    await act(async () => { expect(await view.result.current.restoreSession()).toBe("cancelled") })
    expect(usePlaygroundSessionStore.getState()).toMatchObject(saved)
    expect(useStoreMessageOption.getState()).toMatchObject({ historyId: null, messages: [], queuedMessages: [] })
    await act(async () => { expect(await view.result.current.restoreSession()).toBe("restored") })
    expect(useStoreMessageOption.getState()).toMatchObject({ historyId: saved.historyId, serverChatId: saved.serverChatId })
    view.unmount()
  })

  it("does not clear a replacement session when an old identity lookup rejects", async () => {
    let reject!: (error: Error) => void
    mocks.loadSnapshot.mockImplementationOnce(() => new Promise((_, rejectPromise) => { reject = rejectPromise }))
    usePlaygroundSessionStore.getState().saveSession({ historyId: "old-history", serverChatId: "old-chat", scopeKey: "global" })
    const view = renderHook(() => usePlaygroundSessionPersistence())
    await waitFor(() => expect(view.result.current.sessionScopeReady).toBe(true))
    let pending!: ReturnType<typeof view.result.current.restoreSession>
    act(() => { pending = view.result.current.restoreSession() })
    await waitFor(() => expect(mocks.loadSnapshot).toHaveBeenCalled())
    act(() => {
      usePlaygroundSessionStore.getState().cancelPendingRestore()
      usePlaygroundSessionStore.getState().saveSession({ historyId: "new-history", serverChatId: "new-chat", scopeKey: "global" })
    })
    await act(async () => { reject(new TypeError("Failed to fetch")); expect(await pending).toBe("cancelled") })
    expect(usePlaygroundSessionStore.getState()).toMatchObject({ historyId: "new-history", serverChatId: "new-chat" })
    view.unmount()
  })

  it.each(['["http://chat.test","multi-user","manual",null,"alice",null]', undefined])("does not trust a saved selection pointing to a foreign or unowned cache (%s)", async owner => {
    mocks.getFullChatData.mockResolvedValue({ historyInfo: {
      id: "alice-history", title: "ALICE PRIVATE TITLE", server_chat_id: "alice-chat",
      server_scope_key: owner, last_used_prompt: { prompt_content: "ALICE SECRET PROMPT" }
    }, messages: [{ role: "user", content: "ALICE PRIVATE TRANSCRIPT" }] })
    usePlaygroundSessionStore.getState().saveSession({ historyId: "alice-history", serverChatId: "alice-chat", scopeKey: "global" })
    const view = renderHook(() => usePlaygroundSessionPersistence())
    await waitFor(() => expect(view.result.current.sessionScopeReady).toBe(true))
    await act(async () => { await view.result.current.restoreSession() })
    expect(useStoreMessageOption.getState()).toMatchObject({ historyId: null, serverChatId: null, serverChatTitle: null, messages: [], history: [] })
    expect(mocks.setSystemPrompt).not.toHaveBeenCalled()
    view.unmount()
  })

  it("waits for the assistant account before restoring a saved conversation", async () => {
    mocks.assistantLoading = true
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null, serverChatId: "saved-chat", serverChatTitle: "Saved",
      chatMode: "normal", scopeKey: "global"
    })
    const view = renderHook(() => usePlaygroundSessionPersistence())
    await act(async () => { await Promise.resolve() })
    expect(view.result.current.sessionScopeReady).toBe(false)
    await expect(view.result.current.restoreSession()).resolves.toBe("cancelled")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    mocks.assistantLoading = false
    view.rerender()
    await waitFor(() => expect(view.result.current.sessionScopeReady).toBe(true))
    await act(async () => { await expect(view.result.current.restoreSession()).resolves.toBe("restored") })
    expect(useStoreMessageOption.getState().serverChatId).toBe("saved-chat")
  })

  it("does not stamp a pending private session save with the next account on unmount", async () => {
    useStoreMessageOption.setState({ serverChatId: "alice-private-chat", serverChatTitle: "Alice title" })
    const view = renderHook(() => usePlaygroundSessionPersistence())
    await waitFor(() => expect(view.result.current.sessionScopeReady).toBe(true))
    let resolveConfig!: (config: null) => void
    mocks.getConfig.mockImplementation(() => new Promise(resolve => { resolveConfig = resolve }))
    view.unmount()
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    await act(async () => { resolveConfig(null) })
    expect(usePlaygroundSessionStore.getState().serverChatId).toBeNull()
  })

  it("restores a persisted server-backed character chat even without local Dexie history", async () => {
    useStoreMessageOption.setState({
      historyId: "stale-local-history",
      history: [{ role: "user", content: "stale local message" }],
      messages: [
        {
          isBot: false,
          name: "User",
          role: "user",
          message: "stale local message",
          sources: []
        }
      ]
    })
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "character-chat-42",
      trackedAssistantSelection: {
        kind: "character",
        id: "char-42",
        name: "Captain Redwood",
        metadata: {
          selectionMode: "tracked"
        }
      },
      trackedAssistantKind: "character",
      trackedAssistantId: "char-42",
      trackedCharacterId: "char-42",
      trackedAssistantDisplayName: "Captain Redwood",
      trackedAssistantAvatarUrl: null,
      serverChatPersonaMemoryMode: null,
      scopeKey: "global",
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
      ragMediaIds: null,
      ragSearchMode: "hybrid",
      ragTopK: null,
      ragEnableGeneration: true,
      ragEnableCitations: true,
      queuedMessages: []
    })

    const { result } = renderHook(() => usePlaygroundSessionPersistence())

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
    })

    await expect(result.current.restoreSession()).resolves.toBe("restored")

    await waitFor(() => {
      expect(useStoreMessageOption.getState().serverChatId).toBe(
        "character-chat-42"
      )
      expect(useStoreMessageOption.getState().serverChatAssistantKind).toBe(
        "character"
      )
      expect(useStoreMessageOption.getState().serverChatCharacterId).toBe(
        "char-42"
      )
      expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(false)
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        {
          kind: "character",
          id: "char-42",
          name: "Captain Redwood",
          metadata: {
            selectionMode: "tracked"
          }
        },
        expect.objectContaining({ isCurrent: expect.any(Function) })
      )
    })
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().history).toEqual([])
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(mocks.getFullChatData).not.toHaveBeenCalled()
  })

  it.each([
    { cachedServerId: "persona-chat-7", expectedTitle: "Tracked persona chat" },
    { cachedServerId: "different-chat", expectedTitle: null }
  ])("restores matching cached title while leaving server metadata pending ($cachedServerId)", async ({ cachedServerId, expectedTitle }) => {
    mocks.getFullChatData.mockResolvedValue({
      historyInfo: {
        server_scope_key: '["http://chat.test","multi-user","manual",null,"bob",null]',
        title: "Tracked persona chat",
        server_chat_id: cachedServerId
      },
      messages: [
        {
          id: "message-1",
          role: "user",
          content: "hello"
        }
      ]
    })

    usePlaygroundSessionStore.getState().saveSession({
      historyId: "local-history-7",
      serverChatId: "persona-chat-7",
      trackedAssistantSelection: {
        kind: "persona",
        id: "persona-7",
        name: "Garden Helper",
        metadata: {
          selectionMode: "tracked"
        }
      },
      trackedAssistantKind: "persona",
      trackedAssistantId: "persona-7",
      trackedCharacterId: null,
      trackedAssistantDisplayName: "Garden Helper",
      trackedAssistantAvatarUrl: null,
      serverChatPersonaMemoryMode: "read_only",
      scopeKey: "global",
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
      ragMediaIds: null,
      ragSearchMode: "hybrid",
      ragTopK: null,
      ragEnableGeneration: true,
      ragEnableCitations: true,
      queuedMessages: []
    })

    const { result } = renderHook(() => usePlaygroundSessionPersistence())

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
    })

    await expect(result.current.restoreSession()).resolves.toBe("restored")

    await waitFor(() => {
      expect(useStoreMessageOption.getState().historyId).toBe("local-history-7")
      expect(useStoreMessageOption.getState().serverChatId).toBe("persona-chat-7")
      expect(useStoreMessageOption.getState().serverChatAssistantKind).toBe(
        "persona"
      )
      expect(useStoreMessageOption.getState().serverChatAssistantId).toBe(
        "persona-7"
      )
      expect(useStoreMessageOption.getState().serverChatPersonaMemoryMode).toBe(
        "read_only"
      )
      expect(useStoreMessageOption.getState().serverChatTitle).toBe(expectedTitle)
      expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(false)
    })
  })

  it("preserves canonical metadata loaded while cached assistant persistence is pending", async () => {
    let releaseAssistantWrite: () => void = () => undefined
    mocks.setSelectedAssistant.mockReturnValue(
      new Promise<void>((resolve) => {
        releaseAssistantWrite = resolve
      })
    )
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "restored-chat",
      trackedAssistantSelection: {
        kind: "character",
        id: "5",
        name: "Cached character",
        metadata: { selectionMode: "tracked" }
      },
      trackedAssistantKind: "character",
      trackedAssistantId: "5",
      trackedCharacterId: "5",
      scopeKey: "global",
      queuedMessages: []
    })
    const { result } = renderHook(() => usePlaygroundSessionPersistence())
    await waitFor(() => expect(result.current.sessionScopeReady).toBe(true))
    let restoring: ReturnType<typeof result.current.restoreSession> | undefined
    act(() => {
      restoring = result.current.restoreSession()
    })
    await waitFor(() => expect(mocks.setSelectedAssistant).toHaveBeenCalled())

    // A canonical server response may finish before the cached storage write.
    await act(async () => {
      const state = useStoreMessageOption.getState()
      state.setServerChatTitle("Canonical title")
      state.setServerChatCharacterId("6")
      state.setServerChatAssistantId("6")
      state.setServerChatMetaLoaded(true)
      releaseAssistantWrite()
      await expect(restoring).resolves.toBe("restored")
    })
    const state = useStoreMessageOption.getState()
    expect({
      title: state.serverChatTitle,
      assistantId: state.serverChatAssistantId,
      metaLoaded: state.serverChatMetaLoaded
    }).toEqual({
      title: "Canonical title",
      assistantId: "6",
      metaLoaded: true
    })
  })

  it("does not overwrite a server chat selected while session restore is in flight", async () => {
    let resolveChatData: (value: {
      historyInfo: Record<string, never>
      messages: Array<{ id: string; role: string; content: string }>
    }) => void = () => undefined
    const deferredChatData = new Promise<{
      historyInfo: Record<string, never>
      messages: Array<{ id: string; role: string; content: string }>
    }>((resolve) => {
      resolveChatData = resolve
    })
    mocks.getFullChatData.mockReturnValue(deferredChatData)
    usePlaygroundSessionStore.getState().saveSession({
      historyId: "persisted-history",
      serverChatId: "persisted-chat",
      scopeKey: "global",
      chatMode: "rag",
      ragMediaIds: [42],
      fileRetrievalEnabled: true,
      queuedMessages: []
    })

    const { result } = renderHook(
      () => ({
        persistence: usePlaygroundSessionPersistence(),
        selectServerChat: useSelectServerChat()
      }),
      { wrapper: MemoryRouter }
    )

    await waitFor(() => {
      expect(result.current.persistence.sessionScopeReady).toBe(true)
    })

    let restorePromise:
      | ReturnType<typeof result.current.persistence.restoreSession>
      | undefined
    act(() => {
      restorePromise = result.current.persistence.restoreSession()
    })
    await waitFor(() => {
      expect(mocks.getFullChatData).toHaveBeenCalledWith("persisted-history")
    })

    act(() => {
      result.current.selectServerChat({
        id: "selected-chat",
        title: "Selected from Chats",
        version: 1,
        state: "active",
        topic_label: null,
        cluster_id: null,
        source: "webui",
        external_ref: null
      } as ServerChatSummary)
    })
    await act(async () => {
      resolveChatData({
        historyInfo: {},
        messages: [
          { id: "persisted-message", role: "user", content: "stale" }
        ]
      })
      await expect(restorePromise).resolves.toBe("cancelled")
    })

    expect(useStoreMessageOption.getState().serverChatId).toBe("selected-chat")
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().serverChatTitle).toBe("Selected from Chats")
    expect(useStoreMessageOption.getState().ragMediaIds).toBeNull()
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
  })

  it("reports cancellation when a server chat is selected during assistant persistence", async () => {
    let releaseAssistantWrite = () => undefined
    mocks.setSelectedAssistant.mockReturnValue(
      new Promise<void>((resolve) => {
        releaseAssistantWrite = resolve
      })
    )
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "persisted-chat",
      trackedAssistantSelection: {
        kind: "persona",
        id: "persisted-persona",
        name: "Persisted Persona",
        metadata: { selectionMode: "tracked" }
      },
      trackedAssistantKind: "persona",
      trackedAssistantId: "persisted-persona",
      scopeKey: "global",
      queuedMessages: []
    })
    const { result } = renderHook(
      () => ({
        persistence: usePlaygroundSessionPersistence(),
        selectServerChat: useSelectServerChat()
      }),
      { wrapper: MemoryRouter }
    )
    await waitFor(() => {
      expect(result.current.persistence.sessionScopeReady).toBe(true)
    })

    let restorePromise:
      | ReturnType<typeof result.current.persistence.restoreSession>
      | undefined
    act(() => {
      restorePromise = result.current.persistence.restoreSession()
    })
    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({ id: "persisted-persona" }),
        expect.objectContaining({ isCurrent: expect.any(Function) })
      )
    })

    act(() => {
      result.current.selectServerChat({
        id: "explicit-chat",
        title: "Explicit chat",
        version: 1,
        state: "active",
        topic_label: null,
        cluster_id: null,
        source: "webui",
        external_ref: null
      } as ServerChatSummary)
      releaseAssistantWrite()
    })

    await expect(restorePromise).resolves.toBe("cancelled")
    expect(useStoreMessageOption.getState().serverChatId).toBe("explicit-chat")
    expect(useStoreMessageOption.getState().serverChatTitle).toBe("Explicit chat")
  })

  it("keeps the richer tracked persona snapshot when autosave only has generic metadata", async () => {
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "persona-chat-9",
      trackedAssistantSelection: {
        kind: "persona",
        id: "persona-9",
        name: "Garden Helper",
        metadata: {
          selectionMode: "tracked"
        }
      },
      trackedAssistantKind: "persona",
      trackedAssistantId: "persona-9",
      trackedCharacterId: null,
      trackedAssistantDisplayName: "Garden Helper",
      trackedAssistantAvatarUrl: null,
      serverChatPersonaMemoryMode: "read_only",
      scopeKey: "global",
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
      ragMediaIds: null,
      ragSearchMode: "hybrid",
      ragTopK: null,
      ragEnableGeneration: true,
      ragEnableCitations: true,
      queuedMessages: []
    })

    const { result, rerender } = renderHook(() =>
      usePlaygroundSessionPersistence()
    )

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
    })

    await expect(result.current.restoreSession()).resolves.toBe("restored")

    useStoreMessageOption.setState({
      historyId: "local-history-9",
      serverChatId: "persona-chat-9",
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "persona-9",
      serverChatCharacterId: null,
      serverChatPersonaMemoryMode: "read_only",
      serverChatMetaLoaded: true
    })
    rerender()

    await waitFor(() => {
      const state = usePlaygroundSessionStore.getState()
      expect(state.historyId).toBe("local-history-9")
      expect(state.trackedAssistantKind).toBe("persona")
      expect(state.trackedAssistantId).toBe("persona-9")
      expect(state.trackedAssistantDisplayName).toBe("Garden Helper")
      expect(state.trackedAssistantSelection).toEqual(
        expect.objectContaining({
          kind: "persona",
          id: "persona-9",
          name: "Garden Helper",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      )
    })
  })

  it("allows immediate session persistence after an empty restore attempt", async () => {
    const { result } = renderHook(() => usePlaygroundSessionPersistence())

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
    })

    await expect(result.current.restoreSession()).resolves.toBe("not-restored")

    useStoreMessageOption.setState({
      historyId: "local-history-new",
      serverChatId: "persona-chat-new",
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "persona-new",
      serverChatCharacterId: null,
      serverChatPersonaMemoryMode: "read_only",
      serverChatMetaLoaded: true
    })

    await waitFor(
      () => {
        const state = usePlaygroundSessionStore.getState()
        expect(state.historyId).toBe("local-history-new")
        expect(state.serverChatId).toBe("persona-chat-new")
        expect(state.trackedAssistantKind).toBe("persona")
        expect(state.trackedAssistantId).toBe("persona-new")
      },
      { timeout: 250 }
    )
  })

  it("allows immediate session persistence when no restore attempt is needed", async () => {
    const { result } = renderHook(() => usePlaygroundSessionPersistence())

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
      expect(result.current.hasPersistedSession).toBe(false)
    })

    useStoreMessageOption.setState({
      historyId: "local-history-fresh",
      serverChatId: "character-chat-fresh",
      serverChatAssistantKind: "character",
      serverChatAssistantId: "character-fresh",
      serverChatCharacterId: "character-fresh",
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: true
    })

    await waitFor(
      () => {
        const state = usePlaygroundSessionStore.getState()
        expect(state.historyId).toBe("local-history-fresh")
        expect(state.serverChatId).toBe("character-chat-fresh")
        expect(state.trackedAssistantKind).toBe("character")
        expect(state.trackedAssistantId).toBe("character-fresh")
      },
      { timeout: 250 }
    )
  })

  it("clears stale tracked state when restoring a plain server-backed session", async () => {
    useStoreMessageOption.setState({
      serverChatAssistantKind: "character",
      serverChatAssistantId: "stale-character",
      serverChatCharacterId: "stale-character",
      serverChatPersonaMemoryMode: "read_only",
      serverChatMetaLoaded: true
    })
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "plain-chat-11",
      trackedAssistantSelection: null,
      trackedAssistantKind: null,
      trackedAssistantId: null,
      trackedCharacterId: null,
      trackedAssistantDisplayName: null,
      trackedAssistantAvatarUrl: null,
      serverChatPersonaMemoryMode: null,
      scopeKey: "global",
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
      ragMediaIds: null,
      ragSearchMode: "hybrid",
      ragTopK: null,
      ragEnableGeneration: true,
      ragEnableCitations: true,
      queuedMessages: []
    })

    const { result } = renderHook(() => usePlaygroundSessionPersistence())

    await waitFor(() => {
      expect(result.current.sessionScopeReady).toBe(true)
    })

    await expect(result.current.restoreSession()).resolves.toBe("restored")

    await waitFor(() => {
      const optionState = useStoreMessageOption.getState()
      expect(optionState.serverChatId).toBe("plain-chat-11")
      expect(optionState.serverChatAssistantKind).toBeNull()
      expect(optionState.serverChatAssistantId).toBeNull()
      expect(optionState.serverChatCharacterId).toBeNull()
      expect(optionState.serverChatPersonaMemoryMode).toBeNull()
      expect(optionState.serverChatMetaLoaded).toBe(false)
    })
  })
})
