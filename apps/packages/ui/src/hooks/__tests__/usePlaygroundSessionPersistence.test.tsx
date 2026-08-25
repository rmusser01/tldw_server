// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getFullChatData: vi.fn(),
  getPromptById: vi.fn(),
  setSystemPrompt: vi.fn(),
  setSelectedAssistant: vi.fn()
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
  useSelectedAssistant: () => [null, mocks.setSelectedAssistant]
}))

import { usePlaygroundSessionPersistence } from "../usePlaygroundSessionPersistence"
import { useSelectServerChat } from "../chat/useSelectServerChat"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"

describe("usePlaygroundSessionPersistence", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    mocks.getConfig.mockResolvedValue(null)
    mocks.getFullChatData.mockResolvedValue(null)
    mocks.setSelectedAssistant.mockReset()
    useStoreMessageOption.setState({
      history: [],
      messages: [],
      historyId: null,
      serverChatId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatCharacterId: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      chatMode: "normal",
      webSearch: false,
      compareMode: false,
      compareSelectedModels: [],
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
      expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(true)
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

  it("restores a persisted server-backed chat alongside local history without dropping the server chat id", async () => {
    mocks.getFullChatData.mockResolvedValue({
      historyInfo: {
        title: "Tracked persona chat"
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
