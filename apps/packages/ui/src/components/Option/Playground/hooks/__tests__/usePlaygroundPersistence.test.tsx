import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"

import {
  beginServerChatWrite,
  getServerChatSaveStatus,
  resetServerChatSaveStatus
} from "@/store/server-chat-save-status"
import {
  usePlaygroundPersistence as useActualPlaygroundPersistence,
  type UsePlaygroundPersistenceDeps
} from "../usePlaygroundPersistence"
import { useStoreMessageOption, type ChatHistory } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useTemporaryChatToggle } from "@/hooks/useTemporaryChatToggle"
import { getPersistenceModeLabel } from "@/hooks/playground/usePersistenceMode"
import { normalizeConversationState } from "@/utils/conversation-state"

const selectedHistory = vi.hoisted(() => ({ value: null as any }))
vi.mock('@/hooks/chat/useHistorySelection', () => ({ useHistorySelectionContext: () => selectedHistory.value }))

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  searchCharacters: vi.fn(),
  listCharacters: vi.fn(),
  createCharacter: vi.fn(),
  createChat: vi.fn(),
  addChatMessage: vi.fn(),
  deleteChat: vi.fn(),
  getChatPromotionBlocker: vi.fn(),
  getConfig: vi.fn(),
  savePlaygroundSession: vi.fn(),
  buildChatSurfaceScopeKeyFromConfig: vi.fn(),
  usePersistenceMode: vi.fn()
}))

const modeNotification = vi.hoisted(() => ({
  error: vi.fn(), warning: vi.fn(), info: vi.fn(), success: vi.fn()
}))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => modeNotification
}))
vi.mock("@/utils/is-private-mode", () => ({ isFireFoxPrivateMode: false }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key })
}))

const translate = (
  key: string,
  defaultValue?: string,
  options?: Record<string, unknown>
) => {
  const value = defaultValue || key
  const name = options?.name
  return typeof name === "string" ? value.replace("{{name}}", name) : value
}

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: string[], options?: { signal?: AbortSignal }) => {
    await mocks.initialize()
    return {
      scopeKey: "scope:chat",
      scopeSignal: options?.signal ?? new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      requestScope: { config: await mocks.getConfig(), userId: null },
      release: vi.fn()
    }
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    searchCharacters: mocks.searchCharacters,
    listCharacters: mocks.listCharacters,
    createCharacter: mocks.createCharacter,
    createChat: mocks.createChat,
    addChatMessage: mocks.addChatMessage,
    deleteChat: mocks.deleteChat,
    getConfig: mocks.getConfig
  }
}))

vi.mock("@/db/dexie/chat-promotion", () => ({
  getChatPromotionBlocker: (...args: unknown[]) =>
    (mocks.getChatPromotionBlocker as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/chat-surface-scope", async () => ({
  ...await vi.importActual<typeof import("@/services/chat-surface-scope")>("@/services/chat-surface-scope"),
  buildChatSurfaceScopeKeyFromConfig: mocks.buildChatSurfaceScopeKeyFromConfig
}))

vi.mock("@/hooks/playground", () => ({
  usePersistenceMode: (...args: unknown[]) =>
    (mocks.usePersistenceMode as (...args: unknown[]) => unknown)(...args)
}))

// Declare the original store source once, before mounting. Setter spies record
// calls while the actual setters publish the issued ID and metadata.
const buildDeps = (overrides: Partial<UsePlaygroundPersistenceDeps> = {}) => {
  const state = useStoreMessageOption.getState()
  const deps: UsePlaygroundPersistenceDeps = {
    isFireFoxPrivateMode: false,
    isConnectionReady: true,
    temporaryChat: false,
    setTemporaryChat: vi.fn(state.setTemporaryChat),
    serverChatId: null,
    setServerChatId: vi.fn(state.setServerChatId),
    historyId: null,
    serverChatState: null,
    setServerChatState: vi.fn((value: string) =>
      state.setServerChatState(normalizeConversationState(value))),
    serverChatSource: null,
    setServerChatSource: vi.fn(state.setServerChatSource),
    setServerChatVersion: vi.fn(state.setServerChatVersion),
    setServerChatCharacterId: vi.fn(state.setServerChatCharacterId),
    setServerChatAssistantKind: vi.fn(state.setServerChatAssistantKind),
    setServerChatAssistantId: vi.fn((id: string | number | null) =>
      state.setServerChatAssistantId(id === null ? null : String(id))),
    setServerChatPersonaMemoryMode: vi.fn(state.setServerChatPersonaMemoryMode),
    history: [{ role: "user", content: "Hello" }],
    clearChat: vi.fn(),
    selectedCharacter: null,
    selectedAssistantMode: null,
    assistantOverlayActive: false,
    serverPersistenceHintSeen: false,
    setServerPersistenceHintSeen: vi.fn(),
    invalidateServerChatHistory: vi.fn(),
    navigate: vi.fn(),
    notificationApi: {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    },
    t: translate,
    ...overrides
  }
  state.setTemporaryChat(deps.temporaryChat)
  useStoreMessageOption.setState({
    historyId: deps.historyId,
    serverChatId: deps.serverChatId,
    serverChatState: deps.serverChatState === null
      ? null : normalizeConversationState(deps.serverChatState),
    serverChatSource: deps.serverChatSource,
    history: deps.history.map<ChatHistory[number]>(row => ({
      role: row.role === "assistant" || row.role === "system" ? row.role : "user",
      content: row.content ?? "",
      image: row.image
    }))
  })
  return deps
}

// Reflect acknowledged publication on subsequent renders, as the real caller
// does, while keeping each test's history/input and notification controls.
const usePlaygroundPersistence = (deps: UsePlaygroundPersistenceDeps) => {
  const state = useStoreMessageOption()
  return useActualPlaygroundPersistence({
    ...deps,
    serverChatId: state.serverChatId,
    serverChatState: state.serverChatState,
    serverChatSource: state.serverChatSource
  })
}

describe("usePlaygroundPersistence", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    selectedHistory.value = null
    mocks.initialize.mockReset()
    mocks.searchCharacters.mockReset()
    mocks.listCharacters.mockReset()
    mocks.createCharacter.mockReset()
    mocks.createChat.mockReset()
    mocks.addChatMessage.mockReset()
    mocks.deleteChat.mockReset()
    mocks.getChatPromotionBlocker.mockReset()
    mocks.getConfig.mockReset()
    mocks.savePlaygroundSession.mockReset()
    mocks.buildChatSurfaceScopeKeyFromConfig.mockReset()
    mocks.usePersistenceMode.mockReset()
    resetServerChatSaveStatus()
    useStoreMessageOption.setState(
      { ...useStoreMessageOption.getInitialState(), temporaryChat: false }, true
    )
    useStoreMessageOption.getState().startNewChatDraft(false)
    const session = usePlaygroundSessionStore.getInitialState()
    usePlaygroundSessionStore.setState({
      ...session,
      saveSession: (data) => {
        mocks.savePlaygroundSession(data)
        session.saveSession(data)
      }
    }, true)

    mocks.initialize.mockResolvedValue(undefined)
    mocks.searchCharacters.mockRejectedValue(new Error("search failed"))
    mocks.listCharacters.mockRejectedValue(new Error("list failed"))
    mocks.createCharacter.mockRejectedValue(new Error("create failed"))
    mocks.createChat.mockResolvedValue({ id: "chat-1" })
    mocks.addChatMessage.mockResolvedValue({ id: "saved-message", version: 1 })
    mocks.deleteChat.mockResolvedValue(undefined)
    mocks.getChatPromotionBlocker.mockResolvedValue(null)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.buildChatSurfaceScopeKeyFromConfig.mockReturnValue("scope:chat")
    mocks.usePersistenceMode.mockReturnValue({
      persistenceKind: "server",
      persistenceTooltip: "save to server",
      focusConnectionCard: vi.fn(),
      getPersistenceModeLabel
    })
  })

  it.each([
    ["playground", false], ["playground", true],
    ["sidepanel", false], ["sidepanel", true]
  ] as const)("%s respects clear acceptance (%s) before changing mode", (surface, accepted) => {
    const events: string[] = []
    const clearChat = vi.fn(() => { events.push("clear"); return accepted })
    const setTemporaryChat = vi.fn(() => { events.push("mode") })
    const notificationApi = surface === "sidepanel" ? modeNotification : buildDeps().notificationApi
    const confirm = vi.spyOn(Modal, "confirm").mockReturnValue({ destroy: vi.fn(), update: vi.fn() })
    try {
      const deps = buildDeps({ serverChatId: "source-chat", clearChat, setTemporaryChat, notificationApi })
      const { result } = renderHook(() => {
        const playground = usePlaygroundPersistence(deps)
        const sidepanel = useTemporaryChatToggle({ temporaryChat: false, setTemporaryChat, messagesLength: 1, clearChat })
        return surface === "playground" ? playground : sidepanel
      })

      act(() => { result.current.handleToggleTemporaryChat(true) })
      expect(setTemporaryChat).not.toHaveBeenCalled()
      expect(clearChat).not.toHaveBeenCalled()
      act(() => { confirm.mock.calls[0][0].onOk?.() })

      expect(events).toEqual(accepted ? ["clear", "mode"] : ["clear"])
      expect(notificationApi.info).toHaveBeenCalledTimes(accepted ? 1 : 0)
    } finally { confirm.mockRestore() }
  })

  it.each([
    ["playground", false],
    ["playground", true],
    ["sidepanel", false],
    ["sidepanel", true]
  ] as const)(
    "%s mode-change notice describes eligibility, temporary=%s",
    (surface, next) => {
      const setTemporaryChat = vi.fn()
      const deps = buildDeps({
        temporaryChat: !next,
        history: [],
        serverChatId: "previous-server",
        setTemporaryChat
      })
      const notificationApi =
        surface === "sidepanel" ? modeNotification : deps.notificationApi
      const { result } = renderHook(() => {
        const playground = usePlaygroundPersistence(deps)
        const sidepanel = useTemporaryChatToggle({
          temporaryChat: !next,
          setTemporaryChat,
          messagesLength: 0,
          clearChat: deps.clearChat
        })
        return surface === "playground" ? playground : sidepanel
      })
      act(() => result.current.handleToggleTemporaryChat(next))
      expect(notificationApi.info).toHaveBeenCalledWith(
        expect.objectContaining({
          message: next
            ? "Temporary messages can remain on this device across reloads until you close or clear the chat. They are not added to server history. Draft text may remain on this device."
            : "New messages can be stored in chat history."
        })
      )
      expect(setTemporaryChat).toHaveBeenCalledWith(next)
      expect(deps.clearChat).not.toHaveBeenCalled()
    }
  )

  it("does not autosave tracked character greetings before the send path owns persistence", async () => {
    renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          history: [
            {
              role: "assistant",
              content: "Ready for overlay continuity proof."
            }
          ],
          selectedCharacter: {
            id: "tracked-character",
            name: "Tracked Character",
            avatar_url: "https://example.test/avatar.png",
            greeting: "Ready for overlay continuity proof.",
            system_prompt: "Stay in character."
          },
          selectedAssistantMode: "tracked"
        })
      }
    )

    await waitFor(() => {
      expect(mocks.initialize).not.toHaveBeenCalled()
      expect(mocks.createChat).not.toHaveBeenCalled()
      expect(mocks.addChatMessage).not.toHaveBeenCalled()
    })
    expect(mocks.savePlaygroundSession).not.toHaveBeenCalled()
  })

  it("saves plain chats without requiring a default character", async () => {
    const firstHistory = [{ role: "user", content: "Hello" }]
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }
    const stableDeps = buildDeps({
      notificationApi,
      history: firstHistory
    })

    const { rerender } = renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: stableDeps
      }
    )

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.objectContaining({
          source: "webui-chat"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: expect.anything()
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
    })
    expect(notificationApi.error).not.toHaveBeenCalled()

    rerender({
      ...stableDeps,
      history: [{ role: "user", content: "Hello world" }]
    })

    await waitFor(() => {
      expect(mocks.initialize).toHaveBeenCalledTimes(1)
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(notificationApi.error).not.toHaveBeenCalled()
    })
  })

  it("shows inline persistence feedback without opening a blocking success notification", async () => {
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }
    const setServerPersistenceHintSeen = vi.fn()

    const { result } = renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          notificationApi,
          setServerPersistenceHintSeen,
          history: [{ role: "user", content: "Persist this chat" }]
        })
      }
    )

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(result.current.showServerPersistenceHint).toBe(true)
    })

    expect(setServerPersistenceHintSeen).toHaveBeenCalledWith(true)
    expect(notificationApi.success).not.toHaveBeenCalled()
    expect(notificationApi.error).not.toHaveBeenCalled()
  })

  it("uses current history when the first message arrives after mount", async () => {
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }
    const stableDeps = buildDeps({
      notificationApi,
      history: []
    })

    const { rerender } = renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: stableDeps
      }
    )

    expect(mocks.initialize).not.toHaveBeenCalled()
    expect(notificationApi.error).not.toHaveBeenCalled()

    rerender({
      ...stableDeps,
      history: [{ role: "user", content: "First message" }]
    })

    await waitFor(() => {
      expect(mocks.initialize).toHaveBeenCalledTimes(1)
      expect(mocks.createChat).toHaveBeenCalledTimes(1)
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.objectContaining({
          source: "webui-chat"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
    })
    expect(notificationApi.error).not.toHaveBeenCalled()
  })

  it("autosaves genuinely new transcript work after the original rendered and canonical histories retire", async () => {
    const renderedOriginal = [{ role: "user", content: "Old transcript belongs to the retired draft" }]
    const destroyed = vi.fn()
    const deps = buildDeps({
      historyId: "same-local-history",
      history: renderedOriginal,
      notificationApi: {
        destroy: destroyed, error: vi.fn(), warning: vi.fn(), info: vi.fn(), success: vi.fn()
      }
    })
    const canonicalOriginal = useStoreMessageOption.getState().history
    const originalAddress = useStoreMessageOption.getState().chatDraftAddress
    expect(canonicalOriginal).not.toBe(renderedOriginal)
    let release!: (value: { id: string }) => void
    mocks.createChat.mockResolvedValue({ id: "new-transcript-chat", version: 2 })
    mocks.createChat.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    const view = renderHook(
      ({ history, ready }: { history: UsePlaygroundPersistenceDeps["history"]; ready: boolean }) => {
        const state = useStoreMessageOption()
        return useActualPlaygroundPersistence({
          ...deps, history, isConnectionReady: ready,
          historyId: state.historyId, temporaryChat: state.temporaryChat,
          serverChatId: state.serverChatId, serverChatState: state.serverChatState,
          serverChatSource: state.serverChatSource
        })
      },
      { initialProps: { history: renderedOriginal, ready: true } }
    )
    try {
      await waitFor(() => expect(mocks.createChat).toHaveBeenCalledOnce())
      act(() => {
        useStoreMessageOption.getState().startNewChatDraft(false)
        useStoreMessageOption.setState({ serverChatSource: "Keep replacement source", serverChatVersion: 77 })
      })
      const replacementAddress = useStoreMessageOption.getState().chatDraftAddress
      expect(replacementAddress).not.toBe(originalAddress)
      view.rerender({ history: canonicalOriginal, ready: false })
      await act(async () => { release({ id: "retired-transcript-chat" }) })
      await waitFor(() => expect(destroyed).toHaveBeenCalledWith("chat-promotion-same-local-history"))
      view.rerender({ history: canonicalOriginal, ready: true })
      await act(async () => {})
      expect(mocks.createChat).toHaveBeenCalledOnce()
      expect(mocks.addChatMessage).not.toHaveBeenCalled()
      expect(useStoreMessageOption.getState()).toMatchObject({
        serverChatId: null, serverChatSource: "Keep replacement source", serverChatVersion: 77
      })

      const newHistory: ChatHistory = [
        { role: "user", content: "A genuinely new question" },
        { role: "assistant", content: "A genuinely new answer" }
      ]
      act(() => {
        const state = useStoreMessageOption.getState()
        state.setMessages([
          { id: "new-user", isBot: false, name: "You", message: newHistory[0].content, sources: [] },
          { id: "new-assistant", isBot: true, name: "Assistant", message: newHistory[1].content, sources: [] }
        ])
        state.setHistory(newHistory)
      })
      view.rerender({ history: newHistory, ready: true })
      await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalledTimes(2))
      expect(mocks.createChat).toHaveBeenCalledTimes(2)
      expect(mocks.createChat.mock.calls[1][0].title).toBe("A genuinely new question")
      expect(mocks.addChatMessage.mock.calls.map(([chatId, row]) => [chatId, row.content])).toEqual([
        ["new-transcript-chat", "A genuinely new question"],
        ["new-transcript-chat", "A genuinely new answer"]
      ])
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(replacementAddress)
      expect(useStoreMessageOption.getState().history).toBe(newHistory)
      expect(useStoreMessageOption.getState().serverChatId).toBe("new-transcript-chat")
    } finally {
      release?.({ id: "retired-transcript-chat" })
      view.unmount()
    }
  })

  it("does not autosave tracked character turns because character sends own persistence", async () => {
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }

    renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          notificationApi,
          history: [
            { role: "assistant", content: "Welcome to the archive." },
            { role: "user", content: "Show me the old city." }
          ],
          selectedCharacter: {
            id: "mira",
            name: "Mira"
          },
          selectedAssistantMode: "tracked"
        })
      }
    )

    await waitFor(() => {
      expect(mocks.initialize).not.toHaveBeenCalled()
      expect(mocks.createChat).not.toHaveBeenCalled()
      expect(mocks.addChatMessage).not.toHaveBeenCalled()
    })
    expect(notificationApi.error).not.toHaveBeenCalled()
  })

  it("does not persist a stale selected character as tracked for a plain chat", async () => {
    renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          history: [{ role: "user", content: "Plain conversation" }],
          selectedCharacter: {
            id: "stale-character",
            name: "Stale Character"
          },
          selectedAssistantMode: null
        })
      }
    )

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.objectContaining({
          source: "webui-chat"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: "stale-character"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
    })
    expect(mocks.savePlaygroundSession).not.toHaveBeenCalled()
  })

  it("does not fall back to a plain chat while character workflow is waiting for its tracked selection", async () => {
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }

    renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          notificationApi,
          characterWorkflowActive: true,
          history: [{ role: "user", content: "Continue the character scene" }],
          selectedCharacter: null,
          selectedAssistantMode: null,
          assistantOverlayActive: false
        })
      }
    )

    await waitFor(() => {
      expect(mocks.initialize).not.toHaveBeenCalled()
      expect(mocks.createChat).not.toHaveBeenCalled()
    })
    expect(notificationApi.error).not.toHaveBeenCalled()
    expect(mocks.savePlaygroundSession).not.toHaveBeenCalled()
  })

  it("does not persist overlay character selections as tracked server chats", async () => {
    const { result } = renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          history: [{ role: "user", content: "Hello from overlay" }],
          selectedCharacter: {
            id: "overlay-char",
            name: "Overlay Character"
          },
          selectedAssistantMode: "overlay",
          assistantOverlayActive: true
        })
      }
    )

    await result.current.handleSaveChatToServer()

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: "overlay-char"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
    })
    expect(mocks.createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "webui-chat"
      }),
      expect.objectContaining({
        requestScope: expect.anything(),
        signal: expect.anything()
      })
    )
  })

  it("treats a pending local overlay snapshot as overlay even if the selected assistant mode is not hydrated yet", async () => {
    const { result } = renderHook(
      (deps: ReturnType<typeof buildDeps>) => usePlaygroundPersistence(deps),
      {
        initialProps: buildDeps({
          history: [{ role: "user", content: "Hello from pending overlay" }],
          selectedCharacter: {
            id: "overlay-char",
            name: "Overlay Character"
          },
          selectedAssistantMode: null,
          assistantOverlayActive: true
        })
      }
    )

    await result.current.handleSaveChatToServer()

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: "overlay-char"
        }),
        expect.objectContaining({
          requestScope: expect.anything(),
          signal: expect.anything()
        })
      )
    })
    expect(mocks.createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "webui-chat"
      }),
      expect.objectContaining({
        requestScope: expect.anything(),
        signal: expect.anything()
      })
    )
  })
  it.each(['loading', 'ready'])('does not upload an H1 %s selection through legacy save', async status => {
    selectedHistory.value = { getCurrent: () => ({ status, owner: status === 'ready' ? { kind: 'local' } : null }), fence: () => () => true }
    const deps = buildDeps()
    const { result } = renderHook(() => usePlaygroundPersistence(deps))
    await act(async () => { await result.current.handleSaveChatToServer() })
    expect(mocks.createChat).not.toHaveBeenCalled()
    expect(mocks.addChatMessage).not.toHaveBeenCalled()
  })

  it.each(['initialize', 'createChat'] as const)('does not retarget or copy after a selection replaces a draft during %s', async held => {
    let current = true
    selectedHistory.value = { getCurrent: () => ({ status: current ? 'idle' : 'ready' }), fence: () => () => current }
    let release!: (value?: any) => void
    mocks[held].mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    const deps = buildDeps()
    renderHook(() => usePlaygroundPersistence(deps))
    await waitFor(() => expect(release).toBeTypeOf('function'))
    current = false
    await act(async () => { release({ id: 'late-draft' }) })
    expect(deps.setServerChatId).not.toHaveBeenCalled()
    expect(mocks.addChatMessage).not.toHaveBeenCalled()
    if (held === 'initialize') expect(mocks.createChat).not.toHaveBeenCalled()
    // CS-N3 (#3104): a chat created for a save that was abandoned is deleted,
    // not left on the server empty.
    if (held === 'createChat')
      await waitFor(() =>
        expect(mocks.deleteChat).toHaveBeenCalledWith(
          'late-draft',
          expect.objectContaining({ hardDelete: true })
        )
      )
  })

  // CS-03 (#3104): promotion records whether the server acknowledged the chat,
  // so persistence labels never claim the server for an incomplete copy.
  it("marks a promoted chat as saving until every message is acknowledged, then saved", async () => {
    let releaseMessage!: (value: { id: string; version: number }) => void
    mocks.addChatMessage.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseMessage = resolve
        })
    )
    const deps = buildDeps({
      history: [{ role: "user", content: "Keep this on the server" }]
    })

    renderHook(() => usePlaygroundPersistence(deps))

    await waitFor(() => expect(releaseMessage).toBeTypeOf("function"))
    expect(deps.setServerChatId).toHaveBeenCalledWith("chat-1")
    expect(getServerChatSaveStatus("chat-1")).toBe("saving")

    await act(async () => {
      releaseMessage({ id: "saved-message", version: 1 })
    })

    await waitFor(() => expect(getServerChatSaveStatus("chat-1")).toBe("saved"))
  })

  it("marks a promoted chat as failed when the server rejects a message write", async () => {
    mocks.addChatMessage.mockRejectedValueOnce(new Error("server unavailable"))
    const notificationApi = {
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      success: vi.fn()
    }
    const deps = buildDeps({
      notificationApi,
      history: [{ role: "user", content: "Keep this on the server" }]
    })

    const { result } = renderHook(() => usePlaygroundPersistence(deps))

    await waitFor(() => expect(notificationApi.error).toHaveBeenCalled())
    expect(deps.setServerChatId).toHaveBeenCalledWith("chat-1")
    expect(getServerChatSaveStatus("chat-1")).toBe("failed")
    expect(result.current.showServerPersistenceHint).toBe(false)
  })

  it.each([
    ["the history selection owns the chat", "history_selection_owned"],
    ["it already mirrors a server chat", "server_linked"]
  ] as const)(
    "does not create a server chat when %s (CS-N3)",
    async (_label, blocker) => {
      mocks.getChatPromotionBlocker.mockResolvedValue(blocker)
      const deps = buildDeps({
        historyId: "local-chat",
        history: [{ role: "user", content: "Keep this on the server" }]
      })

      const { result } = renderHook(() => usePlaygroundPersistence(deps))
      await act(async () => {
        await result.current.handleSaveChatToServer()
      })

      expect(mocks.getChatPromotionBlocker).toHaveBeenCalledWith("local-chat")
      expect(mocks.createChat).not.toHaveBeenCalled()
      expect(deps.setServerChatId).not.toHaveBeenCalled()
    }
  )

  it("hides the saved-on-server hint when the server copy is not acknowledged", async () => {
    mocks.usePersistenceMode.mockReturnValue({
      persistenceKind: "serverFailed",
      persistenceTooltip: "not saved to server",
      focusConnectionCard: vi.fn()
    })

    const deps = buildDeps({ history: [{ role: "user", content: "Persist this chat" }] })
    const { result } = renderHook(() => usePlaygroundPersistence(deps))

    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalled())
    await waitFor(() => expect(getServerChatSaveStatus("chat-1")).toBe("saved"))
    expect(result.current.showServerPersistenceHint).toBe(false)
  })

  it.each([
    [
      "a local chat while connected",
      null,
      null,
      "New messages can be stored in chat history."
    ],
    [
      "a server chat whose last write failed",
      "chat-9",
      "failed",
      "New messages can be stored in chat history."
    ],
    [
      "a server chat whose last write was acknowledged",
      "chat-9",
      "saved",
      "New messages can be stored in chat history."
    ]
  ] as const)(
    "announces history eligibility for the fresh chat after leaving temporary mode from %s",
    (_label, serverChatId, outcome, expected) => {
      if (serverChatId && outcome) beginServerChatWrite(serverChatId)(outcome)
      const notificationApi = {
        error: vi.fn(),
        warning: vi.fn(),
        info: vi.fn(),
        success: vi.fn()
      }
      const deps = buildDeps({ notificationApi, temporaryChat: true,
        serverChatId, history: [] })
      const { result } = renderHook(() => usePlaygroundPersistence(deps))

      act(() => result.current.handleToggleTemporaryChat(false))

      expect(notificationApi.info).toHaveBeenCalledWith(
        expect.objectContaining({ message: expected })
      )
    }
  )
})
