import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  beginServerChatWrite,
  getServerChatSaveStatus,
  resetServerChatSaveStatus
} from "@/store/server-chat-save-status"
import { usePlaygroundPersistence } from "../usePlaygroundPersistence"

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
  loadServicePromptSnapshot: async () => {
    await mocks.initialize()
    return {
      scopeSignal: new AbortController().signal,
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

vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: mocks.buildChatSurfaceScopeKeyFromConfig
}))

vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({
      saveSession: mocks.savePlaygroundSession
    })
  }
}))

vi.mock("@/hooks/playground", () => ({
  usePersistenceMode: (...args: unknown[]) =>
    (mocks.usePersistenceMode as (...args: unknown[]) => unknown)(...args)
}))

const buildDeps = (overrides: Record<string, unknown> = {}) => ({
  isFireFoxPrivateMode: false,
  isConnectionReady: true,
  temporaryChat: false,
  setTemporaryChat: vi.fn(),
  serverChatId: null,
  setServerChatId: vi.fn(),
  historyId: null,
  serverChatState: null,
  setServerChatState: vi.fn(),
  serverChatSource: null,
  setServerChatSource: vi.fn(),
  setServerChatVersion: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
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
})

describe("usePlaygroundPersistence", () => {
  beforeEach(() => {
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
      focusConnectionCard: vi.fn()
    })
  })

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
    const { result } = renderHook(() => usePlaygroundPersistence(buildDeps()))
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

    const { result } = renderHook(() =>
      usePlaygroundPersistence(
        buildDeps({ history: [{ role: "user", content: "Persist this chat" }] })
      )
    )

    await waitFor(() => expect(mocks.addChatMessage).toHaveBeenCalled())
    await waitFor(() => expect(getServerChatSaveStatus("chat-1")).toBe("saved"))
    expect(result.current.showServerPersistenceHint).toBe(false)
  })

  it.each([
    [
      "a local chat while connected",
      null,
      null,
      "Saved on this device only. This chat is not on your tldw server."
    ],
    [
      "a server chat whose last write failed",
      "chat-9",
      "failed",
      "Saved on this device. Your tldw server didn't confirm the latest changes."
    ],
    [
      "a server chat whose last write was acknowledged",
      "chat-9",
      "saved",
      "Saved on your tldw server and on this device."
    ]
  ] as const)(
    "announces truthful persistence copy when leaving temporary mode for %s",
    (_label, serverChatId, outcome, expected) => {
      if (serverChatId && outcome) beginServerChatWrite(serverChatId)(outcome)
      const notificationApi = {
        error: vi.fn(),
        warning: vi.fn(),
        info: vi.fn(),
        success: vi.fn()
      }
      const { result } = renderHook(() =>
        usePlaygroundPersistence(
          buildDeps({
            notificationApi,
            temporaryChat: true,
            serverChatId,
            history: []
          })
        )
      )

      act(() => result.current.handleToggleTemporaryChat(false))

      expect(notificationApi.info).toHaveBeenCalledWith(
        expect.objectContaining({ message: expected })
      )
    }
  )
})
