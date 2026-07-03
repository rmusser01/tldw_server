import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { usePlaygroundPersistence } from "../usePlaygroundPersistence"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  searchCharacters: vi.fn(),
  listCharacters: vi.fn(),
  createCharacter: vi.fn(),
  createChat: vi.fn(),
  addChatMessage: vi.fn(),
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

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    searchCharacters: mocks.searchCharacters,
    listCharacters: mocks.listCharacters,
    createCharacter: mocks.createCharacter,
    createChat: mocks.createChat,
    addChatMessage: mocks.addChatMessage,
    getConfig: mocks.getConfig
  }
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
    mocks.initialize.mockReset()
    mocks.searchCharacters.mockReset()
    mocks.listCharacters.mockReset()
    mocks.createCharacter.mockReset()
    mocks.createChat.mockReset()
    mocks.addChatMessage.mockReset()
    mocks.getConfig.mockReset()
    mocks.savePlaygroundSession.mockReset()
    mocks.buildChatSurfaceScopeKeyFromConfig.mockReset()
    mocks.usePersistenceMode.mockReset()

    mocks.initialize.mockResolvedValue(undefined)
    mocks.searchCharacters.mockRejectedValue(new Error("search failed"))
    mocks.listCharacters.mockRejectedValue(new Error("list failed"))
    mocks.createCharacter.mockRejectedValue(new Error("create failed"))
    mocks.createChat.mockResolvedValue({ id: "chat-1" })
    mocks.addChatMessage.mockResolvedValue(undefined)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.buildChatSurfaceScopeKeyFromConfig.mockReturnValue("scope:chat")
    mocks.usePersistenceMode.mockReturnValue({
      persistenceTooltip: "save to server",
      focusConnectionCard: vi.fn(),
      getPersistenceModeLabel: vi.fn(() => "Saved to server")
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
        })
      )
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: expect.anything()
        })
      )
    })
    expect(notificationApi.error).not.toHaveBeenCalled()

    rerender(
      {
        ...stableDeps,
        history: [{ role: "user", content: "Hello world" }],
      }
    )

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
        })
      )
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.not.objectContaining({
          character_id: "stale-character"
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
        })
      )
    })
    expect(mocks.createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "webui-chat"
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
        })
      )
    })
    expect(mocks.createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "webui-chat"
      })
    )
  })
})
