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

  it("immediately persists tracked character identity when greeting auto-creates a server chat", async () => {
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
      expect(mocks.savePlaygroundSession).toHaveBeenCalledWith(
        expect.objectContaining({
          serverChatId: "chat-1",
          trackedAssistantKind: "character",
          trackedAssistantId: "tracked-character",
          trackedCharacterId: "tracked-character",
          trackedAssistantDisplayName: "Tracked Character",
          trackedAssistantAvatarUrl: "https://example.test/avatar.png",
          serverChatPersonaMemoryMode: null,
          scopeKey: "scope:chat"
        })
      )
    })

    expect(mocks.savePlaygroundSession).toHaveBeenCalledWith(
      expect.objectContaining({
        trackedAssistantSelection: expect.objectContaining({
          kind: "character",
          id: "tracked-character",
          name: "Tracked Character",
          greeting: "Ready for overlay continuity proof.",
          system_prompt: "Stay in character.",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      })
    )
    expect(
      mocks.savePlaygroundSession.mock.invocationCallOrder[0]
    ).toBeLessThan(mocks.addChatMessage.mock.invocationCallOrder[0])
  })

  it("shows the server character error notification only once across rerenders for the same pending chat", async () => {
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
      expect(notificationApi.error).toHaveBeenCalledTimes(1)
    })

    rerender(
      {
        ...stableDeps,
        history: [{ role: "user", content: "Hello world" }],
      }
    )

    await waitFor(() => {
      expect(mocks.initialize).toHaveBeenCalledTimes(1)
      expect(notificationApi.error).toHaveBeenCalledTimes(1)
    })
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
      expect(notificationApi.error).toHaveBeenCalledTimes(1)
    })
  })

  it("uses a WebUI character-aware fallback title for server persistence", async () => {
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
          history: [{ role: "assistant", content: "Welcome to the archive." }],
          selectedCharacter: {
            id: "mira",
            name: "Mira"
          }
        })
      }
    )

    await waitFor(() => {
      expect(mocks.createChat).toHaveBeenCalledWith(
        expect.objectContaining({
          character_id: "mira",
          title: "Mira role-play",
          source: "webui-character-chat"
        })
      )
    })
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
