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
  usePersistenceMode: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    searchCharacters: mocks.searchCharacters,
    listCharacters: mocks.listCharacters,
    createCharacter: mocks.createCharacter,
    createChat: mocks.createChat,
    addChatMessage: mocks.addChatMessage
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
  serverChatState: null,
  setServerChatState: vi.fn(),
  serverChatSource: null,
  setServerChatSource: vi.fn(),
  setServerChatVersion: vi.fn(),
  history: [{ role: "user", content: "Hello" }],
  clearChat: vi.fn(),
  selectedCharacter: null,
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
  t: (key: string, defaultValue?: string) => defaultValue || key,
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
    mocks.usePersistenceMode.mockReset()

    mocks.initialize.mockResolvedValue(undefined)
    mocks.searchCharacters.mockRejectedValue(new Error("search failed"))
    mocks.listCharacters.mockRejectedValue(new Error("list failed"))
    mocks.createCharacter.mockRejectedValue(new Error("create failed"))
    mocks.createChat.mockResolvedValue({ id: "chat-1" })
    mocks.addChatMessage.mockResolvedValue(undefined)
    mocks.usePersistenceMode.mockReturnValue({
      persistenceTooltip: "save to server",
      focusConnectionCard: vi.fn(),
      getPersistenceModeLabel: vi.fn(() => "Saved to server")
    })
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
})
