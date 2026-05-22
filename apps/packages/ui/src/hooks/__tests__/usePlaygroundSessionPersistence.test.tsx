// @vitest-environment jsdom
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getFullChatData: vi.fn(),
  getPromptById: vi.fn(),
  setSystemPrompt: vi.fn()
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

import { usePlaygroundSessionPersistence } from "../usePlaygroundSessionPersistence"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"

describe("usePlaygroundSessionPersistence", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    mocks.getConfig.mockResolvedValue(null)
    mocks.getFullChatData.mockResolvedValue(null)
    useStoreMessageOption.setState({
      history: [],
      messages: [],
      historyId: null,
      serverChatId: null,
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
    usePlaygroundSessionStore.getState().saveSession({
      historyId: null,
      serverChatId: "character-chat-42",
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

    await expect(result.current.restoreSession()).resolves.toBe(true)

    expect(useStoreMessageOption.getState().serverChatId).toBe(
      "character-chat-42"
    )
    expect(mocks.getFullChatData).not.toHaveBeenCalled()
  })
})
