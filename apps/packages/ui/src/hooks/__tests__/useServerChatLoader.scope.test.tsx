import React from "react"
// @vitest-environment jsdom
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react"
import type { TFunction } from "i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const store = {
    serverChatId: "chat-a" as string | null,
    serverChatTitle: "Chat A" as string | null,
    serverChatCharacterId: null as string | number | null,
    serverChatAssistantKind: "character" as "character" | "persona" | null,
    serverChatAssistantId: null as string | null,
    serverChatPersonaMemoryMode: null as "read_only" | "read_write" | null,
    serverChatMetaLoaded: true,
    temporaryChat: false,
    setServerChatId: vi.fn(),
    setServerChatLoadState: vi.fn(),
    setServerChatLoadError: vi.fn(),
    setServerChatTitle: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatVersion: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatClusterId: vi.fn(),
    setServerChatSource: vi.fn(),
    setServerChatExternalRef: vi.fn(),
    setServerChatMetaLoaded: vi.fn()
  }

  return {
    store,
    selection: null as any,
    renderedMessages: [] as any[],
    getHistoriesWithMetadata: vi.fn(),
    initialize: vi.fn(),
    listChatMessages: vi.fn(),
    saveMessage: vi.fn(),
    setHistory: vi.fn(),
    setIsLoading: vi.fn(),
    setMessages: vi.fn(),
    setSelectedAssistant: vi.fn(),
    syncChatSettingsForServerChat: vi.fn(),
    updatePageTitle: vi.fn()
  }
})

vi.mock("@/hooks/chat/useHistorySelection", () => ({ useHistorySelectionContext: () => mocks.selection }))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => ({
    messages: mocks.renderedMessages,
    streaming: false,
    isProcessing: false,
    setHistory: mocks.setHistory,
    setMessages: mocks.setMessages,
    setIsLoading: mocks.setIsLoading
  })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof mocks.store) => unknown
  ) => selector(mocks.store)
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, mocks.setSelectedAssistant]
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    listChatMessages: mocks.listChatMessages
  }
}))

vi.mock("@/db/dexie/helpers", () => ({
  getHistoriesWithMetadata: mocks.getHistoriesWithMetadata,
  saveMessage: mocks.saveMessage
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: mocks.syncChatSettingsForServerChat
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: mocks.updatePageTitle
}))

import { useServerChatLoader } from "@/hooks/chat/useServerChatLoader"

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

describe("useServerChatLoader scoped local history", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    mocks.selection = null
    mocks.renderedMessages = []
    mocks.store.serverChatId = "chat-a"
    mocks.store.serverChatTitle = "Chat A"
    mocks.store.serverChatMetaLoaded = true
    mocks.store.temporaryChat = false
    mocks.initialize.mockResolvedValue(undefined)
    mocks.listChatMessages.mockResolvedValue([])
    mocks.getHistoriesWithMetadata.mockResolvedValue(
      new Map([["history-a", { messageCount: 1 }]])
    )
    mocks.syncChatSettingsForServerChat.mockResolvedValue(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("does not publish a superseded load after history linking is aborted", async () => {
    const historyLink = deferred<string | null>()
    const ensureServerChatHistoryId = vi.fn(
      (_chatId: string, _title?: string, signal?: AbortSignal) => {
        if (signal) {
          signal.addEventListener(
            "abort",
            () => {
              const error = Object.assign(new Error("Request scope changed"), {
                status: 412,
                details: {
                  detail: { code: "request_config_scope_changed" }
                }
              })
              historyLink.reject(error)
            },
            { once: true }
          )
        }
        return historyLink.promise
      }
    )
    const notification = { error: vi.fn() }
    const { rerender } = renderHook(() =>
      useServerChatLoader({
        ensureServerChatHistoryId,
        notification,
        t: ((_key: string, options?: { defaultValue?: string }) =>
          options?.defaultValue ?? "Error") as unknown as TFunction
      })
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })
    await vi.waitFor(() =>
      expect(ensureServerChatHistoryId).toHaveBeenCalledTimes(1)
    )

    mocks.store.serverChatId = "chat-b"
    rerender()
    historyLink.resolve("history-a")
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mocks.updatePageTitle).not.toHaveBeenCalled()
    expect(mocks.store.setServerChatLoadState).not.toHaveBeenCalledWith(
      "loaded"
    )
    expect(notification.error).not.toHaveBeenCalled()
  })
  it("does not overwrite the visible selected path when a delayed display load follows an explicit choice", async () => {
    let epoch = 0
    const list = deferred<any[]>()
    mocks.listChatMessages.mockReturnValue(list.promise)
    mocks.selection = { fence: () => { const captured = epoch; return () => captured === epoch }, getCurrent: () => ({ owner: { kind: "native", conversation_id: "chat-a" }, capture: { status: "captured" } }) }
    function Surface() {
      const [rows, setRows] = React.useState<any[]>([])
      mocks.renderedMessages = rows
      mocks.setMessages.mockImplementation(setRows)
      useServerChatLoader({ ensureServerChatHistoryId: async () => "mirror", notification: { error: vi.fn() }, t: ((key: string) => key) as any })
      return <><button onClick={() => { epoch++; setRows([{ id: "selected-a", message: "Selected answer A" }]) }}>Choose A</button><output>{rows.map(row => row.id + ":" + row.message).join("/")}</output></>
    }
    render(<Surface />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    fireEvent.click(screen.getByText("Choose A"))
    await act(async () => { list.resolve([{ id: "latest-b", role: "assistant", content: "Wrong latest B", timestamp: "2026-01-01" }]); await vi.runOnlyPendingTimersAsync() })
    expect(screen.getByRole("status").textContent).toBe("selected-a:Selected answer A")
  })

})

it("keeps an ancestor toolbar loader from fetching or overwriting the selected surface", async () => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  mocks.selection = null
  mocks.renderedMessages = [{ id: "chosen", message: "Chosen answer", isBot: true }]
  const ensureServerChatHistoryId = vi.fn()
  renderHook(() => useServerChatLoader({
    ensureServerChatHistoryId,
    notification: { error: vi.fn() },
    t: ((key: string) => key) as TFunction,
    enabled: false
  }))
  await act(async () => { await vi.advanceTimersByTimeAsync(300) })
  expect(mocks.listChatMessages).not.toHaveBeenCalled()
  expect(mocks.setMessages).not.toHaveBeenCalled()
  expect(mocks.setHistory).not.toHaveBeenCalled()
  expect(ensureServerChatHistoryId).not.toHaveBeenCalled()
  vi.useRealTimers()
})
