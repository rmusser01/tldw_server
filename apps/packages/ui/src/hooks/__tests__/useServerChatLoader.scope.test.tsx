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
    getChat: vi.fn(),
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

vi.mock("@/hooks/chat/useHistorySelection", async importOriginal => ({...(await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>()), useHistorySelectionContext: () => mocks.selection }))
vi.mock("@/db/dexie/history-selection", () => ({ensureLocalProfileId: async () => "profile", loadHistoryBookmark: async () => null, saveHistoryBookmark: async () => {}, loadHistoryTurnRecoveries: async () => []}))
vi.mock("@/db/dexie/fork-operations", () => ({findForkCandidate: async () => null, loadForkOperations: async () => []}))
vi.mock("@/services/chat-history-selection", () => ({captureHistorySnapshot: async (owner: any, view: any) => ({status: "legacy_review_required", code: "legacy_review_required", view, snapshot: {version: 1, owner_key: owner.owner_key, conversation_id: owner.conversation_id, nodes: [], source_digest: "source", storage_context_digest: "storage", fences: {}, interpretation_status: {kind: "legacy_review_required"}}})}))
import {useHistorySelection} from "@/hooks/chat/useHistorySelection"

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
    getChat: mocks.getChat,
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
it("holds native fork owner qualification before any ambient settings import and uses scoped metadata", async () => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  const held = deferred<boolean>()
  let ready = false
  mocks.store.serverChatId = "child"
  mocks.store.serverChatAssistantKind = null
  mocks.store.serverChatCharacterId = null
  mocks.store.serverChatMetaLoaded = true
  mocks.selection = {fence: () => () => true, settingsMode: () => ready ? "fork" : "pending",
    loadConversation: vi.fn(() => held.promise), getCurrent: () => ({owner: {kind: "native", conversation_id: "child", validate_lease: () => true},
      capture: {status: "captured"}, forkSettings: {authorNote: "server only"}})}
  renderHook(() => useServerChatLoader({ensureServerChatHistoryId: vi.fn(), notification: {error: vi.fn()}, t: ((key: string) => key) as any}))
  await act(async () => {await vi.advanceTimersByTimeAsync(300)})
  expect(mocks.syncChatSettingsForServerChat).not.toHaveBeenCalled()
  expect(mocks.initialize).not.toHaveBeenCalled()
  expect(mocks.listChatMessages).not.toHaveBeenCalled()
  await act(async () => {ready = true; held.resolve(true); await vi.runOnlyPendingTimersAsync()})
  expect(mocks.syncChatSettingsForServerChat).not.toHaveBeenCalled()
  expect(mocks.initialize).not.toHaveBeenCalled()
  expect(mocks.listChatMessages).not.toHaveBeenCalled()
  vi.useRealTimers()
})

it("real qualified ordinary legacy controller lets the loader hydrate settings before ancestry review", async () => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  mocks.store.serverChatId = "legacy"
  mocks.store.serverChatAssistantKind = null
  mocks.store.serverChatCharacterId = null
  mocks.store.serverChatMetaLoaded = true
  mocks.store.temporaryChat = false
  mocks.renderedMessages = []
  mocks.initialize.mockResolvedValue(undefined)
  mocks.listChatMessages.mockResolvedValue([])
  mocks.syncChatSettingsForServerChat.mockResolvedValue(null)
  const mounted = renderHook(() => {
    const controller = useHistorySelection()
    mocks.selection = controller
    useServerChatLoader({ensureServerChatHistoryId: vi.fn(), notification: {error: vi.fn()}, t: ((key: string) => key) as any})
    return controller
  })
  await act(async () => {await mounted.result.current.open({kind: "native", owner_key: "owner", conversation_id: "legacy", validate_lease: () => true} as any)})
  await act(async () => {await vi.advanceTimersByTimeAsync(300)})
  expect(mounted.result.current.status).toBe("legacy_review_required")
  expect(mocks.syncChatSettingsForServerChat).toHaveBeenCalledWith({historyId: null, serverChatId: "legacy", scope: undefined, allowScratchFallback: false})
  expect(mocks.listChatMessages).toHaveBeenCalledOnce()
  mounted.unmount()
  vi.useRealTimers()
})

it.each(['scope', 'metadata missing', 'messages missing', 'late scope'] as const)('invalidates only the matching H1 owner after %s rejection', async scenario => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  mocks.store.serverChatId = 'rejected'
  mocks.store.serverChatMetaLoaded = scenario === 'messages missing'
  mocks.store.serverChatAssistantKind = null
  mocks.store.serverChatCharacterId = null
  mocks.store.temporaryChat = false
  mocks.renderedMessages = []
  mocks.initialize.mockResolvedValue(undefined)
  mocks.syncChatSettingsForServerChat.mockResolvedValue(null)
  const response = deferred<any>()
  mocks.getChat.mockImplementation(() => scenario === 'late scope' ? response.promise : scenario === 'metadata missing' ? Promise.reject(Object.assign(new Error('not found'), {status: 404})) : Promise.resolve({id: 'rejected', scope_type: 'workspace', workspace_id: 'foreign'}))
  mocks.listChatMessages.mockImplementation(() => scenario === 'messages missing' ? Promise.reject(Object.assign(new Error('not found'), {status: 404})) : Promise.resolve([]))
  const notification = {error: vi.fn()}
  const mounted = renderHook(() => {
    const control = useHistorySelection()
    mocks.selection = control
    useServerChatLoader({ensureServerChatHistoryId: vi.fn(), notification, t: ((key: string) => key) as any})
    return control
  })
  await act(async () => { await mounted.result.current.open({kind: 'native', owner_key: 'owner', conversation_id: 'rejected', validate_lease: () => true} as any) })
  await act(async () => { await vi.advanceTimersByTimeAsync(300) })
  if (scenario === 'late scope') {
    await act(async () => { await mounted.result.current.open({kind: 'native', owner_key: 'owner', conversation_id: 'later', validate_lease: () => true} as any); response.resolve({id: 'rejected', scope_type: 'workspace', workspace_id: 'foreign'}) })
    expect(mounted.result.current.getCurrent().owner).toMatchObject({kind: 'native', conversation_id: 'later'})
    expect(mocks.store.setServerChatLoadState).not.toHaveBeenCalledWith('failed')
  } else {
    expect(mounted.result.current.getCurrent().owner).toEqual({kind: 'unavailable', code: scenario === 'scope' ? 'server_chat_scope_mismatch' : 'server_chat_not_found'})
    expect(mocks.store.setServerChatLoadState).toHaveBeenCalledWith('failed')
  }
  expect(mocks.store.setServerChatId).not.toHaveBeenCalledWith(null)
  mounted.unmount()
  vi.useRealTimers()
})
