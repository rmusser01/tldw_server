import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"
// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { t as translate, type TFunction } from "i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const store = {
    messages: [],
    historyId: "history-a",
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
    streaming: false,
    getCharacter: vi.fn(),
    getHistoriesWithMetadata: vi.fn(),
    initialize: vi.fn(),
    ensureConfigForRequest: vi.fn(),
    listChatMessages: vi.fn(),
    saveMessage: vi.fn(),
    reconcileServerChatMirror: vi.fn(),
    setHistory: vi.fn(),
    setIsLoading: vi.fn(),
    setMessages: vi.fn(),
    setSelectedAssistant: vi.fn(),
    syncChatSettingsForServerChat: vi.fn(),
    updatePageTitle: vi.fn()
  }
})

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => ({
    messages: [],
    streaming: mocks.streaming,
    isProcessing: false,
    setHistory: mocks.setHistory,
    setMessages: mocks.setMessages,
    setIsLoading: mocks.setIsLoading
  })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign((
    selector: (state: typeof mocks.store) => unknown
  ) => selector(mocks.store), { getState: () => mocks.store })
}))

let selectionRevision = 0
vi.mock("@/hooks/useSelectedAssistant", () => ({ getSelectedAssistantOperationRevision: () => selectionRevision, waitForSelectedAssistantCommit: async () => undefined,
  useSelectedAssistant: () => [null, (...args: unknown[]) => { selectionRevision++; return mocks.setSelectedAssistant(...args) }]
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    ensureConfigForRequest: mocks.ensureConfigForRequest,
    listChatMessages: mocks.listChatMessages,
    getCharacter: mocks.getCharacter
  }
}))

vi.mock("@/db/dexie/helpers", async importOriginal => ({
  ...await importOriginal<typeof import("@/db/dexie/helpers")>(),
  getHistoriesWithMetadata: mocks.getHistoriesWithMetadata,
  saveMessage: mocks.saveMessage
}))
vi.mock("@/db/dexie/server-chat-mirror", async importOriginal => ({
  ...await importOriginal<typeof import("@/db/dexie/server-chat-mirror")>(),
  reconcileServerChatMirror: mocks.reconcileServerChatMirror
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
    scopeKey: "scope-A", scopeSignal: signal, scopeInvalidatedSignal: signal,
    requestScope: { config: { serverUrl: "http://server", authMode: "multi-user" }, userId: "A" }, release: vi.fn()
  })
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: mocks.syncChatSettingsForServerChat
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: mocks.updatePageTitle
}))

import { mapServerChatMessagesToPlaygroundMessages, useServerChatLoader } from "@/hooks/chat/useServerChatLoader"
import { reconcileServerChatMessages } from "@/db/dexie/server-chat-mirror"

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
    mocks.store.setServerChatId.mockReset()
    mocks.streaming = false
    mocks.getCharacter.mockResolvedValue(null)
    mocks.store.serverChatAssistantKind = "character"
    mocks.store.serverChatCharacterId = null
    mocks.store.serverChatId = "chat-a"
    mocks.store.serverChatTitle = "Chat A"
    mocks.store.serverChatMetaLoaded = true
    mocks.store.temporaryChat = false
    mocks.reconcileServerChatMirror.mockResolvedValue({ localIds: new Map(), rows: [] })
    mocks.initialize.mockResolvedValue(undefined)
    mocks.ensureConfigForRequest.mockResolvedValue({ serverUrl: "http://server", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "A" }))}.signature` })
    mocks.listChatMessages.mockResolvedValue([])
    mocks.getHistoriesWithMetadata.mockResolvedValue(
      new Map([["history-a", { messageCount: 1 }]])
    )
    mocks.syncChatSettingsForServerChat.mockResolvedValue(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it.each([401, 403, 404])("removes cached private messages and title when the canonical read returns %s", async status => {
    mocks.listChatMessages.mockRejectedValue(Object.assign(new Error(`HTTP ${status}`), { status }))
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn(), notification: { error: vi.fn() }, t: ((_key: string) => "Error") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.setMessages).toHaveBeenCalledWith([])
    expect(mocks.setHistory).toHaveBeenCalledWith([])
    expect(mocks.store.setServerChatTitle).toHaveBeenCalledWith(null)
  })

  it.each([401, 403, 404])("finishes loading and clears the selected server identifier after HTTP %s", async status => {
    mocks.store.setServerChatId.mockImplementationOnce((id) => { mocks.store.serverChatId = id })
    mocks.listChatMessages.mockRejectedValue(Object.assign(new Error(`HTTP ${status}`), { status }))
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn(), notification: { error: vi.fn() }, t: ((_key: string) => "Error") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.store.serverChatId).toBeNull()
    expect(mocks.setIsLoading).toHaveBeenLastCalledWith(false)
  })

  it.each([401, 403])("does not clear a newer chat selection after a stale HTTP %s rejection", async status => {
    const response = deferred<ServerChatMessage[]>()
    mocks.listChatMessages.mockReturnValue(response.promise)
    mocks.store.setServerChatId.mockImplementationOnce(id => { mocks.store.serverChatId = id })
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn(), notification: { error: vi.fn() }, t: ((_key: string) => "Error") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    mocks.store.serverChatId = "chat-b"
    await act(async () => { response.reject(Object.assign(new Error("Access denied"), { status })) })
    expect(mocks.store.serverChatId).toBe("chat-b")
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(mocks.store.setServerChatTitle).not.toHaveBeenCalled()
  })

  it.each([401, 403])("ignores HTTP %s from an invalidated principal", async status => {
    const response = deferred<ServerChatMessage[]>()
    mocks.listChatMessages.mockReturnValue(response.promise)
    mocks.store.setServerChatId.mockImplementationOnce(id => { mocks.store.serverChatId = id })
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn(), notification: { error: vi.fn() }, t: ((_key: string) => "Error") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    mocks.ensureConfigForRequest.mockResolvedValue({ serverUrl: "http://server", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "B" }))}.signature` })
    await act(async () => {
      window.dispatchEvent(new CustomEvent("tldw:config-updated"))
      await Promise.resolve()
      await Promise.resolve()
    })
    await act(async () => { response.reject(Object.assign(new Error("Access denied"), { status })) })
    expect(mocks.store.serverChatId).toBe("chat-a")
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(mocks.setHistory).not.toHaveBeenCalled()
    expect(mocks.store.setServerChatTitle).not.toHaveBeenCalled()
  })

  it.each(["normal", "persona", "character"] as const)("keeps canonical raw reads limited to %s presentation", async kind => {
    mocks.store.serverChatAssistantKind = kind === "normal" ? null : kind
    mocks.store.serverChatCharacterId = kind === "character" ? 4 : null
    mocks.listChatMessages.mockImplementation(async (_id, params) => [{ id: "user", role: "user", content: params.render_placeholders === "false" ? "Ask {{char}}" : "Ask Cedar", version: 1 }])
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-a"), notification: { error: vi.fn() }, t: ((_key: string) => "Cedar") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.listChatMessages.mock.calls[0][1].render_placeholders).toBe(kind === "character" ? "true" : "false")
    expect(mocks.setMessages.mock.calls[0][0][0].message).toBe(kind === "character" ? "Ask Cedar" : "Ask {{char}}")
  })

  it.each(["", "Question with an image"])("preserves an unacknowledged attachment when the actual mapper lacks image bytes: %s", content => {
    const local = { id: "local-image", role: "user" as const, isBot: false, name: "You", message: content,
      images: ["data:image/png;base64,aW1hZ2U="], sources: [] }
    // The existing standard listing/client shape carries no attachment bytes.
    const mapped = mapServerChatMessagesToPlaygroundMessages({
      serverMessages: [{ id: "canonical-image", role: "user", content: content || "<Image attachment x1>",
        created_at: "2026-09-16T00:00:00Z", version: 1, metadata_extra: { client_message_id: local.id } }],
      assistantName: "Assistant", characterId: null
    })
    const merged = reconcileServerChatMessages([local], mapped)
    expect(merged.find(row => row.id === local.id)).toEqual(local)
    expect(merged.find(row => row.id === local.id)?.serverMessageId).toBeUndefined()
    expect(merged.find(row => row.serverMessageId === "canonical-image")?.id).not.toBe(local.id)
  })

  it("reconciles a nonempty local mirror using acknowledged server identity", async () => {
    mocks.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "08:30", version: 2 }])
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-a")
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: ((_key: string) => "Cedar") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.reconcileServerChatMirror).toHaveBeenCalledWith(expect.objectContaining({ historyId: "history-a", chatId: "chat-a", messages: [expect.objectContaining({ serverMessageId: "answer", serverMessageVersion: 2 })] }))
    expect(ensureServerChatHistoryId.mock.calls[0][3]).toMatchObject({ requestScope: { userId: "A" } })
  })

  it.each(["invalidated", "different-owner", "valid-rotation"])("revalidates canonical %s while an owned response is held", async kind => {
    const response = deferred<ServerChatMessage[]>()
    mocks.listChatMessages.mockReturnValue(response.promise)
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-a")
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: ((_key: string) => "Cedar") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    const signal = mocks.listChatMessages.mock.calls[0][2].signal as AbortSignal
    mocks.ensureConfigForRequest.mockResolvedValue({ serverUrl: "http://server", authMode: "multi-user", accessToken: kind === "invalidated" ? undefined : `new.${btoa(JSON.stringify({ sub: kind === "different-owner" ? "B" : "A" }))}.signature` })
    await act(async () => {
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } }))
      await Promise.resolve(); await Promise.resolve()
    })
    expect(signal.aborted).toBe(kind !== "valid-rotation")
    response.resolve([{ id: "late", role: "assistant", content: "Owned reply", version: 1 }])
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    expect(mocks.reconcileServerChatMirror).toHaveBeenCalledTimes(kind === "valid-rotation" ? 1 : 0)
  })

  it("keeps an active streamed row and skips completed snapshot writes", async () => {
    mocks.streaming = true
    mocks.listChatMessages.mockResolvedValue([{ id: "old", role: "assistant", content: "Server snapshot", version: 1 }])
    const ensureServerChatHistoryId = vi.fn()
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: translate }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.setMessages).not.toHaveBeenCalled()
    expect(ensureServerChatHistoryId).not.toHaveBeenCalled()
    expect(mocks.reconcileServerChatMirror).not.toHaveBeenCalled()
  })

  it("does not apply a late Cedar profile after the current selection changes to Robot", async () => {
    const profile = deferred<{ id: number; name: string }>()
    const rows = deferred<ServerChatMessage[]>()
    mocks.store.serverChatCharacterId = 4
    mocks.getCharacter.mockReturnValueOnce(profile.promise).mockResolvedValue({ id: 5, name: "Robot" })
    mocks.listChatMessages.mockReturnValueOnce(rows.promise).mockResolvedValue([])
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history")
    const { rerender } = renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: translate }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.getCharacter.mock.calls[0][1]).toMatchObject({ requestScope: { userId: "A" } })
    const oldSignal = mocks.getCharacter.mock.calls[0][1].signal as AbortSignal
    mocks.store.serverChatId = "chat-b"
    mocks.store.serverChatCharacterId = 5
    rerender()
    await act(async () => {
      profile.resolve({ id: 4, name: "Cedar" })
      rows.resolve([])
      await vi.advanceTimersByTimeAsync(200)
    })
    expect(oldSignal.aborted).toBe(true)
    expect(mocks.setSelectedAssistant.mock.calls.some(([selection]) => selection?.id === "4")).toBe(false)
    expect(mocks.setSelectedAssistant.mock.calls.some(([selection]) => selection?.id === "5")).toBe(true)
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
  it.each(["found", "missing", "rejected"] as const)(
    "canonical loader %s catalog leaves another plain draft plain",
    async (outcome) => {
      const { resolveEffectiveAssistantState } =
        await import("@/hooks/chat/effective-assistant-state")
      const { preserveAssistantSelectionMode } =
        await import("@/types/assistant-selection")
      let sharedSelection: Parameters<
        typeof resolveEffectiveAssistantState
      >[0]["draftSelection"] = null
      mocks.setSelectedAssistant.mockImplementation(async (next) => {
        sharedSelection = preserveAssistantSelectionMode(next, sharedSelection)
      })
      mocks.store.serverChatCharacterId = 42
      mocks.store.serverChatAssistantId = "42"
      if (outcome === "found")
        mocks.getCharacter.mockResolvedValue({ id: 42, name: "Saved card" })
      if (outcome === "missing") mocks.getCharacter.mockResolvedValue(null)
      if (outcome === "rejected")
        mocks.getCharacter.mockRejectedValue(new Error("catalog unavailable"))
      const view = renderHook(() =>
        useServerChatLoader({
          ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-a"),
          notification: { error: vi.fn() },
          t: translate
        })
      )
      try {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(200)
        })
        expect(mocks.getCharacter).toHaveBeenCalled()
        expect(mocks.setSelectedAssistant).toHaveBeenCalled()
        expect(
          resolveEffectiveAssistantState({
            tracked: {
              assistantKind: "character",
              assistantId: "42",
              characterId: 42
            },
            draftSelection: sharedSelection
          }).mode
        ).toBe("tracked_character")
        expect(
          resolveEffectiveAssistantState({
            tracked: {
              assistantKind: null,
              assistantId: null,
              characterId: null
            },
            draftSelection: sharedSelection
          }).mode
        ).toBe("plain")
      } finally {
        view.unmount()
      }
    }
  )

  it.each(["tracked", "overlay"] as const)(
    "successful same-ID catalog refresh preserves primary explicit %s draft intent",
    async (mode) => {
      const { resolveEffectiveAssistantState } =
        await import("@/hooks/chat/effective-assistant-state")
      const { preserveAssistantSelectionMode } =
        await import("@/types/assistant-selection")
      let sharedSelection: Parameters<
        typeof resolveEffectiveAssistantState
      >[0]["draftSelection"] = {
        kind: "character",
        id: "42",
        name: "Explicitly picked card",
        metadata: { selectionMode: mode }
      }
      mocks.setSelectedAssistant.mockImplementation(async (next) => {
        sharedSelection = preserveAssistantSelectionMode(next, sharedSelection)
      })
      mocks.store.serverChatCharacterId = 42
      mocks.store.serverChatAssistantId = "42"
      mocks.getCharacter.mockResolvedValue({
        id: 42,
        name: "Latest card metadata"
      })
      const view = renderHook(() =>
        useServerChatLoader({
          ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-a"),
          notification: { error: vi.fn() },
          t: translate
        })
      )
      try {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(200)
        })
        expect(mocks.getCharacter).toHaveBeenCalled()
        expect(mocks.setSelectedAssistant).toHaveBeenCalled()
        expect(
          resolveEffectiveAssistantState({
            tracked: {
              assistantKind: null,
              assistantId: null,
              characterId: null
            },
            draftSelection: sharedSelection
          }).mode
        ).toBe(mode === "tracked" ? "tracked_character" : "overlay")
      } finally {
        view.unmount()
      }
    }
  )

})
