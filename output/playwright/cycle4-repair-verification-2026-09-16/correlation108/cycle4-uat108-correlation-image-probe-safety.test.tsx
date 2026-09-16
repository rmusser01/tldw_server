import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"
// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import type { TFunction } from "i18next"
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

  it.each(["normal", "persona", "character"] as const)("keeps canonical raw reads limited to %s presentation", async kind => {
    mocks.store.serverChatAssistantKind = kind === "normal" ? null : kind
    mocks.store.serverChatCharacterId = kind === "character" ? 4 : null
    mocks.listChatMessages.mockImplementation(async (_id, params) => [{ id: "user", role: "user", content: params.render_placeholders === "false" ? "Ask {{char}}" : "Ask Cedar", version: 1 }])
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-a"), notification: { error: vi.fn() }, t: ((_key: string) => "Cedar") as TFunction }))
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(mocks.listChatMessages.mock.calls[0][1].render_placeholders).toBe(kind === "character" ? "true" : "false")
    expect(mocks.setMessages.mock.calls[0][0][0].message).toBe(kind === "character" ? "Ask Cedar" : "Ask {{char}}")
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
    renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: vi.fn() as TFunction }))
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
    const { rerender } = renderHook(() => useServerChatLoader({ ensureServerChatHistoryId, notification: { error: vi.fn() }, t: vi.fn() as TFunction }))
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
})

import { mapServerChatMessagesToPlaygroundMessages } from "@/hooks/chat/useServerChatLoader"
import { reconcileServerChatMessages } from "@/db/dexie/server-chat-mirror"

it.each(["Question", ""])("private actual canonical image preserves unmatched local work: %s", text => {
 const local={id:"local-image-user",isBot:false,role:"user",name:"You",message:text,images:["data:image/png;base64,aGVsbG8="],sources:[]};
 const remote=mapServerChatMessagesToPlaygroundMessages({assistantName:"Assistant",characterId:null,serverMessages:[{id:"canonical-image-user",role:"user",content:text||"<Image attachment x1>",created_at:"2026-09-16T04:49:00Z",has_image:true,metadata_extra:{client_message_id:local.id}} as unknown as ServerChatMessage]});
 const result=reconcileServerChatMessages([local],remote);
 expect(result.filter(m=>m.role==="user")).toHaveLength(2);
 expect(result.find(m=>m.id===local.id)).toMatchObject({message:text,images:local.images});
 expect(result.find(m=>m.id===local.id)?.serverMessageId).toBeUndefined();
});
