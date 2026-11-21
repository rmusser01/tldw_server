// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { StrictMode } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentChat, type DocumentChatCache } from "../useDocumentChat"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import type { HistoryInfo, Message } from "@/db/dexie/types"

const io = vi.hoisted(() => ({
  loadSnapshot: vi.fn(), listHistories: vi.fn(), getHistoryInfo: vi.fn(), getFull: vi.fn(), saveHistory: vi.fn(), saveMessage: vi.fn(), release: vi.fn(),
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => io.loadSnapshot(...args)
}))
vi.mock("@/store/connection", () => ({ useConnectionStore: (select: (state: unknown) => unknown) => select({ state: { isConnected: true, mode: "normal" } }) }))
vi.mock("@/db/dexie/helpers", async (original) => ({
  ...await original<typeof import("@/db/dexie/helpers")>(),
  getFullChatData: (...args: unknown[]) => io.getFull(...args),
  getAllHistoriesByDocId: (...args: unknown[]) => io.listHistories(...args),
  saveHistory: (...args: unknown[]) => io.saveHistory(...args), saveMessage: (...args: unknown[]) => io.saveMessage(...args)
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({ runChatPersistenceTransaction: async (_signal: AbortSignal, operation: () => Promise<unknown>) => operation() }))
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class { getHistoryInfo = io.getHistoryInfo } }))

const operation: QuickIngestOperation = {
  authorityKey: "verified-account",
  authorityRevision: 1,
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" },
  signal: new AbortController().signal,
  isCurrent: () => true,
  assertCurrent: () => {}
}

const savedHistory = (id: string, overrides: Partial<HistoryInfo> = {}): HistoryInfo => ({
  id,
  title: id,
  createdAt: 1,
  is_rag: false,
  doc_id: "document:1",
  server_scope_key: serverChatMirrorOwnerKey(operation),
  ...overrides
})

const savedChat = (historyInfo: HistoryInfo, text: string) => ({
  historyInfo,
  messages: [{
    id: `message-${historyInfo.id}`, history_id: historyInfo.id, name: "Model", role: "assistant",
    createdAt: 5, content: text, images: [], serverMessageId: `server-message-${historyInfo.id}`
  } satisfies Message]
})

describe("document Chat fallback ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    io.listHistories.mockResolvedValue([])
    io.getFull.mockResolvedValue(null)
    io.getHistoryInfo.mockResolvedValue(savedHistory("existing-alice"))
    io.saveHistory.mockResolvedValue(savedHistory("saved-alice-document"))
    io.saveMessage.mockResolvedValue({ id: "saved-message" })
    io.loadSnapshot.mockReset().mockImplementation(async () => {
      const controller = new AbortController()
      return { requestScope: io.requestScope, scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: io.release }
    })
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null, selectedKnowledge: null, ragMediaIds: null, ragSources: ["media_db"] })
  })

  const mountWithUnpersistedMessage = async () => {
    const view = renderHook(({ id }) => useDocumentChat(id, operation), { initialProps: { id: 1 } })
    await waitFor(() => expect(io.listHistories).toHaveBeenCalled())
    await act(async () => { await Promise.resolve() })
    act(() => useStoreMessageOption.setState({
      messages: [{ id: "temp_alice", name: "You", role: "user", message: "Alice document question", isBot: false, sources: [] }],
      history: [{ role: "user", content: "Alice document question" }], historyId: null
    }))
    return view
  }

  it("prepares the active document's private scene for shared chat submission", async () => {
    const view = renderHook(({ id }) => useDocumentChat(id, operation), { initialProps: { id: 1 } })
    await waitFor(() => expect(view.result.current.isChatReady).toBe(true))

    const capture = view.result.current.capturePrivateScene?.(() => true)
    expect(capture?.isCurrent()).toBe(true)
    const send = await capture!.read()
    expect(send.requestScope).toEqual(io.requestScope)
    expect(send.scene).toMatchObject({
      target: { address: { kind: "fresh", id: "document:1" } },
      origin: "default"
    })

    view.rerender({ id: 2 })
    await waitFor(() => expect(view.result.current.isChatReady).toBe(true))
    expect(send.isCurrent()).toBe(false)
    const next = await view.result.current.capturePrivateScene!(() => true)!.read()
    expect(next.scene.target.key).not.toBe(send.scene.target.key)
    view.unmount()
    expect(next.isCurrent()).toBe(false)
  })

  it("waits for the document transcript and revokes a scene when its chat is cleared", async () => {
    let finish!: (histories: HistoryInfo[]) => void
    io.listHistories.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    io.getFull.mockResolvedValue(savedChat(savedHistory("owned-document"), "Owned document question"))
    const view = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(io.listHistories).toHaveBeenCalled())
    expect(view.result.current.isChatReady).toBe(false)
    expect(view.result.current.capturePrivateScene?.(() => true) ?? null).toBeNull()

    await act(async () => { finish([savedHistory("owned-document")]) })
    await waitFor(() => expect(view.result.current.isChatReady).toBe(true))
    const capture = view.result.current.capturePrivateScene!(() => true)!
    act(() => view.result.current.newConversation())
    expect(capture.isCurrent()).toBe(false)
    expect(view.result.current.capturePrivateScene!(() => true)?.isCurrent()).toBe(true)
  })

  it("stamps fallback history with the verified source owner", async () => {
    const view = await mountWithUnpersistedMessage()
    act(() => view.rerender({ id: 2 }))
    await waitFor(() => expect(io.saveHistory).toHaveBeenCalledWith(
      "Alice document question", true, "web-ui", "document:1", undefined, io.requestScope
    ))
    expect(io.listHistories).toHaveBeenCalledWith("document:1", io.requestScope)
    view.unmount()
  })

  it("does not write an old document snapshot under an account selected while verification resolves", async () => {
    const view = await mountWithUnpersistedMessage()
    io.loadSnapshot.mockImplementationOnce(async () => {
      usePlaygroundSessionStore.getState().cancelPendingRestore()
      const controller = new AbortController()
      return { requestScope: { ...io.requestScope, userId: "bob" }, scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: io.release }
    })
    await act(async () => { view.rerender({ id: 2 }); await Promise.resolve() })
    expect(io.saveHistory).not.toHaveBeenCalled()
    view.unmount()
  })

  it.each([undefined, "foreign-owner"])("rejects an existing history with owner %s", async (ownerKey) => {
    const view = await mountWithUnpersistedMessage()
    io.getHistoryInfo.mockResolvedValue({ id: "foreign-history", server_scope_key: ownerKey })
    act(() => useStoreMessageOption.setState({ historyId: "foreign-history" }))
    await act(async () => { view.rerender({ id: 2 }); await Promise.resolve() })
    expect(io.saveMessage).not.toHaveBeenCalled()
    view.unmount()
  })

  it("does not attach an old rejected save to the replacement account's document session", async () => {
    const view = await mountWithUnpersistedMessage()
    act(() => useStoreMessageOption.setState({ historyId: "existing-alice" }))
    let rejectOld!: (error: Error) => void
    io.loadSnapshot.mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectOld = reject }))
    await act(async () => { view.rerender({ id: 2 }); await Promise.resolve() })
    act(() => {
      usePlaygroundSessionStore.getState().cancelPendingRestore()
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    })
    await act(async () => { view.rerender({ id: 1 }); await Promise.resolve() })
    act(() => useStoreMessageOption.setState({
      messages: [{ id: "temp_bob", name: "You", role: "user", message: "Bob question", isBot: false, sources: [] }],
      history: [{ role: "user", content: "Bob question" }], historyId: null
    }))
    const bobHistory = savedHistory("saved-bob-document")
    io.saveHistory.mockResolvedValue(bobHistory)
    io.getFull.mockResolvedValue(savedChat(bobHistory, "Bob question"))
    await act(async () => { view.rerender({ id: 2 }); await Promise.resolve() })
    await waitFor(() => expect(io.saveHistory).toHaveBeenCalledWith("Bob question", true, "web-ui", "document:1", undefined, io.requestScope))
    await act(async () => { rejectOld(new Error("old identity lookup failed")); await Promise.resolve() })
    await act(async () => { view.rerender({ id: 1 }); await Promise.resolve() })
    expect(useStoreMessageOption.getState().historyId).toBe("saved-bob-document")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Bob question")
    view.unmount()
  })

  it("discards document memory and baseline on an account change", async () => {
    const view = await mountWithUnpersistedMessage()
    await act(async () => { view.rerender({ id: 2 }); await Promise.resolve() })
    await waitFor(() => expect(view.result.current.isChatReady).toBe(true))
    const capture = view.result.current.capturePrivateScene(() => true)!
    act(() => { window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
    expect(capture.isCurrent()).toBe(false)
    await act(async () => { view.rerender({ id: 1 }); await Promise.resolve() })
    expect(useStoreMessageOption.getState().messages).toEqual([])
    view.unmount()
    expect(useStoreMessageOption.getState().messages).toEqual([])
  })

  it("does not publish a pending document load after leaving the workspace", async () => {
    const baseline = [{ id: "baseline", name: "You", role: "user" as const, message: "Ordinary Chat", isBot: false, sources: [] }]
    useStoreMessageOption.setState({ messages: baseline })
    let finish!: (histories: HistoryInfo[]) => void
    io.listHistories.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    io.getFull.mockResolvedValue(savedChat(savedHistory("document-history"), "Old document question"))
    const view = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(io.listHistories).toHaveBeenCalled())
    view.unmount()
    await act(async () => { finish([savedHistory("document-history")]); await Promise.resolve() })
    expect(useStoreMessageOption.getState().messages).toEqual(baseline)
  })

  it("restarts document restoration after StrictMode effect cleanup", async () => {
    io.listHistories.mockResolvedValue([savedHistory("owned-document")])
    io.getFull.mockResolvedValue(savedChat(savedHistory("owned-document"), "Owned document question"))
    const view = renderHook(() => useDocumentChat(1, operation), { wrapper: StrictMode })
    await waitFor(() => expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Owned document question"))
    view.unmount()
  })

  it("chooses the newest saved conversation, switches by ID, and keeps each draft and document context", async () => {
    const first = savedHistory("First topic", { createdAt: 2, server_chat_id: "server-first" })
    const second = savedHistory("Second topic")
    io.listHistories.mockResolvedValue([first, second])
    io.getFull.mockImplementation(async id => id === first.id
      ? savedChat(first, "First answer")
      : savedChat(second, "Second answer"))
    const { result } = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    expect(result.current.conversations.map(chat => chat.title)).toEqual([first.title, second.title])
    expect(result.current.messages[0]?.serverMessageId).toBe(`server-message-${first.id}`)
    expect(useStoreMessageOption.getState().serverChatId).toBe("server-first")

    const firstKey = result.current.activeConversationKey!
    act(() => {
      result.current.setInputValue("Draft for first topic")
      result.current.setRagEnabled(true)
      result.current.selectConversation(result.current.conversations[1]!.key)
    })
    await waitFor(() => expect(result.current.messages[0]?.message).toBe("Second answer"))
    expect(result.current.inputValue).toBe("")
    expect(result.current.ragEnabled).toBe(false)
    expect(result.current.ragMediaIds).toEqual([1])
    act(() => result.current.selectConversation(firstKey))
    await waitFor(() => expect(result.current.messages[0]?.message).toBe("First answer"))
    expect(result.current.inputValue).toBe("Draft for first topic")
    expect(result.current.ragEnabled).toBe(true)
    expect(result.current.messages).toHaveLength(1)
  })

  it("keeps the draft tab and its input when its first turn is saved", async () => {
    const old = savedHistory("Earlier topic")
    const fresh = savedHistory("New topic")
    io.listHistories.mockResolvedValue([old])
    io.getFull.mockResolvedValue(savedChat(old, "Earlier answer"))
    io.getHistoryInfo.mockResolvedValue(fresh)
    const { result } = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    const priorScene = result.current.capturePrivateScene(() => true)!
    act(() => result.current.newConversation())
    expect(priorScene.isCurrent()).toBe(false)
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(result.current.messages).toEqual([])
    const draftKey = result.current.activeConversationKey!
    act(() => result.current.setInputValue("Draft follow-up"))
    await act(async () => {
      useStoreMessageOption.setState({ historyId: fresh.id })
      await result.current.onHistorySaved(fresh.id)
    })
    expect(result.current.activeConversationKey).toBe(draftKey)
    expect(result.current.inputValue).toBe("Draft follow-up")
    expect(result.current.conversations).toEqual([
      { key: old.id, title: old.title },
      { key: draftKey, title: fresh.title }
    ])
  })

  it("reloads a stopped turn only in its originating tab after the sidebar remounts", async () => {
    const existing = savedHistory("Existing topic")
    const stopped = savedHistory("Stopped topic")
    io.listHistories.mockResolvedValue([existing])
    io.getFull.mockImplementation(async id => id === stopped.id
      ? savedChat(stopped, "Retained partial")
      : savedChat(existing, "Existing answer"))
    io.getHistoryInfo.mockResolvedValue(stopped)
    const cache: DocumentChatCache = new Map()
    const view = renderHook(() => useDocumentChat(1, operation, cache))
    await waitFor(() => expect(view.result.current.isChatReady).toBe(true))
    act(() => view.result.current.newConversation())
    const originatingKey = view.result.current.activeConversationKey!
    const recordSavedHistory = view.result.current.onHistorySaved
    const oldScene = view.result.current.capturePrivateScene(() => true)!
    let finish!: () => void
    act(() => {
      view.result.current.setInputValue("Unsent draft")
      view.result.current.trackTurn(() => new Promise<void>(resolve => { finish = resolve }))
      useStoreMessageOption.setState({
        messages: [{
          id: "temp_reply", name: "Model", role: "assistant", message: "Outdated stream buffer", isBot: true
        }]
      })
      view.result.current.selectConversation(existing.id)
    })
    await waitFor(() => expect(view.result.current.messages[0]?.message).toBe("Existing answer"))
    expect(oldScene.isCurrent()).toBe(false)
    expect(view.result.current.inputValue).toBe("")
    view.unmount()

    const reopened = renderHook(() => useDocumentChat(1, operation, cache))
    await waitFor(() => expect(reopened.result.current.isChatReady).toBe(true))
    act(() => reopened.result.current.selectConversation(originatingKey))
    expect(reopened.result.current.conversationStatus).toBe("loading")
    await act(async () => {
      await recordSavedHistory(stopped.id)
      finish()
    })
    await waitFor(() => expect(reopened.result.current.messages[0]?.message).toBe("Retained partial"))
    expect(reopened.result.current.conversations.at(-1)).toEqual({ key: originatingKey, title: stopped.title })
    expect(reopened.result.current.inputValue).toBe("Unsent draft")
    expect(io.saveMessage).not.toHaveBeenCalled()
  })

  it("ignores a slow read after a faster conversation takes over", async () => {
    const first = savedHistory("First topic")
    const second = savedHistory("Second topic")
    io.listHistories.mockResolvedValue([first, second])
    const slow = savedChat(first, "Old first answer")
    let finish!: (value: typeof slow) => void
    io.getFull.mockImplementation(async id => id === first.id
      ? new Promise(resolve => { finish = resolve })
      : savedChat(second, "Second answer"))
    const { result } = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(io.getFull).toHaveBeenCalledWith(first.id))
    act(() => result.current.selectConversation(second.id))
    await waitFor(() => expect(result.current.messages[0]?.message).toBe("Second answer"))
    await act(async () => finish(slow))
    expect(result.current.messages[0]?.message).toBe("Second answer")
  })

  it("keeps the completed transcript in a temporary chat when a later reply is stopped", async () => {
    const { result } = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    const source = result.current.activeConversationKey!
    const completed = { id: "existing", name: "Model", role: "assistant" as const, message: "Finished reply", isBot: true }
    let finish!: () => void
    act(() => {
      useStoreMessageOption.setState({ historyId: "temp", messages: [completed] })
      result.current.trackTurn(() => new Promise<void>(resolve => { finish = resolve }))
      useStoreMessageOption.setState({
        messages: [completed, { ...completed, id: "pending", message: "Unfinished" }]
      })
      result.current.newConversation()
    })
    await act(async () => finish())
    act(() => result.current.selectConversation(source))
    await waitFor(() => expect(result.current.messages).toEqual([completed]))
    expect(io.saveMessage).not.toHaveBeenCalled()
  })

  it("restores distinct saved replies with a shared parent while deduplicating server message identities", async () => {
    const history = savedHistory("Regenerated chat")
    const chat = savedChat(history, "Original")
    const originalReply = { ...chat.messages[0]!, parent_message_id: "question" }
    io.listHistories.mockResolvedValue([history])
    io.getFull.mockResolvedValue({
      historyInfo: history,
      messages: [
        originalReply,
        { ...originalReply, id: "same-receipt" },
        { ...originalReply, id: "new-reply", serverMessageId: "new-receipt", content: "New reply" }
      ]
    })
    const { result } = renderHook(() => useDocumentChat(1, operation))
    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    expect(result.current.messages.map(reply => reply.message)).toEqual(["Original", "New reply"])
  })

  it.each([{ doc_id: "document:2" }, { server_scope_key: "foreign-owner" }])(
    "refuses to restore an exact history whose header has %j",
    async invalid => {
      const history = savedHistory("Changed history")
      io.listHistories.mockResolvedValue([history])
      io.getFull.mockResolvedValueOnce(savedChat({ ...history, ...invalid }, "Private text"))
        .mockResolvedValueOnce(savedChat(history, "Restored answer"))
      const { result } = renderHook(() => useDocumentChat(1, operation))
      await waitFor(() => expect(result.current.conversationStatus).toBe("error"))
      expect(result.current.messages).toEqual([])
      expect(result.current.isChatReady).toBe(false)
      act(() => result.current.retryConversation())
      await waitFor(() => expect(result.current.messages[0]?.message).toBe("Restored answer"))
      expect(result.current.isChatReady).toBe(true)
    }
  )
})
