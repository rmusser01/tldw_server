// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { StrictMode } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentChat } from "../useDocumentChat"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"

const io = vi.hoisted(() => ({
  loadSnapshot: vi.fn(), getHistory: vi.fn(), getHistoryInfo: vi.fn(), getFull: vi.fn(), saveHistory: vi.fn(), saveMessage: vi.fn(), release: vi.fn(),
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }
}))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: (...args: unknown[]) => io.loadSnapshot(...args) }))
vi.mock("@/store/connection", () => ({ useConnectionStore: (select: (state: unknown) => unknown) => select({ state: { isConnected: true, mode: "normal" } }) }))
vi.mock("@/db/dexie/helpers", () => ({
  getHistoryByDocId: (...args: unknown[]) => io.getHistory(...args), getFullChatData: (...args: unknown[]) => io.getFull(...args),
  saveHistory: (...args: unknown[]) => io.saveHistory(...args), saveMessage: (...args: unknown[]) => io.saveMessage(...args)
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({ runChatPersistenceTransaction: async (_signal: AbortSignal, operation: () => Promise<unknown>) => operation() }))
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class { getHistoryInfo = io.getHistoryInfo } }))

describe("document Chat fallback ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    io.getHistory.mockResolvedValue(null)
    io.getFull.mockResolvedValue(null)
    io.getHistoryInfo.mockResolvedValue({ id: "existing-alice", server_scope_key: serverChatMirrorOwnerKey({ requestScope: io.requestScope as never }) })
    io.saveHistory.mockResolvedValue({ id: "saved-alice-document" })
    io.saveMessage.mockResolvedValue({ id: "saved-message" })
    io.loadSnapshot.mockReset().mockImplementation(async () => {
      const controller = new AbortController()
      return { requestScope: io.requestScope, scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: io.release }
    })
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null, selectedKnowledge: null, ragMediaIds: null, ragSources: ["media_db"] })
  })

  const mountWithUnpersistedMessage = async () => {
    const view = renderHook(({ id }) => useDocumentChat(id), { initialProps: { id: 1 } })
    await waitFor(() => expect(io.getHistory).toHaveBeenCalled())
    await act(async () => { await Promise.resolve() })
    act(() => useStoreMessageOption.setState({
      messages: [{ id: "temp_alice", name: "You", role: "user", message: "Alice document question", isBot: false, sources: [] }],
      history: [{ role: "user", content: "Alice document question" }], historyId: null
    }))
    return view
  }

  it("stamps fallback history with the verified source owner", async () => {
    const view = await mountWithUnpersistedMessage()
    act(() => view.rerender({ id: 2 }))
    await waitFor(() => expect(io.saveHistory).toHaveBeenCalledWith(
      "Alice document question", true, "web-ui", "document:1", undefined, io.requestScope
    ))
    expect(io.getHistory).toHaveBeenCalledWith("document:1", io.requestScope)
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
    io.saveHistory.mockResolvedValue({ id: "saved-bob-document" })
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
    act(() => { window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
    await act(async () => { view.rerender({ id: 1 }); await Promise.resolve() })
    expect(useStoreMessageOption.getState().messages).toEqual([])
    view.unmount()
    expect(useStoreMessageOption.getState().messages).toEqual([])
  })

  it("does not publish a pending document load after leaving the workspace", async () => {
    const baseline = [{ id: "baseline", name: "You", role: "user" as const, message: "Ordinary Chat", isBot: false, sources: [] }]
    useStoreMessageOption.setState({ messages: baseline })
    let finish!: (history: { id: string }) => void
    io.getHistory.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    io.getFull.mockResolvedValue({ historyInfo: { id: "document-history" }, messages: [{ id: "doc-message", role: "user", content: "Old document question" }] })
    const view = renderHook(() => useDocumentChat(1))
    await waitFor(() => expect(io.getHistory).toHaveBeenCalled())
    view.unmount()
    await act(async () => { finish({ id: "document-history" }); await Promise.resolve() })
    expect(useStoreMessageOption.getState().messages).toEqual(baseline)
  })

  it("restarts document restoration after StrictMode effect cleanup", async () => {
    io.getHistory.mockResolvedValue({ id: "owned-document" })
    io.getFull.mockResolvedValue({ historyInfo: { id: "owned-document" }, messages: [{ id: "owned-message", role: "user", content: "Owned document question" }] })
    const view = renderHook(() => useDocumentChat(1), { wrapper: StrictMode })
    await waitFor(() => expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Owned document question"))
    view.unmount()
  })
})
