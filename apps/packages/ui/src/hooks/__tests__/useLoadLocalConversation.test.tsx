import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { LocalChatList } from "@/components/Common/ChatSidebar/LocalChatList"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useLoadLocalConversation } from "../useLoadLocalConversation"

const io = vi.hoisted(() => ({
  scopeSignal: new AbortController(),
  readMessages: vi.fn(),
  readInfo: vi.fn(),
  readModelPreference: vi.fn(),
  readPrompt: vi.fn(),
  readFiles: vi.fn(),
  notifyError: vi.fn(),
  deps: null as Parameters<typeof useLoadLocalConversation>[0] | null
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "bob" },
    scopeSignal: io.scopeSignal.signal, scopeInvalidatedSignal: io.scopeSignal.signal,
    release: vi.fn()
  })
}))
vi.mock("@/db/dexie/chat", () => ({ pageAssistDatabase: { getChatHistoriesPaginated: async () => ({ histories: [{ id: "owned", title: "Owned local chat", createdAt: Date.now() }], hasMore: false, totalCount: 1 }) }, PageAssistDatabase: class {
  getChatHistory(id: string) { return io.readMessages(id) }
  getHistoryInfo(id: string) { return io.readInfo(id) }
} }))
vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: (rows: unknown[]) => rows,
  formatToMessage: (rows: unknown[]) => rows,
  getPromptById: (id: string) => io.readPrompt(id),
  getSessionFiles: (id: string) => io.readFiles(id),
  getHistoriesWithMetadata: async () => new Map(),
  deleteByHistoryId: vi.fn(), deleteHistoriesByDateRange: vi.fn(), updateHistory: vi.fn(), pinHistory: vi.fn(), getFullChatData: vi.fn(), restoreChat: vi.fn()
}))
vi.mock("@/services/model-settings", () => ({
  lastUsedChatModelEnabled: () => io.readModelPreference()
}))
vi.mock("antd", async importOriginal => ({ ...await importOriginal<typeof import("antd")>(), message: { error: (value: unknown) => io.notifyError(value) } }))
vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => ({ ...io.deps, historyId: null, clearChat: vi.fn() }) }))
vi.mock("@/store/model", () => ({ useStoreChatModelSettings: () => ({ setSystemPrompt: io.deps?.setSystemPrompt }) }))
vi.mock("@/hooks/useConnectionState", () => ({ useIsConnected: () => false }))
vi.mock("@/hooks/useUndoNotification", () => ({ useUndoNotification: () => ({ showUndoNotification: vi.fn() }) }))
vi.mock("@/components/Common/confirm-danger", () => ({ useConfirmDanger: () => vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, options?: string | { defaultValue?: string }) => typeof options === "string" ? options : options?.defaultValue ?? key }) }))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const options = { t: (key: string) => key, errorLogPrefix: "Local read failed", errorDefaultMessage: "Could not load local chat" }
const setup = () => {
  const state = { serverId: "server-before" as string | null, historyId: "before", messages: [] as unknown[], history: [] as unknown[], model: "before", promptId: "before" as string | null, prompt: "before", files: [] as unknown[] }
  const deps = {
    setServerChatId: (id: string | null) => { state.serverId = id },
    setHistoryId: (id: string) => { state.historyId = id },
    setHistory: (rows: unknown[]) => { state.history = rows },
    setMessages: (rows: unknown[]) => { state.messages = rows },
    setSelectedModel: (id: string) => { state.model = id },
    setSelectedSystemPrompt: (id: string | null) => { state.promptId = id },
    setSystemPrompt: (text: string) => { state.prompt = text },
    setContextFiles: (files: unknown[]) => { state.files = files }
  }
  io.deps = deps
  return { ...renderHook(() => useLoadLocalConversation(deps, options)), state }
}
beforeEach(() => {
  vi.clearAllMocks()
  for (const read of [io.readMessages, io.readInfo, io.readModelPreference, io.readPrompt, io.readFiles]) read.mockReset()
  io.scopeSignal = new AbortController()
  usePlaygroundSessionStore.setState({ restoreRevision: 0 })
  document.title = "Current page"
  io.readMessages.mockImplementation(async (id: string) => [{ id, content: `Message ${id}` }])
  io.readInfo.mockImplementation(async (id: string) => ({ id, title: `Title ${id}`, server_scope_key: '["http://chat.test","multi-user","manual",null,"bob",null]', model_id: "saved-model", last_used_prompt: { prompt_id: "saved-prompt", prompt_content: "" } }))
  io.readModelPreference.mockResolvedValue(true)
  io.readPrompt.mockResolvedValue({ id: "saved-prompt", content: "Saved prompt" })
  io.readFiles.mockResolvedValue([{ id: "saved-file" }])
})

describe("local conversation load ownership", () => {
  it.each(['["http://chat.test","multi-user","manual",null,"alice",null]', undefined])("rejects foreign or unowned local history before publishing (%s)", async owner => {
    io.readInfo.mockResolvedValue({ id: "alice", title: "ALICE SECRET", server_scope_key: owner, last_used_prompt: { prompt_content: "ALICE PIRATE" } })
    io.readMessages.mockResolvedValue([{ content: "ALICE PRIVATE" }])
    const { result, state, unmount } = setup()
    let restored = true
    await act(async () => { restored = await result.current("alice") })
    expect({ restored, state, title: document.title }).toEqual({
      restored: false,
      state: { serverId: "server-before", historyId: "before", messages: [], history: [], model: "before", promptId: "before", prompt: "before", files: [] },
      title: "Current page"
    })
    unmount()
  })

  it("restores the complete current local selection", async () => {
    const { result, state, unmount } = setup()
    await act(async () => { await result.current("owned") })
    expect(state).toEqual({ serverId: null, historyId: "owned", messages: [{ id: "owned", content: "Message owned" }], history: [{ id: "owned", content: "Message owned" }], model: "saved-model", promptId: "saved-prompt", prompt: "Saved prompt", files: [{ id: "saved-file" }] })
    expect(document.title).toBe("Title owned")
    unmount()
  })

  it.each(["messages", "model", "prompt", "files"] as const)("stops publication after unmount during the %s read", async boundary => {
    const pending = deferred<unknown>()
    const boundaryRead = { messages: io.readMessages, model: io.readModelPreference, prompt: io.readPrompt, files: io.readFiles }[boundary]
    boundaryRead.mockReturnValueOnce(pending.promise)
    const { result, state, unmount } = setup()
    const loading = result.current("owned")
    await waitFor(() => expect(boundaryRead).toHaveBeenCalled())
    unmount()
    const atUnmount = JSON.parse(JSON.stringify(state))
    await act(async () => { pending.resolve(boundary === "model" ? true : boundary === "prompt" ? { id: "saved-prompt", content: "Saved prompt" } : []); await loading })
    expect(state).toEqual(atUnmount)
    expect(document.title).toBe("Current page")
  })

  it("keeps a replacement local selection when the older read finishes last", async () => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { result, state, unmount } = setup()
    const first = result.current("old")
    await waitFor(() => expect(io.readMessages).toHaveBeenCalledWith("old"))
    await act(async () => { await result.current("replacement") })
    await act(async () => { pending.resolve([{ content: "Late old message" }]); await first })
    expect(state.historyId).toBe("replacement")
    expect(state.messages).toEqual([{ id: "replacement", content: "Message replacement" }])
    expect(document.title).toBe("Title replacement")
    unmount()
  })

  it("honors the existing cancellation revision after another saved chat is selected", async () => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { result, state, unmount } = setup()
    const loading = result.current("old")
    await waitFor(() => expect(io.readMessages).toHaveBeenCalled())
    usePlaygroundSessionStore.getState().cancelPendingRestore()
    await act(async () => { pending.resolve([{ content: "Late old message" }]); await loading })
    expect(state.historyId).toBe("before")
    expect(document.title).toBe("Current page")
    unmount()
  })

  it("suppresses an obsolete read error after route teardown", async () => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const { result, unmount } = setup()
    const loading = result.current("old")
    await waitFor(() => expect(io.readMessages).toHaveBeenCalled())
    unmount()
    await act(async () => { pending.reject(new Error("Synthetic late read error")); await loading })
    expect(io.notifyError).not.toHaveBeenCalled()
    expect(error).not.toHaveBeenCalled()
  })

  it("cancels held results across the explicit principal A-to-B-to-A boundary", async () => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { result, state, unmount } = setup()
    const loading = result.current("old-A")
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    await act(async () => { pending.resolve([{ content: "Old A" }]); await loading })
    expect(state.historyId).toBe("before")
    unmount()
  })

  it("preserves a current local load across a same-owner rotation notification", async () => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { result, state, unmount } = setup()
    const loading = result.current("owned")
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } }))
    await act(async () => { pending.resolve([{ content: "Owned" }]); await loading })
    expect(state.historyId).toBe("owned")
    expect(state.prompt).toBe("Saved prompt")
    unmount()
  })

  it.each(["current", "unmounted", "failed"] as const)("reports only an accepted LocalChatList selection: %s", async outcome => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { state, unmount: unmountUnusedHook } = setup()
    unmountUnusedHook()
    const onSelectChat = vi.fn()
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={queryClient}><LocalChatList searchQuery="" selectedChatId={null} onSelectChat={onSelectChat} /></QueryClientProvider>)
    fireEvent.click(await screen.findByRole("button", { name: "Owned local chat" }))
    await waitFor(() => expect(io.readMessages).toHaveBeenCalledWith("owned"))
    if (outcome === "unmounted") view.unmount()
    if (outcome === "failed") vi.spyOn(console, "error").mockImplementation(() => undefined)
    await act(async () => {
      if (outcome === "failed") pending.reject(new Error("Synthetic read failure"))
      else pending.resolve([{ content: "Current local chat" }])
      await pending.promise.catch(() => undefined)
    })
    if (outcome === "current") {
      await waitFor(() => expect(onSelectChat).toHaveBeenCalledWith("owned"))
      expect(state.historyId).toBe("owned")
      expect(state.prompt).toBe("Saved prompt")
    } else {
      expect(onSelectChat).not.toHaveBeenCalled()
      expect(state.historyId).toBe("before")
      if (outcome === "failed") expect(io.notifyError).toHaveBeenCalled()
    }
    view.unmount()
    queryClient.clear()
  })
})
