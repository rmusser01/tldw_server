import React from "react"
import { MemoryRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { LocalChatList } from "@/components/Common/ChatSidebar/LocalChatList"
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useLoadLocalConversation } from "../useLoadLocalConversation"
import { HistorySelectionProvider, useHistorySelection, useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { db } from "@/db/dexie/schema"
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

const io = vi.hoisted(() => ({
  scopeSignal: new AbortController(),
  profileSnapshot: { status: "ready", profileId: "profile", epoch: 0 },
  selection: null as ReturnType<typeof useHistorySelectionContext>,
  readMessages: vi.fn(),
  readInfo: vi.fn(),
  readModelPreference: vi.fn(),
  readPrompt: vi.fn(),
  readFiles: vi.fn(),
  notifyError: vi.fn(),
  deps: null as Parameters<typeof useLoadLocalConversation>[0] | null
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "bob" }),
  subscribeToServicePromptConfigChanges: () => () => {},
  loadServicePromptSnapshot: async () => ({
    requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "bob" },
    scopeSignal: io.scopeSignal.signal, scopeInvalidatedSignal: io.scopeSignal.signal,
    release: vi.fn()
  })
}))
// MemoryDexie has no IndexedDB subscription; the stored profile is checked by the real reader.
vi.mock("dexie-react-hooks", () => ({ useLiveQuery: () => io.profileSnapshot }))
vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import("@/components/Option/Playground/__tests__/harness/memory-dexie")
  return { db: createMemoryDexie() }
})
vi.mock("@/db/dexie/chat", async importOriginal => {
  const actual = await importOriginal<typeof import("@/db/dexie/chat")>()
  class FixtureDatabase extends actual.PageAssistDatabase {
    getChatHistory(id: string) { return io.readMessages(id) }
    getHistoryInfo(id: string) { return io.readInfo(id) }
    async getOnDeviceChatPreview(...args: Parameters<import("@/db/dexie/chat").PageAssistDatabase["getOnDeviceChatPreview"]>) {
      await io.readMessages(args[0])
      return super.getOnDeviceChatPreview(...args)
    }
  }
  return { ...actual, PageAssistDatabase: FixtureDatabase, pageAssistDatabase: new FixtureDatabase() }
})
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
vi.mock("@/hooks/chat/useSelectedModel", () => ({ useSelectedModel: () => ({ setSelectedModel: io.deps?.setSelectedModel }) }))
vi.mock("@/hooks/chat/useClearChat", () => ({ useClearChat: () => vi.fn() }))
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
function SelectionProbe() {
  io.selection = useHistorySelectionContext()
  return null
}
const memory = () => db as unknown as MemoryDexie
let originalStore: ReturnType<typeof useStoreMessageOption.getState>
let originalSession: ReturnType<typeof usePlaygroundSessionStore.getState>
beforeEach(async () => {
  originalStore = useStoreMessageOption.getState()
  originalSession = usePlaygroundSessionStore.getState()
  useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true)
  useStoreMessageOption.getState().startNewChatDraft(false)
  usePlaygroundSessionStore.setState(usePlaygroundSessionStore.getInitialState(), true)
  memory().resetAll()
  await memory().userSettings.put({ id: "main", history_profile_id: io.profileSnapshot.profileId })
  useStoreMessageOption.setState({ temporaryChat: false, historyId: null, serverChatId: null, messages: [], history: [] })
  vi.spyOn(window, "confirm").mockReturnValue(false)
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

afterEach(() => {
  cleanup()
  useStoreMessageOption.setState(originalStore, true)
  usePlaygroundSessionStore.setState(originalSession, true)
  vi.restoreAllMocks()
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


})

describe("local selection preparation", () => {
  it.each(["model", "prompt", "files"] as const)("keeps the complete source when %s preparation fails", async boundary => {
    const { result, state, unmount } = setup()
    const source = structuredClone(state)
    const failedRead = { model: io.readModelPreference, prompt: io.readPrompt, files: io.readFiles }[boundary]
    failedRead.mockRejectedValueOnce(new Error("Synthetic preparation failure"))
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    await act(async () => { expect(await result.current("owned")).toBe(false) })
    expect(state).toEqual(source)
    expect(document.title).toBe("Current page")
    unmount()
  })
  it.each(["model", "prompt", "files"] as const)("does not partially publish while %s preparation is held", async boundary => {
    const { result, state, unmount } = setup()
    const source = structuredClone(state)
    const pending = deferred<unknown>()
    const heldRead = { model: io.readModelPreference, prompt: io.readPrompt, files: io.readFiles }[boundary]
    heldRead.mockReturnValueOnce(pending.promise)
    const loading = result.current("owned")
    await waitFor(() => expect(heldRead).toHaveBeenCalled())
    expect(state).toEqual(source)
    unmount()
    await act(async () => { pending.resolve(boundary === "model" ? true : boundary === "prompt" ? { id: "saved-prompt", content: "Saved prompt" } : []); await loading })
    expect(state).toEqual(source)
  })
})
describe("accepted LocalChatList selections", () => {
  it.each(["current", "unmounted", "failed"] as const)("reports only an accepted LocalChatList selection: %s", async outcome => {
    const pending = deferred<unknown[]>()
    io.readMessages.mockReturnValueOnce(pending.promise)
    const { state, unmount: unmountUnusedHook } = setup()
    unmountUnusedHook()
    const onSelectChat = vi.fn()
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const owned = { id: "owned", title: "Owned local chat", createdAt: 1, is_rag: false,
      message_source: "web-ui", local_owner_key: "local-history-v1:profile",
      model_id: "saved-model", last_used_prompt: { prompt_id: "saved-prompt", prompt_content: "" } }
    await memory().chatHistories.put(owned)
    await memory().messages.put({ id: "saved-user", history_id: owned.id, role: "user", name: "You",
      content: "Current local chat", createdAt: 1, parent_message_id: null })
    io.readInfo.mockResolvedValue(owned)
    const stopObserving = useStoreMessageOption.subscribe((next, previous) => {
      if (next.historyId !== previous.historyId) state.historyId = next.historyId!
    })
    const view = render(<MemoryRouter><QueryClientProvider client={queryClient}><HistorySelectionProvider><SelectionProbe /><LocalChatList searchQuery="" selectedChatId={null} onSelectChat={onSelectChat} /></HistorySelectionProvider></QueryClientProvider></MemoryRouter>)
    try {
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
        // The selection settles after the read resolves; a slow runner can still see "idle" here.
        await waitFor(() => expect(io.selection?.getCurrent()).toMatchObject({ status: "ready" }))
        await waitFor(() => expect(onSelectChat).toHaveBeenCalledWith("owned"))
        expect(state.historyId).toBe("owned")
        expect(state.prompt).toBe("Saved prompt")
      } else {
        expect(onSelectChat).not.toHaveBeenCalled()
        expect(state.historyId).toBe("before")
        if (outcome === "failed") expect(io.notifyError).toHaveBeenCalled()
      }
    } finally {
      stopObserving()
      view.unmount()
      queryClient.clear()
    }
  })
})


it("keeps the owned Temporary source before local-history reads when selection is cancelled", async () => {
  const rows = [{ id: "temporary", isBot: false, name: "You", role: "user" as const, message: "Temporary question", sources: [] }]
  useStoreMessageOption.setState({ temporaryChat: true, historyId: "temp", messages: rows, history: [{ role: "user", content: "Temporary question" }] })
  const before = useStoreMessageOption.getState()
  const { result, state, unmount } = setup()
  await act(async () => expect(await result.current("owned")).toBe(false))
  expect(window.confirm).toHaveBeenCalledOnce()
  expect(io.readInfo).not.toHaveBeenCalled()
  expect(io.readMessages).not.toHaveBeenCalled()
  expect(useStoreMessageOption.getState()).toBe(before)
  expect(state.historyId).toBe("before")
  expect(document.title).toBe("Current page")
  unmount()
})

it("admits the complete local selection after a Temporary discard is accepted", async () => {
  useStoreMessageOption.setState({ temporaryChat: true, historyId: "temp", history: [{ role: "user", content: "Temporary question" }] })
  vi.mocked(window.confirm).mockReturnValue(true)
  const { result, state, unmount } = setup()
  await act(async () => expect(await result.current("owned")).toBe(true))
  expect(window.confirm).toHaveBeenCalledOnce()
  expect(state.historyId).toBe("owned")
  expect(state.messages).toEqual([{ id: "owned", content: "Message owned" }])
  unmount()
})


it("hands an actual ready local H1 receipt to the captured draft exactly once", async () => {
  const owned = { id: "handoff", title: "Handoff", createdAt: 1, is_rag: false,
    message_source: "web-ui", local_owner_key: "local-history-v1:profile" }
  await memory().chatHistories.put(owned)
  await memory().messages.put({ id: "handoff-user", history_id: owned.id, role: "user", name: "You",
    content: "Keep this question", createdAt: 1, parent_message_id: null })
  io.readInfo.mockResolvedValue(owned)
  const before = useStoreMessageOption.getState()
  const accept = captureChatDraftLoad("switch")
  const onLoaded = vi.fn(accept)
  const { result, unmount } = renderHook(() => useHistorySelection())
  try {
    let loaded = false
    await act(async () => {
      loaded = await result.current.loadConversation({ historyId: owned.id, isCurrent: accept.isCurrent }, null, onLoaded)
    })
    expect(result.current.getCurrent()).toMatchObject({ status: "ready", owner: { kind: "local", conversation_id: owned.id } })
    expect(onLoaded).toHaveBeenCalledOnce()
    expect(onLoaded.mock.results[0].value).toBe(true)
    expect(useStoreMessageOption.getState()).toMatchObject({
      chatDraftAddress: { kind: "local", id: owned.id, profileId: "profile" },
      chatDraftRevision: before.chatDraftRevision + 1
    })
    expect(accept.isCurrent()).toBe(true)
    expect(loaded).toBe(true)
  } finally { unmount() }
})


it.each(["current", "retired"] as const)("publishes an actual H1 local load only for a %s reader", async (readerState) => {
  const owned = { id: "caller-handoff", title: "Caller handoff", createdAt: 1, is_rag: false,
    message_source: "web-ui", local_owner_key: "local-history-v1:profile" }
  await memory().chatHistories.put(owned)
  await memory().messages.put({ id: "caller-user", history_id: owned.id, role: "user", name: "You",
    content: "Keep caller question", createdAt: 1, parent_message_id: null })
  io.readInfo.mockResolvedValue(owned)
  const pending = deferred<unknown[]>()
  io.readMessages.mockReturnValueOnce(pending.promise)
  let readerActive = true
  const state = useStoreMessageOption.getState()
  const { result, unmount } = renderHook(() => useLoadLocalConversation({
    setServerChatId: state.setServerChatId, setHistoryId: state.setHistoryId,
    setMessages: state.setMessages, setHistory: state.setHistory,
    setSelectedSystemPrompt: state.setSelectedSystemPrompt, setContextFiles: state.setContextFiles,
    setSelectedModel: vi.fn(), setSystemPrompt: vi.fn()
  }, options), { wrapper: ({ children }) => <HistorySelectionProvider>{children}</HistorySelectionProvider> })
  try {
    let loading!: Promise<boolean>
    act(() => {
      loading = result.current(owned.id, { profileId: "profile", serverScopeKey: null, isCurrent: () => readerActive })
    })
    await waitFor(() => expect(io.readMessages).toHaveBeenCalledWith(owned.id))
    if (readerState === "retired") readerActive = false
    let loaded = false
    await act(async () => { pending.resolve([]); loaded = await loading })
    expect({ loaded, historyId: useStoreMessageOption.getState().historyId,
      address: useStoreMessageOption.getState().chatDraftAddress, title: document.title }).toEqual(readerState === "current" ? {
      loaded: true, historyId: owned.id,
      address: { kind: "local", id: owned.id, profileId: "profile" }, title: owned.title
    } : { loaded: false, historyId: state.historyId, address: state.chatDraftAddress, title: "Current page" })
  } finally { readerActive = false; unmount() }
})
