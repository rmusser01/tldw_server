import type { HistoryInfo, Message } from "@/db/dexie/types"
import type { TFunction } from "i18next"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest"
const state = vi.hoisted(() => ({ histories: new Map<string, HistoryInfo>(), messages: new Map<string, Message>(), beforeMessagePut: vi.fn(), request: vi.fn(), profile: vi.fn(), selection: vi.fn(), controller: new AbortController() }))
function table<T extends { id: string }>(rows: Map<string, T>) { return {
  get: async (id: string) => rows.get(id),
  add: async (row: T) => { if (rows.has(row.id)) throw new Error("duplicate primary key"); rows.set(row.id, structuredClone(row)); return row.id },
  put: async (row: T) => { await state.beforeMessagePut(row); rows.set(row.id, structuredClone(row)); return row.id },
  update: async (id: string, changes: Partial<T>) => { if (!rows.has(id)) return 0; rows.set(id, { ...rows.get(id)!, ...changes }); return 1 },
  where: (field: string) => ({ equals: (value: unknown) => ({ toArray: async () => [...rows.values()].filter(row => (row as Record<string, unknown>)[field] === value) }) })
} }
vi.mock("@/db/dexie/schema", () => ({ db: {
  chatHistories: table(state.histories), messages: table(state.messages), modelNickname: {}, sessionFiles: {},
  transaction: async (_mode: string, _tables: unknown[], operation: (transaction: { abort: () => void }) => Promise<unknown>) => operation({ abort: vi.fn() })
} }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: state.request, bgStream: vi.fn() }))
vi.mock("@/services/chat-settings", () => ({ syncChatSettingsForServerChat: async () => null }))
let selectionRevision = 0
vi.mock("@/hooks/useSelectedAssistant", () => ({ getSelectedAssistantOperationRevision: () => selectionRevision, waitForSelectedAssistantCommit: async () => undefined, useSelectedAssistant: () => [null, (...args: unknown[]) => { selectionRevision++; return state.selection(...args) }] }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
  scopeKey: "scope-A", scopeSignal: signal, scopeInvalidatedSignal: state.controller.signal,
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "A" }, release: vi.fn()
}) }))
import { useServerChatLoader } from "../chat/useServerChatLoader"
import { useServerChatHistoryId } from "../chat/useServerChatHistoryId"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { formatToMessage, formatToChatHistory } from "@/db/dexie/helpers"
const t = ((_key: string, options?: { defaultValue?: string }) => options?.defaultValue || "Cedar") as TFunction
const notification = { error: vi.fn() }
const mountLoader = () => renderHook(() => {
  const store = useStoreMessageOption()
  const { ensureServerChatHistoryId } = useServerChatHistoryId({ serverChatId: store.serverChatId, historyId: store.historyId, setHistoryId: store.setHistoryId, temporaryChat: store.temporaryChat, t })
  useServerChatLoader({ ensureServerChatHistoryId, notification, t })
})
const remote = [
  { id: "question", sender: "User", content: "When does Cedar open?", timestamp: "2026-09-15T09:00:00Z", version: 1 },
  { id: "answer", sender: "Cedar Guide", content: "Cedar opens at 08:30.", timestamp: "2026-09-15T09:01:00Z", version: 1 }
]
const mirrorRows = () => [...state.messages.values()].filter(row => row.history_id === useStoreMessageOption.getState().historyId)
describe("real adapter → loader → mirror → formatter reload", () => {
  beforeEach(() => {
    state.histories.clear(); state.messages.clear(); state.controller = new AbortController(); notification.error.mockClear(); state.selection.mockClear()
    vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockResolvedValue({ serverUrl: "http://chat.test", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "A" }))}.signature` })
    vi.spyOn(tldwClient, "getCharacter").mockImplementation(state.profile)
    state.beforeMessagePut.mockReset().mockResolvedValue(undefined)
    state.profile.mockResolvedValue({ id: 4, name: "Cedar Guide" })
    state.request.mockImplementation(async ({ path }) => path.includes("/messages") ? { messages: remote } : { id: "cedar", title: "Cedar chat", character_id: 4, assistant_kind: "character", assistant_id: "4", source: "webui-character-chat", scope_type: "global" })
    useStoreMessageOption.setState({ historyId: "legacy", serverChatId: "cedar", serverChatLoadState: "idle", messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantId: null, serverChatAssistantKind: null, streaming: false, isProcessing: false, temporaryChat: false })
    usePlaygroundSessionStore.getState().clearSession()
    usePlaygroundSessionStore.getState().saveSession({ historyId: "legacy", serverChatId: "cedar", scopeKey: "scope-A" })
    state.histories.set("legacy", { id: "legacy", title: "Cedar", server_chat_id: "cedar", is_rag: false, createdAt: 1 })
  })
  afterEach(() => vi.restoreAllMocks())
  it.each([true, false])("recovers an old partial mirror and survives repeated reload (greeting %s)", async greeting => {
    const old = [ ...(greeting ? [{ id: "greeting", history_id: "legacy", name: "Cedar", role: "assistant", content: "Welcome", messageType: "character:greeting", images: [], createdAt: 1 }] : []),
      { id: "question", history_id: "legacy", name: "You", role: "user", content: remote[0].content, images: [], createdAt: Date.parse(remote[0].timestamp) } ]
    old.forEach(row => state.messages.set(row.id, row))
    useStoreMessageOption.setState({ messages: formatToMessage(old), history: formatToChatHistory(old) })
    let view = mountLoader()
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
    expect(useStoreMessageOption.getState().messages.some(row => row.role === "assistant" && row.message === remote[1].content && row.serverMessageId === "answer")).toBe(true)
    expect(state.histories.get("legacy").server_scope_key).toBeTruthy()
    for (const persisted of mirrorRows()) {
      const visible = useStoreMessageOption.getState().messages.find(row => row.serverMessageId === persisted.serverMessageId)
      expect(visible?.id).toBe(persisted.id)
    }
    for (let reload = 0; reload < 2; reload++) {
      view.unmount()
      const saved = mirrorRows()
      act(() => useStoreMessageOption.setState({ messages: formatToMessage(saved), history: formatToChatHistory(saved), serverChatLoadState: "idle", serverChatMetaLoaded: false }))
      view = mountLoader()
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
      expect(useStoreMessageOption.getState().messages.filter(row => row.serverMessageId === "answer")).toHaveLength(1)
    }
    expect(mirrorRows().filter(row => row.serverMessageId === "answer")).toHaveLength(1)
    view.unmount()
  })
  it("retains a real unsynced local draft while adding the missing server reply", async () => {
    const draft = { id: "draft", history_id: "legacy", name: "You", role: "user", content: "Unsent local thought", images: [], createdAt: Date.now() }
    state.messages.set(draft.id, draft)
    useStoreMessageOption.setState({ messages: formatToMessage([draft]) })
    const view = mountLoader()
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
    expect(useStoreMessageOption.getState().messages.map(row => row.message)).toContain("Unsent local thought")
    expect(state.messages.get("draft").content).toBe("Unsent local thought")
    expect(useStoreMessageOption.getState().messages.filter(row => row.id === "draft")).toHaveLength(1)
    expect(useStoreMessageOption.getState().messages.map(row => row.serverMessageId)).toContain("answer")
    view.unmount()
  })

  it.each(["newer-answer", "unsynced-draft"])("publishes the owned persistent %s on a fresh server selection", async kind => {
    state.histories.set("legacy", { ...state.histories.get("legacy")!, server_scope_key: JSON.stringify(["http://chat.test", "multi-user", "manual", null, "A", null]) })
    const stored: Message = kind === "newer-answer"
      ? { id: "local-answer", history_id: "legacy", name: "Cedar", role: "assistant", content: "Newer acknowledged local answer", images: [], createdAt: Date.now(), serverMessageId: "answer", serverMessageVersion: 3 }
      : { id: "draft", history_id: "legacy", name: "You", role: "user", content: "Unsent owned draft", images: [], createdAt: Date.now() }
    state.messages.set(stored.id, stored)
    useStoreMessageOption.setState({ historyId: null, messages: [], history: [] })
    const view = mountLoader()
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
    const result = useStoreMessageOption.getState()
    expect(result.messages.find(message => message.id === stored.id)).toMatchObject({ message: stored.content, serverMessageVersion: stored.serverMessageVersion })
    expect(result.history.map(message => message.content)).toContain(stored.content)
    expect(state.messages.get(stored.id)?.content).toBe(stored.content)
    view.unmount()
  })

  it("keeps typing made during owned mirror reconciliation in both visible and inference history", async () => {
    state.histories.set("legacy", { ...state.histories.get("legacy")!, server_scope_key: JSON.stringify(["http://chat.test", "multi-user", "manual", null, "A", null]) })
    state.messages.set("local-answer", { id: "local-answer", history_id: "legacy", name: "Cedar", role: "assistant", content: "Stored version three", images: [], createdAt: Date.now(), serverMessageId: "answer", serverMessageVersion: 3 })
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    state.beforeMessagePut.mockImplementation(row => row.id === "local-answer" ? pending : undefined)
    useStoreMessageOption.setState({ historyId: null, messages: [], history: [] })
    const view = mountLoader()
    await waitFor(() => expect(state.beforeMessagePut).toHaveBeenCalled())
    act(() => useStoreMessageOption.getState().setMessages(current => current.map(message => message.serverMessageId === "answer" ? { ...message, message: "Typing while restoring" } : message)))
    await act(async () => { finish(); await pending })
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
    expect(useStoreMessageOption.getState().messages.find(message => message.serverMessageId === "answer")).toMatchObject({ id: "local-answer", message: "Typing while restoring" })
    expect(useStoreMessageOption.getState().history.map(message => message.content)).toContain("Typing while restoring")
    view.unmount()
  })

  it.each(["history", "stream", "account"])("does not publish owned persistent rows after %s changes during storage", async change => {
    state.histories.set("legacy", { ...state.histories.get("legacy")!, server_scope_key: JSON.stringify(["http://chat.test", "multi-user", "manual", null, "A", null]) })
    state.messages.set("local-answer", { id: "local-answer", history_id: "legacy", name: "Cedar", role: "assistant", content: "Owned stored answer", images: [], createdAt: Date.now(), serverMessageId: "answer", serverMessageVersion: 3 })
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    state.beforeMessagePut.mockImplementation(row => row.id === "local-answer" ? pending : undefined)
    useStoreMessageOption.setState({ historyId: null, messages: [], history: [] })
    const view = mountLoader()
    await waitFor(() => expect(state.beforeMessagePut).toHaveBeenCalled())
    act(() => {
      if (change === "history") useStoreMessageOption.getState().setHistoryId("replacement", { preserveServerChatId: true })
      if (change === "stream") useStoreMessageOption.setState({ streaming: true })
      if (change === "account") state.controller.abort()
      useStoreMessageOption.setState({ messages: [{ id: "replacement", name: "You", isBot: false, role: "user", message: "Current work" }], history: [{ role: "user", content: "Current work" }] })
    })
    await act(async () => { finish(); await pending; await new Promise(resolve => setTimeout(resolve, 0)) })
    expect(useStoreMessageOption.getState().messages.map(message => message.message)).toEqual(["Current work"])
    expect(useStoreMessageOption.getState().history.map(message => message.content)).toEqual(["Current work"])
    view.unmount()
  })

  it("preserves typing during settings sync before mirror reconciliation begins", async () => {
    state.histories.set("legacy", { ...state.histories.get("legacy")!, server_scope_key: JSON.stringify(["http://chat.test", "multi-user", "manual", null, "A", null]) })
    state.messages.set("local-answer", { id: "local-answer", history_id: "legacy", name: "Cedar", role: "assistant", content: "Stored version three", images: [], createdAt: Date.now(), serverMessageId: "answer", serverMessageVersion: 3 })
    const settings = await import("@/services/chat-settings")
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    const sync = vi.spyOn(settings, "syncChatSettingsForServerChat").mockImplementation(async () => { await pending; return null })
    useStoreMessageOption.setState({ historyId: null, messages: [], history: [] })
    const view = mountLoader()
    await waitFor(() => expect(sync).toHaveBeenCalledWith(expect.objectContaining({ historyId: "legacy" })))
    act(() => useStoreMessageOption.getState().setMessages(current => current.map(message => message.serverMessageId === "answer" ? { ...message, message: "Typing during settings sync" } : message)))
    await act(async () => { finish(); await pending })
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
    expect(useStoreMessageOption.getState().messages.find(message => message.serverMessageId === "answer")).toMatchObject({ id: "local-answer", message: "Typing during settings sync" })
    expect(useStoreMessageOption.getState().history.map(message => message.content)).toContain("Typing during settings sync")
    view.unmount()
  })
})
