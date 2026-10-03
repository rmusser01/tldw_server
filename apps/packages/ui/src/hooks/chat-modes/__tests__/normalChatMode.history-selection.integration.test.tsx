import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatTldwOptions } from "@/models/ChatTldw"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { HistoryInfo, Message } from "@/db/dexie/types"

// Table-engine double only: creation, public sanitization, rereads, mounted
// selection, capture, admission and settlement use production implementations.
// The Critical browser journeys separately qualify native IndexedDB/reload.
const storage = vi.hoisted(() => {
  type Row = Record<string, unknown>
  const tables: Record<string, {
    rows: Map<string, Row>
    get: (id: unknown) => Promise<Row | undefined>
    add: (row: Row) => Promise<void>
    put: (row: Row) => Promise<void>
    bulkPut: (rows: Row[]) => Promise<void>
    update: (id: unknown, patch: Row) => Promise<void>
    toArray: () => Promise<Row[]>
    where: (field: string) => { equals: (value: unknown) => { toArray: () => Promise<Row[]> } }
  }> = {}
  const control = { beforeHistoryAdd: undefined as (() => Promise<void>) | undefined }
  for (const name of ["chatHistories", "messages", "userSettings", "sessionFiles", "compareStates", "historySelections", "historyProjections", "forkOperations", "modelNickname", "prompts"]) {
    const rows = new Map<string, Row>()
    const key = (row: Row) => JSON.stringify(name === "historySelections"
      ? [row.profile_id, row.client_session_id, row.owner_key, row.conversation_id]
      : row.id)
    tables[name] = {
      rows,
      get: async id => structuredClone(rows.get(JSON.stringify(id))),
      add: async row => {
        if (name === "chatHistories") await control.beforeHistoryAdd?.()
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
      },
      put: async row => { rows.set(key(row), structuredClone(row)) },
      bulkPut: async values => { values.forEach(row => rows.set(key(row), structuredClone(row))) },
      update: async (id, patch) => { rows.set(JSON.stringify(id), { ...rows.get(JSON.stringify(id)), ...structuredClone(patch) }) },
      toArray: async () => structuredClone([...rows.values()]),
      where: field => ({ equals: value => ({ toArray: async () => structuredClone([...rows.values()].filter(row => row[field] === value)) }) })
    }
  }
  let tail = Promise.resolve()
  return {
    tables,
    control,
    db: {
      ...tables,
      transaction: (_mode: unknown, _tables: unknown, operation: (tx: { abort: () => void }) => Promise<unknown>) => {
        const result = tail.then(async () => {
          const copies = Object.values(tables).map(table => new Map(table.rows))
          try { return await operation({ abort: () => { throw new Error("aborted") } }) }
          catch (error) {
            Object.values(tables).forEach((table, index) => {
              table.rows.clear()
              copies[index].forEach((row, key) => table.rows.set(key, row))
            })
            throw error
          }
        })
        tail = result.then(() => {}, () => {})
        return result
      }
    }
  }
})
vi.mock("@/db/dexie/schema", () => ({ db: storage.db }))
const prompt = vi.hoisted(() => ({ scope: {
  config: { serverUrl: "https://chat.test", authMode: "multi-user" as const },
  userId: "alice"
} }))
vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => prompt.scope,
  subscribeToServicePromptConfigChanges: () => () => {}
}))
vi.mock("~/services/tldw-server", async original => ({
  ...await original<typeof import("~/services/tldw-server")>(),
  systemPromptForNonRagOption: async () => ""
}))
vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return { pageAssistModel: async (options: ChatTldwOptions) => {
    await preparation.beforeModel()
    return new ChatTldw({ ...options, streaming: true, apiProvider: "openai" })
  } }
})

const preparation = vi.hoisted(() => ({ beforeModel: vi.fn() }))

import { normalChatMode } from "../normalChatMode"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { saveHistory } from "@/db/dexie/helpers"
import { saveMessageOnSuccess, saveMessageOnError } from "@/hooks/chat-helper"
import { captureHistorySnapshot } from "@/services/chat-history-selection"
import { tldwChat } from "@/services/tldw"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreMessageOption } from "@/store/option"
import { buildChatToolFilterState } from "@/utils/chat-tools"

const ACCOUNT = '["https://chat.test","multi-user","manual",null,"alice",null]'
const snapshot = (): ServicePromptSnapshot => ({
  scopeKey: "alice-scope", requestScope: structuredClone(prompt.scope),
  capability: "supported", definitions: {},
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: () => {}
})
const send = (controller: HistorySelectionController, options: {
  historyId?: string
  snapshot?: ServicePromptSnapshot
  originIsCurrent?: () => boolean
  setHistoryId?: (id: string) => void
} = {}) => normalChatMode("First question", "", false, [], [], new AbortController().signal, {
  selectedModel: "test", selectedSystemPrompt: "", useOCR: false,
  currentChatModelSettings: {}, servicePromptSnapshot: options.snapshot ?? snapshot(),
  historySelection: { controller, originIsCurrent: options.originIsCurrent ?? (() => true) },
  historyId: options.historyId ?? null,
  setHistoryId: options.setHistoryId ?? (() => {}),
  setMessages: () => {}, setHistory: () => {}, setIsProcessing: () => {},
  setStreaming: () => {}, setAbortController: () => {},
  saveMessageOnSuccess, saveMessageOnError,
  userMessageId: "first-user", assistantMessageId: "first-answer"
})
const publicHistory = (id: string): HistoryInfo => ({ id, title: "Imported", createdAt: 1, is_rag: false, message_source: "web-ui" })
const savedMessages = async (): Promise<Message[]> => await storage.tables.messages.toArray() as Message[]

beforeEach(() => {
  Object.values(storage.tables).forEach(table => table.rows.clear())
  storage.control.beforeHistoryAdd = undefined
  prompt.scope.userId = "alice"
  preparation.beforeModel.mockReset()
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  useStoreMessageOption.setState({ selectedModel: "test", toolChoice: "none" })
  vi.spyOn(tldwChat, "streamMessage").mockImplementation(async function* () { yield "First answer" })
})

describe("normal Chat owned history creation and H1 admission", () => {
  it.each(["loading", "inactive settings", "MCP health"])("allows a saved send after unrelated %s publication", async publication => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    preparation.beforeModel.mockImplementation(() => {
      if (publication === "loading") useMcpToolsStore.getState().setToolsLoading(true)
      else if (publication === "inactive settings") useStoreChatModelSettings.getState().updateScopedSetting("unused:other", "temperature", 0.8)
      else useMcpToolsStore.setState({ healthState: "healthy", toolCatalog: "background refresh", toolModules: ["module-updated"] })
    })
    await act(async () => {
      expect(await send(mounted.result.current, { historyId: created.id })).toEqual({ status: "submitted" })
    })
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

  it.each(["settings", "model", "tools", "enabled tools", "account"])("rejects a saved send when its %s changes during preparation", async changed => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    const scoped = snapshot()
    if (changed === "enabled tools") {
      useStoreMessageOption.setState({ toolChoice: "auto" })
      useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{
        name: "lookup", canExecute: true, inputSchema: { type: "object", properties: {} }
      }] }).chatTools })
    }
    preparation.beforeModel.mockImplementation(() => {
      if (changed === "settings") useStoreChatModelSettings.getState().setTemperature(0.8)
      if (changed === "model") useStoreMessageOption.getState().setSelectedModel("changed-model")
      if (changed === "tools") useStoreMessageOption.getState().setToolChoice("auto")
      if (changed === "enabled tools") useMcpToolsStore.setState({ chatTools: [] })
      if (changed === "account") {
        const controller = new AbortController()
        controller.abort()
        Object.assign(scoped, { scopeInvalidatedSignal: controller.signal })
      }
    })
    await act(async () => {
      expect((await send(mounted.result.current, { historyId: created.id, snapshot: scoped })).status).not.toBe("submitted")
    })
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it("persists the captured account before first local admission and reopens the exact selected turn", async () => {
    const mounted = renderHook(() => useHistorySelection())
    let historyId = ""
    await act(async () => {
      expect(await send(mounted.result.current, { setHistoryId: id => { historyId = id } })).toEqual({ status: "submitted" })
    })
    const persisted = await new PageAssistDatabase().getHistoryInfo(historyId)
    expect(persisted).toMatchObject({ server_scope_key: ACCOUNT, local_owner_key: expect.stringMatching(/^local-history-v1:/) })
    expect(persisted.server_chat_id).toBeUndefined()
    expect((await savedMessages()).map(({ id, history_id, role, content, parent_message_id }) => ({ id, history_id, role, content, parent_message_id }))).toEqual([
      { id: "first-user", history_id: historyId, role: "user", content: "First question", parent_message_id: null },
      { id: "first-answer", history_id: historyId, role: "assistant", content: "First answer", parent_message_id: "first-user" }
    ])
    expect(mounted.result.current.capture?.status).toBe("captured")
    const reference = mounted.result.current.getReference()!
    expect(mounted.result.current.view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
    mounted.unmount()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => { await reopened.result.current.loadConversation({ historyId }, reference) })
    expect(reopened.result.current.status).toBe("ready")
    expect(reopened.result.current.capture?.status === "captured" && reopened.result.current.capture.rows.map(row => row.id)).toEqual(["first-user", "first-answer"])
    expect(await savedMessages()).toHaveLength(2)
    const dispatched = vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest
    expect(dispatched).toMatchObject({ model: "test", save_to_db: false, messages: [{ role: "user", content: "First question" }] })
    expect(dispatched).not.toHaveProperty("conversation_id")
  })

  it("rereads separately supplied ownership while public and import writes discard supplied authority", async () => {
    const created = await saveHistory("Scoped", false, "web-ui", undefined, undefined, snapshot().requestScope)
    expect((await new PageAssistDatabase().getHistoryInfo(created.id)).server_scope_key).toBe(ACCOUNT)
    const db = new PageAssistDatabase()
    await db.addChatHistory({ ...publicHistory("public"), server_scope_key: ACCOUNT, local_owner_key: "forged" })
    expect(await db.getHistoryInfo("public")).toEqual(publicHistory("public"))
    await db.importChatHistoryV2([{ history: { ...publicHistory("import"), server_scope_key: ACCOUNT, local_owner_key: "forged" }, messages: [] }])
    expect(await db.getHistoryInfo("import")).toMatchObject(publicHistory("import"))
    expect((await db.getHistoryInfo("import")).server_scope_key).toBeUndefined()
    expect((await db.getHistoryInfo("import")).local_owner_key).toBeUndefined()
  })

  it.each(["missing", "foreign"])("does not admit a %s account history", async ownership => {
    const row = ownership === "missing"
      ? await saveHistory("Unowned", false, "web-ui")
      : await saveHistory("Foreign", false, "web-ui", undefined, undefined, { ...snapshot().requestScope, userId: "bob" })
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: row.id }) })
    expect(mounted.result.current.error).toBe("owner_conversation_mismatch")
    await act(async () => { await expect(send(mounted.result.current, { historyId: row.id })).rejects.toThrow("owner_conversation_mismatch") })
    expect(await savedMessages()).toEqual([])
  })

  it("rejects an expired captured account before admitting or generating a user row", async () => {
    const invalidated = new AbortController()
    invalidated.abort()
    const expired = { ...snapshot(), scopeInvalidatedSignal: invalidated.signal }
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { expect(await send(mounted.result.current, { snapshot: expired })).toEqual({ status: "failed", errorMessage: "Request cancelled" }) })
    expect(await savedMessages()).toEqual([])
  })

  it("does not adopt a creation that finishes after navigation", async () => {
    let finish!: () => void
    const gate = new Promise<void>(resolve => { finish = resolve })
    let entered = false
    storage.control.beforeHistoryAdd = async () => { entered = true; await gate }
    const mounted = renderHook(() => useHistorySelection())
    let adopted = ""
    let current = true
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setHistoryId: id => { adopted = id }, originIsCurrent: () => current }) })
    await waitFor(() => expect(entered).toBe(true))
    await act(async () => {
      current = false
      mounted.result.current.reset()
      finish()
      await expect(pending).rejects.toThrow("stale_selection")
    })
    expect(adopted).toBe("")
    expect(mounted.result.current.getCurrent().owner).toBeNull()
    expect(await savedMessages()).toEqual([])
  })

  it("cannot reuse local account authority after an A to B to A boundary", async () => {
    const row = await saveHistory("Owned", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: row.id }) })
    expect(mounted.result.current.status).toBe("ready")
    const owner = mounted.result.current.owner!
    const view = mounted.result.current.view!
    act(() => {
      prompt.scope.userId = "bob"
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      prompt.scope.userId = "alice"
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    expect(mounted.result.current.error).toBe("request_config_scope_changed")
    await expect(captureHistorySnapshot(owner, view, "send")).rejects.toMatchObject({ code: "request_config_scope_changed" })
    await act(async () => { await expect(send(mounted.result.current, { historyId: row.id })).rejects.toThrow("request_config_scope_changed") })
    expect(await savedMessages()).toEqual([])
  })
})
