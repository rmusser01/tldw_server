import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatTldwOptions } from "@/models/ChatTldw"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import * as servicePrompts from "@/services/service-prompts"
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
import { keepHistoryTurnRecovery } from "@/services/history-turn-keep"
import { loadHistoryTurnRecoveries } from "@/db/dexie/history-selection"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { saveHistory } from "@/db/dexie/helpers"
import { saveMessageOnSuccess, saveMessageOnError } from "@/hooks/chat-helper"
import { captureHistorySnapshot } from "@/services/chat-history-selection"
import { assertHistoryConversationTarget, type NativeHistoryOwnerV1 } from "@/services/chat-history-selection"
import { linkServerChatMirror } from "@/db/dexie/server-chat-mirror"
import { tldwChat } from "@/services/tldw"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreMessageOption } from "@/store/option"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { Storage } from "@plasmohq/storage"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { readChatCreationRecovery } from "@/services/pending-chat-promotion"
import type { ChatDraftHandle } from "@/utils/chat-draft-address"

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
  snapshot?: ServicePromptSnapshot | null
  originIsCurrent?: () => boolean
  setHistoryId?: (id: string) => void
  toolChoice?: Parameters<typeof normalChatMode>[6]["toolChoice"]
  fixedOverrides?: boolean
  setServerChatId?: (id: string) => void
  saveOnSuccess?: typeof saveMessageOnSuccess
  surface?: "playground" | "sidepanel"
} = {}) => {
  const params: Parameters<typeof normalChatMode>[6] = {
    selectedModel: "test", selectedSystemPrompt: "", useOCR: false,
    currentChatModelSettings: {}, servicePromptSnapshot: options.snapshot === null ? undefined : options.snapshot ?? snapshot(),
    toolChoice: options.toolChoice ?? useStoreMessageOption.getState().toolChoice,
    selectionSource: {
      model: options.fixedOverrides ? "explicit" : "global",
      toolChoice: options.fixedOverrides ? "explicit" : "global"
    },
    historySelection: { controller, originIsCurrent: options.originIsCurrent ?? (() => true), createServerChat: Boolean(options.setServerChatId) },
    historyId: options.historyId ?? null,
    setServerChatId: options.setServerChatId,
    ...{ surface: options.surface ?? "playground" },
    setHistoryId: options.setHistoryId ?? (() => {}),
    setMessages: () => {}, setHistory: () => {}, setIsProcessing: () => {},
    setStreaming: () => {}, setAbortController: () => {},
    saveMessageOnSuccess: options.saveOnSuccess ?? saveMessageOnSuccess, saveMessageOnError,
    userMessageId: "first-user", assistantMessageId: "first-answer"
  }
  return normalChatMode("First question", "", false, [], [], new AbortController().signal, params)
}
const publicHistory = (id: string): HistoryInfo => ({ id, title: "Imported", createdAt: 1, is_rag: false, message_source: "web-ui" })
const savedMessages = async (): Promise<Message[]> => await storage.tables.messages.toArray() as Message[]
const emptyServerCapture = (request: Parameters<typeof tldwClient.captureHistorySelection>[1], id = "new-server") => {
  const view = { ...request.view, owner_key: "server-owner" }
  return {
    status: "captured", purpose: request.purpose, view, rows: [], selected_content: [],
    snapshot: { version: 1, owner_key: view.owner_key, conversation_id: id, nodes: [],
      fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source",
      storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } },
    storage_context_digest: "storage"
  } as never
}

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear(); sessionStorage.clear()
  useStoreMessageOption.getState().startNewChatDraft()
  const tails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: (key: string, work: () => unknown) => {
    const next = (tails.get(key) ?? Promise.resolve()).then(work); tails.set(key, next.catch(() => undefined)); return next
  } } })
  Object.values(storage.tables).forEach(table => table.rows.clear())
  storage.control.beforeHistoryAdd = undefined
  prompt.scope.userId = "alice"
  preparation.beforeModel.mockReset()
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  useStoreMessageOption.setState({ selectedModel: "test", toolChoice: "none" })
  vi.spyOn(tldwChat, "streamMessage").mockImplementation(async function* () { yield "First answer" })
})

afterEach(() => { vi.restoreAllMocks() })

describe("normal Chat owned history creation and H1 admission", () => {
  it.each(["owned", "foreign", "other conversation"])("checks a historyId-only native mirror against its captured owner (%s)", async kind => {
    const historyId = await linkServerChatMirror({ chatId: kind === "other conversation" ? "other" : "native",
      title: "Mirror", ownerKey: kind === "foreign" ? "other-account" : ACCOUNT })
    const owner: NativeHistoryOwnerV1 = { kind: "native", conversation_id: "native", request_scope: snapshot().requestScope, validate_lease: () => true }
    const check = assertHistoryConversationTarget(owner, { historyId })
    if (kind === "owned") await expect(check).resolves.toBeUndefined()
    else await expect(check).rejects.toThrow("owner_conversation_mismatch")
  })

  it("refuses a ready selection belonging to a different displayed conversation", async () => {
    const source = await saveHistory("Source", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const target = await saveHistory("Target", false, "web-ui", undefined, undefined, snapshot().requestScope)
    const mounted = renderHook(() => useHistorySelection())
    await act(async () => { await mounted.result.current.loadConversation({ historyId: target.id }) })
    await act(async () => {
      await expect(send(mounted.result.current, { historyId: source.id })).rejects.toThrow("owner_conversation_mismatch")
    })
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it.each(["Stop", "navigation", "account"])("does not adopt a native load completing after %s", async boundary => {
    const mounted = renderHook(() => useHistorySelection())
    const scoped = snapshot()
    const stopped = new AbortController()
    const auth = new AbortController()
    Object.assign(scoped, { scopeSignal: stopped.signal, scopeInvalidatedSignal: auth.signal })
    vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
    let finish!: () => void
    const waiting = new Promise<void>(resolve => { finish = resolve })
    const capture = vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      await waiting
      return emptyServerCapture(request)
    })
    const adopt = vi.fn()
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { snapshot: scoped, setServerChatId: adopt, originIsCurrent: mounted.result.current.fence() }) })
    await waitFor(() => expect(capture).toHaveBeenCalledOnce())
    await act(async () => {
      if (boundary === "Stop") stopped.abort()
      if (boundary === "account") auth.abort()
      if (boundary === "navigation") mounted.result.current.reset()
      finish()
      await expect(pending).rejects.toThrow("stale_selection")
    })
    expect(adopt).not.toHaveBeenCalled()
    expect(mounted.result.current.status).toBe(boundary === "navigation" ? "idle" : "stale_selection")
    const owner = mounted.result.current.getCurrent().owner
    if (owner?.kind === "native") expect(owner.validate_lease()).toBe(false)
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    expect(await savedMessages()).toEqual([])
    mounted.unmount()
  })

  it("does not admit after a qualified native receipt is followed by an epoch-only load change", async () => {
    const mounted = renderHook(() => useHistorySelection())
    vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
    const capture = vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
    const admission = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, value) => {
      const selection = value.tldw_history_selection_v1
      return { id: value.id, ...(selection ? { tldw_history_admission_v1: {
        version: 1, owner_key: "server-owner", conversation_id: "new-server", input_message_id: value.id,
        input_message_revision: "1", selection_digest: selection.selection_digest,
        messages: selection.messages, originating_selection_revision: selection.selection_revision
      } } : {}) } as never
    })
    const controller = mounted.result.current
    const load = controller.loadConversation
    const changed = { ...controller, loadConversation: async (...args: Parameters<typeof load>) => {
      const loaded = await load(...args)
      controller.beginLoad()
      return loaded
    } }
    const adopt = vi.fn()
    await act(async () => { await expect(send(changed, { setServerChatId: adopt, originIsCurrent: controller.fence() })).rejects.toThrow("stale_selection") })
    expect(adopt).toHaveBeenCalledWith("new-server")
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    const owner = mounted.result.current.getCurrent().owner
    expect(owner?.kind === "native" && owner.validate_lease()).toBe(false)
    capture.mockClear()
    await act(async () => { await expect(send(controller)).rejects.toThrow("stale_selection") })
    expect(capture).not.toHaveBeenCalled()
    expect(admission).not.toHaveBeenCalled()
    expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it("keeps an enabled-tool first send local without creating an empty native chat", async () => {
    const mounted = renderHook(() => useHistorySelection())
    useStoreMessageOption.setState({ toolChoice: "auto" })
    useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{
      name: "lookup", canExecute: true, inputSchema: { type: "object", properties: {} }
    }] }).chatTools })
    const create = vi.spyOn(tldwClient, "createChat")
    await act(async () => { expect(await send(mounted.result.current, { setServerChatId: vi.fn() })).toEqual({ status: "submitted" }) })
    expect(create).not.toHaveBeenCalled()
    expect(mounted.result.current.owner?.kind).toBe("local")
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

  it.each(["navigation", "account", "Stop", "missing ACK", "creation failure"])(
    "never adopts or generates after native creation crosses %s",
    async boundary => {
      const mounted = renderHook(() => useHistorySelection())
      const scoped = snapshot()
      const cancelled = new AbortController()
      Object.assign(scoped, { scopeSignal: cancelled.signal, scopeInvalidatedSignal: cancelled.signal })
      let finish!: () => void
      const waiting = new Promise<void>(resolve => { finish = resolve })
      let current = true
      const create = vi.spyOn(tldwClient, "createChat").mockImplementation(async () => {
        await waiting
        if (boundary === "creation failure") throw new Error("create unavailable")
        return { id: boundary === "missing ACK" ? "" : "late-chat" } as never
      })
      const adopt = vi.fn()
      let pending!: ReturnType<typeof send>
      act(() => { pending = send(mounted.result.current, { snapshot: scoped, setServerChatId: adopt, originIsCurrent: () => current }) })
      await waitFor(() => expect(create).toHaveBeenCalledOnce())
      await act(async () => {
        if (boundary === "navigation") { current = false; mounted.result.current.reset() }
        if (boundary === "account" || boundary === "Stop") cancelled.abort()
        finish()
        await expect(pending).rejects.toThrow()
      })
      expect(adopt).not.toHaveBeenCalled()
      expect(mounted.result.current.getCurrent().owner).toBeNull()
      expect(vi.mocked(tldwChat.streamMessage)).not.toHaveBeenCalled()
      expect(await savedMessages()).toEqual([])
      expect(await storage.tables.chatHistories.toArray()).toEqual([])
      mounted.unmount()
    }
  )

  it.each(["First answer", "<think>Reasoning text</think>First answer"])("creates a native owner and reopens its server pair, preserving %s", async answer => {
    const source = creationSource()
    const sourceAddress = useStoreMessageOption.getState().chatDraftAddress
    const stopped = new AbortController()
    const scoped = { ...snapshot(), scopeSignal: stopped.signal }
    const rows: Array<{ id: string; role: string; content: string; parent_message_id: string | null }> = []
    const create = vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      const view = { ...request.view, owner_key: "server-owner" }
      const nodes = rows.map(row => ({ id: row.id, role: row.role, parent_id: row.parent_message_id, revision: "1", settled: true }))
      return {
        status: "captured", purpose: request.purpose, view,
        snapshot: { version: 1, owner_key: view.owner_key, conversation_id: "new-server", nodes,
          fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source",
          storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } },
        rows: view.cursor.kind === "empty" ? [] : nodes,
        selected_content: view.cursor.kind === "empty" ? [] : rows.map(row => ({ id: row.id, revision: "1", message: row.content, images: [] })),
        storage_context_digest: "storage"
      } as never
    })
    const append = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, body) => {
      rows.push({ id: body.id!, role: body.role, content: body.content!, parent_message_id: body.parent_message_id ?? null })
      const selection = body.tldw_history_selection_v1
      return { id: body.id, ...(selection ? { tldw_history_admission_v1: {
        version: 1, owner_key: "server-owner", conversation_id: "new-server", input_message_id: body.id,
        input_message_revision: "1", selection_digest: selection.selection_digest,
        messages: selection.messages, originating_selection_revision: selection.selection_revision
      } } : {}) } as never
    })
    if (answer.startsWith("<think>")) {
      vi.mocked(tldwChat.streamMessage).mockImplementation(async function* () {
        yield "<think>Reasoning text"
        await new Promise(resolve => setTimeout(resolve, 10))
        yield "</think>First answer"
      })
    }
    const mounted = renderHook(() => useHistorySelection())
    const adopt = vi.fn()
    await act(async () => {
      expect(await send(mounted.result.current, { setServerChatId: adopt, snapshot: scoped })).toEqual({ status: "submitted" })
    })
    expect(create).toHaveBeenCalledOnce()
    expect(create.mock.calls[0][1]).toMatchObject({ requestScope: prompt.scope, signal: expect.any(AbortSignal) })
    expect(adopt).toHaveBeenCalledWith("new-server")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "native", id: "new-server", ownerKey: "server-owner" })
    expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({ kind: "bind", sourceAddress })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "settled", chatId: "new-server", inputText: "" })
    expect(mounted.result.current.owner).toMatchObject({ kind: "native", conversation_id: "new-server" })
    stopped.abort()
    const owner = mounted.result.current.getCurrent().owner
    expect(owner?.kind === "native" && owner.validate_lease()).toBe(true)
    expect(append).toHaveBeenCalledTimes(2)
    expect(rows.map(row => [row.role, row.content, row.parent_message_id])).toEqual([
      ["user", "First question", null], ["assistant", answer, "first-user"]
    ])
    expect(await savedMessages()).toEqual([])
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).toMatchObject({ save_to_db: false })
    const reference = mounted.result.current.getReference()!
    mounted.unmount()
    const reopened = renderHook(() => useHistorySelection())
    await act(async () => { await reopened.result.current.loadConversation({ serverChatId: "new-server" }, reference) })
    expect(reopened.result.current.status).toBe("ready")
    expect(reopened.result.current.capture?.status === "captured" && reopened.result.current.capture.selected_content.map(row => row.message)).toEqual(["First question", answer])
    reopened.unmount()
  })

  it("dispatches explicit model and tool overrides despite unrelated global selector changes", async () => {
    const mounted = renderHook(() => useHistorySelection())
    const created = await saveHistory("Saved", false, "web-ui", undefined, undefined, snapshot().requestScope)
    await act(async () => { await mounted.result.current.loadConversation({ historyId: created.id }) })
    preparation.beforeModel.mockImplementation(() => {
      useStoreMessageOption.setState({ selectedModel: "unrelated-model", toolChoice: "auto" })
    })
    await act(async () => {
      expect(await send(mounted.result.current, { historyId: created.id, toolChoice: "none", fixedOverrides: true })).toEqual({ status: "submitted" })
    })
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).toMatchObject({ model: "test" })
    expect(vi.mocked(tldwChat.streamMessage).mock.calls[0][1]?.preparedRequest).not.toHaveProperty("tools")
    expect((await savedMessages()).map(row => row.role)).toEqual(["user", "assistant"])
    mounted.unmount()
  })

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
    useStoreMessageOption.getState().startNewChatDraft();
    const sourceDraft = useStoreMessageOption.getState().chatDraftAddress;
    const mounted = renderHook(() => useHistorySelection())
    let historyId = ""
    await act(async () => {
      expect(await send(mounted.result.current, { originIsCurrent: mounted.result.current.fence(), setHistoryId: id => { historyId = id } })).toEqual({ status: "submitted" })
    })
    const persisted = await new PageAssistDatabase().getHistoryInfo(historyId)
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: historyId });
    expect(useStoreMessageOption.getState().chatDraftTransition).toMatchObject({ kind: "bind", sourceAddress: sourceDraft });
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
    useStoreMessageOption.getState().startNewChatDraft();
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
      useStoreMessageOption.getState().startNewChatDraft();
      mounted.result.current.reset()
      finish()
      await expect(pending).rejects.toThrow("stale_selection")
    })
    expect(adopted).toBe("")
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
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


const creationSource = (): ChatDraftHandle => ({
  address: structuredClone(useStoreMessageOption.getState().chatDraftAddress!),
  ownerHash: chatDraftOwnerKey(snapshot().requestScope)
})

describe("ordinary native creation recovery and preparation", () => {
  it.each(["playground", "sidepanel"] as const)("%s keeps unknown create input and blocks repeated and cold dispatch", async surface => {
    const source = creationSource()
    const mounted = renderHook(() => useHistorySelection())
    const create = vi.spyOn(tldwClient, "createChat").mockRejectedValue(new Error("Response lost"))
    const adopt = vi.fn()
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: adopt, surface })).rejects.toThrow() })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "unconfirmed", inputText: "First question", surface })
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: adopt, surface })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" }) })
    mounted.unmount()
    const cold = renderHook(() => useHistorySelection())
    await act(async () => { await expect(send(cold.result.current, { setServerChatId: adopt, surface })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" }) })
    expect(create).toHaveBeenCalledOnce()
    expect(adopt).not.toHaveBeenCalled()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    cold.unmount()
  })

  it.each(["new", "ABA", "account", "Stop"])("keeps a late ordinary create ACK private after %s", async boundary => {
    const source = creationSource()
    const originalAddress = useStoreMessageOption.getState().chatDraftAddress
    const mounted = renderHook(() => useHistorySelection())
    const scoped = snapshot()
    const abort = new AbortController()
    Object.assign(scoped, boundary === "account" ? { scopeInvalidatedSignal: abort.signal } : { scopeSignal: abort.signal })
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const create = vi.spyOn(tldwClient, "createChat").mockImplementation(async () => { await held; return { id: "late-server" } as never })
    const adopt = vi.fn()
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setServerChatId: adopt, snapshot: scoped, originIsCurrent: mounted.result.current.fence() }) })
    await waitFor(() => expect(create).toHaveBeenCalledOnce())
    act(() => {
      if (boundary === "new" || boundary === "ABA") {
        useStoreMessageOption.getState().startNewChatDraft()
        if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(originalAddress)
      } else abort.abort()
    })
    await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
    expect(await readChatCreationRecovery(source)).toMatchObject({ state: "acknowledged", chatId: "late-server", inputText: "First question" })
    expect(adopt).not.toHaveBeenCalled()
    expect(mounted.result.current.getCurrent().owner).toBeNull()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it.each(["create", "capture", "bookmark"].flatMap(stage => ["model", "settings", "tools"].map(boundary => ({ stage, boundary }))))("rejects $boundary drift during held first native $stage", async ({ stage, boundary }) => {
    const source = useStoreMessageOption.getState().chatDraftAddress
    const mounted = renderHook(() => useHistorySelection())
    let enter!: () => void, release!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(tldwClient, "createChat").mockImplementation(async () => {
      if (stage === "create") { enter(); await held }
      return { id: "new-server" } as never
    })
    vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => {
      if (stage === "capture") { enter(); await held }
      return emptyServerCapture(request)
    })
    if (stage === "bookmark") {
      const put = storage.tables.historySelections.put
      vi.spyOn(storage.tables.historySelections, "put").mockImplementation(async value => { enter(); await held; return put(value) })
    }
    const adopt = vi.fn()
    let pending!: ReturnType<typeof send>
    act(() => { pending = send(mounted.result.current, { setServerChatId: adopt }) })
    await entered
    act(() => {
      if (boundary === "model") useStoreMessageOption.getState().setSelectedModel("new-model")
      else if (boundary === "settings") useStoreChatModelSettings.getState().setTemperature(0.73)
      else useStoreMessageOption.getState().setToolChoice("auto")
    })
    await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
    expect(adopt).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })

  it.each(["storage", "locks"])("refuses ordinary %s unavailability before creating", async boundary => {
    const mounted = renderHook(() => useHistorySelection())
    const create = vi.spyOn(tldwClient, "createChat")
    if (boundary === "storage") vi.spyOn(Storage.prototype, "get").mockRejectedValue(new Error("Unavailable"))
    else Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    await act(async () => { await expect(send(mounted.result.current, { setServerChatId: vi.fn() })).rejects.toMatchObject({ code: "chat_creation_recovery_unavailable" }) })
    expect(create).not.toHaveBeenCalled()
    expect(tldwChat.streamMessage).not.toHaveBeenCalled()
    mounted.unmount()
  })
})


it.each(["new", "ABA"])("captures the source before awaited service setup crosses %s", async boundary => {
  const source = useStoreMessageOption.getState().chatDraftAddress
  const mounted = renderHook(() => useHistorySelection())
  let enter!: () => void, release!: () => void
  const entered = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.spyOn(servicePrompts, "loadServicePromptSnapshot").mockImplementation(async () => { enter(); await held; return snapshot() })
  const create = vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
  let pending!: Promise<unknown>
  act(() => { pending = send(mounted.result.current, { setServerChatId: vi.fn(), snapshot: null }).catch(error => error) })
  await entered
  act(() => {
    useStoreMessageOption.getState().startNewChatDraft()
    if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(source)
  })
  let outcome: unknown
  await act(async () => { release(); outcome = await pending })
  expect(create).not.toHaveBeenCalled()
  expect(outcome).toMatchObject({ message: "stale_selection" })
  expect(tldwChat.streamMessage).not.toHaveBeenCalled()
  mounted.unmount()
})


it.each([false, true])("keeps original ordinary input bound when the assistant save acknowledgement is lost (persisted=%s)", async persisted => {
  const serverReplies: unknown[] = []
  const source = creationSource()
  const mounted = renderHook(() => useHistorySelection())
  vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (_id, request) => emptyServerCapture(request))
  const append = vi.spyOn(tldwClient, "addChatMessage").mockImplementation(async (_id, body) => {
    if (body.role === "assistant") {
      if (persisted) serverReplies.push(structuredClone(body))
      throw new Error("Assistant save response lost")
    }
    const selection = body.tldw_history_selection_v1!
    return { id: body.id, tldw_history_admission_v1: {
      version: 1, owner_key: "server-owner", conversation_id: "new-server", input_message_id: body.id,
      input_message_revision: "1", selection_digest: selection.selection_digest,
      messages: selection.messages, originating_selection_revision: selection.selection_revision
    } } as never
  })
  await act(async () => { expect((await send(mounted.result.current, { setServerChatId: vi.fn() })).status).toBe("failed") })
  expect(append).toHaveBeenCalledTimes(2)
  expect(serverReplies).toHaveLength(persisted ? 1 : 0)
  const selected = mounted.result.current.getCurrent()
  const retained = await loadHistoryTurnRecoveries(selected.bookmarkScope!, selected.view!)
  expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({
    state: "generated_unsaved", input_text: "First question", result_text: "First answer"
  }) })])
  expect(retained[0].turn.outcome).toBeUndefined()
  expect(await readChatCreationRecovery(source)).toMatchObject({ state: "bound", chatId: "new-server", inputText: "First question" })
  mounted.unmount()
})

it.each(["new-server", "newer-server"])("a held ordinary bound-record write cannot dispatch after a newer H1 load of %s", async nextId => {
  const source = creationSource()
  const mounted = renderHook(() => useHistorySelection())
  vi.spyOn(tldwClient, "createChat").mockResolvedValue({ id: "new-server" } as never)
  vi.spyOn(tldwClient, "captureHistorySelection").mockImplementation(async (id, request) => {
    return emptyServerCapture(request, id)
  })
  const original = Storage.prototype.set
  let enter!: () => void, release!: () => void
  const entered = new Promise<void>(resolve => { enter = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) {
    const result = await original.call(this, key, value)
    if (key.startsWith("registry:chat-creation:") && value?.value?.state === "bound") { enter(); await held }
    return result
  })
  let pending!: ReturnType<typeof send>
  act(() => { pending = send(mounted.result.current, { setServerChatId: vi.fn() }) })
  await entered
  await act(async () => { await mounted.result.current.loadConversation({ serverChatId: nextId }) })
  const newer = mounted.result.current.getCurrent()
  await act(async () => { release(); await expect(pending).rejects.toThrow("stale_selection") })
  expect(mounted.result.current.getCurrent().owner).toBe(newer.owner)
  expect(mounted.result.current.getCurrent().view).toBe(newer.view)
  expect(newer.owner?.kind === "native" && newer.owner.validate_lease()).toBe(true)
  expect(tldwChat.streamMessage).not.toHaveBeenCalled()
  expect(await readChatCreationRecovery(source)).toMatchObject({ state: "bound", inputText: "First question" })
  mounted.unmount()
})


it.each(["cleanup", "navigation", "account"])("retains an acknowledged local assistant through %s after saving", async boundary => {
  const source = await saveHistory("Source", false, "web-ui", undefined, undefined, snapshot().requestScope)
  const other = await saveHistory("Other", false, "web-ui", undefined, undefined, snapshot().requestScope)
  const mounted = renderHook(() => useHistorySelection())
  await act(async () => { await mounted.result.current.loadConversation({ historyId: source.id }) })
  const origin = mounted.result.current.getCurrent()
  const scoped = snapshot()
  const invalidation = new AbortController()
  // Explicit account-signal I/O double; real event/controller retirement is covered by root ownership checks.
  scoped.scopeInvalidatedSignal = invalidation.signal
  let acknowledgedId: string | undefined
  const save = vi.fn(async (data: Parameters<typeof saveMessageOnSuccess>[0]) => {
    const turn = data.historyTurn!
    const complete = turn.complete!
    turn.complete = async () => {
      await complete()
      acknowledgedId = turn.resultId
      if (boundary !== "cleanup") {
        if (boundary === "account") invalidation.abort()
        await mounted.result.current.loadConversation({ historyId: other.id })
        useStoreMessageOption.getState().startNewChatDraft()
      }
      throw new Error("Ancillary cleanup failed")
    }
    return saveMessageOnSuccess(data)
  })
  await act(async () => {
    expect((await send(mounted.result.current, { historyId: source.id, saveOnSuccess: save, snapshot: scoped })).status).toBe(boundary === "account" ? "failed" : "submitted")
  })
  expect(acknowledgedId).toBe("first-answer")
  expect(save).toHaveBeenCalledOnce()
  expect((await savedMessages()).filter(row => row.role === "assistant")).toHaveLength(1)
  if (boundary === "cleanup") {
    expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
  } else {
    expect(mounted.result.current.getCurrent().owner?.conversation_id).toBe(other.id)
    expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "empty" })
    const retained = await loadHistoryTurnRecoveries(origin.bookmarkScope!, origin.view!)
    if (boundary === "navigation") {
      expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({ settled_message_id: "first-answer", outcome: "complete" }) })])
      await act(async () => {
        await mounted.result.current.loadConversation({ historyId: source.id })
        expect((await keepHistoryTurnRecovery(mounted.result.current, retained[0])).status).toBe("kept")
      })
      expect((await savedMessages()).filter(row => row.role === "assistant")).toHaveLength(1)
      expect(mounted.result.current.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "first-answer" })
    } else expect(retained).toEqual([expect.objectContaining({ turn: expect.objectContaining({ settled_message_id: "first-answer", outcome: "complete" }) })])
  }
  mounted.unmount()
})
