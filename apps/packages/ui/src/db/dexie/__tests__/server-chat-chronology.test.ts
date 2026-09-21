import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Message as ChatMessage } from "@/store/option"
import type { HistoryInfo, Message } from "../types"

// Keep saveMessage, both reconcilers, and formatter real; replace IndexedDB I/O.
const state = vi.hoisted(() => ({
  histories: new Map<string, HistoryInfo>(),
  messages: new Map<string, Message>()
}))
function table<T extends { id: string }>(rows: Map<string, T>) {
  return {
    get: async (id: string) => rows.get(id),
    add: async (row: T) => {
      if (rows.has(row.id)) throw new Error("duplicate primary key")
      rows.set(row.id, structuredClone(row))
      return row.id
    },
    put: async (row: T) => {
      rows.set(row.id, structuredClone(row))
      return row.id
    },
    where: (field: string) => ({
      equals: (value: unknown) => ({
        toArray: async () => [...rows.values()].filter(
          row => (row as Record<string, unknown>)[field] === value
        )
      })
    })
  }
}
vi.mock("../schema", () => ({ db: {
  chatHistories: table(state.histories), messages: table(state.messages),
  modelNickname: {}, sessionFiles: {},
  transaction: async (_mode: string, _tables: unknown[], operation: (transaction: { abort: () => void }) => Promise<unknown>) => operation({ abort: vi.fn() })
} }))
vi.mock("../chat", () => ({ PageAssistDatabase: class {
  async addMessage(row: Message) { state.messages.set(row.id, structuredClone(row)) }
} }))
vi.mock("@/db/index", () => ({}))
vi.mock("../nickname", () => ({ ModelNickname: class {} }))
vi.mock("../models", () => ({ ModelDb: class {} }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({}))

import { formatToMessage, saveMessage } from "../helpers"
import { reconcileServerChatMessages, reconcileServerChatMirror } from "../server-chat-mirror"

const rawQuestion = "Chat with this media: Rowan.md\nSummarize this source."
const providerQuestion = `Context: <doc>Rowan is an observatory run by Mara Chen.</doc>\nQuestion: ${rawQuestion}`
const remote = (id: string, role: "user" | "assistant", message: string, createdAt: number): ChatMessage => ({
  id, serverMessageId: id, serverMessageVersion: 1, role,
  isBot: role === "assistant", name: role, message, sources: [], images: [], createdAt
})
const canonicalTurn = (question = providerQuestion) => [
  remote("server-user", "user", question, 1000),
  remote("server-answer", "assistant", "Rowan answer", 2000)
]
const mirror = (messages: ChatMessage[], localMessages: ChatMessage[] = []) => reconcileServerChatMirror({
  historyId: "history", chatId: "chat", ownerKey: "owner", messages, localMessages
})
const saveLocalTurn = async () => {
  vi.spyOn(Date, "now").mockReturnValue(3000)
  await saveMessage({ id: "local-user", history_id: "history", role: "user", name: "You",
    content: rawQuestion, images: [""], time: 1, serverMessageId: "server-user" })
  await saveMessage({ id: "local-answer", history_id: "history", role: "assistant", name: "Assistant",
    content: "Rowan answer", images: [], time: 2, serverMessageId: "server-answer", parent_message_id: "local-user" })
}

beforeEach(() => {
  state.histories.clear()
  state.messages.clear()
  state.histories.set("history", { id: "history", title: "Cedar", createdAt: 1,
    is_rag: false, server_chat_id: "chat", server_scope_key: "owner" })
})

describe("saved Chat canonical chronology", () => {
  it("keeps an acknowledged raw RAG question before its answer across repeated mirror reloads", async () => {
    await saveLocalTurn()
    const canonical = canonicalTurn()
    const current = formatToMessage([...state.messages.values()])
    const merged = reconcileServerChatMessages(current, canonical)
    const first = await mirror(canonical, merged)
    const reloaded = formatToMessage(first.rows)
    expect(reloaded.map(message => [message.role, message.message, message.createdAt])).toEqual([
      ["user", rawQuestion, 1000], ["assistant", "Rowan answer", 2000]
    ])
    expect(merged[0].createdAt).toBe(1000)
    expect(reloaded[1].parentMessageId).toBe("local-user")
    expect(reloaded.map(message => message.id)).toEqual(["local-user", "local-answer"])
    const second = await mirror(canonical, reloaded)
    expect(formatToMessage(second.rows)).toEqual(reloaded)
  })

  it("keeps the unchanged ordinary turn in canonical order", async () => {
    await saveLocalTurn()
    const result = await mirror(canonicalTurn(rawQuestion))
    expect(formatToMessage(result.rows).map(message => [message.role, message.createdAt])).toEqual([
      ["user", 1000], ["assistant", 2000]
    ])
  })

  it.each([undefined, 1, 3])("preserves protected content and images with local version %s without preserving its late timestamp", async version => {
    await saveLocalTurn()
    const image = "data:image/png;base64,local-attachment"
    const local = { ...state.messages.get("local-user")!, content: "Protected edit", images: [image], serverMessageVersion: version }
    state.messages.set(local.id, local)
    const incoming = canonicalTurn()
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), incoming)
    expect(memory[0]).toMatchObject({ message: "Protected edit", images: [image], createdAt: 1000, id: "local-user", serverMessageId: "server-user" })
    const result = await mirror(incoming, memory)
    expect(formatToMessage(result.rows)[0]).toMatchObject({ message: "Protected edit", images: [image], createdAt: 1000 })
  })

  it("preserves content edited during the snapshot await with canonical chronology", () => {
    const before = [{ ...remote("server-user", "user", "Before edit", 3001), id: "local-user" }]
    const current = [{ ...before[0], message: "Typed during load" }]
    const incoming = [{ ...remote("server-user", "user", "Server revision", 1000), serverMessageVersion: 2 }]
    expect(reconcileServerChatMessages(current, incoming, before)[0]).toMatchObject({
      message: "Typed during load", createdAt: 1000, id: "local-user", serverMessageId: "server-user"
    })
  })

  it.each([undefined, null, NaN, Infinity, -Infinity, "invalid"])("retains local chronology when canonical createdAt is %s", async invalid => {
    await saveLocalTurn()
    // Exercise equal content as well: malformed remote chronology must not replace it.
    const canonical = canonicalTurn(rawQuestion).map(message => ({ ...message, createdAt: invalid as number | undefined }))
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), canonical)
    const result = await mirror(canonical, memory)
    expect(memory.map(message => message.createdAt)).toEqual([3001, 3002])
    expect(formatToMessage(result.rows).map(message => message.createdAt)).toEqual([3001, 3002])
  })

  it("accepts epoch zero as a valid canonical timestamp", async () => {
    await saveLocalTurn()
    const canonical = canonicalTurn()
    canonical[0].createdAt = 0
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), canonical)
    const result = await mirror(canonical, memory)
    expect(memory[0].createdAt).toBe(0)
    expect(formatToMessage(result.rows)[0]).toMatchObject({ message: rawQuestion, createdAt: 0 })
  })

  it("retains distinct repeated drafts and an older local retrieval failure without claiming promotion receipts", async () => {
    await saveLocalTurn()
    for (const id of ["draft-one", "draft-two"]) {
      await saveMessage({ id, history_id: "history", role: "user", name: "You", content: rawQuestion, images: [], createdAt: 4000 })
    }
    await saveMessage({ id: "local-timeout", history_id: "history", role: "assistant", name: "Assistant",
      content: "Could not retrieve evidence", images: [], createdAt: 500 })
    const originalLocalRows = [...state.messages.values()].filter(message => !message.serverMessageId)
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), canonicalTurn())
    const result = await mirror(canonicalTurn(), memory)
    expect(result.rows.filter(message => !message.serverMessageId)).toEqual(originalLocalRows)
    expect(formatToMessage(result.rows).map(message => message.id)).toEqual([
      "local-timeout", "local-user", "local-answer", "draft-one", "draft-two"
    ])
    expect(memory.filter(message => !message.serverMessageId).map(message => message.id)).toEqual([
      "local-timeout", "draft-one", "draft-two"
    ])
  })

  it.each(["owner", "conversation", "history"])("does not adopt chronology across a changed %s", async changed => {
    await saveLocalTurn()
    const existing = state.histories.get("history")!
    if (changed === "history") state.histories.delete("history")
    else state.histories.set("history", { ...existing, ...(changed === "owner" ? { server_scope_key: "other" } : { server_chat_id: "other" }) })
    await expect(mirror(canonicalTurn())).rejects.toThrow()
    expect(state.messages.get("local-user")).toMatchObject({ createdAt: 3001, content: rawQuestion })
  })
})
