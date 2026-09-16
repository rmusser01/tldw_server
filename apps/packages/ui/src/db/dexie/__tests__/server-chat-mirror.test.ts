import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistoryInfo, Message } from "../types"

// Only IndexedDB's storage/transaction boundary is replaced; reconciliation is real.
const state = vi.hoisted(() => ({ histories: new Map<string, HistoryInfo>(), messages: new Map<string, Message>() }))
function table<T extends { id: string }>(rows: Map<string, T>) { return ({
  get: async (id: string) => rows.get(id),
  add: async (row: T) => { if (rows.has(row.id)) throw new Error("duplicate primary key"); rows.set(row.id, structuredClone(row)); return row.id },
  put: async (row: T) => { rows.set(row.id, structuredClone(row)); return row.id },
  update: async (id: string, changes: Partial<T>) => { if (!rows.has(id)) return 0; rows.set(id, { ...rows.get(id)!, ...changes }); return 1 },
  where: (field: string) => ({ equals: (value: unknown) => ({ toArray: async () => [...rows.values()].filter(row => (row as Record<string, unknown>)[field] === value) }) })
}) }
vi.mock("../schema", () => ({ db: {
  chatHistories: table(state.histories), messages: table(state.messages), modelNickname: {}, sessionFiles: {},
  transaction: async (_mode: string, _tables: unknown[], operation: (transaction: { abort: () => void }) => Promise<unknown>) => operation({ abort: vi.fn() })
} }))
vi.mock("../helpers", () => ({ generateID: () => `history-${state.histories.size + 1}` }))
import { linkServerChatMirror, reconcileServerChatMirror, reconcileServerChatMessages } from "../server-chat-mirror"

const history = (id: string, owner?: string): HistoryInfo => ({ id, title: "Cedar", is_rag: false, createdAt: 1, server_chat_id: "chat-1", ...(owner ? { server_scope_key: owner } : {}) })
const row = (id: string, historyId: string, content: string, serverMessageId?: string): Message => ({ id, history_id: historyId, name: "You", role: "user", content, images: [], createdAt: 1, serverMessageId })
const incoming = (id: string, content: string, version = 1) => ({ id, serverMessageId: id, serverMessageVersion: version, role: "user", isBot: false, name: "You", message: content, images: [], createdAt: 1 })
const link = (ownerKey: string, extra = {}) => linkServerChatMirror({ chatId: "chat-1", title: "Cedar", ownerKey, ...extra })

describe("owned server Chat mirror", () => {
  beforeEach(() => { state.histories.clear(); state.messages.clear() })
  it.each([{ images: [] }, { images: [""] }])("recovers an anchored legacy user with images $images and keeps equal-text drafts", async ({ images }) => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images })
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", "answer"), role: "assistant", parent_message_id: "local-q" })
    state.messages.set("draft", { ...row("draft", "alice", "Repeat"), images })
    const remote = [incoming("question", "Repeat"), { ...incoming("answer", "Reply"), isBot: true, role: "assistant" }]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect([...state.messages.values()].map(r => [r.id, r.serverMessageId])).toEqual([
      ["local-q", "question"], ["local-a", "answer"], ["draft", undefined]
    ])
    expect(state.messages.get("local-a")?.parent_message_id).toBe("local-q")
    const current = [
      { ...incoming("local-q", "Repeat"), images, serverMessageId: undefined },
      { ...remote[1], id: "local-a", parentMessageId: "local-q" },
      { ...incoming("draft", "Repeat"), images, serverMessageId: undefined }
    ]
    expect(reconcileServerChatMessages(current, remote).map(r => [r.id, r.serverMessageId])).toEqual([
      ["local-q", "question"], ["local-a", "answer"], ["draft", undefined]
    ])
  })
  it.each([
    { localImages: ["data:image/png;base64,local"], remoteImages: [] },
    { localImages: [""], remoteImages: ["data:image/png;base64,remote"] },
    { localImages: ["data:image/png;base64,local"], remoteImages: ["data:image/png;base64,remote"] }
  ])("preserves anchored work when substantive images differ ($localImages / $remoteImages)", async ({ localImages, remoteImages }) => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images: localImages })
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", "answer"), role: "assistant", parent_message_id: "local-q" })
    const remote = [
      { ...incoming("question", "Repeat"), images: remoteImages },
      { ...incoming("answer", "Reply"), isBot: true, role: "assistant" }
    ]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect(state.messages.get("local-q")).toMatchObject({ serverMessageId: undefined, images: localImages })
    expect(state.messages.size).toBe(3)
  })
  it.each(["missing-parent", "edited-user", "already-mirrored", "wrong-reply", "conflicting-parent"])("keeps ambiguous legacy work: %s", async kind => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", row("local-q", "alice", kind === "edited-user" ? "Edited" : "Repeat"))
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", kind === "wrong-reply" ? "other-answer" : "answer"), role: "assistant", parent_message_id: kind === "missing-parent" ? null : "local-q" })
    if (kind === "already-mirrored") state.messages.set("canonical-q", row("canonical-q", "alice", "Repeat", "question"))
    const remote = [incoming("question", "Repeat"), { ...incoming("answer", "Reply"), isBot: true, role: "assistant", ...(kind === "conflicting-parent" ? { parentMessageId: "other-user" } : {}) }]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect(state.messages.get("local-q")).toMatchObject({ serverMessageId: undefined, content: kind === "edited-user" ? "Edited" : "Repeat" })
  })
  it("keeps identical conversation IDs in different owners' histories", async () => {
    state.histories.set("alice", history("alice", "A"))
    const bob = await link("B", { currentHistoryId: "alice" })
    expect(bob).not.toBe("alice")
    expect(state.histories.get("alice").server_scope_key).toBe("A")
    expect(state.histories.get(bob).server_scope_key).toBe("B")
  })
  it("adopts only the explicitly validated current legacy history", async () => {
    state.histories.set("legacy", history("legacy"))
    expect(await link("A", { currentHistoryId: "legacy", legacyHistoryId: "legacy" })).toBe("legacy")
    expect(state.histories.get("legacy").server_scope_key).toBe("A")
  })
  it("leaves an unvalidated legacy history untouched", async () => {
    state.histories.set("legacy", history("legacy"))
    expect(await link("A", { currentHistoryId: "legacy" })).not.toBe("legacy")
    expect(state.histories.get("legacy").server_scope_key).toBeUndefined()
  })
  it("does not repurpose an owned history from another conversation", async () => {
    state.histories.set("other", { ...history("other", "A"), server_chat_id: "chat-other" })
    expect(await link("A", { currentHistoryId: "other", legacyHistoryId: "other" })).not.toBe("other")
    expect(state.histories.get("other").server_chat_id).toBe("chat-other")
  })
  it("fills a nonempty old mirror, retaining canonical IDs across repeated reads", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("question", row("question", "alice", "When?"))
    const args = { historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "When?"), incoming("answer", "08:30")] }
    await reconcileServerChatMirror(args)
    await reconcileServerChatMirror(args)
    const saved = [...state.messages.values()]
    expect(saved).toHaveLength(2)
    expect(saved.map(r => [r.serverMessageId, r.content])).toEqual([["question", "When?"], ["answer", "08:30"]])
    expect(saved[1].id).not.toBe("answer")
  })
  it("retains normalized metadata through mirror storage", async () => {
    state.histories.set("alice", history("alice", "A"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [{ ...incoming("answer", "08:30"), metadataExtra: { speaker_character_id: 4, mood: "calm" } }] })
    expect([...state.messages.values()][0].metadataExtra).toEqual({ speaker_character_id: 4, mood: "calm" })
  })
  it("adds missing replies without replacing an unsynced visible draft or a newer edit", () => {
    const result = reconcileServerChatMessages([
      { ...incoming("question", "Newer question", 3), id: "local-question" },
      { ...incoming("draft", "Unsent"), serverMessageId: undefined }
    ], [incoming("question", "Old question", 2), incoming("answer", "08:30")])
    expect(result.map(message => [message.id, message.message, message.serverMessageId])).toEqual([
      ["local-question", "Newer question", "question"], ["answer", "08:30", "answer"], ["draft", "Unsent", undefined]
    ])
  })
  it("preserves unsynced drafts and newer acknowledged rows", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("draft", row("draft", "alice", "Unsent"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Newer edit", "question"), serverMessageVersion: 3 })
    state.messages.set("late", row("late", "alice", "Acknowledged after GET", "late"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "Old", 2), incoming("answer", "08:30")] })
    expect([...state.messages.values()].map(r => r.content)).toEqual(["Unsent", "Newer edit", "Acknowledged after GET", "08:30"])
  })
  it("retains an unknown-version local edit when acknowledging an old visible row", () => {
    const merged = reconcileServerChatMessages([{ id: "question", role: "user", isBot: false, name: "You", message: "My unsynced edit" }], [incoming("question", "Original server text"), incoming("answer", "08:30")])
    expect(merged.map(message => message.message)).toEqual(["My unsynced edit", "08:30"])
    expect(merged[0].serverMessageId).toBe("question")
  })
  it("retains an unknown-version local edit in the old persistent mirror", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("question", row("question", "alice", "My unsynced edit"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "Original server text"), incoming("answer", "08:30")] })
    expect(state.messages.get("question")).toMatchObject({ content: "My unsynced edit", serverMessageId: "question" })
    expect(state.messages.size).toBe(2)
  })
  it("never moves another history's global message ID", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("answer", row("answer", "bob", "Bob secret", "answer"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("answer", "Alice answer")] })
    expect(state.messages.get("answer")).toMatchObject({ history_id: "bob", content: "Bob secret" })
    expect([...state.messages.values()].filter(r => r.history_id === "alice")).toMatchObject([{ content: "Alice answer", serverMessageId: "answer" }])
  })
  it("refuses a qualified-ID collision with another history", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("alice:server:answer", row("alice:server:answer", "bob", "Do not move"))
    await expect(reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("answer", "Alice answer")] })).rejects.toThrow()
    expect(state.messages.get("alice:server:answer").history_id).toBe("bob")
  })
  it.each(["wrong-owner", "aborted"])("rejects a %s mirror write", async mode => {
    state.histories.set("alice", history("alice", "A"))
    const controller = new AbortController(); if (mode === "aborted") controller.abort()
    await expect(reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: mode === "wrong-owner" ? "B" : "A", signal: controller.signal, messages: [incoming("answer", "Late")] })).rejects.toThrow()
    expect(state.messages.size).toBe(0)
  })
})
